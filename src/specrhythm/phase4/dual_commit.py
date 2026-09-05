"""Serving-visible commits for the synchronous pinned-vLLM Dual observer.

Bookkeeping supplies rejection-parsed tokens, before serving stop checks. The
physical row is evidence of that delta, never the authority for logical output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

from specrhythm.phase4.serial import AcceptanceDecision, greedy_acceptance


def _tokens(values: Sequence[int]) -> Tuple[int, ...]:
    result = tuple(values)
    if any(type(token) is not int or token < 0 for token in result):
        raise ValueError("valid sampled tokens must be non-negative integers")
    return result


@dataclass(frozen=True)
class DualStopPolicy:
    maximum_new_tokens: int
    eos_token_id: Optional[int]
    stop_token_ids: Tuple[int, ...] = ()

    @classmethod
    def from_sampling_params(
        cls, params: Any, *, maximum: int, prompt_length: int, max_model_len: int
    ) -> DualStopPolicy:
        # Phase4 supplies greedy, single-output SamplingParams. Strings and
        # minimum-length/repetition stopping need other serving machinery.
        if (
            params.stop or params.min_tokens != 0
            or getattr(params, "repetition_detection", None) is not None
            or params.temperature != 0 or params.n != 1
        ):
            raise ValueError("unsupported Dual Phase4 sampling/stop contract")
        if type(maximum) is not int or maximum < 1 or params.max_tokens != maximum:
            raise ValueError("Dual sampling max_tokens differs from frozen workload")
        if prompt_length + maximum > max_model_len:
            raise ValueError("Dual frozen output budget exceeds model context limit")
        eos = params.eos_token_id
        if eos is not None:
            _tokens((eos,))
        stops = _tokens(params.stop_token_ids or ())
        if params.ignore_eos and eos is not None:
            raise ValueError("processed ignore_eos sampling still contains primary EOS")
        return cls(maximum, eos, stops)

    @property
    def terminal_token_ids(self) -> Tuple[int, ...]:
        return tuple(dict.fromkeys(
            (() if self.eos_token_id is None else (self.eos_token_id,))
            + self.stop_token_ids
        ))

    def canonicalize(
        self, previous: Sequence[int], sampled: Sequence[int]
    ) -> tuple[Tuple[int, ...], Optional[str]]:
        prior, delta = _tokens(previous), _tokens(sampled)
        if len(prior) >= self.maximum_new_tokens or any(
            token in self.terminal_token_ids for token in prior
        ):
            raise ValueError("cannot advance an already terminal logical output")
        generated = list(prior)
        for token in delta:
            generated.append(token)
            # Pinned check_stop tests EOS, explicit token stop, then length.
            if token == self.eos_token_id:
                return tuple(generated), "eos"
            if token in self.stop_token_ids:
                return tuple(generated), "stop"
            if len(generated) == self.maximum_new_tokens:
                return tuple(generated), "max_tokens"
        return tuple(generated), None


def dual_greedy_acceptance(
    proposal: Sequence[int], committed_delta: Sequence[int], *, terminal: bool
) -> AcceptanceDecision:
    """Classify a canonical Dual commit, including a terminal Draft prefix.

    A serving stop can retain only a prefix of accepted Draft tokens. The
    uncommitted suffix is rolled back, with no invented correction or bonus.
    Serial's existing acceptance contract is deliberately unchanged.
    """
    drafted, committed = _tokens(proposal), _tokens(committed_delta)
    if terminal and committed and len(committed) < len(drafted) and (
        committed == drafted[:len(committed)]
    ):
        return AcceptanceDecision(committed, drafted[len(committed):], (), (), committed, True)
    return greedy_acceptance(drafted, committed, terminal=terminal)


def phase4_dual_sampling_params(
    definition: Any, sampling_cls: Any, logprobs: Any = None
) -> Any:
    """The explicit request contract shared by the resident runner and observer."""
    return sampling_cls(
        temperature=0.0, top_p=1.0, max_tokens=definition.maximum_new_tokens,
        seed=definition.sampling_seed, n=1, logprobs=logprobs,
    )


def load_dual_stop_policies(
    vllm_config: Any, definitions: Sequence[Any]
) -> dict[str, DualStopPolicy]:
    # Mirror pinned InputProcessor: primary EOS comes from the renderer's
    # tokenizer; additional EOS IDs come from try_get_generation_config().
    # hf_config.eos_token_id alone is not the serving stop contract.
    from vllm import SamplingParams
    from vllm.renderers import renderer_from_config

    if vllm_config.scheduler_config.async_scheduling:
        raise ValueError("Dual logical commits require synchronous sampled-token evidence")
    renderer = renderer_from_config(vllm_config)
    generation = vllm_config.model_config.try_get_generation_config()
    policies = {}
    for definition in definitions:
        params = phase4_dual_sampling_params(definition, SamplingParams)
        params.update_from_generation_config(generation, renderer.get_eos_token_id())
        if renderer.tokenizer is not None:
            params.update_from_tokenizer(renderer.tokenizer)
        policies[definition.request_id] = DualStopPolicy.from_sampling_params(
            params, maximum=definition.maximum_new_tokens,
            prompt_length=len(definition.prompt_token_ids),
            max_model_len=vllm_config.model_config.max_model_len,
        )
    return policies
