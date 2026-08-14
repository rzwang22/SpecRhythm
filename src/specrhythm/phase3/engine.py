"""Correctness-first causal-LM abstraction for Phase-3 trace collection."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from specrhythm.phase3.config import ModelRuntimeConfig, resolve_runtime_path


class EngineUnavailableError(RuntimeError):
    """Raised when a requested real-model backend cannot be initialized."""


@dataclass(frozen=True)
class RankedToken:
    token_id: int
    logit: float
    probability: float


@dataclass(frozen=True)
class NextTokenDistribution:
    ranked_tokens: tuple[RankedToken, ...]
    entropy: float
    top1_top2_margin: float

    @property
    def top1(self) -> RankedToken:
        if not self.ranked_tokens:
            raise ValueError("next-token distribution is empty")
        return self.ranked_tokens[0]


class CausalLMBackend(Protocol):
    model_id: str
    eos_token_id: Optional[int]
    vocab_size: int
    tokenizer_fingerprint: str

    def encode(self, prompt: str) -> list[int]: ...

    def next_token(self, context: list[int], top_k: int) -> NextTokenDistribution: ...

    def next_token_batch(
        self, contexts: list[list[int]], top_k: int
    ) -> tuple[NextTokenDistribution, ...]: ...

    def close(self) -> None: ...


def _event(seed: int, model_id: str, context: list[int], token_id: int) -> float:
    payload = f"{seed}:{model_id}:{','.join(map(str, context))}:{token_id}".encode()
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / float(2**64)


class DryRunBackend:
    """Deterministic CPU backend for schema/lifecycle tests, never for latency claims."""

    def __init__(self, model_id: str, seed: int, *, vocab_size: int = 257) -> None:
        self.model_id = f"dry-run:{model_id}"
        self.seed = seed
        self.vocab_size = vocab_size
        self.eos_token_id = vocab_size - 1
        self.tokenizer_fingerprint = "dry-run-whitespace-sha256-v1"

    def encode(self, prompt: str) -> list[int]:
        if not prompt:
            return [1]
        return [
            1 + int.from_bytes(hashlib.sha256(piece.encode()).digest()[:4], "big")
            % (self.vocab_size - 2)
            for piece in prompt.split()
        ] or [1]

    def next_token(self, context: list[int], top_k: int) -> NextTokenDistribution:
        count = min(max(2, top_k), self.vocab_size)
        scored = [
            (2.0 * _event(self.seed, self.model_id, context, token) - 1.0, token)
            for token in range(self.vocab_size)
        ]
        scored.sort(reverse=True)
        top = scored[:count]
        maximum = top[0][0]
        all_exp = [math.exp(logit - maximum) for logit, _ in scored]
        denominator = sum(all_exp)
        ranked = tuple(
            RankedToken(token, logit, math.exp(logit - maximum) / denominator)
            for logit, token in top
        )
        probabilities = [value / denominator for value in all_exp]
        entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
        return NextTokenDistribution(
            ranked,
            entropy,
            top[0][0] - top[1][0],
        )

    def next_token_batch(
        self, contexts: list[list[int]], top_k: int
    ) -> tuple[NextTokenDistribution, ...]:
        return tuple(self.next_token(context, top_k) for context in contexts)

    def close(self) -> None:
        return None


class TransformersBackend:
    """Pinned optional backend exposing logits needed by the real-trace schema.

    This is a correctness collector, not a serving engine.  It recomputes complete
    contexts and deliberately avoids claiming vLLM/SGLang-equivalent throughput.
    """

    def __init__(self, config: ModelRuntimeConfig, seed: int) -> None:
        try:
            import torch  # type: ignore[import-not-found]
            import transformers  # type: ignore[import-not-found]
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise EngineUnavailableError(
                "Transformers backend requires the optional 'gpu' dependencies"
            ) from error
        if not torch.cuda.is_available():
            raise EngineUnavailableError(
                "Transformers backend requires CUDA; use backend=dry-run for CPU checks"
            )
        self._torch = torch
        self.model_id = resolve_runtime_path(config.model_path)
        self.tp_size = config.tp_size
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if self.tp_size > 1 and world_size != self.tp_size:
            raise EngineUnavailableError(
                f"TP={self.tp_size} requires torchrun WORLD_SIZE={self.tp_size}; got {world_size}"
            )
        if self.tp_size > 1 and not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                "nccl", device_id=torch.device(f"cuda:{local_rank}")
            )
        device_index = local_rank if self.tp_size > 1 else config.gpu_ids[0]
        torch.cuda.set_device(device_index)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        dtype = getattr(torch, config.dtype)
        load_args: dict[str, Any] = {
            "dtype": dtype,
            "trust_remote_code": config.trust_remote_code,
        }
        if config.revision:
            load_args["revision"] = config.revision
        if self.tp_size > 1:
            load_args["tp_plan"] = "auto"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=config.revision,
            trust_remote_code=config.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_args)
        if self.tp_size == 1:
            self.model.to(torch.device(f"cuda:{device_index}"))
        self.model.eval()
        self.device = torch.device(f"cuda:{device_index}")
        self.eos_token_id = self.tokenizer.eos_token_id
        self.vocab_size = int(self.model.config.vocab_size)
        tokenizer_payload = {
            "vocab": self.tokenizer.get_vocab(),
            "special_tokens_map": self.tokenizer.special_tokens_map,
        }
        self.tokenizer_fingerprint = hashlib.sha256(
            json.dumps(
                tokenizer_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        self.transformers_version = transformers.__version__

    def encode(self, prompt: str) -> list[int]:
        return list(self.tokenizer.encode(prompt, add_special_tokens=True))

    def next_token(self, context: list[int], top_k: int) -> NextTokenDistribution:
        return self.next_token_batch([context], top_k)[0]

    def next_token_batch(
        self, contexts: list[list[int]], top_k: int
    ) -> tuple[NextTokenDistribution, ...]:
        torch = self._torch
        if not contexts or any(not context for context in contexts):
            raise ValueError("causal-LM batch must contain non-empty contexts")
        lengths = {len(context) for context in contexts}
        if len(lengths) != 1:
            raise ValueError("correctness backend batching currently requires equal lengths")
        input_ids = torch.tensor(contexts, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.model(input_ids=input_ids, use_cache=False)
            logits = output.logits[:, -1].float()
            probabilities = torch.softmax(logits, dim=-1)
            count = min(max(2, top_k), int(logits.shape[-1]))
            top_logits, top_ids = torch.topk(logits, count, dim=-1)
            top_probabilities = torch.gather(probabilities, 1, top_ids)
            entropy = -(
                probabilities * torch.log(probabilities.clamp_min(1e-30))
            ).sum(dim=-1)
        ids = top_ids.detach().cpu().tolist()
        logits_cpu = top_logits.detach().cpu().tolist()
        probabilities_cpu = top_probabilities.detach().cpu().tolist()
        entropy_cpu = entropy.detach().cpu().tolist()
        return tuple(
            NextTokenDistribution(
                tuple(
                    RankedToken(int(token), float(logit), float(probability))
                    for token, logit, probability in zip(
                        row_ids, row_logits, row_probabilities
                    )
                ),
                float(row_entropy),
                float(row_logits[0] - row_logits[1]),
            )
            for row_ids, row_logits, row_probabilities, row_entropy in zip(
                ids, logits_cpu, probabilities_cpu, entropy_cpu
            )
        )

    def close(self) -> None:
        del self.model
        self._torch.cuda.empty_cache()


def create_backend(
    backend: str, model: ModelRuntimeConfig, seed: int
) -> CausalLMBackend:
    if backend == "dry-run":
        return DryRunBackend(model.model_path, seed)
    if backend == "transformers":
        return TransformersBackend(model, seed)
    raise ValueError(f"unknown Phase-3 backend: {backend}")
