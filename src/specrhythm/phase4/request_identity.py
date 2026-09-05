"""Fail-closed mapping between vLLM-internal and frozen workload identities.

vLLM is free to decorate the request ID supplied by a caller.  Phase-4
artifacts instead use the immutable workload ID.  The only identity bridge is
the frozen prompt-token prefix; internal request-ID strings are opaque.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple


class FrozenPromptIdentityMap:
    """Bind opaque engine request IDs to unique frozen prompt-token prefixes."""

    def __init__(self, stable_prompts: Mapping[str, Sequence[int]]) -> None:
        prompts: dict[str, Tuple[int, ...]] = {}
        for stable_id_value, prompt_value in stable_prompts.items():
            stable_id = str(stable_id_value).strip()
            prompt = tuple(int(item) for item in prompt_value)
            if not stable_id or not prompt:
                raise ValueError("stable request IDs and frozen prompts must be non-empty")
            if stable_id in prompts:
                raise ValueError(f"duplicate stable request ID: {stable_id}")
            prompts[stable_id] = prompt
        if not prompts:
            raise ValueError("at least one frozen request prompt is required")
        if len(set(prompts.values())) != len(prompts):
            raise ValueError("stable request mapping requires unique frozen prompt token IDs")
        self.stable_prompts = prompts
        self.internal_to_stable: dict[str, str] = {}
        self.stable_to_internal: dict[str, str] = {}

    @classmethod
    def from_definitions(cls, definitions: Sequence[Any]) -> "FrozenPromptIdentityMap":
        return cls(
            {
                str(definition.request_id): tuple(definition.prompt_token_ids)
                for definition in definitions
            }
        )

    def match(self, physical_token_prefix: Sequence[int]) -> str:
        """Return the only stable prompt that prefixes the physical token row."""

        tokens = tuple(int(item) for item in physical_token_prefix)
        matches = [
            stable_id
            for stable_id, prompt in self.stable_prompts.items()
            if tokens[: len(prompt)] == prompt
        ]
        if not matches:
            raise RuntimeError("Target token row matches no frozen workload prompt")
        if len(matches) != 1:
            raise RuntimeError("Target token row ambiguously matches frozen workload prompts")
        return matches[0]

    def bind(self, internal_request_id: str, physical_token_prefix: Sequence[int]) -> str:
        """Match and permanently bind one opaque internal ID to one stable ID."""

        internal_id = str(internal_request_id)
        if not internal_id:
            raise RuntimeError("vLLM internal request ID is empty")
        stable_id = self.match(physical_token_prefix)
        previous_stable = self.internal_to_stable.get(internal_id)
        if previous_stable is not None and previous_stable != stable_id:
            raise RuntimeError("vLLM internal request ID changed stable prompt identity")
        previous_internal = self.stable_to_internal.get(stable_id)
        if previous_internal is not None and previous_internal != internal_id:
            raise RuntimeError("multiple vLLM internal request IDs alias one stable request")
        self.internal_to_stable[internal_id] = stable_id
        self.stable_to_internal[stable_id] = internal_id
        return stable_id

    def stable_id(self, internal_request_id: str) -> str:
        internal_id = str(internal_request_id)
        try:
            return self.internal_to_stable[internal_id]
        except KeyError as error:
            raise RuntimeError(
                f"vLLM internal request ID has no frozen prompt binding: {internal_id}"
            ) from error

    def internal_id(self, stable_request_id: str) -> str:
        stable_id = str(stable_request_id)
        try:
            return self.stable_to_internal[stable_id]
        except KeyError as error:
            raise RuntimeError(
                f"stable request has no live vLLM identity binding: {stable_id}"
            ) from error


def resolve_stable_ready_request(
    stable_request_id: str,
    identity: FrozenPromptIdentityMap,
    internal_requests: Mapping[str, Any],
) -> tuple[str, Any]:
    """Resolve Draft-owned stable metadata to exactly one vLLM-owned request."""

    stable_id = str(stable_request_id)
    internal_id = identity.internal_id(stable_id)
    request = internal_requests.get(internal_id)
    if request is None:
        raise RuntimeError(
            f"ready proposal for stable request {stable_id} has no mapped vLLM request"
        )
    if str(getattr(request, "request_id", "")) != internal_id:
        raise RuntimeError("vLLM request table key disagrees with internal request identity")
    return internal_id, request


def resolve_historical_ready_request(
    stable_request_id: str,
    identity: FrozenPromptIdentityMap,
    internal_requests: Mapping[str, Any],
) -> tuple[str, Any]:
    """Resolve an async result, returning None only for a valid retired binding.

    Bindings outlive stock vLLM requests. Absence from the live table is therefore
    expected for late Dual results, but an unknown or inconsistent binding is not.
    The generic live-only resolver deliberately retains its strict semantics.
    """

    if not isinstance(stable_request_id, str) or not stable_request_id.strip():
        raise RuntimeError("ready result stable request ID must be a non-empty string")
    internal_id = identity.internal_id(stable_request_id)
    if (
        stable_request_id not in identity.stable_prompts
        or not isinstance(internal_id, str)
        or not internal_id
        or identity.internal_to_stable.get(internal_id) != stable_request_id
        or sum(value == internal_id for value in identity.stable_to_internal.values()) != 1
        or sum(value == stable_request_id for value in identity.internal_to_stable.values()) != 1
    ):
        raise RuntimeError("ready result historical identity binding is inconsistent")
    if internal_id not in internal_requests:
        return internal_id, None
    return resolve_stable_ready_request(stable_request_id, identity, internal_requests)
