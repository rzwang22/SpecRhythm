"""Dependency-free contracts for disaggregated serving integration."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable


def _token_tuple(name: str, values: Sequence[int], *, allow_empty: bool = True) -> Tuple[int, ...]:
    result = tuple(values)
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in result):
        raise ValueError(f"{name} must contain non-negative integer token IDs")
    return result


def _nonnegative_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class GreedySamplingContract:
    """Frozen deterministic sampling semantics for the stock-engine smoke."""

    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    n: int = 1
    best_of: int = 1
    seed: int = 1664

    def __post_init__(self) -> None:
        if self.do_sample or self.temperature != 0.0 or self.top_p != 1.0:
            raise ValueError("Phase-4 bring-up requires greedy temperature=0/top_p=1")
        if self.n != 1 or self.best_of != 1:
            raise ValueError("Phase-4 bring-up requires exactly one output")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("greedy seed must be an integer")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateNode:
    stable_node_id: str
    parent_id: Optional[str]
    depth: int
    token_id: int
    local_probability: float
    path_probability: float

    def __post_init__(self) -> None:
        if not self.stable_node_id or self.parent_id == "":
            raise ValueError("candidate IDs must be non-empty (or null for a root child)")
        if _nonnegative_int("depth", self.depth) < 1:
            raise ValueError("candidate depth must be positive")
        _nonnegative_int("token_id", self.token_id)
        for name, value in (
            ("local_probability", self.local_probability),
            ("path_probability", self.path_probability),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class CandidateBatch:
    request_id: str
    prefix_epoch: int
    committed_prefix_token_ids: Tuple[int, ...]
    nodes: Tuple[CandidateNode, ...]
    created_monotonic_ns: int
    adapter_backend: str
    fake_data: bool = False

    def __post_init__(self) -> None:
        if not self.request_id or not self.adapter_backend:
            raise ValueError("candidate batch identity/backend must not be empty")
        _nonnegative_int("prefix_epoch", self.prefix_epoch)
        _token_tuple("committed_prefix_token_ids", self.committed_prefix_token_ids)
        _nonnegative_int("created_monotonic_ns", self.created_monotonic_ns)
        ids = [node.stable_node_id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate stable_node_id values must be unique")
        by_id = {node.stable_node_id: node for node in self.nodes}
        for node in self.nodes:
            if node.parent_id is None:
                if node.depth != 1:
                    raise ValueError("root-child candidate depth must be one")
            else:
                parent = by_id.get(node.parent_id)
                if parent is None or parent.depth + 1 != node.depth:
                    raise ValueError("candidate batch must be prefix closed")

    @property
    def drafted_nodes(self) -> int:
        return len(self.nodes)


@dataclass(frozen=True)
class VerificationBatch:
    request_id: str
    prefix_epoch: int
    committed_prefix_token_ids: Tuple[int, ...]
    candidate_nodes: Tuple[CandidateNode, ...]
    selected_node_ids: Tuple[str, ...]
    created_monotonic_ns: int
    source_backend: str

    def __post_init__(self) -> None:
        if not self.request_id or not self.source_backend:
            raise ValueError("verification identity/backend must not be empty")
        _nonnegative_int("prefix_epoch", self.prefix_epoch)
        _token_tuple("committed_prefix_token_ids", self.committed_prefix_token_ids)
        _nonnegative_int("created_monotonic_ns", self.created_monotonic_ns)
        by_id = {node.stable_node_id: node for node in self.candidate_nodes}
        if len(by_id) != len(self.candidate_nodes):
            raise ValueError("verification candidates must have unique IDs")
        if len(set(self.selected_node_ids)) != len(self.selected_node_ids):
            raise ValueError("selected candidate IDs must not repeat")
        selected = set(self.selected_node_ids)
        if not selected.issubset(by_id):
            raise ValueError("verification selected an unknown candidate")
        for node_id in selected:
            parent = by_id[node_id].parent_id
            if parent is not None and parent not in selected:
                raise ValueError("selected verification candidates must be prefix closed")

    @classmethod
    def from_candidates(
        cls, batch: CandidateBatch, selected_node_ids: Sequence[str], timestamp_ns: int
    ) -> "VerificationBatch":
        return cls(
            request_id=batch.request_id,
            prefix_epoch=batch.prefix_epoch,
            committed_prefix_token_ids=batch.committed_prefix_token_ids,
            candidate_nodes=batch.nodes,
            selected_node_ids=tuple(selected_node_ids),
            created_monotonic_ns=timestamp_ns,
            source_backend=batch.adapter_backend,
        )


@dataclass(frozen=True)
class VerificationResult:
    request_id: str
    prefix_epoch: int
    verified_node_ids: Tuple[str, ...]
    accepted_node_ids: Tuple[str, ...]
    committed_token_ids: Tuple[int, ...]
    target_bonus_token_ids: Tuple[int, ...]
    finished: bool
    completed_monotonic_ns: int
    target_backend: str
    fake_data: bool = False

    def __post_init__(self) -> None:
        if not self.request_id or not self.target_backend:
            raise ValueError("verification result identity/backend must not be empty")
        _nonnegative_int("prefix_epoch", self.prefix_epoch)
        _nonnegative_int("completed_monotonic_ns", self.completed_monotonic_ns)
        if len(set(self.verified_node_ids)) != len(self.verified_node_ids):
            raise ValueError("verified node IDs must not repeat")
        if len(set(self.accepted_node_ids)) != len(self.accepted_node_ids):
            raise ValueError("accepted node IDs must not repeat")
        if not set(self.accepted_node_ids).issubset(self.verified_node_ids):
            raise ValueError("accepted nodes must have been verified")
        committed = _token_tuple("committed_token_ids", self.committed_token_ids)
        bonus = _token_tuple("target_bonus_token_ids", self.target_bonus_token_ids)
        if len(bonus) > 1:
            raise ValueError("one verification batch can commit at most one target bonus token")
        if len(committed) != len(self.accepted_node_ids) + len(bonus):
            raise ValueError("committed tokens must equal accepted candidates plus target bonus")

    @property
    def accounting(self) -> dict[str, int]:
        return {
            "verified_nodes": len(self.verified_node_ids),
            "accepted_nodes": len(self.accepted_node_ids),
            "target_bonus_tokens": len(self.target_bonus_token_ids),
            "committed_tokens": len(self.committed_token_ids),
        }


@dataclass(frozen=True)
class RequestState:
    request_id: str
    prompt_token_ids: Tuple[int, ...]
    committed_token_ids: Tuple[int, ...] = ()
    prefix_epoch: int = 0
    finished: bool = False
    last_event_sequence: int = -1

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        _token_tuple("prompt_token_ids", self.prompt_token_ids, allow_empty=False)
        _token_tuple("committed_token_ids", self.committed_token_ids)
        _nonnegative_int("prefix_epoch", self.prefix_epoch)
        if not isinstance(self.last_event_sequence, int) or self.last_event_sequence < -1:
            raise ValueError("last_event_sequence must be at least -1")

    def apply(self, result: VerificationResult, event_sequence: int) -> "RequestState":
        if self.finished:
            raise ValueError("a finished request cannot consume another verification result")
        if result.request_id != self.request_id or result.prefix_epoch != self.prefix_epoch:
            raise ValueError("verification result does not match request identity/prefix epoch")
        if event_sequence <= self.last_event_sequence:
            raise ValueError("engine event sequence must increase monotonically")
        return replace(
            self,
            committed_token_ids=self.committed_token_ids + result.committed_token_ids,
            prefix_epoch=self.prefix_epoch + len(result.committed_token_ids),
            finished=result.finished,
            last_event_sequence=event_sequence,
        )


@dataclass(frozen=True)
class EngineEvent:
    sequence: int
    monotonic_ns: int
    request_id: str
    engine_role: str
    event_type: str
    counters: Mapping[str, int]

    def __post_init__(self) -> None:
        _nonnegative_int("event sequence", self.sequence)
        _nonnegative_int("event monotonic_ns", self.monotonic_ns)
        if not self.request_id or self.engine_role not in {"draft", "target"}:
            raise ValueError("engine event has an invalid request or role")
        if not self.event_type:
            raise ValueError("engine event type must not be empty")
        for name, value in self.counters.items():
            _nonnegative_int(f"event counter {name}", value)


class MonotonicEventClock:
    """Issue strictly increasing process-local event IDs and timestamps."""

    def __init__(self) -> None:
        self._sequence = -1
        self._last_timestamp = -1

    def next(
        self, request_id: str, engine_role: str, event_type: str, **counters: int
    ) -> EngineEvent:
        self._sequence += 1
        now = time.monotonic_ns()
        now = max(now, self._last_timestamp + 1)
        self._last_timestamp = now
        return EngineEvent(self._sequence, now, request_id, engine_role, event_type, counters)


@runtime_checkable
class DraftEngineAdapter(Protocol):
    backend_name: str
    fake_backend: bool

    def propose(self, request: RequestState, candidate_budget: int) -> CandidateBatch: ...


@runtime_checkable
class TargetEngineAdapter(Protocol):
    backend_name: str
    fake_backend: bool

    def verify(self, request: RequestState, batch: VerificationBatch) -> VerificationResult: ...
