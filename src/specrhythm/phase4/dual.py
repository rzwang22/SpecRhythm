"""Dependency-free Phase-4B linear Dual-Batch contracts.

This module is the authority for request/proposal identity and lifecycle
semantics.  GPU adapters may observe or execute transitions, but they must not
weaken them.  In particular, this is linear speculative decoding: a request can
own at most one proposal and Draft never advances through an unverified one.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Deque, Iterable, Mapping, Optional, Sequence, Tuple

from specrhythm.phase4.serial import AcceptanceDecision, greedy_acceptance, token_prefix_hash

DUAL_PROTOCOL_VERSION = "specrhythm.dual-batch.v1"


class RequestState(str, Enum):
    BOOTSTRAP = "BOOTSTRAP"
    DRAFT_READY = "DRAFT_READY"
    DRAFTING = "DRAFTING"
    PROPOSAL_READY = "PROPOSAL_READY"
    VERIFY_READY = "VERIFY_READY"
    VERIFYING = "VERIFYING"
    COMMITTING = "COMMITTING"
    DRAFT_SYNC = "DRAFT_SYNC"
    FINISHED = "FINISHED"
    FAILED = "FAILED"


_LEGAL_TRANSITIONS = {
    RequestState.BOOTSTRAP: {RequestState.DRAFT_READY, RequestState.FINISHED},
    RequestState.DRAFT_READY: {
        RequestState.DRAFTING,
        RequestState.VERIFY_READY,  # proposal-free one-token Target tail
        RequestState.FINISHED,
    },
    RequestState.DRAFTING: {
        RequestState.PROPOSAL_READY,
        RequestState.DRAFT_READY,  # proposal-free one-token Target tail
        RequestState.FAILED,
    },
    RequestState.PROPOSAL_READY: {
        RequestState.VERIFY_READY,
        RequestState.DRAFT_READY,  # fail-closed stale discard
        RequestState.FAILED,
    },
    RequestState.VERIFY_READY: {
        RequestState.VERIFYING,
        RequestState.DRAFT_READY,  # fail-closed stale discard
        RequestState.FAILED,
    },
    RequestState.VERIFYING: {RequestState.COMMITTING, RequestState.FAILED},
    RequestState.COMMITTING: {
        RequestState.DRAFT_SYNC,
        RequestState.FINISHED,
        RequestState.FAILED,
    },
    RequestState.DRAFT_SYNC: {
        RequestState.DRAFT_READY,
        RequestState.FINISHED,
        RequestState.FAILED,
    },
    RequestState.FINISHED: set(),
    RequestState.FAILED: set(),
}


def proposal_identity(
    request_id: str, round_id: int, prefix_version: int, proposal_token_ids: Sequence[int]
) -> str:
    value = {
        "request_id": request_id,
        "round_id": round_id,
        "prefix_version": prefix_version,
        "proposal_token_ids": list(proposal_token_ids),
    }
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _nonnegative(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class DualProposal:
    request_id: str
    round_id: int
    proposal_id: str
    prefix_version: int
    prefix_token_count: int
    prefix_token_sha256: str
    draft_kv_length_before: int
    draft_kv_length_after: int
    proposal_token_ids: Tuple[int, ...]
    created_timestamp_ns: int
    draft_start_ns: int
    draft_end_ns: int

    def __post_init__(self) -> None:
        if not self.request_id or not self.proposal_id or not self.prefix_token_sha256:
            raise ValueError("proposal identity and prefix evidence must not be empty")
        for name in (
            "round_id",
            "prefix_version",
            "prefix_token_count",
            "draft_kv_length_before",
            "draft_kv_length_after",
            "created_timestamp_ns",
            "draft_start_ns",
            "draft_end_ns",
        ):
            _nonnegative(name, getattr(self, name))
        tokens = tuple(self.proposal_token_ids)
        token_prefix_hash(tokens)
        if not tokens or len(tokens) > 4:
            raise ValueError("linear Dual-Batch proposal length must be in [1, 4]")
        if self.draft_kv_length_before != self.prefix_token_count:
            raise ValueError("Draft KV before proposal does not match parent prefix")
        if self.draft_kv_length_after != self.prefix_token_count + len(tokens):
            raise ValueError("Draft KV after proposal does not account for proposal tokens")
        if self.draft_end_ns < self.draft_start_ns:
            raise ValueError("Draft interval is reversed")
        expected = proposal_identity(
            self.request_id, self.round_id, self.prefix_version, tokens
        )
        if self.proposal_id != expected:
            raise ValueError("proposal_id is not the canonical proposal identity")

    @property
    def proposal_length(self) -> int:
        return len(self.proposal_token_ids)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["proposal_token_ids"] = list(self.proposal_token_ids)
        value["proposal_length"] = self.proposal_length
        value["protocol_version"] = DUAL_PROTOCOL_VERSION
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DualProposal":
        if value.get("protocol_version", DUAL_PROTOCOL_VERSION) != DUAL_PROTOCOL_VERSION:
            raise ValueError("unsupported Dual-Batch proposal protocol")
        return cls(
            request_id=str(value.get("request_id", "")),
            round_id=value.get("round_id", -1),
            proposal_id=str(value.get("proposal_id", "")),
            prefix_version=value.get("prefix_version", -1),
            prefix_token_count=value.get("prefix_token_count", -1),
            prefix_token_sha256=str(value.get("prefix_token_sha256", "")),
            draft_kv_length_before=value.get("draft_kv_length_before", -1),
            draft_kv_length_after=value.get("draft_kv_length_after", -1),
            proposal_token_ids=tuple(value.get("proposal_token_ids", ())),
            created_timestamp_ns=value.get("created_timestamp_ns", -1),
            draft_start_ns=value.get("draft_start_ns", -1),
            draft_end_ns=value.get("draft_end_ns", -1),
        )


@dataclass
class DualRequest:
    request_id: str
    committed_token_ids: Tuple[int, ...]
    maximum_generated_tokens: int
    generated_tokens: int = 0
    state: RequestState = RequestState.BOOTSTRAP
    prefix_version: int = 0
    next_round_id: int = 0
    proposal: Optional[DualProposal] = None
    last_error: Optional[str] = None

    @property
    def prefix_sha256(self) -> str:
        return token_prefix_hash(self.committed_token_ids)

    @property
    def committed_token_count(self) -> int:
        return len(self.committed_token_ids)

    def transition(self, destination: RequestState) -> None:
        if destination not in _LEGAL_TRANSITIONS[self.state]:
            raise ValueError(f"illegal request transition {self.state.value}->{destination.value}")
        self.state = destination

    def finish_bootstrap(self, committed_delta: Sequence[int], *, terminal: bool) -> None:
        if self.state is not RequestState.BOOTSTRAP:
            raise ValueError("bootstrap can only finish once")
        delta = tuple(committed_delta)
        token_prefix_hash(delta)
        if not delta:
            raise ValueError("bootstrap must commit at least one Target token")
        self.committed_token_ids += delta
        self.generated_tokens += len(delta)
        self.prefix_version += 1
        self.transition(RequestState.FINISHED if terminal else RequestState.DRAFT_READY)

    def start_draft(self) -> None:
        if self.proposal is not None:
            raise ValueError("request already owns an unverified proposal")
        self.transition(RequestState.DRAFTING)

    def publish_proposal(self, proposal: DualProposal) -> None:
        if self.state is not RequestState.DRAFTING:
            raise ValueError("proposal can only be published by DRAFTING request")
        self._validate_parent(proposal)
        if proposal.round_id != self.next_round_id:
            raise ValueError("proposal round is stale, duplicate, or out of order")
        self.proposal = proposal
        self.transition(RequestState.PROPOSAL_READY)

    def mark_verify_ready(self) -> None:
        if self.proposal is None:
            raise ValueError("request has no proposal to verify")
        self.transition(RequestState.VERIFY_READY)

    def start_verify(self) -> DualProposal:
        if self.proposal is None:
            raise ValueError("request has no proposal to verify")
        self._validate_parent(self.proposal)
        self.transition(RequestState.VERIFYING)
        return self.proposal

    def commit(
        self, committed_delta: Sequence[int], *, terminal: bool
    ) -> AcceptanceDecision:
        if self.state is not RequestState.VERIFYING or self.proposal is None:
            raise ValueError("only VERIFYING request with a proposal can commit")
        self.transition(RequestState.COMMITTING)
        decision = greedy_acceptance(
            self.proposal.proposal_token_ids, committed_delta, terminal=terminal
        )
        self.committed_token_ids += decision.committed_token_ids
        self.generated_tokens += len(decision.committed_token_ids)
        self.prefix_version += 1
        self.next_round_id += 1
        if self.generated_tokens > self.maximum_generated_tokens:
            raise ValueError("committed output exceeds maximum token budget")
        self.transition(RequestState.FINISHED if terminal else RequestState.DRAFT_SYNC)
        return decision

    def complete_draft_sync(self, logical_draft_kv_length: int) -> None:
        if self.state is not RequestState.DRAFT_SYNC or self.proposal is None:
            raise ValueError("Draft synchronization has no committed proposal")
        if logical_draft_kv_length != self.committed_token_count:
            raise ValueError("Draft KV does not exactly match committed prefix")
        self.proposal = None
        self.transition(RequestState.DRAFT_READY)

    def discard_stale(self, reason: str) -> DualProposal:
        if self.state not in {RequestState.PROPOSAL_READY, RequestState.VERIFY_READY}:
            raise ValueError("only an unverified ready proposal can be discarded")
        if self.proposal is None:
            raise ValueError("stale discard has no proposal")
        proposal = self.proposal
        self.proposal = None
        self.last_error = reason
        self.transition(RequestState.DRAFT_READY)
        return proposal

    def _validate_parent(self, proposal: DualProposal) -> None:
        errors = []
        if proposal.request_id != self.request_id:
            errors.append("request_id")
        if proposal.prefix_version != self.prefix_version:
            errors.append("prefix_version")
        if proposal.prefix_token_count != self.committed_token_count:
            errors.append("prefix_token_count")
        if proposal.prefix_token_sha256 != self.prefix_sha256:
            errors.append("prefix_token_sha256")
        if errors:
            raise ValueError("stale proposal parent mismatch: " + ", ".join(errors))


class ProposalReadyQueue:
    """Deterministic, non-blocking ready handoff with stale rejection."""

    def __init__(self) -> None:
        self._order: Deque[str] = deque()
        self._values: dict[str, DualProposal] = {}

    def publish(self, proposal: DualProposal) -> None:
        if proposal.request_id in self._values:
            raise ValueError("request already has a ready proposal")
        self._values[proposal.request_id] = proposal
        self._order.append(proposal.request_id)

    def take(
        self,
        requests: Mapping[str, DualRequest],
        *,
        limit: int,
    ) -> tuple[list[DualProposal], list[dict[str, str]]]:
        if limit < 0:
            raise ValueError("ready queue limit must be non-negative")
        ready: list[DualProposal] = []
        stale: list[dict[str, str]] = []
        while self._order and len(ready) < limit:
            request_id = self._order.popleft()
            proposal = self._values.pop(request_id)
            request = requests.get(request_id)
            if request is None or request.state in {
                RequestState.FINISHED,
                RequestState.FAILED,
            }:
                stale.append({"proposal_id": proposal.proposal_id, "reason": "terminal"})
                continue
            try:
                request._validate_parent(proposal)
            except ValueError as error:
                stale.append({"proposal_id": proposal.proposal_id, "reason": str(error)})
                continue
            ready.append(proposal)
        return ready, stale

    def __len__(self) -> int:
        return len(self._values)


@dataclass(frozen=True)
class DualCycle:
    cycle_id: int
    draft_microbatch_id: Optional[str]
    verify_microbatch_id: Optional[str]
    draft_request_ids: Tuple[str, ...]
    verify_request_ids: Tuple[str, ...]
    draft_start_ns: Optional[int]
    draft_end_ns: Optional[int]
    verify_start_ns: Optional[int]
    verify_end_ns: Optional[int]
    commit_start_ns: Optional[int]
    commit_end_ns: Optional[int]

    def __post_init__(self) -> None:
        _nonnegative("cycle_id", self.cycle_id)
        if set(self.draft_request_ids) & set(self.verify_request_ids):
            raise ValueError("Draft and Verify microbatches are not disjoint")
        for start_name, end_name in (
            ("draft_start_ns", "draft_end_ns"),
            ("verify_start_ns", "verify_end_ns"),
            ("commit_start_ns", "commit_end_ns"),
        ):
            start = getattr(self, start_name)
            end = getattr(self, end_name)
            if (start is None) != (end is None):
                raise ValueError(f"{start_name}/{end_name} must both be present or absent")
            if start is not None and (start < 0 or end < start):
                raise ValueError(f"invalid {start_name}/{end_name} interval")

    @property
    def overlap_interval(self) -> Optional[Tuple[int, int]]:
        if self.draft_start_ns is None or self.verify_start_ns is None:
            return None
        start = max(self.draft_start_ns, self.verify_start_ns)
        end = min(self.draft_end_ns, self.verify_end_ns)  # type: ignore[arg-type]
        return (start, end) if end > start else None

    @property
    def overlap_duration_ns(self) -> int:
        interval = self.overlap_interval
        return 0 if interval is None else interval[1] - interval[0]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["draft_request_ids"] = list(self.draft_request_ids)
        value["verify_request_ids"] = list(self.verify_request_ids)
        interval = self.overlap_interval
        value["overlap_start_ns"] = interval[0] if interval else None
        value["overlap_end_ns"] = interval[1] if interval else None
        value["overlap_duration_ns"] = self.overlap_duration_ns
        return value


def form_dynamic_microbatches(
    requests: Iterable[DualRequest], *, microbatch_size: int
) -> tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Return deterministic FIFO Draft/Verify sets without delaying ready work."""

    if microbatch_size < 1:
        raise ValueError("microbatch_size must be positive")
    ordered = list(requests)
    verify = tuple(
        request.request_id
        for request in ordered
        if request.state in {RequestState.PROPOSAL_READY, RequestState.VERIFY_READY}
    )[:microbatch_size]
    verify_set = set(verify)
    draft = tuple(
        request.request_id
        for request in ordered
        if request.request_id not in verify_set and request.state is RequestState.DRAFT_READY
    )[:microbatch_size]
    return draft, verify


def validate_cycle_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors = []
    cycle_ids = set()
    for row in rows:
        cycle_id = row.get("cycle_id")
        if cycle_id in cycle_ids:
            errors.append(f"duplicate cycle_id: {cycle_id}")
        cycle_ids.add(cycle_id)
        draft = set(row.get("draft_request_ids", ()))
        verify = set(row.get("verify_request_ids", ()))
        if draft & verify:
            errors.append(f"cycle {cycle_id}: Draft/Verify request sets overlap")
        overlap = row.get("overlap_duration_ns", 0)
        if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0:
            errors.append(f"cycle {cycle_id}: invalid overlap duration")
        for key in ("draft_start_ns", "draft_end_ns", "verify_start_ns", "verify_end_ns"):
            value = row.get(key)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                errors.append(f"cycle {cycle_id}: invalid {key}")
    return errors


def finite_positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value > 0
    )


def run_dual_contract_dry_run() -> dict[str, Any]:
    """Exercise a dynamic A/B swap without importing CUDA, Torch, or vLLM."""

    requests = [DualRequest("A", (1, 2), 8), DualRequest("B", (3, 4), 8)]
    for request, bootstrap in zip(requests, (5, 6)):
        request.finish_bootstrap((bootstrap,), terminal=False)
    first = requests[0]
    first.start_draft()
    proposal = DualProposal(
        request_id="A",
        round_id=0,
        proposal_id=proposal_identity("A", 0, 1, (10, 11)),
        prefix_version=1,
        prefix_token_count=3,
        prefix_token_sha256=first.prefix_sha256,
        draft_kv_length_before=3,
        draft_kv_length_after=5,
        proposal_token_ids=(10, 11),
        created_timestamp_ns=2,
        draft_start_ns=1,
        draft_end_ns=2,
    )
    first.publish_proposal(proposal)
    first_sets = form_dynamic_microbatches(requests, microbatch_size=1)
    first.mark_verify_ready()
    first.start_verify()
    first.commit((10, 11, 12), terminal=False)
    first.complete_draft_sync(first.committed_token_count)
    second = requests[1]
    second.start_draft()
    second_proposal = DualProposal(
        request_id="B",
        round_id=0,
        proposal_id=proposal_identity("B", 0, 1, (20,)),
        prefix_version=1,
        prefix_token_count=3,
        prefix_token_sha256=second.prefix_sha256,
        draft_kv_length_before=3,
        draft_kv_length_after=4,
        proposal_token_ids=(20,),
        created_timestamp_ns=4,
        draft_start_ns=3,
        draft_end_ns=4,
    )
    second.publish_proposal(second_proposal)
    second_sets = form_dynamic_microbatches(requests, microbatch_size=1)
    return {
        "schema_version": "specrhythm.phase4b-contract-dry-run.v1",
        "gpu_execution_performed": False,
        "performance_result": False,
        "first_cycle": {
            "draft_request_ids": list(first_sets[0]),
            "verify_request_ids": list(first_sets[1]),
        },
        "second_cycle": {
            "draft_request_ids": list(second_sets[0]),
            "verify_request_ids": list(second_sets[1]),
        },
        "dynamic_swap": first_sets == (("B",), ("A",))
        and second_sets == (("A",), ("B",)),
        "target_only_default_behavior_changed": False,
        "vllm_imported": False,
    }
