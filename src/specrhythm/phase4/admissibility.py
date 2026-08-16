"""Fail-closed request admissibility for the Phase-4B Target scheduler.

The contract is dependency-free so its semantics can be tested without vLLM.
The pinned-vLLM adapter converts each live request into an
``AdmissibilitySnapshot`` immediately before stock scheduling considers it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple


class ExecutionPhase(str, Enum):
    SETUP_PREFILL = "setup-prefill"
    TIMED_DECODE = "timed-decode"


class SchedulerRequestState(str, Enum):
    WAITING_DRAFT = "WAITING_DRAFT"
    DRAFTING = "DRAFTING"
    PROPOSAL_READY = "PROPOSAL_READY"
    VERIFY_READY = "VERIFY_READY"
    TARGET_TAIL_READY = "TARGET_TAIL_READY"
    TERMINAL = "TERMINAL"


class ScheduledOperation(str, Enum):
    PREFILL = "prefill"
    VERIFY = "verify"
    TARGET_TAIL = "legal-target-tail"
    NONE = "none"


@dataclass(frozen=True)
class ProposalEvidence:
    request_id: str
    internal_request_id: str
    prefix_version: int
    prefix_token_count: int
    prefix_token_sha256: str
    round_id: int
    proposal_token_ids: Tuple[int, ...]
    ready_timestamp_ns: int
    expires_timestamp_ns: Optional[int] = None
    consumed: bool = False


@dataclass(frozen=True)
class AdmissibilitySnapshot:
    internal_request_id: str
    stable_request_id: str
    state: SchedulerRequestState
    execution_phase: ExecutionPhase
    prefix_version: int
    round_id: int
    prefix_token_count: int
    prefix_token_sha256: str
    num_computed_tokens: int
    num_output_tokens: int
    spec_token_ids: Tuple[int, ...]
    proposal: Optional[ProposalEvidence] = None
    now_ns: int = 0


@dataclass(frozen=True)
class AdmissibilityDecision:
    admissible: bool
    operation: ScheduledOperation
    reason: str
    proposal_present: bool
    proposal_valid: bool
    proposal_ready_timestamp_ns: Optional[int]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["operation"] = self.operation.value
        return value


def decide_admissibility(snapshot: AdmissibilitySnapshot) -> AdmissibilityDecision:
    """Return the only operations allowed to reach Target execution."""

    proposal = snapshot.proposal
    present = proposal is not None
    ready_ns = proposal.ready_timestamp_ns if proposal is not None else None
    if snapshot.state is SchedulerRequestState.TERMINAL:
        return _deny("terminal request", present, False, ready_ns)
    if snapshot.execution_phase is ExecutionPhase.SETUP_PREFILL:
        return AdmissibilityDecision(
            True,
            ScheduledOperation.PREFILL,
            "setup prefill is independent of Draft readiness",
            present,
            False,
            ready_ns,
        )
    if snapshot.state in {
        SchedulerRequestState.WAITING_DRAFT,
        SchedulerRequestState.DRAFTING,
    }:
        return _deny("waiting for matching Draft proposal", present, False, ready_ns)
    if snapshot.state is SchedulerRequestState.TARGET_TAIL_READY:
        return AdmissibilityDecision(
            True,
            ScheduledOperation.TARGET_TAIL,
            "proposal-free terminal Target tail is ready",
            present,
            False,
            ready_ns,
        )
    if snapshot.state not in {
        SchedulerRequestState.PROPOSAL_READY,
        SchedulerRequestState.VERIFY_READY,
    }:
        return _deny("live decode request has no legal Target operation", present, False, ready_ns)
    errors = proposal_errors(snapshot)
    if errors:
        return _deny("invalid proposal: " + ", ".join(errors), present, False, ready_ns)
    return AdmissibilityDecision(
        True,
        ScheduledOperation.VERIFY,
        "matching unconsumed proposal is ready",
        True,
        True,
        ready_ns,
    )


def proposal_errors(snapshot: AdmissibilitySnapshot) -> list[str]:
    proposal = snapshot.proposal
    if proposal is None:
        return ["missing"]
    errors = []
    if proposal.request_id != snapshot.stable_request_id:
        errors.append("stable request ID mismatch")
    if proposal.internal_request_id != snapshot.internal_request_id:
        errors.append("internal request ID mismatch")
    if proposal.prefix_version != snapshot.prefix_version:
        errors.append("prefix version mismatch")
    if proposal.prefix_token_count != snapshot.prefix_token_count:
        errors.append("prefix token count mismatch")
    if proposal.prefix_token_sha256 != snapshot.prefix_token_sha256:
        errors.append("prefix hash mismatch")
    if proposal.round_id != snapshot.round_id:
        errors.append("round ID mismatch")
    if proposal.consumed:
        errors.append("already consumed")
    if (
        proposal.expires_timestamp_ns is not None
        and snapshot.now_ns > proposal.expires_timestamp_ns
    ):
        errors.append("expired")
    if tuple(proposal.proposal_token_ids) != tuple(snapshot.spec_token_ids):
        errors.append("scheduled proposal tokens mismatch")
    return errors


def select_admissible(
    snapshots: Sequence[AdmissibilitySnapshot], *, token_budget: int
) -> tuple[list[str], int]:
    """Small construction model proving a blocked head does not consume budget."""

    if token_budget < 0:
        raise ValueError("token_budget must be non-negative")
    selected = []
    remaining = token_budget
    for snapshot in snapshots:
        decision = decide_admissibility(snapshot)
        if not decision.admissible or remaining == 0:
            continue
        selected.append(snapshot.stable_request_id)
        remaining -= 1
    return selected, remaining


def decision_event(
    snapshot: AdmissibilitySnapshot,
    decision: AdmissibilityDecision,
    *,
    cycle_id: int,
    scheduler_step: int,
    scheduled: bool,
    target_input_positions: Sequence[int] = (),
    target_forward_start_ns: Optional[int] = None,
    target_forward_end_ns: Optional[int] = None,
) -> dict[str, Any]:
    return {
        "schema_version": "specrhythm.phase4b-scheduler-admissibility.v1",
        "cycle_id": cycle_id,
        "scheduler_step": scheduler_step,
        "internal_request_id": snapshot.internal_request_id,
        "request_id": snapshot.stable_request_id,
        "specrhythm_state": snapshot.state.value,
        "execution_phase": snapshot.execution_phase.value,
        "prefix_version": snapshot.prefix_version,
        "round_id": snapshot.round_id,
        "proposal_present": decision.proposal_present,
        "proposal_valid": decision.proposal_valid,
        "proposal_ready_timestamp_ns": decision.proposal_ready_timestamp_ns,
        "admissible": decision.admissible,
        "inadmissible_reason": None if decision.admissible else decision.reason,
        "scheduled": scheduled,
        "scheduled_operation": (
            decision.operation.value if scheduled else ScheduledOperation.NONE.value
        ),
        "num_computed_tokens": snapshot.num_computed_tokens,
        "num_output_tokens": snapshot.num_output_tokens,
        "spec_token_ids": list(snapshot.spec_token_ids),
        "target_input_token_positions": list(target_input_positions),
        "target_forward_start_ns": target_forward_start_ns,
        "target_forward_end_ns": target_forward_end_ns,
        "forward_timing_correlation": "verification-events-by-request-and-round",
    }


def validate_admissibility_events(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors = []
    for index, row in enumerate(rows):
        prefix = f"row {index}"
        if row.get("schema_version") != "specrhythm.phase4b-scheduler-admissibility.v1":
            errors.append(f"{prefix}: unsupported schema")
        if row.get("scheduled") and not row.get("admissible"):
            errors.append(f"{prefix}: inadmissible request was scheduled")
        if row.get("scheduled") and row.get("scheduled_operation") == "none":
            errors.append(f"{prefix}: scheduled request has no operation")
        if not row.get("scheduled") and row.get("scheduled_operation") != "none":
            errors.append(f"{prefix}: unscheduled request names an operation")
        if row.get("scheduled_operation") == "verify" and not row.get("proposal_valid"):
            errors.append(f"{prefix}: verify lacks a valid proposal")
    return errors


def _deny(
    reason: str,
    proposal_present: bool,
    proposal_valid: bool,
    ready_timestamp_ns: Optional[int],
) -> AdmissibilityDecision:
    return AdmissibilityDecision(
        False,
        ScheduledOperation.NONE,
        reason,
        proposal_present,
        proposal_valid,
        ready_timestamp_ns,
    )
