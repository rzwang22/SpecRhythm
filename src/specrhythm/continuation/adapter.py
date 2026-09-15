"""Connect immutable CPU scheduler intents to the authoritative owner protocol.

This opt-in adapter does no GPU work and is not installed in a production
scheduler. The same owner thread publishes views and revalidates each queued
intent immediately before mutating the request's protocol state.
"""

from __future__ import annotations

from typing import Optional

from .core import DraftWork, Proposal, RollingContinuation
from .policy import EagerStepContext
from .scheduling import (
    DraftTask,
    SchedulingView,
    TargetAdmission,
    revalidate_draft_task,
    revalidate_target_admission,
)


def scheduling_view(
    machine: RollingContinuation,
    request_id: str,
    step_context: Optional[EagerStepContext] = None,
) -> SchedulingView:
    """Publish current policy and legal proposal state, after applying any switch.

    Re-evaluating policy can cancel speculative work in the core. Snapshot only
    after this operation so the view cannot advertise a cancelled continuation.
    A parent that already issued Draft work cannot issue it again, including when
    that work was cancelled and the eager switch has since been re-enabled.
    """

    machine.evaluate_eager_eligibility(request_id, step_context)
    state = machine.state(request_id)
    proposal = state.proposals.get(state.current_proposal_id)
    legal = proposal is not None and (
        proposal.prefix_version == state.committed_prefix_version
        and proposal.parent_prefix == state.committed_prefix
        and proposal.status in ("ready", "verifying")
    )
    return SchedulingView(
        request_id=state.request_id,
        owner_id=state.owner_id,
        home_cohort=state.home_cohort,
        proposal_id=state.current_proposal_id,
        prefix_version=state.committed_prefix_version,
        decision=state.eager_decision,
        is_ready=proposal is not None and proposal.status == "ready",
        is_legal=legal,
        recovery_required=state.recovery_required,
        promoted=proposal is not None and proposal.source_continuation_id is not None,
        verifying=proposal is not None and proposal.status == "verifying",
        active=not state.finished,
        continuation_pending=state.active_continuation_id is not None or any(
            continuation.parent_proposal_id == state.current_proposal_id
            for continuation in state.continuations.values()
        ),
        normal_draft_pending=state.normal_work_id is not None,
    )


def start_admission(
    machine: RollingContinuation, admission: TargetAdmission, *, normal_cohort: str
) -> Proposal:
    """Dispatch a legal, current Target intent once; stale queued work fails."""

    if admission.owner_id != machine.owner_id:
        raise ValueError("stale Target admission for another owner")
    view = scheduling_view(machine, admission.request_id)
    if not revalidate_target_admission(admission, view, normal_cohort=normal_cohort):
        raise ValueError("stale or illegal Target admission")
    return machine.start_verification(
        admission.request_id, admission.proposal_id, eager=admission.kind == "eager"
    )


def start_draft_task(machine: RollingContinuation, task: DraftTask) -> Optional[DraftWork]:
    """Dispatch one normal recovery or dependent Draft role through its owner."""

    if task.owner_id != machine.owner_id:
        raise ValueError("stale Draft task for another owner")
    view = scheduling_view(machine, task.request_id)
    if not revalidate_draft_task(task, view):
        raise ValueError("stale or illegal Draft task")
    if task.kind in ("normal_recovery", "normal_draft"):
        return machine.schedule_normal_recovery(task.request_id)
    return machine.begin_continuation(task.request_id)
