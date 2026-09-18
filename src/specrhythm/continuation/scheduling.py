"""Pure CPU scheduling contracts; no Target or Draft runtime ownership is copied.

The owner publishes immutable views of its authoritative request state. These
selectors return versioned work intents, not READY transitions or GPU launches.
The owner must revalidate an intent immediately before dispatch. Once dispatched,
an in-flight Target verification is resolved by the continuation core even if
the admission switch subsequently changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from .policy import EagerDecision


class SchedulingConflictError(ValueError):
    """One stable identity appeared with conflicting owner snapshots."""


@dataclass(frozen=True)
class SchedulingView:
    request_id: str
    owner_id: str
    home_cohort: str
    proposal_id: Optional[str]
    prefix_version: int
    decision: EagerDecision
    is_ready: bool = False
    is_legal: bool = False
    recovery_required: bool = False
    promoted: bool = False
    verifying: bool = False
    active: bool = True
    continuation_pending: bool = False
    normal_draft_pending: bool = False

    def __post_init__(self) -> None:
        if not self.request_id or not self.owner_id or not self.home_cohort:
            raise ValueError("request_id, owner_id, and home_cohort must be nonempty")
        if self.prefix_version < 0:
            raise ValueError("prefix_version must be nonnegative")


@dataclass(frozen=True)
class TargetAdmission:
    """A proposed Target role for an existing request, not a second request."""

    request_id: str
    owner_id: str
    home_cohort: str
    proposal_id: str
    prefix_version: int
    decision_version: int
    kind: str


@dataclass(frozen=True)
class DraftTask:
    """For eager work proposal_id is the parent; recovery has no old proposal."""

    request_id: str
    owner_id: str
    home_cohort: str
    proposal_id: Optional[str]
    prefix_version: int
    decision_version: int
    kind: str


def _unique_views(views: Iterable[SchedulingView]) -> Tuple[SchedulingView, ...]:
    by_request = {}
    for view in views:
        previous = by_request.get(view.request_id)
        if previous is not None and previous != view:
            raise SchedulingConflictError(
                f"conflicting scheduling snapshots for request {view.request_id!r}"
            )
        by_request[view.request_id] = view
    return tuple(by_request.values())


def _check_capacity(capacity: int) -> None:
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 0:
        raise ValueError("capacity must be a nonnegative integer")


def _target_legal(view: SchedulingView) -> bool:
    return (
        view.active
        and view.is_ready
        and view.is_legal
        and view.proposal_id is not None
        and not view.verifying
        and not view.recovery_required
    )


def select_target_admissions(
    normal_views: Iterable[SchedulingView],
    eager_views: Iterable[SchedulingView] = (),
    *,
    normal_cohort: str,
    capacity: int,
    owner_id: Optional[str] = None,
) -> Tuple[TargetAdmission, ...]:
    """Offer promoted eager proposals next, across A/B; fill from the normal cohort.

    Disabling eager moves a promoted, legal proposal back to its home cohort's
    ordinary scheduling opportunity. Capacity deferral does not mutate a view or
    its eligibility. Conflicting duplicate views fail even when capacity is zero;
    silently choosing a stale proposal would break the owner protocol.
    """

    _check_capacity(capacity)
    views = _unique_views((*normal_views, *eager_views))
    eager: List[Tuple[SchedulingView, str]] = []
    normal: List[Tuple[SchedulingView, str]] = []
    for view in views:
        if (owner_id is not None and view.owner_id != owner_id) or not _target_legal(view):
            continue
        if view.promoted and view.decision.can_admit:
            eager.append((view, "eager"))
        elif view.home_cohort == normal_cohort:
            normal.append((view, "normal"))
    return tuple(
        TargetAdmission(
            request_id=view.request_id,
            owner_id=view.owner_id,
            home_cohort=view.home_cohort,
            proposal_id=view.proposal_id,
            prefix_version=view.prefix_version,
            decision_version=view.decision.decision_version,
            kind=kind,
        )
        for view, kind in (eager + normal)[:capacity]
    )


def select_normal_recovery_tasks(
    views: Iterable[SchedulingView],
    *,
    capacity: int,
    normal_cohort: Optional[str] = None,
    owner_id: Optional[str] = None,
) -> Tuple[DraftTask, ...]:
    """Rebuild from committed prefix, independently of the Target batch.

    A rejected parent's stale proposal_id is deliberately absent from the intent.
    Eager eligibility is irrelevant to obtaining a legal normal recovery proposal.
    """

    _check_capacity(capacity)
    return tuple(
        DraftTask(
            request_id=view.request_id,
            owner_id=view.owner_id,
            home_cohort=view.home_cohort,
            proposal_id=None,
            prefix_version=view.prefix_version,
            decision_version=view.decision.decision_version,
            kind="normal_recovery",
        )
        for view in _unique_views(views)
        if view.active
        and view.recovery_required
        and not view.verifying
        and not view.normal_draft_pending
        and (normal_cohort is None or view.home_cohort == normal_cohort)
        and (owner_id is None or view.owner_id == owner_id)
    )[:capacity]


def select_eager_draft_tasks(
    views: Iterable[SchedulingView],
    *,
    capacity: int,
    owner_id: Optional[str] = None,
) -> Tuple[DraftTask, ...]:
    """Offer at most one dependent Draft role per verifying request identity."""

    _check_capacity(capacity)
    return tuple(
        DraftTask(
            request_id=view.request_id,
            owner_id=view.owner_id,
            home_cohort=view.home_cohort,
            proposal_id=view.proposal_id,
            prefix_version=view.prefix_version,
            decision_version=view.decision.decision_version,
            kind="eager_continuation",
        )
        for view in _unique_views(views)
        if view.active
        and view.verifying
        and view.is_legal
        and view.proposal_id is not None
        and view.decision.can_admit
        and not view.recovery_required
        and not view.continuation_pending
        and (owner_id is None or view.owner_id == owner_id)
    )[:capacity]


def select_normal_draft_tasks(
    views: Iterable[SchedulingView],
    *,
    capacity: int,
    normal_cohort: Optional[str] = None,
    owner_id: Optional[str] = None,
) -> Tuple[DraftTask, ...]:
    """Offer recovery first, then initial/ordinary Draft work from committed state."""

    _check_capacity(capacity)
    unique = _unique_views(views)
    recovery = select_normal_recovery_tasks(
        unique, capacity=capacity, normal_cohort=normal_cohort, owner_id=owner_id
    )
    ordinary = tuple(
        DraftTask(
            request_id=view.request_id,
            owner_id=view.owner_id,
            home_cohort=view.home_cohort,
            proposal_id=None,
            prefix_version=view.prefix_version,
            decision_version=view.decision.decision_version,
            kind="normal_draft",
        )
        for view in unique
        if view.active
        and view.proposal_id is None
        and not view.verifying
        and not view.continuation_pending
        and not view.normal_draft_pending
        and not view.recovery_required
        and (normal_cohort is None or view.home_cohort == normal_cohort)
        and (owner_id is None or view.owner_id == owner_id)
    )
    return (recovery + ordinary)[:capacity]


def revalidate_target_admission(
    admission: TargetAdmission, view: SchedulingView, *, normal_cohort: str
) -> bool:
    """Check a queued intent against the current owner snapshot before dispatch."""

    return admission in select_target_admissions(
        (view,), normal_cohort=normal_cohort, capacity=1
    )


def revalidate_draft_task(task: DraftTask, view: SchedulingView) -> bool:
    """A stale version, switched decision, terminal request, or old parent fails."""

    selector = {
        "normal_recovery": select_normal_recovery_tasks,
        "normal_draft": select_normal_draft_tasks,
        "eager_continuation": select_eager_draft_tasks,
    }.get(task.kind)
    return selector is not None and task in selector((view,), capacity=1)
