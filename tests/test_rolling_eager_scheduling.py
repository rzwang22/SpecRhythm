"""Executable CPU contracts for stable eligibility and cross-cohort work roles."""

from dataclasses import FrozenInstanceError, dataclass, replace

import pytest

from specrhythm.continuation.policy import (
    EagerDecision,
    EagerStepContext,
    StaticEagerEligibility,
)
from specrhythm.continuation.scheduling import (
    SchedulingConflictError,
    SchedulingView,
    revalidate_draft_task,
    revalidate_target_admission,
    select_eager_draft_tasks,
    select_normal_recovery_tasks,
    select_target_admissions,
)


def ready(request_id="req-a", cohort="A", **changes):
    return replace(
        SchedulingView(
            request_id=request_id,
            owner_id="owner-1",
            home_cohort=cohort,
            proposal_id=f"{request_id}/P0",
            prefix_version=0,
            decision=EagerDecision(True, True, 0),
            is_ready=True,
            is_legal=True,
        ),
        **changes,
    )


@dataclass
class ToggleProvider:
    """Test-only mutable provider, exercising the future evaluate interface."""

    static: StaticEagerEligibility
    enabled: bool = True
    decision_version: int = 0

    def set_enabled(self, enabled):
        self.enabled = enabled
        self.decision_version += 1

    def evaluate(self, request_state, step_context):
        eligibility = self.static.evaluate(request_state, step_context)
        return EagerDecision(eligibility.eligible, self.enabled, self.decision_version)


def test_static_eligibility_is_bound_to_request_not_cohort_or_rejection():
    configured = {"req-a"}
    provider = StaticEagerEligibility(configured)
    configured.add("refill")
    before = ready()
    rejected = replace(
        before, home_cohort="B", recovery_required=True, is_ready=False, prefix_version=1
    )
    refill = ready("refill", "A")
    assert isinstance(provider.request_ids, frozenset)
    assert provider.evaluate(before, EagerStepContext(0)).eligible
    assert provider.evaluate(rejected, EagerStepContext(100)).eligible
    assert not provider.evaluate(refill, EagerStepContext(100)).eligible
    with pytest.raises(FrozenInstanceError):
        provider.enabled = False


def test_eligibility_admission_and_validity_are_independent():
    marked = ready(is_legal=False, promoted=True)
    deferred = ready("valid", promoted=True)
    provider = StaticEagerEligibility({marked.request_id, deferred.request_id})
    assert provider.evaluate(marked, EagerStepContext()).can_admit
    assert select_target_admissions((marked,), normal_cohort="B", capacity=3) == ()
    assert select_target_admissions((deferred,), normal_cohort="B", capacity=0) == ()
    assert deferred.decision.eligible
    assert select_target_admissions((deferred,), normal_cohort="B", capacity=1)[0].kind == "eager"


def test_promoted_request_crosses_cohort_and_is_admitted_once():
    promoted_a = ready(promoted=True, proposal_id="req-a/E1", prefix_version=1)
    normal_b = ready("req-b", "B", decision=EagerDecision(False, True, 0))
    admissions = select_target_admissions(
        (promoted_a, normal_b), (promoted_a,), normal_cohort="B", capacity=2
    )
    assert [(item.request_id, item.kind) for item in admissions] == [
        ("req-a", "eager"),
        ("req-b", "normal"),
    ]
    assert admissions[0].home_cohort == "A"
    assert admissions[0].proposal_id == "req-a/E1"
    assert admissions[0].prefix_version == 1
    assert admissions[0].owner_id == "owner-1"
    assert admissions[0].decision_version == 0


def test_next_stage_cross_cohort_admission_has_new_proposal_and_prefix_version():
    first = ready(promoted=True, proposal_id="req-a/E1", prefix_version=1)
    admission = select_target_admissions((first,), normal_cohort="B", capacity=1)[0]
    second = replace(first, proposal_id="req-a/E2", prefix_version=2)
    later = select_target_admissions((second,), normal_cohort="B", capacity=1)[0]
    assert admission.request_id == later.request_id
    assert admission.proposal_id != later.proposal_id
    assert admission.prefix_version + 1 == later.prefix_version
    assert not revalidate_target_admission(admission, second, normal_cohort="B")
    assert revalidate_target_admission(later, second, normal_cohort="B")


@pytest.mark.parametrize(
    "changed",
    [
        {"owner_id": "owner-2"},
        {"prefix_version": 1},
        {"proposal_id": "req-a/P1"},
        {"home_cohort": "B"},
        {"decision": EagerDecision(True, False, 1)},
        {"is_ready": False},
    ],
)
def test_conflicting_snapshots_reject_instead_of_selecting_stale(changed):
    original = ready()
    conflicting = replace(original, **changed)
    with pytest.raises(SchedulingConflictError):
        select_target_admissions(
            (original,), (conflicting,), normal_cohort="A", capacity=0
        )
    with pytest.raises(SchedulingConflictError):
        select_eager_draft_tasks((original, conflicting), capacity=1)
    with pytest.raises(SchedulingConflictError):
        select_normal_recovery_tasks((original, conflicting), capacity=1)


def test_rejection_schedules_fresh_recovery_while_other_request_verifies():
    rejected = ready(recovery_required=True, prefix_version=3, proposal_id="req-a/E2")
    other = ready("req-b", "B")
    target = select_target_admissions(
        (rejected, other), (rejected,), normal_cohort="B", capacity=2
    )
    assert [item.request_id for item in target] == ["req-b"]
    recovery = select_normal_recovery_tasks((rejected, rejected, other), capacity=2)
    assert len(recovery) == 1
    task = recovery[0]
    assert task.kind == "normal_recovery"
    assert task.request_id == rejected.request_id
    assert task.proposal_id is None
    assert task.prefix_version == 3
    assert task.owner_id == rejected.owner_id
    assert task.decision_version == rejected.decision.decision_version
    assert revalidate_draft_task(task, rejected)
    recovered = replace(
        rejected, recovery_required=False, proposal_id="req-a/R3", prefix_version=3
    )
    assert not revalidate_draft_task(task, recovered)
    assert rejected.decision.eligible
    assert select_target_admissions((recovered,), normal_cohort="A", capacity=1)


def test_target_and_draft_are_roles_of_the_same_request():
    initial = ready()
    target = select_target_admissions((initial,), normal_cohort="A", capacity=1)[0]
    verifying = replace(initial, is_ready=False, verifying=True)
    draft = select_eager_draft_tasks((verifying, verifying), capacity=5)
    assert len(draft) == 1
    assert draft[0].kind == "eager_continuation"
    assert (
        draft[0].request_id,
        draft[0].owner_id,
        draft[0].proposal_id,
        draft[0].prefix_version,
    ) == (target.request_id, target.owner_id, target.proposal_id, target.prefix_version)
    assert select_target_admissions((verifying,), normal_cohort="A", capacity=1) == ()
    assert select_eager_draft_tasks(
        (replace(verifying, continuation_pending=True),), capacity=1
    ) == ()
    assert revalidate_draft_task(draft[0], verifying)
    assert not revalidate_draft_task(draft[0], replace(verifying, prefix_version=1))


def test_toggle_disables_new_roles_and_returns_promoted_proposal_to_normal_cohort():
    provider = ToggleProvider(StaticEagerEligibility({"req-a"}))
    promoted = ready(promoted=True, proposal_id="req-a/E1", prefix_version=1)
    promoted = replace(promoted, decision=provider.evaluate(promoted, EagerStepContext()))
    queued = select_target_admissions((promoted,), normal_cohort="B", capacity=1)[0]
    verifying = replace(promoted, is_ready=False, verifying=True)
    draft = select_eager_draft_tasks((verifying,), capacity=1)[0]
    provider.set_enabled(False)
    disabled = replace(promoted, decision=provider.evaluate(promoted, EagerStepContext(1)))
    assert disabled.decision.eligible
    assert not disabled.decision.can_admit
    assert select_target_admissions((disabled,), normal_cohort="B", capacity=1) == ()
    ordinary = select_target_admissions((disabled,), normal_cohort="A", capacity=1)
    assert ordinary[0].kind == "normal"
    assert ordinary[0].proposal_id == promoted.proposal_id
    assert ordinary[0].prefix_version == promoted.prefix_version
    assert not revalidate_target_admission(queued, disabled, normal_cohort="B")
    disabled_verifying = replace(disabled, is_ready=False, verifying=True)
    assert select_eager_draft_tasks((disabled_verifying,), capacity=1) == ()
    assert not revalidate_draft_task(draft, disabled_verifying)
    # An existing immutable Target role is unchanged; the owner resolves it.
    assert queued.proposal_id == disabled_verifying.proposal_id
    assert queued.prefix_version == disabled_verifying.prefix_version
    provider.set_enabled(True)
    enabled = replace(disabled, decision=provider.evaluate(disabled, EagerStepContext(2)))
    renewed = select_target_admissions((enabled,), normal_cohort="B", capacity=1)[0]
    assert renewed.kind == "eager"
    assert renewed.decision_version == 2
    assert not revalidate_target_admission(queued, enabled, normal_cohort="B")
    assert select_eager_draft_tasks(
        (replace(enabled, is_ready=False, verifying=True),), capacity=1
    )


def test_disabled_recovery_remains_available():
    disabled = ready(
        recovery_required=True, decision=EagerDecision(True, False, 1), is_ready=False
    )
    recovery = select_normal_recovery_tasks((disabled,), capacity=1)
    assert len(recovery) == 1
    assert recovery[0].decision_version == 1
    assert recovery[0].kind == "normal_recovery"


@pytest.mark.parametrize("kind", ["target", "recovery", "eager"])
def test_owner_filter_and_terminal_cleanup_exclude_unrelated_work(kind):
    own = ready(promoted=True)
    other = ready("req-b", "B", owner_id="owner-2", promoted=True)
    done = ready("req-done", active=False, promoted=True)
    if kind == "target":
        tasks = select_target_admissions(
            (own, other, done), normal_cohort="B", owner_id="owner-1", capacity=5
        )
    elif kind == "recovery":
        views = tuple(replace(view, recovery_required=True) for view in (own, other, done))
        tasks = select_normal_recovery_tasks(views, owner_id="owner-1", capacity=5)
    else:
        views = tuple(replace(view, verifying=True) for view in (own, other, done))
        tasks = select_eager_draft_tasks(views, owner_id="owner-1", capacity=5)
    assert [task.request_id for task in tasks] == [own.request_id]
    assert tasks[0].owner_id == own.owner_id


@pytest.mark.parametrize("capacity", [-1, 0.5, True])
def test_invalid_capacity_is_rejected(capacity):
    with pytest.raises(ValueError, match="capacity"):
        select_target_admissions((), normal_cohort="A", capacity=capacity)
    with pytest.raises(ValueError, match="capacity"):
        select_normal_recovery_tasks((), capacity=capacity)
    with pytest.raises(ValueError, match="capacity"):
        select_eager_draft_tasks((), capacity=capacity)


def test_unconfirmed_or_invalid_continuation_is_never_target_ready():
    pending = ready(promoted=False, is_ready=False)
    unconfirmed = ready("unconfirmed", promoted=True, is_legal=False)
    assert select_target_admissions(
        (pending, unconfirmed), normal_cohort="A", capacity=2
    ) == ()


def test_static_provider_disabled_preserves_baseline_cohort_order():
    provider = StaticEagerEligibility({"req-a", "req-b"}, enabled=False)
    a = ready()
    b = ready("req-b", "B", promoted=True)
    a = replace(a, decision=provider.evaluate(a, EagerStepContext()))
    b = replace(b, decision=provider.evaluate(b, EagerStepContext()))
    assert [admission.request_id for admission in select_target_admissions(
        (a, b), (b,), normal_cohort="A", capacity=2
    )] == ["req-a"]
    assert [admission.request_id for admission in select_target_admissions(
        (a, b), (b,), normal_cohort="B", capacity=2
    )] == ["req-b"]


@pytest.mark.parametrize("version", [-1, True, False, 1.0, "1", None])
def test_decision_version_requires_a_nonnegative_integer(version):
    with pytest.raises(ValueError, match="decision_version"):
        EagerDecision(True, True, version)
    with pytest.raises(ValueError, match="decision_version"):
        StaticEagerEligibility({"req"}, decision_version=version)


@pytest.mark.parametrize("flag", [1, 0, "true", None])
def test_decision_flags_require_actual_booleans(flag):
    with pytest.raises(ValueError, match="booleans"):
        EagerDecision(flag, True, 0)
    with pytest.raises(ValueError, match="booleans"):
        EagerDecision(True, flag, 0)
    with pytest.raises(ValueError, match="booleans"):
        StaticEagerEligibility({"req"}, enabled=flag)
