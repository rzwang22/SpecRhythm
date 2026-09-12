"""Connected CPU scheduler/owner regression tests using actual protocol work."""

from dataclasses import dataclass, replace

import pytest

from specrhythm.continuation.adapter import (
    scheduling_view,
    start_admission,
    start_draft_task,
)
from specrhythm.continuation.core import (
    ContinuationStatus,
    ParentVerification,
    RollingContinuation,
)
from specrhythm.continuation.cpu import generate
from specrhythm.continuation.policy import EagerDecision, StaticEagerEligibility
from specrhythm.continuation.scheduling import (
    select_eager_draft_tasks,
    select_normal_draft_tasks,
    select_normal_recovery_tasks,
    select_target_admissions,
)


@dataclass
class ToggleProvider:
    enabled: bool = True
    version: int = 0

    def switch(self, enabled):
        self.enabled = enabled
        self.version += 1

    def evaluate(self, request_state, step_context):
        return EagerDecision(request_state.request_id in {"a", "b"}, self.enabled, self.version)


def make_machine(*, provider=None, owner_id="owner"):
    machine = RollingContinuation(owner_id, provider or StaticEagerEligibility({"a", "b"}))
    machine.register("a", (10,), home_cohort="A", max_output_tokens=200)
    machine.register("b", (100,), home_cohort="B", max_output_tokens=200)
    return machine


def normal_proposal(machine, request_id):
    task = select_normal_draft_tasks((scheduling_view(machine, request_id),), capacity=1)[0]
    work = start_draft_task(machine, task)
    return machine.record_normal_completion(work, generate(work, lambda prefix: prefix[-1] + 1))


def target_admission(machine, request_id="a", cohort="A"):
    return select_target_admissions(
        (scheduling_view(machine, request_id),), normal_cohort=cohort, capacity=1
    )[0]


def eager_work(machine, request_id="a"):
    task = select_eager_draft_tasks((scheduling_view(machine, request_id),), capacity=1)[0]
    return start_draft_task(machine, task)


def receipt(machine, request_id, proposal, delta=None):
    return ParentVerification(
        machine.owner_id,
        request_id,
        proposal.proposal_id,
        proposal.prefix_version,
        proposal.parent_prefix,
        delta if delta is not None else proposal.tokens + (proposal.tokens[-1] + 1,),
    )


def roll(machine, request_id="a", cohort="B", *, target_first=False):
    admission = target_admission(machine, request_id, cohort)
    proposal = start_admission(machine, admission, normal_cohort=cohort)
    work = eager_work(machine, request_id)
    completion = generate(work, lambda prefix: prefix[-1] + 1)
    if target_first:
        machine.resolve_parent_verification(receipt(machine, request_id, proposal))
        waiting = scheduling_view(machine, request_id)
        assert not waiting.is_ready
        assert not waiting.is_legal
        assert select_target_admissions((waiting,), normal_cohort=cohort, capacity=1) == ()
        machine.record_continuation_completion(work, completion)
    else:
        machine.record_continuation_completion(work, completion)
        assert scheduling_view(machine, request_id).verifying
        machine.resolve_parent_verification(receipt(machine, request_id, proposal))
    machine.promote_continuation(request_id, work.work_id)
    return admission


def test_real_core_rolls_across_cohorts_after_capacity_deferral_and_revalidates():
    machine = make_machine()
    normal_proposal(machine, "a")
    first = roll(machine, cohort="A")
    assert first.kind == "normal"
    deferred = scheduling_view(machine, "a")
    before = machine.state("a")
    assert select_target_admissions((deferred,), normal_cohort="B", capacity=0) == ()
    assert machine.state("a") == before
    assert deferred.decision.eligible
    assert deferred.promoted and deferred.home_cohort == "A"
    queued = target_admission(machine, cohort="B")
    assert queued.kind == "eager"
    second = roll(machine, target_first=True)
    third = roll(machine)
    assert second.request_id == third.request_id == first.request_id
    assert second.home_cohort == third.home_cohort == "A"
    assert (first.prefix_version, second.prefix_version, third.prefix_version) == (0, 1, 2)
    assert len({first.proposal_id, second.proposal_id, third.proposal_id}) == 3
    with pytest.raises(ValueError, match="stale"):
        start_admission(machine, queued, normal_cohort="B")
    assert machine.state("a").accounting.committed_tokens == 15
    assert machine.state("b").accounting.committed_tokens == 0


def test_rejected_request_gets_one_real_recovery_while_other_target_runs():
    machine = make_machine()
    normal_proposal(machine, "a")
    normal_proposal(machine, "b")
    roll(machine, cohort="A")
    roll(machine)
    doomed = start_admission(machine, target_admission(machine, cohort="B"), normal_cohort="B")
    invalid_work = eager_work(machine)
    completion = generate(invalid_work, lambda prefix: prefix[-1] + 1)
    machine.record_continuation_completion(invalid_work, completion)
    machine.resolve_parent_verification(receipt(machine, "a", doomed, doomed.tokens[:2] + (900,)))
    rejected = scheduling_view(machine, "a")
    other = scheduling_view(machine, "b")
    targets = select_target_admissions(
        (rejected, other), (rejected,), normal_cohort="B", capacity=2
    )
    assert [item.request_id for item in targets] == ["b"]
    other_proposal = start_admission(machine, targets[0], normal_cohort="B")
    assert scheduling_view(machine, "b").verifying
    recovery_task = select_normal_recovery_tasks((rejected, rejected), capacity=2)[0]
    assert recovery_task.proposal_id is None
    recovery = start_draft_task(machine, recovery_task)
    assert recovery.kind == "recovery"
    assert recovery.dependency_prefix == machine.state("a").committed_prefix
    assert recovery.dependency_prefix[-1] == 900
    assert scheduling_view(machine, "a").normal_draft_pending
    assert select_normal_recovery_tasks((scheduling_view(machine, "a"),), capacity=1) == ()
    with pytest.raises(ValueError, match="stale"):
        start_draft_task(machine, recovery_task)
    rebuilt = machine.record_normal_completion(
        recovery, generate(recovery, lambda prefix: prefix[-1] + 1)
    )
    assert rebuilt.proposal_id != doomed.proposal_id
    machine.resolve_parent_verification(receipt(machine, "b", other_proposal))
    assert roll(machine, cohort="A").kind == "normal"
    assert roll(machine).kind == "eager"
    assert roll(machine, target_first=True).kind == "eager"
    state = machine.state("a")
    assert state.eager_decision.eligible
    assert state.accounting.recovery_jobs == 1
    assert state.accounting.recovery_generated_tokens == 4
    assert state.accounting.discarded_early_tokens == 5
    assert state.accounting.committed_tokens == 28
    assert machine.state("b").accounting.committed_tokens == 5


def test_switch_invalidates_queued_admission_and_preserves_legal_promoted_proposal():
    provider = ToggleProvider()
    machine = make_machine(provider=provider)
    normal_proposal(machine, "a")
    roll(machine, cohort="A")
    queued = target_admission(machine, cohort="B")
    committed = machine.state("a").committed_prefix
    provider.switch(False)
    with pytest.raises(ValueError, match="stale"):
        start_admission(machine, queued, normal_cohort="B")
    disabled = scheduling_view(machine, "a")
    assert disabled.promoted and disabled.is_ready and disabled.is_legal
    assert disabled.decision.eligible and not disabled.decision.enabled
    assert select_target_admissions((disabled,), normal_cohort="B", capacity=1) == ()
    normal = target_admission(machine, cohort="A")
    assert normal.kind == "normal"
    assert normal.proposal_id == queued.proposal_id
    started = start_admission(machine, normal, normal_cohort="A")
    assert select_eager_draft_tasks((scheduling_view(machine, "a"),), capacity=1) == ()
    assert machine.state("a").committed_prefix == committed
    with pytest.raises(ValueError, match="stale"):
        start_admission(machine, normal, normal_cohort="A")
    provider.switch(True)
    work = eager_work(machine)
    machine.record_continuation_completion(work, generate(work, lambda prefix: prefix[-1] + 1))
    machine.resolve_parent_verification(receipt(machine, "a", started))
    machine.promote_continuation("a", work.work_id)
    assert roll(machine).decision_version == 2


def test_switch_cancels_running_draft_and_late_result_cannot_reactivate_it():
    provider = ToggleProvider()
    machine = make_machine(provider=provider)
    normal_proposal(machine, "a")
    started = start_admission(machine, target_admission(machine), normal_cohort="A")
    task = select_eager_draft_tasks((scheduling_view(machine, "a"),), capacity=1)[0]
    work = start_draft_task(machine, task)
    completion = generate(work, lambda prefix: prefix[-1] + 1)
    provider.switch(False)
    scheduling_view(machine, "a")
    with pytest.raises(ValueError, match="stale"):
        start_draft_task(machine, task)
    assert machine.record_continuation_completion(work, completion) == ContinuationStatus.CANCELLED
    machine.resolve_parent_verification(receipt(machine, "a", started))
    assert machine.state("a").accounting.committed_tokens == 5
    assert machine.state("a").recovery_required
    provider.switch(True)
    live = scheduling_view(machine, "a")
    assert live.decision.eligible and live.decision.enabled
    assert not live.is_ready
    assert select_target_admissions((live,), normal_cohort="A", capacity=1) == ()
    recovery = start_draft_task(machine, select_normal_recovery_tasks((live,), capacity=1)[0])
    machine.record_normal_completion(recovery, generate(recovery, lambda prefix: prefix[-1] + 1))
    assert roll(machine, cohort="A").kind == "normal"
    assert roll(machine).kind == "eager"


def test_owner_shutdown_fences_queued_and_late_work_without_affecting_other_owner():
    machine = make_machine()
    foreign = make_machine(owner_id="other-owner")
    normal_proposal(machine, "a")
    normal_proposal(foreign, "a")
    foreign_before = foreign.state("a")
    queued = target_admission(machine)
    with pytest.raises(ValueError, match="stale"):
        start_admission(foreign, queued, normal_cohort="A")
    parent = start_admission(machine, queued, normal_cohort="A")
    task = select_eager_draft_tasks((scheduling_view(machine, "a"),), capacity=1)[0]
    work = start_draft_task(machine, task)
    completion = generate(work, lambda prefix: prefix[-1] + 1)
    machine.shutdown()
    for request_id in ("a", "b"):
        view = scheduling_view(machine, request_id)
        assert not view.active
        assert select_target_admissions((view,), normal_cohort="A", capacity=1) == ()
        assert select_normal_recovery_tasks((view,), capacity=1) == ()
        assert select_eager_draft_tasks((view,), capacity=1) == ()
    with pytest.raises(ValueError, match="stale"):
        start_admission(machine, queued, normal_cohort="A")
    with pytest.raises(ValueError, match="stale"):
        start_draft_task(machine, task)
    assert machine.record_continuation_completion(work, completion) == ContinuationStatus.RELEASED
    assert not machine.resolve_parent_verification(receipt(machine, "a", parent))
    assert machine.state("a").accounting.committed_tokens == 0
    assert machine.state("a").materialized_kv_frontier == 0
    assert foreign.state("a") == foreign_before


@pytest.mark.parametrize("field,value", [("prefix_version", 50), ("proposal_id", "old")])
def test_forged_or_stale_task_version_cannot_start_work(field, value):
    machine = make_machine()
    normal_proposal(machine, "a")
    start_admission(machine, target_admission(machine), normal_cohort="A")
    task = select_eager_draft_tasks((scheduling_view(machine, "a"),), capacity=1)[0]
    before = machine.state("a")
    with pytest.raises(ValueError, match="stale"):
        start_draft_task(machine, replace(task, **{field: value}))
    assert machine.state("a") == before


def test_disabled_eager_runs_multiple_normal_rounds_without_recovery_accounting():
    machine = make_machine(provider=StaticEagerEligibility({"a", "b"}, enabled=False))
    for expected_version in range(4):
        view = scheduling_view(machine, "a")
        assert view.decision.eligible and not view.decision.enabled
        tasks = select_normal_draft_tasks((view,), normal_cohort="A", capacity=1)
        assert len(tasks) == 1 and tasks[0].kind == "normal_draft"
        assert tasks[0].prefix_version == expected_version
        work = start_draft_task(machine, tasks[0])
        assert work.kind == "normal"
        assert select_normal_draft_tasks((scheduling_view(machine, "a"),), capacity=1) == ()
        machine.record_normal_completion(work, generate(work, lambda prefix: prefix[-1] + 1))
        admission = target_admission(machine)
        assert admission.kind == "normal"
        started = start_admission(machine, admission, normal_cohort="A")
        assert select_eager_draft_tasks((scheduling_view(machine, "a"),), capacity=1) == ()
        machine.resolve_parent_verification(receipt(machine, "a", started))
    state = machine.state("a")
    assert state.accounting.committed_tokens == 20
    assert state.accounting.normal_generated_tokens == 16
    assert state.accounting.early_generated_tokens == 0
    assert state.accounting.recovery_jobs == 0
    assert state.accounting.recovery_generated_tokens == 0


def test_normal_draft_capacity_prioritizes_recovery_over_initial_request():
    machine = make_machine(provider=StaticEagerEligibility({"a", "b"}, enabled=False))
    normal_proposal(machine, "a")
    started = start_admission(machine, target_admission(machine), normal_cohort="A")
    machine.resolve_parent_verification(receipt(machine, "a", started, (900,)))
    initial = scheduling_view(machine, "b")
    rejected = scheduling_view(machine, "a")
    tasks = select_normal_draft_tasks((initial, rejected), capacity=1)
    assert len(tasks) == 1
    assert tasks[0].request_id == "a" and tasks[0].kind == "normal_recovery"
    assert start_draft_task(machine, tasks[0]).kind == "recovery"


@pytest.mark.parametrize("role", ["target", "draft"])
def test_foreign_intent_rejected_before_local_policy_can_cancel_work(role):
    source = make_machine(owner_id="source")
    provider = ToggleProvider()
    destination = make_machine(owner_id="destination", provider=provider)
    normal_proposal(source, "a")
    foreign_admission = target_admission(source)
    start_admission(source, foreign_admission, normal_cohort="A")
    foreign_task = select_eager_draft_tasks((scheduling_view(source, "a"),), capacity=1)[0]
    normal_proposal(destination, "a")
    start_admission(destination, target_admission(destination), normal_cohort="A")
    local_work = eager_work(destination)
    before = destination.state("a")
    provider.switch(False)
    with pytest.raises(ValueError, match="another owner"):
        if role == "target":
            start_admission(destination, foreign_admission, normal_cohort="A")
        else:
            start_draft_task(destination, foreign_task)
    assert destination.state("a") == before
    continuation = destination.continuation("a", local_work.work_id)
    assert continuation.status == ContinuationStatus.GENERATING
