"""Physical backend coordination using independent fake paged-KV storage."""

from __future__ import annotations

from dataclasses import replace

import pytest
from test_phase4_serial import phase4_config as _config_fixture
from test_phase4_vllm_draft import FakeWorker, commit, next_token

from specrhythm.continuation.core import DraftCompletion, ParentVerification, RollingContinuation
from specrhythm.continuation.gpu_backend import RollingVllmDraftBackend
from specrhythm.continuation.policy import StaticEagerEligibility
from specrhythm.phase4.draft_batch import DraftProposalPlan, commit_frontier

phase4_config = _config_fixture


class FencedWorker(FakeWorker):
    def __init__(self):
        super().__init__()
        self.inflight = False
        self.events = []
        self.during_forward = None

    def materialize(self, rows, purpose):
        assert not self.inflight, "previous write must be fenced"
        self.events.append(("forward", purpose))
        result = super().materialize(rows, purpose)
        self.inflight = True
        if purpose == "eager" and self.during_forward is not None:
            self.during_forward()
        return result

    def fence(self, reason):
        self.events.append(("fence", reason))
        self.inflight = False
        super().fence(reason)

    def greedy(self, logits):
        # Real worker's bulk CPU token transfer also waits on its model writes.
        result = super().greedy(logits)
        self.inflight = False
        return result

    def release(self, request_ids):
        assert not self.inflight, "cannot release in-flight physical writes"
        self.events.append(("release", tuple(request_ids)))
        super().release(request_ids)


def setup(config, *, count=1):
    worker = FencedWorker()
    backend = RollingVllmDraftBackend(config, worker=worker)
    prefixes = {f"req-{i}": (10, 20 + i) for i in range(count)}
    owner = RollingContinuation("owner", StaticEagerEligibility(set(prefixes)))
    backend.initialize_many(tuple(prefixes.items()))
    for rid, prefix in prefixes.items():
        owner.register(rid, prefix, max_output_tokens=100)
    return backend, worker, owner, prefixes


def normal(backend, owner, rid="req-0"):
    work = owner.schedule_normal_recovery(rid)
    result = backend.propose_many((DraftProposalPlan(
        rid, work.prefix_version, work.dependency_prefix, work.candidate_length, work.eos_token_ids
    ),))[rid]
    completion = DraftCompletion(work.work_id, result, backend.states[rid].materialized)
    return owner.record_normal_completion(work, completion)


def begin(backend, owner, proposal, rid="req-0"):
    owner.start_verification(rid, proposal.proposal_id)
    work = owner.begin_continuation(rid)
    backend.begin_gpu_continuation(work)
    return work


def complete(backend, work):
    for _ in range(5):
        result = backend.step_gpu_continuation(work)
        if result is not None:
            return result
    raise AssertionError("fixed continuation did not complete within five GPU steps")


def plan_and_receipt(proposal, *, accepted=4, tail=None, terminal=False):
    tail = (next_token(proposal.parent_prefix + proposal.tokens),) if tail is None else tail
    plan = commit(
        proposal.request_id, proposal.parent_prefix, proposal.tokens, accepted, tail,
        round_id=proposal.prefix_version, terminal=terminal,
    )
    receipt = ParentVerification(
        proposal.owner_id, proposal.request_id, proposal.proposal_id,
        proposal.prefix_version, proposal.parent_prefix,
        proposal.tokens[:accepted] + tuple(tail), "cancelled" if terminal else None,
    )
    return plan, receipt


def greedy_tokens(prefix, count):
    result = []
    for _ in range(count):
        result.append(next_token(prefix + tuple(result)))
    return tuple(result)


def test_enrollment_has_no_gpu_work_and_batched_steps_materialize_bridge_correctly(phase4_config):
    backend, worker, owner, prefixes = setup(phase4_config, count=2)
    proposals = {rid: normal(backend, owner, rid) for rid in prefixes}
    before = len(worker.calls)
    works = [begin(backend, owner, proposal, rid) for rid, proposal in proposals.items()]
    assert len(worker.calls) == before
    assert backend.metrics.counters["eager_enrolled"] == 2
    assert backend.metrics.counters["eager_started"] == 0
    for index in range(5):
        results = backend.step_gpu_continuations(works)
        assert all((result is not None) == (index == 4) for result in results.values())
        assert not worker.inflight
    eager_rows = [rows for purpose, rows in worker.calls if purpose == "eager"]
    assert len(eager_rows) == 5
    assert [len(rows) for rows in eager_rows] == [2] * 5
    assert all(len(row.suffix) == 1 for rows in eager_rows for row in rows)
    for work in works:
        result = results[work.work_id]
        expected = greedy_tokens(work.dependency_prefix, 5)
        assert result.generated_tokens == expected
        assert result.materialized_kv_frontier == len(work.dependency_prefix) + 4
        state = backend.states[work.request_id]
        assert state.prefix == prefixes[work.request_id]
        assert state.proposal == proposals[work.request_id].tokens
        assert tuple(worker.memory[state.internal_id]) == work.dependency_prefix + expected[:-1]
    assert backend.report()["draft_model_forward_count_by_purpose"]["eager"] == 5
    assert backend.metrics.counters["eager_started"] == 2


@pytest.mark.parametrize("target_first", [True, False])
def test_real_backend_rolls_success_rejects_recovers_and_rolls_again(phase4_config, target_first):
    backend, worker, owner, _ = setup(phase4_config)
    proposal = normal(backend, owner)
    outcomes = []
    for round_id in range(7):
        work = begin(backend, owner, proposal)
        rejecting = round_id == 2
        mismatch = round_id == 5
        args = {"accepted": 2, "tail": (900,)} if rejecting else {}
        if mismatch:
            args = {"tail": (901,)}
        plan, receipt = plan_and_receipt(proposal, **args)
        if target_first:
            owner.resolve_parent_verification(receipt)
        completion = complete(backend, work)
        owner.record_continuation_completion(work, completion)
        if not target_first:
            owner.resolve_parent_verification(receipt)
        if rejecting or mismatch:
            assert owner.state("req-0").recovery_required
            backend.rebase_gpu_parent(plan, work=work)
            assert backend.states["req-0"].materialized == len(plan.final_prefix)
            assert backend.states["req-0"].next_logits == next_token(plan.final_prefix)
            proposal = normal(backend, owner)
            outcomes.append("recovered")
        else:
            proposal = owner.promote_continuation("req-0", work.work_id)
            backend.rebase_gpu_parent(plan, work=work, promoted_tokens=proposal.tokens)
            outcomes.append("promoted")
        state = backend.states["req-0"]
        assert proposal.tokens == greedy_tokens(plan.final_prefix, 4)
        assert state.proposal == proposal.tokens
        assert state.prefix == proposal.parent_prefix == plan.final_prefix
        assert state.next_round == round_id + 1
        assert tuple(worker.memory[state.internal_id][:state.materialized]) == (
            plan.final_prefix + proposal.tokens[:-1]
        )
    assert outcomes == ["promoted", "promoted", "recovered", "promoted", "promoted",
                        "recovered", "promoted"]
    assert len([1 for purpose, _ in worker.calls if purpose == "setup"]) == 1
    assert owner.state("req-0").accounting.recovery_jobs == 2
    assert backend.report()["eager_gpu_live_work"] == 0


@pytest.mark.parametrize("steps", [0, 1, 3])
def test_early_rejection_aborts_at_fence_and_repairs_only_true_suffix(phase4_config, steps):
    backend, worker, owner, _ = setup(phase4_config)
    proposal = normal(backend, owner)
    work = begin(backend, owner, proposal)
    for _ in range(steps):
        assert backend.step_gpu_continuation(work) is None
    snapshot = backend.abort_gpu_continuation(work)
    assert backend.metrics.counters["eager_enrolled"] == 1
    assert backend.metrics.counters["eager_started"] == int(steps > 0)
    assert len(snapshot["generated_tokens"]) == steps
    assert snapshot["materialized_kv_frontier"] == len(work.dependency_prefix) + steps - 1
    plan, _ = plan_and_receipt(proposal, accepted=1, tail=(777,))
    backend.rebase_gpu_parent(plan, work=work)
    row = [rows[0] for purpose, rows in worker.calls if purpose == "commit"][-1]
    assert row.valid_length == len(proposal.parent_prefix) + 1
    assert row.suffix == (777,)
    assert backend.states["req-0"].next_logits == next_token(plan.final_prefix)
    before = list(worker.calls)
    with pytest.raises(ValueError, match="retired"):
        backend.step_gpu_continuation(work)
    assert worker.calls == before
    assert backend.abort_gpu_continuation(work)["retired"]
    assert backend.gpu_continuation_snapshot(work)["materialized_kv_frontier"] == (
        snapshot["materialized_kv_frontier"]
    )


def test_rebase_release_and_abort_are_prohibited_during_active_gpu_write(phase4_config):
    backend, worker, owner, _ = setup(phase4_config)
    proposal = normal(backend, owner)
    work = begin(backend, owner, proposal)
    plan, _ = plan_and_receipt(proposal, accepted=1, tail=(777,))
    observed = []

    def during_forward():
        for operation in (
            lambda: backend.rebase_gpu_parent(plan, work=work),
            lambda: backend.finish_many(("req-0",)),
            lambda: backend.abort_gpu_continuation(work),
        ):
            with pytest.raises(RuntimeError, match="active write fence"):
                operation()
            observed.append(True)

    worker.during_forward = during_forward
    backend.step_gpu_continuation(work)
    assert observed == [True, True, True]
    assert not worker.inflight
    backend.rebase_gpu_parent(plan, work=work)
    backend.finish_many(("req-0",))
    assert not worker.memory


def test_promotion_requires_complete_exact_bridge_and_preserves_legacy_commit_guard(phase4_config):
    backend, _, owner, _ = setup(phase4_config)
    proposal = normal(backend, owner)
    work = begin(backend, owner, proposal)
    plan, _ = plan_and_receipt(proposal)
    backend.step_gpu_continuation(work)
    with pytest.raises(ValueError, match="not valid"):
        backend.rebase_gpu_parent(plan, work=work, promoted_tokens=(1, 2, 3, 4))
    completion = complete(backend, work)
    with pytest.raises(ValueError, match="frontier"):
        commit_frontier(plan, completion.materialized_kv_frontier)
    with pytest.raises(ValueError, match="not valid"):
        wrong, _ = plan_and_receipt(proposal, tail=(999,))
        backend.rebase_gpu_parent(
            wrong, work=work, promoted_tokens=completion.generated_tokens[1:]
        )
    result = backend.rebase_gpu_parent(
        plan, work=work, promoted_tokens=completion.generated_tokens[1:]
    )
    assert result["promoted_candidates"] == 4
    assert backend.rebase_gpu_parent(
        plan, work=work, promoted_tokens=completion.generated_tokens[1:]
    ) == result
    with pytest.raises(ValueError, match="conflicting duplicate"):
        backend.rebase_gpu_parent(plan, work=work, promoted_tokens=(999,))


@pytest.mark.parametrize(
    "field", ["owner_id", "request_id", "prefix_version", "dependency_prefix"]
)
def test_immutable_physical_work_identity_fences_stale_results(phase4_config, field):
    backend, _, owner, _ = setup(phase4_config)
    proposal = normal(backend, owner)
    work = begin(backend, owner, proposal)
    mutation = {
        "owner_id": "other-owner", "request_id": "other-request", "prefix_version": 99,
        "dependency_prefix": (999,) + work.dependency_prefix[1:],
    }
    with pytest.raises(ValueError, match="conflicting"):
        backend.step_gpu_continuation(replace(work, **{field: mutation[field]}))
    assert backend.gpu_continuation_snapshot(work)["generated_tokens"] == ()


def test_terminal_materializes_correction_then_releases_speculative_high_water(
    phase4_config,
):
    backend, worker, owner, _ = setup(phase4_config)
    proposal = normal(backend, owner)
    work = begin(backend, owner, proposal)
    complete(backend, work)
    plan, _ = plan_and_receipt(proposal, accepted=1, tail=(900,), terminal=True)
    result = backend.rebase_gpu_parent(plan, work=work)
    assert result["materialized_kv_length"] == len(plan.final_prefix)
    row = [rows[0] for purpose, rows in worker.calls if purpose == "commit"][-1]
    assert row.suffix == (900,)
    assert "req-0" in backend.retired
    assert not worker.memory
    assert backend.report()["eager_gpu_live_work"] == 0
    assert backend.gpu_continuation_snapshot(work)["retired"]


def test_shutdown_physically_releases_live_partial_continuations(phase4_config):
    backend, worker, owner, _ = setup(phase4_config)
    proposal = normal(backend, owner)
    work = begin(backend, owner, proposal)
    backend.step_gpu_continuation(work)
    backend.shutdown()
    assert backend.closed and worker.closed
    assert not backend.states and not worker.memory
    assert backend.report()["eager_gpu_live_work"] == 0
    assert any(kind == "release" for kind, _ in worker.events)


def test_active_work_cannot_bypass_extended_settlement_and_invalid_finish_is_atomic(phase4_config):
    backend, worker, owner, _ = setup(phase4_config)
    proposal = normal(backend, owner)
    work = begin(backend, owner, proposal)
    plan, _ = plan_and_receipt(proposal)
    before = backend.gpu_continuation_snapshot(work)
    with pytest.raises(ValueError, match="explicit GPU parent settlement"):
        backend.commit_many((plan,))
    with pytest.raises(ValueError, match="unknown"):
        backend.finish_many(("req-0", "unknown"))
    assert backend.gpu_continuation_snapshot(work) == before
    assert worker.memory


def test_mixed_batch_promotes_recovers_and_releases_without_cross_request_kv_changes(
    phase4_config,
):
    backend, worker, owner, prefixes = setup(phase4_config, count=3)
    proposals = {rid: normal(backend, owner, rid) for rid in prefixes}
    works = {rid: begin(backend, owner, proposal, rid) for rid, proposal in proposals.items()}
    for _ in range(5):
        completions = backend.step_gpu_continuations(tuple(works.values()))
    plans = {}
    for index, (rid, proposal) in enumerate(proposals.items()):
        args = {} if index == 0 else {"accepted": 1, "tail": (999,), "terminal": index == 2}
        plan, receipt = plan_and_receipt(proposal, **args)
        plans[rid] = plan
        owner.record_continuation_completion(works[rid], completions[works[rid].work_id])
        owner.resolve_parent_verification(receipt)
    promoted = owner.promote_continuation("req-0", works["req-0"].work_id)
    backend.rebase_gpu_parent(plans["req-0"], work=works["req-0"], promoted_tokens=promoted.tokens)
    first_memory = tuple(worker.memory[backend.states["req-0"].internal_id])
    backend.rebase_gpu_parent(plans["req-1"], work=works["req-1"])
    recovered = normal(backend, owner, "req-1")
    backend.rebase_gpu_parent(plans["req-2"], work=works["req-2"])
    assert tuple(worker.memory[backend.states["req-0"].internal_id]) == first_memory
    assert recovered.tokens == greedy_tokens(plans["req-1"].final_prefix, 4)
    assert "req-2" not in backend.states and "req-2" in backend.retired
    assert len(worker.memory) == 2
    assert backend.report()["eager_gpu_live_work"] == 0


def test_real_worker_failure_requires_teardown_and_preserves_live_work_evidence(phase4_config):
    backend, worker, owner, _ = setup(phase4_config)
    proposal = normal(backend, owner)
    work = begin(backend, owner, proposal)
    worker.inject_failure = True
    with pytest.raises(RuntimeError, match="injected forward failure"):
        backend.step_gpu_continuation(work)
    assert backend.failed and backend.report()["eager_gpu_live_work"] == 1
    assert ("fence", "eager_failed_step") in worker.events
    with pytest.raises(RuntimeError, match="failed"):
        backend.abort_gpu_continuation(work)
    backend.shutdown()
    assert backend.closed and worker.closed and not worker.memory
    assert backend.report()["eager_gpu_live_work"] == 0
