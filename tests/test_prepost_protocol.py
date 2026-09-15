"""No GPU claims: actual owner/backend methods with independent physical KV storage."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_rolling_eager_gpu_backend import greedy_tokens
from test_serial_eager_owner import OwnerWorker, proposal_row, settle_row, verify_row

from specrhythm.continuation.prepost import acceptance, canonical_sample
from specrhythm.continuation.prepost_backend import PrePostBackendMixin
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.vllm_draft_backend import VllmBatchedDraftBackend
from specrhythm.serving.prepost_machine import PrePostMachine
from specrhythm.serving.s2_pool import prefix_record


class Backend(PrePostBackendMixin, VllmBatchedDraftBackend):
    def physical_rows(self):
        return {
            rid: prefix_record(s.prefix, s.materialized, [[id(s)]])
            for rid, s in self.states.items()
        }


def machine(eager=True):
    backend = Backend(SimpleNamespace(max_model_len=4096), worker=OwnerWorker())
    result = PrePostMachine(backend, request_ids=("a", "b"), eager=eager)
    for rid in ("a", "b"):
        result.initialize(rid, (10, 20), token_prefix_hash((10, 20)))
    return result


def feedback(proposal, prefix, reject=False, terminal=False):
    tokens = tuple(proposal["proposal_token_ids"])
    delta = (999,) if reject else tokens
    final = prefix + delta
    return dict(
        request_id=proposal["request_id"],
        round_id=proposal["round_id"],
        committed_delta=list(delta),
        committed_prefix_hash=token_prefix_hash(final),
        terminal=terminal,
    ), final


def cleanup(m):
    for rid, state in m.requests.items():
        m.diagnostic_settle(
            settle_row(
                rid, state.committed_token_ids, state.next_round_id, terminal=state.finished
            )
        )
    m.shutdown()
    assert not m.backend.worker.memory


@pytest.mark.parametrize("target_first", [True, False])
@pytest.mark.parametrize("eager", [True, False])
def test_real_machine_mixed_long_short_repeated_failure_and_recovery(target_first, eager):
    m = machine(eager)
    prefixes, budgets = {rid: (10, 20) for rid in m.requests}, {rid: 100 for rid in m.requests}
    proposals = m.batch_propose([proposal_row(rid) for rid in m.requests])["proposals"]
    assert [len(p["proposal_token_ids"]) for p in proposals] == [1, 1]
    try:
        for cycle in range(7):
            m.verify_start([verify_row(p) for p in proposals])
            rows = []
            for p in proposals:
                rid = p["request_id"]
                row, prefixes[rid] = feedback(p, prefixes[rid], rid == "b" and cycle in (1, 2, 3))
                rows.append(row)
                budgets[rid] -= len(row["committed_delta"])
            before = len(m.backend.worker.calls)
            if target_first:
                m.prepare_synchronizations(rows)
            while m.step():
                pass
            if not target_first:
                m.prepare_synchronizations(rows)
            assert m.finish_synchronizations(rows)
            # Duplicate feedback is idempotent and launches no physical work.
            after = list(m.backend.worker.calls)
            m.prepare_synchronizations(rows)
            assert m.finish_synchronizations(rows)
            assert after == m.backend.worker.calls
            proposals = m.batch_propose(
                [proposal_row(rid, prefixes[rid], cycle + 1, budgets[rid]) for rid in m.requests]
            )["proposals"]
            expected = [4, 1 if eager and cycle in (1, 2, 3) else 4]
            assert [len(p["proposal_token_ids"]) for p in proposals] == expected
            for p in proposals:
                rid, tokens = p["request_id"], tuple(p["proposal_token_ids"])
                assert tokens == greedy_tokens(prefixes[rid], len(tokens))
                s = m.backend.states[rid]
                assert s.prefix == prefixes[rid]
                assert tuple(m.backend.worker.memory[s.internal_id][: s.materialized]) == (
                    s.prefix + tokens[:-1]
                )
            calls = m.backend.worker.calls[before:]
            assert sum(purpose == "prepost_post" for purpose, _ in calls) == 1
            assert sum(purpose == "prepost_extension" for purpose, _ in calls) == (
                0 if eager else 3
            )
            assert all(len(batch) == 2 for purpose, batch in calls if purpose == "prepost_post")
            assert not any(purpose in ("proposal", "commit") for purpose, _ in calls)
        assert m.counters["committed_tokens"] == sum(100 - v for v in budgets.values())
        assert (
            m.counters["committed_tokens"]
            == m.counters["accepted_tokens"] + m.counters["correction_tokens"]
        )
        assert (
            m.counters["early_generated_tokens"]
            == m.counters["reused_early_tokens"] + m.counters["discarded_early_tokens"]
        )
    finally:
        cleanup(m)


def test_no_bonus_and_padding_contract():
    decision, record = canonical_sample(
        [2, 3, 4, 5], [2, 3, 4, 5, 9], previous=1, maximum=99, eos=()
    )
    assert decision.committed_token_ids == (2, 3, 4, 5)
    assert record["unused_target_bonus_tokens"] == 1
    with pytest.raises(ValueError):
        acceptance([2], [2, 3])
    with pytest.raises(ValueError):
        canonical_sample([2], [2, -1], previous=1, maximum=99, eos=())


@pytest.mark.parametrize("fault", ["version", "proposal", "frontier", "release", "feedback"])
def test_invalid_dependency_cannot_mutate_or_revive_work(fault):
    from dataclasses import replace

    m = machine()
    proposal = m.batch_propose([proposal_row("a")])["proposals"][0]
    m.verify_start([verify_row(proposal)])
    work = m.works["a"]
    before = list(m.backend.worker.calls)
    try:
        with pytest.raises((ValueError, RuntimeError)):
            if fault == "version":
                m.backend.step_prepost([replace(work, prefix_version=9)])
            elif fault == "proposal":
                m.backend.step_prepost([replace(work, proposal_tokens=(999,))])
            elif fault == "release":
                m.backend.finish_many(["a"])
            elif fault == "feedback":
                row, _ = feedback(proposal, (10, 20))
                m.prepare_synchronizations([{**row, "round_id": 9}])
            else:
                s = m.backend.states["a"]
                s.materialized -= 1
                try:
                    m.backend.step_prepost([work])
                finally:
                    s.materialized += 1
        assert before == m.backend.worker.calls
        m._abort("a", "test")
        with pytest.raises((ValueError, RuntimeError)):
            m.backend.step_prepost([work])
    finally:
        cleanup(m)


def test_actual_owner_enqueue_ack_and_queued_rejection_preempts_next_step():
    import queue
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from specrhythm.serving.eager_owner import EagerOwner

    entered, resume, queued = threading.Event(), threading.Event(), threading.Event()

    class Worker(OwnerWorker):
        def materialize(self, rows, purpose):
            if purpose == "prepost_lookahead":
                entered.set()
                assert resume.wait(3), "test did not release physical step"
            return super().materialize(rows, purpose)

    class Mailbox(queue.Queue):
        def put(self, item, *args, **kwargs):
            super().put(item, *args, **kwargs)
            if item[0] == "synchronize_and_batch_propose":
                queued.set()

    def factory():
        b = Backend(SimpleNamespace(max_model_len=4096), worker=Worker())
        return PrePostMachine(b, request_ids=("a",))

    owner = EagerOwner(factory, timeout_seconds=3)
    try:
        owner.call(
            "initialize",
            dict(
                request_id="a",
                committed_token_ids=[10, 20],
                committed_prefix_hash=token_prefix_hash((10, 20)),
            ),
        )
        proposal = owner.call("batch_propose", {"requests": [proposal_row()]})["proposals"][0]
        assert owner.call("eager_enqueue", {"requests": [verify_row(proposal)]})["enqueued"]
        assert entered.wait(2)
        # Owner is inside its physical call. Replacing its now-empty queue is
        # deterministic; queued event marks actual put, without a polling sleep.
        assert owner.commands.empty()
        owner.commands = Mailbox()
        row, final = feedback(proposal, (10, 20), reject=True)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                owner.call,
                "synchronize_and_batch_propose",
                dict(
                    synchronizations=[row],
                    proposals=[proposal_row(prefix=final, round_id=1, remaining=99)],
                ),
            )
            assert queued.wait(2)
            assert not future.done()
            resume.set()
            result = future.result(timeout=2)
        assert len(result["proposals"][0]["proposal_token_ids"]) == 1
        assert owner.machine.backend.metrics.forwards["prepost_lookahead"] == 1
        assert owner.machine.backend.metrics.forwards["prepost_post"] == 1
        assert owner.machine.counters["discarded_early_tokens"] == 1
    finally:
        resume.set()
        state = owner.machine.requests["a"]
        owner.call(
            "diagnostic_settle", settle_row("a", state.committed_token_ids, state.next_round_id)
        )
        owner.call("shutdown", {"deadline_ns": time.monotonic_ns() + 3_000_000_000})
        assert not owner._thread.is_alive() and not owner.machine.backend.worker.memory


@pytest.mark.parametrize("mode", ["full", "runtime"])
def test_real_runtime_allocator_mixed_post_fence_and_release(mode, tmp_path, monkeypatch):
    from test_fixed_runtime_audit import LifecycleWorker

    from specrhythm.serving import s2_draft
    from specrhythm.serving.fixed_audit import FixedAuditMixin
    from specrhythm.serving.s2_pool import publish

    monkeypatch.setattr(s2_draft, "cuda_memory", lambda _: {"free_memory_bytes": 10**9})
    monkeypatch.setenv("SR_FIXED_DRAFT_AUDIT", mode)
    monkeypatch.setenv("SR_S2_MODE", "serial-eager-prepost3")
    path = tmp_path / "control.json"
    monkeypatch.setenv("SR_S2_CONTROL", str(path))
    monkeypatch.setenv("SR_S2_RUN_DIRECTORY", str(tmp_path))
    states = {str(i): {"state": "STAGED"} for i in range(360)}
    publish(path, {"barrier_ns": None, "requests": states})

    class Audited(PrePostBackendMixin, FixedAuditMixin, s2_draft.S2DraftBackend):
        pass

    b = Audited(SimpleNamespace(max_model_len=4096), worker=LifecycleWorker())
    m = PrePostMachine(b, request_ids=states)
    for rid in ("0", "1"):
        m.initialize(rid, (10, 20), token_prefix_hash((10, 20)))
    b.initialize_many([(rid, (10, 20)) for rid in states if rid not in ("0", "1")])
    for state in states.values():
        state["state"] = "ACTIVE"
    publish(path, {"barrier_ns": 1, "requests": states})
    b.audit_requests(tuple(states))
    proposals = m.batch_propose([proposal_row(rid) for rid in ("0", "1")])["proposals"]
    m.verify_start([verify_row(p) for p in proposals])
    visits = b.audit_full_visits
    for _ in range(3):
        m.step()
    if mode == "runtime":
        assert b.audit_full_visits == visits
    rows = [feedback(p, (10, 20), reject=p["request_id"] == "1")[0] for p in proposals]
    m.prepare_synchronizations(rows)
    assert m.finish_synchronizations(rows)
    assert b.states["0"].proposal == greedy_tokens(b.states["0"].prefix, 4)
    assert len(b.states["1"].proposal) == 1
    b._audit()  # Explicit full reconciliation agrees with incrementally maintained evidence.
    for rid, state in list(m.requests.items()):
        m.diagnostic_settle(settle_row(rid, state.committed_token_ids, state.next_round_id))
    b.finish_many(tuple(b.states))
    m.shutdown()
    assert not b.worker.blocks
    if mode == "runtime":
        assert b.audit_guard.allocations == b.audit_guard.releases
        assert not b.audit_guard.pending and not b.audit_guard.owners


@pytest.mark.parametrize("terminal", ["eos", "limit", "partial"])
def test_terminal_feedback_and_lookahead_never_publish_extra_output(terminal):
    m = machine()
    try:
        p = m.batch_propose([proposal_row("a")])["proposals"][0]
        m.verify_start([verify_row(p)])
        while m.step():
            pass
        row, prefix = feedback(p, (10, 20))
        m.prepare_synchronizations([row])
        m.finish_synchronizations([row])
        state = m.core._state("a")
        p = m.batch_propose([proposal_row("a", prefix, 1, state.remaining)])["proposals"][0]
        m.verify_start([verify_row(p)])
        m.step()
        candidates = tuple(p["proposal_token_ids"])
        delta = candidates[:2] if terminal != "eos" else (999,)
        row = dict(
            request_id="a",
            round_id=1,
            committed_delta=list(delta),
            committed_prefix_hash=token_prefix_hash(prefix + delta),
            terminal=True,
        )
        m.prepare_synchronizations([row])
        m.finish_synchronizations([row])
        assert m.requests["a"].committed_token_ids == prefix + delta
        assert m.requests["a"].pending_proposal is None
        assert "a" in m.backend.retired
        assert not m.works and not m.backend.prepost_jobs
        assert m.finish_authoritative(
            dict(
                request_id="a",
                committed_prefix=prefix + delta,
                committed_prefix_hash=token_prefix_hash(prefix + delta),
            )
        )["finished"]
        m.prepare_synchronizations([row])  # Exact late duplicate is idempotent.
        assert m.finish_synchronizations([row])
        with pytest.raises(ValueError):
            m.prepare_synchronizations([{**row, "committed_delta": [888]}])
    finally:
        cleanup(m)


def test_switch_versions_cancellation_and_refill_seed_do_not_revive_work():
    m = machine()
    try:
        p = m.batch_propose([proposal_row("a")])["proposals"][0]
        m.verify_start([verify_row(p)])
        m.step()
        work = m.works["a"]
        m.set_eager(False, 1)
        with pytest.raises(ValueError):
            m.set_eager(True, 0)
        with pytest.raises(ValueError):
            m.set_eager(True, 1)
        m.set_eager(True, 2)
        assert not m.step()  # Enabled again does not revive the invalidated dependency.
        row, prefix = feedback(p, (10, 20), reject=True)
        m.prepare_synchronizations([row])
        m.finish_synchronizations([row])
        with pytest.raises(ValueError):
            m.backend.step_prepost([work])
        # Previously resident but inactive request enters through the same seed path.
        proposals = m.batch_propose([proposal_row("a", prefix, 1, 99), proposal_row("b")])[
            "proposals"
        ]
        assert [len(p["proposal_token_ids"]) for p in proposals] == [1, 1]
        m.verify_start([verify_row(p) for p in proposals])
        while m.step():
            pass
        assert len(m.works) == 2
    finally:
        cleanup(m)


def test_physical_write_failure_propagates_and_shutdown_waits_for_fence():
    m = machine()
    p = m.batch_propose([proposal_row("a")])["proposals"][0]
    m.verify_start([verify_row(p)])
    m.backend.worker.inject_failure = True
    with pytest.raises(RuntimeError, match="injected forward failure"):
        m.step()
    assert m.backend.failed and m.backend.prepost_jobs
    assert m.requests["a"].committed_token_ids == (10, 20)
    m.backend.shutdown()
    assert m.backend.closed and not m.backend.worker.memory and not m.backend.prepost_jobs


def test_phased_protocol_retention_exceeds_old_limit_and_reports_loss(monkeypatch):
    from specrhythm.continuation.prepost_records import Records
    from specrhythm.continuation.trace import PHASE_BUDGETS, TRACE

    records = Records()
    monkeypatch.setattr(TRACE, "phase", "measurement")
    for i in range(20001):
        records.append({"index": i})
    assert len(records.rows()) == 20001 and records.report()["dropped_rows"] == 0
    for i in range(PHASE_BUDGETS["measurement"] - 20001 + 1):
        records.append({"index": i + 20001})
    assert records.report()["phases"]["measurement"]["dropped_rows"] == 1
    monkeypatch.setattr(TRACE, "phase", "drain")
    records.append({"index": 999999})
    assert records.report()["phases"]["drain"]["retained_rows"] == 1


def test_controlled_drain_with_unpublished_target_delta_preserves_cancellation():
    m = machine()
    p = m.batch_propose([proposal_row("a")])["proposals"][0]
    m.verify_start([verify_row(p)])
    m.step()
    row, final = feedback(p, (10, 20))
    generated_before = m.counters["early_generated_tokens"]
    result = m.diagnostic_settle(settle_row("a", final, 1, terminal=False))
    assert result["disposition"] == "DIAGNOSTIC_CANCELLED"
    assert result["new_proposals_generated"] == 0
    assert m.requests["a"].committed_token_ids == final
    assert m.counters["early_generated_tokens"] == generated_before
    assert m.backend.metrics.forwards["prepost_post"] == 0
    assert not m.backend.prepost_jobs
    cleanup(m)
