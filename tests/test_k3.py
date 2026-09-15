"""Actual K3 owner/backend transitions, CPU KV worker (no GPU overlap claim)."""

from types import SimpleNamespace as NS

import pytest
from test_ping_prepost import admit, run_work
from test_prepost_protocol import cleanup, feedback
from test_rolling_eager_gpu_backend import greedy_tokens
from test_serial_eager_owner import OwnerWorker, proposal_row, verify_row

from specrhythm.continuation.k3_backend import K3BackendMixin
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.vllm_draft_backend import VllmBatchedDraftBackend
from specrhythm.serving.k3_machine import K3Machine
from specrhythm.serving.s2_pool import prefix_record


class Backend(K3BackendMixin, VllmBatchedDraftBackend):
    def physical_rows(self):
        return {
            rid: prefix_record(s.prefix, s.materialized, [[id(s)]])
            for rid, s in self.states.items()
        }


def machine(eager=True, ids=("a", "b"), budget=100, eos=(), worker=None, complete=True, mode=None):
    m = K3Machine(
        Backend(NS(max_model_len=4096), worker=worker or OwnerWorker()),
        request_ids=ids,
        eager=eager, mode=mode,
    )
    for rid in ids:
        m.initialize(rid, (10, 20), token_prefix_hash((10, 20)))
    m.register(
        [
            {
                **proposal_row(rid, remaining=budget),
                "home_cohort": "A" if (mode and mode.startswith("serial-")) or i % 2 == 0 else "B",
                "eos_token_ids": eos,
            }
            for i, rid in enumerate(ids)
        ]
    )
    if complete:
        run_work(m)
    return m


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("target_first", [False, True])
def test_k3_success_rejection_repeated_and_reuse(eager, target_first):
    m = machine(eager)
    try:
        assert len(m.backend.prepost_forwards.rows()) == 2  # seed is cached, plus TWO
        assert m.backend.metrics.counters["proposals"] == 2
        assert m.backend.metrics.counters["proposed_tokens"] == 6
        for rejected_at in [None, 0, 1, 0, None, None]:
            for home in ("A", "B"):
                claim = admit(m, home, 1)["claims"][0]
                p, rid = claim["proposal"], claim["request_id"]
                assert len(p["proposal_token_ids"]) == 3
                m.verify_start([verify_row(p)])
                prefix = m.requests[rid].committed_token_ids
                delta = tuple(p["proposal_token_ids"])
                if rejected_at is not None:
                    delta = delta[:rejected_at] + (999,)
                final = prefix + delta
                row = dict(
                    request_id=rid,
                    round_id=p["round_id"],
                    committed_delta=list(delta),
                    committed_prefix_hash=token_prefix_hash(final),
                    terminal=False,
                )
                if not target_first:
                    run_work(m)
                before = len(m.backend.prepost_forwards.rows())
                m.feedback([row])
                run_work(m)
                records = m.backend.prepost_forwards.rows()[before:]
                if eager and rejected_at is None:
                    assert not any(r["purpose"] == "prepost_post" for r in records)
                else:
                    assert [r["purpose"] for r in records] == [
                        "prepost_post",
                        "prepost_extension",
                        "prepost_extension",
                    ]
                new = m.ready[rid]["proposal"]["proposal_token_ids"]
                assert len(new) == 3 and tuple(new) == greedy_tokens(final, 3)
                state = m.backend.states[rid]
                assert tuple(m.backend.worker.memory[state.internal_id][: state.materialized]) == (
                    final + tuple(new[:-1])
                )
                n = len(m.backend.worker.calls)
                m.feedback([row])
                run_work(m)
                assert len(m.backend.worker.calls) == n
                with pytest.raises(Exception, match="duplicate"):
                    m.feedback([{**row, "committed_delta": [111]}])
                with pytest.raises(Exception, match="feedback without consumed claim"):
                    m.feedback([{**row, "round_id": 999}])
        assert m.counters["committed_tokens"] == (
            m.counters["accepted_tokens"] + m.counters["correction_tokens"]
        )
        assert m.counters["early_generated_tokens"] == (
            m.counters["reused_early_tokens"] + m.counters["discarded_early_tokens"]
        )
        if not eager:
            assert not m.works and not m.counters["admissions"]
    finally:
        cleanup(m)
    assert m.eager_report()["candidate_accounting"]["live_unsubmitted"] == 0


@pytest.mark.parametrize("budget", [1, 2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("eager", [False, True])
def test_k3_output_budget_complete_reference(budget, eager):
    m = machine(eager, ids=("a",), budget=budget)
    try:
        while not m.requests["a"].finished:
            p = admit(m)["claims"][0]["proposal"]
            remaining = m.core._state("a").remaining
            assert len(p["proposal_token_ids"]) == min(3, remaining)
            m.verify_start([verify_row(p)])
            row, final = feedback(p, m.requests["a"].committed_token_ids, terminal=remaining <= 3)
            m.feedback([row])
            run_work(m)
        assert final == (10, 20) + greedy_tokens((10, 20), budget)
        assert not m.backend.states and not m.ready
    finally:
        cleanup(m)
    assert m.eager_report()["candidate_accounting"]["live_unsubmitted"] == 0


@pytest.mark.parametrize("steps", [0, 1, 2])
@pytest.mark.parametrize("consume", [False, True])
def test_k3_cancel_counts_private_candidates_once(steps, consume):
    m = machine(ids=("a",), complete=False)
    for _ in range(steps):
        m.step()
    if consume and steps == 2:
        p = admit(m)["claims"][0]["proposal"]
        m.verify_start([verify_row(p)])
        m.step()  # One pending conditional candidate must be discarded after fence.
    cleanup(m)
    report = m.eager_report()["candidate_accounting"]
    assert report["live_unsubmitted"] == 0, report
    assert report["generated"] == sum(report[k] for k in (
        "submitted", "discarded_early", "discarded_unsubmitted",
    ))
    # Repeating the real settlement cannot inflate discarded counts.
    from test_serial_eager_owner import settle_row

    m.diagnostic_settle(settle_row("a", (10, 20), 0))
    assert m.eager_report()["candidate_accounting"] == report


def test_k3_eos_refill_and_private_incomplete_proposal():
    eos = greedy_tokens((10, 20), 2)[-1]
    m = machine(ids=("a",), eos=(eos,), complete=False)
    try:
        assert not m.ready and not admit(m)["claims"]
        m.step()
        p = admit(m)["claims"][0]["proposal"]
        assert len(p["proposal_token_ids"]) == 2 and p["proposal_eos"]
        m.verify_start([verify_row(p)])
        row, _ = feedback(p, (10, 20), terminal=True)
        m.feedback([row])
        run_work(m)
        m.initialize("refill", (10, 20), token_prefix_hash((10, 20)))
        m.register([{**proposal_row("refill"), "home_cohort": "A"}])
        assert "refill" not in m.ready
        run_work(m)
        assert len(m.ready["refill"]["proposal"]["proposal_token_ids"]) == 3
        assert (
            next(r for r in m.ping_events.rows() if r["event"] == "ready")["short_reason"]
            == "candidate_EOS"
        )
    finally:
        cleanup(m)


@pytest.mark.parametrize("fault", ["version", "frontier", "release", "claim", "ready"])
def test_k3_invalid_work_never_reuses_or_releases_unchecked_state(fault):
    from dataclasses import replace

    m = machine(ids=("a",))
    try:
        p = admit(m)["claims"][0]["proposal"]
        row = verify_row(p)
        m.verify_start([row])
        work = m.works["a"]
        before = len(m.backend.worker.calls)
        with pytest.raises(ValueError):
            if fault == "version":
                m.backend.step_prepost([replace(work, prefix_version=99)])
            elif fault == "frontier":
                s = m.backend.states["a"]
                valid = s.materialized
                try:
                    s.materialized -= 1
                    m.backend.step_prepost([work])
                finally:
                    s.materialized = valid
            elif fault == "release":
                m.backend.finish_many(["a"])
            elif fault == "claim":
                m.verify_start([row])
            else:
                m._publish_proposal("a", m.backend.states["a"].proposal[:1], 0, 0)
        assert len(m.backend.worker.calls) == before
        m.begin_drain()
        # Late physical continuation after cancellation is refused even with old identity.
        with pytest.raises(ValueError):
            m.backend.step_prepost([work])
    finally:
        cleanup(m)
