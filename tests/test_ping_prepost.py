"""Real owner protocol and physical backend; CPU memory stands in for GPU tensors."""

from types import SimpleNamespace as NS

import pytest
from test_prepost_protocol import cleanup, feedback
from test_rolling_eager_gpu_backend import greedy_tokens
from test_serial_eager_owner import OwnerWorker, proposal_row, verify_row

from specrhythm.continuation.ping_prepost_backend import PingPrePostBackendMixin
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.vllm_draft_backend import VllmBatchedDraftBackend
from specrhythm.serving.ping_prepost_controller import PingPrePostController
from specrhythm.serving.ping_prepost_machine import PingPrePostMachine
from specrhythm.serving.s2_pool import prefix_record


class Backend(PingPrePostBackendMixin, VllmBatchedDraftBackend):
    def physical_rows(self):
        return {
            rid: prefix_record(s.prefix, s.materialized, [[id(s)]])
            for rid, s in self.states.items()
        }


def machine(eager=True, ids=("a", "b"), worker=None):
    m = PingPrePostMachine(
        Backend(NS(max_model_len=4096), worker=worker or OwnerWorker()),
        request_ids=ids,
        eager=eager,
    )
    for rid in ids:
        m.initialize(rid, (10, 20), token_prefix_hash((10, 20)))
    m.register(
        [
            {**proposal_row(rid), "home_cohort": "A" if i % 2 == 0 else "B"}
            for i, rid in enumerate(ids)
        ]
    )
    return m


def admit(m, home="A", capacity=8, active=None):
    return m.admit(
        dict(
            active_request_ids=list(active or m.homes),
            opportunity=m.opportunity + 1,
            normal_cohort=home,
            capacity=capacity,
        )
    )


def run_work(m):
    for _ in range(10):
        if not m.step():
            return
    pytest.fail("owner did not complete bounded token work")


def settle(m, claims, rejects=(), *, target_first=False):
    rows = []
    for c in claims:
        p = c["proposal"]
        row, _ = feedback(
            p, m.requests[p["request_id"]].committed_token_ids, p["request_id"] in rejects
        )
        rows.append(row)
    if not target_first:
        run_work(m)
    ack = m.feedback(rows)
    assert not ack["proposals"]
    assert all(r["physical_settlement"] == "PENDING" for r in ack["synchronizations"])
    run_work(m)
    before = list(m.backend.worker.calls)
    m.feedback(rows)  # Exact late duplicates launch no new work.
    run_work(m)
    assert before == m.backend.worker.calls
    return rows


@pytest.mark.parametrize("target_first", [True, False])
@pytest.mark.parametrize("eager", [True, False])
def test_cross_opportunity_success_reject_short_recovery_and_again(eager, target_first):
    m = machine(eager, ("a", "b", "c", "d"))
    try:
        for cycle in range(8):
            a = admit(m, "A" if cycle % 2 == 0 else "B", capacity=2)
            claims = a["claims"]
            assert len(claims) == 2
            if eager and cycle < 3:
                assert [c["request_id"] for c in claims] == ["a", "c"]
                assert [c["opportunity_cohort"] for c in claims] == [a["opportunity_cohort"]] * 2
            m.verify_start([verify_row(c["proposal"]) for c in claims])
            rejected = [claims[-1]["request_id"]] if cycle in (1, 2, 3) else []
            settle(m, claims, rejected, target_first=target_first)
            for c in claims:
                rid = c["request_id"]
                s = m.backend.states[rid]
                p = m.ready[rid]["proposal"]["proposal_token_ids"]
                assert tuple(p) == greedy_tokens(s.prefix, len(p))
                assert len(p) == (1 if eager and rid in rejected else 4)
                assert tuple(m.backend.worker.memory[s.internal_id][: s.materialized]) == (
                    s.prefix + tuple(p[:-1])
                )
        forwards = m.backend.prepost_forwards.rows()
        assert not eager or not any(r["purpose"] == "prepost_extension" for r in forwards)
        assert all(all(n == 1 for n in r["materialized_positions"]) for r in forwards)
        assert m.counters["committed_tokens"] == (
            m.counters["accepted_tokens"] + m.counters["correction_tokens"]
        )
        assert not m.claims and not m.normal
    finally:
        cleanup(m)


def test_ready_other_home_and_mixed_single_gpu_work_not_global_fifo():
    m = machine()
    try:
        a = admit(m, capacity=1)["claims"]
        m.verify_start([verify_row(a[0]["proposal"])])
        # Full parent feedback arriving before lookahead must wait, but B is ready.
        row, _ = feedback(a[0]["proposal"], (10, 20))
        m.feedback([row])
        b = admit(m, "A", capacity=1)["claims"]
        assert b[0]["request_id"] == "b"  # fallback; A's dependency does not block B.
        m.verify_start([verify_row(b[0]["proposal"])])
        reject, _ = feedback(b[0]["proposal"], (10, 20), True)
        m.feedback([reject])
        m.step()  # B correction->P1 + A lookahead #1 are one real physical call.
        physical = m.backend.prepost_forwards.rows()[-1]
        assert physical["purpose"] == "prepost_mixed" and physical["B"] == 2
        assert {r["work_kind"] for r in physical["bindings"]} == {"post", "lookahead"}
        assert "b" in m.ready and "a" not in m.ready
        run_work(m)
        mixed = admit(m)["claims"]
        assert [(r["request_id"], len(r["proposal"]["proposal_token_ids"])) for r in mixed] == [
            ("a", 4)
        ]  # normal B supplements only its own cohort opportunity.
        assert len(admit(m, "B")["claims"]) == 1
    finally:
        cleanup(m)


def test_noneligible_normal_extensions_merge_with_lookahead():
    m = machine(ids=("a", "b"))
    try:
        m.update_eligibility(True, 1, ["a"])
        a = admit(m, "A")["claims"]
        b = admit(m, "B")["claims"]
        m.verify_start([verify_row(c["proposal"]) for c in a + b])
        row, _ = feedback(b[0]["proposal"], (10, 20))
        m.feedback([row])
        m.step()  # b seed with a lookahead step one
        m.step()  # b ordinary extension + a lookahead step two
        physical = m.backend.prepost_forwards.rows()[-1]
        assert {r["work_kind"] for r in physical["bindings"]} == {"normal_extension", "lookahead"}
        assert physical["B"] == 2
        run_work(m)
        assert "b" in m.ready and "a" not in m.ready
        settle(m, a)
        assert len(m.ready["a"]["proposal"]["proposal_token_ids"]) == 4
    finally:
        cleanup(m)


def test_atomic_claim_replay_prefix_and_invalid_enrollment():
    m = machine()
    try:
        a = admit(m, capacity=1)["claims"]
        assert [r["request_id"] for r in admit(m, capacity=1)["claims"]] == ["b"]
        assert not admit(m)["claims"]
        row = verify_row(a[0]["proposal"])
        with pytest.raises(Exception, match="stale"):
            m.verify_start([{**row, "round_id": 9}])
        m.verify_start([row])
        with pytest.raises(Exception, match="consumed"):
            m.verify_start([row])
        rows = settle(m, a)
        with pytest.raises(Exception, match="duplicate Target feedback"):
            m.feedback([{**rows[0], "committed_delta": [991]}])
        with pytest.raises(Exception, match="duplicate/moved home"):
            m.register([{**proposal_row("a"), "home_cohort": "B"}])
        assert m.homes["a"] == "A"
    finally:
        cleanup(m)


def test_control_switch_and_controller_empty_polls_keep_roles():
    m = machine()
    clock = NS(rows={rid: {"state": "ACTIVE"} for rid in m.homes})
    controller = PingPrePostController()
    client = NS(call=lambda op, data: m.admit(data))
    try:
        a = controller.select(clock, client)
        b = controller.select(clock, client)
        empty = controller.select(clock, client)
        assert not empty["claims"] and controller.admitted == 2
        m.verify_start([verify_row(c["proposal"]) for c in a["claims"] + b["claims"]])
        m.step()
        m.update_eligibility(False, 1, ["a"])
        settle(m, a["claims"] + b["claims"])
        assert all(len(r["proposal"]["proposal_token_ids"]) == 4 for r in m.ready.values())
        next_batch = controller.select(clock, client)
        assert next_batch["opportunity_cohort"] == "A"
        m.update_eligibility(True, 2, ["a", "b"])
        assert m.eligible == {"a", "b"}
    finally:
        cleanup(m)


@pytest.mark.parametrize("partial", [False, True])
def test_terminal_budget_and_incomplete_normal_drain(partial):
    from test_serial_eager_owner import settle_row

    m = machine(eager=False)
    try:
        claims = admit(m)["claims"]
        m.verify_start([verify_row(c["proposal"]) for c in claims])
        row, final = feedback(claims[0]["proposal"], (10, 20), terminal=not partial)
        m.feedback([row])
        m.step()
        if partial:
            m.step()  # private normal proposal P2; never Target-ready
            assert "a" in m.normal and "a" not in m.ready
            m.begin_drain()
            assert "a" not in m.ready
            receipt = m.diagnostic_settle(settle_row("a", final, 1))
            assert receipt["discarded_proposal_tokens"] == 2
        else:
            assert "a" in m.backend.retired and "a" not in m.ready
        assert not m.normal
    finally:
        cleanup(m)


def test_finite_output_normal_and_refill_progress_after_priority():
    from test_serial_eager_owner import settle_row

    m = machine(ids=("a", "b"))
    try:
        active = ["a", "b"]
        seen = []
        for cycle in range(6):
            selection = admit(m, "A" if cycle % 2 == 0 else "B", 1, active)["claims"]
            rid = selection[0]["request_id"]
            seen.append(rid)
            m.verify_start([verify_row(selection[0]["proposal"])])
            terminal = cycle in (2, 4, 5)
            row, final = feedback(
                selection[0]["proposal"], m.requests[rid].committed_token_ids, terminal=terminal
            )
            m.feedback([row])
            run_work(m)
            if terminal:
                m.diagnostic_settle(settle_row(rid, final, row["round_id"] + 1, terminal=True))
                active.remove(rid)
                if rid == "a":
                    m.initialize("refill", (10, 20), token_prefix_hash((10, 20)))
                    m.register([{**proposal_row("refill"), "home_cohort": "A"}])
                    active.append("refill")
        assert seen == ["a", "a", "a", "b", "b", "refill"]
    finally:
        cleanup(m)


def test_lookahead_eos_is_uncommitted_and_has_no_fabricated_common_row():
    prefix = (10, 20)
    first = greedy_tokens(prefix, 1)
    eos = greedy_tokens(prefix + first, 2)[-1]
    m = PingPrePostMachine(
        Backend(NS(max_model_len=4096), worker=OwnerWorker()), request_ids=["a"]
    )
    m.initialize("a", prefix, token_prefix_hash(prefix))
    m.register([{**proposal_row("a"), "home_cohort": "A", "eos_token_ids": [eos]}])
    try:
        claim = admit(m)["claims"][0]
        m.verify_start([verify_row(claim["proposal"])])
        run_work(m)
        assert m.requests["a"].committed_token_ids == prefix
        row, final = feedback(claim["proposal"], prefix)
        m.feedback([row])
        count = len(m.backend.worker.calls)
        m.step()
        assert len(m.backend.worker.calls) == count
        next_claim = admit(m, "B")["claims"][0]
        assert next_claim["proposal"]["proposal_eos"]
        assert len(next_claim["proposal"]["proposal_token_ids"]) == 2
        settlement = next(e for e in m.ping_events.rows() if e["event"] == "settlement")
        assert settlement["common_start_ns"] is None
        m.verify_start([verify_row(next_claim["proposal"])])
        row, terminal = feedback(next_claim["proposal"], final, terminal=True)
        m.feedback([row])
        run_work(m)
        assert m.requests["a"].committed_token_ids == terminal
        assert terminal[-1] == eos and len(terminal) == len(prefix) + 3
        assert "a" in m.backend.retired and not m.ready
    finally:
        cleanup(m)
