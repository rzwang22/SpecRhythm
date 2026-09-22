"""Actual owner machine, physical materialize request sets and deferred READY boundary."""

import pytest
from test_k3 import machine
from test_ping_prepost import admit, run_work
from test_prepost_protocol import cleanup, feedback
from test_serial_eager_owner import verify_row


@pytest.mark.parametrize("policy", ["legacy", "unified"])
def test_real_materialize_mixes_rejection_seed_and_ordinary_extension(policy):
    m = machine(draft_dispatch=policy)
    try:
        m.update_eligibility(True, 1, ["a"])
        a = admit(m, "A", 1)["claims"][0]["proposal"]
        b = admit(m, "B", 1)["claims"][0]["proposal"]
        m.verify_start([verify_row(a), verify_row(b)])
        m.feedback([feedback(b, (10, 20))[0]])
        m.step()  # ordinary B seed + A lookahead; both existing production paths.
        m.feedback([feedback(a, (10, 20), reject=True)[0]])
        before = len(m.backend.worker.calls)
        m.step()  # A correction/seed and B extension share ONE real materialize.
        assert len(m.backend.worker.calls) == before + 1
        row = m.backend.prepost_forwards.rows()[-1]
        assert row["B"] == 2 and len(set(row["request_ids"])) == 2
        assert {b["task_kind"] for b in row["bindings"]} == {
            "parent_materialization", "normal_extension"}
        assert {b["source"] for b in row["bindings"]} == {"eager_rejection", "ordinary"}
        assert row["dispatch"]["no_other_executable_work"]
        assert all(r["selected"] for r in row["dispatch"]["inventory"])
        run_work(m)
        assert all(len(m.ready[r]["proposal"]["proposal_token_ids"]) == 3 for r in ("a", "b"))
        exclusive = m.backend.prepost_forwards.rows()[-1]
        assert exclusive["B"] == 1
        assert any(r["reason"] == "already_READY" for r in exclusive["dispatch"]["inventory"])
        assert exclusive["dispatch"]["no_other_executable_work"]
        ids = [r["physical_forward_id"] for r in m.backend.prepost_forwards.rows()]
        assert len(ids) == len(set(ids))
    finally:
        cleanup(m)
    assert not m.backend.worker.memory
    assert m.eager_report()["candidate_accounting"]["live_unsubmitted"] == 0


@pytest.mark.parametrize("policy", ["legacy", "unified"])
def test_promotion_publish_yields_to_owner_before_unrelated_recovery(policy):
    m = machine(draft_dispatch=policy)
    try:
        a = admit(m, "A", 1)["claims"][0]["proposal"]
        m.verify_start([verify_row(a)])
        run_work(m)
        b = admit(m, "B", 1)["claims"][0]["proposal"]
        m.verify_start([verify_row(b)])
        m.feedback([feedback(a, (10, 20))[0], feedback(b, (10, 20), reject=True)[0]])
        before = len(m.backend.worker.calls)
        m.step()
        assert "a" in m.ready and len(m.ready["a"]["proposal"]["proposal_token_ids"]) == 3
        if policy == "unified":
            assert len(m.backend.worker.calls) == before  # No fourth token/other recovery.
            assert "b" not in m.normal  # Still explicitly awaiting physical settlement.
            claim = admit(m, "A", 1)["claims"][0]
            assert claim["request_id"] == "a"
        else:
            assert len(m.backend.worker.calls) == before + 1
            assert "b" in m.normal
        run_work(m)
        assert "b" in m.ready
    finally:
        cleanup(m)


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("target_first", [False, True])
def test_unified_repeated_reject_then_success_conserves_k3(eager, target_first):
    # Exercise the same actual transition checks as the legacy test under opt-in.
    from unittest.mock import patch

    from test_k3 import test_k3_success_rejection_repeated_and_reuse
    original = machine
    with patch("test_k3.machine", side_effect=lambda *a, **kw:
               original(*a, **kw, draft_dispatch="unified")):
        test_k3_success_rejection_repeated_and_reuse(eager, target_first)


@pytest.mark.parametrize("budget", [1, 2, 3, 4, 7])
@pytest.mark.parametrize("eager", [False, True])
def test_unified_legal_short_tails_and_release(budget, eager):
    from unittest.mock import patch

    from test_k3 import test_k3_output_budget_complete_reference

    original = machine
    with patch("test_k3.machine", side_effect=lambda *a, **kw:
               original(*a, **kw, draft_dispatch="unified")):
        test_k3_output_budget_complete_reference(budget, eager)


def test_serial_gate_remains_even_with_promotion_ready():
    m = machine(mode="serial-eager-k3", draft_dispatch="unified")
    try:
        claim = admit(m, "A", 2)["claims"]
        m.verify_start([verify_row(c["proposal"]) for c in claim])
        run_work(m)
        m.feedback([feedback(claim[0]["proposal"], (10, 20))[0],
                    feedback(claim[1]["proposal"], (10, 20), reject=True)[0]])
        m.step()
        assert "a" in m.ready and m.has_work()
        with pytest.raises(Exception, match="before all Draft work is idle"):
            admit(m, "A", 2)
        run_work(m)
        assert len(admit(m, "A", 2)["claims"]) == 2
    finally:
        cleanup(m)


def test_physical_summary_uses_unique_calls_and_missing_is_not_zero():
    from specrhythm.serving.k3_physical_dispatch_evidence import summarize

    m = machine(draft_dispatch="unified")
    try:
        m.update_eligibility(True, 1, ["a"])
        a = admit(m, "A", 1)["claims"][0]["proposal"]
        b = admit(m, "B", 1)["claims"][0]["proposal"]
        m.verify_start([verify_row(a), verify_row(b)])
        m.feedback([feedback(b, (10, 20))[0]])
        m.step()
        m.feedback([feedback(a, (10, 20), reject=True)[0]])
        run_work(m)
        rows = m.backend.prepost_forwards.rows()
        # Native timing is an explicit CPU fixture; no claim of real GPU evidence.
        native = {r["start_ns"]: {"gpu_event_ms": 1.0} for r in rows}
        result = summarize(rows, native, 0, rows[-1]["end_ns"])
        assert result["status"] == "COMPLETE"
        assert result["unique_physical_calls"] == len(rows)
        assert sum(result["composition"].values()) == len(rows)
        assert result["recovery_mixed_calls"] > 0 and result["recovery_exclusive_calls"] == 1
        assert result["missed_compatible_calls"] == 0
        assert summarize(rows + rows[-1:], native, 0, rows[-1]["end_ns"])["status"] == "INCOMPLETE"
        assert summarize(rows, {}, 0, rows[-1]["end_ns"])["status"] == "INCOMPLETE"
        old = summarize([{**r, "dispatch": None} for r in rows], native, 0, rows[-1]["end_ns"])
        assert old["status"] == "NOT_COLLECTED" and old["missed_compatible_calls"] is None
    finally:
        cleanup(m)


def test_inventory_budget_fails_before_next_physical_call_and_can_drain():
    from specrhythm.continuation.trace import TRACE

    m = machine(eager=False, complete=False, draft_dispatch="unified")
    m.dispatch_counts[TRACE.phase] = 4096
    before = len(m.backend.worker.calls)
    try:
        with pytest.raises(Exception, match="inventory budget exhausted"):
            m.step()
        assert len(m.backend.worker.calls) == before
    finally:
        cleanup(m)
    assert not m.backend.worker.memory
