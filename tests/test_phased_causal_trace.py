"""Phase reserves, conservative overflow qualification and the actual control adapter."""

from types import SimpleNamespace

import pytest

from specrhythm.continuation.trace import PHASE_BUDGETS, CausalTrace
from specrhythm.serving import fixed_observe
from specrhythm.serving.eager_latency import trace_rows


def test_setup_overflow_cannot_consume_measurement_reserve_over_old_20000():
    trace = CausalTrace(True, layout="phased")
    for _ in range(22000):
        trace.event("setup", start_ns=1, end_ns=2)
    trace.follow_control({"diagnostic_phase": "measurement"})
    for i in range(28000):
        trace.event("measured", start_ns=100+i, end_ns=101+i)
    trace.follow_control({"diagnostic_phase": "drain"})
    trace.event("release", start_ns=30001, end_ns=30002)
    result = trace.report()
    assert result["status"] == "TRUNCATED"  # Never relabel lifetime evidence.
    assert result["row_limit"] == sum(PHASE_BUDGETS.values())
    assert result["phases"]["measurement"]["retained_rows"] == 28000
    assert result["phases"]["setup"]["dropped_rows"] == 22000-PHASE_BUDGETS["setup"]
    assert len(result["rows"]) <= result["row_limit"]
    scoped = trace_rows({"causal_timeline": result}, 100, 29000)
    assert scoped["status"] == "TRUNCATED" and scoped["measurement_status"] == "COMPLETE"
    assert len(scoped["rows"]) == 28000


def test_measurement_overflow_and_cross_boundary_lost_span_are_not_complete():
    trace = CausalTrace(True, layout="phased")
    for _ in range(PHASE_BUDGETS["setup"]):
        trace.event("setup", start_ns=1, end_ns=2)
    trace.event("cross_boundary", start_ns=2, end_ns=200)
    trace.follow_control({"diagnostic_phase": "measurement"})
    for _ in range(PHASE_BUDGETS["measurement"]+1):
        trace.event("token", start_ns=100, end_ns=101)
    report = trace.report()
    assert report["phases"]["measurement"]["dropped_rows"] == 1
    scoped = trace_rows({"causal_timeline": report}, 100, 200)
    assert scoped["measurement_status"] == "TRUNCATED"
    report["phases"]["measurement"]["dropped_bounds_ns"] = None
    assert trace_rows({"causal_timeline": report}, 100, 200)["measurement_status"] == "MISSING"


def test_real_control_observer_uses_existing_read_and_no_stale_phase_rollback(monkeypatch):
    trace = CausalTrace(True, layout="phased")
    monkeypatch.setattr(fixed_observe, "TRACE", trace)
    calls = []
    owner = SimpleNamespace(read=lambda packet: calls.append(packet) or packet)
    fixed_observe.wrap(owner, "read", "control_json_read")
    packet = {"diagnostic_phase": "warmup", "requests": {"r": {"state": "ACTIVE"}}}
    assert owner.read(packet) is packet  # No cached/substituted functional snapshot.
    with trace.span("across_transition", request_id="r"):
        owner.read({"diagnostic_phase": "measurement"})
    owner.read(packet)
    assert len(calls) == 3 and trace.phase == "measurement"
    row = trace.report()["rows"][0]
    assert row["diagnostic_phase"] == "warmup" and row["request_id"] == "r"
    assert not trace.local.fields
    with pytest.raises(ValueError):
        CausalTrace(True, layout="unbounded")
