"""Closed endpoint statistics, with production protocol records and honest missing data."""

import json

import pytest
import test_ping_prepost_evidence as collector
from test_k3 import machine

from specrhythm.serving.k3_cycle_evidence import (
    cycle_report,
    ledger,
    partition,
    summarize_ledgers,
)


def test_same_sample_boundaries_explain_apparent_negative_step_residual():
    # claim occurs BEFORE complete-step; RPC completion/output commit is AFTER feedback start.
    landmark_values = ledger(
        dict(claim=0, step_start=98, gpu_start=238, gpu_end=350, feedback=414, step_end=493)
    )
    assert landmark_values["closure_residual_ns"] == 0
    assert 414 - 98 + 79 == 493 - 98
    assert landmark_values["segments_ns"]["claim->step_start"] == 98
    assert ledger(dict(start=0, end=None))["status"] == "INCOMPLETE"
    assert ledger(dict(start=5, end=4))["status"] == "INCOMPLETE"
    assert ledger(dict(start=True, end=4))["status"] == "INCOMPLETE"


def test_nested_spans_and_crossing_spans_not_double_added():
    def span(name, a, b):
        return dict(category=name, start_ns=a, end_ns=b)

    p = partition([span("parent", 1, 9), span("child", 2, 4), span("child", 6, 8)], 0, 10)
    assert p["exclusive_ms"] == {"unaccounted": 2e-6, "parent": 4e-6, "child": 4e-6}
    assert p["inclusive_union_ms"] == {"parent": 8e-6, "child": 4e-6}
    assert p["closure_residual_ns"] == 0
    p = partition([span("a", 1, 5), span("b", 3, 7)], 0, 10)
    assert p["exclusive_ms"]["ambiguous_crossing_spans"] == 2e-6


def test_complete_sample_summary_and_predeclared_representatives():
    rows = [
        dict(key=i, ledger=ledger(dict(start=10, middle=12, end=10 + i))) for i in range(2, 12)
    ]
    rows.append(dict(key=13, ledger=ledger(dict(start=10, middle=None, end=40))))
    r = summarize_ledgers(rows)
    assert (r["complete"], r["incomplete"]) == (10, 1)
    assert r["representatives"]["median"]["key"] == 6
    assert r["representatives"]["p90"]["key"] == 10
    assert r["max_absolute_closure_residual_ns"] == 0


@pytest.mark.parametrize("eager", [False, True])
def test_actual_owner_versions_and_refill_never_fabricate_missing_landmarks(monkeypatch, eager):
    monkeypatch.setattr(collector, "machine", machine)
    r, b = collector.collected(eager=eager, mode="pingpong-eager-k3" if eager else "pingpong-k3")
    # Actual owner/backend protocol records, synthetic GPU endpoints. This older
    # collector has no Target RPC capture; the new report must NOT claim closure.
    value = cycle_report(r, b)
    assert value["association_errors"]
    assert value["request_cycle_summary"]["complete"] == 0
    assert value["request_cycle_summary"]["incomplete"] > 0
    assert value["excluded"]["initial_refill_or_missing_predecessor"] > 0
    assert value == cycle_report(json.loads(json.dumps(r)), json.loads(json.dumps(b)))
    for c in value["request_cycles"]:
        assert c["eligibility"]["ready_version"] == c["key"][1]
        assert "NOT_OBSERVED" in c["eligibility"]["earliest_legal_time"]
        assert c["ledger"]["status"] == "INCOMPLETE"


def collected_clock_fixture(monkeypatch):
    """Actual five-round owner transitions; only the worker/transport clocks are synthetic."""
    monkeypatch.setattr(collector, "machine", machine)
    r, b = collector.collected(eager=True, mode="pingpong-eager-k3")
    r.update(measurement_start_ns=0, measurement_end_ns=6000)
    target_trace, draft_trace = [], []
    r["host"] = {"intervals": []}
    events = b["prepost"]["pingpong"]["events"]
    for i, s in enumerate(r["target_steps"]):
        t = i * 1000 + 100
        s.update(start_ns=t + 20, end_ns=t + 100, schedule_start_ns=t + 21, schedule_end_ns=t + 30)
        claims = s["ping_admission"]["claims"]
        s["ping_admission"]["target_available_observed_ns"] = t
        keys = {(c["request_id"], c["prefix_version"]) for c in claims}
        request_refs = [dict(request_id=rid, round_id=v) for rid, v in keys]
        for c in claims:
            c.update(claimed_ns=t + 10, eligibility_observed_ns=t + 9)
        for e in events:
            if e["event"] == "ready" and (e["request_id"], e["prefix_version"]) in keys:
                e.update(ready_ns=t + 1, timestamp_ns=t + 2)
            if e["event"] == "feedback" and any(
                (x["request_id"], x["round_id"]) in keys for x in e["rows"]
            ):
                e["owner_feedback_ns"] = t + 84
        for rank, d in enumerate(r["target_devices"]):
            g = d["device"]["forwards"][i]
            g.update(
                host_start_ns=t + 50,
                start_lower_ns=t + 52,
                start_upper_ns=t + 53,
                end_lower_ns=t + 69,
                end_upper_ns=t + 70,
                physical_forward_id=f"rank{rank}:{i}",
            )
        for category, offset in [
            ("target_sampled_results_received", 75),
            ("target_feedback_payload_ready", 80),
            ("transport_exchange", 81),
        ]:
            target_trace.append(
                dict(
                    category=category,
                    start_ns=t + offset,
                    end_ns=t + offset,
                    operation="pp_feedback",
                    requests=request_refs,
                    service_receive_ns=t + 82,
                )
            )
        draft_trace.append(
            dict(
                category="owner_dequeue",
                start_ns=t + 83,
                end_ns=t + 83,
                operation="pp_feedback",
                requests=request_refs,
            )
        )
    r["target_devices"][0]["host"] = {"causal_timeline": {"rows": target_trace}}
    b["fixed_host"] = {"causal_timeline": {"rows": draft_trace}}
    return r, b


def test_real_owner_same_version_pairing_complete_and_missing(monkeypatch):
    import copy
    from specrhythm.serving.k3_cycle_evidence import compact_report

    r, b = collected_clock_fixture(monkeypatch)
    v = cycle_report(r, b)
    assert not v["association_errors"]
    assert v["request_cycle_summary"]["complete"] > 0
    assert v["request_cycle_summary"]["incomplete"] == 0
    assert v["request_cycle_summary"]["max_absolute_closure_residual_ns"] == 0
    from collections import Counter

    homes = Counter(home for step in v["steps"] for home in step["home_request_counts"])
    for home, count in homes.items():
        if count > 1:
            assert v["cadence_cycle_summary"][home]["complete"] == count - 1
    assert v["cadence_cycle_summary"]["global"]["complete"] == 4
    c = compact_report(v)
    assert "request_cycles" not in c and "cadence_cycles" not in c
    assert c["request_cycle_summary"] == v["request_cycle_summary"]
    for cycle in v["request_cycles"]:
        assert cycle["key"][1] > 0
        assert cycle["landmarks_ns"]["READY_published"] < cycle["landmarks_ns"]["claim"]
    bad = copy.deepcopy(r)
    del bad["target_devices"][0]["host"]["causal_timeline"]["rows"][4]
    assert cycle_report(bad, b)["request_cycle_summary"]["incomplete"] > 0
    bad = copy.deepcopy(r)
    bad["measurement_start_ns"] = 2500
    assert cycle_report(bad, b)["excluded"]["window_boundary"] > 0
    bad = copy.deepcopy(r)
    bad["target_steps"][2]["ping_admission"]["claims"][0]["prefix_version"] = 999
    assert cycle_report(bad, b)["association_errors"]
    bad = copy.deepcopy(b)
    e = next(
        e
        for e in bad["prepost"]["pingpong"]["events"]
        if e["event"] == "ready" and e["prefix_version"] > 0
    )
    bad["prepost"]["pingpong"]["events"].append(copy.deepcopy(e))
    assert cycle_report(r, bad)["request_cycle_summary"]["incomplete"] > 0


def test_lossless_repeated_scope_factoring_preserves_missing():
    from specrhythm.serving.k3_cycle_evidence import factor_wait_scope

    rows = [dict(request_id=str(i), wait_breakdown={'scope': 'observed partial', 'wait': i})
            for i in range(128)]
    original = dict(dispatch={'rows': [dict(request_versions=rows)]})
    compact = factor_wait_scope(original)
    assert compact['dispatch']['request_wait_scope'] == 'observed partial'
    for r, old in zip(compact['dispatch']['rows'][0]['request_versions'], rows):
        assert 'scope' not in r['wait_breakdown']
        r['wait_breakdown']['scope'] = compact['dispatch']['request_wait_scope']
        assert r == old
    del rows[0]['wait_breakdown']['scope']
    assert factor_wait_scope(original) == original  # genuinely missing stays missing
