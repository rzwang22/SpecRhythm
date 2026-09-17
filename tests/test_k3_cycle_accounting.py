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
