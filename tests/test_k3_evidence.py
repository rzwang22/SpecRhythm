"""Collected owner/backend records with explicitly synthetic native endpoints."""

import copy
import json
import tarfile

import pytest
import test_ping_prepost_evidence as collector
from test_k3 import machine

from specrhythm.serving.k3 import MODES
from specrhythm.serving.k3_evidence import pipeline
from specrhythm.serving.ping_prepost_delivery import export
from specrhythm.serving.ping_prepost_evidence import mechanism


@pytest.mark.parametrize("mode", MODES)
def test_k3_real_records_report_export_and_missing_evidence(mode, monkeypatch, tmp_path):
    monkeypatch.setattr(collector, "machine", machine)
    runtime, backend = collector.collected(eager=mode == "pingpong-eager-k3")
    runtime["point"]["mode"] = mode  # Collector label only; actual machine above generated K3.
    result = mechanism(runtime, backend)
    assert result["status"] == "COMPLETE", result["errors"]
    assert all(r["P"] == 3 and r["next_P"] == 3 for c in result["cycles"] for r in c["requests"])
    assert result["generated"] == result["retained"] + result["discarded"]
    p = pipeline(runtime, backend)
    assert p["evidence_integrity"] == "COMPLETE", p["errors"]
    assert p["per_role_forwards"]["rejection_recovery"]["forward_count"] == 6
    assert p["recovery_timeline_status"] == "RECORDED"
    assert any(
        r["role"] == "rejection_recovery" and r["parent_proposal_id"] for r in p["timeline"]
    )
    if mode != "pingpong-eager-k3":
        assert p["per_role_forwards"]["eager_lookahead"]["forward_count"] == 0
    for key in ("ordinary_draft", "rejection_recovery"):
        assert p["cross_home_native_overlap"][key] == {"lower_ms": 0.0, "upper_ms": 0.0}
    # This simulated sequence has no concurrent other-home Target: zero is a real
    # result of the supplied synthetic intervals, never a GPU performance claim.
    assert p["cross_cohort_pipeline_behavior"] == "NOT_DEMONSTRATED"
    for name, value in [("runtime.json", runtime), ("draft-backend-report.json", backend)]:
        (tmp_path / name).write_text(json.dumps(value))
    archive = tmp_path / "k3.tar.gz"
    inv = export(tmp_path, archive, first_code=23, modes=MODES)
    assert inv["first_exit_code"] == 23
    # Owner-only collector intentionally has no real startup/final device snapshot.
    assert inv["export_status"] == "INCOMPLETE"
    with tarfile.open(archive) as t:
        paths = json.load(t.extractfile("inventory.json"))["logical_paths"]
        r, b = [
            json.load(t.extractfile(paths[n]))
            for n in ("runtime.json", "draft-backend-report.json")
        ]
    assert pipeline(r, b) == p and mechanism(r, b) == result
    broken = copy.deepcopy(backend)
    broken["fixed_device"]["forwards"].pop()
    assert pipeline(runtime, broken)["evidence_integrity"] == "INCOMPLETE"
    assert mechanism(runtime, broken)["status"] == "INCOMPLETE"
    bad = copy.deepcopy(backend)
    e = next(e for e in bad["prepost"]["pingpong"]["events"] if e["event"] == "ready")
    e["remaining_output_budget"] = 1
    assert mechanism(runtime, bad)["status"] == "INCOMPLETE"
    bad = copy.deepcopy(backend)
    bad["prepost"]["candidate_accounting"]["submitted"] += 1
    assert mechanism(runtime, bad)["status"] == "INCOMPLETE"
    from specrhythm.serving.k3_gpu_check import coverage

    assert coverage(runtime, backend)["status"] == "COMPLETE"
    with pytest.raises(Exception, match="candidate accounting"):
        coverage(runtime, bad)


@pytest.mark.parametrize(
    "fault",
    [
        "request",
        "rank_missing",
        "rank_duplicate",
        "version",
        "proposal",
        "clock",
        "draft_dependency",
    ],
)
def test_native_request_proposal_rank_dependency_loss_is_not_zero(monkeypatch, fault):
    monkeypatch.setattr(collector, "machine", machine)
    r, b = collector.collected()
    if fault == "request":
        del r["target_devices"][0]["device"]["forwards"][0]["internal_request_ids"]
    elif fault == "rank_missing":
        r["target_devices"].pop()
    elif fault == "rank_duplicate":
        r["target_devices"][1]["device"]["identity"]["global_rank"] = 0
    elif fault in ("version", "proposal"):
        c = r["target_steps"][0]["ping_admission"]["claims"][0]
        c["prefix_version" if fault == "version" else "proposal_id"] = 999
    elif fault == "draft_dependency":
        f = next(
            f
            for f in b["prepost_physical"]["forwards"]
            if any(x.get("work_kind") == "lookahead" for x in f["bindings"])
        )
        f["bindings"][0]["round_id"] = 999
    else:
        del r["target_devices"][0]["device"]["forwards"][0]["start_lower_ns"]
    value = pipeline(r, b)
    assert value["evidence_integrity"] == "INCOMPLETE" and value["errors"]
    assert value["cross_request_native_overlap"]["rejection_recovery"] is None
    assert value["parent_eager_native_overlap"] is None
    assert value["cross_request_pipeline_behavior"] == "INCOMPLETE"


def test_other_request_overlap_precedes_home_classification_and_is_not_double_counted(monkeypatch):
    monkeypatch.setattr(collector, "machine", machine)
    r, b = collector.collected()
    parent = pipeline(r, b)
    assert parent["parent_eager_native_overlap"]["lower_ms"] > 0
    assert parent["rejection_cycle"]["status"] == "COMPLETE"
    assert {x["event"] for x in parent["rejection_cycle"]["rows"]} >= {
        "READY",
        "claim",
        "Target_forward",
        "feedback",
        "Draft_forward",
        "next_READY",
    }
    row = next(x for x in parent["timeline"] if x["role"] == "rejection_recovery")
    rid = row["request_id"]
    other = next(
        s
        for s in r["target_steps"]
        if rid not in {c["request_id"] for c in s["ping_admission"]["claims"]}
    )
    gpu = next(
        g
        for g in r["target_devices"][0]["device"]["forwards"]
        if other["start_ns"] <= g["host_start_ns"] <= other["end_ns"]
    )
    d = b["fixed_device"]["forwards"][row["native_forward_index"]]
    # Explicit simulated device endpoints; collection and logical claims remain real.
    for key in ("start_lower_ns", "start_upper_ns", "end_lower_ns", "end_upper_ns"):
        d[key] = gpu[key]
    for e in b["prepost"]["pingpong"]["events"]:
        for c in e.get("claims", []):
            c["home_cohort"] = "A"
    b["prepost"]["pingpong"]["home_cohorts"] = dict.fromkeys(
        b["prepost"]["pingpong"]["home_cohorts"], "A"
    )
    result = pipeline(r, b)
    overlap = result["cross_request_native_overlap"]["rejection_recovery"]
    assert overlap["lower_ms"] == (gpu["end_lower_ns"] - gpu["start_lower_ns"]) / 1e6
    assert result["cross_home_native_overlap"]["rejection_recovery"]["upper_ms"] == 0
    assert result["cross_request_pipeline_behavior"] == "OBSERVED"
    assert result["cross_cohort_pipeline_behavior"] == "NOT_DEMONSTRATED"
    # Two TP ranks and mixed role participants never multiply wall-clock overlap.
    assert overlap["lower_ms"] == overlap["upper_ms"]


def test_target_claim_must_match_actual_owner_proposal(monkeypatch):
    monkeypatch.setattr(collector, "machine", machine)
    r, b = collector.collected()
    r = copy.deepcopy(r)
    c = r["target_steps"][0]["ping_admission"]["claims"][0]
    c["proposal_id"] = "different-owner-proposal"
    c["proposal"]["runtime_provenance"]["rolling_proposal_id"] = c["proposal_id"]
    result = pipeline(r, b)
    assert result["evidence_integrity"] == "INCOMPLETE"
    assert any("authoritative owner" in e for e in result["errors"])
    assert result["parent_eager_native_overlap"] is None
