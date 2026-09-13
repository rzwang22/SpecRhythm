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
