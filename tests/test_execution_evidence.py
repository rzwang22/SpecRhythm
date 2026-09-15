"""Missing evidence is not zero or execution failure; comparison preserves paired scope."""

import copy
import json

import pytest
from test_audit_layer_report import existing as _existing
from test_audit_layer_report import inputs as _inputs
from test_audit_layer_report import point

from specrhythm.serving.audit_layer_report import analyze, write
from specrhythm.serving.common import DataError
from specrhythm.serving.execution_evidence import compare, qualify

existing = _existing
inputs = _inputs


def qualified(inputs):
    row = analyze(*point(inputs))
    row["coordinator_exclusive"]["status"] = "COMPLETE"
    row["draft_audit"] = dict(
        mode="runtime",
        schema_version="specrhythm.draft-audit.v1",
        full_boundaries=[
            dict(boundary=b)
            for b in ("setup_complete_snapshot", "timing_entry", "before_final_release")
        ],
    )
    for c in row["execution_path"]["cycles"]:
        c["latencies_ms"] = dict.fromkeys(c["latencies_ms"], 1.0)
    for t in row["trace_coverage"].values():
        t.update(
            mode="light",
            layout="phased",
            status="COMPLETE",
            dropped_rows=0,
            measurement_status="COMPLETE",
        )
    return row


def test_missing_landmarks_not_fabricated_and_failure_layer_preserves_run(inputs):
    row = analyze(*point(inputs))
    path = row["execution_path"]
    assert all(
        c["latencies_ms"]["target_GPU_end_upper_to_sampled_hook"] is None for c in path["cycles"]
    )
    assert qualify(row)["failure_layer"] == "diagnostic_evidence"
    assert qualify(row)["original_qualification"]["execution_status"] == "PASS"
    row = qualified(inputs)
    assert qualify(row)["diagnostic_integrity"] == "COMPLETE"
    row["trace_coverage"]["draft"].update(
        status="TRUNCATED", dropped_rows=7897, measurement_status="TRUNCATED"
    )
    failure = qualify(row)
    assert "7897" in failure["errors"][0] and "draft" in failure["errors"][0]
    assert failure["original_qualification"]["measurement_status"] == "PASS"


def test_two_versions_require_same_options_both_baselines_and_no_raw_duplication(inputs, tmp_path):
    row = qualified(inputs)
    paths = []
    commits = ["a" * 40, "b" * 40]
    for sha in commits:
        for mode in ("serial", "serial-eager"):
            r = copy.deepcopy(row)
            r.update(
                source_commit=sha,
                mode=mode,
                workload_sha256="frozen",
                options=dict(draft_audit="runtime", observation="buffered-live"),
            )
            r["draft_audit"]["mode"] = "runtime"
            r["draft_device"]["identity"]["gpu_uuid"] = sha + mode
            p = tmp_path / (sha + mode + ".json")
            write(r, p)
            paths.append(p)
    value = compare(paths, commits)
    assert len(value["points"]) == 4 and not value["raw_events_embedded"]
    with pytest.raises(DataError, match="incomplete paired"):
        compare(paths[:-1], commits)
    r = json.loads(paths[-1].read_text())
    r["options"]["observation"] = "other"
    paths[-1].write_text(json.dumps(r))
    with pytest.raises(DataError, match="options/workload"):
        compare(paths, commits)
