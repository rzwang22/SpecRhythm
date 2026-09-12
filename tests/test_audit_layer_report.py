"""Compact report uses original windows; no duplicate raw events or cross-run UUID gate."""

import copy
import json

import pytest
from test_eager_evidence_export import existing as _existing
from test_eager_retest_report import inputs as _inputs

from specrhythm.serving.audit_layer_report import analyze, compare, exclusive_lanes, main, write

existing = _existing
inputs = _inputs


def test_disjoint_thread_sweep_keeps_nested_checks_and_other_threads_separate():
    rows = [
        dict(pid=1, thread_id=1, category="physical_gpu_audit", start_ns=0, end_ns=100),
        dict(pid=1, thread_id=1, category="resident_block_audit", start_ns=20, end_ns=70),
        dict(pid=1, thread_id=2, category="ipc", start_ns=10, end_ns=90),
    ]
    result = exclusive_lanes(rows, 0, 100)
    a, b = result["lanes"]
    assert a["exclusive_ms"]["resident_block_audit"] == 50 / 1e6
    assert a["exclusive_ms"]["physical_gpu_audit"] == 50 / 1e6
    assert b["exclusive_ms"]["ipc"] == 80 / 1e6
    assert sum(b["exclusive_ms"].values()) == 100 / 1e6
    assert result["status"] == "COMPLETE"
    rows[0].pop("thread_id")
    assert exclusive_lanes(rows, 0, 100)["status"] == "MISSING_LANES"


def point(inputs):
    root, _ = inputs
    p = root / "runs/actual-ended-serial-eager"
    runtime, backend, light = [
        json.loads((p / name).read_text())
        for name in ("runtime.json", "draft-backend-report.json", "light-summary.json")
    ]
    light.update(
        formal_comparison_eligible=True,
        mode="serial-eager",
        measured_window_ms=300,
        execution_status="PASS",
        measurement_status="PASS",
        cleanup_status="PASS",
    )
    return runtime, backend, light


def test_compact_analysis_retains_metrics_and_explicit_missing_trace(inputs):
    r, b, light = point(inputs)
    value = analyze(r, b, light)
    assert value["throughput_tok_s"] == 100
    assert value["tokens_per_step"] == 10
    assert value["native_forward_overlap"] == {"lower_ms": 10, "upper_ms": 10}
    assert value["trace_coverage"]["draft"]["status"] == "MISSING"
    assert value["inclusive_host_lanes"]["draft"]["prefix_hash_and_block_record"]["union_ms"] == 10
    assert value["inclusive_host_lanes"]["draft"]["json_serialization"]["union_ms"] == 6
    payload = json.dumps(value)
    assert "host_events" not in payload and "source_forward_index" not in payload
    assert len(payload) < 50000
    b["fixed_host"]["causal_timeline"] = {
        "mode": "light",
        "status": "TRUNCATED",
        "rows": [],
        "dropped_rows": 100,
    }
    assert analyze(r, b, light)["trace_coverage"]["draft"]["dropped_rows"] == 100
    b["fixed_device"]["forwards"] = [
        f for f in b["fixed_device"]["forwards"] if f.get("purpose") != "eager"
    ]
    absent = analyze(r, b, light)
    assert absent["first_forward_classes"] == {"NO_EAGER_FORWARD_IN_RECORD": light["target_steps"]}
    assert absent["cycles"][0]["before_first_forward_host"] == {}


def test_four_point_same_sha_config_comparison_is_bounded(inputs, tmp_path):
    value = analyze(*point(inputs))
    paths = []
    for i, (mode, audit) in enumerate(
        (
            ("serial", "full"),
            ("serial-eager", "full"),
            ("serial", "runtime"),
            ("serial-eager", "runtime"),
        )
    ):
        p = tmp_path / f"p{i}.json"
        row = copy.deepcopy(value)
        row.update(
            mode=mode,
            source_commit="a" * 40,
            workload_sha256="same",
            options={"draft_audit": audit, "window_seconds": 30, "observation": "buffered-live"},
        )
        row["draft_audit"]["mode"] = audit
        # Different runs legitimately have different UUIDs; no cross-run identity check.
        row["draft_device"]["identity"]["gpu_uuid"] = f"GPU-draft-point-{i}"
        write(row, p)
        paths.append(p)
    output = tmp_path / "four.json"
    compare(paths, output, "a" * 40)
    assert len(json.loads(output.read_text())["points"]) == 4
    assert output.stat().st_size < 200000
    bad = json.loads(paths[-1].read_text())
    bad["options"]["observation"] = "different"
    paths[-1].write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="options differ"):
        compare(paths, tmp_path / "bad.json", "a" * 40)


def test_report_failure_exports_missing_reason_without_overwriting_source(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    output = tmp_path / "error.json"
    with pytest.raises(SystemExit) as error:
        main(["--root", str(root), "--output", str(output), "--expected-commit", "a" * 40])
    assert error.value.code == 1
    assert json.loads(output.read_text())["evidence_status"] == "MISSING_INVALID_OR_LIMIT"
    assert list(root.iterdir()) == []
