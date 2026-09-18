"""Offline source protections, compact evidence and actual interval arithmetic."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from specrhythm.serving.eager_evidence_export import (
    MAX_FILE,
    MAX_ROWS,
    MAX_TOTAL,
    SOURCE_COMMIT,
    Reader,
    bounds,
    device_summary,
    digest,
    duration,
    export,
    intersect,
    select_cycles,
)

NS = 1000000


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def forward(first, last, purpose="target"):
    return {"host_start_ns": first*NS, "host_launch_end_ns": last*NS,
            "start_lower_ns": first*NS, "start_upper_ns": first*NS,
            "end_lower_ns": last*NS, "end_upper_ns": last*NS,
            "gpu_event_ms": last-first, "B": 16, "purpose": purpose,
            "internal_request_ids": ["r"]}


def device(rank, rows):
    return {"identity": {"global_rank": rank, "gpu_uuid": f"GPU-{rank}"},
            "forwards": rows, "anchor_before_ns": 1, "anchor_after_ns": 1,
            "anchor_uncertainty_ns": 0, "projection_version": "integer-anchor-v2"}


@pytest.fixture
def existing(tmp_path):
    root = tmp_path / "existing"
    write(root / "scan-config.json", {"execution": {"git_commit": SOURCE_COMMIT}})
    for mode in ("serial", "serial-eager"):
        path = root / "runs" / ("actual-ended-" + mode)
        point = {"mode": mode, "batch": 16, "kind": "decode-scan", "scan": True,
                 "probe": False}
        write(path / "point.json", point)
        write(path / "light-summary.json", {
            "git_commit": SOURCE_COMMIT, "mode": mode, "batch": 16, "point": point,
            "valid": True, "measurement_status": "PASS", "cleanup_status": "PASS",
            "measurement_snapshot": {"measurement_start_ns": 100*NS,
                                     "measurement_end_ns": 400*NS},
        })
        write(path / "exit-code.json", {"effective_exit_code": 0})
        write(path / "process-lifecycle.json", {"exit_monotonic_ns": 500*NS,
                                               "owned_cleanup_completed": True,
                                               "remaining_owned_pids": []})
        write(path / "runtime.json", {
            "target_steps": [{"start_ns": n*NS, "end_ns": (n+100)*NS,
                              "B": 16, "step": i, "window": True}
                             for i, n in enumerate((100, 200, 300))],
            "host": {"intervals": [{"category": "scheduler", "start_ns": 100*NS,
                                     "end_ns": 110*NS}]},
            "target_devices": [{"device": device(rank, [forward(110, 130)]),
                                "host": {"intervals": []}, "rounds": []}
                               for rank in (0, 1)],
        })
        events = [{"phase": kind, "request_ids": ["r"], "start_ns": stamp*NS,
                   "end_ns": stamp*NS, "counter_delta": {kind: 1}}
                  for kind, stamp in (("promotions", 150), ("parent_rejections", 210),
                                      ("recovery_jobs", 220), ("admissions", 230),
                                      ("bridge_mismatches", 350))]
        write(path / "draft-backend-report.json", {
            "fixed_device": device(2, [forward(120, 140, "eager"),
                                        forward(320, 330, "commit")]),
            "fixed_host": {"intervals": []}, "rolling_eager": {"events": events},
            "eager_gpu_steps": [{"host_start_ns": 120*NS, "host_end_ns": 145*NS,
                                 "request_ids": ["r"], "B": 16, "worker_fenced": True}],
            "fixed_proposals": [],
        })
        row = {"request_id": "r", "round_id": 4, "source_continuation_id": "c",
               "proposal_token_ids": [1, 2, 3, 4],
               "timeline": {"verify_start_ns": 110*NS, "verify_end_ns": 130*NS}}
        row["record_sha256"] = digest(row)
        (path / "round-events.jsonl").write_text(json.dumps(row) + "\n")
        (path / "draft-work-events.jsonl").write_text(json.dumps({
            "operation": "eager_enqueue", "received_ns": 115*NS,
            "completed_ns": 116*NS, "success": True,
        }) + "\n")
    return root, tmp_path / "new-export.json"


def snapshot(root):
    return {str(p): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
            for p in root.rglob("*") if p.is_file()}


def test_export_preserves_read_only_source_and_rank_union_evidence(existing):
    root, output = existing
    before = snapshot(root)
    for path in root.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)
    result = export(root, output)
    assert result["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert snapshot(root) == before
    value = json.loads(output.read_text())
    eager = value["points"]["serial-eager"]
    assert eager["original_result"]["measurement_status"] == "PASS"
    assert eager["recomputed_eager_overlap"]["lower_ms"] == 10
    assert eager["recomputed_eager_overlap"]["upper_ms"] == 10
    assert eager["devices"]["target-0"]["by_purpose"]["target"][
        "launch_selected_event_sum_ms"] == 20
    assert eager["devices"]["draft"]["by_purpose"]["eager"][
        "launch_selected_B_histogram"] == {"16": 1}
    assert eager["same_request_trajectory"]["request_id"] == "r"
    cycle = eager["selected_cycles"][0]
    row = cycle["streams"]["round-events.jsonl"]["rows"][0]
    assert row["proposal_token_ids"] == {"count": 4, "sha256": digest([1, 2, 3, 4])}
    assert row["source_continuation_id"] == "c"
    assert cycle["streams"]["draft-work-events.jsonl"]["rows"][0]["operation"] == (
        "eager_enqueue"
    )
    assert any(r["status"] == "MISSING" for r in value["inventory"])
    assert eager["field_availability"]["owner_dequeue_ns"]["status"] == "MISSING"


def test_output_collision_and_inside_root_rejected(existing):
    root, output = existing
    output.write_text("keep")
    with pytest.raises(ValueError, match="collision"):
        export(root, output)
    assert output.read_text() == "keep"
    with pytest.raises(ValueError, match="outside"):
        export(root, root / "forbidden.json")
    assert not (root / "forbidden.json").exists()


def test_dangling_output_symlink_is_a_collision(existing):
    root, output = existing
    target = output.with_name("must-not-be-created.json")
    output.symlink_to(target)
    with pytest.raises(ValueError, match="collision"):
        export(root, output)
    assert output.is_symlink() and not target.exists()


@pytest.mark.parametrize("target", [
    "scan-config.json", "runs/actual-ended-serial/light-summary.json",
])
def test_source_commit_mismatch_fails_before_export(existing, target):
    root, output = existing
    path = root / target
    value = json.loads(path.read_text())
    value.get("execution", value)["git_commit"] = "bad"
    write(path, value)
    with pytest.raises(ValueError, match="SHA|identity"):
        export(root, output)
    assert not output.exists()


def test_running_point_not_read_as_completed(existing):
    root, output = existing
    path = root / "runs/actual-ended-serial/process-lifecycle.json"
    write(path, {"owned_cleanup_completed": False, "remaining_owned_pids": [10]})
    with pytest.raises(ValueError, match="not ended"):
        export(root, output)
    assert not output.exists()


def test_missing_raw_does_not_replace_original_pass(existing):
    root, output = existing
    for path in (root / "runs").glob("*/*"):
        if path.name in ("runtime.json", "draft-backend-report.json") or path.suffix == ".jsonl":
            path.unlink()
    export(root, output)
    result = json.loads(output.read_text())
    for point in result["points"].values():
        assert point["original_result"]["measurement_status"] == "PASS"
        assert point["recomputed_eager_overlap"]["status"] == "MISSING"
        assert point["selected_cycles"] == []
        assert point["host"]["coordinator"]["status"] == "MISSING"
    assert any(r["path"].endswith("runtime.json") and r["status"] == "MISSING"
               for r in result["inventory"])


@pytest.mark.parametrize("limits", [{"max_file_bytes": MAX_FILE+1},
                                    {"max_total_bytes": MAX_TOTAL+1},
                                    {"max_jsonl_rows": MAX_ROWS+1}, {"selected_cycles": 9}])
def test_hard_limits_cannot_be_raised(existing, limits):
    with pytest.raises(ValueError, match="hard caps"):
        export(*existing, **limits)


def test_reader_limits_hashes_and_row_truncation(tmp_path):
    root = tmp_path.resolve()
    (root / "big.json").write_text("x" * 100)
    (root / "rows.jsonl").write_text('{}\n'*5)
    reader = Reader(root, 80, 90, 2)
    assert reader.read(root / "big.json") is None
    assert reader.inventory["big.json"]["status"] == "LIMIT"
    assert reader.inventory["big.json"]["sha256"] is None
    assert reader.read(root / "rows.jsonl", jsonl=True) == [{}, {}]
    assert reader.inventory["rows.jsonl"]["truncated"]
    assert reader.inventory["rows.jsonl"]["parsed_rows"] == 2
    assert reader.used == 15


def test_jsonl_corruption_and_escape_are_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    write(root / "corrupt.jsonl", {"record_sha256": "wrong"})
    reader = Reader(root, MAX_FILE, MAX_TOTAL, MAX_ROWS)
    with pytest.raises(ValueError, match="partial"):
        reader.read(root / "corrupt.jsonl", jsonl=True)
    (root / "corrupt.jsonl").write_text('{"record_sha256":"wrong"}\n')
    with pytest.raises(ValueError, match="checksum"):
        reader.read(root / "corrupt.jsonl", jsonl=True)
    write(tmp_path / "outside.json", {})
    (root / "escape.json").symlink_to(tmp_path / "outside.json")
    with pytest.raises(ValueError, match="escaped"):
        reader.read(root / "escape.json")


def test_interval_union_clips_edges_and_deduplicates_ranks():
    rows = [forward(90, 130), forward(120, 150)]
    assert duration(bounds(rows, 100*NS, 140*NS, False)) == 40
    assert duration(intersect([(0, 10*NS)]*2, [(5*NS, 15*NS)])) == 5
    assert duration(intersect([(0, 10*NS)], [(11*NS, 15*NS)])) == 0
    bad = {**rows[0], "start_upper_ns": 200*NS}
    with pytest.raises(ValueError, match="INVALID"):
        bounds([bad], 100*NS, 300*NS, False)


def test_cycle_details_are_bounded(existing):
    root, output = existing
    path = root / "runs/actual-ended-serial/round-events.jsonl"
    row = json.loads(path.read_text())
    path.write_text((json.dumps(row) + "\n") * 600)
    export(root, output, selected_cycles=1)
    point = json.loads(output.read_text())["points"]["serial"]
    assert len(point["selected_cycles"]) == 1
    detail = point["selected_cycles"][0]["streams"]["round-events.jsonl"]
    assert len(detail["rows"]) == 512 and detail["truncated"]


def test_window_selected_before_caps_and_cycle_host_uses_all_categories(existing):
    root, output = existing
    path = root / "runs/actual-ended-serial-eager"
    runtime = json.loads((path / "runtime.json").read_text())
    setup = [{"category": "setup", "start_ns": 0, "end_ns": 10*NS}] * 20
    hashes = [{"category": "prefix_hash", "start_ns": 100*NS, "end_ns": 105*NS}] * 600
    late = [{"category": "ipc", "start_ns": 105*NS, "end_ns": 110*NS}]
    runtime["host"]["intervals"] = setup + hashes + late
    for rank in runtime["target_devices"]:
        rank["device"]["forwards"] = [forward(0, 10)] * 20 + rank["device"]["forwards"]
    write(path / "runtime.json", runtime)
    backend = json.loads((path / "draft-backend-report.json").read_text())
    backend["fixed_device"]["forwards"] = [forward(0, 10, "prefill")]*20 + (
        backend["fixed_device"]["forwards"]
    )
    write(path / "draft-backend-report.json", backend)
    export(root, output, max_jsonl_rows=2)
    point = json.loads(output.read_text())["points"]["serial-eager"]
    assert point["devices"]["draft"]["source_forward_count"] == 22
    assert point["devices"]["draft"]["status"] == "COMPLETE"
    assert point["recomputed_eager_overlap"]["lower_ms"] == 10
    cycle = point["selected_cycles"][0]
    assert cycle["host_full_cycle"]["coordinator"]["by_category"]["ipc"][
        "clipped_union_ms"] == 5
    assert cycle["before_first_eager_host_launch"]["boundary_ns"] == [100*NS, 120*NS]
    assert cycle["before_first_eager_host_launch"]["host"]["coordinator"]["by_category"][
        "ipc"]["clipped_union_ms"] == 5
    assert cycle["streams"]["coordinator-host"]["truncated"]


def test_invalid_native_duration_is_not_missing_or_zero(existing):
    root, output = existing
    path = root / "runs/actual-ended-serial-eager/draft-backend-report.json"
    backend = json.loads(path.read_text())
    backend["fixed_device"]["forwards"][0]["gpu_event_ms"] = -1
    write(path, backend)
    export(root, output)
    point = json.loads(output.read_text())["points"]["serial-eager"]
    assert point["recomputed_eager_overlap"]["status"] == "INVALID"
    assert point["original_result"]["measurement_status"] == "PASS"


def test_collector_claim_is_distinct_from_source_proof(existing, monkeypatch):
    root, output = existing
    monkeypatch.setenv("SR_EAGER_DIAG_COMMIT", "diagnostic-collector-claim")
    export(root, output)
    result = json.loads(output.read_text())
    assert result["source_commit"] == SOURCE_COMMIT
    assert result["collector_commit_claimed"] == "diagnostic-collector-claim"
    assert result["exporter_code_sha256"] == hashlib.sha256(Path(
        "src/specrhythm/serving/eager_evidence_export.py"
    ).read_bytes()).hexdigest()


def test_host_launch_and_physical_window_selections_are_independent():
    host_only = {**forward(500, 510, "eager"), "host_start_ns": 120*NS}
    physical_only = {**forward(130, 140, "eager"), "host_start_ns": 90*NS}
    result = device_summary(device(2, [host_only, physical_only]), 100*NS, 400*NS, 10)
    purpose = result["by_purpose"]["eager"]
    assert purpose["host_launch_selected_forwards"] == 1
    assert purpose["launch_selected_event_sum_ms"] == 10
    assert purpose["window_intersecting_forwards"] == 1
    assert purpose["clipped_union_upper_ms"] == 10
    rows = {r["source_forward_index"]: r for r in result["forwards"]}
    assert rows[0]["host_launch_selected"] and not rows[0]["physical_window_intersects"]
    assert not rows[1]["host_launch_selected"] and rows[1]["physical_window_intersects"]


def test_host_only_export_cap_does_not_invalidate_physical_zero(existing):
    root, output = existing
    path = root / "runs/actual-ended-serial-eager/draft-backend-report.json"
    backend = json.loads(path.read_text())
    backend["fixed_device"]["forwards"] = [
        {**forward(500+i*20, 510+i*20, "eager"), "host_start_ns": 120*NS}
        for i in range(4)
    ]
    write(path, backend)
    export(root, output, max_jsonl_rows=1)
    point = json.loads(output.read_text())["points"]["serial-eager"]
    draft = point["devices"]["draft"]
    assert draft["status"] == "PARTIAL" and draft["host_launch_rows_truncated"]
    assert draft["physical_window_status"] == "COMPLETE"
    assert draft["window_forward_count"] == 0
    assert draft["by_purpose"]["eager"]["host_launch_selected_forwards"] == 4
    assert point["recomputed_eager_overlap"]["lower_ms"] == 0
    assert point["recomputed_eager_overlap"]["upper_ms"] == 0


def test_physical_rows_take_precedence_and_partial_physical_is_not_zero(existing):
    root, output = existing
    path = root / "runs/actual-ended-serial-eager/draft-backend-report.json"
    backend = json.loads(path.read_text())
    backend["fixed_device"]["forwards"] = [
        {**forward(500, 510, "eager"), "host_start_ns": 120*NS},
        forward(120, 125, "eager"), forward(125, 130, "eager"),
    ]
    write(path, backend)
    export(root, output, max_jsonl_rows=1)
    point = json.loads(output.read_text())["points"]["serial-eager"]
    draft = point["devices"]["draft"]
    assert draft["forwards"][0]["source_forward_index"] == 1
    assert draft["physical_window_status"] == "PARTIAL"
    assert point["recomputed_eager_overlap"]["status"] == "PARTIAL"
    assert "lower_ms" not in point["recomputed_eager_overlap"]


def test_missing_empty_and_source_truncated_jsonl_remain_distinct_in_cycles(existing):
    root, output = existing
    path = root / "runs/actual-ended-serial-eager"
    (path / "draft-transport.jsonl").write_text("")
    measured = json.loads((path / "round-events.jsonl").read_text())
    setup = {"request_id": "r", "timeline": {"verify_start_ns": 10, "verify_end_ns": 20}}
    (path / "round-events.jsonl").write_text(json.dumps(setup)+"\n"+json.dumps(measured)+"\n")
    export(root, output, max_jsonl_rows=1)
    point = json.loads(output.read_text())["points"]["serial-eager"]
    streams = point["selected_cycles"][0]["streams"]
    assert streams["transport-events.jsonl"]["status"] == "MISSING"
    assert not streams["transport-events.jsonl"]["matching_rows_complete"]
    assert streams["draft-transport.jsonl"]["status"] == "READ"
    assert streams["draft-transport.jsonl"]["matching_rows_complete"]
    truncated = streams["round-events.jsonl"]
    assert truncated["rows"] == [] and truncated["status"] == "PARTIAL"
    assert truncated["source_truncated"] and truncated["truncated"]
    assert not truncated["projection_truncated"] and not truncated["matching_rows_complete"]


def test_byte_limited_stream_and_missing_backend_section_propagate(existing):
    root, output = existing
    path = root / "runs/actual-ended-serial-eager"
    (path / "transport-events.jsonl").write_text(" " * 50001)
    backend = json.loads((path / "draft-backend-report.json").read_text())
    del backend["fixed_proposals"]
    write(path / "draft-backend-report.json", backend)
    export(root, output, max_file_bytes=50000)
    point = json.loads(output.read_text())["points"]["serial-eager"]
    streams = point["selected_cycles"][0]["streams"]
    assert streams["transport-events.jsonl"]["status"] == "LIMIT"
    assert not streams["transport-events.jsonl"]["matching_rows_complete"]
    assert "transport-events.jsonl" in point["limited_files"]
    assert "transport-events.jsonl" not in point["missing_files"]
    assert streams["fixed_proposals"]["status"] == "MISSING"
    assert point["window_eager_records"]["fixed_proposals"]["status"] == "MISSING"


def test_event_selection_prefers_event_classes_over_window_positions():
    steps = [{"start_ns": i*100, "end_ns": (i+1)*100} for i in range(10)]
    events = [{"start_ns": i*100+50, "end_ns": i*100+50, "request_ids": ["r"],
               "counter_delta": {kind: 1}}
              for i, kind in ((2, "promotions"), (4, "parent_rejections"),
                              (5, "recovery_jobs"), (6, "admissions"), (8, "bridge_mismatches"))]
    chosen, trajectory = select_cycles(steps, events, 8)
    assert {2, 4, 5, 6, 8} <= set(chosen)
    assert trajectory["request_id"] == "r"


def test_help_imports_only_offline_stdlib():
    result = subprocess.run([sys.executable, "-m", "specrhythm.serving.eager_evidence_export",
                             "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "never loads CUDA" in result.stdout
    source = Path("src/specrhythm/serving/eager_evidence_export.py").read_text()
    assert "import torch" not in source and "import vllm" not in source
