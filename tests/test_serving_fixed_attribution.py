"""Producer-shaped CPU evidence, actual file entry/export and hard timeout contracts."""

import copy
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

import pytest

from specrhythm.phase4.transport import CheckpointJsonl
from specrhythm.serving.fixed_attribution import (
    analyze_raw,
    host_costs,
    overlap,
    rotations,
)
from specrhythm.serving.fixed_attribution_cli import (
    aggregate_clues,
    attach_events,
    main,
    supervise,
)
from specrhythm.serving.fixed_attribution_io import Inputs, project
from specrhythm.serving.fixed_observe import Timers

NS = 1_000_000
CLOCK = "CUDA elapsed events projected onto bracketed host monotonic anchor"


def forward(ids, begin, end, purpose=None, uncertainty=1):
    return {
        "internal_request_ids": ids,
        "B": len(ids),
        "Q": len(ids),
        "host_start_ns": begin * NS,
        "host_launch_end_ns": (begin + 1) * NS,
        "start_lower_ns": begin * NS,
        "start_upper_ns": begin * NS + uncertainty,
        "end_lower_ns": end * NS,
        "end_upper_ns": end * NS + uncertainty,
        "gpu_event_ms": end - begin,
        **({"purpose": purpose} if purpose else {}),
    }


def device(rank, uuid, forwards):
    return {
        "identity": {"global_rank": rank, "gpu_uuid": uuid},
        "forwards": forwards,
        "clock": CLOCK,
        "anchor_uncertainty_ns": 1,
    }


def artifacts(mode="pingpong", rounds=2):
    """Producer-shaped records; these CPU fixture times are not GPU measurements."""
    requests = [
        {
            "request_id": f"req-{i}",
            "cohort": ("A" if i < 32 else "B") if mode != "serial" else None,
            "commits": [],
        }
        for i in range(64)
    ]
    runtime = {
        "measurement_start_ns": 100 * NS,
        "measurement_end_ns": None,
        "requests": requests,
        "prompt_lengths": {r["request_id"]: 1 for r in requests},
        "target_steps": [],
        "target_devices": [],
        "host": {"intervals": []},
    }
    backend = {
        "fixed_device": device(0, "cpu-fixture-draft", []),
        "fixed_host": {"intervals": []},
        "fixed_proposals": [],
        "s2_work_records": [],
    }
    target_forwards, committed = [], []
    begin = 100
    for round_id in range(rounds):
        for group in [requests] if mode == "serial" else [requests[:32], requests[32:]]:
            ids = [r["request_id"] for r in group]
            prefix = 2 + round_id
            runtime["target_steps"].append(
                {
                    "B": len(ids),
                    "request_ids": ids,
                    "cohort": group[0]["cohort"],
                    "window": True,
                    "start_ns": begin * NS,
                    "end_ns": (begin + 100) * NS,
                    "schedule_start_ns": begin * NS,
                    "schedule_end_ns": (begin + 1) * NS,
                    "rows": [
                        {
                            "request_id": rid,
                            "internal_request_id": "target:" + rid,
                            "context_length": prefix,
                            "candidate_positions": 4,
                        }
                        for rid in ids
                    ],
                }
            )
            target_forwards.append(
                forward(["target:" + rid for rid in ids], begin + 40, begin + 50)
            )
            proposal = {
                "B": len(ids),
                "request_ids": ids,
                "round_ids": [round_id] * len(ids),
                "context_lengths": [prefix] * len(ids),
                "candidate_lengths": [4] * len(ids),
                "budgets": [4] * len(ids),
                "first_token_from_cached_logits": True,
                "start_ns": (begin + 5) * NS,
                "end_ns": (begin + 20) * NS,
            }
            backend["fixed_proposals"].append(proposal)
            for j in range(3):
                backend["fixed_device"]["forwards"].append(
                    forward(
                        ["sr-draft:" + rid for rid in ids],
                        begin + 6 + j * 2,
                        begin + 7 + j * 2,
                        "proposal",
                    )
                )
            backend["fixed_device"]["forwards"].append(
                forward(["sr-draft:" + rid for rid in ids], begin + 1, begin + 2, "commit")
            )
            backend["s2_work_records"].append(
                {
                    "operation": "commit_and_propose",
                    "request_ids": ids,
                    "logical_cohort": group[0]["cohort"],
                    "host_start_ns": begin * NS,
                    "host_end_ns": (begin + 20) * NS,
                }
            )
            for r in group:
                r["commits"].append({"token_ids": [7], "timestamp_ns": (begin + 90) * NS})
                committed.append(
                    {
                        "request_id": r["request_id"],
                        "round_id": round_id,
                        "prefix_token_count": prefix,
                        "prefix_version": round_id + 1,
                        "committed_token_ids": [7],
                    }
                )
            begin += 100
    runtime["measurement_end_ns"] = begin * NS
    runtime["sample_count"] = len(runtime["target_steps"])
    backend["draft_model_forward_count_by_purpose"] = dict(
        Counter(f["purpose"] for f in backend["fixed_device"]["forwards"])
    )
    runtime["target_devices"] = [
        {
            "device": device(
                rank, "cpu-fixture-target-" + str(rank), copy.deepcopy(target_forwards)
            ),
            "rounds": committed if rank == 0 else [],
            "host": {"intervals": []},
            "target_rows": [],
        }
        for rank in (0, 1)
    ]
    return runtime, backend


@pytest.mark.parametrize("mode,step_count", [("serial", 2), ("pingpong", 4)])
def test_real_schema_prefix_commit_batch_join_and_tp_not_summed(mode, step_count):
    runtime, backend = artifacts(mode)
    # Shuffled physical records and scheduler rows must not affect indexed associations.
    runtime["target_steps"].reverse()
    backend["fixed_device"]["forwards"].reverse()
    result = analyze_raw(runtime, backend)
    assert result["status"] == "OBSERVED"
    assert result["step_count"] == step_count
    assert result["rotations"]["complete_count"] == 2
    assert result["rotations"]["rotation_ms"]["mean"] == (100 if mode == "serial" else 200)
    assert result["overlap"]["status"] == "ZERO"
    assert result["steps"][0]["target_gpu_event_ms"] == 10
    assert result["proposal_only_forward_gpu_ms"]["mean"] == 3
    assert result["draft_model_gpu_event_ms_by_purpose"]["commit"]["sum_ms"] == step_count
    assert result["steps"][0]["next_actual_proposal_start_ns_range"] is not None
    assert result["critical_path"].startswith("UNKNOWN")


def test_real_timer_nested_inclusive_union_and_cross_process_union(monkeypatch):
    ticks = iter([0, 2 * NS, 8 * NS, 10 * NS])
    monkeypatch.setattr("specrhythm.serving.fixed_observe.time.monotonic_ns", lambda: next(ticks))
    timers = Timers()
    with timers.span("checkpoint_log_write"):
        with timers.span("log_fsync"):
            pass
    result = host_costs({"rank0": timers.report(), "rank1": timers.report()}, 0, 10 * NS)
    assert result["all_categories_union_ms"] == 10
    assert result["categories"]["checkpoint_log_write"]["inclusive_call_ms"] == 20
    assert result["categories"]["log_fsync"]["cross_process_union_ms"] == 6
    assert result["categories"]["log_fsync"]["exclusive_self_ms"] is None
    assert result["outside_host_observation_ms"] == 0


def test_clock_uncertainty_positive_overlap_missing_and_zero_are_distinct():
    runtime, backend = artifacts(rounds=1)
    targets = [r["device"] for r in runtime["target_devices"]]
    draft = backend["fixed_device"]
    args = targets, draft, 100 * NS, 300 * NS
    assert overlap(*args)["status"] == "ZERO"
    draft["forwards"] = [forward(["sr-draft:req-0"], 140, 145, "proposal")]
    assert overlap(*args)["status"] == "POSITIVE"
    # TP overlap is a union, not 5 ms per rank added together.
    assert overlap(*args)["event_overlap_upper_ms"] < 5.001
    draft["forwards"] = [forward(["sr-draft:req-0"], 135, 140, "proposal", uncertainty=3 * NS)]
    draft["anchor_uncertainty_ns"] = 3 * NS
    assert overlap(*args)["status"] == "UNCERTAIN"
    draft["forwards"] = []
    missing = overlap(*args)
    assert missing["status"] == "UNKNOWN"
    assert missing["event_overlap_upper_ms"] is None
    draft.pop("clock")
    assert overlap(*args)["status"] == "UNKNOWN"


@pytest.mark.parametrize("damage", ["tp", "draft", "cohort", "commit", "prefix", "internal_id"])
def test_missing_or_misassociated_evidence_never_becomes_zero(damage):
    runtime, backend = artifacts()
    if damage == "tp":
        runtime["target_devices"][1]["device"]["forwards"].pop()
    elif damage == "draft":
        backend["fixed_device"]["forwards"].pop(0)
    elif damage == "cohort":
        runtime["requests"][0]["cohort"] = "B"
    elif damage == "commit":
        runtime["requests"][0]["commits"][0]["token_ids"] = [99]
    elif damage == "prefix":
        runtime["target_steps"][0]["rows"][0]["context_length"] += 10
    else:
        backend["fixed_device"]["forwards"][0]["internal_request_ids"][0] = "wrong"
    result = analyze_raw(runtime, backend)
    assert result["overlap"]["status"] == "UNKNOWN"
    assert result["status"] == "INSUFFICIENT"


def test_incomplete_rotations_never_paired_by_adjacent_index():
    runtime, backend = artifacts(rounds=3)
    samples = analyze_raw(runtime, backend)["steps"]
    # Drop A1, leaving B1 next to A2. Only same real round IDs may be paired.
    result = rotations(
        [s for s in samples if not (s["cohort"] == "A" and s["identities"][0]["round_id"] == 1)],
        100 * NS,
    )
    assert result["complete_count"] == 2
    assert result["incomplete_count"] == 1
    assert [r["round_id"] for r in result["rotations"]] == [0, 2]
    assert result["rotations"][1]["rotation_ms"] is None
    assert result["rotations"][0]["rotation_ms"] == 200
    broken = copy.deepcopy(samples[:2])
    broken[1]["identities"][0]["request_id"] = broken[0]["identities"][0]["request_id"]
    with pytest.raises(ValueError, match="shares request"):
        rotations(broken, 100 * NS)


def test_interleaved_cohort_pairs_do_not_get_fabricated_rotation_boundaries():
    runtime, backend = artifacts()
    samples = analyze_raw(runtime, backend)["steps"]
    # A0, A1, B0, B1: neither enclosing pair is one isolated full64 rotation.
    samples[1]["start_ns"], samples[2]["start_ns"] = (
        samples[2]["start_ns"],
        samples[1]["start_ns"],
    )
    samples[1]["end_ns"], samples[2]["end_ns"] = samples[2]["end_ns"], samples[1]["end_ns"]
    result = rotations(samples, 100 * NS)
    assert result["complete_count"] == 2
    assert result["rotation_ms"]["count"] == 0


def test_large_host_intervals_use_sort_scan_not_pairwise():
    rows = [
        {"category": "json_serialization", "start_ns": i, "end_ns": i + 2} for i in range(100_000)
    ]
    started = time.monotonic()
    result = host_costs({"owner": {"intervals": rows}}, 0, 100_001)
    assert result["categories"]["json_serialization"]["count"] == 100_000
    assert result["all_categories_union_ms"] == pytest.approx(0.100001)
    # This generous ceiling detects a Cartesian implementation (10^10 pairs).
    assert time.monotonic() - started < 10


def write_attempt(tmp_path):
    root = tmp_path / "retained"
    attempt = root / "runs" / "continuous-pingpong-cpu-fixture"
    attempt.mkdir(parents=True)
    runtime, backend = artifacts()
    # Large/private structures must not be exported.
    runtime["requests"][0]["generated_token_ids"] = [999] * 100
    runtime["target_pool_final"] = {"private_kv_tensor": "secret"}
    backend["s2_work_records"][0]["terminal_drain"] = {"private_kv_tensor": "secret"}
    for name, value in [("runtime.json", runtime), ("draft-backend-report.json", backend)]:
        (attempt / name).write_text(json.dumps(value))
    event = {
        "operation": "commit_and_propose",
        "request_id": "req-0",
        "success": True,
        "start_ns": 101 * NS,
        "end_ns": 125 * NS,
        "result": {
            "private_kv_tensor": "secret",
            "proposal": {
                "request_id": "req-0",
                "round_id": 0,
                "prefix_token_count": 2,
                "proposal_id": "cpu-fixture-proposal-id",
                "prefix_version": 1,
                "created_timestamp_ns": 120 * NS,
                "draft_start_ns": 105 * NS,
                "draft_end_ns": 120 * NS,
            },
        },
    }
    CheckpointJsonl(attempt / "draft-work-events.jsonl").append(event)
    return root, attempt


def test_export_real_checksummed_events_read_only_reanalysis_and_no_private_kv(tmp_path):
    root, attempt = write_attempt(tmp_path)
    before = {p: p.read_bytes() for p in attempt.iterdir()}
    export = tmp_path / "export"
    assert main(["--input", str(root), "--output", str(export), "--export"]) == 0
    assert before == {p: p.read_bytes() for p in attempt.iterdir()}
    manifest = json.loads((export / "export-manifest.json").read_text())
    assert manifest["status"] == "COMPLETE"
    inputs = Inputs(export / "evidence", ["pingpong"], lambda **kw: None)
    for name in inputs.names:
        value = inputs.read(name)
        assert "private_kv_tensor" not in json.dumps(value)
        assert "generated_token_ids" not in json.dumps(value)
        if "draft-work-events" in name:
            assert value[0]["source_record_sha256"]
            assert value[0]["source_line"] == 1
            assert value[0]["result"]["proposal"]["created_timestamp_ns"] == 120 * NS
    inputs.close()
    output = tmp_path / "analysis"
    assert main(["--input", str(export / "evidence"), "--output", str(output)]) == 0
    report = json.loads((output / "attribution.json").read_text())
    assert report["attempts"][0]["raw"]["rotations"]["complete_count"] == 2
    step = report["attempts"][0]["raw"]["steps"][0]
    assert step["owner_proposal_evidence"][0]["proposal_id"] == "cpu-fixture-proposal-id"
    assert step["proposal_publication_ns"] is None
    assert report["formal_comparison_eligible"] is False
    assert sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) < 10 * 1024 * 1024


def test_small_bundle_reports_missing_raw_not_zero_or_pass(tmp_path):
    archive = tmp_path / "small.tar.gz"
    value = json.dumps(
        {"mode": "pingpong", "overlap": {"physical_overlap_status": "ZERO"}}
    ).encode()
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("runs/continuous-pingpong-fixture/light-summary.json")
        member.size = len(value)
        tar.addfile(member, io.BytesIO(value))
    output = tmp_path / "analysis"
    assert main(["--input", str(archive), "--output", str(output)]) == 0
    result = json.loads((output / "attribution.json").read_text())
    attempt = result["attempts"][0]
    assert attempt["evidence_status"] == "INSUFFICIENT"
    assert any("runtime.json" in gap for gap in attempt["gaps"])
    assert "UNKNOWN" in (output / "summary.csv").read_text()


def test_timeout_retains_partial_result_and_kills_blocked_parser(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "attribution.json").write_text(
        json.dumps({"attempts": [], "errors": [], "marker": 42})
    )
    started = time.monotonic()
    rc = supervise([sys.executable, "-c", "import time; time.sleep(30)"], output, 0.1)
    assert rc == 124
    assert time.monotonic() - started < 5
    report = json.loads((output / "attribution.json").read_text())
    assert report["marker"] == 42 and report["status"] == "TIMEOUT"
    assert report["partial_results_preserved"] is True


def test_input_error_keeps_earlier_summary_and_actual_failure_exit(tmp_path):
    root, attempt = write_attempt(tmp_path)
    (attempt / "light-summary.json").write_text('{"mode":"pingpong","window_ms":23}')
    (attempt / "runtime.json").write_text("{")
    output = tmp_path / "bad-analysis"
    assert main(["--input", str(root), "--output", str(output)]) == 2
    report = json.loads((output / "attribution.json").read_text())
    assert report["attempts"][0]["retained_summary"]["window_ms"] == 23
    assert report["errors"][0].startswith("JSONDecodeError")
    assert report["status"] == "FAILED"


def test_cpu_entry_does_not_import_gpu_or_call_audit():
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "import specrhythm.serving.fixed_attribution_cli; "
            "assert not ({'torch','vllm'} & set(sys.modules))",
        ],
        capture_output=True,
        text=True,
        env=os.environ,
    )
    assert check.returncode == 0, check.stderr
    script = Path("scripts/analyze_fixed_diagnostic.sh").read_text()
    assert "fixed_attribution_cli" in script and "fixed_cli" not in script


def test_failed_points_excluded_and_owner_prefix_mismatch_visible():
    assert "rotation_gap_ms" not in aggregate_clues(
        [{"mode": "serial", "retained_summary": {"valid": False}}]
    )
    runtime, backend = artifacts()
    raw = analyze_raw(runtime, backend)
    with pytest.raises(ValueError, match="owner result prefix"):
        attach_events(
            raw,
            {
                "draft-work-events.jsonl": [
                    {
                        "request_id": "req-0",
                        "result": {
                            "proposal": {
                                "request_id": "req-0",
                                "round_id": 0,
                                "prefix_token_count": 99,
                            }
                        },
                    }
                ]
            },
        )


def test_projection_is_idempotent():
    runtime, backend = artifacts()
    for name, data in [("runtime.json", runtime), ("draft-backend-report.json", backend)]:
        once = project(name, data)
        assert project(name, once) == once
