"""Producer-shaped retained artifacts exercise light accounting and foreground reporting."""

import copy
import json

import pytest
from test_serving_s2_results import fixture

from specrhythm.phase4.transport import CheckpointJsonl
from specrhythm.serving import fixed_observe, fixed_results
from specrhythm.serving.common import read_json
from specrhythm.serving.fixed_plan import capacity_metadata, point, settings
from specrhythm.serving.runtime_profile import load_s2
from specrhythm.serving.s2_plan import sealed


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def retained(tmp_path, monkeypatch, request):
    mode = getattr(request, "param", "serial")
    path, directory, _ = fixture(tmp_path, monkeypatch, mode)
    manifest, definitions = load_s2(str(path))
    manifest.pop("sha256")
    manifest["fixed_diagnostic"] = {
        "capacity": {
            m: capacity_metadata(m) for m in ("target", "serial", "pingpong", "serial-split")
        },
        "options": settings(),
    }
    write(path, sealed(manifest))
    lifecycle = read_json(directory / "process-lifecycle.json")
    lifecycle["owned_cleanup_completed"] = True
    write(directory / "process-lifecycle.json", lifecycle)
    runtime = read_json(directory / "runtime.json")
    backend = read_json(directory / "draft-backend-report.json")
    diag = CheckpointJsonl(directory / "target-diagnostics.jsonl").read()
    monkeypatch.setattr(fixed_observe, "ROUNDS", [])
    monkeypatch.setattr(fixed_observe, "TARGET_ROWS", [])
    for name in ("round-events.jsonl", "proposal-events.jsonl", "target-diagnostics.jsonl"):
        for r in CheckpointJsonl(directory / name).read():
            fixed_observe.capture(r)
    active_diag = [r for r in diag if r["target_forward_start_ns"] >= runtime["start_ns"]]
    batches = {}
    for r in active_diag:
        key = r["target_forward_start_ns"], r["target_forward_end_ns"]
        batches.setdefault(key, {})[r["request_id"]] = r
    steps = []
    devices = []
    for (a, b), members in sorted(batches.items()):
        rows = [
            {
                "request_id": r["request_id"],
                "internal_request_id": r["request_id"],
                "query_positions": r["query_length"],
                "candidate_positions": len(r["proposal_token_ids"]),
                "base_root_positions": r["query_length"] - len(r["proposal_token_ids"]),
                "context_length": len(r["committed_prefix_token_ids"]),
                "position_start": r["position_ids"][0],
                "position_end_exclusive": r["position_ids"][-1] + 1,
            }
            for r in members.values()
        ]
        steps.append(
            dict(
                start_ns=a - 1,
                end_ns=b + 1,
                rows=rows,
                B=len(rows),
                request_ids=list(members),
                cohort=None,
                window=True,
                supply_phase="tail",
                population={"held_slots": 4},
            )
        )
    for rank in (0, 1):
        forwards = [
            {
                "host_start_ns": s["start_ns"] + 1,
                "internal_request_ids": [r["internal_request_id"] for r in s["rows"]],
                "B": s["B"],
                "gpu_event_ms": 0.1,
                "start_lower_ns": s["start_ns"] + 1,
                "start_upper_ns": s["start_ns"] + 2,
                "end_lower_ns": s["end_ns"] - 2,
                "end_upper_ns": s["end_ns"] - 1,
            }
            for s in steps
        ]
        devices.append(
            {
                "device": {
                    "identity": {"global_rank": rank, "gpu_uuid": f"GPU-{rank + 1}"},
                    "forwards": forwards,
                },
                "host": {"intervals": []},
                "rounds": fixed_observe.ROUNDS if rank == 0 else [],
                "target_rows": fixed_observe.TARGET_ROWS if rank == 0 else [],
            }
        )
    runtime.update(
        point=point(mode),
        capacity=capacity_metadata(mode, len(definitions)),
        measurement_start_ns=100,
        measurement_end_ns=500,
        warmup_start_ns=100,
        warmup_end_ns=100,
        warmup_steps=0,
        target_requests_final=0,
        target_devices=devices,
        target_steps=steps,
        prompt_lengths={r.request_id: r.prompt_length for r in definitions},
        stop_reason="all_naturally_completed",
        host={"intervals": []},
    )
    backend.update(
        fixed_host={"intervals": []},
        fixed_proposals=[],
        fixed_device={"identity": {"gpu_uuid": "GPU-0"}, "forwards": []},
    )
    write(directory / "runtime.json", runtime)
    write(directory / "draft-backend-report.json", backend)
    return path, directory, point(mode), runtime, backend


@pytest.mark.parametrize("retained", ["target", "serial", "pingpong"], indirect=True)
def test_light_summary_uses_real_backend_structure_without_full_qualifier(retained, monkeypatch):
    from specrhythm.serving import s2_results

    monkeypatch.setattr(s2_results, "qualify", lambda *a: pytest.fail("heavy qualifier called"))
    path, directory, selected, _, _ = retained
    value = fixed_results.summarize(path, directory, selected)
    assert value["valid"], (value["errors"], value.get("error_details"))
    assert value["execution_status"] == value["measurement_status"] == "PASS"
    assert value["committed_window_tokens"] > 0
    if selected["mode"] != "target":
        assert value["g_per_request_per_verification"] == pytest.approx(
            value["committed_window_tokens"] / sum(s["B"] for s in value["target_samples"])
        )
    assert (
        value["target_base_root_positions"] + value["verified_proposed_tokens"]
        == value["target_query_positions"]
    )
    assert value["full_offline_audit_status"] == "PENDING"
    assert value["cross_run_token_equality"] == "NOT_REQUIRED"
    assert value["full_request_slo_attainment"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("backend_name", "vllm-batched"),
        ("execution_failed", True),
        ("backend_shutdown_complete", False),
        ("draft_live_requests_final", 1),
    ],
)
def test_true_backend_failures_block_with_fields(retained, field, value):
    path, directory, selected, _, backend = retained
    backend[field] = value
    write(directory / "draft-backend-report.json", backend)
    report = fixed_results.summarize(path, directory, selected)
    assert not report["valid"] and report["primary_error"], report
    assert "actual" in report["error_details"], report["errors"]
    assert report["error_details"]["actual"][field]["valid"] is False


@pytest.mark.parametrize("failure", ["tokens", "lifecycle", "root", "TP", "diagnostic"])
def test_material_execution_failures_remain_blocking(retained, failure):
    path, directory, selected, runtime, _ = retained
    if failure == "tokens":
        runtime["requests"][0]["generated_token_ids"].append(500)
    elif failure == "lifecycle":
        runtime["requests"][0]["resources_released"] = False
    elif failure == "root":
        runtime["target_steps"][0]["rows"][0]["base_root_positions"] = 2
    elif failure == "TP":
        runtime["target_devices"].pop()
    else:
        runtime["target_devices"][0]["target_rows"][0]["structural_errors"] = ["broken KV"]
    write(directory / "runtime.json", runtime)
    report = fixed_results.summarize(path, directory, selected)
    assert not report["valid"] and report["errors"]


def test_cancellation_is_not_full_completion_or_tpot_sample(retained):
    path, directory, selected, runtime, _ = retained
    row = runtime["requests"][0]
    row["state"] = "DIAGNOSTIC_CANCELLED"
    row.pop("completion_ns")
    # Deliberately retain a natural-terminal generated sequence: validator must reject.
    write(directory / "runtime.json", runtime)
    assert not fixed_results.summarize(path, directory, selected)["valid"]


def test_foreground_light_json_csv_and_primary_error(retained, capsys):
    _, directory, selected, _, _ = retained
    report = {
        "valid": False,
        "errors": ["original worker failure"],
        "effective_exit_code": 23,
        "primary_error": {"actual": "original worker failure", "artifact": "target.log"},
    }
    value = fixed_results.emit_result(directory, report, selected)
    assert value["execution_status"] == "FAILED"
    assert read_json(directory / "light-summary.json")["effective_exit_code"] == 23
    assert (directory / "light-summary.csv").exists()
    assert "original worker failure" in capsys.readouterr().out


def test_foreground_success_includes_stage_costs_and_rotation(retained, capsys):
    path, directory, selected, _, _ = retained
    report = fixed_results.summarize(path, directory, selected)
    assert report["valid"], report
    fixed_results.emit_result(directory, report, selected)
    shown = json.loads(capsys.readouterr().out.split("[fixed diagnostic] ", 1)[1])
    assert shown["pipeline_stage_gpu_event_ms"] == report["pipeline_stage_gpu_event_ms"]
    assert shown["actual_rotation_ms"] == report["actual_rotation_ms"]


def test_comparison_retains_clean_zero_sample_stop_without_numeric_prediction(retained, tmp_path):
    path, directory, selected, runtime, _ = retained
    runtime["measurement_start_ns"] = None
    write(directory / "runtime.json", runtime)
    report = fixed_results.summarize(path, directory, selected)
    assert report["valid"] and report["measurement_status"] == "INSUFFICIENT"
    destination = tmp_path / "comparison-root/runs/serial"
    destination.mkdir(parents=True)
    write(destination / "result.json", report)
    result = fixed_results.comparisons(destination.parent.parent)
    assert result["points"][0]["measurement_status"] == "INSUFFICIENT"
    assert result["measured_comparisons"]["serial -> pingpong"]["status"] == "PENDING"
    assert result["prediction"]["ideal_serial_gpu_stage_ms"] is None


def test_interval_aggregation_does_not_double_sum_TP_or_duplicate_samples(retained):
    _, _, _, runtime, backend = retained
    start, end = runtime["start_ns"], runtime["end_ns"]
    d = runtime["target_devices"][0]["device"]["forwards"][0]
    backend["fixed_device"]["forwards"] = [{**d, "purpose": "proposal"}]
    one = fixed_results.overlap_metrics(
        runtime["target_devices"][:1], backend["fixed_device"], start, end
    )
    both = fixed_results.overlap_metrics(
        runtime["target_devices"], backend["fixed_device"], start, end
    )
    assert one == both
    duplicate = copy.deepcopy(runtime["target_devices"])
    duplicate[0]["device"]["forwards"] *= 2
    assert fixed_results.overlap_metrics(duplicate, backend["fixed_device"], start, end) == both


def test_split_cost_requires_both_context_matched_halves():
    def r(batch, half, ids, values, time_ms):
        return {
            "valid": True,
            "point": {
                "mode": "serial",
                "kind": "initial-state",
                "batch": batch,
                "half": half,
                "repeat": 0,
            },
            "execution_sha256": "same",
            "dtype_attention_identity": [
                {"role": "target", "dtype": "bf16", "attention_backends": ["FA"]}
            ],
            "stage_shape": {
                "status": "PASS",
                "expected_request_ids": ids,
                "context_lengths": values,
                "target_gpu_event_ms": time_ms,
                "draft_gpu_event_ms": time_ms / 2,
            },
        }

    a = r(32, "A", list(range(32)), [10] * 32, 4)
    b = r(32, "B", list(range(32, 64)), [11] * 32, 6)
    whole = r(64, "A", list(range(64)), [10] * 32 + [11] * 32, 8)
    report = fixed_results.matched_shape_contrasts([a, b, whole])
    assert report["matched_repeats"] == 1
    assert report["split_verification_cost_ms"]["mean"] == 2
    assert report["D32_over_V_SD32"]["mean"] == 0.5
    assert fixed_results.matched_shape_contrasts([a, whole])["matched_repeats"] == 0
    b["stage_shape"]["context_lengths"][0] += 1
    report = fixed_results.matched_shape_contrasts([a, b, whole])
    assert report["matched_repeats"] == 0 and "mismatch" in report["excluded"][0]["reason"]


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), 0, -1])
def test_nonfinite_or_nonpositive_target_timing_blocks(retained, bad):
    path, directory, selected, runtime, _ = retained
    runtime["target_devices"][0]["device"]["forwards"][0]["gpu_event_ms"] = bad
    write(directory / "runtime.json", runtime)
    assert not fixed_results.summarize(path, directory, selected)["valid"]


def test_offline_partial_lifecycle_requires_real_diagnostic_cancellation():
    from specrhythm.serving.common import DataError

    rows = [
        dict(
            proposal_id="p",
            request_id="r",
            round_id=0,
            prefix_version=1,
            proposal_token_ids=[10, 11],
            timestamp_ns=i + 1,
            lifecycle_state=state,
        )
        for i, state in enumerate(("CREATED", "PUBLISHED", "INSTALLED"))
    ]
    cancelled = [{"request_id": "r", "state": "DIAGNOSTIC_CANCELLED"}]
    value = fixed_results.audit_proposal_lifecycle(rows, cancelled, [])
    assert value["cancelled_unverified_proposals"] == 1
    with pytest.raises(DataError, match="lifecycle invalid"):
        fixed_results.audit_proposal_lifecycle(
            rows, [{"request_id": "r", "state": "FINISHED"}], []
        )
    with pytest.raises(DataError, match="lifecycle invalid"):
        fixed_results.audit_proposal_lifecycle(rows, cancelled, [{"proposal_id": "p"}])
