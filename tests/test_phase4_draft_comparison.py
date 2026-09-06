from __future__ import annotations

import copy
import json
import shutil

from specrhythm.phase4.draft_comparison import (
    compare_directories,
    compare_serial_work,
    proposal_equivalence,
    validate_batch_evidence,
)
from specrhythm.phase4.draft_metrics import DraftMetrics
from specrhythm.phase4.manifest import sha256_file
from specrhythm.phase4.transport import CheckpointJsonl


def performance():
    return {
        "schema_version": "specrhythm.phase4b2-decode-performance.v1",
        "mode": "serial",
        "valid": True,
        "errors": [],
        "performance_result": True,
        "cleanup_valid": True,
        "request_count": 1,
        "metrics": {
            "completed_requests": 1,
            "total_measured_committed_output_tokens": 2,
            "decode_makespan_ms": 100.0,
            "aggregate_throughput_tokens_per_second": 20.0,
        },
        "requests": [
            {
                "request_id": "r",
                "token_accounting_valid": True,
                "finish_reason": "length",
                "measured_committed_output_token_ids": [20, 21],
                "measured_committed_output_token_count": 2,
                "bootstrap_token_id": 10,
                "setup_committed_output_tokens": 1,
                "total_generated_token_ids": [10, 20, 21],
                "maximum_new_tokens": 3,
                "prompt_token_count": 1,
                "prompt_token_ids_sha256": "a" * 64,
            }
        ],
        **{
            field: "same"
            for field in (
                "workload_sha256",
                "execution_git_commit",
                "vllm_commit",
                "vllm_version",
                "patch_hashes",
                "models",
                "placement",
                "gpu_topology",
                "correctness_mode",
            )
        },
        "artifact_sha256": {
            key: "same" for key in ("workload", "config", "topology", "patch_manifest")
        },
        "measurement": {
            "setup_excluded": True,
            "measurement_start_ns": 1,
            "bootstrap_excluded_from_measured_token_count": True,
        },
        "mode_semantics": {
            "draft_target_overlap": False,
            "initial_proposal_after_measurement_start": True,
        },
    }


def test_final_target_divergence_is_separate_from_matched_work():
    a = performance()
    b = copy.deepcopy(a)
    b["requests"][0]["measured_committed_output_token_ids"] = [22, 23]
    b["requests"][0]["total_generated_token_ids"] = [10, 22, 23]
    report = compare_serial_work(a, b, 1)
    assert report["valid"]
    assert report["final_target_sequences_equal"] is False


def test_work_budget_change_is_not_numerical_divergence():
    a, b = performance(), performance()
    b["requests"][0]["maximum_new_tokens"] = 4
    assert not compare_serial_work(a, b, 1)["valid"]


def test_existing_qualified_serial_artifacts_remain_valid(tmp_path):
    from test_phase4b2_performance import _measure

    artifact = _measure(tmp_path, "serial")
    result = compare_serial_work(artifact, copy.deepcopy(artifact), artifact["request_count"])
    assert result["valid"], result["errors"]


def test_invalid_termination_or_nonfinite_time_fails_closed():
    a, b = performance(), performance()
    b["requests"][0]["finish_reason"] = "aborted"
    assert not compare_serial_work(a, b, 1)["valid"]
    b = performance()
    b["metrics"]["decode_makespan_ms"] = float("nan")
    assert not compare_serial_work(a, b, 1)["valid"]


def test_directory_comparison_binds_shutdown_startup_and_round_evidence(tmp_path):
    from test_phase4b2_performance import _measure

    artifact = _measure(tmp_path, "serial")
    hf, vllm = tmp_path / "serial", tmp_path / "vllm"
    shutil.copytree(hf, vllm)

    def write(path, value):
        path.write_text(json.dumps(value))

    rounds = CheckpointJsonl(hf / "round-events.jsonl").read()
    for row in rounds:
        row.pop("record_sha256")
        row.update(
            parent_prefix_hash=f"context-{row['round_id']}",
            parent_prefix_len=3,
            proposal_token_ids=[10],
            accepted_draft_tokens=0,
            target_correction_token_ids=row["committed_token_ids"],
            target_bonus_token_ids=[],
        )
    metrics = DraftMetrics()
    metrics.forward("commit", 2, 2)
    metrics.forward("commit", 2, 2)
    metrics.counters.update(
        proposals=len(rounds), commits=len(rounds), proposed_tokens=len(rounds)
    )
    metrics.gpu_ms["commit"] = 1.0
    provenance = json.loads((hf / "draft-service-ready.json").read_text())["provenance"]
    evidence = {
        **metrics.snapshot("vllm-batched-paged-kv-draft"),
        "provenance": provenance,
        "backend_shutdown_complete": True,
        "draft_live_requests_final": 0,
        "worker_resources": {
            "blocks_allocated": 2,
            "blocks_freed": 2,
            "worker_shutdown_complete": True,
            "live_allocator_requests": 0,
        },
    }
    write(vllm / "draft-backend-report.json", evidence)
    write(vllm / "draft-startup.json", provenance)
    for directory, name in (
        (hf, "hf-persistent-kv-correctness-draft"),
        (vllm, "vllm-batched-paged-kv-draft"),
    ):
        (directory / "round-events.jsonl").unlink()
        for row in rounds:
            CheckpointJsonl(directory / "round-events.jsonl").append(row)
        ready = json.loads((directory / "draft-service-ready.json").read_text())
        write(directory / "draft-service-ready.json", {**ready, "backend": name})
        raw = json.loads((directory / "resident-serial.json").read_text())
        raw.update(errors=[], strict_serial_timeline={"round_events": len(rounds)})
        if directory == vllm:
            raw["draft_shutdown"] = {
                "draft_backend_report_sha256": sha256_file(vllm / "draft-backend-report.json")
            }
        write(directory / "resident-serial.json", raw)
        value = copy.deepcopy(artifact)
        value["artifact_sha256"]["raw_run"] = sha256_file(directory / "resident-serial.json")
        write(directory / "decode-performance.json", value)
    report = compare_directories(hf, vllm, artifact["request_count"])
    assert report["ready_for_operator_review"], report["errors"]
    assert report["hf_derived_model_forward_count"] == len(rounds)
    assert report["vllm_actual_model_forward_count"] == 2
    changed = copy.deepcopy(rounds)
    changed[0]["proposal_token_ids"] = [11]
    (vllm / "round-events.jsonl").unlink()
    for row in changed:
        CheckpointJsonl(vllm / "round-events.jsonl").append(row)
    divergent = compare_directories(hf, vllm, artifact["request_count"])
    assert divergent["performance_comparable"]  # Same completed work is a separate fact.
    assert not divergent["performance_interpretation_allowed"]
    assert divergent["decode_makespan_ratio_hf_over_vllm"] is None
    evidence["draft_model_forward_count"] += 1
    write(vllm / "draft-backend-report.json", evidence)
    invalid = compare_directories(hf, vllm, artifact["request_count"])
    assert not invalid["performance_comparable"]
    assert "batched backend final evidence not bound to runtime shutdown" in invalid["errors"]


def test_same_prefix_draft_divergence_is_not_hidden():
    row = {
        "request_id": "r",
        "parent_prefix_hash": "a",
        "parent_prefix_len": 1,
        "proposal_token_ids": [3, 4],
    }
    different_context = {**row, "parent_prefix_hash": "b", "proposal_token_ids": [5, 6]}
    assert proposal_equivalence([row], [different_context])["common_context_count"] == 0
    changed_tokens = {**row, "proposal_token_ids": [3, 5]}
    result = proposal_equivalence([row], [changed_tokens])
    assert not result["exact_on_common_contexts"]
    assert result["mismatches"][0]["hf_tokens"] == [3, 4]


def test_declared_batched_name_cannot_replace_counter_evidence():
    metrics = DraftMetrics()
    metrics.forward("proposal", 2, 2)
    metrics.counters.update(proposals=2, proposed_tokens=4, commits=2)
    metrics.gpu_ms["proposal"] = 1.0
    report = metrics.snapshot("vllm-batched-paged-kv-draft")
    report["worker_resources"] = {
        "blocks_allocated": 2,
        "blocks_freed": 2,
        "worker_shutdown_complete": True,
    }
    rounds = [{"proposal_token_ids": [1, 2]}, {"proposal_token_ids": [3, 4]}]
    assert validate_batch_evidence(report, rounds) == []
    report["draft_batch_size_p50"] = 100
    assert validate_batch_evidence(report, rounds)
