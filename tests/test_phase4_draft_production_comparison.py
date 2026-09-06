from __future__ import annotations

import copy
import json
import shutil

import pytest
from test_phase4_batch_invariant import diagnostic
from test_phase4_draft_comparison import performance
from test_phase4_draft_qualification import write_aggregate_inputs
from test_phase4b2_performance import _measure

from specrhythm.phase4.batched_draft_service import write_immutable_report
from specrhythm.phase4.draft_comparison import compare_directories
from specrhythm.phase4.draft_metrics import DraftMetrics
from specrhythm.phase4.draft_qualification import aggregate_directories, require_progression
from specrhythm.phase4.manifest import sha256_file
from specrhythm.phase4.serial import greedy_acceptance, token_prefix_hash
from specrhythm.phase4.transport import CheckpointJsonl


def write(path, value):
    path.write_text(json.dumps(value))


def append_rows(path, rows):
    path.unlink(missing_ok=True)
    for row in rows:
        CheckpointJsonl(path).append({k: v for k, v in row.items() if k != "record_sha256"})


def qualified_root(root):
    write_aggregate_inputs(root)
    result = aggregate_directories(root)
    assert result["valid"], result["errors"]
    write_immutable_report(root / "qualification.json", result)
    return root / "qualification.json"


def pair(root, stage, count, qualification, *, different_work=False):
    template = root / "template"
    if not template.exists():
        template.mkdir()
        _measure(template, "serial")
    qualified = json.loads(qualification.read_text())
    identity = qualified["run_identity"]
    startup = json.loads((root / "D3-B8/draft-startup.json").read_text())
    for label in ("hf", "vllm"):
        full_accept = different_work and label == "vllm"
        directory = root / stage / label
        shutil.copytree(template / "serial", directory)
        value = performance()
        value["request_count"] = count
        value["workload_sha256"] = "same"
        value["metrics"].update(
            completed_requests=count,
            total_measured_committed_output_tokens=3 * count,
            decode_makespan_ms=100 if label == "hf" else 50,
            aggregate_throughput_tokens_per_second=30 * count if label == "hf" else 60 * count,
            tpot_ms={"mean": 2.0, "p50": 2.0},
        )
        requests, rounds, diagnostics = [], [], []
        for i in range(count):
            rid = f"r{i}"
            request = copy.deepcopy(value["requests"][0])
            request.update(
                request_id=rid,
                maximum_new_tokens=None,  # Historical performance schema omits this value.
                measured_committed_output_token_count=3,
                total_generated_token_count=4,
                measured_committed_output_token_ids=[21, 23, 24],
                total_generated_token_ids=[10, 21, 23, 24],
                prompt_token_ids_sha256=token_prefix_hash([i + 1]),
            )
            requests.append(request)
            prefix = [i + 1, 10]
            deltas = ([21, 23, 24],) if full_accept else ([21], [23, 24])
            for n, committed in enumerate(deltas):
                proposal = (
                    [21, 23]
                    if full_accept
                    else [20 if label == "hf" else 18, 25]
                    if n == 0
                    else [23]
                )
                terminal = full_accept or n == 1
                decision = greedy_acceptance(proposal, committed, terminal=terminal)
                observed = diagnostic(proposal=proposal, prefix=prefix, kv_length=len(prefix))
                observed.update(
                    request_id=rid,
                    round_id=n,
                    target_forward_start_ns=10 + 100 * n,
                    target_forward_end_ns=20 + 100 * n,
                )
                diagnostics.append(observed)
                final = prefix + committed
                rounds.append(
                    {
                        "request_id": rid,
                        "round_id": n,
                        "parent_prefix_hash": token_prefix_hash(prefix),
                        "parent_prefix_len": len(prefix),
                        "proposal_token_ids": proposal,
                        "committed_token_ids": committed,
                        "committed_prefix_hash": token_prefix_hash(final),
                        "logical_draft_kv_length": len(final),
                        "logical_target_kv_length": len(final),
                        "terminal": terminal,
                        "target_authority": True,
                        "target_correction_token_ids": [] if terminal else committed,
                        "target_bonus_token_ids": [24] if terminal else [],
                        "remaining_output_budget": 0 if terminal else 2,
                        **decision.accounting,
                    }
                )
                prefix = final
        value["requests"] = requests
        append_rows(directory / "round-events.jsonl", rounds)
        append_rows(directory / "target-diagnostics.jsonl", diagnostics)
        metrics = DraftMetrics()
        purposes = (
            ("proposal", "commit")
            if full_accept
            else ("proposal", "commit", "commit") + (("commit",) if label == "hf" else ())
        )
        for purpose in purposes:
            for _ in range(count if label == "hf" else 1):
                metrics.forward(
                    purpose, 1 if label == "hf" else count, 1 if label == "hf" else count
                )
        metrics.counters.update(
            proposals=len(rounds),
            proposed_tokens=(2 if full_accept else 3) * count,
            commits=len(rounds),
            invalidated_tokens=0 if full_accept else 2 * count,
        )
        metrics.syncs["greedy"] = 4 * count
        metrics.gpu_ms["proposal"] = 10.0
        backend = {
            **metrics.snapshot(
                "hf-persistent-kv-correctness-draft"
                if label == "hf"
                else "vllm-batched-paged-kv-draft"
            ),
            "provenance": startup,
            "backend_shutdown_complete": True,
            "execution_failed": False,
            "draft_live_requests_final": 0,
            "worker_resources": {
                "blocks_allocated": count,
                "blocks_freed": count,
                "live_allocator_requests": 0,
                "worker_shutdown_complete": True,
            },
        }
        write(directory / "draft-backend-report.json", backend)
        write(directory / "draft-startup.json", startup)
        ready = json.loads((directory / "draft-service-ready.json").read_text())
        ready.update(backend=backend["backend_name"], provenance=startup)
        write(directory / "draft-service-ready.json", ready)
        raw = json.loads((directory / "resident-serial.json").read_text())
        raw.update(
            valid=True,
            errors=[],
            first_target_forward_valid=True,
            accounting={"valid": True},
            kv_monotonicity={"valid": True},
            batch_invariant_validation={"valid": True},
            strict_serial_timeline={"round_events": len(rounds), "validated_in_runner": True},
            draft_shutdown={
                "draft_backend_report_sha256": sha256_file(directory / "draft-backend-report.json")
            },
            stock_reference={"file_sha256": "reference"},
        )
        raw["engine_residency"]["draft"]["service_provenance"] = startup
        write(directory / "resident-serial.json", raw)
        for key, filename in (
            ("raw_run", "resident-serial.json"),
            ("process_lifecycle", "process-lifecycle.json"),
            ("target_diagnostics", "target-diagnostics.jsonl"),
        ):
            value["artifact_sha256"][key] = sha256_file(directory / filename)
        write(directory / "decode-performance.json", value)
        admission = {
            "schema_version": "specrhythm.phase4b3-serial-admission.v1",
            "valid": True,
            "stage": stage,
            "backend": label,
            "run_identity": identity,
            "d3_qualification_sha256": sha256_file(qualification),
            "request_count": count,
            "requests": {
                f"r{i}": {
                    "maximum_new_tokens": 4,
                    "prompt_token_count": 1,
                    "prompt_token_ids_sha256": token_prefix_hash([i + 1]),
                }
                for i in range(count)
            },
            "workload_sha256": "same",
            "reference_sha256": "reference",
            "d4_comparison_sha256": sha256_file(root / "D4/comparison.json")
            if stage == "D5"
            else None,
        }
        write(root / stage / f"{label}-admission.json", admission)


def compare(root, stage, qualification):
    return compare_directories(
        root / stage / "hf",
        root / stage / "vllm",
        5 if stage == "D4" else 100,
        qualification_path=qualification,
        stage=stage,
        d4_path=root / "D4/comparison.json" if stage == "D5" else None,
    )


def test_qualified_d4_and_d5_preserve_divergence_and_report_work(tmp_path):
    qualification = qualified_root(tmp_path)
    pair(tmp_path, "D4", 5, qualification)
    d4 = compare(tmp_path, "D4", qualification)
    assert d4["stage_qualified"], d4["errors"]
    assert not d4["hf_draft_exact"] and d4["hf_vllm_divergent_request_count"] == 5
    assert d4["hf_vllm_divergent_rounds"] == [0]
    assert d4["performance_interpretation_allowed"]
    write(tmp_path / "D4/comparison.json", d4)
    assert require_progression(qualification, "D5", tmp_path / "D4/comparison.json")["valid"]
    pair(tmp_path, "D5", 100, qualification)
    d5 = compare(tmp_path, "D5", qualification)
    assert d5["stage_qualified"] and d5["performance_interpretation_allowed"], d5["errors"]
    assert not d5["pure_batching_speedup_claim"] and not d5["d6_allowed"]
    for label in ("hf", "vllm"):
        work = d5["work_accounting"][label]
        assert work["completed_requests"] == 100 and work["measured_committed_tokens"] == 300
        assert (
            work["target_forward_count"] == 2
        )  # Shared forward counted once across request rows.
        assert work["target_query_tokens"] == 500
        assert work["verification_round_count"] == work["draft_proposal_count"] == 200
        assert work["proposed_tokens"] == 300 and work["rejected_draft_tokens"] == 200
        assert work["accepted_draft_tokens"] == 100 and work["mean_accepted_length"] == 0.5
        assert work["correction_count"] == work["bonus_count"] == 100
    assert d5["work_accounting"]["hf"]["draft_model_forward_count"] == 400
    assert d5["work_accounting"]["vllm"]["draft_model_forward_count"] == 3
    assert d5["execution_time_comparison"]["decode_makespan_reduction_percent"] == 50.0
    assert d5["proposal_acceptance_target_work_nearly_matched"]


def test_different_acceptance_and_target_work_remain_explicit_in_d5(tmp_path):
    qualification = qualified_root(tmp_path)
    pair(tmp_path, "D4", 5, qualification)
    write(tmp_path / "D4/comparison.json", compare(tmp_path, "D4", qualification))
    pair(tmp_path, "D5", 100, qualification, different_work=True)
    result = compare(tmp_path, "D5", qualification)
    assert result["stage_qualified"], result["errors"]
    assert result["performance_interpretation_allowed"]
    assert not result["pure_batching_speedup_claim"]
    assert not result["proposal_acceptance_target_work_nearly_matched"]
    assert result["work_deltas"]["target_forward_count"]["relative_change"] == -0.5
    assert result["work_deltas"]["target_query_tokens"]["vllm_minus_hf"] == -200
    assert result["work_deltas"]["accepted_draft_tokens"]["vllm_minus_hf"] == 100
    assert result["hf_exact_diagnostic"]["hf_only_context_count"] == 100
    assert not result["hf_draft_exact"]
    assert result["production_performance_label"] == (
        "production vLLM Batched Draft end-to-end improvement"
    )


@pytest.mark.parametrize(
    "fault",
    [
        "target-row",
        "frontier",
        "accounting",
        "budget",
        "limit",
        "target-hash",
        "admission",
        "d3-commit",
        "batch",
    ],
)
def test_hf_policy_never_waives_d4_internal_validation(tmp_path, fault):
    qualification = qualified_root(tmp_path)
    pair(tmp_path, "D4", 5, qualification)
    directory = tmp_path / "D4/vllm"
    if fault in ("target-row", "frontier", "accounting", "budget", "target-hash"):
        if fault == "target-hash":
            value = json.loads((directory / "decode-performance.json").read_text())
            value["artifact_sha256"]["target_diagnostics"] = "changed"
            write(directory / "decode-performance.json", value)
        else:
            rows = CheckpointJsonl(directory / "round-events.jsonl").read()
            if fault == "target-row":
                rows[0]["proposal_token_ids"][0] += 1
            elif fault == "frontier":
                rows[0]["logical_draft_kv_length"] += 1
            elif fault == "budget":
                rows[0]["remaining_output_budget"] += 1
            else:
                rows[0]["accepted_draft_tokens"] = 2
            append_rows(directory / "round-events.jsonl", rows)
    elif fault == "limit":
        path = tmp_path / "D4/vllm-admission.json"
        value = json.loads(path.read_text())
        value["requests"]["r0"]["maximum_new_tokens"] += 1
        write(path, value)
    elif fault == "admission":
        (tmp_path / "D4/vllm-admission.json").unlink()
    elif fault == "d3-commit":
        value = json.loads((directory / "decode-performance.json").read_text())
        value["execution_git_commit"] = "wrong"
        write(directory / "decode-performance.json", value)
    else:
        value = json.loads((directory / "draft-backend-report.json").read_text())
        value["draft_batch_statistics_by_purpose"]["proposal"]["max"] = 1
        write(directory / "draft-backend-report.json", value)
    result = compare(tmp_path, "D4", qualification)
    assert not result["stage_qualified"] and result["errors"]
    assert not result["performance_interpretation_allowed"]


def test_d4_d5_cli_requires_new_qualification_and_stage(monkeypatch):
    from specrhythm.phase4 import draft_comparison

    monkeypatch.setattr(
        "sys.argv",
        ["compare", "--hf", "a", "--vllm", "b", "--request-count", "5", "--output", "c"],
    )
    with pytest.raises(SystemExit) as error:
        draft_comparison.main()
    assert error.value.code == 2
