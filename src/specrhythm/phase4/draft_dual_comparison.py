"""D6 offline execution summaries and the fresh three-mode production comparison."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from specrhythm.phase4.batched_draft_service import write_immutable_report
from specrhythm.phase4.draft_metrics import batch_statistics
from specrhythm.phase4.dual_commit import dual_greedy_acceptance
from specrhythm.phase4.manifest import sha256_file
from specrhythm.phase4.matched_work import _request_complete, _tokens
from specrhythm.phase4.process_lifecycle import validate_lifecycle_artifact
from specrhythm.phase4.serial import greedy_acceptance, token_prefix_hash
from specrhythm.phase4.stock_vllm import load_smoke_requests
from specrhythm.phase4.transport import CheckpointJsonl
from specrhythm.phase4.vllm_diagnostics import validate_logits_mapping

BACKEND = "vllm-batched-paged-kv-draft"
RAW = {
    "target": "resident-target.json",
    "serial": "resident-serial.json",
    "dual": "resident-dual.json",
}
CASES = {
    "A": "Dual > Serial > Target: production SD and Dual both improve throughput.",
    "B": (
        "Target > Dual > Serial: Dual helps SD but does not beat Target; examine accepted "
        "length and verification amortization next."
    ),
    "C": (
        "Target > Serial >= Dual: inspect Dual verification fragmentation and rhythm "
        "against the measured Draft cost before any scheduling experiment."
    ),
    "D": (
        "Serial > Target and Dual <= Serial: SD helps, but Dual adds no benefit; "
        "examine scheduling rather than changing the Draft backend."
    ),
}


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _backend(directory, raw):
    path = directory / "draft-backend-report.json"
    value = read(path)
    ready = read(directory / "draft-service-ready.json")
    require(
        value.get("backend_name") == BACKEND
        and ready.get("provenance", {}).get("backend") == BACKEND,
        "wrong production Draft backend",
    )
    require(
        raw.get("draft_shutdown", {}).get("draft_backend_report_sha256") == sha256_file(path),
        "Draft shutdown metrics are not bound to this execution",
    )
    require(
        value.get("execution_failed") is False
        and value.get("backend_shutdown_complete") is True
        and value.get("draft_live_requests_final") == 0
        and value.get("draft_retired_request_count") == raw.get("request_count")
        and value.get("worker_resources", {}).get("live_allocator_requests") == 0,
        "Draft execution/KV cleanup failed",
    )
    histogram = (
        value.get("draft_batch_statistics_by_purpose", {}).get("proposal", {}).get("histogram", {})
    )
    require(
        any(
            int(size) > 1 and type(count) is int and count > 0 for size, count in histogram.items()
        ),
        "no actual multi-request Draft proposal forward",
    )
    return value


def _target_work(directory, boundary, request_ids):
    rows = [
        r
        for r in CheckpointJsonl(directory / "target-diagnostics.jsonl").read()
        if r.get("target_forward_start_ns", -1) >= boundary
    ]
    require(bool(rows), "no measured Target forward evidence")
    for row in rows:
        require(row.get("request_id") in request_ids, "unknown Target verification request")
        require(not validate_logits_mapping(row), "Target proposal/logits mapping invalid")
        query = row.get("query_length")
        require(
            type(query) is int
            and query > 0
            and len(row.get("target_input_token_ids", ())) == query,
            "Target query/input-token accounting invalid",
        )
        require(
            row.get("causal_attention") is True
            and row.get("target_kv_contains_rejected_or_future_tokens") is False,
            "Target causal/KV verification invalid",
        )
        require(
            type(row.get("target_forward_start_ns")) is int
            and type(row.get("target_forward_end_ns")) is int
            and row["target_forward_end_ns"] > row["target_forward_start_ns"],
            "Target forward timing invalid",
        )
    return {
        "target_forward_count": len(
            {(r["target_forward_start_ns"], r["target_forward_end_ns"]) for r in rows}
        ),
        "target_query_tokens": sum(r["query_length"] for r in rows),
        "target_work_scope": (
            "measured rank-0 physical forwards, including Target-only tails; "
            "TP replicas not multiplied"
        ),
    }


def _draft_work(directory, mode, backend):
    path = directory / ("round-events.jsonl" if mode == "serial" else "proposal-events.jsonl")
    rounds = CheckpointJsonl(path).read()
    accepted = proposed = rejected = corrections = bonuses = 0
    for row in rounds:
        decision = (greedy_acceptance if mode == "serial" else dual_greedy_acceptance)(
            row["proposal_token_ids"], row["committed_token_ids"], terminal=row["terminal"]
        )
        require(
            row.get("accepted_draft_tokens") == len(decision.accepted_draft_token_ids)
            and row.get("rejected_draft_tokens") == len(decision.rejected_draft_token_ids),
            "Draft acceptance/token accounting invalid",
        )
        proposed += len(row["proposal_token_ids"])
        accepted += len(decision.accepted_draft_token_ids)
        rejected += len(decision.rejected_draft_token_ids)
        corrections += len(decision.target_correction_token_ids)
        bonuses += len(decision.target_bonus_token_ids)
    require(
        backend.get("draft_proposal_count") == len(rounds)
        and backend.get("draft_proposed_token_count") == proposed
        and backend.get("draft_kv_operations", {}).get("invalidated_tokens") == rejected,
        "Draft proposal/acceptance counters disagree with verified rounds",
    )
    require(positive(backend.get("draft_gpu_event_time_ms")), "invalid Draft GPU event time")
    counts = backend.get("draft_model_forward_count_by_purpose", {})
    require(
        type(backend.get("draft_model_forward_count")) is int
        and backend["draft_model_forward_count"]
        == counts.get("proposal", 0) + counts.get("commit", 0)
        and backend["draft_model_forward_count"] > 0,
        "invalid Draft forward accounting",
    )
    return {
        "draft_proposal_count": len(rounds),
        "proposed_tokens": proposed,
        "accepted_draft_tokens": accepted,
        "rejected_draft_tokens": rejected,
        "mean_accepted_length": accepted / len(rounds) if rounds else None,
        "correction_count": corrections,
        "bonus_count": bonuses,
        "acceptance_counter_scope": (
            "verified proposal rounds; proposal-free Target tails excluded"
        ),
        "draft_model_forward_count": backend["draft_model_forward_count"],
        "draft_proposal_forward_count": counts.get("proposal", 0),
        "draft_commit_forward_count": counts.get("commit", 0),
        "draft_batch_size": {
            k: backend.get("draft_batch_size_" + k)
            for k in ("min", "p10", "p50", "p90", "max", "mean")
        },
        "draft_gpu_event_time_ms": backend["draft_gpu_event_time_ms"],
        "draft_host_sync_count": backend.get("draft_host_sync_count"),
    }


def _dual_work(directory, raw, *, characterization=False):
    overlap = None
    if characterization:
        from specrhythm.phase4.dual_overlap_characterization import read_overlap

        overlap = read_overlap(directory, raw)
        require(overlap["valid"], str(overlap["errors"]))
    else:
        require(raw.get("overlap_gate", {}).get("valid") is True, "physical overlap gate failed")
    require(
        read(directory / "plugin-report.json").get("sampled_row_tp_consensus") is True,
        "Dual sampled-row TP consensus failed",
    )
    overlaps = CheckpointJsonl(directory / "overlap-events.jsonl").read()
    intervals = sorted(
        r["host_interval"]
        for r in overlaps
        if r.get("overlap_duration_ns", 0) > 0 and r.get("host_interval")
    )
    require(characterization or bool(intervals), "no observed physical Draft/Target overlap")
    # Union repeated witness intervals; never sum shared cohort rows as GPU savings.
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    verified = {}
    for row in CheckpointJsonl(directory / "verification-events.jsonl").read():
        verified.setdefault(row["verify_microbatch_id"], len(row["verify_request_ids"]))
    distribution = batch_statistics(Counter(verified.values()))
    return {
        "physical_overlap_valid": overlap["physical_overlap_valid"] if overlap else True,
        **({"overlap_evidence_valid": True} if overlap else {}),
        "observed_overlap_ms": (
            overlap["observed_overlap_ms"] if overlap
            else sum(end - start for start, end in merged) / 1e6
        ),
        "overlap_is_critical_path_time_saved": False,
        "draft_ready_cohorts": read(directory / "draft-backend-report.json").get(
            "dual_cohorts", []
        ),
        "verification_batch_size": distribution,
        "verification_fragmentation": {
            "verification_batch_count": len(verified),
            "singleton_verification_batch_count": sum(size == 1 for size in verified.values()),
            "verified_request_rows": sum(verified.values()),
            "policy_changed": False,
        },
    }


def summarize_run(directory, mode, count, workload, *, smoke=False, characterization=False):
    """Stop at the first material failure; never fabricate downstream failures."""
    result = {
        "schema_version": "specrhythm.phase4b3-d6-run.v1",
        "mode": mode,
        "valid": False,
        "errors": [],
        "performance_result": False,
        "metrics": {},
        "diagnostics": {},
        "input_sha256": {},
    }
    try:
        raw = read(directory / RAW[mode])
        require(
            raw.get("valid") is True and not raw.get("errors"),
            f"runtime failed: {raw.get('errors') or 'valid=false'}",
        )
        lifecycle = read(directory / "process-lifecycle.json")
        require(
            lifecycle.get("run_valid") is True
            and lifecycle.get("target_exit_status") in (None, 0)
            and lifecycle.get("effective_exit_status") in (None, 0),
            "execution return path failed",
        )
        require(
            not validate_lifecycle_artifact(lifecycle)
            and not (directory / "process-lifecycle.active").exists(),
            "request/process cleanup failed",
        )
        require(
            raw.get("request_count") == count and len(raw.get("outputs", [])) == count,
            "runtime completed request count differs",
        )
        backend = _backend(directory, raw) if mode != "target" else None
        if smoke:
            require(mode == "dual", "D6-A smoke is Dual-only")
            result.update(
                valid=True,
                metrics={
                    "completed_requests": count,
                    "draft_batch_p50": backend.get("draft_batch_size_p50"),
                    "draft_live_requests_final": backend["draft_live_requests_final"],
                },
            )
            return result
        performance = read(directory / "decode-performance.json")
        require(
            performance.get("valid") is True
            and not performance.get("errors")
            and performance.get("performance_result") is True
            and performance.get("cleanup_valid") is True,
            f"performance execution invalid: {performance.get('errors') or 'validity flag'}",
        )
        require(
            performance.get("mode") == ("dual-batch" if mode == "dual" else mode),
            "performance mode identity differs",
        )
        for key, name in (
            ("raw_run", RAW[mode]),
            ("process_lifecycle", "process-lifecycle.json"),
            ("target_diagnostics", "target-diagnostics.jsonl"),
        ):
            require(
                performance["artifact_sha256"].get(key) == sha256_file(directory / name),
                f"{key} evidence is not bound to measured execution",
            )
        require(
            performance.get("workload_sha256") == sha256_file(workload),
            "workload identity differs",
        )
        requests = load_smoke_requests(workload, count)
        by_id = {r.request_id: r for r in requests}
        rows, metrics = performance["requests"], performance["metrics"]
        require(
            performance["request_count"] == metrics["completed_requests"] == len(rows) == count
            and {r["request_id"] for r in rows} == set(by_id),
            "completed request set differs",
        )
        for row in rows:
            contract = by_id[row["request_id"]]
            ids = row.get("measured_committed_output_token_ids", [])
            require(
                _request_complete(row)
                and _tokens(ids)
                and row.get("measured_committed_output_token_count") == len(ids)
                and row.get("total_generated_token_ids") == [row.get("bootstrap_token_id"), *ids]
                and len(row["total_generated_token_ids"]) <= contract.maximum_new_tokens
                and (
                    row.get("finish_reason") not in ("length", "max_tokens")
                    or len(row["total_generated_token_ids"]) == contract.maximum_new_tokens
                )
                and row.get("prompt_token_ids_sha256")
                == token_prefix_hash(contract.prompt_token_ids),
                "request token accounting/prompt/output limit invalid",
            )
        require(
            metrics["total_measured_committed_output_tokens"]
            == sum(len(r["measured_committed_output_token_ids"]) for r in rows),
            "total token accounting invalid",
        )
        for key in ("decode_makespan_ms", "aggregate_throughput_tokens_per_second"):
            require(positive(metrics.get(key)), f"{key} must be finite and positive")
        for key in ("mean", "p50", "p90"):
            value = metrics.get("tpot_ms", {}).get(key)
            require(
                positive(value)
                or (
                    value is None and metrics.get("tpot_ms", {}).get("defined_request_count") == 0
                ),
                f"TPOT {key} invalid",
            )
        result["metrics"] = {
            "completed_requests": count,
            "measured_committed_tokens": metrics["total_measured_committed_output_tokens"],
            "decode_makespan_ms": metrics["decode_makespan_ms"],
            "throughput_tokens_per_second": metrics["aggregate_throughput_tokens_per_second"],
            "tpot_ms": metrics["tpot_ms"],
            **_target_work(
                directory, performance["measurement"]["measurement_start_ns"], set(by_id)
            ),
        }
        if backend is not None:
            result["metrics"].update(_draft_work(directory, mode, backend))
        else:
            require(
                read(directory / "plugin-report.json").get("proposal_generation") is False,
                "Target-only performed Draft proposals",
            )
            result["diagnostics"]["target_only_draft_scope"] = (
                "existing shared DecodeReady setup retained; no measured Draft proposals"
            )
        if mode == "dual":
            result["metrics"].update(_dual_work(directory, raw, characterization=characterization))
        result["experiment_identity"] = {
            key: performance.get(key)
            for key in (
                "execution_git_commit",
                "workload_sha256",
                "models",
                "vllm_commit",
                "vllm_version",
                "patch_hashes",
                "placement",
                "gpu_topology",
                "correctness_mode",
            )
        }
        result["experiment_identity"]["config_sha256"] = performance["artifact_sha256"].get(
            "config"
        )
        result["input_sha256"] = {
            "performance": sha256_file(directory / "decode-performance.json"),
            "workload": sha256_file(workload),
        }
        result["diagnostics"].update(
            jit_warmup=performance.get("jit_observation"),
            final_sequences={r["request_id"]: r["total_generated_token_ids"] for r in rows},
        )
        result.update(valid=True, performance_result=True)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        result["errors"] = [str(error)]
    return result


def compare_three(root):
    modes = {mode: read(root / mode / "qualification.json") for mode in RAW}
    errors = []
    for mode, value in modes.items():
        if (
            value.get("mode") != mode
            or value.get("valid") is not True
            or value.get("errors")
            or value.get("performance_result") is not True
        ):
            errors.append(f"{mode}: execution not qualified: {value.get('errors')}")
        if value.get("input_sha256", {}).get("performance") != sha256_file(
            root / mode / "decode-performance.json"
        ):
            errors.append(f"{mode}: current performance summary input differs")
    identities = [r.get("experiment_identity") for r in modes.values()]
    if not identities[0] or any(i != identities[0] for i in identities[1:]):
        errors.append("three-mode commit/server/workload/model/config identity differs")
    if any(r.get("metrics", {}).get("completed_requests") != 100 for r in modes.values()):
        errors.append("three-mode comparison requires 100 completed requests per mode")
    ratios, work_differences, case = {}, {}, None
    if not errors:
        m = {k: v["metrics"] for k, v in modes.items()}
        for numerator, denominator in (
            ("serial", "target"),
            ("dual", "target"),
            ("dual", "serial"),
        ):
            ratios[f"{numerator}_over_{denominator}_throughput"] = (
                m[numerator]["throughput_tokens_per_second"]
                / m[denominator]["throughput_tokens_per_second"]
            )
            ratios[f"{denominator}_over_{numerator}_makespan"] = (
                m[denominator]["decode_makespan_ms"] / m[numerator]["decode_makespan_ms"]
            )
            for quantile in ("mean", "p50", "p90"):
                a, b = m[numerator]["tpot_ms"][quantile], m[denominator]["tpot_ms"][quantile]
                ratios[f"{numerator}_vs_{denominator}_tpot_{quantile}_change_percent"] = (
                    100 * (a / b - 1) if a is not None and b is not None else None
                )
        target, serial, dual = [m[k]["throughput_tokens_per_second"] for k in RAW]
        work_differences = {
            key: m["dual"][key] - m["serial"][key]
            for key in (
                "measured_committed_tokens",
                "draft_proposal_count",
                "proposed_tokens",
                "accepted_draft_tokens",
                "rejected_draft_tokens",
                "correction_count",
                "bonus_count",
                "target_forward_count",
                "target_query_tokens",
            )
        }
        case = (
            "A"
            if dual > serial > target
            else "B"
            if target > dual > serial
            else "C"
            if target > serial >= dual
            else "D"
            if serial > target and dual <= serial
            else "unclassified/tie"
        )
    return {
        "schema_version": "specrhythm.phase4b3-d6-comparison.v1",
        "valid": not errors,
        "errors": errors,
        "modes": modes,
        "ratios": ratios,
        "dual_minus_serial_work": work_differences,
        "dual_serial_work_exactly_matched": (
            not any(work_differences.values()) if work_differences else None
        ),
        "interpretation_case": case,
        "interpretation": CASES.get(case, "No predefined strict ordering established."),
        "interpretation_is_descriptive_not_causal_proof": True,
        "production_performance_label": "production vLLM Batched Draft end-to-end improvement",
        "pure_batching_speedup_claim": False,
        "tuning_performed": False,
    }


def render_summary(report):
    lines = ["Metric | Target | Serial-vLLM | Dual-vLLM", "--- | ---: | ---: | ---:"]
    for key in (
        "completed_requests",
        "measured_committed_tokens",
        "decode_makespan_ms",
        "throughput_tokens_per_second",
        "tpot_ms.mean",
        "tpot_ms.p50",
        "tpot_ms.p90",
        "target_forward_count",
        "target_query_tokens",
        "draft_model_forward_count",
        "draft_batch_size.p50",
        "draft_gpu_event_time_ms",
        "accepted_draft_tokens",
        "mean_accepted_length",
        "physical_overlap_valid",
        "observed_overlap_ms",
    ):
        values = []
        for mode in RAW:
            value = report["modes"][mode].get("metrics", {})
            for part in key.split("."):
                value = value.get(part, "—") if isinstance(value, dict) else "—"
            values.append(f"{value:.6g}" if type(value) is float else str(value))
        lines.append(key + " | " + " | ".join(values))
    jit = [
        (report["modes"][mode].get("diagnostics", {}).get("jit_warmup") or {}).get(
            "post_measurement_jit_event_count", "unavailable"
        )
        for mode in RAW
    ]
    lines.append("post_measurement_jit_event_count | " + " | ".join(map(str, jit)))
    lines.extend(["", f"Valid: {report['valid']}; case: {report['interpretation_case']}", ""])
    lines.extend([report["interpretation"], ""])
    lines.extend(f"- {k}: {v}" for k, v in report["ratios"].items())
    lines.extend(["", "Dual minus Serial work: " + json.dumps(report["dual_minus_serial_work"])])
    lines.append(
        "\nObserved overlap is not critical-path time saved. No tuning or pure-batching claim."
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--run-root", type=Path, required=True)
    validate.add_argument("--mode", choices=tuple(RAW), required=True)
    validate.add_argument("--request-count", type=int, required=True)
    validate.add_argument("--workload", type=Path, required=True)
    validate.add_argument("--smoke", action="store_true")
    compare = sub.add_parser("compare")
    compare.add_argument("--root", type=Path, required=True)
    compare.add_argument("--markdown", type=Path, required=True)
    for p in (validate, compare):
        p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "output must be fresh")
    if args.command == "compare":
        require(not args.markdown.exists(), "Markdown output must be fresh")
        result = compare_three(args.root)
        args.markdown.write_text(render_summary(result))
        print(render_summary(result))
    else:
        result = summarize_run(
            args.run_root, args.mode, args.request_count, args.workload, smoke=args.smoke
        )
        print(json.dumps({k: result[k] for k in ("valid", "errors", "metrics")}, indent=2))
    write_immutable_report(args.output, result)
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
