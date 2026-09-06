"""D4/D5 production-regime qualification and explicit performance/work accounting."""

from __future__ import annotations

import math

from specrhythm.phase4.draft_metrics import batch_statistics
from specrhythm.phase4.draft_qualification import NUMERICAL, read_json
from specrhythm.phase4.manifest import sha256_file
from specrhythm.phase4.serial import greedy_acceptance, token_prefix_hash
from specrhythm.phase4.transport import CheckpointJsonl
from specrhythm.phase4.vllm_diagnostics import validate_target_diagnostic


def production_work(directory, performance, raw, rounds, backend, request_contracts=None):
    """Count rank-0 physical Target forwards once, and query tokens across rows."""
    errors = []
    metrics = performance["metrics"]
    boundary = performance["measurement"]["measurement_start_ns"]
    path = directory / "target-diagnostics.jsonl"
    if performance["artifact_sha256"].get("target_diagnostics") != sha256_file(path):
        errors.append("Target diagnostic evidence hash mismatch")
    diagnostics = [
        r for r in CheckpointJsonl(path).read() if r.get("target_forward_start_ns", -1) >= boundary
    ]
    by_round = {}
    request_ids = {r["request_id"] for r in performance["requests"]}
    if not isinstance(request_contracts, dict) or set(request_contracts) != request_ids:
        errors.append("bound workload request/limit contracts are missing or incomplete")
        request_contracts = {}
    for row in performance["requests"]:
        contract = request_contracts.get(row["request_id"], {})
        limit = contract.get("maximum_new_tokens")
        if (
            type(limit) is not int
            or limit < 1
            or row.get("maximum_new_tokens") not in (None, limit)
            or row.get("prompt_token_count") != contract.get("prompt_token_count")
            or row.get("prompt_token_ids_sha256") != contract.get("prompt_token_ids_sha256")
            or len(row["total_generated_token_ids"]) > limit
            or (
                row["finish_reason"] in ("length", "max_tokens")
                and len(row["total_generated_token_ids"]) != limit
            )
        ):
            errors.append("Serial output differs from the declared workload limit/prompt")
    for row in diagnostics:
        errors.extend(validate_target_diagnostic(row))
        if row.get("request_id") not in request_ids:
            errors.append("Target verification contains an unknown request")
        if row.get("proposal_token_ids"):
            key = (row["request_id"], row.get("round_id"))
            if key in by_round:
                errors.append("duplicate Target verification request/round")
            by_round[key] = row
    if not diagnostics or set(by_round) != {(r["request_id"], r["round_id"]) for r in rounds}:
        errors.append("Target verification/Serial round identity set differs")
    previous = {}
    accepted = rejected = proposed = corrections = bonuses = 0
    for row in rounds:
        try:
            rid = row["request_id"]
            observed = by_round[(rid, row["round_id"])]
            prefix = observed["committed_prefix_token_ids"]
            if (
                observed["proposal_token_ids"] != row["proposal_token_ids"]
                or token_prefix_hash(prefix) != row["parent_prefix_hash"]
                or len(prefix) != row["parent_prefix_len"]
            ):
                errors.append("Target verification proposal/prefix differs from Draft round")
            if rid in previous and (
                row["round_id"] != previous[rid][0] + 1
                or prefix != previous[rid][1]
                or previous[rid][2]
            ):
                errors.append("Serial committed-prefix/round/retirement chain differs")
            if rid not in previous and row["round_id"] != 0:
                errors.append("Serial first round is not zero")
            decision = greedy_acceptance(
                row["proposal_token_ids"], row["committed_token_ids"], terminal=row["terminal"]
            )
            for key, value in decision.accounting.items():
                if row.get(key) != value:
                    errors.append(f"Serial token accounting differs: {key}")
            final = prefix + row["committed_token_ids"]
            contract = request_contracts[rid]
            remaining = contract["maximum_new_tokens"] - (
                len(prefix) - contract["prompt_token_count"]
            )
            if row.get("remaining_output_budget") != remaining - len(
                row["committed_token_ids"]
            ) or not 0 < len(row["proposal_token_ids"]) <= min(4, remaining - 1):
                errors.append("Serial per-round output/proposal budget differs from workload")
            if (
                token_prefix_hash(final) != row["committed_prefix_hash"]
                or len(final) != row["logical_draft_kv_length"]
                or len(final) != row["logical_target_kv_length"]
            ):
                errors.append("Serial committed hash/Draft/Target KV frontier differs")
            if row.get("target_authority") is not True:
                errors.append("Target verification authority missing")
            previous[rid] = (row["round_id"], final, row["terminal"])
            proposed += len(row["proposal_token_ids"])
            accepted += decision.accounting["accepted_draft_tokens"]
            rejected += decision.accounting["rejected_draft_tokens"]
            corrections += decision.accounting["target_correction_tokens"]
            bonuses += decision.accounting["target_bonus_tokens"]
        except (ValueError, KeyError, TypeError) as error:
            errors.append(f"invalid Serial round state: {error}")
    for field in ("accounting", "kv_monotonicity", "batch_invariant_validation"):
        if raw.get(field, {}).get("valid") is not True:
            errors.append(f"raw Serial {field} invalid")
    if raw.get("strict_serial_timeline", {}).get("validated_in_runner") is not True:
        errors.append("Serial lifecycle/timeline was not validated")
    if raw.get("first_target_forward_valid") is not True:
        errors.append("resident first Target verification invalid")
    counters = backend.get("draft_kv_operations", {})
    if (
        backend.get("draft_proposal_count") != len(rounds)
        or backend.get("draft_proposed_token_count") != proposed
        or counters.get("invalidated_tokens") != rejected
    ):
        errors.append("Draft backend proposal/acceptance counters differ from Target rounds")
    for key in ("draft_model_forward_count", "draft_host_sync_count"):
        if type(backend.get(key)) is not int or backend[key] <= 0:
            errors.append(f"missing actual Draft counter: {key}")
    duration = backend.get("draft_gpu_event_time_ms")
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        errors.append("missing actual Draft GPU event duration")
    value = {
        "completed_requests": metrics["completed_requests"],
        "measured_committed_tokens": metrics["total_measured_committed_output_tokens"],
        "requested_output_limits_by_request": {
            rid: r["maximum_new_tokens"] for rid, r in request_contracts.items()
        },
        "decode_makespan_ms": metrics["decode_makespan_ms"],
        "throughput_tokens_per_second": metrics["aggregate_throughput_tokens_per_second"],
        "tpot_ms": metrics.get("tpot_ms"),
        "draft_proposal_count": len(rounds),
        "proposed_tokens": proposed,
        "accepted_draft_tokens": accepted,
        "rejected_draft_tokens": rejected,
        "mean_accepted_length": accepted / len(rounds) if rounds else None,
        "verification_round_count": len(rounds),
        "target_forward_count": len(
            {(r["target_forward_start_ns"], r["target_forward_end_ns"]) for r in diagnostics}
        ),
        "target_query_tokens": sum(r["query_length"] for r in diagnostics),
        "target_work_scope": (
            "measured rank-0 diagnostic forwards, including Target-only tails; "
            "TP replicas not multiplied"
        ),
        "draft_model_forward_count": backend.get("draft_model_forward_count"),
        "draft_batch_size": {
            key: backend.get(f"draft_batch_size_{key}")
            for key in ("min", "p10", "p50", "p90", "max", "mean")
        },
        "draft_gpu_event_time_ms": duration,
        "draft_host_sync_count": backend.get("draft_host_sync_count"),
        "draft_host_sync_scope": backend.get("draft_sync_coverage"),
        "correction_count": corrections,
        "bonus_count": bonuses,
        "jit_warmup": {
            "target_log_evidence": performance.get("jit_observation"),
            "draft_forward_counts_by_purpose": backend.get("draft_model_forward_count_by_purpose"),
            "draft_gpu_event_ms_by_purpose": backend.get("draft_gpu_event_time_ms_by_purpose"),
            "draft_startup": {
                key: backend.get("provenance", {}).get(key)
                for key in ("startup_warmup_ns", "warmup_model_forward_count")
            },
            "hf_startup_load_ns": backend.get("startup_load_ns"),
            "hf_explicit_warmup_forward_count": backend.get("explicit_warmup_model_forward_count"),
        },
    }
    return value, errors


def qualify_serial_pair(
    base, directories, values, raw, rounds, aggregate, qualification_path, stage, d4_path
):
    from collections import Counter

    errors = list(base["errors"])
    work = {}
    for label, directory in directories.items():
        try:
            admission_path = directory.parent / f"{label}-admission.json"
            admission = read_json(admission_path)
            if not (
                admission.get("schema_version") == "specrhythm.phase4b3-serial-admission.v1"
                and admission.get("valid") is True
                and admission.get("stage") == stage
                and admission.get("backend") == label
                and admission.get("run_identity") == aggregate["run_identity"]
                and admission.get("d3_qualification_sha256") == sha256_file(qualification_path)
                and admission.get("request_count") == values[label]["request_count"]
                and admission.get("workload_sha256") == values[label]["workload_sha256"]
                and admission.get("reference_sha256")
                == raw[label].get("stock_reference", {}).get("file_sha256")
                and admission.get("d4_comparison_sha256")
                == (sha256_file(d4_path) if d4_path else None)
            ):
                errors.append(f"{label}: missing/incompatible pre-execution admission")
            backend = read_json(directory / "draft-backend-report.json")
            ready = read_json(directory / "draft-service-ready.json")
            if raw[label].get("draft_shutdown", {}).get(
                "draft_backend_report_sha256"
            ) != sha256_file(directory / "draft-backend-report.json"):
                errors.append(f"{label}: Draft final metrics are not bound to shutdown")
            if (
                backend.get("backend_shutdown_complete") is not True
                or backend.get("execution_failed") is not False
                or backend.get("draft_live_requests_final") != 0
            ):
                errors.append(f"{label}: Draft state/cleanup invalid")
            if backend.get("provenance") != ready.get("provenance"):
                errors.append(f"{label}: Draft metrics/startup provenance differs")
            stats = batch_statistics(
                Counter({int(k): v for k, v in backend["draft_batch_size_histogram"].items()})
            )
            if stats["count"] != backend.get("draft_model_forward_count") or any(
                stats[key] != backend.get(f"draft_batch_size_{key}")
                for key in ("min", "p10", "p50", "p90", "max", "mean")
            ):
                errors.append(f"{label}: actual-forward histogram/metrics disagree")
            if label == "vllm":
                qualified = aggregate["run_identity"]
                for field, expected in (
                    ("model", qualified["model_path"]),
                    ("tokenizer", qualified["model_path"]),
                ):
                    if backend["provenance"].get(field, {}).get("path") != expected:
                        errors.append(f"vllm: {field} differs from D3 qualification")
                if backend["provenance"].get("vllm_api") != qualified["vllm_api"]:
                    errors.append(
                        "vllm: installed production source differs from D3 qualification"
                    )
                proposal_batch = backend.get("draft_batch_statistics_by_purpose", {}).get(
                    "proposal", {}
                )
                if not proposal_batch.get("max") or proposal_batch["max"] <= 1:
                    errors.append("vllm: no real multi-request Draft proposal forward")
            work[label], invalid = production_work(
                directory,
                values[label],
                raw[label],
                rounds[label],
                backend,
                admission.get("requests"),
            )
            errors.extend(f"{label}: {e}" for e in invalid)
            if (
                values[label]["execution_git_commit"]
                != aggregate["run_identity"]["execution_commit"]
            ):
                errors.append(f"{label}: Serial execution differs from D3 qualification commit")
            if (
                values[label]["artifact_sha256"].get("config")
                != aggregate["run_identity"]["config_sha256"]
            ):
                errors.append(f"{label}: Serial config differs from D3 qualification")
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f"{label}: production work evidence invalid: {error}")
    if not base["draft_batch_p50_greater_than_one"]:
        errors.append("vllm: measured Draft batch p50 is not greater than one")
    if (
        len(work) == 2
        and work["hf"]["requested_output_limits_by_request"]
        != work["vllm"]["requested_output_limits_by_request"]
    ):
        errors.append("A/B declared workload output limits differ")
    equivalence = base["draft_backend_equivalence"]
    mismatches = equivalence["mismatches"]
    divergent = {(r["request_id"], r["parent_prefix_hash"]) for r in mismatches}
    deltas = {}
    if len(work) == 2:
        for field in (
            "draft_proposal_count",
            "proposed_tokens",
            "accepted_draft_tokens",
            "rejected_draft_tokens",
            "verification_round_count",
            "target_forward_count",
            "target_query_tokens",
        ):
            left, right = work["hf"][field], work["vllm"][field]
            deltas[field] = {
                "hf": left,
                "vllm": right,
                "vllm_minus_hf": right - left,
                "relative_change": (right - left) / left
                if left
                else (0.0 if right == 0 else None),
            }
    valid = not errors
    timings = None
    if valid:
        left, right = work["hf"]["decode_makespan_ms"], work["vllm"]["decode_makespan_ms"]
        timings = {
            "decode_makespan_reduction_ms": left - right,
            "decode_makespan_reduction_percent": 100 * (left - right) / left,
            "decode_makespan_ratio_hf_over_vllm": left / right,
            "draft_gpu_event_reduction_ms": work["hf"]["draft_gpu_event_time_ms"]
            - work["vllm"]["draft_gpu_event_time_ms"],
        }
    return {
        **base,
        "schema_version": "specrhythm.phase4b3-serial-backend-comparison.v2",
        "stage": stage,
        "errors": errors,
        "stage_qualified": valid,
        "performance_comparable": valid,
        "performance_interpretation_allowed": valid and stage == "D5",
        "ready_for_operator_review": valid,
        "d4_qualified": valid and stage == "D4",
        "d5_qualified": valid and stage == "D5",
        "hf_draft_exact": equivalence["exact_on_common_contexts"]
        and not equivalence["hf_only_context_count"]
        and not equivalence["vllm_only_context_count"],
        "hf_exact_diagnostic": equivalence,
        "hf_vllm_divergent_request_count": len({r["request_id"] for r in mismatches}),
        "hf_vllm_divergent_rounds": sorted(
            {
                r["round_id"]
                for r in rounds["vllm"]
                if (r["request_id"], r["parent_prefix_hash"]) in divergent
            }
        ),
        "hf_vllm_divergence_classification": NUMERICAL,
        "hf_vllm_divergence_evidence_status": "strong_evidence",
        "hf_proposal_equality_is_blocking": False,
        "cross_backend_divergence_qualified": True,
        "mechanistic_root_cause_proven": False,
        "production_performance_label": "production vLLM Batched Draft end-to-end improvement",
        "pure_batching_speedup_claim": False,
        "work_accounting": work,
        "work_deltas": deltas,
        "proposal_acceptance_target_work_nearly_matched": bool(deltas)
        and all(
            d["relative_change"] is not None and abs(d["relative_change"]) <= 0.01
            for d in deltas.values()
        ),
        "nearly_matched_rule": (
            "all listed proposal/acceptance/Target-work counters differ by at most 1%; "
            "descriptive only"
        ),
        "execution_time_comparison": timings,
        "decode_makespan_ratio_hf_over_vllm": timings["decode_makespan_ratio_hf_over_vllm"]
        if timings
        else None,
        "d6_allowed": False,
        "input_sha256": {
            **base["input_sha256"],
            "d3_qualification": sha256_file(qualification_path),
            "d4_comparison": sha256_file(d4_path) if d4_path else None,
            **{
                f"{label}_{name}": sha256_file(directory / name)
                for label, directory in directories.items()
                for name in ("draft-backend-report.json", "target-diagnostics.jsonl")
            },
            **{
                f"{label}_admission": sha256_file(directory.parent / f"{label}-admission.json")
                if (directory.parent / f"{label}-admission.json").is_file()
                else None
                for label, directory in directories.items()
            },
        },
    }
