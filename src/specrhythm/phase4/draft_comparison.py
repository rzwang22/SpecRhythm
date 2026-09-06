"""Offline HF Serial / batched Serial comparison, with separate equality evidence."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from specrhythm.phase4.batched_draft_service import write_immutable_report
from specrhythm.phase4.draft_metrics import batch_statistics
from specrhythm.phase4.manifest import sha256_file
from specrhythm.phase4.matched_work import _request_complete, _sha, _tokens
from specrhythm.phase4.process_lifecycle import validate_lifecycle_artifact
from specrhythm.phase4.transport import CheckpointJsonl


def compare_serial_work(a: Mapping[str, Any], b: Mapping[str, Any], count: int) -> dict:
    """Same-mode form of the existing completed-work policy; never relabel a mode."""
    errors = [] if type(count) is int and count > 0 else ["invalid expected request count"]
    work = []
    for label, value in (("hf", a), ("vllm", b)):
        if (
            value.get("schema_version") != "specrhythm.phase4b2-decode-performance.v1"
            or value.get("mode") != "serial"
            or value.get("valid") is not True
            or value.get("errors") != []
            or value.get("performance_result") is not True
            or value.get("cleanup_valid") is not True
        ):
            errors.append(f"{label}: invalid Serial performance artifact")
        rows = value.get("requests", [])
        by_id = {row["request_id"]: row for row in rows}
        metrics = value.get("metrics", {})
        for field in ("decode_makespan_ms", "aggregate_throughput_tokens_per_second"):
            number = metrics.get(field)
            if type(number) not in (int, float) or not math.isfinite(number) or number <= 0:
                errors.append(f"{label}: {field} must be finite and positive")
        if not (
            len(rows)
            == len(by_id)
            == count
            == value.get("request_count")
            == metrics.get("completed_requests")
        ):
            errors.append(f"{label}: completed request count differs")
        for rid, row in by_id.items():
            ids = row.get("measured_committed_output_token_ids", [])
            if (
                not _request_complete(row)
                or not _tokens(ids)
                or type(row.get("bootstrap_token_id")) is not int
                or row.get("bootstrap_token_id", -1) < 0
                or not _sha(row.get("prompt_token_ids_sha256"))
                or row.get("setup_committed_output_tokens") != 1
                or row.get("measured_committed_output_token_count") != len(ids)
                or row.get("total_generated_token_ids") != [row.get("bootstrap_token_id"), *ids]
            ):
                errors.append(f"{label}/{rid}: token accounting differs")
        boundary = value.get("measurement", {})
        if (
            boundary.get("setup_excluded") is not True
            or boundary.get("bootstrap_excluded_from_measured_token_count") is not True
            or value.get("mode_semantics", {}).get("draft_target_overlap") is not False
            or value.get("mode_semantics", {}).get("initial_proposal_after_measurement_start")
            is not True
        ):
            errors.append(f"{label}: invalid Serial measurement semantics")
        if metrics.get("total_measured_committed_output_tokens") != sum(
            row.get("measured_committed_output_token_count", -1) for row in rows
        ):
            errors.append(f"{label}: total committed-token accounting differs")
        work.append(
            {
                rid: {
                    field: row.get(field)
                    for field in (
                        "prompt_token_count",
                        "prompt_token_ids_sha256",
                        "bootstrap_token_id",
                        "maximum_new_tokens",
                        "measured_committed_output_token_count",
                    )
                }
                for rid, row in by_id.items()
            }
        )
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
        "mode_semantics",
    ):
        if not a.get(field) or a.get(field) != b.get(field):
            errors.append(f"A/B {field} differs or is missing")
    for field in ("workload", "config", "topology", "patch_manifest"):
        if not a.get("artifact_sha256", {}).get(field) or a["artifact_sha256"][field] != b.get(
            "artifact_sha256", {}
        ).get(field):
            errors.append(f"A/B {field} artifact hash differs")
    boundaries = [
        {
            k: v
            for k, v in value.get("measurement", {}).items()
            if k not in ("measurement_start_ns", "measurement_end_ns")
        }
        for value in (a, b)
    ]
    if not boundaries[0] or boundaries[0] != boundaries[1]:
        errors.append("A/B measurement boundary differs")
    if work[0] != work[1]:
        errors.append("A/B per-request completed work differs")
    sequences = [
        {r["request_id"]: r["total_generated_token_ids"] for r in v.get("requests", [])}
        for v in (a, b)
    ]
    return {
        "valid": not errors,
        "errors": errors,
        "final_target_sequences_equal": sequences[0] == sequences[1],
        "final_target_equality_is_performance_gate": False,
    }


def proposal_equivalence(a, b) -> dict:
    def index(rows):
        result = {}
        for row in rows:
            key = (row["request_id"], row["parent_prefix_hash"], row["parent_prefix_len"])
            if key in result:
                raise ValueError("duplicate proposal context in Serial evidence")
            result[key] = row["proposal_token_ids"]
        return result

    left, right = index(a), index(b)
    common = left.keys() & right.keys()
    mismatches = [
        {
            "request_id": k[0],
            "parent_prefix_hash": k[1],
            "hf_tokens": left[k],
            "vllm_tokens": right[k],
        }
        for k in sorted(common)
        if left[k] != right[k]
    ]
    return {
        "common_context_count": len(common),
        "hf_only_context_count": len(left.keys() - common),
        "vllm_only_context_count": len(right.keys() - common),
        "mismatches": mismatches,
        "exact_on_common_contexts": bool(common) and not mismatches,
        "unmatched_contexts_are_not_equivalence_evidence": True,
    }


def validate_batch_evidence(evidence, rounds) -> list[str]:
    errors = []
    hist = evidence.get("draft_batch_size_histogram", {})
    try:
        histogram = Counter({int(k): v for k, v in hist.items()})
        if any(k < 1 or type(v) is not int or v < 1 for k, v in histogram.items()):
            raise ValueError("invalid histogram")
        stats = batch_statistics(histogram)
        if any(
            evidence.get(f"draft_batch_size_{k}") != stats[k]
            for k in ("min", "p10", "p50", "p90", "max", "mean")
        ):
            errors.append("Draft batch statistics disagree with actual-forward histogram")
        if stats["count"] != evidence.get("draft_model_forward_count"):
            errors.append("Draft histogram/forward count mismatch")
    except (TypeError, ValueError):
        errors.append("malformed Draft batch histogram")
    by_purpose = evidence.get("draft_model_forward_count_by_purpose", {})
    if evidence.get("draft_model_forward_count") != sum(
        by_purpose.get(p, 0) for p in ("proposal", "commit")
    ):
        errors.append("Draft forward counts disagree across purposes")
    if (
        evidence.get("draft_proposal_count") != len(rounds)
        or evidence.get("draft_proposed_token_count")
        != sum(len(r["proposal_token_ids"]) for r in rounds)
        or evidence.get("draft_kv_operations", {}).get("commits") != len(rounds)
    ):
        errors.append("Draft counters disagree with completed Serial round evidence")
    duration = evidence.get("draft_gpu_event_time_ms")
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        errors.append("Draft GPU event duration is missing or invalid")
    resources = evidence.get("worker_resources", {})
    if (
        resources.get("blocks_allocated") != resources.get("blocks_freed")
        or resources.get("worker_shutdown_complete") is not True
    ):
        errors.append("Draft physical allocation cleanup evidence is incomplete")
    return errors


def compare_directories(
    hf: Path, vllm: Path, count: int, *, qualification_path=None, stage=None, d4_path=None
) -> dict:
    aggregate = None
    if qualification_path is not None or stage is not None:
        from specrhythm.phase4.draft_qualification import require_progression

        if stage not in ("D4", "D5") or count != (5 if stage == "D4" else 100):
            raise ValueError("D4 requires corrected-five; D5 requires corrected-100")
        aggregate = require_progression(qualification_path, stage, d4_path)

    def read(directory, name):
        return json.loads((directory / name).read_text())

    values, rounds, raw = {}, {}, {}
    errors = []
    for label, directory in (("hf", hf), ("vllm", vllm)):
        values[label] = read(directory, "decode-performance.json")
        raw[label] = read(directory, "resident-serial.json")
        lifecycle = read(directory, "process-lifecycle.json")
        errors.extend(validate_lifecycle_artifact(lifecycle))
        if raw[label].get("valid") is not True or raw[label].get("errors") != []:
            errors.append(f"{label}: raw Serial invalid")
        for key, filename in (
            ("raw_run", "resident-serial.json"),
            ("process_lifecycle", "process-lifecycle.json"),
        ):
            if values[label]["artifact_sha256"].get(key) != sha256_file(directory / filename):
                errors.append(f"{label}: {key} evidence hash mismatch")
        if (directory / "process-lifecycle.active").exists():
            errors.append(f"{label}: process lifecycle guard still active")
        rounds[label] = CheckpointJsonl(directory / "round-events.jsonl").read()
        if raw[label].get("strict_serial_timeline", {}).get("round_events") != len(rounds[label]):
            errors.append(f"{label}: round count disagrees with raw runtime evidence")
    matched = compare_serial_work(values["hf"], values["vllm"], count)
    errors.extend(matched["errors"])
    evidence = read(vllm, "draft-backend-report.json")
    errors.extend(validate_batch_evidence(evidence, rounds["vllm"]))
    shutdown = raw["vllm"].get("draft_shutdown", {})
    if shutdown.get("draft_backend_report_sha256") != sha256_file(
        vllm / "draft-backend-report.json"
    ):
        errors.append("batched backend final evidence not bound to runtime shutdown")
    if (
        evidence.get("execution_failed") is not False
        or evidence.get("backend_shutdown_complete") is not True
        or evidence.get("draft_live_requests_final") != 0
        or evidence.get("worker_resources", {}).get("live_allocator_requests") != 0
    ):
        errors.append("batched backend execution/cleanup invalid")
    for label, directory, expected in (
        ("hf", hf, "hf-persistent-kv-correctness-draft"),
        ("vllm", vllm, "vllm-batched-paged-kv-draft"),
    ):
        ready = read(directory, "draft-service-ready.json")
        if ready.get("backend") != expected:
            errors.append(f"{label}: wrong Draft backend")
        if label == "vllm":
            startup = read(directory, "draft-startup.json")
            raw_provenance = (
                raw[label].get("engine_residency", {}).get("draft", {}).get("service_provenance")
            )
            if not (
                startup == ready.get("provenance") == evidence.get("provenance") == raw_provenance
            ):
                errors.append("batched startup identity differs across runtime artifacts")
    equivalence = proposal_equivalence(rounds["hf"], rounds["vllm"])
    p50 = evidence.get("draft_batch_size_p50")
    true_batch = type(p50) in (int, float) and p50 > 1
    # This estimate is explicitly tied to the retained HF implementation, not a GPU counter.
    hf_forwards = sum(
        max(len(r["proposal_token_ids"]) - 1, 0)
        + int(
            bool(r["proposal_token_ids"])
            and r["accepted_draft_tokens"] == len(r["proposal_token_ids"])
        )
        + len(r["target_correction_token_ids"])
        + len(r["target_bonus_token_ids"])
        for r in rounds["hf"]
    )
    result = {
        "schema_version": "specrhythm.phase4b3-serial-backend-comparison.v1",
        "performance_comparable": not errors,
        "performance_interpretation_allowed": not errors
        and equivalence["exact_on_common_contexts"],
        "errors": errors,
        "matched_work": matched,
        "draft_backend_equivalence": equivalence,
        "draft_batch_p50_greater_than_one": true_batch,
        "ready_for_operator_review": not errors
        and true_batch
        and equivalence["exact_on_common_contexts"],
        "dual_go_no_go": "operator decision after D5; no automatic speedup threshold",
        "metrics": {k: v["metrics"] for k, v in values.items()},
        "hf_derived_model_forward_count": hf_forwards,
        "vllm_actual_model_forward_count": evidence.get("draft_model_forward_count"),
        "vllm_draft_evidence": evidence,
        "decode_makespan_ratio_hf_over_vllm": (
            values["hf"]["metrics"]["decode_makespan_ms"]
            / values["vllm"]["metrics"]["decode_makespan_ms"]
            if not errors and equivalence["exact_on_common_contexts"]
            else None
        ),
        "jit_warmup": {k: v.get("jit_observation") for k, v in values.items()},
        "input_sha256": {
            "hf": sha256_file(hf / "decode-performance.json"),
            "vllm": sha256_file(vllm / "decode-performance.json"),
            "draft_backend": sha256_file(vllm / "draft-backend-report.json"),
            "hf_rounds": sha256_file(hf / "round-events.jsonl"),
            "vllm_rounds": sha256_file(vllm / "round-events.jsonl"),
            "draft_startup": sha256_file(vllm / "draft-startup.json"),
        },
    }
    if aggregate is not None:
        from specrhythm.phase4.draft_production_comparison import qualify_serial_pair

        return qualify_serial_pair(
            result,
            {"hf": hf, "vllm": vllm},
            values,
            raw,
            rounds,
            aggregate,
            qualification_path,
            stage,
            d4_path,
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", type=Path, required=True)
    parser.add_argument("--vllm", type=Path, required=True)
    parser.add_argument("--request-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--stage", choices=("D4", "D5"), required=True)
    parser.add_argument("--d4", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("comparison output must be fresh")
    report = compare_directories(
        args.hf,
        args.vllm,
        args.request_count,
        qualification_path=args.qualification,
        stage=args.stage,
        d4_path=args.d4,
    )
    write_immutable_report(args.output, report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "performance_comparable",
                    "errors",
                    "draft_batch_p50_greater_than_one",
                    "ready_for_operator_review",
                )
            }
        )
    )
    return 0 if report["ready_for_operator_review"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
