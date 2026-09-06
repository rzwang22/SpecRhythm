"""Versioned production-Draft qualification; CPU-only evidence and admission checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from specrhythm.phase4.batched_draft_service import write_immutable_report
from specrhythm.phase4.draft_logits_contract import PATHS, classify_paths, load_probe_fixture
from specrhythm.phase4.manifest import sha256_file

D3_SCHEMA = "specrhythm.phase4b3-draft-qualification.v2"
AGGREGATE_SCHEMA = "specrhythm.phase4b3-d1-d3-qualification.v2"
REGIME_SCHEMA = "specrhythm.phase4b3-draft-execution-regime.v1"
NUMERICAL = "hf_vllm_execution_numerical_divergence"
STRUCTURAL_CHECKS = (
    "request_identity",
    "batch_row_mapping",
    "committed_prefix_hash",
    "round_identity",
    "output_budget",
    "eos_subset",
    "request_retirement",
    "physical_kv_ownership",
    "no_observed_block_alias",
    "materialized_frontier",
    "committed_kv_length",
    "actual_proposal_batch",
    "cleanup",
    "provenance",
    "no_internal_inconsistency",
)


def read_json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def qualify_probe_reports(reports):
    """Recompute from all four complete vectors, never trust top1 or equality flags."""
    result = classify_paths(load_probe_fixture(), reports)
    errors = list(result["errors"])
    if result["classification"] != NUMERICAL or result["evidence_status"] != "strong_evidence":
        errors.append("cross-backend divergence has not been qualified")
    deltas = result["raw_logit_deltas"]
    equal = []
    for left, right in ((PATHS[1], PATHS[2]), (PATHS[2], PATHS[3])):
        delta = deltas.get(f"{right}_minus_{left}", {})
        valid = (
            delta.get("full_vector_equal") is True
            and delta.get("full_vector_max_abs_delta") == 0.0
        )
        equal.append(valid)
        if not valid:
            errors.append(f"vLLM internal raw-vector mismatch: {left}/{right}")
    for path in PATHS[1:]:
        diagnostic = reports.get(path, {}).get("diagnostic_frames", {})
        if (
            diagnostic.get("all_structural_checks_passed") is not True
            or diagnostic.get("observer_removed") is not True
        ):
            errors.append(f"{path}: structural diagnostic/cleanup did not pass")
    return {
        "schema_version": REGIME_SCHEMA,
        "valid": not errors,
        "errors": errors,
        "classification": result["classification"],
        "evidence_status": result["evidence_status"],
        "mechanistic_root_cause_proven": False,
        "vllm_batch_invariant_valid": equal[0],
        "vllm_persistent_history_valid": equal[1],
        "run_identity": reports.get(PATHS[0], {}).get("run_identity"),
        "gpu_identity": reports.get(PATHS[0], {}).get("gpu_identity"),
        "top1_pattern": result["top1_pattern"],
        "raw_logit_deltas": deltas,
        "fixture_sha256": result["fixture_sha256"],
        "source_diagnostic_sha256": result["source_diagnostic_sha256"],
        "vectors": {
            path: {
                key: report["logits"][key]
                for key in (
                    "raw_logits_sha256_le_f32",
                    "vocab_size",
                    "token_logits",
                    "top1_token",
                    "original_dtype",
                    "margin_16_minus_227",
                )
            }
            for path, report in reports.items()
            if "logits" in report
        },
        "scope": "pinned production vLLM execution regime; HF remains diagnostic",
    }


def load_probe_qualification(root):
    root = Path(root)
    reports = {path: read_json(root / f"{path}.json") for path in PATHS}
    envelope = read_json(root / "probe-inputs.json")
    original = read_json(root / "classification.json")
    result = qualify_probe_reports(reports)
    if envelope != {"fixture": load_probe_fixture(), "run_identity": result["run_identity"]}:
        result["errors"].append("probe input envelope identity mismatch")
    for path in PATHS:
        if original.get("path_artifact_sha256", {}).get(path) != sha256_file(
            root / f"{path}.json"
        ):
            result["errors"].append(f"probe artifact hash mismatch: {path}")
    for field in ("classification", "evidence_status", "top1_pattern", "raw_logit_deltas"):
        if original.get(field) != result[field]:
            result["errors"].append(f"probe classification disagrees with raw evidence: {field}")
    result["source_artifact_sha256"] = {
        name: sha256_file(root / name)
        for name in ("probe-inputs.json", "classification.json", *(f"{p}.json" for p in PATHS))
    }
    result["valid"] = not result["errors"]
    return result


def compatible_identity(current, qualified):
    # A new diagnostics/qualification commit is intentional. Every numerical input,
    # installed source file, patch, weight and version must still be identical.
    return {k: v for k, v in current.items() if k != "execution_commit"} == {
        k: v for k, v in qualified.items() if k != "execution_commit"
    }


def hf_diagnostics(comparisons):
    mismatches = [
        r for r in comparisons if r.get("same_prefix") is True and r.get("exact") is False
    ]
    unmatched = [r for r in comparisons if r.get("same_prefix") is not True]
    exact = bool(comparisons) and not mismatches and not unmatched
    return {
        "hf_draft_exact": exact,
        "hf_exact_diagnostic": {
            "exact": exact,
            "same_prefix_comparison_count": len(comparisons) - len(unmatched),
            "unmatched_context_count": len(unmatched),
            "unmatched_contexts_are_not_equivalence_evidence": True,
        },
        "hf_vllm_divergent_request_count": len({r["request_id"] for r in mismatches}),
        "hf_vllm_divergent_rounds": sorted({r["round"] for r in mismatches}),
    }


def qualify_d3(observation, backend, regime):
    errors = list(observation.get("errors", []))
    checks = observation.get("structural_checks", {})
    for name in STRUCTURAL_CHECKS:
        if checks.get(name) is not True:
            errors.append(f"D3 structural gate failed: {name}")
    resources = backend.get("worker_resources", {})
    if not (
        backend.get("execution_failed") is False
        and backend.get("backend_shutdown_complete") is True
        and backend.get("draft_live_requests_final") == 0
        and resources.get("live_allocator_requests") == 0
        and resources.get("worker_shutdown_complete") is True
        and type(resources.get("blocks_allocated")) is int
        and resources["blocks_allocated"] == resources.get("blocks_freed")
    ):
        errors.append("D3 backend execution/resource cleanup invalid")
    count = observation.get("requested_batch_size")
    histogram = (
        backend.get("draft_batch_statistics_by_purpose", {})
        .get("proposal", {})
        .get("histogram", {})
    )
    if count not in (2, 4, 8) or histogram.get(str(count), 0) < 1:
        errors.append("D3 did not execute the requested B2/B4/B8 proposal batch")
    if (
        observation.get("diagnostic_only") is not False
        or observation.get("materialization_observer_installed") is not False
    ):
        errors.append("D3 must be a fresh uninstrumented execution")
    if observation.get("completed_requests") != count:
        errors.append("D3 completed request count mismatch")
    if (
        observation.get("runtime_batch_invariance", {}).get("batch_invariant_env_resolved")
        is not True
    ):
        errors.append("D3 worker batch-invariant mode is not resolved")
    compatible = compatible_identity(
        observation.get("run_identity", {}), regime.get("run_identity", {})
    )
    qualified = (
        regime.get("schema_version") == REGIME_SCHEMA
        and regime.get("valid") is True
        and regime.get("errors") == []
        and regime.get("classification") == NUMERICAL
        and regime.get("evidence_status") == "strong_evidence"
        and compatible
        and regime.get("vllm_batch_invariant_valid") is True
        and regime.get("vllm_persistent_history_valid") is True
    )
    semantic = not errors
    if not qualified:
        errors.append("missing/incompatible cross-backend execution-regime qualification")
    return {
        "schema_version": D3_SCHEMA,
        "gate": "D3",
        "requested_batch_size": count,
        "valid": not errors,
        "errors": errors,
        "vllm_semantic_valid": semantic,
        "vllm_batch_invariant_valid": qualified and regime["vllm_batch_invariant_valid"],
        "vllm_persistent_history_valid": qualified and regime["vllm_persistent_history_valid"],
        "cross_backend_divergence_qualified": qualified,
        "d3_qualified": not errors,
        **hf_diagnostics(observation.get("comparisons", [])),
        "hf_vllm_divergence_classification": regime.get("classification"),
        "hf_vllm_divergence_evidence_status": regime.get("evidence_status"),
        "mechanistic_root_cause_proven": False,
        "production_draft_backend": "VllmBatchedDraftBackend",
        "hf_proposal_equality_is_blocking": False,
        "performance_result": False,
        "run_identity": observation.get("run_identity"),
        "structural_checks": checks,
        "regime_qualification": regime,
    }


def validate_d3_directory(root):
    root = Path(root)
    report = read_json(root / "gate.json")
    errors = []
    for name in (
        "observation.json",
        "draft-backend-report.json",
        "regime-qualification.json",
        "draft-startup.json",
        "hf-oracle.json",
    ):
        if report.get("artifact_sha256", {}).get(name) != sha256_file(root / name):
            errors.append(f"D3 bound artifact changed: {name}")
    recomputed = qualify_d3(
        read_json(root / "observation.json"),
        read_json(root / "draft-backend-report.json"),
        read_json(root / "regime-qualification.json"),
    )
    if any(report.get(k) != v for k, v in recomputed.items()):
        errors.append("D3 qualification differs from its underlying observations")
    errors.extend(recomputed["errors"])
    return report, errors


def aggregate_directories(root):
    root = Path(root)
    errors, gates, hashes = [], {}, {}
    for name in ("D1", "D2", "D3-B2", "D3-B4", "D3-B8"):
        directory = root / name
        try:
            gate = read_json(directory / "gate.json")
            startup = read_json(directory / "draft-startup.json")
            backend = read_json(directory / "draft-backend-report.json")
            if name.startswith("D3"):
                gate, invalid = validate_d3_directory(directory)
                errors.extend(f"{name}: {e}" for e in invalid)
                if gate.get("requested_batch_size") != int(name[-1]):
                    errors.append(f"{name}: wrong requested batch size")
            elif not (
                gate.get("schema_version") == "specrhythm.phase4b3-draft-gate.v1"
                and gate.get("gate") == name
                and gate.get("valid") is True
                and gate.get("errors") == []
                and gate.get("diagnostic_only") is False
            ):
                errors.append(f"{name}: legacy gate invalid")
            if name == "D2" and (
                not gate.get("comparisons")
                or any(r.get("draft_proposals_exact") is not True for r in gate["comparisons"])
            ):
                errors.append("D2: retained HF single-request gate did not pass")
            resources = backend.get("worker_resources", {})
            if not (
                backend.get("execution_failed") is False
                and backend.get("backend_shutdown_complete") is True
                and backend.get("draft_live_requests_final") == 0
                and resources.get("live_allocator_requests") == 0
                and resources.get("worker_shutdown_complete") is True
                and resources.get("blocks_allocated") == resources.get("blocks_freed")
            ):
                errors.append(f"{name}: cleanup failed")
            if startup != backend.get("provenance"):
                errors.append(f"{name}: startup/final provenance differs")
            gates[name] = {"gate": gate, "startup": startup}
            for filename in ("gate.json", "draft-startup.json", "draft-backend-report.json"):
                hashes[f"{name}/{filename}"] = sha256_file(directory / filename)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f"{name}: {error}")
    identity = gates.get("D3-B8", {}).get("gate", {}).get("run_identity")
    if len(gates) == 5:
        base = gates["D3-B8"]["startup"]
        stable = (
            "model",
            "tokenizer",
            "dtype",
            "gpu_uuid",
            "physical_gpu_id",
            "logical_cuda_index",
            "world_size",
            "tensor_parallel_size",
            "runner_class",
            "vllm_api",
            "enforce_eager",
            "prefix_caching",
        )
        for name, entry in gates.items():
            if any(base.get(k) is None or entry["startup"].get(k) != base[k] for k in stable):
                errors.append(f"{name}: model/vLLM/config/device provenance incompatible")
            if name.startswith("D3") and entry["gate"].get("run_identity") != identity:
                errors.append(f"{name}: qualification execution identity differs")
    return {
        "schema_version": AGGREGATE_SCHEMA,
        "valid": not errors,
        "errors": errors,
        "d3_qualified": not errors,
        "d4_allowed": not errors,
        "d5_requires_d4": True,
        "d6_allowed": False,
        "run_identity": identity,
        "input_sha256": hashes,
        "gates": {
            name: {
                k: entry["gate"].get(k)
                for k in (
                    "schema_version",
                    "valid",
                    "d3_qualified",
                    "hf_draft_exact",
                    "hf_vllm_divergent_request_count",
                )
            }
            for name, entry in gates.items()
        },
        "regime_qualification": gates.get("D3-B8", {}).get("gate", {}).get("regime_qualification"),
    }


def require_progression(qualification_path, stage, d4_path=None, expected_commit=None):
    path = Path(qualification_path)
    aggregate = read_json(path)
    actual = aggregate_directories(path.parent)
    if (
        aggregate != actual
        or actual["valid"] is not True
        or actual["schema_version"] != AGGREGATE_SCHEMA
    ):
        raise ValueError("D4/D5 require a valid, bound v2 D1-D3 qualification")
    if expected_commit and actual["run_identity"]["execution_commit"] != expected_commit:
        raise ValueError("D3 qualification was not executed at this commit")
    if stage not in ("D4", "D5"):
        raise ValueError("only D4/D5 progression is authorized")
    if stage == "D5":
        if d4_path is None:
            raise ValueError("D5 requires a valid D4 corrected-five report")
        d4 = read_json(d4_path)
        from specrhythm.phase4.draft_comparison import compare_directories

        root = Path(d4_path).parent
        rebuilt = compare_directories(
            root / "hf", root / "vllm", 5, qualification_path=path, stage="D4"
        )
        if d4 != rebuilt or d4.get("stage_qualified") is not True:
            raise ValueError("D5 requires a bound valid D4 corrected-five comparison")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evidence = sub.add_parser("evidence")
    evidence.add_argument("--probe-root", type=Path, required=True)
    evidence.add_argument("--output", type=Path, required=True)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--root", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("check")
    check.add_argument("--qualification", type=Path, required=True)
    check.add_argument("--stage", choices=("D4", "D5"), required=True)
    check.add_argument("--d4", type=Path)
    check.add_argument("--expected-commit", required=True)
    prepare = sub.add_parser("prepare-serial")
    prepare.add_argument("--qualification", type=Path, required=True)
    prepare.add_argument("--stage", choices=("D4", "D5"), required=True)
    prepare.add_argument("--d4", type=Path)
    prepare.add_argument("--expected-commit", required=True)
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--workload", type=Path, required=True)
    prepare.add_argument("--reference", type=Path, required=True)
    prepare.add_argument("--backend", choices=("hf", "vllm"), required=True)
    prepare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "check":
        require_progression(args.qualification, args.stage, args.d4, args.expected_commit)
        print(f"{args.stage} prerequisites valid")
        return 0
    if args.command == "prepare-serial":
        from specrhythm.phase4.draft_logits_probe import preflight
        from specrhythm.phase4.serial import token_prefix_hash
        from specrhythm.phase4.stock_vllm import load_smoke_requests

        prior = require_progression(args.qualification, args.stage, args.d4, args.expected_commit)
        _, identity = preflight(args.config, args.expected_commit, load_probe_fixture())
        if identity != prior["run_identity"]:
            raise ValueError("Serial model/config/weights/source differ from D3 qualification")
        count = 5 if args.stage == "D4" else 100
        if len([line for line in args.workload.read_text().splitlines() if line.strip()]) != count:
            raise ValueError("Serial stage requires the exact corrected workload size")
        requests = load_smoke_requests(args.workload, count)
        result = {
            "schema_version": "specrhythm.phase4b3-serial-admission.v1",
            "valid": True,
            "stage": args.stage,
            "backend": args.backend,
            "run_identity": identity,
            "d3_qualification_sha256": sha256_file(args.qualification),
            "d4_comparison_sha256": sha256_file(args.d4) if args.d4 else None,
            "workload_sha256": sha256_file(args.workload),
            "reference_sha256": sha256_file(args.reference),
            "request_count": count,
            "requests": {
                r.request_id: {
                    "prompt_token_count": len(r.prompt_token_ids),
                    "prompt_token_ids_sha256": token_prefix_hash(r.prompt_token_ids),
                    "maximum_new_tokens": r.maximum_new_tokens,
                }
                for r in requests
            },
        }
        write_immutable_report(args.output, result)
        print("Serial admission valid; no GPU initialized")
        return 0
    result = (
        load_probe_qualification(args.probe_root)
        if args.command == "evidence"
        else aggregate_directories(args.root)
    )
    write_immutable_report(args.output, result)
    print(json.dumps({k: result[k] for k in ("valid", "errors")}))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
