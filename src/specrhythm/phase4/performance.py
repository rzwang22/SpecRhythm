"""Dependency-free Phase-4B.2 decode-only measurement and comparison layer."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Optional, Sequence

from specrhythm.phase4.decode_ready import load_decode_ready_manifest
from specrhythm.phase4.manifest import atomic_write_json, sha256_file
from specrhythm.phase4.matched_work import compare_matched_work, exact_sequence_diagnostic
from specrhythm.phase4.performance_boundary import (
    PERFORMANCE_COMMIT_SCHEMA,
    extract_performance_boundary,
)
from specrhythm.phase4.process_lifecycle import validate_lifecycle_artifact
from specrhythm.phase4.resident_setup import load_deferred_initial_proposals_ready
from specrhythm.phase4.transport import CheckpointJsonl

PERFORMANCE_SCHEMA = "specrhythm.phase4b2-decode-performance.v1"
COMPARISON_SCHEMA = "specrhythm.phase4b2-decode-performance-comparison.v2"
MODES = ("target", "serial", "dual-batch")
CONSUMERS = {
    "target": "target-only",
    "serial": "serial",
    "dual-batch": "dual-batch",
}
RUN_FILES = {
    "target": "resident-target.json",
    "serial": "resident-serial.json",
    "dual-batch": "resident-dual.json",
}
JIT_MARKERS = ("triton", "jit", "compil", "persistent matmul")
GATE3_QUALIFICATION = {
    "gate3_exact_stock_equivalence": False,
    "exact_stock_trajectory": "96/100",
    "logical_correctness_qualification": "pass",
    "numerical_qualification": "complete",
    "classification": "cross-execution-regime bootstrap numerical divergence",
    "async_scheduling_ruled_out": True,
    "further_micro_diagnostics": "deferred",
    "phase4b2_progression_permitted": True,
}
LEGACY_SERIAL_METADATA_COMMIT = "56bd0a50e3b5f33cf30e32564532b1483ea7e34d"
SERIAL_RUN_SCHEMA = "specrhythm.phase4-serial-disaggregated-run.v1"
RUNTIME_BUNDLE_SCHEMA = "specrhythm.phase4-runtime-bundle.v1"


def _measurement_code_git_commit() -> Optional[str]:
    repository = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        capture_output=True,
        check=False,
        text=True,
    )
    value = completed.stdout.strip()
    if completed.returncode != 0 or len(value) != 40:
        return None
    try:
        int(value, 16)
    except ValueError:
        return None
    return value


def percentile(values: Sequence[float], probability: float) -> Optional[float]:
    """Return a deterministic linearly interpolated percentile."""

    if not values:
        return None
    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("percentile values must be finite")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_metrics(
    request_rows: Sequence[Mapping[str, Any]], measurement_start_ns: int
) -> dict[str, Any]:
    if not request_rows:
        raise ValueError("decode performance requires at least one request")
    final_commits = [int(row["final_measured_commit_ns"]) for row in request_rows]
    if any(timestamp < measurement_start_ns for timestamp in final_commits):
        raise ValueError("a final commit predates the performance boundary")
    makespan_ns = max(final_commits) - measurement_start_ns
    if makespan_ns <= 0:
        raise ValueError("decode makespan must be positive")
    latencies = [float(row["decode_latency_ms"]) for row in request_rows]
    tpots = [float(row["tpot_ms"]) for row in request_rows if row.get("tpot_ms") is not None]
    total = sum(int(row["measured_committed_output_token_count"]) for row in request_rows)
    return {
        "completed_requests": len(request_rows),
        "total_measured_committed_output_tokens": total,
        "decode_makespan_ms": makespan_ns / 1_000_000,
        "aggregate_throughput_tokens_per_second": total * 1_000_000_000 / makespan_ns,
        "decode_latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p90": percentile(latencies, 0.90),
            "p99": percentile(latencies, 0.99),
        },
        "tpot_ms": {
            "defined_request_count": len(tpots),
            "undefined_one_token_request_count": len(request_rows) - len(tpots),
            "mean": mean(tpots) if tpots else None,
            "p50": percentile(tpots, 0.50),
            "p90": percentile(tpots, 0.90),
            "p99": percentile(tpots, 0.99),
        },
    }


def build_decode_performance_result(
    *,
    mode: str,
    run_root: Path,
    workload_path: Path,
    config_path: Path,
    topology_path: Path,
    patch_manifest_path: Path,
    output_path: Path,
    terminal_revalidation_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Validate one immutable resident run and derive decode-only metrics."""

    if mode not in MODES:
        raise ValueError("Phase-4B.2 mode must be target, serial, or dual-batch")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite performance result {output_path}")
    terminal_revalidation = None
    if terminal_revalidation_path is not None:
        if mode != "dual-batch":
            raise ValueError("terminal-state revalidation is restricted to Dual-Batch")
        if run_root.resolve().parent in output_path.resolve().parents:
            raise ValueError("recovered performance must be outside the immutable source root")
        from specrhythm.phase4.dual_terminal_recovery import validate_recovery_certificate

        terminal_revalidation = validate_recovery_certificate(
            terminal_revalidation_path,
            run_root=run_root,
            workload_path=workload_path,
            config_path=config_path,
            topology_path=topology_path,
            patch_manifest_path=patch_manifest_path,
        )
    raw_path = run_root / RUN_FILES[mode]
    manifest_path = run_root / "decode-ready-manifest.json"
    setup_ready_path = run_root / "setup-ready.json"
    timing_path = run_root / "timing-events.jsonl"
    diagnostic_path = run_root / "target-diagnostics.jsonl"
    lifecycle_path = run_root / "process-lifecycle.json"
    plugin_path = run_root / "plugin-report.json"
    timestamped_log_path = run_root / "timestamped-target-log.jsonl"
    raw = _read_object(raw_path)
    manifest = load_decode_ready_manifest(_read_object(manifest_path))
    setup_ready = _read_object(setup_ready_path)
    lifecycle = _read_object(lifecycle_path)
    plugin = _read_object(plugin_path)
    timing_rows = CheckpointJsonl(timing_path).read()
    CheckpointJsonl(diagnostic_path).read()
    boundary, errors = extract_performance_boundary(timing_rows, consumer=CONSUMERS[mode])
    errors = list(errors)
    (
        performance_candidate,
        final_sync_value,
        metadata_provenance,
        metadata_errors,
    ) = _resolve_phase4b2_execution_metadata(
        mode=mode,
        run_root=run_root,
        raw=raw,
        manifest=manifest,
        workload_path=workload_path,
        config_path=config_path,
        topology_path=topology_path,
        patch_manifest_path=patch_manifest_path,
    )
    errors.extend(metadata_errors)
    if raw.get("valid") is not True and terminal_revalidation is None:
        errors.append("underlying resident run is invalid")
    if performance_candidate is not True:
        errors.append("underlying run did not request Phase-4B.2 measurement")
    errors.extend(validate_lifecycle_artifact(lifecycle))
    if lifecycle.get("run_valid") is not True and terminal_revalidation is None:
        errors.append("owned Target process run is invalid")
    if setup_ready.get("global_decode_ready") is not True:
        errors.append("setup-ready did not prove global decode readiness")
    if boundary is None:
        boundary = -1
    if boundary <= int(setup_ready.get("ready_published_ns", -1)):
        errors.append("performance boundary did not exclude setup-ready publication")
    if boundary <= manifest.measurement_start_ns:
        errors.append("performance boundary did not follow the correctness boundary")
    if manifest.workload_sha256 != sha256_file(workload_path):
        errors.append("workload checksum differs from decode-ready provenance")
    output_rows = raw.get("outputs")
    if not isinstance(output_rows, list):
        errors.append("resident run outputs are missing")
        output_rows = []
    event_rows, event_errors = _commit_events(
        mode,
        run_root,
        timing_rows,
        boundary,
    )
    errors.extend(event_errors)
    request_metrics, accounting_errors = _request_metrics(
        output_rows,
        manifest.requests,
        event_rows,
        boundary,
        workload_path,
    )
    errors.extend(accounting_errors)
    errors.extend(_validate_mode_boundary(mode, run_root, raw, plugin, boundary))
    final_sync, final_sync_errors = _validate_final_sync(
        final_sync_value,
        manifest.target_tensor_parallel_size,
        manifest.target_physical_gpu_ids,
        max(
            (int(row["final_measured_commit_ns"]) for row in request_metrics),
            default=boundary,
        ),
    )
    errors.extend(final_sync_errors)
    measurement_end_ns = max(
        (
            int(row["final_cuda_synchronize_complete_ns"])
            for row in final_sync
            if isinstance(row.get("final_cuda_synchronize_complete_ns"), int)
        ),
        default=None,
    )
    jit = _jit_evidence(timestamped_log_path, boundary)
    errors.extend(jit["errors"])
    measurement_code_git_commit = _measurement_code_git_commit()
    if measurement_code_git_commit is None:
        errors.append("measurement code git commit is unavailable")
    metrics = None
    if request_metrics:
        try:
            metrics = summarize_metrics(request_metrics, boundary)
        except ValueError as error:
            errors.append(str(error))
    valid = not errors and metrics is not None
    result = {
        "schema_version": PERFORMANCE_SCHEMA,
        "mode": mode,
        "valid": valid,
        "errors": errors,
        "performance_result": valid,
        "reports_speedup": False,
        "requires_exact_cross_mode_comparison": False,
        "requires_matched_work_comparison": True,
        "stage": "phase4b2-decode-only-performance-bringup",
        "git_commit": manifest.specrhythm_git_commit,
        "execution_git_commit": manifest.specrhythm_git_commit,
        "measurement_code_git_commit": measurement_code_git_commit,
        "vllm_version": manifest.vllm_version,
        "vllm_commit": manifest.vllm_commit,
        "patch_hashes": list(manifest.vllm_patch_stack_sha256),
        "models": {
            "target": {
                "path": manifest.target_model_path,
                "revision": manifest.target_model_revision,
            },
            "draft": {
                "path": manifest.draft_model_path,
                "revision": manifest.draft_model_revision,
            },
        },
        "workload_sha256": manifest.workload_sha256,
        "correctness_mode": "batch-invariant",
        "placement": {
            "target_physical_gpu_ids": list(manifest.target_physical_gpu_ids),
            "draft_physical_gpu_ids": list(manifest.draft_physical_gpu_ids),
            "target_tensor_parallel_size": manifest.target_tensor_parallel_size,
            "draft_tensor_parallel_size": manifest.draft_tensor_parallel_size,
        },
        "gpu_topology": _read_object(topology_path),
        "measurement": {
            "clock": "time.monotonic_ns",
            "measurement_start_ns": boundary,
            "measurement_end_ns": measurement_end_ns,
            "setup_excluded": True,
            "bootstrap_excluded_from_measured_token_count": True,
            "first_measured_target_forward_consumes_pending_bootstrap": True,
            "first_post_bootstrap_token_counted": True,
            "pre_measurement_tp_barrier": True,
            "pre_measurement_target_cuda_synchronize": True,
            "per_token_cuda_synchronize": False,
            "final_all_target_rank_cuda_synchronize": True,
        },
        "request_count": len(request_metrics),
        "requests": request_metrics,
        "metrics": metrics,
        "warmup_clean": jit["warmup_clean"],
        "jit_observation": jit,
        "cleanup_valid": not validate_lifecycle_artifact(lifecycle),
        "mode_semantics": _mode_semantics(mode, plugin),
        "phase4b2_metadata_provenance": metadata_provenance,
        "gate3_qualification": dict(GATE3_QUALIFICATION),
        "artifact_sha256": {
            "raw_run": sha256_file(raw_path),
            "decode_ready_manifest": sha256_file(manifest_path),
            "setup_ready": sha256_file(setup_ready_path),
            "timing_events": sha256_file(timing_path),
            "target_diagnostics": sha256_file(diagnostic_path),
            "process_lifecycle": sha256_file(lifecycle_path),
            "timestamped_target_log": sha256_file(timestamped_log_path),
            "workload": sha256_file(workload_path),
            "config": sha256_file(config_path),
            "topology": sha256_file(topology_path),
            "patch_manifest": sha256_file(patch_manifest_path),
            **(
                {"runtime_manifest": sha256_file(run_root / "runtime-manifest.json")}
                if mode == "serial" and (run_root / "runtime-manifest.json").is_file()
                else {}
            ),
            **(
                {"initial_proposals_ready": sha256_file(run_root / "initial-proposals-ready.json")}
                if mode == "serial" and (run_root / "initial-proposals-ready.json").is_file()
                else {}
            ),
        },
    }
    if terminal_revalidation is not None:
        result["terminal_state_reconciliation"] = {
            **terminal_revalidation["terminal_state_reconciliation"],
            "certificate_path": str(terminal_revalidation_path.resolve()),
            "certificate_sha256": sha256_file(terminal_revalidation_path),
            "source_resident_valid": raw["valid"],
            "source_process_run_valid": lifecycle["run_valid"],
            "source_target_exit_status": lifecycle["target_exit_status"],
        }
    atomic_write_json(output_path, result)
    return result


def compare_decode_performance_results(
    *,
    target_path: Path,
    serial_path: Path,
    output_path: Path,
    markdown_path: Path,
    dual_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Gate speedups on matched work while retaining exact sequence diagnostics."""

    for path in (output_path, markdown_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite comparison artifact {path}")
    values = {
        "target": _read_object(target_path),
        "serial": _read_object(serial_path),
    }
    if dual_path is not None:
        values["dual-batch"] = _read_object(dual_path)
    matched_work = compare_matched_work(values)
    errors = matched_work["errors"]
    equality = _cross_mode_equality(values)
    sequence_diagnostic = exact_sequence_diagnostic(values)
    complete = "dual-batch" in values
    performance_valid = complete and not errors
    speedups = None
    if not errors:
        target = values["target"]["metrics"]
        serial = values["serial"]["metrics"]
        speedups = {
            "primary_denominator": "decode_makespan_ms",
            "target_vs_serial": _speedup(target, serial),
        }
        if complete:
            dual = values["dual-batch"]["metrics"]
            speedups.update(
                {
                    "target_vs_dual_batch": _speedup(target, dual),
                    "serial_vs_dual_batch": _speedup(serial, dual),
                }
            )
    report = {
        "schema_version": COMPARISON_SCHEMA,
        "valid": not errors,
        "errors": errors,
        "comparison_complete": complete,
        "performance_valid": performance_valid,
        "performance_valid_for_pair": not errors if not complete else None,
        "metrics": {mode: value.get("metrics") for mode, value in values.items()},
        "warmup": {
            mode: {
                "warmup_clean": value.get("warmup_clean"),
                "post_measurement_jit_event_count": value.get("jit_observation", {}).get(
                    "post_measurement_jit_event_count"
                ),
            }
            for mode, value in values.items()
        },
        "execution_provenance": {
            mode: {
                "execution_git_commit": value.get("execution_git_commit", value.get("git_commit")),
                "measurement_code_git_commit": value.get("measurement_code_git_commit"),
            }
            for mode, value in values.items()
        },
        "matched_work_comparability": matched_work,
        "exact_sequence_diagnostic": sequence_diagnostic,
        "exact_correctness_triangle": equality,
        "exact_correctness_triangle_role": "legacy exact diagnostic only; not a performance gate",
        "speedups": speedups,
        "gate3_qualification": dict(GATE3_QUALIFICATION),
        "claim_boundary": (
            "preliminary Phase 4B.2 matched-work decode-only bring-up; "
            "performance-only matched-work comparison; exact generated-token equivalence "
            "and output quality equivalence are not claimed; not a final paper benchmark "
            "or steady-state result"
            if not errors
            else "no performance or speedup claim"
        ),
        "input_sha256": {
            "target": sha256_file(target_path),
            "serial": sha256_file(serial_path),
            **({"dual-batch": sha256_file(dual_path)} if dual_path else {}),
        },
    }
    atomic_write_json(output_path, report)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_comparison_markdown(report), encoding="utf-8")
    return report


def _resolve_phase4b2_execution_metadata(
    *,
    mode: str,
    run_root: Path,
    raw: Mapping[str, Any],
    manifest: Any,
    workload_path: Path,
    config_path: Path,
    topology_path: Path,
    patch_manifest_path: Path,
) -> tuple[Any, Any, dict[str, Any], list[str]]:
    candidate_present = "phase4b2_performance_candidate" in raw
    sync_present = "phase4b2_final_sync" in raw
    provenance = {
        "performance_candidate_source": "raw-run",
        "final_sync_source": "raw-run",
        "legacy_serial_metadata_recovered": False,
        "recovery_allowed_mode": "serial-only",
        "legacy_execution_commit": LEGACY_SERIAL_METADATA_COMMIT,
    }
    if mode != "serial" or (candidate_present and sync_present):
        return (
            raw.get("phase4b2_performance_candidate"),
            raw.get("phase4b2_final_sync"),
            provenance,
            [],
        )
    if candidate_present != sync_present:
        return (
            raw.get("phase4b2_performance_candidate"),
            raw.get("phase4b2_final_sync"),
            provenance,
            [
                "Serial Phase-4B.2 raw metadata is partially present; "
                "runtime-manifest recovery is forbidden"
            ],
        )
    runtime_path = run_root / "runtime-manifest.json"
    provenance.update(
        {
            "performance_candidate_source": "runtime-manifest.phase4a1",
            "final_sync_source": "runtime-manifest.phase4a1",
            "runtime_manifest_file": runtime_path.name,
            "runtime_manifest_sha256": (
                sha256_file(runtime_path) if runtime_path.is_file() else None
            ),
        }
    )
    try:
        runtime = _read_object(runtime_path)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
        return None, None, provenance, [f"legacy Serial metadata recovery failed: {error}"]
    candidate, final_sync, errors = _recover_legacy_serial_metadata(
        raw=raw,
        runtime=runtime,
        manifest=manifest,
        run_root=run_root,
        workload_path=workload_path,
        config_path=config_path,
        topology_path=topology_path,
        patch_manifest_path=patch_manifest_path,
    )
    provenance["legacy_serial_metadata_recovered"] = not errors
    provenance["recovery_validation_errors"] = errors
    return candidate, final_sync, provenance, errors


def _recover_legacy_serial_metadata(
    *,
    raw: Mapping[str, Any],
    runtime: Mapping[str, Any],
    manifest: Any,
    run_root: Path,
    workload_path: Path,
    config_path: Path,
    topology_path: Path,
    patch_manifest_path: Path,
) -> tuple[Any, Any, list[str]]:
    errors = []
    if raw.get("schema_version") != SERIAL_RUN_SCHEMA:
        errors.append("legacy Serial raw artifact schema differs")
    if raw.get("mode") != "serial-disaggregated":
        errors.append("legacy Serial raw artifact mode differs")
    if raw.get("provider_kind") != "resident-warm-start":
        errors.append("legacy Serial raw artifact is not resident")
    if raw.get("valid") is not True:
        errors.append("legacy Serial raw artifact is invalid")
    if runtime.get("schema_version") != RUNTIME_BUNDLE_SCHEMA:
        errors.append("legacy Serial runtime manifest schema differs")
    if runtime.get("stage") != "phase4a1-serial-disaggregated-correctness":
        errors.append("legacy Serial runtime manifest stage differs")
    phase = runtime.get("phase4a1")
    roles = runtime.get("roles")
    role = roles.get("target") if isinstance(roles, Mapping) else None
    if not isinstance(phase, Mapping):
        errors.append("legacy Serial runtime manifest phase4a1 is missing")
        phase = {}
    if not isinstance(role, Mapping):
        errors.append("legacy Serial Target runtime role is missing")
        role = {}
    if phase.get("mode") != "serial-disaggregated":
        errors.append("legacy Serial phase4a1 mode differs")
    if role.get("role") != "target":
        errors.append("legacy Serial runtime role is not Target")

    raw_provenance = _nested_mapping(raw, "provenance")
    role_inputs = _nested_mapping(role, "inputs")
    role_framework = _nested_mapping(role, "framework")
    role_correctness = _nested_mapping(role, "correctness")
    role_engine = _nested_mapping(role, "engine")
    raw_target = _nested_mapping(raw, "target_runtime_configuration")
    raw_patch = _nested_mapping(raw, "patch_manifest")
    raw_reference = _nested_mapping(raw, "stock_reference")

    execution_commit = manifest.specrhythm_git_commit
    _require_same(
        errors,
        "execution git commit",
        execution_commit,
        raw_provenance.get("git_commit"),
        role.get("git_commit"),
        LEGACY_SERIAL_METADATA_COMMIT,
    )
    workload_sha256 = sha256_file(workload_path)
    _require_same(
        errors,
        "workload SHA256",
        workload_sha256,
        manifest.workload_sha256,
        raw_provenance.get("workload_sha256"),
        role_inputs.get("workload_sha256"),
    )
    config_sha256 = sha256_file(config_path)
    _require_same(
        errors,
        "config SHA256",
        config_sha256,
        raw_provenance.get("config_sha256"),
        role_inputs.get("config_sha256"),
    )
    topology_sha256 = sha256_file(topology_path)
    _require_same(
        errors,
        "topology SHA256",
        topology_sha256,
        role_inputs.get("topology_sha256"),
    )
    patch_sha256 = sha256_file(patch_manifest_path)
    _require_same(
        errors,
        "patch-manifest SHA256",
        patch_sha256,
        raw_patch.get("file_sha256"),
        phase.get("patch_manifest_sha256"),
    )
    _require_same(
        errors,
        "patch-manifest filename",
        patch_manifest_path.name,
        raw_patch.get("file"),
        phase.get("patch_manifest_file"),
    )
    _require_same(
        errors,
        "active patch SHA256",
        raw_patch.get("patch_sha256"),
        role_framework.get("vllm_patch_sha256"),
    )
    _require_same(
        errors,
        "stock-reference SHA256",
        raw_reference.get("file_sha256"),
        phase.get("stock_reference_sha256"),
    )
    _require_same(
        errors,
        "stock-reference filename",
        raw_reference.get("file"),
        phase.get("stock_reference_file"),
    )

    draft_ready_path = run_root / "draft-service-ready.json"
    draft_ready_sha256 = sha256_file(draft_ready_path) if draft_ready_path.is_file() else None
    _require_same(
        errors,
        "Draft-ready SHA256",
        draft_ready_sha256,
        phase.get("draft_service_ready_sha256"),
    )
    if phase.get("draft_service_ready_file") != draft_ready_path.name:
        errors.append("Draft-ready filename differs")
    phase_draft = phase.get("draft_service")
    raw_residency = _nested_mapping(raw, "engine_residency")
    raw_draft = _nested_mapping(raw_residency, "draft")
    if not isinstance(phase_draft, Mapping) or phase_draft.get(
        "provenance"
    ) != raw_draft.get("service_provenance"):
        errors.append("Draft-ready provenance differs between raw and runtime artifacts")

    _require_same(
        errors,
        "correctness mode",
        raw.get("correctness_mode"),
        phase.get("correctness_mode"),
        role_correctness.get("mode"),
        "batch-invariant",
    )
    _require_same(
        errors,
        "vLLM source commit",
        raw_provenance.get("vllm_source_commit"),
        role_framework.get("source_commit"),
        manifest.vllm_commit,
        "752a3a504485790a2e8491cacbb35c137339ad34",
    )

    output_rows = raw.get("outputs")
    workload_count = sum(
        1 for line in workload_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    _require_same(
        errors,
        "request count",
        raw.get("request_count"),
        len(output_rows) if isinstance(output_rows, list) else None,
        len(manifest.requests),
        workload_count,
    )
    _require_same(
        errors,
        "Target physical GPU placement",
        raw_target.get("physical_gpu_ids"),
        role_engine.get("physical_gpu_ids"),
        list(manifest.target_physical_gpu_ids),
        [1, 2],
    )
    _require_same(
        errors,
        "Target tensor-parallel size",
        raw_target.get("tensor_parallel_size"),
        role_engine.get("tensor_parallel_size"),
        manifest.target_tensor_parallel_size,
        2,
    )
    if phase.get("phase4b2_performance_candidate") is not True:
        errors.append("legacy Serial runtime performance candidate is not true")
    final_sync = phase.get("phase4b2_final_sync")
    errors.extend(_validate_legacy_serial_sync_shape(final_sync))
    return phase.get("phase4b2_performance_candidate"), final_sync, errors


def _validate_legacy_serial_sync_shape(value: Any) -> list[str]:
    if not isinstance(value, list) or len(value) != 2:
        return ["legacy Serial runtime final synchronization must contain two rows"]
    if any(not isinstance(row, Mapping) for row in value):
        return ["legacy Serial runtime final synchronization row is malformed"]
    errors = []
    if {row.get("global_rank") for row in value} != {0, 1}:
        errors.append("legacy Serial runtime final synchronization global ranks differ")
    if {row.get("local_rank") for row in value} != {0, 1}:
        errors.append("legacy Serial runtime final synchronization local ranks differ")
    if {row.get("physical_gpu_id") for row in value} != {1, 2}:
        errors.append("legacy Serial runtime final synchronization physical GPUs differ")
    if {row.get("world_size") for row in value} != {2}:
        errors.append("legacy Serial runtime final synchronization world size differs")
    if any(
        not isinstance(row.get("final_cuda_synchronize_complete_ns"), int)
        or isinstance(row.get("final_cuda_synchronize_complete_ns"), bool)
        or row["final_cuda_synchronize_complete_ns"] <= 0
        for row in value
    ):
        errors.append("legacy Serial runtime final synchronization timestamps are invalid")
    return errors


def _nested_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    nested = value.get(key)
    return nested if isinstance(nested, Mapping) else {}


def _require_same(errors: list[str], label: str, *values: Any) -> None:
    if not values or any(value is None or value == "" for value in values):
        errors.append(f"legacy Serial {label} is missing")
    elif any(value != values[0] for value in values[1:]):
        errors.append(f"legacy Serial {label} differs")


def _commit_events(
    mode: str,
    run_root: Path,
    timing_rows: Sequence[Mapping[str, Any]],
    boundary: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    errors = []
    for row in timing_rows:
        if (
            row.get("schema_version") == PERFORMANCE_COMMIT_SCHEMA
            and row.get("per_token_cuda_synchronize") is not False
        ):
            errors.append("a measured commit performed per-token CUDA synchronization")
    if mode == "target":
        rows.extend(_explicit_commit_events(timing_rows, boundary))
    elif mode == "serial":
        for row in CheckpointJsonl(run_root / "round-events.jsonl").read():
            timeline = row.get("timeline")
            timestamp = (
                timeline.get("state_sync_end_ns")
                if isinstance(timeline, Mapping)
                else None
            )
            rows.append(
                _commit_row(
                    row.get("request_id"),
                    row.get("committed_token_ids"),
                    timestamp,
                    "serial-round-commit",
                )
            )
        rows.extend(_explicit_commit_events(timing_rows, boundary))
    else:
        for row in CheckpointJsonl(run_root / "proposal-events.jsonl").read():
            rows.append(
                _commit_row(
                    row.get("request_id"),
                    row.get("committed_token_ids"),
                    row.get("commit_end_ns"),
                    "dual-proposal-commit",
                )
            )
        rows.extend(_explicit_commit_events(timing_rows, boundary))
    for row in rows:
        timestamp = row.get("timestamp_ns")
        tokens = row.get("token_ids")
        if not isinstance(timestamp, int) or timestamp < boundary:
            errors.append("a measured commit predates the performance boundary")
        if not isinstance(tokens, list) or not tokens:
            errors.append("a measured commit has no token IDs")
    return rows, errors


def _explicit_commit_events(
    timing_rows: Sequence[Mapping[str, Any]], boundary: int
) -> list[dict[str, Any]]:
    rows = []
    for row in timing_rows:
        if row.get("schema_version") != PERFORMANCE_COMMIT_SCHEMA:
            continue
        if int(row.get("timestamp_ns", -1)) < boundary:
            continue
        rows.append(
            _commit_row(
                row.get("request_id"),
                row.get("token_ids"),
                row.get("timestamp_ns"),
                str(row.get("source", "explicit-resident-commit")),
            )
        )
    return rows


def _commit_row(
    request_id: Any, tokens: Any, timestamp: Any, source: str
) -> dict[str, Any]:
    return {
        "request_id": str(request_id or ""),
        "token_ids": list(tokens) if isinstance(tokens, (list, tuple)) else tokens,
        "timestamp_ns": timestamp,
        "source": source,
    }


def _request_metrics(
    outputs: Sequence[Mapping[str, Any]],
    manifest_requests: Sequence[Any],
    event_rows: Sequence[Mapping[str, Any]],
    boundary: int,
    workload_path: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    errors = []
    output_by_id = _unique_by_id(outputs, "output", errors)
    manifest_by_id = {row.request_id: row for row in manifest_requests}
    workload_by_id = _workload_rows(workload_path, errors)
    events: dict[str, list[Mapping[str, Any]]] = {}
    for row in event_rows:
        events.setdefault(str(row.get("request_id", "")), []).append(row)
    if set(output_by_id) != set(manifest_by_id):
        errors.append("output and decode-ready request sets differ")
    if set(output_by_id) != set(workload_by_id):
        errors.append("output and workload request sets differ")
    if set(events) - set(output_by_id):
        errors.append("measured commit events contain an unknown request ID")
    result = []
    for request_id in sorted(set(output_by_id) & set(manifest_by_id)):
        output = output_by_id[request_id]
        ready = manifest_by_id[request_id]
        generated = _integer_tokens(output.get("generated_token_ids"), errors, request_id)
        expected = generated[1:] if generated else []
        request_events = sorted(
            events.get(request_id, ()), key=lambda row: int(row.get("timestamp_ns", -1))
        )
        observed = [token for row in request_events for token in row.get("token_ids", ())]
        timestamps = [
            int(row["timestamp_ns"])
            for row in request_events
            for _ in row.get("token_ids", ())
            if isinstance(row.get("timestamp_ns"), int)
        ]
        if not generated or generated[0] != ready.bootstrap_token_id:
            errors.append(f"{request_id}: final output does not begin with bootstrap")
        if observed != expected:
            errors.append(f"{request_id}: measured commit tokens differ from final output")
        if len(timestamps) != len(observed) or not timestamps:
            errors.append(f"{request_id}: measured commit timestamps are incomplete")
            continue
        if timestamps != sorted(timestamps) or timestamps[0] < boundary:
            errors.append(f"{request_id}: measured commit timestamps are invalid")
        final = timestamps[-1]
        tpot_ms = (
            (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1) / 1_000_000
            if len(timestamps) >= 2
            else None
        )
        workload = workload_by_id.get(request_id, {})
        result.append(
            {
                "request_id": request_id,
                "prompt_token_count": ready.prompt_token_count,
                "prompt_token_ids_sha256": ready.prompt_token_ids_sha256,
                "bootstrap_token_id": ready.bootstrap_token_id,
                "setup_committed_output_tokens": 1,
                "measured_committed_output_token_ids": observed,
                "measured_committed_output_token_count": len(observed),
                "measured_commit_timestamps_ns": timestamps,
                "first_measured_commit_ns": timestamps[0],
                "final_measured_commit_ns": final,
                "decode_latency_ms": (final - boundary) / 1_000_000,
                "tpot_ms": tpot_ms,
                "total_generated_token_ids": generated,
                "maximum_new_tokens": workload.get("output_tokens"),
                "finish_reason": output.get("finish_reason"),
                "termination_reason": output.get("stop_reason"),
                "token_accounting_valid": (
                    [ready.bootstrap_token_id] + observed == generated
                ),
                "commit_events": [dict(row) for row in request_events],
            }
        )
    return result, errors


def _validate_mode_boundary(
    mode: str,
    run_root: Path,
    raw: Mapping[str, Any],
    plugin: Mapping[str, Any],
    boundary: int,
) -> list[str]:
    errors = []
    if mode == "target":
        if plugin.get("proposal_generation") is not False:
            errors.append("Target mode performed measured Draft proposal generation")
        for name in (
            "round-events.jsonl",
            "proposal-events.jsonl",
            "proposal-lifecycle-events.jsonl",
        ):
            path = run_root / name
            if path.is_file() and CheckpointJsonl(path).read():
                errors.append(f"Target mode contains measured proposal evidence in {name}")
    elif mode == "serial":
        deferred = run_root / "initial-proposals-ready.json"
        evidence = raw.get("phase4b2_initial_proposals_ready")
        if not deferred.is_file():
            errors.append("Serial deferred initial-proposal artifact is missing")
        elif not isinstance(evidence, Mapping) or evidence.get("sha256") != sha256_file(
            deferred
        ):
            errors.append("Serial deferred initial-proposal provenance differs")
        else:
            try:
                load_deferred_initial_proposals_ready(
                    deferred,
                    manifest_path=run_root / "decode-ready-manifest.json",
                    expected_request_ids=[
                        str(row.get("request_id", ""))
                        for row in raw.get("outputs", ())
                        if isinstance(row, Mapping)
                    ],
                )
            except (RuntimeError, TypeError, ValueError) as error:
                errors.append(str(error))
        rounds = CheckpointJsonl(run_root / "round-events.jsonl").read()
        first = [
            row.get("timeline", {}).get("draft_start_ns")
            for row in rounds
            if row.get("round_id") == 0 and isinstance(row.get("timeline"), Mapping)
        ]
        if not first or any(not isinstance(value, int) or value < boundary for value in first):
            errors.append("Serial initial proposal did not start after the boundary")
    else:
        lifecycle = CheckpointJsonl(run_root / "proposal-lifecycle-events.jsonl").read()
        starts = [
            row.get("draft_start_ns")
            for row in lifecycle
            if row.get("lifecycle_state") == "CREATED"
        ]
        if not starts or any(not isinstance(value, int) or value < boundary for value in starts):
            errors.append("Dual initial proposal did not start after the boundary")
        overlaps = CheckpointJsonl(run_root / "overlap-events.jsonl").read()
        if not any(int(row.get("overlap_duration_ns", 0)) > 0 for row in overlaps):
            errors.append("Dual run has no physical Draft/Target overlap witness")
    return errors


def _validate_final_sync(
    value: Any,
    tp_size: int,
    physical_gpu_ids: Sequence[int],
    last_commit_ns: int,
) -> tuple[list[Mapping[str, Any]], list[str]]:
    rows = value if isinstance(value, list) else []
    errors = []
    if len(rows) != tp_size:
        errors.append("final Target synchronization does not cover every TP rank")
    if {row.get("local_rank") for row in rows if isinstance(row, Mapping)} != set(
        range(tp_size)
    ):
        errors.append("final Target synchronization TP ranks are incomplete")
    if {row.get("physical_gpu_id") for row in rows if isinstance(row, Mapping)} != set(
        physical_gpu_ids
    ):
        errors.append("final Target synchronization GPU placement differs")
    for row in rows:
        timestamp = row.get("final_cuda_synchronize_complete_ns")
        if not isinstance(timestamp, int) or timestamp < last_commit_ns:
            errors.append("final Target synchronization predates a measured commit")
    return rows, errors


def _jit_evidence(path: Path, boundary: int) -> dict[str, Any]:
    errors = []
    events = []
    if not path.is_file():
        errors.append("timestamped Target log is missing")
    else:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"timestamped Target log line {line_number} is invalid")
                continue
            if not isinstance(row, Mapping) or not isinstance(row.get("timestamp_ns"), int):
                errors.append(f"timestamped Target log line {line_number} lacks monotonic time")
                continue
            text = str(row.get("line", ""))
            if row["timestamp_ns"] >= boundary and any(
                marker in text.lower() for marker in JIT_MARKERS
            ):
                events.append({"timestamp_ns": row["timestamp_ns"], "line": text})
    return {
        "observer": "timestamped merged Target stdout/stderr; metrics never use log time",
        "timestamp_clock": "time.monotonic_ns",
        "post_measurement_jit_events": events,
        "post_measurement_jit_event_count": len(events),
        "warmup_clean": not events and not errors,
        "errors": errors,
    }


def _cross_mode_equality(values: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    errors = []
    target = values["target"]
    target_rows = _result_requests(target)
    pair_checks = {}
    shared_fields = (
        "git_commit",
        "vllm_commit",
        "patch_hashes",
        "models",
        "workload_sha256",
        "correctness_mode",
        "placement",
    )
    target_checksums = target.get("artifact_sha256")
    for mode, value in values.items():
        if mode == "target":
            continue
        metadata_equal = all(target.get(field) == value.get(field) for field in shared_fields)
        current_checksums = value.get("artifact_sha256")
        provenance_equal = isinstance(target_checksums, Mapping) and isinstance(
            current_checksums, Mapping
        ) and all(
            target_checksums.get(field) == current_checksums.get(field)
            for field in ("config", "topology", "patch_manifest")
        )
        current = _result_requests(value)
        request_ids_equal = set(target_rows) == set(current)
        mismatches = []
        fields = (
            "prompt_token_count",
            "prompt_token_ids_sha256",
            "bootstrap_token_id",
            "setup_committed_output_tokens",
            "measured_committed_output_token_ids",
            "measured_committed_output_token_count",
            "total_generated_token_ids",
            "maximum_new_tokens",
            "finish_reason",
            "termination_reason",
        )
        for request_id in sorted(set(target_rows) | set(current)):
            if request_id not in target_rows or request_id not in current:
                mismatches.append({"request_id": request_id, "field": "request_presence"})
                continue
            for field in fields:
                if target_rows[request_id].get(field) != current[request_id].get(field):
                    mismatches.append({"request_id": request_id, "field": field})
                    break
        equal = metadata_equal and provenance_equal and request_ids_equal and not mismatches
        if not equal:
            errors.append(f"Target and {mode} resident semantics differ exactly")
        pair_checks[f"target_equals_{mode}"] = {
            "equal": equal,
            "metadata_equal": metadata_equal,
            "provenance_equal": provenance_equal,
            "request_ids_equal": request_ids_equal,
            "first_mismatches": mismatches[:10],
        }
    boundary_equivalent = all(
        value.get("measurement", {}).get("setup_excluded") is True
        and value.get("measurement", {}).get(
            "bootstrap_excluded_from_measured_token_count"
        )
        is True
        for value in values.values()
    )
    workload_equivalent = len(
        {value.get("workload_sha256") for value in values.values()}
    ) == 1
    topology_equivalent = len(
        {
            value.get("artifact_sha256", {}).get("topology")
            for value in values.values()
            if isinstance(value.get("artifact_sha256"), Mapping)
        }
    ) == 1
    if not boundary_equivalent:
        errors.append("measurement-boundary contracts differ")
    if not workload_equivalent:
        errors.append("workload provenance differs")
    if not topology_equivalent:
        errors.append("GPU topology provenance differs")
    return {
        "valid": not errors,
        "errors": errors,
        "no_tolerance": True,
        "token_accounting_equality": not errors,
        "measurement_boundary_contract_equivalent": boundary_equivalent,
        "workload_equivalent": workload_equivalent,
        "topology_equivalent": topology_equivalent,
        **pair_checks,
    }


def _speedup(baseline: Mapping[str, Any], mode: Mapping[str, Any]) -> dict[str, float]:
    return {
        "makespan_speedup": float(baseline["decode_makespan_ms"])
        / float(mode["decode_makespan_ms"]),
        "throughput_ratio": float(mode["aggregate_throughput_tokens_per_second"])
        / float(baseline["aggregate_throughput_tokens_per_second"]),
    }


def _mode_semantics(mode: str, plugin: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "target": {
            "execution": "resident target autoregressive decode",
            "draft_measured_work": False,
            "proposals": False,
        },
        "serial": {
            "execution": "existing resident serial state machine",
            "draft_target_overlap": False,
            "initial_proposal_after_measurement_start": True,
        },
        "dual-batch": {
            "execution": "existing Phase-4B.1 resident Dual-Batch state machine",
            "natural_draft_target_overlap": True,
            "per_round_global_cuda_synchronize": False,
            "dual_eager": False,
        },
    }[mode] | {"plugin_schema_version": plugin.get("schema_version")}


def _workload_rows(path: Path, errors: list[str]) -> dict[str, Mapping[str, Any]]:
    rows = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        request_id = str(value.get("request_id", "")) if isinstance(value, Mapping) else ""
        if not request_id or request_id in rows:
            errors.append(f"workload request ID is empty or duplicate at line {line_number}")
            continue
        rows[request_id] = value
    return rows


def _unique_by_id(
    rows: Sequence[Mapping[str, Any]], label: str, errors: list[str]
) -> dict[str, Mapping[str, Any]]:
    result = {}
    for row in rows:
        request_id = str(row.get("request_id", ""))
        if not request_id or request_id in result:
            errors.append(f"{label} request IDs are empty or duplicate")
            continue
        result[request_id] = row
    return result


def _integer_tokens(value: Any, errors: list[str], request_id: str) -> list[int]:
    if not isinstance(value, list) or any(
        not isinstance(token, int) or isinstance(token, bool) or token < 0
        for token in value
    ):
        errors.append(f"{request_id}: generated token IDs are invalid")
        return []
    return list(value)


def _result_requests(value: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = value.get("requests")
    rows = rows if isinstance(rows, list) else []
    return {
        str(row.get("request_id", "")): row
        for row in rows
        if isinstance(row, Mapping) and row.get("request_id")
    }


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _comparison_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 4B.2 decode-only performance comparison",
        "",
        f"- Matched-work comparable: `{report['matched_work_comparability']['valid']}`",
        "- Exact sequence equal (diagnostic): "
        f"`{report['exact_sequence_diagnostic']['all_equal']}`",
        f"- Divergent requests: {report['exact_sequence_diagnostic']['divergent_request_count']}",
        f"- Comparison complete: `{report['comparison_complete']}`",
        f"- Performance valid: `{report['performance_valid']}`",
        f"- Pair performance valid: `{report['performance_valid_for_pair']}`",
        f"- Claim boundary: {report['claim_boundary']}",
    ]
    lines.extend(
        [
            "",
            "## Per-mode metrics",
            "",
            "| Mode | Requests | Measured tokens | Makespan ms | Throughput tok/s | "
            "Latency p50/p90/p99 ms | TPOT mean/p50/p90/p99 ms | Warmup clean | JIT count |",
            "|---|---:|---:|---:|---:|---|---|---|---:|",
        ]
    )
    for mode, metrics in report["metrics"].items():
        if not isinstance(metrics, Mapping):
            continue
        latency = metrics.get("decode_latency_ms", {})
        tpot = metrics.get("tpot_ms", {})
        warmup = report["warmup"][mode]
        latency_text = "/".join(str(latency.get(key)) for key in ("p50", "p90", "p99"))
        tpot_text = "/".join(str(tpot.get(key)) for key in ("mean", "p50", "p90", "p99"))
        lines.append(
            f"| {mode} | {metrics.get('completed_requests')} | "
            f"{metrics.get('total_measured_committed_output_tokens')} | "
            f"{metrics.get('decode_makespan_ms')} | "
            f"{metrics.get('aggregate_throughput_tokens_per_second')} | "
            f"{latency_text} | {tpot_text} | {warmup['warmup_clean']} | "
            f"{warmup['post_measurement_jit_event_count']} |"
        )
    speedups = report.get("speedups")
    if isinstance(speedups, Mapping):
        lines.extend(["", "## Speedups", ""])
        for name in ("target_vs_serial", "target_vs_dual_batch", "serial_vs_dual_batch"):
            if name not in speedups:
                continue
            row = speedups[name]
            lines.append(
                f"- {name}: makespan speedup (baseline/mode) `{row['makespan_speedup']:.6f}x`, "
                f"throughput `{row['throughput_ratio']:.6f}x`"
            )
    else:
        lines.extend(["", "No speedup is reported because matched-work comparability failed."])
    lines.extend(
        [
            "",
            "## Diagnostics and gate errors",
            "",
            "```json",
            json.dumps(
                {
                    "errors": report["errors"],
                    "exact_sequence_diagnostic": report["exact_sequence_diagnostic"],
                    "termination_differences": report["matched_work_comparability"][
                        "termination_differences"
                    ],
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"
