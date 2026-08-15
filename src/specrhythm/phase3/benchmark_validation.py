"""Validation and same-implementation comparison for Phase-3B.1 reports."""

from __future__ import annotations

import json
import math
import statistics
from typing import Any, Iterable, Optional

_STAT_KEYS = (
    "mean_ms",
    "std_ms",
    "cv",
    "min_ms",
    "p50_ms",
    "p90_ms",
    "p95_ms",
    "p99_ms",
    "max_ms",
)


def _cell_name(row: dict[str, Any]) -> str:
    dimensions = json.dumps(row.get("dimensions", {}), sort_keys=True, separators=(",", ":"))
    return f"{row.get('operation')}:{dimensions}"


def _close(first: float, second: float) -> bool:
    return math.isclose(first, second, rel_tol=1e-6, abs_tol=1e-6)


def _nearest_rank(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = math.ceil(fraction * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def _validate_statistics(
    value: Any,
    *,
    measured_iterations: int,
    label: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    if not isinstance(value, dict):
        errors.append(f"{label}: timing statistics are missing")
        return
    samples = value.get("raw_samples_ms")
    if not isinstance(samples, (list, tuple)) or len(samples) != measured_iterations:
        errors.append(
            f"{label}: expected {measured_iterations} raw samples, got "
            f"{len(samples) if isinstance(samples, (list, tuple)) else 'missing'}"
        )
        return
    if any(
        not isinstance(sample, (int, float))
        or isinstance(sample, bool)
        or not math.isfinite(float(sample))
        or float(sample) <= 0
        for sample in samples
    ):
        errors.append(f"{label}: all timing samples must be finite and positive")
        return
    numeric = [float(sample) for sample in samples]
    expected = {
        "mean_ms": statistics.mean(numeric),
        "std_ms": statistics.pstdev(numeric),
        "min_ms": min(numeric),
        "p50_ms": _nearest_rank(numeric, 0.50),
        "p90_ms": _nearest_rank(numeric, 0.90),
        "p95_ms": _nearest_rank(numeric, 0.95),
        "p99_ms": _nearest_rank(numeric, 0.99),
        "max_ms": max(numeric),
    }
    expected["cv"] = expected["std_ms"] / expected["mean_ms"]
    for key in _STAT_KEYS:
        actual = value.get(key)
        if not isinstance(actual, (int, float)) or not math.isfinite(float(actual)):
            errors.append(f"{label}: {key} is missing or non-finite")
        elif not _close(float(actual), expected[key]):
            errors.append(f"{label}: {key} does not match retained raw samples")
    quantiles = [
        value.get("min_ms"),
        value.get("p50_ms"),
        value.get("p90_ms"),
        value.get("p95_ms"),
        value.get("p99_ms"),
        value.get("max_ms"),
    ]
    if all(isinstance(item, (int, float)) for item in quantiles) and any(
        float(left) > float(right) for left, right in zip(quantiles, quantiles[1:])
    ):
        errors.append(f"{label}: quantiles are not monotonic")
    outliers = value.get("outlier_indices")
    if not isinstance(outliers, (list, tuple)):
        errors.append(f"{label}: outlier_indices must be retained")
    elif any(
        not isinstance(index, int) or index < 0 or index >= len(samples)
        for index in outliers
    ):
        errors.append(f"{label}: outlier_indices contains an invalid sample index")
    elif outliers:
        warnings.append(f"{label}: retained outlier samples at indices {outliers}")


def _validate_rank_records(
    row: dict[str, Any],
    *,
    expected_ranks: int,
    measured_iterations: int,
    label: str,
    errors: list[str],
) -> None:
    ranks = row.get("rank_measurements")
    if not isinstance(ranks, list):
        errors.append(f"{label}: rank_measurements are missing")
        return
    global_ranks = [rank.get("global_rank") for rank in ranks if isinstance(rank, dict)]
    if len(ranks) != expected_ranks or sorted(global_ranks) != list(range(expected_ranks)):
        errors.append(
            f"{label}: expected complete ranks 0..{expected_ranks - 1}, got {global_ranks}"
        )
    logical_devices = [
        rank.get("logical_cuda_index") for rank in ranks if isinstance(rank, dict)
    ]
    physical_devices = [
        rank.get("physical_gpu_id") for rank in ranks if isinstance(rank, dict)
    ]
    gpu_uuids = [rank.get("gpu_uuid") for rank in ranks if isinstance(rank, dict)]
    if len(set(logical_devices)) != len(logical_devices):
        errors.append(f"{label}: participating ranks do not use distinct logical CUDA devices")
    if len(set(physical_devices)) != len(physical_devices):
        errors.append(f"{label}: participating ranks do not map to distinct physical GPUs")
    if len(set(gpu_uuids)) != len(gpu_uuids):
        errors.append(f"{label}: participating ranks do not report distinct GPU UUIDs")
    for rank in ranks:
        if not isinstance(rank, dict):
            errors.append(f"{label}: rank record is not an object")
            continue
        rank_label = f"{label}/rank{rank.get('global_rank')}"
        if rank.get("world_size") != expected_ranks:
            errors.append(f"{rank_label}: world_size does not match expected TP")
        for key in ("logical_cuda_index", "physical_gpu_id", "gpu_uuid"):
            if rank.get(key) is None:
                errors.append(f"{rank_label}: {key} is missing")
        mapping = rank.get("cuda_visible_devices_mapping")
        if not isinstance(mapping, list) or not mapping:
            errors.append(f"{rank_label}: logical/physical CUDA mapping is missing")
        elif not any(
            row.get("logical_cuda_index") == rank.get("logical_cuda_index")
            and str(row.get("physical_gpu_id")) == str(rank.get("physical_gpu_id"))
            for row in mapping
        ):
            errors.append(f"{rank_label}: logical/physical CUDA mapping is inconsistent")
        for sample_key in ("cuda_samples_ms", "host_samples_ms"):
            samples = rank.get(sample_key)
            if not isinstance(samples, list) or len(samples) != measured_iterations:
                errors.append(
                    f"{rank_label}: {sample_key} must contain {measured_iterations} samples"
                )
            elif any(
                not isinstance(sample, (int, float))
                or isinstance(sample, bool)
                or not math.isfinite(float(sample))
                or float(sample) <= 0
                for sample in samples
            ):
                errors.append(f"{rank_label}: {sample_key} contains an invalid sample")
        if not row.get("requires_model_rank_evidence"):
            continue
        for key in (
            "model_parameter_count",
            "parameter_bytes",
            "allocated_memory_bytes",
            "reserved_memory_bytes",
            "max_allocated_memory_bytes",
            "max_reserved_memory_bytes",
        ):
            value = rank.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                errors.append(f"{rank_label}: {key} must be positive")
        if rank.get("model_parameters_on_expected_device") is not True:
            errors.append(f"{rank_label}: model parameters are not on the expected device")
        if not rank.get("parameter_devices"):
            errors.append(f"{rank_label}: parameter device evidence is missing")
        if not rank.get("forward_input_shape") or not rank.get("forward_output_shape"):
            errors.append(f"{rank_label}: forward input/output shapes are missing")
        if not rank.get("output_checksum"):
            errors.append(f"{rank_label}: output checksum is missing")
        if (
            not isinstance(rank.get("forward_invocations"), int)
            or rank["forward_invocations"] <= 0
        ):
            errors.append(f"{rank_label}: no model forward invocation was recorded")
    if row.get("requires_model_rank_evidence") and ranks:
        checksums = {rank.get("output_checksum") for rank in ranks if isinstance(rank, dict)}
        input_shapes = {
            tuple(rank.get("forward_input_shape", []))
            for rank in ranks
            if isinstance(rank, dict)
        }
        output_shapes = {
            tuple(rank.get("forward_output_shape", []))
            for rank in ranks
            if isinstance(rank, dict)
        }
        if len(checksums) != 1:
            errors.append(f"{label}: rank output checksums disagree")
        if len(input_shapes) != 1 or len(output_shapes) != 1:
            errors.append(f"{label}: rank forward shapes disagree")
    if not ranks or any(
        not isinstance(rank, dict)
        or not isinstance(rank.get("cuda_samples_ms"), list)
        or len(rank["cuda_samples_ms"]) != measured_iterations
        for rank in ranks
    ):
        return
    for sample_key, aggregate_key in (
        ("cuda_samples_ms", "cuda_event"),
        ("host_samples_ms", "host_wall"),
    ):
        expected = [
            max(float(rank[sample_key][index]) for rank in ranks)
            for index in range(measured_iterations)
        ]
        actual = row.get(aggregate_key, {}).get("raw_samples_ms")
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected) or any(
            not _close(float(left), float(right)) for left, right in zip(actual, expected)
        ):
            errors.append(
                f"{label}: {aggregate_key} is not the per-iteration max-rank latency"
            )


def validate_benchmark_report(report: Any) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(report, dict):
        return {
            "schema_version": "specrhythm.gpu-latency-validation.v1",
            "valid": False,
            "errors": ["benchmark report must be an object"],
            "warnings": [],
        }
    if report.get("schema_version") != "specrhythm.gpu-latency.v2":
        errors.append("benchmark schema_version must be specrhythm.gpu-latency.v2")
    semantics = report.get("backend_semantics")
    required_semantics = {
        "backend": "hf_correctness",
        "serving_engine": False,
        "kv_cache_reuse": False,
        "packed_tree_verification": False,
        "simulator_latency_surface_compatible": False,
    }
    if not isinstance(semantics, dict):
        errors.append("backend_semantics are missing")
    else:
        for key, expected in required_semantics.items():
            if semantics.get(key) != expected:
                errors.append(f"backend_semantics.{key} must be {expected!r}")
    if report.get("simulator_latency_surface_compatible") is not False:
        errors.append("correctness-backend latency must be forbidden from simulator surfaces")
    model_identity = report.get("model_identity")
    if not isinstance(model_identity, dict):
        errors.append("model_identity is missing")
    else:
        for role in ("draft", "target"):
            identity = model_identity.get(role)
            if not isinstance(identity, dict):
                errors.append(f"model_identity.{role} is missing")
                continue
            if "configured_revision" not in identity:
                errors.append(f"model_identity.{role}.configured_revision is missing")
            checksum = identity.get("config_sha256")
            if not isinstance(checksum, str) or len(checksum) != 64:
                errors.append(f"model_identity.{role}.config_sha256 is invalid")
    for snapshot_name in ("hardware_state_before", "hardware_state_after"):
        snapshot = report.get(snapshot_name)
        if not isinstance(snapshot, dict):
            errors.append(f"{snapshot_name} is missing")
            continue
        for key in (
            "clock_locked",
            "gpus",
            "peer_access",
            "nvlink_pcie_topology",
            "errors",
        ):
            if key not in snapshot:
                errors.append(f"{snapshot_name}.{key} is missing")
        if snapshot.get("clock_locked") is not False:
            errors.append(f"{snapshot_name}.clock_locked must be false")
        gpus = snapshot.get("gpus")
        if not isinstance(gpus, list) or not gpus:
            errors.append(f"{snapshot_name}.gpus must contain observed GPU identities")
        else:
            hardware_fields = (
                "physical_gpu_id",
                "gpu_name",
                "gpu_uuid",
                "temperature_c",
                "power_draw_w",
                "power_limit_w",
                "sm_clock_mhz",
                "memory_clock_mhz",
                "p_state",
                "memory_used_mib",
                "ecc_status",
                "pcie_generation_current",
                "pcie_generation_max",
                "pcie_width_current",
                "pcie_width_max",
            )
            for gpu in gpus:
                for key in hardware_fields:
                    if key not in gpu:
                        errors.append(f"{snapshot_name}: GPU field {key} is missing")
                if not gpu.get("gpu_name") or not gpu.get("gpu_uuid"):
                    errors.append(
                        f"{snapshot_name}: GPU name and UUID are required identities"
                    )
        for error in snapshot.get("errors", []):
            warnings.append(f"{snapshot_name}: {error}")
    measurements = report.get("measurements")
    if not isinstance(measurements, list) or not measurements:
        errors.append("measurements must be a non-empty list")
        measurements = []
    for row in measurements:
        if not isinstance(row, dict):
            errors.append("measurement row is not an object")
            continue
        label = _cell_name(row)
        measured = row.get("measured_iterations")
        warmup = row.get("warmup_iterations")
        if not isinstance(warmup, int) or warmup < 5:
            errors.append(f"{label}: warmup_iterations must be at least 5")
        if not isinstance(measured, int) or measured < 30:
            errors.append(f"{label}: measured_iterations must be at least 30")
            continue
        _validate_statistics(
            row.get("cuda_event"),
            measured_iterations=measured,
            label=f"{label}/cuda",
            errors=errors,
            warnings=warnings,
        )
        _validate_statistics(
            row.get("host_wall"),
            measured_iterations=measured,
            label=f"{label}/host",
            errors=errors,
            warnings=warnings,
        )
        cuda_mean = row.get("cuda_event", {}).get("mean_ms")
        host_mean = row.get("host_wall", {}).get("mean_ms")
        if isinstance(cuda_mean, (int, float)) and isinstance(host_mean, (int, float)):
            tolerance = max(0.1, 0.01 * float(cuda_mean))
            if float(host_mean) + tolerance < float(cuda_mean):
                errors.append(f"{label}: host latency is clearly below CUDA critical path")
        protocol = row.get("timing_protocol")
        required_protocol = (
            "distributed_barrier_before_iteration",
            "cuda_synchronize_before_iteration",
            "cuda_event_per_rank",
            "host_clock_per_rank",
            "cuda_event_synchronize_before_host_stop",
            "distributed_barrier_after_iteration",
            "raw_samples_retained",
        )
        if not isinstance(protocol, dict) or any(
            protocol.get(key) is not True for key in required_protocol
        ):
            errors.append(f"{label}: barrier/synchronization timing protocol is incomplete")
        if isinstance(protocol, dict) and protocol.get("outliers_removed") is not False:
            errors.append(f"{label}: outlier samples must not be removed")
        operation = row.get("operation")
        if operation == "T_verify":
            expected_ranks = int(row.get("dimensions", {}).get("TP", 0))
            semantics_value = row.get("operation_semantics", {})
            if semantics_value.get("verify_implementation") != "serial_full_context_replay":
                errors.append(f"{label}: verify implementation semantics are incomplete")
        elif operation == "T_draft":
            expected_ranks = int(report.get("draft_model", {}).get("tp_size", 1))
            semantics_value = row.get("operation_semantics", {})
            for key in (
                "search_generation_semantics",
                "number_of_model_forwards",
                "tokens_or_nodes_per_forward",
                "N_search_definition",
            ):
                if key not in semantics_value:
                    errors.append(f"{label}: draft semantic {key} is missing")
        else:
            expected_ranks = 1
        _validate_rank_records(
            row,
            expected_ranks=expected_ranks,
            measured_iterations=measured,
            label=label,
            errors=errors,
        )
        if operation == "T_select":
            selection = row.get("operation_semantics", {})
            expected_selection = {
                "selector_backend": "synthetic_topk",
                "prefix_closure": False,
                "tree_materialization": False,
            }
            for key, expected in expected_selection.items():
                if selection.get(key) != expected:
                    errors.append(f"{label}: selector semantic {key} must be {expected!r}")
        if operation == "T_transfer":
            transfer = row.get("transfer_metadata")
            if not isinstance(transfer, dict):
                errors.append(f"{label}: transfer P2P metadata is missing")
            else:
                for key in (
                    "source_logical_cuda_index",
                    "source_physical_gpu_id",
                    "destination_logical_cuda_index",
                    "destination_physical_gpu_id",
                    "cuda_device_can_access_peer",
                    "p2p_enabled",
                    "copy_direction",
                    "host_staging",
                    "effective_bandwidth_gbps",
                    "topology_source",
                ):
                    if key not in transfer:
                        errors.append(f"{label}: transfer metadata {key} is missing")
    return {
        "schema_version": "specrhythm.gpu-latency-validation.v1",
        "valid": not errors,
        "measurement_count": len(measurements),
        "errors": errors,
        "warnings": warnings,
    }


def _gpu_models(report: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(row.get("gpu_name"))
        for row in report.get("hardware_state_before", {}).get("gpus", [])
    )


def _compatibility_value(report: dict[str, Any], key: str) -> Any:
    if key == "gpu_models":
        return _gpu_models(report)
    return report.get(key)


def compare_benchmark_reports(
    reports: Iterable[dict[str, Any]], source_names: Optional[Iterable[str]] = None
) -> dict[str, Any]:
    values = list(reports)
    names = list(source_names or [f"run-{index + 1}" for index in range(len(values))])
    errors: list[str] = []
    warnings: list[str] = []
    if len(values) < 2:
        errors.append("comparison requires at least two independent benchmark runs")
    if len(names) != len(values):
        errors.append("source name count does not match report count")
    for index, report in enumerate(values):
        validation = validate_benchmark_report(report)
        if not validation["valid"]:
            errors.append(f"{names[index]} failed validation: {validation['errors']}")
    compatibility_keys = (
        "git_commit",
        "config_sha256",
        "backend_semantics",
        "draft_model",
        "target_model",
        "model_identity",
        "runtime_versions",
        "benchmark_config",
        "gpu_models",
    )
    compatibility: dict[str, Any] = {}
    if values:
        for key in compatibility_keys:
            observed = [_compatibility_value(report, key) for report in values]
            compatibility[key] = observed[0]
            if key in {"git_commit", "config_sha256"} and not observed[0]:
                errors.append(f"comparison requires non-empty {key}")
            if any(value != observed[0] for value in observed[1:]):
                errors.append(f"cannot compare runs with different {key}")
    keyed_runs = []
    for report in values:
        keyed_runs.append({_cell_name(row): row for row in report.get("measurements", [])})
    if keyed_runs:
        expected_cells = set(keyed_runs[0])
        for index, keyed in enumerate(keyed_runs[1:], start=1):
            if set(keyed) != expected_cells:
                errors.append(f"{names[index]} has a different measurement cell set")
        for cell in sorted(expected_cells):
            semantics = [keyed[cell].get("operation_semantics") for keyed in keyed_runs]
            implementations = [keyed[cell].get("implementation") for keyed in keyed_runs]
            if any(value != semantics[0] for value in semantics[1:]):
                errors.append(f"{cell}: operation semantics differ across runs")
            if any(value != implementations[0] for value in implementations[1:]):
                errors.append(f"{cell}: implementation differs across runs")
    cells = []
    if not errors and keyed_runs:
        for cell in sorted(keyed_runs[0]):
            rows = [keyed[cell] for keyed in keyed_runs]
            means = [float(row["cuda_event"]["mean_ms"]) for row in rows]
            run_mean = statistics.mean(means)
            run_std = statistics.pstdev(means)
            variation = (max(means) - min(means)) / run_mean if run_mean else 0.0
            cells.append(
                {
                    "cell": cell,
                    "operation": rows[0]["operation"],
                    "dimensions": rows[0]["dimensions"],
                    "per_run": [
                        {
                            "source": name,
                            "mean_ms": row["cuda_event"]["mean_ms"],
                            "std_ms": row["cuda_event"]["std_ms"],
                            "cv": row["cuda_event"]["cv"],
                        }
                        for name, row in zip(names, rows)
                    ],
                    "run_mean_ms": run_mean,
                    "run_std_ms": run_std,
                    "run_cv": run_std / run_mean if run_mean else 0.0,
                    "run_to_run_variation": variation,
                    "run_to_run_variation_percent": 100.0 * variation,
                    "raw_samples_combined": False,
                }
            )
            if variation > 0.10:
                warnings.append(f"{cell}: run-to-run variation exceeds 10%")
    return {
        "schema_version": "specrhythm.gpu-latency-comparison.v1",
        "valid": not errors,
        "run_count": len(values),
        "sources": names,
        "compatibility": compatibility,
        "comparison_semantics": (
            "compares per-run statistics without pooling or deleting raw samples"
        ),
        "cells": cells,
        "errors": errors,
        "warnings": warnings,
    }


def comparison_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 3B.1 repeated-run comparison",
        "",
        f"Validation: **{'PASS' if report.get('valid') else 'FAIL'}**",
        "",
    ]
    if report.get("errors"):
        lines.extend(["## Errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "| Operation | Dimensions | Run means ms | Across-run mean/CV | Variation |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for cell in report.get("cells", []):
        dimensions = ", ".join(
            f"{key}={value}" for key, value in cell["dimensions"].items()
        )
        run_means = ", ".join(
            f"{row['mean_ms']:.3f}" for row in cell["per_run"]
        )
        lines.append(
            f"| {cell['operation']} | {dimensions} | {run_means} | "
            f"{cell['run_mean_ms']:.3f}/{cell['run_cv']:.4f} | "
            f"{cell['run_to_run_variation_percent']:.2f}% |"
        )
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"
