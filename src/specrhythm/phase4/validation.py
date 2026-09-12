"""Strict validation for Phase-4A stock-engine bring-up artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from specrhythm.phase4.config import Phase4Config
from specrhythm.phase4.manifest import (
    validate_environment,
    validate_runtime_manifest,
    validate_topology,
)
from specrhythm.phase4.stock_vllm import validate_worker_ranks


def _read(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _validate_smoke(
    value: Mapping[str, Any], config: Phase4Config, expected_role: str
) -> tuple[list[str], list[str]]:
    errors = []
    warnings = []
    prefix = f"{expected_role} smoke"
    if value.get("schema_version") != "specrhythm.phase4-stock-smoke.v1":
        errors.append(f"{prefix}: unsupported schema")
    if value.get("role") != expected_role:
        errors.append(f"{prefix}: wrong engine role")
    for key in (
        "fake_data",
        "serving_performance_result",
        "built_in_speculative_decoding",
        "vllm_dbo_enabled",
        "specrhythm_dual_batch_implemented",
    ):
        if value.get(key) is not False:
            errors.append(f"{prefix}: {key} must be false")
    if value.get("gpu_result") is not True:
        errors.append(f"{prefix}: real GPU result flag is missing")
    if value.get("request_count") != config.smoke_request_count:
        errors.append(f"{prefix}: request count is not {config.smoke_request_count}")
    if value.get("prompt_token_ids_revalidated") is not True:
        errors.append(f"{prefix}: vLLM tokenizer did not revalidate frozen prompt token IDs")
    if value.get("repeated_run_deterministic") is not True:
        errors.append(f"{prefix}: repeated greedy run is not deterministic")
    engine = config.draft if expected_role == "draft" else config.target
    rows = value.get("worker_ranks")
    if not isinstance(rows, list):
        errors.append(f"{prefix}: worker rank records are missing")
    else:
        errors.extend(f"{prefix}: {error}" for error in validate_worker_ranks(rows, engine))
    runs = value.get("runs")
    if not isinstance(runs, list) or len(runs) != 2:
        errors.append(f"{prefix}: exactly two repeated runs are required")
        runs = []
    for run_index, run in enumerate(runs):
        if not isinstance(run, list) or len(run) != config.smoke_request_count:
            errors.append(f"{prefix}: run {run_index} has missing request outputs")
            continue
        request_ids = set()
        for output in run:
            if not isinstance(output, Mapping) or not output.get("request_id"):
                errors.append(f"{prefix}: run {run_index} has an invalid output")
                continue
            request_ids.add(output["request_id"])
            token_ids = output.get("generated_token_ids")
            accounting = output.get("token_accounting")
            if not isinstance(token_ids, list) or not token_ids:
                errors.append(f"{prefix}: request output has no generated tokens")
            if not isinstance(accounting, Mapping):
                errors.append(f"{prefix}: request token accounting is missing")
            elif accounting.get("generated_tokens") != len(token_ids or []):
                errors.append(f"{prefix}: generated-token accounting is not conserved")
            elif accounting.get("total_tokens") != accounting.get(
                "prompt_tokens", 0
            ) + accounting.get("generated_tokens", 0):
                errors.append(f"{prefix}: total-token accounting is not conserved")
            timestamps = output.get("timestamps")
            if not isinstance(timestamps, Mapping) or timestamps.get("available") is not True:
                errors.append(f"{prefix}: per-request vLLM timestamps are missing")
            else:
                scheduled = timestamps.get("scheduled_ts")
                first = timestamps.get("first_token_ts")
                last = timestamps.get("last_token_ts")
                if (
                    not all(
                        isinstance(item, (int, float)) and item > 0
                        for item in (scheduled, first, last)
                    )
                    or not scheduled <= first <= last
                ):
                    errors.append(f"{prefix}: per-request timestamps are invalid")
        if len(request_ids) != config.smoke_request_count:
            errors.append(f"{prefix}: run {run_index} request IDs are not unique/complete")
    comparison = value.get("frozen_hf_target_comparison")
    if expected_role == "target":
        if (
            isinstance(comparison, Mapping)
            and comparison.get("performed") is True
            and not comparison.get("all_tokens_equal")
        ):
            warnings.append(
                "legacy HF trajectory differs from stock vLLM; it is advisory provenance "
                "and cannot fail Phase-4 serving correctness"
            )
        elif not isinstance(comparison, Mapping) or comparison.get("performed") is not True:
            warnings.append("legacy HF trajectory comparison was not supplied (advisory only)")
    for name in ("startup_ms", "total_wall_ms"):
        if not isinstance(value.get(name), (int, float)) or value[name] <= 0:
            errors.append(f"{prefix}: {name} must be positive")
    run_wall = value.get("run_wall_ms")
    if (
        not isinstance(run_wall, list)
        or len(run_wall) != 2
        or not all(isinstance(item, (int, float)) and item > 0 for item in run_wall)
    ):
        errors.append(f"{prefix}: two positive run wall-clock samples are required")
    return errors, warnings


def validate_artifacts(
    config: Phase4Config,
    *,
    environment_path: Path,
    topology_path: Path,
    runtime_manifest_path: Path,
    draft_smoke_path: Path,
    target_smoke_path: Path,
) -> dict[str, Any]:
    errors = []
    warnings = []
    environment = _read(environment_path)
    topology = _read(topology_path)
    runtime = _read(runtime_manifest_path)
    draft = _read(draft_smoke_path)
    target = _read(target_smoke_path)
    environment_validation = validate_environment(environment, config)
    topology_validation = validate_topology(topology, config)
    errors.extend(environment_validation["errors"])
    errors.extend(topology_validation["errors"])
    if runtime.get("schema_version") != "specrhythm.phase4-runtime-bundle.v1":
        errors.append("unsupported runtime bundle schema")
    roles = runtime.get("roles")
    roles = roles if isinstance(roles, Mapping) else {}
    if set(roles) != {"draft", "target"}:
        errors.append("runtime bundle must contain exactly draft and target roles")
    for role in ("draft", "target"):
        role_value = roles.get(role)
        if not isinstance(role_value, Mapping):
            continue
        errors.extend(
            f"{role} runtime: {item}" for item in validate_runtime_manifest(role_value, config)
        )
    if set(roles) == {"draft", "target"} and all(
        isinstance(roles[role], Mapping) for role in ("draft", "target")
    ):
        draft_runtime = roles["draft"]
        target_runtime = roles["target"]
        if draft_runtime.get("git_commit") != target_runtime.get("git_commit"):
            errors.append("draft and target runtime manifests come from different commits")
        for key in (
            "config_sha256",
            "workload_sha256",
            "environment_sha256",
            "topology_sha256",
        ):
            draft_inputs = draft_runtime.get("inputs")
            target_inputs = target_runtime.get("inputs")
            draft_inputs = draft_inputs if isinstance(draft_inputs, Mapping) else {}
            target_inputs = target_inputs if isinstance(target_inputs, Mapping) else {}
            if draft_inputs.get(key) != target_inputs.get(key):
                errors.append(f"draft and target runtime manifests differ in {key}")
    for role, smoke in (("draft", draft), ("target", target)):
        runtime_role = roles.get(role, {})
        provenance = smoke.get("provenance", {})
        if isinstance(runtime_role, Mapping) and isinstance(provenance, Mapping):
            if provenance.get("git_commit") != runtime_role.get("git_commit"):
                errors.append(f"{role} smoke/runtime git commits differ")
            runtime_inputs = runtime_role.get("inputs")
            runtime_inputs = runtime_inputs if isinstance(runtime_inputs, Mapping) else {}
            for key in ("config_sha256", "workload_sha256"):
                if provenance.get(key) != runtime_inputs.get(key):
                    errors.append(f"{role} smoke/runtime {key} values differ")
    draft_errors, draft_warnings = _validate_smoke(draft, config, "draft")
    target_errors, target_warnings = _validate_smoke(target, config, "target")
    errors.extend(draft_errors)
    errors.extend(target_errors)
    warnings.extend(draft_warnings)
    warnings.extend(target_warnings)
    return {
        "schema_version": "specrhythm.phase4-validation.v1",
        "stage": "phase4a-stock-vllm-bringup",
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "serving_performance_result": False,
        "checks": {
            "environment": environment_validation,
            "topology": topology_validation,
            "draft_request_count": draft.get("request_count"),
            "target_request_count": target.get("request_count"),
            "draft_deterministic": draft.get("repeated_run_deterministic"),
            "target_deterministic": target.get("repeated_run_deterministic"),
            "hf_target_comparison": target.get("frozen_hf_target_comparison"),
        },
    }


def validation_markdown(report: Mapping[str, Any]) -> str:
    status = "PASS" if report.get("valid") else "FAIL"
    lines = [
        "# Phase 4A.0 stock vLLM bring-up",
        "",
        f"Validation: **{status}**",
        "",
        "This is an engine bring-up and token-correctness artifact, not a serving "
        "performance result.",
        "It does not implement serial-disaggregated verification, SpecRhythm "
        "Dual-Batch, packed-tree verification, or eager execution.",
        "",
        "## Checks",
        "",
    ]
    checks = report.get("checks", {})
    for key in (
        "draft_request_count",
        "target_request_count",
        "draft_deterministic",
        "target_deterministic",
    ):
        lines.append(f"- `{key}`: `{checks.get(key) if isinstance(checks, Mapping) else None}`")
    if report.get("warnings"):
        lines.extend(("", "## Warnings", ""))
        lines.extend(f"- {item}" for item in report["warnings"])
    if report.get("errors"):
        lines.extend(("", "## Errors", ""))
        lines.extend(f"- {item}" for item in report["errors"])
    return "\n".join(lines) + "\n"
