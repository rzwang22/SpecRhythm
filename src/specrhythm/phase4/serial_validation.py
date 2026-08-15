"""Artifact validation and summary for Phase-4A.1 correctness runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from specrhythm.phase4.config import Phase4Config
from specrhythm.phase4.manifest import sha256_file
from specrhythm.phase4.reference import load_reference, validate_stock_reference
from specrhythm.phase4.serial_runner import load_patch_manifest
from specrhythm.phase4.transport import (
    CheckpointJsonl,
    payload_sha256,
    validate_transport_event,
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def validate_serial_artifacts(
    config: Phase4Config,
    *,
    reference_path: Path,
    patch_manifest_path: Path,
    target_regression_path: Path,
    run_paths: Sequence[Path],
    round_event_paths: Sequence[Path],
    transport_event_paths: Sequence[Path],
) -> dict[str, Any]:
    errors = []
    warnings = []
    if len(run_paths) != 2 or len(round_event_paths) != 2 or len(transport_event_paths) != 2:
        raise ValueError("Phase-4A.1 validation requires exactly two Serial runs")
    reference = load_reference(reference_path)
    errors.extend(validate_stock_reference(reference))
    patch_manifest = load_patch_manifest(patch_manifest_path, config)
    regression = _json(target_regression_path)
    if regression.get("schema_version") != "specrhythm.phase4-patched-target-regression.v1":
        errors.append("patched Target-only regression schema is unsupported")
    if regression.get("valid") is not True or regression.get("comparison", {}).get(
        "all_sequences_equal"
    ) is not True:
        errors.append("patched Target-only does not equal frozen stock Target-only")
    if regression.get("repeated_run_deterministic") is not True:
        errors.append("patched Target-only repeated run is not deterministic")
    if regression.get("patch_manifest_sha256") != payload_sha256(patch_manifest):
        errors.append("patched Target-only regression used a different patch manifest")
    runs = [_json(path) for path in run_paths]
    reference_file_sha = sha256_file(reference_path)
    sequences = []
    round_semantics = []
    total_rounds = []
    for index, run in enumerate(runs, 1):
        prefix = f"Serial run {index}"
        if run.get("schema_version") != "specrhythm.phase4-serial-disaggregated-run.v1":
            errors.append(f"{prefix}: unsupported schema")
        if run.get("mode") != "serial-disaggregated":
            errors.append(f"{prefix}: wrong execution mode")
        for flag, expected in (
            ("gpu_correctness_result", True),
            ("gpu_performance_result", False),
            ("reports_goodput", False),
            ("reports_slo_attainment", False),
            ("reports_speedup", False),
            ("packed_tree_verification", False),
            ("dual_batch_overlap", False),
            ("eager", False),
        ):
            if run.get(flag) is not expected:
                errors.append(f"{prefix}: {flag} must be {expected}")
        if run.get("candidate_budget") != config.proposal_budget:
            errors.append(f"{prefix}: wrong linear proposal budget")
        if run.get("request_count") != config.smoke_request_count:
            errors.append(f"{prefix}: wrong request count")
        if run.get("exact_sequence_match") is not True or run.get("valid") is not True:
            errors.append(f"{prefix}: does not exactly match stock vLLM reference")
        stock = run.get("stock_reference")
        if not isinstance(stock, Mapping) or stock.get("file_sha256") != reference_file_sha:
            errors.append(f"{prefix}: stock reference changed or is not pinned")
        run_patch = run.get("patch_manifest")
        if (
            not isinstance(run_patch, Mapping)
            or run_patch.get("patch_sha256") != patch_manifest.get("patch_sha256")
            or run_patch.get("file_sha256") != sha256_file(patch_manifest_path)
        ):
            errors.append(f"{prefix}: patch provenance changed or is not pinned")
        residency = run.get("engine_residency")
        residency = residency if isinstance(residency, Mapping) else {}
        draft = residency.get("draft")
        target = residency.get("target")
        if not isinstance(draft, Mapping) or draft.get("physical_gpu_ids") != [0]:
            errors.append(f"{prefix}: Draft residency is not GPU 0")
        if not isinstance(target, Mapping) or target.get("physical_gpu_ids") != [1, 2]:
            errors.append(f"{prefix}: Target residency is not GPUs 1,2")
        if isinstance(draft, Mapping) and draft.get("target_model_loaded") is not False:
            errors.append(f"{prefix}: Target model residency leaked to Draft GPU")
        if isinstance(target, Mapping) and (
            target.get("draft_model_loaded") is not False
            or target.get("remote_proposer_parameter_count") != 0
        ):
            errors.append(f"{prefix}: Draft model residency leaked to Target GPU")
        plugin = run.get("plugin_report")
        plugin = plugin if isinstance(plugin, Mapping) else {}
        for key in (
            "target_logits_observed",
            "target_future_tokens_observed",
            "oracle_labels_observed",
        ):
            if plugin.get(key) is not False:
                errors.append(f"{prefix}: remote Draft target-information isolation failed")
        hooks = plugin.get("hook_counts")
        hooks = hooks if isinstance(hooks, Mapping) else {}
        if hooks.get("verify_start", 0) <= 0 or hooks.get("verify_start") != hooks.get(
            "verify_end"
        ):
            errors.append(f"{prefix}: Target verification hooks are incomplete")
        accounting = run.get("accounting")
        if not isinstance(accounting, Mapping) or accounting.get("valid") is not True:
            errors.append(f"{prefix}: token accounting is not conserved")
        outputs = run.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != config.smoke_request_count:
            errors.append(f"{prefix}: final outputs are incomplete")
            sequences.append({})
        else:
            sequences.append(
                {
                    row.get("request_id"): (
                        row.get("generated_token_ids"),
                        row.get("finish_reason"),
                        row.get("stop_reason"),
                    )
                    for row in outputs
                    if isinstance(row, Mapping)
                }
            )
        try:
            rounds = CheckpointJsonl(round_event_paths[index - 1]).read()
        except ValueError as error:
            errors.append(f"{prefix}: {error}")
            rounds = []
        total_rounds.append(len(rounds))
        round_semantics.append(
            [
                (
                    row.get("request_id"),
                    row.get("round_id"),
                    row.get("parent_prefix_hash"),
                    row.get("proposal_token_ids"),
                    row.get("accepted_draft_token_ids"),
                    row.get("rejected_draft_token_ids"),
                    row.get("target_correction_token_ids"),
                    row.get("target_bonus_token_ids"),
                    row.get("committed_token_ids"),
                )
                for row in rounds
            ]
        )
        errors.extend(f"{prefix}: {item}" for item in _validate_round_rows(rounds))
        try:
            transport = CheckpointJsonl(transport_event_paths[index - 1]).read()
        except ValueError as error:
            errors.append(f"{prefix}: {error}")
            transport = []
        if not transport:
            errors.append(f"{prefix}: transport events are missing")
        for event in transport:
            errors.extend(
                f"{prefix}: {item}" for item in validate_transport_event(event)
            )
    if len(sequences) == 2 and sequences[0] != sequences[1]:
        errors.append("two identical Serial runs are not deterministic")
    if len(round_semantics) == 2 and round_semantics[0] != round_semantics[1]:
        errors.append("two Serial runs produced different Draft proposals or Target outcomes")
    provenances = [run.get("provenance") for run in runs]
    if len(provenances) == 2 and provenances[0] != provenances[1]:
        errors.append("two Serial runs do not have identical input/runtime provenance")
    legacy = reference.get("legacy_hf_trajectory")
    if isinstance(legacy, Mapping):
        comparison = legacy.get("comparison")
        if (
            isinstance(comparison, Mapping)
            and comparison.get("performed") is True
            and not comparison.get("all_tokens_equal")
        ):
            warnings.append(
                "legacy HF trajectory differs from stock vLLM; this remains advisory only"
            )
    if os_insecure_serialization_enabled(patch_manifest):
        warnings.append(
            "VLLM_ALLOW_INSECURE_SERIALIZATION was enabled; artifacts are valid only for the "
            "local trusted experiment boundary"
        )
    return {
        "schema_version": "specrhythm.phase4a1-validation.v1",
        "stage": "phase4a1-serial-disaggregated-correctness",
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "serving_correctness_reference": "stock-vllm-target-only",
        "legacy_hf_trajectory_role": "provenance-and-divergence-diagnosis-only",
        "gpu_correctness_result": True,
        "gpu_performance_result": False,
        "reports_goodput": False,
        "reports_slo_attainment": False,
        "reports_speedup": False,
        "checks": {
            "patched_target_equals_stock": regression.get("valid"),
            "serial_runs_equal_stock": [run.get("exact_sequence_match") for run in runs],
            "serial_runs_deterministic": len(sequences) == 2 and sequences[0] == sequences[1],
            "draft_proposals_deterministic": len(round_semantics) == 2
            and round_semantics[0] == round_semantics[1],
            "round_counts": total_rounds,
            "reference_file_sha256": reference_file_sha,
            "patch_sha256": patch_manifest.get("patch_sha256"),
        },
    }


def _validate_round_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors = []
    next_round: dict[str, int] = {}
    terminal = set()
    for row in rows:
        request_id = str(row.get("request_id", ""))
        if not request_id:
            errors.append("round event request_id is missing")
            continue
        if request_id in terminal:
            errors.append(f"{request_id}: work exists after request termination")
        expected = next_round.get(request_id, 0)
        if row.get("round_id") != expected:
            errors.append(f"{request_id}: stale, duplicate, or out-of-order round")
        next_round[request_id] = expected + 1
        proposed = row.get("proposed_tokens")
        accepted = row.get("accepted_draft_tokens")
        rejected = row.get("rejected_draft_tokens")
        committed = row.get("committed_tokens")
        correction = row.get("target_correction_tokens")
        bonus = row.get("target_bonus_tokens")
        counts = (proposed, accepted, rejected, committed, correction, bonus)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in counts
        ):
            errors.append(f"{request_id}: token accounting fields are invalid")
            continue
        if proposed != accepted + rejected:
            errors.append(f"{request_id}: proposed token accounting mismatch")
        if committed != accepted + correction + bonus:
            errors.append(f"{request_id}: committed token accounting mismatch")
        if correction + bonus > 1 or correction and bonus:
            errors.append(f"{request_id}: correction/bonus mutual exclusion failed")
        if row.get("verified_candidate_tokens") != proposed:
            errors.append(f"{request_id}: verified/proposed accounting mismatch")
        if row.get("logical_target_kv_length") != row.get("logical_draft_kv_length"):
            errors.append(f"{request_id}: logical Target/Draft KV length mismatch")
        if not row.get("target_microbatch_id") or not isinstance(
            row.get("target_batch_request_ids"), list
        ):
            errors.append(f"{request_id}: Target batch/microbatch metadata is missing")
        timeline = row.get("timeline")
        if not isinstance(timeline, Mapping):
            errors.append(f"{request_id}: serial timeline is missing")
        else:
            values = [
                timeline.get(name)
                for name in (
                    "draft_start_ns",
                    "draft_end_ns",
                    "transfer_start_ns",
                    "transfer_end_ns",
                    "verify_start_ns",
                    "verify_end_ns",
                    "state_sync_start_ns",
                    "state_sync_end_ns",
                    "next_round_draft_start_ns",
                )
            ]
            if any(not isinstance(value, int) for value in values) or values != sorted(
                values
            ):
                errors.append(f"{request_id}: strict serial timeline overlaps")
        if row.get("terminal") is True:
            terminal.add(request_id)
    draft_intervals = []
    verify_intervals = []
    for row in rows:
        timeline = row.get("timeline")
        if not isinstance(timeline, Mapping):
            continue
        draft_intervals.append((timeline.get("draft_start_ns"), timeline.get("draft_end_ns")))
        verify_intervals.append(
            (timeline.get("verify_start_ns"), timeline.get("verify_end_ns"))
        )
    for draft_start, draft_end in draft_intervals:
        for verify_start, verify_end in verify_intervals:
            if not all(
                isinstance(value, int)
                for value in (draft_start, draft_end, verify_start, verify_end)
            ):
                continue
            if max(draft_start, verify_start) < min(draft_end, verify_end):
                errors.append("global Draft and Target verification phases overlap")
                return errors
    return errors


def os_insecure_serialization_enabled(patch_manifest: Mapping[str, Any]) -> bool:
    runtime = patch_manifest.get("runtime_security")
    return isinstance(runtime, Mapping) and runtime.get(
        "VLLM_ALLOW_INSECURE_SERIALIZATION"
    ) is True


def serial_summary_markdown(report: Mapping[str, Any], runs: Sequence[Mapping[str, Any]]) -> str:
    status = "PASS" if report.get("valid") else "FAIL"
    lines = [
        "# Phase 4A.1 Serial Disaggregated correctness",
        "",
        f"Validation: **{status}**",
        "",
        "Serving correctness reference: stock vLLM v0.25.1 Target-only greedy output.",
        "Legacy HF trajectories are advisory provenance only.",
        "This is a GPU correctness result, not a performance, goodput, SLO, or speedup result.",
        "",
        "## Runs",
        "",
        "| Run | Requests | Rounds | Proposed | Accepted | Rejected | "
        "Correction | Bonus | Exact |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for index, run in enumerate(runs, 1):
        accounting = run.get("accounting", {})
        timeline = run.get("strict_serial_timeline", {})
        lines.append(
            "| {index} | {requests} | {rounds} | {proposed} | {accepted} | {rejected} | "
            "{correction} | {bonus} | {exact} |".format(
                index=index,
                requests=run.get("request_count"),
                rounds=timeline.get("round_events"),
                proposed=accounting.get("proposed_tokens"),
                accepted=accounting.get("accepted_draft_tokens"),
                rejected=accounting.get("rejected_draft_tokens"),
                correction=accounting.get("target_correction_tokens"),
                bonus=accounting.get("target_bonus_tokens"),
                exact=run.get("exact_sequence_match"),
            )
        )
    if report.get("warnings"):
        lines.extend(("", "## Warnings", ""))
        lines.extend(f"- {item}" for item in report["warnings"])
    if report.get("errors"):
        lines.extend(("", "## Errors", ""))
        lines.extend(f"- {item}" for item in report["errors"])
    return "\n".join(lines) + "\n"
