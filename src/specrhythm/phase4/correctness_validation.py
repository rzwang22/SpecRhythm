"""A/B/C/D artifact validation for Phase 4A.1.1 correctness hardening."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from specrhythm.phase4.batch_invariant import reference_correctness_mode
from specrhythm.phase4.reference import load_reference
from specrhythm.phase4.transport import CheckpointJsonl
from specrhythm.phase4.vllm_diagnostics import (
    compare_divergence_diagnostics,
    compare_fixed_proposal_controls,
    validate_kv_monotonicity,
    validate_target_diagnostic,
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _trajectories(outputs: Any) -> dict[str, tuple[Any, Any, Any]]:
    if not isinstance(outputs, list):
        return {}
    return {
        str(row.get("request_id")): (
            row.get("generated_token_ids"),
            row.get("finish_reason"),
            row.get("stop_reason"),
        )
        for row in outputs
        if isinstance(row, Mapping) and row.get("request_id")
    }


def validate_batch_invariant_experiment(
    *,
    stock_reference_paths: Sequence[Path],
    target_regression_paths: Sequence[Path],
    serial_run_paths: Sequence[Path],
    round_event_paths: Sequence[Path],
    target_diagnostic_paths: Sequence[Path],
    serial_diagnostic_paths: Sequence[Path],
) -> dict[str, Any]:
    if any(
        len(paths) != 2
        for paths in (
            stock_reference_paths,
            target_regression_paths,
            serial_run_paths,
            round_event_paths,
            target_diagnostic_paths,
            serial_diagnostic_paths,
        )
    ):
        raise ValueError("Phase-4A.1.1 C/D validation requires exactly two artifacts per input")
    errors = []
    references = [load_reference(path) for path in stock_reference_paths]
    for index, reference in enumerate(references, 1):
        if reference_correctness_mode(reference) != "batch-invariant":
            errors.append(f"C{index} is not a batch-invariant stock reference")
        runtime = reference.get("target_runtime_configuration", {})
        if runtime.get("batch_invariant_effective") is not True:
            errors.append(f"C{index} does not prove effective batch invariance")
    stock_trajectories = [_trajectories(reference.get("outputs")) for reference in references]
    if stock_trajectories[0] != stock_trajectories[1]:
        errors.append("C stock Target-only repeats are not exactly deterministic")
    regressions = [_json(path) for path in target_regression_paths]
    for index, regression in enumerate(regressions, 1):
        smoke = regression.get("smoke")
        smoke = smoke if isinstance(smoke, Mapping) else {}
        if regression.get("valid") is not True:
            errors.append(f"patched C{index} Target-only does not equal its stock reference")
        if smoke.get("correctness_mode") != "batch-invariant":
            errors.append(f"patched C{index} Target-only has the wrong correctness mode")
        if smoke.get("batch_invariant_effective") is not True:
            errors.append(f"patched C{index} Target-only lacks effective-mode proof")
    runs = [_json(path) for path in serial_run_paths]
    serial_trajectories = []
    stock_runtime = references[0].get("target_runtime_configuration", {})
    stock_attention = stock_runtime.get("attention_backends")
    stock_all_reduce = stock_runtime.get("all_reduce_backends")
    stock_dtype = stock_runtime.get("dtype")
    stock_sampling = references[0].get("sampling_configuration")
    stock_workload = references[0].get("workload", {})
    stock_vllm = references[0].get("vllm", {})
    if not isinstance(stock_attention, list) or not any(
        "FLASH" in str(backend).upper() for backend in stock_attention
    ):
        errors.append("C did not retain a FlashAttention backend")
    for index, run in enumerate(runs, 1):
        if run.get("correctness_mode") != "batch-invariant":
            errors.append(f"D{index} is not labeled batch-invariant")
        if run.get("batch_invariant_effective") is not True:
            errors.append(f"D{index} does not prove effective batch invariance")
        if run.get("accounting", {}).get("valid") is not True:
            errors.append(f"D{index} token accounting failed")
        if run.get("strict_serial_timeline", {}).get("validated_in_runner") is not True:
            errors.append(f"D{index} strict serial timeline failed")
        if run.get("kv_monotonicity", {}).get("valid") is not True:
            errors.append(f"D{index} Target/Draft KV monotonicity failed")
        if run.get("sampling_configuration") != stock_sampling:
            errors.append(f"D{index} sampling configuration differs from C")
        provenance = run.get("provenance")
        provenance = provenance if isinstance(provenance, Mapping) else {}
        if provenance.get("workload_sha256") != stock_workload.get("sha256"):
            errors.append(f"D{index} workload differs from C")
        if provenance.get("vllm_source_commit") != stock_vllm.get("source_commit"):
            errors.append(f"D{index} vLLM source commit differs from C")
        runtime = run.get("target_runtime_configuration")
        runtime = runtime if isinstance(runtime, Mapping) else {}
        for key in ("physical_gpu_ids", "tensor_parallel_size", "max_model_len"):
            if runtime.get(key) != stock_runtime.get(key):
                errors.append(f"D{index} target runtime {key} differs from C")
        workers = run.get("worker_ranks")
        workers = workers if isinstance(workers, list) else []
        serial_attention = sorted(
            {
                str(backend)
                for row in workers
                if isinstance(row, Mapping)
                for backend in row.get("attention_backends", ())
            }
        )
        serial_all_reduce = sorted(
            {
                str(backend)
                for row in workers
                if isinstance(row, Mapping)
                for backend in row.get("all_reduce_backends", ())
            }
        )
        serial_dtypes = {
            row.get("dtype") for row in workers if isinstance(row, Mapping)
        }
        if serial_attention != stock_attention:
            errors.append(f"D{index} attention backend differs from C")
        if serial_all_reduce != stock_all_reduce:
            errors.append(f"D{index} all-reduce backend differs from C")
        if len(serial_dtypes) != 1 or next(iter(serial_dtypes), None) not in {
            stock_dtype,
            f"torch.{stock_dtype}",
        }:
            errors.append(f"D{index} dtype differs from C")
        serial_trajectories.append(_trajectories(run.get("outputs")))
    serial_repeated = serial_trajectories[0] == serial_trajectories[1]
    if not serial_repeated:
        errors.append("D Serial repeats are not exactly deterministic")
    serial_equals_stock = [row == stock_trajectories[0] for row in serial_trajectories]

    round_rows = []
    for index, path in enumerate(round_event_paths, 1):
        rows = CheckpointJsonl(path).read()
        round_rows.append(rows)
        errors.extend(f"D{index}: {item}" for item in validate_kv_monotonicity(rows))
    round_repeated = [
        [
            (
                row.get("request_id"),
                row.get("round_id"),
                row.get("proposal_token_ids"),
                row.get("committed_token_ids"),
                row.get("terminal"),
            )
            for row in rows
        ]
        for rows in round_rows
    ]
    if round_repeated[0] != round_repeated[1]:
        errors.append("D repeats have different proposal/acceptance/termination semantics")

    diagnostics_valid = []
    for label, paths in (
        ("C", target_diagnostic_paths),
        ("D", serial_diagnostic_paths),
    ):
        for index, path in enumerate(paths, 1):
            rows = CheckpointJsonl(path).read()
            row_errors = [
                error for row in rows for error in validate_target_diagnostic(row)
            ]
            if not rows:
                row_errors.append("diagnostic log is empty")
            diagnostics_valid.append(not row_errors)
            errors.extend(f"{label}{index} diagnostic: {item}" for item in row_errors)

    exact_pass = all(serial_equals_stock) and serial_repeated
    outcome = (
        "invalid-artifacts"
        if errors
        else "A"
        if exact_pass
        else "controls-required"
    )
    return {
        "schema_version": "specrhythm.phase4a1.1-batch-invariant-validation.v1",
        "stage": "phase4a1.1-batch-invariant-correctness-hardening",
        "valid": outcome == "A",
        "outcome": outcome,
        "errors": errors,
        "checks": {
            "target_only_repeated_exact_equality": stock_trajectories[0]
            == stock_trajectories[1],
            "serial_repeated_exact_equality": serial_repeated,
            "serial_equals_stock": serial_equals_stock,
            "termination_included_in_trajectory_comparison": True,
            "round_semantics_repeated": round_repeated[0] == round_repeated[1],
            "target_diagnostics_valid": all(diagnostics_valid),
        },
        "next_action": (
            "Phase 4A.1 exact correctness passes; retain batch-invariant mode for future "
            "exact baselines"
            if outcome == "A"
            else (
                "Run K=1/2/4 single-request local-static versus remote-fixed controls; "
                "do not relax exact equality"
                if outcome == "controls-required"
                else "Reject inconsistent artifacts and rerun C/D before classification"
            )
        ),
        "gpu_performance_result": False,
        "reports_goodput": False,
        "reports_slo_attainment": False,
        "dual_batch": False,
        "packed_tree_verification": False,
    }


def correctness_markdown(report: Mapping[str, Any]) -> str:
    checks = report.get("checks", {})
    lines = [
        "# Phase 4A.1.1 batch-invariant correctness",
        "",
        f"- Outcome: `{report.get('outcome')}`",
        f"- Validation pass: `{str(report.get('valid')).lower()}`",
        "- Scope: exact-token correctness only; no serving-performance result",
        "",
        "## Checks",
        "",
    ]
    if isinstance(checks, Mapping):
        lines.extend(f"- {key}: `{value}`" for key, value in checks.items())
    errors = report.get("errors", ())
    if errors:
        lines.extend(("", "## Errors", ""))
        lines.extend(f"- {error}" for error in errors)
    lines.extend(("", "## Next action", "", str(report.get("next_action", "")), ""))
    return "\n".join(lines)


def validate_fixed_control_matrix(
    *,
    local_run_paths: Sequence[Path],
    remote_run_paths: Sequence[Path],
    local_diagnostic_paths: Sequence[Path],
    remote_diagnostic_paths: Sequence[Path],
) -> dict[str, Any]:
    groups = (
        local_run_paths,
        remote_run_paths,
        local_diagnostic_paths,
        remote_diagnostic_paths,
    )
    if any(len(group) != 3 for group in groups):
        raise ValueError("fixed controls require local and remote K=1/2/4 artifacts")
    errors = []
    comparisons = []
    expected_budgets = (1, 2, 4)
    for index, budget in enumerate(expected_budgets):
        local_run = _json(local_run_paths[index])
        remote_run = _json(remote_run_paths[index])
        if local_run.get("proposal_budget") != budget or remote_run.get(
            "proposal_budget"
        ) != budget:
            errors.append(f"K={budget} controls carry the wrong budget")
        if local_run.get("proposer") != "local-static":
            errors.append(f"K={budget} local artifact has the wrong proposer")
        if remote_run.get("proposer") != "remote-fixed":
            errors.append(f"K={budget} remote artifact has the wrong proposer")
        local_rows = CheckpointJsonl(local_diagnostic_paths[index]).read()
        remote_rows = CheckpointJsonl(remote_diagnostic_paths[index]).read()
        local_diag = next((row for row in local_rows if row.get("proposal_token_ids")), {})
        remote_diag = next((row for row in remote_rows if row.get("proposal_token_ids")), {})

        def evidence(run: Mapping[str, Any], diagnostic: Mapping[str, Any]) -> dict[str, Any]:
            outputs = run.get("outputs")
            output = outputs[0] if isinstance(outputs, list) and outputs else {}
            generated = list(output.get("generated_token_ids", ()))
            proposal = list(run.get("proposal_token_ids", ()))
            post_bootstrap = generated[1:]
            accepted = 0
            for candidate, target in zip(proposal, post_bootstrap):
                if candidate != target:
                    break
                accepted += 1
            committed_count = min(len(post_bootstrap), accepted + 1)
            return {
                "proposal_token_ids": proposal,
                "top_raw_logits": diagnostic.get("top_raw_logits"),
                "top_target_logprobs": diagnostic.get("top_target_logprobs"),
                "accepted_prefix_length": accepted,
                "committed_token_ids": post_bootstrap[:committed_count],
                "final_trajectory": _trajectories(outputs),
            }

        local_evidence = evidence(local_run, local_diag)
        remote_evidence = evidence(remote_run, remote_diag)
        comparison = compare_fixed_proposal_controls(local_evidence, remote_evidence)
        comparison["proposal_budget"] = budget
        comparison["final_trajectory_equal"] = (
            local_evidence["final_trajectory"] == remote_evidence["final_trajectory"]
        )
        if not comparison["local_remote_equal"] or not comparison[
            "final_trajectory_equal"
        ]:
            errors.append(f"K={budget} local and remote fixed controls differ")
        comparisons.append(comparison)
    all_equal = not errors
    return {
        "schema_version": "specrhythm.phase4-fixed-control-matrix.v1",
        "valid": all_equal,
        "local_remote_all_equal": all_equal,
        "comparisons": comparisons,
        "errors": errors,
        "classification_if_c_ne_d": (
            "Outcome B is eligible only if the separate prefix/position/KV/mask proof also "
            "passes"
            if all_equal
            else "Outcome C: integration bug; fix and rerun Phase 4A.1"
        ),
        "gpu_performance_result": False,
    }


def diagnose_first_divergence(
    *, stock_diagnostics_path: Path, serial_diagnostics_path: Path, serial_run_path: Path
) -> dict[str, Any]:
    run = _json(serial_run_path)
    comparisons = run.get("comparison", {}).get("requests", ())
    divergence = next(
        (
            row
            for row in comparisons
            if isinstance(row, Mapping)
            and isinstance(row.get("first_divergence_position"), int)
        ),
        None,
    )
    if divergence is None:
        return {
            "schema_version": "specrhythm.phase4-first-divergence.v1",
            "divergence_found": False,
            "valid": run.get("exact_sequence_match") is True,
            "outcome": "A" if run.get("exact_sequence_match") is True else "invalid",
        }
    prefix_hash = divergence.get("prefix_hash")
    if not isinstance(prefix_hash, str) or not prefix_hash:
        raise ValueError("Serial divergence is not linked to a committed prefix hash")
    proof = compare_divergence_diagnostics(
        CheckpointJsonl(stock_diagnostics_path).read(),
        CheckpointJsonl(serial_diagnostics_path).read(),
        request_id=str(divergence.get("request_id", "")),
        committed_prefix_sha256=prefix_hash,
    )
    proof.update(
        {
            "schema_version": "specrhythm.phase4-first-divergence.v1",
            "divergence_found": True,
            "first_divergence_position": divergence["first_divergence_position"],
            "stock_token_id": divergence.get("stock_token_id"),
            "serial_token_id": divergence.get("actual_token_id"),
            "round_id": divergence.get("round_id"),
            "proposal_token_ids": divergence.get("proposal_token_ids"),
            "accepted_prefix_length": divergence.get("accepted_prefix_length"),
            "outcome_if_controls_equal": "B" if proof["valid"] else "C",
        }
    )
    return proof
