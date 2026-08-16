"""Read-only Phase-4B Dual-Batch correctness and overlap validator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from specrhythm.phase4.correctness_validation import compare_round_semantics
from specrhythm.phase4.dual import RequestState, validate_cycle_rows
from specrhythm.phase4.manifest import atomic_write_json, sha256_file
from specrhythm.phase4.reference import load_reference
from specrhythm.phase4.transport import CheckpointJsonl
from specrhythm.phase4.vllm_diagnostics import (
    validate_kv_monotonicity,
    validate_target_diagnostic,
)


def validate_dual_batch_runs(
    *,
    stock_references: Sequence[Path],
    target_regression_path: Path,
    run_paths: Sequence[Path],
    state_event_paths: Sequence[Path],
    proposal_event_paths: Sequence[Path],
    cycle_event_paths: Sequence[Path],
    overlap_event_paths: Sequence[Path],
    draft_work_event_paths: Sequence[Path],
    target_diagnostic_paths: Sequence[Path],
    output_path: Path,
    markdown_path: Path,
) -> dict[str, Any]:
    inputs = [
        *stock_references,
        target_regression_path,
        *run_paths,
        *state_event_paths,
        *proposal_event_paths,
        *cycle_event_paths,
        *overlap_event_paths,
        *draft_work_event_paths,
        *target_diagnostic_paths,
    ]
    before = {str(path.resolve()): sha256_file(path) for path in inputs}
    if len(stock_references) != 2 or len(run_paths) != 2:
        raise ValueError("validator requires two stock references and two Dual-Batch runs")
    paired_lengths = {
        len(state_event_paths),
        len(proposal_event_paths),
        len(cycle_event_paths),
        len(overlap_event_paths),
        len(draft_work_event_paths),
        len(target_diagnostic_paths),
        len(run_paths),
    }
    if paired_lengths != {2}:
        raise ValueError("every Dual-Batch run requires one complete artifact set")
    references = [load_reference(path) for path in stock_references]
    runs = [_load_object(path) for path in run_paths]
    target_regression = _load_object(target_regression_path)
    state_rows = [CheckpointJsonl(path).read() for path in state_event_paths]
    proposal_rows = [CheckpointJsonl(path).read() for path in proposal_event_paths]
    cycle_rows = [CheckpointJsonl(path).read() for path in cycle_event_paths]
    overlap_rows = [CheckpointJsonl(path).read() for path in overlap_event_paths]
    draft_rows = [CheckpointJsonl(path).read() for path in draft_work_event_paths]
    diagnostic_rows = [CheckpointJsonl(path).read() for path in target_diagnostic_paths]
    errors: list[str] = []

    target_only_equal = _outputs(references[0]) == _outputs(references[1])
    if not target_only_equal:
        errors.append("stock Target-only references differ")
    dual_equal = _outputs(runs[0]) == _outputs(runs[1])
    if not dual_equal:
        errors.append("Dual-Batch repeated final outputs differ")
    dual_equals_stock = [_outputs(run) == _outputs(references[0]) for run in runs]
    if not all(dual_equals_stock):
        errors.append("Dual-Batch final output differs from stock Target-only")
    termination_equal = [
        _terminations(run) == _terminations(references[0]) for run in runs
    ]
    if not all(termination_equal):
        errors.append("Dual-Batch termination differs from stock Target-only")

    round_comparison = compare_round_semantics(proposal_rows[0], proposal_rows[1])
    keyed_round_equal = bool(round_comparison.get("valid"))
    if not keyed_round_equal:
        errors.append("keyed proposal/acceptance/commit semantics differ across runs")
    state_errors = [_validate_state_events(rows) for rows in state_rows]
    for index, values in enumerate(state_errors):
        errors.extend(f"run {index + 1} state: {item}" for item in values)
    cycle_errors = [validate_cycle_rows(rows) for rows in cycle_rows]
    for index, values in enumerate(cycle_errors):
        errors.extend(f"run {index + 1} cycle: {item}" for item in values)
    prefix_errors = [_validate_prefix_and_accounting(rows) for rows in proposal_rows]
    for index, values in enumerate(prefix_errors):
        errors.extend(f"run {index + 1} proposal: {item}" for item in values)
    commit_sequence_errors = [
        _validate_final_commit_concat(run, rows)
        for run, rows in zip(runs, proposal_rows)
    ]
    for index, values in enumerate(commit_sequence_errors):
        errors.extend(f"run {index + 1} commit sequence: {item}" for item in values)
    draft_kv_errors = [_validate_draft_kv(rows) for rows in draft_rows]
    for index, values in enumerate(draft_kv_errors):
        errors.extend(f"run {index + 1} Draft KV: {item}" for item in values)
    target_kv_errors = [_validate_target_diagnostics(rows) for rows in diagnostic_rows]
    for index, values in enumerate(target_kv_errors):
        errors.extend(f"run {index + 1} Target KV: {item}" for item in values)
    identity_errors = [
        _validate_identity_domains(
            run,
            state_rows[index],
            proposal_rows[index],
            cycle_rows[index],
            draft_rows[index],
            diagnostic_rows[index],
        )
        for index, run in enumerate(runs)
    ]
    for index, values in enumerate(identity_errors):
        errors.extend(f"run {index + 1} identity: {item}" for item in values)
    batch_effective = [
        run.get("batch_invariant_effective") is True
        and all(
            row.get("batch_invariant_effective") is True
            for row in run.get("worker_ranks", ())
        )
        for run in runs
    ]
    if not all(batch_effective):
        errors.append("batch-invariant execution is not effective on every Target rank")
    patch_regression_valid = _target_regression_valid(target_regression)
    if not patch_regression_valid:
        errors.append("patched Target-only regression is invalid")
    placement_errors = [_validate_placement(rows) for rows in overlap_rows]
    for index, values in enumerate(placement_errors):
        errors.extend(f"run {index + 1} placement: {item}" for item in values)
    overlap_observed = [
        any(int(row.get("overlap_duration_ns", 0)) > 0 for row in rows)
        for rows in overlap_rows
    ]
    if not all(overlap_observed):
        errors.append("at least one run lacks a positive real-GPU overlap interval")
    raw_event_order_equal = bool(round_comparison.get("raw_event_order_equal"))
    request_count_ok = [
        int(run.get("request_count", 0)) == len(_outputs(run)) for run in runs
    ]
    if not all(request_count_ok):
        errors.append("run output coverage is incomplete")

    exact_and_semantic = (
        target_only_equal
        and dual_equal
        and all(dual_equals_stock)
        and all(termination_equal)
        and keyed_round_equal
        and not any(state_errors)
        and not any(cycle_errors)
        and not any(prefix_errors)
        and not any(commit_sequence_errors)
        and not any(draft_kv_errors)
        and not any(target_kv_errors)
        and not any(identity_errors)
        and all(batch_effective)
        and patch_regression_valid
        and not any(placement_errors)
    )
    if exact_and_semantic and all(overlap_observed):
        outcome = "A"
    elif exact_and_semantic:
        outcome = "B"
    else:
        outcome = "C"
    after = {str(path.resolve()): sha256_file(path) for path in inputs}
    immutable = before == after
    if not immutable:
        errors.append("validator mutated an input artifact")
    valid = outcome == "A" and immutable and not errors
    result = {
        "schema_version": "specrhythm.phase4b-dual-batch-validation.v1",
        "valid": valid,
        "outcome": outcome,
        "performance_result": False,
        "target_only_repeated_exact_equality": target_only_equal,
        "dual_batch_repeated_exact_equality": dual_equal,
        "dual_batch_equals_stock": dual_equals_stock,
        "termination_equality": termination_equal,
        "keyed_round_semantics_repeated": keyed_round_equal,
        "round_semantics_comparison": round_comparison,
        "request_state_machine_valid": not any(state_errors),
        "draft_verify_request_sets_disjoint": not any(cycle_errors),
        "prefix_versions_monotonic": not any(prefix_errors),
        "no_stale_proposal_committed": not any(prefix_errors),
        "proposal_accounting_valid": not any(prefix_errors),
        "commit_accounting_valid": not any(prefix_errors)
        and not any(commit_sequence_errors),
        "draft_kv_valid": not any(draft_kv_errors),
        "target_kv_valid": not any(target_kv_errors),
        "stable_request_identity_valid": not any(identity_errors),
        "batch_invariant_effective": batch_effective,
        "target_patch_regression_valid": patch_regression_valid,
        "gpu_placement_valid": not any(placement_errors),
        "real_gpu_overlap_observed": overlap_observed,
        "raw_event_order_equal": raw_event_order_equal,
        "key_sets_equal": round_comparison.get("key_sets_equal"),
        "semantic_mismatches": round_comparison.get("semantic_mismatches", []),
        "missing_events": round_comparison.get("missing_in_d2", []),
        "extra_events": round_comparison.get("extra_in_d2", []),
        "input_artifacts_immutable": immutable,
        "input_sha256_before": before,
        "input_sha256_after": after,
        "errors": errors,
        "claim_boundary": (
            "Phase 4B.1 validates correctness and existence of Draft/Verify GPU "
            "overlap only; it does not establish speedup, goodput, SLO, or production claims."
        ),
    }
    atomic_write_json(output_path, result)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_markdown(result), encoding="utf-8")
    return result


def _validate_state_events(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors = []
    by_request: dict[str, list[Mapping[str, Any]]] = {}
    legal = {
        state.value: {item.value for item in destinations}
        for state, destinations in _legal_transitions().items()
    }
    for row in rows:
        by_request.setdefault(str(row.get("request_id", "")), []).append(row)
    for request_id, values in by_request.items():
        previous = "BOOTSTRAP"
        prior_version = -1
        for row in values:
            source = str(row.get("source_state", ""))
            destination = str(row.get("destination_state", ""))
            if source != previous:
                errors.append(f"{request_id}: non-contiguous state event")
            if destination not in legal.get(source, set()):
                errors.append(f"{request_id}: illegal {source}->{destination}")
            version = row.get("prefix_version")
            if not isinstance(version, int) or version < prior_version:
                errors.append(f"{request_id}: prefix version regressed")
            previous = destination
            prior_version = version if isinstance(version, int) else prior_version
        if previous not in {"FINISHED", "DRAFTING"}:
            errors.append(f"{request_id}: checkpoint ended in unsupported state {previous}")
    return errors


def _legal_transitions() -> Mapping[RequestState, set[RequestState]]:
    # Kept local to avoid exposing mutation of the contract module's table.
    return {
        RequestState.BOOTSTRAP: {RequestState.DRAFT_READY, RequestState.FINISHED},
        RequestState.DRAFT_READY: {
            RequestState.DRAFTING,
            RequestState.VERIFY_READY,
            RequestState.FINISHED,
        },
        RequestState.DRAFTING: {
            RequestState.PROPOSAL_READY,
            RequestState.DRAFT_READY,
            RequestState.FAILED,
        },
        RequestState.PROPOSAL_READY: {RequestState.VERIFY_READY, RequestState.FAILED},
        RequestState.VERIFY_READY: {RequestState.VERIFYING, RequestState.FAILED},
        RequestState.VERIFYING: {RequestState.COMMITTING, RequestState.FAILED},
        RequestState.COMMITTING: {
            RequestState.DRAFT_SYNC,
            RequestState.FINISHED,
            RequestState.FAILED,
        },
        RequestState.DRAFT_SYNC: {
            RequestState.DRAFT_READY,
            RequestState.FINISHED,
            RequestState.FAILED,
        },
        RequestState.FINISHED: set(),
        RequestState.FAILED: set(),
    }


def _validate_prefix_and_accounting(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors = []
    by_request: dict[str, list[Mapping[str, Any]]] = {}
    keys = set()
    for row in rows:
        key = (str(row.get("request_id", "")), row.get("round_id"))
        if key in keys:
            errors.append(f"duplicate proposal event key {key}")
        keys.add(key)
        by_request.setdefault(key[0], []).append(row)
        proposed = len(row.get("proposal_token_ids", ()))
        accepted = row.get("accepted_draft_tokens")
        rejected = row.get("rejected_draft_tokens")
        counts_valid = isinstance(accepted, int) and isinstance(rejected, int)
        if not counts_valid or proposed != accepted + rejected:
            errors.append(f"{key}: accepted + rejected != proposed")
        committed = len(row.get("committed_token_ids", ()))
        expected = accepted + len(row.get("target_correction_token_ids", ())) + len(
            row.get("target_bonus_token_ids", ())
        ) if isinstance(accepted, int) else -1
        if committed != expected:
            errors.append(f"{key}: committed-token accounting mismatch")
        if row.get("stale") is not False:
            errors.append(f"{key}: stale proposal was recorded as committed")
    for request_id, values in by_request.items():
        ordered = sorted(values, key=lambda row: row.get("round_id", -1))
        if [row.get("round_id") for row in ordered] != list(range(len(ordered))):
            errors.append(f"{request_id}: rounds are not contiguous")
        versions = [row.get("prefix_version") for row in ordered]
        if any(not isinstance(value, int) for value in versions) or versions != list(
            range(1, len(versions) + 1)
        ):
            errors.append(f"{request_id}: proposal prefix versions are not monotonic")
    return errors


def _validate_draft_kv(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors = []
    for row in rows:
        if row.get("success") is not True:
            errors.append(f"{row.get('request_id')}: Draft work failed")
            continue
        result = row.get("result")
        if not isinstance(result, Mapping):
            errors.append(f"{row.get('request_id')}: Draft work result missing")
            continue
        proposal = result.get("proposal")
        if isinstance(proposal, Mapping):
            before = proposal.get("draft_kv_length_before")
            after = proposal.get("draft_kv_length_after")
            length = proposal.get("proposal_length")
            valid_lengths = all(
                isinstance(value, int) for value in (before, after, length)
            )
            if not valid_lengths or after != before + length:
                errors.append(f"{row.get('request_id')}: Draft proposal KV accounting mismatch")
        logical = result.get("logical_draft_kv_length")
        if logical is not None and (not isinstance(logical, int) or logical <= 0):
            errors.append(f"{row.get('request_id')}: invalid synchronized Draft KV length")
    return errors


def _validate_final_commit_concat(
    run: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> list[str]:
    errors = []
    rounds: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        rounds.setdefault(str(row.get("request_id", "")), []).append(row)
    for output in run.get("outputs", ()):
        request_id = str(output.get("request_id", ""))
        generated = list(output.get("generated_token_ids", ()))
        committed = []
        for row in sorted(rounds.get(request_id, ()), key=lambda item: item["round_id"]):
            committed.extend(row.get("committed_token_ids", ()))
        if not generated:
            errors.append(f"{request_id}: final generated sequence is empty")
            continue
        body = generated[1 : 1 + len(committed)]
        if body != committed:
            errors.append(
                f"{request_id}: final sequence is not bootstrap + round commits + tail"
            )
        if len(generated) - 1 - len(committed) not in {0, 1}:
            errors.append(f"{request_id}: invalid proposal-free Target tail length")
    return errors


def _validate_target_diagnostics(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors = list(validate_kv_monotonicity(rows))
    for row in rows:
        errors.extend(validate_target_diagnostic(row))
        if row.get("target_kv_contains_rejected_or_future_tokens") is not False:
            errors.append(f"{row.get('request_id')}: rejected/future Target KV is visible")
        if row.get("batch_invariant_requested") is not True:
            errors.append(f"{row.get('request_id')}: diagnostic is not batch-invariant")
    return errors


def _validate_placement(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors = []
    for row in rows:
        if row.get("request_sets_disjoint") is not True:
            errors.append(f"cycle {row.get('cycle_id')}: request sets overlap")
        if row.get("draft_physical_gpu_ids") not in ([0], []):
            errors.append(f"cycle {row.get('cycle_id')}: Draft is not on GPU 0")
        target = row.get("target_physical_gpu_ids")
        if target and target != [1, 2]:
            errors.append(f"cycle {row.get('cycle_id')}: Target is not on GPUs 1,2")
        if int(row.get("overlap_duration_ns", 0)) > 0:
            if row.get("draft_physical_gpu_ids") != [0] or target != [1, 2]:
                errors.append(f"cycle {row.get('cycle_id')}: positive overlap lacks GPU proof")
            if row.get("draft_cuda_events") is not True:
                errors.append(f"cycle {row.get('cycle_id')}: Draft CUDA event evidence missing")
            rank_rows = row.get("target_rank_intervals", ())
            if len(rank_rows) != 2 or any(
                item.get("cuda_events") is not True for item in rank_rows
            ):
                errors.append(f"cycle {row.get('cycle_id')}: Target rank CUDA evidence missing")
            elif len({item.get("physical_gpu_id") for item in rank_rows}) != 2:
                errors.append(
                    f"cycle {row.get('cycle_id')}: Target rank GPU identities are not unique"
                )
    return errors


def _validate_identity_domains(
    run: Mapping[str, Any],
    state_rows: Sequence[Mapping[str, Any]],
    proposal_rows: Sequence[Mapping[str, Any]],
    cycle_rows: Sequence[Mapping[str, Any]],
    draft_rows: Sequence[Mapping[str, Any]],
    diagnostic_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    errors = []
    stable_ids = {str(row.get("request_id", "")) for row in run.get("outputs", ())}
    identity = run.get("request_identity")
    if not stable_ids or "" in stable_ids:
        errors.append("final outputs lack stable request IDs")
    if not isinstance(identity, Mapping):
        errors.append("run lacks explicit internal/stable identity evidence")
    else:
        if identity.get("mapping_source") != "unique frozen prompt_token_ids":
            errors.append("run did not use frozen prompt-token identity mapping")
        if identity.get("suffix_parsing") is not False:
            errors.append("run parsed opaque vLLM request IDs")
        bindings = identity.get("bindings", ())
        if not isinstance(bindings, list):
            errors.append("run identity bindings are not a list")
        elif any(not isinstance(row, Mapping) for row in bindings):
            errors.append("run identity binding is not an object")
        else:
            internal = [str(row.get("internal_request_id", "")) for row in bindings]
            stable = [str(row.get("request_id", "")) for row in bindings]
            if any(not item for item in internal + stable):
                errors.append("run identity binding contains an empty ID")
            if len(internal) != len(set(internal)):
                errors.append("internal request identity is not one-to-one")
            if len(stable) != len(set(stable)):
                errors.append("stable request identity is aliased")
            if set(stable) != stable_ids:
                errors.append("identity bindings do not cover final outputs")
    for label, rows in (
        ("state", state_rows),
        ("proposal", proposal_rows),
        ("Draft", draft_rows),
        ("Target diagnostic", diagnostic_rows),
    ):
        unknown = {
            str(row.get("request_id", "")) for row in rows
        } - stable_ids
        if unknown:
            errors.append(f"{label} events use non-stable request IDs: {sorted(unknown)}")
    for row in cycle_rows:
        observed = {
            str(item)
            for field in ("draft_request_ids", "verify_request_ids")
            for item in row.get(field, ())
        }
        unknown = observed - stable_ids
        if unknown:
            errors.append(f"cycle events use non-stable request IDs: {sorted(unknown)}")
    return errors


def _target_regression_valid(value: Mapping[str, Any]) -> bool:
    comparison = value.get("comparison")
    if isinstance(comparison, Mapping):
        return comparison.get("all_sequences_equal") is True
    return value.get("exact_sequence_match") is True


def _outputs(value: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    return {
        str(row.get("request_id")): tuple(row.get("generated_token_ids", ()))
        for row in value.get("outputs", ())
    }


def _terminations(value: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
    return {
        str(row.get("request_id")): (row.get("finish_reason"), row.get("stop_reason"))
        for row in value.get("outputs", ())
    }


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _markdown(value: Mapping[str, Any]) -> str:
    status = "PASS" if value["valid"] else "FAIL"
    state_kv_accounting = (
        value["request_state_machine_valid"]
        and value["draft_kv_valid"]
        and value["target_kv_valid"]
    )
    lines = [
        "# Phase 4B.1 Dual-Batch validation",
        "",
        f"- Validation: **{status}**",
        f"- Outcome: **{value['outcome']}**",
        f"- Exact Dual-vs-stock: `{value['dual_batch_equals_stock']}`",
        f"- Repeated keyed semantics: `{value['keyed_round_semantics_repeated']}`",
        f"- State/KV/accounting valid: `{state_kv_accounting}`",
        f"- Positive real-GPU overlap per run: `{value['real_gpu_overlap_observed']}`",
        f"- Raw event order equal (diagnostic only): `{value['raw_event_order_equal']}`",
        "",
        str(value["claim_boundary"]),
        "",
    ]
    if value["errors"]:
        lines.extend(["## Errors", "", *[f"- {item}" for item in value["errors"]], ""])
    return "\n".join(lines)
