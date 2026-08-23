"""Read-only Phase-4B.1 decode-only Dual-Batch correctness authority.

The validator deliberately compares logical serving artifacts, not Phase-3 HF
trajectories and not cross-request JSONL write order. Every failure is explicit;
the only successful classification is outcome A.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from specrhythm.phase4.admissibility import validate_admissibility_events
from specrhythm.phase4.correctness_validation import compare_round_semantics
from specrhythm.phase4.decode_ready import load_decode_ready_manifest
from specrhythm.phase4.manifest import atomic_write_json, sha256_file
from specrhythm.phase4.process_lifecycle import validate_lifecycle_artifact
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.transport import CheckpointJsonl

VALIDATION_SCHEMA = "specrhythm.phase4b1-dual-correctness-validation.v1"
CONTROLLED_SCHEMA = "specrhythm.phase4b1-controlled-gate-validation.v1"

_LEGAL_STATES = {
    "BOOTSTRAP": {"DRAFT_READY", "TERMINAL"},
    "DRAFT_READY": {"DRAFTING", "TARGET_TAIL_READY", "TERMINAL"},
    "DRAFTING": {"PROPOSAL_READY", "TARGET_TAIL_READY", "FAILED"},
    "PROPOSAL_READY": {"VERIFY_READY", "DROPPED_STALE", "FAILED"},
    "VERIFY_READY": {"VERIFYING", "DROPPED_STALE", "FAILED"},
    "VERIFYING": {"COMMITTING", "FAILED"},
    "COMMITTING": {"DRAFT_SYNC", "TERMINAL", "FAILED"},
    "DRAFT_SYNC": {"DRAFT_READY", "TARGET_TAIL_READY", "TERMINAL", "FAILED"},
    "TARGET_TAIL_READY": {"VERIFYING", "TERMINAL", "FAILED"},
    "DROPPED_STALE": {"DRAFT_READY", "TERMINAL", "FAILED"},
    "TERMINAL": set(),
    "FAILED": set(),
}

def validate_phase4b1_dual_correctness(
    *,
    target_path: Path,
    serial_path: Path,
    dual_paths: Sequence[Path],
    target_manifest_path: Path,
    serial_manifest_path: Path,
    target_process_lifecycle_path: Path,
    serial_process_lifecycle_path: Path,
    dual_manifest_paths: Sequence[Path],
    state_event_paths: Sequence[Path],
    proposal_event_paths: Sequence[Path],
    proposal_lifecycle_paths: Sequence[Path],
    scheduler_event_paths: Sequence[Path],
    verification_event_paths: Sequence[Path],
    draft_work_event_paths: Sequence[Path],
    target_diagnostic_paths: Sequence[Path],
    overlap_event_paths: Sequence[Path],
    process_lifecycle_paths: Sequence[Path],
    output_path: Path,
    markdown_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Validate one Target, one Serial, and one or more immutable Dual runs."""

    if not dual_paths:
        raise ValueError("at least one decode-only Dual run is required")
    paired = (
        dual_manifest_paths,
        state_event_paths,
        proposal_event_paths,
        proposal_lifecycle_paths,
        scheduler_event_paths,
        verification_event_paths,
        draft_work_event_paths,
        target_diagnostic_paths,
        overlap_event_paths,
        process_lifecycle_paths,
    )
    if any(len(values) != len(dual_paths) for values in paired):
        raise ValueError("every Dual run requires one complete evidence set")
    inputs = [
        target_path,
        serial_path,
        target_manifest_path,
        serial_manifest_path,
        target_process_lifecycle_path,
        serial_process_lifecycle_path,
        *dual_paths,
        *dual_manifest_paths,
        *state_event_paths,
        *proposal_event_paths,
        *proposal_lifecycle_paths,
        *scheduler_event_paths,
        *verification_event_paths,
        *draft_work_event_paths,
        *target_diagnostic_paths,
        *overlap_event_paths,
        *process_lifecycle_paths,
    ]
    before = {str(path.resolve()): sha256_file(path) for path in inputs}
    target = _read_object(target_path)
    serial = _read_object(serial_path)
    duals = [_read_object(path) for path in dual_paths]
    manifests = [
        load_decode_ready_manifest(_read_object(target_manifest_path)),
        load_decode_ready_manifest(_read_object(serial_manifest_path)),
        *[
            load_decode_ready_manifest(_read_object(path))
            for path in dual_manifest_paths
        ],
    ]
    states = [CheckpointJsonl(path).read() for path in state_event_paths]
    rounds = [CheckpointJsonl(path).read() for path in proposal_event_paths]
    lifecycles = [CheckpointJsonl(path).read() for path in proposal_lifecycle_paths]
    schedulers = [CheckpointJsonl(path).read() for path in scheduler_event_paths]
    verifications = [CheckpointJsonl(path).read() for path in verification_event_paths]
    draft_work = [CheckpointJsonl(path).read() for path in draft_work_event_paths]
    diagnostics = [CheckpointJsonl(path).read() for path in target_diagnostic_paths]
    overlaps = [CheckpointJsonl(path).read() for path in overlap_event_paths]
    process_lifecycles = [_read_object(path) for path in process_lifecycle_paths]
    baseline_lifecycles = [
        _read_object(target_process_lifecycle_path),
        _read_object(serial_process_lifecycle_path),
    ]
    errors: list[str] = []

    for label, run in [("Target", target), ("Serial", serial), *[
        (f"Dual-{index + 1}", run) for index, run in enumerate(duals)
    ]]:
        if run.get("valid") is not True:
            errors.append(f"{label} run is not valid")

    identities = [_manifest_identity(item) for item in manifests]
    manifest_identity_equal = all(item == identities[0] for item in identities[1:])
    if not manifest_identity_equal:
        errors.append("Target/Serial/Dual decode-ready logical manifest identities differ")
    for label, lifecycle in zip(("Target", "Serial"), baseline_lifecycles):
        errors.extend(f"{label} cleanup: {item}" for item in validate_cleanup(lifecycle))

    consumers = [target, serial, *duals]
    triangle = compare_consumer_triangle(consumers)
    if not triangle["valid"]:
        errors.extend(triangle["errors"])

    component_results = []
    for index in range(len(duals)):
        values = {
            "state_machine": validate_request_state_events(states[index]),
            "proposal_lifecycle": validate_proposal_lifecycle_events(
                lifecycles[index]
            ),
            "scheduler": validate_scheduler_cycles(schedulers[index]),
            "token_accounting": validate_round_accounting(rounds[index]),
            "verification_contract": validate_verification_contracts(
                rounds[index], verifications[index], diagnostics[index]
            ),
            "draft_sync": validate_draft_sync(
                draft_work[index], rounds[index], states[index]
            ),
            "target_blind": validate_target_blind_isolation(
                duals[index], draft_work[index], rounds[index]
            ),
            "measurement_boundary": validate_measurement_boundary(
                manifests[index + 2], lifecycles[index], draft_work[index]
            ),
            "overlap": validate_overlap_witness(overlaps[index]),
            "cleanup": validate_cleanup(process_lifecycles[index]),
        }
        values["token_accounting"].extend(
            validate_final_commit_sequence(
                duals[index],
                manifests[index + 2],
                rounds[index],
                draft_work[index],
                states[index],
            )
        )
        for name, component_errors in values.items():
            errors.extend(
                f"Dual-{index + 1} {name}: {item}" for item in component_errors
            )
        component_results.append(
            {name: {"valid": not value, "errors": value} for name, value in values.items()}
        )

    repeats = []
    for index in range(1, len(duals)):
        comparison = compare_round_semantics(rounds[0], rounds[index])
        first_versions = _round_prefix_versions(rounds[0])
        current_versions = _round_prefix_versions(rounds[index])
        prefix_versions_equal = first_versions == current_versions
        output_equal = _canonical_outputs(duals[0]) == _canonical_outputs(duals[index])
        valid = (
            comparison.get("valid") is True
            and prefix_versions_equal
            and output_equal
        )
        if not valid:
            errors.append(f"Dual-1/Dual-{index + 1} repeated semantics differ")
        repeats.append(
            {
                "left": 1,
                "right": index + 1,
                "valid": valid,
                "per_request_outputs_equal": output_equal,
                "prefix_version_transitions_equal": prefix_versions_equal,
                "keyed_round_semantics": comparison,
                "raw_event_order_equal": comparison.get("raw_event_order_equal"),
            }
        )

    after = {str(path.resolve()): sha256_file(path) for path in inputs}
    immutable = before == after
    if not immutable:
        errors.append("validator mutated one or more input artifacts")
    errors = list(dict.fromkeys(errors))
    valid = not errors
    result = {
        "schema_version": VALIDATION_SCHEMA,
        "valid": valid,
        "outcome": "A" if valid else "FAIL",
        "errors": errors,
        "performance_result": False,
        "correctness_scope": "real-decode-only-dual-batch",
        "decode_ready_manifest_identity_equal": manifest_identity_equal,
        "triangle": triangle,
        "dual_runs": component_results,
        "repeat_comparisons": repeats,
        "input_artifacts_immutable": immutable,
        "input_sha256_before": before,
        "input_sha256_after": after,
        "claim_boundary": (
            "Correctness and real cross-request overlap existence only; no TPOT, "
            "throughput, goodput, SLO, speedup, or overlap-benefit claim."
        ),
    }
    atomic_write_json(output_path, result)
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(_markdown(result), encoding="utf-8")
    return result


def compare_consumer_triangle(consumers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(consumers) < 3:
        raise ValueError("triangle comparison requires Target, Serial, and Dual")
    labels = ["target", "serial", *[f"dual-{index}" for index in range(1, len(consumers) - 1)]]
    outputs = [_canonical_outputs(item) for item in consumers]
    request_sets = [set(item) for item in outputs]
    union = set().union(*request_sets)
    intersection = set.intersection(*request_sets) if request_sets else set()
    errors = []
    comparisons = []
    baseline = outputs[0]
    for index, current in enumerate(outputs[1:], 1):
        missing = sorted(set(baseline) - set(current))
        extra = sorted(set(current) - set(baseline))
        divergent = []
        for request_id in sorted(set(baseline) & set(current)):
            if baseline[request_id] != current[request_id]:
                divergent.append(
                    {
                        "request_id": request_id,
                        **_first_divergence(
                            baseline[request_id][0], current[request_id][0]
                        ),
                        "target_termination": list(baseline[request_id][1:]),
                        "consumer_termination": list(current[request_id][1:]),
                    }
                )
        equal = not missing and not extra and not divergent
        if not equal:
            errors.append(f"target != {labels[index]} exact decode output")
        comparisons.append(
            {
                "left": "target",
                "right": labels[index],
                "equal": equal,
                "missing_requests": missing,
                "extra_requests": extra,
                "divergences": divergent,
            }
        )
    return {
        "valid": not errors,
        "errors": errors,
        "completed_request_sets_equal": all(item == request_sets[0] for item in request_sets[1:]),
        "completed_request_intersection": sorted(intersection),
        "completed_request_union": sorted(union),
        "comparisons": comparisons,
    }


def validate_controlled_gate(
    *,
    asynchronous_scheduler_path: Path,
    coordinated_scheduler_path: Path,
    state_event_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Validate Gate-1 A/B/C constructions without interpreting performance."""

    async_rows = CheckpointJsonl(asynchronous_scheduler_path).read()
    coordinated_rows = CheckpointJsonl(coordinated_scheduler_path).read()
    state_rows = CheckpointJsonl(state_event_path).read()
    case_a_cycles = []
    for cycle in async_rows:
        decisions = cycle.get("request_admissibility", ())
        waiting = [
            row
            for row in decisions
            if row.get("specrhythm_state") in {"WAITING_DRAFT", "DRAFTING"}
            and row.get("scheduled") is False
        ]
        verifying = [
            row
            for row in decisions
            if row.get("scheduled_operation") == "verify"
            and row.get("scheduled") is True
            and row.get("proposal_valid") is True
        ]
        if waiting and verifying and {
            str(row.get("request_id")) for row in waiting
        }.isdisjoint(str(row.get("request_id")) for row in verifying):
            case_a_cycles.append(cycle.get("cycle_id"))
    case_b_cycles = []
    for cycle in coordinated_rows:
        verifying = [
            row
            for row in cycle.get("request_admissibility", ())
            if row.get("scheduled_operation") == "verify"
            and row.get("scheduled") is True
            and row.get("proposal_valid") is True
        ]
        if len(verifying) >= 2 and len(
            {str(row.get("request_id")) for row in verifying}
        ) == len(verifying):
            case_b_cycles.append(cycle.get("cycle_id"))
    terminal_timestamps = {
        str(row.get("request_id")): row.get("timestamp_ns")
        for row in state_rows
        if row.get("destination_state") == "TERMINAL"
    }
    case_c_witnesses = []
    for terminal_id, timestamp in terminal_timestamps.items():
        if not isinstance(timestamp, int):
            continue
        terminal_scheduled_later = False
        continuing_ids = set()
        for cycle in async_rows:
            cycle_time = cycle.get("poll_start_ns")
            if not isinstance(cycle_time, int) or cycle_time < timestamp:
                continue
            for row in cycle.get("request_admissibility", ()):
                request_id = str(row.get("request_id", ""))
                if row.get("scheduled") is not True:
                    continue
                if request_id == terminal_id:
                    terminal_scheduled_later = True
                elif request_id != terminal_id:
                    continuing_ids.add(request_id)
        if continuing_ids and not terminal_scheduled_later:
            case_c_witnesses.append(
                {
                    "terminal_request_id": terminal_id,
                    "continuing_request_ids": sorted(continuing_ids),
                }
            )
    errors = []
    if not case_a_cycles:
        errors.append("Case A missing: no ready request advanced while another waited")
    if not case_b_cycles:
        errors.append("Case B missing: no coordinated two-proposal verification batch")
    if not case_c_witnesses:
        errors.append("Case C missing: terminal removal/remaining progress not proven")
    result = {
        "schema_version": CONTROLLED_SCHEMA,
        "valid": not errors,
        "outcome": "A" if not errors else "FAIL",
        "errors": errors,
        "case_a_one_ready_one_waiting_cycle_ids": case_a_cycles,
        "case_b_two_ready_cycle_ids": case_b_cycles,
        "case_c_terminal_removal_witnesses": case_c_witnesses,
        "test_only_coordination_is_performance_evidence": False,
        "performance_result": False,
    }
    atomic_write_json(output_path, result)
    return result


def validate_request_state_events(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors = []
    by_request: dict[str, list[Mapping[str, Any]]] = {}
    for index, row in enumerate(rows):
        request_id = str(row.get("request_id", ""))
        if not request_id:
            errors.append(f"state row {index} has no stable request ID")
            continue
        if not str(row.get("internal_request_id", "")):
            errors.append(f"{request_id}: state row has no internal request ID")
        if not str(row.get("reason", "")):
            errors.append(f"{request_id}: state transition has no reason")
        by_request.setdefault(request_id, []).append(row)
    for request_id, events in by_request.items():
        expected = "BOOTSTRAP"
        last_timestamp = -1
        last_version = -1
        for row in events:
            source = str(row.get("source_state", ""))
            destination = str(row.get("destination_state", ""))
            if source != expected:
                errors.append(f"{request_id}: non-contiguous state transition")
            if destination not in _LEGAL_STATES.get(source, set()):
                errors.append(f"{request_id}: illegal {source}->{destination}")
            timestamp = row.get("timestamp_ns")
            if not isinstance(timestamp, int) or timestamp <= last_timestamp:
                errors.append(f"{request_id}: state timestamps are not strictly monotonic")
            version = row.get("prefix_version")
            if not isinstance(version, int) or version < last_version:
                errors.append(f"{request_id}: prefix version regressed")
            if not str(row.get("committed_prefix_sha256", "")):
                errors.append(f"{request_id}: state row lacks logical prefix hash")
            expected = destination
            last_timestamp = timestamp if isinstance(timestamp, int) else last_timestamp
            last_version = version if isinstance(version, int) else last_version
        if expected != "TERMINAL":
            errors.append(f"{request_id}: final state is {expected}, not TERMINAL")
    if not by_request:
        errors.append("request state evidence is empty")
    return errors


def validate_proposal_lifecycle_events(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors = []
    by_id: dict[str, list[Mapping[str, Any]]] = {}
    round_keys: dict[tuple[str, int], str] = {}
    for index, row in enumerate(rows):
        proposal_id = str(row.get("proposal_id", ""))
        request_id = str(row.get("request_id", ""))
        round_id = row.get("round_id")
        if not proposal_id or not request_id or not isinstance(round_id, int):
            errors.append(f"proposal lifecycle row {index} has invalid identity")
            continue
        key = (request_id, round_id)
        previous = round_keys.setdefault(key, proposal_id)
        if previous != proposal_id:
            errors.append(f"{key}: more than one proposal ID was published")
        by_id.setdefault(proposal_id, []).append(row)
    published = consumed = dropped = 0
    for proposal_id, events in by_id.items():
        states = [str(row.get("lifecycle_state", "")) for row in events]
        expected = ["CREATED", "PUBLISHED", "INSTALLED", "CONSUMED"]
        if states in (
            ["CREATED", "PUBLISHED", "DROPPED_STALE"],
            ["CREATED", "PUBLISHED", "INSTALLED", "DROPPED_STALE"],
        ):
            dropped += 1
        elif states == expected:
            consumed += 1
        else:
            errors.append(f"{proposal_id}: invalid proposal lifecycle {states}")
        published += int("PUBLISHED" in states)
        timestamps = [row.get("timestamp_ns") for row in events]
        if any(not isinstance(item, int) for item in timestamps) or timestamps != sorted(
            timestamps
        ):
            errors.append(f"{proposal_id}: lifecycle timestamps are not monotonic")
        identity = {
            (
                row.get("request_id"),
                row.get("round_id"),
                row.get("prefix_version"),
                tuple(row.get("proposal_token_ids", ())),
            )
            for row in events
        }
        if len(identity) != 1:
            errors.append(f"{proposal_id}: lifecycle identity changed")
    if published != consumed + dropped:
        errors.append("published proposals != consumed + explicitly dropped proposals")
    if not by_id:
        errors.append("proposal lifecycle evidence is empty")
    return errors


def validate_scheduler_cycles(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors = []
    decisions = []
    ready_deferrals: dict[str, int] = {}
    seen_terminal: set[str] = set()
    seen_cycles = set()
    for cycle in rows:
        cycle_id = cycle.get("cycle_id")
        if cycle_id in seen_cycles:
            errors.append(f"duplicate scheduler cycle ID {cycle_id}")
        seen_cycles.add(cycle_id)
        current = cycle.get("request_admissibility")
        if not isinstance(current, list):
            errors.append(f"cycle {cycle_id}: request decisions are missing")
            continue
        scheduled_ids = set(cycle.get("scheduled_request_ids", ()))
        for row in current:
            decisions.append(row)
            request_id = str(row.get("request_id", ""))
            scheduled = row.get("scheduled") is True
            operation = row.get("scheduled_operation")
            state = row.get("specrhythm_state")
            if state in {"WAITING_DRAFT", "DRAFTING"} and scheduled:
                errors.append(f"{request_id}: waiting request consumed Target budget")
            if state in {"WAITING_DRAFT", "DRAFTING"} and row.get(
                "target_input_token_positions"
            ):
                errors.append(f"{request_id}: waiting request owns Target positions")
            if state == "TERMINAL":
                seen_terminal.add(request_id)
                if scheduled:
                    errors.append(f"{request_id}: terminal request was scheduled")
            if request_id in seen_terminal and scheduled:
                errors.append(f"{request_id}: request was scheduled after terminal evidence")
            if operation == "verify" and not row.get("proposal_valid"):
                errors.append(f"{request_id}: Target verification lacks matching proposal")
            if (
                scheduled
                and row.get("execution_phase") == "timed-decode"
                and operation not in {"verify", "target-tail"}
            ):
                errors.append(f"{request_id}: unexplained live Target advancement")
            if scheduled != (request_id in scheduled_ids):
                errors.append(f"{request_id}: cycle summary/decision scheduling mismatch")
            if row.get("admissible") and not scheduled:
                ready_deferrals[request_id] = ready_deferrals.get(request_id, 0) + 1
            elif scheduled:
                ready_deferrals[request_id] = 0
        if any(value > max(8, len(current) * 4) for value in ready_deferrals.values()):
            errors.append(f"cycle {cycle_id}: proposal-ready request deferred without bound")
    errors.extend(validate_admissibility_events(decisions))
    if not rows:
        errors.append("scheduler decision evidence is empty")
    return errors


def validate_round_accounting(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors = []
    seen = set()
    by_request: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        request_id = str(row.get("request_id", ""))
        key = (request_id, row.get("round_id"))
        if key in seen:
            errors.append(f"duplicate round key {key}")
        seen.add(key)
        by_request.setdefault(request_id, []).append(row)
        proposal = list(row.get("proposal_token_ids", ()))
        accepted_ids = list(row.get("accepted_draft_token_ids", ()))
        rejected_ids = list(row.get("rejected_draft_token_ids", ()))
        accepted = row.get("accepted_draft_tokens")
        rejected = row.get("rejected_draft_tokens")
        correction = list(row.get("target_correction_token_ids", ()))
        bonus = list(row.get("target_bonus_token_ids", ()))
        committed = list(row.get("committed_token_ids", ()))
        if accepted_ids != proposal[: len(accepted_ids)]:
            errors.append(f"{key}: accepted Draft tokens are not a proposal prefix")
        if rejected_ids != proposal[len(accepted_ids) :]:
            errors.append(f"{key}: rejected Draft tokens are not the proposal suffix")
        if accepted != len(accepted_ids) or rejected != len(rejected_ids):
            errors.append(f"{key}: accepted/rejected count differs from token evidence")
        if len(proposal) != len(accepted_ids) + len(rejected_ids):
            errors.append(f"{key}: proposal conservation failed")
        if correction and bonus:
            errors.append(f"{key}: correction and bonus are not mutually exclusive")
        if committed != accepted_ids + correction + bonus:
            errors.append(f"{key}: committed-token accounting mismatch")
        truncation = row.get("terminal_truncation_reason")
        if truncation is not None and truncation not in {"eos", "stop", "max_tokens"}:
            errors.append(f"{key}: invalid terminal truncation reason")
    for request_id, request_rows in by_request.items():
        ordered = sorted(request_rows, key=lambda row: row.get("round_id", -1))
        if [row.get("round_id") for row in ordered] != list(range(len(ordered))):
            errors.append(f"{request_id}: round IDs are not contiguous")
        versions = [row.get("prefix_version") for row in ordered]
        if versions != list(range(1, len(versions) + 1)):
            errors.append(f"{request_id}: proposal prefix versions are not contiguous")
    if not rows:
        errors.append("round accounting evidence is empty")
    return errors


def validate_verification_contracts(
    rounds: Sequence[Mapping[str, Any]],
    verification_rows: Sequence[Mapping[str, Any]],
    diagnostic_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    errors = []
    round_by_proposal = {str(row.get("proposal_id", "")): row for row in rounds}
    verify_by_proposal = {
        str(row.get("proposal_id", "")): row for row in verification_rows
    }
    diagnostics_by_proposal = {
        str(row.get("proposal_id", "")): row
        for row in diagnostic_rows
        if row.get("proposal_id")
    }
    for proposal_id, row in round_by_proposal.items():
        verify = verify_by_proposal.get(proposal_id)
        if verify is None:
            errors.append(f"{proposal_id}: verification timing evidence is missing")
            continue
        if list(verify.get("proposal_token_ids", ())) != list(
            row.get("proposal_token_ids", ())
        ):
            errors.append(f"{proposal_id}: verification proposal tokens differ")
        diagnostic = diagnostics_by_proposal.get(proposal_id)
        if diagnostic is None:
            # Older observer rows can be matched by request/round while still
            # requiring exact identity.
            diagnostic = next(
                (
                    item
                    for item in diagnostic_rows
                    if item.get("request_id") == row.get("request_id")
                    and item.get("round_id") == row.get("round_id")
                ),
                None,
            )
        if diagnostic is None:
            errors.append(f"{proposal_id}: Target input diagnostic is missing")
            continue
        proposal = list(row.get("proposal_token_ids", ()))
        target_input = list(diagnostic.get("target_input_token_ids", ()))
        positions = list(diagnostic.get("position_ids", ()))
        computed = diagnostic.get("physical_kv_num_computed_tokens")
        logical = diagnostic.get("logical_committed_prefix_count")
        pending = diagnostic.get("target_pending_input_token_id")
        if target_input != [pending, *proposal]:
            errors.append(f"{proposal_id}: verification input is not pending + proposal")
        if not isinstance(computed, int) or positions != list(
            range(computed, computed + len(proposal) + 1)
        ):
            errors.append(f"{proposal_id}: verification positions are not contiguous")
        if logical is not None and logical != computed + 1:
            errors.append(f"{proposal_id}: Target KV/pending logical invariant failed")
        prefix = list(diagnostic.get("committed_prefix_token_ids", ()))
        if prefix and pending != prefix[-1]:
            errors.append(f"{proposal_id}: pending token differs from logical prefix tail")
        if diagnostic.get("target_kv_contains_rejected_or_future_tokens") is not False:
            errors.append(f"{proposal_id}: rejected/future positions entered live Target KV")
    return errors


def validate_final_commit_sequence(
    run: Mapping[str, Any],
    manifest: Any,
    rounds: Sequence[Mapping[str, Any]],
    draft_rows: Sequence[Mapping[str, Any]],
    state_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    errors = []
    by_request: dict[str, list[Mapping[str, Any]]] = {}
    for row in rounds:
        by_request.setdefault(str(row.get("request_id", "")), []).append(row)
    tails: dict[str, list[int]] = {}
    for row in draft_rows:
        if row.get("operation") != "finish_tail" or row.get("success") is not True:
            continue
        result = row.get("result")
        if isinstance(result, Mapping):
            tails[str(row.get("request_id", ""))] = list(
                result.get("committed_token_ids", ())
            )
    outputs = _canonical_outputs(run)
    manifest_by_id = {row.request_id: row for row in manifest.requests}
    terminal_states = {
        str(row.get("request_id", "")): row
        for row in state_rows
        if row.get("destination_state") == "TERMINAL"
    }
    for request_id, output in outputs.items():
        generated = list(output[0])
        request = manifest_by_id.get(request_id)
        if request is None:
            errors.append(f"{request_id}: final output has no decode-ready request")
            continue
        committed = []
        for row in sorted(
            by_request.get(request_id, ()), key=lambda item: item.get("round_id", -1)
        ):
            committed.extend(row.get("committed_token_ids", ()))
        committed.extend(tails.get(request_id, ()))
        if generated != [request.bootstrap_token_id, *committed]:
            errors.append(
                f"{request_id}: bootstrap + round/tail commits != final generated sequence"
            )
        final_prefix = request.logical_committed_prefix_token_ids[:-1] + tuple(generated)
        if len(final_prefix) != output[-1]:
            errors.append(f"{request_id}: final logical prefix length differs")
        terminal_state = terminal_states.get(request_id)
        if terminal_state is None:
            errors.append(f"{request_id}: final TERMINAL state evidence is missing")
        elif (
            terminal_state.get("committed_prefix_length") != len(final_prefix)
            or terminal_state.get("committed_prefix_sha256")
            != token_prefix_hash(final_prefix)
        ):
            errors.append(f"{request_id}: final TERMINAL prefix state differs")
    return errors


def validate_draft_sync(
    draft_rows: Sequence[Mapping[str, Any]],
    rounds: Sequence[Mapping[str, Any]],
    state_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    errors = []
    commits = {
        (str(row.get("request_id", "")), row.get("round_id")): row for row in rounds
    }
    for row in draft_rows:
        if row.get("success") is not True:
            errors.append(f"{row.get('request_id')}: Draft work failed")
            continue
        if row.get("operation") != "commit_and_propose":
            continue
        result = row.get("result")
        if not isinstance(result, Mapping):
            errors.append(f"{row.get('request_id')}: Draft sync result is missing")
            continue
        key = (str(row.get("request_id", "")), result.get("round_id"))
        committed = commits.get(key)
        if committed is None:
            errors.append(f"{key}: Draft sync has no Target commit")
            continue
        if list(result.get("committed_token_ids", ())) != list(
            committed.get("committed_token_ids", ())
        ):
            errors.append(f"{key}: Draft synchronized different committed tokens")
        logical = result.get("logical_draft_kv_length")
        if not isinstance(logical, int) or logical <= 0:
            errors.append(f"{key}: Draft logical KV length is invalid")
        if result.get("rollback_length") != committed.get("rejected_draft_tokens"):
            errors.append(f"{key}: Draft rollback length differs from rejection count")
        sync_complete = result.get("draft_sync_complete_ns")
        next_interval = result.get("draft_gpu_interval")
        if not isinstance(sync_complete, int):
            errors.append(f"{key}: Draft sync completion timestamp is missing")
        elif (
            isinstance(next_interval, Mapping)
            and isinstance(next_interval.get("host_start_ns"), int)
            and next_interval["host_start_ns"] < sync_complete
        ):
            errors.append(f"{key}: next Draft started before correction/bonus sync")
        proposal = result.get("proposal")
        if isinstance(proposal, Mapping):
            proposal_id = str(proposal.get("proposal_id", ""))
            transitions = {
                str(event.get("destination_state")): event
                for event in state_rows
                if event.get("request_id") == key[0]
                and event.get("proposal_id") == proposal_id
            }
            ready = transitions.get("DRAFT_READY")
            drafting = transitions.get("DRAFTING")
            if ready is None or ready.get("timestamp_ns") != sync_complete:
                errors.append(f"{key}: DRAFT_READY is not aligned with Draft sync")
            if (
                drafting is None
                or not isinstance(next_interval, Mapping)
                or drafting.get("timestamp_ns") != next_interval.get("host_start_ns")
            ):
                errors.append(f"{key}: DRAFTING is not aligned with real Draft work")
        elif result.get("target_tail") is True and result.get("terminal") is not True:
            ready_ns = result.get("target_tail_ready_ns")
            tails = [
                event
                for event in state_rows
                if event.get("request_id") == key[0]
                and event.get("destination_state") == "TARGET_TAIL_READY"
                and event.get("timestamp_ns") == ready_ns
            ]
            if not tails:
                errors.append(f"{key}: Target tail readiness is not state-aligned")
    return errors


def validate_target_blind_isolation(
    run: Mapping[str, Any],
    draft_rows: Sequence[Mapping[str, Any]],
    rounds: Sequence[Mapping[str, Any]],
) -> list[str]:
    errors = []
    semantics = run.get("runtime_semantics", {})
    if not isinstance(semantics, Mapping) or semantics.get("target_blind_draft") is not True:
        errors.append("run lacks affirmative target-blind Draft metadata")
    forbidden = {
        "target_logits",
        "target_only_reference_output",
        "acceptance_outcome",
        "future_target_token_ids",
        "oracle_labels",
    }
    for row in draft_rows:
        if forbidden & set(_nested_keys(row)):
            errors.append(f"{row.get('request_id')}: Draft work contains forbidden Target data")
    verify_end = {
        (str(row.get("request_id", "")), row.get("round_id")): row.get("commit_end_ns")
        for row in rounds
    }
    for row in draft_rows:
        if row.get("operation") != "commit_and_propose":
            continue
        result = row.get("result", {})
        if not isinstance(result, Mapping):
            continue
        key = (str(row.get("request_id", "")), result.get("round_id"))
        ended = verify_end.get(key)
        started = row.get("start_ns")
        if isinstance(ended, int) and isinstance(started, int) and started < ended:
            errors.append(f"{key}: Draft sync/propose began before Target commit completed")
    return errors


def validate_overlap_witness(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    for row in rows:
        if (
            isinstance(row.get("overlap_duration_ns"), int)
            and row["overlap_duration_ns"] > 0
            and row.get("request_sets_disjoint") is True
            and row.get("draft_physical_gpu_ids") == [0]
            and row.get("target_physical_gpu_ids") == [1, 2]
            and row.get("draft_cuda_events") is True
            and len(row.get("target_rank_intervals", ())) == 2
        ):
            return []
    return ["no positive cross-request GPU Draft/Target overlap witness exists"]


def validate_measurement_boundary(
    manifest: Any,
    lifecycle_rows: Sequence[Mapping[str, Any]],
    draft_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Reject any proposal work that began before global decode readiness."""

    errors = []
    boundary = manifest.measurement_start_ns
    created = [
        row for row in lifecycle_rows if row.get("lifecycle_state") == "CREATED"
    ]
    for row in created:
        start = row.get("draft_start_ns")
        if not isinstance(start, int) or start < boundary:
            errors.append(
                f"{row.get('proposal_id')}: proposal began before measurement_start"
            )
    for row in draft_rows:
        result = row.get("result")
        proposal = result.get("proposal") if isinstance(result, Mapping) else None
        if not isinstance(proposal, Mapping):
            continue
        start = proposal.get("draft_start_ns")
        if not isinstance(start, int) or start < boundary:
            errors.append(
                f"{proposal.get('proposal_id')}: Draft work predates measurement_start"
            )
    if not created:
        errors.append("proposal creation evidence is empty")
    return errors


def validate_cleanup(run: Mapping[str, Any]) -> list[str]:
    errors = list(validate_lifecycle_artifact(run))
    draft = run.get("draft_shutdown_result")
    if not isinstance(draft, Mapping) or draft.get("valid") is not True:
        errors.append("Draft service/socket cleanup is invalid")
    elif (
        draft.get("required") is not True
        or not isinstance(draft.get("socket_path"), str)
        or draft.get("socket_exists_after_cleanup") is not False
        or draft.get("alive_after_cleanup") is not False
    ):
        errors.append("Draft cleanup lacks explicit PID/socket absence evidence")
    return errors


def _manifest_identity(manifest: Any) -> tuple[Any, ...]:
    return (
        manifest.provider_kind,
        manifest.specrhythm_git_commit,
        manifest.vllm_version,
        manifest.vllm_commit,
        manifest.vllm_patch_stack_sha256,
        manifest.target_model_path,
        manifest.target_model_revision,
        manifest.draft_model_path,
        manifest.draft_model_revision,
        manifest.tokenizer_revision,
        manifest.workload_sha256,
        manifest.sampling_configuration_json,
        manifest.batch_invariant_configuration_json,
        manifest.target_physical_gpu_ids,
        manifest.draft_physical_gpu_ids,
        manifest.target_tensor_parallel_size,
        manifest.draft_tensor_parallel_size,
        manifest.kv_connector_handoff,
        tuple(
            (
                row.request_id,
                row.prompt_token_ids_sha256,
                row.bootstrap_token_id,
                row.logical_committed_prefix_count,
                row.logical_committed_prefix_sha256,
                row.logical_committed_prefix_token_ids,
                row.target_materialized_kv_token_count,
                row.target_pending_input_token_id,
                row.draft_materialized_kv_token_count,
                row.prefix_version,
                row.next_round_id,
            )
            for row in manifest.requests
        ),
    )


def _canonical_outputs(value: Mapping[str, Any]) -> dict[str, tuple[Any, ...]]:
    rows = value.get("decode_only_outputs", value.get("outputs", ()))
    result = {}
    for row in rows:
        request_id = str(row.get("request_id", ""))
        if not request_id:
            raise ValueError("decode-only output has an empty request ID")
        if request_id in result:
            raise ValueError(f"duplicate decode-only output request: {request_id}")
        generated = tuple(int(item) for item in row.get("generated_token_ids", ()))
        result[request_id] = (
            generated,
            row.get("finish_reason"),
            row.get("eos_token_id", row.get("stop_reason")),
            row.get("max_token_termination", row.get("finish_reason") == "length"),
            row.get("completed", row.get("finish_reason") is not None),
            row.get("final_logical_length"),
        )
    if not result:
        raise ValueError("decode-only output set is empty")
    return result


def _first_divergence(left: Sequence[int], right: Sequence[int]) -> dict[str, Any]:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return {
                "first_divergence_position": index,
                "target_token_id": left_token,
                "consumer_token_id": right_token,
            }
    index = min(len(left), len(right))
    return {
        "first_divergence_position": index,
        "target_token_id": left[index] if index < len(left) else None,
        "consumer_token_id": right[index] if index < len(right) else None,
    }


def _round_prefix_versions(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], Any]:
    return {
        (str(row.get("request_id", "")), int(row.get("round_id", -1))): row.get(
            "prefix_version"
        )
        for row in rows
    }


def _nested_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _nested_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _nested_keys(item)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _markdown(value: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 4B.1 decode-only Dual-Batch correctness",
        "",
        f"- Validation: **{'PASS' if value['valid'] else 'FAIL'}**",
        f"- Outcome: **{value['outcome']}**",
        f"- Exact Target/Serial/Dual triangle: `{value['triangle']['valid']}`",
        f"- Decode-ready identity equal: `{value['decode_ready_manifest_identity_equal']}`",
        f"- Input artifacts immutable: `{value['input_artifacts_immutable']}`",
        "",
        str(value["claim_boundary"]),
        "",
    ]
    if value["errors"]:
        lines.extend(["## Errors", "", *[f"- {item}" for item in value["errors"]], ""])
    return "\n".join(lines)
