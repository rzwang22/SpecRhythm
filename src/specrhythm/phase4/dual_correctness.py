"""Read-only Phase-4B.1 decode-only Dual-Batch correctness authority.

The validator deliberately compares logical serving artifacts, not Phase-3 HF
trajectories and not cross-request JSONL write order. Every failure is explicit;
the only successful classification is outcome A.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from specrhythm.phase4.admissibility import (
    ExecutionPhase,
    ScheduledOperation,
    SchedulerRequestState,
    validate_admissibility_events,
)
from specrhythm.phase4.correctness_validation import compare_round_semantics
from specrhythm.phase4.decode_ready import load_decode_ready_manifest
from specrhythm.phase4.manifest import atomic_write_json, sha256_file
from specrhythm.phase4.process_lifecycle import validate_lifecycle_artifact
from specrhythm.phase4.resident_setup import validate_setup_ready
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.transport import CheckpointJsonl
from specrhythm.phase4.vllm_dual import validate_target_rank_identity

VALIDATION_SCHEMA = "specrhythm.phase4b1-dual-correctness-validation.v2"
CONTROLLED_SCHEMA = "specrhythm.phase4b1-controlled-gate-validation.v1"
OVERLAP_DIAGNOSTIC_SCHEMA = "specrhythm.phase4b1-overlap-timing-diagnostic.v1"
LEGACY_GATE1_COMMIT = "3ee1c3ec4007d3e835bc7d7f385d2d3b5c3c3e8a"

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
    overlap_requirement: str = "required",
    legacy_source_commit: Optional[str] = None,
) -> dict[str, Any]:
    """Validate one Target, one Serial, and one or more immutable Dual runs."""

    if not dual_paths:
        raise ValueError("at least one decode-only Dual run is required")
    if overlap_requirement not in {"required", "separate-gate"}:
        raise ValueError("unknown overlap requirement")
    if legacy_source_commit not in {None, LEGACY_GATE1_COMMIT}:
        raise ValueError("unsupported legacy read-only revalidation source")
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
    if legacy_source_commit is not None:
        preserved_root = Path(
            os.path.commonpath([str(path.resolve()) for path in inputs])
        )
        destinations = [output_path, *([markdown_path] if markdown_path else [])]
        if any(
            destination.resolve() == preserved_root
            or preserved_root in destination.resolve().parents
            for destination in destinations
        ):
            raise ValueError(
                "legacy revalidation outputs must be outside the preserved artifact root"
            )
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

    for label, run in [("Target", target), ("Serial", serial)]:
        if run.get("valid") is not True:
            errors.append(f"{label} run is not valid")

    identities = [_manifest_identity(item) for item in manifests]
    manifest_identity_equal = all(item == identities[0] for item in identities[1:])
    if not manifest_identity_equal:
        errors.append("Target/Serial/Dual decode-ready logical manifest identities differ")
    if legacy_source_commit is not None and any(
        item.specrhythm_git_commit != legacy_source_commit for item in manifests[2:]
    ):
        errors.append("Dual decode-ready provenance differs from legacy source commit")
    for label, lifecycle in zip(("Target", "Serial"), baseline_lifecycles):
        errors.extend(f"{label} cleanup: {item}" for item in validate_cleanup(lifecycle))

    consumers = [target, serial, *duals]
    triangle = compare_consumer_triangle(consumers)
    if not triangle["valid"]:
        errors.extend(triangle["errors"])

    component_results = []
    overlap_evaluations = []
    for index in range(len(duals)):
        worker_rows = duals[index].get("worker_ranks", ())
        if not isinstance(worker_rows, list):
            worker_rows = []
        overlap_evaluation = evaluate_overlap_witness(
            overlaps[index],
            authoritative_worker_rows=worker_rows,
            expected_target_physical_gpu_ids=manifests[
                index + 2
            ].target_physical_gpu_ids,
            allow_legacy_device_supersession=legacy_source_commit is not None,
        )
        overlap_evaluations.append(overlap_evaluation)
        values = {
            "state_machine": validate_request_state_events(states[index]),
            "proposal_lifecycle": validate_proposal_lifecycle_events(
                lifecycles[index]
            ),
            "scheduler": validate_scheduler_cycles(
                schedulers[index],
                proposal_lifecycle_rows=lifecycles[index],
                state_rows=states[index],
                draft_rows=draft_work[index],
            ),
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
            "runner_invariants": validate_dual_runner_evidence(
                duals[index],
                manifests[index + 2],
                dual_manifest_paths[index],
            ),
            "overlap": overlap_evaluation["errors"],
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
            if name == "overlap":
                continue
            errors.extend(
                f"Dual-{index + 1} {name}: {item}" for item in component_errors
            )
        recomputed = {
            name: {"valid": not value, "errors": value}
            for name, value in values.items()
        }
        authority = classify_embedded_dual_verdict(
            duals[index],
            scheduler_rows=schedulers[index],
            scheduler_errors=values["scheduler"],
            overlap_requirement=overlap_requirement,
            overlap_evaluation=overlap_evaluation,
            legacy_source_commit=legacy_source_commit,
        )
        authority["recomputed_components"] = {
            name: value["valid"] for name, value in recomputed.items()
        }
        authority["recomputed_semantic_valid"] = all(
            value["valid"]
            for name, value in recomputed.items()
            if name != "overlap"
        )
        if authority["remaining_embedded_errors"]:
            errors.extend(
                f"Dual-{index + 1} embedded verdict: {item}"
                for item in authority["remaining_embedded_errors"]
            )
        component_results.append(
            {
                **recomputed,
                "overlap_evidence": overlap_evaluation,
                "embedded_verdict_authority": authority,
                "recomputed_semantic_valid": authority[
                    "recomputed_semantic_valid"
                ],
            }
        )

    if overlap_requirement == "required" and not any(
        value["hardware_placement_qualified_overlap_observed"]
        for value in overlap_evaluations
    ):
        errors.append(
            "no Dual run has a hardware-qualified positive cross-request GPU overlap witness"
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
    temporal_overlap_observed = [
        value["temporal_overlap_observed"] for value in overlap_evaluations
    ]
    qualified_overlap_observed = [
        value["hardware_placement_qualified_overlap_observed"]
        for value in overlap_evaluations
    ]
    result = {
        "schema_version": VALIDATION_SCHEMA,
        "valid": valid,
        "outcome": "A" if valid else "FAIL",
        "errors": errors,
        "performance_result": False,
        "correctness_scope": "real-decode-only-dual-batch",
        "gate_profile": (
            "controlled-correctness"
            if overlap_requirement == "separate-gate"
            else "correctness-and-overlap-existence"
        ),
        "overlap_requirement": overlap_requirement,
        "legacy_read_only_revalidation": {
            "enabled": legacy_source_commit is not None,
            "source_commit": legacy_source_commit,
        },
        "overlap_gate": {
            "required_for_validation": overlap_requirement == "required",
            "valid": any(qualified_overlap_observed),
            "temporal_observed_per_run": temporal_overlap_observed,
            "hardware_qualified_observed_per_run": qualified_overlap_observed,
            "at_least_one_temporal_overlap": any(temporal_overlap_observed),
            "at_least_one_hardware_qualified_overlap": any(
                qualified_overlap_observed
            ),
            "claim_permitted": (
                overlap_requirement == "required"
                and any(qualified_overlap_observed)
            ),
        },
        "decode_ready_manifest_identity_equal": manifest_identity_equal,
        "triangle": triangle,
        "dual_runs": component_results,
        "repeat_comparisons": repeats,
        "input_artifacts_immutable": immutable,
        "input_sha256_before": before,
        "input_sha256_after": after,
        "claim_boundary": (
            "Controlled semantic correctness only; real cross-request GPU overlap "
            "is a separate required gate and is not established by this result. No "
            "TPOT, throughput, goodput, SLO, speedup, or overlap-benefit claim."
            if overlap_requirement == "separate-gate"
            else "Correctness and real cross-request overlap existence only; no TPOT, "
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


def validate_scheduler_cycles(
    rows: Sequence[Mapping[str, Any]],
    *,
    proposal_lifecycle_rows: Sequence[Mapping[str, Any]] = (),
    state_rows: Sequence[Mapping[str, Any]] = (),
    draft_rows: Sequence[Mapping[str, Any]] = (),
) -> list[str]:
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
            phase = row.get("execution_phase")
            waiting = state in {
                SchedulerRequestState.WAITING_DRAFT.value,
                SchedulerRequestState.DRAFTING.value,
            }
            timed_waiting = waiting and phase == ExecutionPhase.TIMED_DECODE.value
            setup_prefill = (
                waiting
                and phase == ExecutionPhase.SETUP_PREFILL.value
                and scheduled
                and operation == ScheduledOperation.PREFILL.value
            )
            if timed_waiting and scheduled:
                errors.append(f"{request_id}: waiting request consumed Target budget")
            if timed_waiting and row.get(
                "target_input_token_positions"
            ):
                errors.append(f"{request_id}: waiting request owns Target positions")
            if waiting and phase == ExecutionPhase.SETUP_PREFILL.value and scheduled:
                if not setup_prefill:
                    errors.append(f"{request_id}: invalid setup-prefill advancement")
            if (
                waiting
                and phase == ExecutionPhase.SETUP_PREFILL.value
                and row.get("target_input_token_positions")
                and not setup_prefill
            ):
                errors.append(f"{request_id}: setup-prefill positions lack legal prefill")
            if state == SchedulerRequestState.TERMINAL.value:
                seen_terminal.add(request_id)
                if scheduled:
                    errors.append(f"{request_id}: terminal request was scheduled")
            if request_id in seen_terminal and scheduled:
                errors.append(f"{request_id}: request was scheduled after terminal evidence")
            if (
                operation == ScheduledOperation.VERIFY.value
                and not row.get("proposal_valid")
            ):
                errors.append(f"{request_id}: Target verification lacks matching proposal")
            if (
                scheduled
                and phase == ExecutionPhase.TIMED_DECODE.value
                and operation
                not in {
                    ScheduledOperation.VERIFY.value,
                    ScheduledOperation.TARGET_TAIL.value,
                }
            ):
                errors.append(f"{request_id}: unexplained live Target advancement")
            if operation == ScheduledOperation.TARGET_TAIL.value:
                tail_errors = _validate_target_tail_decision(
                    row,
                    cycle,
                    proposal_lifecycle_rows=proposal_lifecycle_rows,
                    state_rows=state_rows,
                    draft_rows=draft_rows,
                )
                if tail_errors:
                    errors.append(
                        f"{request_id}: legal Target tail contract is incomplete: "
                        + ", ".join(tail_errors)
                    )
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


def _validate_target_tail_decision(
    row: Mapping[str, Any],
    cycle: Mapping[str, Any],
    *,
    proposal_lifecycle_rows: Sequence[Mapping[str, Any]],
    state_rows: Sequence[Mapping[str, Any]],
    draft_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Prove a proposal-free tail while allowing consumed proposal history."""

    errors = []
    request_id = str(row.get("request_id", ""))
    if row.get("execution_phase") != ExecutionPhase.TIMED_DECODE.value:
        errors.append("not timed decode")
    if row.get("scheduled") is not True:
        errors.append("not scheduled")
    if row.get("admissible") is not True:
        errors.append("not admissible")
    if row.get("specrhythm_state") != SchedulerRequestState.TARGET_TAIL_READY.value:
        errors.append("state is not TARGET_TAIL_READY")
    if row.get("proposal_valid") is not False:
        errors.append("proposal is still valid")
    if row.get("spec_token_ids"):
        errors.append("speculative tokens are non-empty")
    if len(row.get("target_input_token_positions", ())) != 1:
        errors.append("Target advancement is not exactly one position")

    proposal_present = row.get("proposal_present") is True
    explicit_live = row.get("live_proposal_present")
    if explicit_live is True:
        errors.append("a live proposal remains")
    consumed = _consumed_proposal_before_tail(
        row, cycle, proposal_lifecycle_rows
    )
    if proposal_present and not consumed:
        errors.append("retained proposal is not proven consumed")
    if (
        proposal_present
        and "proposal_consumed" in row
        and row.get("proposal_consumed") is not True
    ):
        errors.append("retained proposal is marked unconsumed")

    ready_ns = _tail_ready_timestamp(row, draft_rows)
    if ready_ns is None:
        errors.append("Draft target-tail readiness is missing")
    else:
        ready_states = [
            event
            for event in state_rows
            if event.get("request_id") == request_id
            and event.get("destination_state")
            == SchedulerRequestState.TARGET_TAIL_READY.value
            and event.get("timestamp_ns") == ready_ns
        ]
        if not ready_states:
            errors.append("Draft readiness is not state-aligned")
        if not _terminal_tail_state_sequence(request_id, ready_ns, state_rows):
            errors.append("one-token tail lacks ordered terminal state evidence")
    return errors


def _consumed_proposal_before_tail(
    row: Mapping[str, Any],
    cycle: Mapping[str, Any],
    lifecycle_rows: Sequence[Mapping[str, Any]],
) -> bool:
    request_id = str(row.get("request_id", ""))
    tail_boundary = cycle.get("poll_end_ns", cycle.get("poll_start_ns"))
    candidates = [
        event
        for event in lifecycle_rows
        if event.get("request_id") == request_id
        and event.get("lifecycle_state") == "CONSUMED"
        and event.get("round_id") == row.get("round_id")
        and event.get("prefix_version") == row.get("prefix_version")
    ]
    if not candidates:
        return False
    if not isinstance(tail_boundary, int):
        return False
    return any(
        isinstance(event.get("timestamp_ns"), int)
        and event["timestamp_ns"] <= tail_boundary
        for event in candidates
    )


def _tail_ready_timestamp(
    row: Mapping[str, Any], draft_rows: Sequence[Mapping[str, Any]]
) -> Optional[int]:
    request_id = str(row.get("request_id", ""))
    explicit = row.get("target_tail_ready_timestamp_ns")
    candidates = []
    for draft in draft_rows:
        result = draft.get("result")
        if draft.get("request_id") != request_id or not isinstance(result, Mapping):
            continue
        if (
            result.get("target_tail") is True
            and result.get("proposal") is None
            and result.get("terminal") is not True
            and isinstance(result.get("target_tail_ready_ns"), int)
        ):
            candidates.append(result["target_tail_ready_ns"])
    if isinstance(explicit, int):
        if row.get("target_tail_ready") is not True or explicit not in candidates:
            return None
        return explicit
    return candidates[-1] if len(candidates) == 1 else None


def _terminal_tail_state_sequence(
    request_id: str,
    ready_ns: int,
    state_rows: Sequence[Mapping[str, Any]],
) -> bool:
    required = (
        SchedulerRequestState.TARGET_TAIL_READY.value,
        "VERIFYING",
        "COMMITTING",
        SchedulerRequestState.TERMINAL.value,
    )
    transitions = [
        event
        for event in state_rows
        if event.get("request_id") == request_id
        and isinstance(event.get("timestamp_ns"), int)
        and event["timestamp_ns"] >= ready_ns
        and event.get("destination_state") in required
    ]
    transitions.sort(key=lambda event: event["timestamp_ns"])
    observed = tuple(event.get("destination_state") for event in transitions)
    return observed[: len(required)] == required


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


def validate_dual_runner_evidence(
    run: Mapping[str, Any],
    manifest: Any,
    manifest_path: Path,
) -> list[str]:
    """Recompute runner-only invariants instead of trusting run.valid."""

    errors = []
    expected_ids = [row.request_id for row in manifest.requests]
    workers = run.get("worker_ranks")
    if not isinstance(workers, list):
        errors.append("authoritative Target worker ranks are missing")
        workers = []
    errors.extend(
        _validate_authoritative_worker_rows(
            workers,
            manifest.target_tensor_parallel_size,
            manifest.target_physical_gpu_ids,
        )
    )

    identity = run.get("request_identity")
    if not isinstance(identity, Mapping):
        errors.append("request identity/plugin binding evidence is missing")
    else:
        if identity.get("mapping_source") != "unique frozen prompt_token_ids":
            errors.append("request identity did not use frozen prompt tokens")
        if identity.get("suffix_parsing") is not False:
            errors.append("opaque internal request IDs were parsed")
        bindings = identity.get("bindings")
        if not isinstance(bindings, list) or any(
            not isinstance(row, Mapping) for row in bindings
        ):
            errors.append("request identity bindings are missing or malformed")
        else:
            internal = [str(row.get("internal_request_id", "")) for row in bindings]
            stable = [str(row.get("request_id", "")) for row in bindings]
            if (
                any(not item for item in internal)
                or len(internal) != len(set(internal))
                or len(stable) != len(set(stable))
                or set(stable) != set(expected_ids)
                or identity.get("bound_request_count") != len(bindings)
            ):
                errors.append("request identity bindings do not cover the cohort exactly")

    setup_ready = run.get("global_setup_ready")
    if not isinstance(setup_ready, Mapping):
        errors.append("global setup-ready evidence is missing")
    else:
        errors.extend(
            f"setup-ready: {item}"
            for item in validate_setup_ready(
                setup_ready,
                manifest_path=manifest_path,
                consumer="dual-batch",
                expected_request_ids=expected_ids,
            )
        )

    shutdown = run.get("draft_shutdown")
    if not isinstance(shutdown, Mapping):
        errors.append("Draft service shutdown evidence is missing")
    elif (
        shutdown.get("shutdown") is not True
        or shutdown.get("request_count") != len(expected_ids)
        or shutdown.get("failures") != {}
        or shutdown.get("inflight_request_ids") != []
        or shutdown.get("work_queue_depth") != 0
    ):
        errors.append("Draft service shutdown evidence is incomplete")

    outputs = run.get("decode_only_outputs", run.get("outputs", ()))
    output_ids = [
        str(row.get("request_id", ""))
        for row in outputs
        if isinstance(row, Mapping)
    ] if isinstance(outputs, list) else []
    if (
        run.get("request_count") != len(expected_ids)
        or len(output_ids) != len(set(output_ids))
        or set(output_ids) != set(expected_ids)
    ):
        errors.append("runner output/request cohort identity is incomplete")
    return list(dict.fromkeys(errors))


def classify_embedded_dual_verdict(
    run: Mapping[str, Any],
    *,
    scheduler_rows: Sequence[Mapping[str, Any]],
    scheduler_errors: Sequence[str],
    overlap_requirement: str,
    overlap_evaluation: Mapping[str, Any],
    legacy_source_commit: Optional[str],
) -> dict[str, Any]:
    """Classify every historical embedded error using structural raw evidence."""

    embedded_valid = run.get("valid") is True
    raw_errors = run.get("errors", ())
    embedded_errors = [str(item) for item in raw_errors] if isinstance(
        raw_errors, list
    ) else ["embedded run errors are missing or malformed"]
    supersedable = {}
    if legacy_source_commit is not None and not scheduler_errors:
        supersedable.update(_legacy_scheduler_error_proofs(scheduler_rows))
    overlap_error = "no positive cross-request GPU Draft/Target overlap witness exists"
    if legacy_source_commit is not None:
        if overlap_requirement == "separate-gate":
            supersedable[overlap_error] = "excluded-by-controlled-semantic-gate"
        elif overlap_evaluation.get(
            "hardware_placement_qualified_overlap_observed"
        ) is True:
            supersedable[overlap_error] = (
                "raw-temporal-witness-plus-authoritative-worker-topology"
            )

    superseded = []
    remaining = []
    if embedded_valid and embedded_errors:
        remaining.append("embedded valid=true artifact contains errors")
    elif not embedded_valid and not embedded_errors:
        remaining.append("embedded valid=false artifact has no classified errors")
    for error in embedded_errors:
        proof = supersedable.get(error)
        if not embedded_valid and proof is not None:
            superseded.append({"error": error, "proof": proof})
        elif error:
            remaining.append(error)
    if legacy_source_commit is None and not embedded_valid:
        remaining = embedded_errors or ["embedded Dual run is not valid"]
        superseded = []
    return {
        "embedded_run_valid": embedded_valid,
        "embedded_run_errors": embedded_errors,
        "superseded_legacy_errors": superseded,
        "remaining_embedded_errors": list(dict.fromkeys(remaining)),
        "legacy_source_commit": legacy_source_commit,
    }


def _legacy_scheduler_error_proofs(
    scheduler_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    proofs = {}
    for cycle in scheduler_rows:
        for row in cycle.get("request_admissibility", ()):
            request_id = str(row.get("request_id", ""))
            legal_setup = (
                row.get("specrhythm_state")
                in {
                    SchedulerRequestState.WAITING_DRAFT.value,
                    SchedulerRequestState.DRAFTING.value,
                }
                and row.get("execution_phase") == ExecutionPhase.SETUP_PREFILL.value
                and row.get("scheduled") is True
                and row.get("scheduled_operation")
                == ScheduledOperation.PREFILL.value
            )
            if legal_setup:
                proofs[f"{request_id}: waiting request consumed Target budget"] = (
                    "current-phase-aware-scheduler-validation"
                )
                if row.get("target_input_token_positions"):
                    proofs[f"{request_id}: waiting request owns Target positions"] = (
                        "current-phase-aware-scheduler-validation"
                    )
            if (
                row.get("scheduled_operation")
                == ScheduledOperation.TARGET_TAIL.value
                and row.get("specrhythm_state")
                == SchedulerRequestState.TARGET_TAIL_READY.value
            ):
                proofs[f"{request_id}: unexplained live Target advancement"] = (
                    "authoritative-scheduled-operation-enum"
                )
                proofs[f"{request_id}: legal Target tail contract is incomplete"] = (
                    "consumed-lifecycle-plus-draft-readiness-plus-terminal-state-proof"
                )
    return proofs


def evaluate_overlap_witness(
    rows: Sequence[Mapping[str, Any]],
    *,
    authoritative_worker_rows: Sequence[Mapping[str, Any]] = (),
    expected_target_physical_gpu_ids: Sequence[int] = (1, 2),
    allow_legacy_device_supersession: bool = False,
) -> dict[str, Any]:
    """Separate temporal overlap from hardware-placement qualification."""

    expected = sorted(int(item) for item in expected_target_physical_gpu_ids)
    workers = list(authoritative_worker_rows)
    worker_errors = _validate_authoritative_worker_rows(
        workers, len(expected), expected
    ) if workers else ["authoritative Target worker evidence is missing"]
    temporal = []
    qualified = []
    invalid_attribution = []
    for row in rows:
        duration = row.get("overlap_duration_ns")
        if not (
            isinstance(duration, int)
            and duration > 0
            and row.get("request_sets_disjoint") is True
            and row.get("draft_physical_gpu_ids") == [0]
            and row.get("draft_cuda_events") is True
        ):
            continue
        temporal.append(dict(row))
        rank_rows = row.get("target_rank_intervals", ())
        direct_errors = validate_target_rank_identity(
            rank_rows if isinstance(rank_rows, list) else (),
            len(expected),
            workers,
        )
        direct_placement = (
            row.get("target_physical_gpu_ids") == expected and not direct_errors
        )
        if direct_placement:
            qualified.append(
                {
                    "row": dict(row),
                    "attribution_source": "verification-rank-events",
                    "historical_event_instrumentation_invalid": False,
                }
            )
            continue
        legacy_signature = _legacy_aliased_verify_identity(
            rank_rows if isinstance(rank_rows, list) else (),
            len(expected),
            workers,
        ) and row.get("target_physical_gpu_ids") == sorted(
            {item.get("physical_gpu_id") for item in rank_rows}
        )
        if legacy_signature:
            invalid_attribution.append(dict(row))
        if (
            allow_legacy_device_supersession
            and legacy_signature
            and not worker_errors
        ):
            qualified.append(
                {
                    "row": dict(row),
                    "attribution_source": "authoritative-worker-ranks-supersede-legacy-event",
                    "historical_event_instrumentation_invalid": True,
                }
            )
    return {
        "temporal_overlap_observed": bool(temporal),
        "hardware_placement_qualified_overlap_observed": bool(qualified),
        "temporal_witness_count": len(temporal),
        "qualified_witness_count": len(qualified),
        "historical_event_instrumentation_invalid": bool(invalid_attribution),
        "authoritative_worker_topology_valid": not worker_errors,
        "authoritative_worker_errors": worker_errors,
        "qualified_witnesses": qualified,
        "errors": (
            []
            if qualified
            else ["no hardware-qualified positive cross-request GPU overlap witness exists"]
        ),
    }


def validate_overlap_witness(
    rows: Sequence[Mapping[str, Any]],
    *,
    authoritative_worker_rows: Sequence[Mapping[str, Any]] = (),
    expected_target_physical_gpu_ids: Sequence[int] = (1, 2),
    allow_legacy_device_supersession: bool = False,
) -> list[str]:
    return evaluate_overlap_witness(
        rows,
        authoritative_worker_rows=authoritative_worker_rows,
        expected_target_physical_gpu_ids=expected_target_physical_gpu_ids,
        allow_legacy_device_supersession=allow_legacy_device_supersession,
    )["errors"]


def _legacy_aliased_verify_identity(
    rows: Sequence[Mapping[str, Any]],
    tensor_parallel_size: int,
    authoritative_worker_rows: Sequence[Mapping[str, Any]],
) -> bool:
    aliased_pair = {
        (row.get("physical_gpu_id"), _canonical_gpu_uuid(row.get("gpu_uuid")))
        for row in rows
    }
    worker_pairs = {
        (
            row.get("physical_gpu_id"),
            _canonical_gpu_uuid(row.get("gpu_uuid")),
        )
        for row in authoritative_worker_rows
    }
    return bool(
        len(rows) == tensor_parallel_size
        and {row.get("tp_rank") for row in rows} == set(range(tensor_parallel_size))
        and len({row.get("global_rank") for row in rows}) == tensor_parallel_size
        and len({row.get("physical_gpu_id") for row in rows}) == 1
        and len({row.get("gpu_uuid") for row in rows}) == 1
        and len(aliased_pair) == 1
        and aliased_pair <= worker_pairs
        and all(
            row.get("cuda_events") is True
            and row.get("cuda_synchronized") is True
            for row in rows
        )
    )


def _canonical_gpu_uuid(value: Any) -> str:
    result = str(value or "").strip().lower()
    return result[4:] if result.startswith("gpu-") else result


def _validate_authoritative_worker_rows(
    rows: Sequence[Mapping[str, Any]],
    tensor_parallel_size: int,
    expected_physical_gpu_ids: Sequence[int],
) -> list[str]:
    errors = []
    if len(rows) != tensor_parallel_size:
        errors.append("worker rank count differs from Target TP size")
    if {row.get("global_rank") for row in rows} != set(range(tensor_parallel_size)):
        errors.append("worker global ranks are incomplete")
    if {row.get("physical_gpu_id") for row in rows} != set(
        expected_physical_gpu_ids
    ):
        errors.append("worker physical GPU set differs from decode-ready placement")
    uuids = [str(row.get("gpu_uuid", "")) for row in rows]
    if any(not item for item in uuids) or len(set(uuids)) != len(rows):
        errors.append("worker GPU UUIDs are missing or aliased")
    for row in rows:
        if row.get("world_size") != tensor_parallel_size:
            errors.append("worker world size differs from Target TP size")
        if row.get("parameter_count", 0) <= 0 or row.get("parameter_bytes", 0) <= 0:
            errors.append("worker rank lacks model parameters")
        if row.get("allocated_memory_bytes", 0) <= 0:
            errors.append("worker rank lacks allocated CUDA memory")
        if row.get("all_parameters_on_expected_device") is not True:
            errors.append("worker parameters are not on the active CUDA device")
    return list(dict.fromkeys(errors))


def diagnose_overlap_timing(
    draft_rows: Sequence[Mapping[str, Any]],
    verification_rows: Sequence[Mapping[str, Any]],
    recorded_overlap_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Find the exact nearest disjoint-request Draft/Verify interval pair."""

    errors = []
    drafts = []
    for row in draft_rows:
        result = row.get("result")
        proposal = result.get("proposal") if isinstance(result, Mapping) else None
        interval = (
            result.get("draft_gpu_interval")
            if isinstance(result, Mapping)
            else None
        )
        if not isinstance(proposal, Mapping) or not isinstance(interval, Mapping):
            continue
        start = interval.get("host_start_ns")
        end = interval.get("host_end_ns")
        if not isinstance(start, int) or not isinstance(end, int) or start >= end:
            errors.append(f"{row.get('request_id')}: invalid Draft host interval")
            continue
        drafts.append(
            {
                "request_id": str(row.get("request_id", "")),
                "proposal_id": str(proposal.get("proposal_id", "")),
                "round_id": proposal.get("round_id"),
                "host_start_ns": start,
                "host_end_ns": end,
                "physical_gpu_id": interval.get("physical_gpu_id"),
            }
        )
    verifications = []
    seen_batches = set()
    for row in verification_rows:
        batch_id = row.get("verify_microbatch_id")
        if batch_id in seen_batches:
            continue
        seen_batches.add(batch_id)
        start = row.get("verify_host_start_ns")
        end = row.get("verify_host_end_ns")
        if not isinstance(start, int) or not isinstance(end, int) or start >= end:
            errors.append(f"{batch_id}: invalid Verify host interval")
            continue
        verifications.append(
            {
                "verify_microbatch_id": batch_id,
                "request_ids": [
                    str(item) for item in row.get("verify_request_ids", ())
                ],
                "host_start_ns": start,
                "host_end_ns": end,
                "target_physical_gpu_ids": row.get("target_physical_gpu_ids", []),
            }
        )
    pairs = []
    for draft in drafts:
        for verification in verifications:
            if draft["request_id"] in verification["request_ids"]:
                continue
            intersection_ns = min(
                draft["host_end_ns"], verification["host_end_ns"]
            ) - max(draft["host_start_ns"], verification["host_start_ns"])
            if draft["host_end_ns"] <= verification["host_start_ns"]:
                ordering = "draft-before-verify"
            elif verification["host_end_ns"] <= draft["host_start_ns"]:
                ordering = "verify-before-draft"
            else:
                ordering = "overlap"
            pairs.append(
                {
                    "draft_request_id": draft["request_id"],
                    "proposal_id": draft["proposal_id"],
                    "draft_round_id": draft["round_id"],
                    "verify_microbatch_id": verification["verify_microbatch_id"],
                    "verify_request_ids": verification["request_ids"],
                    "draft_host_interval_ns": [
                        draft["host_start_ns"],
                        draft["host_end_ns"],
                    ],
                    "verify_host_interval_ns": [
                        verification["host_start_ns"],
                        verification["host_end_ns"],
                    ],
                    "signed_intersection_ns": intersection_ns,
                    "overlap_duration_ns": max(0, intersection_ns),
                    "separation_ns": max(0, -intersection_ns),
                    "ordering": ordering,
                    "draft_physical_gpu_id": draft["physical_gpu_id"],
                    "target_physical_gpu_ids": verification[
                        "target_physical_gpu_ids"
                    ],
                }
            )
    if not pairs:
        errors.append("no disjoint-request Draft/Verify interval pair exists")
    pairs.sort(
        key=lambda row: (
            row["separation_ns"],
            -row["overlap_duration_ns"],
            row["draft_host_interval_ns"][0],
            str(row["verify_microbatch_id"]),
        )
    )
    positive = any(row["overlap_duration_ns"] > 0 for row in pairs)
    recorded_positive = any(
        isinstance(row.get("overlap_duration_ns"), int)
        and row["overlap_duration_ns"] > 0
        for row in recorded_overlap_rows
    )
    if positive != recorded_positive:
        errors.append("raw intervals and recorded overlap events disagree")
    return {
        "schema_version": OVERLAP_DIAGNOSTIC_SCHEMA,
        "valid": not errors,
        "errors": errors,
        "draft_interval_count": len(drafts),
        "verification_interval_count": len(verifications),
        "disjoint_pair_count": len(pairs),
        "positive_overlap_observed": positive,
        "recorded_positive_overlap_observed": recorded_positive,
        "nearest_pair": pairs[0] if pairs else None,
        "zero_duration_explanation": (
            None
            if positive
            else "every disjoint pair has a non-positive signed interval intersection"
        ),
        "performance_result": False,
    }


def diagnose_overlap_artifacts(
    *,
    draft_work_paths: Sequence[Path],
    verification_paths: Sequence[Path],
    overlap_paths: Sequence[Path],
    output_path: Path,
) -> dict[str, Any]:
    if not draft_work_paths or not (
        len(draft_work_paths) == len(verification_paths) == len(overlap_paths)
    ):
        raise ValueError("overlap diagnosis requires equally paired non-empty inputs")
    inputs = [*draft_work_paths, *verification_paths, *overlap_paths]
    before = {str(path.resolve()): sha256_file(path) for path in inputs}
    runs = [
        diagnose_overlap_timing(
            CheckpointJsonl(draft_path).read(),
            CheckpointJsonl(verify_path).read(),
            CheckpointJsonl(overlap_path).read(),
        )
        for draft_path, verify_path, overlap_path in zip(
            draft_work_paths, verification_paths, overlap_paths
        )
    ]
    after = {str(path.resolve()): sha256_file(path) for path in inputs}
    immutable = before == after
    errors = [
        f"run {index + 1}: {error}"
        for index, run in enumerate(runs)
        for error in run["errors"]
    ]
    if not immutable:
        errors.append("diagnosis mutated an input artifact")
    result = {
        "schema_version": OVERLAP_DIAGNOSTIC_SCHEMA,
        "valid": not errors,
        "errors": errors,
        "runs": runs,
        "input_artifacts_immutable": immutable,
        "input_sha256_before": before,
        "input_sha256_after": after,
        "performance_result": False,
    }
    atomic_write_json(output_path, result)
    return result


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
        f"- Gate profile: `{value['gate_profile']}`",
        f"- Overlap requirement: `{value['overlap_requirement']}`",
        f"- Positive overlap gate: `{value['overlap_gate']['valid']}`",
        f"- Input artifacts immutable: `{value['input_artifacts_immutable']}`",
        "",
        str(value["claim_boundary"]),
        "",
    ]
    if value["errors"]:
        lines.extend(["## Errors", "", *[f"- {item}" for item in value["errors"]], ""])
    return "\n".join(lines)
