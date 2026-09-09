"""S1-P internal execution validation and resident decode-only performance observations."""

from __future__ import annotations

import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

from specrhythm.phase4.decode_ready import load_decode_ready_manifest
from specrhythm.phase4.dual_terminal import completed_output_prefixes
from specrhythm.phase4.manifest import sha256_file
from specrhythm.phase4.performance import _commit_events, _validate_final_sync, percentile
from specrhythm.phase4.performance_boundary import extract_performance_boundary
from specrhythm.phase4.process_lifecycle import validate_lifecycle_artifact
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.vllm_diagnostics import validate_target_diagnostic
from specrhythm.serving.common import integer, read_json, require
from specrhythm.serving.s1_policy import (
    COMPARISON_SCHEMA,
    EFFECTIVE_SCHEMA,
    RESULT_SCHEMA,
    SEAL_SCHEMA,
    policy_fields,
    validate_policy,
)
from specrhythm.serving.s1_workload import MODES, PROFILE_ENV, load_execution, write_once


def jsonl(directory, name, *, optional=False):
    import json

    path = directory / (name + ".jsonl")
    if optional and not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def distribution(values):
    values = list(values)
    return {
        "count": len(values),
        "p50": percentile(values, 0.5),
        "p90": percentile(values, 0.9),
        "p99": percentile(values, 0.99),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def union_intervals(intervals):
    merged = []
    for start, end in sorted(set(map(tuple, intervals))):
        require(integer(start) and integer(end) and end >= start, "invalid physical time interval")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def interval_duration(intervals):
    return sum(b - a for a, b in union_intervals(intervals))


def validate_outputs(definitions, outputs, eos_ids, ready, events, start, end):
    """No fixed output budget/count, no EOS or numerical equivalence waiver."""
    prefixes = completed_output_prefixes(definitions, outputs, ready)
    by_id = {r["request_id"]: r for r in outputs}
    ready_by_id = {r.request_id: r for r in ready.requests}
    by_request = defaultdict(list)
    for event in events:
        require(event["request_id"] in by_id, "commit belongs to unknown request")
        require(
            integer(event["timestamp_ns"]) and start <= event["timestamp_ns"] <= end,
            "commit lies outside measured boundary",
            request_id=event["request_id"],
        )
        by_request[event["request_id"]].append(event)
    result = []
    for definition in definitions:
        rid = definition.request_id
        row, bootstrap = by_id[rid], ready_by_id[rid]
        tokens = row["generated_token_ids"]
        require(
            not any(t in eos_ids for t in tokens[:-1]),
            "tokens generated after EOS",
            request_id=rid,
        )
        eos = tokens[-1] in eos_ids
        require(
            row["finish_reason"] == ("stop" if eos else "length"),
            "termination contradicts frozen natural EOS policy",
            request_id=rid,
        )
        require(
            row["stop_reason"] is None or row["stop_reason"] == tokens[-1],
            "stop reason differs from terminal token",
            request_id=rid,
        )
        untimed = list(bootstrap.logical_committed_prefix_token_ids)[definition.prompt_length :]
        require(
            untimed == tokens[: len(untimed)] and len(untimed) == 1,
            "actual untimed bootstrap differs from final output",
            request_id=rid,
        )
        ordered = sorted(by_request[rid], key=lambda r: r["timestamp_ns"])
        measured = [t for e in ordered for t in e["token_ids"]]
        require(
            measured == tokens[len(untimed) :], "timed token accounting differs", request_id=rid
        )
        terminal_in_setup = len(tokens) == len(untimed)
        require(
            not terminal_in_setup or eos or len(tokens) == definition.maximum_new_tokens,
            "zero timed work lacks terminal evidence",
            request_id=rid,
        )
        final = ordered[-1]["timestamp_ns"] if ordered else None
        result.append(
            {
                "request_id": rid,
                "task_class": definition.task_class,
                "generated_token_ids": tokens,
                "generated_tokens": len(tokens),
                "bootstrap_token_ids": untimed,
                "untimed_output_tokens": len(untimed),
                "timed_committed_tokens": len(measured),
                "finish_reason": row["finish_reason"],
                "stop_reason": row["stop_reason"],
                "eos_terminated": eos,
                "maximum_new_tokens": definition.maximum_new_tokens,
                "max_token_termination": row["finish_reason"] == "length",
                "final_prefix_sha256": token_prefix_hash(prefixes[rid]),
                "terminal_in_setup": terminal_in_setup,
                "last_commit_ns": final,
                "decode_barrier_to_completion_ms": (final - start) / 1e6 if final else 0.0,
                "completion_clock_source": "last logical commit; setup terminal => 0",
                "commit_events": ordered,
            }
        )
    return result


def validate_rounds(rows, request_results, prompt_lengths, eos):
    final = {r["request_id"]: r for r in request_results}
    seen = set()
    next_offset = defaultdict(lambda: 1)
    semantics = []
    for row in sorted(rows, key=lambda r: (r["request_id"], r["round_id"])):
        rid, round_id = row["request_id"], row["round_id"]
        require(rid in final and integer(round_id), "invalid verification round identity")
        key = (rid, round_id)
        require(key not in seen, "duplicate verification round", request_id=rid, round_id=round_id)
        require(round_id == sum(r == rid for r, _ in seen), "noncontiguous request round")
        seen.add(key)
        proposal, committed = row["proposal_token_ids"], row["committed_token_ids"]
        accepted, rejected = row["accepted_draft_token_ids"], row["rejected_draft_token_ids"]
        correction, bonus = row["target_correction_token_ids"], row["target_bonus_token_ids"]
        require(
            accepted == proposal[: len(accepted)] and rejected == proposal[len(accepted) :],
            "proposal conservation failed",
            request_id=rid,
            round_id=round_id,
        )
        require(
            len(accepted) == row["accepted_draft_tokens"]
            and len(rejected) == row["rejected_draft_tokens"],
            "proposal count mismatch",
        )
        require(
            committed
            and committed == accepted + correction + bonus
            and len(correction) + len(bonus) <= 1,
            "correction/bonus accounting invalid",
        )
        parent_length = row.get("parent_prefix_len", row.get("prefix_token_count"))
        # Serial records parent_prefix_len; Dual records it in the proposal payload too.
        if parent_length is None:
            parent_length = row.get("parent_prefix_length")
        require(integer(parent_length, 1), "round lacks parent prefix length")
        offset = parent_length - prompt_lengths[rid]
        tokens = final[rid]["generated_token_ids"]
        require(
            offset == next_offset[rid] and committed == tokens[offset : offset + len(committed)],
            "round commit differs from final sequence",
            request_id=rid,
            round_id=round_id,
        )
        next_offset[rid] = offset + len(committed)
        terminal = offset + len(committed) == len(tokens)
        require(
            len(correction) + len(bonus) == 1 or terminal,
            "nonterminal verification lacks correction/bonus",
        )
        require(not any(t in eos for t in tokens[:offset]), "verification after terminal EOS")
        if "logical_target_kv_length" in row:
            require(
                row["logical_target_kv_length"]
                == row["logical_draft_kv_length"]
                == parent_length + len(committed),
                "Serial logical Target/Draft KV accounting differs",
            )
        if "prefix_version" in row:
            require(row["prefix_version"] == round_id + 1, "stale proposal prefix version")
        semantics.append(
            {
                "request_id": rid,
                "round_id": round_id,
                "prefix_length": parent_length,
                "prefix_sha256": row.get("parent_prefix_hash", row.get("prefix_token_sha256")),
                "proposal": proposal,
                "accepted": accepted,
                "rejected": rejected,
                "correction": correction,
                "bonus": bonus,
                "committed": committed,
                "terminal": terminal,
            }
        )
    return semantics


def target_forwards(diagnostics, start, rounds):
    batches = defaultdict(dict)
    progress = {(r["request_id"], r["prefix_length"]): len(r["committed"]) for r in rounds}
    for row in diagnostics:
        a, b = row["target_forward_start_ns"], row["target_forward_end_ns"]
        if a < start:
            continue
        require(
            not validate_target_diagnostic(row),
            "Target structural diagnostic invalid",
            errors=validate_target_diagnostic(row),
        )
        key = (a, b)
        previous = batches[key].get(row["request_id"])
        require(
            previous is None
            or previous["target_input_token_ids"] == row["target_input_token_ids"],
            "TP Target row input disagreement",
        )
        batches[key][row["request_id"]] = row
        if row["proposal_token_ids"]:
            require(
                (row["request_id"], len(row["committed_prefix_token_ids"])) in progress,
                "speculative Target forward lacks matching committed round",
            )
    result = []
    for (a, b), members in sorted(batches.items()):
        rows = list(members.values())
        result.append(
            {
                "start_ns": a,
                "end_ns": b,
                "request_ids": list(members),
                "B": len(members),
                "Q": sum(len(r["position_ids"]) for r in rows),
                "proposal_positions": sum(len(r["proposal_token_ids"]) for r in rows),
                "kind": "verify" if any(r["proposal_token_ids"] for r in rows) else "tail",
                "context": distribution(len(r["committed_prefix_token_ids"]) for r in rows),
                "committed_progress": sum(
                    progress.get((r["request_id"], len(r["committed_prefix_token_ids"])), 1)
                    for r in rows
                ),
                "host_envelope_ms": (b - a) / 1e6,
                "gpu_event_ms": None,
                "gpu_event_status": "unavailable in row diagnostic",
            }
        )
    return result


def draft_measurement(backend, start):
    """Join existing forward events/fences, without adding synchronization or file writes."""
    records = [
        r for r in backend.get("s1_forward_records", []) if r["purpose"] in ("proposal", "commit")
    ]
    require(
        len(records) == backend["draft_model_forward_count"],
        "Draft measured forward evidence/count differs",
    )
    for row in records:
        require(integer(row["B"], 1) and integer(row["Q"], row["B"]), "Draft physical B/Q invalid")
        require(
            integer(row["cuda_completion_observed_ns"])
            and start
            <= row["host_start_ns"]
            <= row["host_launch_end_ns"]
            <= row["cuda_completion_observed_ns"],
            "Draft work lacks ordered measured completion evidence",
        )
        require(
            type(row["gpu_event_ms"]) in (int, float)
            and math.isfinite(row["gpu_event_ms"])
            and row["gpu_event_ms"] > 0,
            "Draft forward lacks positive actual CUDA event duration",
        )
    for purpose in ("proposal", "commit"):
        selected = [r for r in records if r["purpose"] == purpose]
        require(
            len(selected) == backend["draft_model_forward_count_by_purpose"].get(purpose, 0)
            and sum(r["Q"] for r in selected)
            == backend["draft_scheduled_token_count_by_purpose"].get(purpose, 0),
            "Draft aggregate forward/query accounting differs",
            purpose=purpose,
        )
    require(
        math.isclose(
            sum(r["gpu_event_ms"] for r in records),
            backend["draft_gpu_event_time_ms"],
            rel_tol=1e-9,
            abs_tol=1e-9,
        ),
        "Draft aggregate GPU duration differs",
    )
    return records


def workload_metrics(requests, start, end):
    timed = sum(r["timed_committed_tokens"] for r in requests)
    require(
        type(start) is int and type(end) is int and end > start,
        "measurement requires ordered real runtime boundaries",
    )
    makespan = (end - start) / 1e6
    classes = {}
    for task in ("overall", "chat", "code", "summarization", "reasoning"):
        selected = [r for r in requests if task == "overall" or r["task_class"] == task]
        n = len(selected)
        classes[task] = {
            "completed_requests": n,
            "output_length": distribution(r["generated_tokens"] for r in selected),
            "generated_tokens": sum(r["generated_tokens"] for r in selected),
            "untimed_bootstrap_tokens": sum(r["untimed_output_tokens"] for r in selected),
            "eos_terminated_requests": sum(r["eos_terminated"] for r in selected),
            "max_token_terminated_requests": sum(r["max_token_termination"] for r in selected),
            "setup_terminal_requests": sum(r["terminal_in_setup"] for r in selected),
            "eos_ratio": sum(r["eos_terminated"] for r in selected) / n if n else None,
            "cap_ratio": sum(r["max_token_termination"] for r in selected) / n if n else None,
            "timed_committed_tokens": sum(r["timed_committed_tokens"] for r in selected),
        }
    return {
        "completed_requests": len(requests),
        "generated_tokens": classes["overall"]["generated_tokens"],
        "untimed_bootstrap_tokens": classes["overall"]["untimed_bootstrap_tokens"],
        "setup_terminal_requests": classes["overall"]["setup_terminal_requests"],
        "eos_terminated_requests": classes["overall"]["eos_terminated_requests"],
        "max_token_terminated_requests": classes["overall"]["max_token_terminated_requests"],
        "decode_makespan_ms": makespan,
        "timed_committed_tokens": timed,
        "throughput_tokens_per_second": timed / (makespan / 1000) if timed else None,
        "decode_barrier_to_completion_ms": distribution(
            r["decode_barrier_to_completion_ms"] for r in requests
        ),
        "aggregate_ms_per_timed_token": makespan / timed if timed else None,
        "per_token_denominator": "total actual timed output tokens, excluding manifest bootstrap",
        "time_scope": "decode barrier through final all-rank GPU completion; not serving TPOT",
        "output_by_task": classes,
    }


def inspect_run(manifest_path: Path, directory: Path, mode: str):
    manifest, definitions = load_execution(manifest_path, verify_parent=True)
    os.environ[PROFILE_ENV] = str(manifest_path.resolve())
    checks = {
        k: {"valid": False, "errors": []} for k in ("correctness", "measurement", "lifecycle")
    }
    report = {
        "schema_version": RESULT_SCHEMA,
        **policy_fields(),
        "mode": mode,
        "valid": False,
        "performance_result": False,
        "gpu_execution": True,
        "synthetic": False,
        "execution_sha256": manifest["manifest_sha256"],
        "logical_sha256": manifest["logical_sha256"],
        "arrival_replay_enabled": False,
        "execution_git_commit": manifest["execution"]["git_commit"],
        "execution_configuration": manifest["execution"],
        "checks": checks,
        "requests": [],
        "errors": [],
    }
    try:
        raw = read_json(directory / "raw.json")
        require(
            raw.get("valid") is True and not raw.get("errors"),
            "raw execution invalid",
            raw_errors=raw.get("errors", raw.get("resident_errors")),
        )
        lifecycle = read_json(directory / "process-lifecycle.json")
        require(
            not validate_lifecycle_artifact(lifecycle)
            and lifecycle.get("run_valid") is True
            and lifecycle.get("target_exit_status") == 0,
            "owned execution/cleanup invalid",
        )
        exit_code = read_json(directory / "exit-code.json")
        require(
            exit_code.get("effective_exit_code") == 0
            and exit_code.get("coordinator_exit_code") == 0,
            "nonzero owned execution exit code",
            exit_code=exit_code,
        )
        checks["lifecycle"]["valid"] = True
        effective = read_json(directory / "s1-effective-runtime.json")
        validate_policy(effective, EFFECTIVE_SCHEMA)
        require(
            effective["execution_sha256"] == manifest["manifest_sha256"],
            "effective runtime execution identity mismatch",
        )
        ready = load_decode_ready_manifest(read_json(directory / "decode-ready-manifest.json"))
        require(
            ready.workload_sha256 == manifest["logical"]["workload_sha256"]
            and ready.specrhythm_git_commit == manifest["execution"]["git_commit"],
            "decode-ready input/code identity mismatch",
        )
        setup = read_json(directory / "setup-ready.json")
        require(setup.get("global_decode_ready") is True, "global readiness not proven")
        timing = jsonl(directory, "timing-events")
        consumer = {"target": "target-only", "serial": "serial", "pingpong": "dual-batch"}[mode]
        start, errors = extract_performance_boundary(timing, consumer=consumer)
        require(
            not errors and start > setup["ready_published_ns"],
            "invalid measured start",
            errors=errors,
        )
        events, errors = _commit_events(
            "dual-batch" if mode == "pingpong" else mode, directory, timing, start
        )
        require(not errors, "invalid commit evidence", errors=errors)
        sync, errors = _validate_final_sync(
            raw.get("phase4b2_final_sync"),
            2,
            [1, 2],
            max([start] + [e["timestamp_ns"] for e in events]),
        )
        require(not errors, "invalid final GPU synchronization", errors=errors)
        backend = read_json(directory / "draft-backend-report.json")
        draft_forwards = draft_measurement(backend, start)
        end = max(
            [r["final_cuda_synchronize_complete_ns"] for r in sync]
            + [r["cuda_completion_observed_ns"] for r in draft_forwards]
        )
        requests = validate_outputs(
            definitions,
            raw["outputs"],
            manifest["execution"]["eos_token_ids"],
            ready,
            events,
            start,
            end,
        )
        rounds = jsonl(
            directory,
            "round-events" if mode == "serial" else "proposal-events",
            optional=True,
        )
        semantics = validate_rounds(
            rounds,
            requests,
            {r.request_id: r.prompt_length for r in definitions},
            manifest["execution"]["eos_token_ids"],
        )
        definition_by_id = {r.request_id: r for r in definitions}
        result_by_id = {r["request_id"]: r for r in requests}
        for row in semantics:
            definition = definition_by_id[row["request_id"]]
            offset = row["prefix_length"] - definition.prompt_length
            prefix = [
                *definition.prompt_token_ids,
                *result_by_id[row["request_id"]]["generated_token_ids"][:offset],
            ]
            require(
                row["prefix_sha256"] == token_prefix_hash(prefix),
                "proposal parent prefix differs from exact final prefix",
                round=row,
            )
        diagnostics = jsonl(directory, "target-diagnostics", optional=not events)
        forwards = target_forwards(diagnostics, start, semantics)
        require(forwards or not events, "timed commits lack mandatory Target forward evidence")
        require(all(r["end_ns"] <= end for r in forwards), "Target forward beyond measurement end")
        require(
            backend.get("backend_name") == "vllm-batched"
            and backend.get("backend_shutdown_complete") is True
            and backend.get("draft_live_requests_final") == 0
            and backend.get("execution_failed") is False,
            "Draft backend identity/cleanup invalid",
        )
        if mode == "target":
            require(
                not rounds and backend["draft_model_forward_count"] == 0,
                "Target timed window performed Draft proposal/commit computation",
            )
        elif semantics and any(
            sum(
                not r["terminal_in_setup"] and r["maximum_new_tokens"] > 2
                for r in requests
                if mode == "serial" or manifest["assignment"][r["request_id"]] == cohort
            )
            > 1
            for cohort in ("A", "B")
        ):
            require(
                backend.get("true_batching_observed") is True,
                "production Draft did not perform multi-request batching",
            )
        for row in rounds:
            draft_start = row.get("draft_start_ns", row.get("timeline", {}).get("draft_start_ns"))
            require(
                integer(draft_start) and draft_start >= start,
                "Draft initial/round proposal predates measurement",
            )
        overlap = {"status": "not applicable", "physical_overlap_observed": False}
        cohort = None
        if mode == "pingpong":
            cohort, overlap = pingpong_evidence(
                directory, manifest["assignment"], raw, ready, start, bool(semantics)
            )
        report.update(
            requests=requests,
            round_semantics=semantics,
            metrics=workload_metrics(requests, start, end),
            measurement={
                "start_ns": start,
                "end_ns": end,
                "end_source": (
                    "latest Target final sync / necessary Draft forward fence completion"
                ),
                "setup_bootstrap_excluded": True,
                "cleanup_excluded": True,
                "initial_fill_drain_included": True,
            },
            work={
                "Target_physical_forwards": forwards,
                "Target_forward_count": len(forwards),
                "Target_query_tokens": sum(r["Q"] for r in forwards),
                "Target_batch_B": distribution(r["B"] for r in forwards),
                "Target_query_Q": distribution(r["Q"] for r in forwards),
                "Target_prefill_forwards": None,
                "Target_prefill_status": "untimed; chunked prefill not fully captured",
                "Target_host_interval_union_ms": interval_duration(
                    (r["start_ns"], r["end_ns"]) for r in forwards
                )
                / 1e6,
                "Draft": backend,
                "Draft_physical_forwards": draft_forwards,
                "verification_rounds": len(semantics),
                "proposed_tokens": sum(len(r["proposal"]) for r in semantics),
                "accepted_tokens": sum(len(r["accepted"]) for r in semantics),
                "rejected_tokens": sum(len(r["rejected"]) for r in semantics),
                "mean_accepted_draft_prefix": statistics.mean(
                    len(r["accepted"]) for r in semantics
                )
                if semantics
                else None,
                "mean_committed_tokens_per_verification": statistics.mean(
                    len(r["committed"]) for r in semantics
                )
                if semantics
                else None,
            },
            effective_runtime=effective,
            physical_setup_sha256=ready.manifest_sha256,
            overlap=overlap,
            cohort=cohort,
            resources={
                "Target_GPU_count": 2,
                "Draft_GPU_reserved": 1,
                "Draft_GPU_used_in_setup": 1,
                "Draft_GPU_used_in_timed_window": 0 if mode == "target" else 1,
                "equal_total_GPU_comparison": False,
            },
            attribution={
                "scheduler_IPC_wait_commit_sync": "unavailable as exclusive spans",
                "unattributed_ms": None,
                "reason": "overlapping host envelopes cannot be subtracted as disjoint GPU time",
                "cohort_clipping_reason": "unavailable",
            },
        )
        checks["correctness"]["valid"] = checks["measurement"]["valid"] = True
        report["valid"] = True
        report["performance_result"] = report["metrics"]["timed_committed_tokens"] > 0
        verify_finite_metrics(report["metrics"])
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        report["valid"] = report["performance_result"] = False
        report["errors"].append(f"S1 execution evidence: {error}")
        report["error_details"] = getattr(error, "details", {})
        for check in checks.values():
            if not check["valid"]:
                check["errors"].append(str(error))
    report["artifact_sha256"] = {
        p.name: sha256_file(p)
        for p in sorted(directory.iterdir())
        if p.is_file() and p.name not in ("result.json", "seal.json")
    }
    return report


def verify_finite_metrics(metrics):
    for key in (
        "decode_makespan_ms",
        "throughput_tokens_per_second",
        "aggregate_ms_per_timed_token",
    ):
        value = metrics[key]
        require(
            value is None or (type(value) in (int, float) and math.isfinite(value) and value > 0),
            "nonfinite/nonpositive performance metric",
            field=key,
        )


def pingpong_evidence(directory, assignment, raw, ready, boundary, has_verifications):
    from specrhythm.phase4.dual_correctness import (
        validate_draft_sync,
        validate_final_commit_sequence,
        validate_proposal_lifecycle_events,
        validate_request_state_events,
        validate_verification_contracts,
    )
    from specrhythm.phase4.dual_overlap_characterization import read_overlap
    from specrhythm.phase4.dual_rhythm import cohort_for

    states = jsonl(directory, "request-state-events")
    rounds = jsonl(directory, "proposal-events", optional=not has_verifications)
    drafts = jsonl(directory, "draft-work-events", optional=not has_verifications)
    require(
        not validate_request_state_events(states),
        "PingPong request lifecycle invalid",
    )
    final_errors = validate_final_commit_sequence(raw, ready, rounds, drafts, states)
    require(
        not final_errors, "PingPong terminal prefix/commit sequence invalid", errors=final_errors
    )
    terminal_outputs = {r["request_id"]: r for r in raw["decode_only_outputs"]}
    require(
        len(terminal_outputs) == len(raw["outputs"])
        and all(
            terminal_outputs[r["request_id"]]["generated_token_ids"] == r["generated_token_ids"]
            for r in raw["outputs"]
        ),
        "PingPong terminal and serialized outputs differ",
    )
    require(not validate_draft_sync(drafts, rounds, states), "PingPong Draft sync invalid")
    if has_verifications:
        require(
            read_json(directory / "plugin-report.json").get("sampled_row_tp_consensus") is True,
            "PingPong sampled-row TP consensus failed",
        )
        require(
            not validate_verification_contracts(
                rounds,
                jsonl(directory, "verification-events"),
                jsonl(directory, "target-diagnostics"),
            ),
            "PingPong Target pending/proposal/row mapping contract invalid",
        )
    lifecycle = jsonl(directory, "proposal-lifecycle-events", optional=not has_verifications)
    require(
        not lifecycle or not validate_proposal_lifecycle_events(lifecycle),
        "PingPong proposal lifecycle invalid",
    )
    backend = read_json(directory / "draft-backend-report.json")
    require(backend["assignment"] == assignment, "PingPong Draft assignment changed")
    require(
        raw.get("dual_rhythm") == "pingpong" and raw.get("cohort_assignment") == assignment,
        "PingPong Target assignment/rhythm changed",
    )
    verifies = jsonl(directory, "verification-events", optional=not has_verifications)
    require(
        len(verifies) == len(rounds)
        and {r["proposal_id"] for r in verifies}
        == {r["proposal_id"] for r in rounds}
        == {r["proposal_id"] for r in lifecycle if r["lifecycle_state"] == "CONSUMED"},
        "PingPong consumed/verified/committed proposal coverage differs",
    )
    cycles = jsonl(directory, "scheduler-events")
    selected, previous = [], None
    for row in cycles:
        ids = row.get("attained_cohort_request_ids", [])
        if not ids:
            continue
        cohort = cohort_for(assignment, ids)
        require(
            cohort == row["logical_cohort"]
            and set(ids) <= set(row["eligible_cohort_request_ids"]),
            "PingPong scheduled ineligible/cross-cohort request",
        )
        opposite = "B" if cohort == "A" else "A"
        require(
            previous != cohort or row["active_cohort_sizes"][opposite] == 0,
            "PingPong failed fixed alternation before drain",
        )
        previous = cohort
        selected.append(
            {
                "cycle_id": row["cycle_id"],
                "cohort": cohort,
                "candidate_sizes": row["active_cohort_sizes"],
                "admissible_ids": row["eligible_cohort_request_ids"],
                "scheduled_ids": ids,
                "clipping_reason": "unavailable",
            }
        )
    works = backend.get("pingpong_work_records", [])
    for work in works:
        require(
            cohort_for(assignment, [r["request_id"] for r in work["rows"]])
            == work["logical_cohort"],
            "PingPong Draft mixed cohorts",
        )
        require(work["host_start_ns"] >= boundary, "PingPong fill/work predates measurement")
        for verify in verifies:
            ranks = verify["target_rank_intervals"]
            require(bool(ranks), "PingPong verification lacks Target rank timing")
            if work["host_start_ns"] < max(r["host_end_ns"] for r in ranks) and (
                work["host_end_ns"] > min(r["host_start_ns"] for r in ranks)
            ):
                require(
                    work["logical_cohort"] != cohort_for(assignment, verify["verify_request_ids"]),
                    "PingPong same-cohort Draft/Target concurrency",
                )
    overlap = (
        read_overlap(directory, raw)
        if has_verifications
        else {
            "valid": True,
            "errors": [],
            "physical_overlap_valid": False,
            "observed_overlap_ms": 0.0,
            "reason": "no speculative verification work",
        }
    )
    require(overlap["valid"], "invalid physical overlap evidence", errors=overlap["errors"])
    return {
        "assignment": assignment,
        "selected_cycles": selected,
        "pipeline_fill_drain": "measured; separate GPU critical-path attribution unavailable",
    }, overlap


def first_divergence(a, b):
    tokens_a, tokens_b = a["generated_token_ids"], b["generated_token_ids"]
    for index in range(max(len(tokens_a), len(tokens_b))):
        x = tokens_a[index] if index < len(tokens_a) else None
        y = tokens_b[index] if index < len(tokens_b) else None
        if x != y:
            return {"position": index, "expected_token": x, "actual_token": y}
    for field in ("finish_reason", "stop_reason", "bootstrap_token_ids", "final_prefix_sha256"):
        if a.get(field) != b.get(field):
            return {"field": field, "expected": a.get(field), "actual": b.get(field)}
    return None


def round_differences(first, repeated):
    a = {(r["request_id"], r["round_id"]): r for r in first}
    b = {(r["request_id"], r["round_id"]): r for r in repeated}
    differences = []
    for key in sorted(set(a) | set(b)):
        if a.get(key) == b.get(key):
            continue
        before, after = a.get(key, {}), b.get(key, {})
        changed = {
            k: {"first": before.get(k), "repeat": after.get(k)}
            for k in sorted(set(before) | set(after))
            if before.get(k) != after.get(k)
        }
        classification = (
            "verification round added/absent"
            if not before or not after
            else "verification boundary differs"
            if "prefix_length" in changed
            else "same-prefix Draft proposal differs"
            if "proposal" in changed
            else "acceptance/commit boundary differs"
        )
        differences.append(
            {
                "request_id": key[0],
                "round_id": key[1],
                "classification": classification,
                "changed_fields": changed,
                "scheduling_or_numerical_root_cause": "unavailable without event audit",
            }
        )
    return differences


def compare_results(results):
    """Compare input identity and actual rates; never compare independent output sequences."""
    require(set(results) == set(MODES) and all(results.values()), "three S1-P modes are required")
    baseline = results["target"][0]
    expected = {r["request_id"] for r in baseline["requests"]}
    errors, observations = [], {}
    for mode, runs in results.items():
        observations[mode] = []
        for index, result in enumerate(runs):
            validate_policy(result, RESULT_SCHEMA)
            if result.get("mode") != mode or result.get("valid") is not True or result.get(
                "errors"
            ) or not all(
                result["checks"][k]["valid"] is True and not result["checks"][k].get("errors")
                for k in ("correctness", "measurement", "lifecycle")
            ):
                errors.append(f"{mode}/{index}: invalid execution evidence")
            if (
                result["execution_sha256"] != baseline["execution_sha256"]
                or result.get("logical_sha256") != baseline.get("logical_sha256")
            ):
                errors.append(f"{mode}/{index}: execution/config/input identity differs")
            if result.get("effective_runtime", {}).get("sampling") != baseline.get(
                "effective_runtime", {}
            ).get("sampling"):
                errors.append(f"{mode}/{index}: effective sampling differs")
            actual = {r["request_id"] for r in result["requests"]}
            if actual != expected or len(actual) != len(result["requests"]):
                errors.append(f"{mode}/{index}: completed request set differs")
            # This is within-run metric accounting, not equality with another run's output.
            metrics = result["metrics"]
            timed = sum(r["timed_committed_tokens"] for r in result["requests"])
            try:
                verify_finite_metrics(metrics)
                require(timed == metrics["timed_committed_tokens"], "timed token metric differs")
                rate = timed / (metrics["decode_makespan_ms"] / 1000) if timed else None
                require(
                    metrics["throughput_tokens_per_second"] == rate,
                    "throughput does not use this run's actual timed tokens",
                )
            except (ValueError, TypeError, ZeroDivisionError) as error:
                errors.append(f"{mode}/{index}: invalid measurement: {error}")
            observations[mode].append(
                {
                    "repeat": index,
                    "metrics": metrics,
                    "request_output_lengths": [
                        {k: r[k] for k in (
                            "request_id", "task_class", "generated_tokens",
                            "untimed_output_tokens", "timed_committed_tokens",
                            "finish_reason", "stop_reason", "terminal_in_setup",
                        )}
                        for r in result["requests"]
                    ],
                    "work": result.get("work"),
                    "overlap": result.get("overlap"),
                    "resources": result.get("resources"),
                    "effective_runtime": result.get("effective_runtime"),
                    "execution_configuration": result.get("execution_configuration"),
                }
            )
    aggregation = {}
    for mode, runs in results.items():
        if not errors and all(r.get("performance_result") for r in runs):
            aggregation[mode] = {
                k: {
                    "raw": [r["metrics"][k] for r in runs],
                    "median": statistics.median(r["metrics"][k] for r in runs),
                    "population_stdev": statistics.pstdev(r["metrics"][k] for r in runs),
                }
                for k in (
                    "decode_makespan_ms", "throughput_tokens_per_second",
                    "generated_tokens", "timed_committed_tokens", "untimed_bootstrap_tokens",
                )
            }
    ratios = None
    if not errors and set(aggregation) == set(MODES):
        ratios = {
            mode: {
                "median_makespan_ratio": aggregation["target"]["decode_makespan_ms"]["median"]
                / aggregation[mode]["decode_makespan_ms"]["median"],
                "median_throughput_ratio": aggregation[mode]["throughput_tokens_per_second"][
                    "median"
                ] / aggregation["target"]["throughput_tokens_per_second"]["median"],
                "target_timed_tokens_by_repeat": aggregation["target"]["timed_committed_tokens"][
                    "raw"
                ],
                "mode_timed_tokens_by_repeat": aggregation[mode]["timed_committed_tokens"]["raw"],
                "throughput_above_target": aggregation[mode]["throughput_tokens_per_second"][
                    "median"
                ] > aggregation["target"]["throughput_tokens_per_second"]["median"],
            }
            for mode in MODES
        }
    counts = [r["metrics"]["timed_committed_tokens"] for runs in results.values() for r in runs]
    return {
        "schema_version": COMPARISON_SCHEMA,
        **policy_fields(),
        "execution_sha256": baseline["execution_sha256"],
        "logical_sha256": baseline.get("logical_sha256"),
        "valid": not errors,
        "errors": errors,
        "repeatability": {
            "performed": False,
            "status": "NOT_REQUIRED",
            "exact_tokens_and_termination": None,
            "round_comparison_performed": False,
        },
        "raw_reference_checked": False,
        "metrics": aggregation,
        "observations": observations,
        "matched_work": {
            "status": "NOT_ASSESSED",
            "exact_timed_output_equal": None,
            "equal_timed_token_counts": len(set(counts)) == 1,
            "interpretation": (
                "Equal budgets or token counts do not prove equal sequences or computation. "
                "Compare each run's actual tokens, B, Q, proposal/acceptance work and time."
            ),
        },
        "performance_ratios": ratios,
        "ratio_interpretation": (
            "Ratios of mode-level medians, using each run's actual timed tok/s. "
            "Makespan ratios are accompanied by actual output counts, not equal-work speedup."
        ),
        "label": "resident decode-only three-mode performance observation",
        "pure_batching_claim": False,
        "end_to_end_improvement_claim": False,
        "new_serving_SLO_result": False,
    }


def seal_run(directory, report):
    validate_policy(report, RESULT_SCHEMA)
    write_once(directory / "result.json", report)
    write_once(
        directory / "seal.json",
        {
            "schema_version": SEAL_SCHEMA,
            **policy_fields(),
            "execution_sha256": report["execution_sha256"],
            "files": {
                p.name: sha256_file(p)
                for p in sorted(directory.iterdir())
                if p.is_file() and p.name != "seal.json"
            },
        },
    )


def verify_seal(directory):
    sealed = read_json(directory / "seal.json")
    validate_policy(sealed, SEAL_SCHEMA)
    inventory = sealed["files"]
    require(
        set(inventory)
        == {p.name for p in directory.iterdir() if p.is_file() and p.name != "seal.json"},
        "sealed run file inventory changed",
    )
    for name, checksum in inventory.items():
        require(
            Path(name).name == name and sha256_file(directory / name) == checksum,
            "sealed run checksum mismatch",
            file=name,
        )
    report = read_json(directory / "result.json")
    validate_policy(report, RESULT_SCHEMA)
    require(
        report["execution_sha256"] == sealed["execution_sha256"],
        "sealed policy/execution identity differs",
    )
    return report
