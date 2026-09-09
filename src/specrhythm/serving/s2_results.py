"""Offline S2 internal validity, queue-inclusive engineering SLO and finite-trace reports."""

from __future__ import annotations

import math
from collections import Counter
from statistics import mean

from specrhythm.phase4.decode_ready import load_decode_ready_manifest
from specrhythm.phase4.manifest import sha256_file
from specrhythm.phase4.performance import _commit_events
from specrhythm.phase4.process_lifecycle import validate_lifecycle_artifact
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.serving.common import TASKS, read_json, require
from specrhythm.serving.runtime_profile import load_s2
from specrhythm.serving.s1_results import (
    distribution,
    draft_backend_checks,
    draft_measurement,
    interval_duration,
    jsonl,
    target_forwards,
    validate_rounds,
)
from specrhythm.serving.s1_workload import write_once
from specrhythm.serving.s2_plan import BOUNDARY, LABEL, check_seal, sealed


def stats(values):
    values = [v for v in values if v is not None]
    return {**distribution(values), "mean": mean(values) if values else None}


def request_metrics(definitions, runtime, trace, eos, policy):
    start, end = runtime["start_ns"], runtime["end_ns"]
    require(
        type(start) is int and type(end) is int and end > start,
        "S2 observation boundaries invalid",
    )
    check_seal(trace)
    actual = {r["request_id"]: r for r in runtime["requests"]}
    require(
        len(actual) == len(runtime["requests"]) == len(definitions)
        and set(actual) == {r.request_id for r in definitions},
        "S2 completed request set/count differs",
    )
    arrivals = {
        r["request_id"]: start + round(r["arrival_offset_seconds"] * 1e9) for r in trace["rows"]
    }
    result = []
    for definition in definitions:
        rid = definition.request_id
        row = actual[rid]
        tokens = row["generated_token_ids"]
        token_prefix_hash(tokens)
        require(
            row["state"] == "FINISHED" and row["resources_released"] is True,
            "S2 request lifecycle/release invalid",
            request_id=rid,
        )
        require(
            0 < len(tokens) <= definition.maximum_new_tokens
            and not any(t in eos for t in tokens[:-1]),
            "S2 token budget/EOS/accounting invalid",
            request_id=rid,
        )
        eos_end = tokens[-1] in eos
        require(
            row["finish_reason"] == ("stop" if eos_end else "length")
            and (eos_end or len(tokens) == definition.maximum_new_tokens),
            "S2 natural termination invalid",
            request_id=rid,
        )
        arrival, observed, completion = (
            row["arrival_ns"],
            row["observed_arrival_ns"],
            row["completion_ns"],
        )
        require(
            arrival == arrivals[rid] and start <= arrival <= observed <= completion <= end,
            "S2 planned/observed/completion time invalid",
            request_id=rid,
        )
        commits = row["commits"]
        require(
            [t for c in commits for t in c["token_ids"]] == tokens[1:],
            "S2 timed commit/bootstrap conservation failed",
            request_id=rid,
        )
        times = [c["timestamp_ns"] for c in commits for _ in c["token_ids"]]
        n = len(times)
        admission = row["admission_ns"]
        if n:
            require(
                type(admission) is int
                and observed <= admission <= times[0]
                and times == sorted(times)
                and times[-1] == completion,
                "S2 commit preceded arrival/admission or completion differs",
                request_id=rid,
            )
        else:
            require(
                admission is None and completion == observed and len(tokens) == 1,
                "S2 bootstrap-terminal has artificial timed work",
                request_id=rid,
            )
        avg = (completion - arrival) / 1e6 / n if n else None
        threshold = policy["threshold_ms_per_token"][definition.task_class] if policy else None
        good = avg <= threshold if avg is not None and threshold is not None else None
        result.append(
            {
                "request_id": rid,
                "task_class": definition.task_class,
                "arrival_ns": arrival,
                "observed_arrival_ns": observed,
                "admission_ns": admission,
                "completion_ns": completion,
                "bootstrap_tokens": 1,
                "timed_tokens": n,
                "total_generated_tokens": len(tokens),
                "maximum_new_tokens": definition.maximum_new_tokens,
                "eos_terminated": eos_end,
                "cap_terminated": not eos_end,
                "cohort": row["cohort"],
                "arrival_handling_lag_ms": (observed - arrival) / 1e6,
                "queue_ms": (admission - arrival) / 1e6 if n else None,
                "first_timed_token_wait_ms": (times[0] - arrival) / 1e6 if n else None,
                "request_tpot_ms": (times[-1] - times[0]) / 1e6 / (n - 1) if n > 1 else None,
                "token_intervals_ms": [(b - a) / 1e6 for a, b in zip(times, times[1:])],
                "queue_inclusive_decode_avg_ms_per_token": avg,
                "slo_threshold_ms_per_token": threshold,
                "slo_good": good,
                "zero_timed_token_request": not n,
            }
        )
    return result


def observation_metrics(rows, start, end):
    seconds = (end - start) / 1e9
    require(math.isfinite(seconds) and seconds > 0, "S2 observation duration must be positive")
    groups = {}
    for task in ("overall", *TASKS):
        chosen = [r for r in rows if task == "overall" or r["task_class"] == task]
        eligible = [r for r in chosen if r["slo_good"] is not None]
        good = [r for r in eligible if r["slo_good"]]
        groups[task] = {
            "completed_requests": len(chosen),
            "slo_eligible_requests": len(eligible),
            "zero_timed_token_requests": sum(r["zero_timed_token_request"] for r in chosen),
            "slo_good_requests": len(good),
            "request_attainment": len(good) / len(eligible) if eligible else None,
            "slo_good_timed_tokens": sum(r["timed_tokens"] for r in good),
            "timed_tokens": sum(r["timed_tokens"] for r in chosen),
            "bootstrap_tokens": len(chosen),
            "total_generated_tokens": sum(r["total_generated_tokens"] for r in chosen),
            "throughput_tok_s": sum(r["timed_tokens"] for r in chosen) / seconds,
            "token_goodput_tok_s": sum(r["timed_tokens"] for r in good) / seconds,
            "request_goodput_req_s": len(good) / seconds,
            "eos_requests": sum(r["eos_terminated"] for r in chosen),
            "cap_requests": sum(r["cap_terminated"] for r in chosen),
            "output_length": stats(r["total_generated_tokens"] for r in chosen),
            **{
                k: stats(r[k] for r in chosen)
                for k in (
                    "queue_ms",
                    "first_timed_token_wait_ms",
                    "request_tpot_ms",
                    "arrival_handling_lag_ms",
                    "queue_inclusive_decode_avg_ms_per_token",
                )
            },
            "token_intervals_ms": stats(t for r in chosen for t in r["token_intervals_ms"]),
        }
    last = max(r["arrival_ns"] for r in rows)
    return {
        "observation_makespan_ms": seconds * 1000,
        "arrival_span_ms": (last - start) / 1e6,
        "observed_arrival_span_ms": (
            max(r["observed_arrival_ns"] for r in rows)
            - min(r["observed_arrival_ns"] for r in rows)
        )
        / 1e6,
        "drain_ms": (end - last) / 1e6,
        "by_task": groups,
        "duration_scope": (
            "one uninterrupted monotonic barrier to all-request/resource "
            "drain; includes trace idle"
        ),
    }


def validate_population(runtime, definitions, limit):
    # The output timestamp is captured at engine return. An independent arrival may
    # be appended before the coordinator publishes that already-observed output.
    events = sorted(runtime["events"], key=lambda e: e["timestamp_ns"])
    actual = {r["request_id"]: r for r in runtime["requests"]}
    state = dict.fromkeys(actual, "STAGED")
    held = set()
    population = []
    for event in events:
        rid = event.get("request_id")
        kind = event["event"]
        if kind == "arrived":
            require(state[rid] == "STAGED", "S2 duplicated arrival")
            state[rid] = "QUEUED"
        elif kind == "admitted":
            require(state[rid] == "QUEUED", "S2 admission before arrival")
            state[rid] = "ACTIVE"
            held.add(rid)
        elif kind == "commit":
            require(state[rid] == "ACTIVE", "S2 commit outside active lifecycle")
        elif kind in ("finished", "finished-in-setup"):
            require(
                state[rid] == ("ACTIVE" if kind == "finished" else "QUEUED"),
                "S2 invalid terminal event",
            )
            state[rid] = "FINISHED"
        elif kind == "resources-released":
            require(state[rid] == "FINISHED" and rid in held, "S2 invalid resource release")
            held.remove(rid)
        require(len(held) <= limit, "S2 common active limit exceeded")
        population.append(
            {
                "timestamp_ns": event["timestamp_ns"],
                "active_held": len(held),
                "queued": sum(v == "QUEUED" for v in state.values()),
            }
        )
    require(
        not held and all(v == "FINISHED" for v in state.values()), "S2 event ledger incomplete"
    )
    return population


def qualify(manifest_path, directory, mode, policy):
    manifest, definitions = load_s2(str(manifest_path))
    report = {
        "schema_version": "specrhythm.s2-result.v1",
        "mode": mode,
        "label": LABEL,
        "measurement_boundary": BOUNDARY,
        "valid": False,
        "execution_sha256": manifest["sha256"],
        "execution_configuration": manifest["execution"],
        "trace_sha256": manifest["trace"]["sha256"],
        "slo_sha256": policy["sha256"] if policy else None,
        "requested_N": manifest["requested_N"],
        "actual_N": len(definitions),
        "class_counts": dict(Counter(r.task_class for r in definitions)),
        "offered_rate_qps": manifest["trace"]["arrival_rate_qps"],
        "arrival_seed": manifest["trace"]["arrival_seed"],
        "order_seed": manifest["trace"]["order_seed"],
        "cross_run_token_length_EOS_round_equality": "NOT_REQUIRED",
        "errors": [],
    }
    artifact = directory / "exit-code.json"
    try:
        exit_code = read_json(artifact)
        require(
            exit_code["effective_exit_code"]
            == exit_code["coordinator_exit_code"]
            == exit_code["draft_exit_code"]
            == 0,
            "S2 nonzero execution exit",
            expected=0,
            actual=exit_code,
        )
        artifact = directory / "process-lifecycle.json"
        lifecycle = read_json(artifact)
        errors = validate_lifecycle_artifact(lifecycle)
        require(
            not errors and lifecycle["run_valid"] is True,
            "S2 owned cleanup invalid",
            expected=[],
            actual=errors,
        )
        artifact = directory / "draft-backend-report.json"
        backend = read_json(artifact)
        checks = draft_backend_checks(backend, artifact)
        checks["s2_live_requests_before_shutdown"] = {
            "field": "s2_live_requests_before_shutdown",
            "expected": 0,
            "actual": backend.get("s2_live_requests_before_shutdown"),
            "artifact": str(artifact),
            "valid": backend.get("s2_live_requests_before_shutdown") == 0,
        }
        report["draft_backend_checks"] = checks
        require(
            all(c["valid"] for c in checks.values()),
            "S2 Draft identity/cleanup failed",
            actual=[c for c in checks.values() if not c["valid"]],
        )
        artifact = directory / "runtime.json"
        runtime = read_json(artifact)
        raw_events = read_json(directory / "arrival-output-events.json")
        require(
            raw_events["events"] == runtime["events"]
            and raw_events["requests"] == runtime["requests"]
            and raw_events["failure"] is None,
            "S2 arrival/output raw evidence differs",
        )
        metrics = request_metrics(
            definitions, runtime, manifest["trace"], manifest["execution"]["eos_token_ids"], policy
        )
        population = validate_population(runtime, definitions, manifest["active_limit"])
        start, end = runtime["start_ns"], runtime["end_ns"]
        require(runtime["draft_shutdown"]["shutdown"] is True, "S2 Draft shutdown RPC missing")
        require(
            len(runtime["target_final_sync"]) == 2
            and all(start <= r["timestamp_ns"] <= end for r in runtime["target_final_sync"]),
            "S2 final TP GPU fence missing",
        )
        ready = load_decode_ready_manifest(read_json(directory / "decode-ready-manifest.json"))
        require(
            ready.workload_sha256 == manifest["workload_sha256"]
            and ready.specrhythm_git_commit == manifest["execution"]["git_commit"],
            "S2 resident identity mismatch",
        )
        pool = read_json(directory / "resident-pool.json")
        require(
            pool["selected_requests"] == len(definitions) and pool["prefill_setup_ns"] > 0,
            "S2 initial pool evidence incomplete",
        )
        native_commits, native_errors = _commit_events(
            "dual-batch" if mode == "pingpong" else mode,
            directory,
            jsonl(directory, "timing-events"),
            start,
        )
        require(not native_errors, "S2 native commit evidence invalid", actual=native_errors)
        for r in runtime["requests"]:
            native = sorted(
                [e for e in native_commits if e["request_id"] == r["request_id"]],
                key=lambda e: e["timestamp_ns"],
            )
            require(
                [t for e in native for t in e["token_ids"]] == r["generated_token_ids"][1:],
                "S2 native worker commits differ from observed outputs",
                request_id=r["request_id"],
            )
            require(
                all(r["admission_ns"] <= e["timestamp_ns"] <= r["completion_ns"] for e in native),
                "S2 native commit outside admission/completion",
                request_id=r["request_id"],
            )
        require(
            all(e["request_id"] in {r.request_id for r in definitions} for e in native_commits),
            "S2 native commit belongs to unknown request",
        )
        # Native worker diagnostics independently validate positions/KV/greedy commits.
        artifact = directory / "target-diagnostics.jsonl"
        rounds = jsonl(
            directory, "round-events" if mode == "serial" else "proposal-events", optional=True
        )
        semantics = validate_rounds(
            rounds,
            runtime["requests"],
            {r.request_id: r.prompt_length for r in definitions},
            manifest["execution"]["eos_token_ids"],
        )
        raw = {r["request_id"]: r for r in runtime["requests"]}
        defs = {r.request_id: r for r in definitions}
        for r in semantics:
            prefix = (
                list(defs[r["request_id"]].prompt_token_ids)
                + raw[r["request_id"]]["generated_token_ids"][
                    : r["prefix_length"] - defs[r["request_id"]].prompt_length
                ]
            )
            require(
                r["prefix_sha256"] == token_prefix_hash(prefix),
                "S2 verification parent hash differs",
            )
        forwards = target_forwards(
            jsonl(directory, "target-diagnostics", optional=True), start, semantics
        )
        require(
            bool(forwards) or not any(r["timed_tokens"] for r in metrics),
            "S2 timed commits lack Target evidence",
        )
        for f in forwards:
            require(
                f["B"] <= manifest["active_limit"] and f["Q"] <= 4096,
                "S2 Target B/Q exceeds common resource contract",
                actual=f,
            )
            require(
                f["end_ns"] <= end
                and all(raw[r]["admission_ns"] <= f["start_ns"] for r in f["request_ids"]),
                "S2 Target execution outside admitted observation interval",
            )
        draft_forwards = draft_measurement(backend, start)
        require(
            all(r["cuda_completion_observed_ns"] <= end for r in draft_forwards),
            "S2 Draft forward beyond drain",
        )
        if mode == "target":
            require(
                not rounds and not draft_forwards, "S2 Target performed timed speculative work"
            )
        if mode == "pingpong":
            from specrhythm.phase4.dual_correctness import (
                validate_draft_sync,
                validate_final_commit_sequence,
                validate_proposal_lifecycle_events,
                validate_request_state_events,
                validate_verification_contracts,
            )

            states = jsonl(directory, "request-state-events")
            verifies = jsonl(directory, "verification-events", optional=not rounds)
            drafts = jsonl(directory, "draft-work-events")
            lifecycle_rows = jsonl(directory, "proposal-lifecycle-events", optional=not rounds)
            require(not validate_request_state_events(states), "S2 PingPong request state invalid")
            terminal_errors = validate_final_commit_sequence(
                {
                    "outputs": [
                        {
                            **r,
                            "final_logical_length": defs[r["request_id"]].prompt_length
                            + len(r["generated_token_ids"]),
                        }
                        for r in runtime["requests"]
                    ]
                },
                ready,
                rounds,
                drafts,
                states,
            )
            require(
                not terminal_errors,
                "S2 final Target/Draft prefix mismatch",
                actual=terminal_errors,
            )
            require(
                not validate_draft_sync(drafts, rounds, states), "S2 PingPong Draft sync invalid"
            )
            lifecycle_errors = (
                validate_proposal_lifecycle_events(lifecycle_rows) if lifecycle_rows else []
            )
            require(
                (bool(lifecycle_rows) or not rounds) and not lifecycle_errors,
                "S2 proposal lifecycle invalid",
                field="proposal_lifecycle_events",
                expected="valid lifecycle for every verified proposal",
                actual=lifecycle_errors
                or {"round_count": len(rounds), "lifecycle_count": len(lifecycle_rows)},
                artifact=str(directory / "proposal-lifecycle-events.jsonl"),
            )
            require(
                not validate_verification_contracts(
                    rounds, verifies, jsonl(directory, "target-diagnostics", optional=True)
                ),
                "S2 Target verification/TP row contract invalid",
            )
            require(
                read_json(directory / "plugin-report.json")["sampled_row_tp_consensus"] is True,
                "S2 sampled-row TP consensus false",
            )
        physical_overlap = {
            "valid": True,
            "observed_overlap_ms": 0.0,
            "definition": "no speculative CUDA stage pair",
            "pair_count": 0,
        }
        if mode == "pingpong" and rounds:
            from specrhythm.phase4.dual_overlap_characterization import characterize_overlap
            from specrhythm.phase4.dual_runner import build_cycle_and_overlap_events

            _, overlap_rows = build_cycle_and_overlap_events(drafts, verifies)
            physical_overlap = characterize_overlap(
                drafts,
                verifies,
                overlap_rows,
                read_json(directory / "actual-capacity.json")["target_worker_ranks"],
            )
            require(
                physical_overlap["valid"],
                "S2 CUDA stage timing evidence invalid",
                actual=physical_overlap.get("errors"),
            )
        # Synchronized host envelopes are an overlap observation, not kernel self time.
        overlaps = []
        for work in backend.get("s2_work_records", []):
            for f in forwards:
                a, b = (
                    max(work["host_start_ns"], f["start_ns"]),
                    min(work["host_end_ns"], f["end_ns"]),
                )
                if b > a:
                    require(
                        not (set(work["request_ids"]) & set(f["request_ids"])),
                        "S2 concurrent Draft/Target ownership collision",
                    )
                    require(
                        all(raw[r]["cohort"] != work["logical_cohort"] for r in f["request_ids"]),
                        "S2 same-cohort stages overlap",
                    )
                    overlaps.append((a, b))
        report.update(
            valid=True,
            metrics=observation_metrics(metrics, start, end),
            requests=[{k: v for k, v in r.items() if k != "token_intervals_ms"} for r in metrics],
            active_limit=manifest["active_limit"],
            pool_size=len(definitions),
            query_token_limit=4096,
            active_held_peak=max(r["active_held"] for r in population),
            verify_batch=stats(f["B"] for f in forwards if f["kind"] == "verify"),
            target_forward_count=len(forwards),
            target_forward_batch=stats(f["B"] for f in forwards),
            target_query_tokens=sum(f["Q"] for f in forwards),
            draft_forward_count=len(draft_forwards),
            draft_batch=stats(r["B"] for r in draft_forwards),
            draft_gpu_event_ms=backend["draft_gpu_event_time_ms"],
            proposed_tokens=sum(len(r["proposal"]) for r in semantics),
            accepted_tokens=sum(len(r["accepted"]) for r in semantics),
            verified_proposal_tokens=sum(len(r["proposal"]) for r in semantics),
            acceptance_rate=(
                sum(len(r["accepted"]) for r in semantics)
                / sum(len(r["proposal"]) for r in semantics)
                if semantics
                else None
            ),
            rejected_tokens=sum(len(r["rejected"]) for r in semantics),
            verified_rounds=len(semantics),
            committed_progress_per_round=stats(len(r["committed"]) for r in semantics),
            overlap_ms=physical_overlap["observed_overlap_ms"],
            physical_overlap=physical_overlap,
            stage_host_overlap_ms=interval_duration(overlaps) / 1e6,
            overlap_definition=(
                "union of disjoint opposite-cohort Draft work / Target forward "
                "host envelopes; synchronized CUDA evidence; not kernel self time"
            ),
            zero_overlap_or_attainment_blocks=False,
            prefill_setup_ns=pool["prefill_setup_ns"],
            target_kv=runtime["target_pool_final"],
            draft_kv=backend["s2_resident_pool"],
            target_rank_memory=runtime["target_final_memory"],
            target_rank_initial_memory=pool["target_rank_initial_memory"],
            draft_memory=backend["s2_memory_final"],
            population=population,
        )
        write_once(directory / "population.json", {"rows": population}) if not (
            directory / "population.json"
        ).exists() else None
    except (ValueError, KeyError, TypeError, OSError, AssertionError) as error:
        report["valid"] = False
        report["errors"].append(str(error))
        report["primary_error"] = {
            "field": str(artifact.name),
            "expected": "valid complete execution evidence",
            "actual": str(error),
            "artifact": str(artifact),
            **getattr(error, "details", {}),
        }
    return report


def seal_result(directory, report):
    write_once(directory / "result.json", report)
    files = {
        p.name: sha256_file(p)
        for p in sorted(directory.iterdir())
        if p.is_file() and p.name != "seal.json"
    }
    write_once(
        directory / "seal.json",
        sealed({"schema_version": "specrhythm.s2-seal.v1", "files": files}),
    )


def verify_result(directory):
    seal = read_json(directory / "seal.json")
    check_seal(seal)
    for name, sha in seal["files"].items():
        require(
            sha256_file(directory / name) == sha,
            "S2 sealed evidence changed",
            artifact=str(directory / name),
        )
    return read_json(directory / "result.json")


def light_summary(root):
    results = []
    for p in sorted(root.glob("**/seal.json")):
        value = verify_result(p.parent)
        reduced = {
            k: v
            for k, v in value.items()
            if k not in ("requests", "population", "target_kv", "draft_kv", "target_rank_memory")
        }
        for role in ("target_kv", "draft_kv"):
            if role in value:
                reduced[role] = {k: v for k, v in value[role].items() if k != "initial"}
        if "target_rank_memory" in value:
            reduced["target_rank_memory"] = [
                {"rank": r.get("rank"), **r["s2_capacity"]} for r in value["target_rank_memory"]
            ]
        reduced["raw_evidence"] = {
            "directory": str(p.parent),
            "seal_sha256": sha256_file(p),
            "files": read_json(p)["files"],
        }
        results.append(reduced)
    inputs = {}
    for name in ("preparation", "selection", "capacity-plan", "slo-policy", "g0"):
        path = root / (name + ".json")
        if path.exists():
            value = read_json(path)
            check_seal(value)
            inputs[name] = {"path": str(path), "file_sha256": sha256_file(path)}
            if name == "capacity-plan":
                inputs[name]["sizes"] = {
                    size: {
                        k: v
                        for k, v in row.items()
                        if k not in ("request_ids", "capacity_attempts")
                    }
                    for size, row in value["sizes"].items()
                }
            elif name == "slo-policy":
                inputs[name]["policy"] = value
    return {
        "schema_version": "specrhythm.s2-light-summary.v1",
        "label": LABEL,
        "boundary": BOUNDARY,
        "cross_run_equality": "NOT_REQUIRED",
        "frozen_inputs": inputs,
        "unsealed_attempts": [
            {"directory": str(p), "valid": False, "status": "requires owned recovery"}
            for p in sorted(root.glob("runs/**/attempt-*"))
            if p.is_dir() and not (p / "seal.json").exists()
        ],
        "results": results,
        "comparison_rule": (
            "report actual output tokens, tok/s, SLO goodput and makespan "
            "together; no equal-work speedup inferred"
        ),
    }
