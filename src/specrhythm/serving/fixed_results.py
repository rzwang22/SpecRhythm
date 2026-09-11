"""Linear/sorted light summaries, separate CPU audit, and honest stage comparisons."""

from __future__ import annotations

import bisect
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from specrhythm.phase4.process_lifecycle import validate_lifecycle_artifact
from specrhythm.serving.common import DataError, read_json, require
from specrhythm.serving.fixed_plan import POLICY, SCENARIO
from specrhythm.serving.runtime_profile import load_s2
from specrhythm.serving.s1_results import draft_backend_checks
from specrhythm.serving.s1_workload import write_once


def union(intervals):
    result = []
    for a, b in sorted(intervals):
        require(
            type(a) in (int, float)
            and type(b) in (int, float)
            and math.isfinite(a)
            and math.isfinite(b)
            and a <= b,
            "invalid observed interval",
            actual=[a, b],
        )
        if a == b:
            continue
        if result and a <= result[-1][1]:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return result


def duration(intervals):
    return sum(b - a for a, b in union(intervals)) / 1e6


def intersections(left, right):
    """Two sorted unions, O(n log n + m log m), never a Cartesian product."""
    left, right = union(left), union(right)
    i = j = 0
    result = []
    while i < len(left) and j < len(right):
        a, b = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if b > a:
            result.append((a, b))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return result


def clipped(intervals, start, end):
    return intersections(intervals, [(start, end)])


def stats(values, reason="no eligible samples"):
    values = sorted(values)
    return {
        "count": len(values),
        "mean": mean(values) if values else None,
        "p50": median(values) if values else None,
        "p90": values[max(0, math.ceil(0.9 * len(values)) - 1)] if values else None,
        "max": max(values) if values else None,
        "reason": None if values else reason,
    }


def positions(row):
    q, k, root = row["query_positions"], row["candidate_positions"], row["base_root_positions"]
    require(
        type(q) is int and type(k) is int and 0 <= k <= 4 and root == 1 and q == k + root,
        "Target query/base/candidate conservation failed",
        actual=row,
    )
    require(
        row["position_end_exclusive"] - row["position_start"] == q,
        "Target position range differs from query count",
        actual=row,
    )
    return q, k, root


def rounds_by_prefix(rows):
    result = {}
    for r in rows:
        prefix = r.get("parent_prefix_len", r.get("prefix_token_count"))
        key = (r["request_id"], prefix)
        require(key not in result, "duplicate committed verification", actual=key)
        p, a, rejected = (
            r[k]
            for k in ("proposal_token_ids", "accepted_draft_token_ids", "rejected_draft_token_ids")
        )
        c, bonus = r["target_correction_token_ids"], r["target_bonus_token_ids"]
        require(
            a == p[: len(a)]
            and rejected == p[len(a) :]
            and r["committed_token_ids"] == a + c + bonus
            and r["committed_token_ids"]
            and len(c) + len(bonus) <= 1
            and len(a) == r["accepted_draft_tokens"]
            and len(rejected) == r["rejected_draft_tokens"],
            "verification token accounting failed",
            actual=r,
        )
        result[key] = r
    return result


def device_batches(devices, steps):
    """Join each real GPU forward to the one containing synchronous Target step."""
    starts = [s["start_ns"] for s in steps]
    by_step = defaultdict(dict)
    for evidence in devices:
        device = evidence["device"]
        rank = device["identity"]["global_rank"]
        for f in device["forwards"]:
            idx = bisect.bisect_right(starts, f["host_start_ns"]) - 1
            if idx < 0 or f["host_start_ns"] > steps[idx]["end_ns"]:
                continue  # Real setup forward, outside decode steps.
            require(
                rank not in by_step[idx],
                "multiple Target model forwards in one step",
                actual={"step": idx, "rank": rank},
            )
            require(
                type(f["gpu_event_ms"]) in (int, float)
                and math.isfinite(f["gpu_event_ms"])
                and f["gpu_event_ms"] > 0,
                "Target model forward lacks finite positive CUDA timing",
                actual=f,
            )
            require(
                set(f["internal_request_ids"])
                == {r["internal_request_id"] for r in steps[idx]["rows"]},
                "Target TP actual request-row identity differs",
                actual=f,
            )
            require(
                f["B"] == steps[idx]["B"],
                "Target physical/scheduled batch mismatch",
                actual=f,
                expected=steps[idx]["B"],
            )
            by_step[idx][rank] = f
    for idx, step in enumerate(steps):
        if step["B"]:
            require(
                set(by_step[idx]) == {0, 1},
                "Target forward lacks both TP rank CUDA events",
                actual={"step": idx, "ranks": list(by_step[idx])},
            )
    return by_step


def execution_checks(runtime, definitions, backend, lifecycle, eos=()):
    require(
        not validate_lifecycle_artifact(lifecycle)
        and lifecycle["run_valid"]
        and lifecycle["cleanup_valid"]
        and lifecycle.get("owned_cleanup_completed") is True,
        "diagnostic owned execution/cleanup failed",
        artifact="process-lifecycle.json",
    )
    checks = draft_backend_checks(backend, Path("draft-backend-report.json"))
    require(
        all(c["valid"] for c in checks.values()),
        "diagnostic backend/cleanup failed",
        actual=checks,
        artifact="draft-backend-report.json",
    )
    if runtime.get("probe"):
        return checks
    ids = {r.request_id: r for r in definitions}
    rows = runtime["requests"]
    require(
        len(rows) == len(ids) and {r["request_id"] for r in rows} == set(ids),
        "diagnostic request accounting incomplete",
    )
    releases = {
        e["request_id"]: e["timestamp_ns"]
        for e in runtime["events"]
        if e["event"] in ("resources-released", "cancelled-resources-released")
    }
    if runtime.get("schema_version") == "specrhythm.fixed-runtime.v2":
        settlement_checks(runtime, ids, backend, releases)
    for r in rows:
        rid = r["request_id"]
        require(
            r["state"] in ("FINISHED", "DIAGNOSTIC_CANCELLED") and r["resources_released"],
            "diagnostic request remains live/failed",
            request_id=rid,
        )
        require(
            1 <= len(r["generated_token_ids"]) <= ids[rid].maximum_new_tokens,
            "diagnostic output budget invalid",
            request_id=rid,
        )
        require(
            sum(len(c["token_ids"]) for c in r["commits"]) == len(r["generated_token_ids"]) - 1,
            "diagnostic committed-token count differs",
            request_id=rid,
        )
        if r.get("admission_ns") is not None:
            require(
                rid in releases
                and releases[rid] <= runtime["end_ns"]
                and releases[rid] >= r.get("completion_ns", runtime["measurement_end_ns"]),
                "diagnostic resource release/order evidence missing",
                request_id=rid,
            )
        tokens = r["generated_token_ids"]
        require(
            not any(t in eos for t in tokens[:-1]), "diagnostic output after EOS", request_id=rid
        )
        natural = len(tokens) == ids[rid].maximum_new_tokens or tokens[-1] in eos
        require(
            (r["state"] == "FINISHED") == natural,
            "diagnostic natural termination/cancellation status differs",
            request_id=rid,
        )
        if r["state"] == "DIAGNOSTIC_CANCELLED":
            require("completion_ns" not in r, "diagnostic cancellation fabricated completion")
    require(runtime["target_requests_final"] == 0, "Target request cleanup incomplete")
    return checks


def settlement_checks(runtime, definitions, backend, releases):
    from specrhythm.phase4.serial import token_prefix_hash

    drain = runtime["diagnostic_drain"]
    receipts = drain["receipts"]
    require(
        drain["status"] == "COMPLETE"
        and drain["settled_requests"] == len(definitions)
        and len(receipts) == len(definitions)
        and {r["request_id"] for r in receipts} == set(definitions)
        and backend.get("s2_live_requests_before_shutdown") == 0,
        "diagnostic settlement/physical pre-shutdown release incomplete",
        artifact="drain-state.json / draft-backend-report.json",
    )
    require(
        runtime["measurement_end_ns"] <= drain["start_ns"] < drain["end_ns"] <= runtime["end_ns"]
        and drain["end_ns"] <= drain["deadline_ns"],
        "diagnostic measurement/drain/deadline boundaries differ",
        artifact="drain-state.json",
    )
    final = {r["request_id"]: r for r in runtime["requests"]}
    for receipt in receipts:
        rid = receipt["request_id"]
        r = final[rid]
        prefix = (*definitions[rid].prompt_token_ids, *r["generated_token_ids"])
        require(
            receipt["released"] is True
            and receipt["new_proposals_generated"] == 0
            and receipt["authoritative_prefix_hash"] == token_prefix_hash(prefix)
            and receipt["authoritative_prefix_count"] == len(prefix)
            and receipt["natural_terminal"] == (r["state"] == "FINISHED")
            and receipt["disposition"]
            == ("NATURAL_TERMINAL" if r["state"] == "FINISHED" else "DIAGNOSTIC_CANCELLED")
            and receipt["resources_released_ns"] <= runtime["end_ns"],
            "diagnostic settlement authority/disposition differs",
            request_id=rid,
            artifact="drain-state.json",
        )
        if r.get("admission_ns") is not None:
            require(
                receipt["resources_released_ns"] <= releases.get(rid, -1),
                "logical release precedes physical Draft release",
                request_id=rid,
                artifact="drain-state.json / arrival-output-events.json",
            )


def summarize(manifest_path, directory, point, *, probe=False):
    manifest, definitions = load_s2(str(manifest_path))
    base = {
        "schema_version": "specrhythm.fixed-result.v1",
        "mode": point["mode"],
        "point": point,
        "scenario": SCENARIO,
        **POLICY,
        "execution_sha256": manifest["sha256"],
        "effective_exit_code": 0,
        "full_offline_audit_status": "PENDING",
        "artifact": str(directory),
    }
    try:
        runtime = read_json(directory / "runtime.json")
        base["capacity"] = runtime["capacity"]
        require(bool(runtime.get("probe")) == probe, "capacity/execution artifact kind differs")
        if not probe:
            require(runtime["point"] == point, "runtime mode/point identity differs")
        backend = read_json(directory / "draft-backend-report.json")
        lifecycle = read_json(directory / "process-lifecycle.json")
        checks = execution_checks(
            runtime, definitions, backend, lifecycle, manifest["execution"]["eos_token_ids"]
        )
        base.update(draft_backend_checks=checks, execution_status="PASS")
        if probe:
            return {
                **base,
                "valid": True,
                "errors": [],
                "probe": True,
                "measurement_status": "NOT_APPLICABLE",
            }
        if runtime["measurement_start_ns"] is None or not any(
            s["window"] and s["B"] for s in runtime["target_steps"]
        ):
            return {
                **base,
                "valid": True,
                "errors": [],
                "measurement_status": "INSUFFICIENT",
                "reason": "stopped before any measured forward/commit",
                "valid_samples": 0,
                "window_throughput_tok_s": None,
                "stop_reason": runtime["stop_reason"],
            }
        report = measurements(manifest, runtime, backend, point)
        if runtime["stop_reason"] == "operator_stop":
            report.update(
                measurement_status="INSUFFICIENT",
                reason="operator stopped before the requested budget completed",
            )
        if point["kind"] == "initial-state" and report["stage_shape"]["status"] != "PASS":
            report["measurement_status"] = "INSUFFICIENT"
        return {**base, **report, "valid": True, "errors": []}
    except (DataError, KeyError, TypeError, ValueError) as error:
        if isinstance(error, KeyError):
            error = DataError(
                "missing required diagnostic evidence field",
                field=str(error),
                artifact=str(directory),
                expected="present",
                actual="missing",
            )
        return {
            **base,
            "valid": False,
            "errors": [str(error)],
            "execution_status": "FAILED",
            "measurement_status": "INVALID",
            "error_details": getattr(error, "details", {}),
            "primary_error": {
                "field": "diagnostic evidence",
                "expected": "valid",
                "actual": str(error),
                "artifact": str(directory),
                **getattr(error, "details", {}),
            },
        }


def measurements(manifest, runtime, backend, point):
    start, end = runtime["measurement_start_ns"], runtime["measurement_end_ns"]
    require(start is not None and end > start, "no positive diagnostic measurement window")
    wall_ms = (end - start) / 1e6
    steps = runtime["target_steps"]
    devices = runtime["target_devices"]
    require(
        len(devices) == 2 and {d["device"]["identity"]["global_rank"] for d in devices} == {0, 1},
        "diagnostic missing TP worker evidence",
    )
    require(
        len(
            {d["device"]["identity"]["gpu_uuid"] for d in devices}
            | {backend["fixed_device"]["identity"]["gpu_uuid"]}
        )
        == 3,
        "diagnostic GPU identities are not distinct",
    )
    rounds = rounds_by_prefix([r for d in devices for r in d["rounds"]])
    final = {r["request_id"]: r for r in runtime["requests"]}

    # Workload-independent prefix offsets come from the retained setup artifact's
    # bootstrap count indirectly: each runtime row retains only generated tokens.
    # Explicit prompt lengths are frozen into the runtime below.
    for (rid, prefix), r in rounds.items():
        offset = prefix - runtime["prompt_lengths"][rid]
        require(
            offset >= 1
            and r["committed_token_ids"]
            == final[rid]["generated_token_ids"][offset : offset + len(r["committed_token_ids"])],
            "verification commits differ from actual coordinator output",
            actual=r,
        )

    batches = device_batches(devices, steps)
    diagnostics = {}
    for d in devices:
        for r in d["target_rows"]:
            if r["target_forward_start_ns"] < runtime["start_ns"]:
                continue
            require(
                not r["structural_errors"],
                "Target structural validation failed",
                request_id=r["request_id"],
                actual=r["structural_errors"],
                artifact="target-diagnostics.jsonl",
            )
            key = (r["request_id"], r["context_length"])
            require(
                key not in diagnostics or diagnostics[key] == r,
                "TP Target diagnostic disagreement",
                actual=key,
            )
            diagnostics[key] = r
    accepted = rejected = proposed = committed_verified = verifies = 0
    q = roots = 0
    target_samples, rotations = [], []
    pending_cohort = None
    rotation_boundary = start
    for i, s in enumerate(steps):
        if not s["B"]:
            continue
        sample_q = sample_k = sample_roots = 0
        progress = 0
        for r in s["rows"]:
            rq, rk, rr = positions(r)
            sample_q += rq
            sample_k += rk
            sample_roots += rr
            key = (r["request_id"], r["context_length"])
            require(
                key in diagnostics, "Target scheduled row lacks actual input diagnostic", actual=r
            )
            diag = diagnostics[key]
            require(
                diag["query_length"] == rq
                and len(diag["proposal_token_ids"]) == rk
                and diag["position_ids"]
                == list(range(r["position_start"], r["position_end_exclusive"])),
                "actual Target input positions differ from scheduler",
                actual=r,
            )
            if rk:
                require(key in rounds, "Target speculative row lacks committed round", actual=r)
                commit = rounds[key]
                require(
                    len(commit["proposal_token_ids"]) == rk,
                    "Target candidates differ from accepted proposal",
                    actual=r,
                )
                g = len(commit["committed_token_ids"])
                progress += g
                if s["window"]:
                    accepted += len(commit["accepted_draft_token_ids"])
                    rejected += len(commit["rejected_draft_token_ids"])
                    proposed += rk
                    committed_verified += g
                    verifies += 1
            else:
                progress += 1
        if not s["window"]:
            continue
        q += sample_q
        roots += sample_roots
        by_rank = {str(rank): value["gpu_event_ms"] for rank, value in batches[i].items()}
        sample = {
            "B": s["B"],
            "Q": sample_q,
            "candidate_positions": sample_k,
            "base_root_positions": sample_roots,
            "request_ids": s["request_ids"],
            "context_lengths": [r["context_length"] for r in s["rows"]],
            "candidate_lengths": [r["candidate_positions"] for r in s["rows"]],
            "gpu_event_ms_by_rank": by_rank,
            "gpu_event_ms": max(by_rank.values()),
            "tp_aggregation": "max rank duration, not sum",
            "kind": "V_SD" if sample_k else "V_AR",
            "cohort": s["cohort"],
            "start_ns": s["start_ns"],
            "end_ns": s["end_ns"],
            "step_wall_ms": (s["end_ns"] - s["start_ns"]) / 1e6,
            "committed_progress": progress,
            "g": progress / s["B"],
        }
        target_samples.append(sample)
        if s["B"] == 32 and s["cohort"] in ("A", "B"):
            if pending_cohort and pending_cohort["cohort"] != s["cohort"]:
                rotations.append((s["end_ns"] - rotation_boundary) / 1e6)
                rotation_boundary = s["end_ns"]
                pending_cohort = None
            else:
                pending_cohort = s
        else:
            pending_cohort = None
            if s["B"] == 64 and s["cohort"] is None:
                rotations.append((s["end_ns"] - rotation_boundary) / 1e6)
            rotation_boundary = s["end_ns"]
    tokens = sum(
        len(c["token_ids"])
        for r in runtime["requests"]
        for c in r["commits"]
        if start <= c["timestamp_ns"] <= end
    )
    require(tokens > 0 and target_samples, "diagnostic has no measured committed progress")
    measured_progress = sum(s["committed_progress"] for s in target_samples)
    measured_requests = sum(s["B"] for s in target_samples)
    require(
        tokens == measured_progress,
        "measured Target progress differs from actual window commits",
        expected=measured_progress,
        actual=tokens,
    )
    draft = [
        r for r in backend["fixed_proposals"] if r["start_ns"] >= start and r["end_ns"] <= end
    ]
    draft_forwards = [
        r for r in backend["fixed_device"]["forwards"] if r["purpose"] in ("proposal", "commit")
    ]
    for p in backend["fixed_proposals"]:
        # Sorted cumulative sums avoid proposal x forward joins.
        p["wall_ms"] = (p["end_ns"] - p["start_ns"]) / 1e6
    starts = [f["host_start_ns"] for f in draft_forwards]
    cumulative = [0.0]
    for f in draft_forwards:
        require(
            type(f["gpu_event_ms"]) in (int, float)
            and math.isfinite(f["gpu_event_ms"])
            and f["gpu_event_ms"] > 0,
            "Draft forward lacks finite positive CUDA timing",
        )
        cumulative.append(cumulative[-1] + f["gpu_event_ms"])
    for p in backend["fixed_proposals"]:
        a, b = bisect.bisect_left(starts, p["start_ns"]), bisect.bisect_right(starts, p["end_ns"])
        p["gpu_event_ms"] = cumulative[b] - cumulative[a]
    overlap = overlap_metrics(devices, backend["fixed_device"], start, end)
    if point["mode"] == "serial-split":
        require(
            overlap["event_overlap_lower_ms"] == 0,
            "serial-split has physical Draft/Target overlap",
            actual=overlap,
        )
        # Native Target diagnostics span completed forwards. Owner operation spans
        # include its required completion fence, so this is also a strict host gate.
        draft_work = [(r["host_start_ns"], r["host_end_ns"]) for r in backend["s2_work_records"]]
        target_work = [
            (r["target_forward_start_ns"], r["target_forward_end_ns"])
            for r in diagnostics.values()
        ]
        require(
            not clipped(intersections(draft_work, target_work), start, end),
            "serial-split owner work overlaps Target forward",
            artifact="draft-backend-report.json",
        )
    host = host_summary(runtime, backend, start, end)
    supply = population_metrics(runtime, start, end)
    intervals = [b["start_ns"] - a["end_ns"] for a, b in zip(target_samples, target_samples[1:])]
    request_tpots, token_intervals = [], []
    for r in runtime["requests"]:
        times = [c["timestamp_ns"] for c in r["commits"] if start <= c["timestamp_ns"] <= end]
        token_intervals.extend((b - a) / 1e6 for a, b in zip(times, times[1:]))
        if r["state"] == "FINISHED" and len(r["commits"]) >= 2:
            count = len(r["generated_token_ids"]) - 1
            request_tpots.append(
                (r["completion_ns"] - r["commits"][0]["timestamp_ns"])
                / 1e6
                / (count - len(r["commits"][0]["token_ids"]))
            )
    target_intervals = [(s["start_ns"], s["end_ns"]) for s in target_samples]
    return {
        "measurement_status": "PASS",
        "valid_samples": len(target_samples),
        "warmup_steps": runtime["warmup_steps"],
        "warmup_ms": (runtime["warmup_end_ns"] - runtime["warmup_start_ns"]) / 1e6,
        "startup_and_state_preparation_ms": runtime.get("startup_and_state_preparation_ms"),
        "reset_method": "new process, fresh engine and real prefill at each point",
        "reset_ms_separate": None,
        "reset_ms_reason": "restoration is included in fresh startup/state preparation",
        "stop_reason": runtime["stop_reason"],
        "window_ms": wall_ms,
        "committed_window_tokens": tokens,
        "window_throughput_tok_s": tokens * 1000 / wall_ms,
        "arrival_to_drain_ms": (runtime["end_ns"] - runtime["start_ns"]) / 1e6,
        "drain_ms": (runtime["end_ns"] - end) / 1e6,
        "natural_completed_requests": sum(r["state"] == "FINISHED" for r in runtime["requests"]),
        "diagnostic_cancelled_requests": sum(
            r["state"] == "DIAGNOSTIC_CANCELLED" for r in runtime["requests"]
        ),
        "full_request_tpot_ms": stats(request_tpots, "insufficient naturally completed requests"),
        "window_commit_interval_ms": stats(token_intervals, "insufficient commit batches"),
        "window_commit_interval_semantics": "commit batches; individual token times unavailable",
        "full_request_slo_attainment": None,
        "full_request_slo_reason": "short diagnostic window with controlled cancellations",
        "actual_target_batch": stats([s["B"] for s in target_samples]),
        "actual_target_batch_histogram": dict(Counter(s["B"] for s in target_samples)),
        "actual_draft_forward_batch": stats(
            [r["B"] for r in draft_forwards if start <= r["host_start_ns"] <= end]
        ),
        **supply,
        "full_target_batch_fraction": mean(
            s["B"]
            == manifest["fixed_diagnostic"]["capacity"][point["mode"]][
                "max_requests_per_target_forward"
            ]
            for s in target_samples
        ),
        "target_step_gap_ms": stats([n / 1e6 for n in intervals]),
        "target_model_forward_gap_ms_by_rank": {
            str(d["device"]["identity"]["global_rank"]): forward_gaps(
                d["device"]["forwards"], start, end
            )
            for d in devices
        },
        "draft_model_forward_gap_ms": forward_gaps(draft_forwards, start, end),
        "concurrent_initial_sample": concurrent_sample(
            manifest, point, target_samples, backend["fixed_proposals"]
        ),
        "accepted_tokens": accepted,
        "rejected_tokens": rejected,
        "verified_proposed_tokens": proposed,
        "committed_verified_tokens": measured_progress if point["mode"] != "target" else 0,
        "verification_request_count": measured_requests if point["mode"] != "target" else 0,
        "proposal_verification_request_count": verifies,
        "committed_proposal_verification_tokens": committed_verified,
        "g_per_request_per_verification": (
            measured_progress / measured_requests if point["mode"] != "target" else None
        ),
        "g_semantics": "own actual committed progress, including base-only terminal steps",
        "target_forward_count": len(target_samples),
        "target_query_positions": q,
        "target_base_root_positions": roots,
        "draft_forward_count": sum(start <= r["host_start_ns"] <= end for r in draft_forwards),
        "draft_issued_candidate_tokens_including_drain": sum(
            sum(r["candidate_lengths"]) for r in backend["fixed_proposals"]
        ),
        "actual_rotation_ms": stats(rotations, "no complete full A/B rotation"),
        "target_samples": target_samples,
        "draft_samples": draft,
        "overlap": overlap,
        "host_observation": host,
        "target_step_union_ms": duration(clipped(target_intervals, start, end)),
        "uuid_query_by_rank": [
            r["dual_uuid_query"]
            for r in runtime.get("target_final_memory", [])
            if "dual_uuid_query" in r
        ],
        "pipeline_stage_gpu_event_ms": {
            **{
                kind + str(b): stats(
                    [
                        s["gpu_event_ms"]
                        for s in target_samples
                        if s["kind"] == kind
                        and s["B"] == b
                        and all(k == (4 if kind == "V_SD" else 0) for k in s["candidate_lengths"])
                    ]
                )
                for kind in ("V_AR", "V_SD")
                for b in (32, 64)
            },
            **{
                "D" + str(b): stats(
                    [
                        s["gpu_event_ms"]
                        for s in draft
                        if s["B"] == b and all(k == 4 for k in s["candidate_lengths"])
                    ]
                )
                for b in (32, 64)
            },
        },
        "pipeline_interpretation": "measured event/host overlap is not critical-path time saved",
        "dtype_attention_identity": [d["device"]["identity"] for d in devices]
        + [backend["fixed_device"]["identity"]],
        "stage_shape": stage_shape(manifest, point, target_samples, draft),
    }


def population_metrics(runtime, start, end):
    """Sweep real admission/completion/release times, including inter-step waits."""
    releases = {
        e["request_id"]: e["timestamp_ns"]
        for e in runtime["events"]
        if e["event"] in ("resources-released", "cancelled-resources-released")
    }
    deltas = defaultdict(lambda: [0, 0])  # Active and held are separate populations.
    deltas[runtime["start_ns"]]
    deltas[end]
    for row in runtime["requests"]:
        admitted = row.get("admission_ns")
        if admitted is None or admitted >= end:
            continue
        rid = row["request_id"]
        finished = min(row.get("completion_ns", end), end)
        released = min(releases[rid], end)
        require(admitted <= finished <= released, "invalid diagnostic slot lifetime", actual=rid)
        deltas[admitted][0] += 1
        deltas[admitted][1] += 1
        deltas[finished][0] -= 1
        deltas[released][1] -= 1
    ordered = sorted(deltas)
    active = held = 0
    seen_full = False
    phase_ms = {p: 0.0 for p in ("fill", "full-load", "tail")}
    full_held_ms = 0.0
    for a, b in zip(ordered, ordered[1:]):
        da, dh = deltas[a]
        active, held = active + da, held + dh
        require(
            0 <= active <= held <= 64,
            "diagnostic active/held slot limit violated",
            actual={"timestamp_ns": a, "active": active, "held": held},
        )
        seen_full |= active == 64
        elapsed = max(0, min(b, end) - max(a, start)) / 1e6
        phase = "full-load" if active == 64 else ("tail" if seen_full else "fill")
        phase_ms[phase] += elapsed
        if held == 64:
            full_held_ms += elapsed
    return {
        "supply_phase_ms": phase_ms,
        "supply_phase_semantics": (
            "64 active is full-load; pre-full fill and post-full tail; includes waits"
        ),
        "full_held_load_fraction": full_held_ms / ((end - start) / 1e6),
        "full_active_load_fraction": phase_ms["full-load"] / ((end - start) / 1e6),
    }


def overlap_metrics(devices, draft, start, end):
    def intervals(rows, inner):
        a, b = ("start_upper_ns", "end_lower_ns") if inner else ("start_lower_ns", "end_upper_ns")
        return clipped([(r[a], r[b]) for r in rows if r[b] > r[a]], start, end)

    targets = [f for d in devices for f in d["device"]["forwards"]]
    drafts = [f for f in draft["forwards"] if f.get("purpose") in ("proposal", "commit")]
    low = duration(intersections(intervals(targets, True), intervals(drafts, True)))
    high = duration(intersections(intervals(targets, False), intervals(drafts, False)))
    return {
        "event_overlap_lower_ms": low,
        "event_overlap_upper_ms": high,
        "physical_overlap_status": "OBSERVED"
        if low > 0
        else ("ZERO" if high == 0 else "UNRESOLVED_WITHIN_ANCHOR_UNCERTAINTY"),
        "definition": "union across TP ranks, intersection with Draft union; clock-anchor bounds",
        "exact_kernel_overlap": False,
        "critical_path_time_saved": None,
    }


def host_summary(runtime, backend, start, end):
    by_category = defaultdict(list)
    sources = [runtime["host"], backend["fixed_host"]] + [
        r["host"] for r in runtime["target_devices"]
    ]
    for source in sources:
        for row in source["intervals"]:
            by_category[row["category"]].append((row["start_ns"], row["end_ns"]))
    for category in (
        "scheduler",
        "ipc",
        "live_uuid_validation",
        "nvidia_smi_subprocess",
        "resident_block_audit",
        "prefix_hash_and_block_record",
        "control_json_read",
        "control_json_write",
        "json_serialization",
        "required_cuda_synchronize",
        "required_cuda_event_synchronize",
        "required_tp_barrier",
        "log_fsync",
        "checkpoint_log_write",
        "target_diagnostic_contract_scan",
    ):
        by_category.setdefault(category, [])
    return {
        key: {
            "count": sum(a <= end and b >= start for a, b in rows),
            "union_host_ms": duration(clipped(rows, start, end)),
        }
        for key, rows in by_category.items()
    }


def stage_shape(manifest, point, target, draft):
    if point["kind"] != "initial-state":
        return {"status": "NOT_APPLICABLE", "reason": "continuous decode, separate population"}
    diag = manifest["fixed_diagnostic"]
    ids = diag["initial_request_ids"] if point["batch"] == 64 else diag["cohorts"][point["half"]]
    selected = [s for s in target if set(s["request_ids"]) == set(ids)]
    valid = len(selected) == 1 and all(
        k == (0 if point["mode"] == "target" else 4) for k in selected[0]["candidate_lengths"]
    )
    contexts = (
        dict(zip(selected[0]["request_ids"], selected[0]["context_lengths"]))
        if len(selected) == 1
        else {}
    )
    # Draft EOS may shorten individual candidates; do not pretend it was a full K4 sample.
    d = [
        s
        for s in draft
        if set(s["request_ids"]) == set(ids)
        and dict(zip(s["request_ids"], s["context_lengths"])) == contexts
        and set(s["round_ids"]) == {0}
        and all(k == 4 for k in s["candidate_lengths"])
    ]
    return {
        "status": "PASS" if valid else "INSUFFICIENT",
        "reason": None if valid else "actual IDs, B or effective K differs from requested shape",
        "expected_request_ids": ids,
        "target_gpu_event_ms": selected[0]["gpu_event_ms"] if valid else None,
        "draft_gpu_event_ms": d[0]["gpu_event_ms"] if len(d) == 1 else None,
        "draft_wall_ms": d[0]["wall_ms"] if len(d) == 1 else None,
        "draft_reason": None if len(d) == 1 else "no complete initial K4 proposal in window",
        "actual_target_request_ids": selected[0]["request_ids"] if selected else None,
        "context_lengths": [contexts[rid] for rid in ids] if contexts else None,
        "kind": "fresh initial-state microbenchmark; no continuation replay",
    }


def emit_result(directory, report, point):
    from specrhythm.serving.fixed_artifacts import retained_report

    report = retained_report(directory, report)
    report = {
        **report,
        "point": point,
        "mode": point["mode"],
        **POLICY,
        "full_offline_audit_status": "PENDING",
    }
    if "execution_status" not in report:
        report.update(execution_status="FAILED", measurement_status="INVALID")
    write_once(directory / "result.json", report)
    # Keep large sample tables out of the immediate JSON/CSV and default small bundle.
    light = {k: v for k, v in report.items() if k not in ("target_samples", "draft_samples")}
    write_once(directory / "light-summary.json", light)
    fields = (
        "mode",
        "execution_status",
        "measurement_status",
        "valid_samples",
        "window_ms",
        "committed_window_tokens",
        "window_throughput_tok_s",
        "g_per_request_per_verification",
        "draft_forward_count",
        "target_forward_count",
        "target_query_positions",
        "accepted_tokens",
        "verified_proposed_tokens",
        "rejected_tokens",
    )
    with (directory / "light-summary.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({k: light.get(k) for k in fields})
    print(
        "[fixed diagnostic] "
        + json.dumps(
            {
                **{k: light.get(k) for k in fields},
                "actual_batch": light.get("actual_target_batch"),
                "stage": light.get("stage_shape"),
                "pipeline_stage_gpu_event_ms": light.get("pipeline_stage_gpu_event_ms"),
                "actual_rotation_ms": light.get("actual_rotation_ms"),
                "effective_exit_code": light.get("effective_exit_code"),
                "primary_error": light.get("primary_error"),
                "measurement_availability": light.get("measurement_availability"),
                "partial_window_tokens": light.get("measurement_snapshot", {}).get(
                    "committed_window_tokens"
                )
                if not light.get("valid")
                else None,
                "artifact": str(directory),
                "full_offline_audit_status": "PENDING",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return report


def comparisons(root):
    from specrhythm.serving.fixed_artifacts import point_reports

    reports = point_reports(root)
    continuous = defaultdict(list)
    stages = defaultdict(list)
    for r in reports:
        if not r.get("valid") or r.get("probe") or r.get("point", {}).get("discard_warmup"):
            continue
        if r.get("measurement_status") != "PASS":
            continue  # A clean stop without samples is not a numeric timing population.
        if r["point"]["kind"] == "continuous":
            continuous[r["mode"]].append(r)
        elif r.get("stage_shape", {}).get("status") == "PASS":
            shape = r["stage_shape"]
            b = r["point"]["batch"]
            if r["mode"] in ("target", "serial"):
                stages[("V_AR" if r["mode"] == "target" else "V_SD") + str(b)].append(
                    shape["target_gpu_event_ms"]
                )
            if r["mode"] == "serial" and shape["draft_gpu_event_ms"] is not None:
                stages["D" + str(b)].append(shape["draft_gpu_event_ms"])
    stage_values = {
        name: stats(stages[name], "no qualified initial-state B/K sample")
        for name in ("D32", "D64", "V_SD32", "V_SD64", "V_AR32", "V_AR64")
    }
    stage_means = {k: v["mean"] for k, v in stage_values.items()}
    pairs = {}
    for before, after in (
        ("serial", "serial-split"),
        ("serial-split", "pingpong"),
        ("serial", "pingpong"),
        ("target", "serial"),
        ("target", "serial-split"),
        ("target", "pingpong"),
    ):
        a, b = continuous[before], continuous[after]
        key = before + " -> " + after
        if not a or not b:
            pairs[key] = {"status": "PENDING", "reason": "both measured modes required"}
            continue
        throughput_a, throughput_b = (
            mean(r["window_throughput_tok_s"] for r in rows) for rows in (a, b)
        )
        wall_a, wall_b = (mean(r["window_ms"] for r in rows) for rows in (a, b))
        pairs[key] = {
            "status": "MEASURED",
            "window_ms_before": wall_a,
            "window_ms_after": wall_b,
            "window_time_change_ms": wall_b - wall_a,
            "window_throughput_ratio": throughput_b / throughput_a,
            "window_throughput_change_percent": (throughput_b / throughput_a - 1) * 100,
            "tokens_before": stats([r["committed_window_tokens"] for r in a]),
            "tokens_after": stats([r["committed_window_tokens"] for r in b]),
            "g_before": stats(
                [
                    r["g_per_request_per_verification"]
                    for r in a
                    if r["g_per_request_per_verification"] is not None
                ]
            ),
            "g_after": stats(
                [
                    r["g_per_request_per_verification"]
                    for r in b
                    if r["g_per_request_per_verification"] is not None
                ]
            ),
            "label": "production vLLM Batched Draft end-to-end improvement",
            "pure_batching_claim": False,
            "work_matching": "different modes may commit different work; actual counters retained",
        }
    contrasts = matched_shape_contrasts(reports)
    v32, v64, d32, d64 = (stage_means[k] for k in ("V_SD32", "V_SD64", "D32", "D64"))
    ideal_serial = d64 + v64 if d64 is not None and v64 is not None else None
    ideal_ping = 2 * max(d32, v32) if d32 is not None and v32 is not None else None
    return {
        "schema_version": "specrhythm.fixed-comparison.v1",
        "scenario": SCENARIO,
        **POLICY,
        "partial_measurements": [
            {
                "artifact": r["artifact"],
                "mode": r["mode"],
                "measurement": r["measurement_snapshot"],
                "execution_status": r["execution_status"],
                "cleanup_status": r["cleanup_status"],
                "primary_error": r.get("primary_error"),
                "formal_comparison_eligible": False,
            }
            for r in reports
            if r.get("measurement_snapshot")
            and (not r.get("valid") or r.get("measurement_status") != "PASS")
        ],
        "points": [
            {
                k: r.get(k)
                for k in (
                    "mode",
                    "point",
                    "artifact",
                    "execution_status",
                    "measurement_status",
                    "full_offline_audit_status",
                    "actual_target_batch",
                    "actual_rotation_ms",
                    "window_throughput_tok_s",
                    "g_per_request_per_verification",
                )
            }
            for r in reports
        ],
        "stage_gpu_event_ms": stage_values,
        "split_verification_cost_ms": contrasts["split_verification_cost_ms"]["mean"],
        "matched_shape_contrasts": contrasts,
        "D32_over_V_SD32": contrasts["D32_over_V_SD32"]["mean"],
        "measured_comparisons": pairs,
        "prediction": {
            "T_serial": "D64 + V_SD64 + H_serial",
            "T_pingpong": "2 * (max(D32, V_SD32) + H_pingpong)",
            "assumptions": "balanced full A/B, comparable context/K, own g per mode",
            "unexplained_time": "report measured wall residual; no automatic causal attribution",
            "H_is_not_sum_of_nested_host_timers": True,
            "ideal_serial_gpu_stage_ms": ideal_serial,
            "ideal_pingpong_gpu_stage_ms": ideal_ping,
            "measured_wall_minus_idealized_gpu_ms": {
                m: stats(
                    [
                        r["actual_rotation_ms"]["mean"] - ideal_serial
                        for r in continuous[m]
                        if r["actual_rotation_ms"]["count"]
                    ]
                )
                if m == "serial" and ideal_serial is not None
                else stats(
                    [
                        r["actual_rotation_ms"]["mean"] - ideal_ping
                        for r in continuous[m]
                        if r["actual_rotation_ms"]["count"]
                    ]
                )
                if m == "pingpong" and ideal_ping is not None
                else stats([])
                for m in ("serial", "pingpong")
            },
        },
    }


def offline_audit(manifest_path, directory):
    """Explicit CPU-only full artifact scan. Never called by execute/summary/bundle."""
    from specrhythm.phase4.decode_ready import load_decode_ready_manifest
    from specrhythm.phase4.dual_correctness import _LEGAL_STATES, validate_verification_contracts
    from specrhythm.phase4.manifest import sha256_file
    from specrhythm.phase4.transport import CheckpointJsonl
    from specrhythm.phase4.vllm_diagnostics import validate_target_diagnostic
    from specrhythm.phase4.vllm_dual import validate_target_rank_identity
    from specrhythm.serving.s2_overlap import stage_overlap

    point = read_json(directory / "point.json")
    report = summarize(manifest_path, directory, point, probe=point.get("probe", False))
    require(report["valid"], "light execution evidence invalid", actual=report.get("errors"))
    runtime = read_json(directory / "runtime.json")
    if runtime.get("probe"):
        return {"status": "NOT_APPLICABLE", "reason": "model-loaded capacity probe"}
    ready = load_decode_ready_manifest(read_json(directory / "decode-ready-manifest.json"))
    require(
        {r.request_id for r in ready.requests} == {r["request_id"] for r in runtime["requests"]},
        "offline resident/observed request identity mismatch",
    )
    logs = {p.name: CheckpointJsonl(p).read() for p in sorted(directory.glob("*.jsonl"))}
    diagnostics = logs.get("target-diagnostics.jsonl", [])
    require(diagnostics, "offline audit missing Target diagnostic log")
    for r in diagnostics:
        require(
            not validate_target_diagnostic(r),
            "offline Target diagnostic invalid",
            artifact=str(directory / "target-diagnostics.jsonl"),
            actual=validate_target_diagnostic(r),
        )
    backend = read_json(directory / "draft-backend-report.json")
    checks = {}
    if point["runtime_mode"] == "pingpong":
        proposals = logs.get("proposal-events.jsonl", [])
        verifies = logs.get("verification-events.jsonl", [])
        errors = validate_verification_contracts(proposals, verifies, diagnostics)
        require(not errors, "offline proposal/verification/input contract invalid", actual=errors)
        for verify in verifies:
            require(
                not validate_target_rank_identity(verify["target_rank_intervals"], 2),
                "offline actual Target TP/device verification invalid",
                actual=verify,
            )
        checks["proposal_lifecycle"] = audit_proposal_lifecycle(
            logs.get("proposal-lifecycle-events.jsonl", []), runtime["requests"], proposals
        )
        require(
            all(r.get("success") is True for r in logs.get("draft-work-events.jsonl", [])),
            "offline Draft owner work failed",
        )
        states = logs.get("request-state-events.jsonl", [])
        require(states, "offline audit missing request-state evidence")
        last = {}
        for r in states:
            rid = r["request_id"]
            prior = last.get(rid)
            require(
                r["source_state"] == (prior["destination_state"] if prior else "BOOTSTRAP")
                and r["destination_state"] in _LEGAL_STATES.get(r["source_state"], ())
                and (
                    prior is None
                    or (
                        r["timestamp_ns"] > prior["timestamp_ns"]
                        and r["prefix_version"] >= prior["prefix_version"]
                    )
                ),
                "offline request lifecycle/prefix order invalid",
                actual=r,
            )
            last[rid] = r
        for r in runtime["requests"]:
            require(r["request_id"] in last, "offline request lacks native state")
            final = last[r["request_id"]]["destination_state"]
            require(
                (r["state"] == "FINISHED" and final == "TERMINAL")
                or (r["state"] == "DIAGNOSTIC_CANCELLED" and final not in ("FAILED", "VERIFYING")),
                "offline natural completion/controlled cancellation mismatch",
                actual=r,
            )
        forwards = {}
        for r in diagnostics:
            a, b = r["target_forward_start_ns"], r["target_forward_end_ns"]
            if a < runtime["start_ns"]:
                continue
            f = forwards.setdefault(
                (a, b),
                {
                    "start_ns": a,
                    "end_ns": b,
                    "request_ids": [],
                    "kind": "verify" if r["proposal_token_ids"] else "tail",
                },
            )
            if r["request_id"] not in f["request_ids"]:
                f["request_ids"].append(r["request_id"])
        checks["stage_dependency"] = stage_overlap(
            backend,
            list(forwards.values()),
            runtime,
            states,
            logs.get("draft-work-events.jsonl", []),
            read_json(directory / "actual-capacity.json")["target_worker_ranks"],
            directory,
        )
    for name in ("proposal-events.jsonl", "round-events.jsonl"):
        rows = [r for r in logs.get(name, []) if "committed_token_ids" in r]
        if rows:
            rounds_by_prefix(rows)
    return {
        "schema_version": "specrhythm.fixed-offline-audit.v1",
        "status": "PASS",
        "scope": "diagnostic-window accounting, actual input/KV/state/owned cleanup; "
        "controlled cancellation is not natural TERMINAL",
        "checks": checks,
        "artifact_sha256": {
            p.name: sha256_file(p)
            for p in sorted(directory.iterdir())
            if p.is_file() and p.name != "offline-audit.json"
        },
        **POLICY,
    }


def matched_shape_contrasts(reports):
    """Only derive split cost from the exact union and same per-request contexts."""
    groups = defaultdict(dict)
    for r in reports:
        p = r.get("point", {})
        if (
            r.get("valid")
            and not r.get("probe")
            and p.get("kind") == "initial-state"
            and not p.get("discard_warmup")
            and r.get("stage_shape", {}).get("status") == "PASS"
            and p["mode"] == "serial"
        ):
            groups[p["repeat"]][(p["batch"], p["half"])] = r
    costs, ratios = [], []
    rejected = []
    for repeat, group in groups.items():
        keys = ((32, "A"), (32, "B"), (64, "A"))
        if not all(k in group for k in keys):
            rejected.append({"repeat": repeat, "reason": "need both 32 halves and their 64 union"})
            continue
        a, b, whole = [group[k] for k in keys]
        shapes = [r["stage_shape"] for r in (a, b, whole)]
        contexts = [dict(zip(s["expected_request_ids"], s["context_lengths"])) for s in shapes]
        identities = [
            [
                (i.get("role"), i.get("dtype"), i.get("attention_backends"))
                for i in r["dtype_attention_identity"]
            ]
            for r in (a, b, whole)
        ]
        if (
            set(contexts[0]) & set(contexts[1])
            or contexts[0] | contexts[1] != contexts[2]
            or len({r["execution_sha256"] for r in (a, b, whole)}) != 1
            or identities[0] != identities[1]
            or identities[1] != identities[2]
        ):
            rejected.append(
                {"repeat": repeat, "reason": "union/context/dtype/backend/config mismatch"}
            )
            continue
        v = [s["target_gpu_event_ms"] for s in shapes]
        costs.append(v[0] + v[1] - v[2])
        d = [s["draft_gpu_event_ms"] for s in shapes[:2]]
        if all(x is not None for x in d):
            ratios.append(sum(d) / sum(v[:2]))
    return {
        "matched_repeats": len(costs),
        "split_verification_cost_ms": stats(costs),
        "D32_over_V_SD32": stats(ratios),
        "excluded": rejected,
        "definition": "D32/V32 uses both halves; B64 is their exact context-matched union",
    }


def forward_gaps(forwards, start, end):
    selected = [r for r in forwards if start <= r["host_start_ns"] <= end]
    return stats(
        [
            max(0, b["start_lower_ns"] - a["end_lower_ns"]) / 1e6
            for a, b in zip(selected, selected[1:])
        ],
        "fewer than two actual model forwards",
    )


def concurrent_sample(manifest, point, targets, drafts):
    if point["kind"] != "initial-state" or point["mode"] != "pingpong":
        return {"status": "NOT_APPLICABLE"}
    groups = manifest["fixed_diagnostic"]["cohorts"]
    a = [
        s
        for s in targets
        if set(s["request_ids"]) == set(groups["A"])
        and all(k == 4 for k in s["candidate_lengths"])
    ]
    b = [
        s
        for s in drafts
        if set(s["request_ids"]) == set(groups["B"])
        and set(s["round_ids"]) == {0}
        and all(k == 4 for k in s["candidate_lengths"])
    ]
    if len(a) != 1 or len(b) != 1:
        return {
            "status": "INSUFFICIENT",
            "reason": "actual A verification/B initial Draft B32 K4 missing",
        }
    a, b = a[0], b[0]
    return {
        "status": "MEASURED",
        "Draft_B32_gpu_event_ms": b["gpu_event_ms"],
        "Draft_B32_host_ms": b["wall_ms"],
        "Target_A32_gpu_event_ms": a["gpu_event_ms"],
        "pair_host_envelope_ms": (
            max(a["end_ns"], b["end_ns"]) - min(a["start_ns"], b["start_ns"])
        )
        / 1e6,
        "definition": "actual synchronous Target step and B proposal span; includes control work",
        "physical_overlap": "see clock-bounded event union, not async submission evidence",
    }


def audit_proposal_lifecycle(events, requests, rounds):
    """A cancelled unverified suffix is explicit; completed proposals retain full checks."""
    cancelled = {r["request_id"] for r in requests if r["state"] == "DIAGNOSTIC_CANCELLED"}
    grouped, round_keys = defaultdict(list), {}
    committed = {r["proposal_id"] for r in rounds}
    for r in events:
        key = (r["request_id"], r["round_id"])
        require(
            round_keys.setdefault(key, r["proposal_id"]) == r["proposal_id"],
            "multiple proposal identities for one request round",
        )
        grouped[r["proposal_id"]].append(r)
    require(committed <= set(grouped), "committed proposal missing native lifecycle")
    cancelled_pending = 0
    for pid, rows in grouped.items():
        phases = [r["lifecycle_state"] for r in rows]
        done = phases == ["CREATED", "PUBLISHED", "INSTALLED", "CONSUMED"]
        dropped = phases in (
            ["CREATED", "PUBLISHED", "DROPPED_STALE"],
            ["CREATED", "PUBLISHED", "INSTALLED", "DROPPED_STALE"],
        )
        pending = (
            phases in (["CREATED", "PUBLISHED"], ["CREATED", "PUBLISHED", "INSTALLED"])
            and rows[0]["request_id"] in cancelled
            and pid not in committed
        )
        identity = {
            (r["request_id"], r["round_id"], r["prefix_version"], tuple(r["proposal_token_ids"]))
            for r in rows
        }
        times = [r["timestamp_ns"] for r in rows]
        require(
            len(identity) == 1
            and times == sorted(times)
            and (done or dropped or pending)
            and (pid not in committed or done),
            "offline proposal lifecycle invalid",
            actual=rows,
        )
        cancelled_pending += int(pending)
    return {
        "checked_proposals": len(grouped),
        "cancelled_unverified_proposals": cancelled_pending,
        "cancelled_proposals_counted_as_commits": False,
    }
