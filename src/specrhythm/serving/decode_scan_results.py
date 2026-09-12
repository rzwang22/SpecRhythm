"""Small scan qualification: actual batch/commit/identity/lifecycle, no offline audit."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter

from specrhythm.serving.common import DataError, read_json, require
from specrhythm.serving.decode_scan_plan import BOUNDARY, POOL_SIZE, SCHEMA
from specrhythm.serving.decode_scan_window import ScanWindow, pair_step
from specrhythm.serving.fixed_artifacts import retained_report
from specrhythm.serving.fixed_identity import qualify as identity_checks
from specrhythm.serving.fixed_logging import qualify as logging_checks
from specrhythm.serving.fixed_plan import POLICY
from specrhythm.serving.fixed_results import (
    device_batches,
    execution_checks,
    positions,
    rounds_by_prefix,
    stats,
)
from specrhythm.serving.runtime_profile import load_s2
from specrhythm.serving.s1_workload import write_once


def prepared_checks(manifest, runtime, directory, point):
    actual = read_json(directory / "actual-capacity.json")
    require(
        actual["execution_sha256"] == manifest["sha256"]
        and actual["workload_sha256"] == manifest["workload_sha256"]
        and actual["point"] == point,
        "scan capacity point/workload binding differs",
    )
    require(
        len(actual["checks"]) == 3
        and all(
            c["valid"] and c["pool_size"] == POOL_SIZE and c["active_limit"] == point["batch"]
            for c in actual["checks"]
        ),
        "scan physical capacity insufficient",
        artifact="actual-capacity.json",
    )
    pool = read_json(directory / "resident-pool.json")
    ids = set(manifest["request_ids"])
    initial = pool["target"]["initial"]
    require(
        pool["selected_requests"] == POOL_SIZE
        and pool["resident_requests"] + pool["bootstrap_terminal_requests"] == POOL_SIZE
        and len(initial) == pool["resident_requests"]
        and set(initial) == set(pool["draft"]["rows"]) <= ids,
        "scan all-request physical prefill evidence incomplete",
        artifact="resident-pool.json",
    )
    require(
        runtime["capacity"]["resident_request_count"] == pool["resident_requests"],
        "scan actual resident count differs",
    )
    require(
        pool["target"]["timed_reprefill_allowed"] is False, "scan permits timed initial prefill"
    )
    require(
        runtime["decode_scan"]["prefill_complete_ns"] <= runtime["start_ns"],
        "scan prefill boundary occurs after decode clock start",
    )
    return {
        k: pool[k]
        for k in (
            "selected_requests",
            "resident_requests",
            "bootstrap_terminal_requests",
            "prefill_setup_ns",
        )
    }


def full_load_fraction(rows, start, end, batch):
    """Sweep actual admission/completion boundaries; refill/drain gaps stay in denominator."""
    changes = Counter()
    for r in rows:
        a = r.get("admission_ns")
        b = r.get("completion_ns", end)
        if a is not None and max(a, start) < min(b, end):
            changes[max(a, start)] += 1
            changes[min(b, end)] -= 1
    held, last, full = 0, start, 0
    for timestamp, delta in sorted(changes.items()):
        if held == batch:
            full += timestamp - last
        held += delta
        require(0 <= held <= batch, "scan active request ceiling exceeded")
        last = timestamp
    return full / (end - start)


def rotations(steps, expected, mode):
    """Pair opposite cohorts using actual disjoint request sets; retain partial rotations."""
    complete, partial, pending = 0, 0, None
    for s in steps:
        if s["B"] != expected:
            partial += 1
            continue
        rotation, historical, pending = pair_step(pending, s, mode == "pingpong")
        complete += int(rotation is not None)
        partial += int(historical is not None)
    return complete, partial + int(pending is not None)


def warmup_boundary(runtime, point, opts):
    """Replay actual full steps, not the historical-partial count, at the start boundary."""
    scan, start = runtime["decode_scan"], runtime["measurement_start_ns"]
    retained = scan.get("warmup_boundary")
    require(isinstance(retained, dict), "scan mandatory warmup boundary evidence missing",
            field="decode_scan.warmup_boundary", artifact="runtime.json")
    replay = ScanWindow(opts, point["batch"], point["mode"] == "pingpong")
    previous_end = runtime["start_ns"]
    for row in runtime["target_steps"]:
        if row["window"] or not row["B"]:
            continue
        require(previous_end <= row["start_ns"] < row["end_ns"],
                "scan warmup step timestamps overlap or are invalid", actual=row["start_ns"])
        replay.step_completed(row, row["end_ns"])
        previous_end = row["end_ns"]
    require(replay.warmup_rotations == scan["warmup_rotations"]
            and replay.warmup_steps == runtime["warmup_steps"], "scan warmup evidence differs")
    if start is not None:
        require(replay.pending is None, "scan window start has a pending warmup half-rotation",
                actual=replay.pending, artifact="runtime.json", field="target_steps")
        population = scan["window_initial_population"]
        require(isinstance(population, dict) and replay.ready(start, population=population),
                "scan window start lacks required full warmup/population",
                artifact="runtime.json", field="decode_scan.window_initial_population")
        active = {r["request_id"]: r for r in runtime["requests"]
                  if r.get("admission_ns") is not None and r["admission_ns"] <= start
                  and (r.get("completion_ns") is None or start < r["completion_ns"])}
        require(set(population["request_ids"]) == set(active),
                "scan initial active identities differ from request lifecycle")
        if point["mode"] == "pingpong":
            require(all(set(population["cohorts"][c]) ==
                        {rid for rid, r in active.items() if r["cohort"] == c}
                        for c in ("A", "B")), "scan initial cohort identities differ")
    expected = replay.warmup_boundary()
    require(retained == expected, "scan warmup boundary evidence differs from actual steps",
            artifact="runtime.json", field="decode_scan.warmup_boundary",
            expected=expected, actual=retained)
    return expected


def timing(runtime, backend, point, opts):
    start, end = runtime["measurement_start_ns"], runtime["measurement_end_ns"]
    all_steps = runtime["target_steps"]
    measured = [s for s in all_steps if s["window"]]
    nonempty = [s for s in measured if s["B"]]
    expected = point["batch"] // 2 if point["mode"] == "pingpong" else point["batch"]
    devices = runtime["target_devices"]
    require(
        len(devices) == 2 and {d["device"]["identity"]["global_rank"] for d in devices} == {0, 1},
        "scan Target TP evidence missing",
    )
    require(
        len(
            {d["device"]["identity"]["gpu_uuid"] for d in devices}
            | {backend["fixed_device"]["identity"]["gpu_uuid"]}
        )
        == 3,
        "scan GPU identities are not isolated",
    )
    device_batches(
        devices, all_steps
    )  # Actual forward B/IDs and positive CUDA events per TP rank.
    rounds = rounds_by_prefix([r for d in devices for r in d["rounds"]])
    diagnostics = {}
    for d in devices:
        for r in d["target_rows"]:
            if r["target_forward_start_ns"] < runtime["start_ns"]:
                continue
            require(
                not r["structural_errors"],
                "scan Target structural verification invalid",
                request_id=r["request_id"],
                actual=r["structural_errors"],
            )
            key = (r["request_id"], r["context_length"])
            require(
                key not in diagnostics or diagnostics[key] == r,
                "scan Target TP diagnostic disagreement",
                actual=key,
            )
            diagnostics[key] = r
    final = {r["request_id"]: r for r in runtime["requests"]}
    total = roots = candidates = accepted = rejected = 0
    for s in all_steps:
        step_tokens = 0
        require(
            0 <= s["B"] <= expected and len(set(s["request_ids"])) == s["B"],
            "scan actual batch ceiling/identity invalid",
            actual=s["B"],
        )
        require(s["output_commit_complete"], "scan Target output commit incomplete")
        require(s["population"]["held_slots"] <= point["batch"], "scan held capacity exceeded")
        if point["mode"] == "pingpong":
            require(
                all(n <= expected for n in s["population"]["cohort_held"].values()),
                "scan cohort capacity exceeded",
            )
            require(not s["B"] or s["cohort"] in ("A", "B"), "scan cohort missing")
        if s["window"]:
            require(
                start is not None and start <= s["start_ns"] < s["end_ns"] <= end,
                "scan measured step crosses window boundary",
            )
        if s["window"]:
            require(
                s["start_ns"] < start + opts["window_seconds"] * 1e9,
                "scan issued another step after time budget",
            )
        elif start is not None:
            require(s["end_ns"] <= start, "scan excluded a step inside the measured window")
        for row in s["rows"]:
            rid = row["request_id"]
            require(
                point["mode"] != "pingpong" or final[rid]["cohort"] == s["cohort"],
                "scan cohort/request assignment differs",
            )
            q, k, root = positions(row)
            key = (rid, row["context_length"])
            require(key in diagnostics, "scan scheduled row lacks actual Target input", actual=key)
            d = diagnostics[key]
            require(
                d["query_length"] == q
                and len(d["proposal_token_ids"]) == k
                and d["position_ids"]
                == list(range(row["position_start"], row["position_end_exclusive"])),
                "scan actual Target token positions differ",
                actual=key,
            )
            progress = 1
            if k:
                require(key in rounds, "scan verification lacks commit record", actual=key)
                r = rounds[key]
                offset = row["context_length"] - runtime["prompt_lengths"][rid]
                progress = len(r["committed_token_ids"])
                require(
                    len(r["proposal_token_ids"]) == k
                    and offset >= 1
                    and final[rid]["generated_token_ids"][offset : offset + progress]
                    == r["committed_token_ids"],
                    "scan committed prefix/accounting differs",
                )
                if s["window"]:
                    accepted += r["accepted_draft_tokens"]
                    rejected += r["rejected_draft_tokens"]
            if s["window"]:
                total += progress
                roots += root
                candidates += k
            step_tokens += progress
        if not s["window"] and s["B"]:
            committed = sum(len(c["token_ids"]) for rid in s["request_ids"]
                            for c in final[rid]["commits"]
                            if s["start_ns"] <= c["timestamp_ns"] <= s["end_ns"])
            require(s.get("committed_tokens") == step_tokens == committed,
                    "scan warmup committed token/time accounting differs",
                    expected=step_tokens, actual=s.get("committed_tokens"))
    tokens = sum(
        len(c["token_ids"])
        for r in final.values()
        for c in r["commits"]
        if start is not None and start <= c["timestamp_ns"] <= end
    )
    require(
        tokens == total and candidates == accepted + rejected,
        "scan committed/root/candidate accounting differs",
        expected=total,
        actual=tokens,
    )
    scan = runtime["decode_scan"]
    complete, partial = rotations(nonempty, expected, point["mode"])
    require(
        complete == len(scan["complete_rotations"]) and partial == len(scan["partial_rotations"]),
        "scan rotation evidence differs",
    )
    boundary = warmup_boundary(runtime, point, opts)
    wall = (end - start) / 1e6 if start is not None else None
    reason = runtime["stop_reason"]
    full = bool(nonempty) and all(s["B"] == expected for s in nonempty)
    qualified = (
        reason == "time_budget"
        and wall is not None
        and wall >= opts["window_seconds"] * 1000
        and full
        and scan["warmup_rotations"] == opts["warmup_steps"]
    )
    if qualified:
        require(tokens > 0 and math.isfinite(wall) and wall > 0, "scan throughput nonpositive")
    if not full and nonempty:
        reason = "unexpected_partial_batch"
    admissions = [e for e in runtime["events"] if e["event"] == "admitted"]
    return {
        "measurement_status": "PASS" if qualified else "INSUFFICIENT",
        "stop_reason": reason,
        "formal_comparison_eligible": qualified,
        "measurement_available": bool(nonempty),
        "measured_window_ms": wall,
        "window_budget_seconds": opts["window_seconds"],
        "window_overrun_ms": max(0.0, wall - opts["window_seconds"] * 1000) if wall else None,
        "committed_window_tokens": tokens,
        "decode_throughput_tok_s": tokens * 1000 / wall if wall and tokens else None,
        "actual_target_batch": {
            "min": min((s["B"] for s in nonempty), default=None),
            **stats([s["B"] for s in nonempty]),
        },
        "full_active_time_fraction": full_load_fraction(final.values(), start, end, point["batch"])
        if start is not None and end > start
        else None,
        "target_steps": len(nonempty),
        "empty_scheduler_steps": len(measured) - len(nonempty),
        "complete_rotations": complete,
        "partial_rotations": partial,
        "warmup_rotations": scan["warmup_rotations"],
        "warmup_steps": runtime["warmup_steps"],
        "scan_warmup_boundary": boundary,
        "root_positions": roots,
        "candidate_positions": candidates,
        "accepted_tokens": accepted,
        "rejected_tokens": rejected,
        "natural_completed_requests": sum(r["state"] == "FINISHED" for r in final.values()),
        "diagnostic_cancelled_requests": sum(
            r["state"] == "DIAGNOSTIC_CANCELLED" for r in final.values()
        ),
        "refill_requests": max(0, len(admissions) - point["batch"]),
        "admission_order": [
            {
                "request_id": e["request_id"],
                "cohort": e["cohort"],
                "timestamp_ns": e["timestamp_ns"],
            }
            for e in admissions
        ],
        "window_initial_population": scan["window_initial_population"],
        "rejected_step": scan["rejected_step"],
        "scan_readiness": scan.get("readiness"),
        "drain_ms": (runtime["end_ns"] - end) / 1e6,
        "full_execution_ms": runtime.get("engine_and_execution_host_ms"),
    }


def summarize(manifest_path, directory, point, *, probe=False):
    m, definitions = load_s2(str(manifest_path))
    opts = m["fixed_diagnostic"]["options"]
    base = {
        "schema_version": SCHEMA,
        "mode": point["mode"],
        "point": point,
        "probe": probe,
        "git_commit": m["execution"]["git_commit"],
        "execution_sha256": m["sha256"],
        "workload_sha256": m["workload_sha256"],
        "pool_size": POOL_SIZE,
        "batch": point["batch"],
        "sub_batch": point["batch"] // 2 if point["mode"] == "pingpong" else None,
        "boundary": BOUNDARY,
        "artifact": str(directory),
        "effective_exit_code": 0,
        "formal_comparison_eligible": False,
        **POLICY,
    }
    try:
        r = read_json(directory / "runtime.json")
        b = read_json(directory / "draft-backend-report.json")
        if point["mode"] == "serial-eager":
            from specrhythm.serving.eager_results import eager_columns, summarize_eager

            base["rolling_eager"] = summarize_eager(b, r)
            base.update(eager_columns(base))
        life = read_json(directory / "process-lifecycle.json")
        require(
            r["point"] == point and bool(r.get("probe")) == probe, "scan runtime point differs"
        )
        base["draft_backend_checks"] = execution_checks(
            {**r, "probe": False}, definitions, b, life, m["execution"]["eos_token_ids"]
        )
        base["prepared_pool"] = prepared_checks(m, r, directory, point)
        base["diagnostic_logging"] = logging_checks(directory, opts["observation"])
        base["identity_matching"] = identity_checks(r, opts["identity_matching"])
        if point["mode"] == "pingpong":
            verifications = sum(
                any(row["candidate_positions"] for row in s["rows"]) for s in r["target_steps"]
            )
            ranks = [x["dual_uuid_query"] for x in r["target_final_memory"]]
            require(
                len(ranks) == 2
                and all(
                    x["uuid_query_mode"] == "live"
                    and x["uuid_initial_validation_count"] == 1
                    and x["uuid_cache_hit_count"] == 0
                    and x["uuid_verification_subprocess_query_count"]
                    == x["uuid_verification_access_count"]
                    and x["uuid_verification_access_count"] == verifications
                    for x in ranks
                ),
                "scan live UUID evidence invalid",
            )
            base["uuid_query_by_rank"] = ranks
        if probe:
            require(
                r["diagnostic_drain"]["status"] == "COMPLETE"
                and r["diagnostic_drain"]["settled_requests"] == POOL_SIZE
                and r["target_requests_final"] == 0
                and not r["target_steps"],
                "scan prefilled probe did not settle all360 without verification",
            )
            result = {"measurement_status": "NOT_APPLICABLE", "stop_reason": "capacity_probe"}
        else:
            result = timing(r, b, point, opts)
        return {**base, **result, "valid": True, "errors": [], "execution_status": "PASS"}
    except (DataError, KeyError, TypeError, ValueError, OSError) as error:
        return {
            **base,
            "valid": False,
            "errors": [str(error)],
            "execution_status": "FAILED",
            "measurement_status": "INVALID",
            "primary_error": {
                "error": str(error),
                "artifact": str(directory),
                **getattr(error, "details", {}),
            },
        }


def compact(value):
    """Keep runtime originals; summaries carry settlement completeness without 360 receipts."""
    from specrhythm.serving.common import digest

    if isinstance(value, list):
        return [compact(v) for v in value]
    if not isinstance(value, dict):
        return value
    project = len(value.get("receipts", [])) > 32 and all(
        "request_id" in r for r in value.get("receipts", [])
    )
    result = {k: compact(v) for k, v in value.items() if not (project and k == "receipts")}
    if project:
        rows = value["receipts"]
        result["receipt_summary"] = {
            "count": len(rows),
            "sha256": digest(rows),
            "released": sum(bool(r.get("released")) for r in rows),
            "new_proposals_generated": sum(r.get("new_proposals_generated", 0) for r in rows),
            "original_receipts_retained_in_run_directory": True,
        }
    return result


def emit_result(directory, result, point):
    value = retained_report(
        directory,
        {
            **result,
            "point": point,
            "batch": point["batch"],
            "mode": point["mode"],
            "probe": point.get("probe", False),
        },
    )
    capacity_path = directory / "actual-capacity.json"
    if capacity_path.exists():
        capacity = read_json(capacity_path)
        value["capacity_status"] = (
            "PASS" if all(c["valid"] for c in capacity["checks"]) else "INSUFFICIENT"
        )
        if value["capacity_status"] == "INSUFFICIENT":
            value["stop_reason"] = "capacity_insufficient"
    else:
        value["capacity_status"] = "UNKNOWN"
    if value.get("stop_reason") == "operator_stop" and value.get("valid"):
        value["execution_status"] = "STOPPED"
        value["measurement_status"] = "STOPPED"
    value["formal_comparison_eligible"] = bool(
        value.get("valid")
        and value.get("measurement_status") == "PASS"
        and value.get("cleanup_status") == "PASS"
    )
    value = compact(value)
    print("[decode scan] " + json.dumps(value, ensure_ascii=False), flush=True)
    write_once(directory / "result.json", value)
    write_once(directory / "light-summary.json", value)
    flat = {k: v for k, v in value.items() if not isinstance(v, (dict, list))}
    with (directory / "light-summary.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)
    return value
