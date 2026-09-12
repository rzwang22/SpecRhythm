"""Bounded four-point audit comparison; raw events remain once in source bundles.

Metrics are original measured values. Host categories are inclusive per lane;
only the coordinator partition is disjoint. No audit-subtracted throughput.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

from specrhythm.serving.common import require
from specrhythm.serving.eager_evidence_export import Reader, bounds, duration, intersect, touching
from specrhythm.serving.eager_latency import causal_cycle, exclusive_main, trace_rows
from specrhythm.serving.fixed_results import stats

MAX_FILE = 512 * 1024 * 1024
MAX_POINT = 1024 * 1024 * 1024
MAX_OUTPUT = 8 * 1024 * 1024


def categories(rows, start, end):
    groups = {}
    for category in sorted({r["category"] for r in rows}):
        selected = [r for r in rows if r["category"] == category and touching(r, start, end)]
        spans = [(max(start, r["start_ns"]), min(end, r["end_ns"])) for r in selected]
        groups[category] = dict(
            count=len(selected),
            union_ms=duration(spans),
            inclusive_sum_ms=sum(b - a for a, b in spans) / 1e6,
        )
    return groups


def exclusive_lanes(rows, start, end):
    """Sweep endpoints in O(N log N); each thread is its own wall-time budget."""
    precedence = [
        "prefix_hash_and_block_record",
        "resident_block_audit",
        "runtime_resident_check",
        "runtime_write_check",
        "json_serialization",
        "log_fsync",
        "checkpoint_log_write",
        "required_cuda_synchronize",
        "required_cuda_event_synchronize",
        "required_tp_barrier",
        "draft_required_fence",
        "target_diagnostic_contract_scan",
        "eligibility_snapshot",
        "eligibility_provider",
        "control_json_read",
        "control_json_write",
        "draft_pool_audit",
        "physical_gpu_audit",
        "parent_kv_materialize",
        "eager_token_forward",
        "normal_recovery_proposal",
        "owner_batch_gate",
        "owner_response_wait",
        "owner_dispatch",
        "ipc",
        "output_commit",
        "coordinator_checkpoint",
        "target_step",
    ]
    lanes = {}
    missing = 0
    for row in rows:
        if row["category"] not in precedence or not touching(row, start, end):
            continue
        if row.get("pid") is None or row.get("thread_id") is None:
            missing += 1
            continue
        a, b = max(start, row["start_ns"]), min(end, row["end_ns"])
        if a < b:
            events = lanes.setdefault((row["pid"], row["thread_id"]), [])
            events.extend(((a, 1, row["category"]), (b, -1, row["category"])))
    output = []
    for (pid, tid), events in sorted(lanes.items()):
        active, totals, previous = Counter(), Counter(), start
        for stamp, delta, category in sorted(events):
            label = next((p for p in precedence if active[p]), "unaccounted")
            totals[label] += stamp - previous
            active[category] += delta
            previous = stamp
        totals["unaccounted"] += end - previous
        output.append(
            dict(
                pid=pid,
                thread_id=tid,
                window_ms=(end - start) / 1e6,
                exclusive_ms={k: v / 1e6 for k, v in totals.items()},
            )
        )
    return dict(
        status="COMPLETE" if output and not missing else "MISSING_LANES",
        missing_lane_rows=missing,
        precedence=precedence,
        lanes=output,
        semantics="mutually exclusive within each thread only; never sum thread budgets; "
        "parent residual is not recorder overhead or GIL evidence",
    )


def device(device, start, end):
    rows = [r for r in device.get("forwards", []) if start <= r["host_start_ns"] <= end]
    require(
        all(
            type(r.get("gpu_event_ms")) in (int, float)
            and math.isfinite(r["gpu_event_ms"])
            and r["gpu_event_ms"] > 0
            and type(r.get("B")) is int
            and r["B"] > 0
            for r in rows
        ),
        "invalid native GPU duration or batch",
    )
    native = [
        r
        for r in rows
        if all(
            type(r.get(k)) is int
            for k in ("start_lower_ns", "start_upper_ns", "end_lower_ns", "end_upper_ns")
        )
    ]
    return {
        "identity": device.get("identity"),
        "source_status": device.get("status"),
        "native_coverage": "COMPLETE" if len(native) == len(rows) and rows else "MISSING",
        "forwards": len(rows),
        "GPU_forward_ms": stats([r["gpu_event_ms"] for r in rows]),
        "native_union": {
            side: duration(bounds(native, start, end, inner))
            for side, inner in (("lower_ms", True), ("upper_ms", False))
        },
        "by_purpose": {
            purpose: {
                "count": len(selected),
                "B_histogram": dict(Counter(str(r["B"]) for r in selected)),
                "gpu_event_sum_ms": sum(r["gpu_event_ms"] for r in selected),
            }
            for purpose in sorted({r.get("purpose") for r in rows})
            for selected in [[r for r in rows if r.get("purpose") == purpose]]
        },
    }


def analyze(runtime, backend, light):
    start, end = (
        light["measurement_snapshot"][k] for k in ("measurement_start_ns", "measurement_end_ns")
    )
    steps = [s for s in runtime["target_steps"] if s.get("window") and touching(s, start, end)]
    require(0 < len(steps) <= 512 and len(steps) == light["target_steps"], "invalid bounded steps")
    targets = [r["device"] for r in runtime["target_devices"]]
    require(
        len(targets) == 2 and {d["identity"]["global_rank"] for d in targets} == {0, 1},
        "missing native TP ranks",
    )
    draft = backend["fixed_device"]
    require(
        len({d["identity"]["gpu_uuid"] for d in targets + [draft]}) == 3,
        "invalid within-point Draft/TP identities",
    )
    lanes = {"coordinator": runtime.get("host", {}), "draft": backend.get("fixed_host", {})}
    lanes.update(
        {
            f"target-rank-{r['device']['identity']['global_rank']}": r.get("host", {})
            for r in runtime["target_devices"]
        }
    )
    traces = {name: trace_rows(host, start, end) for name, host in lanes.items()}
    all_trace = [r for t in traces.values() for r in t["rows"]]
    draft_trace = traces["draft"]["rows"]
    audit = backend.get(
        "draft_audit",
        {"mode": "full", "schema_version": "legacy-full", "metadata_status": "NOT_RECORDED"},
    )
    cycles = []
    for index, step in enumerate(steps):
        # Compute metrics, then discard repeated raw host/TP rows immediately.
        causal = causal_cycle(step, targets, draft, all_trace)
        causal.pop("target_ranks", None)
        causal.pop("host_events", None)
        a, b = step["start_ns"], step["end_ns"]
        eager = [
            f
            for f in draft["forwards"]
            if f.get("purpose") == "eager" and a <= f["host_start_ns"] <= b
        ]
        first = min((f["host_start_ns"] for f in eager), default=b)
        host = backend["fixed_host"]["intervals"]
        before = categories(host, a, first) if eager else {}
        if not eager:
            causal["classification"] = (
                "NOT_APPLICABLE" if light["mode"] == "serial" else "NO_EAGER_FORWARD_IN_RECORD"
            )
        own = categories(draft_trace, a, b)
        scan = [
            r for r in draft_trace if r["category"] == "draft_full_pool_scan" and touching(r, a, b)
        ]
        resident = categories(host, a, b).get("resident_block_audit", {})
        cycle = {
            "index": index,
            "boundary_ns": [a, b],
            "committed_tokens": step["committed_tokens"],
            "step_wall_ms": (b - a) / 1e6,
            "draft_GPU_by_purpose": device(draft, a, b)["by_purpose"],
            "causal": causal,
            "draft_causal_categories": own,
            "before_first_forward_host": before,
            "full_scans": len(scan) if audit["schema_version"] != "legacy-full" else None,
            "full_prefix_visits": sum(r.get("prefix_visits", 0) for r in scan)
            if audit["schema_version"] != "legacy-full"
            else None,
            "resident_pool_checks": resident.get("count", 0),
            "runtime_affected_request_checks": sum(
                r.get("affected_requests", 0)
                for r in draft_trace
                if r["category"] == "runtime_checked_requests" and touching(r, a, b)
            ),
            "allocator_validation_ms": sum(
                r.get("allocator_check_ns", 0)
                for r in draft_trace
                if r["category"] == "runtime_allocator_operation" and touching(r, a, b)
            )
            / 1e6,
        }
        cycles.append(cycle)
    target_rows = [r for d in targets for r in d["forwards"]]
    eager = [r for r in draft["forwards"] if r.get("purpose") == "eager"]
    overlap = {
        side: duration(
            intersect(bounds(target_rows, start, end, inner), bounds(eager, start, end, inner))
        )
        for side, inner in (("lower_ms", True), ("upper_ms", False))
    }
    window_host = backend["fixed_host"]["intervals"]
    coverage = bounds(draft["forwards"] + target_rows, start, end, False) + [
        (max(start, r["start_ns"]), min(end, r["end_ns"]))
        for r in window_host
        if touching(r, start, end)
    ]
    classified = Counter(c["causal"].get("classification", "MISSING") for c in cycles)
    if light["mode"] == "serial":
        classified = Counter(NOT_APPLICABLE=len(steps))
    queue_rows = [c["causal"].get("owner_queue", {}) for c in cycles]
    queue_rows = [q for q in queue_rows if q.get("status") == "OBSERVED"]
    normal_groups = {}
    proposal_forwards = [
        f
        for f in draft["forwards"]
        if f.get("purpose") == "proposal" and start <= f["host_start_ns"] <= end
    ]
    for kind in ("normal", "recovery", "mixed"):
        spans = [
            r
            for r in draft_trace
            if r["category"] == "normal_recovery_proposal"
            and r.get("request_kinds")
            and (
                next(iter(set(r["request_kinds"].values())))
                if len(set(r["request_kinds"].values())) == 1
                else "mixed"
            )
            == kind
        ]
        forwards = [
            f
            for f in proposal_forwards
            if any(r["start_ns"] <= f["host_start_ns"] <= r["end_ns"] for r in spans)
        ]
        if light["mode"] == "serial" and kind == "normal":
            forwards = proposal_forwards
        normal_groups[kind] = dict(
            host_union_ms=duration(
                [(max(start, r["start_ns"]), min(end, r["end_ns"])) for r in spans]
            ),
            GPU_forward_count=len(forwards),
            GPU_event_sum_ms=sum(f["gpu_event_ms"] for f in forwards),
            B_histogram=dict(Counter(str(f["B"]) for f in forwards)),
            status="OBSERVED" if spans or forwards else "NOT_OBSERVED_OR_MISSING",
        )
    return {
        "exclusive_process_threads": exclusive_lanes(
            [r for h in lanes.values() for r in h.get("intervals", [])] + all_trace, start, end
        ),
        "normal_recovery": normal_groups,
        "startup": dict(
            observed_cycles=len(queue_rows),
            expected_cycles=len(steps) if light["mode"] == "serial-eager" else 0,
            submit_to_dequeue_ms=stats([q["submit_to_dequeue_ms"] for q in queue_rows]),
            dequeue_to_first_GPU={
                side: stats([q["dequeue_to_first_gpu"][side] for q in queue_rows])
                for side in ("lower_ms", "upper_ms")
            },
            admission_host_ms=stats(
                [
                    c["draft_causal_categories"]["physical_begin_gpu_continuations"]["union_ms"]
                    for c in cycles
                    if c["draft_causal_categories"]
                    .get("physical_begin_gpu_continuations", {})
                    .get("count")
                ]
            ),
        ),
        "mode": light["mode"],
        "draft_audit": audit,
        "original_qualification": {
            k: light[k]
            for k in (
                "execution_status",
                "measurement_status",
                "cleanup_status",
                "formal_comparison_eligible",
            )
        },
        "window_ms": light["measured_window_ms"],
        "steps": len(steps),
        "committed_tokens": light["committed_window_tokens"],
        "throughput_tok_s": light["decode_throughput_tok_s"],
        "tokens_per_step": light["committed_window_tokens"] / len(steps),
        "complete_step_wall_ms": stats([c["step_wall_ms"] for c in cycles]),
        "outside_complete_steps_ms": (end - start) / 1e6
        - duration([(s["start_ns"], s["end_ns"]) for s in steps]),
        "target_ranks": {
            str(d["identity"]["global_rank"]): device(d, start, end) for d in targets
        },
        "target_TP_union": {
            side: duration(bounds(target_rows, start, end, inner))
            for side, inner in (("lower_ms", True), ("upper_ms", False))
        },
        "draft_device": device(draft, start, end),
        "native_forward_overlap": overlap,
        "overlap_denominator": "window and native Target TP union, not rank duration sum",
        "first_forward_classes": dict(classified),
        "definite_overlap_cycle_fraction": classified["RECORDED_OVERLAP"] / len(steps)
        if light["mode"] == "serial-eager"
        else None,
        "coordinator_exclusive": exclusive_main(
            runtime.get("host", {}).get("intervals", []), traces["coordinator"]["rows"], start, end
        ),
        "inclusive_host_lanes": {
            name: categories(h.get("intervals", []), start, end) for name, h in lanes.items()
        },
        "inclusive_causal_lanes": {
            name: categories(t["rows"], start, end) for name, t in traces.items()
        },
        "trace_coverage": {
            name: {k: v for k, v in t.items() if k != "rows"} for name, t in traces.items()
        },
        "uncovered_by_draft_host_or_recorded_gpu_outer_ms": (end - start) / 1e6
        - duration(coverage),
        "eager_window": light.get("rolling_eager", {}).get("window_counters"),
        "cycles": cycles,
        "semantics": [
            "no audit-subtracted throughput; source qualification unchanged",
            "physical_gpu_audit is inclusive executed work, not recorder overhead",
            "host lanes, child spans, RPC waits and GPU overlap; do not add",
            "allocator_validation_ms is timed callback CPU sum within worker operation",
            "full/runtime deltas are not an additive causal wall-time decomposition",
            "raw native/causal events retained once in bounded evidence bundle",
            "uncovered is not inferred Python, GIL or idle; no observer on/off isolation",
        ],
        "recording_self_cost": "NOT_ISOLATED: timer/TRACE append and final export; "
        "inclusive instrumented functions are not observer overhead",
    }


def write(value, output):
    payload = (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()
    require(len(payload) <= MAX_OUTPUT, "compact report exceeds 8 MiB bound")
    with Path(output).open("xb") as f:
        f.write(payload)
    return {
        "output": str(output),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def report(root, output, expected_commit):
    root = Path(root).resolve()
    require(not Path(output).resolve().is_relative_to(root), "report must be outside source root")
    reader = Reader(root, MAX_FILE, MAX_POINT, 100000)
    config = reader.read(root / "scan-config.json", required=True)
    require(config["execution"]["git_commit"] == expected_commit, "scan SHA differs")
    found = []
    for p in sorted((root / "runs").glob("*/point.json")):
        point = reader.read(p, required=True)
        if point.get("probe"):
            continue
        require(
            point["mode"] in ("serial", "serial-eager") and point["batch"] == 16, "B16 pair only"
        )
        found.append(p.parent)
    require(len(found) == 1, "audit point root must contain exactly one performance point")
    directory = found[0]
    light = reader.read(directory / "light-summary.json", required=True)
    life = reader.read(directory / "process-lifecycle.json", required=True)
    require(
        life["owned_cleanup_completed"] and not life["remaining_owned_pids"],
        "point must have completed cleanup",
    )
    require(
        light["git_commit"] == expected_commit
        and light["workload_sha256"] == config["workload_sha256"],
        "point identity differs",
    )
    runtime = reader.read(directory / "runtime.json", required=True)
    backend = reader.read(directory / "draft-backend-report.json", required=True)
    value = analyze(runtime, backend, light)
    require(
        value["draft_audit"]["mode"] == config["options"].get("draft_audit", "full"),
        "backend audit mode differs from frozen metadata",
    )
    value.update(
        schema_version="specrhythm.audit-point.v1",
        source_commit=expected_commit,
        workload_sha256=config["workload_sha256"],
        options=config["options"],
        inventory=list(reader.inventory.values()),
        source_root=str(root),
        raw_events_embedded=False,
        original_results_unchanged=True,
    )
    return write(value, output)


def compare(paths, output, expected_commit):
    points = {}
    for path in paths:
        require(Path(path).stat().st_size <= MAX_OUTPUT, "point report too large")
        value = json.loads(Path(path).read_text())
        require(value["source_commit"] == expected_commit, "four-point SHA mismatch")
        key = value["mode"] + "+" + value["draft_audit"]["mode"]
        require(key not in points, "duplicate comparison point")
        points[key] = value
    require(
        set(points)
        == {m + "+" + a for m in ("serial", "serial-eager") for a in ("full", "runtime")},
        "need four mode/audit points",
    )
    require(len({p["workload_sha256"] for p in points.values()}) == 1, "workload mismatch")
    opts = [{k: v for k, v in p["options"].items() if k != "draft_audit"} for p in points.values()]
    require(all(o == opts[0] for o in opts), "four-point measurement/observation options differ")
    # Each point appears exactly once. No re-embedding raw event arrays or historical results.
    return write(
        dict(
            schema_version="specrhythm.audit-four-point.v1",
            points=points,
            source_commit=expected_commit,
            original_results_unchanged=True,
        ),
        output,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--reports", nargs=4, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args(argv)
    require(bool(args.root) != bool(args.reports), "choose one root or four reports")
    try:
        result = (
            report(args.root, args.output, args.expected_commit)
            if args.root
            else compare(args.reports, args.output, args.expected_commit)
        )
        print(json.dumps(result))
    except (OSError, ValueError, KeyError, TypeError) as error:
        failure = dict(
            schema_version="specrhythm.audit-report-error.v1",
            evidence_status="MISSING_INVALID_OR_LIMIT",
            error=str(error),
            original_results_unchanged=True,
        )
        if not args.output.exists():
            write(failure, args.output)
        print(json.dumps(failure), file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
