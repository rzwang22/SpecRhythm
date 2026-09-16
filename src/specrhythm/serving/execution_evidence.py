"""Compact per-cycle execution evidence, preserving missing endpoints and lane scope."""

from collections import Counter

from specrhythm.serving.eager_evidence_export import duration, touching
from specrhythm.serving.eager_latency import trace_rows
from specrhythm.serving.fixed_results import stats


def unique(rows, category):
    found = [r for r in rows if r.get("category") == category]
    return found[0] if len(found) == 1 else None


def delta(a, b):
    return (b - a) / 1e6 if a is not None and b is not None else None


def execution_path(
    runtime, backend, start, end, *, feedback_operation="synchronize_and_batch_propose"
):
    targets = runtime.get("target_devices", [])
    target_rows = [r for t in targets for r in t.get("rounds", [])]
    target_trace = [r for t in targets for r in trace_rows(t.get("host", {}), start, end)["rows"]]
    draft_trace = trace_rows(backend.get("fixed_host", {}), start, end)["rows"]
    forwards = backend.get("fixed_device", {}).get("forwards", [])
    native_target = [r for t in targets for r in t.get("device", {}).get("forwards", [])]
    prepost = "prepost" in backend
    protocol = backend.get("prepost" if prepost else "rolling_eager", {}).get("events")
    cycles = []
    outcomes = Counter()
    steps = [s for s in runtime.get("target_steps", []) if s.get("window")]
    for index, step in enumerate(steps):
        a, b = step["start_ns"], step["end_ns"]
        tr = [r for r in target_trace if a <= r["start_ns"] <= b]
        dt = [r for r in draft_trace if a <= r["start_ns"] <= b]
        records = [
            r for r in target_rows if a <= r.get("timeline", {}).get("verify_start_ns", -1) <= b
        ]
        gpu = [r for r in forwards if a <= r["host_start_ns"] <= b]
        tg = [r for r in native_target if a <= r["host_start_ns"] <= b]
        gpu_end = max((r.get("end_upper_ns", -1) for r in tg), default=None)
        exchanges = [
            r
            for r in tr
            if r["category"] == "transport_exchange"
            and r.get("operation") == feedback_operation
        ]
        rpc = exchanges[0] if len(exchanges) == 1 else {}
        sampled = unique(tr, "target_sampled_results_received") or {}
        packed = unique(tr, "target_feedback_payload_ready") or {}
        dequeues = [
            r
            for r in dt
            if r["category"] == "owner_dequeue"
            and r.get("operation") == feedback_operation
        ]
        dequeue = dequeues[0] if len(dequeues) == 1 else {}
        ev = [r for r in (protocol or []) if a <= r["end_ns"] <= b]
        parents = [r for r in ev if r["phase"] == "parent_result"]
        settled = [r for r in ev if r["phase"] == ("settled" if prepost else "parent_settled")]
        counts = Counter()
        for r in ev:
            counts.update(r["counter_delta"])
        admissions = {
            rid for r in ev
            if r["phase"] == ("admission" if prepost else "eager_admission")
            for rid in r["request_ids"]
        }
        cycle_outcomes = Counter()
        for rid in admissions:
            pr = [r for r in parents if rid in r["request_ids"]]
            st = [r for r in settled if rid in r["request_ids"]]
            rd = [r for r in records if r["request_id"] == rid]
            # Join only unambiguous request/parent rounds inside a complete step.
            # No bridge inference from a residual counter or truncated causal rows.
            reason = "MISSING_OR_AMBIGUOUS"
            if len(pr) == len(st) == len(rd) == 1:
                reason = (
                    "parent_rejection"
                    if pr[0]["counter_delta"].get("parent_rejections")
                    else "promotion"
                    if st[0]["counter_delta"].get("promotions")
                    else "bridge_mismatch"
                    if st[0]["counter_delta"].get("bridge_mismatches")
                    else "terminal"
                    if rd[0].get("terminal")
                    else "other"
                )
            cycle_outcomes[reason] += 1
        outcomes.update(cycle_outcomes)
        waits = [
            delta(r["end_ns"], rpc.get("end_ns"))
            for r in settled
            if r["counter_delta"].get("promotions") and rpc
        ]
        proposal = [r for r in gpu if r.get("purpose") == "proposal"]
        kinds = Counter()
        for r in proposal:
            values = set(r.get("causal_context", {}).get("request_kinds", {}).values())
            kinds[
                next(iter(values)) if len(values) == 1 else "mixed" if values else "MISSING"
            ] += 1
        upcoming = steps[index + 1] if index + 1 < len(steps) else None
        nxt = [
            r
            for r in native_target
            if upcoming and upcoming["start_ns"] <= r["host_start_ns"] <= upcoming["end_ns"]
        ]
        cycles.append(
            dict(
                index=index,
                boundary_ns=[a, b],
                counters=dict(counts) if protocol is not None else None,
                admitted_outcomes=dict(cycle_outcomes) if protocol is not None else None,
                proposal_forward_B=[r["B"] for r in proposal],
                proposal_forward_kind=dict(kinds),
                promoted_ready_to_response_ms=stats(waits),
                latencies_ms=dict(
                    target_GPU_end_upper_to_sampled_hook=delta(gpu_end, sampled.get("start_ns")),
                    sampled_hook_to_payload_ready=delta(
                        sampled.get("start_ns"), packed.get("start_ns")
                    ),
                    payload_ready_to_transport=delta(packed.get("start_ns"), rpc.get("start_ns")),
                    transport_to_service_receive=delta(
                        rpc.get("start_ns"), rpc.get("service_receive_ns")
                    ),
                    service_receive_to_owner_dequeue=delta(
                        rpc.get("service_receive_ns"), dequeue.get("start_ns")
                    ),
                    transport_to_first_parent_result=delta(
                        rpc.get("start_ns"), min((r["end_ns"] for r in parents), default=None)
                    ),
                    last_settlement_to_transport_response=delta(
                        max((r["end_ns"] for r in settled), default=None), rpc.get("end_ns")
                    ),
                    response_to_next_Target=delta(
                        rpc.get("end_ns"),
                        min((r.get("start_lower_ns") for r in nxt), default=None),
                    ),
                ),
                legacy_GPU_end_to_verify_end_ms=stats(
                    [
                        delta(gpu_end, r["timeline"]["verify_end_ns"])
                        for r in records
                        if gpu_end is not None and "verify_end_ns" in r["timeline"]
                    ]
                ),
                legacy_verify_end_to_state_sync_ms=stats(
                    [
                        delta(r["timeline"]["verify_end_ns"], r["timeline"]["state_sync_start_ns"])
                        for r in records
                        if all(
                            k in r["timeline"] for k in ("verify_end_ns", "state_sync_start_ns")
                        )
                    ]
                ),
            )
        )
    logs = []
    hosts = {"coordinator": runtime.get("host", {}), "draft": backend.get("fixed_host", {})}
    hosts.update(
        {
            "target-" + str(t["device"]["identity"]["global_rank"]): t.get("host", {})
            for t in targets
        }
    )
    for role, host in hosts.items():
        rows = [
            r
            for r in host.get("intervals", [])
            if r["category"] == "log_fsync" and touching(r, start, end)
        ]
        def attribution(row):
            return (
                row.get("log_name") or "MISSING_FILENAME",
                row.get("log_path"), row.get("physical_path"), row.get("write_kind"),
            )

        for key in sorted({attribution(r) for r in rows}, key=repr):
            name, target_path, physical_path, write_kind = key
            selected = [r for r in rows if attribution(r) == key]
            logs.append(
                dict(
                    role=role,
                    log_name=name,
                    log_path=target_path,
                    physical_path=physical_path,
                    write_kind=write_kind,
                    count=len(selected),
                    union_ms=duration(
                        [(max(start, r["start_ns"]), min(end, r["end_ns"])) for r in selected]
                    ),
                )
            )
    window_proposal = [
        r
        for r in forwards
        if r.get("purpose") == "proposal" and start <= r["host_start_ns"] <= end
    ]
    extra = [
        r
        for r in window_proposal
        if not any(s["start_ns"] <= r["host_start_ns"] <= s["end_ns"] for s in steps)
    ]
    return dict(
        schema_version="specrhythm.execution-path.v1",
        cycles=cycles,
        fsync_by_file=logs,
        admitted_outcomes=dict(outcomes) if protocol is not None else None,
        proposal_forwards_outside_complete_steps=[
            dict(
                B=r["B"],
                GPU_ms=r["gpu_event_ms"],
                request_kinds=r.get("causal_context", {}).get("request_kinds", {}),
            )
            for r in extra
        ],
        semantics="counter updates by end timestamp; native forwards by host launch; "
        "per-step unique request joins, not equality across event sets; null is missing; "
        "latencies are landmarks, not additive costs; ready means parent_settled promotion",
        limitations=[
            "sampled hook is after CPU sampler result transfer, not exact decision time",
            "transport span starts before wire write; no exact last-byte clock",
            "legacy traces lack sampled/payload/file markers; do not fill missing with zero",
            "same-file fsync union is per process; do not sum process budgets",
        ],
    )


def qualify(report):
    """Diagnostic integrity is a separate layer from immutable run qualification."""
    errors = []
    audit = report.get("draft_audit", {})
    required_boundaries = {"setup_complete_snapshot", "timing_entry", "before_final_release"}
    if (audit.get("schema_version") != "specrhythm.draft-audit.v1"
            or audit.get("mode") != "runtime"
            or not required_boundaries.issubset(
                {b.get("boundary") for b in audit.get("full_boundaries", [])})):
        errors.append("runtime audit metadata/full boundaries incomplete")
    traces = report.get("trace_coverage", {})
    expected = {"coordinator", "draft", "target-rank-0", "target-rank-1"}
    if set(traces) != expected:
        errors.append("missing host producers")
    for role, trace in traces.items():
        if trace.get("mode") != "light" or trace.get("measurement_status") != "COMPLETE":
            errors.append(
                f"{role}: measurement host trace {trace.get('measurement_status', 'MISSING')}; "
                f"lifetime dropped_rows={trace.get('dropped_rows', 'MISSING')}"
            )
        if trace.get("layout") != "phased":
            errors.append(f"{role}: phased retention metadata missing")
    targets = report.get("target_ranks", {})
    if set(targets) != {"0", "1"} or any(
        t.get("native_coverage") != "COMPLETE" for t in targets.values()
    ):
        errors.append("native Target TP evidence incomplete")
    if report.get("draft_device", {}).get("native_coverage") != "COMPLETE":
        errors.append("native Draft evidence incomplete")
    if report.get("coordinator_exclusive", {}).get("status") != "COMPLETE":
        errors.append("coordinator lane evidence incomplete")
    path = report.get("execution_path", {})
    cycles = path.get("cycles", [])
    if not cycles or len(cycles) != report.get("steps") or any(
        c.get("latencies_ms", {}).get(k) is None for c in cycles for k in (
            "target_GPU_end_upper_to_sampled_hook", "sampled_hook_to_payload_ready",
            "payload_ready_to_transport", "transport_to_service_receive")):
        errors.append("per-cycle Target feedback landmarks incomplete")
    if report.get("mode") in ("serial-eager", "serial-prepost3", "serial-eager-prepost3") and any(
        c.get("latencies_ms", {}).get("service_receive_to_owner_dequeue") is None
        for c in cycles
    ):
        errors.append("per-cycle owner feedback landmarks incomplete")
    if report.get("mode") in ("serial-prepost3", "serial-eager-prepost3"):
        prepost = report.get("prepost", {})
        if prepost.get("status") != "COMPLETE":
            errors.append("prepost protocol/forward evidence incomplete: "
                          + str(prepost.get("errors")))
    if report.get("mode") in ("pingpong-prepost3", "pingpong-eager-prepost3",
                              "serial-k3", "serial-eager-k3", "pingpong-k3", "pingpong-eager-k3"):
        ping = report.get("pingpong", {})
        if ping.get("status") != "COMPLETE":
            errors.append("PingPong protocol evidence incomplete: " + str(ping.get("errors")))
        if any(c.get("latencies_ms", {}).get("service_receive_to_owner_dequeue") is None
               for c in cycles):
            errors.append("per-cycle PingPong owner feedback landmarks incomplete")
    if report.get("mode") in ("serial-k3", "serial-eager-k3", "pingpong-k3", "pingpong-eager-k3"):
        resident = report.get("pingpong", {}).get("pipeline", {}).get(
            "dispatch", {}).get("resident_schedule", {})
        if resident.get("status") != "COMPLETE":
            errors.append("K3 resident scheduling evidence incomplete: "
                          + str(resident.get("errors", "MISSING")))
    missing_fsync = [
        r for r in path.get("fsync_by_file", [])
        if not r.get("log_name") or r["log_name"] == "MISSING_FILENAME"
    ]
    if missing_fsync:
        errors.append("fsync file attribution incomplete")
    from specrhythm.serving.k3_validation import EXPLORATION, not_run, profile_of

    policy_fields = {}
    if "validation_profile" in report:
        policy_fields = not_run(report)
        if profile_of(report) == EXPLORATION:
            from specrhythm.serving.k3 import configuration_of
            from specrhythm.serving.k3_acceptance import full_batch_receipt

            if (report.get("full_output_comparison_run") is not False
                    or report.get("output_equivalence_status") != "NOT_RUN"
                    or report.get("native_geometry_status") != "PASS"
                    or not full_batch_receipt(report.get("native_target_geometry"),
                                              report["mode"], configuration_of(report))):
                errors.append("performance exploration policy/native geometry evidence invalid")
        policy_fields.update(measurement_valid=report.get("measurement_valid"),
                             native_geometry_status=report.get("native_geometry_status"))
    return dict(
        **policy_fields,
        schema_version="specrhythm.execution-evidence-status.v1",
        original_qualification=report.get("original_qualification"),
        original_run_details=report.get("original_run_details"),
        diagnostic_integrity="FAILED" if errors else "COMPLETE",
        failure_layer="diagnostic_evidence" if errors else None,
        errors=errors,
        missing_fsync_attribution=missing_fsync,
        trace_coverage=traces,
        GPU_correctness="see separate physical correctness result",
        performance_conclusion="PENDING; no speedup threshold",
    )


def compare(paths, commits, *, modes=("serial", "serial-eager")):
    import json
    from pathlib import Path

    from specrhythm.serving.audit_layer_report import MAX_OUTPUT
    from specrhythm.serving.common import require

    require(
        len(commits) in (1, 2) and len(set(commits)) == len(commits), "one or two SHAs required"
    )
    points = {}
    for path in paths:
        require(Path(path).stat().st_size <= MAX_OUTPUT, "point report too large")
        row = json.loads(Path(path).read_text())
        key = row["source_commit"] + ":" + row["mode"]
        require(
            key not in points and row["draft_audit"]["mode"] == "runtime",
            "duplicate point or non-runtime audit",
        )
        require(qualify(row)["diagnostic_integrity"] == "COMPLETE", "incomplete point evidence")
        require(
            all(
                row["original_qualification"][k] == "PASS"
                for k in ("execution_status", "measurement_status", "cleanup_status")
            ),
            "invalid execution point",
        )
        points[key] = row
    require(
        set(points)
        == {sha + ":" + mode for sha in commits for mode in modes},
        "incomplete paired versions",
    )
    rows = list(points.values())
    require(
        all(
            r["options"] == rows[0]["options"]
            and r["workload_sha256"] == rows[0]["workload_sha256"]
            for r in rows
        ),
        "paired options/workload differ",
    )
    return dict(
        schema_version="specrhythm.execution-comparison.v1",
        points=points,
        commits=commits,
        raw_events_embedded=False,
        interpretation="one window checks mechanism; stable speedup requires repeats",
    )


def main(argv=None):
    import argparse
    import json
    from pathlib import Path

    from specrhythm.serving.audit_layer_report import MAX_OUTPUT, write

    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--qualify", type=Path)
    group.add_argument("--reports", nargs="+", type=Path)
    parser.add_argument("--commits", nargs="+")
    parser.add_argument("--prepost", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.qualify:
        if args.qualify.stat().st_size > MAX_OUTPUT:
            raise ValueError("point report too large")
        result = qualify(json.loads(args.qualify.read_text()))
    else:
        result = compare(args.reports, args.commits or [],
                         modes=("serial-prepost3", "serial-eager-prepost3") if args.prepost
                         else ("serial", "serial-eager"))
    print(json.dumps(write(result, args.output)))
    if result.get("diagnostic_integrity") == "FAILED":
        print(json.dumps({
            k: result[k] for k in (
                "failure_layer", "original_qualification", "diagnostic_integrity",
                "missing_fsync_attribution", "performance_conclusion",
            )
        }, sort_keys=True))
        print("evidence_status=" + str(args.output))
        raise SystemExit("diagnostic_evidence FAILED: " + "; ".join(result["errors"]))


if __name__ == "__main__":
    main()
