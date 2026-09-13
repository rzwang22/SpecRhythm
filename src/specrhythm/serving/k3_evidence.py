"""Offline native cross-home joins; physical intervals are unioned, never summed."""

from collections import Counter, defaultdict

from specrhythm.serving.eager_evidence_export import bounds, duration, intersect


def pipeline(runtime, backend):
    start, end = runtime["measurement_start_ns"], runtime["measurement_end_ns"]
    ping = backend["prepost"]["pingpong"]
    events, homes = ping["events"], ping["home_cohorts"]
    settlements = {
        (r["request_id"], r["prefix_version"]): r for r in events if r["event"] == "settlement"
    }
    claim_map = {
        (c["request_id"], c["prefix_version"]): c
        for e in events
        if e["event"] == "admission"
        for c in e["claims"]
    }
    native = backend["fixed_device"]["forwards"]
    targets = [g for d in runtime["target_devices"] for g in d["device"]["forwards"]]
    groups, pairs, timeline, errors = defaultdict(dict), defaultdict(list), [], []
    first_rows, preceding, captured_recovery = [], [], False
    for index, f in enumerate(backend["prepost_physical"]["forwards"]):
        if not start <= f["start_ns"] <= end:
            continue
        match = [
            (i, g)
            for i, g in enumerate(native)
            if g["purpose"] == f["purpose"]
            and g["B"] == f["B"]
            and f["start_ns"] <= g["host_start_ns"] <= f["end_ns"]
        ]
        if len(match) != 1:
            errors.append("Draft native binding missing/ambiguous: physical index " + str(index))
            continue
        ni, g = match[0]
        for binding in f["bindings"]:
            rid, version = binding["request_id"], binding["round_id"]
            kind = binding.get("work_kind")
            parent = settlements.get((rid, version - int(kind == "normal_extension")), {})
            role = (
                "eager_lookahead"
                if kind == "lookahead"
                else "KV_repair"
                if binding.get("role") == "terminal_repair" or kind == "repair"
                else "rejection_recovery"
                if parent.get("correction")
                else "ordinary_draft"
            )
            groups[role][ni] = g
            row = dict(
                request_id=rid,
                prefix_version=version,
                home_cohort=homes.get(rid),
                role=role,
                physical_record_index=index,
                native_forward_index=ni,
                forward_id="draft-native-index:" + str(ni),
                forward_id_source="fixed_device.forwards array index in original producer",
                materialized_positions=f["materialized_positions"],
                B=f["B"],
                proposal_id=claim_map.get((rid, version), {}).get("proposal_id"),
                parent_proposal_id=claim_map.get(
                    (rid, version - int(kind == "normal_extension")), {}
                ).get("proposal_id"),
                continuation_id=binding.get("continuation_id"),
                feedback_ns=parent.get("feedback_ns"),
                host_start_ns=f["start_ns"],
                host_end_ns=f["end_ns"],
            )
            if len(first_rows) < 128:
                first_rows.append(row)
            if role == "rejection_recovery" and not captured_recovery:
                captured_recovery = True
                timeline = preceding[:]
            if captured_recovery and len(timeline) < 128:
                timeline.append(row)
            preceding = (preceding + [row])[-16:]
            for s in runtime["target_steps"]:
                claims = s.get("ping_admission", {}).get("claims", [])
                if not claims or homes.get(rid) not in ("A", "B"):
                    continue
                if any(c["home_cohort"] == homes[rid] for c in claims):
                    continue  # Strict other-home interval, not just same-device overlap.
                tg = [t for t in targets if s["start_ns"] <= t["host_start_ns"] <= s["end_ns"]]
                pairs[role].append((g, tg))
    complete = (
        bool(native and targets)
        and not errors
        and all(
            type(g.get(k)) is int
            for g in native + targets
            for k in ("start_lower_ns", "start_upper_ns", "end_lower_ns", "end_upper_ns")
        )
    )
    if not complete and not errors:
        errors.append("native Draft/Target intervals absent or bounds missing")
    overlap, uncovered, costs = {}, {}, {}
    for role in ("ordinary_draft", "rejection_recovery", "eager_lookahead", "KV_repair"):
        rows = list(groups[role].values())
        costs[role] = dict(
            forward_count=len(rows),
            B_histogram=dict(Counter(str(r["B"]) for r in rows)),
            GPU_event_sum_ms=sum(r["gpu_event_ms"] for r in rows),
            count_scope="physical batch participating in role; mixed roles not additive",
        )
        overlap[role] = (
            {
                side: duration(
                    [
                        interval
                        for g, tg in pairs[role]
                        for interval in intersect(
                            bounds([g], start, end, inner), bounds(tg, start, end, inner)
                        )
                    ]
                )
                for side, inner in (("lower_ms", True), ("upper_ms", False))
            }
            if complete
            else None
        )
        if role == "rejection_recovery":
            # Union difference bounds, not subtraction from throughput or from inclusive spans.
            uncovered[role] = (
                dict(
                    lower_ms=max(
                        0,
                        duration(bounds(rows, start, end, True))
                        - duration(
                            intersect(
                                bounds(rows, start, end, True), bounds(targets, start, end, False)
                            )
                        ),
                    ),
                    upper_ms=max(
                        0,
                        duration(bounds(rows, start, end, False))
                        - duration(
                            intersect(
                                bounds(rows, start, end, False), bounds(targets, start, end, True)
                            )
                        ),
                    ),
                )
                if complete
                else None
            )
    polls = [e for e in events if e["event"] == "admission" and start <= e["timestamp_ns"] <= end]
    pending = [r for e in polls for r in e.get("waiting_inventory", [])]
    observed = any(
        v and v["lower_ms"] > 0
        for k, v in overlap.items()
        if k in ("ordinary_draft", "rejection_recovery")
    )
    return dict(
        evidence_integrity="COMPLETE" if complete else "INCOMPLETE",
        errors=errors,
        cross_cohort_pipeline_behavior="OBSERVED" if observed else "NOT_DEMONSTRATED",
        cross_home_native_overlap=overlap,
        per_role_forwards=costs,
        recovery_GPU_union_not_hidden_by_any_Target=uncovered,
        wait_reasons=dict(Counter(r["reason"] for r in pending)),
        empty_admissions_with_other_READY=sum(
            not e["claims"] and bool(e["ready_inventory"]) for e in polls
        ),
        timeline=timeline if captured_recovery else first_rows,
        timeline_limit=128,
        recovery_timeline_status="RECORDED" if captured_recovery else "NO_RECORDED_RECOVERY",
        timeline_scope="16 rows preceding first recovery then bounded following rows; "
        "fallback first128; raw phased retained",
        timeline_omitted_rows=max(
            0,
            sum(
                len(f["bindings"])
                for f in backend["prepost_physical"]["forwards"]
                if start <= f["start_ns"] <= end
            )
            - len(timeline if captured_recovery else first_rows),
        ),
        zero_overlap_interpretation="join admission waiting_inventory/READY and raw spans; "
        "zero alone cannot distinguish no opportunity from scheduling delay",
        unaccounted="report outside_complete_steps_ms and exclusive process/thread lanes; "
        "not attributed to Python/GIL; missing is not zero",
    )
