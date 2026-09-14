"""Offline native cross-home joins; physical intervals are unioned, never summed."""

from collections import Counter, defaultdict

from specrhythm.serving.eager_evidence_export import bounds, duration, intersect
from specrhythm.serving.k3_dispatch_evidence import dispatch, rejection_timeline, target_groups


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
    target_batches = target_groups(runtime, errors)
    for batch in target_batches:
        for key, c in batch["claims"].items():
            owner = claim_map.get((key, c["prefix_version"]), {})
            if any(
                owner.get(k) != c[k]
                for k in ("request_id", "prefix_version", "proposal_id", "claim_id")
            ):
                errors.append(
                    "Target claim differs from authoritative owner: step " + str(batch["index"])
                )
    request_pairs, parent_pairs, draft_rows = defaultdict(list), [], []
    cross_steps = defaultdict(list)
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
            parent_version = version - int(kind == "normal_extension")
            if (
                rid not in homes
                or (
                    kind in ("lookahead", "post", "repair")
                    or (kind == "normal_extension" and version > 0)
                )
                and ((rid, parent_version) not in claim_map)
            ):
                errors.append(
                    "Draft request/parent proposal dependency missing: physical index "
                    + str(index)
                )
            if kind == "lookahead" and (
                not binding.get("continuation_id")
                or binding["continuation_id"] != parent.get("continuation_id")
            ):
                errors.append(
                    "Draft continuation/settled parent mismatch: physical index " + str(index)
                )
            groups[role][ni] = g
            row = dict(
                request_id=rid,
                prefix_version=version,
                version_scope="work round; normal_extension uses next proposal version; "
                "post/lookahead use parent version",
                produces_prefix_version=(
                    version + 1 if kind in ("lookahead", "post") else version
                ),
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
                native_bounds={
                    k: g[k]
                    for k in ("start_lower_ns", "start_upper_ns", "end_lower_ns", "end_upper_ns")
                    if k in g
                },
            )
            draft_rows.append(row)
            if len(first_rows) < 128:
                first_rows.append(row)
            if role == "rejection_recovery" and not captured_recovery:
                captured_recovery = True
                timeline = preceding[:]
            if captured_recovery and len(timeline) < 128:
                timeline.append(row)
            preceding = (preceding + [row])[-16:]
            for batch in target_batches:
                claims, tg = batch["claims"], batch["native"]
                if rid not in claims:
                    request_pairs[role].append((g, tg))
                    if batch["step"].get("window") and role in (
                        "ordinary_draft", "rejection_recovery"
                    ):
                        cross_steps[batch["index"]].append((g, tg))
                    if homes.get(rid) in ("A", "B") and any(
                        c["home_cohort"] != homes[rid] for c in claims.values()
                    ):
                        pairs[role].append((g, tg))
                elif role == "eager_lookahead":
                    c = claims[rid]
                    if (
                        c["prefix_version"] == version
                        and c["proposal_id"] == row["parent_proposal_id"]
                    ):
                        parent_pairs.append((g, tg))
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

    def union_overlap(joined):
        return (
            {
                side: duration(
                    [
                        interval
                        for g, tg in joined
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

    def not_hidden(rows):
        return (
            {
                side: max(
                    0,
                    duration(bounds(rows, start, end, inner))
                    - duration(
                        intersect(
                            bounds(rows, start, end, inner), bounds(targets, start, end, not inner)
                        )
                    ),
                )
                for side, inner in (("lower_ms", True), ("upper_ms", False))
            }
            if complete
            else None
        )

    overlap, uncovered, costs = {}, {}, {}
    for role in ("ordinary_draft", "rejection_recovery", "eager_lookahead", "KV_repair"):
        rows = list(groups[role].values())
        costs[role] = dict(
            forward_count=len(rows),
            B_histogram=dict(Counter(str(r["B"]) for r in rows)),
            GPU_event_sum_ms=sum(r["gpu_event_ms"] for r in rows),
            count_scope="physical batch participating in role; mixed roles not additive",
        )
        overlap[role] = union_overlap(pairs[role])
        if role == "rejection_recovery":
            uncovered[role] = not_hidden(rows)
    recovery_rows = list(groups["rejection_recovery"].values())
    recovery_union = ({side: duration(bounds(recovery_rows, start, end, inner))
                       for side, inner in (("lower_ms", True), ("upper_ms", False))}
                      if complete else None)
    recovery_covered = union_overlap(request_pairs["rejection_recovery"])
    coverage = dict(
        status=("INCOMPLETE" if not complete else
                "NO_RECOVERY" if not recovery_rows else "COMPLETE"),
        denominator_ms=recovery_union, covered_ms=recovery_covered,
        fraction=None if not complete or not recovery_union["lower_ms"] else dict(
            lower=recovery_covered["lower_ms"] / recovery_union["upper_ms"],
            upper=min(1.0, recovery_covered["upper_ms"] / recovery_union["lower_ms"])),
        measurement_window_ns=[start, end],
        denominator="union of native physical Draft intervals participating in rejection "
        "recovery; unique native forward indices; selected host launches in measurement, "
        "device bounds clipped to measurement window; mixed batches counted once",
        numerator="union of intersections with both TP ranks where the recovery request "
        "is absent from Target claims; duplicate bindings/ranks/roles do not multiply time",
        uncertainty="lower fraction = covered lower / denominator upper; upper fraction "
        "= covered upper / denominator lower, capped at 1; absent bounds never become zero",
    )
    measured_batches = [b for b in target_batches if b["step"].get("window")]
    cross_counts = dict(
        total_measured_Target_steps=len(measured_batches),
        definite=None if not complete else sum(
            union_overlap(cross_steps[b["index"]])["lower_ms"] > 0 for b in measured_batches),
        possible_only=None if not complete else sum(
            union_overlap(cross_steps[b["index"]])["lower_ms"] == 0
            and union_overlap(cross_steps[b["index"]])["upper_ms"] > 0 for b in measured_batches),
        meaning="OBSERVED means any definite nonzero other-request ordinary/recovery overlap; "
        "it does not mean recovery is sufficiently hidden",
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
        cross_cohort_pipeline_behavior=(
            "INCOMPLETE" if not complete else "OBSERVED" if observed else "NOT_DEMONSTRATED"
        ),
        cross_request_pipeline_behavior=(
            "INCOMPLETE"
            if not complete
            else "OBSERVED"
            if any(
                union_overlap(request_pairs[role])["lower_ms"] > 0
                for role in ("ordinary_draft", "rejection_recovery")
            )
            else "NOT_DEMONSTRATED"
        ),
        cross_home_native_overlap=overlap,
        cross_request_native_overlap={
            r: union_overlap(v) for r, v in ((role, request_pairs[role]) for role in costs)
        },
        parent_eager_native_overlap=union_overlap(parent_pairs),
        cross_request_overlap_steps=cross_counts,
        recovery_coverage_by_other_requests=coverage,
        all_Draft_GPU_union_not_hidden_by_any_Target=not_hidden(native),
        association="native internal IDs -> scheduler stable IDs -> authoritative claim "
        "proposal/version; Draft physical binding and parent dependency; then home; "
        "mixed physical batches participate in multiple roles, never additive",
        dispatch=dispatch(runtime, target_batches, events),
        rejection_cycle=rejection_timeline(runtime, backend, target_batches, draft_rows),
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
