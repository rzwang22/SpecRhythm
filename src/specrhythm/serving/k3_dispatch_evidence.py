"""Offline request/TP association and bounded dispatch landmarks, no host/GPU sum."""

from specrhythm.serving.eager_evidence_export import bounds, duration, intersect
from specrhythm.serving.fixed_results import stats
from specrhythm.serving.k3_resident_evidence import resident_step, summarize


def target_groups(runtime, errors):
    groups = []
    for index, step in enumerate(runtime["target_steps"]):
        claims = step.get("ping_admission", {}).get("claims", [])
        if not claims:
            continue
        by_id = {c["request_id"]: c for c in claims}
        mapping = {r["internal_request_id"]: r["request_id"] for r in step["rows"]}
        if len(by_id) != len(claims) or set(mapping.values()) != set(by_id):
            errors.append("Target step/claim identity mismatch: " + str(index))
            continue
        for c in claims:
            p = c.get("proposal", {})
            if (
                p.get("request_id") != c["request_id"]
                or p.get("round_id") != c["prefix_version"]
                or p.get("runtime_provenance", {}).get("rolling_proposal_id") != c["proposal_id"]
            ):
                errors.append("Target proposal/version/claim mismatch: step " + str(index))
        native, ranks, indices = [], [], []
        for d in runtime["target_devices"]:
            rank = d["device"]["identity"]["global_rank"]
            matches = [
                (ni, g)
                for ni, g in enumerate(d["device"]["forwards"])
                if step["start_ns"] <= g["host_start_ns"] <= step["end_ns"]
            ]
            if len(matches) != 1:
                errors.append(
                    "Target native interval missing/ambiguous: step/rank " + str((index, rank))
                )
                continue
            ni, g = matches[0]
            if set(g.get("internal_request_ids", [])) != set(mapping) or len(
                g.get("internal_request_ids", [])
            ) != len(mapping):
                errors.append(
                    "Target native request identity missing/different: step/rank "
                    + str((index, rank))
                )
                continue
            if not all(
                type(g.get(k)) is int
                for k in ("start_lower_ns", "start_upper_ns", "end_lower_ns", "end_upper_ns")
            ):
                errors.append(
                    "Target native clock bounds missing: step/rank " + str((index, rank))
                )
                continue
            ranks.append(rank)
            native.append(g)
            indices.append(ni)
        if sorted(ranks) != [0, 1]:
            errors.append("Target TP rank missing/duplicated: step " + str(index))
        groups.append(
            dict(
                index=index,
                step=step,
                claims=by_id,
                native=native,
                native_ranks=ranks,
                native_indices=indices,
            )
        )
    return groups


def dispatch(runtime, groups, events):
    published = {
        (e["request_id"], e["prefix_version"]): e["timestamp_ns"]
        for e in events
        if e["event"] == "ready"
    }
    host = runtime.get("host", {})
    spans = host.get("causal_timeline", {}).get("rows", [])
    timers = host.get("intervals", [])
    out = []
    all_targets = [g for group in groups for g in group["native"]]
    for group in groups:
        step, native = group["step"], group["native"]
        if not step.get("window") or len(native) != 2:
            continue
        claim = min(c["claimed_ns"] for c in group["claims"].values())
        gpu = min(g["start_lower_ns"] for g in native)
        sched_a, sched_b = step.get("schedule_start_ns"), step.get("schedule_end_ns")

        def selected(rows, category, step_start=step["start_ns"], gpu=gpu):
            return [
                r for r in rows if r["category"] == category and step_start <= r["start_ns"] <= gpu
            ]

        def union_ms(rows, claim=claim, gpu=gpu):
            return duration(
                [
                    (max(claim, r["start_ns"]), min(gpu, r["end_ns"]))
                    for r in rows
                    if r["end_ns"] > claim and r["start_ns"] < gpu
                ]
            )

        ready_times = [
            published.get((c["request_id"], c["prefix_version"])) for c in group["claims"].values()
        ]
        first_ready = (
            max(runtime["measurement_start_ns"], min(ready_times))
            if all(t is not None for t in ready_times)
            else None
        )
        ready_idle = None if first_ready is None else {}
        if first_ready is not None:
            for side, inner in (("lower_ms", False), ("upper_ms", True)):
                stop = min(g["start_upper_ns" if inner else "start_lower_ns"] for g in native)
                ready_idle[side] = max(
                    0,
                    (stop - first_ready) / 1e6
                    - duration(
                        intersect(
                            [(first_ready, stop)], bounds(all_targets, first_ready, stop, inner)
                        )
                    ),
                )
        row = dict(
            resident_schedule=resident_step(step, spans),
            READY_while_Target_idle_ms=ready_idle,
            step_index=group["index"],
            request_versions=[
                dict(
                    request_id=c["request_id"],
                    proposal_id=c["proposal_id"],
                    prefix_version=c["prefix_version"],
                    home_cohort=c["home_cohort"],
                    ready_ns=c["ready_ns"],
                    owner_ready_published_ns=published.get((c["request_id"], c["prefix_version"])),
                    claimed_ns=c["claimed_ns"],
                )
                for c in group["claims"].values()
            ],
            claim_ns=claim,
            Target_start_lower_ns=gpu,
            claim_to_Target_ms=(gpu - claim) / 1e6,
            scheduler_ms=(
                (sched_b - sched_a) / 1e6 if sched_a is not None and sched_b is not None else None
            ),
            prefix_hash_visits=(
                len(selected(timers, "prefix_hash_and_block_record"))
                if "intervals" in host
                else None
            ),
            prefix_hash_union_ms=(
                union_ms(selected(timers, "prefix_hash_and_block_record"))
                if "intervals" in host
                else None
            ),
            block_audit_union_ms=(
                union_ms(selected(timers, "resident_block_audit")) if "intervals" in host else None
            ),
            scheduler_phases_ms={
                k: union_ms(selected(spans, k)) if selected(spans, k) else None
                for k in (
                    "target_pool_pre_schedule",
                    "target_resident_stock_schedule",
                    "target_pool_post_schedule",
                )
            },
            prompt_proofs=[
                {
                    k: r[k]
                    for k in (
                        "resident_rows",
                        "prompt_hashes",
                        "prompt_proofs_reused",
                        "current_prompt_tokens_compared",
                    )
                }
                for r in selected(spans, "target_pool_prompt_proof")
            ],
        )
        # The broad scheduler span is only one coordinator lane. Internal unknown
        # cost stays unknown; it is not assigned to RPC, GIL or another GPU.
        row["outside_scheduler_ms"] = (
            (gpu - claim) / 1e6 - max(0, min(gpu, sched_b) - max(claim, sched_a)) / 1e6
            if sched_a is not None and sched_b is not None
            else None
        )
        out.append(row)
    return dict(
        rows=out,
        resident_schedule=summarize(out),
        claim_to_Target_ms=stats([r["claim_to_Target_ms"] for r in out]),
        published_READY_to_claim_ms=stats(
            [
                (c["claimed_ns"] - c["owner_ready_published_ns"]) / 1e6
                for r in out
                for c in r["request_versions"]
                if c["owner_ready_published_ns"] is not None
            ]
        ),
        missing_READY_publications=sum(
            c["owner_ready_published_ns"] is None for r in out for c in r["request_versions"]
        ),
        scheduler_ms=stats([r["scheduler_ms"] for r in out if r["scheduler_ms"] is not None]),
        missing_scheduler_phase_steps=sum(
            any(v is None for v in r["scheduler_phases_ms"].values()) for r in out
        ),
        semantics="one coordinator lane; hash/audit are nested in scheduler phases; "
        "do not sum these columns or add owner/Target GPU time",
        READY_idle_scope="owner publication -> claimed Target; clipped to measurement; "
        "GPU-idle does not prove CPU sampling/dispatch is free. Across steps not additive. "
        "ready_ns: physical completion; owner_ready_published_ns: actual READY event; "
        "neither is substituted for authoritative claim",
        native_clock="lower endpoint; bounded CUDA anchor uncertainty retained in raw records",
    )


def rejection_timeline(runtime, backend, groups, draft_rows):
    """One real request/version rejection chain, bounded selected A/B surroundings."""
    a, b = runtime["measurement_start_ns"], runtime["measurement_end_ns"]
    events = backend["prepost"]["pingpong"]["events"]
    rejection = next(
        (
            e
            for e in events
            if e["event"] == "settlement" and e.get("correction") and a <= e["timestamp_ns"] <= b
        ),
        None,
    )
    if rejection is None:
        return dict(status="NO_RECORDED_REJECTION", rows=[])
    rid, version = rejection["request_id"], rejection["prefix_version"]
    parent = next(
        (
            g
            for g in groups
            if rid in g["claims"] and g["claims"][rid]["prefix_version"] == version
        ),
        None,
    )
    if parent is None:
        return dict(status="MISSING_PARENT_TARGET", rows=[])
    following = next(
        (
            e
            for e in events
            if e["event"] == "ready"
            and e["request_id"] == rid
            and e["prefix_version"] == version + 1
        ),
        None,
    )
    end = following["timestamp_ns"] if following else rejection["timestamp_ns"]
    adjacent = [g for g in groups if parent["step"]["start_ns"] <= g["step"]["start_ns"] <= end]
    # Include the next other-home dispatch even if it starts after READY (old zero-overlap).
    later = next((g for g in groups if g["step"]["start_ns"] > parent["step"]["end_ns"]), None)
    if later and later not in adjacent:
        adjacent.append(later)
    publications = {
        (e["request_id"], e["prefix_version"]): e["timestamp_ns"]
        for e in events
        if e["event"] == "ready"
    }
    rows = []
    for g in adjacent:
        for c in g["claims"].values():
            binding = {
                k: c[k]
                for k in ("request_id", "proposal_id", "prefix_version", "home_cohort", "claim_id")
            }
            publication = publications.get((c["request_id"], c["prefix_version"]))
            if publication is not None:
                rows.append(dict(event="READY_publication", timestamp_ns=publication, **binding))
            rows += [
                dict(event=kind, timestamp_ns=c[key], **binding)
                for kind, key in (("READY", "ready_ns"), ("claim", "claimed_ns"))
            ]
        for rank, ni, native in zip(g["native_ranks"], g["native_indices"], g["native"]):
            rows.append(
                dict(
                    event="Target_forward",
                    timestamp_ns=native["start_lower_ns"],
                    step_index=g["index"],
                    TP_rank=rank,
                    native_forward_index=ni,
                    forward_id=f"target-rank:{rank}-native-index:{ni}",
                    forward_id_source="target_devices rank device.forwards original array index",
                    requests=[
                        {k: c[k] for k in ("request_id", "prefix_version", "proposal_id")}
                        for c in g["claims"].values()
                    ],
                    **{k: v for k, v in native.items() if k.endswith(("_lower_ns", "_upper_ns"))},
                )
            )
    rows.append(
        dict(
            event="feedback",
            boundary="owner validated feedback, not transport enqueue",
            timestamp_ns=rejection["feedback_ns"],
            request_id=rid,
            prefix_version=version,
            proposal_id=parent["claims"][rid]["proposal_id"],
        )
    )
    transport = [
        r
        for r in backend.get("fixed_host", {}).get("causal_timeline", {}).get("rows", [])
        if r.get("operation") == "pp_feedback"
        and r["category"] in ("owner_queue_submit", "owner_dequeue", "owner_dispatch")
        and any(
            x.get("request_id") == rid and x.get("round_id") == version
            for x in r.get("requests", [])
        )
    ]
    for r in transport:
        rows.append(
            dict(
                event=r["category"],
                timestamp_ns=r["start_ns"],
                start_ns=r["start_ns"],
                end_ns=r["end_ns"],
                pid=r["pid"],
                thread_id=r["thread_id"],
                request_id=rid,
                prefix_version=version,
                proposal_id=parent["claims"][rid]["proposal_id"],
            )
        )
    rows.extend(
        dict(event="Draft_forward", timestamp_ns=r["host_start_ns"], **r)
        for r in draft_rows
        if parent["step"]["start_ns"] <= r["host_start_ns"] <= end
    )
    if following:
        rows.append({**following, "event": "next_READY"})
    rows.sort(key=lambda r: r["timestamp_ns"])
    return dict(
        status="COMPLETE" if following else "MISSING_NEXT_READY",
        owner_feedback_transport_status=(
            "COMPLETE"
            if {r["category"] for r in transport}
            == {"owner_queue_submit", "owner_dequeue", "owner_dispatch"}
            else "MISSING_LANDMARKS"
        ),
        selected_request_id=rid,
        selected_prefix_version=version,
        rows=rows[:128],
        omitted_rows=max(0, len(rows) - 128),
        row_limit=128,
        clock_note="host landmarks and native bounds are distinct; no summed lane times",
    )
