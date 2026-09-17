"""Offline, version-joined feedback cycles. No interpolation or cross-thread CPU sums."""

from collections import Counter, defaultdict

from specrhythm.serving.eager_evidence_export import duration, intersect
from specrhythm.serving.fixed_results import stats
from specrhythm.serving.k3_dispatch_evidence import target_groups

SCHEMA = "specrhythm.k3-cycle-ledger.v1"


def ledger(landmarks):
    """Adjacent endpoints telescope on the SAME sample; missing is never zero."""
    missing = [k for k, v in landmarks.items() if type(v) is not int]
    pairs = list(landmarks.items())
    reversed_pairs = [
        a + "->" + b
        for (a, x), (b, y) in zip(pairs, pairs[1:])
        if type(x) is int and type(y) is int and y < x
    ]
    if missing or reversed_pairs:
        return dict(status="INCOMPLETE", missing=missing, reversed=reversed_pairs)
    parts = {a + "->" + b: y - x for (a, x), (b, y) in zip(pairs, pairs[1:])}
    total = pairs[-1][1] - pairs[0][1]
    return dict(
        status="COMPLETE",
        total_ns=total,
        segments_ns=parts,
        closure_residual_ns=total - sum(parts.values()),
    )


def partition(rows, start, end):
    """A thread's covered wall time, innermost span wins; crossing labels are ambiguous.

    Sweep endpoints rather than quadratic token/event sampling. Inclusive spans
    are reported separately; uninstrumented time is unaccounted, never 'Python'.
    """
    edges, inclusive = defaultdict(list), defaultdict(list)
    for i, r in enumerate(rows):
        a, b = max(start, r["start_ns"]), min(end, r["end_ns"])
        if b <= a:
            continue
        edges[a].append((True, i, r))
        edges[b].append((False, i, r))
        inclusive[r["category"]].append((a, b))
    active, totals, previous = {}, Counter(), start
    for t in sorted(set(edges) | {start, end}):
        if t > previous:
            if not active:
                label = "unaccounted"
            else:
                chosen = min(
                    active.values(), key=lambda r: (r["end_ns"] - r["start_ns"], r["category"])
                )
                crossing = any(
                    not (r["start_ns"] <= chosen["start_ns"] and chosen["end_ns"] <= r["end_ns"])
                    for r in active.values()
                )
                label = "ambiguous_crossing_spans" if crossing else chosen["category"]
            totals[label] += t - previous
        for add, i, r in edges[t]:
            if add:
                active[i] = r
            else:
                active.pop(i, None)
        previous = t
    return dict(
        exclusive_ms={k: v / 1e6 for k, v in totals.items()},
        inclusive_union_ms={k: duration(v) for k, v in inclusive.items()},
        closure_residual_ns=end - start - sum(totals.values()),
    )


def traces(host):
    return host.get("causal_timeline", {}).get("rows", [])


def one(rows, category, start, end, operation=None):
    found = [
        r
        for r in rows
        if r["category"] == category
        and start <= r["start_ns"] <= end
        and (operation is None or r.get("operation") == operation)
    ]
    return found[0] if len(found) == 1 else {}


def refs(row):
    return {(r.get("request_id"), r.get("round_id")) for r in row.get("requests", [])}


def summarize_ledgers(rows):
    complete = [r for r in rows if r["ledger"]["status"] == "COMPLETE"]
    values = [r["ledger"]["total_ns"] / 1e6 for r in complete]
    names = set(k for r in complete for k in r["ledger"]["segments_ns"])
    ordered = sorted(complete, key=lambda r: (r["ledger"]["total_ns"], str(r["key"])))
    representatives = {}
    for name, percentile in (("median", 50), ("p90", 90)):
        if ordered:
            # Nearest-rank observed sample; all complete samples determine rank.
            representatives[name] = ordered[max(0, (len(ordered) * percentile + 99) // 100 - 1)]
    return dict(
        complete=len(complete),
        incomplete=len(rows) - len(complete),
        total_ms=stats(values),
        segments_ms={
            k: stats(
                [
                    r["ledger"]["segments_ns"][k] / 1e6
                    for r in complete
                    if k in r["ledger"]["segments_ns"]
                ]
            )
            for k in sorted(names)
        },
        max_absolute_closure_residual_ns=max(
            (abs(r["ledger"]["closure_residual_ns"]) for r in complete), default=None
        ),
        representative_rule="nearest rank median/P90 of all complete samples; ties by key",
        representatives=representatives,
    )


def cycle_report(runtime, backend):
    errors, groups = [], []
    groups = target_groups(runtime, errors)
    start, end = runtime["measurement_start_ns"], runtime["measurement_end_ns"]
    target = next(
        (d for d in runtime["target_devices"] if d["device"]["identity"]["global_rank"] == 0), {}
    )
    host = runtime.get("host", {})
    tt, dt = traces(target.get("host", {})), traces(backend.get("fixed_host", {}))
    events = backend.get("prepost", {}).get("pingpong", {}).get("events", [])
    ready, owner = defaultdict(list), defaultdict(list)
    for e in events:
        if e["event"] == "ready":
            ready[e["request_id"], e["prefix_version"]].append(e)
        elif e["event"] == "feedback":
            for r in e["rows"]:
                owner[r["request_id"], r["round_id"]].append(e["owner_feedback_ns"])
    all_rows, steps, queue = {}, [], []
    native_intervals = backend.get("prepost_physical", {}).get("forwards", [])
    for g in groups:
        s, natives = g["step"], g["native"]
        if len(natives) != 2:
            continue
        a, b = s["start_ns"], s["end_ns"]
        rpc = one(tt, "transport_exchange", a, b, "pp_feedback")
        sampled = one(tt, "target_sampled_results_received", a, b)
        packed = one(tt, "target_feedback_payload_ready", a, b)
        deq = one(dt, "owner_dequeue", a, b, "pp_feedback")
        expected = {(c["request_id"], c["prefix_version"]) for c in g["claims"].values()}
        if refs(rpc) != expected or refs(deq) != expected:
            errors.append(
                f"step {g['index']}: feedback/dequeue request-version association missing"
            )
            rpc, deq = {}, {}
        gpu_a = min(n["start_lower_ns"] for n in natives)
        gpu_b = max(n["end_upper_ns"] for n in natives)
        claim = min(c["claimed_ns"] for c in g["claims"].values())
        lm = dict(
            claim=claim,
            complete_step_start=a,
            scheduler_start=s.get("schedule_start_ns"),
            scheduler_end=s.get("schedule_end_ns"),
            GPU_start_lower=gpu_a,
            GPU_end_upper=gpu_b,
            sampled=sampled.get("start_ns"),
            feedback_generated=packed.get("start_ns"),
            feedback_RPC=rpc.get("start_ns"),
            complete_step_end=b,
        )
        row = dict(
            key=g["index"],
            landmarks_ns=lm,
            ledger=ledger(lm),
            home_request_counts=dict(Counter(c["home_cohort"] for c in g["claims"].values())),
            physical_forward_ids={
                str(rank): n.get("physical_forward_id")
                for rank, n in zip(g["native_ranks"], natives)
            },
            GPU_clock_bounds=[
                {
                    k: n[k]
                    for k in ("start_lower_ns", "start_upper_ns", "end_lower_ns", "end_upper_ns")
                }
                for n in natives
            ],
        )
        if s.get("window"):
            # Different lanes share the same claim->GPU boundary, but are never
            # added together. Control publication before complete_step is included.
            row["claim_to_gpu_threads"] = {}
            for label, h in (("coordinator", host), ("target-rank0", target.get("host", {}))):
                lane_rows = [*h.get("intervals", []), *traces(h)]
                lanes_here = defaultdict(list)
                for span in lane_rows:
                    if (
                        type(span.get("thread_id")) is int
                        and span["end_ns"] > claim
                        and span["start_ns"] < gpu_a
                    ):
                        lanes_here[span.get("pid"), span["thread_id"]].append(span)
                row["claim_to_gpu_threads"][label] = {
                    str(k): partition(v, claim, gpu_a) for k, v in lanes_here.items()
                }
            steps.append(row)
            recv, dequeue = rpc.get("service_receive_ns"), deq.get("start_ns")
            if type(recv) is int and type(dequeue) is int and dequeue >= recv:
                physical = duration(
                    intersect(
                        [(recv, dequeue)], [(p["start_ns"], p["end_ns"]) for p in native_intervals]
                    )
                )
                queue.append(
                    dict(
                        step=g["index"],
                        service_receive_ns=recv,
                        owner_dequeue_ns=dequeue,
                        wait_ms=(dequeue - recv) / 1e6,
                        physical_host_call_union_ms=physical,
                        other_or_unaccounted_ms=(dequeue - recv) / 1e6 - physical,
                        physical_ids=[
                            p.get("physical_forward_id")
                            for p in native_intervals
                            if p["start_ns"] < dequeue and p["end_ns"] > recv
                        ],
                    )
                )
        for c in g["claims"].values():
            key = c["request_id"], c["prefix_version"]
            if key in all_rows:
                errors.append("duplicate request/version claim: " + str(key))
                all_rows[key] = None
                continue
            rs, os = ready[key], owner[key]
            r = rs[0] if len(rs) == 1 and rs[0]["proposal_id"] == c["proposal_id"] else {}
            all_rows[key] = dict(
                group=g,
                claim=c,
                ready=r,
                row=row,
                rpc=rpc,
                dequeue=deq,
                owner_ns=os[0] if len(os) == 1 else None,
            )
    cycles, excluded = [], Counter()
    for key, cur in all_rows.items():
        if cur is None:
            excluded["ambiguous_claim"] += 1
            continue
        previous = all_rows.get((key[0], key[1] - 1))
        if previous is None:
            excluded["initial_refill_or_missing_predecessor"] += 1
            continue
        first = previous["row"]["landmarks_ns"]["feedback_generated"]
        last = cur["row"]["landmarks_ns"]["feedback_generated"]
        if type(first) is int and type(last) is int and not start <= first <= last <= end:
            excluded["window_boundary"] += 1
            continue
        if previous["claim"]["home_cohort"] != cur["claim"]["home_cohort"]:
            errors.append("request home changed within cycle: " + str(key))
            continue
        landmark_values = cur["row"]["landmarks_ns"]
        r, c = cur["ready"], cur["claim"]
        landmarks = dict(
            previous_feedback_generated=first,
            previous_feedback_RPC=previous["rpc"].get("start_ns"),
            service_receive=previous["rpc"].get("service_receive_ns"),
            owner_dequeue=previous["dequeue"].get("start_ns"),
            owner_processed=previous["owner_ns"],
            physical_READY=r.get("ready_ns"),
            READY_published=r.get("timestamp_ns"),
            claim=c["claimed_ns"],
            GPU_start_lower=landmark_values["GPU_start_lower"],
            GPU_end_upper=landmark_values["GPU_end_upper"],
            next_feedback_generated=last,
        )
        # Readiness is valid for this proposal/version only; the atomic selection
        # additionally enforces home/capacity/gate. This is a necessary-condition
        # bound, not a fabricated earliest eligibility timestamp.
        available = cur["group"]["step"]["ping_admission"].get("target_available_observed_ns")
        lower = max(r["timestamp_ns"], available) if r and type(available) is int else None
        upper = c.get("eligibility_observed_ns")
        cycles.append(
            dict(
                key=list(key),
                home=c["home_cohort"],
                proposal_id=c["proposal_id"],
                claim_id=c["claim_id"],
                previous_step=previous["group"]["index"],
                next_step=cur["group"]["index"],
                landmarks_ns=landmarks,
                ledger=ledger(landmarks),
                eligibility=dict(
                    ready_version=key[1],
                    valid_until_claim_ns=c["claimed_ns"],
                    jointly_observed_necessary_conditions_ns=lower,
                    atomic_selection_observed_ns=upper,
                    earliest_legal_time="NOT_OBSERVED; home/capacity/Serial gate "
                    "only checked at selection; READY alone is insufficient",
                ),
            )
        )
    # Cadences are start-to-start, not denominators synthesized from opportunity counts.
    cadences = defaultdict(list)
    previous_global, previous_home = None, {}
    for row in steps:
        t = row["landmarks_ns"]["GPU_start_lower"]
        if previous_global is not None:
            cadences["global"].append((t - previous_global) / 1e6)
        previous_global = t
        for home in row["home_request_counts"]:
            if home in previous_home:
                cadences[home].append((t - previous_home[home]) / 1e6)
            previous_home[home] = t
    lanes = {}
    producers = {"coordinator": host, "draft": backend.get("fixed_host", {})}
    producers.update(
        {
            "target-" + str(d["device"]["identity"]["global_rank"]): d.get("host", {})
            for d in runtime["target_devices"]
        }
    )
    for producer, h in producers.items():
        by_lane = defaultdict(list)
        for r in [*h.get("intervals", []), *traces(h)]:
            if type(r.get("pid")) is int and type(r.get("thread_id")) is int:
                by_lane[r["pid"], r["thread_id"]].append(r)
        lanes[producer] = {str(k): partition(v, start, end) for k, v in by_lane.items()}
    return dict(
        schema_version=SCHEMA,
        association_errors=errors,
        measurement_boundary_ns=[start, end],
        window_ms=(end - start) / 1e6,
        steps=steps,
        matched_step_summary=summarize_ledgers(steps),
        request_cycles=cycles,
        request_cycle_summary=summarize_ledgers(cycles),
        excluded=dict(excluded),
        cadence_start_lower_ms={k: stats(v) for k, v in cadences.items()},
        feedback_owner_queue=queue,
        thread_partitions=lanes,
        scope="complete feedback-generated to next feedback-generated request/version cycles; "
        "no interpolation; home cadence may include mixed-home steps; no alternation assumption",
        limitations="GPU lower/upper envelope closes algebraically, not exact device wall time; "
        "host clocks are same-machine monotonic. CPU lane totals are never added across threads. "
        "Unobserved IPC/input subwork and legal eligibility remain unaccounted; terminal requests "
        "without a next claim have no next cycle. Native overlap is reported separately.",
    )
