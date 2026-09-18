"""Ordinary CPU comparison only; offline accounting with unchanged window boundaries."""

from collections import defaultdict

from specrhythm.serving.fixed_results import stats


def compare_smoke(rows, *, cpu_comparison=False):
    by = {r.get("case", r["target_dispatch"]): r for r in rows}
    pairs = (("P0", "P1"), ("S0", "S1")) if cpu_comparison else (("reference", "dual-batch"),)
    differences = []
    for a, b in pairs:
        before, after = by[a]["outputs"], by[b]["outputs"]
        for rid in sorted(set(before) | set(after)):
            if before.get(rid) != after.get(rid):
                left, right = before.get(rid, {}), after.get(rid, {})
                lt, rt = left.get("tokens", []), right.get("tokens", [])
                first = next(
                    (i for i, (x, y) in enumerate(zip(lt, rt)) if x != y), min(len(lt), len(rt))
                )
                differences.append(
                    dict(
                        reference=a,
                        actual=b,
                        request_id=rid,
                        first_difference_index=first,
                        reference_output=left,
                        actual_output=right,
                        sources=[by[a]["run_directory"], by[b]["run_directory"]],
                    )
                )
    return dict(
        status="FAILED" if differences else "PASS",
        runs=rows,
        differences=differences,
        comparison_stage="smoke_comparison",
        pairs=pairs,
        scope="paired same-mode bounded greedy paths; not Target-only or cross-geometry",
    )


def window_metrics(runtime):
    """Same full-request TPOT as fixed_results, censored requests never completed.

    S2 includes the resident bootstrap token in generated_token_ids; preserve the
    fixed_results exclusion of bootstrap + first timed commit group. Full request
    TPOT is a completion-selected statistic, NOT the reciprocal of throughput.
    """
    a, b = runtime["measurement_start_ns"], runtime["measurement_end_ns"]
    completed, tpots, unsupported = [], [], 0
    for r in runtime["requests"]:
        if r["state"] != "FINISHED" or not a <= r["completion_ns"] <= b:
            continue
        completed.append(r["request_id"])
        commits = r["commits"]
        denominator = (
            (len(r["generated_token_ids"]) - 1 - len(commits[0]["token_ids"])) if commits else 0
        )
        if len(commits) >= 2 and denominator > 0:
            tpots.append((r["completion_ns"] - commits[0]["timestamp_ns"]) / 1e6 / denominator)
        else:
            unsupported += 1
    return dict(
        natural_completions_in_window=len(completed),
        completed_request_ids=completed,
        completed_requests_per_second=len(completed) / ((b - a) / 1e9),
        full_request_tpot_ms=stats(tpots, "insufficient naturally completed requests"),
        tpot_unavailable_completed_requests=unsupported,
        scope="completion_ns inside window; full-request TPOT uses fixed_results formula; "
        "includes pre-window history if request started earlier; no SLO claim",
    )


def aggregate(windows):
    groups = defaultdict(list)
    for w in windows:
        groups[w.get("case", w["target_dispatch"])].append(w)
    result = {}
    for case, rows in groups.items():
        points = [r["points"][0] for r in rows]
        tokens = sum(p["committed_tokens"] for p in points)
        ms = sum(p["window_ms"] for p in points)
        result[case] = dict(
            windows=len(rows),
            committed_tokens=tokens,
            window_ms=ms,
            pooled_throughput_tok_s=tokens / (ms / 1000),
            throughput_range=[
                min(p["throughput_tok_s"] for p in points),
                max(p["throughput_tok_s"] for p in points),
            ],
            complete_requests=sum(
                r["request_measurement"]["natural_completions_in_window"] for r in rows
            ),
            geometry_note="P0/P1 A64/B64 Target64; S1 A128 Target128; same active128/resources; "
            "system comparison includes batch efficiency, not pure overlap",
        )
    return result


def recovery_during_prepare(runtime, backend):
    """A conservative, observable class, not a complete eligibility attribution.

    Count unique physical recovery calls that finish wholly after another home's
    claim and before its earliest possible GPU start. No rank/row duplication.
    Remaining calls are unclassified, not automatically late/blocked/too long.
    """
    from specrhythm.serving.k3_dispatch_evidence import target_groups

    errors = []
    groups = target_groups(runtime, errors)
    if errors:
        return dict(status="INCOMPLETE", errors=errors)
    homes = backend["prepost"]["pingpong"]["home_cohorts"]
    native = backend["fixed_device"]["forwards"]
    start, end = runtime["measurement_start_ns"], runtime["measurement_end_ns"]
    total, classified, missing = 0, [], 0
    for f in backend["prepost_physical"]["forwards"]:
        if not start <= f["start_ns"] <= end:
            continue
        recoveries = [
            r for r in f["bindings"] if r.get("source") in ("rejection", "eager_rejection")
        ]
        if not recoveries:
            continue
        total += 1
        matches = [
            g
            for g in native
            if g["purpose"] == f["purpose"]
            and g["B"] == f["B"]
            and f["start_ns"] <= g["host_start_ns"] <= f["end_ns"]
        ]
        if len(matches) != 1:
            missing += 1
            continue
        g = matches[0]
        for batch in groups:
            if not batch["step"].get("window"):
                continue
            other = {c["home_cohort"] for c in batch["claims"].values()}
            claim = min(c["claimed_ns"] for c in batch["claims"].values())
            gpu = min(t["start_lower_ns"] for t in batch["native"])
            if (
                all(homes[r["request_id"]] not in other for r in recoveries)
                and claim <= g["start_lower_ns"]
                and g["end_upper_ns"] <= gpu
            ):
                classified.append(
                    dict(
                        physical_forward_id=f["physical_forward_id"],
                        B=f["B"],
                        target_step=batch["index"],
                        claim_ns=claim,
                        draft_start_lower_ns=g["start_lower_ns"],
                        draft_end_upper_ns=g["end_upper_ns"],
                        target_start_lower_ns=gpu,
                        draft_finish_before_target_ms=(gpu - g["end_upper_ns"]) / 1e6,
                        example_binding=recoveries[0],
                    )
                )
                break
    ordered = sorted(classified, key=lambda r: r["draft_finish_before_target_ms"])
    examples = [ordered[int((len(ordered) - 1) * q)] for q in (0.5, 0.9)] if ordered else []
    events = backend["prepost"]["pingpong"].get("events", [])
    for row in examples:
        binding = row["example_binding"]
        rid = binding["request_id"]
        version = binding.get("round_id")
        parent_version = (
            version - int(binding.get("work_kind") == "normal_extension")
            if type(version) is int
            else None
        )
        parent = next(
            (
                g
                for g in groups
                if rid in g["claims"] and g["claims"][rid].get("prefix_version") == parent_version
            ),
            None,
        )
        settled = next(
            (
                e
                for e in events
                if e["event"] == "settlement"
                and e["request_id"] == rid
                and e["prefix_version"] == parent_version
            ),
            None,
        )
        ready = next(
            (
                e
                for e in events
                if e["event"] == "ready"
                and e["request_id"] == rid
                and type(parent_version) is int
                and e["prefix_version"] == parent_version + 1
            ),
            None,
        )
        row["parent_and_READY"] = dict(
            status="COMPLETE" if parent and settled and ready else "MISSING_LANDMARKS",
            parent_prefix_version=parent_version,
            parent_claim={
                k: parent["claims"][rid][k]
                for k in (
                    "request_id",
                    "prefix_version",
                    "proposal_id",
                    "claim_id",
                    "home_cohort",
                    "claimed_ns",
                )
            }
            if parent
            else None,
            parent_GPU_end_upper_ns=max(g["end_upper_ns"] for g in parent["native"])
            if parent
            else None,
            owner_feedback_ns=settled.get("feedback_ns") if settled else None,
            next_READY=ready,
        )
    return dict(
        cross_home_applicability="OTHER_HOME_PRESENT"
        if len(set(homes.values())) > 1
        else "NO_OTHER_HOME",
        status="INCOMPLETE" if missing else "COMPLETE",
        physical_recovery_calls=total,
        finished_wholly_in_other_home_claim_to_GPU=len(classified),
        unclassified_calls=total - len(classified),
        missing_native=missing,
        examples=examples,
        example_rule="median/P90 finish-to-Target gap",
        limit="classification of individual physical calls, not whole K3 jobs; "
        "no earliest eligibility, no claim that all remainder is avoidable; "
        "host monotonic / calibrated native uncertainty bounds",
    )
