"""Indexed CPU timing attribution. Observations never imply removable critical-path time."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections import Counter, defaultdict

from specrhythm.serving.fixed_results import clipped, duration, intersections, stats

CATEGORIES = (
    "checkpoint_log_write",
    "log_fsync",
    "json_serialization",
    "live_uuid_validation",
    "nvidia_smi_subprocess",
    "ipc",
    "required_tp_barrier",
    "resident_block_audit",
    "scheduler",
)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def span(row):
    a, b = row["start_ns"], row["end_ns"]
    check(type(a) is int and type(b) is int and a <= b, "invalid host interval")
    return a, b


def host_costs(sources, start, end):
    """One scan then sorted unions per process/category; no cross-event Cartesian join."""
    by_category, by_source = defaultdict(list), {}
    inclusive, counts = defaultdict(float), defaultdict(int)
    all_intervals = []
    for label, source in sources.items():
        rows = source.get("intervals", [])
        categories = defaultdict(list)
        for row in rows:
            a, b = span(row)
            if b < start or a > end:
                continue
            interval = max(a, start), min(b, end)
            category = row["category"]
            counts[category] += 1
            inclusive[category] += (interval[1] - interval[0]) / 1e6
            categories[category].append(interval)
            by_category[category].append(interval)
            all_intervals.append(interval)
        by_source[label] = {
            "observed_union_ms": duration([v for values in categories.values() for v in values]),
            "categories": {
                k: {"count": len(v), "union_ms": duration(v)}
                for k, v in sorted(categories.items())
            },
            "exclusive_self_ms": None,
            "exclusive_self_status": "UNKNOWN: legacy timers have no thread/parent-call IDs",
        }
    observed = duration(all_intervals)
    return {
        "categories": {
            k: {
                "count": counts[k],
                "inclusive_call_ms": inclusive[k],
                "cross_process_union_ms": duration(by_category[k]),
                "exclusive_self_ms": None,
            }
            for k in sorted(set(by_category) | set(CATEGORIES))
        },
        "sources": by_source,
        "all_categories_union_ms": observed,
        "outside_host_observation_ms": (end - start) / 1e6 - observed,
        "semantics": "inclusive sums retain nesting; unions deduplicate nesting/concurrent lanes; "
        "outside observation is not CPU time; no critical-path saving inferred",
    }


def gpu_bounds(device, start, end):
    check(
        device.get("clock")
        == "CUDA elapsed events projected onto bracketed host monotonic anchor",
        "missing/unknown CUDA-to-host clock contract",
    )
    uncertainty = device.get("anchor_uncertainty_ns")
    check(type(uncertainty) is int and uncertainty >= 0, "missing CUDA anchor uncertainty")
    integer_anchor = device.get("projection_version") == "integer-anchor-v2"
    if integer_anchor:
        check(
            device["anchor_after_ns"] - device["anchor_before_ns"] == uncertainty,
            "integer CUDA anchor bracket disagrees with uncertainty",
        )
    lower, upper = [], []
    for row in device.get("forwards", []):
        a, b, c, d = (
            row[k] for k in ("start_lower_ns", "start_upper_ns", "end_lower_ns", "end_upper_ns")
        )
        elapsed = row["gpu_event_ms"]
        check(
            all(type(v) is int for v in (a, b, c, d)) and a <= b and c <= d and a < c,
            "malformed projected CUDA bounds",
        )
        check(
            abs(b - a - uncertainty) <= (0 if integer_anchor else 2)
            and abs(d - c - uncertainty) <= (0 if integer_anchor else 2),
            "CUDA bounds disagree with anchor uncertainty",
        )
        check(
            type(elapsed) in (float, int)
            and math.isfinite(elapsed)
            and elapsed > 0
            and abs((c - a) / 1e6 - elapsed) <= 0.000003,
            "CUDA elapsed duration disagrees with projected bounds",
        )
        if c > b:
            lower.append((b, c))
        upper.append((a, d))
    return clipped(lower, start, end), clipped(upper, start, end)


def overlap(targets, draft, start, end, coverage_errors=()):
    errors = list(coverage_errors)
    if not targets or not draft or not draft.get("forwards"):
        errors.append("missing Target/Draft CUDA forward coverage")
    lows, highs = [], []
    try:
        ranks = [t["identity"]["global_rank"] for t in targets]
        check(sorted(ranks) == [0, 1], "missing/duplicated Target TP rank evidence")
        identities = [d["identity"]["gpu_uuid"] for d in targets + [draft]]
        check(
            len(set(identities)) == 3 and all(identities), "GPU identity evidence is not distinct"
        )
        for t in targets:
            check(bool(t.get("forwards")), "empty Target rank CUDA forward list")
            lo, hi = gpu_bounds(t, start, end)
            lows.extend(lo)
            highs.extend(hi)
        dl, dh = gpu_bounds(draft, start, end)
        low = duration(intersections(lows, dl))
        high = duration(intersections(highs, dh))
    except (KeyError, TypeError, ValueError) as error:
        errors.append(str(error))
        low = high = None
    return {
        "status": "UNKNOWN"
        if errors
        else ("POSITIVE" if low > 0 else "ZERO" if high == 0 else "UNCERTAIN"),
        "event_overlap_lower_ms": low if not errors else None,
        "event_overlap_upper_ms": high if not errors else None,
        "coverage_errors": errors,
        "exact_kernel_overlap": False,
        "clock_scope": "one retained attempt on one host; bracketed CUDA anchor projections",
        "critical_path_time_saved_ms": None,
    }


def target_index(runtime):
    steps = sorted(runtime["target_steps"], key=lambda r: r["start_ns"])
    starts = [s["start_ns"] for s in steps]
    check(
        all(a["end_ns"] <= b["start_ns"] for a, b in zip(steps, steps[1:])),
        "overlapping coordinator Target steps",
    )
    batches = defaultdict(dict)
    for evidence in runtime["target_devices"]:
        rank = evidence["device"]["identity"]["global_rank"]
        for f in evidence["device"]["forwards"]:
            i = bisect.bisect_right(starts, f["host_start_ns"]) - 1
            if i < 0 or f["host_start_ns"] > steps[i]["end_ns"]:
                continue
            check(rank not in batches[i], "multiple Target forwards per step/rank")
            check(
                f["B"] == steps[i]["B"] == len(f["internal_request_ids"])
                and set(f["internal_request_ids"])
                == {r["internal_request_id"] for r in steps[i]["rows"]},
                "Target physical batch/request identity mismatch",
            )
            batches[i][rank] = f
    return steps, batches


def proposal_index(backend):
    index, summaries, errors = {}, [], []
    forwards = sorted(backend["fixed_device"]["forwards"], key=lambda r: r["host_start_ns"])
    starts = [f["host_start_ns"] for f in forwards]
    actual_counts = Counter(f.get("purpose") for f in forwards)
    declared = backend.get("draft_model_forward_count_by_purpose")
    if declared is None or any(
        actual_counts[p] != declared.get(p, 0) for p in ("proposal", "commit")
    ):
        errors.append(
            "Draft proposal/commit CUDA coverage differs from independent backend counters"
        )
    previous_end = None
    covered = set()
    for p in sorted(backend["fixed_proposals"], key=lambda p: p["start_ns"]):
        # Proposals execute on one owner. Nonoverlap bounds total slice work to O(F+P).
        check(previous_end is None or previous_end <= p["start_ns"], "overlapping Draft proposals")
        previous_end = p["end_ns"]
        check(
            p["B"]
            == len(set(p["request_ids"]))
            == len(p["round_ids"])
            == len(p["context_lengths"])
            == len(p["candidate_lengths"]),
            "Draft proposal row metadata lengths disagree",
        )
        a = bisect.bisect_left(starts, p["start_ns"])
        b = bisect.bisect_left(starts, p["end_ns"])
        selected = [f for f in forwards[a:b] if f["purpose"] == "proposal"]
        expected = max(p["candidate_lengths"], default=0) - 1
        if not p.get("first_token_from_cached_logits") or len(selected) != max(0, expected):
            errors.append("Draft cached-first-token proposal forward coverage is incomplete")
        ids = {"sr-draft:" + rid for rid in p["request_ids"]}
        for offset, f in enumerate(selected):
            expected_ids = {
                "sr-draft:" + rid
                for rid, count in zip(p["request_ids"], p["candidate_lengths"])
                if count > offset + 1
            }
            if set(f["internal_request_ids"]) != expected_ids or f["B"] != len(expected_ids):
                errors.append("Draft proposal actual internal request/batch coverage mismatch")
            covered.add(id(f))
        check(bool(ids), "empty Draft proposal request set")
        summary = {
            **p,
            "proposal_forward_count": len(selected),
            "proposal_forward_gpu_ms": sum(f["gpu_event_ms"] for f in selected),
            "other_forward_gpu_ms": sum(
                f["gpu_event_ms"] for f in forwards[a:b] if f["purpose"] != "proposal"
            ),
        }
        summaries.append(summary)
        for rid, round_id, prefix in zip(p["request_ids"], p["round_ids"], p["context_lengths"]):
            key = rid, round_id, prefix
            check(key not in index, "duplicate Draft request/round/prefix proposal")
            index[key] = summary
    if any(f["purpose"] == "proposal" and id(f) not in covered for f in forwards):
        errors.append("unassociated Draft proposal forwards")
    return index, summaries, sorted(set(errors))


def joined_steps(runtime, backend):
    steps, forwards = target_index(runtime)
    proposals, draft_samples, coverage_errors = proposal_index(backend)
    measured_steps = [s for s in steps if s.get("window") and s["B"]]
    if not measured_steps or len(measured_steps) != runtime.get("sample_count"):
        coverage_errors.append("measured Target steps differ from runtime sample_count")
    if any(
        s["start_ns"] < runtime["measurement_start_ns"]
        or s["end_ns"] > runtime["measurement_end_ns"]
        for s in measured_steps
    ):
        coverage_errors.append("measured Target steps lie outside the retained window")
    rounds, commits, requests = {}, {}, {r["request_id"]: r for r in runtime["requests"]}
    for evidence in runtime["target_devices"]:
        for r in evidence["rounds"]:
            key = r["request_id"], r.get("parent_prefix_len", r.get("prefix_token_count"))
            check(key not in rounds or rounds[key] == r, "conflicting committed round identity")
            rounds[key] = r
    for rid, r in requests.items():
        prefix = runtime["prompt_lengths"][rid] + 1  # Real decode-ready bootstrap is one token.
        for c in sorted(r["commits"], key=lambda c: c["timestamp_ns"]):
            commits[rid, prefix] = c
            prefix += len(c["token_ids"])
    samples, errors = [], []
    for i, step in enumerate(steps):
        if not step.get("window") or not step["B"]:
            continue
        try:
            check(set(forwards[i]) == {0, 1}, "Target step lacks both physical TP forwards")
            check(
                len(set(step["request_ids"])) == step["B"] == len(step["rows"]),
                "duplicate/missing scheduled request IDs",
            )
            check(
                set(step["request_ids"]) == {r["request_id"] for r in step["rows"]},
                "scheduled request IDs differ from rows",
            )
            identities, proposal_spans, output_times, next_starts = [], set(), [], []
            for row in step["rows"]:
                rid, prefix = row["request_id"], row["context_length"]
                check(
                    requests[rid]["cohort"] == step["cohort"], "request/cohort assignment mismatch"
                )
                r, c = rounds[rid, prefix], commits[rid, prefix]
                check(
                    c["token_ids"] == r["committed_token_ids"],
                    "actual output/round commit mismatch",
                )
                check(
                    step["start_ns"] <= c["timestamp_ns"] <= step["end_ns"],
                    "actual output commit lies outside its Target step",
                )
                p = proposals[rid, r["round_id"], prefix]
                identities.append(
                    {
                        "request_id": rid,
                        "cohort": step["cohort"],
                        "round_id": r["round_id"],
                        "prefix_version": r.get("prefix_version"),
                        "prefix_length": prefix,
                        "committed_count": len(c["token_ids"]),
                        "worker_commit_start_ns": r.get("commit_start_ns"),
                        "worker_commit_end_ns": r.get("commit_end_ns"),
                        "coordinator_output_commit_ns": c["timestamp_ns"],
                        "serial_timeline": r.get("timeline"),
                    }
                )
                proposal_spans.add((p["start_ns"], p["end_ns"]))
                output_times.append(c["timestamp_ns"])
                following = proposals.get((rid, r["round_id"] + 1, prefix + len(c["token_ids"])))
                if following:
                    next_starts.append(following["start_ns"])
            first = min(f["host_start_ns"] for f in forwards[i].values())
            last_launch = max(f["host_launch_end_ns"] for f in forwards[i].values())
            check(
                step["start_ns"] <= first <= last_launch <= step["end_ns"],
                "Target host launch bounds lie outside the step",
            )
            samples.append(
                {
                    "start_ns": step["start_ns"],
                    "end_ns": step["end_ns"],
                    "B": step["B"],
                    "cohort": step["cohort"],
                    "identities": identities,
                    "scheduler_start_ns": step.get("schedule_start_ns"),
                    "scheduler_end_ns": step.get("schedule_end_ns"),
                    "target_model_host_start_ns": first,
                    "target_model_host_launch_end_ns": last_launch,
                    "target_gpu_event_ms_by_rank": {
                        str(k): f["gpu_event_ms"] for k, f in forwards[i].items()
                    },
                    "target_gpu_event_ms": max(f["gpu_event_ms"] for f in forwards[i].values()),
                    "before_model_host_ms": (first - step["start_ns"]) / 1e6,
                    "after_model_launch_host_ms": (step["end_ns"] - last_launch) / 1e6,
                    "output_commit_first_ns": min(output_times),
                    "output_commit_last_ns": max(output_times),
                    "draft_proposal_host_intervals": sorted(proposal_spans),
                    "next_actual_proposal_start_ns_range": [min(next_starts), max(next_starts)]
                    if next_starts
                    else None,
                    "next_proposal_request_count": len(next_starts),
                    "enqueue_ns": None,
                    "owner_receive_ns": None,
                    "proposal_publication_ns": None,
                    "unavailable": "legacy RPC has no request correlation/receive event; "
                    "model launch end is not GPU completion; "
                    "proposal end is not ready publication",
                }
            )
        except (KeyError, TypeError, ValueError) as error:
            errors.append({"step_start_ns": step.get("start_ns"), "error": str(error)})
    return samples, draft_samples, coverage_errors, errors


def rotations(samples, start):
    """Join equal real round IDs with complete disjoint A32/B32 cohorts, not list offsets."""
    pending, complete, incomplete = defaultdict(dict), [], []
    for sample in samples:
        ids = sample["identities"]
        rounds = {r["round_id"] for r in ids}
        if len(rounds) != 1 or len({r["request_id"] for r in ids}) != sample["B"]:
            incomplete.append(
                {"start_ns": sample["start_ns"], "reason": "nonuniform round/full batch"}
            )
            continue
        round_id = next(iter(rounds))
        if sample["B"] == 64 and sample["cohort"] is None:
            complete.append((round_id, [sample]))
        elif sample["B"] == 32 and sample["cohort"] in ("A", "B"):
            cohort = sample["cohort"]
            check(cohort not in pending[round_id], "duplicate cohort verification for one round")
            pending[round_id][cohort] = sample
        else:
            incomplete.append({"start_ns": sample["start_ns"], "reason": "not full64 or cohort32"})
    for round_id, cohorts in sorted(pending.items()):
        if set(cohorts) != {"A", "B"}:
            incomplete.append({"round_id": round_id, "reason": "missing matching A32/B32 cohort"})
            continue
        pair = list(cohorts.values())
        ids = [r["request_id"] for s in pair for r in s["identities"]]
        check(len(set(ids)) == 64, "cohort pair shares request IDs")
        complete.append((round_id, pair))
    result, previous, previous_round = [], {}, None
    all_starts = sorted(s["start_ns"] for s in samples)
    for round_id, group in sorted(complete, key=lambda p: max(s["end_ns"] for s in p[1])):
        identities = sorted(
            [r for s in group for r in s["identities"]], key=lambda r: r["request_id"]
        )
        for row in identities:
            old = previous.get(row["request_id"])
            if old is not None and row["round_id"] == old["round_id"] + 1:
                check(
                    row["prefix_length"] == old["prefix_length"] + old["committed_count"],
                    "successive actual committed prefix lengths are inconsistent",
                )
                if row["prefix_version"] is not None and old["prefix_version"] is not None:
                    check(
                        row["prefix_version"] == old["prefix_version"] + 1,
                        "successive prefix versions are inconsistent",
                    )
            previous[row["request_id"]] = row
        end = max(s["end_ns"] for s in group)
        valid_boundary = (
            (previous_round is None or previous_round == round_id - 1)
            and min(s["start_ns"] for s in group) >= start
            and bisect.bisect_left(all_starts, end) - bisect.bisect_left(all_starts, start)
            == len(group)
        )
        result.append(
            {
                "round_id": round_id,
                "identity_sha256": hashlib.sha256(
                    json.dumps(identities, sort_keys=True).encode()
                ).hexdigest(),
                "request_count": len(identities),
                "start_boundary_ns": start if valid_boundary else None,
                "end_ns": end,
                "rotation_ms": (end - start) / 1e6 if valid_boundary else None,
                "boundary_status": "OBSERVED"
                if valid_boundary
                else "INSUFFICIENT: missing round or interleaved cohort rotation boundary",
                "target_step_union_ms": duration([(s["start_ns"], s["end_ns"]) for s in group]),
                "committed_tokens": sum(r["committed_count"] for r in identities),
            }
        )
        start, previous_round = end, round_id
    return {
        "complete_count": len(result),
        "incomplete_count": len(incomplete),
        "rotation_ms": stats([r["rotation_ms"] for r in result if r["rotation_ms"] is not None]),
        "rotations": result[:128],
        "incomplete": incomplete[:128],
        "details_truncated": len(result) > 128 or len(incomplete) > 128,
    }


def analyze_raw(runtime, backend):
    start, end = runtime["measurement_start_ns"], runtime["measurement_end_ns"]
    check(
        type(start) is int and type(end) is int and end > start,
        "missing positive measurement boundary",
    )
    samples, proposals, coverage, join_errors = joined_steps(runtime, backend)
    sources = {"coordinator": runtime["host"], "draft-owner-and-server": backend["fixed_host"]}
    for r in runtime["target_devices"]:
        sources["target-rank-" + str(r["device"]["identity"]["global_rank"])] = r["host"]
    costs = host_costs(sources, start, end)
    gpu = overlap(
        [r["device"] for r in runtime["target_devices"]],
        backend["fixed_device"],
        start,
        end,
        coverage + (["request/round/commit joins incomplete"] if join_errors else []),
    )
    by_purpose = defaultdict(list)
    for f in backend["fixed_device"]["forwards"]:
        if start <= f["host_start_ns"] <= end:
            by_purpose[f["purpose"]].append(f["gpu_event_ms"])
    # Cross-cohort Draft/model relation: sorted disjoint owner proposals, prefix interval sums.
    owner = sorted(backend.get("s2_work_records", []), key=lambda r: r["host_start_ns"])
    begins = [r["host_start_ns"] for r in owner]
    ends = [r["host_end_ns"] for r in owner]
    check(all(a <= b for a, b in zip(ends, begins[1:])), "owner work intervals overlap")
    pre_model = []
    for s in samples:
        a = bisect.bisect_right(ends, s["start_ns"])
        b = bisect.bisect_left(begins, s["target_model_host_start_ns"])
        for r in owner[a:b]:
            if r.get("logical_cohort") in ("A", "B") and r["logical_cohort"] != s["cohort"]:
                pre_model.append(
                    {
                        "target_step_start_ns": s["start_ns"],
                        "operation": r["operation"],
                        "draft_start_ns": r["host_start_ns"],
                        "draft_end_ns": r["host_end_ns"],
                        "draft_cohort": r["logical_cohort"],
                        "target_cohort": s["cohort"],
                        "owner_finished_before_model_launch": r["host_end_ns"]
                        <= s["target_model_host_start_ns"],
                    }
                )
    from specrhythm.serving.fixed_timing import recorded_costs

    rotation_report = rotations(samples, start)
    return {
        "recorded_gpu_costs": recorded_costs(
            [r["device"] for r in runtime["target_devices"]],
            backend["fixed_device"],
            [{"gpu_event_ms": r["target_gpu_event_ms"]} for r in samples],
            start,
            end,
            rotation_report["complete_count"],
        ),
        "clock_failure_details": clock_failures(runtime, backend, start, end),
        "status": "INSUFFICIENT" if join_errors or gpu["status"] == "UNKNOWN" else "OBSERVED",
        "measurement_start_ns": start,
        "measurement_end_ns": end,
        "window_ms": (end - start) / 1e6,
        "rotations": rotation_report,
        "steps": samples[:32],
        "join_errors": join_errors[:128],
        "join_error_count": len(join_errors),
        "step_count": len(samples),
        "step_details_truncated": len(samples) > 32,
        "host_costs": costs,
        "overlap": gpu,
        "draft_model_gpu_event_ms_by_purpose": {
            k: {**stats(v), "sum_ms": sum(v)} for k, v in sorted(by_purpose.items())
        },
        "proposal_only_forward_gpu_ms": stats(
            [
                p["proposal_forward_gpu_ms"]
                for p in proposals
                if start <= p["start_ns"] and p["end_ns"] <= end
            ]
        ),
        "owner_during_target_pre_model": pre_model[:128],
        "owner_during_target_pre_model_count": len(pre_model),
        "owner_relation_status": "OBSERVED" if owner else "UNKNOWN: owner work records absent",
        "critical_path": "UNKNOWN: no correlated RPC enqueue/receive IDs "
        "or thread/parent timer IDs",
        "target_pre_post_semantics": "host envelopes, including waits; "
        "not isolated CPU or GPU kernels",
        "required_next_evidence": [
            "request-correlated enqueue/owner receive and ready publication "
            "are not timestamped by the legacy producer"
        ],
    }


def clock_failures(runtime, backend, start, end, limit=16):
    """Bounded raw bad rows, including old producer values, without relaxing validation."""
    bad, count, inspected = [], 0, 0
    devices = [("runtime.json", r["device"]) for r in runtime["target_devices"]]
    devices.append(("draft-backend-report.json", backend["fixed_device"]))
    for artifact, device in devices:
        for i, row in enumerate(device.get("forwards", [])):
            inspected += 1
            try:
                gpu_bounds({**device, "forwards": [row]}, start, end)
            except (KeyError, ValueError, TypeError) as error:
                count += 1
                if len(bad) < limit:
                    u = device.get("anchor_uncertainty_ns")
                    fields = {
                        k: row.get(k)
                        for k in (
                            "host_start_ns",
                            "host_launch_end_ns",
                            "start_lower_ns",
                            "start_upper_ns",
                            "end_lower_ns",
                            "end_upper_ns",
                            "gpu_event_ms",
                            "purpose",
                            "B",
                        )
                    }
                    delta = {}
                    if type(u) is int:
                        for side in ("start", "end"):
                            a, b = (row.get(side + k) for k in ("_lower_ns", "_upper_ns"))
                            if type(a) is int and type(b) is int:
                                delta[side + "_width_minus_uncertainty_ns"] = b - a - u
                    bad.append(
                        dict(
                            artifact=artifact,
                            forward_index=i,
                            identity=device.get("identity"),
                            error=str(error),
                            **fields,
                            anchor_uncertainty_ns=u,
                            **delta,
                            projection_version=device.get("projection_version", "legacy"),
                            anchor_before_ns=device.get("anchor_before_ns"),
                            anchor_after_ns=device.get("anchor_after_ns"),
                        )
                    )
    return {
        "inspected_forwards": inspected,
        "failure_count": count,
        "rows": bad,
        "truncated": count > len(bad),
        "legacy_anchor_offsets_not_retained": True,
    }
