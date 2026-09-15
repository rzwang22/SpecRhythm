"""Window-bounded rolling-eager counters, physical evidence, and cleanup checks."""

from __future__ import annotations

import math

from specrhythm.serving.common import digest, require

COUNTERS = (
    "admissions", "started", "completed", "parent_full_accepts", "parent_rejections",
    "bridge_matches", "bridge_mismatches", "promotions", "verified_promoted_candidates",
    "accepted_promoted_candidates", "discarded_early_tokens", "recovery_jobs",
    "recovery_generated_tokens", "early_generated_tokens", "bridge_generated_tokens",
    "draft_materialized_tokens", "committed_tokens", "parent_accepted_tokens",
    "correction_tokens", "bonus_tokens", "unhidden_wait_ns",
    "normal_generated_tokens", "reused_candidate_tokens", "reused_bridge_tokens",
)


def summarize_eager(backend, runtime):
    """Never equate generation, promotion, verification, or acceptance with output.

    Counter events are attributed at their completion/update timestamp. The
    committed-output throughput remains the existing coordinator measurement.
    Cross-device overlap is only qualified from native CUDA clock bounds; a
    queued task or host call interval never establishes physical overlap.
    """

    source = backend.get("rolling_eager")
    require(isinstance(source, dict) and source.get("schema_version"),
            "serial-eager backend report missing")
    counters = source.get("counters")
    require(isinstance(counters, dict) and all(
        type(counters.get(name)) is int and counters[name] >= 0 for name in COUNTERS
    ), "serial-eager counters incomplete or invalid")
    events = source.get("events")
    require(isinstance(events, list), "serial-eager event evidence missing")
    lifetime = dict.fromkeys(COUNTERS, 0)
    window = dict.fromkeys(COUNTERS, 0)
    start, end = runtime.get("measurement_start_ns"), runtime.get("measurement_end_ns")
    window_available = type(start) is int and type(end) is int and end >= start
    selected_events = 0
    clipped_wait_ns = 0
    for event in events:
        first, last = event.get("start_ns"), event.get("end_ns")
        require(type(first) is int and type(last) is int and 0 <= first <= last,
                "serial-eager event interval invalid")
        delta = event.get("counter_delta")
        require(isinstance(delta, dict) and all(
            name in COUNTERS and type(value) is int and value >= 0
            for name, value in delta.items()
        ), "serial-eager event counter delta invalid")
        measured = window_available and start <= last <= end
        selected_events += int(measured)
        for name, value in delta.items():
            lifetime[name] += value
            if measured:
                window[name] += value
        if window_available and delta.get("unhidden_wait_ns", 0):
            require(delta["unhidden_wait_ns"] == last - first,
                    "serial-eager wait counter lacks its actual wait interval")
            clipped_wait_ns += max(0, min(last, end) - max(first, start))
    require(all(lifetime[name] == counters[name] for name in COUNTERS),
            "serial-eager counters differ from retained events")
    require(counters["committed_tokens"] == counters["parent_accepted_tokens"]
            + counters["correction_tokens"] + counters["bonus_tokens"],
            "serial-eager committed accounting does not conserve tokens")
    if "requests" in runtime:
        committed = sum(len(commit["token_ids"]) for row in runtime["requests"]
                        for commit in row["commits"])
        require(counters["committed_tokens"] == committed,
                "serial-eager committed count differs from authoritative coordinator output")
    require(counters["accepted_promoted_candidates"] <= counters["verified_promoted_candidates"]
            <= 4 * counters["promotions"],
            "serial-eager promoted/verified/accepted accounting invalid")
    require(counters["completed"] <= counters["started"] <= counters["admissions"],
            "serial-eager admission/start/completion accounting invalid")
    pending = source.get("pending_work")
    require((type(pending) is int and pending == 0) or pending == [],
            "serial-eager owner retains pending work", actual=pending)
    require(source.get("owner_stopped") is True, "serial-eager owner did not stop")
    overlap = eager_overlap(runtime.get("target_devices", []), backend.get("fixed_device", {}),
                            start, end)
    return {
        "schema_version": source["schema_version"],
        "lifetime_counters": {name: counters[name] for name in COUNTERS},
        "window_counters": window if window_available else None,
        "window_counter_boundary": "event completion/update within inclusive measured boundary",
        "window_unhidden_wait_ns": clipped_wait_ns if window_available else None,
        "window_wait_boundary": "actual wait interval clipped to measured window",
        "event_count": len(events),
        "window_event_count": selected_events,
        "event_sha256": digest(events),
        "raw_events_artifact": "draft-backend-report.json:rolling_eager.events",
        "observation": {
            name: "OBSERVED" if window.get(name, 0) else "NOT_OBSERVED"
            for name in ("started", "promotions", "verified_promoted_candidates",
                         "accepted_promoted_candidates", "parent_rejections", "bridge_mismatches")
        },
        "GPU_overlap": overlap,
        "cleanup_status": "PASS",
        "pending_work": pending,
        "owner_stopped": True,
        "output_accounting": "only Target commits count as output; bridge never recounted",
    }


def eager_overlap(devices, draft, start, end):
    """Require real, clock-bounded continuation forwards on both devices."""

    required = ("start_lower_ns", "start_upper_ns", "end_lower_ns", "end_upper_ns",
                "gpu_event_ms")
    candidates = [row for row in draft.get("forwards", ())
                  if row.get("purpose") in (
                      "eager", "continuation", "eager_continuation", "rolling_eager")]
    targets = [row for device in devices for row in device.get("device", {}).get("forwards", ())]
    identities = [device.get("device", {}).get("identity", {}) for device in devices]
    identities_valid = (
        len(identities) == 2 and {identity.get("global_rank") for identity in identities} == {0, 1}
        and len({identity.get("gpu_uuid") for identity in identities}
                | {draft.get("identity", {}).get("gpu_uuid")}) == 3
        and all(identity.get("gpu_uuid") for identity in identities)
        and bool(draft.get("identity", {}).get("gpu_uuid"))
    )
    if (start is None or end is None or not candidates or not targets
            or not identities_valid
            or any(any(name not in row for name in required) for row in candidates + targets)):
        return {"status": "UNKNOWN", "reason": "native eager/Target CUDA bounds unavailable",
                "host_overlap_is_gpu_evidence": False}
    require(all(
        all(type(row[name]) is int and row[name] >= 0 for name in required[:-1])
        and type(row["gpu_event_ms"]) in (int, float)
        and math.isfinite(row["gpu_event_ms"]) and row["gpu_event_ms"] > 0
        and row["start_lower_ns"] <= row["start_upper_ns"]
        and row["end_lower_ns"] <= row["end_upper_ns"]
        and row["start_lower_ns"] < row["end_lower_ns"]
        and row["start_upper_ns"] < row["end_upper_ns"]
        for row in candidates + targets
    ), "invalid native eager/Target CUDA timing evidence")
    from specrhythm.serving.fixed_results import overlap_metrics

    # The shared interval engine filters proposal/commit; project only the already
    # selected native continuation records, retaining every measured device bound.
    evidence = overlap_metrics(devices, {"forwards": [
        {**row, "purpose": "proposal"} for row in candidates
    ]}, start, end)
    return {"status": "OBSERVED" if evidence["event_overlap_lower_ms"] > 0 else "UNKNOWN",
            "reason": "native CUDA clock-bound intersection",
            "host_overlap_is_gpu_evidence": False, **evidence}


def eager_columns(value):
    """A compact CSV/status projection; absence means this was a baseline mode."""

    eager = value.get("rolling_eager")
    if eager is None:
        return {}
    counters = eager.get("window_counters") or {}
    return {
        "eager_started": counters.get("started"),
        "eager_completed": counters.get("completed"),
        "eager_promotions": counters.get("promotions"),
        "eager_verified_candidates": counters.get("verified_promoted_candidates"),
        "eager_accepted_candidates": counters.get("accepted_promoted_candidates"),
        "eager_recovery_jobs": counters.get("recovery_jobs"),
        "eager_unhidden_wait_ms": eager["window_unhidden_wait_ns"] / 1e6
        if eager.get("window_unhidden_wait_ns") is not None else None,
        "eager_gpu_overlap_status": eager.get("GPU_overlap", {}).get("status", "UNKNOWN"),
        "eager_cleanup_status": eager.get("cleanup_status"),
    }
