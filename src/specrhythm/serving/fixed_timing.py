"""Observed Draft purpose costs and integer CUDA-anchor projection, CPU only."""

from __future__ import annotations

import math
from collections import defaultdict


def project_anchor(before_ns, after_ns, start_elapsed_ns, end_elapsed_ns):
    # Round only the CUDA offset. Adding an integer monotonic clock to a float first
    # loses low bits above 2**53. Each offset error is <= 0.5 ns, width is exact and
    # duration rounding error <= 1 ns (plus the source CUDA elapsed measurement error).
    if not (
        type(before_ns) is int
        and type(after_ns) is int
        and before_ns <= after_ns
        and all(math.isfinite(v) for v in (start_elapsed_ns, end_elapsed_ns))
        and 0 <= start_elapsed_ns < end_elapsed_ns
    ):
        raise ValueError("invalid CUDA anchor/elapsed offsets")
    a, b = round(start_elapsed_ns), round(end_elapsed_ns)
    return dict(
        start_lower_ns=before_ns + a,
        start_upper_ns=after_ns + a,
        end_lower_ns=before_ns + b,
        end_upper_ns=after_ns + b,
    )


def recorded_costs(devices, draft, target_samples, start, end, rotations):
    from specrhythm.serving.fixed_attribution import gpu_bounds
    from specrhythm.serving.fixed_results import duration

    groups = defaultdict(list)
    for f in draft.get("forwards", []):
        if start <= f["host_start_ns"] <= end:
            groups[f.get("purpose", "unknown")].append(f)
    draft_purposes = {
        k: {"count": len(rows), "launch_selected_sum_ms": sum(r["gpu_event_ms"] for r in rows)}
        for k, rows in groups.items()
    }
    per_rank = {
        str(d["identity"]["global_rank"]): sum(
            f["gpu_event_ms"] for f in d["forwards"] if start <= f["host_start_ns"] <= end
        )
        for d in devices
    }
    parts = {
        "D_proposal": sum(r["gpu_event_ms"] for r in groups["proposal"]),
        "D_commit_or_prefix_sync": sum(r["gpu_event_ms"] for r in groups["commit"]),
        "V_target": sum(s["gpu_event_ms"] for s in target_samples),
        "other_recorded_Draft_GPU": sum(
            r["gpu_event_ms"]
            for k, rows in groups.items()
            if k not in ("proposal", "commit")
            for r in rows
        ),
    }
    lower, upper, errors = [], [], []
    for d in [*devices, draft]:
        try:
            lo, hi = gpu_bounds(d, start, end)
            lower.extend(lo)
            upper.extend(hi)
        except (KeyError, TypeError, ValueError) as error:
            errors.append(str(error))
    return {
        "schema_version": "specrhythm.fixed-recorded-gpu-costs.v1",
        "window_launch_selected_sum_ms": parts,
        "approximate_ms_per_64_rotation": {
            k: v / rotations if rotations else None for k, v in parts.items()
        },
        "complete_rotation_count": rotations,
        "draft_by_purpose": draft_purposes,
        "target_forward_sum_ms_by_rank": per_rank,
        "target_aggregation": "max duration per joined step; never sum TP ranks",
        "clipped_all_recorded_gpu_union_lower_ms": duration(lower) if not errors else None,
        "clipped_all_recorded_gpu_union_upper_ms": duration(upper) if not errors else None,
        "outside_recorded_gpu_upper_union_ms": (end - start) / 1e6 - duration(upper)
        if not errors
        else None,
        "clock_errors": errors,
        "semantics": "purpose sums select host launches in window, may straddle its edge; "
        "union bounds clip physical intervals to window. Rotation division is approximate, "
        "not a per-rotation dependency join or critical path. Outside union is not CPU time. "
        "Only hooked model forwards are covered; other GPU work is UNKNOWN.",
        "all_gpu_work_covered": False,
    }
