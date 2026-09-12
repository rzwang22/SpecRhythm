"""Offline timing checks for fixed-cohort characterization, including zero overlap."""

from __future__ import annotations

import math

from specrhythm.phase4.transport import CheckpointJsonl
from specrhythm.phase4.vllm_dual import validate_target_rank_identity


def _interval(start, end):
    if not (type(start) is int and type(end) is int and 0 <= start < end):
        raise ValueError("malformed Dual host timing interval")
    return start, end


def _cuda(row):
    elapsed = row.get("cuda_elapsed_ns")
    if not (
        row.get("cuda_events") is True
        and row.get("cuda_synchronized") is True
        and type(elapsed) in (int, float)
        and math.isfinite(elapsed)
        and elapsed > 0
    ):
        raise ValueError("Dual timing lacks positive synchronized CUDA event evidence")


def characterize_overlap(drafts, verifies, overlaps, workers=()):
    # Import lazily: the runner also uses this CPU-only check after execution.
    from specrhythm.phase4.dual_runner import build_cycle_and_overlap_events

    try:
        draft_intervals = []
        for row in drafts:
            result = row.get("result", {})
            if not result.get("proposal"):
                continue
            interval = result["draft_gpu_interval"]
            start, end = _interval(interval["host_start_ns"], interval["host_end_ns"])
            _cuda(interval)
            if interval.get("physical_gpu_id") != 0:
                raise ValueError("Draft overlap timing is not from physical GPU 0")
            draft_intervals.append((row["request_id"], start, end))
        if not draft_intervals or not verifies:
            raise ValueError("Dual characterization timing evidence is missing")
        batches, intersections = {}, []
        for row in verifies:
            start, end = _interval(row["verify_host_start_ns"], row["verify_host_end_ns"])
            ranks = row["target_rank_intervals"]
            identity_errors = validate_target_rank_identity(ranks, 2, workers)
            if identity_errors or row.get("target_physical_gpu_ids") != [1, 2]:
                raise ValueError("Target overlap GPU/TP identity invalid: " + str(identity_errors))
            for rank in ranks:
                a, b = _interval(rank["host_start_ns"], rank["host_end_ns"])
                _cuda(rank)
                if not a <= start < end <= b:
                    raise ValueError("verification interval is outside synchronized TP intervals")
            ids = tuple(row["verify_request_ids"])
            if not ids or len(set(ids)) != len(ids) or row.get("request_id") not in ids:
                raise ValueError("verification cohort identity invalid")
            signature = ids
            batch_id = row["verify_microbatch_id"]
            if batch_id in batches and batches[batch_id] != signature:
                raise ValueError("verification cohort identity disagrees between rows")
            batches[batch_id] = signature
            intersections.extend(
                (max(start, a), min(end, b))
                for rid, a, b in draft_intervals
                if rid not in ids and a < end and b > start
            )
        _, expected = build_cycle_and_overlap_events(drafts, verifies)
        if len(expected) != len(overlaps):
            raise ValueError("overlap evidence does not cover verification batches")
        for actual, wanted in zip(overlaps, expected):
            # Reproduce the existing witness envelope without modifying its logger.
            material = (
                "draft_request_ids",
                "verify_request_ids",
                "request_sets_disjoint",
                "draft_physical_gpu_ids",
                "target_physical_gpu_ids",
                "draft_cuda_events",
                "target_rank_intervals",
                "host_interval",
                "overlap_duration_ns",
            )
            if any(actual.get(key) != wanted[key] for key in material):
                raise ValueError("overlap claim is unsupported by recorded Draft/Target timing")
        merged = []
        for start, end in sorted(set(intersections)):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(end, merged[-1][1])
            else:
                merged.append([start, end])
        return {
            "valid": True,
            "errors": [],
            "physical_overlap_valid": bool(merged),
            "observed_overlap_ms": sum(b - a for a, b in merged) / 1e6,
            "overlap_is_critical_path_time_saved": False,
            "overlap_duration_method": "union of actual disjoint-request interval intersections",
        }
    except (KeyError, TypeError, ValueError) as error:
        return {"valid": False, "errors": [str(error)]}


def read_overlap(directory, raw):
    return characterize_overlap(
        *[
            CheckpointJsonl(directory / name).read()
            for name in (
                "draft-work-events.jsonl",
                "verification-events.jsonl",
                "overlap-events.jsonl",
            )
        ],
        raw.get("worker_ranks", ()),
    )
