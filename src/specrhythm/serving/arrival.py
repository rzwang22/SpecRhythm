"""Timestamp-only Mooncake composition. This module never submits a request."""

from __future__ import annotations

import json
from pathlib import Path

from specrhythm.serving.common import distribution, finite, require, source_path


def read_arrivals(lock: dict, root: Path) -> list[dict]:
    source = lock["sources"]["arrival"]
    files = [r for r in source["files"] if r["role"] == "data"]
    require(len(files) == 1, "exactly one arrival trace is required")
    file = files[0]
    rows = []
    with source_path(root, "arrival", file["path"]).open(encoding="utf-8") as handle:
        for line, raw in enumerate(handle, 1):
            require(raw.strip(), "blank Mooncake trace row", line=line)
            record = json.loads(raw)
            value = record.get("timestamp") if isinstance(record, dict) else None
            require(finite(value), "invalid Mooncake timestamp in milliseconds", line=line)
            rows.append(
                {
                    "repository_id": source["repository_id"],
                    "revision": source["revision"],
                    "file": file["path"],
                    "file_sha256": file["sha256"],
                    "original_line_index": line,
                    "source_timestamp_ms": value,
                }
            )
    return sorted(rows, key=lambda r: (r["source_timestamp_ms"], r["original_line_index"]))


def windows(rows: list[dict], config: dict) -> dict:
    counts = {s: sum(config["splits"][s]["quotas"].values()) for s in ("main", "calibration")}
    require(
        len(rows) >= sum(counts.values()),
        "insufficient original Mooncake arrivals",
        available=len(rows),
        required=sum(counts.values()),
    )
    result, offset = {}, 0
    for split in ("main", "calibration"):
        selected = rows[offset : offset + counts[split]]
        origin = selected[0]["source_timestamp_ms"]
        result[split] = [
            {**r, "sorted_index": i + offset, "selected_first_timestamp_ms": origin}
            for i, r in enumerate(selected)
        ]
        offset += counts[split]
    return result


def scheduled_arrival(arrival_time_ms, time_scale=1.0):
    require(
        finite(arrival_time_ms) and finite(time_scale) and time_scale > 0,
        "arrival/time_scale must be finite with a positive scale",
    )
    scheduled = arrival_time_ms / time_scale
    require(finite(scheduled), "scaled arrival overflows finite milliseconds")
    return scheduled


def arrival_summary(rows: list) -> dict:
    times = [r.arrival_time_ms for r in rows]
    gaps = [b - a for a, b in zip(times, times[1:])]
    span = times[-1] - times[0] if times else 0
    return {
        "count": len(times),
        "coverage_ms": span,
        "interval_ms": distribution(gaps),
        "same_timestamp_adjacent_pairs": sum(g == 0 for g in gaps),
        "observed_iat_rate_per_second": (len(times) - 1) * 1000 / span
        if len(times) > 1 and span > 0
        else None,
        "rate_definition": "(request_count - 1) / observed span in seconds",
        "scope": "Mooncake arrival times only; no source lengths or prefix identities",
    }
