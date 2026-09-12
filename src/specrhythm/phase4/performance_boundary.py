"""Phase-4B.2 decode-only measurement-boundary contracts.

The Phase-4B.1 decode-ready manifest keeps its historical correctness boundary.
Performance uses a later boundary, after setup-ready publication and a final TP
barrier/CUDA synchronization.  Serial and Dual are forbidden to draft before
this second boundary.
"""

from __future__ import annotations

import os
import time
from typing import Any, Mapping, Optional, Sequence

from specrhythm.phase4.transport import CheckpointJsonl

PERFORMANCE_ENV = "SR_PHASE4B2_PERFORMANCE"
PERFORMANCE_EVENT_SCHEMA = "specrhythm.phase4b2-measurement-boundary.v1"
PERFORMANCE_EVENT = "phase4b2-performance-measurement-start"
PERFORMANCE_COMMIT_SCHEMA = "specrhythm.phase4b2-token-commit.v1"


def performance_requested() -> bool:
    value = os.environ.get(PERFORMANCE_ENV, "0")
    if value not in {"0", "1"}:
        raise RuntimeError(f"{PERFORMANCE_ENV} must resolve to 0 or 1")
    return value == "1"


def publish_performance_boundary(
    *,
    tp_group: Any,
    torch_module: Any,
    tp_rank: int,
    timing_log: CheckpointJsonl,
    consumer: str,
    ready_published_ns: int,
) -> int:
    """Publish one rank-zero monotonic boundary after all setup work is visible."""

    tp_group.barrier()
    torch_module.cuda.synchronize()
    boundary: Optional[int] = time.monotonic_ns() if tp_rank == 0 else None
    boundary = tp_group.broadcast_object(boundary, src=0)
    if not isinstance(boundary, int) or boundary <= ready_published_ns:
        raise RuntimeError("Phase-4B.2 boundary did not follow setup-ready publication")
    if tp_rank == 0:
        timing_log.append(
            {
                "schema_version": PERFORMANCE_EVENT_SCHEMA,
                "event": PERFORMANCE_EVENT,
                "timestamp_ns": boundary,
                "consumer": consumer,
                "setup_ready_published_ns": ready_published_ns,
                "pre_measurement_tp_barrier": True,
                "pre_measurement_target_cuda_synchronize": True,
                "setup_excluded": True,
                "bootstrap_excluded_from_measured_tokens": True,
            }
        )
    return boundary


def extract_performance_boundary(
    rows: Sequence[Mapping[str, Any]], *, consumer: str
) -> tuple[Optional[int], list[str]]:
    matches = [
        row
        for row in rows
        if row.get("schema_version") == PERFORMANCE_EVENT_SCHEMA
        and row.get("event") == PERFORMANCE_EVENT
    ]
    errors = []
    if len(matches) != 1:
        errors.append("exactly one Phase-4B.2 measurement boundary is required")
        return None, errors
    row = matches[0]
    if row.get("consumer") != consumer:
        errors.append("performance boundary consumer differs")
    timestamp = row.get("timestamp_ns")
    ready = row.get("setup_ready_published_ns")
    if (
        not isinstance(timestamp, int)
        or not isinstance(ready, int)
        or timestamp <= ready
    ):
        errors.append("performance boundary does not follow setup-ready publication")
    if row.get("pre_measurement_tp_barrier") is not True:
        errors.append("performance boundary lacks the final TP barrier")
    if row.get("pre_measurement_target_cuda_synchronize") is not True:
        errors.append("performance boundary lacks Target CUDA synchronization")
    if row.get("setup_excluded") is not True:
        errors.append("performance boundary does not exclude setup")
    return timestamp if isinstance(timestamp, int) else None, errors


def record_performance_commit(
    timing_log: CheckpointJsonl,
    *,
    request_id: str,
    token_ids: Sequence[int],
    source: str,
) -> None:
    """Record an explicit semantic commit without synchronizing CUDA."""

    if not performance_requested():
        return
    tokens = list(token_ids)
    if not request_id or not tokens or any(
        not isinstance(token, int) or isinstance(token, bool) or token < 0
        for token in tokens
    ):
        raise RuntimeError("Phase-4B.2 commit identity or token IDs are invalid")
    timing_log.append(
        {
            "schema_version": PERFORMANCE_COMMIT_SCHEMA,
            "event": "measured-token-commit",
            "timestamp_ns": time.monotonic_ns(),
            "request_id": request_id,
            "token_ids": tokens,
            "source": source,
            "per_token_cuda_synchronize": False,
        }
    )
