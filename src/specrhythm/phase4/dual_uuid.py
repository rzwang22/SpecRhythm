"""Dual-only UUID experiment; worker snapshots remain the startup authority."""

from __future__ import annotations

import os
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from specrhythm.phase4.stock_vllm import (
    _visible_physical_ids,
    _worker_runtime_snapshot,
    active_cuda_device_identity,
)

UUID_QUERY_MODE_ENV = "SR_PHASE4_DUAL_UUID_QUERY_MODE"
_IDENTITY_FIELDS = (
    "logical_cuda_index", "physical_gpu_id", "gpu_uuid", "gpu_name", "cuda_visible_devices",
)
_COUNTERS = (
    "uuid_initial_validation_count",
    "uuid_verification_subprocess_query_count",
    "uuid_cache_hit_count",
    "uuid_verification_access_count",
)
_GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")


def dual_uuid_query_mode() -> str:
    mode = os.environ.get(UUID_QUERY_MODE_ENV, "live")
    if mode not in {"live", "cached"}:
        raise RuntimeError(f"{UUID_QUERY_MODE_ENV} must be live or cached, got {mode!r}")
    return mode


class DualVerificationUuidQuery:
    """One identity and in-memory counters owned by one initialized TP worker.

    Construction always performs the original real worker snapshot. There is no
    constructor taking a UUID, a rank-derived device ID, or a prefilled cache.
    """

    def __init__(self, worker: Any) -> None:
        self.mode = dual_uuid_query_mode()
        self.torch = worker.model_runner.drafter.torch
        self._initial_validations = 0
        self.worker_snapshot = _worker_runtime_snapshot(worker)
        self._identity = MappingProxyType({
            key: self.worker_snapshot[key] for key in _IDENTITY_FIELDS
        })
        self._ranks = MappingProxyType({
            "global_rank": int(worker.rank),
            "local_rank": int(worker.local_rank),
            "tp_rank": int(worker.model_runner.drafter.tp_rank),
        })
        if self.mode == "cached":
            if not _GPU_UUID.fullmatch(str(self._identity["gpu_uuid"])):
                raise RuntimeError("cached Dual startup UUID is malformed")
            self._validate_cached_binding()
        self._initial_validations += 1
        self._verification_queries = 0
        self._cache_hits = 0

    def _validate_cached_binding(self) -> None:
        logical = int(self.torch.cuda.current_device())
        visible = _visible_physical_ids()
        if logical < 0 or logical >= len(visible):
            raise RuntimeError("active CUDA device is outside CUDA_VISIBLE_DEVICES")
        # Keep checking the actual active device, never use TP/local rank as an
        # index. A changed binding requires a new worker and real startup query.
        if (
            logical != self._identity["logical_cuda_index"]
            or visible[logical] != self._identity["physical_gpu_id"]
            or os.environ.get("CUDA_VISIBLE_DEVICES") != self._identity["cuda_visible_devices"]
            or self.torch.cuda.get_device_properties(logical).name != self._identity["gpu_name"]
        ):
            raise RuntimeError("cached Dual CUDA device binding changed since startup")

    def for_verification(self) -> dict[str, Any]:
        if self.mode == "live":
            # Exactly the old verification lookup, including its error behavior.
            identity = active_cuda_device_identity(self.torch)
            self._verification_queries += 1
            return identity
        self._validate_cached_binding()
        self._cache_hits += 1
        return dict(self._identity)

    def evidence(self) -> dict[str, Any]:
        return {
            **self._ranks,
            **self._identity,
            "uuid_query_mode": self.mode,
            "uuid_initial_validation_count": self._initial_validations,
            "uuid_verification_subprocess_query_count": self._verification_queries,
            "uuid_cache_hit_count": self._cache_hits,
            "uuid_verification_access_count": self._verification_queries + self._cache_hits,
        }


def worker_dual_runtime_snapshot(worker: Any) -> dict[str, Any]:
    """Replace only Dual's startup RPC, retaining its original snapshot/query."""

    drafter = worker.model_runner.drafter
    if getattr(drafter, "uuid_queries", None) is not None:
        raise RuntimeError("Dual UUID identity is already initialized for this worker")
    queries = DualVerificationUuidQuery(worker)
    drafter.uuid_queries = queries
    return queries.worker_snapshot


def worker_dual_uuid_evidence(worker: Any) -> dict[str, Any]:
    """Read counters once after decode, the original final sync and shutdown."""

    return worker.model_runner.drafter.uuid_queries.evidence()


def build_dual_uuid_query_report(
    rows: Sequence[Mapping[str, Any]],
    worker_rows: Sequence[Mapping[str, Any]],
    verification_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Cross-check counters against the unchanged startup and verification logs."""

    errors = []
    modes = {row["uuid_query_mode"] for row in rows}
    mode = next(iter(modes)) if len(modes) == 1 else None
    if mode not in {"live", "cached"}:
        errors.append("UUID query modes disagree or are missing")
    workers = {row["global_rank"]: row for row in worker_rows}
    if (
        len(rows) != len(workers)
        or {row["global_rank"] for row in rows} != set(workers)
        or {row["tp_rank"] for row in rows} != set(range(len(workers)))
    ):
        errors.append("UUID query evidence does not cover every Target TP rank")
    batches = {row["verify_sequence"] for row in verification_rows}
    if not batches:
        errors.append("no completed verification batches; UUID A/B evidence is inconclusive")
    evidence_by_rank = {row["global_rank"]: row for row in rows}
    for verification in verification_rows:
        intervals = verification.get("target_rank_intervals", ())
        if (
            len(intervals) != len(workers)
            or {row.get("global_rank") for row in intervals} != set(workers)
        ):
            errors.append("UUID verification log does not cover every Target rank")
        for interval in intervals:
            evidence = evidence_by_rank.get(interval.get("global_rank"), {})
            if any(interval.get(key) != evidence.get(key) for key in (
                "logical_cuda_index", "physical_gpu_id", "gpu_uuid",
                "cuda_visible_devices", "local_rank", "tp_rank",
            )):
                errors.append("UUID verification log identity disagrees with validated worker")
    for row in rows:
        worker = workers.get(row["global_rank"], {})
        if any(row[key] != worker.get(key) for key in (*_IDENTITY_FIELDS, "local_rank")):
            errors.append("UUID query identity disagrees with authoritative startup worker")
        queries = row["uuid_verification_subprocess_query_count"]
        hits = row["uuid_cache_hit_count"]
        if row["uuid_initial_validation_count"] != 1:
            errors.append("UUID startup validation count must be one per worker")
        if row["uuid_verification_access_count"] != len(batches) or queries + hits != len(batches):
            errors.append("UUID counters disagree with completed verification batches")
        if mode == "live" and (queries != len(batches) or hits != 0):
            errors.append("live UUID verification must use subprocess queries exclusively")
        if mode == "cached" and (queries != 0 or hits != len(batches)):
            errors.append("cached UUID verification must use validated cache hits exclusively")
    return {
        "schema_version": "specrhythm.phase4b2-dual-uuid-query.v1",
        "valid": not errors,
        "errors": list(dict.fromkeys(errors)),
        "uuid_query_mode": mode,
        **{key: sum(row[key] for row in rows) for key in _COUNTERS},
        "uuid_query_by_rank": sorted((dict(row) for row in rows), key=lambda row: row["tp_rank"]),
        "verification_batch_count": len(batches),
        "counter_scope": "startup worker snapshot and successful verification UUID accesses",
        "final_synchronization_uuid_queries_included": False,
    }
