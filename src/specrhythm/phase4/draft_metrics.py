"""In-memory Draft execution counters; no timed-path filesystem writes."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


def batch_statistics(histogram: Counter) -> dict[str, Any]:
    count = sum(histogram.values())

    def quantile(q: float) -> Any:
        rank = max(1, math.ceil(q * count))
        cumulative = 0
        for size, frequency in sorted(histogram.items()):
            cumulative += frequency
            if cumulative >= rank:
                return size
        return None

    return {
        "count": count,
        "min": min(histogram) if count else None,
        "p10": quantile(0.1),
        "p50": quantile(0.5),
        "p90": quantile(0.9),
        "max": max(histogram) if count else None,
        "mean": sum(k * v for k, v in histogram.items()) / count if count else None,
        "histogram": {str(k): v for k, v in sorted(histogram.items())},
    }


class DraftMetrics:
    def __init__(self) -> None:
        self.forwards: Counter = Counter()
        self.batches = {purpose: Counter() for purpose in ("setup", "proposal", "commit")}
        self.query_tokens: Counter = Counter()
        self.active_steps: Counter = Counter()
        self.counters: Counter = Counter()
        self.syncs: Counter = Counter()
        self.gpu_ms: Counter = Counter()
        self.host_gap_ns = 0
        self.host_gap_count = 0
        self.failed = False
        self.s1_forward_records: list[dict[str, Any]] = []

    def forward(self, purpose: str, requests: int, tokens: int) -> None:
        if purpose not in (*self.batches, "warmup") or requests < 1 or tokens < requests:
            raise ValueError("invalid actual Draft forward evidence")
        self.forwards[purpose] += 1
        if purpose in self.batches:
            self.batches[purpose][requests] += 1
            self.query_tokens[purpose] += tokens

    def snapshot(self, backend_name: str) -> dict[str, Any]:
        measured = self.batches["proposal"] + self.batches["commit"]
        stats = batch_statistics(measured)
        return {
            **({"s1_forward_records": list(self.s1_forward_records)}
               if self.s1_forward_records else {}),
            "schema_version": "specrhythm.phase4b3-draft-backend.v1",
            "backend_name": backend_name,
            "serving_performance_backend": True,
            "gpu_performance_validated": False,
            "execution_failed": self.failed,
            "draft_model_forward_count": sum(self.forwards[p] for p in ("proposal", "commit")),
            "draft_model_forward_count_total": sum(self.forwards.values()),
            "draft_model_forward_count_by_purpose": dict(self.forwards),
            "draft_batch_count": stats["count"],
            **{
                f"draft_batch_size_{key}": stats[key]
                for key in ("min", "p10", "p50", "p90", "max", "mean")
            },
            "draft_batch_size_histogram": stats["histogram"],
            "draft_batch_statistics_by_purpose": {
                p: batch_statistics(h) for p, h in self.batches.items()
            },
            "draft_autoregressive_step_count": sum(self.active_steps.values()),
            "draft_active_rows_per_step": batch_statistics(self.active_steps),
            "draft_proposed_token_count": self.counters["proposed_tokens"],
            "draft_proposal_count": self.counters["proposals"],
            "draft_kv_materialization_forward_count": self.forwards["commit"],
            "draft_correction_bonus_forward_count": self.counters["correction_bonus_batches"],
            "draft_kv_operations": dict(self.counters),
            "draft_scheduled_token_count_by_purpose": dict(self.query_tokens),
            "draft_host_sync_count": sum(self.syncs.values()),
            "draft_host_sync_count_by_reason": dict(self.syncs),
            "draft_scalar_item_count": 0,
            "draft_internal_host_sync_count": None,
            "draft_sync_coverage": "explicit adapter fences and bulk token D2H only",
            "draft_gpu_event_time_ms": self.gpu_ms["proposal"] + self.gpu_ms["commit"],
            "draft_gpu_event_time_ms_by_purpose": dict(self.gpu_ms),
            "draft_gpu_event_scope": "actual model forward hooks, not kernel self time",
            "draft_step_host_gap_ns_total": self.host_gap_ns,
            "draft_step_host_gap_count": self.host_gap_count,
            "counter_scope": "proposal/commit; setup and warmup separately reported",
            "quantile_method": "nearest rank over actual model forwards",
            "true_batching_observed": bool(stats["max"] and stats["max"] > 1),
        }
