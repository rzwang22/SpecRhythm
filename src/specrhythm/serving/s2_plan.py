"""Frozen S2 selection, Poisson arrivals, conservative resident capacity and engineering SLO."""

from __future__ import annotations

import math
import random
from collections import Counter
from statistics import median

from specrhythm.serving.common import TASKS, digest, require

RATIO = {"chat": 3, "code": 3, "reasoning": 2, "summarization": 2}
MODES = ("target", "serial", "pingpong")
RATES = (0.25, 0.5, 1.0)
ACTIVE_LIMIT = 128
POOL_SLOTS = 512
QUERY_LIMIT = 4096
BOUNDARY = (
    "GPU-resident prefilled-KV delivery; prefill/import excluded; queue/decode/drain included"
)
LABEL = "finite-trace serving observation"


def sealed(value, field="sha256"):
    return {**value, field: digest(value)}


def check_seal(value, field="sha256"):
    require(
        value.get(field) == digest({k: v for k, v in value.items() if k != field}),
        "S2 immutable JSON hash mismatch",
        field=field,
    )


def selection_order(main, small_ids, seed=1666):
    """Prioritize the validated mixed100 IDs; append by independent seeded hash ranking."""
    by_id = {r.request_id: r for r in main}
    require(len(by_id) == len(main) and len(main) == 1000, "S2 requires unique main1000")
    require(all(r.split == "main" for r in main), "calibration requests cannot enter S2 main")
    require(
        len(small_ids) == len(set(small_ids)) == 100 and set(small_ids) <= set(by_id),
        "S1 mixed100 request identity mismatch",
    )
    require(
        Counter(by_id[r].task_class for r in small_ids) == {k: 10 * v for k, v in RATIO.items()},
        "S1 mixed100 class ratio differs",
    )
    preferred = set(small_ids)
    groups = {}
    for task in RATIO:
        first = [r for r in small_ids if by_id[r].task_class == task]
        rest = [
            r.request_id for r in main if r.task_class == task and r.request_id not in preferred
        ]
        groups[task] = first + sorted(rest, key=lambda rid: (digest([seed, task, rid]), rid))
    # Every prefix of 10 preserves 3:3:2:2; source rows themselves are never edited.
    order = []
    for block in range(50):
        for task, quota in RATIO.items():
            order.extend(groups[task][block * quota : (block + 1) * quota])
    require(
        len(order) == len(set(order)) == 500 and set(order[:100]) == preferred,
        "S2 nested selection failed",
    )
    return sealed(
        {
            "schema_version": "specrhythm.s2-selection.v1",
            "selection_seed": seed,
            "algorithm": "S1-mixed100-first; class hash rank; nested ratio blocks of ten",
            "request_ids": order,
            "request_set_sha256": digest(sorted(order)),
            "preferred_s1_ids_sha256": digest(sorted(small_ids)),
        }
    )


def poisson_trace(ids, arrival_rate_qps, arrival_seed=1667, order_seed=1668):
    require(
        isinstance(arrival_rate_qps, (int, float))
        and math.isfinite(arrival_rate_qps)
        and arrival_rate_qps > 0,
        "invalid arrival_rate_qps",
    )
    require(ids and len(ids) == len(set(ids)), "trace request IDs must be unique/nonempty")
    order = list(ids)
    random.Random(order_seed).shuffle(order)
    stream = random.Random(arrival_seed)
    units = [0.0] + [stream.expovariate(1.0) for _ in order[1:]]
    cumulative, rows = 0.0, []
    for rid, interval in zip(order, units):
        cumulative += interval
        rows.append(
            {
                "request_id": rid,
                "unit_exponential_interval": interval,
                "arrival_offset_seconds": cumulative / arrival_rate_qps,
            }
        )
    return sealed(
        {
            "schema_version": "specrhythm.s2-trace.v1",
            "kind": "Poisson exponential intervals",
            "arrival_rate_qps": arrival_rate_qps,
            "arrival_seed": arrival_seed,
            "order_seed": order_seed,
            "order_sha256": digest(order),
            "request_set_sha256": digest(sorted(ids)),
            "unit_samples_sha256": digest(units),
            "first_arrival_seconds": 0.0,
            "rows": rows,
            "original_S0_arrivals_modified": False,
        }
    )


def capacity_for(rows, rank, *, active_limit=ACTIVE_LIMIT, speculative_tokens=4):
    """All prefixes + largest active growth, including speculative and allocator margins.

    A fresh engine rebuilds initial state; there are no KV snapshot tensor copies.
    Cache capacity is measured after model/workspace profiling, not inferred from RAM size.
    """
    block = rank["block_size"]
    available = rank["num_gpu_blocks"]
    require(
        type(block) is int and block > 0 and type(available) is int and available > 0,
        "invalid actual KV block capacity",
        rank=rank,
    )
    require(rank["role"] in ("target", "draft"), "unknown capacity rank role")
    require(type(speculative_tokens) is int and speculative_tokens >= 4,
            "speculative capacity cannot be less than the baseline K4 reserve")

    def ceil(n):
        return (n + block - 1) // block

    initial = [ceil(r.prompt_length + (rank["role"] == "draft")) for r in rows]
    growth = [ceil(r.prompt_length + r.maximum_new_tokens + speculative_tokens) - p
              for r, p in zip(rows, initial)]
    active = min(active_limit, len(rows))
    reserve = max(32, math.ceil(available * 0.05))
    partial = active  # Conservative one extra private partial/copy block per active request.
    required = sum(initial) + sum(sorted(growth, reverse=True)[:active]) + partial + reserve
    logits = len(rows) * rank["vocab_size"] * 4 if rank["role"] == "draft" else 0
    workspace = 512 * 1024**2
    free_bytes = rank["free_memory_bytes"]
    require(type(free_bytes) is int and free_bytes >= 0, "actual rank free memory missing")
    return {
        "role": rank["role"],
        "mode": rank["mode"],
        "physical_gpu_id": rank["physical_gpu_id"],
        "gpu_uuid": rank["gpu_uuid"],
        "block_size": block,
        "num_gpu_blocks": available,
        "prefix_blocks": sum(initial),
        "worst_active_growth_blocks": sum(sorted(growth, reverse=True)[:active]),
        **({"speculative_capacity_tokens": speculative_tokens,
            "extra_speculative_tokens": speculative_tokens - 4}
           if speculative_tokens != 4 else {}),
        "active_limit": active_limit,
        "pool_size": len(rows),
        "partial_block_copy_margin": partial,
        "safety_blocks": reserve,
        "snapshot_copy_blocks": 0,
        "required_blocks": required,
        "additional_workspace_reserve_bytes": workspace,
        "resident_draft_logits_bytes": logits,
        "free_memory_bytes_after_engine_init": free_bytes,
        "early_EOS_assumed": False,
        "prefix_eviction_allowed": False,
        "valid": required <= available and logits + workspace <= free_bytes,
    }


def freeze_sizes(main, selection, ranks, *, active_limit=ACTIVE_LIMIT):
    check_seal(selection)
    expected = {
        (m, r, gpu) for m in MODES for r, gpu in (("draft", 0), ("target", 1), ("target", 2))
    }
    require(
        {(r["mode"], r["role"], r["physical_gpu_id"]) for r in ranks} == expected
        and len(ranks) == len(expected),
        "capacity must cover all three modes and all GPU ranks",
    )
    by_id = {r.request_id: r for r in main}
    result = {}
    for name, requested in (("small", 100), ("large", 500)):
        attempts = []
        for n in range(requested, 0, -10):
            chosen = [by_id[r] for r in selection["request_ids"][:n]]
            evidence = [capacity_for(chosen, rank, active_limit=active_limit) for rank in ranks]
            attempts.append(
                {"N": n, "ranks": evidence, "valid": all(r["valid"] for r in evidence)}
            )
            if attempts[-1]["valid"]:
                break
        else:
            raise ValueError(f"S2 capacity cannot hold even ten requests for {name}")
        result[name] = {
            "requested_N": requested,
            "actual_N": n,
            "request_ids": [r.request_id for r in chosen],
            "class_counts": dict(Counter(r.task_class for r in chosen)),
            "capacity_attempts": attempts,
            "shrink_reason": None if n == requested else "common all-mode/rank resident capacity",
        }
    result["large"]["distinct_scale"] = result["large"]["actual_N"] > result["small"]["actual_N"]
    return sealed(
        {
            "schema_version": "specrhythm.s2-capacity-plan.v1",
            "sizes": result,
            "active_limit": active_limit,
            "pool_slots": POOL_SLOTS,
            "query_token_limit": QUERY_LIMIT,
            "all_modes_share_actual_N": True,
        }
    )


def slo_policy(thresholds, *, source="explicit operator thresholds", evidence=None):
    require(
        set(thresholds) == set(TASKS)
        and all(
            type(v) in (int, float) and math.isfinite(v) and v > 0 for v in thresholds.values()
        ),
        "invalid class SLO thresholds",
    )
    return sealed(
        {
            "schema_version": "specrhythm.s2-engineering-slo.v1",
            "source": source,
            "threshold_ms_per_token": dict(thresholds),
            "metric": "queue-inclusive decode average ms per timed token",
            "definition": "(completion - planned arrival) / actual timed tokens",
            "paper_SLO": False,
            "new_engineering_experiment_policy": True,
            "calibration_evidence": evidence,
        }
    )


def calibrated_policy(requests, evidence):
    groups = {task: [] for task in TASKS}
    for row in requests:
        if row["timed_tokens"]:
            groups[row["task_class"]].append(
                (row["completion_ns"] - row["admission_ns"]) / 1e6 / row["timed_tokens"]
            )
    require(
        all(groups.values()), "calibration has no timed tokens for a class; supply explicit SLO"
    )
    baselines = {task: median(values) for task, values in groups.items()}
    return slo_policy(
        {task: 1.5 * v for task, v in baselines.items()},
        source="Target-only active=1; class median active decode ms/token × 1.5",
        evidence={
            **evidence,
            "baselines": baselines,
            "calibration_queue_excluded_from_baseline": True,
        },
    )
