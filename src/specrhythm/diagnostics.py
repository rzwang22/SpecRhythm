"""Compact reporting helpers for simulator capacity and eager sensitivity sweeps."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from specrhythm.cli import POLICY_ORDER, _policy
from specrhythm.schema import Workload
from specrhythm.simulator import SimulatorConfig, simulate


def capacity_knee_report(
    workload_paths: Iterable[tuple[float, str]], config: SimulatorConfig
) -> dict[str, Any]:
    rows = []
    for scale, workload_path in workload_paths:
        workload = Workload.load_jsonl(workload_path)
        for policy_name in POLICY_ORDER:
            summary = simulate(workload, _policy(policy_name, config), config).summary
            rows.append(
                {
                    "time_scale": scale,
                    "policy": policy_name,
                    "allocator": summary.allocator,
                    "execution_mode": summary.execution_mode,
                    "eager_semantics": summary.eager_semantics,
                    "goodput_tokens_per_s": summary.goodput_tokens_per_s,
                    "slo_attainment": summary.slo_attainment,
                    "raw_throughput_tokens_per_s": summary.raw_throughput_tokens_per_s,
                    "slo_good_tokens": summary.slo_good_tokens,
                    "makespan_ms": summary.makespan_ms,
                    "mean_queueing_latency_ms": summary.mean_queueing_latency_ms,
                    "mean_service_latency_ms": summary.mean_service_latency_ms,
                    "mean_decode_latency_ms": summary.mean_decode_latency_ms,
                    "slo_class_metrics": summary.slo_class_metrics,
                    "tree": {
                        "mean_width": summary.mean_candidate_tree_width,
                        "mean_depth": summary.mean_candidate_tree_depth,
                        "mean_nodes": summary.mean_candidate_tree_nodes,
                        "selected_path_probability_distribution": (
                            summary.selected_path_probability_distribution
                        ),
                    },
                }
            )
    return {
        "schema_version": "specrhythm.capacity-knee.v1",
        "model_status": "simulator-proxy-not-gpu-measured",
        "rows": rows,
    }


def eager_grid_report(
    workload: Workload,
    config: SimulatorConfig,
    *,
    eager_budgets: Iterable[int] = (1, 2, 4),
    dependency_probabilities: Iterable[float] = (0.1, 0.2, 0.3, 0.5),
) -> dict[str, Any]:
    rows = []
    for eager_budget in eager_budgets:
        for probability in dependency_probabilities:
            configured = replace(
                config,
                max_eager_budget=eager_budget,
                min_dependency_path_probability=probability,
            )
            for policy_name in ("dual-eager", "specrhythm"):
                summary = simulate(
                    workload, _policy(policy_name, configured), configured
                ).summary
                rows.append(
                    {
                        "max_eager_budget": eager_budget,
                        "minimum_dependency_path_probability": probability,
                        "policy": policy_name,
                        "goodput_tokens_per_s": summary.goodput_tokens_per_s,
                        "slo_attainment": summary.slo_attainment,
                        "eager_drafted_tokens": summary.eager_drafted_tokens,
                        "eager_promotion_token_ratio": summary.eager_promotion_token_ratio,
                        "eager_invalidation_token_ratio": (
                            summary.eager_invalidation_token_ratio
                        ),
                        "eager_compute_waste_ratio": summary.eager_compute_waste_ratio,
                    }
                )
    return {
        "schema_version": "specrhythm.eager-grid.v1",
        "model_status": "simulator-proxy-not-gpu-measured",
        "rows": rows,
    }


def shaping_loss_report(
    baseline: dict[str, Any], shaped: dict[str, Any]
) -> dict[str, Any]:
    """Decompose class resource transfer and goodput numerator/denominator changes."""

    class_rows = {}
    for label in sorted(set(baseline["slo_class_metrics"]) | set(shaped["slo_class_metrics"])):
        before = baseline["slo_class_metrics"].get(label, {})
        after = shaped["slo_class_metrics"].get(label, {})
        class_rows[label] = {
            "candidate_budget_delta": (
                after.get("drafted_tokens", 0) - before.get("drafted_tokens", 0)
            ),
            "expected_progress_delta": (
                after.get("mean_expected_progress", 0)
                * after.get("scheduled_request_ratio", 0)
                - before.get("mean_expected_progress", 0)
                * before.get("scheduled_request_ratio", 0)
            ),
            "realized_progress_delta": (
                after.get("accepted_tokens", 0) - before.get("accepted_tokens", 0)
            ),
            "attained_requests_delta": (
                after.get("attainment", 0) * after.get("requests", 0)
                - before.get("attainment", 0) * before.get("requests", 0)
            ),
        }
    return {
        "class_deltas": class_rows,
        "slo_good_tokens_delta": shaped["slo_good_tokens"]
        - baseline["slo_good_tokens"],
        "makespan_ms_delta": shaped["makespan_ms"] - baseline["makespan_ms"],
        "goodput_tokens_per_s_delta": shaped["goodput_tokens_per_s"]
        - baseline["goodput_tokens_per_s"],
    }


def request_level_shaping_loss_report(
    baseline: dict[str, Any], shaped: dict[str, Any]
) -> dict[str, Any]:
    before = {
        row["request_id"]: row for row in baseline["request_allocation_diagnostics"]
    }
    after = {
        row["request_id"]: row for row in shaped["request_allocation_diagnostics"]
    }
    rows = []
    tight_extra_but_missed = 0
    relaxed_lost_attainment = 0
    for request_id in sorted(before.keys() & after.keys()):
        old = before[request_id]
        new = after[request_id]
        node_delta = (
            new["allocated_candidate_nodes"] - old["allocated_candidate_nodes"]
        )
        if new["slo_tpot_ms"] in {40, 50} and node_delta > 0 and not new["attained"]:
            tight_extra_but_missed += 1
        if new["slo_tpot_ms"] == 150 and old["attained"] and not new["attained"]:
            relaxed_lost_attainment += 1
        rows.append(
            {
                "request_id": request_id,
                "slo_tpot_ms": new["slo_tpot_ms"],
                "candidate_node_delta": node_delta,
                "expected_progress_delta": new["expected_progress"]
                - old["expected_progress"],
                "realized_progress_delta": new["realized_candidate_progress"]
                - old["realized_candidate_progress"],
                "baseline_attained": old["attained"],
                "shaped_attained": new["attained"],
            }
        )
    return {
        "tight_extra_budget_but_still_missed": tight_extra_but_missed,
        "relaxed_150ms_lost_attainment": relaxed_lost_attainment,
        "rows": rows,
    }


def write_report(report: dict[str, Any], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
