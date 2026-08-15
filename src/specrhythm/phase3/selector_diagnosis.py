"""Target-blind and oracle selector replay for Phase-3C real candidate forests."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from specrhythm.phase3.phase3c_config import Phase3CConfig, load_frozen_pool_dimensions
from specrhythm.phase3.real_candidate_trace import (
    ImmutableRequestStore,
    LabeledCandidateNode,
    LabeledTraceRecord,
    RuntimeCandidateNode,
    TargetFeatureLeakageError,
    labeled_store,
)

TARGET_BLIND_SELECTORS = (
    "residual-probability",
    "local-probability",
    "depth-normalized-log-path",
    "round-robin-branch-coverage",
    "entropy-margin-heuristic",
)
ORACLE_SELECTOR = "within-request-target-oracle"
SELECTOR_ORDER = TARGET_BLIND_SELECTORS + (ORACLE_SELECTOR,)
POOL_ORDER = ("1x", "2x", "4x")
COVERAGE_DEFINITION_VERSION = "specrhythm.target-coverage.v2"
SUMMARY_SCHEMA_VERSION = "specrhythm.phase3c-selector-diagnosis.v2"


@dataclass(frozen=True)
class SelectionResult:
    selected_node_ids: tuple[str, ...]
    prefix_closure_overhead: int


def _ensure_runtime_nodes(
    values: Sequence[RuntimeCandidateNode],
) -> tuple[RuntimeCandidateNode, ...]:
    if any(not isinstance(node, RuntimeCandidateNode) for node in values):
        raise TargetFeatureLeakageError(
            "target-blind selector accepts RuntimeCandidateNode only; labels are forbidden"
        )
    return tuple(values)


def _score(name: str, node: RuntimeCandidateNode) -> float:
    if name == "residual-probability":
        return node.path_probability
    if name == "local-probability":
        return node.local_probability
    if name == "depth-normalized-log-path":
        return node.log_path_probability / node.depth
    if name == "entropy-margin-heuristic":
        # Frozen before the pilot. No target labels, training, or post-hoc tuning.
        return (
            node.log_path_probability
            + 0.25 * node.top1_top2_margin
            - 0.10 * (node.cumulative_entropy / node.depth)
        )
    raise ValueError(f"selector {name} does not use a scalar score")


def _rank_key(name: str, node: RuntimeCandidateNode) -> tuple[Any, ...]:
    if name == "round-robin-branch-coverage":
        return (
            node.depth,
            node.sibling_rank,
            node.branch_rank,
            -node.path_probability,
            node.stable_node_id,
        )
    return (-_score(name, node), node.depth, node.stable_node_id)


def select_target_blind(
    name: str, values: Sequence[RuntimeCandidateNode], budget: int
) -> SelectionResult:
    if name not in TARGET_BLIND_SELECTORS:
        raise ValueError(f"unknown target-blind selector: {name}")
    nodes = _ensure_runtime_nodes(values)
    if budget < 0:
        raise ValueError("verification budget must be non-negative")
    by_id = {node.stable_node_id: node for node in nodes}
    selected: list[str] = []
    selected_set: set[str] = set()
    while len(selected) < min(budget, len(nodes)):
        eligible = [
            node
            for node in nodes
            if node.stable_node_id not in selected_set
            and (node.parent_id is None or node.parent_id in selected_set)
        ]
        if not eligible:
            break
        node = min(eligible, key=lambda item: _rank_key(name, item))
        selected.append(node.stable_node_id)
        selected_set.add(node.stable_node_id)
    unconstrained = {
        node.stable_node_id
        for node in sorted(nodes, key=lambda item: _rank_key(name, item))[:budget]
    }
    overhead = sum(node_id not in unconstrained for node_id in selected)
    for node_id in selected:
        parent = by_id[node_id].parent_id
        if parent is not None and parent not in selected_set:
            raise AssertionError("target-blind selection lost prefix closure")
    return SelectionResult(tuple(selected), overhead)


def select_within_request_oracle(
    values: Sequence[LabeledCandidateNode], budget: int
) -> SelectionResult:
    if budget < 0:
        raise ValueError("verification budget must be non-negative")
    nodes = tuple(values)
    by_id = {node.runtime_features.stable_node_id: node for node in nodes}
    selected: list[str] = []
    selected_set: set[str] = set()
    while len(selected) < min(budget, len(nodes)):
        eligible = [
            node
            for node in nodes
            if node.runtime_features.stable_node_id not in selected_set
            and (
                node.runtime_features.parent_id is None
                or node.runtime_features.parent_id in selected_set
            )
        ]
        if not eligible:
            break
        node = min(
            eligible,
            key=lambda item: (
                not bool(item.target_only_labels["on_target_path"]),
                item.runtime_features.depth,
                -item.runtime_features.path_probability,
                item.runtime_features.stable_node_id,
            ),
        )
        node_id = node.runtime_features.stable_node_id
        selected.append(node_id)
        selected_set.add(node_id)
    for node_id in selected:
        parent = by_id[node_id].runtime_features.parent_id
        if parent is not None and parent not in selected_set:
            raise AssertionError("oracle selection lost prefix closure")
    return SelectionResult(tuple(selected), 0)


def _longest_accepted_prefix(
    record: LabeledTraceRecord, selected: set[str], pool: set[str]
) -> int:
    by_depth = {
        node.runtime_features.depth: node.runtime_features.stable_node_id
        for node in record.nodes
        if node.target_only_labels["on_target_path"]
        and node.runtime_features.stable_node_id in pool
    }
    accepted = 0
    for depth in range(1, record.target_trajectory.target_path_length + 1):
        node_id = by_depth.get(depth)
        if node_id is None or node_id not in selected:
            break
        accepted += 1
    return accepted


def _target_nodes_by_depth(record: LabeledTraceRecord, pool: set[str]) -> dict[int, str]:
    return {
        node.runtime_features.depth: node.runtime_features.stable_node_id
        for node in record.nodes
        if node.target_only_labels["on_target_path"]
        and node.runtime_features.stable_node_id in pool
    }


def coverage_metrics(
    record: LabeledTraceRecord, pool: set[str], selected: set[str]
) -> dict[str, Any]:
    """Return v2 coverage metrics with one denominator shared by nested pools."""

    by_id = {node.runtime_features.stable_node_id: node for node in record.nodes}
    target_by_depth = _target_nodes_by_depth(record, pool)
    target_length = record.target_trajectory.target_path_length
    forest_depth = max((node.runtime_features.depth for node in record.nodes), default=0)
    eligible = min(target_length, forest_depth)
    present_depths = set(target_by_depth)
    selected_target = selected & set(target_by_depth.values())
    first_missing = next(
        (depth for depth in range(1, target_length + 1) if depth not in present_depths),
        None,
    )
    horizon = {}
    for budget in (4, 8, 16):
        opportunities = min(target_length, budget)
        present = sum(depth in present_depths for depth in range(1, opportunities + 1))
        horizon[str(budget)] = {
            "eligible_target_nodes": opportunities,
            "target_nodes_present": present,
            "target_path_recall": present / opportunities if opportunities else 1.0,
            "first_missing_target_depth": next(
                (depth for depth in range(1, opportunities + 1) if depth not in present_depths),
                None,
            ),
        }
    maximum_pool_depth = max(
        (by_id[node_id].runtime_features.depth for node_id in pool), default=0
    )
    legacy_opportunities = min(target_length, maximum_pool_depth)
    return {
        "eligible_target_trajectory_nodes": eligible,
        "target_path_nodes_present": len(target_by_depth),
        "target_path_recall": len(target_by_depth) / eligible if eligible else 1.0,
        "target_node_density": len(target_by_depth) / len(pool) if pool else 0.0,
        "selected_target_nodes": len(selected_target),
        "selected_target_precision": len(selected_target) / len(selected) if selected else 0.0,
        "selected_target_recall": len(selected_target) / eligible if eligible else 1.0,
        "full_target_trajectory_covered": first_missing is None,
        "full_eligible_target_path_covered": all(
            depth in present_depths for depth in range(1, eligible + 1)
        ),
        "first_missing_target_depth": first_missing,
        "first_missing_within_verification_horizon": horizon["4"]["first_missing_target_depth"],
        "verification_horizon_target_recall": horizon,
        # Preserve the exact v1 value. This was variable-depth-normalized recall,
        # not density, so it is not expected to be monotonic across nested pools.
        "target_path_pool_coverage": len(target_by_depth) / max(1, legacy_opportunities),
        "target_path_opportunities": legacy_opportunities,
    }


def _evaluate_selection(
    record: LabeledTraceRecord,
    ratio: str,
    selector: str,
    selection: SelectionResult,
) -> dict[str, Any]:
    pool = set(record.pool_node_ids[ratio])
    selected = set(selection.selected_node_ids)
    if len(selected) != len(selection.selected_node_ids):
        raise AssertionError("selector returned duplicate node IDs")
    if not selected.issubset(pool):
        raise AssertionError("selector returned a node outside the shared pool")
    if len(selected) > len(pool):
        raise AssertionError("selector exceeded candidate pool")
    by_id = {node.runtime_features.stable_node_id: node for node in record.nodes}
    for node_id in selected:
        parent = by_id[node_id].runtime_features.parent_id
        if parent is not None and parent not in selected:
            raise AssertionError("selector output is not prefix closed")
    accepted = _longest_accepted_prefix(record, selected, pool)
    target_length = record.target_trajectory.target_path_length
    committed = min(target_length, accepted + int(accepted < target_length))
    pool_target = set(record.target_path_node_ids_by_pool[ratio])
    selected_target = selected & pool_target
    coverage = coverage_metrics(record, pool, selected)
    speculative = tuple(
        record.target_trajectory.target_token_ids[:accepted]
        + record.target_trajectory.target_token_ids[accepted:]
    )
    if speculative != record.target_trajectory.target_token_ids:
        raise AssertionError("speculative and target token semantics differ")
    return {
        "request_id": record.request_id,
        "task_class": record.task_class,
        "data_split": record.data_split,
        "pool_ratio": ratio,
        "selector": selector,
        "forest_sha256": record.forest_sha256,
        "target_trajectory_sha256": record.target_trajectory_sha256,
        "selected_node_ids": list(selection.selected_node_ids),
        "search_nodes": len(pool),
        "verified_nodes": len(selected),
        "request_roots": 1,
        "coverage_definition_version": COVERAGE_DEFINITION_VERSION,
        "target_path_opportunities": coverage["target_path_opportunities"],
        "target_path_pool_nodes": len(pool_target),
        "selected_target_path_nodes": len(selected_target),
        "target_path_pool_coverage": coverage["target_path_pool_coverage"],
        "selected_target_path_coverage": len(selected_target)
        / max(1, coverage["target_path_opportunities"]),
        **{
            key: value
            for key, value in coverage.items()
            if key not in {"target_path_pool_coverage", "target_path_opportunities"}
        },
        "accepted_draft_tokens_per_proposal": accepted,
        "committed_tokens_per_proposal": committed,
        "accepted_per_verified": accepted / max(1, len(selected)),
        # v1 aliases are retained for migration; v2 names above use fixed denominators.
        "selected_node_target_precision": len(selected_target) / max(1, len(selected)),
        "selected_node_target_recall": len(selected_target) / max(1, len(pool_target)),
        "first_error_depth": accepted + 1 if accepted < target_length else None,
        "prefix_closure_overhead": selection.prefix_closure_overhead,
        "candidate_accounting": {
            "search_pool_nodes": len(pool),
            "selected_verify_nodes": len(selected),
            "accepted_candidate_tokens": accepted,
            "committed_candidate_tokens": accepted,
            "committed_target_root_tokens": int(accepted < target_length),
        },
    }


def replay_request(record: LabeledTraceRecord, verification_budget: int) -> dict[str, Any]:
    rows = []
    for ratio in POOL_ORDER:
        pool_ids = set(record.pool_node_ids[ratio])
        labeled = tuple(
            node for node in record.nodes if node.runtime_features.stable_node_id in pool_ids
        )
        runtime = tuple(node.runtime_features for node in labeled)
        selections = {
            name: select_target_blind(name, runtime, verification_budget)
            for name in TARGET_BLIND_SELECTORS
        }
        selections[ORACLE_SELECTOR] = select_within_request_oracle(labeled, verification_budget)
        ratio_rows = {
            name: _evaluate_selection(record, ratio, name, selection)
            for name, selection in selections.items()
        }
        oracle = ratio_rows[ORACLE_SELECTOR]["accepted_draft_tokens_per_proposal"]
        baseline = ratio_rows["residual-probability"]["accepted_draft_tokens_per_proposal"]
        gap = oracle - baseline
        for name in SELECTOR_ORDER:
            row = ratio_rows[name]
            accepted = row["accepted_draft_tokens_per_proposal"]
            row["oracle_regret"] = oracle - accepted
            row["oracle_gap_recovery"] = (
                (accepted - baseline) / gap if gap > 0 else (1.0 if accepted == oracle else 0.0)
            )
            row["oracle_gap_absolute"] = oracle - accepted
            row["oracle_gap_relative_to_accepted"] = (
                (oracle - accepted) / accepted if accepted > 0 else None
            )
            committed = row["committed_tokens_per_proposal"]
            row["oracle_gap_relative_to_committed"] = (
                (oracle - accepted) / committed if committed > 0 else None
            )
            rows.append(row)
    return {
        "schema_version": "specrhythm.phase3c-selector-request.v2",
        "coverage_definition_version": COVERAGE_DEFINITION_VERSION,
        "request_id": record.request_id,
        "task_class": record.task_class,
        "data_split": record.data_split,
        "forest_sha256": record.forest_sha256,
        "target_trajectory_sha256": record.target_trajectory_sha256,
        "verification_budget": verification_budget,
        "selectors": list(SELECTOR_ORDER),
        "target_blind_selectors": list(TARGET_BLIND_SELECTORS),
        "rows": rows,
    }


def selector_store(path: Path) -> ImmutableRequestStore[dict[str, Any]]:
    return ImmutableRequestStore(path, lambda value: dict(value), lambda value: value)


def run_selector_replay_stage(
    config: Phase3CConfig,
    *,
    labeled_dir: Path,
    output_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    labels = labeled_store(labeled_dir)
    output = selector_store(output_dir)
    if not resume and any(output.records_dir.glob("*.json")):
        raise FileExistsError("selector output is non-empty; pass --resume to continue")
    budget = int(load_frozen_pool_dimensions(config)["verification_budget"])
    written = 0
    for record in labels.records():
        if output.has(record.request_id):
            if not resume:
                raise FileExistsError(output.path(record.request_id))
            continue
        result = replay_request(record, budget)
        written += int(output.write(record.request_id, result))
    return {
        "schema_version": "specrhythm.phase3c-stage-summary.v1",
        "stage": "selector-replay",
        "new_records": written,
        "completed_records": len(output.records()),
        "verification_budget": budget,
        "target_blind_selectors": list(TARGET_BLIND_SELECTORS),
        "oracle_selector": ORACLE_SELECTOR,
        "gpu_measurement": False,
        "reports_goodput_or_slo_or_speedup": False,
    }


def validate_selector_artifacts(*, labeled_dir: Path, selector_dir: Path) -> dict[str, Any]:
    labels = {record.request_id: record for record in labeled_store(labeled_dir).records()}
    selector_records = selector_store(selector_dir).records()
    errors = []
    if {str(record.get("request_id")) for record in selector_records} != set(labels):
        errors.append("selector and labeled request sets differ")
    for result in selector_records:
        request_id = str(result.get("request_id", ""))
        record = labels.get(request_id)
        if record is None:
            continue
        budget = result.get("verification_budget")
        rows = result.get("rows", [])
        if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
            errors.append(f"{request_id}: invalid verification budget")
            continue
        expected_pairs = {
            (ratio, selector) for ratio in ("1x", "2x", "4x") for selector in SELECTOR_ORDER
        }
        actual_pairs = {
            (row.get("pool_ratio"), row.get("selector"))
            for row in rows
            if isinstance(row, Mapping)
        }
        if actual_pairs != expected_pairs or len(rows) != len(expected_pairs):
            errors.append(f"{request_id}: selector row matrix is incomplete or duplicated")
            continue
        by_id = {node.runtime_features.stable_node_id: node for node in record.nodes}
        for row in rows:
            ratio = str(row["pool_ratio"])
            selected_ids = tuple(row.get("selected_node_ids", ()))
            selected = set(selected_ids)
            accounting = row.get("candidate_accounting", {})
            if len(selected) != len(selected_ids) or len(selected) > budget:
                errors.append(f"{request_id}/{ratio}: selected-node budget violation")
            if not selected.issubset(record.pool_node_ids[ratio]):
                errors.append(f"{request_id}/{ratio}: selected node is outside pool")
            for node_id in selected:
                if node_id not in by_id:
                    continue
                parent = by_id[node_id].runtime_features.parent_id
                if parent is not None and parent not in selected:
                    errors.append(f"{request_id}/{ratio}: selection is not prefix closed")
            if accounting.get("selected_verify_nodes") != len(selected):
                errors.append(f"{request_id}/{ratio}: verified-node accounting differs")
            accepted = accounting.get("accepted_candidate_tokens")
            if not isinstance(accepted, int) or not 0 <= accepted <= len(selected):
                errors.append(f"{request_id}/{ratio}: accepted-token accounting invalid")
            if accounting.get("committed_candidate_tokens") != accepted:
                errors.append(f"{request_id}/{ratio}: candidate commit accounting differs")
            if accounting.get("committed_target_root_tokens") not in {0, 1}:
                errors.append(f"{request_id}/{ratio}: target-root accounting invalid")
            if row.get("request_roots") != 1:
                errors.append(f"{request_id}/{ratio}: request-root accounting invalid")
            if (
                row.get("forest_sha256") != record.forest_sha256
                or row.get("target_trajectory_sha256") != record.target_trajectory_sha256
            ):
                errors.append(f"{request_id}/{ratio}: shared forest/outcome identity differs")
    return {
        "valid": not errors,
        "errors": errors,
        "request_count": len(selector_records),
        "prefix_closure_checked": True,
        "candidate_token_root_accounting_checked": True,
        "shared_forest_target_outcome_checked": True,
    }


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return sum(values) / len(values) if values else 0.0


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def stratified_bootstrap_ci(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    seed: int = 1664,
    iterations: int = 2000,
) -> list[float]:
    """Bootstrap requests, preserving each task stratum; never resample nodes."""

    return _stratified_bootstrap_cis(rows, (field,), seed=seed, iterations=iterations)[field]


def _stratified_bootstrap_cis(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    *,
    seed: int,
    iterations: int = 2000,
) -> dict[str, list[float]]:
    strata: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[str(row["task_class"])].append(row)
    if not strata:
        return {field: [0.0, 0.0] for field in fields}
    rng = random.Random(seed)
    samples = {field: [] for field in fields}
    for _ in range(iterations):
        sampled = [
            rng.choice(stratum)
            for task in sorted(strata)
            for stratum in (strata[task],)
            for _ in range(len(stratum))
        ]
        for field in fields:
            values = [float(row[field]) for row in sampled if row.get(field) is not None]
            samples[field].append(sum(values) / len(values) if values else 0.0)
    return {
        field: [_percentile(values, 0.025), _percentile(values, 0.975)]
        for field, values in samples.items()
    }


def _distribution(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    seed: int,
    bootstrap_ci: Optional[Sequence[float]] = None,
) -> dict[str, Any]:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "std": 0.0,
            "request_bootstrap_95_ci": [0.0, 0.0],
        }
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p25": _percentile(values, 0.25),
        "p75": _percentile(values, 0.75),
        "p90": _percentile(values, 0.90),
        "std": statistics.pstdev(values),
        "request_bootstrap_95_ci": (
            list(bootstrap_ci)
            if bootstrap_ci is not None
            else stratified_bootstrap_ci(rows, field, seed=seed)
        ),
    }


def _paired_values(
    rows: Sequence[Mapping[str, Any]],
    reference: Mapping[tuple[str, str], Mapping[str, Any]],
    field: str,
) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        other = reference[(str(row["request_id"]), str(row["pool_ratio"]))]
        result.append(
            {
                "task_class": row["task_class"],
                "value": float(row[field]) - float(other[field]),
            }
        )
    return result


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "target_path_pool_coverage",
        "selected_target_path_coverage",
        "accepted_draft_tokens_per_proposal",
        "committed_tokens_per_proposal",
        "accepted_per_verified",
        "oracle_regret",
        "oracle_gap_recovery",
        "selected_node_target_precision",
        "selected_node_target_recall",
        "first_error_depth",
        "prefix_closure_overhead",
        "search_nodes",
        "verified_nodes",
    )
    fields = fields + (
        "target_path_recall",
        "target_node_density",
        "selected_target_precision",
        "selected_target_recall",
        "oracle_gap_absolute",
        "oracle_gap_relative_to_accepted",
        "oracle_gap_relative_to_committed",
    )
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        for task in (str(row["task_class"]), "all"):
            groups[(task, str(row["pool_ratio"]), str(row["selector"]))].append(row)
    residual = {
        (str(row["request_id"]), str(row["pool_ratio"])): row
        for row in rows
        if row["selector"] == "residual-probability"
    }
    oracle = {
        (str(row["request_id"]), str(row["pool_ratio"])): row
        for row in rows
        if row["selector"] == ORACLE_SELECTOR
    }
    output = []
    for (task, ratio, selector), values in sorted(groups.items()):
        seed = 1664 + int(
            hashlib.sha256(f"{task}/{ratio}/{selector}".encode()).hexdigest()[:8], 16
        )
        accepted_field = "accepted_draft_tokens_per_proposal"
        wins = ties = losses = 0
        for row in values:
            reference = residual[(str(row["request_id"]), ratio)]
            delta = float(row[accepted_field]) - float(reference[accepted_field])
            wins += int(delta > 0)
            ties += int(delta == 0)
            losses += int(delta < 0)
        paired = _paired_values(values, oracle, accepted_field)
        paired_rows = [
            {"task_class": row["task_class"], "paired_delta": row["value"]} for row in paired
        ]
        bootstrap_cis = _stratified_bootstrap_cis(values, fields, seed=seed)
        distributions = {
            field: _distribution(
                values,
                field,
                seed=seed + index,
                bootstrap_ci=bootstrap_cis[field],
            )
            for index, field in enumerate(fields)
        }
        paired_distribution = _distribution(paired_rows, "paired_delta", seed=seed + len(fields))
        output.append(
            {
                "task_class": task,
                "pool_ratio": ratio,
                "selector": selector,
                "requests": len(values),
                **{field: _mean(values, field) for field in fields},
                "statistics": distributions,
                "win_tie_loss_vs_residual_probability": {
                    "win": wins,
                    "tie": ties,
                    "loss": losses,
                },
                "paired_delta_vs_within_request_oracle": paired_distribution,
            }
        )
    return output


def _fresh_rows(
    labels: Sequence[LabeledTraceRecord],
    selector_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Migrate v1 selector checkpoints without changing their selected node IDs."""

    by_request = {record.request_id: record for record in labels}
    rows = []
    for result in selector_records:
        record = by_request[str(result["request_id"])]
        migrated: dict[tuple[str, str], dict[str, Any]] = {}
        for old in result["rows"]:
            ratio = str(old["pool_ratio"])
            selector = str(old["selector"])
            selection = SelectionResult(
                tuple(str(value) for value in old["selected_node_ids"]),
                int(old.get("prefix_closure_overhead", 0)),
            )
            migrated[(ratio, selector)] = _evaluate_selection(record, ratio, selector, selection)
        for ratio in POOL_ORDER:
            oracle = migrated[(ratio, ORACLE_SELECTOR)]["accepted_draft_tokens_per_proposal"]
            baseline = migrated[(ratio, "residual-probability")][
                "accepted_draft_tokens_per_proposal"
            ]
            baseline_gap = oracle - baseline
            for selector in SELECTOR_ORDER:
                row = migrated[(ratio, selector)]
                accepted = row["accepted_draft_tokens_per_proposal"]
                committed = row["committed_tokens_per_proposal"]
                gap = oracle - accepted
                row.update(
                    {
                        "oracle_regret": gap,
                        "oracle_gap_absolute": gap,
                        "oracle_gap_relative_to_accepted": (gap / accepted if accepted else None),
                        "oracle_gap_relative_to_committed": (
                            gap / committed if committed else None
                        ),
                        "oracle_gap_recovery": (
                            (accepted - baseline) / baseline_gap
                            if baseline_gap > 0
                            else (1.0 if accepted == oracle else 0.0)
                        ),
                    }
                )
                rows.append(row)
    return rows


def _shell_decomposition(
    labels: Sequence[LabeledTraceRecord], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    selected = {
        (str(row["request_id"]), str(row["pool_ratio"]), str(row["selector"])): set(
            row["selected_node_ids"]
        )
        for row in rows
    }
    request_rows = []
    shell_specs = (("1x-base", "1x", None), ("2x-shell", "2x", "1x"), ("4x-shell", "4x", "2x"))
    for record in labels:
        by_id = {node.runtime_features.stable_node_id: node for node in record.nodes}
        full_target = set(record.target_path_node_ids_by_pool["4x"])
        maximum_target_depth = max(
            (by_id[node_id].runtime_features.depth for node_id in full_target), default=0
        )
        for shell_name, ratio, previous in shell_specs:
            cumulative = set(record.pool_node_ids[ratio])
            shell = cumulative - (set(record.pool_node_ids[previous]) if previous else set())
            target_shell = shell & full_target
            depths = sorted(by_id[node_id].runtime_features.depth for node_id in target_shell)
            reachable = sum(
                depth <= 4
                and all(
                    any(
                        node.runtime_features.depth == prefix_depth
                        and node.runtime_features.stable_node_id in cumulative
                        and node.target_only_labels["on_target_path"]
                        for node in record.nodes
                    )
                    for prefix_depth in range(1, depth + 1)
                )
                for depth in depths
            )
            request_rows.append(
                {
                    "request_id": record.request_id,
                    "task_class": record.task_class,
                    "shell": shell_name,
                    "pool_ratio": ratio,
                    "node_count": len(shell),
                    "on_target_node_count": len(target_shell),
                    "target_node_density": len(target_shell) / len(shell) if shell else 0.0,
                    "minimum_target_depth": min(depths) if depths else None,
                    "maximum_target_depth": max(depths) if depths else None,
                    "reachable_target_nodes_under_budget_4": reachable,
                    "prefix_enabling_nodes": sum(depth < maximum_target_depth for depth in depths),
                    "oracle_selected_shell_nodes": len(
                        selected[(record.request_id, ratio, ORACLE_SELECTOR)] & shell
                    ),
                    "target_blind_selected_shell_nodes": {
                        selector: len(selected[(record.request_id, ratio, selector)] & shell)
                        for selector in TARGET_BLIND_SELECTORS
                    },
                    "target_blind_selected_shell_target_hits": {
                        selector: len(
                            selected[(record.request_id, ratio, selector)] & target_shell
                        )
                        for selector in TARGET_BLIND_SELECTORS
                    },
                }
            )
    aggregate = []
    for shell_name in ("1x-base", "2x-shell", "4x-shell"):
        values = [row for row in request_rows if row["shell"] == shell_name]
        aggregate.append(
            {
                "shell": shell_name,
                "requests": len(values),
                "node_count": sum(row["node_count"] for row in values),
                "on_target_node_count": sum(row["on_target_node_count"] for row in values),
                "target_node_density": (
                    sum(row["on_target_node_count"] for row in values)
                    / max(1, sum(row["node_count"] for row in values))
                ),
                "requests_with_target_nodes": sum(
                    row["on_target_node_count"] > 0 for row in values
                ),
                "reachable_target_nodes_under_budget_4": sum(
                    row["reachable_target_nodes_under_budget_4"] for row in values
                ),
                "prefix_enabling_nodes": sum(row["prefix_enabling_nodes"] for row in values),
                "oracle_selected_shell_nodes": sum(
                    row["oracle_selected_shell_nodes"] for row in values
                ),
            }
        )
    return {"per_request": request_rows, "aggregate": aggregate}


def _selection_stability(
    labels: Sequence[LabeledTraceRecord], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    by_key = {
        (str(row["request_id"]), str(row["pool_ratio"]), str(row["selector"])): row for row in rows
    }
    request_rows = []
    requests = sorted({str(row["request_id"]) for row in rows})
    target_ids = {
        record.request_id: set(record.target_path_node_ids_by_pool["4x"]) for record in labels
    }
    pool_ids = {
        (record.request_id, ratio): set(record.pool_node_ids[ratio])
        for record in labels
        for ratio in POOL_ORDER
    }
    pairs = (("1x", "2x"), ("2x", "4x"), ("1x", "4x"))
    for request_id in requests:
        for selector in TARGET_BLIND_SELECTORS:
            for old_ratio, new_ratio in pairs:
                old = by_key[(request_id, old_ratio, selector)]
                new = by_key[(request_id, new_ratio, selector)]
                old_ids = set(old["selected_node_ids"])
                new_ids = set(new["selected_node_ids"])
                union = old_ids | new_ids
                added_pool_shell = (
                    pool_ids[(request_id, new_ratio)] - pool_ids[(request_id, old_ratio)]
                )
                shell = new_ids & added_pool_shell
                target_hits = len(shell & target_ids[request_id])
                request_rows.append(
                    {
                        "request_id": request_id,
                        "task_class": old["task_class"],
                        "selector": selector,
                        "from_pool": old_ratio,
                        "to_pool": new_ratio,
                        "exact_match": old_ids == new_ids,
                        "selected_set_jaccard": len(old_ids & new_ids) / len(union)
                        if union
                        else 1.0,
                        "selected_budget_displacement": len(old_ids - new_ids),
                        "new_shell_selection_count": len(shell),
                        "new_shell_target_hit_count": target_hits,
                        "accepted_outcome_equal": old["accepted_draft_tokens_per_proposal"]
                        == new["accepted_draft_tokens_per_proposal"],
                    }
                )
    aggregate = []
    for selector in TARGET_BLIND_SELECTORS:
        for old_ratio, new_ratio in pairs:
            values = [
                row
                for row in request_rows
                if row["selector"] == selector
                and row["from_pool"] == old_ratio
                and row["to_pool"] == new_ratio
            ]
            exact = sum(row["exact_match"] for row in values)
            outcome_equal = sum(row["accepted_outcome_equal"] for row in values)
            if exact == len(values):
                reason = "selected sets are identical request-by-request"
            elif outcome_equal == len(values):
                reason = (
                    "selected sets differ but accepted outcomes are identical request-by-request"
                )
            else:
                reason = (
                    "accepted outcomes change for at least one request; equal means may be "
                    "aggregate cancellation"
                )
            aggregate.append(
                {
                    "selector": selector,
                    "from_pool": old_ratio,
                    "to_pool": new_ratio,
                    "requests": len(values),
                    "selected_node_id_exact_match_ratio": exact / max(1, len(values)),
                    "selected_set_jaccard": statistics.fmean(
                        row["selected_set_jaccard"] for row in values
                    ),
                    "selected_budget_displacement": statistics.fmean(
                        row["selected_budget_displacement"] for row in values
                    ),
                    "new_shell_selection_count": sum(
                        row["new_shell_selection_count"] for row in values
                    ),
                    "new_shell_target_hit_count": sum(
                        row["new_shell_target_hit_count"] for row in values
                    ),
                    "accepted_outcome_exact_match_ratio": outcome_equal / max(1, len(values)),
                    "interpretation": reason,
                }
            )
    residual_pairs = [row for row in request_rows if row["selector"] == "residual-probability"]
    residual_exact_requests = sum(
        all(
            row["exact_match"]
            for row in residual_pairs
            if row["request_id"] == request_id
            and (row["from_pool"], row["to_pool"]) in {("1x", "2x"), ("2x", "4x")}
        )
        for request_id in requests
    )
    residual_equal_outcome_requests = sum(
        all(
            row["accepted_outcome_equal"]
            for row in residual_pairs
            if row["request_id"] == request_id
            and (row["from_pool"], row["to_pool"]) in {("1x", "2x"), ("2x", "4x")}
        )
        for request_id in requests
    )
    if residual_exact_requests == len(requests):
        residual_reason = "all three selected sets are identical request-by-request"
    elif residual_equal_outcome_requests == len(requests):
        residual_reason = (
            "selected sets change, but accepted outcomes match across all pools request-by-request"
        )
    else:
        residual_reason = (
            "at least one request changes accepted outcome; equal aggregate means may be "
            "cancellation"
        )
    return {
        "per_request": request_rows,
        "aggregate": aggregate,
        "residual_probability_three_pool_explanation": {
            "requests": len(requests),
            "identical_selected_sets": residual_exact_requests,
            "identical_accepted_outcomes": residual_equal_outcome_requests,
            "classification": residual_reason,
        },
    }


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*.json") if item.is_file())
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def headroom_decomposition(
    labels: Sequence[LabeledTraceRecord],
    rows: Sequence[Mapping[str, Any]],
    *,
    prefix_depth: int = 0,
) -> dict[str, Any]:
    """Decompose generator, selector, and budget limits on common snapshots."""

    by_key = {
        (str(row["request_id"]), str(row["pool_ratio"]), str(row["selector"])): row for row in rows
    }
    request_rows = []
    for record in labels:
        full_pool_accepted = {}
        for ratio in POOL_ORDER:
            depths = {
                node.runtime_features.depth
                for node in record.nodes
                if node.target_only_labels["on_target_path"]
                and node.runtime_features.stable_node_id in record.pool_node_ids[ratio]
            }
            accepted = 0
            for depth in range(1, record.target_trajectory.target_path_length + 1):
                if depth not in depths:
                    break
                accepted += 1
            full_pool_accepted[ratio] = accepted
        for ratio_index, ratio in enumerate(POOL_ORDER):
            previous = POOL_ORDER[ratio_index - 1] if ratio_index else None
            oracle = float(
                by_key[(record.request_id, ratio, ORACLE_SELECTOR)][
                    "accepted_draft_tokens_per_proposal"
                ]
            )
            previous_oracle = (
                float(
                    by_key[(record.request_id, previous, ORACLE_SELECTOR)][
                        "accepted_draft_tokens_per_proposal"
                    ]
                )
                if previous
                else None
            )
            generator_gain = oracle - previous_oracle if previous_oracle is not None else None
            budget_constraint = full_pool_accepted[ratio] - oracle
            for selector in TARGET_BLIND_SELECTORS:
                selected = float(
                    by_key[(record.request_id, ratio, selector)][
                        "accepted_draft_tokens_per_proposal"
                    ]
                )
                previous_selected = (
                    float(
                        by_key[(record.request_id, previous, selector)][
                            "accepted_draft_tokens_per_proposal"
                        ]
                    )
                    if previous
                    else None
                )
                selected_gain = (
                    selected - previous_selected if previous_selected is not None else None
                )
                identifiable = generator_gain is not None and generator_gain != 0
                utilization = selected_gain / generator_gain if identifiable else None
                if full_pool_accepted[ratio] == 0:
                    case = "pool does not contain useful target nodes"
                elif budget_constraint > 0:
                    case = "pool contains target nodes but budget cannot reach them"
                elif oracle > selected:
                    case = "budget can reach target nodes but selector cannot identify them"
                else:
                    case = "selector identifies target nodes and realizes available gain"
                request_rows.append(
                    {
                        "request_id": record.request_id,
                        "task_class": record.task_class,
                        "prefix_depth": prefix_depth,
                        "pool_ratio": ratio,
                        "selector": selector,
                        "generator_coverage_ceiling": generator_gain,
                        "selector_regret": oracle - selected,
                        "budget_constraint": budget_constraint,
                        "full_pool_target_path_accepted": full_pool_accepted[ratio],
                        "budget_4_oracle_accepted": oracle,
                        "pool_expansion_utilization": utilization,
                        "pool_expansion_utilization_identifiable": identifiable,
                        "case": case,
                    }
                )
    aggregate = []
    tasks = sorted({record.task_class for record in labels}) + ["all"]
    for task in tasks:
        for ratio in POOL_ORDER:
            for selector in TARGET_BLIND_SELECTORS:
                values = [
                    row
                    for row in request_rows
                    if (task == "all" or row["task_class"] == task)
                    and row["pool_ratio"] == ratio
                    and row["selector"] == selector
                ]
                identifiable_values = [
                    float(row["pool_expansion_utilization"])
                    for row in values
                    if row["pool_expansion_utilization_identifiable"]
                ]
                generator_values = [
                    float(row["generator_coverage_ceiling"])
                    for row in values
                    if row["generator_coverage_ceiling"] is not None
                ]
                aggregate.append(
                    {
                        "task_class": task,
                        "prefix_depth": prefix_depth,
                        "pool_ratio": ratio,
                        "selector": selector,
                        "requests": len(values),
                        "generator_coverage_ceiling": (
                            statistics.fmean(generator_values) if generator_values else None
                        ),
                        "selector_regret": statistics.fmean(
                            float(row["selector_regret"]) for row in values
                        ),
                        "budget_constraint": statistics.fmean(
                            float(row["budget_constraint"]) for row in values
                        ),
                        "pool_expansion_utilization": (
                            statistics.fmean(identifiable_values) if identifiable_values else None
                        ),
                        "pool_expansion_utilization_identifiable_requests": len(
                            identifiable_values
                        ),
                        "case_counts": {
                            case: sum(row["case"] == case for row in values)
                            for case in sorted({str(row["case"]) for row in values})
                        },
                    }
                )
    return {"per_request": request_rows, "aggregate": aggregate}


def _bin_name(value: float, edges: Sequence[float]) -> str:
    lower = 0.0
    for upper in edges:
        if value <= upper:
            return f"({lower:g},{upper:g}]"
        lower = upper
    return f"({lower:g},inf]"


def _calibration(
    labels: Sequence[LabeledTraceRecord], selector_records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    selected_by_key = {
        (row["request_id"], row["pool_ratio"], row["selector"]): set(row["selected_node_ids"])
        for record in selector_records
        for row in record["rows"]
    }
    probability: dict[tuple[int, str], list[int]] = defaultdict(lambda: [0, 0])
    entropy: dict[tuple[int, str], list[int]] = defaultdict(lambda: [0, 0])
    sibling: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    selection: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    depth_rows: dict[tuple[str, str, int, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    for record in labels:
        for ratio, ids in record.pool_node_ids.items():
            pool = set(ids)
            for node in record.nodes:
                runtime = node.runtime_features
                if runtime.stable_node_id not in pool:
                    continue
                hit = int(node.target_only_labels["on_target_path"])
                key = (
                    runtime.depth,
                    _bin_name(
                        runtime.path_probability,
                        (0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
                    ),
                )
                probability[key][0] += hit
                probability[key][1] += 1
                ekey = (
                    runtime.depth,
                    _bin_name(runtime.entropy, (1.0, 2.0, 4.0, 6.0, 8.0, 12.0)),
                )
                entropy[ekey][0] += hit
                entropy[ekey][1] += 1
                sibling[runtime.sibling_rank][0] += hit
                sibling[runtime.sibling_rank][1] += 1
                for selector in SELECTOR_ORDER:
                    selected = (
                        runtime.stable_node_id
                        in selected_by_key[(record.request_id, ratio, selector)]
                    )
                    status = "selected" if selected else "unselected"
                    selection[(ratio, selector, status)][0] += hit
                    selection[(ratio, selector, status)][1] += 1
                    depth_rows[(record.task_class, ratio, runtime.depth, selector)][0] += hit
                    depth_rows[(record.task_class, ratio, runtime.depth, selector)][1] += int(
                        selected and hit
                    )
                    depth_rows[(record.task_class, ratio, runtime.depth, selector)][2] += int(
                        selected
                    )

    def rates(values: Mapping[Any, list[int]], names: Callable[[Any], dict[str, Any]]):
        return [
            {
                **names(key),
                "target_hits": counts[0],
                "nodes": counts[1],
                "target_hit_rate": counts[0] / max(1, counts[1]),
            }
            for key, counts in sorted(values.items())
        ]

    return {
        "depth_probability_bins": rates(
            probability, lambda key: {"depth": key[0], "probability_bin": key[1]}
        ),
        "depth_entropy_bins": rates(entropy, lambda key: {"depth": key[0], "entropy_bin": key[1]}),
        "sibling_rank_target_hit_rate": rates(sibling, lambda key: {"sibling_rank": key}),
        "selected_vs_unselected": rates(
            selection,
            lambda key: {"pool_ratio": key[0], "selector": key[1], "status": key[2]},
        ),
        "depth_metrics": [
            {
                "task_class": key[0],
                "pool_ratio": key[1],
                "depth": key[2],
                "selector": key[3],
                "pool_target_nodes": counts[0],
                "selected_target_nodes": counts[1],
                "selected_nodes": counts[2],
                "selected_target_precision": counts[1] / max(1, counts[2]),
                "selected_target_recall": counts[1] / max(1, counts[0]),
            }
            for key, counts in sorted(depth_rows.items())
        ],
    }


def summarize_selector_diagnosis(
    *,
    labeled_dir: Path,
    selector_dir: Path,
    source_trace_commit: Optional[str] = None,
    workload_manifest_path: Optional[Path] = None,
    draft_dir: Optional[Path] = None,
    target_dir: Optional[Path] = None,
) -> dict[str, Any]:
    validation = validate_selector_artifacts(labeled_dir=labeled_dir, selector_dir=selector_dir)
    if not validation["valid"]:
        raise ValueError(f"invalid selector artifacts: {validation['errors'][:3]}")
    labels = labeled_store(labeled_dir).records()
    selector_records = selector_store(selector_dir).records()
    label_ids = {record.request_id for record in labels}
    selector_ids = {str(record["request_id"]) for record in selector_records}
    if label_ids != selector_ids:
        raise ValueError("selector and labeled request sets differ")
    rows = _fresh_rows(labels, selector_records)
    aggregate = _aggregate_rows(rows)
    by_selector_ratio = {
        (row["selector"], row["pool_ratio"]): row
        for row in aggregate
        if row["task_class"] == "all"
    }
    robustness = []
    for selector in SELECTOR_ORDER:
        base = by_selector_ratio[(selector, "1x")]
        for ratio in ("2x", "4x"):
            current = by_selector_ratio[(selector, ratio)]
            robustness.append(
                {
                    "selector": selector,
                    "from_pool": "1x",
                    "to_pool": ratio,
                    "accepted_tokens_delta": current["accepted_draft_tokens_per_proposal"]
                    - base["accepted_draft_tokens_per_proposal"],
                    "target_path_recall_delta": current["target_path_recall"]
                    - base["target_path_recall"],
                    "legacy_variable_depth_coverage_delta": current["target_path_pool_coverage"]
                    - base["target_path_pool_coverage"],
                    "oracle_regret_delta": current["oracle_regret"] - base["oracle_regret"],
                }
            )
    recall_errors = []
    for record in labels:
        request_rows = {
            str(row["pool_ratio"]): row
            for row in rows
            if row["request_id"] == record.request_id and row["selector"] == "residual-probability"
        }
        recalls = [request_rows[ratio]["target_path_recall"] for ratio in POOL_ORDER]
        if recalls != sorted(recalls):
            recall_errors.append(record.request_id)
    if recall_errors:
        raise ValueError(
            "nested target_path_recall is non-monotonic for "
            f"{recall_errors[:5]}"
        )
    shell = _shell_decomposition(labels, rows)
    stability = _selection_stability(labels, rows)
    headroom = headroom_decomposition(labels, rows)
    oracle_by_ratio = {
        ratio: by_selector_ratio[(ORACLE_SELECTOR, ratio)]["accepted_draft_tokens_per_proposal"]
        for ratio in POOL_ORDER
    }
    oracle_sources = {
        "1x_to_2x_accepted_delta": oracle_by_ratio["2x"] - oracle_by_ratio["1x"],
        "2x_to_4x_accepted_delta": oracle_by_ratio["4x"] - oracle_by_ratio["2x"],
        "shell_evidence": shell["aggregate"],
        "per_request": [
            {
                "request_id": record.request_id,
                "task_class": record.task_class,
                "1x_to_2x_accepted_delta": next(
                    row["accepted_draft_tokens_per_proposal"]
                    for row in rows
                    if row["request_id"] == record.request_id
                    and row["pool_ratio"] == "2x"
                    and row["selector"] == ORACLE_SELECTOR
                )
                - next(
                    row["accepted_draft_tokens_per_proposal"]
                    for row in rows
                    if row["request_id"] == record.request_id
                    and row["pool_ratio"] == "1x"
                    and row["selector"] == ORACLE_SELECTOR
                ),
                "2x_to_4x_accepted_delta": next(
                    row["accepted_draft_tokens_per_proposal"]
                    for row in rows
                    if row["request_id"] == record.request_id
                    and row["pool_ratio"] == "4x"
                    and row["selector"] == ORACLE_SELECTOR
                )
                - next(
                    row["accepted_draft_tokens_per_proposal"]
                    for row in rows
                    if row["request_id"] == record.request_id
                    and row["pool_ratio"] == "2x"
                    and row["selector"] == ORACLE_SELECTOR
                ),
            }
            for record in labels
        ],
        "interpretation_rule": (
            "A zero oracle delta means the new shell did not add a prefix-closed, "
            "budget-4 target continuation on these snapshots; it does not imply the "
            "shell contained no target-labeled nodes."
        ),
    }
    manifest = None
    if workload_manifest_path is not None:
        manifest = json.loads(workload_manifest_path.read_text(encoding="utf-8"))
    trace_hashes = {
        "labeled": _directory_sha256(labeled_dir),
        "selectors": _directory_sha256(selector_dir),
    }
    if draft_dir is not None:
        trace_hashes["draft"] = _directory_sha256(draft_dir)
    if target_dir is not None:
        trace_hashes["target"] = _directory_sha256(target_dir)
    trace_hashes["combined"] = hashlib.sha256(
        json.dumps(trace_hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_schema = manifest.get("schema_version") if manifest is not None else None
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "coverage_definition_version": COVERAGE_DEFINITION_VERSION,
        "source_trace_commit": source_trace_commit or "not-provided",
        "source_trace_sha256": trace_hashes,
        "source_workload_manifest": (
            {
                "file": workload_manifest_path.name,
                "sha256": hashlib.sha256(workload_manifest_path.read_bytes()).hexdigest(),
                "prompt_rendering_audit": manifest.get("prompt_rendering_audit"),
                "chat_trace_compatibility": manifest.get("chat_trace_compatibility")
                or (
                    "legacy v1 workload: ShareGPT used an untemplated first user turn; "
                    "retain only as legacy diagnostics and do not mix with corrected traces"
                    if manifest_schema == "specrhythm.r3-real-workload-manifest.v1"
                    else "prompt audit unavailable"
                ),
            }
            if workload_manifest_path is not None and manifest is not None
            else None
        ),
        "evidence_scope": (f"{len(labels)}-request-coverage-and-selector-signal-only"),
        "gpu_performance_result": False,
        "reports_goodput": False,
        "reports_slo_attainment": False,
        "reports_speedup": False,
        "request_count": len(labels),
        "validation": validation,
        "request_level_splits": dict(
            sorted(
                (
                    split,
                    sum(record.data_split == split for record in labels),
                )
                for split in {record.data_split for record in labels}
            )
        ),
        "selector_order": list(SELECTOR_ORDER),
        "target_blind_selectors": list(TARGET_BLIND_SELECTORS),
        "oracle_selector": ORACLE_SELECTOR,
        "entropy_margin_formula": (
            "log_path_probability + 0.25*top1_top2_margin - 0.10*(cumulative_entropy/depth)"
        ),
        "per_request_rows": rows,
        "aggregate_metrics": aggregate,
        "calibration": _calibration(labels, selector_records),
        "pool_expansion_robustness": robustness,
        "coverage_semantics_audit": {
            "legacy_target_path_pool_coverage_definition": (
                "target nodes present divided by min(target length, each pool's own "
                "maximum realized depth); retained exactly for v1 migration and not monotonic"
            ),
            "target_path_recall_definition": (
                "target nodes present divided by min(target length, shared 4x forest depth)"
            ),
            "target_node_density_definition": "target nodes present divided by pool nodes",
            "nested_target_path_recall_monotonic": not recall_errors,
            "non_monotonic_request_ids": recall_errors,
            "requests_missing_full_1x_target_trajectory": sum(
                not next(
                    row["full_target_trajectory_covered"]
                    for row in rows
                    if row["request_id"] == record.request_id
                    and row["pool_ratio"] == "1x"
                    and row["selector"] == "residual-probability"
                )
                for record in labels
            ),
            "requests_missing_1x_within_k4": sum(
                next(
                    row["first_missing_within_verification_horizon"] is not None
                    for row in rows
                    if row["request_id"] == record.request_id
                    and row["pool_ratio"] == "1x"
                    and row["selector"] == "residual-probability"
                )
                for record in labels
            ),
            "interpretation": (
                "Missing the full 1x target trajectory includes target depths beyond the "
                "first verification horizon and therefore is not a first-round failure rate."
            ),
        },
        "pool_shell_decomposition": shell,
        "oracle_pool_expansion_sources": oracle_sources,
        "selection_set_stability": stability,
        "headroom_decomposition": headroom,
    }


def diagnosis_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 3C.2 coverage and selector diagnosis",
        "",
        (
            "Pilot schema/signal evidence only. No goodput, SLO attainment, GPU speedup, "
            "or serving-engine claim."
        ),
        "",
        (
            "| Task | Pool | Selector | Target recall | Target density | Selected precision | "
            "Selected recall | Accepted mean [95% CI] | Paired delta vs oracle | "
            "W/T/L vs residual |"
        ),
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["aggregate_metrics"]:
        accepted = row["statistics"]["accepted_draft_tokens_per_proposal"]
        paired = row["paired_delta_vs_within_request_oracle"]
        wtl = row["win_tie_loss_vs_residual_probability"]
        lines.append(
            "| {task} | {pool} | {selector} | {recall:.4f} | {density:.4f} | "
            "{precision:.4f} | {selected_recall:.4f} | {mean:.4f} [{low:.4f}, {high:.4f}] | "
            "{paired:.4f} | {win}/{tie}/{loss} |".format(
                task=row["task_class"],
                pool=row["pool_ratio"],
                selector=row["selector"],
                recall=row["target_path_recall"],
                density=row["target_node_density"],
                precision=row["selected_target_precision"],
                selected_recall=row["selected_target_recall"],
                mean=accepted["mean"],
                low=accepted["request_bootstrap_95_ci"][0],
                high=accepted["request_bootstrap_95_ci"][1],
                paired=paired["mean"],
                win=wtl["win"],
                tie=wtl["tie"],
                loss=wtl["loss"],
            )
        )
    audit = report["coverage_semantics_audit"]
    lines.extend(
        [
            "",
            "## Coverage semantics",
            "",
            audit["interpretation"],
            "",
            (
                f"Nested target-path recall monotonic: "
                f"{audit['nested_target_path_recall_monotonic']}. Full-trajectory missing at "
                f"1x: {audit['requests_missing_full_1x_target_trajectory']}; missing within "
                f"K=4: {audit['requests_missing_1x_within_k4']}."
            ),
            "",
            "## Pool expansion source",
            "",
            (
                "Oracle accepted-token delta: 1x→2x = "
                f"{report['oracle_pool_expansion_sources']['1x_to_2x_accepted_delta']:.4f}; "
                "2x→4x = "
                f"{report['oracle_pool_expansion_sources']['2x_to_4x_accepted_delta']:.4f}."
            ),
            "",
            "| Shell | Nodes | Target nodes | Density | Requests with targets | "
            "Budget-4 reachable targets | Oracle-selected shell nodes |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["pool_shell_decomposition"]["aggregate"]:
        lines.append(
            f"| {row['shell']} | {row['node_count']} | {row['on_target_node_count']} | "
            f"{row['target_node_density']:.4f} | {row['requests_with_target_nodes']} | "
            f"{row['reachable_target_nodes_under_budget_4']} | "
            f"{row['oracle_selected_shell_nodes']} |"
        )
    residual = report["selection_set_stability"][
        "residual_probability_three_pool_explanation"
    ]
    lines.extend(
        [
            "",
            "## Residual-Probability stability",
            "",
            (
                f"{residual['classification']}. Identical selected sets: "
                f"{residual['identical_selected_sets']}/{residual['requests']}; identical "
                f"accepted outcomes: {residual['identical_accepted_outcomes']}/"
                f"{residual['requests']}."
            ),
            "",
            (
                "Pool-expansion utilization is null/not identifiable whenever the paired "
                "oracle expansion gain is zero; it is never imputed as zero."
            ),
        ]
    )
    return "\n".join(lines) + "\n"
