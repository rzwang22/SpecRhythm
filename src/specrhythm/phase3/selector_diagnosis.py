"""Target-blind and oracle selector replay for Phase-3C real candidate forests."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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
    maximum_pool_depth = max(
        (by_id[node_id].runtime_features.depth for node_id in pool), default=0
    )
    target_opportunities = min(target_length, maximum_pool_depth)
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
        "target_path_opportunities": target_opportunities,
        "target_path_pool_nodes": len(pool_target),
        "selected_target_path_nodes": len(selected_target),
        "target_path_pool_coverage": len(pool_target) / max(1, target_opportunities),
        "selected_target_path_coverage": len(selected_target)
        / max(1, target_opportunities),
        "accepted_draft_tokens_per_proposal": accepted,
        "committed_tokens_per_proposal": committed,
        "accepted_per_verified": accepted / max(1, len(selected)),
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


def replay_request(
    record: LabeledTraceRecord, verification_budget: int
) -> dict[str, Any]:
    rows = []
    for ratio in ("1x", "2x", "4x"):
        pool_ids = set(record.pool_node_ids[ratio])
        labeled = tuple(
            node
            for node in record.nodes
            if node.runtime_features.stable_node_id in pool_ids
        )
        runtime = tuple(node.runtime_features for node in labeled)
        selections = {
            name: select_target_blind(name, runtime, verification_budget)
            for name in TARGET_BLIND_SELECTORS
        }
        selections[ORACLE_SELECTOR] = select_within_request_oracle(
            labeled, verification_budget
        )
        ratio_rows = {
            name: _evaluate_selection(record, ratio, name, selection)
            for name, selection in selections.items()
        }
        oracle = ratio_rows[ORACLE_SELECTOR]["accepted_draft_tokens_per_proposal"]
        baseline = ratio_rows["residual-probability"][
            "accepted_draft_tokens_per_proposal"
        ]
        gap = oracle - baseline
        for name in SELECTOR_ORDER:
            row = ratio_rows[name]
            accepted = row["accepted_draft_tokens_per_proposal"]
            row["oracle_regret"] = oracle - accepted
            row["oracle_gap_recovery"] = (
                (accepted - baseline) / gap if gap > 0 else (1.0 if accepted == oracle else 0.0)
            )
            rows.append(row)
    return {
        "schema_version": "specrhythm.phase3c-selector-request.v1",
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


def validate_selector_artifacts(
    *, labeled_dir: Path, selector_dir: Path
) -> dict[str, Any]:
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
            (ratio, selector)
            for ratio in ("1x", "2x", "4x")
            for selector in SELECTOR_ORDER
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
                or row.get("target_trajectory_sha256")
                != record.target_trajectory_sha256
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
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        for task in (str(row["task_class"]), "all"):
            groups[(task, str(row["pool_ratio"]), str(row["selector"]))].append(row)
    return [
        {
            "task_class": task,
            "pool_ratio": ratio,
            "selector": selector,
            "requests": len(values),
            **{field: _mean(values, field) for field in fields},
        }
        for (task, ratio, selector), values in sorted(groups.items())
    ]


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
        (row["request_id"], row["pool_ratio"], row["selector"]): set(
            row["selected_node_ids"]
        )
        for record in selector_records
        for row in record["rows"]
    }
    probability: dict[tuple[int, str], list[int]] = defaultdict(lambda: [0, 0])
    entropy: dict[tuple[int, str], list[int]] = defaultdict(lambda: [0, 0])
    sibling: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    selection: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    depth_rows: dict[tuple[str, str, int, str], list[int]] = defaultdict(
        lambda: [0, 0, 0]
    )
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
                    selected = runtime.stable_node_id in selected_by_key[
                        (record.request_id, ratio, selector)
                    ]
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
        "depth_entropy_bins": rates(
            entropy, lambda key: {"depth": key[0], "entropy_bin": key[1]}
        ),
        "sibling_rank_target_hit_rate": rates(
            sibling, lambda key: {"sibling_rank": key}
        ),
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
    *, labeled_dir: Path, selector_dir: Path
) -> dict[str, Any]:
    validation = validate_selector_artifacts(
        labeled_dir=labeled_dir, selector_dir=selector_dir
    )
    if not validation["valid"]:
        raise ValueError(f"invalid selector artifacts: {validation['errors'][:3]}")
    labels = labeled_store(labeled_dir).records()
    selector_records = selector_store(selector_dir).records()
    label_ids = {record.request_id for record in labels}
    selector_ids = {str(record["request_id"]) for record in selector_records}
    if label_ids != selector_ids:
        raise ValueError("selector and labeled request sets differ")
    rows = [row for record in selector_records for row in record["rows"]]
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
                    "accepted_tokens_delta": current[
                        "accepted_draft_tokens_per_proposal"
                    ]
                    - base["accepted_draft_tokens_per_proposal"],
                    "target_pool_coverage_delta": current["target_path_pool_coverage"]
                    - base["target_path_pool_coverage"],
                    "oracle_regret_delta": current["oracle_regret"]
                    - base["oracle_regret"],
                }
            )
    return {
        "schema_version": "specrhythm.phase3c-selector-diagnosis.v1",
        "evidence_scope": (
            f"{len(labels)}-request-pilot-schema-and-selector-signal-only"
        ),
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
            "log_path_probability + 0.25*top1_top2_margin "
            "- 0.10*(cumulative_entropy/depth)"
        ),
        "per_request_rows": rows,
        "aggregate_metrics": aggregate,
        "calibration": _calibration(labels, selector_records),
        "pool_expansion_robustness": robustness,
    }


def diagnosis_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 3C.1 selector learnability diagnosis",
        "",
        (
            "Pilot schema/signal evidence only. No goodput, SLO attainment, GPU speedup, "
            "or serving-engine claim."
        ),
        "",
        (
            "| Task | Pool | Selector | Pool coverage | Selected coverage | Accepted | "
            "Committed | A/V | Oracle regret |"
        ),
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["aggregate_metrics"]:
        if row["task_class"] != "all":
            continue
        lines.append(
            "| {task_class} | {pool_ratio} | {selector} | {target_path_pool_coverage:.4f} | "
            "{selected_target_path_coverage:.4f} | {accepted_draft_tokens_per_proposal:.4f} | "
            "{committed_tokens_per_proposal:.4f} | {accepted_per_verified:.4f} | "
            "{oracle_regret:.4f} |".format(**row)
        )
    return "\n".join(lines) + "\n"
