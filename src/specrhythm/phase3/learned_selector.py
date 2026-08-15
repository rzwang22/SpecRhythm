"""Phase-3C.3 shell diagnostics and a leakage-safe learned selector pilot."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from specrhythm.phase3.multiround import (
    _canonical_sha,
    _directory_sha,
    sequential_store,
    snapshot_store,
)
from specrhythm.phase3.r3_workload import R3RealRequest
from specrhythm.phase3.real_candidate_trace import (
    ImmutableRequestStore,
    LabeledTraceRecord,
    RuntimeCandidateNode,
    TargetFeatureLeakageError,
    target_store,
)
from specrhythm.phase3.selector_diagnosis import (
    ORACLE_SELECTOR,
    SelectionResult,
    _evaluate_selection,
    select_target_blind,
    select_within_request_oracle,
    stratified_bootstrap_ci,
)
from specrhythm.workload import apportion_counts

LEARNED_SELECTOR = "learned-shell-ranker"
MODEL_SCHEMA = "specrhythm.phase3c-learned-shell-ranker.v1"
FEATURE_DATASET_SCHEMA = "specrhythm.phase3c-runtime-feature-row.v1"
LEARNED_REPLAY_SCHEMA = "specrhythm.phase3c-learned-selector-request.v1"
FEATURE_NAMES = (
    "local_probability",
    "log_local_probability",
    "path_probability",
    "log_path_probability",
    "depth",
    "sibling_rank",
    "parent_probability",
    "cumulative_entropy",
    "local_entropy",
    "top1_top2_margin",
    "branching_factor",
    "remaining_output_length",
    "round_index",
    "is_shell",
    "task_code",
    "task_chat",
    "task_summarization",
)


def _payload_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def stratified_request_splits(
    requests: Iterable[R3RealRequest], *, seed: int = 1664
) -> dict[str, str]:
    """Return exact task-stratified, request-level 70/15/15 splits."""

    by_task: dict[str, list[R3RealRequest]] = defaultdict(list)
    for request in requests:
        by_task[request.task_class].append(request)
    splits = {}
    names = ("train", "validation", "test")
    for task in sorted(by_task):
        ranked = sorted(
            by_task[task],
            key=lambda request: (
                hashlib.sha256(
                    f"{seed}\0{task}\0{request.request_id}".encode()
                ).hexdigest(),
                request.request_id,
            ),
        )
        counts = apportion_counts((0.70, 0.15, 0.15), len(ranked))
        cursor = 0
        for name, count in zip(names, counts):
            for request in ranked[cursor : cursor + count]:
                splits[request.request_id] = name
            cursor += count
    return splits


def _branching_factors(
    nodes: Sequence[RuntimeCandidateNode],
) -> dict[str, int]:
    result = {node.stable_node_id: 0 for node in nodes}
    for node in nodes:
        if node.parent_id is not None:
            result[node.parent_id] = result.get(node.parent_id, 0) + 1
    return result


def _runtime_feature_map(
    node: RuntimeCandidateNode,
    *,
    branching_factor: int,
    remaining_output_length: int,
    round_index: int,
    task_class: str,
    is_shell: bool,
) -> dict[str, float]:
    return {
        "local_probability": node.local_probability,
        "log_local_probability": node.log_local_probability,
        "path_probability": node.path_probability,
        "log_path_probability": node.log_path_probability,
        "depth": float(node.depth),
        "sibling_rank": float(node.sibling_rank),
        "parent_probability": node.parent_probability,
        "cumulative_entropy": node.cumulative_entropy,
        "local_entropy": node.entropy,
        "top1_top2_margin": node.top1_top2_margin,
        "branching_factor": float(branching_factor),
        # This is the configured generation budget remaining, not target EOS distance.
        "remaining_output_length": float(remaining_output_length),
        "round_index": float(round_index),
        "is_shell": float(is_shell),
        "task_code": float(task_class == "code"),
        "task_chat": float(task_class == "chat"),
        "task_summarization": float(task_class == "summarization"),
    }


def _pool_nodes(
    labeled: LabeledTraceRecord, ratio: str
) -> tuple[Any, ...]:
    pool = set(labeled.pool_node_ids[ratio])
    return tuple(
        node for node in labeled.nodes if node.runtime_features.stable_node_id in pool
    )


def _selections_for_snapshot(
    labeled: LabeledTraceRecord, budget: int
) -> dict[tuple[str, str], SelectionResult]:
    result = {}
    for ratio in ("1x", "2x"):
        nodes = _pool_nodes(labeled, ratio)
        runtime = tuple(node.runtime_features for node in nodes)
        for selector in ("residual-probability", "entropy-margin-heuristic"):
            result[(ratio, selector)] = select_target_blind(
                selector, runtime, budget
            )
        result[(ratio, ORACLE_SELECTOR)] = select_within_request_oracle(
            nodes, budget
        )
    return result


def shell_opportunity_decomposition(
    snapshots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = []
    for snapshot in snapshots:
        labeled = LabeledTraceRecord.from_dict(snapshot["labeled_trace"])
        budget = int(snapshot["verification_budget"])
        if budget != 4:
            raise ValueError("Phase-3C.3 shell decomposition requires frozen budget=4")
        base = set(labeled.pool_node_ids["1x"])
        pool2 = set(labeled.pool_node_ids["2x"])
        shell = pool2 - base
        by_id = {
            node.runtime_features.stable_node_id: node for node in labeled.nodes
        }
        target = {
            node.runtime_features.stable_node_id
            for node in labeled.nodes
            if node.target_only_labels["on_target_path"]
        }
        base_reachable = {
            node_id for node_id in base if by_id[node_id].runtime_features.depth <= budget
        }
        shell_reachable = {
            node_id for node_id in shell if by_id[node_id].runtime_features.depth <= budget
        }
        selections = _selections_for_snapshot(labeled, budget)
        evaluations = {
            key: _evaluate_selection(labeled, key[0], key[1], selection)
            for key, selection in selections.items()
        }
        oracle1 = evaluations[("1x", ORACLE_SELECTOR)][
            "accepted_draft_tokens_per_proposal"
        ]
        oracle2 = evaluations[("2x", ORACLE_SELECTOR)][
            "accepted_draft_tokens_per_proposal"
        ]
        oracle_selected = set(
            selections[("2x", ORACLE_SELECTOR)].selected_node_ids
        )
        shell_target = shell & target
        reachable_shell_target = shell_reachable & target
        row = {
            "snapshot_id": (
                f"{snapshot['request_id']}@prefix-{int(snapshot['prefix_position']):02d}"
            ),
            "request_id": snapshot["request_id"],
            "task_class": snapshot["task_class"],
            "round_index": int(snapshot["prefix_position"]),
            "round_index_semantics": "common target-prefix snapshot ordinal",
            "prefix_length": int(snapshot["prefix_position"]),
            "remaining_output_length": int(snapshot["remaining_target_tokens"]),
            "base_node_count": len(base),
            "shell_node_count": len(shell),
            "base_target_path_node_count": len(base & target),
            "shell_target_path_node_count": len(shell_target),
            "base_nodes_reachable_under_budget_4": len(base_reachable),
            "shell_nodes_reachable_under_budget_4": len(shell_reachable),
            "reachable_shell_target_node_count": len(reachable_shell_target),
            "reachable_shell_target_node_ids": sorted(reachable_shell_target),
            "selected_nodes": {
                selector: {
                    "base": len(
                        set(selections[("2x", selector)].selected_node_ids) & base
                    ),
                    "shell": len(
                        set(selections[("2x", selector)].selected_node_ids) & shell
                    ),
                    "reachable_shell_target": len(
                        set(selections[("2x", selector)].selected_node_ids)
                        & reachable_shell_target
                    ),
                }
                for selector in (
                    "residual-probability",
                    "entropy-margin-heuristic",
                    ORACLE_SELECTOR,
                )
            },
            "accepted_progress": {
                selector: evaluations[("2x", selector)][
                    "accepted_draft_tokens_per_proposal"
                ]
                for selector in (
                    "residual-probability",
                    "entropy-margin-heuristic",
                    ORACLE_SELECTOR,
                )
            },
            "accepted_gain_2x_minus_1x": {
                selector: evaluations[("2x", selector)][
                    "accepted_draft_tokens_per_proposal"
                ]
                - evaluations[("1x", selector)][
                    "accepted_draft_tokens_per_proposal"
                ]
                for selector in (
                    "residual-probability",
                    "entropy-margin-heuristic",
                    ORACLE_SELECTOR,
                )
            },
            "shell_target_present": bool(shell_target),
            "shell_target_budget_reachable": bool(reachable_shell_target),
            "oracle_uses_shell": bool(oracle_selected & shell),
            "oracle_shell_gain_positive": oracle2 > oracle1,
        }
        rows.append(row)

    aggregate = []
    tasks = sorted({str(row["task_class"]) for row in rows}) + ["all"]
    conditions = (
        "shell_target_present",
        "shell_target_budget_reachable",
        "oracle_uses_shell",
    )
    for task in tasks:
        values = [row for row in rows if task == "all" or row["task_class"] == task]
        reachable_targets = sum(
            row["reachable_shell_target_node_count"] for row in values
        )
        aggregate.append(
            {
                "task_class": task,
                "snapshots": len(values),
                **{
                    f"{condition}_snapshot_ratio": sum(
                        bool(row[condition]) for row in values
                    )
                    / max(1, len(values))
                    for condition in conditions + ("oracle_shell_gain_positive",)
                },
                "residual_shell_selection_recall": sum(
                    row["selected_nodes"]["residual-probability"][
                        "reachable_shell_target"
                    ]
                    for row in values
                )
                / max(1, reachable_targets),
                "entropy_margin_shell_selection_recall": sum(
                    row["selected_nodes"]["entropy-margin-heuristic"][
                        "reachable_shell_target"
                    ]
                    for row in values
                )
                / max(1, reachable_targets),
                "reachable_shell_target_nodes": reachable_targets,
                "conditioned_mean_accepted_gain": {
                    condition: {
                        selector: (
                            statistics.fmean(
                                row["accepted_gain_2x_minus_1x"][selector]
                                for row in values
                                if row[condition]
                            )
                            if any(row[condition] for row in values)
                            else None
                        )
                        for selector in (
                            "residual-probability",
                            "entropy-margin-heuristic",
                            ORACLE_SELECTOR,
                        )
                    }
                    for condition in conditions
                },
            }
        )
    return {
        "schema_version": "specrhythm.phase3c-shell-opportunity.v1",
        "pool": "2x-minus-1x",
        "verification_budget": 4,
        "per_snapshot": rows,
        "aggregate": aggregate,
        "separates": (
            "pool coverage headroom, budget/prefix-closure reachability, "
            "and selector ranking failure"
        ),
    }


def build_runtime_feature_rows(
    snapshots: Sequence[Mapping[str, Any]],
    requests: Mapping[str, R3RealRequest],
    splits: Mapping[str, str],
) -> list[dict[str, Any]]:
    rows = []
    for snapshot in snapshots:
        request_id = str(snapshot["request_id"])
        request = requests[request_id]
        labeled = LabeledTraceRecord.from_dict(snapshot["labeled_trace"])
        budget = int(snapshot["verification_budget"])
        if budget != 4:
            raise ValueError("Phase-3C.3 feature rows require frozen budget=4")
        base = set(labeled.pool_node_ids["1x"])
        pool2 = set(labeled.pool_node_ids["2x"])
        shell = pool2 - base
        nodes = tuple(
            node.runtime_features
            for node in labeled.nodes
            if node.runtime_features.stable_node_id in pool2
        )
        factors = _branching_factors(nodes)
        oracle1 = select_within_request_oracle(_pool_nodes(labeled, "1x"), budget)
        oracle2 = select_within_request_oracle(_pool_nodes(labeled, "2x"), budget)
        eval1 = _evaluate_selection(labeled, "1x", ORACLE_SELECTOR, oracle1)
        eval2 = _evaluate_selection(labeled, "2x", ORACLE_SELECTOR, oracle2)
        oracle_selected = set(oracle2.selected_node_ids)
        labels = {
            node.runtime_features.stable_node_id: node.target_only_labels
            for node in labeled.nodes
        }
        remaining_budget = max(
            1, request.maximum_new_tokens - int(snapshot["prefix_position"])
        )
        for node in nodes:
            if node.depth > budget:
                continue
            is_shell = node.stable_node_id in shell
            target = bool(labels[node.stable_node_id]["on_target_path"])
            rows.append(
                {
                    "schema_version": FEATURE_DATASET_SCHEMA,
                    "request_id": request_id,
                    "task_class": request.task_class,
                    "request_split": splits[request_id],
                    "snapshot_id": (
                        f"{request_id}@prefix-"
                        f"{int(snapshot['prefix_position']):02d}"
                    ),
                    "stable_node_id": node.stable_node_id,
                    "runtime_features": _runtime_feature_map(
                        node,
                        branching_factor=factors[node.stable_node_id],
                        remaining_output_length=remaining_budget,
                        round_index=int(snapshot["prefix_position"]),
                        task_class=request.task_class,
                        is_shell=is_shell,
                    ),
                    "target_only_labels": {
                        "node_on_target_path": target,
                        "node_selected_by_oracle": node.stable_node_id
                        in oracle_selected,
                        "node_contributes_to_oracle_shell_gain": (
                            target
                            and is_shell
                            and node.stable_node_id in oracle_selected
                            and eval2["accepted_draft_tokens_per_proposal"]
                            > eval1["accepted_draft_tokens_per_proposal"]
                        ),
                    },
                    "scope_flags": {
                        "all_reachable_nodes": True,
                        "2x_shell_reachable_nodes": is_shell,
                        "oracle_shell_opportunity_snapshot": (
                            eval2["accepted_draft_tokens_per_proposal"]
                            > eval1["accepted_draft_tokens_per_proposal"]
                        ),
                    },
                }
            )
    return rows


@dataclass(frozen=True)
class LearnedShellModel:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    metadata: dict[str, Any]

    def score(self, features: Mapping[str, float]) -> float:
        values = [float(features[name]) for name in self.feature_names]
        return self.intercept + sum(
            coefficient * ((value - mean) / scale)
            for value, mean, scale, coefficient in zip(
                values, self.means, self.scales, self.coefficients
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MODEL_SCHEMA,
            "selector": LEARNED_SELECTOR,
            "model_type": "deterministic-linear-logistic",
            "runtime_feature_names": list(self.feature_names),
            "means": list(self.means),
            "scales": list(self.scales),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "target_labels_available_at_inference": False,
            **self.metadata,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LearnedShellModel":
        if value.get("schema_version") != MODEL_SCHEMA:
            raise ValueError("unsupported learned-shell model schema")
        if value.get("target_labels_available_at_inference") is not False:
            raise ValueError("learned-shell model permits target-label leakage")
        names = tuple(value["runtime_feature_names"])
        if names != FEATURE_NAMES:
            raise ValueError("learned-shell runtime feature list changed")
        excluded = {
            "schema_version",
            "selector",
            "model_type",
            "runtime_feature_names",
            "means",
            "scales",
            "coefficients",
            "intercept",
            "target_labels_available_at_inference",
        }
        return cls(
            names,
            tuple(float(item) for item in value["means"]),
            tuple(float(item) for item in value["scales"]),
            tuple(float(item) for item in value["coefficients"]),
            float(value["intercept"]),
            {key: item for key, item in value.items() if key not in excluded},
        )


def train_learned_shell_ranker(
    feature_rows: Sequence[Mapping[str, Any]],
    *,
    split_hash: str,
    iterations: int = 400,
    learning_rate: float = 0.10,
    l2: float = 0.001,
) -> LearnedShellModel:
    train = [row for row in feature_rows if row["request_split"] == "train"]
    if not train:
        raise ValueError("learned-shell training split is empty")
    matrix = [
        [float(row["runtime_features"][name]) for name in FEATURE_NAMES]
        for row in train
    ]
    labels = [float(row["target_only_labels"]["node_on_target_path"]) for row in train]
    positives = sum(labels)
    if positives == 0 or positives == len(labels):
        raise ValueError("learned-shell training requires positive and negative nodes")
    means = tuple(
        statistics.fmean(row[index] for row in matrix)
        for index in range(len(FEATURE_NAMES))
    )
    scales = tuple(
        max(
            1e-12,
            math.sqrt(
                statistics.fmean(
                    (row[index] - means[index]) ** 2 for row in matrix
                )
            ),
        )
        for index in range(len(FEATURE_NAMES))
    )
    standardized = [
        [(value - means[index]) / scales[index] for index, value in enumerate(row)]
        for row in matrix
    ]
    positive_weight = len(labels) / (2.0 * positives)
    negative_weight = len(labels) / (2.0 * (len(labels) - positives))
    coefficients = [0.0] * len(FEATURE_NAMES)
    intercept = math.log(positives / (len(labels) - positives))
    for _ in range(iterations):
        gradients = [0.0] * len(FEATURE_NAMES)
        intercept_gradient = 0.0
        for row, label in zip(standardized, labels):
            logit = intercept + sum(
                coefficient * value
                for coefficient, value in zip(coefficients, row)
            )
            probability = (
                1.0 / (1.0 + math.exp(-logit))
                if logit >= 0
                else math.exp(logit) / (1.0 + math.exp(logit))
            )
            weight = positive_weight if label else negative_weight
            error = weight * (probability - label)
            intercept_gradient += error
            for index, value in enumerate(row):
                gradients[index] += error * value
        intercept -= learning_rate * intercept_gradient / len(labels)
        for index in range(len(coefficients)):
            gradient = gradients[index] / len(labels) + l2 * coefficients[index]
            coefficients[index] -= learning_rate * gradient
    training_rows = [
        {
            "request_id": row["request_id"],
            "snapshot_id": row["snapshot_id"],
            "stable_node_id": row["stable_node_id"],
            "runtime_features": row["runtime_features"],
            "label": row["target_only_labels"]["node_on_target_path"],
        }
        for row in train
    ]
    return LearnedShellModel(
        FEATURE_NAMES,
        means,
        scales,
        tuple(coefficients),
        intercept,
        {
            "training_iterations": iterations,
            "learning_rate": learning_rate,
            "l2": l2,
            "class_weighting": "balanced-from-training-split",
            "training_nodes": len(train),
            "training_positive_nodes": int(positives),
            "training_set_sha256": _payload_sha(training_rows),
            "request_split_sha256": split_hash,
            "validation_used_for_tuning": False,
            "test_used_for_tuning": False,
        },
    )


def select_learned_shell_ranker(
    nodes: Sequence[RuntimeCandidateNode],
    budget: int,
    model: LearnedShellModel,
    *,
    base_node_ids: set[str],
    task_class: str,
    round_index: int,
    remaining_output_length: int,
) -> SelectionResult:
    if any(not isinstance(node, RuntimeCandidateNode) for node in nodes):
        raise TargetFeatureLeakageError(
            "learned selector accepts RuntimeCandidateNode only; target labels are forbidden"
        )
    factors = _branching_factors(nodes)
    scores = {
        node.stable_node_id: model.score(
            _runtime_feature_map(
                node,
                branching_factor=factors[node.stable_node_id],
                remaining_output_length=remaining_output_length,
                round_index=round_index,
                task_class=task_class,
                is_shell=node.stable_node_id not in base_node_ids,
            )
        )
        for node in nodes
    }
    selected = []
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
        node = min(
            eligible,
            key=lambda item: (
                -scores[item.stable_node_id], item.depth, item.stable_node_id
            ),
        )
        selected.append(node.stable_node_id)
        selected_set.add(node.stable_node_id)
    return SelectionResult(tuple(selected), 0)


def _auroc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and scores[ordered[end]] == scores[ordered[cursor]]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for index in ordered[cursor:end]:
            ranks[index] = average_rank
        cursor = end
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _auprc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    positives = sum(labels)
    if positives == 0:
        return None
    ordered = sorted(
        range(len(scores)), key=lambda index: (-scores[index], index)
    )
    hits = 0
    precisions = []
    for rank, index in enumerate(ordered, start=1):
        if labels[index]:
            hits += 1
            precisions.append(hits / rank)
    return sum(precisions) / positives


def _ranking_metrics(
    rows: Sequence[Mapping[str, Any]], scores: Mapping[tuple[str, str], float]
) -> dict[str, Any]:
    labels = [int(row["target_only_labels"]["node_on_target_path"]) for row in rows]
    values = [scores[(str(row["snapshot_id"]), str(row["stable_node_id"]))] for row in rows]
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["snapshot_id"])].append(index)
    precisions = []
    recalls = []
    ndcgs = []
    for indices in grouped.values():
        selected = sorted(indices, key=lambda index: (-values[index], index))[:4]
        positives = sum(labels[index] for index in indices)
        hits = sum(labels[index] for index in selected)
        precisions.append(hits / max(1, len(selected)))
        recalls.append(hits / positives if positives else 0.0)
        dcg = sum(
            labels[index] / math.log2(rank + 2)
            for rank, index in enumerate(selected)
        )
        ideal = sum(
            1.0 / math.log2(rank + 2) for rank in range(min(4, positives))
        )
        ndcgs.append(dcg / ideal if ideal else 0.0)
    positives = sum(labels)
    return {
        "nodes": len(rows),
        "positive_nodes": positives,
        "positive_prevalence": positives / max(1, len(rows)),
        "auroc": _auroc(labels, values),
        "auprc": _auprc(labels, values),
        "precision_at_budget_4": statistics.fmean(precisions) if precisions else None,
        "recall_at_budget_4": statistics.fmean(recalls) if recalls else None,
        "ndcg_at_budget_4": statistics.fmean(ndcgs) if ndcgs else None,
    }


def feature_separability_report(
    rows: Sequence[Mapping[str, Any]], model: LearnedShellModel
) -> list[dict[str, Any]]:
    orientations = {
        "local_probability": 1.0,
        "log_local_probability": 1.0,
        "path_probability": 1.0,
        "log_path_probability": 1.0,
        "depth": -1.0,
        "sibling_rank": -1.0,
        "parent_probability": 1.0,
        "cumulative_entropy": -1.0,
        "local_entropy": -1.0,
        "top1_top2_margin": 1.0,
        "branching_factor": 1.0,
        "remaining_output_length": 1.0,
        "round_index": -1.0,
        "is_shell": 1.0,
        "task_code": 1.0,
        "task_chat": 1.0,
        "task_summarization": 1.0,
    }
    scorers = {
        name: {
            (str(row["snapshot_id"]), str(row["stable_node_id"])): orientation
            * float(row["runtime_features"][name])
            for row in rows
        }
        for name, orientation in orientations.items()
    }
    scorers["residual-probability"] = {
        (str(row["snapshot_id"]), str(row["stable_node_id"])): float(
            row["runtime_features"]["path_probability"]
        )
        for row in rows
    }
    scorers["entropy-margin-heuristic"] = {
        (str(row["snapshot_id"]), str(row["stable_node_id"])): (
            float(row["runtime_features"]["log_path_probability"])
            + 0.25 * float(row["runtime_features"]["top1_top2_margin"])
            - 0.10
            * float(row["runtime_features"]["cumulative_entropy"])
            / max(1.0, float(row["runtime_features"]["depth"]))
        )
        for row in rows
    }
    scorers[LEARNED_SELECTOR] = {
        (str(row["snapshot_id"]), str(row["stable_node_id"])): model.score(
            row["runtime_features"]
        )
        for row in rows
    }
    scopes = {
        "all-reachable-nodes": lambda row: True,
        "2x-shell-reachable-nodes": lambda row: bool(
            row["scope_flags"]["2x_shell_reachable_nodes"]
        ),
        "oracle-shell-opportunity-snapshots": lambda row: bool(
            row["scope_flags"]["oracle_shell_opportunity_snapshot"]
        ),
    }
    output = []
    for scope, predicate in scopes.items():
        scoped = [row for row in rows if predicate(row)]
        for scorer, scores in scorers.items():
            evaluation_rows = [
                row for row in scoped if row["request_split"] == "test"
            ]
            output.append(
                {
                    "scope": scope,
                    "scorer": scorer,
                    "request_split": "test",
                    **_ranking_metrics(evaluation_rows, scores),
                }
            )
    return output


def _parse_learned_replay(value: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(value)
    if copied.get("schema_version") != LEARNED_REPLAY_SCHEMA:
        raise ValueError("unsupported learned-selector replay schema")
    if copied.get("runtime_features_only_at_inference") is not True:
        raise TargetFeatureLeakageError(
            "learned replay does not guarantee target-label isolation"
        )
    if not copied.get("request_id") or copied.get("request_split") not in {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("learned replay request identity or split is invalid")
    if len(str(copied.get("model_sha256", ""))) != 64:
        raise ValueError("learned replay model checksum is missing")
    result = copied.get("result")
    if not isinstance(result, Mapping):
        raise ValueError("learned replay result is missing")
    rounds = result.get("rounds", ())
    if not rounds or result.get("proposal_rounds") != len(rounds):
        raise ValueError("learned replay proposal-round accounting differs")
    position = 0
    for row in rounds:
        accepted = row.get("accepted_tokens")
        committed = row.get("committed_tokens")
        selected = row.get("selected_node_ids", ())
        if row.get("prefix_position") != position:
            raise ValueError("learned replay prefix progression is discontinuous")
        if len(selected) != len(set(selected)) or len(selected) > 4:
            raise ValueError("learned replay selection violates budget or uniqueness")
        if not isinstance(accepted, int) or not isinstance(committed, int):
            raise ValueError("learned replay accounting must use integers")
        if not 0 <= accepted <= len(selected) or committed - accepted not in {0, 1}:
            raise ValueError("learned replay accepted/committed accounting is invalid")
        position += committed
    if result.get("committed_tokens") != position:
        raise ValueError("learned replay committed total differs")
    if result.get("final_sequence_matches_target") is not True:
        raise ValueError("learned replay final target equality flag is false")
    if len(result.get("final_committed_target_token_ids", ())) != position:
        raise ValueError("learned replay final token count differs")
    return copied


def _learned_replay_store(path: Path) -> ImmutableRequestStore[dict[str, Any]]:
    return ImmutableRequestStore(path, _parse_learned_replay, lambda value: value)


def _learned_sequential_one(
    request: R3RealRequest,
    target: Any,
    snapshots: Mapping[int, Mapping[str, Any]],
    model: LearnedShellModel,
) -> dict[str, Any]:
    position = 0
    committed_sequence = []
    rounds = []
    while position < target.target_path_length:
        snapshot = snapshots[position]
        labeled = LabeledTraceRecord.from_dict(snapshot["labeled_trace"])
        pool2 = set(labeled.pool_node_ids["2x"])
        runtime = tuple(
            node.runtime_features
            for node in labeled.nodes
            if node.runtime_features.stable_node_id in pool2
        )
        selection = select_learned_shell_ranker(
            runtime,
            int(snapshot["verification_budget"]),
            model,
            base_node_ids=set(labeled.pool_node_ids["1x"]),
            task_class=request.task_class,
            round_index=position,
            remaining_output_length=max(
                1, request.maximum_new_tokens - position
            ),
        )
        evaluation = _evaluate_selection(
            labeled, "2x", LEARNED_SELECTOR, selection
        )
        remaining = target.target_path_length - position
        accepted = int(evaluation["accepted_draft_tokens_per_proposal"])
        committed = min(remaining, accepted + int(accepted < remaining))
        committed_sequence.extend(
            target.target_token_ids[position : position + committed]
        )
        rounds.append(
            {
                "round_index": len(rounds),
                "prefix_position": position,
                "forest_sha256": snapshot["forest_sha256"],
                "selected_node_ids": list(selection.selected_node_ids),
                "accepted_tokens": accepted,
                "committed_tokens": committed,
                "verified_nodes": len(selection.selected_node_ids),
            }
        )
        position += committed
    if tuple(committed_sequence) != target.target_token_ids:
        raise AssertionError("learned selector committed sequence differs from target")
    return {
        "pool_ratio": "2x",
        "selector": LEARNED_SELECTOR,
        "proposal_rounds": len(rounds),
        "accepted_tokens": sum(row["accepted_tokens"] for row in rounds),
        "committed_tokens": sum(row["committed_tokens"] for row in rounds),
        "verified_nodes": sum(row["verified_nodes"] for row in rounds),
        "accepted_tokens_per_proposal": statistics.fmean(
            row["accepted_tokens"] for row in rounds
        ),
        "committed_tokens_per_proposal": statistics.fmean(
            row["committed_tokens"] for row in rounds
        ),
        "final_committed_target_token_ids": list(committed_sequence),
        "final_sequence_matches_target": True,
        "rounds": rounds,
    }


def _immutable_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"immutable artifact differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            if path.read_text(encoding="utf-8") != payload:
                raise FileExistsError(f"immutable artifact differs: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _metric_rows(
    baseline: Sequence[Mapping[str, Any]],
    learned: Sequence[Mapping[str, Any]],
    splits: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_by_request = {str(row["request_id"]): row for row in baseline}
    learned_by_request = {str(row["request_id"]): row for row in learned}
    comparison_rows = []
    recovery_rows = []
    for request_id, record in baseline_by_request.items():
        if splits[request_id] != "test":
            continue
        by_key = {
            (str(row["pool_ratio"]), str(row["selector"])): row
            for row in record["results"]
        }
        learned_result = learned_by_request[request_id]["result"]
        residual = by_key[("2x", "residual-probability")]
        entropy = by_key[("2x", "entropy-margin-heuristic")]
        oracle = by_key[("2x", ORACLE_SELECTOR)]
        base_value = float(residual["accepted_tokens_per_proposal"])
        oracle_value = float(oracle["accepted_tokens_per_proposal"])
        learned_value = float(learned_result["accepted_tokens_per_proposal"])
        comparison_rows.append(
            {
                "request_id": request_id,
                "task_class": record["task_class"],
                "residual": base_value,
                "entropy": float(entropy["accepted_tokens_per_proposal"]),
                "learned": learned_value,
                "oracle": oracle_value,
                "learned_minus_residual": learned_value - base_value,
                "entropy_minus_residual": float(
                    entropy["accepted_tokens_per_proposal"]
                )
                - base_value,
            }
        )
        denominator = oracle_value - base_value
        if denominator > 0:
            recovery_rows.append(
                {
                    "request_id": request_id,
                    "task_class": record["task_class"],
                    "learned_gap_recovery": (learned_value - base_value)
                    / denominator,
                    "entropy_gap_recovery": (
                        float(entropy["accepted_tokens_per_proposal"])
                        - base_value
                    )
                    / denominator,
                }
            )
    return comparison_rows, recovery_rows


def _simple_stats(
    rows: Sequence[Mapping[str, Any]], field: str, seed: int
) -> dict[str, Any]:
    observed = [row for row in rows if row.get(field) is not None]
    values = [float(row[field]) for row in observed]
    return {
        "requests": len(rows),
        "identifiable_requests": len(values),
        "mean": statistics.fmean(values) if values else None,
        "request_bootstrap_95_ci": (
            stratified_bootstrap_ci(observed, field, seed=seed)
            if values
            else [None, None]
        ),
    }


def _heldout_report(
    baseline: Sequence[Mapping[str, Any]],
    learned: Sequence[Mapping[str, Any]],
    splits: Mapping[str, str],
) -> dict[str, Any]:
    baseline_by_request = {str(row["request_id"]): row for row in baseline}
    learned_by_request = {str(row["request_id"]): row for row in learned}
    test_ids = sorted(
        request_id for request_id, split in splits.items() if split == "test"
    )
    if not test_ids:
        raise ValueError("learned-shell held-out test split is empty")
    tasks = sorted(
        {str(baseline_by_request[item]["task_class"]) for item in test_ids}
    ) + ["all"]
    metrics = []
    for task in tasks:
        ids = [
            request_id
            for request_id in test_ids
            if task == "all" or baseline_by_request[request_id]["task_class"] == task
        ]
        for selector in (
            "residual-probability",
            "entropy-margin-heuristic",
            LEARNED_SELECTOR,
            ORACLE_SELECTOR,
        ):
            rows = []
            for request_id in ids:
                baseline_record = baseline_by_request[request_id]
                if selector == LEARNED_SELECTOR:
                    result = learned_by_request[request_id]["result"]
                else:
                    result = next(
                        row
                        for row in baseline_record["results"]
                        if row["pool_ratio"] == "2x"
                        and row["selector"] == selector
                    )
                rows.append(
                    {
                        "request_id": request_id,
                        "task_class": baseline_record["task_class"],
                        "accepted_tokens_per_proposal": result[
                            "accepted_tokens_per_proposal"
                        ],
                        "committed_tokens_per_proposal": result[
                            "committed_tokens_per_proposal"
                        ],
                        "proposal_rounds_per_request": result["proposal_rounds"],
                        "verified_nodes_per_request": result["verified_nodes"],
                        "oracle_regret_per_request": (
                            next(
                                row["accepted_tokens"]
                                for row in baseline_record["results"]
                                if row["pool_ratio"] == "2x"
                                and row["selector"] == ORACLE_SELECTOR
                            )
                            - result["accepted_tokens"]
                        ),
                    }
                )
            metrics.append(
                {
                    "task_class": task,
                    "selector": selector,
                    "pool_ratio": "2x",
                    "requests": len(rows),
                    "statistics": {
                        field: _simple_stats(rows, field, 1664 + index)
                        for index, field in enumerate(
                            (
                                "accepted_tokens_per_proposal",
                                "committed_tokens_per_proposal",
                                "proposal_rounds_per_request",
                                "verified_nodes_per_request",
                                "oracle_regret_per_request",
                            )
                        )
                    },
                }
            )
    comparisons, recovery = _metric_rows(baseline, learned, splits)
    paired = []
    for task in tasks:
        values = [
            row for row in comparisons if task == "all" or row["task_class"] == task
        ]
        recovery_values = [
            row for row in recovery if task == "all" or row["task_class"] == task
        ]
        paired.append(
            {
                "task_class": task,
                "learned_minus_residual": _simple_stats(
                    values, "learned_minus_residual", 1800
                ),
                "entropy_minus_residual": _simple_stats(
                    values, "entropy_minus_residual", 1801
                ),
                "learned_oracle_gap_recovery": _simple_stats(
                    recovery_values, "learned_gap_recovery", 1802
                ),
                "entropy_oracle_gap_recovery": _simple_stats(
                    recovery_values, "entropy_gap_recovery", 1803
                ),
            }
        )
    return {"metrics": metrics, "paired": paired}


def _decision(heldout: Mapping[str, Any]) -> dict[str, Any]:
    paired = {row["task_class"]: row for row in heldout["paired"]}
    candidates = {}
    for selector, delta_key, recovery_key in (
        (LEARNED_SELECTOR, "learned_minus_residual", "learned_oracle_gap_recovery"),
        (
            "entropy-margin-heuristic",
            "entropy_minus_residual",
            "entropy_oracle_gap_recovery",
        ),
    ):
        overall_delta = paired["all"][delta_key]["request_bootstrap_95_ci"]
        recovery = paired["all"][recovery_key]["request_bootstrap_95_ci"]
        task_rows = [row for task, row in paired.items() if task != "all"]
        task_ci_lower_bounds = [
            row[delta_key]["request_bootstrap_95_ci"][0] for row in task_rows
        ]
        no_task_harm = all(
            lower is not None and lower >= 0 for lower in task_ci_lower_bounds
        )
        qualifies = (
            overall_delta[0] is not None
            and overall_delta[0] > 0
            and recovery[0] is not None
            and recovery[0] > 0
            and no_task_harm
        )
        candidates[selector] = {
            "qualifies": qualifies,
            "overall_delta_95_ci": overall_delta,
            "gap_recovery_95_ci": recovery,
            "no_task_harm_95_ci": no_task_harm,
            "task_delta_95_ci_lower_bounds": task_ci_lower_bounds,
        }
    outcome_a = any(row["qualifies"] for row in candidates.values())
    return {
        "outcome": "A" if outcome_a else "B",
        "fixed_rule": (
            "Outcome A requires overall paired improvement CI lower bound > 0, "
            "gap-recovery CI lower bound > 0, and every represented task's paired "
            "delta CI lower bound >= 0. The rule is fixed before server evaluation."
        ),
        "candidate_checks": candidates,
        "recommendation": (
            "2x Overdraft-and-Prune packed-tree prototype"
            if outcome_a
            else (
                "Do not expand the pool or promote the learned selector; return to 1x "
                "Residual-Probability and implement only the Dual-Batch/overlap baseline."
            )
        ),
        "test_set_used_for_rule_definition": False,
    }


def run_learned_selector_pilot(
    requests: Iterable[R3RealRequest],
    *,
    workload_path: Path,
    target_dir: Path,
    snapshot_dir: Path,
    sequential_dir: Path,
    output_dir: Path,
    resume: bool,
    seed: int = 1664,
    source_trace_commit: Optional[str] = None,
) -> dict[str, Any]:
    if source_trace_commit is not None and (
        len(source_trace_commit) != 40
        or any(
            character not in "0123456789abcdef"
            for character in source_trace_commit.lower()
        )
    ):
        raise ValueError("source_trace_commit must be a full 40-character Git SHA")
    request_list = list(requests)
    request_by_id = {request.request_id: request for request in request_list}
    if len(request_by_id) != len(request_list):
        raise ValueError("learned pilot workload request IDs are duplicated")
    snapshots = snapshot_store(snapshot_dir).records()
    baseline = sequential_store(sequential_dir).records()
    targets = target_store(target_dir)
    request_ids = set(request_by_id)
    if {str(row["request_id"]) for row in baseline} != request_ids:
        raise ValueError("learned pilot baseline request set differs from workload")
    if {record.request_id for record in targets.records()} != request_ids:
        raise ValueError("learned pilot target request set differs from workload")
    if {str(row["request_id"]) for row in snapshots} != request_ids:
        raise ValueError("learned pilot snapshot request set differs from workload")
    splits = stratified_request_splits(request_list, seed=seed)
    split_payload = {
        "schema_version": "specrhythm.phase3c-request-splits.v1",
        "seed": seed,
        "unit": "request_id",
        "task_stratified": True,
        "ratios": [0.70, 0.15, 0.15],
        "assignments": dict(sorted(splits.items())),
    }
    split_hash = _payload_sha(split_payload)
    feature_rows = build_runtime_feature_rows(
        snapshots, request_by_id, splits
    )
    model = train_learned_shell_ranker(feature_rows, split_hash=split_hash)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in feature_rows
    )
    _immutable_text(output_dir / "runtime-feature-dataset.jsonl", feature_payload)
    _immutable_text(
        output_dir / "request-splits.json",
        json.dumps(split_payload, indent=2, sort_keys=True) + "\n",
    )
    _immutable_text(
        output_dir / "learned-shell-model.json",
        json.dumps(model.to_dict(), indent=2, sort_keys=True) + "\n",
    )
    model_hash = _payload_sha(model.to_dict())
    by_request: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for snapshot in snapshots:
        by_request[str(snapshot["request_id"])][
            int(snapshot["prefix_position"])
        ] = snapshot
    replay_store = _learned_replay_store(output_dir / "learned-replay")
    if not resume and replay_store.records():
        raise FileExistsError("learned replay output is non-empty; pass --resume")
    written = 0
    for request in request_list:
        if replay_store.has(request.request_id):
            if not resume:
                raise FileExistsError(replay_store.path(request.request_id))
            existing = replay_store.read(request.request_id)
            target_sha = _canonical_sha(targets.read(request.request_id).to_dict())
            if (
                existing["model_sha256"] != model_hash
                or existing["request_split"] != splits[request.request_id]
                or existing["target_trajectory_sha256"] != target_sha
            ):
                raise ValueError(
                    f"learned replay resume identity differs for {request.request_id}"
                )
            continue
        target = targets.read(request.request_id)
        result = _learned_sequential_one(
            request, target, by_request[request.request_id], model
        )
        record = {
            "schema_version": LEARNED_REPLAY_SCHEMA,
            "request_id": request.request_id,
            "task_class": request.task_class,
            "request_split": splits[request.request_id],
            "target_trajectory_sha256": _canonical_sha(target.to_dict()),
            "model_sha256": model_hash,
            "runtime_features_only_at_inference": True,
            "result": result,
        }
        written += int(replay_store.write(request.request_id, record))
    learned = replay_store.records()
    if {str(row["request_id"]) for row in learned} != request_ids:
        raise ValueError("learned replay request set differs from workload")
    shell = shell_opportunity_decomposition(snapshots)
    separability = feature_separability_report(feature_rows, model)
    heldout = _heldout_report(baseline, learned, splits)
    _immutable_text(
        output_dir / "shell-opportunity.json",
        json.dumps(shell, indent=2, sort_keys=True) + "\n",
    )
    _immutable_text(
        output_dir / "feature-separability.json",
        json.dumps(separability, indent=2, sort_keys=True) + "\n",
    )
    _immutable_text(
        output_dir / "heldout-test.json",
        json.dumps(heldout, indent=2, sort_keys=True) + "\n",
    )
    artifacts = {
        "workload_sha256": hashlib.sha256(workload_path.read_bytes()).hexdigest(),
        "runtime_feature_dataset_sha256": hashlib.sha256(
            feature_payload.encode()
        ).hexdigest(),
        "request_splits_file_sha256": hashlib.sha256(
            (output_dir / "request-splits.json").read_bytes()
        ).hexdigest(),
        "model_file_sha256": hashlib.sha256(
            (output_dir / "learned-shell-model.json").read_bytes()
        ).hexdigest(),
        "learned_replay_sha256": _directory_sha(output_dir / "learned-replay"),
        "shell_opportunity_file_sha256": hashlib.sha256(
            (output_dir / "shell-opportunity.json").read_bytes()
        ).hexdigest(),
        "feature_separability_file_sha256": hashlib.sha256(
            (output_dir / "feature-separability.json").read_bytes()
        ).hexdigest(),
        "heldout_test_file_sha256": hashlib.sha256(
            (output_dir / "heldout-test.json").read_bytes()
        ).hexdigest(),
    }
    report = {
        "schema_version": "specrhythm.phase3c-learned-pilot.v1",
        "evidence_scope": "corrected-multiround-selector-diagnostic-only",
        "request_count": len(request_list),
        "snapshot_count": len(snapshots),
        "new_learned_replay_records": written,
        "source_trace_commit": source_trace_commit,
        "split_counts": {
            split: sum(value == split for value in splits.values())
            for split in ("train", "validation", "test")
        },
        "model": model.to_dict(),
        "feature_dataset": {
            "file": "runtime-feature-dataset.jsonl",
            "sha256": artifacts["runtime_feature_dataset_sha256"],
            "rows": len(feature_rows),
            "runtime_and_target_fields_serialized_separately": True,
        },
        "source_trace_sha256": {
            "targets": _directory_sha(target_dir),
            "snapshots": _directory_sha(snapshot_dir),
            "sequential": _directory_sha(sequential_dir),
        },
        "provenance": {
            "request_split_sha256": split_hash,
            "model_sha256": model_hash,
            "training_set_sha256": model.metadata["training_set_sha256"],
            "source_trace_commit": source_trace_commit,
            "artifacts": artifacts,
        },
        "shell_opportunity_decomposition": shell,
        "feature_separability": separability,
        "heldout_test": heldout,
        "decision": _decision(heldout),
        "gpu_performance_result": False,
        "reports_latency_goodput_slo_or_speedup": False,
        "serving_engine": False,
    }
    manifest = {
        "schema_version": "specrhythm.phase3c-learned-pilot-manifest.v1",
        "source_trace_commit": source_trace_commit,
        "request_count": len(request_list),
        "snapshot_count": len(snapshots),
        "selector": LEARNED_SELECTOR,
        "runtime_features_only_at_inference": True,
        "target_labels_used_for_training_only": True,
        "artifacts": artifacts,
        "request_split_sha256": split_hash,
        "model_payload_sha256": model_hash,
        "gpu_performance_result": False,
    }
    _immutable_text(
        output_dir / "artifact-manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    report["artifact_manifest"] = manifest
    return report


def learned_pilot_markdown(report: Mapping[str, Any]) -> str:
    def value_text(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.4f}"

    lines = [
        "# Phase 3C.3 2x shell learnability pilot",
        "",
        (
            "Corrected real-model selector diagnostics only. No latency, goodput, SLO, "
            "serving-engine, or speedup claim."
        ),
        "",
        "## 2x minus 1x shell opportunity",
        "",
        (
            "| Task | Snapshots | Target present | Budget reachable | Oracle uses shell | "
            "Oracle gain | Residual recall | Entropy recall |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["shell_opportunity_decomposition"]["aggregate"]:
        lines.append(
            f"| {row['task_class']} | {row['snapshots']} | "
            f"{row['shell_target_present_snapshot_ratio']:.4f} | "
            f"{row['shell_target_budget_reachable_snapshot_ratio']:.4f} | "
            f"{row['oracle_uses_shell_snapshot_ratio']:.4f} | "
            f"{row['oracle_shell_gain_positive_snapshot_ratio']:.4f} | "
            f"{row['residual_shell_selection_recall']:.4f} | "
            f"{row['entropy_margin_shell_selection_recall']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Held-out node separability",
            "",
            "| Scope | Scorer | Prevalence | AUROC | AUPRC | P@4 | R@4 | NDCG@4 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["feature_separability"]:
        if row["scorer"] not in {
            "residual-probability",
            "entropy-margin-heuristic",
            LEARNED_SELECTOR,
        }:
            continue
        lines.append(
            f"| {row['scope']} | {row['scorer']} | "
            f"{value_text(row['positive_prevalence'])} | "
            f"{value_text(row['auroc'])} | {value_text(row['auprc'])} | "
            f"{value_text(row['precision_at_budget_4'])} | "
            f"{value_text(row['recall_at_budget_4'])} | "
            f"{value_text(row['ndcg_at_budget_4'])} |"
        )
    lines.extend(
        [
            "",
        "## Held-out request-level results",
        "",
        (
            "| Task | Selector | Accepted/proposal [95% CI] | Committed/proposal | "
            "Rounds/request | Verified/request | Oracle regret/request |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["heldout_test"]["metrics"]:
        stats = row["statistics"]
        accepted = stats["accepted_tokens_per_proposal"]
        ci = accepted["request_bootstrap_95_ci"]
        lines.append(
            f"| {row['task_class']} | {row['selector']} | "
            f"{accepted['mean']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] | "
            f"{stats['committed_tokens_per_proposal']['mean']:.4f} | "
            f"{stats['proposal_rounds_per_request']['mean']:.4f} | "
            f"{stats['verified_nodes_per_request']['mean']:.4f} | "
            f"{stats['oracle_regret_per_request']['mean']:.4f} |"
        )
    decision = report["decision"]
    lines.extend(
        [
            "",
            "## Fixed decision gate",
            "",
            f"Outcome **{decision['outcome']}**. {decision['recommendation']}",
            "",
            decision["fixed_rule"],
            "",
            (
                "The within-request oracle reads target outcomes and is an upper bound. "
                "The learned selector uses target labels only during training; held-out "
                "inference receives runtime features only."
            ),
        ]
    )
    return "\n".join(lines) + "\n"
