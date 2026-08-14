"""Phase-2 candidate-pool and target-outcome oracle diagnostics.

These helpers are intentionally isolated from the default policy registry.  They expose
structural and fully-hidden-search upper bounds, never a deployable scheduling policy.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import statistics
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Optional

from specrhythm.policies.base import PolicySnapshot, StepPlan
from specrhythm.policies.baselines import allocate_round_robin
from specrhythm.policies.specrhythm import ShapingDiagnosticPolicy
from specrhythm.schema import Workload
from specrhythm.simulator import (
    PlanningDiagnostic,
    SimulationResult,
    SimulatorConfig,
    simulate,
)
from specrhythm.tree import (
    CandidateTree,
    CandidateTreeNode,
    CandidateTreeOracle,
    SelectedProposalTree,
    TreeVerificationOutcome,
    expected_tree_progress,
    make_selected_tree,
    select_sequence_path,
    validate_prefix_closure,
)

SEARCH_RATIOS = (1, 2, 4, 8)
PHASE2_VARIANTS = {
    "a": "residual-probability-current-selector",
    "b": "oracle-within-request-residual",
    "c": "oracle-global-residual",
    "d": "oracle-full-tree-ceiling",
}


def _root_branch_ids(tree: CandidateTree, width: int) -> tuple[str, ...]:
    return tuple(
        node.node_id
        for node in tree.children_by_parent.get(tree.root.node_id, ())[:width]
    )


def base_pool(tree: CandidateTree, width: int) -> CandidateTree:
    """Project an expanded tree back to the immutable 1x candidate pool."""

    roots = set(_root_branch_ids(tree, width))
    retained = {tree.root.node_id, *roots}
    changed = True
    while changed:
        changed = False
        for node in tree.candidate_nodes:
            if node.parent_id in retained and node.node_id not in retained:
                # Extra branches are rooted outside the first ``width`` children.
                root_ancestor = node
                while root_ancestor.parent_id not in {None, tree.root.node_id}:
                    root_ancestor = tree.by_id[root_ancestor.parent_id or ""]
                if root_ancestor.node_id in roots:
                    retained.add(node.node_id)
                    changed = True
    nodes = tuple(node for node in tree.nodes if node.node_id in retained)
    return CandidateTree(
        tree.request_id,
        tree.root,
        nodes,
        requested_candidate_nodes=len(nodes) - 1,
        search_budget_ratio=1,
    )


class Phase2CandidateTreeOracle(CandidateTreeOracle):
    """Nested candidate pools with a target trajectory frozen on the 1x tree.

    The historical oracle chooses a target child from the children in the supplied tree.
    That is valid for fixed-pool policy comparison but would make target truth depend on
    Phase-2 search ratio.  This oracle expands only after freezing the historical 1x
    trajectory, so ratio 1 is bit-for-bit compatible while larger pools cannot rewrite it.
    """

    def __init__(self, seed: int, *, search_ratio: int, base_width: int) -> None:
        if search_ratio not in SEARCH_RATIOS:
            raise ValueError("Phase-2 search ratio must be one of 1, 2, 4, or 8")
        if base_width < 1:
            raise ValueError("Phase-2 base width must be positive")
        super().__init__(seed)
        self.search_ratio = search_ratio
        self.base_width = base_width
        self.target_access_count = 0
        self._target_cache: dict[tuple[str, int, float, int], tuple[str, ...]] = {}

    def tree(
        self,
        request_id: str,
        committed_prefix_len: int,
        *,
        width: int,
        depth: int,
        draft_confidence: float,
    ) -> CandidateTree:
        if width != self.base_width:
            raise ValueError("Phase-2 oracle width must match its frozen 1x width")
        original = super().tree(
            request_id,
            committed_prefix_len,
            width=width,
            depth=depth,
            draft_confidence=draft_confidence,
        )
        return self.expand_pool(
            original,
            draft_confidence=draft_confidence,
            committed_prefix_len=committed_prefix_len,
        )

    def expand_pool(
        self,
        original: CandidateTree,
        *,
        draft_confidence: float,
        committed_prefix_len: int,
    ) -> CandidateTree:
        """Append deterministic root-to-depth branches without changing 1x nodes."""

        if self.search_ratio == 1 or original.depth == 0:
            return CandidateTree(
                original.request_id,
                original.root,
                original.nodes,
                requested_candidate_nodes=len(original.candidate_nodes),
                search_budget_ratio=1,
            )
        base = base_pool(original, self.base_width)
        confidence = max(0.0, min(1.0, draft_confidence))
        # Node probabilities for appended branches use the same deterministic events
        # and 1x normalization. Existing probabilities are copied, never regenerated.
        raw_base = [
            0.5
            + self._event(
                original.request_id, committed_prefix_len, 1, branch + 1
            )
            for branch in range(self.base_width)
        ]
        scale = confidence / sum(raw_base)
        nodes = list(base.nodes)
        target_width = self.base_width * self.search_ratio
        for branch in range(self.base_width, target_width):
            raw = 0.5 + self._event(
                original.request_id, committed_prefix_len, 1, branch + 1
            )
            conditional = min(1.0, raw * scale)
            parent = original.root
            for level in range(1, original.depth + 1):
                token_confidence = conditional if level == 1 else confidence
                node = CandidateTreeNode(
                    node_id=f"d{level}-p{parent.node_id}-b{branch if level == 1 else 0}",
                    parent_id=parent.node_id,
                    depth=level,
                    token_confidence=token_confidence,
                    path_probability=parent.path_probability * token_confidence,
                )
                nodes.append(node)
                parent = node
        requested = len(base.candidate_nodes) * self.search_ratio
        return CandidateTree(
            original.request_id,
            original.root,
            tuple(nodes),
            requested_candidate_nodes=requested,
            search_budget_ratio=self.search_ratio,
        )

    def canonical_trajectory(
        self,
        tree: CandidateTree,
        *,
        committed_prefix_len: int,
        acceptance_probability: float,
    ) -> tuple[str, ...]:
        """Return target-matching draft nodes chosen only from the immutable 1x pool."""

        self.target_access_count += 1
        if not 0 <= acceptance_probability <= 1 or not math.isfinite(
            acceptance_probability
        ):
            raise ValueError("acceptance probability must be finite and in [0, 1]")
        key = (
            tree.request_id,
            committed_prefix_len,
            float(acceptance_probability),
            tree.depth,
        )
        if key in self._target_cache:
            return self._target_cache[key]
        frozen = base_pool(tree, self.base_width)
        accepted = []
        parent_id = frozen.root.node_id
        for depth in range(1, frozen.depth + 1):
            if (
                self._event(tree.request_id, committed_prefix_len, depth, 1)
                >= acceptance_probability
            ):
                break
            children = frozen.children_by_parent.get(parent_id, ())
            if not children:
                break
            choice = int(
                self._event(tree.request_id, committed_prefix_len, depth, 2)
                * len(children)
            )
            node = children[min(choice, len(children) - 1)]
            accepted.append(node.node_id)
            parent_id = node.node_id
        result = tuple(accepted)
        self._target_cache[key] = result
        return result

    def verify(
        self,
        tree: CandidateTree,
        selected: SelectedProposalTree,
        *,
        committed_prefix_len: int,
        acceptance_probability: float,
    ) -> TreeVerificationOutcome:
        validate_prefix_closure(tree, selected)
        target = self.canonical_trajectory(
            tree,
            committed_prefix_len=committed_prefix_len,
            acceptance_probability=acceptance_probability,
        )
        selected_ids = set(selected.selected_node_ids)
        accepted = []
        for node_id in target:
            if node_id not in selected_ids:
                break
            accepted.append(node_id)
        return TreeVerificationOutcome(tuple(accepted), len(accepted) + 1)


def realized_candidate_progress(
    selected: SelectedProposalTree, target: tuple[str, ...]
) -> int:
    selected_ids = set(selected.selected_node_ids)
    accepted = 0
    for node_id in target:
        if node_id not in selected_ids:
            break
        accepted += 1
    return accepted


def _eligible_nodes(
    tree: CandidateTree, selected: list[str], cap: int
) -> list[CandidateTreeNode]:
    if len(selected) >= cap:
        return []
    selected_set = set(selected)
    parents = (tree.root.node_id, *selected)
    return [
        node
        for parent in parents
        for node in tree.children_by_parent.get(parent, ())
        if node.node_id not in selected_set
    ]


def _fill_request_by_probability(
    tree: CandidateTree, selected: list[str], budget: int
) -> list[str]:
    while len(selected) < budget:
        eligible = _eligible_nodes(tree, selected, budget)
        if not eligible:
            break
        node = max(eligible, key=lambda item: (item.path_probability, item.node_id))
        selected.append(node.node_id)
    return selected


def _within_request_oracle_selection(
    tree: CandidateTree,
    base: SelectedProposalTree,
    total_budget: int,
    target: tuple[str, ...],
) -> SelectedProposalTree:
    selected = list(base.selected_node_ids)
    selected_set = set(selected)
    for node_id in target:
        required = tree.ancestors(node_id)
        missing = [item for item in required if item not in selected_set]
        if len(selected) + len(missing) > total_budget:
            break
        selected.extend(missing)
        selected_set.update(missing)
    _fill_request_by_probability(tree, selected, total_budget)
    return make_selected_tree(tree, selected)


def _base_trees(
    snapshot: PolicySnapshot,
    oracle: Phase2CandidateTreeOracle,
    per_request_budget: int,
) -> dict[str, SelectedProposalTree]:
    budgets = allocate_round_robin(snapshot, per_request_budget)
    result = {}
    for request in snapshot.normal_requests:
        tree = request.candidate_tree
        if tree is None:
            raise ValueError("Phase-2 diagnostics require explicit candidate trees")
        frozen = base_pool(tree, oracle.base_width)
        result[request.request_id] = select_sequence_path(
            frozen, budgets[request.request_id]
        )
    return result


def _build_plan(
    snapshot: PolicySnapshot,
    trees: dict[str, CandidateTree],
    selections: dict[str, SelectedProposalTree],
    base_trees: dict[str, SelectedProposalTree],
) -> StepPlan:
    by_id = {request.request_id: request for request in snapshot.normal_requests}
    return StepPlan(
        normal_budgets={key: value.candidate_budget for key, value in selections.items()},
        normal_trees=selections,
        candidate_trees=trees,
        expected_progress={
            key: expected_tree_progress(trees[key], value)
            for key, value in selections.items()
        },
        requested_progress_gap={
            key: by_id[key].continuous_progress_gap for key in selections
        },
        slo_stage_budgets={key: 0 for key in selections},
        residual_stage_budgets={
            key: max(0, value.candidate_budget - base_trees[key].candidate_budget)
            for key, value in selections.items()
        },
        base_normal_budgets={
            key: value.candidate_budget for key, value in base_trees.items()
        },
        base_normal_trees=base_trees,
        required_total_progress={
            key: by_id[key].required_total_progress for key in selections
        },
        required_candidate_progress={
            key: by_id[key].required_candidate_progress for key in selections
        },
        maximum_attainable_candidate_progress={
            key: by_id[key].maximum_attainable_candidate_progress for key in selections
        },
        maximum_attainable_total_progress={
            key: by_id[key].maximum_attainable_total_progress for key in selections
        },
        one_cycle_feasible={key: by_id[key].one_cycle_feasible for key in selections},
    )


def allocate_phase2_current(
    snapshot: PolicySnapshot,
    *,
    oracle: Phase2CandidateTreeOracle,
    per_request_budget: int,
) -> StepPlan:
    """A_r: frozen Dual-Batch base plus target-blind path-probability fill."""

    base_trees = _base_trees(snapshot, oracle, per_request_budget)
    trees = {
        request.request_id: request.candidate_tree
        for request in snapshot.normal_requests
        if request.candidate_tree is not None
    }
    selected = {
        key: list(value.selected_node_ids) for key, value in base_trees.items()
    }
    remaining = snapshot.roof_candidate_budget - sum(map(len, selected.values()))
    versions = {key: 0 for key in selected}
    heap: list[tuple[float, str, str, int]] = []
    by_id = {request.request_id: request for request in snapshot.normal_requests}

    def push(request_id: str) -> None:
        request = by_id[request_id]
        for node in _eligible_nodes(
            trees[request_id], selected[request_id], request.max_budget
        ):
            heapq.heappush(
                heap,
                (-node.path_probability, request_id, node.node_id, versions[request_id]),
            )

    for request_id in selected:
        push(request_id)
    while remaining > 0 and heap:
        _, request_id, node_id, version = heapq.heappop(heap)
        if version != versions[request_id]:
            continue
        request = by_id[request_id]
        eligible = {
            node.node_id: node
            for node in _eligible_nodes(
                trees[request_id], selected[request_id], request.max_budget
            )
        }
        node = eligible.get(node_id)
        if node is None:
            continue
        selected[request_id].append(node.node_id)
        versions[request_id] += 1
        remaining -= 1
        push(request_id)
    selections = {
        key: make_selected_tree(trees[key], value) for key, value in selected.items()
    }
    return _build_plan(snapshot, trees, selections, base_trees)


def _targets(
    snapshot: PolicySnapshot, oracle: Phase2CandidateTreeOracle
) -> dict[str, tuple[str, ...]]:
    return {
        request.request_id: oracle.canonical_trajectory(
            request.candidate_tree,
            committed_prefix_len=request.committed_prefix_len,
            acceptance_probability=request.acceptance_probability,
        )
        for request in snapshot.normal_requests
        if request.candidate_tree is not None
    }


def allocate_phase2_within_request(
    snapshot: PolicySnapshot,
    current: StepPlan,
    *,
    oracle: Phase2CandidateTreeOracle,
    targets: Optional[dict[str, tuple[str, ...]]] = None,
) -> StepPlan:
    """B_r: target-optimal selection under A_r's per-request residual vector."""

    targets = targets or _targets(snapshot, oracle)
    selections = {
        request_id: _within_request_oracle_selection(
            current.candidate_trees[request_id],
            current.base_normal_trees[request_id],
            current.normal_budgets[request_id],
            targets[request_id],
        )
        for request_id in current.normal_budgets
    }
    return _build_plan(
        snapshot, current.candidate_trees, selections, current.base_normal_trees
    )


def _globally_optimal_plan(
    snapshot: PolicySnapshot,
    current: StepPlan,
    *,
    oracle: Phase2CandidateTreeOracle,
    replace_base: bool,
    targets: Optional[dict[str, tuple[str, ...]]] = None,
) -> StepPlan:
    targets = targets or _targets(snapshot, oracle)
    request_ids = sorted(current.normal_budgets)
    by_id = {request.request_id: request for request in snapshot.normal_requests}
    selected = {
        request_id: (
            []
            if replace_base
            else list(current.base_normal_trees[request_id].selected_node_ids)
        )
        for request_id in request_ids
    }
    target_total = current.total_candidates
    remaining = target_total - sum(len(value) for value in selected.values())
    # Every previously uncovered canonical target node costs one slot and yields one
    # additional committed candidate token once its target-prefix predecessors exist.
    # Therefore greedily taking these unit gains is the exact global optimum; the
    # deterministic request-ID order is only a tie break between equal-gain choices.
    while remaining > 0:
        added = False
        for request_id in request_ids:
            if len(selected[request_id]) >= by_id[request_id].max_budget:
                continue
            chosen = set(selected[request_id])
            next_node = next(
                (node for node in targets[request_id] if node not in chosen),
                None,
            )
            if next_node is None:
                continue
            missing = [
                node
                for node in current.candidate_trees[request_id].ancestors(next_node)
                if node not in chosen
            ]
            if len(missing) != 1:
                raise AssertionError("canonical target path lost unit-cost prefix closure")
            selected[request_id].append(next_node)
            remaining -= 1
            added = True
            if remaining == 0:
                break
        if not added:
            break

    versions = {request_id: 0 for request_id in request_ids}
    heap: list[tuple[float, str, str, int]] = []

    def push(request_id: str) -> None:
        for node in _eligible_nodes(
            current.candidate_trees[request_id],
            selected[request_id],
            by_id[request_id].max_budget,
        ):
            heapq.heappush(
                heap,
                (-node.path_probability, request_id, node.node_id, versions[request_id]),
            )

    for request_id in request_ids:
        push(request_id)
    while remaining > 0 and heap:
        _, request_id, node_id, version = heapq.heappop(heap)
        if version != versions[request_id]:
            continue
        eligible = {
            node.node_id: node
            for node in _eligible_nodes(
                current.candidate_trees[request_id],
                selected[request_id],
                by_id[request_id].max_budget,
            )
        }
        if node_id not in eligible:
            continue
        selected[request_id].append(node_id)
        versions[request_id] += 1
        remaining -= 1
        push(request_id)
    if remaining:
        raise AssertionError("oracle allocation could not preserve verification budget")
    selections = {
        request_id: make_selected_tree(
            current.candidate_trees[request_id], selected[request_id]
        )
        for request_id in request_ids
    }
    return _build_plan(
        snapshot,
        current.candidate_trees,
        selections,
        current.base_normal_trees,
    )


def allocate_phase2_global(
    snapshot: PolicySnapshot,
    current: StepPlan,
    *,
    oracle: Phase2CandidateTreeOracle,
    targets: Optional[dict[str, tuple[str, ...]]] = None,
) -> StepPlan:
    """C_r: globally reallocate only residual slots under target leakage."""

    return _globally_optimal_plan(
        snapshot,
        current,
        oracle=oracle,
        replace_base=False,
        targets=targets,
    )


def allocate_phase2_full_tree(
    snapshot: PolicySnapshot,
    current: StepPlan,
    *,
    oracle: Phase2CandidateTreeOracle,
    targets: Optional[dict[str, tuple[str, ...]]] = None,
) -> StepPlan:
    """D_r: target-aware full verification-tree ceiling at fixed request roots/B_verify."""

    return _globally_optimal_plan(
        snapshot,
        current,
        oracle=oracle,
        replace_base=True,
        targets=targets,
    )


class Phase2OraclePolicy:
    """Diagnostic-only end-to-end policy for fully-hidden-search upper bounds."""

    execution_mode = "dual"
    eager_enabled = False
    eager_semantics = "none"
    diagnostic_only = True
    assumes_fully_hidden_search = True
    search_latency_mode = "metadata_only"
    base_allocator = "dual-batch-slo-unaware-round-robin"

    def __init__(
        self,
        variant: str,
        *,
        oracle: Phase2CandidateTreeOracle,
        speculative_budget: int,
    ) -> None:
        if variant not in PHASE2_VARIANTS:
            raise ValueError("unknown Phase-2 oracle variant")
        self.variant = variant
        self.name = PHASE2_VARIANTS[variant]
        self.display_name = self.name.replace("-", " ").title()
        self.allocator = self.name + "-diagnostic"
        self.residual_selector = self.name
        self.uses_target_outcome = variant != "a"
        self.allows_base_node_replacement = variant == "d"
        self.oracle = oracle
        self.speculative_budget = speculative_budget
        self.search_budget_ratio = oracle.search_ratio

    def plan(self, snapshot: PolicySnapshot) -> StepPlan:
        targets = _targets(snapshot, self.oracle) if self.variant != "a" else None
        current = allocate_phase2_current(
            snapshot,
            oracle=self.oracle,
            per_request_budget=self.speculative_budget,
        )
        if self.variant == "a":
            return current
        if self.variant == "b":
            return allocate_phase2_within_request(
                snapshot, current, oracle=self.oracle, targets=targets
            )
        if self.variant == "c":
            return allocate_phase2_global(
                snapshot, current, oracle=self.oracle, targets=targets
            )
        return allocate_phase2_full_tree(
            snapshot, current, oracle=self.oracle, targets=targets
        )


@dataclass(frozen=True)
class _SnapshotDescriptor:
    eligible_index: int
    cycle: int
    now_ms: float
    active_requests: int
    pending_requests: int


def _active_bin(value: int, maximum: int) -> str:
    quarter = max(1, math.ceil(maximum / 4))
    return f"{min(3, value // quarter)}"


def _queue_bin(value: int, maximum: int) -> str:
    if value == 0:
        return "0"
    if value < maximum:
        return "1-to-active-cap"
    if value < maximum * 8:
        return "active-cap-to-8x"
    return "8x-plus"


def _sample_indices(
    descriptors: list[_SnapshotDescriptor], sample_size: int, max_active: int
) -> tuple[set[int], dict[str, Any]]:
    if sample_size < 1:
        raise ValueError("Phase-2 sample size must be positive")
    count = len(descriptors)
    target = min(sample_size, count)
    if target == count:
        selected = {item.eligible_index for item in descriptors}
        rule = "all eligible planning snapshots"
    else:
        groups: dict[tuple[str, str, str], list[int]] = {}
        for item in descriptors:
            phase = str(min(3, item.eligible_index * 4 // max(count, 1)))
            key = (
                phase,
                _active_bin(item.active_requests, max_active),
                _queue_bin(item.pending_requests, max_active),
            )
            groups.setdefault(key, []).append(item.eligible_index)
        if len(groups) >= target:
            retained = sorted(
                groups, key=lambda key: (-len(groups[key]), key)
            )[:target]
            quotas = {key: 1 for key in retained}
            groups = {key: groups[key] for key in retained}
        else:
            quotas = {
                key: max(1, math.floor(target * len(values) / count))
                for key, values in groups.items()
            }
            while sum(quotas.values()) > target:
                key = max(
                    (key for key in quotas if quotas[key] > 1),
                    key=lambda item: (quotas[item], item),
                )
                quotas[key] -= 1
        remainders = sorted(
            groups,
            key=lambda key: (
                target * len(groups[key]) / count - quotas[key],
                len(groups[key]),
                key,
            ),
            reverse=True,
        )
        cursor = 0
        while sum(quotas.values()) < target:
            key = remainders[cursor % len(remainders)]
            if quotas[key] < len(groups[key]):
                quotas[key] += 1
            cursor += 1
        selected = set()
        for key, values in groups.items():
            quota = quotas[key]
            for index in range(quota):
                position = min(
                    len(values) - 1,
                    math.floor((index + 0.5) * len(values) / quota),
                )
                selected.add(values[position])
        if len(selected) != target:
            raise AssertionError("stratified Phase-2 sample did not reach target size")
        rule = (
            "proportional deterministic sampling across temporal quartile, "
            "active-batch quartile, and queue-depth bin"
        )

    selected_descriptors = [
        item for item in descriptors if item.eligible_index in selected
    ]

    def coverage(items: Iterable[_SnapshotDescriptor]) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {
            "temporal_quartile": {},
            "active_batch_bin": {},
            "queue_depth_bin": {},
        }
        for item in items:
            phase = str(min(3, item.eligible_index * 4 // max(count, 1)))
            active = _active_bin(item.active_requests, max_active)
            queue = _queue_bin(item.pending_requests, max_active)
            for category, value in (
                ("temporal_quartile", phase),
                ("active_batch_bin", active),
                ("queue_depth_bin", queue),
            ):
                result[category][value] = result[category].get(value, 0) + 1
        return result

    return selected, {
        "rule": rule,
        "eligible_snapshots": count,
        "requested_sample_size": sample_size,
        "sampled_snapshots": len(selected),
        "population_coverage": coverage(descriptors),
        "sample_coverage": coverage(selected_descriptors),
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


@dataclass
class _ReplayAccumulator:
    snapshots: int = 0
    requests: int = 0
    requested_pool_nodes: int = 0
    realized_pool_nodes: int = 0
    base_pool_nodes: int = 0
    selected_nodes: int = 0
    committed_nodes: int = 0
    root_progress: int = 0
    pool_width_sum: int = 0
    pool_depth_sum: int = 0
    maximum_pool_width: int = 0
    maximum_pool_depth: int = 0
    accepted_lengths: list[float] = field(default_factory=list)
    residual_budgets: list[float] = field(default_factory=list)
    snapshot_committed: list[float] = field(default_factory=list)
    snapshot_efficiency: list[float] = field(default_factory=list)
    snapshot_total_progress: list[float] = field(default_factory=list)
    pool_coverage: dict[int, list[int]] = field(default_factory=dict)
    selected_coverage: dict[int, list[int]] = field(default_factory=dict)

    def add(
        self,
        snapshot: PolicySnapshot,
        plan: StepPlan,
        targets: dict[str, tuple[str, ...]],
    ) -> None:
        self.snapshots += 1
        self.requests += len(snapshot.normal_requests)
        committed = 0
        for request in snapshot.normal_requests:
            request_id = request.request_id
            tree = plan.candidate_trees[request_id]
            selected = plan.normal_trees[request_id]
            requested = tree.requested_candidate_nodes
            realized = len(tree.candidate_nodes)
            self.requested_pool_nodes += requested if requested is not None else realized
            self.realized_pool_nodes += realized
            self.base_pool_nodes += (
                (requested // tree.search_budget_ratio)
                if requested is not None
                else realized
            )
            self.pool_width_sum += tree.width
            self.pool_depth_sum += tree.depth
            self.maximum_pool_width = max(self.maximum_pool_width, tree.width)
            self.maximum_pool_depth = max(self.maximum_pool_depth, tree.depth)
            accepted = realized_candidate_progress(selected, targets[request_id])
            committed += accepted
            self.accepted_lengths.append(float(accepted))
            self.residual_budgets.append(
                float(
                    selected.candidate_budget
                    - plan.base_normal_trees[request_id].candidate_budget
                )
            )
            selected_ids = set(selected.selected_node_ids)
            pool_ids = set(tree.by_id)
            for depth, node_id in enumerate(targets[request_id], start=1):
                pool = self.pool_coverage.setdefault(depth, [0, 0])
                pool[1] += 1
                pool[0] += int(node_id in pool_ids)
                chosen = self.selected_coverage.setdefault(depth, [0, 0])
                chosen[1] += 1
                chosen[0] += int(node_id in selected_ids)
        selected_nodes = plan.total_candidates
        root_progress = len(snapshot.normal_requests)
        self.selected_nodes += selected_nodes
        self.committed_nodes += committed
        self.root_progress += root_progress
        self.snapshot_committed.append(float(committed))
        self.snapshot_efficiency.append(
            committed / selected_nodes if selected_nodes else 0.0
        )
        self.snapshot_total_progress.append(float(root_progress + committed))

    def finish(self) -> dict[str, Any]:
        return {
            "snapshots": self.snapshots,
            "candidate_pool_nodes_per_request": (
                self.realized_pool_nodes / self.requests if self.requests else 0.0
            ),
            "candidate_pool_nodes_per_cycle": (
                self.realized_pool_nodes / self.snapshots if self.snapshots else 0.0
            ),
            "requested_search_nodes": self.requested_pool_nodes,
            "realized_search_nodes": self.realized_pool_nodes,
            "achieved_search_ratio": (
                self.realized_pool_nodes / max(1, self.base_pool_nodes)
            ),
            "selected_verified_candidate_nodes": self.selected_nodes,
            "candidate_committed_tokens": self.committed_nodes,
            "candidate_committed_per_cycle": (
                self.committed_nodes / self.snapshots if self.snapshots else 0.0
            ),
            "candidate_committed_per_verified": (
                self.committed_nodes / self.selected_nodes if self.selected_nodes else 0.0
            ),
            "mean_accepted_candidate_length": (
                statistics.mean(self.accepted_lengths) if self.accepted_lengths else 0.0
            ),
            "p50_accepted_candidate_length": _percentile(self.accepted_lengths, 0.50),
            "p90_accepted_candidate_length": _percentile(self.accepted_lengths, 0.90),
            "root_progress": self.root_progress,
            "root_progress_per_cycle": (
                self.root_progress / self.snapshots if self.snapshots else 0.0
            ),
            "total_progress_per_cycle": (
                (self.root_progress + self.committed_nodes) / self.snapshots
                if self.snapshots
                else 0.0
            ),
            "target_path_pool_coverage_by_depth": {
                str(depth): covered / total
                for depth, (covered, total) in sorted(self.pool_coverage.items())
            },
            "target_path_selected_coverage_by_depth": {
                str(depth): covered / total
                for depth, (covered, total) in sorted(self.selected_coverage.items())
            },
            "residual_budget_distribution": {
                "mean": statistics.mean(self.residual_budgets)
                if self.residual_budgets
                else 0.0,
                "p50": _percentile(self.residual_budgets, 0.50),
                "p90": _percentile(self.residual_budgets, 0.90),
            },
            "maximum_pool_depth": self.maximum_pool_depth,
            "maximum_pool_width": self.maximum_pool_width,
            "mean_pool_depth": (
                self.pool_depth_sum / self.requests if self.requests else 0.0
            ),
            "mean_pool_width": (
                self.pool_width_sum / self.requests if self.requests else 0.0
            ),
        }


def _tree_hash(tree: CandidateTree) -> str:
    digest = hashlib.sha256()
    for node in tree.nodes:
        digest.update(
            (
                f"{node.node_id}|{node.parent_id}|{node.depth}|"
                f"{node.token_confidence:.17g}|{node.path_probability:.17g}\n"
            ).encode()
        )
    return digest.hexdigest()


def _validate_plan(
    snapshot: PolicySnapshot,
    plan: StepPlan,
    *,
    allow_base_replacement: bool,
) -> None:
    request_ids = {request.request_id for request in snapshot.normal_requests}
    if set(plan.normal_trees) != request_ids:
        raise AssertionError("Phase-2 plan changed the base request/root set")
    if plan.total_candidates > snapshot.roof_candidate_budget:
        raise AssertionError("Phase-2 plan exceeded B_verify")
    for request_id, selected in plan.normal_trees.items():
        tree = plan.candidate_trees[request_id]
        validate_prefix_closure(tree, selected)
        if not set(selected.selected_node_ids).issubset(tree.by_id):
            raise AssertionError("Phase-2 selected node is outside the candidate pool")
        if not allow_base_replacement and not set(
            plan.base_normal_trees[request_id].selected_node_ids
        ).issubset(selected.selected_node_ids):
            raise AssertionError("Phase-2 residual oracle removed base work")


def _replay_snapshot(
    diagnostic: PlanningDiagnostic,
    *,
    config: SimulatorConfig,
    search_ratio: int,
) -> tuple[
    PolicySnapshot,
    dict[str, StepPlan],
    dict[str, tuple[str, ...]],
    Phase2CandidateTreeOracle,
]:
    oracle = Phase2CandidateTreeOracle(
        config.seed,
        search_ratio=search_ratio,
        base_width=config.candidate_tree_width,
    )
    expanded_requests = []
    for request in diagnostic.snapshot.normal_requests:
        if request.candidate_tree is None:
            raise AssertionError("common replay requires a candidate tree")
        expanded_requests.append(
            replace(
                request,
                candidate_tree=oracle.expand_pool(
                    request.candidate_tree,
                    draft_confidence=request.draft_confidence,
                    committed_prefix_len=request.committed_prefix_len,
                ),
            )
        )
    snapshot = PolicySnapshot(
        tuple(expanded_requests),
        (),
        diagnostic.snapshot.roof_candidate_budget,
        diagnostic.snapshot.residual_draft_tokens,
    )
    targets = _targets(snapshot, oracle)
    target_accesses = oracle.target_access_count
    current = allocate_phase2_current(
        snapshot,
        oracle=oracle,
        per_request_budget=config.speculative_budget,
    )
    if oracle.target_access_count != target_accesses:
        raise AssertionError("A_r accessed the canonical target outcome")
    if search_ratio == 1:
        if (
            current.normal_budgets != diagnostic.plan.normal_budgets
            or current.normal_trees != diagnostic.plan.normal_trees
            or current.base_normal_trees != diagnostic.plan.base_normal_trees
        ):
            raise AssertionError("A_1x failed to reproduce Residual-Probability")
    within = allocate_phase2_within_request(
        snapshot, current, oracle=oracle, targets=targets
    )
    global_plan = allocate_phase2_global(
        snapshot, current, oracle=oracle, targets=targets
    )
    full = allocate_phase2_full_tree(
        snapshot, current, oracle=oracle, targets=targets
    )
    plans = {"a": current, "b": within, "c": global_plan, "d": full}
    for variant, plan in plans.items():
        _validate_plan(
            snapshot,
            plan,
            allow_base_replacement=variant == "d",
        )
    progress = {
        variant: sum(
            realized_candidate_progress(
                plan.normal_trees[request_id], targets[request_id]
            )
            for request_id in plan.normal_trees
        )
        for variant, plan in plans.items()
    }
    if not progress["b"] >= progress["a"]:
        raise AssertionError(f"B<A counterexample at cycle {diagnostic.cycle}")
    if not progress["c"] >= progress["b"]:
        raise AssertionError(f"C<B counterexample at cycle {diagnostic.cycle}")
    if not progress["d"] >= progress["c"]:
        raise AssertionError(f"D<C counterexample at cycle {diagnostic.cycle}")
    return snapshot, plans, targets, oracle


def _snapshot_record(
    diagnostic: PlanningDiagnostic,
    replayed: dict[
        int,
        tuple[
            PolicySnapshot,
            dict[str, StepPlan],
            dict[str, tuple[str, ...]],
            Phase2CandidateTreeOracle,
        ],
    ],
    config: SimulatorConfig,
) -> dict[str, Any]:
    largest_snapshot, largest_plans, targets, _ = replayed[8]
    requests = []
    for request in largest_snapshot.normal_requests:
        request_id = request.request_id
        pools = {}
        for ratio in SEARCH_RATIOS:
            tree = replayed[ratio][0].normal_requests[
                [
                    item.request_id
                    for item in replayed[ratio][0].normal_requests
                ].index(request_id)
            ].candidate_tree
            if tree is None:
                raise AssertionError("serialized Phase-2 pool is missing")
            pools[f"{ratio}x"] = {
                "requested_nodes": tree.requested_candidate_nodes,
                "realized_nodes": len(tree.candidate_nodes),
                "sha256": _tree_hash(tree),
            }
        target = targets[request_id]
        tree = largest_plans["a"].candidate_trees[request_id]
        requests.append(
            {
                "request_id": request_id,
                "prefix_epoch": request.prefix_epoch,
                "committed_prefix_len": request.committed_prefix_len,
                "slo_tpot_ms": request.slo_tpot_ms,
                "current_generated_tokens": request.committed_prefix_len,
                "acceptance_probability": request.acceptance_probability,
                "recent_acceptance_ratio": request.recent_acceptance_ratio,
                "draft_confidence": request.draft_confidence,
                "max_candidate_budget": request.max_budget,
                "base_budget": largest_plans["a"].base_normal_budgets[request_id],
                "base_selected_nodes": largest_plans["a"]
                .base_normal_trees[request_id]
                .selected_node_ids,
                "canonical_forest": {
                    "generator": "phase2-nested-root-branches-v1",
                    "seed": config.seed,
                    "base_width": config.candidate_tree_width,
                    "depth": tree.depth,
                    "pools": pools,
                },
                "canonical_target_trajectory": target,
                "target_path_probabilities": [
                    tree.by_id[node_id].path_probability for node_id in target
                ],
            }
        )
    return {
        "schema_version": "specrhythm.phase2-snapshot.v1",
        "cycle": diagnostic.cycle,
        "time_ms": diagnostic.now_ms,
        "active_request_ids": diagnostic.active_request_ids,
        "pending_arrived_requests": diagnostic.pending_requests,
        "base_request_ids": tuple(
            request.request_id for request in largest_snapshot.normal_requests
        ),
        "residual_roof": (
            largest_snapshot.roof_candidate_budget
            - sum(largest_plans["a"].base_normal_budgets.values())
        ),
        "candidate_roof": largest_snapshot.roof_candidate_budget,
        "verification_surface_inputs": {
            "surface": "T_verify(B_req, B_cand, C)",
            "request_root_positions": diagnostic.verification_request_count,
            "candidate_positions": diagnostic.verified_candidate_nodes,
            "context_modeled": False,
            "verify_latency_ms": diagnostic.verify_latency_ms,
        },
        "requests": requests,
    }


def _paired_gain(
    later: _ReplayAccumulator, earlier: _ReplayAccumulator
) -> dict[str, Any]:
    committed = [
        right - left
        for right, left in zip(later.snapshot_committed, earlier.snapshot_committed)
    ]
    efficiency = [
        right - left
        for right, left in zip(later.snapshot_efficiency, earlier.snapshot_efficiency)
    ]
    total = [
        right - left
        for right, left in zip(
            later.snapshot_total_progress, earlier.snapshot_total_progress
        )
    ]

    def values(rows: list[float]) -> dict[str, float]:
        return {
            "mean": statistics.mean(rows) if rows else 0.0,
            "p50": _percentile(rows, 0.50),
            "p90": _percentile(rows, 0.90),
        }

    return {
        "candidate_committed_per_cycle": values(committed),
        "candidate_committed_per_verified": values(efficiency),
        "total_progress_per_cycle": values(total),
        "fraction_snapshots_with_improvement": (
            sum(value > 0 for value in committed) / len(committed)
            if committed
            else 0.0
        ),
    }


def common_snapshot_replay(
    workload: Workload,
    config: SimulatorConfig,
    *,
    sample_size: int = 10_000,
    snapshot_sink: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    """Run two-pass baseline capture and paired Phase-2 counterfactual replay."""

    baseline_policy = ShapingDiagnosticPolicy(
        "residual-probability",
        speculative_budget=config.speculative_budget,
        n_max_slo=config.n_max_slo,
        residual_score=config.specrhythm_residual_score,
    )
    descriptors: list[_SnapshotDescriptor] = []
    eligible_index = 0

    def describe(diagnostic: PlanningDiagnostic) -> None:
        nonlocal eligible_index
        if not diagnostic.snapshot.normal_requests:
            return
        descriptors.append(
            _SnapshotDescriptor(
                eligible_index,
                diagnostic.cycle,
                diagnostic.now_ms,
                diagnostic.active_requests,
                diagnostic.pending_requests,
            )
        )
        eligible_index += 1

    first_pass = simulate(
        workload,
        baseline_policy,
        config,
        planning_sink=describe,
    )
    selected, sampling = _sample_indices(
        descriptors, sample_size, config.max_active_requests
    )
    accumulators = {
        (ratio, variant): _ReplayAccumulator()
        for ratio in SEARCH_RATIOS
        for variant in PHASE2_VARIANTS
    }
    replay_index = 0

    def replay(diagnostic: PlanningDiagnostic) -> None:
        nonlocal replay_index
        if not diagnostic.snapshot.normal_requests:
            return
        index = replay_index
        replay_index += 1
        if index not in selected:
            return
        replayed = {
            ratio: _replay_snapshot(
                diagnostic,
                config=config,
                search_ratio=ratio,
            )
            for ratio in SEARCH_RATIOS
        }
        for ratio, (snapshot, plans, targets, _) in replayed.items():
            for variant, plan in plans.items():
                accumulators[(ratio, variant)].add(snapshot, plan, targets)
        if snapshot_sink is not None:
            snapshot_sink(_snapshot_record(diagnostic, replayed, config))

    second_pass = simulate(
        workload,
        baseline_policy,
        config,
        planning_sink=replay,
    )
    if first_pass.summary.to_dict() != second_pass.summary.to_dict():
        raise AssertionError("two-pass baseline capture was not deterministic")

    rows = []
    headroom = []
    for ratio in SEARCH_RATIOS:
        for variant in PHASE2_VARIANTS:
            row = accumulators[(ratio, variant)].finish()
            row.update(
                {
                    "search_budget_ratio": ratio,
                    "variant": PHASE2_VARIANTS[variant],
                    "diagnostic_only": True,
                    "uses_target_outcome": variant != "a",
                    "assumes_fully_hidden_search": True,
                    "search_latency_mode": "metadata_only",
                }
            )
            rows.append(row)
        comparisons = (
            ("pool_gain", (ratio, "a"), (1, "a")),
            ("selector_gap", (ratio, "b"), (ratio, "a")),
            ("allocation_gap", (ratio, "c"), (ratio, "b")),
            ("base_tree_gap", (ratio, "d"), (ratio, "c")),
        )
        for label, later_key, earlier_key in comparisons:
            headroom.append(
                {
                    "search_budget_ratio": ratio,
                    "component": label,
                    **_paired_gain(
                        accumulators[later_key], accumulators[earlier_key]
                    ),
                }
            )
    return {
        "schema_version": "specrhythm.phase2-common-replay.v1",
        "model_status": "simulator-proxy-not-gpu-measured",
        "evidence_kind": "common-snapshot-structural-acceptance-headroom",
        "canonical_target_audit": {
            "historical_semantics": (
                "target child is selected from the candidate tree passed to verify"
            ),
            "phase2_semantics": (
                "target trajectory is frozen on the immutable 1x tree before pool expansion"
            ),
            "a_1x_strict_reproduction_checked_per_snapshot": True,
            "target_independent_of_search_ratio": True,
        },
        "sampling": sampling,
        "baseline_summary": first_pass.summary.to_dict(),
        "rows": rows,
        "headroom_decomposition": headroom,
    }


def run_phase2_end_to_end(
    workload: Workload,
    config: SimulatorConfig,
    *,
    variant: str,
    search_ratio: int,
) -> SimulationResult:
    oracle = Phase2CandidateTreeOracle(
        config.seed,
        search_ratio=search_ratio,
        base_width=config.candidate_tree_width,
    )
    policy = Phase2OraclePolicy(
        variant,
        oracle=oracle,
        speculative_budget=config.speculative_budget,
    )
    return simulate(
        workload,
        policy,
        config,
        candidate_tree_oracle=oracle,
    )
