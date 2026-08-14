"""Frozen tree-aware allocation specifications for Phase-A control-plane tests."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Callable, Optional

from specrhythm.policies.base import PolicySnapshot, RequestView, StepPlan
from specrhythm.policies.baselines import allocate_round_robin
from specrhythm.tree import (
    CandidateTree,
    CandidateTreeNode,
    SelectedProposalTree,
    expected_tree_progress,
    make_selected_tree,
    select_sequence_path,
)


def _fallback_sequence_tree(request: RequestView) -> CandidateTree:
    root = CandidateTreeNode("root", None, 0, 1.0, 1.0)
    nodes = [root]
    parent = root
    for depth in range(1, request.max_budget + 1):
        node = CandidateTreeNode(
            node_id=f"seq-{depth}",
            parent_id=parent.node_id,
            depth=depth,
            token_confidence=request.acceptance_benefit,
            path_probability=request.acceptance_benefit**depth,
        )
        nodes.append(node)
        parent = node
    return CandidateTree(request.request_id, root, tuple(nodes))


@dataclass
class _SelectionState:
    request: RequestView
    tree: CandidateTree
    selected: list[str]
    expected: float = 0.0

    @property
    def selected_set(self) -> set[str]:
        return set(self.selected)

    def eligible(self, cap: int) -> list[CandidateTreeNode]:
        if len(self.selected) >= cap:
            return []
        chosen = self.selected_set
        parent_ids = (self.tree.root.node_id, *self.selected)
        return [
            node
            for parent_id in parent_ids
            for node in self.tree.children_by_parent.get(parent_id, ())
            if node.node_id not in chosen
        ]

    def add(self, node: CandidateTreeNode) -> None:
        self.selected.append(node.node_id)
        self.expected += node.path_probability

    def proposal(self) -> SelectedProposalTree:
        return make_selected_tree(self.tree, self.selected)


def _states(
    snapshot: PolicySnapshot,
    initial_trees: Optional[dict[str, SelectedProposalTree]] = None,
) -> dict[str, _SelectionState]:
    result = {}
    for request in snapshot.normal_requests:
        state = _SelectionState(
            request=request,
            tree=request.candidate_tree or _fallback_sequence_tree(request),
            selected=[],
        )
        initial = (initial_trees or {}).get(request.request_id)
        if initial is not None:
            for node_id in initial.selected_node_ids:
                eligible = {node.node_id: node for node in state.eligible(request.max_budget)}
                if node_id not in eligible:
                    raise ValueError("initial proposal tree is not prefix closed")
                state.add(eligible[node_id])
        result[request.request_id] = state
    return result


def _best_eligible(
    state: _SelectionState,
    cap: int,
    score: Callable[[CandidateTreeNode], tuple[float, str]],
) -> CandidateTreeNode:
    return max(state.eligible(cap), key=score)


def _allocate_frontier(
    states: dict[str, _SelectionState],
    remaining: int,
    *,
    cap: Callable[[_SelectionState], int],
    enabled: Callable[[_SelectionState], bool],
    score: Callable[[_SelectionState, CandidateTreeNode], float],
    on_add: Callable[[_SelectionState], None],
) -> int:
    """Allocate from prefix frontiers with lazy score-version invalidation."""

    versions = {request_id: 0 for request_id in states}
    heap: list[tuple[float, str, str, int]] = []

    def push(state: _SelectionState) -> None:
        if not enabled(state):
            return
        request_id = state.request.request_id
        version = versions[request_id]
        for node in state.eligible(cap(state)):
            heapq.heappush(
                heap,
                (-score(state, node), request_id, node.node_id, version),
            )

    for state in states.values():
        push(state)
    while remaining > 0 and heap:
        _, request_id, node_id, version = heapq.heappop(heap)
        state = states[request_id]
        if version != versions[request_id] or not enabled(state):
            continue
        eligible = {node.node_id: node for node in state.eligible(cap(state))}
        node = eligible.get(node_id)
        if node is None:
            continue
        state.add(node)
        remaining -= 1
        on_add(state)
        versions[request_id] += 1
        push(state)
    return remaining


def _finish(
    states: dict[str, _SelectionState],
    slo_stage: dict[str, int],
    base_trees: Optional[dict[str, SelectedProposalTree]] = None,
) -> StepPlan:
    has_base = base_trees is not None
    base_trees = base_trees or {}
    trees = {request_id: state.proposal() for request_id, state in states.items()}
    return StepPlan(
        normal_budgets={key: tree.candidate_budget for key, tree in trees.items()},
        normal_trees=trees,
        candidate_trees={key: state.tree for key, state in states.items()},
        expected_progress={key: state.expected for key, state in states.items()},
        requested_progress_gap={
            key: state.request.continuous_progress_gap for key, state in states.items()
        },
        slo_stage_budgets=slo_stage,
        residual_stage_budgets={
            key: len(state.selected)
            - base_trees.get(key, SelectedProposalTree((), (), 0)).candidate_budget
            - slo_stage[key]
            for key, state in states.items()
        },
        base_normal_budgets={
            key: tree.candidate_budget for key, tree in base_trees.items()
        },
        base_normal_trees=dict(base_trees) if has_base else {},
        required_total_progress={
            key: state.request.required_total_progress for key, state in states.items()
        },
        required_candidate_progress={
            key: state.request.required_candidate_progress for key, state in states.items()
        },
        maximum_attainable_candidate_progress={
            key: state.request.maximum_attainable_candidate_progress
            for key, state in states.items()
        },
        maximum_attainable_total_progress={
            key: state.request.maximum_attainable_total_progress
            for key, state in states.items()
        },
        one_cycle_feasible={
            key: state.request.one_cycle_feasible for key, state in states.items()
        },
    )


def allocate_adaserve_tree_aware(
    snapshot: PolicySnapshot, *, n_max_slo: int
) -> StepPlan:
    """AdaServe two-stage tree selection using unrounded continuous gaps.

    Root/baseline progress is reported separately by the simulator and is not charged to
    this repository's candidate roof. Stage 1 processes requests by decreasing A_i and
    accumulates path-probability mass. Stage 2 ranks remaining nodes globally by path
    probability only; SLO urgency is deliberately absent from the throughput stage.
    """

    states = _states(snapshot)
    slo_stage = {request_id: 0 for request_id in states}
    remaining = snapshot.roof_candidate_budget
    ordered = sorted(
        states.values(),
        key=lambda state: (-state.request.continuous_progress_gap, state.request.request_id),
    )
    for state in ordered:
        attainable = sum(node.path_probability for node in state.tree.candidate_nodes)
        target = min(state.request.candidate_progress_gap, attainable)
        cap = min(state.request.max_budget, n_max_slo)
        while remaining > 0 and state.expected + 1e-12 < target:
            eligible = state.eligible(cap)
            if not eligible:
                break
            state.add(max(eligible, key=lambda node: (node.path_probability, node.node_id)))
            slo_stage[state.request.request_id] += 1
            remaining -= 1

    remaining = _allocate_frontier(
        states,
        remaining,
        cap=lambda state: state.request.max_budget,
        enabled=lambda state: bool(state.eligible(state.request.max_budget)),
        score=lambda state, node: node.path_probability,
        on_add=lambda state: None,
    )
    return _finish(states, slo_stage)


def _residual_urgency(state: _SelectionState) -> float:
    gap = state.request.candidate_progress_gap
    return 1.0 + max(0.0, gap - state.expected)


def allocate_specrhythm_tree_aware(
    snapshot: PolicySnapshot,
    *,
    n_max_slo: int,
    residual_score: str = "urgency-path-probability",
    stage1_feasible_only: bool = False,
    base_trees: Optional[dict[str, SelectedProposalTree]] = None,
) -> StepPlan:
    """SpecRhythm §4.4 control-plane interpretation frozen in the design document."""

    if residual_score not in {"path-probability", "urgency-path-probability"}:
        raise ValueError("unknown SpecRhythm residual score")
    states = _states(snapshot, base_trees)
    slo_stage = {request_id: 0 for request_id in states}
    base_candidates = sum(
        tree.candidate_budget for tree in (base_trees or {}).values()
    )
    if base_candidates > snapshot.roof_candidate_budget:
        raise ValueError("base proposal trees exceed the candidate roof")
    remaining = snapshot.roof_candidate_budget - base_candidates

    # Stage 1: among requests with an uncovered projected gap, greedily select the
    # eligible node with greatest residual-urgency-weighted expected progress.
    remaining = _allocate_frontier(
        states,
        remaining,
        cap=lambda state: min(state.request.max_budget, n_max_slo),
        enabled=lambda state: (
            state.request.candidate_progress_gap > 0
            and state.expected + 1e-12 < state.request.candidate_progress_gap
            and (not stage1_feasible_only or state.request.one_cycle_feasible)
        ),
        score=lambda state, node: _residual_urgency(state) * node.path_probability,
        on_add=lambda state: slo_stage.__setitem__(
            state.request.request_id, slo_stage[state.request.request_id] + 1
        ),
    )

    # Stage 2 diagnostic ablation: the default is the paper-style residual urgency
    # multiplied by node expected progress. The path-only variant is retained verbatim.
    remaining = _allocate_frontier(
        states,
        remaining,
        cap=lambda state: state.request.max_budget,
        enabled=lambda state: bool(state.eligible(state.request.max_budget)),
        score=lambda state, node: node.path_probability
        * (
            _residual_urgency(state)
            if residual_score == "urgency-path-probability"
            else 1.0
        ),
        on_add=lambda state: None,
    )
    return _finish(states, slo_stage, base_trees)


def allocate_specrhythm_residual(
    snapshot: PolicySnapshot,
    *,
    per_request_budget: int,
    n_max_slo: int,
    residual_score: str = "urgency-path-probability",
    stage1_feasible_only: bool = False,
) -> StepPlan:
    """Freeze Dual-Batch work, then shape only otherwise-unused candidate roof."""

    base_trees = _dual_batch_base_trees(snapshot, per_request_budget)
    return allocate_specrhythm_tree_aware(
        snapshot,
        n_max_slo=n_max_slo,
        residual_score=residual_score,
        stage1_feasible_only=stage1_feasible_only,
        base_trees=base_trees,
    )


def _dual_batch_base_trees(
    snapshot: PolicySnapshot, per_request_budget: int
) -> dict[str, SelectedProposalTree]:
    """Materialize the exact Dual-Batch allocation on the shared candidate forest."""

    base_budgets = allocate_round_robin(snapshot, per_request_budget)
    base_trees = {}
    for request in snapshot.normal_requests:
        tree = request.candidate_tree or _fallback_sequence_tree(request)
        base_trees[request.request_id] = select_sequence_path(
            tree, base_budgets[request.request_id]
        )
    return base_trees


def allocate_round_robin_residual(
    snapshot: PolicySnapshot, *, per_request_budget: int
) -> StepPlan:
    """Freeze Dual-Batch, then fill residual roof uniformly across requests."""

    base_trees = _dual_batch_base_trees(snapshot, per_request_budget)
    states = _states(snapshot, base_trees)
    remaining = snapshot.roof_candidate_budget - sum(
        tree.candidate_budget for tree in base_trees.values()
    )
    while remaining > 0:
        added = False
        for request in snapshot.normal_requests:
            state = states[request.request_id]
            eligible = state.eligible(request.max_budget)
            if not eligible:
                continue
            state.add(
                max(eligible, key=lambda node: (node.path_probability, node.node_id))
            )
            remaining -= 1
            added = True
            if remaining == 0:
                break
        if not added:
            break
    return _finish(states, {request_id: 0 for request_id in states}, base_trees)


def allocate_probability_residual(
    snapshot: PolicySnapshot, *, per_request_budget: int
) -> StepPlan:
    """Freeze Dual-Batch, then fill residual roof by path probability only."""

    base_trees = _dual_batch_base_trees(snapshot, per_request_budget)
    states = _states(snapshot, base_trees)
    remaining = snapshot.roof_candidate_budget - sum(
        tree.candidate_budget for tree in base_trees.values()
    )
    _allocate_frontier(
        states,
        remaining,
        cap=lambda state: state.request.max_budget,
        enabled=lambda state: bool(state.eligible(state.request.max_budget)),
        score=lambda state, node: node.path_probability,
        on_add=lambda state: None,
    )
    return _finish(states, {request_id: 0 for request_id in states}, base_trees)


def expected_progress_for_plan(
    requests: tuple[RequestView, ...], plan: StepPlan
) -> dict[str, float]:
    by_id = {request.request_id: request for request in requests}
    return {
        request_id: expected_tree_progress(by_id[request_id].candidate_tree, selected)
        for request_id, selected in plan.normal_trees.items()
        if by_id[request_id].candidate_tree is not None
    }
