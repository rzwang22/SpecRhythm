import math

import pytest

from specrhythm.policies.base import PolicySnapshot, RequestView
from specrhythm.policies.tree_aware import (
    allocate_adaserve_tree_aware,
    allocate_probability_residual,
    allocate_round_robin_residual,
    allocate_specrhythm_residual,
    allocate_specrhythm_tree_aware,
    expected_progress_for_plan,
)
from specrhythm.schema import WorkloadRequest
from specrhythm.simulator import Proposal, RuntimeRequest, eager_is_promotable
from specrhythm.tree import (
    CandidateTree,
    CandidateTreeNode,
    CandidateTreeOracle,
    expected_tree_progress,
    make_selected_tree,
)


def _tree(request_id, rows):
    root = CandidateTreeNode("root", None, 0, 1.0, 1.0)
    return CandidateTree(
        request_id,
        root,
        (root, *(CandidateTreeNode(*row) for row in rows)),
    )


def _view(request_id, slo, tree, elapsed=0.0, committed=0, max_budget=8):
    return RequestView(
        request_id=request_id,
        committed_prefix_len=committed,
        elapsed_decode_ms=elapsed,
        slo_tpot_ms=slo,
        recent_acceptance_ratio=0.8,
        draft_confidence=0.8,
        waiting_time_ms=0,
        max_budget=max_budget,
        candidate_tree=tree,
    )


def _snapshot(*requests, roof=8):
    return PolicySnapshot(tuple(requests), (), roof, roof)


def test_tree_prefix_closure_and_expected_progress():
    tree = _tree(
        "r",
        [
            ("a", "root", 1, 0.7, 0.7),
            ("b", "a", 2, 0.6, 0.42),
        ],
    )
    selected = make_selected_tree(tree, ("a", "b"))
    assert math.isclose(expected_tree_progress(tree, selected), 1.12)
    with pytest.raises(ValueError, match="prefix closed"):
        make_selected_tree(tree, ("b",))


def test_one_cycle_feasibility_boundary_counts_root_once():
    tree = _tree("r", [("a", "root", 1, 0.8, 0.8)])
    boundary = _view("r", 1, tree, elapsed=1.8, max_budget=1)
    above = _view("r", 1, tree, elapsed=1.800001, max_budget=1)
    assert math.isclose(boundary.required_total_progress, 1.8)
    assert math.isclose(boundary.required_candidate_progress, 0.8)
    assert math.isclose(boundary.maximum_attainable_candidate_progress, 0.8)
    assert math.isclose(boundary.maximum_attainable_total_progress, 1.8)
    assert boundary.one_cycle_feasible
    assert not above.one_cycle_feasible


def test_feasible_guard_excludes_infeasible_only_from_stage1():
    tree = _tree(
        "r",
        [
            ("a", "root", 1, 0.8, 0.8),
            ("b", "a", 2, 0.8, 0.64),
        ],
    )
    infeasible = _view("r", 1, tree, elapsed=4, max_budget=2)
    plan = allocate_specrhythm_tree_aware(
        _snapshot(infeasible, roof=2),
        n_max_slo=2,
        stage1_feasible_only=True,
    )
    assert not plan.one_cycle_feasible["r"]
    assert plan.slo_stage_budgets["r"] == 0
    assert plan.residual_stage_budgets["r"] == 2
    assert plan.normal_budgets["r"] == 2


def test_residual_zero_strictly_preserves_dual_batch_allocation():
    oracle = CandidateTreeOracle(1664)
    requests = tuple(
        _view(
            f"r{index}",
            40,
            oracle.tree(f"r{index}", 0, width=2, depth=4, draft_confidence=0.8),
            max_budget=4,
        )
        for index in range(2)
    )
    plan = allocate_specrhythm_residual(
        _snapshot(*requests, roof=4),
        per_request_budget=2,
        n_max_slo=4,
    )
    assert plan.normal_budgets == {"r0": 2, "r1": 2}
    assert plan.normal_trees == plan.base_normal_trees
    assert plan.slo_stage_budgets == {"r0": 0, "r1": 0}
    assert plan.residual_stage_budgets == {"r0": 0, "r1": 0}


def test_residual_additions_preserve_base_prefixes_and_roof():
    tree = CandidateTreeOracle(7).tree(
        "r", 0, width=2, depth=3, draft_confidence=0.9
    )
    plan = allocate_specrhythm_residual(
        _snapshot(_view("r", 1, tree, elapsed=1.5, max_budget=3), roof=3),
        per_request_budget=1,
        n_max_slo=3,
    )
    base = plan.base_normal_trees["r"]
    selected = plan.normal_trees["r"]
    assert base.candidate_budget == 1
    assert set(base.selected_node_ids).issubset(selected.selected_node_ids)
    assert selected.candidate_budget == 3
    assert plan.total_candidates <= 3
    # Construction validates prefix closure; rebuilding makes the invariant explicit.
    assert make_selected_tree(tree, selected.selected_node_ids) == selected


def test_residual_selectors_share_exact_base_forest_roof_and_utilization():
    oracle = CandidateTreeOracle(1664)
    requests = tuple(
        _view(
            f"r{index}",
            40 if index == 0 else 150,
            oracle.tree(f"r{index}", 0, width=2, depth=5, draft_confidence=0.8),
            elapsed=3 if index == 0 else 0,
            max_budget=5,
        )
        for index in range(3)
    )
    snapshot = _snapshot(*requests, roof=12)
    plans = (
        allocate_round_robin_residual(snapshot, per_request_budget=2),
        allocate_probability_residual(snapshot, per_request_budget=2),
        allocate_specrhythm_residual(
            snapshot, per_request_budget=2, n_max_slo=5
        ),
        allocate_specrhythm_residual(
            snapshot,
            per_request_budget=2,
            n_max_slo=5,
            stage1_feasible_only=True,
        ),
    )
    expected_base = plans[0].base_normal_trees
    expected_forest = plans[0].candidate_trees
    for plan in plans:
        assert plan.base_normal_trees == expected_base
        assert plan.candidate_trees == expected_forest
        assert plan.total_candidates == snapshot.roof_candidate_budget
        for request_id, base in expected_base.items():
            assert set(base.selected_node_ids).issubset(
                plan.normal_trees[request_id].selected_node_ids
            )
            assert make_selected_tree(
                expected_forest[request_id],
                plan.normal_trees[request_id].selected_node_ids,
            ) == plan.normal_trees[request_id]


def test_probability_residual_is_slo_unaware_and_can_differ_from_shaping():
    high_probability = _tree(
        "loose", [("loose-a", "root", 1, 0.95, 0.95)]
    )
    urgent = _tree("tight", [("tight-a", "root", 1, 0.4, 0.4)])
    snapshot = _snapshot(
        _view("loose", 150, high_probability, max_budget=1),
        _view("tight", 1, urgent, elapsed=2, max_budget=1),
        roof=1,
    )
    probability = allocate_probability_residual(snapshot, per_request_budget=0)
    shaping = allocate_specrhythm_residual(
        snapshot, per_request_budget=0, n_max_slo=1
    )
    assert probability.normal_budgets == {"loose": 1, "tight": 0}
    assert shaping.normal_budgets == {"loose": 0, "tight": 1}


def test_adaserve_figure5_style_selection():
    r0 = _tree(
        "r0",
        [
            ("r0-a", "root", 1, 0.7, 0.7),
            ("r0-b", "root", 1, 0.2, 0.2),
            ("r0-c", "r0-a", 2, 0.6, 0.42),
            ("r0-d", "r0-b", 2, 0.9, 0.18),
            ("r0-e", "r0-a", 2, 0.4, 0.28),
        ],
    )
    r1 = _tree(
        "r1",
        [
            ("r1-a", "root", 1, 0.5, 0.5),
            ("r1-b", "root", 1, 0.4, 0.4),
            ("r1-c", "r1-a", 2, 0.7, 0.35),
            ("r1-d", "r1-b", 2, 0.6, 0.24),
            ("r1-e", "r1-a", 2, 0.4, 0.2),
        ],
    )
    # Roots provide baseline progress 1.0. Candidate gaps 0.6 and 0.8 therefore
    # require r0-a, then r1-a+r1-b; the three residual picks are globally highest.
    p0 = _view("r0", 1, r0, elapsed=1.6)
    p1 = _view("r1", 1, r1, elapsed=1.8)
    plan = allocate_adaserve_tree_aware(_snapshot(p0, p1, roof=6), n_max_slo=4)
    assert plan.normal_trees["r0"].selected_node_ids == ("r0-a", "r0-c", "r0-e")
    assert plan.normal_trees["r1"].selected_node_ids == ("r1-a", "r1-b", "r1-c")


def test_width_one_degenerates_to_sequence_sd():
    oracle = CandidateTreeOracle(1664)
    tree = oracle.tree("r", 0, width=1, depth=4, draft_confidence=0.8)
    request = _view("r", 100, tree)
    plan = allocate_adaserve_tree_aware(_snapshot(request, roof=4), n_max_slo=4)
    selected = plan.normal_trees["r"]
    assert selected.candidate_budget == 4
    assert [tree.by_id[node].depth for node in selected.selected_node_ids] == [1, 2, 3, 4]


def test_adaserve_and_specrhythm_have_distinct_residual_allocators():
    tree_a = _tree("a", [("a1", "root", 1, 0.9, 0.9)])
    tree_b = _tree("b", [("b1", "root", 1, 0.8, 0.8)])
    relaxed = _view("a", 100, tree_a)
    urgent = _view("b", 1, tree_b, elapsed=2.1)
    snapshot = _snapshot(relaxed, urgent, roof=1)
    ada = allocate_adaserve_tree_aware(snapshot, n_max_slo=0)
    spec = allocate_specrhythm_tree_aware(
        snapshot, n_max_slo=0, residual_score="urgency-path-probability"
    )
    assert ada.normal_budgets == {"a": 1, "b": 0}
    assert spec.normal_budgets == {"a": 0, "b": 1}


def test_adaserve_throughput_stage_ignores_urgency():
    high = _tree("high", [("h", "root", 1, 0.9, 0.9)])
    low = _tree("low", [("l", "root", 1, 0.4, 0.4)])
    plan = allocate_adaserve_tree_aware(
        _snapshot(_view("high", 100, high), _view("low", 1, low, elapsed=5), roof=1),
        n_max_slo=0,
    )
    assert plan.normal_budgets == {"high": 1, "low": 0}


def test_slo_stage_respects_per_request_cap():
    tree = CandidateTreeOracle(1).tree("r", 0, width=1, depth=4, draft_confidence=1)
    plan = allocate_adaserve_tree_aware(
        _snapshot(_view("r", 1, tree, elapsed=10), roof=4), n_max_slo=2
    )
    # Stage 2 may fill after the SLO cap; the first two nodes are sufficient to expose
    # the cap by repeating with no throughput budget left for another request.
    assert plan.normal_trees["r"].selected_node_ids[:2] == tuple(
        node.node_id for node in tree.candidate_nodes[:2]
    )
    assert plan.slo_stage_budgets["r"] == 2


def test_tree_node_accounting_is_conserved_in_simulation():
    from specrhythm.policies import SpecRhythmPolicy
    from specrhythm.schema import Workload
    from specrhythm.simulator import SimulatorConfig, simulate

    workload = Workload(
        [WorkloadRequest("r", 0, 1, 8, 40, acceptance_probability=0.7)]
    )
    summary = simulate(
        workload,
        SpecRhythmPolicy(enable_eager=False),
        SimulatorConfig(
            max_active_requests=1,
            roof_candidate_budget=4,
            max_request_budget=4,
            candidate_tree_width=2,
            candidate_tree_depth=4,
        ),
    ).summary
    assert summary.tree_drafted_nodes == (
        summary.tree_verified_nodes
        + summary.tree_invalidated_nodes
        + summary.tree_discarded_at_eos_nodes
    )
    assert summary.tree_accepted_nodes <= summary.tree_verified_nodes


def test_tree_eager_depends_only_on_predicted_path():
    tree = _tree(
        "r",
        [
            ("a", "root", 1, 0.8, 0.8),
            ("x", "root", 1, 0.2, 0.2),
            ("b", "a", 2, 0.7, 0.56),
        ],
    )
    selected = make_selected_tree(tree, ("a", "x", "b"))
    parent = Proposal("r", 0, 0, 3, 3, "normal", 0, tree, selected)
    eager = Proposal("r", 3, 1, 1, 1, "eager", 0, None, None, ("a", "b"))
    runtime = RuntimeRequest(
        WorkloadRequest("r", 0, 1, 10, 40),
        slot=0,
        admitted_at_ms=0,
        committed_prefix_len=3,
        prefix_epoch=1,
    )
    assert eager_is_promotable(
        runtime,
        parent,
        eager,
        parent_fully_accepted=False,
        accepted_branch_node_ids=("a", "b"),
    )
    assert not eager_is_promotable(
        runtime,
        parent,
        eager,
        parent_fully_accepted=False,
        accepted_branch_node_ids=("a",),
    )


def test_expected_progress_helper_matches_selected_probabilities():
    tree = CandidateTreeOracle(2).tree("r", 0, width=2, depth=2, draft_confidence=0.8)
    request = _view("r", 10, tree)
    plan = allocate_specrhythm_tree_aware(
        _snapshot(request, roof=2), n_max_slo=2
    )
    measured = expected_progress_for_plan((request,), plan)
    assert math.isclose(measured["r"], sum(
        tree.by_id[node].path_probability
        for node in plan.normal_trees["r"].selected_node_ids
    ))


def test_tree_verifier_returns_branch_and_committed_progress():
    tree = _tree(
        "r",
        [
            ("a", "root", 1, 0.8, 0.8),
            ("x", "root", 1, 0.2, 0.2),
            ("b", "a", 2, 0.7, 0.56),
        ],
    )
    selected = make_selected_tree(tree, ("a", "x", "b"))
    outcome = CandidateTreeOracle(
        1, injected_branches={("r", 0): ("a", "b")}
    ).verify(tree, selected, committed_prefix_len=0, acceptance_probability=0.0)
    assert outcome.accepted_branch_node_ids == ("a", "b")
    assert outcome.committed_progress == 3
