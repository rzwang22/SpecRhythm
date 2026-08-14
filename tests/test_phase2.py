import gzip
import json
from dataclasses import asdict, replace

from specrhythm.cli import SIMULATION_POLICY_ORDER, main
from specrhythm.phase2 import (
    PHASE2_VARIANTS,
    SEARCH_RATIOS,
    Phase2CandidateTreeOracle,
    Phase2OraclePolicy,
    allocate_phase2_current,
    allocate_phase2_full_tree,
    allocate_phase2_global,
    allocate_phase2_within_request,
    common_snapshot_replay,
    realized_candidate_progress,
    run_phase2_end_to_end,
)
from specrhythm.policies.base import PolicySnapshot, RequestView
from specrhythm.policies.specrhythm import ShapingDiagnosticPolicy
from specrhythm.policies.tree_aware import allocate_probability_residual
from specrhythm.schema import Workload, WorkloadRequest
from specrhythm.simulator import SimulatorConfig, simulate
from specrhythm.tree import CandidateTreeOracle, make_selected_tree, validate_prefix_closure


def _config(**overrides):
    values = {
        "max_active_requests": 4,
        "roof_candidate_budget": 8,
        "max_request_budget": 4,
        "speculative_budget": 2,
        "candidate_tree_width": 2,
        "candidate_tree_depth": 4,
        "verify_base_ms": 4,
        "verify_per_request_ms": 0.1,
        "verify_per_candidate_ms": 0.05,
        "draft_per_candidate_ms": 0.02,
        "seed": 1664,
    }
    values.update(overrides)
    return SimulatorConfig(**values)


def _workload(count=12):
    return Workload(
        [
            WorkloadRequest(
                request_id=f"r{index}",
                arrival_time_ms=index * 2,
                input_tokens=32,
                output_tokens=8,
                slo_tpot_ms=40 if index % 2 == 0 else 150,
                task="code" if index % 2 == 0 else "summarization",
                acceptance_probability=0.75,
                draft_confidence=0.8,
            )
            for index in range(count)
        ]
    )


def _snapshot(config):
    oracle = CandidateTreeOracle(config.seed)
    requests = tuple(
        RequestView(
            request_id=f"r{index}",
            committed_prefix_len=index,
            elapsed_decode_ms=10,
            slo_tpot_ms=40 if index == 0 else 150,
            recent_acceptance_ratio=0.75,
            draft_confidence=0.8,
            waiting_time_ms=0,
            max_budget=config.max_request_budget,
            candidate_tree=oracle.tree(
                f"r{index}",
                index,
                width=config.candidate_tree_width,
                depth=config.candidate_tree_depth,
                draft_confidence=0.8,
            ),
            acceptance_probability=1.0,
            prefix_epoch=index,
        )
        for index in range(2)
    )
    return PolicySnapshot(requests, (), config.roof_candidate_budget, 100)


def test_phase2_pools_are_nested_prefix_closed_and_node_stable():
    trees = []
    for ratio in SEARCH_RATIOS:
        oracle = Phase2CandidateTreeOracle(1664, search_ratio=ratio, base_width=2)
        trees.append(
            oracle.tree(
                "r",
                7,
                width=2,
                depth=4,
                draft_confidence=0.8,
            )
        )
    for smaller, larger in zip(trees, trees[1:]):
        assert set(smaller.by_id).issubset(larger.by_id)
        for node_id, node in smaller.by_id.items():
            assert larger.by_id[node_id] == node
    for ratio, tree in zip(SEARCH_RATIOS, trees):
        assert len(tree.candidate_nodes) == len(trees[0].candidate_nodes) * ratio
        assert tree.requested_candidate_nodes == len(trees[0].candidate_nodes) * ratio
        selected = make_selected_tree(
            tree, tuple(node.node_id for node in tree.candidate_nodes)
        )
        validate_prefix_closure(tree, selected)


def test_canonical_target_is_stable_across_pool_ratios():
    trajectories = []
    for ratio in SEARCH_RATIOS:
        oracle = Phase2CandidateTreeOracle(1664, search_ratio=ratio, base_width=2)
        tree = oracle.tree(
            "r", 3, width=2, depth=4, draft_confidence=0.8
        )
        trajectories.append(
            oracle.canonical_trajectory(
                tree,
                committed_prefix_len=3,
                acceptance_probability=1.0,
            )
        )
    assert len(set(trajectories)) == 1
    assert len(trajectories[0]) == 4


def test_a1_reproduces_residual_probability_without_target_access():
    config = _config()
    snapshot = _snapshot(config)
    historical = allocate_probability_residual(
        snapshot, per_request_budget=config.speculative_budget
    )
    oracle = Phase2CandidateTreeOracle(
        config.seed, search_ratio=1, base_width=config.candidate_tree_width
    )
    expanded = PolicySnapshot(
        tuple(
            replace(
                request,
                candidate_tree=oracle.expand_pool(
                    request.candidate_tree,
                    draft_confidence=request.draft_confidence,
                    committed_prefix_len=request.committed_prefix_len,
                ),
            )
            for request in snapshot.normal_requests
        ),
        (),
        snapshot.roof_candidate_budget,
        snapshot.residual_draft_tokens,
    )
    current = allocate_phase2_current(
        expanded,
        oracle=oracle,
        per_request_budget=config.speculative_budget,
    )
    assert oracle.target_access_count == 0
    assert current.normal_budgets == historical.normal_budgets
    assert current.normal_trees == historical.normal_trees
    assert current.base_normal_trees == historical.base_normal_trees


def test_oracle_variants_preserve_constraints_and_dominate_per_snapshot():
    config = _config(roof_candidate_budget=6)
    original = _snapshot(config)
    oracle = Phase2CandidateTreeOracle(
        config.seed, search_ratio=4, base_width=config.candidate_tree_width
    )
    snapshot = PolicySnapshot(
        tuple(
            replace(
                request,
                candidate_tree=oracle.expand_pool(
                    request.candidate_tree,
                    draft_confidence=request.draft_confidence,
                    committed_prefix_len=request.committed_prefix_len,
                ),
            )
            for request in original.normal_requests
        ),
        (),
        original.roof_candidate_budget,
        original.residual_draft_tokens,
    )
    current = allocate_phase2_current(
        snapshot, oracle=oracle, per_request_budget=config.speculative_budget
    )
    within = allocate_phase2_within_request(snapshot, current, oracle=oracle)
    global_plan = allocate_phase2_global(snapshot, current, oracle=oracle)
    full = allocate_phase2_full_tree(snapshot, current, oracle=oracle)
    assert within.normal_budgets == current.normal_budgets
    assert global_plan.total_candidates == current.total_candidates
    assert full.total_candidates == current.total_candidates
    assert set(full.normal_trees) == set(current.normal_trees)
    for plan in (current, within, global_plan, full):
        assert plan.total_candidates <= snapshot.roof_candidate_budget
        for request_id, selected in plan.normal_trees.items():
            assert set(selected.selected_node_ids).issubset(
                plan.candidate_trees[request_id].by_id
            )
            validate_prefix_closure(plan.candidate_trees[request_id], selected)
    for plan in (current, within, global_plan):
        for request_id, base in plan.base_normal_trees.items():
            assert set(base.selected_node_ids).issubset(
                plan.normal_trees[request_id].selected_node_ids
            )
    targets = {
        request.request_id: oracle.canonical_trajectory(
            request.candidate_tree,
            committed_prefix_len=request.committed_prefix_len,
            acceptance_probability=request.acceptance_probability,
        )
        for request in snapshot.normal_requests
    }
    progress = [
        sum(
            realized_candidate_progress(plan.normal_trees[key], targets[key])
            for key in plan.normal_trees
        )
        for plan in (current, within, global_plan, full)
    ]
    assert progress == sorted(progress)


def test_phase2_policy_metadata_is_diagnostic_and_not_in_default_order():
    config = _config()
    for variant in PHASE2_VARIANTS:
        oracle = Phase2CandidateTreeOracle(
            config.seed, search_ratio=2, base_width=config.candidate_tree_width
        )
        policy = Phase2OraclePolicy(
            variant,
            oracle=oracle,
            speculative_budget=config.speculative_budget,
        )
        assert policy.diagnostic_only
        assert policy.assumes_fully_hidden_search
        assert policy.search_latency_mode == "metadata_only"
        assert policy.uses_target_outcome is (variant != "a")
        assert policy.name not in SIMULATION_POLICY_ORDER


def test_common_replay_is_deterministic_and_reports_all_headroom_components():
    snapshots = []
    first = common_snapshot_replay(
        _workload(), _config(), sample_size=20, snapshot_sink=snapshots.append
    )
    second = common_snapshot_replay(_workload(), _config(), sample_size=20)
    assert first == second
    assert first["sampling"]["sampled_snapshots"] == 20
    assert len(first["rows"]) == len(SEARCH_RATIOS) * len(PHASE2_VARIANTS)
    assert {row["component"] for row in first["headroom_decomposition"]} == {
        "pool_gain",
        "selector_gap",
        "allocation_gap",
        "base_tree_gap",
    }
    assert first["canonical_target_audit"][
        "a_1x_strict_reproduction_checked_per_snapshot"
    ]
    assert "pending_arrived_requests" in snapshots[0]
    request_snapshot = snapshots[0]["requests"][0]
    assert "acceptance_probability" in request_snapshot
    assert "recent_acceptance_ratio" in request_snapshot
    assert "draft_confidence" in request_snapshot
    assert "max_candidate_budget" in request_snapshot


def test_planning_queue_depth_excludes_future_arrivals():
    config = _config(max_active_requests=1)
    policy = ShapingDiagnosticPolicy(
        "residual-probability",
        speculative_budget=config.speculative_budget,
        n_max_slo=config.n_max_slo,
    )

    def request(request_id, arrival_time_ms):
        return WorkloadRequest(
            request_id=request_id,
            arrival_time_ms=arrival_time_ms,
            input_tokens=32,
            output_tokens=2,
            slo_tpot_ms=40,
            task="code",
            acceptance_probability=0.75,
            draft_confidence=0.8,
        )

    future_depths = []
    simulate(
        Workload([request("r0", 0), request("r1", 1_000)]),
        policy,
        config,
        planning_sink=lambda diagnostic: future_depths.append(
            diagnostic.pending_requests
        ),
    )
    overloaded_depths = []
    simulate(
        Workload([request("r0", 0), request("r1", 0)]),
        policy,
        config,
        planning_sink=lambda diagnostic: overloaded_depths.append(
            diagnostic.pending_requests
        ),
    )
    assert future_depths[0] == 0
    assert overloaded_depths[0] == 1


def test_a1_end_to_end_reproduces_historical_metrics_and_all_variants_conserve():
    config = _config()
    workload = _workload()
    baseline_policy = ShapingDiagnosticPolicy(
        "residual-probability",
        speculative_budget=config.speculative_budget,
        n_max_slo=config.n_max_slo,
    )
    baseline = simulate(workload, baseline_policy, config)
    a1 = run_phase2_end_to_end(workload, config, variant="a", search_ratio=1)
    assert a1.requests == baseline.requests
    for field in (
        "goodput_tokens_per_s",
        "slo_attainment",
        "raw_throughput_tokens_per_s",
        "mean_queueing_latency_ms",
        "candidate_committed_tokens_per_cycle",
        "total_progress_per_cycle",
    ):
        assert getattr(a1.summary, field) == getattr(baseline.summary, field)

    for variant in PHASE2_VARIANTS:
        result = run_phase2_end_to_end(
            workload, config, variant=variant, search_ratio=2
        )
        summary = result.summary
        assert summary.diagnostic_only
        assert summary.uses_target_outcome is (variant != "a")
        assert summary.assumes_fully_hidden_search
        assert summary.search_latency_mode == "metadata_only"
        assert summary.normal_drafted_tokens == (
            summary.verified_tokens
            + summary.invalidated_tokens
            + summary.discarded_at_eos_tokens
        )
        assert summary.tree_drafted_nodes == (
            summary.tree_verified_nodes
            + summary.tree_invalidated_nodes
            + summary.tree_discarded_at_eos_nodes
        )
        assert summary.accepted_tokens <= summary.verified_tokens


def test_phase2_cli_commands_write_diagnostic_outputs(tmp_path):
    workload_path = tmp_path / "workload.jsonl"
    config_path = tmp_path / "simulator.json"
    replay_path = tmp_path / "replay.json"
    snapshots_path = tmp_path / "snapshots.jsonl.gz"
    simulation_path = tmp_path / "simulation.json"
    _workload(count=4).save_jsonl(workload_path)
    config_path.write_text(json.dumps(asdict(_config())))

    replay_status = main(
        [
            "phase2-replay",
            "--workload",
            str(workload_path),
            "--config",
            str(config_path),
            "--sample-size",
            "4",
            "--output",
            str(replay_path),
            "--snapshot-output",
            str(snapshots_path),
        ]
    )
    assert replay_status == 0
    replay = json.loads(replay_path.read_text())
    assert replay["schema_version"] == "specrhythm.phase2-common-replay.v1"
    assert replay["sampling"]["sampled_snapshots"] == 4
    with gzip.open(snapshots_path, "rt", encoding="utf-8") as handle:
        snapshots = [json.loads(line) for line in handle]
    assert len(snapshots) == 4

    simulation_status = main(
        [
            "phase2-simulate",
            "--workload",
            str(workload_path),
            "--config",
            str(config_path),
            "--variant",
            PHASE2_VARIANTS["b"],
            "--search-ratio",
            "2",
            "--output",
            str(simulation_path),
        ]
    )
    assert simulation_status == 0
    simulation = json.loads(simulation_path.read_text())
    assert simulation["policy"] == PHASE2_VARIANTS["b"]
    assert simulation["search_budget_ratio"] == 2
    assert simulation["evidence_kind"] == "fully-hidden-search-system-upper-bound"
    assert simulation["deployable_measured_result"] is False
