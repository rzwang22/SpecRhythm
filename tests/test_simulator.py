import math

from specrhythm.cli import POLICY_ORDER, _policy
from specrhythm.policies import (
    AdaServeStylePolicy,
    ARPolicy,
    DualBatchPolicy,
    DualEagerPolicy,
    SerialSDPolicy,
    SpecRhythmPolicy,
)
from specrhythm.schema import Workload, WorkloadRequest
from specrhythm.simulator import (
    AcceptanceOracle,
    Proposal,
    RuntimeRequest,
    SimulatorConfig,
    cycle_latency_ms,
    eager_is_promotable,
    simulate,
)


def _workload(output_tokens=12, acceptance=0.8, count=8):
    return Workload(
        [
            WorkloadRequest(
                request_id=f"r{index}",
                arrival_time_ms=index * 2,
                input_tokens=64,
                output_tokens=output_tokens,
                slo_tpot_ms=40 if index % 2 == 0 else 100,
                task="tight" if index % 2 == 0 else "relaxed",
                acceptance_probability=acceptance,
            )
            for index in range(count)
        ]
    )


def _config(**overrides):
    values = {
        "max_active_requests": 4,
        "roof_candidate_budget": 12,
        "max_request_budget": 4,
        "verify_base_ms": 10,
        "verify_per_request_ms": 0,
        "verify_per_candidate_ms": 0.1,
        "draft_per_candidate_ms": 0.2,
        "seed": 9,
    }
    values.update(overrides)
    return SimulatorConfig(**values)


def test_simulation_completes_and_accounts_metrics():
    result = simulate(_workload(), SpecRhythmPolicy(), _config())
    assert result.summary.completed_requests == 8
    assert 0 <= result.summary.slo_attainment <= 1
    assert result.summary.throughput_tokens_per_s >= result.summary.goodput_tokens_per_s
    assert result.summary.verified_tokens >= result.summary.accepted_tokens
    assert {item.request_id for item in result.requests} == {f"r{index}" for index in range(8)}


def test_cumulative_ablation_order_and_metadata_are_explicit():
    expected = (
        "ar",
        "serial-sd",
        "adaserve",
        "dual-batch",
        "dual-eager",
        "shaping",
        "specrhythm",
    )
    assert POLICY_ORDER == expected
    metadata = {
        name: (
            _policy(name, _config()).execution_mode,
            _policy(name, _config()).allocator,
            _policy(name, _config()).eager_semantics,
        )
        for name in expected
    }
    assert metadata == {
        "ar": ("ar", "none", "none"),
        "serial-sd": ("serial", "slo-unaware-round-robin", "none"),
        "adaserve": ("serial", "slo-aware-two-pass-proxy", "none"),
        "dual-batch": ("dual", "slo-unaware-round-robin", "none"),
        "dual-eager": ("dual", "slo-unaware-round-robin", "guarded-rolling"),
        "shaping": ("dual", "slo-aware-two-pass-proxy", "none"),
        "specrhythm": ("dual", "slo-aware-two-pass-proxy", "guarded-rolling"),
    }
    assert _policy("adaserve", _config()).display_name == (
        "AdaServe-style simulator baseline"
    )


def test_simulation_is_deterministic():
    first = simulate(_workload(), SpecRhythmPolicy(), _config())
    second = simulate(_workload(), SpecRhythmPolicy(), _config())
    assert first == second


def test_ar_proposes_no_draft_tokens():
    result = simulate(_workload(), ARPolicy(), _config())
    assert result.summary.execution_mode == "ar"
    assert result.summary.proposed_draft_tokens == 0
    assert result.summary.accepted_draft_tokens == 0
    assert result.summary.normal_drafted_proposals == 0
    assert result.summary.eager_drafted_proposals == 0


def test_ar_processes_the_full_active_batch_each_step():
    workload = _workload(output_tokens=2, count=4)
    workload = Workload(
        [
            WorkloadRequest(
                request.request_id,
                0,
                request.input_tokens,
                request.output_tokens,
                request.slo_tpot_ms,
            )
            for request in workload.requests
        ]
    )
    config = _config(
        max_active_requests=4,
        verify_base_ms=10,
        verify_per_request_ms=0,
        verify_per_candidate_ms=0,
    )
    result = simulate(workload, ARPolicy(), config)
    assert result.summary.measurement_ms == 20
    assert {request.service_latency_ms for request in result.requests} == {20}


def test_output_length_one_has_no_eager_promotion_or_post_eos_proposal():
    result = simulate(
        _workload(output_tokens=1, acceptance=1.0, count=1),
        SpecRhythmPolicy(),
        _config(),
    )
    assert result.summary.eager_promotions == 0
    assert result.summary.eager_drafted_tokens == 0
    assert all(
        state.finished
        and state.normal_proposal is None
        and state.eager_proposal is None
        for state in result.final_states
    )


def test_partial_acceptance_invalidates_eager_proposal():
    oracle = AcceptanceOracle(
        seed=0,
        max_k=2,
        traces={
            ("r0", 0): (True, False),
            ("r0", 2): (True, True),
        },
    )
    result = simulate(
        _workload(output_tokens=8, acceptance=1.0, count=1),
        SpecRhythmPolicy(),
        _config(roof_candidate_budget=2, max_request_budget=2),
        acceptance_oracle=oracle,
    )
    assert result.summary.eager_invalidations >= 1
    assert result.summary.invalidated_tokens >= 1


def test_full_acceptance_only_promotes_the_corresponding_parent():
    request = WorkloadRequest("r", 0, 1, 10, 40)
    runtime = RuntimeRequest(
        request, slot=0, admitted_at_ms=0, committed_prefix_len=3, prefix_epoch=1
    )
    parent = Proposal("r", 0, 0, 2, 2, "normal", 0)
    corresponding = Proposal("r", 3, 1, 2, 2, "eager", 0)
    wrong_parent = Proposal("r", 4, 1, 2, 2, "eager", 0)
    assert eager_is_promotable(runtime, parent, corresponding, parent_fully_accepted=True)
    assert not eager_is_promotable(runtime, parent, wrong_parent, parent_fully_accepted=True)

    oracle = AcceptanceOracle(
        seed=0,
        max_k=2,
        traces={
            ("r0", 0): (True, True),
            ("r0", 3): (True, True),
            ("r0", 6): (True, True),
        },
    )
    result = simulate(
        _workload(output_tokens=8, acceptance=1.0, count=1),
        SpecRhythmPolicy(),
        _config(roof_candidate_budget=2, max_request_budget=2),
        acceptance_oracle=oracle,
    )
    assert result.summary.eager_promotions == 1
    assert result.summary.promoted_tokens > 0
    assert result.summary.verified_tokens == (
        result.summary.normal_drafted_tokens + result.summary.promoted_tokens
    )


def test_prefix_epoch_mismatch_prevents_promotion():
    request = WorkloadRequest("r", 0, 1, 10, 40)
    runtime = RuntimeRequest(
        request, slot=0, admitted_at_ms=0, committed_prefix_len=3, prefix_epoch=1
    )
    parent = Proposal("r", 0, 0, 2, 2, "normal", 0)
    stale = Proposal("r", 3, 2, 2, 2, "eager", 0)
    assert not eager_is_promotable(runtime, parent, stale, parent_fully_accepted=True)


def test_different_budgets_replay_the_same_prefix_trace():
    oracle = AcceptanceOracle(seed=1664, max_k=8)
    trace = oracle.trace("request", 12, 0.7)
    for budget in (1, 3, 8):
        expected = 0
        for accepted in trace[:budget]:
            if not accepted:
                break
            expected += 1
        assert oracle.accepted_prefix("request", 12, budget, 0.7) == expected


def test_serial_and_dual_batch_use_distinct_latency_formulas():
    assert cycle_latency_ms("serial", draft_ms=7, verify_ms=11) == 18
    assert cycle_latency_ms("dual", draft_ms=7, verify_ms=11) == 11
    serial = simulate(_workload(count=1), SerialSDPolicy(2), _config())
    dual = simulate(_workload(count=1), DualBatchPolicy(2), _config())
    assert serial.summary.execution_mode == "serial"
    assert dual.summary.execution_mode == "dual"

    paired = _workload(output_tokens=8, acceptance=1.0, count=2)
    latency_config = _config(
        max_active_requests=2,
        roof_candidate_budget=4,
        max_request_budget=2,
        verify_base_ms=4,
        verify_per_candidate_ms=0,
        draft_per_candidate_ms=2,
    )
    serial = simulate(paired, SerialSDPolicy(2), latency_config)
    dual = simulate(paired, DualBatchPolicy(2), latency_config)
    assert serial.summary.measurement_ms > dual.summary.measurement_ms


def test_serial_and_dual_share_the_exact_candidate_allocation():
    workload = Workload(
        [
            WorkloadRequest(
                f"r{index}",
                0,
                64,
                9,
                40 if index % 2 == 0 else 100,
                acceptance_probability=0.75,
            )
            for index in range(4)
        ]
    )
    config = _config(
        max_active_requests=4,
        roof_candidate_budget=6,
        max_request_budget=3,
    )
    serial = simulate(workload, SerialSDPolicy(3), config)
    dual = simulate(workload, DualBatchPolicy(3), config)
    assert serial.summary.normal_drafted_tokens == dual.summary.normal_drafted_tokens
    assert serial.summary.verified_tokens == dual.summary.verified_tokens
    assert serial.summary.accepted_tokens == dual.summary.accepted_tokens
    assert serial.summary.draft_compute_ms == dual.summary.draft_compute_ms
    assert serial.summary.verify_compute_ms == dual.summary.verify_compute_ms
    assert serial.summary.measurement_ms > dual.summary.measurement_ms


def test_adaserve_uses_serial_latency_and_never_eager():
    result = simulate(_workload(), AdaServeStylePolicy(), _config())
    assert cycle_latency_ms(result.summary.execution_mode, 7, 11) == 18
    assert result.summary.execution_mode == "serial"
    assert result.summary.allocator == "slo-aware-two-pass-proxy"
    assert result.summary.eager_semantics == "none"
    assert result.summary.eager_drafted_tokens == 0
    assert result.summary.eager_promotions == 0
    assert result.summary.eager_invalidations == 0


def test_summary_marks_unmodeled_context_and_proxy_parameters():
    summary = simulate(_workload(count=1), ARPolicy(), _config()).summary
    assert summary.input_tokens_modeled is False
    assert summary.context_dependent_latency_modeled is False
    assert summary.proxy_parameter_status == {
        "draft_latency": "context-independent simulator proxy",
        "verify_latency": "context-independent simulator proxy",
        "acceptance": "workload proxy; not GPU-measured",
        "draft_confidence": "workload proxy; not GPU-calibrated",
        "candidate_roof": "configured simulator proxy",
    }
    assert summary.simulator_parameters["speculative_budget"] == 4


def _tight_loose_pressure_workload():
    return Workload(
        [
            WorkloadRequest(
                f"zz-tight-{index}" if index % 2 == 0 else f"aa-loose-{index}",
                0,
                1,
                4,
                3 if index % 2 == 0 else 100,
                task="tight" if index % 2 == 0 else "loose",
                acceptance_probability=1.0,
                draft_confidence=1.0,
            )
            for index in range(4)
        ]
    )


def _tight_loose_pressure_config():
    return _config(
        max_active_requests=4,
        roof_candidate_budget=3,
        max_request_budget=4,
        verify_base_ms=4,
        verify_per_request_ms=0,
        verify_per_candidate_ms=0,
        draft_per_candidate_ms=1,
        speculative_budget=4,
        seed=1,
    )


def test_adaserve_improves_tight_attainment_over_serial_sd_constructively():
    workload = _tight_loose_pressure_workload()
    config = _tight_loose_pressure_config()
    serial = simulate(workload, SerialSDPolicy(4), config)
    adaserve = simulate(workload, AdaServeStylePolicy(), config)
    assert adaserve.summary.class_metrics["tight"]["attainment"] > (
        serial.summary.class_metrics["tight"]["attainment"]
    )


def test_shaping_improves_tight_attainment_over_dual_batch_constructively():
    workload = _tight_loose_pressure_workload()
    config = _tight_loose_pressure_config()
    dual = simulate(workload, DualBatchPolicy(4), config)
    shaping = simulate(workload, SpecRhythmPolicy(enable_eager=False), config)
    assert shaping.summary.class_metrics["tight"]["attainment"] > (
        dual.summary.class_metrics["tight"]["attainment"]
    )


def test_high_confidence_short_parent_eager_can_improve_goodput():
    workload = Workload(
        [
            WorkloadRequest(
                "r0",
                0,
                1,
                4,
                3,
                task="tight",
                acceptance_probability=1.0,
                draft_confidence=1.0,
            )
        ]
    )
    config = _config(
        max_active_requests=1,
        roof_candidate_budget=2,
        max_request_budget=4,
        verify_base_ms=4,
        verify_per_request_ms=0,
        verify_per_candidate_ms=0,
        draft_per_candidate_ms=0.5,
        speculative_budget=2,
        seed=1,
    )
    dual = simulate(workload, DualBatchPolicy(2), config)
    eager = simulate(workload, DualEagerPolicy(2), config)
    assert eager.summary.eager_promotions > 0
    assert eager.summary.goodput_tokens_per_s > dual.summary.goodput_tokens_per_s


def test_all_proposal_tokens_have_one_terminal_disposition():
    result = simulate(_workload(output_tokens=9, count=4), SpecRhythmPolicy(), _config())
    summary = result.summary
    assert summary.normal_drafted_tokens + summary.eager_drafted_tokens == (
        summary.verified_tokens
        + summary.invalidated_tokens
        + summary.discarded_at_eos_tokens
    )
    assert summary.accepted_tokens <= summary.verified_tokens
    assert summary.promoted_tokens <= summary.eager_drafted_tokens
    assert math.isclose(
        summary.draft_compute_ms,
        (summary.normal_drafted_tokens + summary.eager_drafted_tokens)
        * _config().draft_per_candidate_ms,
    )


def test_finished_requests_never_retain_or_generate_proposals():
    for policy in (
        ARPolicy(),
        SerialSDPolicy(2),
        AdaServeStylePolicy(),
        DualBatchPolicy(2),
        DualEagerPolicy(2),
        SpecRhythmPolicy(enable_eager=False),
        SpecRhythmPolicy(),
    ):
        result = simulate(_workload(output_tokens=3, count=2), policy, _config())
        assert all(
            state.finished
            and state.normal_proposal is None
            and state.eager_proposal is None
            and state.committed_prefix_len == 3
            for state in result.final_states
        )


def test_request_latency_splits_queueing_service_and_end_to_end():
    workload = Workload(
        [
            WorkloadRequest("first", 0, 1, 2, 100),
            WorkloadRequest("queued", 1, 1, 1, 100),
        ]
    )
    result = simulate(
        workload,
        ARPolicy(),
        _config(
            max_active_requests=1,
            verify_base_ms=10,
            verify_per_request_ms=0,
            verify_per_candidate_ms=0,
        ),
    )
    by_id = {request.request_id: request for request in result.requests}
    assert by_id["queued"].queueing_latency_ms == 19
    assert by_id["queued"].service_latency_ms == 10
    assert by_id["queued"].decode_latency_ms == 29
    assert by_id["queued"].decode_latency_ms == (
        by_id["queued"].queueing_latency_ms + by_id["queued"].service_latency_ms
    )
    for request in result.requests:
        assert request.decode_latency_ms == (
            request.queueing_latency_ms + request.service_latency_ms
        )


def test_overload_increases_queueing_delay_and_slo_violations():
    def arrivals(gap):
        return Workload(
            [
                    WorkloadRequest(f"r{index}", index * gap, 1, 4, 12)
                for index in range(10)
            ]
        )

    config = _config(
        max_active_requests=1,
        verify_base_ms=10,
        verify_per_request_ms=0,
        verify_per_candidate_ms=0,
    )
    low = simulate(arrivals(50), ARPolicy(), config)
    high = simulate(arrivals(1), ARPolicy(), config)
    assert high.summary.mean_queueing_latency_ms > low.summary.mean_queueing_latency_ms
    assert high.summary.slo_attainment < low.summary.slo_attainment


def test_all_modes_satisfy_proposal_and_token_accounting():
    policies = (
        ARPolicy(),
        SerialSDPolicy(2),
        AdaServeStylePolicy(),
        DualBatchPolicy(2),
        DualEagerPolicy(2),
        SpecRhythmPolicy(enable_eager=False),
        SpecRhythmPolicy(),
    )
    for policy in policies:
        summary = simulate(_workload(output_tokens=7), policy, _config()).summary
        assert summary.normal_drafted_proposals + summary.eager_drafted_proposals == (
            summary.verified_proposals
            + summary.invalidated_proposals
            + summary.discarded_at_eos_proposals
        )
        assert summary.normal_drafted_tokens + summary.eager_drafted_tokens == (
            summary.verified_tokens
            + summary.invalidated_tokens
            + summary.discarded_at_eos_tokens
        )
        assert (
            summary.promoted_proposals
            + summary.eager_invalidations
            + summary.eager_discarded_at_eos_proposals
            == summary.eager_drafted_proposals
        )
        assert (
            summary.promoted_tokens
            + summary.eager_invalidated_tokens
            + summary.eager_discarded_at_eos_tokens
            == summary.eager_drafted_tokens
        )
        assert math.isclose(
            summary.eager_promotion_proposal_ratio
            + summary.eager_invalidation_proposal_ratio
            + summary.eager_eos_discard_proposal_ratio,
            1.0 if summary.eager_drafted_proposals else 0.0,
        )
        assert math.isclose(
            summary.eager_promotion_token_ratio
            + summary.eager_invalidation_token_ratio
            + summary.eager_eos_discard_token_ratio,
            1.0 if summary.eager_drafted_tokens else 0.0,
        )
        assert 0 <= summary.draft_compute_waste_ratio <= 1
        assert 0 <= summary.eager_compute_waste_ratio <= 1
