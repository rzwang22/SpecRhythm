import math

from specrhythm.policies import ARPolicy, DualBatchPolicy, SerialSDPolicy, SpecRhythmPolicy
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


def test_simulation_is_deterministic():
    first = simulate(_workload(), SpecRhythmPolicy(), _config())
    second = simulate(_workload(), SpecRhythmPolicy(), _config())
    assert first == second


def test_ar_proposes_no_draft_tokens():
    result = simulate(_workload(), ARPolicy(), _config())
    assert result.summary.execution_mode == "ar"
    assert result.summary.proposed_draft_tokens == 0
    assert result.summary.accepted_draft_tokens == 0


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
    assert result.summary.promoted_tokens == 2
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
    assert serial.summary.measurement_ms == 40
    assert dual.summary.measurement_ms == 28
    assert serial.summary.measurement_ms > dual.summary.measurement_ms


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
    for policy in (ARPolicy(), SerialSDPolicy(2), DualBatchPolicy(2), SpecRhythmPolicy()):
        result = simulate(_workload(output_tokens=3, count=2), policy, _config())
        assert all(
            state.finished
            and state.normal_proposal is None
            and state.eager_proposal is None
            and state.committed_prefix_len == 3
            for state in result.final_states
        )
