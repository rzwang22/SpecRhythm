from specrhythm.policies import ARPolicy, SpecRhythmPolicy
from specrhythm.schema import Workload, WorkloadRequest
from specrhythm.simulator import SimulatorConfig, simulate


def _workload():
    return Workload(
        [
            WorkloadRequest(
                request_id=f"r{index}",
                arrival_time_ms=index * 2,
                input_tokens=64,
                output_tokens=12,
                slo_tpot_ms=40 if index % 2 == 0 else 100,
                task="tight" if index % 2 == 0 else "relaxed",
                acceptance_probability=0.8,
            )
            for index in range(8)
        ]
    )


def test_simulation_completes_and_accounts_metrics():
    config = SimulatorConfig(max_active_requests=4, roof_candidate_budget=12, max_request_budget=4)
    result = simulate(_workload(), SpecRhythmPolicy(), config)
    assert result.summary.completed_requests == 8
    assert 0 <= result.summary.slo_attainment <= 1
    assert result.summary.throughput_tokens_per_s >= result.summary.goodput_tokens_per_s
    assert result.summary.proposed_draft_tokens >= result.summary.accepted_draft_tokens
    assert {item.request_id for item in result.requests} == {f"r{index}" for index in range(8)}


def test_simulation_is_deterministic():
    config = SimulatorConfig(seed=9, roof_candidate_budget=8, max_request_budget=3)
    first = simulate(_workload(), SpecRhythmPolicy(), config)
    second = simulate(_workload(), SpecRhythmPolicy(), config)
    assert first == second


def test_ar_proposes_no_draft_tokens():
    result = simulate(_workload(), ARPolicy(), SimulatorConfig())
    assert result.summary.proposed_draft_tokens == 0
    assert result.summary.accepted_draft_tokens == 0


def test_eager_continuation_requires_a_fully_accepted_prefix():
    config = SimulatorConfig(
        roof_candidate_budget=2,
        max_request_budget=2,
        verify_base_ms=10,
        verify_per_request_ms=0,
        verify_per_candidate_ms=0,
        draft_per_candidate_ms=0.01,
        seed=0,
    )
    accepting = Workload([WorkloadRequest("accept", 0, 8, 6, 1, acceptance_probability=1.0)])
    rejecting = Workload([WorkloadRequest("reject", 0, 8, 6, 1, acceptance_probability=0.5)])

    accepted_result = simulate(accepting, SpecRhythmPolicy(), config)
    rejected_result = simulate(rejecting, SpecRhythmPolicy(), config)

    assert accepted_result.summary.eager_promotions > 0
    assert accepted_result.summary.eager_invalidations == 0
    assert rejected_result.summary.eager_invalidations > 0
