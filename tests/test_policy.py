from specrhythm.policies.base import PolicySnapshot, RequestView
from specrhythm.policies.baselines import DualBatchPolicy, SerialSDPolicy
from specrhythm.policies.specrhythm import (
    EAGER_MIN_FULL_ACCEPTANCE_PROBABILITY,
    AdaServeStylePolicy,
    DualEagerPolicy,
    SpecRhythmPolicy,
)


def _view(
    request_id,
    slo,
    acceptance=0.8,
    confidence=0.9,
    proposal_budget=0,
    full_probability=0.0,
    elapsed=100,
    waiting=20,
    max_budget=4,
):
    return RequestView(
        request_id=request_id,
        committed_prefix_len=1,
        elapsed_decode_ms=elapsed,
        slo_tpot_ms=slo,
        recent_acceptance_ratio=acceptance,
        draft_confidence=confidence,
        waiting_time_ms=waiting,
        max_budget=max_budget,
        proposal_budget=proposal_budget,
        parent_full_acceptance_probability=full_probability,
    )


def _snapshot(normal=(), eager=(), roof=4, residual=8):
    return PolicySnapshot(
        normal_requests=tuple(normal),
        eager_requests=tuple(eager),
        roof_candidate_budget=roof,
        residual_draft_tokens=residual,
    )


def test_tight_slo_gets_more_budget_under_candidate_pressure():
    snapshot = _snapshot(
        normal=(_view("tight", 20), _view("relaxed", 200)),
        roof=2,
        residual=2,
    )
    plan = SpecRhythmPolicy().plan(snapshot)
    assert plan.normal_budgets == {"tight": 2, "relaxed": 0}


def test_policy_respects_global_and_per_request_budgets():
    requests = tuple(_view(f"r{index}", 40, 0.9) for index in range(10))
    snapshot = _snapshot(normal=requests, roof=7, residual=7)
    plan = SpecRhythmPolicy().plan(snapshot)
    assert sum(plan.normal_budgets.values()) <= 7
    assert all(0 <= value <= 4 for value in plan.normal_budgets.values())


def test_shaping_only_disables_eager_continuation():
    snapshot = _snapshot(
        normal=(_view("normal", 40),),
        eager=(
            _view(
                "eager",
                40,
                proposal_budget=2,
                full_probability=0.9,
            ),
        ),
        roof=2,
        residual=4,
    )
    plan = SpecRhythmPolicy(enable_eager=False).plan(snapshot)
    assert plan.normal_budgets["normal"] > 0
    assert plan.eager_budgets == {}


def test_eager_budget_is_not_copied_from_parent_budget():
    snapshot = _snapshot(
        eager=(
            _view(
                "tight",
                10,
                confidence=1.0,
                proposal_budget=4,
                full_probability=0.6,
                elapsed=100,
                waiting=20,
            ),
        ),
        roof=4,
        residual=4,
    )
    plan = SpecRhythmPolicy().plan(snapshot)
    assert 0 < plan.eager_budgets["tight"] < 4


def test_rejection_heavy_long_parent_suppresses_eager():
    low = _view(
        "low",
        10,
        acceptance=0.2,
        confidence=0.9,
        proposal_budget=4,
        full_probability=0.2**4 * 0.9,
    )
    plan = SpecRhythmPolicy().plan(_snapshot(eager=(low,), roof=4, residual=4))
    assert plan.eager_budgets == {}


def test_eager_admission_threshold_is_explicit():
    assert EAGER_MIN_FULL_ACCEPTANCE_PROBABILITY == 0.10


def test_high_confidence_short_parent_admits_eager():
    high = _view(
        "high",
        10,
        acceptance=0.95,
        confidence=0.95,
        proposal_budget=1,
        full_probability=0.95 * 0.95,
    )
    plan = SpecRhythmPolicy().plan(_snapshot(eager=(high,), roof=4, residual=4))
    assert plan.eager_budgets["high"] > 0


def test_residual_budget_prefers_higher_acceptance_value():
    high = _view("high", 100, 0.95)
    low = _view("low", 100, 0.2)
    snapshot = _snapshot(normal=(high, low), roof=3, residual=3)
    plan = SpecRhythmPolicy().plan(snapshot)
    assert plan.normal_budgets["high"] > plan.normal_budgets["low"]


def test_adaserve_style_is_serial_slo_aware_and_never_eager():
    policy = AdaServeStylePolicy()
    snapshot = _snapshot(
        normal=(_view("tight", 20), _view("loose", 200)),
        eager=(
            _view("eager", 20, proposal_budget=1, full_probability=0.95),
        ),
        roof=2,
        residual=2,
    )
    plan = policy.plan(snapshot)
    assert policy.execution_mode == "serial"
    assert policy.allocator == "slo-aware-two-pass-proxy"
    assert plan.normal_budgets["tight"] > plan.normal_budgets["loose"]
    assert plan.eager_budgets == {}


def test_dual_eager_keeps_slo_unaware_normal_allocation():
    policy = DualEagerPolicy(budget=2)
    plan = policy.plan(
        _snapshot(
            normal=(_view("tight", 20), _view("loose", 200)),
            roof=2,
            residual=2,
        )
    )
    assert plan.normal_budgets == {"tight": 1, "loose": 1}


def test_serial_and_dual_policies_return_identical_candidate_plans():
    snapshot = _snapshot(
        normal=(
            _view("a", 20, max_budget=4),
            _view("b", 200, max_budget=2),
            _view("c", 50, max_budget=3),
        ),
        roof=7,
        residual=7,
    )
    assert SerialSDPolicy(4).plan(snapshot) == DualBatchPolicy(4).plan(snapshot)
