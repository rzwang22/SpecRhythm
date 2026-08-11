from specrhythm.policies.base import PolicySnapshot, RequestView
from specrhythm.policies.specrhythm import SpecRhythmPolicy


def _view(request_id, slo, acceptance=0.8):
    return RequestView(
        request_id=request_id,
        delivered_tokens=1,
        elapsed_decode_ms=100,
        slo_tpot_ms=slo,
        acceptance_ratio=acceptance,
        draft_confidence=acceptance,
        waiting_time_ms=20,
        max_budget=4,
    )


def test_tight_slo_gets_no_less_budget_than_relaxed_slo():
    snapshot = PolicySnapshot(
        requests=(_view("tight", 40), _view("relaxed", 150)),
        roof_candidate_budget=4,
        residual_draft_tokens=8,
    )
    plan = SpecRhythmPolicy().plan(snapshot)
    assert plan.budgets["tight"] >= plan.budgets["relaxed"]
    assert "tight" in plan.eager_request_ids


def test_policy_respects_global_and_per_request_budgets():
    requests = tuple(_view(f"r{index}", 40, 0.9) for index in range(10))
    snapshot = PolicySnapshot(requests, roof_candidate_budget=7, residual_draft_tokens=3)
    plan = SpecRhythmPolicy().plan(snapshot)
    assert sum(plan.budgets.values()) <= 7
    assert all(0 <= value <= 4 for value in plan.budgets.values())
    assert plan.eager_request_ids == ()


def test_shaping_only_disables_eager_continuation():
    snapshot = PolicySnapshot(
        requests=(_view("tight", 40),),
        roof_candidate_budget=4,
        residual_draft_tokens=4,
    )
    plan = SpecRhythmPolicy(enable_eager=False).plan(snapshot)
    assert plan.budgets["tight"] > 0
    assert plan.eager_request_ids == ()


def test_residual_budget_prefers_higher_acceptance_value():
    high = RequestView("high", 1, 10, 100, 0.95, 0.95, 0, 4)
    low = RequestView("low", 1, 10, 100, 0.2, 0.2, 0, 4)
    snapshot = PolicySnapshot((high, low), roof_candidate_budget=2, residual_draft_tokens=0)
    plan = SpecRhythmPolicy().plan(snapshot)
    assert plan.budgets["high"] > plan.budgets["low"]
