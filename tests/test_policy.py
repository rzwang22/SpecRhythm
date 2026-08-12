from specrhythm.policies.base import PolicySnapshot, RequestView
from specrhythm.policies.specrhythm import SpecRhythmPolicy


def _view(request_id, slo, acceptance=0.8, proposal_budget=0):
    return RequestView(
        request_id=request_id,
        committed_prefix_len=1,
        elapsed_decode_ms=100,
        slo_tpot_ms=slo,
        acceptance_estimate=acceptance,
        waiting_time_ms=20,
        max_budget=4,
        proposal_budget=proposal_budget,
    )


def _snapshot(normal=(), eager=(), roof=4, residual=8):
    return PolicySnapshot(
        normal_requests=tuple(normal),
        eager_requests=tuple(eager),
        roof_candidate_budget=roof,
        residual_draft_tokens=residual,
    )


def test_tight_slo_gets_no_less_budget_than_relaxed_slo():
    snapshot = _snapshot(normal=(_view("tight", 40), _view("relaxed", 150)))
    plan = SpecRhythmPolicy().plan(snapshot)
    assert plan.normal_budgets["tight"] >= plan.normal_budgets["relaxed"]


def test_policy_respects_global_and_per_request_budgets():
    requests = tuple(_view(f"r{index}", 40, 0.9) for index in range(10))
    snapshot = _snapshot(normal=requests, roof=7, residual=7)
    plan = SpecRhythmPolicy().plan(snapshot)
    assert sum(plan.normal_budgets.values()) <= 7
    assert all(0 <= value <= 4 for value in plan.normal_budgets.values())


def test_shaping_only_disables_eager_continuation():
    snapshot = _snapshot(
        normal=(_view("normal", 40),),
        eager=(_view("eager", 40, proposal_budget=2),),
        roof=2,
        residual=4,
    )
    plan = SpecRhythmPolicy(enable_eager=False).plan(snapshot)
    assert plan.normal_budgets["normal"] > 0
    assert plan.eager_budgets == {}


def test_eager_uses_parent_proposal_budget_from_separate_view():
    snapshot = _snapshot(
        eager=(_view("tight", 40, proposal_budget=2),),
        roof=2,
        residual=2,
    )
    plan = SpecRhythmPolicy().plan(snapshot)
    assert plan.normal_budgets == {}
    assert plan.eager_budgets == {"tight": 2}


def test_residual_budget_prefers_higher_acceptance_value():
    high = _view("high", 100, 0.95)
    low = _view("low", 100, 0.2)
    snapshot = _snapshot(normal=(high, low), roof=3, residual=3)
    plan = SpecRhythmPolicy().plan(snapshot)
    assert plan.normal_budgets["high"] > plan.normal_budgets["low"]
