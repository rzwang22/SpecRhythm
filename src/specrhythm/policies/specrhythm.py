"""SLO-aware allocation and guarded rolling-eager policy variants."""

from __future__ import annotations

import math
from dataclasses import replace

from specrhythm.policies.base import PolicySnapshot, RequestView, StepPlan
from specrhythm.policies.baselines import allocate_round_robin

EAGER_MIN_FULL_ACCEPTANCE_PROBABILITY = 0.10


def _urgency(request: RequestView) -> float:
    slack = request.normalized_slack
    return 1.0 + max(0.0, -slack) + 1.0 / (1.0 + max(0.0, slack))


def _marginal_value(request: RequestView, next_depth: int) -> float:
    return _urgency(request) * request.acceptance_benefit**next_depth


def allocate_slo_aware(snapshot: PolicySnapshot) -> dict[str, int]:
    """Two-pass SLO-aware allocator shared by AdaServe-style and shaping modes."""

    budgets = {request.request_id: 0 for request in snapshot.normal_requests}
    remaining = max(0, snapshot.roof_candidate_budget)

    urgent = sorted(
        (request for request in snapshot.normal_requests if request.progress_gap > 0),
        key=lambda request: (
            request.progress_gap * request.acceptance_benefit,
            -request.normalized_slack,
            request.request_id,
        ),
        reverse=True,
    )
    unmet = {request.request_id: request.progress_gap for request in urgent}
    while remaining > 0 and any(value > 0 for value in unmet.values()):
        eligible = [
            request
            for request in urgent
            if unmet[request.request_id] > 0
            and budgets[request.request_id] < request.max_budget
        ]
        if not eligible:
            break
        request = max(
            eligible,
            key=lambda item: (
                unmet[item.request_id]
                * _marginal_value(item, budgets[item.request_id] + 1),
                -item.normalized_slack,
                item.request_id,
            ),
        )
        budgets[request.request_id] += 1
        remaining -= 1
        unmet[request.request_id] -= 1

    while remaining > 0:
        eligible = [
            request
            for request in snapshot.normal_requests
            if budgets[request.request_id] < request.max_budget
        ]
        if not eligible:
            break
        request = max(
            eligible,
            key=lambda item: (
                _marginal_value(item, budgets[item.request_id] + 1),
                -item.normalized_slack,
                item.request_id,
            ),
        )
        if _marginal_value(request, budgets[request.request_id] + 1) <= 0:
            break
        budgets[request.request_id] += 1
        remaining -= 1
    return budgets


def allocate_guarded_eager(snapshot: PolicySnapshot) -> dict[str, int]:
    """Admit eager work using urgency and parent full-acceptance probability.

    The parent budget affects the full-acceptance probability already carried in the
    RequestView, but the continuation budget is derived independently from progress gap,
    confidence, and that probability. It is never copied from the parent budget.
    """

    draft_remaining = max(0, snapshot.residual_draft_tokens)
    roof_remaining = max(0, snapshot.roof_candidate_budget)
    eager_budgets: dict[str, int] = {}
    eager_order = sorted(
        snapshot.eager_requests,
        key=lambda request: (
            request.progress_gap * request.parent_full_acceptance_probability,
            request.parent_full_acceptance_probability,
            request.draft_confidence,
            -request.normalized_slack,
            request.request_id,
        ),
        reverse=True,
    )
    for request in eager_order:
        full_probability = request.parent_full_acceptance_probability
        if (
            request.progress_gap <= 0
            or full_probability < EAGER_MIN_FULL_ACCEPTANCE_PROBABILITY
        ):
            continue
        gap_budget = max(
            1,
            math.ceil(min(request.progress_gap, request.max_budget) * full_probability),
        )
        confidence_cap = math.floor(request.max_budget * request.draft_confidence)
        provisional = min(
            request.max_budget,
            gap_budget,
            confidence_cap,
            draft_remaining,
            roof_remaining,
        )
        if provisional <= 0:
            continue
        eager_budgets[request.request_id] = provisional
        draft_remaining -= provisional
        roof_remaining -= provisional
    return eager_budgets


class AdaServeStylePolicy:
    """Serial SLO-aware simulator baseline, not a full AdaServe reproduction."""

    name = "adaserve"
    display_name = "AdaServe-style simulator baseline"
    execution_mode = "serial"
    allocator = "slo-aware-two-pass-proxy"
    eager_enabled = False
    eager_semantics = "none"

    def plan(self, snapshot: PolicySnapshot) -> StepPlan:
        return StepPlan(normal_budgets=allocate_slo_aware(snapshot))


class DualEagerPolicy:
    """SLO-unaware dual-batch allocation with guarded rolling eager."""

    name = "dual-eager"
    display_name = "Dual-Batch + Rolling Eager"
    execution_mode = "dual"
    allocator = "slo-unaware-round-robin"
    eager_enabled = True
    eager_semantics = "guarded-rolling"

    def __init__(self, budget: int = 4) -> None:
        if budget < 0:
            raise ValueError("budget must be non-negative")
        self.budget = budget

    def plan(self, snapshot: PolicySnapshot) -> StepPlan:
        eager = allocate_guarded_eager(snapshot)
        normal_snapshot = replace(
            snapshot,
            roof_candidate_budget=snapshot.roof_candidate_budget - sum(eager.values()),
        )
        normal = allocate_round_robin(normal_snapshot, self.budget)
        return StepPlan(
            normal_budgets=normal,
            eager_budgets=eager,
        )


class SpecRhythmPolicy:
    """Two-pass SLO-aware allocation with optional guarded rolling eager."""

    execution_mode = "dual"
    allocator = "slo-aware-two-pass-proxy"

    def __init__(self, enable_eager: bool = True) -> None:
        self.enable_eager = enable_eager
        self.eager_enabled = enable_eager
        self.name = "specrhythm" if enable_eager else "shaping"
        self.display_name = (
            "SpecRhythm simulator policy"
            if enable_eager
            else "Dual-Batch + Individual Budget Shaping"
        )
        self.eager_semantics = "guarded-rolling" if enable_eager else "none"

    def plan(self, snapshot: PolicySnapshot) -> StepPlan:
        eager = allocate_guarded_eager(snapshot) if self.enable_eager else {}
        normal_snapshot = replace(
            snapshot,
            roof_candidate_budget=snapshot.roof_candidate_budget - sum(eager.values()),
        )
        normal = allocate_slo_aware(normal_snapshot)
        return StepPlan(normal_budgets=normal, eager_budgets=eager)
