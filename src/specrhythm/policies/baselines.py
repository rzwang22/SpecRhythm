"""Autoregressive and SLO-unaware speculative baselines."""

from __future__ import annotations

from specrhythm.policies.base import PolicySnapshot, StepPlan


class ARPolicy:
    name = "ar"

    def plan(self, snapshot: PolicySnapshot) -> StepPlan:
        return StepPlan({request.request_id: 0 for request in snapshot.requests})


class FixedBudgetPolicy:
    name = "fixed"

    def __init__(self, budget: int = 4) -> None:
        if budget < 0:
            raise ValueError("budget must be non-negative")
        self.budget = budget

    def plan(self, snapshot: PolicySnapshot) -> StepPlan:
        remaining = snapshot.roof_candidate_budget
        budgets: dict[str, int] = {}
        for request in snapshot.requests:
            budget = min(self.budget, request.max_budget, remaining)
            budgets[request.request_id] = budget
            remaining -= budget
        return StepPlan(budgets)


class MineDraftPolicy:
    """Dual-batch, SLO-unaware round-robin speculative allocation."""

    name = "minedraft"

    def __init__(self, budget: int = 4) -> None:
        if budget < 0:
            raise ValueError("budget must be non-negative")
        self.budget = budget

    def plan(self, snapshot: PolicySnapshot) -> StepPlan:
        budgets = {request.request_id: 0 for request in snapshot.requests}
        remaining = snapshot.roof_candidate_budget
        for depth in range(self.budget):
            for request in snapshot.requests:
                if remaining <= 0:
                    return StepPlan(budgets)
                if depth < request.max_budget:
                    budgets[request.request_id] += 1
                    remaining -= 1
        return StepPlan(budgets)
