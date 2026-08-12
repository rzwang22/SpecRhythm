"""Autoregressive and SLO-unaware speculative baselines."""

from __future__ import annotations

from specrhythm.policies.base import PolicySnapshot, StepPlan


class ARPolicy:
    name = "ar"
    execution_mode = "ar"
    eager_enabled = False

    def plan(self, snapshot: PolicySnapshot) -> StepPlan:
        return StepPlan(
            normal_budgets={request.request_id: 0 for request in snapshot.normal_requests}
        )


class SerialSDPolicy:
    """Fixed-budget speculative decoding with exposed serial draft compute."""

    name = "serial-sd"
    execution_mode = "serial"
    eager_enabled = False

    def __init__(self, budget: int = 4) -> None:
        if budget < 0:
            raise ValueError("budget must be non-negative")
        self.budget = budget

    def plan(self, snapshot: PolicySnapshot) -> StepPlan:
        remaining = snapshot.roof_candidate_budget
        budgets: dict[str, int] = {}
        for request in snapshot.normal_requests:
            budget = min(self.budget, request.max_budget, remaining)
            budgets[request.request_id] = budget
            remaining -= budget
        return StepPlan(normal_budgets=budgets)


class DualBatchPolicy:
    """Dual-batch, SLO-unaware round-robin speculative allocation."""

    name = "dual-batch"
    execution_mode = "dual"
    eager_enabled = False

    def __init__(self, budget: int = 4) -> None:
        if budget < 0:
            raise ValueError("budget must be non-negative")
        self.budget = budget

    def plan(self, snapshot: PolicySnapshot) -> StepPlan:
        budgets = {request.request_id: 0 for request in snapshot.requests}
        remaining = snapshot.roof_candidate_budget
        for depth in range(self.budget):
            for request in snapshot.normal_requests:
                if remaining <= 0:
                    return StepPlan(normal_budgets=budgets)
                if depth < request.max_budget:
                    budgets[request.request_id] += 1
                    remaining -= 1
        return StepPlan(normal_budgets=budgets)
