"""Autoregressive and SLO-unaware speculative baselines."""

from __future__ import annotations

from specrhythm.policies.base import PolicySnapshot, StepPlan


def allocate_round_robin(
    snapshot: PolicySnapshot, per_request_budget: int
) -> dict[str, int]:
    """Allocate the same SLO-unaware candidate plan across serial and dual modes."""

    budgets = {request.request_id: 0 for request in snapshot.normal_requests}
    remaining = snapshot.roof_candidate_budget
    for depth in range(per_request_budget):
        for request in snapshot.normal_requests:
            if remaining <= 0:
                return budgets
            if depth < request.max_budget:
                budgets[request.request_id] += 1
                remaining -= 1
    return budgets


class ARPolicy:
    name = "ar"
    display_name = "AR"
    execution_mode = "ar"
    allocator = "none"
    eager_enabled = False
    eager_semantics = "none"

    def plan(self, snapshot: PolicySnapshot) -> StepPlan:
        return StepPlan(
            normal_budgets={request.request_id: 0 for request in snapshot.normal_requests}
        )


class SerialSDPolicy:
    """Fixed-budget speculative decoding with exposed serial draft compute."""

    name = "serial-sd"
    display_name = "Serial SD"
    execution_mode = "serial"
    allocator = "slo-unaware-round-robin"
    eager_enabled = False
    eager_semantics = "none"

    def __init__(self, budget: int = 4) -> None:
        if budget < 0:
            raise ValueError("budget must be non-negative")
        self.budget = budget

    def plan(self, snapshot: PolicySnapshot) -> StepPlan:
        return StepPlan(normal_budgets=allocate_round_robin(snapshot, self.budget))


class DualBatchPolicy:
    """Dual-batch, SLO-unaware round-robin speculative allocation."""

    name = "dual-batch"
    display_name = "Dual-Batch"
    execution_mode = "dual"
    allocator = "slo-unaware-round-robin"
    eager_enabled = False
    eager_semantics = "none"

    def __init__(self, budget: int = 4) -> None:
        if budget < 0:
            raise ValueError("budget must be non-negative")
        self.budget = budget

    def plan(self, snapshot: PolicySnapshot) -> StepPlan:
        return StepPlan(normal_budgets=allocate_round_robin(snapshot, self.budget))
