"""SLO-aware allocation and guarded rolling-eager policy variants."""

from __future__ import annotations

import math
from dataclasses import replace

from specrhythm.policies.base import PolicySnapshot, RequestView, StepPlan
from specrhythm.policies.baselines import allocate_round_robin
from specrhythm.policies.tree_aware import (
    allocate_adaserve_tree_aware,
    allocate_specrhythm_tree_aware,
)
from specrhythm.tree import predicted_dependency_path

EAGER_MIN_FULL_ACCEPTANCE_PROBABILITY = 0.10


def _urgency(request: RequestView) -> float:
    slack = request.normalized_slack
    return 1.0 + max(0.0, -slack) + 1.0 / (1.0 + max(0.0, slack))


def _marginal_value(request: RequestView, next_depth: int) -> float:
    return _urgency(request) * request.acceptance_benefit**next_depth


def allocate_slo_aware(snapshot: PolicySnapshot) -> dict[str, int]:
    """Legacy flat-sequence allocator retained only for proxy diagnostics."""

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


def allocate_guarded_eager(
    snapshot: PolicySnapshot,
    *,
    max_eager_budget: int = 4,
    min_dependency_path_probability: float = EAGER_MIN_FULL_ACCEPTANCE_PROBABILITY,
) -> dict[str, int]:
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
            or full_probability < min_dependency_path_probability
        ):
            continue
        gap_budget = max(
            1,
            math.ceil(min(request.progress_gap, request.max_budget) * full_probability),
        )
        confidence_cap = math.floor(request.max_budget * request.draft_confidence)
        provisional = min(
            request.max_budget,
            max_eager_budget,
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


def _attribute_normal_displacement(
    eager_budgets: dict[str, int],
    *,
    actual_normal_candidates: int,
    counterfactual_normal_candidates: int,
) -> dict[str, int]:
    """Attribute only normal work that eager admission actually displaced."""

    remaining = max(0, counterfactual_normal_candidates - actual_normal_candidates)
    result: dict[str, int] = {}
    for request_id, budget in eager_budgets.items():
        displaced = min(budget, remaining)
        result[request_id] = displaced
        remaining -= displaced
    return result


class AdaServeStylePolicy:
    """Tree-aware AdaServe control-plane allocator under proxy latency inputs."""

    name = "adaserve"
    display_name = "AdaServe tree-aware simulator baseline"
    execution_mode = "serial"
    allocator = "adaserve-tree-aware"
    eager_enabled = False
    eager_semantics = "none"

    def __init__(self, n_max_slo: int = 8) -> None:
        self.n_max_slo = n_max_slo

    def plan(self, snapshot: PolicySnapshot) -> StepPlan:
        return allocate_adaserve_tree_aware(snapshot, n_max_slo=self.n_max_slo)


class AdaServeFlatProxyPolicy:
    """Legacy flat-sequence shaping proxy retained only for diagnostics."""

    name = "adaserve-flat-proxy"
    display_name = "AdaServe legacy flat-sequence shaping proxy"
    execution_mode = "serial"
    allocator = "legacy-flat-sequence-shaping-proxy"
    eager_enabled = False
    eager_semantics = "none"

    def plan(self, snapshot: PolicySnapshot) -> StepPlan:
        return StepPlan(normal_budgets=allocate_slo_aware(snapshot))


class LegacyFlatShapingProxyPolicy:
    """Legacy flat-sequence shaping/eager proxy retained for result provenance."""

    execution_mode = "dual"
    allocator = "legacy-flat-sequence-shaping-proxy"

    def __init__(self, enable_eager: bool = False) -> None:
        self.enable_eager = enable_eager
        self.eager_enabled = enable_eager
        self.name = "specrhythm-flat-proxy" if enable_eager else "shaping-flat-proxy"
        self.display_name = (
            "SpecRhythm legacy flat-sequence shaping proxy"
            if enable_eager
            else "Shaping legacy flat-sequence proxy"
        )
        self.eager_semantics = "guarded-rolling" if enable_eager else "none"

    def plan(self, snapshot: PolicySnapshot) -> StepPlan:
        eager = allocate_guarded_eager(snapshot) if self.enable_eager else {}
        normal_snapshot = replace(
            snapshot,
            roof_candidate_budget=snapshot.roof_candidate_budget - sum(eager.values()),
        )
        normal = allocate_slo_aware(normal_snapshot)
        counterfactual = allocate_slo_aware(snapshot)
        return StepPlan(
            normal_budgets=normal,
            eager_budgets=eager,
            eager_dependency_paths={
                request.request_id: predicted_dependency_path(
                    request.candidate_tree, request.parent_selected_tree
                )
                for request in snapshot.eager_requests
                if request.request_id in eager
                and request.candidate_tree is not None
                and request.parent_selected_tree is not None
            },
            normal_budget_displaced_by_eager=_attribute_normal_displacement(
                eager,
                actual_normal_candidates=sum(normal.values()),
                counterfactual_normal_candidates=sum(counterfactual.values()),
            ),
        )


class DualEagerPolicy:
    """SLO-unaware dual-batch allocation with guarded rolling eager."""

    name = "dual-eager"
    display_name = "Dual-Batch + Rolling Eager"
    execution_mode = "dual"
    allocator = "slo-unaware-round-robin"
    eager_enabled = True
    eager_semantics = "guarded-rolling"

    def __init__(
        self,
        budget: int = 4,
        *,
        max_eager_budget: int = 4,
        min_dependency_path_probability: float = EAGER_MIN_FULL_ACCEPTANCE_PROBABILITY,
    ) -> None:
        if budget < 0:
            raise ValueError("budget must be non-negative")
        self.budget = budget
        self.max_eager_budget = max_eager_budget
        self.min_dependency_path_probability = min_dependency_path_probability

    def plan(self, snapshot: PolicySnapshot) -> StepPlan:
        eager = allocate_guarded_eager(
            snapshot,
            max_eager_budget=self.max_eager_budget,
            min_dependency_path_probability=self.min_dependency_path_probability,
        )
        normal_snapshot = replace(
            snapshot,
            roof_candidate_budget=snapshot.roof_candidate_budget - sum(eager.values()),
        )
        normal = allocate_round_robin(normal_snapshot, self.budget)
        counterfactual = allocate_round_robin(snapshot, self.budget)
        return StepPlan(
            normal_budgets=normal,
            eager_budgets=eager,
            eager_dependency_paths={
                request.request_id: predicted_dependency_path(
                    request.candidate_tree, request.parent_selected_tree
                )
                for request in snapshot.eager_requests
                if request.request_id in eager
                and request.candidate_tree is not None
                and request.parent_selected_tree is not None
            },
            normal_budget_displaced_by_eager=_attribute_normal_displacement(
                eager,
                actual_normal_candidates=sum(normal.values()),
                counterfactual_normal_candidates=sum(counterfactual.values()),
            ),
        )


class SpecRhythmPolicy:
    """Two-pass SLO-aware allocation with optional guarded rolling eager."""

    execution_mode = "dual"
    allocator = "specrhythm-tree-aware"

    def __init__(
        self,
        enable_eager: bool = True,
        *,
        residual_score: str = "urgency-path-probability",
        n_max_slo: int = 8,
        max_eager_budget: int = 4,
        min_dependency_path_probability: float = EAGER_MIN_FULL_ACCEPTANCE_PROBABILITY,
    ) -> None:
        self.enable_eager = enable_eager
        self.eager_enabled = enable_eager
        self.name = "specrhythm" if enable_eager else "shaping"
        self.display_name = (
            "SpecRhythm simulator policy"
            if enable_eager
            else "Dual-Batch + Individual Budget Shaping"
        )
        self.eager_semantics = "guarded-rolling" if enable_eager else "none"
        self.residual_score = residual_score
        self.n_max_slo = n_max_slo
        self.max_eager_budget = max_eager_budget
        self.min_dependency_path_probability = min_dependency_path_probability

    def plan(self, snapshot: PolicySnapshot) -> StepPlan:
        eager = (
            allocate_guarded_eager(
                snapshot,
                max_eager_budget=self.max_eager_budget,
                min_dependency_path_probability=self.min_dependency_path_probability,
            )
            if self.enable_eager
            else {}
        )
        normal_snapshot = replace(
            snapshot,
            roof_candidate_budget=snapshot.roof_candidate_budget - sum(eager.values()),
        )
        normal = allocate_specrhythm_tree_aware(
            normal_snapshot,
            n_max_slo=self.n_max_slo,
            residual_score=self.residual_score,
        )
        counterfactual = allocate_specrhythm_tree_aware(
            snapshot,
            n_max_slo=self.n_max_slo,
            residual_score=self.residual_score,
        )
        dependencies = {}
        by_id = {request.request_id: request for request in snapshot.eager_requests}
        for request_id in eager:
            request = by_id[request_id]
            if request.candidate_tree is not None and request.parent_selected_tree is not None:
                dependencies[request_id] = predicted_dependency_path(
                    request.candidate_tree, request.parent_selected_tree
                )
        return StepPlan(
            normal_budgets=normal.normal_budgets,
            eager_budgets=eager,
            normal_trees=normal.normal_trees,
            candidate_trees=normal.candidate_trees,
            eager_dependency_paths=dependencies,
            expected_progress=normal.expected_progress,
            requested_progress_gap=normal.requested_progress_gap,
            slo_stage_budgets=normal.slo_stage_budgets,
            residual_stage_budgets=normal.residual_stage_budgets,
            normal_budget_displaced_by_eager=_attribute_normal_displacement(
                eager,
                actual_normal_candidates=normal.total_candidates,
                counterfactual_normal_candidates=counterfactual.total_candidates,
            ),
        )
