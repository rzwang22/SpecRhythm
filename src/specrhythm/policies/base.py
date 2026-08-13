"""Policy contracts for the proposal-lifecycle simulator."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Protocol

from specrhythm.tree import (
    CandidateTree,
    SelectedProposalTree,
    maximum_expected_candidate_progress,
)


@dataclass(frozen=True)
class RequestView:
    request_id: str
    committed_prefix_len: int
    elapsed_decode_ms: float
    slo_tpot_ms: float
    recent_acceptance_ratio: float
    draft_confidence: float
    waiting_time_ms: float
    max_budget: int
    proposal_budget: int = 0
    parent_full_acceptance_probability: float = 0.0
    candidate_tree: Optional[CandidateTree] = None
    parent_selected_tree: Optional[SelectedProposalTree] = None
    estimated_next_iteration_latency_ms: float = 0.0

    @property
    def estimated_tpot_ms(self) -> float:
        return self.elapsed_decode_ms / max(self.committed_prefix_len, 1)

    @property
    def normalized_slack(self) -> float:
        return 1.0 - self.estimated_tpot_ms / self.slo_tpot_ms

    @property
    def progress_gap(self) -> int:
        return max(0, math.ceil(self.continuous_progress_gap))

    @property
    def continuous_progress_gap(self) -> float:
        gap = (
            self.elapsed_decode_ms + self.estimated_next_iteration_latency_ms
        ) / self.slo_tpot_ms
        gap -= self.committed_prefix_len
        return max(0.0, gap)

    @property
    def candidate_progress_gap(self) -> float:
        """Expected candidate progress needed after the guaranteed root token."""

        return max(0.0, self.continuous_progress_gap - 1.0)

    @property
    def required_total_progress(self) -> float:
        """Total root-plus-candidate progress required by the frozen SLO gap."""

        return self.continuous_progress_gap

    @property
    def required_candidate_progress(self) -> float:
        """Candidate progress required after counting one guaranteed root once."""

        return self.candidate_progress_gap

    @property
    def maximum_attainable_candidate_progress(self) -> float:
        if self.candidate_tree is None:
            return sum(
                self.acceptance_benefit**depth
                for depth in range(1, self.max_budget + 1)
            )
        return maximum_expected_candidate_progress(
            self.candidate_tree, self.max_budget
        )

    @property
    def maximum_attainable_total_progress(self) -> float:
        return 1.0 + self.maximum_attainable_candidate_progress

    @property
    def one_cycle_feasible(self) -> bool:
        return (
            self.required_total_progress
            <= self.maximum_attainable_total_progress + 1e-12
        )

    @property
    def acceptance_benefit(self) -> float:
        recent = max(0.0, min(1.0, self.recent_acceptance_ratio))
        confidence = max(0.0, min(1.0, self.draft_confidence))
        return recent * confidence


@dataclass(frozen=True)
class PolicySnapshot:
    normal_requests: tuple[RequestView, ...]
    eager_requests: tuple[RequestView, ...]
    roof_candidate_budget: int
    residual_draft_tokens: int

    @property
    def requests(self) -> tuple[RequestView, ...]:
        """Compatibility alias for policies that allocate normal proposals."""

        return self.normal_requests


@dataclass(frozen=True)
class StepPlan:
    normal_budgets: dict[str, int] = field(default_factory=dict)
    eager_budgets: dict[str, int] = field(default_factory=dict)
    normal_trees: dict[str, SelectedProposalTree] = field(default_factory=dict)
    candidate_trees: dict[str, CandidateTree] = field(default_factory=dict)
    eager_dependency_paths: dict[str, tuple[str, ...]] = field(default_factory=dict)
    expected_progress: dict[str, float] = field(default_factory=dict)
    requested_progress_gap: dict[str, float] = field(default_factory=dict)
    slo_stage_budgets: dict[str, int] = field(default_factory=dict)
    residual_stage_budgets: dict[str, int] = field(default_factory=dict)
    normal_budget_displaced_by_eager: dict[str, int] = field(default_factory=dict)
    base_normal_budgets: dict[str, int] = field(default_factory=dict)
    base_normal_trees: dict[str, SelectedProposalTree] = field(default_factory=dict)
    required_total_progress: dict[str, float] = field(default_factory=dict)
    required_candidate_progress: dict[str, float] = field(default_factory=dict)
    maximum_attainable_candidate_progress: dict[str, float] = field(
        default_factory=dict
    )
    maximum_attainable_total_progress: dict[str, float] = field(default_factory=dict)
    one_cycle_feasible: dict[str, bool] = field(default_factory=dict)

    @property
    def budgets(self) -> dict[str, int]:
        """Compatibility alias for normal proposal budgets."""

        return self.normal_budgets

    @property
    def total_candidates(self) -> int:
        if self.normal_trees:
            return sum(tree.candidate_budget for tree in self.normal_trees.values())
        return sum(self.normal_budgets.values())

    @property
    def total_eager_candidates(self) -> int:
        return sum(self.eager_budgets.values())


class SchedulingPolicy(Protocol):
    name: str
    display_name: str
    execution_mode: str
    allocator: str
    eager_enabled: bool
    eager_semantics: str

    def plan(self, snapshot: PolicySnapshot) -> StepPlan: ...
