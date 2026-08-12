"""Policy contracts for the proposal-lifecycle simulator."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol


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

    @property
    def estimated_tpot_ms(self) -> float:
        return self.elapsed_decode_ms / max(self.committed_prefix_len, 1)

    @property
    def normalized_slack(self) -> float:
        return 1.0 - self.estimated_tpot_ms / self.slo_tpot_ms

    @property
    def progress_gap(self) -> int:
        gap = (self.elapsed_decode_ms + self.waiting_time_ms) / self.slo_tpot_ms
        gap -= self.committed_prefix_len
        return max(0, math.ceil(gap))

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

    @property
    def budgets(self) -> dict[str, int]:
        """Compatibility alias for normal proposal budgets."""

        return self.normal_budgets

    @property
    def total_candidates(self) -> int:
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
