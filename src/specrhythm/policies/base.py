"""Policy contracts shared by the simulator and future engine adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class RequestView:
    request_id: str
    delivered_tokens: int
    elapsed_decode_ms: float
    slo_tpot_ms: float
    acceptance_ratio: float
    draft_confidence: float
    waiting_time_ms: float
    max_budget: int

    @property
    def estimated_tpot_ms(self) -> float:
        return self.elapsed_decode_ms / max(self.delivered_tokens, 1)

    @property
    def normalized_slack(self) -> float:
        return 1.0 - self.estimated_tpot_ms / self.slo_tpot_ms

    @property
    def progress_gap(self) -> int:
        gap = (self.elapsed_decode_ms + self.waiting_time_ms) / self.slo_tpot_ms
        gap -= self.delivered_tokens
        return max(0, math.ceil(gap))

    @property
    def acceptance_benefit(self) -> float:
        return max(0.0, min(1.0, self.acceptance_ratio * self.draft_confidence))


@dataclass(frozen=True)
class PolicySnapshot:
    requests: tuple[RequestView, ...]
    roof_candidate_budget: int
    residual_draft_tokens: int


@dataclass(frozen=True)
class StepPlan:
    budgets: dict[str, int] = field(default_factory=dict)
    eager_request_ids: tuple[str, ...] = ()

    @property
    def total_candidates(self) -> int:
        return sum(self.budgets.values())


class SchedulingPolicy(Protocol):
    name: str

    def plan(self, snapshot: PolicySnapshot) -> StepPlan: ...
