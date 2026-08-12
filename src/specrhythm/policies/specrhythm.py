"""SLO-aware two-pass speculative budget shaping."""

from __future__ import annotations

from specrhythm.policies.base import PolicySnapshot, RequestView, StepPlan


class SpecRhythmPolicy:
    """Testable control-plane interpretation of the paper's policy.

    Pass one closes projected progress gaps. Pass two spends residual roofline capacity on the
    largest urgency-weighted marginal acceptance benefit. Prefix dependencies are preserved by
    allocating only the next depth of any request.
    """

    name = "specrhythm"
    execution_mode = "dual"

    def __init__(self, enable_eager: bool = True) -> None:
        self.enable_eager = enable_eager
        self.eager_enabled = enable_eager
        self.name = "specrhythm" if enable_eager else "shaping"

    @staticmethod
    def _urgency(request: RequestView) -> float:
        slack = request.normalized_slack
        return 1.0 + max(0.0, -slack) + 1.0 / (1.0 + max(0.0, slack))

    @staticmethod
    def _marginal_value(request: RequestView, next_depth: int) -> float:
        probability = request.acceptance_benefit**next_depth
        return SpecRhythmPolicy._urgency(request) * probability

    def plan(self, snapshot: PolicySnapshot) -> StepPlan:
        budgets = {request.request_id: 0 for request in snapshot.normal_requests}
        remaining = max(0, snapshot.roof_candidate_budget)
        by_id = {request.request_id: request for request in snapshot.normal_requests}

        # Pass 1: requests projected to fall behind compete by gap times acceptance benefit.
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
                    * self._marginal_value(item, budgets[item.request_id] + 1),
                    -item.normalized_slack,
                    item.request_id,
                ),
            )
            budgets[request.request_id] += 1
            remaining -= 1
            # One candidate can contribute at most one extra committed token.
            unmet[request.request_id] -= 1

        # Pass 2: allocate each prefix-valid next token by marginal goodput value.
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
                    self._marginal_value(item, budgets[item.request_id] + 1),
                    -item.normalized_slack,
                    item.request_id,
                ),
            )
            if self._marginal_value(request, budgets[request.request_id] + 1) <= 0:
                break
            budgets[request.request_id] += 1
            remaining -= 1

        # Eager continuation uses only the hidden drafting window left after normal work.
        eager_budgets: dict[str, int] = {}
        draft_remaining = max(0, snapshot.residual_draft_tokens - sum(budgets.values()))
        eager_remaining = max(0, snapshot.roof_candidate_budget - sum(budgets.values()))
        eager_order = sorted(
            (request for request in snapshot.eager_requests if request.progress_gap > 0),
            key=lambda request: (
                request.progress_gap * request.acceptance_benefit,
                -request.normalized_slack,
                request.request_id,
            ),
            reverse=True,
        )
        if self.enable_eager:
            for request in eager_order:
                provisional = min(request.proposal_budget, request.max_budget)
                if (
                    provisional > 0
                    and provisional <= draft_remaining
                    and provisional <= eager_remaining
                ):
                    eager_budgets[request.request_id] = provisional
                    draft_remaining -= provisional
                    eager_remaining -= provisional

        assert sum(budgets.values()) <= snapshot.roof_candidate_budget
        assert all(0 <= value <= by_id[key].max_budget for key, value in budgets.items())
        assert sum(budgets.values()) + sum(eager_budgets.values()) <= (
            snapshot.roof_candidate_budget
        )
        return StepPlan(normal_budgets=budgets, eager_budgets=eager_budgets)
