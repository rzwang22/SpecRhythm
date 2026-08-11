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

    def __init__(self, enable_eager: bool = True) -> None:
        self.enable_eager = enable_eager
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
        budgets = {request.request_id: 0 for request in snapshot.requests}
        remaining = max(0, snapshot.roof_candidate_budget)
        by_id = {request.request_id: request for request in snapshot.requests}

        # Pass 1: requests projected to fall behind compete by gap times acceptance benefit.
        urgent = sorted(
            (request for request in snapshot.requests if request.progress_gap > 0),
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
                for request in snapshot.requests
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

        # Eager continuation is admitted only for urgent, useful requests that fit the hidden
        # drafting window. The current proposal must later pass guarded commit.
        eager: list[str] = []
        # Normal draft work consumes the overlap window before eager continuations.
        draft_remaining = max(0, snapshot.residual_draft_tokens - sum(budgets.values()))
        eager_order = sorted(
            urgent,
            key=lambda request: (
                request.progress_gap * request.acceptance_benefit,
                -request.normalized_slack,
                request.request_id,
            ),
            reverse=True,
        )
        if self.enable_eager:
            for request in eager_order:
                provisional = budgets[request.request_id]
                if provisional > 0 and provisional <= draft_remaining:
                    eager.append(request.request_id)
                    draft_remaining -= provisional

        assert sum(budgets.values()) <= snapshot.roof_candidate_budget
        assert all(0 <= value <= by_id[key].max_budget for key, value in budgets.items())
        return StepPlan(budgets=budgets, eager_request_ids=tuple(eager))
