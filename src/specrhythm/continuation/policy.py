"""Eligibility providers for the opt-in rolling continuation protocol.

Eligibility belongs to a stable request identity. Admission is a separate,
per-opportunity choice, and neither eligibility nor admission proves that a
continuation's dependency is valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Protocol


class EagerRequestState(Protocol):
    """The request identity contract exposed to eligibility providers."""

    request_id: str


@dataclass(frozen=True)
class EagerRequestView:
    """Isolated provider input; no live state, prefix or historical objects.

    Create a fresh view at each decision boundary. Provider decisions are not
    cached: enabled/version changes still take effect at every original call.
    """

    request_id: str


@dataclass(frozen=True)
class EagerStepContext:
    """Extension point for future providers; static eligibility ignores it."""

    step_index: int = 0


@dataclass(frozen=True)
class EagerDecision:
    """Persistent eligibility and the current admission switch are distinct."""

    eligible: bool
    enabled: bool
    decision_version: int

    def __post_init__(self) -> None:
        if type(self.eligible) is not bool or type(self.enabled) is not bool:
            raise ValueError("eligible and enabled must be booleans")
        if type(self.decision_version) is not int or self.decision_version < 0:
            raise ValueError("decision_version must be a nonnegative integer")

    @property
    def can_admit(self) -> bool:
        """Whether a new eager operation may be considered, before capacity."""

        return self.eligible and self.enabled


class EagerEligibilityProvider(Protocol):
    def evaluate(
        self, request_state: EagerRequestState, step_context: EagerStepContext
    ) -> EagerDecision:
        """Return a versioned decision without committing or scheduling work."""


@dataclass(frozen=True)
class StaticEagerEligibility:
    """Use only configured request IDs; rejection and cohort movement do not matter.

    Configuration is immutable, including a defensive frozen copy of request_ids.
    A later provider can change enabled and increment decision_version without
    changing the meaning of eligible. No dynamic selection algorithm lives here.
    """

    request_ids: FrozenSet[str]
    enabled: bool = True
    decision_version: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_ids", frozenset(self.request_ids))
        if any(
            not isinstance(request_id, str) or not request_id for request_id in self.request_ids
        ):
            raise ValueError("request IDs must be nonempty strings")
        EagerDecision(False, self.enabled, self.decision_version)

    def evaluate(
        self, request_state: EagerRequestState, step_context: EagerStepContext
    ) -> EagerDecision:
        return EagerDecision(
            eligible=request_state.request_id in self.request_ids,
            enabled=self.enabled,
            decision_version=self.decision_version,
        )
