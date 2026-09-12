"""CPU-only plans for the Serial batched Draft data plane.

Acceptance/stop decisions remain in ``serial.greedy_acceptance``. These plans
only project already validated tokens onto the physical KV frontier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple

from specrhythm.phase4.serial import token_prefix_hash


def token_tuple(values: Sequence[int]) -> Tuple[int, ...]:
    result = tuple(values)
    if any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in result):
        raise ValueError("Draft token IDs must be non-negative integers")
    return result


def unique_ids(ids: Sequence[str]) -> None:
    if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("Draft batch requires unique nonempty request IDs")


@dataclass(frozen=True)
class DraftProposalPlan:
    request_id: str
    round_id: int
    prefix: Tuple[int, ...]
    budget: int
    eos_token_ids: Tuple[int, ...]

    def __post_init__(self) -> None:
        unique_ids((self.request_id,))
        token_tuple(self.prefix)
        token_tuple(self.eos_token_ids)
        if not self.prefix or type(self.round_id) is not int or self.round_id < 0:
            raise ValueError("Draft proposal needs a prefix and non-negative round")
        if type(self.budget) is not int or not 0 <= self.budget <= 4:
            raise ValueError("Draft proposal budget must be in [0, 4]")


@dataclass(frozen=True)
class DraftCommitPlan:
    request_id: str
    round_id: int
    parent_prefix: Tuple[int, ...]
    proposal: Tuple[int, ...]
    accepted: int
    target_tail: Tuple[int, ...]
    final_prefix_hash: str
    terminal: bool
    correction_count: int = 0
    bonus_count: int = 0

    def __post_init__(self) -> None:
        unique_ids((self.request_id,))
        for values in (self.parent_prefix, self.proposal, self.target_tail):
            token_tuple(values)
        if not self.parent_prefix or type(self.round_id) is not int or self.round_id < 0:
            raise ValueError("Draft commit needs a prefix and non-negative round")
        if type(self.accepted) is not int or not 0 <= self.accepted <= len(self.proposal):
            raise ValueError("accepted count is outside the pending proposal")
        if type(self.terminal) is not bool or len(self.target_tail) > 1:
            raise ValueError("invalid Draft terminal/Target tail")
        if (
            self.correction_count not in (0, 1)
            or self.bonus_count not in (0, 1)
            or self.correction_count + self.bonus_count != len(self.target_tail)
        ):
            raise ValueError("correction/bonus accounting disagrees with Target tail")
        if token_prefix_hash(self.final_prefix) != self.final_prefix_hash:
            raise ValueError("Draft commit final prefix hash mismatch")

    @property
    def final_prefix(self) -> Tuple[int, ...]:
        return self.parent_prefix + self.proposal[: self.accepted] + self.target_tail


@dataclass(frozen=True)
class DraftMaterialization:
    """One ragged row: KV below ``valid_length`` is already valid."""

    request_id: str  # Draft internal ID, never a physical row index.
    context: Tuple[int, ...]
    valid_length: int

    def __post_init__(self) -> None:
        unique_ids((self.request_id,))
        token_tuple(self.context)
        if type(self.valid_length) is not int or not 0 <= self.valid_length < len(self.context):
            raise ValueError("materialization must contain a nonempty missing suffix")

    @property
    def suffix(self) -> Tuple[int, ...]:
        return self.context[self.valid_length :]


def commit_frontier(plan: DraftCommitPlan, materialized: int) -> int:
    length = len(plan.parent_prefix)
    expected = length + max(len(plan.proposal) - 1, 0)
    if type(materialized) is not int or materialized != expected:
        raise ValueError("Draft materialized frontier disagrees with proposal progress")
    return min(materialized, length + plan.accepted)


def mapped_rows(expected: Sequence[str], actual: Sequence[str], values: Sequence[Any]) -> dict:
    """Project actual runner rows through identity, including after compaction."""
    unique_ids(expected)
    unique_ids(actual)
    if set(expected) != set(actual) or len(actual) != len(values):
        raise RuntimeError("Draft runner row identity/count mismatch")
    by_id = dict(zip(actual, values))
    return {request_id: by_id[request_id] for request_id in expected}
