"""Explicit no-bonus pre/post protocol; independent of rolling-eager bridge semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from specrhythm.phase4.draft_batch import token_tuple
from specrhythm.phase4.serial import AcceptanceDecision, token_prefix_hash

MODES = ("serial-prepost3", "serial-eager-prepost3")
PROTOCOL = "specrhythm.serial-prepost3.v1"
PARAMETERS = dict(
    eager_lookahead_steps=3,
    post_verify_batch_steps=1,
    long_proposal_tokens=4,
    short_proposal_tokens=1,
    serial_extension_steps=3,
)
PURPOSES = ("prepost_lookahead", "prepost_post", "prepost_extension", "prepost_repair")


def acceptance(proposal, committed, *, terminal=False):
    """Only accepted candidates or accepted prefix + correction can be committed."""
    drafted, delta = token_tuple(proposal), token_tuple(committed)
    if not delta or not 1 <= len(drafted) <= 4 or type(terminal) is not bool:
        raise ValueError("prepost requires real candidates and nonempty committed output")
    matched = 0
    while matched < min(len(drafted), len(delta)) and drafted[matched] == delta[matched]:
        matched += 1
    if matched == len(delta):
        if matched != len(drafted) and not terminal:
            raise ValueError("nonterminal partial acceptance lacks correction")
        correction = ()
    elif matched < len(drafted) and len(delta) == matched + 1:
        correction = delta[matched:]
    else:
        raise ValueError("prepost cannot commit a bonus or unverified suffix")
    return AcceptanceDecision(
        drafted[:matched], drafted[matched:], correction, (), delta, terminal
    )


def canonical_sample(proposal, sampled, *, previous, maximum, eos):
    """Project parsed stock rejection output before any downstream publication.

    Stock computes a bonus on full accept. It is explicitly unused, never becomes
    a candidate and never reaches committed output. Padding must already have
    been removed by the stock rejection parser (negative token IDs fail here).
    """
    drafted, raw = token_tuple(proposal), token_tuple(sampled)
    if not raw or not 1 <= len(drafted) <= 4 or not 0 <= previous < maximum:
        raise ValueError("invalid prepost sampled row/remaining budget")
    full = len(raw) == len(drafted) + 1 and raw[:-1] == drafted
    values = raw[:-1] if full else raw
    values = values[: maximum - previous]
    for i, token in enumerate(values):
        if token in eos:
            values = values[: i + 1]
            break
    terminal = previous + len(values) == maximum or bool(values and values[-1] in eos)
    decision = acceptance(drafted, values, terminal=terminal)
    return decision, dict(
        unused_target_bonus_tokens=int(full),
        stop_trimmed_tokens=len(raw) - int(full) - len(values),
    )


@dataclass(frozen=True)
class PrePostWork:
    work_id: str
    request_id: str
    prefix_version: int
    parent_proposal_id: str
    parent_prefix: Tuple[int, ...]
    proposal_tokens: Tuple[int, ...]
    limit: int
    eos_token_ids: Tuple[int, ...] = ()

    @property
    def dependency_prefix(self):
        return self.parent_prefix + self.proposal_tokens


@dataclass
class Lookahead:
    work: PrePostWork
    generated: list[int] = field(default_factory=list)
    completed: bool = False
    status: str = "running"
    started_ns: int = 0
    completed_ns: int = 0
    aborted: bool = False


@dataclass
class PrePostState:
    committed_prefix: Tuple[int, ...]
    prefix_version: int
    remaining: int
    eos_token_ids: Tuple[int, ...]
    proposal_id: str = ""
    proposal_tokens: Tuple[int, ...] = ()
    continuations: dict[str, Lookahead] = field(default_factory=dict)
    terminal: bool = False

    def check(self, prefix, version):
        if self.terminal or self.committed_prefix != prefix or self.prefix_version != version:
            raise ValueError("stale prepost request/prefix/version")

    @property
    def prefix_hash(self):
        return token_prefix_hash(self.committed_prefix)


class PrePostLedger:
    def __init__(self):
        self.states = {}

    def _state(self, rid):
        if rid not in self.states:
            raise ValueError("unknown prepost request")
        return self.states[rid]
