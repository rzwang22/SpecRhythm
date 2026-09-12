"""Dependency-free Phase-4A.1 serial speculative-decoding semantics.

The GPU integration imports these records, but the state transitions are kept
independent of vLLM, Transformers, PyTorch, and CUDA so they can be exhaustively
tested on CPU.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

PROTOCOL_VERSION = "specrhythm.serial-disaggregated.v1"


def token_prefix_hash(token_ids: Sequence[int]) -> str:
    """Hash a token sequence with an unambiguous, portable encoding."""

    normalized = []
    for token_id in token_ids:
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
            raise ValueError("token IDs must be non-negative integers")
        normalized.append(token_id)
    payload = json.dumps(normalized, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tokens(name: str, values: Sequence[int]) -> Tuple[int, ...]:
    result = tuple(values)
    token_prefix_hash(result)
    return result


def _nonnegative(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class Proposal:
    protocol_version: str
    request_id: str
    round_id: int
    parent_prefix_len: int
    parent_prefix_hash: str
    proposal_token_ids: Tuple[int, ...]
    proposal_eos: bool
    draft_start_ns: int
    draft_end_ns: int
    transport_payload_bytes: int
    model_provenance: Mapping[str, Any]
    runtime_provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("unsupported serial proposal protocol")
        if not self.request_id or not self.parent_prefix_hash:
            raise ValueError("proposal identity and prefix hash must not be empty")
        _nonnegative("round_id", self.round_id)
        _nonnegative("parent_prefix_len", self.parent_prefix_len)
        proposal = _tokens("proposal_token_ids", self.proposal_token_ids)
        if len(proposal) > 4:
            raise ValueError("Phase-4A.1 linear proposal budget is at most four")
        for name, value in (
            ("draft_start_ns", self.draft_start_ns),
            ("draft_end_ns", self.draft_end_ns),
            ("transport_payload_bytes", self.transport_payload_bytes),
        ):
            _nonnegative(name, value)
        if self.draft_end_ns < self.draft_start_ns:
            raise ValueError("draft end precedes draft start")
        if self.proposal_eos and not proposal:
            raise ValueError("an empty proposal cannot contain Draft EOS")
        if not isinstance(self.model_provenance, Mapping) or not isinstance(
            self.runtime_provenance, Mapping
        ):
            raise ValueError("proposal provenance must be mappings")

    @property
    def proposal_length(self) -> int:
        return len(self.proposal_token_ids)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["proposal_token_ids"] = list(self.proposal_token_ids)
        value["proposal_length"] = self.proposal_length
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Proposal":
        return cls(
            protocol_version=str(value.get("protocol_version", "")),
            request_id=str(value.get("request_id", "")),
            round_id=value.get("round_id", -1),
            parent_prefix_len=value.get("parent_prefix_len", -1),
            parent_prefix_hash=str(value.get("parent_prefix_hash", "")),
            proposal_token_ids=tuple(value.get("proposal_token_ids", ())),
            proposal_eos=value.get("proposal_eos", False),
            draft_start_ns=value.get("draft_start_ns", -1),
            draft_end_ns=value.get("draft_end_ns", -1),
            transport_payload_bytes=value.get("transport_payload_bytes", -1),
            model_provenance=value.get("model_provenance", {}),
            runtime_provenance=value.get("runtime_provenance", {}),
        )


@dataclass(frozen=True)
class AcceptanceDecision:
    accepted_draft_token_ids: Tuple[int, ...]
    rejected_draft_token_ids: Tuple[int, ...]
    target_correction_token_ids: Tuple[int, ...]
    target_bonus_token_ids: Tuple[int, ...]
    committed_token_ids: Tuple[int, ...]
    terminal: bool

    def __post_init__(self) -> None:
        for name in (
            "accepted_draft_token_ids",
            "rejected_draft_token_ids",
            "target_correction_token_ids",
            "target_bonus_token_ids",
            "committed_token_ids",
        ):
            _tokens(name, getattr(self, name))
        if len(self.target_correction_token_ids) > 1 or len(self.target_bonus_token_ids) > 1:
            raise ValueError("a result contains at most one correction or bonus")
        if self.target_correction_token_ids and self.target_bonus_token_ids:
            raise ValueError("target correction and bonus are mutually exclusive")
        expected = (
            self.accepted_draft_token_ids
            + self.target_correction_token_ids
            + self.target_bonus_token_ids
        )
        if self.committed_token_ids != expected:
            raise ValueError("committed token accounting is not conserved")

    @property
    def accounting(self) -> dict[str, int]:
        accepted = len(self.accepted_draft_token_ids)
        rejected = len(self.rejected_draft_token_ids)
        return {
            "proposed_tokens": accepted + rejected,
            "verified_candidate_tokens": accepted + rejected,
            "accepted_draft_tokens": accepted,
            "rejected_draft_tokens": rejected,
            "target_correction_tokens": len(self.target_correction_token_ids),
            "target_bonus_tokens": len(self.target_bonus_token_ids),
            "committed_tokens": len(self.committed_token_ids),
        }


def greedy_acceptance(
    proposal: Sequence[int],
    committed_delta: Sequence[int],
    *,
    terminal: bool,
) -> AcceptanceDecision:
    """Classify vLLM's committed greedy output for one linear proposal.

    `committed_delta` is authoritative Target output after EOS/max-token
    trimming. It may equal a full Draft proposal without a bonus only when the
    request terminates on the accepted Draft EOS token.
    """

    drafted = _tokens("proposal", proposal)
    committed = _tokens("committed_delta", committed_delta)
    matched = 0
    while matched < min(len(drafted), len(committed)):
        if drafted[matched] != committed[matched]:
            break
        matched += 1

    accepted = drafted[:matched]
    rejected = drafted[matched:]
    correction: Tuple[int, ...] = ()
    bonus: Tuple[int, ...] = ()
    if matched < len(drafted):
        if len(committed) != matched + 1:
            raise ValueError("partial acceptance must commit exactly one Target correction")
        correction = committed[matched:]
    else:
        if not drafted and len(committed) == 1:
            correction = committed
        elif len(committed) == len(drafted) + 1:
            bonus = committed[-1:]
        elif len(committed) == len(drafted) and terminal:
            pass
        else:
            raise ValueError("full acceptance must commit one bonus unless terminal")
    return AcceptanceDecision(
        accepted_draft_token_ids=accepted,
        rejected_draft_token_ids=rejected,
        target_correction_token_ids=correction,
        target_bonus_token_ids=bonus,
        committed_token_ids=committed,
        terminal=terminal,
    )


@dataclass(frozen=True)
class SerialTimeline:
    draft_start_ns: int
    draft_end_ns: int
    transfer_start_ns: int
    transfer_end_ns: int
    verify_start_ns: int
    verify_end_ns: int
    state_sync_start_ns: int
    state_sync_end_ns: int
    next_round_draft_start_ns: int

    def __post_init__(self) -> None:
        values = [
            self.draft_start_ns,
            self.draft_end_ns,
            self.transfer_start_ns,
            self.transfer_end_ns,
            self.verify_start_ns,
            self.verify_end_ns,
            self.state_sync_start_ns,
            self.state_sync_end_ns,
            self.next_round_draft_start_ns,
        ]
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values
        ):
            raise ValueError("serial timeline timestamps must be non-negative integers")
        if values != sorted(values):
            raise ValueError("Draft/transfer/verify/state-sync phases overlap or are out of order")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class RoundRecord:
    request_id: str
    round_id: int
    parent_prefix_len: int
    parent_prefix_hash: str
    proposal_token_ids: Tuple[int, ...]
    decision: AcceptanceDecision
    remaining_output_budget: int
    logical_target_kv_length: int
    logical_draft_kv_length: int
    timeline: SerialTimeline
    target_microbatch_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("round request_id must not be empty")
        _nonnegative("round_id", self.round_id)
        _nonnegative("parent_prefix_len", self.parent_prefix_len)
        _tokens("proposal_token_ids", self.proposal_token_ids)
        for name in (
            "remaining_output_budget",
            "logical_target_kv_length",
            "logical_draft_kv_length",
        ):
            _nonnegative(name, getattr(self, name))
        if self.decision.accounting["proposed_tokens"] != len(self.proposal_token_ids):
            raise ValueError("round proposed-token accounting is not conserved")
        if self.logical_target_kv_length != self.logical_draft_kv_length:
            raise ValueError("Draft and Target logical KV lengths differ")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "round_id": self.round_id,
            "parent_prefix_len": self.parent_prefix_len,
            "parent_prefix_hash": self.parent_prefix_hash,
            "proposal_token_ids": list(self.proposal_token_ids),
            "proposal_length": len(self.proposal_token_ids),
            **self.decision.accounting,
            "accepted_draft_token_ids": list(self.decision.accepted_draft_token_ids),
            "rejected_draft_token_ids": list(self.decision.rejected_draft_token_ids),
            "target_correction_token_ids": list(self.decision.target_correction_token_ids),
            "target_bonus_token_ids": list(self.decision.target_bonus_token_ids),
            "committed_token_ids": list(self.decision.committed_token_ids),
            "remaining_output_budget": self.remaining_output_budget,
            "logical_target_kv_length": self.logical_target_kv_length,
            "logical_draft_kv_length": self.logical_draft_kv_length,
            "terminal": self.decision.terminal,
            "timeline": self.timeline.to_dict(),
            "target_microbatch_id": self.target_microbatch_id,
        }


def validate_round_sequence(records: Sequence[RoundRecord]) -> list[str]:
    errors = []
    by_request: dict[str, list[RoundRecord]] = {}
    for record in records:
        by_request.setdefault(record.request_id, []).append(record)
    for request_id, rows in by_request.items():
        expected_round = 0
        prior_end = -1
        for row in rows:
            if row.round_id != expected_round:
                errors.append(f"{request_id}: stale, duplicate, or out-of-order round")
            if row.timeline.draft_start_ns < prior_end:
                errors.append(f"{request_id}: adjacent rounds overlap")
            expected_round += 1
            prior_end = row.timeline.state_sync_end_ns
            accounting = row.decision.accounting
            if (
                accounting["proposed_tokens"]
                != accounting["accepted_draft_tokens"] + accounting["rejected_draft_tokens"]
            ):
                errors.append(f"{request_id}: proposed accounting mismatch")
            if (
                accounting["committed_tokens"]
                != accounting["accepted_draft_tokens"]
                + accounting["target_correction_tokens"]
                + accounting["target_bonus_tokens"]
            ):
                errors.append(f"{request_id}: committed accounting mismatch")
        terminal_indices = [index for index, row in enumerate(rows) if row.decision.terminal]
        if terminal_indices and terminal_indices[-1] != len(rows) - 1:
            errors.append(f"{request_id}: work exists after a terminal round")
    return errors


def finite_nonnegative_number(name: str, value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result
