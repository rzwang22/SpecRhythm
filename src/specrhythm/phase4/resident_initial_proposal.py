"""Fail-closed lifecycle for resident Serial round-zero proposals.

The lifecycle is dependency-free and operates on the small request surface used
by pinned vLLM.  The scheduler remains responsible for calling it immediately
before and after the stock scheduling decision.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Sequence

from specrhythm.phase4.serial import Proposal, token_prefix_hash

INITIAL_PROPOSAL_EVENT_SCHEMA = (
    "specrhythm.phase4b-resident-initial-proposal-event.v1"
)


class InitialProposalState(str, Enum):
    PUBLISHED = "published"
    INSTALLED = "installed"
    CONSUMED = "consumed"


@dataclass
class _InitialProposalRecord:
    proposal: Proposal
    internal_request_id: str
    state: InitialProposalState = InitialProposalState.PUBLISHED


class ResidentInitialProposalLifecycle:
    """Own one round-zero proposal until stock vLLM schedules it exactly once."""

    def __init__(
        self,
        *,
        expected_request_ids: Sequence[str],
        stable_to_internal_request_id: Mapping[str, Any],
        proposals: Sequence[Proposal],
        emit: Callable[[Mapping[str, Any]], None],
    ) -> None:
        expected = tuple(str(item) for item in expected_request_ids)
        if not expected or len(set(expected)) != len(expected):
            raise RuntimeError("resident initial proposal request IDs are invalid")
        by_id = {proposal.request_id: proposal for proposal in proposals}
        if len(by_id) != len(tuple(proposals)) or set(by_id) != set(expected):
            raise RuntimeError("resident initial proposal request set is invalid")
        self._records: dict[str, _InitialProposalRecord] = {}
        for stable_id in expected:
            internal_id = str(stable_to_internal_request_id.get(stable_id, ""))
            if not internal_id:
                raise RuntimeError(
                    f"resident initial proposal mapping is missing for {stable_id}"
                )
            proposal = by_id[stable_id]
            if proposal.round_id != 0 or not proposal.proposal_token_ids:
                raise RuntimeError(
                    f"resident initial proposal is not a non-empty round zero for {stable_id}"
                )
            self._records[stable_id] = _InitialProposalRecord(
                proposal=proposal,
                internal_request_id=internal_id,
            )
        self._expected_request_ids = expected
        self._emit = emit
        self._published = False

    @property
    def expected_request_ids(self) -> tuple[str, ...]:
        return self._expected_request_ids

    def state_for(self, stable_request_id: str) -> InitialProposalState:
        return self._records[stable_request_id].state

    def was_installed(self, stable_request_id: str) -> bool:
        return self.state_for(stable_request_id) is not InitialProposalState.PUBLISHED

    def available_for_first_verification(
        self, stable_request_id: str, request: Any
    ) -> bool:
        record = self._records[stable_request_id]
        return (
            record.state is InitialProposalState.INSTALLED
            and self._spec_tokens(request) == record.proposal.proposal_token_ids
        )

    def publish(self, requests: Mapping[str, Any], *, cycle_id: int) -> None:
        if self._published:
            raise RuntimeError("resident initial proposals were published twice")
        for stable_id in self._expected_request_ids:
            record = self._records[stable_id]
            request = self._require_live_request(record, requests, cycle_id)
            self._emit_event(
                record,
                request,
                cycle_id=cycle_id,
                event="published",
                previous_state=None,
                scheduled_spec_tokens=None,
            )
        self._published = True

    def prepare_for_schedule(
        self, requests: Mapping[str, Any], *, cycle_id: int
    ) -> None:
        if not self._published:
            raise RuntimeError("resident initial proposals were not published")
        for stable_id in self._expected_request_ids:
            record = self._records[stable_id]
            if record.state is InitialProposalState.CONSUMED:
                # Later prefixes and proposals belong exclusively to the normal
                # RemoteDraftProposer progression.
                continue
            request = self._require_live_request(record, requests, cycle_id)
            self._validate_parent(record, request, cycle_id)
            spec_tokens = self._spec_tokens(request)
            expected_tokens = record.proposal.proposal_token_ids
            if record.state is InitialProposalState.PUBLISHED:
                if spec_tokens and spec_tokens != expected_tokens:
                    self._fail(
                        record,
                        request,
                        cycle_id,
                        "worker/scheduler initial proposal tokens differ before installation",
                    )
                request.spec_token_ids = list(expected_tokens)
                record.state = InitialProposalState.INSTALLED
                self._emit_event(
                    record,
                    request,
                    cycle_id=cycle_id,
                    event="installed",
                    previous_state=InitialProposalState.PUBLISHED,
                    scheduled_spec_tokens=None,
                )
                continue
            if spec_tokens != expected_tokens:
                self._fail(
                    record,
                    request,
                    cycle_id,
                    "installed resident Serial initial proposal disappeared or changed",
                )
            self._emit_event(
                record,
                request,
                cycle_id=cycle_id,
                event="installed-validated-before-schedule",
                previous_state=InitialProposalState.INSTALLED,
                scheduled_spec_tokens=None,
            )

    def finish_schedule(
        self,
        requests: Mapping[str, Any],
        *,
        scheduled_tokens: Mapping[str, Any],
        scheduled_spec_decode_tokens: Mapping[str, Sequence[int]],
        cycle_id: int,
    ) -> None:
        for stable_id in self._expected_request_ids:
            record = self._records[stable_id]
            if record.state is not InitialProposalState.INSTALLED:
                continue
            request = self._require_live_request(record, requests, cycle_id)
            internal_id = record.internal_request_id
            scheduled_count = int(scheduled_tokens.get(internal_id, 0))
            if internal_id in scheduled_spec_decode_tokens:
                actual = tuple(
                    int(item) for item in scheduled_spec_decode_tokens[internal_id]
                )
                expected = record.proposal.proposal_token_ids
                if not actual or actual != expected[: len(actual)]:
                    self._fail(
                        record,
                        request,
                        cycle_id,
                        "scheduled resident Serial initial proposal tokens differ",
                        scheduled_spec_tokens=actual,
                    )
                record.state = InitialProposalState.CONSUMED
                self._emit_event(
                    record,
                    request,
                    cycle_id=cycle_id,
                    event="consumed-by-speculative-verification",
                    previous_state=InitialProposalState.INSTALLED,
                    scheduled_spec_tokens=actual,
                )
                continue
            if scheduled_count > 0:
                self._fail(
                    record,
                    request,
                    cycle_id,
                    "resident Serial request was scheduled without its initial proposal",
                    scheduled_spec_tokens=(),
                )
            self._validate_parent(record, request, cycle_id)
            if self._spec_tokens(request) != record.proposal.proposal_token_ids:
                self._fail(
                    record,
                    request,
                    cycle_id,
                    "unscheduled resident Serial initial proposal disappeared or changed",
                    scheduled_spec_tokens=(),
                )
            self._emit_event(
                record,
                request,
                cycle_id=cycle_id,
                event="installed-not-scheduled",
                previous_state=InitialProposalState.INSTALLED,
                scheduled_spec_tokens=(),
            )

    def _require_live_request(
        self,
        record: _InitialProposalRecord,
        requests: Mapping[str, Any],
        cycle_id: int,
    ) -> Any:
        request = requests.get(record.internal_request_id)
        if request is None or request.is_finished():
            self._fail(
                record,
                request,
                cycle_id,
                "resident initial proposal has no live Target request",
            )
        if str(request.request_id) != record.internal_request_id:
            self._fail(
                record,
                request,
                cycle_id,
                "resident initial proposal internal request identity differs",
            )
        return request

    def _validate_parent(
        self, record: _InitialProposalRecord, request: Any, cycle_id: int
    ) -> None:
        prefix = self._prefix(request)
        proposal = record.proposal
        if (
            proposal.parent_prefix_len != len(prefix)
            or proposal.parent_prefix_hash != token_prefix_hash(prefix)
        ):
            self._fail(
                record,
                request,
                cycle_id,
                "resident Serial initial proposal parent is stale before consumption",
            )

    def _fail(
        self,
        record: _InitialProposalRecord,
        request: Optional[Any],
        cycle_id: int,
        message: str,
        *,
        scheduled_spec_tokens: Optional[Sequence[int]] = None,
    ) -> None:
        event = self._event(
            record,
            request,
            cycle_id=cycle_id,
            event="validation-failed",
            previous_state=record.state,
            scheduled_spec_tokens=scheduled_spec_tokens,
        )
        event["error"] = message
        self._emit(event)
        evidence = json.dumps(event, sort_keys=True, separators=(",", ":"))
        raise RuntimeError(f"{message}; evidence={evidence}")

    def _emit_event(
        self,
        record: _InitialProposalRecord,
        request: Optional[Any],
        *,
        cycle_id: int,
        event: str,
        previous_state: Optional[InitialProposalState],
        scheduled_spec_tokens: Optional[Sequence[int]],
    ) -> None:
        self._emit(
            self._event(
                record,
                request,
                cycle_id=cycle_id,
                event=event,
                previous_state=previous_state,
                scheduled_spec_tokens=scheduled_spec_tokens,
            )
        )

    def _event(
        self,
        record: _InitialProposalRecord,
        request: Optional[Any],
        *,
        cycle_id: int,
        event: str,
        previous_state: Optional[InitialProposalState],
        scheduled_spec_tokens: Optional[Sequence[int]],
    ) -> dict[str, Any]:
        prefix = self._prefix(request) if request is not None else ()
        spec_tokens = self._spec_tokens(request) if request is not None else ()
        proposal = record.proposal
        return {
            "schema_version": INITIAL_PROPOSAL_EVENT_SCHEMA,
            "timestamp_ns": time.monotonic_ns(),
            "scheduler_cycle_id": cycle_id,
            "consumer": "serial",
            "event": event,
            "request_id": proposal.request_id,
            "internal_request_id": record.internal_request_id,
            "previous_lifecycle_state": (
                previous_state.value if previous_state is not None else None
            ),
            "lifecycle_state": record.state.value,
            "proposal_id": f"{proposal.request_id}:round-{proposal.round_id}",
            "round_id": proposal.round_id,
            "proposal_parent_prefix_len": proposal.parent_prefix_len,
            "proposal_parent_prefix_hash": proposal.parent_prefix_hash,
            "live_request_prefix_len": len(prefix) if request is not None else None,
            "live_request_prefix_hash": (
                token_prefix_hash(prefix) if request is not None else None
            ),
            "num_output_tokens": (
                int(request.num_output_tokens) if request is not None else None
            ),
            "num_computed_tokens": (
                int(request.num_computed_tokens) if request is not None else None
            ),
            "request_spec_token_ids": list(spec_tokens),
            "initial_proposal_token_ids": list(proposal.proposal_token_ids),
            "scheduled_spec_decode_tokens_member": (
                scheduled_spec_tokens is not None
            ),
            "scheduled_spec_decode_token_ids": (
                list(scheduled_spec_tokens)
                if scheduled_spec_tokens is not None
                else None
            ),
        }

    @staticmethod
    def _prefix(request: Any) -> tuple[int, ...]:
        return tuple(int(item) for item in request.all_token_ids)

    @staticmethod
    def _spec_tokens(request: Any) -> tuple[int, ...]:
        return tuple(int(item) for item in request.spec_token_ids)


def validate_initial_proposal_lifecycle_events(
    rows: Sequence[Mapping[str, Any]], *, expected_request_ids: Sequence[str]
) -> list[str]:
    """Validate that every initial proposal was published, installed, and consumed."""

    expected = tuple(str(item) for item in expected_request_ids)
    errors = []
    transitions: dict[str, list[str]] = {request_id: [] for request_id in expected}
    consumed_counts = {request_id: 0 for request_id in expected}
    for index, row in enumerate(rows):
        label = f"initial proposal lifecycle row {index}"
        if row.get("schema_version") != INITIAL_PROPOSAL_EVENT_SCHEMA:
            errors.append(f"{label} has an unsupported schema")
            continue
        request_id = str(row.get("request_id", ""))
        if request_id not in transitions:
            errors.append(f"{label} belongs to an unexpected request")
            continue
        event = str(row.get("event", ""))
        if event == "validation-failed":
            errors.append(f"{label} records a fail-closed lifecycle error")
        if event in {"published", "installed", "consumed-by-speculative-verification"}:
            transitions[request_id].append(event)
        if event == "consumed-by-speculative-verification":
            consumed_counts[request_id] += 1
            scheduled = row.get("scheduled_spec_decode_token_ids")
            if (
                row.get("scheduled_spec_decode_tokens_member") is not True
                or not isinstance(scheduled, list)
                or not scheduled
            ):
                errors.append(f"{label} lacks speculative scheduling evidence")
    for request_id in expected:
        if transitions[request_id] != [
            "published",
            "installed",
            "consumed-by-speculative-verification",
        ]:
            errors.append(f"{request_id}: initial proposal lifecycle is incomplete")
        if consumed_counts[request_id] != 1:
            errors.append(f"{request_id}: initial proposal was not consumed exactly once")
    return errors
