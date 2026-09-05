"""Pinned-vLLM ready-only scheduler plugin for Phase 4B.

This module is imported by the vLLM EngineCore process only.  It subclasses the
stock scheduler and gates decode eligibility; allocation, preemption, paged KV,
and the stock scheduling algorithm remain owned by vLLM.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Mapping, Optional

from specrhythm.phase4.admissibility import (
    AdmissibilityDecision,
    AdmissibilitySnapshot,
    ExecutionPhase,
    ProposalEvidence,
    ScheduledOperation,
    SchedulerRequestState,
    decide_admissibility,
    decision_event,
)
from specrhythm.phase4.dual import DualProposal
from specrhythm.phase4.dual_service import DualDraftClient
from specrhythm.phase4.request_identity import (
    FrozenPromptIdentityMap,
    resolve_historical_ready_request,
)
from specrhythm.phase4.resident_setup import load_setup_ready
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.stock_vllm import load_smoke_requests
from specrhythm.phase4.transport import CheckpointJsonl

try:
    from vllm.v1.core.sched.scheduler import Scheduler
except ImportError as error:  # pragma: no cover - exercised only in GPU environment
    raise RuntimeError("DualBatchScheduler requires pinned vLLM v0.25.1") from error


class DualBatchScheduler(Scheduler):
    """Stock scheduler with a default-off proposal-readiness decode gate."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if os.environ.get("SR_PHASE4_DUAL_BATCH") != "1":
            raise RuntimeError("DualBatchScheduler is fail-closed unless explicitly enabled")
        socket_value = os.environ.get("SR_PHASE4_DUAL_DRAFT_SOCKET")
        event_value = os.environ.get("SR_PHASE4_DUAL_SCHEDULER_EVENTS")
        if not socket_value or not event_value:
            raise RuntimeError("Dual-Batch scheduler socket/event paths are required")
        self._dual_client = DualDraftClient(Path(socket_value), timeout_seconds=2.0)
        self._dual_events = CheckpointJsonl(Path(event_value))
        lifecycle_value = os.environ.get("SR_PHASE4_PROPOSAL_LIFECYCLE_EVENTS")
        self._dual_proposal_lifecycle = (
            CheckpointJsonl(Path(lifecycle_value)) if lifecycle_value else None
        )
        self._dual_resident = os.environ.get("SR_PHASE4_DUAL_RESIDENT") == "1"
        if self._dual_resident and self._dual_proposal_lifecycle is None:
            raise RuntimeError("resident Dual requires proposal lifecycle evidence")
        self._dual_microbatch_size = int(
            os.environ.get("SR_PHASE4_DUAL_MICROBATCH_SIZE", "1")
        )
        if self._dual_microbatch_size < 1:
            raise RuntimeError("Dual-Batch microbatch size must be positive")
        workload_value = os.environ.get("SR_PHASE4_WORKLOAD")
        if not workload_value:
            raise RuntimeError("Dual-Batch scheduler requires the frozen workload path")
        request_count = int(os.environ.get("SR_PHASE4_REQUEST_COUNT", "5"))
        definitions = load_smoke_requests(
            Path(workload_value),
            request_count,
            require_task_mixture=request_count in {5, 100},
        )
        self._dual_identity = FrozenPromptIdentityMap.from_definitions(definitions)
        self._dual_expected_ids = tuple(row.request_id for row in definitions)
        self._dual_setup_ready: Optional[Mapping[str, Any]] = None
        if self._dual_resident:
            self._dual_setup_ready_path = _required_path(
                "SR_PHASE4_RESIDENT_SETUP_READY"
            )
            self._dual_manifest_path = _required_path(
                "SR_PHASE4_DECODE_READY_MANIFEST"
            )
        # These collections are stable-ID keyed. vLLM-owned request tables and
        # cadence fields remain internal-ID keyed.
        self._dual_tail_ready: set[str] = set()
        self._dual_tail_ready_ns: dict[str, int] = {}
        self._dual_proposals: dict[str, DualProposal] = {}
        self._dual_consumed_proposals: set[str] = set()
        self._dual_drafting: set[str] = set()
        self._dual_retired_ready_events: list[dict[str, Any]] = []
        self._dual_retired_results: dict[tuple[Any, ...], Optional[DualProposal]] = {}
        self._dual_dropped_proposals: dict[str, DualProposal] = {}
        self._dual_decisions: dict[str, tuple[AdmissibilitySnapshot, AdmissibilityDecision]] = {}
        ttl_seconds = float(os.environ.get("SR_PHASE4_PROPOSAL_TTL_SECONDS", "0"))
        if ttl_seconds < 0:
            raise RuntimeError("proposal TTL must be non-negative")
        self._dual_proposal_ttl_ns = int(ttl_seconds * 1_000_000_000)
        self._dual_test_coordination = os.environ.get(
            "SR_PHASE4_DUAL_TEST_COORDINATION", "none"
        )
        if self._dual_test_coordination not in {
            "none",
            "one-ready",
            "two-ready",
        }:
            raise RuntimeError("unknown test-only Dual readiness coordination mode")
        self._dual_test_coordination_satisfied = False
        existing_cycles = self._dual_events.read()
        self._dual_cycle_id = 1 + max(
            (int(row.get("cycle_id", -1)) for row in existing_cycles), default=-1
        )

    def schedule(self, *args: Any, **kwargs: Any) -> Any:
        self._bind_vllm_requests()
        self._refresh_resident_readiness()
        poll_start = time.monotonic_ns()
        already_ready = sum(
            bool(request.spec_token_ids)
            or self._dual_identity.stable_id(str(request.request_id))
            in self._dual_tail_ready
            for request in self.running
        )
        available = max(0, self._dual_microbatch_size - already_ready)
        if self._dual_resident and self._dual_setup_ready is None:
            available = 0
        if (
            available
            and self._dual_test_coordination == "one-ready"
            and not self._dual_test_coordination_satisfied
        ):
            available = min(available, 1)
        if (
            available
            and self._dual_test_coordination == "two-ready"
            and not self._dual_test_coordination_satisfied
        ):
            status = self._dual_client.call("status", {})
            if len(status.get("ready_request_ids", ())) < 2:
                available = 0
            else:
                self._dual_test_coordination_satisfied = True
        response = (
            self._dual_client.call("poll_ready", {"limit": available})
            if available
            else {
                "ready": [],
                "pending_request_ids": [],
                "failures": {},
                "blocking_on_draft_gpu": False,
            }
        )
        failures = response.get("failures", {})
        if failures:
            raise RuntimeError(f"asynchronous Draft failure: {failures}")
        for result in response.get("ready", ()):
            self._accept_ready_result(result)
        if (
            self._dual_test_coordination == "one-ready"
            and response.get("ready")
        ):
            self._dual_test_coordination_satisfied = True
        if available:
            pending = response.get("pending_request_ids", ())
            # A poll snapshot can still mention work for a request retired by
            # stock vLLM. Do not reintroduce it after ready-result cleanup.
            self._dual_drafting = {
                item
                for item in pending
                if (request := resolve_historical_ready_request(
                    item, self._dual_identity, self.requests
                )[1]) is not None and not request.is_finished()
            }
        self._dual_decisions = {
            str(request.request_id): self._decision_for(request)
            for request in self.requests.values()
        }

        # The independent pinned-vLLM scheduler patch invokes
        # _request_admissible_for_schedule immediately before stock allocation.
        # Queue traversal, token budgets, KV allocation, preemption and cadence
        # remain stock behavior.  In particular, this does not mutate
        # next_decode_eligible_step or depend on current_step arithmetic.
        output = super().schedule(*args, **kwargs)

        scheduled_ids = tuple(output.num_scheduled_tokens)
        verified_ids = tuple(output.scheduled_spec_decode_tokens)
        stable_scheduled_ids = tuple(
            self._dual_identity.stable_id(str(request_id))
            for request_id in scheduled_ids
        )
        stable_verified_ids = tuple(
            self._dual_identity.stable_id(str(request_id))
            for request_id in verified_ids
        )
        decision_rows = []
        for internal_id, (snapshot, decision) in self._dual_decisions.items():
            scheduled = internal_id in scheduled_ids
            scheduled_count = int(output.num_scheduled_tokens.get(internal_id, 0))
            positions = (
                range(
                    snapshot.num_computed_tokens,
                    snapshot.num_computed_tokens + scheduled_count,
                )
                if scheduled
                else ()
            )
            decision_rows.append(
                decision_event(
                    snapshot,
                    decision,
                    cycle_id=self._dual_cycle_id,
                    scheduler_step=int(self.current_step),
                    scheduled=scheduled,
                    target_input_positions=positions,
                )
            )
        for stable_id in stable_scheduled_ids:
            self._dual_tail_ready.discard(stable_id)
            self._dual_tail_ready_ns.pop(stable_id, None)
        for stable_id in stable_verified_ids:
            proposal = self._dual_proposals.get(stable_id)
            if proposal is not None:
                if proposal.proposal_id in self._dual_consumed_proposals:
                    raise RuntimeError("one Dual proposal was consumed more than once")
                self._dual_consumed_proposals.add(proposal.proposal_id)
                self._emit_proposal_lifecycle(
                    proposal,
                    "CONSUMED",
                    internal_request_id=self._dual_identity.internal_id(stable_id),
                    reason="stock-scheduled-spec-decode-evidence",
                )
        self._dual_events.append(
            {
                "schema_version": "specrhythm.phase4b-scheduler-cycle.v1",
                "cycle_id": self._dual_cycle_id,
                "poll_start_ns": poll_start,
                "poll_end_ns": time.monotonic_ns(),
                "poll_blocking_on_draft_gpu": False,
                "scheduled_request_ids": list(stable_scheduled_ids),
                "internal_scheduled_request_ids": list(scheduled_ids),
                "verify_request_ids": list(stable_verified_ids),
                "internal_verify_request_ids": list(verified_ids),
                "ready_proposal_ids": [
                    self._dual_proposals[item].proposal_id
                    for item in stable_verified_ids
                    if item in self._dual_proposals
                ],
                "request_admissibility": decision_rows,
                "retired_ready_results": list(self._dual_retired_ready_events),
                "admissibility_hook": "explicit-request-predicate",
                "cadence_field_used_for_draft_readiness": False,
                "scheduler_class": type(self).__name__,
                "stock_scheduler_delegated": True,
                "global_decode_ready": self._dual_setup_ready is not None,
                "test_only_readiness_coordination": self._dual_test_coordination,
            }
        )
        self._dual_retired_ready_events.clear()
        self._dual_cycle_id += 1
        return output

    def _request_admissible_for_schedule(self, request: Any) -> bool:
        """Pinned scheduler hook; default stock schedulers have no such hook."""

        internal_id = str(request.request_id)
        snapshot_and_decision = self._dual_decisions.get(internal_id)
        if snapshot_and_decision is None:
            snapshot_and_decision = self._decision_for(request)
            self._dual_decisions[internal_id] = snapshot_and_decision
        return snapshot_and_decision[1].admissible

    def _decision_for(
        self, request: Any
    ) -> tuple[AdmissibilitySnapshot, AdmissibilityDecision]:
        internal_id = str(request.request_id)
        stable_id = self._dual_identity.stable_id(internal_id)
        prompt_count = int(
            getattr(request, "num_prompt_tokens", len(request.prompt_token_ids))
        )
        computed = int(request.num_computed_tokens)
        phase = (
            ExecutionPhase.SETUP_PREFILL
            if computed < prompt_count
            else ExecutionPhase.TIMED_DECODE
        )
        proposal = self._dual_proposals.get(stable_id)
        prefix = tuple(int(item) for item in request.all_token_ids)
        expected_version = 1 if proposal is None else proposal.prefix_version
        expected_round = 0 if proposal is None else proposal.round_id
        if request.is_finished():
            state = SchedulerRequestState.TERMINAL
        elif phase is ExecutionPhase.SETUP_PREFILL:
            state = SchedulerRequestState.WAITING_DRAFT
        elif stable_id in self._dual_tail_ready:
            state = SchedulerRequestState.TARGET_TAIL_READY
        elif request.spec_token_ids and proposal is not None:
            state = SchedulerRequestState.VERIFY_READY
        elif stable_id in self._dual_drafting:
            state = SchedulerRequestState.DRAFTING
        else:
            state = SchedulerRequestState.WAITING_DRAFT
        evidence = None
        if proposal is not None:
            expires = (
                proposal.created_timestamp_ns + self._dual_proposal_ttl_ns
                if self._dual_proposal_ttl_ns
                else None
            )
            evidence = ProposalEvidence(
                request_id=proposal.request_id,
                internal_request_id=internal_id,
                prefix_version=proposal.prefix_version,
                prefix_token_count=proposal.prefix_token_count,
                prefix_token_sha256=proposal.prefix_token_sha256,
                round_id=proposal.round_id,
                proposal_token_ids=proposal.proposal_token_ids,
                ready_timestamp_ns=proposal.created_timestamp_ns,
                expires_timestamp_ns=expires,
                consumed=proposal.proposal_id in self._dual_consumed_proposals,
            )
        snapshot = AdmissibilitySnapshot(
            internal_request_id=internal_id,
            stable_request_id=stable_id,
            state=state,
            execution_phase=phase,
            prefix_version=expected_version,
            round_id=expected_round,
            prefix_token_count=len(prefix),
            prefix_token_sha256=token_prefix_hash(prefix),
            num_computed_tokens=computed,
            num_output_tokens=int(request.num_output_tokens),
            spec_token_ids=tuple(int(item) for item in request.spec_token_ids),
            proposal=evidence,
            now_ns=time.monotonic_ns(),
            target_tail_ready_timestamp_ns=self._dual_tail_ready_ns.get(stable_id),
        )
        decision = decide_admissibility(snapshot)
        if (
            self._dual_resident
            and self._dual_setup_ready is None
            and phase is ExecutionPhase.TIMED_DECODE
        ):
            decision = AdmissibilityDecision(
                admissible=False,
                operation=ScheduledOperation.NONE,
                reason="bootstrap-ready-awaiting-global-boundary",
                proposal_present=evidence is not None,
                proposal_valid=False,
                proposal_ready_timestamp_ns=(
                    evidence.ready_timestamp_ns if evidence is not None else None
                ),
            )
        return snapshot, decision

    def _accept_ready_result(self, result: Mapping[str, Any]) -> None:
        if not isinstance(result, Mapping):
            raise RuntimeError("ready Draft result is not an object")
        request_id = result.get("request_id")
        internal_request_id, request = resolve_historical_ready_request(
            request_id, self._dual_identity, self.requests
        )
        proposal_value = result.get("proposal")
        if result.get("terminal", False) is not False:
            raise RuntimeError("ready Draft result must contain nonterminal work")
        proposal = None
        ready_ns = None
        if proposal_value is None:
            if result.get("target_tail") is not True:
                raise RuntimeError("ready Draft result contains neither proposal nor tail")
            ready_ns = result.get("target_tail_ready_ns")
            if type(ready_ns) is not int or ready_ns <= 0:
                raise RuntimeError("ready Draft target tail lacks a readiness timestamp")
        else:
            if not isinstance(proposal_value, Mapping):
                raise RuntimeError("ready proposal payload is not an object")
            for field in ("request_id", "proposal_id", "prefix_token_sha256"):
                if not isinstance(proposal_value.get(field), str):
                    raise ValueError(f"ready proposal {field} must be a string")
            if re.fullmatch(r"[0-9a-f]{64}", proposal_value["prefix_token_sha256"]) is None:
                raise ValueError("ready proposal prefix_token_sha256 is malformed")
            proposal = DualProposal.from_dict(proposal_value)
            if "proposal_length" in proposal_value and (
                type(proposal_value["proposal_length"]) is not int
                or proposal_value["proposal_length"] != proposal.proposal_length
            ):
                raise ValueError("ready proposal_length disagrees with proposal tokens")
            if proposal.request_id != request_id:
                raise RuntimeError("ready proposal request_id disagrees with ready result")
            if result.get("target_tail", False) is not False:
                raise RuntimeError("ready proposal also claims a target tail")
            if proposal.proposal_id in self._dual_consumed_proposals:
                raise RuntimeError("one Dual proposal was consumed more than once")
        if request is None or request.is_finished():
            self._drop_retired_ready_result(
                request_id,
                internal_request_id,
                proposal=proposal,
                ready_ns=ready_ns,
                reason=(
                    "request-retired-before-ready" if request is None else "terminal-request"
                ),
            )
            return
        if proposal is None:
            self._dual_tail_ready.add(request_id)
            self._dual_tail_ready_ns[request_id] = ready_ns
            self._dual_drafting.discard(request_id)
            return
        self._emit_new_proposal(proposal, internal_request_id)
        prefix = tuple(int(item) for item in request.all_token_ids)
        errors = []
        if proposal.request_id != request_id:
            errors.append("request_id")
        if proposal.prefix_token_count != len(prefix):
            errors.append("prefix_token_count")
        if proposal.prefix_token_sha256 != token_prefix_hash(prefix):
            errors.append("prefix_token_sha256")
        previous = self._dual_proposals.get(request_id)
        expected_version = 1 if previous is None else previous.prefix_version + 1
        expected_round = 0 if previous is None else previous.round_id + 1
        if proposal.prefix_version != expected_version:
            errors.append("prefix_version")
        if proposal.round_id != expected_round:
            errors.append("round_id")
        if request.spec_token_ids:
            errors.append("second_unverified_proposal")
        if errors:
            self._emit_proposal_lifecycle(
                proposal,
                "DROPPED_STALE",
                internal_request_id=internal_request_id,
                reason=",".join(errors),
            )
            self._dual_events.append(
                {
                    "schema_version": "specrhythm.phase4b-stale-proposal.v1",
                    "request_id": request_id,
                    "internal_request_id": internal_request_id,
                    "proposal_id": proposal.proposal_id,
                    "discarded": True,
                    "verified": False,
                    "reasons": errors,
                }
            )
            raise RuntimeError("stale proposal rejected before verification: " + ", ".join(errors))
        request.spec_token_ids = list(proposal.proposal_token_ids)
        self._dual_proposals[request_id] = proposal
        self._dual_drafting.discard(request_id)
        self._emit_proposal_lifecycle(
            proposal,
            "INSTALLED",
            internal_request_id=internal_request_id,
            reason="vllm-spec-token-ids-installed",
        )

    def _drop_retired_ready_result(
        self,
        request_id: str,
        internal_request_id: str,
        *,
        proposal: Optional[DualProposal],
        ready_ns: Optional[int],
        reason: str,
    ) -> None:
        """Discard validated late work without touching any vLLM request."""

        key = (
            ("proposal", proposal.proposal_id)
            if proposal is not None
            else ("target-tail", request_id, ready_ns)
        )
        if key in self._dual_retired_results:
            if self._dual_retired_results[key] != proposal:
                raise RuntimeError("replayed retired proposal changed its payload")
            return
        previous = self._dual_proposals.get(request_id)
        if proposal is not None:
            for dropped in self._dual_dropped_proposals.values():
                if dropped.proposal_id == proposal.proposal_id and dropped != proposal:
                    raise RuntimeError("replayed retired proposal changed its payload")
                if (
                    dropped.request_id == request_id
                    and dropped.round_id == proposal.round_id
                    and dropped.proposal_id != proposal.proposal_id
                ):
                    raise RuntimeError("multiple Dual proposals claim one request round")
        if previous is not None and proposal is not None:
            if previous.proposal_id == proposal.proposal_id and previous != proposal:
                raise RuntimeError("retired proposal changed its installed payload")
            if (
                previous.round_id == proposal.round_id
                and previous.proposal_id != proposal.proposal_id
            ):
                raise RuntimeError("multiple Dual proposals claim one request round")
        # Consumed proposals already have a complete lifecycle. An installed but
        # unconsumed proposal instead needs a single terminal DROP before cleanup.
        if previous is not None and previous.proposal_id not in self._dual_consumed_proposals:
            self._drop_retired_proposal(previous, internal_request_id, reason=reason)
        if proposal is not None:
            if (
                proposal.proposal_id not in self._dual_dropped_proposals
                and (previous is None or previous.proposal_id != proposal.proposal_id)
            ):
                self._emit_new_proposal(proposal, internal_request_id)
            self._drop_retired_proposal(proposal, internal_request_id, reason=reason)
        self._dual_drafting.discard(request_id)
        self._dual_tail_ready.discard(request_id)
        self._dual_tail_ready_ns.pop(request_id, None)
        self._dual_proposals.pop(request_id, None)
        self._dual_retired_results[key] = proposal
        self._dual_retired_ready_events.append(
            {
                "schema_version": "specrhythm.phase4b2-retired-ready-result.v1",
                "request_id": request_id,
                "internal_request_id": internal_request_id,
                "result_kind": "proposal" if proposal is not None else "target-tail",
                "proposal_id": proposal.proposal_id if proposal is not None else None,
                "target_tail_ready_ns": ready_ns,
                "timestamp_ns": time.monotonic_ns(),
                "reason": reason,
                "discarded": True,
                "installed": False,
                "verified": False,
            }
        )

    def _drop_retired_proposal(
        self, proposal: DualProposal, internal_request_id: str, *, reason: str
    ) -> None:
        if proposal.proposal_id not in self._dual_dropped_proposals:
            self._emit_proposal_lifecycle(
                proposal,
                "DROPPED_STALE",
                internal_request_id=internal_request_id,
                reason=reason,
            )
            self._dual_dropped_proposals[proposal.proposal_id] = proposal

    def _emit_new_proposal(
        self, proposal: DualProposal, internal_request_id: str
    ) -> None:
        self._emit_proposal_lifecycle(
            proposal,
            "CREATED",
            internal_request_id=internal_request_id,
            timestamp_ns=proposal.created_timestamp_ns,
            reason="draft-model-proposal-created",
        )
        self._emit_proposal_lifecycle(
            proposal,
            "PUBLISHED",
            internal_request_id=internal_request_id,
            reason="scheduler-polled-ready-service",
        )

    def _refresh_resident_readiness(self) -> None:
        if (
            not self._dual_resident
            or self._dual_setup_ready is not None
            or not self._dual_setup_ready_path.is_file()
        ):
            return
        self._dual_setup_ready = load_setup_ready(
            self._dual_setup_ready_path,
            manifest_path=self._dual_manifest_path,
            consumer="dual-batch",
            expected_request_ids=self._dual_expected_ids,
        )

    def _emit_proposal_lifecycle(
        self,
        proposal: DualProposal,
        lifecycle_state: str,
        *,
        internal_request_id: str,
        reason: str,
        timestamp_ns: Optional[int] = None,
    ) -> None:
        if self._dual_proposal_lifecycle is None:
            return
        self._dual_proposal_lifecycle.append(
            {
                "schema_version": "specrhythm.phase4b1-proposal-lifecycle.v1",
                "proposal_id": proposal.proposal_id,
                "request_id": proposal.request_id,
                "internal_request_id": internal_request_id,
                "round_id": proposal.round_id,
                "prefix_version": proposal.prefix_version,
                "prefix_token_count": proposal.prefix_token_count,
                "prefix_token_sha256": proposal.prefix_token_sha256,
                "proposal_token_ids": list(proposal.proposal_token_ids),
                "proposal_length": proposal.proposal_length,
                "draft_start_ns": proposal.draft_start_ns,
                "draft_end_ns": proposal.draft_end_ns,
                "lifecycle_state": lifecycle_state,
                "timestamp_ns": timestamp_ns or time.monotonic_ns(),
                "reason": reason,
            }
        )

    def _bind_vllm_requests(self) -> None:
        """Bind opaque scheduler IDs from token prefixes; never parse ID text."""

        for internal_request_id, request in self.requests.items():
            if request.is_finished():
                continue
            table_id = str(internal_request_id)
            object_id = str(request.request_id)
            if table_id != object_id:
                raise RuntimeError("vLLM request table key disagrees with request.request_id")
            self._dual_identity.bind(table_id, tuple(int(item) for item in request.all_token_ids))


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required resident Dual environment variable is missing: {name}")
    return Path(value).resolve()
