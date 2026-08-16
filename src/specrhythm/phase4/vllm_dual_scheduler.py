"""Pinned-vLLM ready-only scheduler plugin for Phase 4B.

This module is imported by the vLLM EngineCore process only.  It subclasses the
stock scheduler and gates decode eligibility; allocation, preemption, paged KV,
and the stock scheduling algorithm remain owned by vLLM.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Mapping

from specrhythm.phase4.admissibility import (
    AdmissibilityDecision,
    AdmissibilitySnapshot,
    ExecutionPhase,
    ProposalEvidence,
    SchedulerRequestState,
    decide_admissibility,
    decision_event,
)
from specrhythm.phase4.dual import DualProposal
from specrhythm.phase4.dual_service import DualDraftClient
from specrhythm.phase4.request_identity import (
    FrozenPromptIdentityMap,
    resolve_stable_ready_request,
)
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
        # These collections are stable-ID keyed. vLLM-owned request tables and
        # cadence fields remain internal-ID keyed.
        self._dual_tail_ready: set[str] = set()
        self._dual_proposals: dict[str, DualProposal] = {}
        self._dual_consumed_proposals: set[str] = set()
        self._dual_drafting: set[str] = set()
        self._dual_decisions: dict[str, tuple[AdmissibilitySnapshot, AdmissibilityDecision]] = {}
        ttl_seconds = float(os.environ.get("SR_PHASE4_PROPOSAL_TTL_SECONDS", "0"))
        if ttl_seconds < 0:
            raise RuntimeError("proposal TTL must be non-negative")
        self._dual_proposal_ttl_ns = int(ttl_seconds * 1_000_000_000)
        existing_cycles = self._dual_events.read()
        self._dual_cycle_id = 1 + max(
            (int(row.get("cycle_id", -1)) for row in existing_cycles), default=-1
        )

    def schedule(self, *args: Any, **kwargs: Any) -> Any:
        self._bind_vllm_requests()
        poll_start = time.monotonic_ns()
        already_ready = sum(
            bool(request.spec_token_ids)
            or self._dual_identity.stable_id(str(request.request_id))
            in self._dual_tail_ready
            for request in self.running
        )
        available = max(0, self._dual_microbatch_size - already_ready)
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
        if available:
            pending = response.get("pending_request_ids", ())
            self._dual_drafting = {str(item) for item in pending}
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
        for stable_id in stable_verified_ids:
            proposal = self._dual_proposals.get(stable_id)
            if proposal is not None:
                self._dual_consumed_proposals.add(proposal.proposal_id)
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
                "admissibility_hook": "explicit-request-predicate",
                "cadence_field_used_for_draft_readiness": False,
                "scheduler_class": type(self).__name__,
                "stock_scheduler_delegated": True,
            }
        )
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
        )
        return snapshot, decide_admissibility(snapshot)

    def _accept_ready_result(self, result: Mapping[str, Any]) -> None:
        request_id = str(result.get("request_id", ""))
        internal_request_id, request = resolve_stable_ready_request(
            request_id, self._dual_identity, self.requests
        )
        if request.is_finished():
            raise RuntimeError(
                f"ready proposal belongs to terminal stable request {request_id}"
            )
        proposal_value = result.get("proposal")
        if proposal_value is None:
            if result.get("target_tail") is not True:
                raise RuntimeError("ready Draft result contains neither proposal nor tail")
            self._dual_tail_ready.add(request_id)
            self._dual_drafting.discard(request_id)
            return
        if not isinstance(proposal_value, Mapping):
            raise RuntimeError("ready proposal payload is not an object")
        proposal = DualProposal.from_dict(proposal_value)
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
