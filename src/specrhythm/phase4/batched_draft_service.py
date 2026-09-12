"""Opt-in Serial dispatch; the HF and Dual state machines remain unchanged."""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from specrhythm.phase4.draft_batch import DraftCommitPlan, DraftProposalPlan, unique_ids
from specrhythm.phase4.draft_service import DraftStateMachine
from specrhythm.phase4.serial import (
    PROTOCOL_VERSION,
    AcceptanceDecision,
    Proposal,
    greedy_acceptance,
    token_prefix_hash,
)
from specrhythm.phase4.transport import canonical_json_bytes


def write_immutable_report(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def serve_batched_draft(config, *, socket_path, event_log_path, ready_path) -> None:
    from specrhythm.phase4.draft_service import DraftUnixServer
    from specrhythm.phase4.transport import CheckpointJsonl
    from specrhythm.phase4.vllm_draft_backend import VllmBatchedDraftBackend

    report_path = ready_path.with_name("draft-backend-report.json")
    startup_path = ready_path.with_name("draft-startup.json")
    if any(p.exists() for p in (report_path, startup_path, ready_path, socket_path)):
        raise FileExistsError("Draft socket/startup/report artifacts must be fresh")
    backend = VllmBatchedDraftBackend(config)
    try:
        write_immutable_report(startup_path, backend.provenance)
        machine = BatchedDraftStateMachine(
            backend, candidate_budget=config.proposal_budget, report_path=report_path
        )
        DraftUnixServer(socket_path, machine, event_log=CheckpointJsonl(event_log_path)).serve(
            ready_path
        )
    finally:
        # Normal shutdown has already written the final report through the RPC.
        # A failed bind/serve must also release the one resident worker.
        if not backend.closed:
            backend._fail()
            try:
                backend.shutdown()
            finally:
                if not report_path.exists():
                    write_immutable_report(report_path, backend.report())


class BatchedDraftStateMachine(DraftStateMachine):
    def __init__(
        self, backend: Any, *, candidate_budget: int = 4, report_path: Path = None
    ) -> None:
        super().__init__(backend, candidate_budget=candidate_budget)
        self.report_path = report_path

    def batch_propose(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        unique_ids([str(row.get("request_id", "")) for row in rows])
        plans = []
        batch_started = time.monotonic_ns()
        for row in rows:
            request_id = str(row.get("request_id", ""))
            state = self._state(request_id)
            if state.finished or state.pending_proposal is not None:
                raise ValueError("finished/pending request cannot produce another proposal")
            if row.get("round_id") != state.next_round_id:
                raise ValueError("stale, duplicate, or out-of-order Draft round")
            if row.get("committed_prefix_len") != len(state.committed_token_ids):
                raise ValueError("Draft proposal parent prefix length mismatch")
            if row.get("committed_prefix_hash") != token_prefix_hash(state.committed_token_ids):
                raise ValueError("Draft proposal parent prefix hash mismatch")
            remaining = row.get("remaining_output_budget")
            if type(remaining) is not int or remaining < 1:
                raise ValueError("remaining output budget must be positive")
            budget = min(self.candidate_budget, max(remaining - 1, 0))
            if budget:
                plans.append(
                    DraftProposalPlan(
                        request_id,
                        state.next_round_id,
                        state.committed_token_ids,
                        budget,
                        tuple(row.get("eos_token_ids", ())),
                    )
                )
        started = time.monotonic_ns()
        before = self.backend.metrics.forwards["proposal"]
        tokens = self.backend.propose_many(plans)
        ended = time.monotonic_ns()
        actual_forwards = self.backend.metrics.forwards["proposal"] - before
        proposals = []
        for plan in plans:
            state = self._state(plan.request_id)
            values = tokens[plan.request_id]
            proposal = Proposal(
                protocol_version=PROTOCOL_VERSION,
                request_id=plan.request_id,
                round_id=plan.round_id,
                parent_prefix_len=len(plan.prefix),
                parent_prefix_hash=token_prefix_hash(plan.prefix),
                proposal_token_ids=values,
                proposal_eos=bool(values and values[-1] in plan.eos_token_ids),
                draft_start_ns=started,
                draft_end_ns=ended,
                transport_payload_bytes=0,
                model_provenance=self.backend.provenance,
                runtime_provenance={
                    "backend": self.backend.backend_name,
                    "number_of_model_forwards": max(len(values) - 1, 0),
                    "model_forwards_are_shared_batch_participation": True,
                    "actual_batch_model_forward_count": actual_forwards,
                    "batch_request_ids": [p.request_id for p in plans],
                    "full_context_replay": False,
                    "persistent_cross_round_kv": True,
                    "materialized_kv_length": self.backend.states[plan.request_id].materialized,
                },
            )
            for _ in range(3):
                size = len(canonical_json_bytes(proposal.to_dict()))
                if size == proposal.transport_payload_bytes:
                    break
                proposal = replace(proposal, transport_payload_bytes=size)
            state.pending_proposal = proposal
            state.proposal_count += 1
            proposals.append(proposal.to_dict())
        return {
            "proposals": proposals,
            "draft_batch_start_ns": batch_started,
            "draft_batch_end_ns": time.monotonic_ns(),
            "draft_microbatch_count": int(bool(plans)),
            "actual_batch_model_forward_count": actual_forwards,
            "single_persistent_model": True,
        }

    def _commit_plan(
        self, request_id: str, round_id: int, decision: AcceptanceDecision, final_hash: str
    ) -> DraftCommitPlan:
        state = self._state(request_id)
        proposal = state.pending_proposal
        assert proposal is not None
        return DraftCommitPlan(
            request_id,
            round_id,
            state.committed_token_ids,
            proposal.proposal_token_ids,
            len(decision.accepted_draft_token_ids),
            decision.target_correction_token_ids + decision.target_bonus_token_ids,
            final_hash,
            decision.terminal,
            len(decision.target_correction_token_ids),
            len(decision.target_bonus_token_ids),
        )

    def _publish_commit(
        self, plan: DraftCommitPlan, decision: AcceptanceDecision, physical: Mapping[str, Any]
    ) -> dict[str, Any]:
        state = self._state(plan.request_id)
        state.committed_token_ids = plan.final_prefix
        state.next_round_id += 1
        state.finished = plan.terminal
        state.pending_proposal = None
        state.pending_decision = None
        state.rollback_applied = False
        appended = {
            "request_id": plan.request_id,
            "round_id": plan.round_id,
            "committed_prefix_len": len(plan.final_prefix),
            "committed_prefix_hash": plan.final_prefix_hash,
            "logical_draft_kv_length": len(plan.final_prefix),
            "finished": plan.terminal,
            **physical,
        }
        return {
            "request_id": plan.request_id,
            "round_id": plan.round_id,
            "decision": {
                **{
                    key: list(getattr(decision, key))
                    for key in (
                        "accepted_draft_token_ids",
                        "rejected_draft_token_ids",
                        "target_correction_token_ids",
                        "target_bonus_token_ids",
                        "committed_token_ids",
                    )
                },
                **decision.accounting,
                "terminal": plan.terminal,
            },
            "rollback": {
                "request_id": plan.request_id,
                "round_id": plan.round_id,
                "accepted_draft_tokens": plan.accepted,
                "invalidated_draft_tokens": len(plan.proposal) - plan.accepted,
            },
            "state": appended,
        }

    def synchronize_and_batch_propose(
        self,
        synchronizations: Sequence[Mapping[str, Any]],
        proposals: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        unique_ids([str(row.get("request_id", "")) for row in synchronizations])
        started = time.monotonic_ns()
        plans = []
        decisions = []
        # Validate the whole cohort before any KV or control state mutation.
        for row in synchronizations:
            request_id = str(row.get("request_id", ""))
            round_id = row.get("round_id", -1)
            state = self._pending(request_id, round_id)
            decision = greedy_acceptance(
                state.pending_proposal.proposal_token_ids,
                row.get("committed_delta", ()),
                terminal=bool(row.get("terminal", False)),
            )
            plans.append(
                self._commit_plan(
                    request_id, round_id, decision, str(row.get("committed_prefix_hash", ""))
                )
            )
            decisions.append(decision)
        physical = self.backend.commit_many(plans)
        sync = [
            self._publish_commit(p, d, physical[p.request_id]) for p, d in zip(plans, decisions)
        ]
        ended = time.monotonic_ns()
        return {
            "synchronizations": sync,
            "state_sync_start_ns": started,
            "state_sync_end_ns": ended,
            **self.batch_propose(proposals),
        }

    def rollback_rejected_suffix(self, request_id: str, round_id: int) -> dict[str, Any]:
        state = self._decision(request_id, round_id)
        if state.rollback_applied:
            raise ValueError("rejected Draft suffix was already rolled back")
        state.rollback_applied = True
        return {
            "request_id": request_id,
            "round_id": round_id,
            "accepted_draft_tokens": len(state.pending_decision.accepted_draft_token_ids),
            "invalidated_draft_tokens": len(state.pending_decision.rejected_draft_token_ids),
            "physical_materialization_deferred_until_append": True,
        }

    def append_target_correction_or_bonus(self, request_id: str, round_id: int) -> dict[str, Any]:
        state = self._decision(request_id, round_id)
        if not state.rollback_applied:
            raise ValueError("Draft KV rollback must precede correction/bonus append")
        decision = state.pending_decision
        plan = self._commit_plan(
            request_id,
            round_id,
            decision,
            token_prefix_hash(state.committed_token_ids + decision.committed_token_ids),
        )
        physical = self.backend.commit_many((plan,))
        return self._publish_commit(plan, decision, physical[request_id])["state"]

    def shutdown(self) -> dict[str, Any]:
        result = super().shutdown()
        if self.report_path is not None:
            from specrhythm.phase4.manifest import sha256_file

            write_immutable_report(self.report_path, self.backend.report())
            result["draft_backend_report_file"] = self.report_path.name
            result["draft_backend_report_sha256"] = sha256_file(self.report_path)
        return result
