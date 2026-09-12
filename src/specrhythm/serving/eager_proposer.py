"""Explicit Serial-eager Target rank-zero submitter, before real model forward."""

from __future__ import annotations

from specrhythm.continuation.trace import TRACE, references
from specrhythm.phase4.dual_commit import dual_greedy_acceptance
from specrhythm.phase4.serial import RoundRecord, token_prefix_hash
from specrhythm.serving.common import require
from specrhythm.serving.s2_proposer import S2SerialProposer


class EagerTimeline:
    """Actual independent phase intervals; no invented Serial non-overlap ordering."""

    def __init__(self, **values):
        require(
            all(type(v) is int and v >= 0 for v in values.values()), "invalid eager timestamps"
        )
        for prefix in ("draft", "transfer", "verify", "state_sync"):
            require(
                values[prefix + "_start_ns"] <= values[prefix + "_end_ns"],
                "reversed eager phase interval",
            )
        require(
            values["draft_end_ns"]
            <= values["transfer_end_ns"]
            <= values["verify_start_ns"]
            <= values["verify_end_ns"]
            <= values["state_sync_start_ns"],
            "eager proposal was used before ready",
        )
        self.values = values

    def to_dict(self):
        return dict(self.values)


class EagerSerialProposer(S2SerialProposer):
    def _gpu_qualification_report(self):
        return {
            "gpu_correctness_result": False,
            "gpu_performance_result": False,
            "rolling_eager_qualification": "PENDING",
        }

    @TRACE.observe("target_verify_start_hook")
    def on_target_verify_start(self, *, request_ids, scheduled_spec_token_ids):
        # Installs initial proposals, validates live identity and runs existing TP
        # barriers first. This callback itself is inserted before _model_forward.
        super().on_target_verify_start(
            request_ids=request_ids, scheduled_spec_token_ids=scheduled_spec_token_ids
        )
        if self.tp_rank != 0:
            # Rank zero must acknowledge the enqueue before any TP rank enters
            # the model forward. This barrier waits only for enqueue, not Draft.
            self.tp_group.barrier()
            return
        rows = []
        for internal in request_ids:
            if internal not in scheduled_spec_token_ids:
                continue
            rid = self.identity.stable_id(str(internal))
            proposal = self.requests[rid].pending_proposal
            require(
                tuple(scheduled_spec_token_ids[internal]) == proposal.proposal_token_ids,
                "Target froze different eager proposal tokens",
            )
            rows.append(
                {
                    "request_id": rid,
                    "round_id": proposal.round_id,
                    "proposal_id": proposal.runtime_provenance["rolling_proposal_id"],
                    "parent_prefix_hash": proposal.parent_prefix_hash,
                    "parent_prefix_len": proposal.parent_prefix_len,
                    "proposal_tokens": list(proposal.proposal_token_ids),
                }
            )
        if rows:
            with TRACE.span("target_enqueue_rpc", requests=references(rows),
                            batch_id=getattr(self.requests[rows[0]["request_id"]],
                                             "target_batch_id", None)):
                payload = {"requests": rows}
                if TRACE.enabled:
                    payload["_causal_batch_id"] = getattr(
                        self.requests[rows[0]["request_id"]], "target_batch_id", None)
                result = self.client.call("eager_enqueue", payload)
            require(result.get("enqueued") is True, "Draft continuation was not enqueued")
        self.tp_group.barrier()

    @TRACE.observe("target_feedback_hook")
    def on_target_verify_end(self, **kwargs):
        value = super().on_target_verify_end(**kwargs)
        if self.tp_rank == 0 and TRACE.enabled:
            rows = []
            for internal in kwargs["request_ids"]:
                rid = self.internal_to_stable.get(str(internal))
                state = self.requests.get(rid)
                if state is not None and state.pending_proposal is not None:
                    rows.append({"request_id": rid, "round_id": state.pending_proposal.round_id})
            TRACE.event("target_feedback_available", requests=rows)
        return value

    def _finalize_round(
        self,
        state,
        logical_prefix,
        generated,
        synchronization,
        response,
        next_proposal,
        *,
        terminal,
    ):
        proposal = state.pending_proposal
        require(
            proposal is not None and state.verify_start_ns and state.verify_end_ns,
            "eager finalization lacks live verified proposal",
        )
        decision = dual_greedy_acceptance(
            proposal.proposal_token_ids,
            logical_prefix[len(state.committed_token_ids) :],
            terminal=terminal,
        )
        require(
            list(decision.committed_token_ids)
            == synchronization.get("decision", {}).get("committed_token_ids"),
            "Target and eager owner disagree on committed tokens",
        )
        timeline = EagerTimeline(
            draft_start_ns=proposal.draft_start_ns,
            draft_end_ns=proposal.draft_end_ns,
            transfer_start_ns=state.transfer_start_ns,
            transfer_end_ns=state.transfer_end_ns,
            verify_start_ns=state.verify_start_ns,
            verify_end_ns=state.verify_end_ns,
            state_sync_start_ns=response["state_sync_start_ns"],
            state_sync_end_ns=response["state_sync_end_ns"],
            next_round_draft_start_ns=(
                next_proposal.draft_start_ns if next_proposal else response["state_sync_end_ns"]
            ),
        )
        record = RoundRecord(
            state.request_id,
            proposal.round_id,
            proposal.parent_prefix_len,
            proposal.parent_prefix_hash,
            proposal.proposal_token_ids,
            decision,
            state.maximum_new_tokens - len(generated),
            len(logical_prefix),
            synchronization["state"]["logical_draft_kv_length"],
            timeline,
            state.target_batch_id,
        )
        row = {
            "schema_version": "specrhythm.phase4-round-event.v1",
            **record.to_dict(),
            "committed_prefix_hash": token_prefix_hash(logical_prefix),
            "target_authority": True,
            "candidate_linear_sequence": True,
            "packed_tree_verification": False,
            "target_batch_request_ids": list(state.target_batch_request_ids),
            "vllm_dbo_microbatching": False,
            "rolling_eager": True,
            "source_continuation_id": proposal.runtime_provenance.get("source_continuation_id"),
        }
        self.round_log.append(row)
        self.round_records.append(row)
        state.committed_token_ids, state.generated_token_ids = logical_prefix, generated
        state.next_round_id += 1
        state.pending_proposal = None
