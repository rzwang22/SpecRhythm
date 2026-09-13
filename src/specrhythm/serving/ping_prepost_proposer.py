"""Target publication is asynchronous to Draft settlement; only claimed proposals verify."""

import os
import time

from specrhythm.phase4.serial import Proposal, token_prefix_hash
from specrhythm.serving.common import require
from specrhythm.serving.ping_prepost import PROTOCOL
from specrhythm.serving.prepost_proposer import PrePostProposer
from specrhythm.serving.s2_pool import control


class FeedbackClient:
    def __init__(self, client):
        self.client = client

    def call(self, operation, payload):
        if operation == "synchronize_and_batch_propose":
            return self.client.call("pp_feedback", payload)
        return self.client.call(operation, payload)


class PingPrePostProposer(PrePostProposer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = FeedbackClient(self.client)

    def on_target_verify_start(self, *, request_ids, scheduled_spec_token_ids):
        if self.tp_rank == 0:
            packet = control()
            claims = packet["pp_admission"]["claims"]
            by_id = {r["request_id"]: r for r in claims}
            require(len(by_id) == len(claims), "duplicate Target claim")
            for internal in request_ids:
                rid = self.identity.stable_id(str(internal))
                claim, state = by_id[rid], self.requests[rid]
                proposal = Proposal.from_dict(claim["proposal"])
                require(
                    state.pending_proposal is None
                    and proposal.round_id == state.next_round_id
                    and proposal.parent_prefix_hash == token_prefix_hash(state.committed_token_ids)
                    and list(proposal.proposal_token_ids) == scheduled_spec_token_ids[internal],
                    "claimed proposal differs from Target state",
                )
                state.pending_proposal = proposal
                state.transfer_start_ns = claim["claimed_ns"]
                state.transfer_end_ns = time.monotonic_ns()
                state.verify_start_ns = state.verify_end_ns = 0
                state.ping_claim = claim
        super().on_target_verify_start(
            request_ids=request_ids, scheduled_spec_token_ids=scheduled_spec_token_ids
        )

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
        decision = self.acceptance_rule(
            proposal.proposal_token_ids,
            logical_prefix[len(state.committed_token_ids) :],
            terminal=terminal,
        )
        require(
            next_proposal is None
            and synchronization["physical_settlement"] == "PENDING"
            and list(decision.committed_token_ids)
            == synchronization["decision"]["committed_token_ids"],
            "asynchronous feedback acknowledgement differs",
        )
        row = dict(
            schema_version="specrhythm.phase4-round-event.v1",
            request_id=state.request_id,
            round_id=proposal.round_id,
            parent_prefix_len=proposal.parent_prefix_len,
            parent_prefix_hash=proposal.parent_prefix_hash,
            proposal_token_ids=list(proposal.proposal_token_ids),
            proposal_length=len(proposal.proposal_token_ids),
            **decision.accounting,
            accepted_draft_token_ids=list(decision.accepted_draft_token_ids),
            rejected_draft_token_ids=list(decision.rejected_draft_token_ids),
            target_correction_token_ids=list(decision.target_correction_token_ids),
            target_bonus_token_ids=[],
            committed_token_ids=list(decision.committed_token_ids),
            committed_prefix_hash=token_prefix_hash(logical_prefix),
            terminal=terminal,
            remaining_output_budget=state.maximum_new_tokens - len(generated),
            logical_target_kv_length=len(logical_prefix),
            logical_draft_kv_length=None,
            draft_settlement_status="PENDING; join owner settlement by request/version",
            target_microbatch_id=state.target_batch_id,
            ping_target_batch_id=state.ping_claim["target_batch_id"],
            home_cohort=state.ping_claim["home_cohort"],
            ping_prepost_protocol=("specrhythm.uniform-k3.v1"
                if os.environ.get("SR_S2_MODE", "").endswith("-k3") else PROTOCOL),
            target_authority=True,
            timeline=dict(
                draft_start_ns=proposal.draft_start_ns,
                draft_end_ns=proposal.draft_end_ns,
                transfer_start_ns=state.transfer_start_ns,
                transfer_end_ns=state.transfer_end_ns,
                verify_start_ns=state.verify_start_ns,
                verify_end_ns=state.verify_end_ns,
                state_sync_start_ns=response["state_sync_start_ns"],
                state_sync_end_ns=response["state_sync_end_ns"],
                state_sync_scope="validated feedback acknowledgement only",
            ),
        )
        self.round_log.append(row)
        self.round_records.append(row)
        state.committed_token_ids, state.generated_token_ids = logical_prefix, generated
        state.next_round_id += 1
        state.pending_proposal = None
