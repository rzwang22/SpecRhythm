"""S2 admission-time initial proposals; verification/acceptance algorithms are inherited."""

from specrhythm.phase4.dual_rhythm import AnnotatedLog, CohortDraftClient, cohort_for
from specrhythm.phase4.resident_vllm import ResidentTargetProposer
from specrhythm.phase4.serial import Proposal, token_prefix_hash
from specrhythm.phase4.vllm_dual import DualBatchRemoteProposer
from specrhythm.phase4.vllm_remote import RemoteDraftProposer
from specrhythm.serving.common import require
from specrhythm.serving.s2_pool import control


def assignment():
    return {
        rid: r["cohort"] for rid, r in control()["requests"].items() if r["cohort"] is not None
    }


class DynamicClient:
    def __init__(self, client):
        self.client = client

    def call(self, operation, payload):
        return CohortDraftClient(self.client, assignment()).call(operation, payload)


class S2TargetProposer(ResidentTargetProposer):
    pass


class S2SerialProposer(RemoteDraftProposer):
    def _generate_initial_resident_proposals(self, measurement_start_ns):
        return ()

    def on_target_verify_start(self, *, request_ids, scheduled_spec_token_ids):
        if self.tp_rank == 0:
            packet = control()
            for internal in request_ids:
                if internal not in scheduled_spec_token_ids:
                    continue
                rid = self.identity.stable_id(str(internal))
                state = self.requests[rid]
                if state.pending_proposal is None and len(state.generated_token_ids) == 1:
                    require(
                        packet["requests"][rid]["state"] == "ACTIVE",
                        "initial Serial verification before admission",
                        request_id=rid,
                    )
                    row = packet["initial_proposals"][rid]
                    proposal = Proposal.from_dict(row["proposal"])
                    require(
                        proposal.parent_prefix_len == len(state.committed_token_ids)
                        and proposal.parent_prefix_hash
                        == token_prefix_hash(state.committed_token_ids),
                        "S2 worker initial proposal prefix mismatch",
                        request_id=rid,
                    )
                    state.pending_proposal = proposal
                    state.transfer_start_ns = row["service_send_ns"]
                    state.transfer_end_ns = row["transport_end_ns"]
        return super().on_target_verify_start(
            request_ids=request_ids, scheduled_spec_token_ids=scheduled_spec_token_ids
        )


class S2PingProposer(DualBatchRemoteProposer):
    def __init__(self, config):
        super().__init__(config)
        self.client = DynamicClient(self.client)
        self.proposal_log = AnnotatedLog(self.proposal_log, self._cohort)
        self.verification_log = AnnotatedLog(self.verification_log, self._cohort)

    def _cohort(self, row):
        ids = row.get("verify_request_ids") or [row["request_id"]]
        return {
            "logical_cohort": cohort_for(assignment(), ids),
            "dual_rhythm": "s2-dynamic-pingpong",
        }

    def _enqueue_initial_proposals(self, measurement_start_ns):
        return

    def _admitted_initial(self):
        if self.tp_rank != 0:
            return
        packet = control()
        for rid, row in packet.get("initial_enqueues", {}).items():
            state = self.requests.get(rid)
            if state is not None and state.lifecycle == "DRAFT_READY":
                self._transition(
                    rid,
                    "DRAFTING",
                    reason="S2 admitted initial proposal",
                    timestamp_ns=row["enqueue_start_ns"],
                )

    def on_target_verify_start(self, **kwargs):
        self._admitted_initial()
        return super().on_target_verify_start(**kwargs)

    def propose(self, *args, **kwargs):
        self._admitted_initial()
        return super().propose(*args, **kwargs)
