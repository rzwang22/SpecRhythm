"""Install owner-claimed ragged proposals into the actual resident Target scheduler."""

from specrhythm.phase4.serial import Proposal, token_prefix_hash
from specrhythm.serving.common import require
from specrhythm.serving.fixed_scheduler import FixedBatch
from specrhythm.serving.s2_pool import control
from specrhythm.serving.s2_scheduler import S2SerialScheduler


class PingPrePostScheduler(FixedBatch, S2SerialScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pp_consumed = set()
        self.selected_cohort = "A"

    def schedule(self, *args, **kwargs):
        packet = control()
        admission = packet.get("pp_admission", {})
        claims = admission.get("claims", [])
        self.pp_claims = {r["request_id"]: r for r in claims}
        require(len(self.pp_claims) == len(claims) <= 8, "invalid Target claim batch")
        if packet["barrier_ns"] is not None:
            self.selected_cohort = admission["opportunity_cohort"]
            require(
                all(r["claim_id"] not in self.pp_consumed for r in claims),
                "duplicate Target claim/consume",
            )
        output = super().schedule(*args, **kwargs)
        if packet["barrier_ns"] is not None:
            ids = [self._identity().stable_id(str(i)) for i in output.num_scheduled_tokens]
            require(
                set(ids) == set(self.pp_claims), "stock Target did not consume exact claimed batch"
            )
            self.pp_consumed.update(r["claim_id"] for r in claims)
            self.s2_steps[-1].update(
                ping_admission=admission,
                home_cohorts={rid: self.pp_claims[rid]["home_cohort"] for rid in ids},
            )
        return output

    def _initial_proposal_available(self, stable_id, request):
        claim = getattr(self, "pp_claims", {}).get(stable_id)
        if claim is None:
            return False
        proposal = Proposal.from_dict(claim["proposal"])
        require(
            proposal.parent_prefix_len == len(request.all_token_ids)
            and proposal.parent_prefix_hash == token_prefix_hash(request.all_token_ids),
            "stale claimed Target prefix",
        )
        request.spec_token_ids = list(proposal.proposal_token_ids)
        return True

    def _initial_proposal_was_installed(self, stable_id):
        return stable_id in getattr(self, "pp_claims", {})

    def _request_admissible_for_schedule(self, request):
        if int(request.num_output_tokens) and self.s2_control["barrier_ns"] is not None:
            rid = self._identity().stable_id(str(request.request_id))
            if rid not in self.pp_claims:
                return False
        return super()._request_admissible_for_schedule(request)
