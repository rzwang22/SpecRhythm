"""Install owner-claimed ragged proposals into the actual resident Target scheduler."""

import os
from contextlib import contextmanager

from specrhythm.continuation.trace import TRACE
from specrhythm.phase4.request_identity import _NormalizedTokenRow
from specrhythm.phase4.serial import Proposal, token_prefix_hash
from specrhythm.serving.common import require
from specrhythm.serving.fixed_scheduler import FixedBatch
from specrhythm.serving.k3 import RESIDENT_POLICY
from specrhythm.serving.s2_pool import control
from specrhythm.serving.s2_scheduler import S2SerialScheduler
from specrhythm.serving.shared_control import control_transaction


class PingPrePostScheduler(FixedBatch, S2SerialScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pp_consumed = set()
        self.selected_cohort = "A"

    def _binding_input(self):
        if os.environ.get("SR_S2_MODE", "").endswith("-k3"):
            return _NormalizedTokenRow
        return super()._binding_input()

    @contextmanager
    def _resident_span(self, name, work=None):
        if not os.environ.get("SR_S2_MODE", "").endswith("-k3"):
            yield
            return
        # One bounded span per phase, never per token/request. The mutable work
        # dictionary is call-local and only populated before the span is recorded.
        work = {} if work is None else work
        with TRACE.span("target_resident_" + name, work=work,
                        resident_cycle=self._resident_cycle_id,
                        request_table_size=len(self.requests),
                        resident_schedule_policy=RESIDENT_POLICY):
            yield
            if name in ("decisions", "admission_records"):
                work["decision_rows"] = len(self._resident_decisions)
                if name == "admission_records":
                    work["admission_records"] = len(self._resident_decisions)
                    work["checkpoint_payload_encodings"] = len(self._resident_decisions)
                    work["shared_field_snapshots"] = 1

    def _resident_record_factory(self, shared):
        if not os.environ.get("SR_S2_MODE", "").endswith("-k3"):
            return super()._resident_record_factory(shared)
        from functools import partial
        from types import MappingProxyType

        from specrhythm.phase4.admission_record import PreparedAdmission

        # Owned scalar snapshot for this call only. No request/claim/KV cache.
        return partial(PreparedAdmission, MappingProxyType(shared))

    def physical_rows(self):
        if os.environ.get("SR_S2_MODE", "").endswith("-k3"):
            from specrhythm.serving.k3_prompt_proof import physical_rows

            return physical_rows(self)
        return super().physical_rows()

    @control_transaction
    def schedule(self, *args, **kwargs):
        packet = control()
        admission = packet.get("pp_admission", {})
        claims = admission.get("claims", [])
        self.pp_claims = {r["request_id"]: r for r in claims}
        from specrhythm.serving.k3 import MODES, configuration_of, geometry

        mode = os.environ.get("SR_S2_MODE")
        ceiling = (geometry(mode, configuration_of(packet))["target_request_ceiling"]
                   if mode in MODES else 8)
        require(len(self.pp_claims) == len(claims) <= ceiling, "invalid Target claim batch")
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
        if os.environ.get("SR_S2_MODE", "").endswith("-k3"):
            budget = min(3, request.sampling_params.max_tokens - request.num_output_tokens)
            require(len(proposal.proposal_token_ids) == budget or (
                0 < len(proposal.proposal_token_ids) < budget and proposal.proposal_eos),
                "Target refuses incomplete K3 claim")
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
