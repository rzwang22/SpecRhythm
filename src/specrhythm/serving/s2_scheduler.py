"""S2-only gates over pinned stock allocation and qualified resident readiness."""

from __future__ import annotations

import time
from dataclasses import replace

from specrhythm.phase4.admissibility import ExecutionPhase, ScheduledOperation
from specrhythm.phase4.resident_scheduler import ResidentSetupScheduler
from specrhythm.phase4.serial import Proposal
from specrhythm.phase4.vllm_dual_scheduler import DualBatchScheduler
from specrhythm.serving.common import require
from specrhythm.serving.s2_pool import ResidentPoolAudit, control, prefix_record


class PoolScheduler:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.s2_pool = ResidentPoolAudit("target")
        self.s2_steps = []
        self.s2_control = control()

    def _identity(self):
        return getattr(self, "_resident_identity", None) or self._dual_identity

    def physical_rows(self):
        identity = self._identity()
        return {
            identity.stable_id(str(i)): prefix_record(
                r.prompt_token_ids, r.num_computed_tokens, self.kv_cache_manager.get_block_ids(i)
            )
            for i, r in self.requests.items()
            if not r.is_finished() and r.num_output_tokens
        }

    def freeze_pool(self):
        self.s2_control = control()
        self.s2_pool.check(self.physical_rows(), self.s2_control["requests"], freeze=True)
        return self.s2_pool.report()

    def schedule(self, *args, **kwargs):
        self.s2_control = control()
        timed = self.s2_control["barrier_ns"] is not None
        if timed:
            self.s2_pool.check(self.physical_rows(), self.s2_control["requests"])
        before = {i: int(r.num_computed_tokens) for i, r in self.requests.items()}
        output = super().schedule(*args, **kwargs)
        if timed:
            require(not output.preempted_req_ids, "S2 Target preemption would destroy resident KV")
            ids = [self._identity().stable_id(str(i)) for i in output.num_scheduled_tokens]
            states = self.s2_control["requests"]
            require(
                len(ids) <= self.s2_control["active_limit"]
                and all(states[r]["state"] == "ACTIVE" for r in ids),
                "S2 scheduled request before arrival/admission or beyond active limit",
            )
            for i in output.num_scheduled_tokens:
                require(
                    before[i] >= self.requests[i].num_prompt_tokens,
                    "S2 timed Target attempted prefix prefill",
                    request_id=str(i),
                )
            self.s2_pool.check(self.physical_rows(), states)
            self.s2_steps.append(
                {
                    "timestamp_ns": time.monotonic_ns(),
                    "request_ids": ids,
                    "query_tokens": sum(output.num_scheduled_tokens.values()),
                    "verify_requests": len(output.scheduled_spec_decode_tokens),
                    "cohort": getattr(self, "selected_cohort", None),
                }
            )
            if ids and hasattr(self, "selected_cohort"):
                self.selected_cohort = "B" if self.selected_cohort == "A" else "A"
        return output

    def _preempt_request(self, request, timestamp):
        require(
            self.s2_control["barrier_ns"] is None,
            "S2 Target preemption forbidden after resident barrier",
        )
        return super()._preempt_request(request, timestamp)

    def _request_admissible_for_schedule(self, request):
        if int(request.num_output_tokens):
            rid = self._identity().stable_id(str(request.request_id))
            if (
                self.s2_control["barrier_ns"] is None
                or self.s2_control["requests"][rid]["state"] != "ACTIVE"
            ):
                return False
        return super()._request_admissible_for_schedule(request)


class S2TargetScheduler(PoolScheduler, ResidentSetupScheduler):
    pass


class S2SerialScheduler(PoolScheduler, ResidentSetupScheduler):
    def _refresh_deferred_initial_proposals(self):
        # S2 owns initial proposal publication on dynamic admission, not global setup.
        return

    def _initial_proposal_available(self, stable_id, request):
        if request.num_output_tokens != 1:
            return False
        record = self.s2_control.get("initial_proposals", {}).get(stable_id)
        if record is None:
            return False
        proposal = Proposal.from_dict(record["proposal"])
        from specrhythm.phase4.serial import token_prefix_hash

        require(
            proposal.parent_prefix_len == len(request.all_token_ids)
            and proposal.parent_prefix_hash == token_prefix_hash(request.all_token_ids),
            "S2 initial Serial proposal differs from resident prefix",
            request_id=stable_id,
        )
        request.spec_token_ids = list(proposal.proposal_token_ids)
        return True

    def _initial_proposal_was_installed(self, stable_id):
        return stable_id in self.s2_control.get("initial_proposals", {})


class S2PingScheduler(PoolScheduler, DualBatchScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selected_cohort = "A"
        self._chosen_cycle = -1

    def _decision_for(self, request):
        snapshot, decision = super()._decision_for(request)
        if snapshot.execution_phase is ExecutionPhase.SETUP_PREFILL:
            return snapshot, decision
        states = self.s2_control["requests"]
        if self._chosen_cycle != self._dual_cycle_id:
            self._chosen_cycle = self._dual_cycle_id
            busy = {states[r]["cohort"] for r in self._dual_drafting}
            eligible = set()
            for candidate in self.requests.values():
                snapshot_candidate, decision_candidate = super()._decision_for(candidate)
                rid_candidate = snapshot_candidate.stable_request_id
                row_candidate = states[rid_candidate]
                if (
                    decision_candidate.admissible
                    and row_candidate["state"] == "ACTIVE"
                    and row_candidate["cohort"] not in busy
                ):
                    eligible.add(row_candidate["cohort"])
            if self.selected_cohort not in eligible and eligible:
                self.selected_cohort = sorted(eligible)[0]
        rid = snapshot.stable_request_id
        row = states[rid]
        allowed = (
            self.s2_control["barrier_ns"] is not None
            and row["state"] == "ACTIVE"
            and row["cohort"] == self.selected_cohort
            and not any(states[r]["cohort"] == row["cohort"] for r in self._dual_drafting)
        )
        if not allowed:
            decision = replace(
                decision,
                admissible=False,
                operation=ScheduledOperation.NONE,
                reason="S2 arrival/active/cohort dependency",
            )
        return snapshot, decision
