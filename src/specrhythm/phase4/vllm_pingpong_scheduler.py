"""Isolated persistent-cohort gate; stock allocation and Dual validity are inherited."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from specrhythm.phase4.admissibility import ExecutionPhase, ScheduledOperation
from specrhythm.phase4.dual_rhythm import COHORTS, AnnotatedLog, load_assignment
from specrhythm.phase4.vllm_dual_scheduler import DualBatchScheduler


class PingPongScheduler(DualBatchScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.assignment = load_assignment(
            workload=Path(os.environ["SR_PHASE4_WORKLOAD"]),
            count=len(self._dual_expected_ids),
        )
        if self._dual_microbatch_size != len(self.assignment):
            raise ValueError("pingpong readiness capacity must cover the whole frozen burst")
        if not self._dual_resident or self._dual_test_coordination != "none":
            raise ValueError("pingpong requires resident execution without test coordination")
        self.selected_cohort = "A"
        self.initial_ready = set()
        self.fill_complete = False
        self._dual_events = AnnotatedLog(self._dual_events, self._cycle_evidence)

    def _active(self):
        return {
            self._dual_identity.stable_id(str(r.request_id))
            for r in self.requests.values()
            if not r.is_finished()
        }

    def _accept_ready_result(self, result):
        super()._accept_ready_result(result)
        self.initial_ready.add(result["request_id"])

    def _decision_for(self, request):
        snapshot, decision = super()._decision_for(request)
        if not decision.admissible or snapshot.execution_phase is ExecutionPhase.SETUP_PREFILL:
            return snapshot, decision
        active = self._active()
        if not self.fill_complete:
            self.fill_complete = bool(self._dual_setup_ready) and active <= self.initial_ready
        if not any(self.assignment[r] == self.selected_cohort for r in active):
            self.selected_cohort = "B" if self.selected_cohort == "A" else "A"
        # A capacity-clipped earlier unit can leave other ready members behind.
        # Do not overlap those members with still-running Draft work from their
        # own cohort. This is a stage dependency, not a wait to fill a batch.
        reason = (
            "pingpong initial pipeline fill"
            if not self.fill_complete
            else "pingpong opposite verification cohort"
            if self.assignment[snapshot.stable_request_id] != self.selected_cohort
            else "pingpong same-cohort Draft dependency"
            if any(self.assignment[r] == self.selected_cohort for r in self._dual_drafting)
            else None
        )
        if reason:
            decision = replace(
                decision, admissible=False, operation=ScheduledOperation.NONE, reason=reason
            )
        return snapshot, decision

    def _cycle_evidence(self, row):
        if "cycle_id" not in row:
            return {}
        active = self._active()
        sizes = {c: sum(self.assignment[r] == c for r in active) for c in COHORTS}
        timed = {
            s.stable_request_id
            for s, _ in self._dual_decisions.values()
            if s.execution_phase is ExecutionPhase.TIMED_DECODE
        }
        selected = [r for r in row["scheduled_request_ids"] if r in timed]
        eligible = [
            s.stable_request_id
            for s, d in self._dual_decisions.values()
            if d.admissible
            and s.stable_request_id in timed
            and self.fill_complete
            and self.assignment[s.stable_request_id] == self.selected_cohort
            and not any(self.assignment[r] == self.selected_cohort for r in self._dual_drafting)
        ]
        if any(self.assignment[r] != self.selected_cohort for r in selected):
            raise RuntimeError("Target scheduled outside the selected logical cohort")
        phase = (
            "fill" if not self.fill_complete else "drain" if not all(sizes.values()) else "steady"
        )
        result = {
            "dual_rhythm": "pingpong",
            "logical_cohort": self.selected_cohort,
            "pipeline_phase": phase,
            "active_cohort_sizes": sizes,
            "active_request_ids": sorted(active),
            "initial_ready_request_ids": sorted(self.initial_ready),
            "cohort_imbalance": abs(sizes["A"] - sizes["B"]),
            "eligible_cohort_request_ids": eligible,
            "attained_cohort_request_ids": selected,
            "capacity_clipped": len(selected) < len(eligible),
            "capacity_constraint_attribution": (
                "stock token/sequence/KV budgets; see dual_scheduler_constraints"
            ),
            "target_waiting_for_draft": bool(active) and not selected and not eligible,
            "cohort_switch_after_schedule": bool(selected),
        }
        if selected:
            self.selected_cohort = "B" if self.selected_cohort == "A" else "A"
        return result
