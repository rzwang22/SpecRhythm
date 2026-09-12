"""Diagnostic-only batch ceiling and stage evidence over the existing S2 schedulers."""

from __future__ import annotations

import time

from specrhythm.serving.common import require
from specrhythm.serving.fixed_identity import COUNTERS, install, scheduler_report
from specrhythm.serving.fixed_observe import TIMERS
from specrhythm.serving.s2_scheduler import S2PingScheduler, S2SerialScheduler, S2TargetScheduler


class FixedBatch:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        install(self, "_resident_identity" if hasattr(self, "_resident_identity")
                else "_dual_identity")

    def schedule(self, *args, **kwargs):
        # Control is refreshed by PoolScheduler. The preceding snapshot is used
        # only to tag evidence after super has read the current atomic packet.
        before = {
            i: (int(r.num_computed_tokens), len(r.all_token_ids)) for i, r in self.requests.items()
        }
        start = time.monotonic_ns()
        identity_before = scheduler_report(self)
        with TIMERS.span("scheduler"):
            output = super().schedule(*args, **kwargs)
        if self.s2_control["barrier_ns"] is not None:
            ids = list(output.num_scheduled_tokens)
            maximum = self.s2_control["max_requests_per_target_forward"]
            require(
                len(ids) <= maximum,
                "fixed Target batch ceiling exceeded",
                expected=maximum,
                actual=len(ids),
            )
            candidates = output.scheduled_spec_decode_tokens
            rows = [
                {
                    "request_id": self._identity().stable_id(str(i)),
                    "internal_request_id": str(i),
                    "query_positions": int(output.num_scheduled_tokens[i]),
                    "candidate_positions": len(candidates.get(i, ())),
                    "base_root_positions": int(output.num_scheduled_tokens[i])
                    - len(candidates.get(i, ())),
                    "context_length": before[i][1],
                    "position_start": before[i][0],
                    "position_end_exclusive": before[i][0] + output.num_scheduled_tokens[i],
                }
                for i in ids
            ]
            require(
                all(r["base_root_positions"] == 1 for r in rows),
                "fixed decode requires exactly one uncached base/root per request",
                actual=rows,
            )
            self.s2_steps[-1].update(
                schedule_start_ns=start,
                schedule_end_ns=time.monotonic_ns(),
                rows=rows,
                B=len(ids),
                population=self.s2_control.get("population"),
                phase=self.s2_control.get("diagnostic_phase"),
            )
            identity_after = scheduler_report(self)
            self.s2_steps[-1]["identity_matching"] = {
                "mode": identity_after["mode"],
                **{k: identity_after[k] - identity_before[k] for k in COUNTERS
                   if k in identity_after},
            }
            expected = self.s2_control.get("decode_scan_full_batch")
            if expected is not None and ids and len(ids) != expected:
                # Scan-only stop after real stock selection but BEFORE model dispatch.
                # No rollback or fake commit: the bounded drain aborts/releases Target
                # scheduling state and explicitly settles every unused Draft proposal.
                from specrhythm.serving.decode_scan_window import ScanShapeStop

                raise ScanShapeStop(expected, self.s2_steps[-1])
        return output


class FixedTargetScheduler(FixedBatch, S2TargetScheduler):
    pass


class FixedSerialScheduler(FixedBatch, S2SerialScheduler):
    pass


class FixedPingScheduler(FixedBatch, S2PingScheduler):
    def _ready_poll_limit(self, available):
        if self.s2_control.get("decode_scan_full_batch") is not None:
            # poll_ready is a global FIFO, not cohort-filtered. Drain published
            # results independently of the verification batch quota. At most one
            # result per prepared request; retired results use the same validated
            # discard path. No wait for work in flight and no new proposal.
            return len(self._dual_expected_ids)
        return super()._ready_poll_limit(available)

    def _before_stock_schedule(self, available, collected):
        expected = self.s2_control.get("decode_scan_full_batch")
        if expected is None:
            return super()._before_stock_schedule(available, collected)
        from specrhythm.serving.decode_scan_readiness import ScanBatchWait

        states = self.s2_control["requests"]
        counts = {c: dict(active=0, held=0, drafting=0, ready=0, verifiable=0)
                  for c in ("A", "B")}
        for rid, row in states.items():
            c = row["cohort"]
            if c in counts:
                counts[c]["active"] += row["state"] == "ACTIVE"
                counts[c]["held"] += row["state"] in ("ACTIVE", "FINISHED") and not row.get(
                    "resources_released", False)
                counts[c]["drafting"] += rid in self._dual_drafting
        for snapshot, decision in self._dual_decisions.values():
            row = states[snapshot.stable_request_id]
            if row["state"] == "ACTIVE":
                c = row["cohort"]
                require(c in counts, "scan active request lacks a valid cohort",
                        request_id=snapshot.stable_request_id)
                counts[c]["ready"] += snapshot.state.value in (
                    "VERIFY_READY", "TARGET_TAIL_READY")
                counts[c]["verifiable"] += decision.admissible
        now = time.monotonic_ns()
        selected = counts[self.selected_cohort]
        deadline = self.s2_control.get("decode_scan_deadline_ns")
        reason = None
        if deadline is not None and now >= deadline:
            reason = "time_budget"
        elif selected["active"] < expected:
            reason = "release_or_refill_pending"
        elif selected["drafting"]:
            reason = "draft_inflight"
        elif selected["verifiable"] < expected:
            # poll_ready atomically returns all published results AND in-flight
            # IDs. With a full active cohort and no owner work, a missing/invalid
            # proposal is a state-contract error, not an ordinary timed wait.
            require(False, "scan full idle cohort lacks verifiable proposals",
                    selected_cohort=self.selected_cohort, actual=counts,
                    ready_poll_quota=available, ready_collected=collected,
                    reason="cohort_ready_state_inconsistent")
        require(all(r["active"] <= expected for r in counts.values()),
                "scan cohort active capacity exceeded", actual=counts)
        evidence = {"timestamp_ns": now, "selected_cohort": self.selected_cohort,
                    "cohorts": counts, "expected_batch": expected,
                    "verifiable_scope": "current S2 selected-cohort admissibility",
                    "ready_poll_quota": available, "ready_collected": collected,
                    "ready_poll_scope": "entire prepared request pool; FIFO",
                    "reason": reason, "stock_allocation_issued": False,
                    "model_forward_issued": False}
        self.scan_readiness = evidence
        if reason:
            # S2's cohort decision cache is cycle-scoped. A wait is a new poll
            # next time, despite not advancing the stock scheduler's current_step.
            self._dual_cycle_id += 1
            raise ScanBatchWait(evidence)
