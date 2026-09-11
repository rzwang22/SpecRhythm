"""Diagnostic-only batch ceiling and stage evidence over the existing S2 schedulers."""

from __future__ import annotations

import time

from specrhythm.serving.common import require
from specrhythm.serving.fixed_observe import TIMERS
from specrhythm.serving.s2_scheduler import S2PingScheduler, S2SerialScheduler, S2TargetScheduler


class FixedBatch:
    def schedule(self, *args, **kwargs):
        # Control is refreshed by PoolScheduler. The preceding snapshot is used
        # only to tag evidence after super has read the current atomic packet.
        before = {
            i: (int(r.num_computed_tokens), len(r.all_token_ids)) for i, r in self.requests.items()
        }
        start = time.monotonic_ns()
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
        return output


class FixedTargetScheduler(FixedBatch, S2TargetScheduler):
    pass


class FixedSerialScheduler(FixedBatch, S2SerialScheduler):
    pass


class FixedPingScheduler(FixedBatch, S2PingScheduler):
    pass
