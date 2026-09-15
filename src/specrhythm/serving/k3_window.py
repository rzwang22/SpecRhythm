"""Matched warmup work: sixteen request-verification opportunities per unit."""

from collections import Counter

from specrhythm.serving.common import require
from specrhythm.serving.decode_scan_window import ScanWindow, warmup_step
from specrhythm.serving.k3 import B16, geometry


class K3Window(ScanWindow):
    def __init__(self, options, batch, mode, configuration=B16):
        self.geometry = geometry(mode, configuration)
        require(type(batch) is int and batch == self.geometry["active_limit"],
                "K3 window active/configuration mismatch")
        super().__init__(options, batch, not self.geometry["serial_idle_gate"])
        self.opportunities = Counter()

    def step_completed(self, row, now):
        if not row["B"]:
            return self.time_expired(now)
        require(0 < row["B"] <= self.expected
                and len(set(row["request_ids"])) == row["B"], "K3 warmup/step batch invalid")
        require(self.samples + self.warmup_steps < 10000, "K3 step evidence capacity exceeded")
        if self.start_ns is None:
            self.warmup_steps += 1
            self.opportunities.update(row["request_ids"])
            self.warmup_history.append(warmup_step({**row, "end_ns": now}, self.warmup_steps))
            self.warmup_rotations = sum(self.opportunities.values()) // self.batch
        else:
            self.samples += 1
            # A rotation is a reporting unit only; no new scheduling or barrier.
            item = dict(request_ids=list(row["request_ids"]), cohort=row["cohort"],
                        start_ns=row["start_ns"], end_ns=now, step=self.samples)
            if not self.grouped:
                self.rotations.append([item])
            elif self.pending is None:
                self.pending = item
            else:
                self.rotations.append([self.pending, item])
                self.pending = None
        return self.time_expired(now)

    def warmup_boundary(self):
        return {**super().warmup_boundary(),
                "schema_version": "specrhythm.k3-request-opportunities.v1",
                "rotation_semantics": self.geometry["warmup_unit"],
                "request_opportunities": sum(self.opportunities.values()),
                "opportunities_by_request": dict(self.opportunities),
                "unique_requests": len(self.opportunities)}
