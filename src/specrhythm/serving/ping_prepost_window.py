"""Two actual admission opportunities per warmup rotation; repeated request IDs are legal."""

from specrhythm.serving.common import require
from specrhythm.serving.decode_scan_window import ScanWindow, warmup_step


class PingPrePostWindow(ScanWindow):
    def step_completed(self, row, now):
        if not row["B"]:
            return self.time_expired(now)
        require(
            0 < row["B"] <= self.expected and len(set(row["request_ids"])) == row["B"],
            "PingPong actual Target ceiling/identity invalid",
        )
        require(
            self.samples + self.warmup_steps < 10000, "PingPong step evidence capacity exceeded"
        )
        measured = self.start_ns is not None
        if measured:
            self.samples += 1
        else:
            self.warmup_steps += 1
            self.warmup_history.append(warmup_step({**row, "end_ns": now}, self.warmup_steps))
        item = dict(
            request_ids=list(row["request_ids"]),
            cohort=row["cohort"],
            start_ns=row["start_ns"],
            end_ns=now,
            step=self.samples if measured else self.warmup_steps,
        )
        if self.pending is None:
            self.pending = item
        else:
            if measured:
                self.rotations.append([self.pending, item])
            else:
                self.warmup_rotations += 1
            self.pending = None
        return self.time_expired(now)

    def warmup_boundary(self):
        return {
            **super().warmup_boundary(),
            "schema_version": "specrhythm.ping-prepost-warmup.v1",
            "rotation_semantics": "two actual Target opportunities; homes/IDs may repeat",
        }
