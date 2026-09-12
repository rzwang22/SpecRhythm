"""CPU time-window state and pre-forward full-shape stop signal, scan only."""

from __future__ import annotations

import time

from specrhythm.serving.common import require


class ScanShapeStop(RuntimeError):
    """Stock selection is retained, but no partial model forward may be issued."""

    def __init__(self, expected, row):
        super().__init__("decode scan stopped before a partial Target forward")
        self.evidence = {
            "expected_batch": expected,
            "scheduled_batch": row["B"],
            "request_ids": row["request_ids"],
            "cohort": row["cohort"],
            "timestamp_ns": time.monotonic_ns(),
            "model_forward_issued": False,
        }


class ScanWindow:
    def __init__(self, options, batch, grouped):
        self.options, self.batch, self.grouped = options, batch, grouped
        self.expected = batch // 2 if grouped else batch
        self.start_ns = self.end_ns = self.warmup_end_ns = None
        self.warmup_start_ns = time.monotonic_ns()
        self.warmup_steps = self.samples = self.warmup_rotations = 0
        self.reason = None
        self.pending = None
        self.rotations = []
        self.partial_rotations = []
        self.window_state = None
        self.rejected_step = None

    def ready(self, now, *, population):
        if (
            self.start_ns is None
            and self.warmup_rotations >= self.options["warmup_steps"]
            and population["active_requests"] == self.batch
        ):
            self.start_ns = self.warmup_end_ns = now
            self.window_state = dict(population)
        return self.start_ns is not None

    def time_expired(self, now):
        if (
            self.start_ns is not None
            and now - self.start_ns >= self.options["window_seconds"] * 1e9
        ):
            self.reason = "time_budget"
            return True
        return False

    def step_completed(self, row, now):
        if not row["B"]:
            return self.time_expired(now)
        require(row["B"] == self.expected, "scan executed unexpected partial batch", actual=row)
        require(
            self.samples + self.warmup_steps < 10000,
            "scan step evidence capacity exceeded; incomplete, never silently truncated",
        )
        measured = self.start_ns is not None
        if measured:
            self.samples += 1
        else:
            self.warmup_steps += 1
        item = {
            "request_ids": list(row["request_ids"]),
            "cohort": row["cohort"],
            "start_ns": row["start_ns"],
            "end_ns": now,
            "step": self.samples if measured else self.warmup_steps,
        }
        rotation = None
        if self.grouped:
            require(row["cohort"] in ("A", "B"), "scan PingPong cohort evidence missing")
            if self.pending is not None and self.pending["cohort"] != item["cohort"]:
                require(
                    not (set(self.pending["request_ids"]) & set(item["request_ids"])),
                    "scan A/B rotation reused request identities",
                )
                rotation = [self.pending, item]
                self.pending = None
            else:
                if self.pending is not None and measured:
                    self.partial_rotations.append([self.pending])
                self.pending = item
        else:
            require(row["cohort"] is None, "ungrouped scan has a cohort")
            rotation = [item]
        if rotation:
            if measured:
                self.rotations.append(rotation)
            else:
                self.warmup_rotations += 1
        return self.time_expired(now)

    def evidence(self):
        return {
            **({"readiness": self.readiness.report()} if hasattr(self, "readiness") else {}),
            "warmup_rotations": self.warmup_rotations,
            "complete_rotations": self.rotations,
            "partial_rotations": self.partial_rotations
            + ([[self.pending]] if self.pending is not None and self.start_ns is not None else []),
            "window_initial_population": self.window_state,
            "rejected_step": self.rejected_step,
            "sample_limit": None,
            "boundary": "after full warmup rotations and refill; existing pipeline state retained",
        }
