"""CPU time-window state and pre-forward full-shape stop signal, scan only."""

from __future__ import annotations

import time
from copy import deepcopy

from specrhythm.serving.common import digest, require


def pair_step(pending, item, grouped):
    """Latest opposite-cohort pair; a superseded half is historical, not pending."""
    if not grouped:
        require(item["cohort"] is None, "ungrouped scan has a cohort")
        return [item], None, None
    require(item["cohort"] in ("A", "B"), "scan PingPong cohort evidence missing")
    if pending is not None and pending["cohort"] != item["cohort"]:
        require(not (set(pending["request_ids"]) & set(item["request_ids"])),
                "scan A/B rotation reused request identities")
        return [pending, item], None, None
    return None, pending, item


def full_population(population, batch, grouped):
    ids = population.get("request_ids", [])
    if (population.get("active_requests") != batch
            or len(ids) != len(set(ids)) or len(ids) != batch):
        return False
    if grouped:
        cohorts = population.get("cohorts", {})
        a, b = cohorts.get("A", []), cohorts.get("B", [])
        return (len(a) == len(set(a)) == batch // 2 == len(b) == len(set(b))
                and not (set(a) & set(b)) and set(a) | set(b) == set(ids))
    return True


def warmup_step(row, number):
    """Bounded metadata only: no token prefixes or repeated resident/KV dumps."""
    return {"step": number, "cohort": row["cohort"], "B": row["B"],
            "start_ns": row["start_ns"], "end_ns": row["end_ns"],
            "committed_tokens": row["committed_tokens"],
            "request_ids_sha256": digest(row["request_ids"])}


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
        self.warmup_history = []
        self.warmup_unpaired = []

    def ready(self, now, *, population):
        if (
            self.start_ns is None
            and self.warmup_rotations >= self.options["warmup_steps"]
            and self.pending is None
            and (not self.warmup_history or self.warmup_history[-1]["end_ns"] <= now)
            and full_population(population, self.batch, self.grouped)
        ):
            self.start_ns = self.warmup_end_ns = now
            self.window_state = deepcopy(population)
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
        require(len(row["request_ids"]) == len(set(row["request_ids"])) == row["B"],
                "scan actual batch ceiling/identity invalid")
        require(
            self.samples + self.warmup_steps < 10000,
            "scan step evidence capacity exceeded; incomplete, never silently truncated",
        )
        measured = self.start_ns is not None
        if measured:
            self.samples += 1
        else:
            self.warmup_steps += 1
            self.warmup_history.append(warmup_step({**row, "end_ns": now}, self.warmup_steps))
        item = {
            "request_ids": list(row["request_ids"]),
            "cohort": row["cohort"],
            "start_ns": row["start_ns"],
            "end_ns": now,
            "step": self.samples if measured else self.warmup_steps,
        }
        rotation, historical, self.pending = pair_step(self.pending, item, self.grouped)
        if historical is not None:
            if measured:
                self.partial_rotations.append([historical])
            else:
                self.warmup_unpaired.append(historical["step"])
        if rotation:
            if measured:
                self.rotations.append(rotation)
            else:
                self.warmup_rotations += 1
        return self.time_expired(now)

    def warmup_boundary(self):
        # Once open, measurement's pending half must never rewrite the start boundary.
        pending = self.pending if self.start_ns is None else None
        return {
            "schema_version": "specrhythm.decode-scan-warmup-boundary.v1",
            "status": "OPEN" if self.start_ns is not None else "NOT_OPEN",
            "measurement_start_ns": self.start_ns,
            "required_complete_rotations": self.options["warmup_steps"],
            "completed_rotations": self.warmup_rotations,
            "completed_steps": self.warmup_steps,
            "historical_unpaired_steps": list(self.warmup_unpaired),
            "pending_step": pending["step"] if pending else None,
            "steps": list(self.warmup_history),
            "committed_tokens_excluded": sum(s["committed_tokens"] for s in self.warmup_history),
            "initial_population": self.window_state,
            "max_step_records": 10000,
        }

    def evidence(self):
        return {
            **({"readiness": self.readiness.report()} if hasattr(self, "readiness") else {}),
            "warmup_rotations": self.warmup_rotations,
            "warmup_boundary": self.warmup_boundary(),
            "complete_rotations": self.rotations,
            "partial_rotations": self.partial_rotations
            + ([[self.pending]] if self.pending is not None and self.start_ns is not None else []),
            "window_initial_population": self.window_state,
            "rejected_step": self.rejected_step,
            "sample_limit": None,
            "boundary": "after full warmup rotations and refill; existing pipeline state retained",
        }
