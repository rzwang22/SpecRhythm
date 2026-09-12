"""One monotonic arrival clock, FIFO admission and safe dynamic cohort membership."""

from __future__ import annotations

import threading
import time
from collections import deque

from specrhythm.serving.common import require
from specrhythm.serving.s2_plan import check_seal


class ServingClock:
    def __init__(
        self, definitions, trace, bootstraps, *, active_limit=128, mode="target", emit=None,
        per_cohort_capacity=None, fixed_assignment=None
    ):
        check_seal(trace)
        require(mode in ("target", "serial", "pingpong", "serial-eager"), "unknown S2 mode")
        require(type(active_limit) is int and 0 < active_limit <= 128, "invalid active limit")
        self.definitions = {r.request_id: r for r in definitions}
        ids = [r["request_id"] for r in trace["rows"]]
        require(
            len(ids) == len(set(ids)) and set(ids) == set(self.definitions) == set(bootstraps),
            "S2 trace/bootstrap/request identity mismatch",
        )
        self.trace, self.bootstraps = trace, bootstraps
        self.active_limit, self.mode = active_limit, mode
        require(
            per_cohort_capacity is None or (
                mode == "pingpong" and type(per_cohort_capacity) is int
                and per_cohort_capacity > 0 and 2 * per_cohort_capacity >= active_limit
            ), "invalid explicit cohort capacity",
        )
        self.per_cohort_capacity = per_cohort_capacity
        self.fixed_assignment = dict(fixed_assignment or {})
        require(
            set(self.fixed_assignment) <= set(ids)
            and all(c in ("A", "B") for c in self.fixed_assignment.values()),
            "invalid frozen cohort assignment",
        )
        self.emit = emit or (lambda row: None)
        self.lock = threading.RLock()
        self.events, self.queue = [], deque()
        self.rows = {
            rid: {
                "request_id": rid,
                "state": "STAGED",
                "cohort": None,
                "resources_released": False,
                "generated_token_ids": [bootstraps[rid]["token"]],
                "commits": [],
            }
            for rid in ids
        }
        self.offset = 0
        self.barrier_ns = None
        self.stop_event = threading.Event()
        self.thread = None
        self.observer_error = None

    def _event(self, event, timestamp_ns, **fields):
        row = {
            "schema_version": "specrhythm.s2-event.v1",
            "event": event,
            "timestamp_ns": timestamp_ns,
            **fields,
        }
        self.events.append(row)
        self.emit(row)

    def start(self, barrier_ns, *, threaded=True):
        require(
            self.barrier_ns is None and type(barrier_ns) is int and barrier_ns > 0,
            "S2 barrier must be published exactly once",
        )
        self.barrier_ns = barrier_ns
        for trace in self.trace["rows"]:
            self.rows[trace["request_id"]]["arrival_ns"] = barrier_ns + round(
                trace["arrival_offset_seconds"] * 1e9
            )
        self._event("barrier", barrier_ns, trace_sha256=self.trace["sha256"])
        if threaded:
            self.thread = threading.Thread(
                target=self._observe, name="s2-monotonic-arrivals", daemon=True
            )
            self.thread.start()

    def _observe(self):
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    self.observe(time.monotonic_ns())
                    if self.offset == len(self.trace["rows"]):
                        return
                    rid = self.trace["rows"][self.offset]["request_id"]
                    delay = (self.rows[rid]["arrival_ns"] - time.monotonic_ns()) / 1e9
                self.stop_event.wait(max(0.00001, min(delay, 0.1)))
        except Exception as error:
            self.observer_error = error

    def observe(self, now_ns):
        """Runs independently of GPU step completion; planned arrivals never shift."""
        with self.lock:
            require(self.barrier_ns is not None, "arrival observation before S2 barrier")
            while self.offset < len(self.trace["rows"]):
                rid = self.trace["rows"][self.offset]["request_id"]
                row = self.rows[rid]
                if row["arrival_ns"] > now_ns:
                    break
                self.offset += 1
                row.update(state="QUEUED", observed_arrival_ns=now_ns)
                self._event(
                    "arrived",
                    now_ns,
                    request_id=rid,
                    planned_arrival_ns=row["arrival_ns"],
                    arrival_handling_lag_ns=now_ns - row["arrival_ns"],
                )
                if self.bootstraps[rid]["terminal"]:
                    row.update(
                        state="FINISHED",
                        completion_ns=now_ns,
                        admission_ns=None,
                        resources_released=True,
                        finish_reason=self.bootstraps[rid]["finish_reason"],
                    )
                    self._event("finished-in-setup", now_ns, request_id=rid, timed_tokens=0)
                else:
                    self.queue.append(rid)

    def admit(self, now_ns, *, busy_cohorts=()):
        """Only the coordinator between completed Target steps calls this mutation."""
        require(
            self.observer_error is None,
            "S2 arrival observer failed",
            error=str(self.observer_error),
        )
        admitted = []
        with self.lock:
            held = [
                r
                for r in self.rows.values()
                if r.get("admission_ns") is not None and not r["resources_released"]
            ]
            while self.queue and len(held) < self.active_limit:
                cohort = None
                if self.mode == "pingpong":
                    choices = [
                        c for c in ("A", "B") if c not in busy_cohorts
                        and (self.per_cohort_capacity is None or
                             sum(r["cohort"] == c for r in held) < self.per_cohort_capacity)
                        and self.fixed_assignment.get(self.queue[0], c) == c
                    ]
                    if not choices:
                        break
                    cohort = min(choices, key=lambda c: (sum(r["cohort"] == c for r in held), c))
                rid = self.queue.popleft()
                row = self.rows[rid]
                require(
                    row["state"] == "QUEUED" and now_ns >= row["observed_arrival_ns"],
                    "admission preceded observed arrival",
                    request_id=rid,
                )
                row.update(state="ACTIVE", admission_ns=now_ns, cohort=cohort)
                held.append(row)
                admitted.append(rid)
                self._event(
                    "admitted",
                    now_ns,
                    request_id=rid,
                    cohort=cohort,
                    queue_ns=now_ns - row["arrival_ns"],
                    safe_boundary="between Target steps; no in-flight Draft in this cohort",
                )
            if admitted:
                self._event(
                    "population",
                    now_ns,
                    active=len(held),
                    queued=len(self.queue),
                    runnable=sum(r["state"] == "ACTIVE" for r in self.rows.values()),
                )
        return admitted

    def commit(self, rid, delta, now_ns, *, finished=False, finish_reason=None):
        with self.lock:
            row = self.rows[rid]
            definition = self.definitions[rid]
            require(
                row["state"] == "ACTIVE" and now_ns >= row["admission_ns"],
                "timed execution before arrival/admission or after terminal",
                request_id=rid,
            )
            require(
                delta and all(type(t) is int and t >= 0 for t in delta), "invalid commit tokens"
            )
            row["generated_token_ids"].extend(delta)
            require(
                len(row["generated_token_ids"]) <= definition.maximum_new_tokens,
                "output budget exceeded",
                request_id=rid,
            )
            row["commits"].append({"timestamp_ns": now_ns, "token_ids": list(delta)})
            self._event(
                "commit", now_ns, request_id=rid, token_ids=list(delta), batch_timestamp=True
            )
            if finished:
                row.update(state="FINISHED", completion_ns=now_ns, finish_reason=finish_reason)
                self._event(
                    "finished",
                    now_ns,
                    request_id=rid,
                    timed_tokens=len(row["generated_token_ids"]) - 1,
                )

    def released(self, ids, now_ns):
        with self.lock:
            for rid in ids:
                row = self.rows[rid]
                require(row["state"] == "FINISHED", "release before completion", request_id=rid)
                if not row["resources_released"]:
                    row["resources_released"] = True
                    self._event("resources-released", now_ns, request_id=rid)

    def control(self):
        with self.lock:
            return {
                "schema_version": "specrhythm.s2-control.v1",
                "barrier_ns": self.barrier_ns,
                "active_limit": self.active_limit,
                "requests": {
                    rid: {
                        k: row.get(k)
                        for k in (
                            "state",
                            "arrival_ns",
                            "admission_ns",
                            "cohort",
                            "resources_released",
                        )
                    }
                    for rid, row in self.rows.items()
                },
            }

    @property
    def complete(self):
        with self.lock:
            return all(
                r["state"] == "FINISHED" and r["resources_released"] for r in self.rows.values()
            )

    def close(self, *, failure=None):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)
        if failure is not None:
            with self.lock:
                for rid, row in self.rows.items():
                    if row["state"] != "FINISHED":
                        row["state"] = "FAILED"
                        self._event(
                            "failed", time.monotonic_ns(), request_id=rid, error=str(failure)
                        )
