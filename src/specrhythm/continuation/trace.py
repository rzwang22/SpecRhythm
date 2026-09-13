"""Opt-in bounded host causality, with no I/O, GPU synchronization or protocol state.

Host spans are never GPU evidence. The existing native device recorder supplies
calibrated intervals at the final fence. All observers use the same light mode.
"""

from __future__ import annotations

import functools
import itertools
import os
import threading
import time
from contextlib import contextmanager

PHASE_BUDGETS = dict(setup=8192, warmup=4096, measurement=65536, drain=8192)


class CausalTrace:
    def __init__(self, enabled=False, limit=20000, *, layout="legacy"):
        if type(limit) is not int or not 1 <= limit <= 20000:
            raise ValueError("causal trace limit must be 1..20000")
        if layout not in ("legacy", "phased"):
            raise ValueError("unknown causal trace layout")
        self.enabled, self.limit = enabled, limit
        self.layout, self.phase = layout, "setup"
        self.rows = []
        self.attempts = {}
        self.sequence = itertools.count()
        self.local = threading.local()
        self.buckets = {p: dict(rows=[], attempts={}, sequence=itertools.count(), lost={})
                        for p in PHASE_BUDGETS}
        self.phase_changes = []

    def follow_control(self, packet):
        """Observe an already-read/published control packet; never fetch control here.

        Monotone phase transitions are observation only. Each span retains its entry
        phase; dropped span endpoints conservatively expose cross-boundary loss.
        """
        phase = packet.get("diagnostic_phase") if isinstance(packet, dict) else None
        if not self.enabled or self.layout != "phased" or phase not in PHASE_BUDGETS:
            return
        order = tuple(PHASE_BUDGETS)
        if order.index(phase) > order.index(self.phase):
            self.phase = phase
            self.phase_changes.append(dict(phase=phase, observed_ns=time.monotonic_ns()))

    @contextmanager
    def span(self, category, **fields):
        if not self.enabled:
            yield
            return
        old = getattr(self.local, 'fields', {})
        self.local.fields = {**old, **fields}
        start = time.monotonic_ns()
        phase = self.phase
        try:
            yield
        finally:
            end = time.monotonic_ns()
            self.event(category, start_ns=start, end_ns=end, _phase=phase)
            self.local.fields = old

    def event(self, category, **fields):
        if not self.enabled:
            return
        tid = threading.get_ident()
        phase = fields.pop("_phase", self.phase)
        if self.layout == "phased":
            bucket = self.buckets[phase]
            bucket["attempts"][tid] = bucket["attempts"].get(tid, 0) + 1
            if next(bucket["sequence"]) >= PHASE_BUDGETS[phase]:
                now = time.monotonic_ns()
                a, b = fields.get("start_ns", now), fields.get("end_ns", now)
                # Per-thread bounds avoid shared read/modify/write races and keep
                # memory constant even after overflow. No observer lock or I/O.
                old = bucket["lost"].get(tid, (a, b))
                bucket["lost"][tid] = min(old[0], a), max(old[1], b)
                return
            destination = bucket["rows"]
        else:
            destination = self.rows
        self.attempts[tid] = self.attempts.get(tid, 0)+1
        # CPython next(count) assigns one bounded slot atomically without an
        # observer lock; per-thread attempt counters account for omitted rows.
        if self.layout == "legacy" and next(self.sequence) >= self.limit:
            return
        now = time.monotonic_ns()
        destination.append({
            'category': category, 'start_ns': now, 'end_ns': now,
            'pid': os.getpid(), 'thread_id': threading.get_ident(),
            'thread_name': threading.current_thread().name,
            **getattr(self.local, 'fields', {}), **fields,
            **({"diagnostic_phase": phase} if self.layout == "phased" else {}),
        })

    def observe(self, category):
        def decorate(function):
            @functools.wraps(function)
            def call(*args, **kwargs):
                if not self.enabled:
                    return function(*args, **kwargs)
                with self.span(category):
                    return function(*args, **kwargs)
            return call
        return decorate

    def report(self):
        if self.layout == "phased":
            phases, rows = {}, []
            for phase, budget in PHASE_BUDGETS.items():
                bucket = self.buckets[phase]
                kept = list(bucket["rows"])
                attempted = sum(dict(bucket["attempts"]).values())
                dropped = max(0, attempted-len(kept))
                lost = list(dict(bucket["lost"]).values())
                phases[phase] = dict(row_budget=budget, retained_rows=len(kept),
                                     attempted_rows=attempted, dropped_rows=dropped,
                                     dropped_bounds_ns=([min(a for a, _ in lost),
                                                         max(b for _, b in lost)]
                                                        if lost else None))
                rows.extend(kept)
            dropped = sum(p["dropped_rows"] for p in phases.values())
            return dict(schema_version="specrhythm.eager-causal-host.v2",
                        layout="phased", mode="light" if self.enabled else "off",
                        status=("TRUNCATED" if dropped else "COMPLETE")
                        if self.enabled else "DISABLED",
                        row_limit=sum(PHASE_BUDGETS.values()), dropped_rows=dropped,
                        phases=phases, phase_changes=list(self.phase_changes),
                        phase_source="already observed control packet; span entry phase",
                        clock="host monotonic_ns; separate PID/thread lanes",
                        semantics="inclusive host spans, no GPU timing; lost bounds conservative",
                        rows=rows)
        dropped = max(0, sum(dict(self.attempts).values())-len(self.rows))
        return {
            'schema_version': 'specrhythm.eager-causal-host.v1',
            'mode': 'light' if self.enabled else 'off',
            'status': 'TRUNCATED' if dropped else 'COMPLETE' if self.enabled else 'DISABLED',
            'row_limit': self.limit, 'dropped_rows': dropped,
            'clock': 'host monotonic_ns; same machine, separate PID/thread lanes',
            'semantics': 'inclusive nested host spans; no GPU timing or GIL inference',
            'rows': self.rows[:self.limit],
        }


TRACE = CausalTrace(os.environ.get('SR_EAGER_CAUSAL_TRACE', 'off') == 'light',
                    layout=os.environ.get('SR_EAGER_CAUSAL_LAYOUT', 'legacy'))


def references(rows):
    """Bounded identity metadata only; never copy token/prefix/history payloads."""
    return [{k: row[k] for k in ('request_id', 'round_id', 'proposal_id') if k in row}
            for row in rows[:128]]
