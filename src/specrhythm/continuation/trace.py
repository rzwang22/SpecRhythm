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


class CausalTrace:
    def __init__(self, enabled=False, limit=20000):
        if type(limit) is not int or not 1 <= limit <= 20000:
            raise ValueError("causal trace limit must be 1..20000")
        self.enabled, self.limit = enabled, limit
        self.rows = []
        self.attempts = {}
        self.sequence = itertools.count()
        self.local = threading.local()

    @contextmanager
    def span(self, category, **fields):
        if not self.enabled:
            yield
            return
        old = getattr(self.local, 'fields', {})
        self.local.fields = {**old, **fields}
        start = time.monotonic_ns()
        try:
            yield
        finally:
            end = time.monotonic_ns()
            self.event(category, start_ns=start, end_ns=end)
            self.local.fields = old

    def event(self, category, **fields):
        if not self.enabled:
            return
        tid = threading.get_ident()
        self.attempts[tid] = self.attempts.get(tid, 0)+1
        # CPython next(count) assigns one bounded slot atomically without an
        # observer lock; per-thread attempt counters account for omitted rows.
        if next(self.sequence) >= self.limit:
            return
        now = time.monotonic_ns()
        self.rows.append({
            'category': category, 'start_ns': now, 'end_ns': now,
            'pid': os.getpid(), 'thread_id': threading.get_ident(),
            'thread_name': threading.current_thread().name,
            **getattr(self.local, 'fields', {}), **fields,
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


TRACE = CausalTrace(os.environ.get('SR_EAGER_CAUSAL_TRACE', 'off') == 'light')


def references(rows):
    """Bounded identity metadata only; never copy token/prefix/history payloads."""
    return [{k: row[k] for k in ('request_id', 'round_id', 'proposal_id') if k in row}
            for row in rows[:128]]
