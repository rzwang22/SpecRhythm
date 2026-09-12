"""Explicit fixed-diagnostic Draft audit layers; S1/S2/default algorithms stay intact."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager

from specrhythm.continuation.trace import TRACE
from specrhythm.serving.common import require
from specrhythm.serving.runtime_kv_audit import RuntimeKVGuard
from specrhythm.serving.s2_pool import control

MODES = ("full", "runtime")
VERSION = "specrhythm.draft-audit.v1"


class FixedAuditMixin:
    def __init__(self, *args, **kwargs):
        self.audit_mode = os.environ.get("SR_FIXED_DRAFT_AUDIT", "full")
        require(self.audit_mode in MODES, "unknown Draft audit mode")
        self.audit_ids = None
        self.audit_boundaries = []
        self.audit_setup_count, self.audit_setup_last = 0, None
        self.audit_full_checks = self.audit_full_visits = 0
        self.audit_guard = None
        super().__init__(*args, **kwargs)
        if self.audit_mode == "runtime":
            require(
                os.environ.get("SR_S2_MODE") in ("serial", "serial-eager"),
                "runtime Draft audit is limited to Serial/Serial-eager",
            )
            self.audit_guard = RuntimeKVGuard(self)
        self._provenance["draft_audit"] = self.audit_metadata()

    def audit_metadata(self):
        return dict(
            schema_version=VERSION,
            mode=self.audit_mode,
            scope="Draft private full-attention KV; Target retains full audit",
            control="fresh atomic control snapshot at every audit/write; no cache",
            full_boundaries="setup snapshots, first timing boundary, before final release, "
            "shutdown; terminal receipt full audit retained",
            observation=os.environ.get("SR_FIXED_OBSERVATION", "original-live"),
            causal_observation=os.environ.get("SR_EAGER_CAUSAL_TRACE", "off"),
        )

    def audit_control(self):
        return control()

    def audit_setup_allowed(self, rid, packet):
        return self.pool.initial is None and packet["barrier_ns"] is None

    @contextmanager
    def audit_scope(self, ids):
        old, self.audit_ids = self.audit_ids, tuple(ids)
        try:
            yield
        finally:
            self.audit_ids = old

    def audit_requests(self, ids):
        with self.audit_scope(ids):
            return self._audit()

    def physical_rows(self):
        start = time.monotonic_ns()
        value = super().physical_rows()
        self.audit_full_visits += len(value)
        TRACE.event(
            "draft_full_pool_scan",
            start_ns=start,
            end_ns=time.monotonic_ns(),
            prefix_visits=len(value),
            audit_mode=self.audit_mode,
        )
        return value

    @TRACE.observe("draft_pool_audit")
    def _audit(self):
        # Keep the original complete implementation and its freeze/check side effects.
        if self.audit_mode == "full" or self.pool.initial is None or self.audit_ids is None:
            was_frozen = self.pool.initial is not None
            packet = super()._audit()
            if packet["barrier_ns"] is not None:
                self.audit_full_checks += 1
                if self.audit_guard is not None:
                    self.audit_guard.check(tuple(self.states), packet)
                    self.audit_guard.reconcile(self.physical_rows())
                if not was_frozen:
                    self.audit_boundaries.append(
                        dict(
                            boundary="timing_entry",
                            timestamp_ns=time.monotonic_ns(),
                            requests=len(self.states),
                        )
                    )
            return packet
        packet = self.audit_control()  # Also preserves control commands in the return value.
        require(packet["barrier_ns"] is not None, "runtime timing barrier regressed")
        self.audit_guard.check(self.audit_ids, packet)
        return packet

    def initialize_many(self, rows):
        if self.audit_mode == "runtime":
            require(self.pool.initial is None, "runtime re-initialization after timing entry")
        result = super().initialize_many(rows)
        self.audit_setup_count += 1
        self.audit_setup_last = dict(
            boundary="setup_complete_snapshot",
            timestamp_ns=time.monotonic_ns(),
            requests=len(self.states),
        )
        return result

    def propose_many(self, plans):
        with self.audit_scope([p.request_id for p in plans]):
            return super().propose_many(plans)

    def commit_many(self, plans):
        with self.audit_scope([p.request_id for p in plans]):
            return super().commit_many(plans)

    def finish_many(self, ids):
        if self.audit_guard is not None and ids:
            require(not self.audit_guard.pending, "runtime release before completed write fence")
            # Full terminal receipt checks below are intentionally retained. They
            # feed diagnostic drain consumers and are not just timing records.
            self.audit_guard.check(tuple(ids), self.audit_control())
        if ids and len(ids) == len(self.states):
            old, self.audit_ids = self.audit_ids, None
            try:
                self._audit()
                self.audit_boundaries.append(
                    dict(
                        boundary="before_final_release",
                        timestamp_ns=time.monotonic_ns(),
                        requests=len(self.states),
                    )
                )
            finally:
                self.audit_ids = old
        return super().finish_many(ids)

    def shutdown(self):
        if not self.closed:
            # S2.shutdown performs the complete audit before physical teardown.
            # A failed worker may need its existing cleanup fence before release.
            if self.audit_guard is not None and self.audit_guard.pending:
                self.worker.fence("runtime_failed_cleanup")
            old, self.audit_ids = self.audit_ids, None
            try:
                return super().shutdown()
            finally:
                self.audit_ids = old
                self.audit_boundaries.append(
                    dict(
                        boundary="shutdown",
                        timestamp_ns=time.monotonic_ns(),
                        requests=len(self.states),
                        closed=self.closed,
                        failed=self.failed,
                    )
                )
        return super().shutdown()

    def report(self):
        return {
            **super().report(),
            "draft_audit": {
                **self.audit_metadata(),
                "full_checks": self.audit_full_checks,
                "full_prefix_visits": self.audit_full_visits,
                "setup_full_snapshots": self.audit_setup_count,
                "full_boundaries": ([self.audit_setup_last] if self.audit_setup_last else [])
                + list(self.audit_boundaries),
                "runtime": self.audit_guard.report() if self.audit_guard is not None else None,
            },
        }
