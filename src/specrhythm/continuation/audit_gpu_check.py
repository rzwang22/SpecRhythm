"""Opt-in GPU correctness adapter for the exact production Draft audit layer.

Independent same-worker reference prefixes remain explicit diagnostic allocations;
only this adapter permits them after the initial freeze. Production never does.
"""

from __future__ import annotations

import os

from specrhythm.continuation.gpu_backend import GPUContinuationBackendMixin
from specrhythm.phase4.vllm_draft_backend import VllmBatchedDraftBackend
from specrhythm.serving.common import require
from specrhythm.serving.fixed_audit import FixedAuditMixin
from specrhythm.serving.s2_draft import S2DraftBackend
from specrhythm.serving.s2_pool import control, publish


def is_reference(rid):
    return rid.startswith(("gpu-check-reference:", "gpu-check-bonus-reference:"))


class AuditCheckBackend(GPUContinuationBackendMixin, FixedAuditMixin, S2DraftBackend):
    def audit_setup_allowed(self, rid, packet):
        return is_reference(rid) or super().audit_setup_allowed(rid, packet)

    def initialize_many(self, rows):
        if self.pool.initial is not None:
            require(
                all(is_reference(rid) for rid, _ in rows),
                "audit correctness may replay only independent reference IDs",
            )
            packet = control()
            for rid, _ in rows:
                require(rid not in packet["requests"], "reference ID reused")
                packet["requests"][rid] = {"state": "ACTIVE"}
            publish(os.environ["SR_S2_CONTROL"], packet)
            # The existing live-request path remains unchanged. Independent
            # reference prefill is explicitly outside the performance workload.
            VllmBatchedDraftBackend.initialize_many(self, rows)
            self.audit_requests([rid for rid, _ in rows])
            return
        return super().initialize_many(rows)


def prepare(config, directory, requests, mode, *, worker=None):
    require(mode in ("full", "runtime"), "unknown correctness audit mode")
    os.environ.update(
        SR_FIXED_DRAFT_AUDIT=mode,
        SR_S2_MODE="serial-eager",
        SR_S2_RUN_DIRECTORY=str(directory),
        SR_S2_CONTROL=str(directory / "audit-control.json"),
    )
    publish(
        directory / "audit-control.json",
        {"barrier_ns": None, "requests": {r["request_id"]: {"state": "STAGED"} for r in requests}},
    )
    return AuditCheckBackend(config, worker=worker)


def initialized(event):
    if event["event"] == "initialized":
        packet = control()
        packet["barrier_ns"] = 1
        for row in packet["requests"].values():
            row["state"] = "ACTIVE"
        publish(os.environ["SR_S2_CONTROL"], packet)
