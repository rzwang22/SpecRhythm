"""Explicit Serial A/B measurement of existing HF calls; no token-policy changes."""

from __future__ import annotations

import time

from specrhythm.phase4.batched_draft_service import write_immutable_report
from specrhythm.phase4.draft_metrics import DraftMetrics
from specrhythm.phase4.draft_service import (
    DraftStateMachine,
    DraftUnixServer,
    HFPersistentDraftBackend,
)
from specrhythm.phase4.manifest import sha256_file
from specrhythm.phase4.transport import CheckpointJsonl


class MeasuredHFDraftBackend(HFPersistentDraftBackend):
    """Model events are queued in memory and read once at final shutdown.

    No per-forward synchronization is added. Scalar greedy D2H calls are counted
    at the existing proposal boundary; framework-internal syncs are not claimed.
    """

    def __init__(self, config):
        started = time.monotonic_ns()
        super().__init__(config)
        self.metrics = DraftMetrics()
        self.phase = "setup"
        self.events = []
        self.closed = self.failed = False
        self.startup_load_ns = time.monotonic_ns() - started
        self._provenance["serial_ab_forward_event_observer"] = True

        def before(_model, _args, kwargs):
            self.event_start = self.torch.cuda.Event(enable_timing=True)
            self.event_start.record()
            self.query_count = int(kwargs["input_ids"].numel())

        def after(_model, _args, _output):
            end = self.torch.cuda.Event(enable_timing=True)
            end.record()
            self.events.append((self.phase, self.event_start, end))
            self.metrics.forward(self.phase, 1, self.query_count)

        self.hooks = [
            self.model.register_forward_pre_hook(before, with_kwargs=True),
            self.model.register_forward_hook(after),
        ]

    def initialize(self, request_id, committed_token_ids):
        self.phase = "setup"
        return super().initialize(request_id, committed_token_ids)

    def propose(self, request_id, budget, eos_token_ids):
        self.phase = "proposal"
        values, forwards = super().propose(request_id, budget, eos_token_ids)
        self.metrics.counters["proposals"] += 1
        self.metrics.counters["proposed_tokens"] += len(values)
        self.metrics.syncs["existing_greedy_scalar_item"] += len(values)
        return values, forwards

    def rollback(self, request_id, accepted_draft_tokens):
        self.phase = "commit"
        self.metrics.counters["commits"] += 1
        self.metrics.counters["invalidated_tokens"] += (
            len(self.states[request_id].proposal) - accepted_draft_tokens
        )
        return super().rollback(request_id, accepted_draft_tokens)

    def append_target_token(self, request_id, token_id):
        self.phase = "commit"
        return super().append_target_token(request_id, token_id)

    def shutdown(self):
        if self.closed:
            return
        try:
            self.torch.cuda.synchronize()
            self.metrics.syncs["final_event_collection"] += 1
            for purpose, start, end in self.events:
                self.metrics.gpu_ms[purpose] += start.elapsed_time(end)
            self.events.clear()
        finally:
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()
            super().shutdown()
            self.closed = True

    def report(self):
        return {
            **self.metrics.snapshot(self.backend_name),
            "serving_performance_backend": False,
            "provenance": self.provenance,
            "backend_shutdown_complete": self.closed,
            "execution_failed": self.failed,
            "draft_live_requests_final": len(self.states),
            "draft_scalar_item_count": self.metrics.syncs["existing_greedy_scalar_item"],
            "draft_sync_coverage": (
                "existing HF greedy scalar item calls plus final event collection; "
                "internal framework syncs unobserved"
            ),
            "startup_load_ns": self.startup_load_ns,
            "explicit_warmup_model_forward_count": 0,
            "measurement_observer": "nonblocking CUDA events; no additional measured-path fences",
        }


class MeasuredHFStateMachine(DraftStateMachine):
    def __init__(self, backend, candidate_budget, report_path):
        super().__init__(backend, candidate_budget=candidate_budget)
        self.report_path = report_path

    def shutdown(self):
        result = super().shutdown()
        write_immutable_report(self.report_path, self.backend.report())
        return {**result, "draft_backend_report_sha256": sha256_file(self.report_path)}


def serve_measured_hf(config, *, socket_path, event_log_path, ready_path):
    report_path = ready_path.with_name("draft-backend-report.json")
    if any(p.exists() for p in (report_path, ready_path, socket_path)):
        raise FileExistsError("HF A/B artifacts must be fresh")
    backend = MeasuredHFDraftBackend(config)
    machine = MeasuredHFStateMachine(backend, config.proposal_budget, report_path)
    try:
        DraftUnixServer(socket_path, machine, event_log=CheckpointJsonl(event_log_path)).serve(
            ready_path
        )
    finally:
        if not backend.closed:
            backend.failed = True
            try:
                backend.shutdown()
            finally:
                if not report_path.exists():
                    write_immutable_report(report_path, backend.report())
