"""Observed production Draft; unchanged proposal, commit, EOS and release algorithms."""

from __future__ import annotations

import time

from specrhythm.serving.fixed_audit import FixedAuditMixin
from specrhythm.serving.fixed_observe import TIMERS, DeviceTimeline
from specrhythm.serving.fixed_settle import (
    DiagnosticSerialMachine as diagnostic_serial_machine,
)
from specrhythm.serving.fixed_settle import (
    DiagnosticSerialServer,
)
from specrhythm.serving.s2_draft import S2DraftBackend


def serve(config, directory, socket_path, mode, *, backend_class=None):
    """Only the fixed diagnostic service opts into the explicit stop protocol."""
    if mode == "serial-eager":
        from specrhythm.serving.eager_draft import serve as serve_eager

        return serve_eager(config, directory, socket_path, backend_class=backend_class)
    from specrhythm.phase4.dual_service import DualDraftUnixServer
    from specrhythm.phase4.transport import CheckpointJsonl
    from specrhythm.serving.fixed_artifacts import record_error
    from specrhythm.serving.fixed_settle import DiagnosticDualController, DiagnosticDualMachine
    from specrhythm.serving.s1_workload import write_once

    report = directory / "draft-backend-report.json"
    ready = directory / "draft-service-ready.json"
    events = CheckpointJsonl(directory / "draft-work-events.jsonl")

    def factory():
        backend = (backend_class or FixedDraftBackend)(config)
        write_once(directory / "draft-startup.json", backend.provenance)
        cls = DiagnosticDualMachine if mode == "pingpong" else diagnostic_serial_machine
        return cls(backend, candidate_budget=4, report_path=report)

    if mode == "pingpong":
        controller = DiagnosticDualController(factory, events)
        server = DualDraftUnixServer(
            socket_path,
            controller,
            ready_path=ready,
            transport_log=CheckpointJsonl(directory / "draft-transport.jsonl"),
        )
        failure = None
        try:
            server.serve()
        except BaseException as error:
            failure = error
            record_error(directory, error, "draft_server")
            raise
        finally:
            try:
                controller.shutdown(failed=server.running)
            except Exception as error:
                record_error(directory, error, "draft_owner_cleanup")
                if failure is None:
                    raise
    else:
        machine = factory()
        failure = None
        try:
            DiagnosticSerialServer(socket_path, machine, event_log=events).serve(ready)
        except BaseException as error:
            failure = error
            record_error(directory, error, "draft_server")
            raise
        finally:
            try:
                if not machine.backend.closed:
                    machine.backend._fail()
                    machine.backend.shutdown()
                    if not report.exists():
                        write_once(report, machine.backend.report())
            except Exception as error:
                record_error(directory, error, "draft_backend_cleanup")
                if failure is None:
                    raise
    # The final socket response precedes its transport log; finalize only after
    # server exit and owner join. The coordinator waits for this receipt in drain.
    from specrhythm.serving.fixed_logging import finish_current

    finish_current("draft")


class FixedDraftBackend(FixedAuditMixin, S2DraftBackend):
    def __init__(self, config, *, worker=None):
        super().__init__(config, worker=worker)
        self.fixed_proposals = []
        self.fixed_device_report = None
        worker = self.worker
        self._provenance["fixed_engine_limits"] = {
            "max_num_seqs": worker.vllm_config.scheduler_config.max_num_seqs,
            "max_num_batched_tokens": worker.vllm_config.scheduler_config.max_num_batched_tokens,
            "max_model_len": worker.vllm_config.model_config.max_model_len,
            "num_gpu_blocks": self.provenance["kv_cache_num_blocks"],
            "block_size": self.provenance["block_size"],
        }
        self.fixed_device = DeviceTimeline(
            worker.torch,
            worker.model,
            lambda: {
                "role": "draft",
                "purpose": worker.phase,
                "internal_request_ids": list(worker.active_rows),
                "B": len(worker.active_rows),
                "Q": worker.active_tokens,
            },
            identity={
                "role": "draft",
                "global_rank": 0,
                "gpu_uuid": self.provenance["gpu_uuid"],
                "physical_gpu_id": self.provenance["physical_gpu_id"],
                "dtype": str(worker.vllm_config.model_config.dtype),
                "attention_backends": sorted(
                    {g.backend.__name__ for groups in worker.runner.attn_groups for g in groups}
                ),
            },
        )

    def propose_many(self, plans):
        start = time.monotonic_ns()
        result = super().propose_many(plans)
        self.fixed_proposals.append(
            {
                "start_ns": start,
                "end_ns": time.monotonic_ns(),
                "B": len(plans),
                "request_ids": [p.request_id for p in plans],
                "context_lengths": [len(p.prefix) for p in plans],
                "budgets": [p.budget for p in plans],
                "candidate_lengths": [len(result[p.request_id]) for p in plans],
                "round_ids": [p.round_id for p in plans],
                "first_token_from_cached_logits": True,
            }
        )
        return result

    def shutdown(self):
        if not self.closed:
            # Existing shutdown fences the worker before tearing it down. Read our
            # events at this safe boundary, not once per Draft round.
            self.worker.fence("fixed_diagnostic_window_end")
            self.fixed_device_report = self.fixed_device.report()
        super().shutdown()

    def report(self):
        from specrhythm.serving.fixed_logging import current

        logs = current()
        return {
            **super().report(),
            "fixed_proposals": self.fixed_proposals,
            "diagnostic_logging": logs.snapshot() if logs else None,
            "fixed_device": self.fixed_device_report,
            "fixed_host": TIMERS.report(),
        }
