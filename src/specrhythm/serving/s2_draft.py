"""Real production Draft KV with S2-only residency/admission guards."""

from __future__ import annotations

import os
import time
from pathlib import Path

from specrhythm.phase4.batched_draft_service import BatchedDraftStateMachine
from specrhythm.phase4.draft_service import DraftUnixServer
from specrhythm.phase4.dual_batched_draft import (
    BatchedDualDraftController,
    BatchedDualDraftMachine,
)
from specrhythm.phase4.dual_rhythm import cohort_for
from specrhythm.phase4.dual_service import DualDraftUnixServer
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.transport import CheckpointJsonl
from specrhythm.phase4.vllm_draft_backend import VllmBatchedDraftBackend
from specrhythm.serving.common import digest, require
from specrhythm.serving.s1_workload import write_once
from specrhythm.serving.s2_pool import ResidentPoolAudit, control, prefix_record, publish


def cuda_memory(torch):
    free, total = torch.cuda.mem_get_info()
    return {
        "free_memory_bytes": int(free),
        "total_memory_bytes": int(total),
        "allocated_memory_bytes": int(torch.cuda.memory_allocated()),
        "reserved_memory_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_memory_bytes": int(torch.cuda.max_memory_reserved()),
    }


class S2DraftBackend(VllmBatchedDraftBackend):
    def __init__(self, config, *, worker=None):
        super().__init__(config, worker=worker)
        self.pool = ResidentPoolAudit("draft")
        self.pool_path = Path(os.environ["SR_S2_RUN_DIRECTORY"]) / "draft-pool.json"
        self.history = []
        self.terminal_release = None
        self.memory_initial = cuda_memory(self.worker.torch)
        self._provenance["s2_capacity"] = {
            **self.memory_initial,
            "role": "draft",
            "mode": os.environ["SR_S2_MODE"],
            "physical_gpu_id": self.provenance["physical_gpu_id"],
            "gpu_uuid": self.provenance["gpu_uuid"],
            "block_size": self.provenance["block_size"],
            "num_gpu_blocks": self.provenance["kv_cache_num_blocks"],
            "vocab_size": self.worker.vllm_config.model_config.get_vocab_size(),
        }

    def physical_rows(self):
        return {
            rid: prefix_record(
                state.prefix, state.materialized, self.worker.kv.get_block_ids(state.internal_id)
            )
            for rid, state in self.states.items()
        }

    def _audit(self):
        packet = control()
        states = {
            rid: ({**r, "state": "FINISHED"} if rid in self.retired else r)
            for rid, r in packet["requests"].items()
        }
        if packet["barrier_ns"] is not None:
            if self.pool.initial is None:
                # This is the identical snapshot written after the final setup initialization.
                require(self.pool_path.is_file(), "S2 Draft resident snapshot missing")
                from specrhythm.serving.common import read_json

                self.pool.check(read_json(self.pool_path)["rows"], states, freeze=True)
            self.pool.check(self.physical_rows(), states)
        return packet

    def initialize_many(self, rows):
        require(
            control()["barrier_ns"] is None, "S2 Draft prefill/re-initialization inside timing"
        )
        super().initialize_many(rows)
        physical = self.physical_rows()
        self.pool.check(physical, control()["requests"])
        publish(
            self.pool_path,
            {
                "rows": physical,
                "timestamp_ns": time.monotonic_ns(),
                "memory": cuda_memory(self.worker.torch),
            },
        )

    def propose_many(self, plans):
        packet = self._audit()
        require(
            packet["barrier_ns"] is not None
            and all(packet["requests"][p.request_id]["state"] == "ACTIVE" for p in plans),
            "S2 Draft proposal before arrival/admission",
        )
        result = super().propose_many(plans)
        self._audit()
        return result

    def commit_many(self, plans):
        packet = self._audit()
        require(
            all(
                packet["requests"][p.request_id]["state"] in ("ACTIVE", "FINISHED") for p in plans
            ),
            "S2 Draft commit before admission",
        )
        result = super().commit_many(plans)
        self._audit()
        return result

    def finish_many(self, ids):
        receipt = self.terminal_release
        if ids and receipt is not None:
            # The real commit materialization/fence has completed. Check the whole
            # allocator, including newly allocated tail blocks, before freeing any.
            physical = self.physical_rows()
            ResidentPoolAudit("draft-terminal-release").check(physical, {})
            require(set(ids) == set(receipt["request_ids"]), "S2 tail release scope changed")
            receipt.update(
                materialized={rid: physical[rid] for rid in ids},
                unrelated_before_release_sha256=digest(
                    {rid: row for rid, row in physical.items() if rid not in ids}
                ),
                private_kv_checked_requests=len(physical),
            )
        super().finish_many(ids)
        if ids and receipt is not None:
            receipt["resources_released_ns"] = time.monotonic_ns()
        if control()["barrier_ns"] is None:
            publish(
                self.pool_path,
                {
                    "rows": self.physical_rows(),
                    "timestamp_ns": time.monotonic_ns(),
                    "memory": cuda_memory(self.worker.torch),
                },
            )

    def report(self):
        return {
            **super().report(),
            "s2_resident_pool": self.pool.report(),
            "s2_memory_initial": self.memory_initial,
            "s2_live_requests_before_shutdown": getattr(self, "live_before_shutdown", None),
            "s2_memory_final": getattr(self, "memory_final", None),
            "s2_work_records": self.history,
        }

    def shutdown(self):
        try:
            if not self.closed:
                self.live_before_shutdown = len(self.states)
                if control()["barrier_ns"] is not None:
                    self._audit()
                self.memory_final = cuda_memory(self.worker.torch)
        finally:
            super().shutdown()


class S2DualMachine(BatchedDualDraftMachine):
    def execute_batch(self, operation, rows):
        packet = control()
        assigned = {rid: r["cohort"] for rid, r in packet["requests"].items()}
        cohort = cohort_for(assigned, [r["request_id"] for r in rows])
        require(
            cohort in ("A", "B") and all(r["logical_cohort"] == cohort for r in rows),
            "S2 Draft cohort mutation/mixed execution unit",
        )
        started = time.monotonic_ns()
        tail = operation == "finish_tail"
        ids = [r["request_id"] for r in rows]
        terminal_drain = None
        if tail:
            before = self.backend.physical_rows()
            before_states = {
                rid: {
                    "finished": self._state(rid).finished,
                    "proposal_present": self._state(rid).proposal is not None,
                    "prefix_version": self._state(rid).prefix_version,
                    "prefix_length": len(self._state(rid).committed_token_ids),
                    "prefix_sha256": token_prefix_hash(self._state(rid).committed_token_ids),
                }
                for rid in ids
            }
            unrelated = {rid: r for rid, r in before.items() if rid not in ids}
            before_proposals = self.backend.metrics.forwards["proposal"]
            self.backend.terminal_release = {"request_ids": ids}
        try:
            # Unchanged dispatcher enforces terminal one-token/no-proposal and
            # prefix/version contracts before the real commit, fence and release.
            results = super().execute_batch(operation, rows)
            if tail:
                after = self.backend.physical_rows()
                receipt = self.backend.terminal_release
                require(
                    after == unrelated
                    and digest(unrelated) == receipt.get("unrelated_before_release_sha256")
                    and all(rid in self.backend.retired for rid in ids),
                    "S2 terminal drain changed unrelated KV or failed to retire its requests",
                )
                terminal_drain = {
                    "schema_version": "specrhythm.s2-terminal-drain.v1",
                    "rows": [dict(r) for r in rows],
                    "before": before_states,
                    "results": results,
                    "release": dict(receipt),
                    "unrelated_request_ids": sorted(unrelated),
                    "unrelated_before_sha256": digest(unrelated),
                    "unrelated_after_sha256": digest(after),
                    "retired_request_ids": sorted(set(ids) & self.backend.retired),
                    "proposal_forward_count": (
                        self.backend.metrics.forwards["proposal"] - before_proposals
                    ),
                    "physical_gpu_id": self.backend.provenance["physical_gpu_id"],
                    "gpu_uuid": self.backend.provenance["gpu_uuid"],
                }
        finally:
            self.backend.terminal_release = None
        self.backend.history.append(
            {
                "operation": operation,
                "logical_cohort": cohort,
                "request_ids": [r["request_id"] for r in rows],
                "host_start_ns": started,
                "host_end_ns": time.monotonic_ns(),
                "terminal_by_request": {r["request_id"]: r.get("terminal", False) for r in rows},
                **({"terminal_drain": terminal_drain} if tail else {}),
            }
        )
        return [{**r, "logical_cohort": cohort} for r in results]


def serve(config, directory, socket_path, mode, *, backend_class=S2DraftBackend):
    report = directory / "draft-backend-report.json"
    ready = directory / "draft-service-ready.json"
    events = CheckpointJsonl(directory / "draft-work-events.jsonl")

    def factory():
        backend = backend_class(config)
        write_once(directory / "draft-startup.json", backend.provenance)
        cls = S2DualMachine if mode == "pingpong" else BatchedDraftStateMachine
        return cls(backend, candidate_budget=4, report_path=report)

    if mode == "pingpong":
        controller = BatchedDualDraftController(factory, events)
        server = DualDraftUnixServer(
            socket_path,
            controller,
            ready_path=ready,
            transport_log=CheckpointJsonl(directory / "draft-transport.jsonl"),
        )
        try:
            server.serve()
        finally:
            controller.shutdown(failed=server.running)
    else:
        machine = factory()
        try:
            DraftUnixServer(socket_path, machine, event_log=events).serve(ready)
        finally:
            if not machine.backend.closed:
                machine.backend._fail()
                machine.backend.shutdown()
                if not report.exists():
                    write_once(report, machine.backend.report())
