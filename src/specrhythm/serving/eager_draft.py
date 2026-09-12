"""Explicit fixed-diagnostic Serial-eager service on one existing GPU owner."""

from __future__ import annotations

import os
import socket
import time

from specrhythm.continuation.gpu_backend import GPUContinuationBackendMixin
from specrhythm.phase4.batched_draft_service import write_immutable_report
from specrhythm.phase4.draft_service import DraftUnixServer
from specrhythm.phase4.manifest import atomic_write_json
from specrhythm.phase4.serial import PROTOCOL_VERSION
from specrhythm.phase4.transport import CheckpointJsonl
from specrhythm.serving.eager_machine import EagerSerialMachine
from specrhythm.serving.eager_owner import EagerOwner
from specrhythm.serving.fixed_artifacts import record_error
from specrhythm.serving.fixed_draft import FixedDraftBackend
from specrhythm.serving.s1_workload import write_once
from specrhythm.serving.s2_pool import control


class EagerFixedDraftBackend(GPUContinuationBackendMixin, FixedDraftBackend):
    pass


class EagerSerialServer(DraftUnixServer):
    deadline_ns = None

    def _dispatch(self, operation, payload):
        if "deadline_ns" in payload:
            if self.deadline_ns is None:
                self.deadline_ns = payload["deadline_ns"]
            elif self.deadline_ns != payload["deadline_ns"]:
                raise ValueError("eager drain must retain its original absolute deadline")
        try:
            result = self.machine.call(operation, payload)
        finally:
            if self.machine.failure is not None:
                self.running = False
        if operation == "shutdown":
            self.running = False
        return result

    def serve(self, ready_path):
        if self.socket_path.exists():
            raise FileExistsError("refusing to replace existing eager socket")
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o600)
            server.listen(8)
            atomic_write_json(
                ready_path,
                {
                    "schema_version": "specrhythm.phase4-draft-service-ready.v1",
                    "protocol_version": PROTOCOL_VERSION,
                    "socket_file": self.socket_path.name,
                    "pid": os.getpid(),
                    "backend": self.machine.backend.backend_name,
                    "provenance": self.machine.backend.provenance,
                    "gpu_correctness_result": False,
                    "gpu_performance_result": False,
                    "rolling_eager_qualification": "PENDING",
                },
            )
            while self.running:
                connection, _ = server.accept()
                with connection:
                    self._handle(connection)
        self.socket_path.unlink(missing_ok=True)


def serve(config, directory, socket_path, *, backend_class=None):
    report = directory / "draft-backend-report.json"
    cls = backend_class or EagerFixedDraftBackend
    if not issubclass(cls, GPUContinuationBackendMixin):
        cls = type("OptInEagerDraftBackend", (GPUContinuationBackendMixin, cls), {})

    def factory():
        backend = cls(config)
        write_once(directory / "draft-startup.json", backend.provenance)
        return EagerSerialMachine(
            backend, request_ids=tuple(control()["requests"]), report_path=report
        )

    owner = EagerOwner(factory)
    server = EagerSerialServer(
        socket_path, owner, event_log=CheckpointJsonl(directory / "draft-work-events.jsonl")
    )
    try:
        server.serve(directory / "draft-service-ready.json")
        if owner.failure is not None:
            raise RuntimeError(str(owner.failure))
    except BaseException as error:
        record_error(directory, error, "eager_draft_server")
        raise
    finally:
        try:
            # Reuse the coordinator's shared deadline, including join and log flush.
            deadline = server.deadline_ns or time.monotonic_ns() + 60_000_000_000
            owner.close(deadline)
            from specrhythm.serving.fixed_logging import finish_current
            from specrhythm.serving.fixed_settle import remaining

            remaining(deadline)
            finish_current("draft")
            remaining(deadline)
        finally:
            if owner.machine is not None and not owner._thread.is_alive() and not report.exists():
                write_immutable_report(
                    report,
                    {
                        **owner.machine.backend.report(),
                        "rolling_eager": owner.machine.eager_report(),
                    },
                )
