"""Explicit fixed-diagnostic Serial-eager service on one existing GPU owner."""

from __future__ import annotations

import os
import socket
import sys
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
    deadline_contract = None
    drain_failure = None

    def _dispatch(self, operation, payload):
        guarded = self.deadline_contract is not None
        try:
            if guarded and ("deadline_ns" in payload or operation in
                            ("pp_stop", "diagnostic_settle", "shutdown")):
                if self.drain_failure is not None:
                    raise self.drain_failure
                self.deadline_ns = self.deadline_contract.bind(payload, operation)
            elif "deadline_ns" in payload:
                # Legacy non-K3 protocol remains unchanged.
                if self.deadline_ns is None:
                    self.deadline_ns = payload["deadline_ns"]
                elif self.deadline_ns != payload["deadline_ns"]:
                    raise ValueError("eager drain must retain its original absolute deadline")
            result = self.machine.call(operation, payload)
        except BaseException as error:
            if guarded and (hasattr(error, "deadline_context") or operation in
                            ("pp_stop", "diagnostic_settle", "shutdown")):
                self.drain_failure = self.drain_failure or error
                self.running = False
            raise
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


def serve(config, directory, socket_path, *, backend_class=None, prepost_mode=None):
    from specrhythm.serving.ping_prepost import SCHEDULED_MODES as PING_MODES

    report = directory / "draft-backend-report.json"
    cls = backend_class or EagerFixedDraftBackend
    if not issubclass(cls, GPUContinuationBackendMixin):
        cls = type("OptInEagerDraftBackend", (GPUContinuationBackendMixin, cls), {})

    if prepost_mode is not None:
        from specrhythm.continuation.prepost import MODES
        from specrhythm.continuation.prepost_backend import PrePostBackendMixin

        if prepost_mode not in (*MODES, *PING_MODES):
            raise ValueError("unknown explicit prepost mode")
        if prepost_mode in PING_MODES:
            from specrhythm.continuation.ping_prepost_backend import PingPrePostBackendMixin

            PrePostBackendMixin = PingPrePostBackendMixin
        if prepost_mode.endswith("-k3"):
            from specrhythm.continuation.k3_backend import K3BackendMixin

            PrePostBackendMixin = K3BackendMixin
        cls = type("PrePostFixedDraftBackend", (PrePostBackendMixin,
                   backend_class or FixedDraftBackend), {})

    def factory():
        backend = cls(config)
        write_once(directory / "draft-startup.json", backend.provenance)
        if prepost_mode is not None:
            from specrhythm.serving.prepost_machine import PrePostMachine

            if prepost_mode in PING_MODES:
                from specrhythm.serving.ping_prepost_machine import PingPrePostMachine

                PrePostMachine = PingPrePostMachine
            if prepost_mode.endswith("-k3"):
                from specrhythm.serving.k3 import configuration_of
                from specrhythm.serving.k3_machine import K3Machine

                PrePostMachine = K3Machine
            return PrePostMachine(
                backend,
                request_ids=tuple(control()["requests"]),
                eager=prepost_mode
                in (
                    "serial-eager-prepost3",
                    "pingpong-eager-prepost3",
                    "serial-eager-k3",
                    "pingpong-eager-k3",
                ),
                report_path=report,
                **({"mode": prepost_mode, "configuration": configuration_of(control()),
                    "draft_dispatch": os.environ.get("SR_K3_DRAFT_DISPATCH")}
                   if prepost_mode.endswith("-k3") else {}),
            )
        return EagerSerialMachine(
            backend, request_ids=tuple(control()["requests"]), report_path=report
        )

    if prepost_mode in PING_MODES:
        from specrhythm.serving.ping_prepost_owner import PingPrePostOwner

        if prepost_mode.endswith("-k3"):
            from specrhythm.serving.k3_owner import K3Owner

            PingPrePostOwner = K3Owner
        owner = PingPrePostOwner(factory)
    else:
        owner = EagerOwner(factory)
    server = EagerSerialServer(
        socket_path, owner, event_log=CheckpointJsonl(directory / "draft-work-events.jsonl")
    )
    strict_deadline = bool(prepost_mode and prepost_mode.endswith("-k3"))
    if strict_deadline:
        from specrhythm.phase4.drain_deadline import DrainDeadline

        server.deadline_contract = DrainDeadline(prepost_mode, directory.resolve())
    try:
        server.serve(directory / "draft-service-ready.json")
        if server.drain_failure is not None:
            raise server.drain_failure
        if owner.failure is not None:
            raise RuntimeError(str(owner.failure))
    except BaseException as error:
        record_error(directory, error, "eager_draft_server")
        raise
    finally:
        original_error = sys.exc_info()[1]
        cleanup_error = None
        deadline = server.deadline_ns
        if not strict_deadline:
            deadline = deadline or time.monotonic_ns() + 60_000_000_000
        try:
            # Reuse the coordinator's shared deadline, including join and log flush.
            if strict_deadline and original_error is not None:
                # Signal the existing owner fault path at its next fenced command boundary.
                # No new budget if a valid coordinator deadline never arrived: the external
                # process supervisor still owns the original setup/drain absolute bound.
                owner.fail_drain(original_error, deadline)
                raise original_error
            if strict_deadline:
                from specrhythm.phase4.drain_deadline import MISSING, validate_deadline

                validate_deadline(deadline if deadline is not None else MISSING,
                                  mode=prepost_mode, directory=directory.resolve(),
                                  phase="owner_close")
            owner.close(deadline)
            from specrhythm.serving.fixed_logging import finish_current
            from specrhythm.serving.fixed_settle import remaining

            remaining(deadline)
            finish_current("draft")
            remaining(deadline)
        except BaseException as error:
            cleanup_error = error
            record_error(directory, error, "draft_owner_close_or_final_flush")
            if original_error is None:
                raise
        finally:
            if (owner.machine is not None and not owner._thread.is_alive() and not report.exists()
                    and not (strict_deadline and (original_error or cleanup_error))):
                def build_report():
                    return {**owner.machine.backend.report(),
                            ("prepost" if prepost_mode else "rolling_eager"):
                                owner.machine.eager_report()}
                try:
                    if prepost_mode and prepost_mode.endswith("-k3"):
                        from specrhythm.phase4.report_publication import publish_final_report

                        publish_final_report(report, build_report, deadline_ns=deadline,
                                             owner_stopped=not owner._thread.is_alive(),
                                             mode=prepost_mode)
                    else:
                        write_immutable_report(report, build_report())
                except BaseException as error:
                    record_error(directory, error, "draft_final_report")
                    if original_error is None and cleanup_error is None:
                        raise
