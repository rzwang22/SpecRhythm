"""Coordinator payload -> real framed RPC dispatch -> factory -> report receipts (CPU)."""

import json
import socket
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from test_serial_eager_owner import OwnerWorker

from specrhythm.phase4.transport import UnixDraftClient
from specrhythm.phase4.vllm_draft_backend import VllmBatchedDraftBackend
from specrhythm.serving import eager_draft, fixed_draft, fixed_drain, fixed_logging
from specrhythm.serving.fixed_settle import (
    DiagnosticSerialServer,
    PublishedDiagnosticSerialMachine,
)
from specrhythm.serving.k3 import MODES
from specrhythm.serving.s2_pool import prefix_record


class Hardware(VllmBatchedDraftBackend):
    def __init__(self, config):
        super().__init__(config, worker=OwnerWorker())
        self.initialize_many((("r", (1, 2, 3)),))

    def physical_rows(self):
        return {rid: prefix_record(s.prefix, s.materialized, [[id(s)]])
                for rid, s in self.states.items()}


def run_chain(directory, monkeypatch, mode, *, transform=lambda op, row: row,
              repeat_settlement=False, before_failure=None):
    """Only GPU and accept-loop orchestration are substituted; all messages are real RPC."""
    from specrhythm.phase4.serial import token_prefix_hash

    operations, results = [], []
    monkeypatch.setenv("SR_K3_MANAGED_LOCAL", "1")
    monkeypatch.setattr(fixed_logging, "_CURRENT", None)
    monkeypatch.setattr(eager_draft, "control", lambda: {"requests": {"r": {}}})

    def serve(server, ready):
        machine = server.machine if mode == "target" else server.machine.machine
        assert isinstance(machine, PublishedDiagnosticSerialMachine) == (mode == "target")
        with tempfile.TemporaryDirectory(prefix="sr-deadline-", dir="/tmp") as short:
            path = Path(short) / "rpc"
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                listener.bind(str(path))
                listener.listen(1)
                listener.settimeout(3)
                real_client = UnixDraftClient(path, timeout_seconds=3)
                with ThreadPoolExecutor(max_workers=1) as executor:
                    def call(operation, row):
                        operations.append((operation, dict(row)))
                        future = executor.submit(
                            real_client.call, operation, transform(operation, row))
                        connection, _ = listener.accept()
                        with connection:
                            server._handle(connection)
                        try:
                            result = future.result(timeout=3)
                        except RuntimeError:
                            if before_failure is not None:
                                before_failure(server, machine)
                            raise
                        if repeat_settlement and operation == "diagnostic_settle":
                            future = executor.submit(real_client.call, operation, row)
                            connection, _ = listener.accept()
                            with connection:
                                server._handle(connection)
                            again = future.result(timeout=3)
                            assert again["binding"] == result["binding"]
                            assert (again["resources_released_ns"]
                                    == result["resources_released_ns"])
                        return result

                    scheduler = NS(requests={}, deferred_frees=[])
                    engine = NS(abort_request=lambda ids: None,
                                has_unfinished_requests=lambda: False)
                    clock = NS(
                        rows={"r": dict(state="STAGED", generated_token_ids=[], commits=[],
                                        admission_ns=None, resources_released=False)},
                        definitions={"r": NS(prompt_token_ids=(1, 2, 3))},
                        _event=lambda *a, **kw: None,
                    )
                    warm = {"r": NS(prefix_version=0,
                                     logical_committed_prefix_sha256=token_prefix_hash((1, 2, 3)))}
                    result = fixed_drain.settle(
                        NS(collective_rpc=lambda *a, **kw: []), engine, scheduler,
                        NS(call=call), clock, warm, [], directory, mode, 60,
                        lambda: None, lambda *a, **kw: None,
                    )
                    results.append(result)
        assert machine.backend.closed and not machine.backend.states

    monkeypatch.setattr(DiagnosticSerialServer, "serve", serve)
    monkeypatch.setattr(eager_draft.EagerSerialServer, "serve", serve)
    fixed_draft.serve(NS(max_model_len=4096), directory, directory / "unused", mode,
                      backend_class=Hardware)
    return operations, results


@pytest.mark.parametrize("mode", ("target", *MODES))
def test_coordinator_service_rpc_report_roundtrip(tmp_path, monkeypatch, mode):
    from specrhythm.phase4.report_publication import qualify_final_report

    operations, results = run_chain(tmp_path, monkeypatch, mode)
    state = json.loads((tmp_path / "drain-state.json").read_text())
    assert state["status"] == "COMPLETE"
    deadline = state["deadline_ns"]
    assert deadline - state["start_ns"] == 60_000_000_000
    for operation, row in operations:
        if operation in ("pp_stop", "diagnostic_settle", "shutdown"):
            assert type(row["deadline_ns"]) is int and row["deadline_ns"] == deadline
    receipt = qualify_final_report(tmp_path / "draft-backend-report.json")
    assert receipt["deadline_ns"] == deadline and receipt["owner_stopped"]
    assert results[0][1]["shutdown"] is True


def test_settlement_repeat_and_shutdown_keep_original_deadline(tmp_path, monkeypatch):
    from specrhythm.phase4.drain_deadline import DrainDeadline

    observed = []
    bind = DrainDeadline.bind

    def repeated(contract, row, phase):
        value = bind(contract, row, phase)
        assert bind(contract, dict(row), phase) == value
        observed.append((phase, value))
        return value

    monkeypatch.setattr(DrainDeadline, "bind", repeated)
    run_chain(tmp_path, monkeypatch, "target", repeat_settlement=True)
    assert {phase for phase, _ in observed} >= {"diagnostic_settle", "shutdown"}
    assert len({deadline for _, deadline in observed}) == 1


@pytest.mark.parametrize("mode", ("target", *MODES))
@pytest.mark.parametrize("bad", ["missing", None, True, "123", 1.5, 0, -1, 2**63, "conflict"])
def test_shutdown_rejects_bad_contract_through_rpc_and_export(
    tmp_path, monkeypatch, mode, bad
):
    from test_k3_report_publication import payload

    from specrhythm.phase4.report_publication import qualify_final_report
    from specrhythm.serving.fixed_artifacts import record_error, retained_report
    from specrhythm.serving.k3_local_run import validate_archive
    from specrhythm.serving.ping_prepost_delivery import export

    def transform(operation, row):
        if operation != "shutdown":
            return row
        if bad == "missing":
            return {}
        return {"deadline_ns": row["deadline_ns"] + 1 if bad == "conflict" else bad}

    def before_failure(server, machine):
        # Contract rejected before backend.shutdown; no retry may turn it into success.
        assert not machine.backend.closed and server.running is False
        with pytest.raises(ValueError, match="deadline"):
            server._dispatch("shutdown", {"deadline_ns":
                json.loads((tmp_path / "drain-state.json").read_text())["deadline_ns"]})

    with pytest.raises(RuntimeError, match="DeadlineContractError") as caught:
        run_chain(tmp_path, monkeypatch, mode, transform=transform,
                  before_failure=before_failure)
    error = caught.value
    assert "TimeoutError" not in str(error) and "expired_deadline" not in str(error)
    context = error.deadline_context
    assert context["mode"] == mode and context["phase"] == "shutdown"
    assert context["run_directory"] == str(tmp_path.resolve())
    assert ("remaining_ns" in context) == (bad == "conflict")
    assert context["received_deadline_type"] == (
        "missing" if bad == "missing" else "int" if bad == "conflict" else type(bad).__name__)
    with pytest.raises((OSError, ValueError)):
        qualify_final_report(tmp_path / "draft-backend-report.json")
    state = json.loads((tmp_path / "drain-state.json").read_text())
    assert state["status"] == "FAILED" and state["deadline_ns"] > state["end_ns"]
    record_error(tmp_path, BrokenPipeError("secondary closed socket"), "after_shutdown_failure")
    summary = retained_report(tmp_path, {"effective_exit_code": 17})
    assert summary["effective_exit_code"] == 17
    first = json.loads((tmp_path / "diagnostic-primary-error.json").read_text())
    assert first["deadline_context"] == context and "secondary closed socket" not in first["error"]
    # Real summarization and exporter keep original error/exit code; no fake report completion.
    (tmp_path / "failure.json").write_text(json.dumps(summary))
    archive = tmp_path.parent / (tmp_path.name + ".tar.gz")
    exported = export(tmp_path, archive, first_code=17, stage="joint_gpu_correctness", modes=MODES)
    assert exported["first_exit_code"] == 17
    assert validate_archive(archive)["archive_integrity"] == "VERIFIED"
    assert json.loads(payload(archive, "diagnostic-primary-error.json")[0]) == first
    assert json.loads(payload(archive, "failure.json")[0])["effective_exit_code"] == 17


@pytest.mark.parametrize("mode", ("target", *MODES))
def test_expired_valid_deadline_is_timeout_before_shutdown(tmp_path, monkeypatch, mode):
    from specrhythm.phase4 import drain_deadline

    bind = drain_deadline.DrainDeadline.bind

    def expired(contract, payload, phase):
        if phase == "shutdown":
            return drain_deadline.validate_deadline(
                payload["deadline_ns"], mode=mode, directory=tmp_path.resolve(), phase=phase,
                original=contract.value, now_ns=payload["deadline_ns"])
        return bind(contract, payload, phase)

    monkeypatch.setattr(drain_deadline.DrainDeadline, "bind", expired)
    with pytest.raises(RuntimeError, match="DeadlineExpired: expired_deadline") as error:
        run_chain(tmp_path, monkeypatch, mode)
    assert error.value.deadline_context["remaining_ns"] == 0


@pytest.mark.parametrize("bad", [None, False, "123", 1.0, 0, -5, 2**63])
def test_publisher_invalid_parameter_is_not_timeout(tmp_path, bad):
    from specrhythm.phase4.drain_deadline import DeadlineContractError
    from specrhythm.phase4.report_publication import publish_final_report

    called = []
    with pytest.raises(DeadlineContractError, match="invalid_deadline") as error:
        publish_final_report(tmp_path / "draft-backend-report.json",
                             lambda: called.append(True), deadline_ns=bad,
                             owner_stopped=True, mode="target")
    assert not called and "remaining_ns" not in error.value.deadline_context
    state = json.loads((tmp_path / "draft-report-state.json").read_text())
    assert state["status"] == "FAILED" and state["failed_stage"] == "deadline_validation"
    assert state["deadline_ns"] == bad


@pytest.mark.parametrize("phase", ["build", "serialize", "fsync", "publish", "verify"])
@pytest.mark.parametrize("expiry", [False, True])
def test_publication_failures_preserve_phase_and_never_qualify(
    tmp_path, monkeypatch, phase, expiry
):
    import time

    from specrhythm.phase4 import report_publication as pub

    path = tmp_path / "draft-backend-report.json"
    deadline = time.monotonic_ns() + 10**10

    def fault():
        if expiry:
            monkeypatch.setattr(pub.time, "monotonic_ns", lambda: deadline)
        else:
            raise OSError("injected " + phase)

    def build():
        if phase == "build":
            fault()
        return {"provenance": {}}

    if phase == "serialize":
        dump = pub.json.dump

        def interrupted(value, handle, **kwargs):
            result = dump(value, handle, **kwargs)
            if "report_publication" in value:
                fault()
            return result

        monkeypatch.setattr(pub.json, "dump", interrupted)
    elif phase == "fsync":
        from specrhythm.io_context import IO_CONTEXT

        fsync = pub.os.fsync

        def interrupted(fd):
            result = fsync(fd)
            if getattr(IO_CONTEXT, "name", None) == path.name:
                fault()
            return result

        monkeypatch.setattr(pub.os, "fsync", interrupted)
    elif phase in ("publish", "verify"):
        module, name = (pub.os, "link") if phase == "publish" else (pub, "sha256_file")
        operation = getattr(module, name)

        def interrupted(*args):
            result = operation(*args)
            fault()
            return result

        monkeypatch.setattr(module, name, interrupted)
    with pytest.raises(TimeoutError if expiry else OSError) as error:
        pub.publish_final_report(path, build, deadline_ns=deadline, owner_stopped=True,
                                 mode="target")
    state = json.loads((tmp_path / "draft-report-state.json").read_text())
    assert state["status"] == "FAILED"
    assert error.value.report_publication_context["mode"] == "target"
    assert state["failed_stage"] == ("fsync" if expiry and phase == "serialize" else phase)
    with pytest.raises(ValueError, match="incomplete"):
        pub.qualify_final_report(path)
    assert (tmp_path / state["temporary_file"]).exists()


@pytest.mark.parametrize("mode", ("target", *MODES))
def test_missing_first_deadline_uses_fault_cleanup_without_new_budget(
    tmp_path, monkeypatch, mode
):
    import threading

    stopped, backends = threading.Event(), []
    shutdown = Hardware.shutdown

    def observed(backend):
        try:
            return shutdown(backend)
        finally:
            backends.append(backend)
            stopped.set()

    monkeypatch.setattr(Hardware, "shutdown", observed)

    def transform(operation, row):
        assert operation == ("diagnostic_settle" if mode == "target" else "pp_stop")
        return {k: v for k, v in row.items() if k != "deadline_ns"}

    with pytest.raises(RuntimeError, match="missing_deadline") as error:
        run_chain(tmp_path, monkeypatch, mode, transform=transform)
    assert error.value.deadline_context["original_deadline_ns"] is None
    # Test-only wait observes the real asynchronous owner fault completion; production
    # does not generate or wait for a replacement 60s budget when no deadline arrived.
    assert stopped.wait(3)
    assert all(b.closed and b.failed for b in backends)
    assert not (tmp_path / "draft-report-state.json").exists()
