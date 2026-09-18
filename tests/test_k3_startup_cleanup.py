"""Real pre-drive failure and real socket -> owner shutdown; no GPU execution."""

import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from test_k3 import Backend
from test_k3_capacity import capacity_run as _capacity_run
from test_serial_eager_owner import OwnerWorker

from specrhythm.phase4.transport import CheckpointJsonl, UnixDraftClient
from specrhythm.serving import eager_draft, fixed_runtime
from specrhythm.serving.common import DataError, read_json
from specrhythm.serving.fixed_artifacts import record_error, retained_report
from specrhythm.serving.k3 import MODES
from specrhythm.serving.k3_machine import K3Machine
from specrhythm.serving.k3_owner import K3Owner
from specrhythm.serving.k3_startup_cleanup import StartupCleanup
from specrhythm.serving.ping_prepost_delivery import export

capacity_run = _capacity_run


@pytest.fixture
def service(tmp_path, monkeypatch):
    ready, errors = threading.Event(), []
    original = eager_draft.atomic_write_json
    def publish(path, value):
        original(path, value)
        ready.set()
    monkeypatch.setattr(eager_draft, "atomic_write_json", publish)
    owner = K3Owner(lambda: K3Machine(Backend(NS(max_model_len=4096), worker=OwnerWorker()),
                                    request_ids=(), eager=False))
    with tempfile.TemporaryDirectory(prefix="sr-k3-") as name:
        socket = Path(name) / "draft.sock"
        server = eager_draft.EagerSerialServer(socket, owner,
            event_log=CheckpointJsonl(tmp_path / "transport-events.jsonl"))
        def serve():
            try:
                server.serve(tmp_path / "ready.json")
            except BaseException as error:
                errors.append(error)
                ready.set()
        thread = threading.Thread(target=serve)
        thread.start()
        assert ready.wait(3) and not errors
        client = UnixDraftClient(socket, timeout_seconds=3)
        try:
            yield NS(client=client, owner=owner, socket=socket, thread=thread)
        finally:
            if not owner.closed:
                client.call("shutdown", {"deadline_ns": time.monotonic_ns() + 3_000_000_000})
            thread.join(3)
            assert not thread.is_alive() and not errors
            assert not owner._thread.is_alive() and not socket.exists()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("secondary", [None, "target", "transport", "constructor"])
def test_initialized_capacity_failure_preserves_original_and_shuts_empty_draft(
    mode, secondary, capacity_run, service, tmp_path, monkeypatch
):
    h = capacity_run(mode)
    h.raw[0]["block_size"] = False  # Actual rank producer data, not a mocked capacity exception.
    monkeypatch.setattr(fixed_runtime, "client_for", lambda mode: service.client)
    if secondary == "transport":
        def unavailable(mode):
            raise OSError("missing service endpoint")
        monkeypatch.setattr(fixed_runtime, "client_for", unavailable)
    if secondary == "target":
        def broken(**kw):
            raise OSError("Target shutdown failed")
        h.llm.llm_engine.engine_core.shutdown = broken
    if secondary == "constructor":
        def constructor(*a, **kw):
            raise DataError("model constructor failed before returning handle")
        monkeypatch.setattr(fixed_runtime, "make_engine", constructor)
    expected = ("model constructor" if secondary == "constructor"
                else "invalid K3 raw rank capacity")
    with pytest.raises(DataError, match=expected):
        fixed_runtime.run(tmp_path, None, tmp_path, h.point, probe=True)
    primary = read_json(tmp_path / "diagnostic-primary-error.json")
    assert expected in primary["error"]
    cleanup = read_json(tmp_path / "startup-cleanup.json")
    assert cleanup["original_error"] == primary["error"] and cleanup["execution_failed"]
    assert cleanup["cleanup_qualification"] == "PENDING_SUPERVISOR"
    assert cleanup["release_attempts_complete"] is (secondary is None)
    assert cleanup["actions"]["target_engine_shutdown"]["status"] == (
        "FAILED" if secondary == "target" else "NO_HANDLE" if secondary == "constructor"
        else "RETURNED")
    if secondary != "transport":
        assert service.owner.closed and not service.owner._thread.is_alive()
        assert cleanup["actions"]["draft_shutdown"]["receipt"]["physical_live_requests"] == 0
    else:
        assert cleanup["actions"]["draft_shutdown"]["status"] == "FAILED"
    # Retention/export cannot replace capacity failure with secondary cleanup diagnostics.
    report = retained_report(tmp_path, dict(mode=mode))
    assert report["primary_error"]["error"] == primary["error"]
    package = tmp_path / "failure.tar.gz"
    result = export(tmp_path, package, first_code=1, stage="capacity", modes=MODES)
    assert result["first_exit_code"] == 1 and "startup-cleanup.json" in result["logical_paths"]


def test_cleanup_is_once_only_and_recording_failure_does_not_mask_primary(
    tmp_path, monkeypatch
):
    from specrhythm.serving import k3_startup_cleanup

    calls = []
    primary = DataError("original capacity error")
    record_error(tmp_path, primary, "runtime")
    def cannot_write(*a):
        raise OSError("cleanup report disk failure")
    monkeypatch.setattr(k3_startup_cleanup, "publish", cannot_write)
    client = NS(call=lambda *a: calls.append("Draft") or dict(
        shutdown=True, physical_live_requests=0, pending_work=[]))
    engine = NS(llm_engine=NS(engine_core=NS(shutdown=lambda **kw: calls.append("Target"))))
    c = StartupCleanup(tmp_path, MODES[0], primary, time.monotonic_ns() + 3_000_000_000)
    result = c.finish(engine, lambda: client)
    assert c.finish(engine, lambda: client) is result and calls == ["Draft", "Target"]
    assert result["recording_errors"] and result["original_error"] == str(primary)
    assert read_json(tmp_path / "diagnostic-primary-error.json")["error"] == str(primary)
    assert read_json(tmp_path / "diagnostic-secondary-errors.json")
