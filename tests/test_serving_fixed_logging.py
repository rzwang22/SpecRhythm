"""Bounded persistence, real checksummed framing and diagnostic stop contracts (CPU)."""

import hashlib
import json
import os
import sys
import time
from types import SimpleNamespace

import pytest
from test_serving_fixed_settle import owner as _owner

from specrhythm.phase4.transport import CheckpointJsonl
from specrhythm.serving import fixed_logging
from specrhythm.serving.common import DataError, read_json
from specrhythm.serving.fixed_artifacts import record_error
from specrhythm.serving.fixed_logging import DiagnosticLogs
from specrhythm.serving.fixed_observe import Timers
from specrhythm.serving.fixed_plan import settings
from specrhythm.serving.s2_pool import publish

owner = _owner


def deadline():
    return time.monotonic_ns() + 5_000_000_000


@pytest.mark.parametrize("mode", ["original-live", "buffered-live"])
def test_real_framing_order_count_bounds_and_default(tmp_path, monkeypatch, mode):
    assert settings()["observation"] == "original-live"
    assert settings(observation="buffered-live")["observation"] == "buffered-live"
    calls = []
    monkeypatch.setattr(os, "fsync", lambda fd: calls.append(fd))
    logs = DiagnosticLogs(tmp_path, mode, max_records=3, max_bytes=512)
    log = CheckpointJsonl(tmp_path / "verification-events.jsonl")
    rows = [
        {"sequence": i, "timestamp_ns": time.monotonic_ns(), "request_id": "A"} for i in range(10)
    ]
    for r in rows:
        logs.append(log, r, CheckpointJsonl.append)
    before = len(calls)
    result = logs.finish(deadline())
    decoded = log.read()
    assert [{k: v for k, v in r.items() if k != "record_sha256"} for r in decoded] == rows
    baseline = CheckpointJsonl(tmp_path / "native.jsonl")
    for row in rows:
        CheckpointJsonl.append(baseline, row)
    assert baseline.path.read_bytes() == log.path.read_bytes()
    del calls[before + (1 if mode == "buffered-live" else 0) :]
    assert result["produced_records"] == result["written_records"] == 10
    assert result["integrity_complete"]
    assert logs.finish(deadline()) == result  # No second physical flush/release.
    if mode == "original-live":
        assert before == len(calls) == 10 and result["peak_buffer_records"] == 0
    else:
        assert len(calls) == 4 and before == 3
        assert result["peak_buffer_records"] <= 3 and result["peak_buffer_bytes"] <= 512
        assert (
            result["streams"][log.path.name]["written_bytes_sha256"]
            == hashlib.sha256(log.path.read_bytes()).hexdigest()
        )
    assert result["fsync_count"] == len(calls)


def test_byte_threshold_oversized_read_your_writes_and_protocol_immediate(tmp_path):
    logs = DiagnosticLogs(tmp_path, "buffered-live", max_records=100, max_bytes=256)
    log = CheckpointJsonl(tmp_path / "draft-work-events.jsonl")
    for r in [{"n": 1}, {"n": 2}, {"large": "x" * 1024}, {"n": 3}]:
        logs.append(log, r, CheckpointJsonl.append)
    protocol = CheckpointJsonl(tmp_path / "admission-events.jsonl")
    logs.append(protocol, {"released": True}, CheckpointJsonl.append)
    assert len(protocol.read()) == 1 and protocol.read()[0]["released"] is True
    publish(tmp_path / "control.json", {"ready": True})
    assert read_json(tmp_path / "control.json") == {"ready": True}
    assert logs.pending  # Protocol publication did not depend on final diagnostic flush.
    logs.before_read(log)
    assert len(log.read()) == 4
    r = logs.finish(deadline())
    assert r["oversized_direct_records"] == 1 and r["peak_buffer_bytes"] <= 256
    assert r["flush_reasons"]["read-your-writes"] == 1


@pytest.mark.parametrize("failure", ["fsync", "timeout", "publication"])
def test_flush_error_sticky_primary_and_counts_never_claim_completion(tmp_path, failure):
    logs = DiagnosticLogs(tmp_path, "buffered-live")
    logs.append(
        CheckpointJsonl(tmp_path / "proposal-events.jsonl"), {"a": 1}, CheckpointJsonl.append
    )
    record_error(tmp_path, RuntimeError("first execution fault"), "worker")
    if failure == "fsync":
        logs.fsync = lambda fd: (_ for _ in ()).throw(OSError("disk failed"))
    if failure == "publication":
        original = logs._publish
        logs._publish = lambda state: (
            original(state)
            if state != "COMPLETE"
            else (_ for _ in ()).throw(OSError("receipt failed"))
        )
    with pytest.raises((OSError, DataError)):
        logs.finish(time.monotonic_ns() - 1 if failure == "timeout" else deadline())
    row = read_json(logs.receipt_path)
    assert row["status"] == "FAILED" and not row["integrity_complete"]
    assert (
        read_json(tmp_path / "diagnostic-primary-error.json")["error"] == "first execution fault"
    )
    with pytest.raises(DataError):
        logs.finish(deadline())
    with pytest.raises(DataError):
        logs.append(
            CheckpointJsonl(tmp_path / "proposal-events.jsonl"), {}, CheckpointJsonl.append
        )


def install(tmp_path, monkeypatch):
    monkeypatch.setenv("SR_FIXED_POINT", str(tmp_path / "point.json"))
    monkeypatch.setenv("SR_FIXED_OBSERVATION", "buffered-live")
    monkeypatch.setattr(fixed_logging, "_CURRENT", None)
    # Restore the actual class after the same adapter used by real startup patches it.
    monkeypatch.setattr(CheckpointJsonl, "append", CheckpointJsonl.append)
    monkeypatch.setattr(CheckpointJsonl, "read", CheckpointJsonl.read)
    captured = []
    logs = fixed_logging.install_checkpoint_logging("coordinator", captured.append, Timers())
    return logs, captured


@pytest.mark.parametrize("case", ["sample", "unsynced", "time_before_verify", "initial", "target"])
def test_real_serial_stop_state_machine_with_buffered_diagnostics(tmp_path, monkeypatch, case):
    from test_serving_fixed_stop import test_serial_sample_stop_settles_real_pending_proposals

    logs, captured = install(tmp_path, monkeypatch)
    # Hardware fixture has in-process Draft and no TP processes. Use the same finalizer,
    # substituting only receipt acquisition and collective RPC, not stop or shutdown.
    monkeypatch.setattr(fixed_logging, "wait_draft_receipt", lambda directory, deadline: {})
    CheckpointJsonl(tmp_path / "proposal-events.jsonl").append({"sequence": 0})
    assert captured == [{"sequence": 0}] and logs.pending
    test_serial_sample_stop_settles_real_pending_proposals(tmp_path, monkeypatch, case)
    drain = read_json(tmp_path / "drain-state.json")
    assert drain["logging_finalization"]["coordinator"]["integrity_complete"]
    assert drain["start_ns"] <= logs.final_flush["start_ns"] <= logs.final_flush["end_ns"]
    assert logs.final_flush["end_ns"] <= drain["end_ns"] <= drain["deadline_ns"]
    assert logs.produced == logs.written and not logs.pending


@pytest.mark.parametrize("mode", ["serial-split", "pingpong"])
def test_real_owner_ready_inflight_stop_with_buffered_logs(tmp_path, monkeypatch, mode):
    from test_serving_fixed_runtime import (
        test_real_owner_dispatch_initial_work_wait_and_window_drain,
    )

    logs, _ = install(tmp_path, monkeypatch)
    monkeypatch.setattr(fixed_logging, "wait_draft_receipt", lambda directory, deadline: {})
    # Fixture work log is explicitly renamed to the audited real owner filename.
    native_init = CheckpointJsonl.__init__

    def initialize(log, path):
        native_init(
            log, path.with_name("draft-work-events.jsonl") if path.name == "work.jsonl" else path
        )

    monkeypatch.setattr(CheckpointJsonl, "__init__", initialize)
    test_real_owner_dispatch_initial_work_wait_and_window_drain(tmp_path, monkeypatch, mode)
    assert logs.produced > 100 and logs.produced == logs.written and logs.closed


def test_worker_finalizer_wait_order_remaining_budget_and_qualification(tmp_path, monkeypatch):
    monkeypatch.setenv("SR_FIXED_OBSERVATION", "buffered-live")
    start, limit = time.monotonic_ns(), deadline()
    publish(tmp_path / "drain-state.json", {"start_ns": start, "deadline_ns": limit})
    managers = {}
    for i, role in enumerate(("draft", "target-rank-0", "target-rank-1", "coordinator")):
        manager = DiagnosticLogs(tmp_path, "buffered-live", role)
        manager.pid = 100 + i  # Distinct mocked OS processes, not rank-derived device identities.
        manager.receipt_path = tmp_path / f"fixed-logging-{manager.pid}.json"
        manager.append(
            CheckpointJsonl(tmp_path / "verification-events.jsonl"),
            {"role": role},
            CheckpointJsonl.append,
        )
        managers[role] = manager
    (tmp_path / f"fixed-logging-{os.getpid()}.json").unlink()
    managers["draft"].finish(limit)  # The server/owner have completed their last append.
    calls = []

    def collective(callback, timeout):
        assert callback is fixed_logging.finish_worker and 0 < timeout <= 5
        calls.append("workers")
        result = []
        for rank in (0, 1):
            monkeypatch.setattr(fixed_logging, "_CURRENT", managers[f"target-rank-{rank}"])
            monkeypatch.setattr(
                fixed_logging, "current", lambda role=None: managers[role or "coordinator"]
            )
            result.append(callback(SimpleNamespace(rank=rank)))
        return result

    monkeypatch.setenv("SR_FIXED_POINT", str(tmp_path / "point.json"))
    fixed_logging.finalize_drain(SimpleNamespace(collective_rpc=collective), tmp_path, limit)
    publish(
        tmp_path / "drain-state.json",
        {"start_ns": start, "end_ns": time.monotonic_ns(), "deadline_ns": limit},
    )
    assert calls == ["workers"]
    assert len(fixed_logging.qualify(tmp_path, "buffered-live")["receipts"]) == 4
    managers["draft"].receipt_path.unlink()
    with pytest.raises(DataError, match="completion receipt"):
        fixed_logging.qualify(tmp_path, "buffered-live")


def test_real_supervisor_bounds_blocked_final_fsync_and_marks_incomplete(tmp_path):
    from specrhythm.phase4.process_lifecycle import run_owned_target

    script = """
import pathlib, signal, sys, time
from specrhythm.serving.fixed_logging import DiagnosticLogs
from specrhythm.phase4.transport import CheckpointJsonl
from specrhythm.serving.s2_pool import publish
p=pathlib.Path(sys.argv[1])
logs=DiagnosticLogs(p, "buffered-live", "coordinator", fsync=lambda fd: time.sleep(60))
logs.append(CheckpointJsonl(p / "proposal-events.jsonl"), {"sequence": 1}, CheckpointJsonl.append)
publish(p / "measurement-snapshot.json", {"committed_window_tokens": 64})
signal.signal(signal.SIGTERM, signal.SIG_IGN)
limit=time.monotonic_ns()+200_000_000
publish(p / "drain-state.json", {"deadline_ns": limit, "phase": "diagnostic_final_flush"})
logs.finish(limit)
"""
    rc, life = run_owned_target(
        [sys.executable, "-c", script, str(tmp_path)],
        target_log=tmp_path / "target.log",
        artifact_path=tmp_path / "process-lifecycle.json",
        timeout_seconds=10,
        phase_deadline_path=tmp_path / "drain-state.json",
        graceful_seconds=0.1,
        kill_seconds=1,
        poll_seconds=0.01,
    )
    assert rc == 124 and not life["run_valid"] and life["owned_cleanup_completed"]
    row = json.loads(next(tmp_path.glob("fixed-logging-*.json")).read_text())
    assert row["status"] == "RUNNING" and not row["counts_final"] and not row["integrity_complete"]
    assert read_json(tmp_path / "measurement-snapshot.json")["committed_window_tokens"] == 64


def test_real_capacity_zero_verification_finalizes_empty_buffers(owner, tmp_path, monkeypatch):
    from test_serving_fixed_settle import (
        test_capacity_runtime_zero_verification_uses_actual_empty_machine_shutdown,
    )

    logs, _ = install(tmp_path, monkeypatch)
    monkeypatch.setattr(fixed_logging, "wait_draft_receipt", lambda directory, deadline: {})
    test_capacity_runtime_zero_verification_uses_actual_empty_machine_shutdown(
        owner, tmp_path, monkeypatch
    )
    assert logs.closed and logs.produced == logs.written == 0
    assert read_json(tmp_path / "drain-state.json")["logging_finalization"]["coordinator"][
        "integrity_complete"
    ]


def test_actual_fork_cannot_share_parent_buffer_or_forge_child_complete(tmp_path):
    import subprocess

    script = """
import os, pathlib, sys, time
from specrhythm.phase4.transport import CheckpointJsonl
from specrhythm.serving.fixed_logging import install_checkpoint_logging, current
from specrhythm.serving.fixed_observe import Timers
p=pathlib.Path(sys.argv[1])
os.environ["SR_FIXED_POINT"]=str(p/"point.json")
os.environ["SR_FIXED_OBSERVATION"]="buffered-live"
parent=install_checkpoint_logging("coordinator", lambda row: None, Timers())
CheckpointJsonl(p/"proposal-events.jsonl").append({"from": "parent"})
child=os.fork()
if child==0:
    try:
        own=current("target-rank-0")
        CheckpointJsonl(p/"target-diagnostics.jsonl").append({"from": "child"})
        assert own.produced==1 and parent.produced==1
        assert own.pid==os.getpid() and own.pid!=parent.pid
        own.finish(time.monotonic_ns()+2_000_000_000)
        os._exit(0)
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(1)
_, status=os.waitpid(child,0)
assert os.waitstatus_to_exitcode(status)==0
assert parent.produced==1 and len(parent.pending)==1
parent.finish(time.monotonic_ns()+2_000_000_000)
assert len(CheckpointJsonl(p/"proposal-events.jsonl").read())==1
assert len(CheckpointJsonl(p/"target-diagnostics.jsonl").read())==1
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    rows = [read_json(p) for p in tmp_path.glob("fixed-logging-*.json")]
    assert len(rows) == 2 and len({r["pid"] for r in rows}) == 2
    assert all(r["produced_records"] == r["written_records"] == 1 for r in rows)
    assert all(r["integrity_complete"] for r in rows)
