"""Production writer/adapter/finalizer, no GPU or timing-threshold assertions."""

import hashlib
import json
import os
import time
from types import SimpleNamespace

import pytest
from test_fsync_attribution import proposer

from specrhythm import diagnostic_report
from specrhythm.continuation.trace import TRACE
from specrhythm.phase4.transport import CheckpointJsonl
from specrhythm.serving import fixed_logging, fixed_observe
from specrhythm.serving.common import DataError
from specrhythm.serving.deferred_logging import DeferredLogs
from specrhythm.serving.ping_prepost_proposer import PingPrePostProposer
from specrhythm.serving.s2_pool import publish


@pytest.fixture
def installed(tmp_path, monkeypatch):
    monkeypatch.setenv("SR_FIXED_POINT", str(tmp_path / "point.json"))
    monkeypatch.setenv("SR_FIXED_OBSERVATION", "deferred-window")
    monkeypatch.setattr(fixed_logging, "_CURRENT", None)
    monkeypatch.setattr(diagnostic_report, "_SINK", None)
    monkeypatch.setattr(CheckpointJsonl, "append", CheckpointJsonl.append)
    monkeypatch.setattr(CheckpointJsonl, "read", CheckpointJsonl.read)
    monkeypatch.setattr(TRACE, "phase", "measurement")
    timers = fixed_observe.Timers()
    monkeypatch.setattr(fixed_observe, "TIMERS", timers)
    native = os.fsync
    monkeypatch.setattr(os, "fsync", native)
    fixed_observe.wrap(os, "fsync", "log_fsync")
    captured = []
    logs = fixed_logging.install_checkpoint_logging("target-rank-0", captured.append, timers)
    return logs, captured, timers


@pytest.mark.parametrize("mode", ["serial-k3", "serial-eager-k3",
                                   "pingpong-k3", "pingpong-eager-k3"])
def test_actual_proposer_buffer_control_and_final_worker_receipt(installed, tmp_path,
                                                               monkeypatch, mode):
    logs, captured, timers = installed
    monkeypatch.setenv("SR_S2_MODE", mode)
    p = proposer(PingPrePostProposer, tmp_path / "plugin-report.json")
    p.protocol_metadata = {"prepost_protocol": "specrhythm.uniform-k3.v1"}
    count = []
    build = p._build_report
    p._build_report = lambda: (count.append(1), build())[1]
    log = CheckpointJsonl(tmp_path / "round-events.jsonl")
    for i in range(600):
        p.hooks_seen["on_target_sampled"] = i
        log.append({"sequence": i})
        p._write_report()
    assert not count and not log.path.exists() and not p.report_path.exists()
    assert len(captured) == len(log.read()) == 600  # Real read-your-writes, no flush.
    assert not [r for r in timers.rows if r.get("category") == "log_fsync" and r.get("log_name") in
                (log.path.name, p.report_path.name)]
    publish(tmp_path / "s2-control.json", {"live": True})
    assert json.loads((tmp_path / "s2-control.json").read_text()) == {"live": True}
    start = time.monotonic_ns()
    deadline = start + 5_000_000_000
    publish(tmp_path / "drain-state.json", {"deadline_ns": deadline})
    result = fixed_logging.finish_worker(SimpleNamespace(rank=0))
    assert count == [1] and result["integrity_complete"]
    assert result["produced_records"] == result["written_records"] == 600
    assert [r["sequence"] for r in log.read()] == list(range(600))
    assert result["streams"][log.path.name]["written_bytes_sha256"] == hashlib.sha256(
        log.path.read_bytes()).hexdigest()
    report = result["reports"][p.report_path.name]
    assert report["deadline_ns"] == deadline and report["end_ns"] <= deadline
    assert report["sha256"] == hashlib.sha256(p.report_path.read_bytes()).hexdigest()
    assert json.loads(p.report_path.read_bytes())["hook_counts"]["on_target_sampled"] == 599
    assert result["phase_costs"]["measurement"]["write_records"] == 0
    assert result["phase_costs"]["drain"]["write_records"] == 600
    assert fixed_logging.finish_worker(SimpleNamespace(rank=0)) == result


@pytest.mark.parametrize("kind", ["records", "bytes"])
def test_overflow_fails_without_hot_path_capacity_flush(tmp_path, kind):
    logs = DeferredLogs(tmp_path, max_records=1 if kind == "records" else 100,
                        max_bytes=10000 if kind == "records" else 120)
    log = CheckpointJsonl(tmp_path / "round-events.jsonl")
    logs.append(log, {"n": 1}, CheckpointJsonl.append)
    with pytest.raises(DataError, match="buffer exhausted"):
        logs.append(log, {"n": 2}, CheckpointJsonl.append)
    result = json.loads(logs.receipt_path.read_text())
    assert result["status"] == "FAILED" and not result["integrity_complete"]
    assert result["produced_records"] == 2 and result["written_records"] == 0
    assert result["peak_buffer_records"] <= result["max_buffer_records"]


@pytest.mark.parametrize("failure", ["fsync", "build", "expired", "build_expiry"])
def test_final_failure_never_publishes_complete(installed, tmp_path, monkeypatch, failure):
    logs, _, _ = installed
    CheckpointJsonl(tmp_path / "round-events.jsonl").append({"n": 1})
    limit = time.monotonic_ns() + 5_000_000_000
    def build():
        if failure == "build":
            raise RuntimeError("report construction failed")
        if failure == "build_expiry":
            monkeypatch.setattr(time, "monotonic_ns", lambda: limit + 1)
        return {"ok": True}
    logs.defer_report(tmp_path / "plugin-report.json", build)
    if failure == "fsync":
        logs.fsync = lambda _: (_ for _ in ()).throw(OSError("write failed"))
    if failure == "expired":
        limit = time.monotonic_ns() - 1
    with pytest.raises((OSError, RuntimeError, ValueError)):
        logs.finish(limit)
    assert not logs.snapshot()["integrity_complete"]
    assert not (tmp_path / "plugin-report.json").exists()


def test_buffer_bytes_immutable_and_unaudited_admission_stays_live(installed, tmp_path):
    logs, _, _ = installed
    log = CheckpointJsonl(tmp_path / "round-events.jsonl")
    row = {"nested": [1]}
    log.append(row)
    row["nested"].append(2)
    assert log.read()[0]["nested"] == [1]
    control = CheckpointJsonl(tmp_path / "admission-events.jsonl")
    control.append({"control": "not resident audit schema"})
    assert control.path.exists() and control.read()[0]["control"]
    assert logs.pending
