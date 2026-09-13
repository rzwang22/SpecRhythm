"""Real writers -> installed observer -> raw host report -> strict qualification (CPU)."""

import json
import os
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from test_execution_evidence import existing as _existing
from test_execution_evidence import inputs as _inputs
from test_execution_evidence import qualified

from specrhythm import io_context
from specrhythm.phase4 import manifest, vllm_remote
from specrhythm.phase4.transport import CheckpointJsonl
from specrhythm.serving import fixed_logging, fixed_observe
from specrhythm.serving.eager_proposer import EagerSerialProposer
from specrhythm.serving.execution_evidence import execution_path, main, qualify
from specrhythm.serving.execution_failure import summarize
from specrhythm.serving.s2_proposer import S2SerialProposer

existing = _existing
inputs = _inputs


@pytest.fixture
def observed(monkeypatch):
    timers, calls = fixed_observe.Timers(), []
    original = os.fsync

    def sync(fd):
        calls.append(fd)
        return original(fd)

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(fixed_observe, "TIMERS", timers)
    fixed_observe.wrap(os, "fsync", "log_fsync")
    return timers, calls


def proposer(cls, path):
    # Only GPU construction is omitted. Both production classes inherit this writer;
    # the imported function alias predates observation installation.
    p = cls.__new__(cls)
    p.tp_rank, p.tp_world_size, p.report_path = 0, 2, path
    p.hooks_seen, p.round_records = {"on_target_sampled": 1}, [{}]
    p.requests = {"A": SimpleNamespace(generated_token_ids=(1, 2), bootstrap_target_tokens=1,
                                       tail_target_tokens=0, next_round_id=1, finished=False)}
    p.resident_mode, p.resident_setup_complete = True, True
    p.resident_setup_tracker = SimpleNamespace(observations=["A"])
    p.measurement_start_ns, p.performance_measurement_start_ns = 1, 2
    return p


def report_from_capture(inputs, timers):
    # Other GPU/trace evidence is the existing qualified CPU fixture. File attribution
    # is never fabricated: collect and serialize Timers.report(), then run the actual
    # raw-host aggregation and CLI qualifier used after GPU runs.
    report = qualified(inputs)
    host = json.loads(json.dumps(timers.report()))
    end = max(r["end_ns"] for r in host["intervals"]) + 1
    runtime = dict(host={}, target_devices=[dict(host=host, device=dict(
        identity=dict(global_rank=0)))])
    captured = execution_path(runtime, {}, 0, end)
    report["execution_path"]["fsync_by_file"] = captured["fsync_by_file"]
    return report


@pytest.mark.parametrize("cls", [S2SerialProposer, EagerSerialProposer])
def test_real_proposer_alias_collect_report_qualify_and_reproduce_old_gap(
    tmp_path, monkeypatch, observed, inputs, cls, capsys
):
    timers, calls = observed
    path = tmp_path / "remote-proposer-report.json"
    p = proposer(cls, path)
    assert vllm_remote.atomic_write_json is manifest.atomic_write_json
    assert cls._write_report is vllm_remote.RemoteDraftProposer._write_report
    with monkeypatch.context() as old:
        old.setattr(manifest, "file_context", lambda *a, **k: nullcontext())
        p._write_report()  # Exactly the old uncovered sync through the real imported alias.
    old_bytes = path.read_bytes()
    bad = report_from_capture(inputs, timers)
    assert bad["execution_path"]["fsync_by_file"][0]["log_name"] == "MISSING_FILENAME"
    source, status = tmp_path / "old-report.json", tmp_path / "old-status.json"
    source.write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="fsync file attribution incomplete"):
        main(["--qualify", str(source), "--output", str(status)])
    failure = json.loads(status.read_text())
    assert failure["errors"] == ["fsync file attribution incomplete"]
    assert failure["failure_layer"] == "diagnostic_evidence"
    assert failure["original_qualification"]["execution_status"] == "PASS"
    assert failure["missing_fsync_attribution"][0]["role"] == "target-0"
    assert failure["missing_fsync_attribution"][0]["count"] == 1
    assert '"union_ms"' in capsys.readouterr().out
    timers.rows.clear()
    p._write_report()
    assert path.read_bytes() == old_bytes and len(calls) == 2
    row = timers.rows[0]
    assert row["log_name"] == path.name and row["log_path"] == str(path)
    assert row["physical_path"] == str(path.with_name(f".{path.name}.{os.getpid()}.tmp"))
    assert row["write_kind"] == "atomic_json"
    good = report_from_capture(inputs, timers)
    source, status = tmp_path / "new-report.json", tmp_path / "new-status.json"
    source.write_text(json.dumps(good))
    main(["--qualify", str(source), "--output", str(status)])
    assert json.loads(status.read_text())["diagnostic_integrity"] == "COMPLETE"
    # Even after fixing all known writers, an unknown raw fsync still fails.
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
    assert qualify(report_from_capture(inputs, timers))["diagnostic_integrity"] == "FAILED"


@pytest.mark.parametrize("mode", ["original-live", "buffered-live"])
def test_installed_jsonl_adapter_repeated_install_and_flush_attribution(
    tmp_path, monkeypatch, observed, mode
):
    timers, calls = observed
    monkeypatch.setenv("SR_FIXED_POINT", str(tmp_path / "point.json"))
    monkeypatch.setenv("SR_FIXED_OBSERVATION", mode)
    monkeypatch.setattr(fixed_logging, "_CURRENT", None)
    # Undo class installation after the test, as the real installation is process-global.
    monkeypatch.setattr(CheckpointJsonl, "append", CheckpointJsonl.append)
    monkeypatch.setattr(CheckpointJsonl, "read", CheckpointJsonl.read)
    captured = []
    logs = fixed_logging.install_checkpoint_logging("target", captured.append, timers)
    fixed_logging.install_checkpoint_logging("target", captured.append, timers)
    wrapper = os.fsync
    assert fixed_observe.wrap(os, "fsync", "log_fsync") is wrapper
    log = CheckpointJsonl(tmp_path / "proposal-events.jsonl")
    log.append({"n": 1})
    log.append({"n": 2})
    assert captured == [{"n": 1}, {"n": 2}]
    assert len(log.read()) == 2  # Production read-your-writes flush.
    rows = [r for r in timers.rows if r["category"] == "log_fsync"]
    assert len(rows) == len(calls) == (1 if mode == "buffered-live" else 2)
    assert all(r["log_path"] == r["physical_path"] == str(log.path) for r in rows)
    assert {r["write_kind"] for r in rows} == {
        "buffered_checkpoint_jsonl" if mode == "buffered-live" else "checkpoint_jsonl"}
    assert logs.snapshot()["produced_records"] == logs.snapshot()["written_records"] == 2


def test_nested_exception_and_two_threads_do_not_share_context(tmp_path, observed):
    timers, calls = observed
    outer, inner = tmp_path / "outer.json", tmp_path / "inner.json"
    with io_context.file_context(outer, write_kind="outer"):
        previous = io_context.sync_attribution()
        with pytest.raises(RuntimeError):
            with io_context.file_context(inner, write_kind="inner"):
                assert io_context.sync_attribution()["log_path"] == str(inner)
                raise RuntimeError("restore")
        assert io_context.sync_attribution() == previous
    assert io_context.sync_attribution()["log_name"] is None
    ready, go = threading.Event(), threading.Event()
    errors = []

    def other_thread():
        try:
            with io_context.file_context(inner, write_kind="thread"):
                ready.set()
                assert go.wait(5), "test coordination did not release worker"
                # Reentrant actual writer restores this thread's outer context.
                manifest.atomic_write_json(inner, {"inner": True})
                assert io_context.sync_attribution()["write_kind"] == "thread"
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=other_thread)
    worker.start()
    try:
        assert ready.wait(5), "test worker did not enter context"
        assert io_context.sync_attribution()["log_name"] is None
        manifest.atomic_write_json(outer, {"outer": True})
    finally:
        go.set()
        worker.join(5)
    assert not worker.is_alive() and not errors
    assert len(calls) == 2
    assert {r["log_path"] for r in timers.rows} == {str(inner), str(outer)}
    assert io_context.sync_attribution()["log_name"] is None


@pytest.mark.parametrize("failure", ["json", "fsync", "replace"])
@pytest.mark.parametrize("cls", [S2SerialProposer, EagerSerialProposer])
def test_failed_report_preserves_published_snapshot_and_propagates(
    tmp_path, monkeypatch, observed, failure, cls
):
    timers, calls = observed
    path = tmp_path / "report.json"
    path.write_text('{"previous":true}\n')
    old_bytes = path.read_bytes()
    p = proposer(cls, path)
    replacements = []
    original_replace = os.replace

    def replace(a, b):
        replacements.append((a, b))
        if failure == "replace":
            raise OSError("replace failed")
        return original_replace(a, b)

    monkeypatch.setattr(os, "replace", replace)
    if failure == "json":
        monkeypatch.setattr(manifest.json, "dump", lambda *a, **k: (_ for _ in ()).throw(
            OSError("write failed")))
    if failure == "fsync":
        monkeypatch.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("fsync failed")))
        fixed_observe.wrap(os, "fsync", "log_fsync")
    with pytest.raises(OSError, match="failed"):
        p._write_report()
    assert path.read_bytes() == old_bytes
    assert len(replacements) == (1 if failure == "replace" else 0)
    assert io_context.sync_attribution()["log_name"] is None
    if failure == "fsync":
        assert timers.rows[0]["log_path"] == str(path)


def test_same_bytes_flush_sync_close_replace_order_with_and_without_context(tmp_path, monkeypatch):
    from pathlib import Path

    path = tmp_path / "snapshot.json"
    original_open, original_replace = Path.open, os.replace
    original_sync = os.fsync
    events = []

    class Handle:
        def __init__(self, native):
            self.native = native

        def __enter__(self):
            self.native.__enter__()
            return self

        def __exit__(self, *args):
            result = self.native.__exit__(*args)
            events.append("close")
            return result

        def write(self, value):
            events.append("write")
            return self.native.write(value)

        def flush(self):
            events.append("flush")
            return self.native.flush()

        def fileno(self):
            return self.native.fileno()

    def open_file(p, *a, **kw):
        native = original_open(p, *a, **kw)
        return Handle(native) if p.name.startswith(".snapshot.json.") and a == ("w",) else native

    def sync(fd):
        events.append("fsync")
        # Flush must have made the complete bytes visible before sync and replace.
        temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        assert json.loads(temp.read_text()) == {"n": [1, 2]}
        return original_sync(fd)

    def replace(a, b):
        events.append("replace")
        return original_replace(a, b)

    monkeypatch.setattr(Path, "open", open_file)
    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "replace", replace)
    with monkeypatch.context() as old:
        old.setattr(manifest, "file_context", lambda *a, **k: nullcontext())
        manifest.atomic_write_json(path, {"n": [1, 2]})
    baseline, baseline_events = path.read_bytes(), list(events)
    events.clear()
    manifest.atomic_write_json(path, {"n": [1, 2]})
    assert path.read_bytes() == baseline and events == baseline_events
    assert events[-4:] == ["flush", "fsync", "close", "replace"]
    assert events.count("fsync") == events.count("replace") == 1


def test_other_phase4_sync_writers_are_attributed(tmp_path, observed):
    from specrhythm.phase4.batched_draft_service import write_immutable_report
    from specrhythm.phase4.process_lifecycle import _atomic_json
    from specrhythm.phase4.reference import _exclusive_copy, _exclusive_freeze

    timers, calls = observed
    _atomic_json(tmp_path / "lifecycle.json", {"n": 1})
    write_immutable_report(tmp_path / "backend.json", {"n": 2})
    _exclusive_freeze(tmp_path / "reference.json", {"n": 3})
    _exclusive_copy(tmp_path / "reference.json", tmp_path / "copy.json")
    assert len(calls) == 4
    assert [r["log_name"] for r in timers.rows] == [
        "lifecycle.json", "backend.json", "reference.json", "copy.json"]
    assert timers.rows[0]["physical_path"] != timers.rows[0]["log_path"]
    assert timers.rows[-1]["write_kind"] == "immutable_copy"


def test_first_failure_preserves_run_and_missing_detail_even_for_old_status(tmp_path, inputs):
    root = tmp_path / "new-root"
    root.mkdir()
    status = qualify(qualified(inputs))
    status.update(diagnostic_integrity="FAILED", failure_layer="diagnostic_evidence",
                  errors=["fsync file attribution incomplete"])
    status.pop("missing_fsync_attribution")
    (tmp_path / "new-root-evidence-status.json").write_text(json.dumps(status))
    missing = dict(role="target-0", log_name="MISSING_FILENAME", count=45, union_ms=123.609629)
    (tmp_path / "new-root-audit-report.json").write_text(json.dumps(
        dict(execution_path=dict(fsync_by_file=[missing]))))
    report = summarize(root, 23, "diagnostic_evidence")
    assert report["first_exit_code"] == 23
    assert report["failure_layer"] == "diagnostic_evidence"
    assert report["original_qualification"]["cleanup_status"] == "PASS"
    assert report["missing_fsync_attribution"] == [missing]
    assert report["evidence_status"] == str(tmp_path / "new-root-evidence-status.json")
    assert len(report["evidence_packages"]) == 2
