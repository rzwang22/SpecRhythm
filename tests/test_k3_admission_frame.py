"""Actual scheduler -> installed logger -> checksummed rows; no timing thresholds."""

import json
import os
from dataclasses import FrozenInstanceError

import pytest
from test_k3_resident_schedule import fixed_schedulers as _fixed
from test_k3_resident_schedule import prepare
from test_k3_resident_schedule import s2_schedulers as _s2
from test_k3_resident_schedule import target_pool as _pool
from test_serving_fixed_logging import deadline, install

from specrhythm.phase4.admission_record import PreparedAdmission
from specrhythm.phase4.resident_setup import ADMISSION_EVENT_SCHEMA
from specrhythm.phase4.transport import CheckpointJsonl
from specrhythm.serving import fixed_logging
from specrhythm.serving.k3 import MODES

fixed_schedulers, s2_schedulers, target_pool = _fixed, _s2, _pool


@pytest.mark.parametrize("mode", MODES)
def test_real_schedule_encodes_every_logical_record_once(target_pool, monkeypatch, mode):
    s, packet, path = target_pool
    monkeypatch.setenv("SR_S2_MODE", mode)
    prepare(s, packet, path)
    logs, captured = install(s._resident_events.path.parent, monkeypatch)
    original, encodings, rows = json.dumps, [], []

    def encode(value, *args, **kwargs):
        if isinstance(value, dict) and value.get("schema_version") == ADMISSION_EVENT_SCHEMA:
            encodings.append(dict(value))
        return original(value, *args, **kwargs)

    monkeypatch.setattr(json, "dumps", encode)
    append = s._resident_events.append

    def collect(value):
        assert type(value) is PreparedAdmission
        rows.append(dict(value))
        append(value)

    monkeypatch.setattr(s._resident_events, "append", collect)
    before = s.s2_pool.checks
    s.schedule()
    assert s.s2_pool.checks == before + 2
    assert len(rows) == len(encodings) == 360
    assert rows == encodings
    logs.finish(deadline())
    # Native read verifies checksum and exact logical rows, no sidecar decompressor.
    actual = s._resident_events.read()[-360:]
    assert [{k: v for k, v in r.items() if k != "record_sha256"} for r in actual] == rows


@pytest.mark.parametrize("observation", ["original-live", "buffered-live"])
@pytest.mark.parametrize("stop", [False, True])
def test_prepared_bytes_fsync_order_and_end_or_stop_identical(tmp_path, monkeypatch,
                                                           observation, stop):
    monkeypatch.setattr(os, "fsync", lambda fd: None)
    results = []
    common = dict(schema_version=ADMISSION_EVENT_SCHEMA, consumer="serial", cycle_id=1)
    for prepared in (False, True):
        directory = tmp_path / str(prepared)
        directory.mkdir()
        log = CheckpointJsonl(directory / "admission-events.jsonl")
        logs = fixed_logging.DiagnosticLogs(directory, observation, max_records=2)
        rows = []
        for i in range(7):
            dynamic = dict(request_id='汉字,"request_id":\\' + str(i), timestamp_ns=i,
                           reason="ready", scheduled=i % 2 == 0, num_output_tokens=None)
            row = PreparedAdmission(common, dynamic) if prepared else {**common, **dynamic}
            rows.append(dict(row))
            logs.append(log, row, CheckpointJsonl.append)
        # Same finish path is used for bounded operator stop and natural completion.
        if stop:
            logs.before_read(log)
        receipt = logs.finish(deadline())
        assert receipt["integrity_complete"]
        assert logs.finish(deadline()) == receipt
        assert [{k: v for k, v in r.items() if k != "record_sha256"} for r in log.read()] == rows
        results.append((log.path.read_bytes(), receipt["fsync_count"],
                        receipt["flush_reasons"], receipt["written_records"]))
    assert results[0] == results[1]


def test_prepared_record_owns_immutable_scalars_and_errors_remain_sticky(tmp_path):
    common = dict(schema_version=ADMISSION_EVENT_SCHEMA, consumer="serial")
    dynamic = dict(request_id="r", reason="ready")
    row = PreparedAdmission(common, dynamic)
    common["consumer"], dynamic["reason"] = "changed", "changed"
    assert row["consumer"] == "serial" and row["reason"] == "ready"
    with pytest.raises(TypeError):
        row._values["request_id"] = "x"
    with pytest.raises(FrozenInstanceError):
        row.line = b"changed"
    with pytest.raises(ValueError, match="immutable"):
        PreparedAdmission(common, {"request_id": "r", "mutable": []})
    logs = fixed_logging.DiagnosticLogs(tmp_path, "buffered-live")
    logs.append(CheckpointJsonl(tmp_path / "admission-events.jsonl"), row, CheckpointJsonl.append)
    logs.fsync = lambda fd: (_ for _ in ()).throw(OSError("original disk error"))
    with pytest.raises(OSError, match="original disk error"):
        logs.finish(deadline())
    receipt = json.loads(logs.receipt_path.read_text())
    assert receipt["status"] == "FAILED" and not receipt["integrity_complete"]
