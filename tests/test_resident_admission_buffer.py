"""Actual fixed/S2/resident scheduler → CheckpointJsonl → bounded logger, CPU stock stub."""

import importlib.util
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_phase4_dual_scheduler import Request
from test_serving_fixed_logging import deadline, install
from test_serving_s2 import s2_schedulers as _s2_schedulers

from specrhythm.phase4.resident_setup import (
    ADMISSION_EVENT_SCHEMA,
    validate_resident_admission_events,
)
from specrhythm.phase4.transport import CheckpointJsonl
from specrhythm.serving.common import read_json
from specrhythm.serving.fixed_logging import DiagnosticLogs
from specrhythm.serving.s2_pool import publish

s2_schedulers = _s2_schedulers


@pytest.mark.parametrize("serving", ["serial", "serial-eager"])
@pytest.mark.parametrize("observation", ["original-live", "buffered-live"])
def test_real_resident360_scheduler_decisions_and_all_records_preserved(
    s2_schedulers, tmp_path, monkeypatch, serving, observation
):
    module, control_path = s2_schedulers
    resident = sys.modules["specrhythm.phase4.resident_scheduler"]
    definitions = [
        SimpleNamespace(request_id=f"r{i}", prompt_token_ids=(i + 1, i + 10000))
        for i in range(360)
    ]
    monkeypatch.setattr(resident, "load_smoke_requests", lambda *a, **kw: definitions)
    for key, value in dict(
        SR_PHASE4_RESIDENT_SETUP="1",
        SR_PHASE4_RESIDENT_CONSUMER="serial",
        SR_PHASE4_RESIDENT_SETUP_READY=str(tmp_path / "unused-ready.json"),
        SR_PHASE4_DECODE_READY_MANIFEST=str(tmp_path / "unused-manifest.json"),
        SR_PHASE4_RESIDENT_ADMISSION_EVENTS=str(tmp_path / "admission-events.jsonl"),
        SR_PHASE4_RESIDENT_INITIAL_PROPOSAL_EVENTS=str(tmp_path / "initial-proposals.jsonl"),
        SR_PHASE4_WORKLOAD=str(tmp_path / "workload.jsonl"),
        SR_PHASE4_REQUEST_COUNT="360",
        SR_S2_MODE=serving,
        SR_FIXED_IDENTITY_MATCHING="bound-prefix",
        SR_FIXED_POINT=str(tmp_path / "point.json"),
    ).items():
        monkeypatch.setenv(key, value)
    packet = dict(
        barrier_ns=100,
        active_limit=16,
        max_requests_per_target_forward=16,
        diagnostic_phase="measurement",
        requests={
            d.request_id: {"state": "ACTIVE" if i < 16 else "QUEUED"}
            for i, d in enumerate(definitions)
        },
    )
    publish(control_path, packet)
    # Load the actual FixedSerialScheduler against the fixture's stock GPU substitute.
    name = "cpu_fixed_resident_scheduler"
    spec = importlib.util.spec_from_file_location(
        name, Path("src/specrhythm/serving/fixed_scheduler.py")
    )
    fixed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixed)
    scheduler = fixed.FixedSerialScheduler()
    assert isinstance(scheduler, module.S2SerialScheduler)
    scheduler.requests = {
        f"opaque-{i}": Request(f"opaque-{i}", d.prompt_token_ids)
        for i, d in enumerate(definitions)
    }
    for req in scheduler.requests.values():
        req.all_token_ids.append(11)
        req.num_computed_tokens, req.num_output_tokens = 3, 2
        req.spec_token_ids = [20, 21, 22, 23]
    scheduler.running = list(scheduler.requests.values())
    scheduler._resident_ready = {"measurement_start_ns": 100}
    scheduler.kv_cache_manager = SimpleNamespace(
        get_block_ids=lambda rid: [[int(rid.removeprefix("opaque-"))]]
    )
    scheduler._bind_requests()
    scheduler.freeze_pool()
    syncs = []
    monkeypatch.setattr(os, "fsync", lambda _: syncs.append(1))
    logs, captured = install(tmp_path, monkeypatch)
    logs.mode = observation
    # 58*360 > old causal cap. This also validates real byte/checksum flush batches;
    # it is a structural count assertion, not a timing threshold or GPU overlap test.
    for _ in range(58):
        output = scheduler.schedule()
        assert set(output.num_scheduled_tokens) == {f"opaque-{i}" for i in range(16)}
        assert all(len(v) == 4 for v in output.scheduled_spec_decode_tokens.values())
    before = len(syncs)
    if observation == "buffered-live":
        assert before == (58 * 360) // logs.max_records
        assert logs.pending
    else:
        assert before == 58 * 360 and not logs.pending
    result = logs.finish(deadline())
    rows = scheduler._resident_events.read()
    assert len(rows) == len(captured) == 58 * 360
    assert not validate_resident_admission_events(rows, consumer="serial")
    assert all(r["schema_version"] == ADMISSION_EVENT_SCHEMA for r in rows)
    assert sum(r["scheduled"] for r in rows) == 58 * 16
    assert result["integrity_complete"] and result["pending_records"] == 0
    assert result["produced_records"] == result["written_records"] == len(rows)
    assert len(syncs) == (
        math.ceil(len(rows) / 256) if observation == "buffered-live" else len(rows)
    )
    assert result["peak_buffer_records"] <= 256 and result["peak_buffer_bytes"] <= 1024 * 1024


@pytest.mark.parametrize("ending", ["normal", "controlled_stop", "write_failure"])
def test_admission_final_boundary_and_failed_write_never_claim_complete(
    tmp_path, monkeypatch, ending
):
    logs, captured = install(tmp_path, monkeypatch)
    path = tmp_path / "admission-events.jsonl"
    log = CheckpointJsonl(path)
    log.append(dict(schema_version=ADMISSION_EVENT_SCHEMA, consumer="serial", cycle_id=0))
    assert logs.pending and captured  # Immediate validation/capture, delayed persistence.
    if ending == "controlled_stop":
        publish(tmp_path / "stop-request.json", {"requested": True})
        assert read_json(tmp_path / "stop-request.json")["requested"]
    if ending == "write_failure":
        logs.fsync = lambda _: (_ for _ in ()).throw(OSError("admission disk failure"))
        with pytest.raises(OSError, match="admission disk failure"):
            logs.finish(deadline())
        receipt = read_json(logs.receipt_path)
        assert receipt["status"] == "FAILED" and not receipt["integrity_complete"]
        assert receipt["produced_records"] > receipt["written_records"]
    else:
        receipt = logs.finish(deadline())
        assert receipt["integrity_complete"] and len(log.read()) == 1


def test_unrecognized_admission_records_and_target_only_stay_synchronous(tmp_path):
    logs = DiagnosticLogs(tmp_path, "buffered-live")
    log = CheckpointJsonl(tmp_path / "admission-events.jsonl")
    for row in (
        {"released": True},
        {"schema_version": ADMISSION_EVENT_SCHEMA, "consumer": "target-only"},
    ):
        logs.append(log, row, CheckpointJsonl.append)
    assert len(log.read()) == 2 and not logs.pending


def test_mixed_same_file_preserves_order_and_never_claims_a_partial_digest(tmp_path):
    logs = DiagnosticLogs(tmp_path, "buffered-live")
    log = CheckpointJsonl(tmp_path / "admission-events.jsonl")
    known = dict(schema_version=ADMISSION_EVENT_SCHEMA, consumer="serial")
    for row in (dict(known, order=0), {"order": 1, "released": True}, dict(known, order=2)):
        logs.append(log, row, CheckpointJsonl.append)
    logs.before_read(log)
    assert [row["order"] for row in log.read()] == [0, 1, 2]
    receipt = logs.finish(deadline())
    assert receipt["integrity_complete"]
    assert receipt["flush_reasons"]["unbuffered-same-file"] == 1
    stream = receipt["streams"][log.path.name]
    assert stream["written_bytes_sha256"] is None and "native framing" in stream["digest_scope"]
