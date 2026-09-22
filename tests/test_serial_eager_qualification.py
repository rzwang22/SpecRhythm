"""The actual proposer report writer publishes qualification in its first write."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from specrhythm.phase4 import vllm_remote
from specrhythm.phase4.vllm_remote import RemoteDraftProposer
from specrhythm.serving.eager_proposer import EagerSerialProposer
from specrhythm.serving.s2_proposer import S2SerialProposer


def report_proposer(cls, path, *, rank=0):
    # Substitute only construction of a GPU host; use the unchanged real report
    # writer, subclass dispatch and atomic file publication.
    proposer = object.__new__(cls)
    proposer.tp_rank = rank
    proposer.tp_world_size = 2
    proposer.report_path = path
    proposer.hooks_seen = {"verify_start": 0, "verify_end": 0}
    proposer.requests = {"request": SimpleNamespace(
        generated_token_ids=(10,), bootstrap_target_tokens=1, tail_target_tokens=0,
        next_round_id=0, finished=False,
    )}
    proposer.round_records = []
    proposer.resident_mode = True
    proposer.resident_setup_complete = False
    proposer.resident_setup_tracker = None
    proposer.measurement_start_ns = None
    proposer.performance_measurement_start_ns = None
    return proposer


@pytest.mark.parametrize("cls", [RemoteDraftProposer, S2SerialProposer, EagerSerialProposer])
def test_actual_report_first_write_keeps_baseline_and_marks_eager_pending(
    tmp_path, monkeypatch, cls,
):
    path = tmp_path / "proposer-report.json"
    proposer = report_proposer(cls, path)
    writes = []
    atomic_write = vllm_remote.atomic_write_json
    expected_correctness = cls is not EagerSerialProposer

    def observed_write(target, value):
        # Check before publication, ruling out an unsafe write-then-rewrite fix.
        assert value["gpu_correctness_result"] is expected_correctness
        assert value["gpu_performance_result"] is False
        if cls is EagerSerialProposer:
            assert value["rolling_eager_qualification"] == "PENDING"
        else:
            assert "rolling_eager_qualification" not in value
        writes.append(value)
        atomic_write(target, value)

    monkeypatch.setattr(vllm_remote, "atomic_write_json", observed_write)
    monkeypatch.setattr(vllm_remote, "performance_requested", lambda: False)
    proposer._write_report()
    assert len(writes) == 1
    report = json.loads(path.read_text())
    assert report == writes[0]
    assert report["schema_version"] == "specrhythm.phase4-remote-proposer-report.v1"
    assert report["request_count"] == 1 and report["round_count"] == 0
    assert report["requests"]["request"]["generated_token_ids"] == [10]
    assert report["target_rank0_only_transport"] is True
    proposer.resident_setup_complete = True
    proposer._write_report()
    assert len(writes) == 2
    assert json.loads(path.read_text())["gpu_correctness_result"] is expected_correctness


def test_nonzero_target_rank_does_not_publish_eager_qualification(tmp_path):
    path = tmp_path / "rank-one-report.json"
    report_proposer(EagerSerialProposer, path, rank=1)._write_report()
    assert not path.exists()
