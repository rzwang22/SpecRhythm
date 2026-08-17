from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import pytest

from specrhythm.phase4.decode_ready import (
    DecodeReadyProvenance,
    ResidentSetupObservation,
    ResidentWarmStartProvider,
)
from specrhythm.phase4.manifest import atomic_write_json
from specrhythm.phase4.resident_setup import (
    IncrementalResidentSetup,
    build_setup_ready,
    load_setup_ready,
    resident_admission_decision,
    validate_resident_admission_events,
)
from specrhythm.phase4.serial import PROTOCOL_VERSION, Proposal


def _observation(request_id: str, offset: int) -> ResidentSetupObservation:
    prompt = (offset, offset + 1)
    return ResidentSetupObservation(
        request_id=request_id,
        internal_target_request_id=f"opaque-{request_id}",
        prompt_token_ids=prompt,
        bootstrap_token_id=offset + 2,
        target_materialized_kv_token_count=2,
        target_num_computed_tokens=2,
        draft_materialized_kv_token_count=3,
        bootstrap_ready_ns=10 + offset,
        draft_initialization_complete_ns=11 + offset,
    )


def _provenance() -> DecodeReadyProvenance:
    return DecodeReadyProvenance(
        specrhythm_git_commit="1" * 40,
        vllm_version="0.25.1",
        vllm_commit="752a3a504485790a2e8491cacbb35c137339ad34",
        vllm_patch_stack_sha256=("a" * 64, "b" * 64, "c" * 64),
        target_model_path="/target",
        target_model_revision=None,
        draft_model_path="/draft",
        draft_model_revision=None,
        tokenizer_revision=None,
        workload_sha256="d" * 64,
        sampling_configuration={"temperature": 0.0},
        batch_invariant_configuration={"requested": True},
        target_physical_gpu_ids=(1, 2),
        draft_physical_gpu_ids=(0,),
        target_tensor_parallel_size=2,
        draft_tensor_parallel_size=1,
    )


def _manifest(observations: tuple[ResidentSetupObservation, ...]):
    return ResidentWarmStartProvider().prepare(
        observations,
        _provenance(),
        setup_start_ns=5,
        setup_complete_ns=30,
        global_barrier_ns=31,
        measurement_start_ns=32,
    )


def _proposal(request_id: str, parent_len: int, parent_hash: str, start: int = 33):
    return Proposal(
        protocol_version=PROTOCOL_VERSION,
        request_id=request_id,
        round_id=0,
        parent_prefix_len=parent_len,
        parent_prefix_hash=parent_hash,
        proposal_token_ids=(90, 91),
        proposal_eos=False,
        draft_start_ns=start,
        draft_end_ns=start + 1,
        transport_payload_bytes=10,
        model_provenance={"model": "draft"},
        runtime_provenance={"device": 0},
    )


def test_incremental_l2_records_a_then_b_and_freezes_a():
    tracker = IncrementalResidentSetup(("A", "B"), setup_start_ns=5)
    assert tracker.record(_observation("A", 1)) is True
    assert tracker.complete is False
    assert tracker.observed_request_ids == ("A",)
    assert resident_admission_decision(
        num_output_tokens=1,
        global_decode_ready=False,
        consumer="target-only",
        has_initial_proposal=False,
    ) == (False, "bootstrap-ready-awaiting-global-boundary")
    assert tracker.record(_observation("B", 4)) is True
    assert tracker.complete is True
    assert tracker.observed_request_ids == ("A", "B")
    assert tracker.completion_transition_count == 1


def test_incremental_l5_mixed_subsets_have_one_completion_transition():
    request_ids = ("A", "B", "C", "D", "E")
    tracker = IncrementalResidentSetup(request_ids, setup_start_ns=5)
    offsets = {request_id: index * 3 + 1 for index, request_id in enumerate(request_ids)}
    for subset in (("A",), ("B", "C"), ("D",), ("E",)):
        for request_id in subset:
            tracker.record(_observation(request_id, offsets[request_id]))
    assert tracker.complete is True
    assert tracker.observed_request_ids == request_ids
    assert tracker.completion_transition_count == 1
    assert tracker.record(tracker.observations[-1]) is False
    assert tracker.completion_transition_count == 1


def test_duplicate_or_stale_bootstrap_fails_closed():
    tracker = IncrementalResidentSetup(("A",), setup_start_ns=5)
    original = _observation("A", 1)
    tracker.record(original)
    assert tracker.record(original) is False
    with pytest.raises(RuntimeError, match="observation changed"):
        tracker.record(replace(original, bootstrap_token_id=999))
    with pytest.raises(RuntimeError, match="unexpected resident setup request"):
        tracker.record(replace(original, request_id="B"))


def test_serial_proposal_before_measurement_start_fails(tmp_path):
    observations = (_observation("A", 1), _observation("B", 4))
    manifest = _manifest(observations)
    manifest_path = tmp_path / "manifest.json"
    atomic_write_json(manifest_path, manifest.to_dict())
    proposals = tuple(
        _proposal(
            row.request_id,
            row.logical_committed_prefix_count,
            row.logical_committed_prefix_sha256,
            start=31,
        )
        for row in manifest.requests
    )
    with pytest.raises(ValueError, match="predates measurement_start"):
        build_setup_ready(
            manifest,
            consumer="serial",
            manifest_path=manifest_path,
            initial_proposals=proposals,
            ready_published_ns=40,
        )


def test_target_and_serial_share_manifest_state_and_auditable_ready(tmp_path):
    observations = (_observation("A", 1), _observation("B", 4))
    target_manifest = _manifest(observations)
    serial_manifest = _manifest(observations)
    assert target_manifest.requests == serial_manifest.requests
    manifest_path = tmp_path / "manifest.json"
    ready_path = tmp_path / "ready.json"
    atomic_write_json(manifest_path, serial_manifest.to_dict())
    proposals = tuple(
        _proposal(
            row.request_id,
            row.logical_committed_prefix_count,
            row.logical_committed_prefix_sha256,
        )
        for row in serial_manifest.requests
    )
    ready = build_setup_ready(
        serial_manifest,
        consumer="serial",
        manifest_path=manifest_path,
        initial_proposals=proposals,
        ready_published_ns=40,
    )
    atomic_write_json(ready_path, ready)
    loaded = load_setup_ready(
        ready_path,
        manifest_path=manifest_path,
        consumer="serial",
        expected_request_ids=("A", "B"),
    )
    assert loaded["global_decode_ready"] is True
    assert len(loaded["initial_proposals"]) == 2


def test_admission_evidence_rejects_preboundary_target_advance():
    base = {
        "schema_version": "specrhythm.phase4b-resident-admission-event.v1",
        "cycle_id": 0,
        "timestamp_ns": 20,
        "consumer": "target-only",
        "request_id": "A",
        "internal_request_id": "opaque-A",
        "num_output_tokens": 1,
        "global_decode_ready": False,
        "measurement_start_ns": None,
        "initial_proposal_installed": False,
        "admissible": False,
        "reason": "bootstrap-ready-awaiting-global-boundary",
        "scheduled": False,
        "scheduled_token_count": 0,
        "explicit_request_predicate": True,
        "current_step_arithmetic": False,
    }
    released = {
        **base,
        "cycle_id": 1,
        "timestamp_ns": 40,
        "global_decode_ready": True,
        "measurement_start_ns": 32,
        "admissible": True,
        "reason": "global-decode-ready",
        "scheduled": True,
        "scheduled_token_count": 1,
    }
    assert validate_resident_admission_events(
        (base, released), consumer="target-only"
    ) == []
    broken = {**base, "admissible": True, "scheduled": True}
    errors = validate_resident_admission_events(
        (broken, released), consumer="target-only"
    )
    assert any("advanced after bootstrap" in error for error in errors)


def test_serial_admission_requires_initial_proposal_after_global_ready():
    assert resident_admission_decision(
        num_output_tokens=1,
        global_decode_ready=True,
        consumer="serial",
        has_initial_proposal=False,
    ) == (False, "serial-initial-proposal-not-installed")
    assert resident_admission_decision(
        num_output_tokens=1,
        global_decode_ready=True,
        consumer="serial",
        has_initial_proposal=True,
    ) == (True, "global-decode-ready")


def test_both_gpu_consumers_use_incremental_setup_and_resident_scheduler():
    root = Path(__file__).parents[1]
    target = (root / "src/specrhythm/phase4/resident_vllm.py").read_text()
    serial = (root / "src/specrhythm/phase4/vllm_remote.py").read_text()
    target_runner = (root / "src/specrhythm/phase4/resident_runner.py").read_text()
    serial_runner = (root / "src/specrhythm/phase4/serial_runner.py").read_text()
    forbidden = "requires every frozen request in one initial prefill batch"
    assert forbidden not in target
    assert "requires all requests in one initial prefill batch" not in serial
    assert "IncrementalResidentSetup" in target
    assert "IncrementalResidentSetup" in serial
    scheduler = "specrhythm.phase4.resident_scheduler.ResidentSetupScheduler"
    assert scheduler in target_runner
    assert scheduler in serial_runner

    from specrhythm.phase4.resident_vllm import ResidentTargetProposer
    from specrhythm.phase4.vllm_remote import RemoteDraftProposer

    target_completion = inspect.getsource(
        ResidentTargetProposer._complete_global_setup
    )
    serial_completion = inspect.getsource(
        RemoteDraftProposer._complete_resident_setup
    )
    assert target_completion.count("self.tp_group.barrier()") == 1
    assert serial_completion.count("self.tp_group.barrier()") == 1
