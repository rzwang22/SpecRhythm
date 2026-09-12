from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest

from specrhythm.phase4.decode_ready import (
    DecodeReadyProvenance,
    ResidentSetupObservation,
    ResidentWarmStartProvider,
)
from specrhythm.phase4.manifest import atomic_write_json
from specrhythm.phase4.request_identity import FrozenPromptIdentityMap
from specrhythm.phase4.resident_setup import (
    IncrementalResidentSetup,
    ResidentSetupStage,
    build_setup_ready,
    classify_resident_setup_wave,
    load_setup_ready,
    observation_to_dict,
    resident_admission_decision,
    validate_resident_admission_events,
)
from specrhythm.phase4.serial import PROTOCOL_VERSION, Proposal


class _RowSlice:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


class _TokenRows:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]

    def __getitem__(self, item):
        row, selected = item
        return _RowSlice(self.rows[row][selected])


def _classify(request_ids, sampled, logical_rows, materialized, prompts):
    return classify_resident_setup_wave(
        request_ids=request_ids,
        sampled_token_ids=sampled,
        num_tokens_no_spec=[len(row) for row in logical_rows],
        token_ids_cpu=_TokenRows(logical_rows),
        target_materialized_token_counts=materialized,
        frozen_prompts=prompts,
    )


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


def _roundtrip(
    observation: ResidentSetupObservation,
) -> ResidentSetupObservation:
    serialized = json.loads(json.dumps(observation_to_dict(observation)))
    assert isinstance(serialized["prompt_token_ids"], list)
    return ResidentSetupObservation.from_dict(serialized)


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
    assert tracker.record(_roundtrip(_observation("A", 1))) is True
    assert tracker.complete is False
    assert tracker.observed_request_ids == ("A",)
    assert resident_admission_decision(
        num_output_tokens=1,
        global_decode_ready=False,
        consumer="target-only",
        has_initial_proposal=False,
    ) == (False, "bootstrap-ready-awaiting-global-boundary")
    assert tracker.record(_roundtrip(_observation("B", 4))) is True
    assert tracker.complete is True
    assert tracker.observed_request_ids == ("A", "B")
    assert tracker.completion_transition_count == 1


def test_incremental_l5_mixed_subsets_have_one_completion_transition():
    request_ids = ("A", "B", "C", "D", "E")
    tracker = IncrementalResidentSetup(request_ids, setup_start_ns=5)
    offsets = {request_id: index * 3 + 1 for index, request_id in enumerate(request_ids)}
    for subset in (("A",), ("B", "C"), ("D",), ("E",)):
        for request_id in subset:
            tracker.record(
                _roundtrip(_observation(request_id, offsets[request_id]))
            )
    assert tracker.complete is True
    assert tracker.observed_request_ids == request_ids
    assert tracker.completion_transition_count == 1
    assert tracker.record(tracker.observations[-1]) is False
    assert tracker.completion_transition_count == 1


def test_chunked_prefill_mixed_wave_waits_for_exact_bootstrap_evidence():
    prompts = {"A": (1, 2, 3, 4), "B": (5, 6), "C": (7, 8)}
    rows = _classify(
        ("opaque-A", "opaque-B", "opaque-C"),
        ((), (), (9,)),
        ((1, 2), (5, 6), (7, 8, 9)),
        (2, 2, 2),
        prompts,
    )
    assert [row.stage for row in rows] == [
        ResidentSetupStage.PARTIAL_PREFILL,
        ResidentSetupStage.FULL_PROMPT_NO_BOOTSTRAP,
        ResidentSetupStage.BOOTSTRAP_READY,
    ]
    assert rows[0].stable_request_id is None
    assert rows[1].stable_request_id == "B"
    assert rows[2].stable_request_id == "C"
    assert rows[2].bootstrap_token_id == 9


def test_chunked_prefill_subsequent_waves_record_once_in_frozen_order():
    prompts = {"A": (1, 2, 3, 4), "B": (5, 6), "C": (7, 8)}
    tracker = IncrementalResidentSetup(tuple(prompts), setup_start_ns=5)
    initializations = {request_id: 0 for request_id in prompts}

    def consume(classified, timestamp):
        for row in classified:
            if row.stage is not ResidentSetupStage.BOOTSTRAP_READY:
                continue
            assert row.stable_request_id is not None
            request_id = row.stable_request_id
            existing = tracker.get(request_id)
            if existing is None:
                initializations[request_id] += 1
                prompt = prompts[request_id]
                tracker.record(
                    ResidentSetupObservation(
                        request_id=request_id,
                        internal_target_request_id=row.internal_request_id,
                        prompt_token_ids=prompt,
                        bootstrap_token_id=int(row.bootstrap_token_id),
                        target_materialized_kv_token_count=len(prompt),
                        target_num_computed_tokens=len(prompt),
                        draft_materialized_kv_token_count=len(prompt) + 1,
                        bootstrap_ready_ns=timestamp,
                        draft_initialization_complete_ns=timestamp + 1,
                    )
                )
            else:
                assert existing.bootstrap_token_id == row.bootstrap_token_id

    consume(
        _classify(
            ("opaque-A", "opaque-B", "opaque-C"),
            ((), (), (9,)),
            ((1, 2), (5, 6), (7, 8, 9)),
            (2, 2, 2),
            prompts,
        ),
        10,
    )
    assert tracker.observed_request_ids == ("C",)
    assert tracker.complete is False
    consume(
        _classify(
            ("opaque-A", "opaque-B"),
            ((10,), (11,)),
            ((1, 2, 3, 4, 10), (5, 6, 11)),
            (4, 2),
            prompts,
        ),
        20,
    )
    assert tracker.observed_request_ids == ("A", "B", "C")
    assert tracker.complete is True
    assert tracker.completion_transition_count == 1
    assert initializations == {"A": 1, "B": 1, "C": 1}

    # An identical repeated hook does not authorize a second Draft initialize.
    consume(
        _classify(
            ("opaque-C",), ((9,),), ((7, 8, 9),), (2,), prompts
        ),
        30,
    )
    assert initializations["C"] == 1


def test_partial_and_ambiguous_prompt_rows_never_become_bootstrap_ready():
    prompts = {"A": (1, 2, 3), "B": (1, 2, 4)}
    row = _classify(("opaque",), ((),), ((1, 2),), (2,), prompts)[0]
    assert row.stage is ResidentSetupStage.PARTIAL_PREFILL
    assert row.stable_request_id is None
    assert row.candidate_request_ids == ("A", "B")
    identity = FrozenPromptIdentityMap(prompts)
    assert identity.internal_to_stable == {}

    buffered = _classify(
        ("opaque-full-buffer",), ((),), ((8, 9, 10, 11),), (2,), {"C": (8, 9, 10, 11)}
    )[0]
    assert buffered.stage is ResidentSetupStage.PARTIAL_PREFILL
    assert buffered.stable_request_id is None
    assert buffered.candidate_request_ids == ("C",)


def test_setup_classifier_rejects_unknown_changed_or_illegally_advanced_rows():
    prompts = {"A": (1, 2)}
    with pytest.raises(RuntimeError, match="matches no frozen"):
        _classify(("opaque",), ((),), ((8,),), (1,), prompts)
    with pytest.raises(RuntimeError, match="beyond one bootstrap"):
        _classify(("opaque",), ((),), ((1, 2, 3, 4),), (2,), prompts)
    with pytest.raises(RuntimeError, match="does not identify one frozen prompt"):
        _classify(("opaque",), ((4,),), ((1, 2, 3),), (2,), prompts)
    with pytest.raises(RuntimeError, match="invalid token"):
        _classify(("opaque",), ((),), ((1, -2),), (1,), prompts)


def test_duplicate_bootstrap_change_and_early_completion_fail_closed():
    tracker = IncrementalResidentSetup(("A", "B"), setup_start_ns=5)
    first = _observation("A", 1)
    tracker.record(first)
    assert tracker.complete is False
    with pytest.raises(RuntimeError, match="observation changed"):
        tracker.record(replace(first, bootstrap_token_id=999))
    assert tracker.completion_transition_count == 0


def test_scale_wave_exceeds_vllm_prefill_budget_without_false_bootstrap():
    prompts = {
        f"R{index:03d}": (index + 1,) + tuple(range(1000, 1164))
        for index in range(100)
    }
    assert sum(len(prompt) for prompt in prompts.values()) > 16384
    request_ids = tuple(f"opaque-{index:03d}" for index in range(100))
    logical_rows = tuple(tuple(prompt[:32]) for prompt in prompts.values())
    rows = _classify(
        request_ids,
        tuple(() for _ in request_ids),
        logical_rows,
        tuple(32 for _ in request_ids),
        prompts,
    )
    assert len(rows) == 100
    assert all(row.stage is ResidentSetupStage.PARTIAL_PREFILL for row in rows)
    assert all(row.bootstrap_token_id is None for row in rows)


def test_duplicate_or_stale_bootstrap_fails_closed():
    tracker = IncrementalResidentSetup(("A",), setup_start_ns=5)
    original = _observation("A", 1)
    tracker.record(original)
    assert tracker.record(original) is False
    with pytest.raises(RuntimeError, match="observation changed"):
        tracker.record(replace(original, bootstrap_token_id=999))
    with pytest.raises(RuntimeError, match="unexpected resident setup request"):
        tracker.record(replace(original, request_id="B"))


def test_observation_json_roundtrip_restores_tuple_and_builds_manifest():
    original = _observation("A", 1)
    restored = _roundtrip(original)
    assert restored == original
    assert isinstance(restored.prompt_token_ids, tuple)
    manifest = ResidentWarmStartProvider().prepare(
        (restored,),
        _provenance(),
        setup_start_ns=5,
        setup_complete_ns=20,
        global_barrier_ns=21,
        measurement_start_ns=22,
    )
    assert manifest.requests[0].logical_committed_prefix_token_ids == (1, 2, 3)


@pytest.mark.parametrize(
    "prompt", [[1, "bad"], [1, 2.5], [1, True], [1, -2], "1,2"]
)
def test_observation_loader_rejects_malformed_prompt_tokens(prompt):
    value = observation_to_dict(_observation("A", 1))
    value["prompt_token_ids"] = prompt
    with pytest.raises(ValueError, match="prompt_token_ids"):
        ResidentSetupObservation.from_dict(value)


def test_observation_loader_canonicalizes_request_ids_to_strings():
    value = observation_to_dict(_observation("A", 1))
    value["request_id"] = 7
    value["internal_target_request_id"] = 8
    restored = ResidentSetupObservation.from_dict(value)
    assert restored.request_id == "7"
    assert restored.internal_target_request_id == "8"


def test_in_memory_observation_rejects_list_prompt_tokens():
    with pytest.raises(ValueError, match="non-empty tuple"):
        ResidentSetupObservation(
            request_id="A",
            internal_target_request_id="opaque-A",
            prompt_token_ids=[1, 2],
            bootstrap_token_id=3,
            target_materialized_kv_token_count=2,
            target_num_computed_tokens=2,
            draft_materialized_kv_token_count=3,
            bootstrap_ready_ns=11,
            draft_initialization_complete_ns=12,
        )


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
    dual = (root / "src/specrhythm/phase4/vllm_dual.py").read_text()
    target_runner = (root / "src/specrhythm/phase4/resident_runner.py").read_text()
    serial_runner = (root / "src/specrhythm/phase4/serial_runner.py").read_text()
    forbidden = "requires every frozen request in one initial prefill batch"
    assert forbidden not in target
    assert "requires all requests in one initial prefill batch" not in serial
    assert "IncrementalResidentSetup" in target
    assert "IncrementalResidentSetup" in serial
    assert "IncrementalResidentSetup" in dual
    assert target.count("classify_resident_setup_wave(") == 1
    assert serial.count("classify_resident_setup_wave(") == 1
    assert dual.count("classify_resident_setup_wave(") == 1
    for source in (target, serial, dual):
        assert "target_materialized_token_counts" in source
        assert "ResidentSetupStage.BOOTSTRAP_READY" in source
    assert "ResidentSetupObservation.from_dict" in target
    assert "ResidentSetupObservation.from_dict" in serial
    assert "ResidentSetupObservation.from_dict" in dual
    assert "ResidentSetupObservation(**row)" not in target
    assert "ResidentSetupObservation(**row)" not in serial
    assert "ResidentSetupObservation(**row)" not in dual
    scheduler = "specrhythm.phase4.resident_scheduler.ResidentSetupScheduler"
    assert scheduler in target_runner
    assert scheduler in serial_runner

    patch = (
        root
        / "integrations/vllm/patches/0001-custom-proposer-request-and-verify-hooks.patch"
    ).read_text()
    assert "target_materialized_token_counts=[" in patch
    assert "num_computed_tokens_cpu[index]" in patch
    assert "scheduler_output.num_scheduled_tokens[request_id]" in patch

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
