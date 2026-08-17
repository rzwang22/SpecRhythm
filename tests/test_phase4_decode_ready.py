from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from specrhythm import cli
from specrhythm.phase4.decode_ready import (
    DecodeReadyManifest,
    DecodeReadyProvenance,
    ResidentSetupObservation,
    ResidentWarmStartProvider,
    build_first_target_forward_contract,
    compare_decode_consumers,
    compare_raw_and_decode_outputs,
    load_decode_ready_manifest,
    validate_decode_ready_manifest,
    validate_first_target_forward_contract,
    validate_measurement_boundary,
)


def make_manifest() -> DecodeReadyManifest:
    provenance = DecodeReadyProvenance(
        specrhythm_git_commit="1" * 40,
        vllm_version="0.25.1",
        vllm_commit="752a3a504485790a2e8491cacbb35c137339ad34",
        vllm_patch_stack_sha256=("a" * 64, "b" * 64, "c" * 64),
        target_model_path="/models/Qwen3-32B",
        target_model_revision="target-revision",
        draft_model_path="/models/Qwen3-0.6B",
        draft_model_revision="draft-revision",
        tokenizer_revision="tokenizer-revision",
        workload_sha256="d" * 64,
        sampling_configuration={"temperature": 0.0, "seed": 1664},
        batch_invariant_configuration={
            "requested": True,
            "VLLM_BATCH_INVARIANT": "1",
            "enable_dbo": False,
        },
        target_physical_gpu_ids=(1, 2),
        draft_physical_gpu_ids=(0,),
        target_tensor_parallel_size=2,
        draft_tensor_parallel_size=1,
    )
    observations = [
        ResidentSetupObservation(
            request_id="A",
            internal_target_request_id="opaque-A",
            prompt_token_ids=(1, 2, 3),
            bootstrap_token_id=4,
            target_materialized_kv_token_count=3,
            target_num_computed_tokens=3,
            draft_materialized_kv_token_count=4,
            bootstrap_ready_ns=12,
            draft_initialization_complete_ns=14,
        ),
        ResidentSetupObservation(
            request_id="B",
            internal_target_request_id="opaque-B",
            prompt_token_ids=(5, 6),
            bootstrap_token_id=7,
            target_materialized_kv_token_count=2,
            target_num_computed_tokens=2,
            draft_materialized_kv_token_count=3,
            bootstrap_ready_ns=15,
            draft_initialization_complete_ns=18,
        ),
    ]
    return ResidentWarmStartProvider().prepare(
        observations,
        provenance,
        setup_start_ns=10,
        setup_complete_ns=20,
        global_barrier_ns=30,
        measurement_start_ns=40,
    )


def test_manifest_round_trip_hash_and_immutability():
    manifest = make_manifest()
    assert validate_decode_ready_manifest(manifest) == []
    assert load_decode_ready_manifest(manifest.to_dict()) == manifest
    assert len(manifest.manifest_sha256) == 64
    with pytest.raises(FrozenInstanceError):
        manifest.measurement_start_ns = 99
    tampered = manifest.to_dict()
    tampered["requests"][0]["bootstrap_token_id"] = 99
    with pytest.raises(ValueError, match="invalid DecodeReadyManifest"):
        load_decode_ready_manifest(tampered)


def test_prompt_bootstrap_target_pending_and_draft_kv_accounting():
    request = make_manifest().requests[0]
    assert request.logical_committed_prefix_count == request.prompt_token_count + 1
    assert request.target_materialized_kv_token_count + 1 == (
        request.logical_committed_prefix_count
    )
    assert request.target_pending_input_token_id == request.bootstrap_token_id
    assert request.target_pending_input_position == 3
    assert request.logical_committed_prefix_token_ids[-1] == request.bootstrap_token_id
    assert request.draft_materialized_kv_token_count == (
        request.logical_committed_prefix_count
    )
    assert request.initial_proposal_generated is False


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("target_materialized_kv_token_count", 4, "Target KV + 1"),
        ("draft_materialized_kv_token_count", 3, "Draft KV"),
        ("target_pending_input_position", 2, "pending Target position"),
        ("initial_proposal_generated", True, "proposal was generated"),
    ],
)
def test_manifest_fails_closed_on_decode_ready_invariant(field, value, error):
    manifest = make_manifest()
    changed = replace(manifest.requests[0], **{field: value})
    invalid = replace(manifest, requests=(changed,) + manifest.requests[1:]).with_hash()
    assert any(error in item for item in validate_decode_ready_manifest(invalid))


def test_first_target_and_serial_verification_inputs_are_exact():
    request = make_manifest().requests[0]
    target = build_first_target_forward_contract(
        request,
        consumer="target-only",
        target_forward_start_ns=41,
        target_forward_end_ns=42,
    )
    assert target["verification_input_token_ids"] == [4]
    assert target["input_positions"] == [3]
    serial = build_first_target_forward_contract(
        request,
        consumer="serial",
        proposal_token_ids=(8, 9),
        target_forward_start_ns=43,
        target_forward_end_ns=44,
    )
    assert serial["verification_input_token_ids"] == [4, 8, 9]
    assert serial["input_positions"] == [3, 4, 5]
    broken = {**serial, "input_positions": [4, 5, 6]}
    assert any(
        "off by one" in error
        for error in validate_first_target_forward_contract(broken, request)
    )


def test_measurement_boundary_rejects_early_proposal_or_forward():
    manifest = make_manifest()
    assert (
        validate_measurement_boundary(
            manifest,
            first_draft_start_ns=41,
            first_draft_end_ns=42,
            first_target_decode_start_ns=43,
            proposal_created_timestamps_ns=(41,),
        )
        == []
    )
    errors = validate_measurement_boundary(
        manifest,
        first_draft_start_ns=39,
        first_draft_end_ns=42,
        first_target_decode_start_ns=39,
        proposal_created_timestamps_ns=(39,),
    )
    assert len(errors) == 3


def test_raw_decode_and_triangular_comparators_include_termination():
    manifest = make_manifest()
    raw = [
        {
            "request_id": "A",
            "generated_token_ids": [4, 8],
            "finish_reason": "stop",
            "eos_token_id": 8,
            "max_token_termination": False,
            "final_logical_length": 5,
        },
        {
            "request_id": "B",
            "generated_token_ids": [7, 9],
            "finish_reason": "length",
            "eos_token_id": None,
            "max_token_termination": True,
            "final_logical_length": 4,
        },
    ]
    decode = [
        {**row, "continuation_token_ids": row["generated_token_ids"][1:]}
        for row in raw
    ]
    assert compare_raw_and_decode_outputs(raw, decode, manifest)["valid"] is True
    assert compare_decode_consumers(decode, decode)["valid"] is True
    assert compare_decode_consumers(decode, decode, decode)["valid"] is True
    changed = [{**decode[0], "finish_reason": "length"}, decode[1]]
    assert compare_decode_consumers(decode, changed)["valid"] is False


def test_decode_ready_dry_run_cli_is_cuda_free_and_deterministic(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    assert (
        cli.main(
            ["phase4-decode-ready-contract-dry-run", "--output", str(first)]
        )
        == 0
    )
    assert (
        cli.main(
            ["phase4-decode-ready-contract-dry-run", "--output", str(second)]
        )
        == 0
    )
    one = json.loads(first.read_text(encoding="utf-8"))
    two = json.loads(second.read_text(encoding="utf-8"))
    assert one == two
    assert one["gpu_execution_performed"] is False
    assert one["manifest_validation"]["valid"] is True
    assert one["kv_connector_implemented"] is False


def test_patch_stack_contains_independent_scheduler_and_timing_patches():
    root = Path(__file__).parents[1]
    manager = (root / "integrations/vllm/manage_patch.py").read_text(
        encoding="utf-8"
    )
    assert "0002-scheduler-request-admissibility-hook.patch" in manager
    assert "0003-target-forward-timing-observer.patch" in manager
    assert "5cd618de8826e15ef00ca1735101a29af06029b7ce9d54cede00bf2b401cc257" in manager
