from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path

import pytest

from specrhythm import cli
from specrhythm.phase4.config import (
    PYTORCH_VERSION,
    VLLM_COMMIT,
    VLLM_VERSION,
    GreedySamplingContract,
    load_phase4_config,
)
from specrhythm.phase4.contracts import (
    DraftEngineAdapter,
    RequestState,
    TargetEngineAdapter,
    VerificationBatch,
)
from specrhythm.phase4.fake import (
    FakeDraftEngineAdapter,
    FakeTargetEngineAdapter,
    run_fake_contract,
)
from specrhythm.phase4.manifest import (
    build_runtime_manifest,
    validate_environment,
    validate_runtime_manifest,
    validate_topology,
)
from specrhythm.phase4.stock_vllm import load_smoke_requests, validate_worker_ranks
from specrhythm.phase4.validation import validate_artifacts

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "phase4a_target_fair_1d2v.yaml"
WORKLOAD = ROOT / "tests" / "fixtures" / "phase4-r3-smoke.jsonl"


@pytest.fixture
def phase4_config(tmp_path, monkeypatch):
    draft = tmp_path / "Qwen3-0.6B"
    target = tmp_path / "Qwen3-32B"
    for path, model_type in ((draft, "qwen3-draft"), (target, "qwen3-target")):
        path.mkdir()
        (path / "config.json").write_text(
            json.dumps({"model_type": model_type}), encoding="utf-8"
        )
        (path / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "Qwen3Tokenizer"}), encoding="utf-8"
        )
    monkeypatch.setenv("SR_DRAFT_MODEL", str(draft))
    monkeypatch.setenv("SR_TARGET_MODEL", str(target))
    return load_phase4_config(str(CONFIG))


def valid_environment():
    return {
        "schema_version": "specrhythm.phase4-environment.v1",
        "python_version": "3.11.10",
        "pytorch_version": PYTORCH_VERSION + "+cu130",
        "torch_cuda_available": True,
        "nccl_version": "2.28.9",
        "vllm": {"version": VLLM_VERSION, "record_sha256": "a" * 64},
        "vllm_source": {
            "commit": VLLM_COMMIT,
            "exact_tag": "v" + VLLM_VERSION,
            "tracked_tree_clean": True,
        },
        "transformers": {"version": "5.5.3"},
    }


def valid_topology():
    return {
        "schema_version": "specrhythm.phase4-topology.v1",
        "gpus": [
            {"physical_gpu_id": 0, "uuid": "GPU-draft"},
            {"physical_gpu_id": 1, "uuid": "GPU-target-0"},
            {"physical_gpu_id": 2, "uuid": "GPU-target-1"},
        ],
    }


def worker(rank, physical, tp):
    return {
        "global_rank": rank,
        "local_rank": rank,
        "world_size": tp,
        "logical_cuda_index": rank,
        "physical_gpu_id": physical,
        "gpu_uuid": f"GPU-{physical}",
        "parameter_count": 10,
        "parameter_bytes": 20,
        "allocated_memory_bytes": 30,
        "reserved_memory_bytes": 40,
        "all_parameters_on_expected_device": True,
        "attention_backends": ["FLASH_ATTN"],
    }


def test_config_freezes_target_fair_layout_and_modes(phase4_config):
    assert phase4_config.draft.physical_gpu_ids == (0,)
    assert phase4_config.target.physical_gpu_ids == (1, 2)
    assert phase4_config.target.tensor_parallel_size == 2
    assert [mode.name for mode in phase4_config.modes] == [
        "target-only",
        "serial-disaggregated",
        "dual-batch",
    ]
    assert all(not mode.built_in_vllm_speculative for mode in phase4_config.modes)
    assert all(not mode.vllm_dbo for mode in phase4_config.modes)


def test_wrong_vllm_commit_python_and_torch_are_rejected(phase4_config):
    report = valid_environment()
    report["vllm_source"] = {
        "commit": "bad",
        "exact_tag": "v" + VLLM_VERSION,
        "tracked_tree_clean": True,
    }
    assert any("commit" in item for item in validate_environment(report, phase4_config)["errors"])
    report = valid_environment()
    report["python_version"] = "3.12.0"
    report["pytorch_version"] = "2.10.0"
    errors = validate_environment(report, phase4_config)["errors"]
    assert any("Python" in item for item in errors)
    assert any("PyTorch" in item for item in errors)


def test_topology_overlap_and_missing_are_rejected(phase4_config):
    missing = valid_topology()
    missing["gpus"] = missing["gpus"][:2]
    assert not validate_topology(missing, phase4_config)["valid"]
    with pytest.raises(ValueError, match="disjoint"):
        replace(
            phase4_config,
            target=replace(phase4_config.target, physical_gpu_ids=(0, 2)),
        )


def test_adapter_protocol_and_candidate_accounting():
    state = RequestState("request", (1, 2))
    draft = FakeDraftEngineAdapter()
    target = FakeTargetEngineAdapter()
    assert isinstance(draft, DraftEngineAdapter)
    assert isinstance(target, TargetEngineAdapter)
    candidates = draft.propose(state, 3)
    batch = VerificationBatch.from_candidates(
        candidates,
        [node.stable_node_id for node in candidates.nodes[:2]],
        candidates.created_monotonic_ns + 1,
    )
    result = target.verify(state, batch)
    assert candidates.drafted_nodes == 3
    assert result.accounting == {
        "verified_nodes": 2,
        "accepted_nodes": 1,
        "target_bonus_tokens": 0,
        "committed_tokens": 1,
    }
    updated = state.apply(result, 0)
    assert updated.prefix_epoch == 1
    with pytest.raises(ValueError, match="monotonically"):
        updated.apply(replace(result, prefix_epoch=1), 0)


def test_deterministic_greedy_and_fake_labels():
    assert GreedySamplingContract().to_dict()["do_sample"] is False
    with pytest.raises(ValueError, match="greedy"):
        GreedySamplingContract(do_sample=True)
    report = run_fake_contract()
    assert report["fake_data"] is True
    assert report["gpu_result"] is False
    assert report["serving_performance_result"] is False


def test_builtin_speculative_and_vllm_dbo_cannot_label_future_modes(tmp_path):
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    value["future_modes"]["serial-disaggregated"]["built_in_vllm_speculative"] = True
    path = tmp_path / "bad-spec.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="colocated speculative"):
        load_phase4_config(str(path))
    value["future_modes"]["serial-disaggregated"]["built_in_vllm_speculative"] = False
    value["future_modes"]["dual-batch"]["vllm_dbo"] = True
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="vLLM DBO"):
        load_phase4_config(str(path))


def test_worker_rank_validation_detects_missing_or_empty_rank(phase4_config):
    rows = [worker(0, 1, 2)]
    assert validate_worker_ranks(rows, phase4_config.target)
    rows.append(worker(1, 2, 2))
    assert not validate_worker_ranks(rows, phase4_config.target)
    rows[1]["parameter_count"] = 0
    assert any("parameters" in item for item in validate_worker_ranks(rows, phase4_config.target))


def test_runtime_manifest_schema(phase4_config, tmp_path):
    environment = tmp_path / "environment.json"
    topology = tmp_path / "topology.json"
    environment.write_text(json.dumps(valid_environment()), encoding="utf-8")
    topology.write_text(json.dumps(valid_topology()), encoding="utf-8")
    manifest = build_runtime_manifest(
        phase4_config,
        role="target",
        git_commit="f" * 40,
        workload_path=WORKLOAD,
        environment_path=environment,
        topology_path=topology,
        worker_ranks=[worker(0, 1, 2), worker(1, 2, 2)],
        attention_backend="FLASH_ATTN",
        mode_setup={"configured_before_vllm_import": True},
    )
    assert manifest["schema_version"] == "specrhythm.phase4-runtime-manifest.v1"
    assert manifest["serving_performance_result"] is False
    assert not validate_runtime_manifest(manifest, phase4_config)


def test_full_artifact_validation_accepts_real_bringup_schema(phase4_config, tmp_path):
    environment_path = tmp_path / "environment.json"
    topology_path = tmp_path / "topology.json"
    environment_path.write_text(json.dumps(valid_environment()), encoding="utf-8")
    topology_path.write_text(json.dumps(valid_topology()), encoding="utf-8")
    roles = {
        "draft": build_runtime_manifest(
            phase4_config,
            role="draft",
            git_commit="f" * 40,
            workload_path=WORKLOAD,
            environment_path=environment_path,
            topology_path=topology_path,
                worker_ranks=[worker(0, 0, 1)],
                attention_backend="FLASH_ATTN",
                mode_setup={"configured_before_vllm_import": True},
        ),
        "target": build_runtime_manifest(
            phase4_config,
            role="target",
            git_commit="f" * 40,
            workload_path=WORKLOAD,
            environment_path=environment_path,
            topology_path=topology_path,
                worker_ranks=[worker(0, 1, 2), worker(1, 2, 2)],
                attention_backend="FLASH_ATTN",
                mode_setup={"configured_before_vllm_import": True},
        ),
    }
    runtime_path = tmp_path / "runtime-manifest.json"
    runtime_path.write_text(
        json.dumps(
            {
                "schema_version": "specrhythm.phase4-runtime-bundle.v1",
                "stage": "phase4a-stock-vllm-bringup",
                "serving_performance_result": False,
                "roles": roles,
            }
        ),
        encoding="utf-8",
    )
    requests = load_smoke_requests(WORKLOAD)
    outputs = [
        {
            "request_id": request.request_id,
            "generated_token_ids": [42],
            "token_accounting": {
                "prompt_tokens": len(request.prompt_token_ids),
                "generated_tokens": 1,
                "total_tokens": len(request.prompt_token_ids) + 1,
            },
            "timestamps": {
                "available": True,
                "scheduled_ts": 1.0,
                "first_token_ts": 2.0,
                "last_token_ts": 3.0,
            },
        }
        for request in requests
    ]

    def smoke(role, ranks):
        return {
            "schema_version": "specrhythm.phase4-stock-smoke.v1",
            "role": role,
            "fake_data": False,
            "gpu_result": True,
            "serving_performance_result": False,
            "built_in_speculative_decoding": False,
            "vllm_dbo_enabled": False,
            "specrhythm_dual_batch_implemented": False,
            "request_count": 5,
            "prompt_token_ids_revalidated": True,
            "repeated_run_deterministic": True,
            "startup_ms": 1.0,
            "run_wall_ms": [2.0, 2.0],
            "total_wall_ms": 5.0,
            "worker_ranks": ranks,
            "runs": [outputs, outputs],
            "frozen_hf_target_comparison": {
                "performed": role == "target",
                "reference_coverage_complete": role == "target",
                "all_tokens_equal": role == "target",
            },
            "provenance": {
                "git_commit": "f" * 40,
                "config_sha256": roles[role]["inputs"]["config_sha256"],
                "workload_sha256": roles[role]["inputs"]["workload_sha256"],
            },
        }

    draft_path = tmp_path / "draft.json"
    target_path = tmp_path / "target.json"
    draft_path.write_text(json.dumps(smoke("draft", [worker(0, 0, 1)])), encoding="utf-8")
    target_path.write_text(
        json.dumps(smoke("target", [worker(0, 1, 2), worker(1, 2, 2)])),
        encoding="utf-8",
    )
    report = validate_artifacts(
        phase4_config,
        environment_path=environment_path,
        topology_path=topology_path,
        runtime_manifest_path=runtime_path,
        draft_smoke_path=draft_path,
        target_smoke_path=target_path,
    )
    assert report["valid"], report


def test_r3_smoke_loader_uses_five_existing_tokenized_prompts():
    requests = load_smoke_requests(WORKLOAD)
    assert len(requests) == 5
    assert requests[3].prompt_text.startswith("<|im_start|>user")
    assert len({request.tokenizer_fingerprint for request in requests}) == 1


def test_no_cuda_probe_is_explicit_and_does_not_emit_smoke(
    phase4_config, tmp_path, monkeypatch
):
    from specrhythm.phase4 import manifest as phase4_manifest

    environment = valid_environment()
    environment["torch_cuda_available"] = False
    monkeypatch.setattr(phase4_manifest, "collect_environment", lambda _path: environment)
    monkeypatch.setattr(phase4_manifest, "collect_topology", lambda: {"gpus": []})
    environment_path = tmp_path / "environment.json"
    topology_path = tmp_path / "topology.json"
    validation_path = tmp_path / "validation.json"
    code = cli.main(
        [
            "phase4-probe",
            "--config",
            str(phase4_config.path),
            "--vllm-source",
            str(tmp_path),
            "--environment-output",
            str(environment_path),
            "--topology-output",
            str(topology_path),
            "--validation-output",
            str(validation_path),
        ]
    )
    assert code == 2
    assert json.loads(validation_path.read_text())["valid"] is False
    assert not (tmp_path / "draft-smoke.json").exists()


def test_phase4_contract_cli_is_explicit_fake(tmp_path):
    output = tmp_path / "contract.json"
    assert cli.main(["phase4-contract-dry-run", "--output", str(output)]) == 0
    value = json.loads(output.read_text())
    assert value["backend"] == "fake-contract-test-only"
    assert value["gpu_result"] is False


def test_phase4_sources_parse_as_python39():
    for path in (ROOT / "src" / "specrhythm" / "phase4").glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 9))
