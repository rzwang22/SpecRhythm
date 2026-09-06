from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_phase4_draft_qualification import (
    gate_inputs,
    regime,
    runtime_provenance,
    write_aggregate_inputs,
)
from test_phase4_serial import phase4_config as _config_fixture

from specrhythm.phase4.draft_admission import admit_device, collect_runtime
from specrhythm.phase4.draft_qualification import (
    STRUCTURAL_CHECKS,
    aggregate_directories,
    qualify_d3,
    validate_d3_directory,
)
from specrhythm.phase4.draft_qualification_gate import run_qualification

phase4_config = _config_fixture
NEW_UUID = "GPU-abcdef01-2345-6789-abcd-ef0123456789"


@pytest.mark.parametrize("changed", [False, True])
def test_current_a800_uuid_does_not_bind_retained_execution_regime(changed):
    proof, current = regime(), runtime_provenance()
    before = copy.deepcopy(proof)
    if changed:
        current["gpu_uuid"] = NEW_UUID
    admission = admit_device(current, proof["run_identity"], proof)
    assert admission["valid"], admission["errors"]
    assert admission["same_physical_gpu"] is not changed
    assert admission["current_device_binding_valid"]
    assert admission["execution_regime_device_compatible"]
    assert admission["historical_probe_gpu_uuid"] == before["gpu_identity"]["gpu_uuid"]
    assert admission["current_gpu_uuid"] == current["gpu_uuid"]
    assert proof == before


def test_current_run_aggregate_accepts_a_different_historical_gpu_uuid(tmp_path):
    write_aggregate_inputs(tmp_path, current_uuid=NEW_UUID)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    result = aggregate_directories(tmp_path)
    assert result["valid"], result["errors"]
    for name in ("D3-B2", "D3-B4", "D3-B8"):
        gate = json.loads((tmp_path / name / "gate.json").read_text())
        assert gate["d3_qualified"] and gate["execution_regime_device_compatible"]
        assert not gate["same_physical_gpu"]
    assert all(p.read_bytes() == data for p, data in before.items())


@pytest.mark.parametrize(
    "field,value",
    [
        ("gpu_name", "NVIDIA A100-SXM4-80GB"),
        ("physical_gpu_id", 1),
        ("logical_cuda_index", 1),
        ("cuda_visible_devices", "1"),
        ("gpu_uuid", "GPU-0"),
        ("gpu_uuid", ""),
        ("gpu_uuid", None),
        ("world_size", 2),
        ("tensor_parallel_size", 2),
        ("global_rank", 1),
        ("startup_uuid_validation_count", 0),
        ("python_major_minor", [3, 12]),
        ("dtype", "float16"),
        ("enforce_eager", False),
        ("prefix_caching", True),
        ("runner_class", "MRV2"),
    ],
)
def test_current_device_and_runtime_validation_remain_strict(field, value):
    proof, current = regime(), runtime_provenance()
    current[field] = value
    assert not admit_device(current, proof["run_identity"], proof)["valid"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("compute_capability", "9.0"),
        ("batch_invariant_effective", False),
        ("batch_invariant_env_resolved", False),
        ("batch_invariant_env_raw", "0"),
        ("batch_invariant_validation", {"valid": False}),
    ],
)
def test_sm80_and_effective_batch_invariance_are_required(field, value):
    proof, current = regime(), runtime_provenance()
    current["batch_invariance"][field] = value
    assert not admit_device(current, proof["run_identity"], proof)["valid"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("implementation", "different"),
        ("flash_attention_version", 3),
        ("batch_invariant_enabled", False),
    ],
)
def test_captured_attention_properties_remain_exact(field, value):
    proof, current = regime(), runtime_provenance()
    current["attention_implementations"][0][field] = value
    assert not admit_device(current, proof["run_identity"], proof)["valid"]


@pytest.mark.parametrize(
    "field",
    [
        "vllm_api",
        "numerical_source_sha256",
        "five_patch_sha256",
        "model_files_sha256",
        "config_sha256",
        "versions",
        "tokenizer_revision",
    ],
)
def test_cross_run_file_version_model_config_identity_remains_exact(field):
    proof, current = regime(), runtime_provenance()
    identity = copy.deepcopy(proof["run_identity"])
    identity[field] = {"changed": True}
    assert not admit_device(current, identity, proof)["valid"]


@pytest.mark.parametrize("index", range(25))
def test_each_installed_api_source_hash_remains_binding(index):
    proof, current = regime(), runtime_provenance()
    api = json.loads(
        (Path(__file__).parents[1] / "src/specrhythm/phase4/vllm_draft_api.json").read_text()
    )
    proof["run_identity"]["vllm_api"] = {
        "vllm_commit": api["vllm_commit"],
        "files": [
            {"path": r["path"], "sha256": r["required_installed_sha256"]} for r in api["files"]
        ],
    }
    identity = copy.deepcopy(proof["run_identity"])
    identity["vllm_api"]["files"][index]["sha256"] = "changed"
    current["vllm_api"] = identity["vllm_api"]
    assert not admit_device(current, identity, proof)["valid"]


def fake_setup(monkeypatch, current):
    from specrhythm.phase4 import draft_qualification_gate as gate

    calls = []
    _, report = gate_inputs()
    report["provenance"] = current
    backend = SimpleNamespace(
        provenance=current, shutdown=lambda: calls.append("shutdown"), report=lambda: report
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: object())),
    )
    monkeypatch.setattr(gate, "_hf_fixture", lambda *a: [{}])
    monkeypatch.setattr(gate, "VllmBatchedDraftBackend", lambda *a: backend)
    monkeypatch.setattr(gate, "collect_runtime", lambda *a: current)
    monkeypatch.setattr(gate, "BatchedDraftStateMachine", lambda *a: backend)

    def replay(*args):
        calls.append("replay")
        return {"completed_requests": 8, "comparisons": gate_inputs()[0]["comparisons"]}

    monkeypatch.setattr(gate, "replay_production", replay)
    return calls


@pytest.mark.parametrize("invalid", [False, True])
def test_d3_entry_admits_new_uuid_or_reports_one_unexecuted_admission_failure(
    tmp_path, monkeypatch, phase4_config, invalid
):
    current, proof = runtime_provenance(), regime()
    current["gpu_uuid"] = NEW_UUID
    if invalid:
        current["gpu_name"] = "incompatible GPU"
    calls = fake_setup(monkeypatch, current)
    model_config = phase4_config.draft.resolved_model_path / "config.json"
    value = json.loads(model_config.read_text())
    value["vocab_size"] = 151936
    model_config.write_text(json.dumps(value))
    output = tmp_path / "D3-B8"
    result = run_qualification(phase4_config, 8, output, proof["run_identity"], proof)
    assert result["d3_qualified"] is not invalid, result["errors"]
    assert ("replay" in calls) is not invalid
    assert "shutdown" in calls
    assert result["execution_started"] is not invalid
    assert result["admission_valid"] is not invalid
    assert result["structural_checks_executed"] is not invalid
    assert not result["same_physical_gpu"]
    if invalid:
        assert result["vllm_semantic_valid"] is None
        assert result["structural_checks"] == dict.fromkeys(STRUCTURAL_CHECKS, None)
        assert len(result["errors"]) == 1
        assert "structural gate failed" not in str(result["errors"])
        aggregate = aggregate_directories(tmp_path)
        assert aggregate["gates"]["D3-B8"]["execution_started"] is False
        assert "structural gate failed" not in str(aggregate["errors"])
        assert validate_d3_directory(output)[1] == result["errors"]
    else:
        assert not validate_d3_directory(output)[1]


def test_preflight_failure_writes_unavailable_results_without_loading_backend(tmp_path):
    result = run_qualification(
        None, 8, tmp_path / "gate", {}, {}, preflight_error="source mismatch"
    )
    assert result["errors"] == ["ValueError: source mismatch"]
    assert not result["execution_started"] and not result["admission_valid"]
    assert set(result["structural_checks"].values()) == {None}
    assert validate_d3_directory(tmp_path / "gate")[1] == result["errors"]


def test_success_flags_cannot_override_invalid_device_evidence():
    observation, backend = gate_inputs()
    observation["admission"]["current_runtime_provenance"]["physical_gpu_id"] = 2
    assert not qualify_d3(observation, backend, regime())["d3_qualified"]


def test_runtime_collection_uses_actual_worker_metadata_without_a_forward(monkeypatch):
    from specrhythm.phase4 import batch_invariant

    batch = runtime_provenance()["batch_invariance"]
    monkeypatch.setattr(batch_invariant, "worker_batch_invariant_evidence", lambda _: batch)
    impl = SimpleNamespace(vllm_flash_attn_version=2, batch_invariant_enabled=True)
    backend = SimpleNamespace(
        provenance={"gpu_uuid": NEW_UUID},
        worker=SimpleNamespace(
            executor=SimpleNamespace(driver_worker=SimpleNamespace(worker=object())),
            vllm_config=SimpleNamespace(
                compilation_config=SimpleNamespace(
                    static_forward_context={"actual-layer": SimpleNamespace(impl=impl)}
                )
            ),
        ),
    )
    result = collect_runtime(backend)
    assert result["gpu_uuid"] == NEW_UUID
    assert result["batch_invariance"] == batch
    assert result["attention_implementations"][0]["layer"] == "actual-layer"
    assert result["python_major_minor"] == list(sys.version_info[:2])
