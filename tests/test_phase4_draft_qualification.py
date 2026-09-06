from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
from test_phase4_d3_diagnostics import Tensor
from test_phase4_draft_logits_probe import reports_for, vector
from test_phase4_serial import phase4_config as _config_fixture
from test_phase4_vllm_draft import FakeWorker, next_token

from specrhythm.phase4.batched_draft_service import (
    BatchedDraftStateMachine,
    write_immutable_report,
)
from specrhythm.phase4.draft_admission import FLASH_IMPL, GPU_MODEL, MRV1, admit_device
from specrhythm.phase4.draft_logits_contract import PATHS, load_probe_fixture, summarize_logits
from specrhythm.phase4.draft_qualification import (
    D3_SCHEMA,
    STRUCTURAL_CHECKS,
    aggregate_directories,
    compatible_identity,
    hf_diagnostics,
    load_probe_qualification,
    qualify_d3,
    qualify_probe_reports,
    require_progression,
    validate_d3_directory,
)
from specrhythm.phase4.draft_qualification_gate import replay_production, state_snapshot
from specrhythm.phase4.manifest import sha256_file
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.vllm_draft_backend import VllmBatchedDraftBackend

phase4_config = _config_fixture


@pytest.mark.parametrize("invalid_limit", [False, True])
def test_serial_admission_binds_declared_workload_limits_before_execution(
    tmp_path, monkeypatch, invalid_limit
):
    from specrhythm.phase4 import draft_logits_probe, draft_qualification

    identity = {"execution_commit": "qualified"}
    monkeypatch.setattr(
        draft_qualification, "require_progression", lambda *a: {"run_identity": identity}
    )
    monkeypatch.setattr(draft_logits_probe, "preflight", lambda *a: (None, identity))
    requests = []
    for i, task in enumerate(("code", "code", "code", "chat", "summarization")):
        requests.append(
            {
                "request_id": f"r{i}",
                "task_class": task,
                "prompt_text": "<|im_start|>user\nHi<|im_start|>assistant"
                if task == "chat"
                else "Hi",
                "prompt_token_ids": [i + 1],
                "prompt_length": 1,
                "maximum_new_tokens": None if invalid_limit and i == 0 else 10 + i,
                "sampling_seed": 1,
                "tokenizer_fingerprint": "frozen",
            }
        )
    workload, reference, qualification, output = [
        tmp_path / n for n in ("workload", "reference", "qualification", "admission")
    ]
    workload.write_text("\n".join(json.dumps(r) for r in requests))
    reference.write_text("reference")
    qualification.write_text("qualification")
    monkeypatch.setattr(
        "sys.argv",
        [
            "qualify",
            "prepare-serial",
            "--qualification",
            str(qualification),
            "--stage",
            "D4",
            "--expected-commit",
            "qualified",
            "--config",
            "unused",
            "--workload",
            str(workload),
            "--reference",
            str(reference),
            "--backend",
            "hf",
            "--output",
            str(output),
        ],
    )
    if invalid_limit:
        with pytest.raises(ValueError, match="maximum_new_tokens"):
            draft_qualification.main()
        assert not output.exists()
    else:
        assert draft_qualification.main() == 0
        result = json.loads(output.read_text())
        assert result["workload_sha256"] == sha256_file(workload)
        assert result["requests"]["r4"] == {
            "maximum_new_tokens": 14,
            "prompt_token_count": 1,
            "prompt_token_ids_sha256": token_prefix_hash([5]),
        }


def runtime_provenance():
    return {
        "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
        "gpu_name": GPU_MODEL,
        "physical_gpu_id": 0,
        "logical_cuda_index": 0,
        "cuda_visible_devices": "0",
        "world_size": 1,
        "tensor_parallel_size": 1,
        "global_rank": 0,
        "startup_uuid_validation_count": 1,
        "python_major_minor": [3, 11],
        "python_version": "3.11.0",
        "dtype": "bfloat16",
        "runner_class": MRV1,
        "enforce_eager": True,
        "prefix_caching": False,
        "cache_initialized": True,
        "model_instance_count": 1,
        "block_size": 16,
        "kv_cache_group_count": 1,
        "max_num_seqs": 128,
        "max_num_batched_tokens": 4096,
        "vllm_api": {"pinned": True},
        "model": {"path": "/model"},
        "tokenizer": {"path": "/model"},
        "batch_invariance": {
            "compute_capability": "8.0",
            "dtype": "torch.bfloat16",
            "batch_invariant_env_raw": "1",
            "batch_invariant_requested": True,
            "batch_invariant_env_resolved": True,
            "batch_invariant_effective": True,
            "batch_invariant_validation": {"valid": True, "reasons": []},
            "cascade_attention_enabled": False,
            "vllm_dbo_enabled": False,
        },
        "attention_implementations": [
            {
                "layer": "layer.0",
                "implementation": FLASH_IMPL,
                "flash_attention_version": 2,
                "batch_invariant_enabled": True,
            }
        ],
    }


def probe_reports():
    rows = reports_for([16, 227, 227, 227])
    for value in rows.values():
        value["run_identity"] = {
            "execution_commit": "same",
            "config_sha256": "same",
            "model_path": "/model",
            "vllm_api": {"pinned": True},
            "model_files_sha256": {"model.safetensors": "frozen"},
        }
        value["diagnostic_frames"] = {
            "all_structural_checks_passed": True,
            "observer_removed": True,
        }
        value["runtime_provenance"] = runtime_provenance()
        value["gpu_identity"] = {
            k: runtime_provenance()[k]
            for k in (
                "gpu_uuid",
                "gpu_name",
                "physical_gpu_id",
                "logical_cuda_index",
                "cuda_visible_devices",
            )
        }
    return rows


def regime():
    return qualify_probe_reports(probe_reports())


def gate_inputs(count=8):
    proof, runtime = regime(), runtime_provenance()
    observation = {
        "requested_batch_size": count,
        "completed_requests": count,
        "run_identity": proof["run_identity"],
        "runtime_batch_invariance": runtime["batch_invariance"],
        "execution_started": True,
        "admission_valid": True,
        "structural_checks_executed": True,
        "admission": admit_device(runtime, proof["run_identity"], proof),
        "errors": [],
        "diagnostic_only": False,
        "materialization_observer_installed": False,
        "structural_checks": dict.fromkeys(STRUCTURAL_CHECKS, True),
        "comparisons": [
            {
                "request_id": "batch-8-7",
                "round": 2,
                "same_prefix": True,
                "exact": False,
                "hf_tokens": [11, 16, 17, 11],
                "vllm_tokens": [11, 227, 11, 227],
            }
        ],
    }
    backend = {
        "provenance": runtime,
        "execution_failed": False,
        "backend_shutdown_complete": True,
        "draft_live_requests_final": 0,
        "worker_resources": {
            "live_allocator_requests": 0,
            "worker_shutdown_complete": True,
            "blocks_allocated": 20,
            "blocks_freed": 20,
        },
        "draft_batch_statistics_by_purpose": {"proposal": {"histogram": {str(count): 3, "1": 1}}},
    }
    return observation, backend


def test_qualified_full_vllm_vectors_allow_hf_diagnostic_divergence():
    reports = probe_reports()
    before = copy.deepcopy(reports)
    qualified = qualify_probe_reports(reports)
    assert qualified["valid"] and qualified["vllm_batch_invariant_valid"]
    assert qualified["vllm_persistent_history_valid"]
    result = qualify_d3(*gate_inputs(), qualified)
    assert result["schema_version"] == D3_SCHEMA and result["d3_qualified"]
    assert result["vllm_semantic_valid"] and not result["hf_draft_exact"]
    assert result["hf_vllm_divergent_request_count"] == 1
    assert result["hf_vllm_divergent_rounds"] == [2]
    assert result["hf_vllm_divergence_classification"] == "hf_vllm_execution_numerical_divergence"
    assert result["hf_exact_diagnostic"]["exact"] is False
    assert result["mechanistic_root_cause_proven"] is False
    assert reports == before


@pytest.mark.parametrize("path", [PATHS[2], PATHS[3]])
def test_internal_vector_difference_blocks_even_if_top1_unchanged(path):
    rows = probe_reports()
    values = vector(227)
    values[42] += 0.125
    rows[path]["logits"] = summarize_logits(values, "torch.bfloat16")
    proof = qualify_probe_reports(rows)
    assert not proof["valid"]
    result = qualify_d3(*gate_inputs(), proof)
    assert not result["d3_qualified"]


@pytest.mark.parametrize(
    "fault", ["unclassified", "missing", "provenance", "structure", "raw-summary"]
)
def test_unclassified_or_invalid_execution_evidence_blocks(fault):
    reports = probe_reports()
    if fault == "unclassified":
        reports[PATHS[0]]["logits"] = summarize_logits(vector(227), "torch.bfloat16")
        reports[PATHS[0]]["top1_by_request"]["batch-8-7"] = 227
    elif fault == "missing":
        del reports[PATHS[2]]
    elif fault == "provenance":
        reports[PATHS[2]]["run_identity"] = {"changed_weights": True}
    elif fault == "structure":
        reports[PATHS[3]]["diagnostic_frames"]["all_structural_checks_passed"] = False
    else:
        reports[PATHS[2]]["logits"]["raw_logits_sha256_le_f32"] = "0" * 64
    assert not qualify_d3(*gate_inputs(), qualify_probe_reports(reports))["d3_qualified"]


@pytest.mark.parametrize("field", STRUCTURAL_CHECKS)
def test_no_structural_state_gate_can_be_waived_by_hf_policy(field):
    observation, backend = gate_inputs()
    observation["structural_checks"][field] = False
    result = qualify_d3(observation, backend, regime())
    assert not result["d3_qualified"] and not result["vllm_semantic_valid"]


@pytest.mark.parametrize("fault", ["cleanup", "fake-batch", "diagnostic", "count", "weights"])
def test_d3_runtime_controls_fail_closed(fault):
    observation, backend = gate_inputs()
    if fault == "cleanup":
        backend["worker_resources"]["blocks_freed"] = 19
    elif fault == "fake-batch":
        backend["draft_batch_statistics_by_purpose"]["proposal"]["histogram"] = {"1": 24}
    elif fault == "diagnostic":
        observation["materialization_observer_installed"] = True
    elif fault == "count":
        observation["completed_requests"] = 7
    else:
        observation["run_identity"] = {"changed_weights": True}
    assert not qualify_d3(observation, backend, regime())["d3_qualified"]


def test_provenance_allows_only_execution_commit_difference():
    assert compatible_identity(
        {"execution_commit": "old", "weights": "a"}, {"execution_commit": "new", "weights": "a"}
    )
    assert not compatible_identity({"weights": "a"}, {"weights": "b"})


def test_unmatched_hf_context_is_not_reported_as_exact():
    result = hf_diagnostics([{"same_prefix": False, "exact": None}])
    assert not result["hf_draft_exact"]
    assert result["hf_exact_diagnostic"]["unmatched_context_count"] == 1
    assert result["hf_vllm_divergent_request_count"] == 0


class HostStateWorker(FakeWorker):
    def __init__(self, oracle=None):
        super().__init__()
        self.views = {}
        self.runner = SimpleNamespace(requests={})
        self.pages = {}
        self.kv = SimpleNamespace(get_block_ids=lambda rid: (self.pages[rid],))
        self.oracle = oracle or {}
        self.sync_batch([])

    def sync_batch(self, ids):
        table = SimpleNamespace(
            blocks_per_kv_block=1,
            num_blocks_per_row=[len(self.pages[r]) for r in ids],
            get_numpy_array=lambda: Tensor([self.pages[r] for r in ids]),
        )
        self.runner.input_batch = SimpleNamespace(
            req_ids=ids,
            req_id_to_index={r: i for i, r in enumerate(ids)},
            num_prompt_tokens=[len(self.views[r].prompt_token_ids) for r in ids],
            token_ids_cpu=Tensor([list(self.views[r].prompt_token_ids) for r in ids]),
            num_computed_tokens_cpu=[self.runner.requests[r].num_computed_tokens for r in ids],
            block_table=SimpleNamespace(block_tables=[table]),
        )

    def materialize(self, rows, purpose):
        result = super().materialize(rows, purpose)
        for row in rows:
            self.pages.setdefault(row.request_id, [len(self.pages) + 1])
            self.views[row.request_id] = SimpleNamespace(prompt_token_ids=list(row.context))
            self.runner.requests[row.request_id] = SimpleNamespace(
                prompt_token_ids=list(row.context), num_computed_tokens=row.valid_length
            )
            result[row.request_id] = self.oracle.get(row.context, next_token(row.context))
        self.sync_batch([r.request_id for r in reversed(rows)])
        return result

    def request_evidence(self, rid):
        return {
            "draft_internal_request_id": rid,
            "draft_physical_request_block_observable": True,
            "draft_physical_request_block_identity": [self.pages[rid]],
        }

    def release(self, ids):
        super().release(ids)
        for rid in ids:
            del self.views[rid], self.pages[rid], self.runner.requests[rid]
        self.sync_batch([r for r in self.runner.input_batch.req_ids if r not in ids])


def original_oracle():
    history = load_probe_fixture()["history"]
    prefixes = {r: tuple(p) for r, p in history["initial"].items()}
    oracle = {}
    for item in history["rounds"]:
        for rid, tokens in item["expected"].items():
            for n, token in enumerate(tokens):
                oracle[prefixes[rid] + tuple(tokens[:n])] = token
        for row in item["synchronizations"]:
            prefixes[row["request_id"]] += tuple(row["committed_delta"])
    return history, oracle


def test_production_replay_commits_vllm_tokens_and_finishes_after_hf_divergence(phase4_config):
    history, oracle = original_oracle()
    prefix = tuple(load_probe_fixture()["rows"][-1]["committed_prefix"])
    actual = [11, 227, 11, 227]
    for index, token in enumerate(actual):
        oracle[prefix + tuple(actual[:index])] = token
    backend = VllmBatchedDraftBackend(phase4_config, worker=HostStateWorker(oracle))
    machine = BatchedDraftStateMachine(backend)
    result = replay_production(machine, history, 151936, state_snapshot)
    diagnostic = hf_diagnostics(result["comparisons"])
    assert diagnostic["hf_vllm_divergent_request_count"] == 1
    assert diagnostic["hf_vllm_divergent_rounds"] == [2]
    assert diagnostic["hf_exact_diagnostic"]["unmatched_context_count"] == 1
    assert result["completed_requests"] == 8 and result["cohort_sizes"] == [8, 7, 7, 7]
    assert result["eos_retired_requests"] == ["batch-8-0"]
    committed = next(
        r for r in result["commits"] if r["request_id"] == "batch-8-7" and r["round_id"] == 2
    )
    assert committed["decision"]["accepted_draft_token_ids"] == actual
    assert all(s["checks_passed"] for s in result["state_snapshots"])
    assert backend.report()["draft_batch_statistics_by_purpose"]["proposal"]["max"] == 8
    machine.shutdown()


@pytest.mark.parametrize("fault", ["alias", "row", "frontier", "prefix", "round", "retired"])
def test_host_state_gate_detects_real_identity_frontier_and_owner_faults(phase4_config, fault):
    backend = VllmBatchedDraftBackend(phase4_config, worker=HostStateWorker())
    machine = BatchedDraftStateMachine(backend)
    for rid, prefix in (("r1", (1, 2)), ("r2", (3, 4))):
        machine.initialize(rid, prefix, token_prefix_hash(prefix))
    worker = backend.worker
    if fault == "alias":
        worker.pages["sr-draft:r1"] = worker.pages["sr-draft:r2"]
    elif fault == "row":
        worker.runner.input_batch.req_id_to_index["sr-draft:r2"] = 42
    elif fault == "frontier":
        backend.states["r1"].materialized -= 1
    elif fault == "prefix":
        worker.views["sr-draft:r1"].prompt_token_ids[0] = 42
    elif fault == "round":
        backend.states["r1"].next_round += 1
    else:
        backend.retired.add("r1")
    with pytest.raises((ValueError, RuntimeError)):
        state_snapshot(backend, machine, committed=True)


def write_d3(directory, count=8, *, current_uuid=None):
    observation, backend = gate_inputs(count)
    startup = backend["provenance"]
    proof = regime()
    if current_uuid:
        startup["gpu_uuid"] = current_uuid
        observation["admission"] = admit_device(startup, proof["run_identity"], proof)
    for name, value in (
        ("observation.json", observation),
        ("draft-backend-report.json", backend),
        ("regime-qualification.json", proof),
        ("admission.json", observation["admission"]),
        ("draft-startup.json", startup),
        ("hf-oracle.json", {"immutable": True}),
    ):
        write_immutable_report(directory / name, value)
    gate = qualify_d3(observation, backend, proof)
    gate["artifact_sha256"] = {p.name: sha256_file(p) for p in directory.glob("*.json")}
    write_immutable_report(directory / "gate.json", gate)
    return backend, startup


def write_aggregate_inputs(tmp_path, *, current_uuid=None):
    for count in (2, 4, 8):
        backend, startup = write_d3(tmp_path / f"D3-B{count}", count, current_uuid=current_uuid)
    for name in ("D1", "D2"):
        old = {
            "schema_version": "specrhythm.phase4b3-draft-gate.v1",
            "gate": name,
            "valid": True,
            "errors": [],
            "diagnostic_only": False,
            "comparisons": [{"draft_proposals_exact": True}],
        }
        for filename, value in (
            ("gate.json", old),
            ("draft-startup.json", startup),
            ("draft-backend-report.json", backend),
        ):
            write_immutable_report(tmp_path / name / filename, value)


def test_aggregator_binds_five_gates_without_mutating_old_artifacts(tmp_path):
    write_aggregate_inputs(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    result = aggregate_directories(tmp_path)
    assert result["valid"], result["errors"]
    path = tmp_path / "qualification.json"
    write_immutable_report(path, result)
    assert require_progression(path, "D4") == result
    with pytest.raises(ValueError, match="D5 requires"):
        require_progression(path, "D5")
    assert all(p.read_bytes() == b for p, b in before.items())
    (tmp_path / "D3-B8/observation.json").write_text("{}")
    with pytest.raises(ValueError, match="D4/D5"):
        require_progression(path, "D4")


def test_old_or_missing_d3_report_cannot_admit_d4_d5(tmp_path):
    write_immutable_report(
        tmp_path / "qualification.json", {"schema_version": "old", "valid": True}
    )
    for stage in ("D4", "D5"):
        with pytest.raises(ValueError, match="D4/D5"):
            require_progression(tmp_path / "qualification.json", stage)


def test_d3_bound_artifact_mutation_is_detected(tmp_path):
    write_d3(tmp_path)
    assert not validate_d3_directory(tmp_path)[1]
    (tmp_path / "hf-oracle.json").write_text('{"changed":true}')
    assert "D3 bound artifact changed: hf-oracle.json" in validate_d3_directory(tmp_path)[1]


def test_probe_import_recomputes_vectors_and_checks_immutable_artifact_hashes(tmp_path):
    from specrhythm.phase4.draft_logits_contract import classify_paths

    reports = probe_reports()
    for path, value in reports.items():
        write_immutable_report(tmp_path / f"{path}.json", value)
    write_immutable_report(
        tmp_path / "probe-inputs.json",
        {"fixture": load_probe_fixture(), "run_identity": reports[PATHS[0]]["run_identity"]},
    )
    classification = classify_paths(load_probe_fixture(), reports)
    classification["path_artifact_sha256"] = {
        p: sha256_file(tmp_path / f"{p}.json") for p in PATHS
    }
    write_immutable_report(tmp_path / "classification.json", classification)
    assert load_probe_qualification(tmp_path)["valid"]
    before = (tmp_path / "classification.json").read_bytes()
    changed = reports[PATHS[3]]
    changed["logits"]["raw_logits_float32"][42] = 9.0
    (tmp_path / f"{PATHS[3]}.json").write_text(json.dumps(changed))
    assert not load_probe_qualification(tmp_path)["valid"]
    assert (tmp_path / "classification.json").read_bytes() == before
