from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Optional

import pytest

from specrhythm.cli import build_parser
from specrhythm.phase4.decode_ready import (
    DecodeReadyProvenance,
    ResidentSetupObservation,
    ResidentWarmStartProvider,
)
from specrhythm.phase4.performance import (
    GATE3_QUALIFICATION,
    LEGACY_SERIAL_METADATA_COMMIT,
    build_decode_performance_result,
    compare_decode_performance_results,
    percentile,
    summarize_metrics,
)
from specrhythm.phase4.performance_boundary import (
    PERFORMANCE_COMMIT_SCHEMA,
    PERFORMANCE_EVENT,
    PERFORMANCE_EVENT_SCHEMA,
    extract_performance_boundary,
    publish_performance_boundary,
)
from specrhythm.phase4.process_lifecycle import LIFECYCLE_SCHEMA
from specrhythm.phase4.resident_setup import (
    build_deferred_initial_proposals_ready,
    build_setup_ready,
    load_deferred_initial_proposals_ready,
    validate_setup_ready,
)
from specrhythm.phase4.serial import PROTOCOL_VERSION, Proposal
from specrhythm.phase4.serial_runner import _phase4b2_serial_execution_evidence
from specrhythm.phase4.stock_vllm import SmokeRequest
from specrhythm.phase4.transport import CheckpointJsonl

START = 1_000_000_000
REQUESTS = ("a", "b")
PINNED_VLLM_COMMIT = "752a3a504485790a2e8491cacbb35c137339ad34"


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _checkpoint(path: Path, rows: list[dict[str, object]]) -> None:
    log = CheckpointJsonl(path)
    if not rows:
        path.touch()
    for row in rows:
        log.append(row)


def _make_run(
    tmp_path: Path,
    mode: str,
    *,
    one_token: bool = False,
    execution_commit: str = "f" * 40,
    native_workload: bool = False,
) -> tuple[Path, dict[str, Path]]:
    root = tmp_path / mode
    root.mkdir()
    workload = tmp_path / "workload.jsonl"
    if not workload.exists():
        workload.write_text(
            "".join(
                json.dumps(
                    {
                        "request_id": request_id,
                        "task_class": "code",
                        "prompt_text": "source-a" if request_id == "a" else "source-b",
                        "prompt_token_ids": [1, 2],
                        "prompt_length": 2,
                        "maximum_new_tokens": 2 if one_token else 3,
                        "sampling_seed": 1664,
                        "tokenizer_fingerprint": "fixture-tokenizer",
                    } if native_workload else {
                        "request_id": request_id,
                        "input_tokens": 2,
                        "output_tokens": 2 if one_token else 3,
                    },
                    sort_keys=True,
                )
                + "\n"
                for request_id in REQUESTS
            ),
            encoding="utf-8",
        )
    config = tmp_path / "config.json"
    topology = tmp_path / "topology.json"
    patch = tmp_path / "patch.json"
    for path, value in (
        (config, {"config": "same"}),
        (topology, {"gpus": [0, 1, 2]}),
        (patch, {"patch_stack": ["same"]}),
    ):
        if not path.exists():
            _json(path, value)
    from specrhythm.phase4.manifest import sha256_file

    observations = tuple(
        ResidentSetupObservation(
            request_id=request_id,
            internal_target_request_id=f"internal-{request_id}",
            prompt_token_ids=(1, 2),
            bootstrap_token_id=100 + index,
            target_materialized_kv_token_count=2,
            target_num_computed_tokens=2,
            draft_materialized_kv_token_count=3,
            bootstrap_ready_ns=20 + index,
            draft_initialization_complete_ns=30 + index,
        )
        for index, request_id in enumerate(REQUESTS)
    )
    manifest = ResidentWarmStartProvider().prepare(
        observations,
        DecodeReadyProvenance(
            specrhythm_git_commit=execution_commit,
            vllm_version="0.25.1",
            vllm_commit=PINNED_VLLM_COMMIT,
            vllm_patch_stack_sha256=("a" * 64, "b" * 64, "c" * 64),
            target_model_path="/models/target",
            target_model_revision="target-revision",
            draft_model_path="/models/draft",
            draft_model_revision="draft-revision",
            tokenizer_revision="tokenizer-revision",
            workload_sha256=sha256_file(workload),
            sampling_configuration={"temperature": 0.0},
            batch_invariant_configuration={"requested": True},
            target_physical_gpu_ids=(1, 2),
            draft_physical_gpu_ids=(0,),
            target_tensor_parallel_size=2,
            draft_tensor_parallel_size=1,
        ),
        setup_start_ns=10,
        setup_complete_ns=40,
        global_barrier_ns=50,
        measurement_start_ns=60,
    )
    _json(root / "decode-ready-manifest.json", manifest.to_dict())
    _json(
        root / "setup-ready.json",
        {
            "global_decode_ready": True,
            "ready_published_ns": START - 10,
        },
    )
    _checkpoint(
        root / "timing-events.jsonl",
        [
            {
                "schema_version": PERFORMANCE_EVENT_SCHEMA,
                "event": PERFORMANCE_EVENT,
                "timestamp_ns": START,
                "consumer": {
                    "target": "target-only",
                    "serial": "serial",
                    "dual-batch": "dual-batch",
                }[mode],
                "setup_ready_published_ns": START - 10,
                "pre_measurement_tp_barrier": True,
                "pre_measurement_target_cuda_synchronize": True,
                "setup_excluded": True,
                "bootstrap_excluded_from_measured_tokens": True,
            }
        ],
    )
    tokens_per_request = 1 if one_token else 2
    outputs = []
    diagnostics = []
    round_rows = []
    proposal_rows = []
    lifecycle_rows = []
    for request_index, request_id in enumerate(REQUESTS):
        bootstrap = 100 + request_index
        measured = [200 + request_index * 10 + offset for offset in range(tokens_per_request)]
        outputs.append(
            {
                "request_id": request_id,
                "prompt_length": 2,
                "generated_token_ids": [bootstrap, *measured],
                "finish_reason": "length",
                "stop_reason": None,
            }
        )
        for offset, token in enumerate(measured):
            timestamp = START + (request_index + 1) * 1_000_000 + offset * 2_000_000
            if mode == "target":
                diagnostics.append(
                    {
                        "request_id": request_id,
                        "proposal_token_ids": [],
                        "selected_target_token_id": [token],
                        "target_forward_start_ns": timestamp - 100,
                        "target_forward_end_ns": timestamp,
                    }
                )
                CheckpointJsonl(root / "timing-events.jsonl").append(
                    {
                        "schema_version": PERFORMANCE_COMMIT_SCHEMA,
                        "event": "measured-token-commit",
                        "timestamp_ns": timestamp,
                        "request_id": request_id,
                        "token_ids": [token],
                        "source": "resident-target-sampled-commit",
                        "per_token_cuda_synchronize": False,
                    }
                )
            elif mode == "serial":
                round_rows.append(
                    {
                        "request_id": request_id,
                        "round_id": offset,
                        "committed_token_ids": [token],
                        "timeline": {
                            "draft_start_ns": START + 100 + offset,
                            "state_sync_end_ns": timestamp,
                        },
                    }
                )
            else:
                proposal_rows.append(
                    {
                        "request_id": request_id,
                        "round_id": offset,
                        "committed_token_ids": [token],
                        "commit_end_ns": timestamp,
                    }
                )
                lifecycle_rows.append(
                    {
                        "request_id": request_id,
                        "round_id": offset,
                        "lifecycle_state": "CREATED",
                        "draft_start_ns": START + 100 + offset,
                    }
                )
    _checkpoint(root / "target-diagnostics.jsonl", diagnostics)
    if mode == "serial":
        _checkpoint(root / "round-events.jsonl", round_rows)
        ready_by_id = {row.request_id: row for row in manifest.requests}
        proposals = tuple(
            Proposal(
                protocol_version=PROTOCOL_VERSION,
                request_id=request_id,
                round_id=0,
                parent_prefix_len=ready_by_id[
                    request_id
                ].logical_committed_prefix_count,
                parent_prefix_hash=ready_by_id[
                    request_id
                ].logical_committed_prefix_sha256,
                proposal_token_ids=(300 + index,),
                proposal_eos=False,
                draft_start_ns=START + 100,
                draft_end_ns=START + 200,
                transport_payload_bytes=1,
                model_provenance={"model": "draft"},
                runtime_provenance={"runtime": "test"},
            )
            for index, request_id in enumerate(REQUESTS)
        )
        _json(
            root / "initial-proposals-ready.json",
            build_deferred_initial_proposals_ready(
                manifest,
                proposals=proposals,
                performance_measurement_start_ns=START,
                published_ns=START + 300,
            ),
        )
    if mode == "dual-batch":
        _checkpoint(root / "proposal-events.jsonl", proposal_rows)
        _checkpoint(root / "proposal-lifecycle-events.jsonl", lifecycle_rows)
        _checkpoint(
            root / "overlap-events.jsonl",
            [{"overlap_duration_ns": 500, "cycle_id": 0}],
        )
    plugin = {
        "schema_version": f"test-{mode}",
        "proposal_generation": False if mode == "target" else None,
    }
    _json(root / "plugin-report.json", plugin)
    _json(
        root / "process-lifecycle.json",
        {
            "schema_version": LIFECYCLE_SCHEMA,
            "coordinator_pid": 10,
            "pgid": 10,
            "session_id": 10,
            "child_reap_result": {"coordinator_reaped": True},
            "remaining_owned_pids": [],
            "cleanup_valid": True,
            "run_valid": True,
        },
    )
    (root / "timestamped-target-log.jsonl").write_text(
        json.dumps({"timestamp_ns": START + 1, "line": "decode begins"}) + "\n",
        encoding="utf-8",
    )
    raw_name = {
        "target": "resident-target.json",
        "serial": "resident-serial.json",
        "dual-batch": "resident-dual.json",
    }[mode]
    final_sync = [
        {
            "global_rank": rank,
            "local_rank": rank,
            "world_size": 2,
            "logical_cuda_index": rank,
            "physical_gpu_id": rank + 1,
            "gpu_uuid": f"GPU-{rank + 1}",
            "final_cuda_synchronize_complete_ns": START + 10_000_000 + rank,
        }
        for rank in range(2)
    ]
    paths = {
        "workload": workload,
        "config": config,
        "topology": topology,
        "patch": patch,
    }
    raw = {
        "valid": True,
        "phase4b2_performance_candidate": True,
        "phase4b2_initial_proposals_ready": (
            {
                "file": "initial-proposals-ready.json",
                "sha256": sha256_file(root / "initial-proposals-ready.json"),
            }
            if mode == "serial"
            else None
        ),
        "outputs": outputs,
        "phase4b2_final_sync": final_sync,
    }
    if mode == "serial":
        reference = tmp_path / "stock-target-reference.json"
        if not reference.exists():
            _json(reference, {"reference": "same"})
        draft_ready = root / "draft-service-ready.json"
        draft_ready_value = {
            "schema_version": "specrhythm.phase4-draft-service-ready.v1",
            "provenance": {"physical_gpu_id": 0, "model": "draft"},
        }
        _json(draft_ready, draft_ready_value)
        active_patch_sha = "d" * 64
        raw.update(
            {
                "schema_version": "specrhythm.phase4-serial-disaggregated-run.v1",
                "mode": "serial-disaggregated",
                "provider_kind": "resident-warm-start",
                "correctness_mode": "batch-invariant",
                "request_count": len(REQUESTS),
                "target_runtime_configuration": {
                    "physical_gpu_ids": [1, 2],
                    "tensor_parallel_size": 2,
                },
                "engine_residency": {
                    "draft": {
                        "service_provenance": draft_ready_value["provenance"]
                    }
                },
                "stock_reference": {
                    "file": reference.name,
                    "file_sha256": sha256_file(reference),
                },
                "patch_manifest": {
                    "file": patch.name,
                    "file_sha256": sha256_file(patch),
                    "patch_sha256": active_patch_sha,
                },
                "provenance": {
                    "git_commit": execution_commit,
                    "config_sha256": sha256_file(config),
                    "workload_sha256": sha256_file(workload),
                    "vllm_source_commit": PINNED_VLLM_COMMIT,
                },
            }
        )
        _json(
            root / "runtime-manifest.json",
            {
                "schema_version": "specrhythm.phase4-runtime-bundle.v1",
                "stage": "phase4a1-serial-disaggregated-correctness",
                "roles": {
                    "target": {
                        "role": "target",
                        "git_commit": execution_commit,
                        "framework": {
                            "source_commit": PINNED_VLLM_COMMIT,
                            "vllm_patch_sha256": active_patch_sha,
                        },
                        "correctness": {"mode": "batch-invariant"},
                        "engine": {
                            "physical_gpu_ids": [1, 2],
                            "tensor_parallel_size": 2,
                        },
                        "inputs": {
                            "config_sha256": sha256_file(config),
                            "workload_sha256": sha256_file(workload),
                            "topology_sha256": sha256_file(topology),
                        },
                    }
                },
                "phase4a1": {
                    "mode": "serial-disaggregated",
                    "correctness_mode": "batch-invariant",
                    "phase4b2_performance_candidate": True,
                    "phase4b2_final_sync": final_sync,
                    "draft_service_ready_file": draft_ready.name,
                    "draft_service_ready_sha256": sha256_file(draft_ready),
                    "draft_service": draft_ready_value,
                    "stock_reference_file": reference.name,
                    "stock_reference_sha256": sha256_file(reference),
                    "patch_manifest_file": patch.name,
                    "patch_manifest_sha256": sha256_file(patch),
                },
            },
        )
        paths["reference"] = reference
        paths["draft_ready"] = draft_ready
    _json(root / raw_name, raw)
    return root, paths


def _measure(tmp_path: Path, mode: str, *, one_token: bool = False) -> dict[str, object]:
    root, paths = _make_run(tmp_path, mode, one_token=one_token)
    return build_decode_performance_result(
        mode=mode,
        run_root=root,
        workload_path=paths["workload"],
        config_path=paths["config"],
        topology_path=paths["topology"],
        patch_manifest_path=paths["patch"],
        output_path=root / "decode-performance.json",
    )


def _legacy_serial_run(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    root, paths = _make_run(
        tmp_path,
        "serial",
        execution_commit=LEGACY_SERIAL_METADATA_COMMIT,
    )
    raw_path = root / "resident-serial.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw.pop("phase4b2_performance_candidate")
    raw.pop("phase4b2_final_sync")
    _json(raw_path, raw)
    return root, paths


def _build_result(root: Path, paths: dict[str, Path], mode: str) -> dict[str, object]:
    return build_decode_performance_result(
        mode=mode,
        run_root=root,
        workload_path=paths["workload"],
        config_path=paths["config"],
        topology_path=paths["topology"],
        patch_manifest_path=paths["patch"],
        output_path=root / "decode-performance.json",
    )


def test_serial_execution_evidence_is_shared_by_runtime_and_top_level_result() -> None:
    rows = [{"global_rank": 0}]
    evidence = _phase4b2_serial_execution_evidence(
        phase4b2_performance=True,
        final_sync_rows=rows,
        stock_comparison_exact=False,
    )
    assert evidence["phase4b2_performance_candidate"] is True
    assert evidence["phase4b2_final_sync"] == rows
    assert evidence["historical_gate3_qualification"][
        "phase4b2_progression_permitted"
    ] is True
    source = (
        Path(__file__).resolve().parents[1]
        / "src/specrhythm/phase4/serial_runner.py"
    ).read_text(encoding="utf-8")
    assert source.count("**phase4b2_evidence") == 2


def test_future_serial_uses_raw_phase4b2_metadata(tmp_path: Path) -> None:
    result = _measure(tmp_path, "serial")
    assert result["valid"] is True
    assert result["phase4b2_metadata_provenance"] == {
        "performance_candidate_source": "raw-run",
        "final_sync_source": "raw-run",
        "legacy_serial_metadata_recovered": False,
        "recovery_allowed_mode": "serial-only",
        "legacy_execution_commit": LEGACY_SERIAL_METADATA_COMMIT,
    }


def test_matching_legacy_serial_metadata_is_recovered_auditably(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    measurement_commit = "e" * 40
    monkeypatch.setattr(
        "specrhythm.phase4.performance._measurement_code_git_commit",
        lambda: measurement_commit,
    )
    root, paths = _legacy_serial_run(tmp_path)
    result = _build_result(root, paths, "serial")
    provenance = result["phase4b2_metadata_provenance"]
    assert result["valid"] is True
    assert result["errors"] == []
    assert result["performance_result"] is True
    assert result["execution_git_commit"] == LEGACY_SERIAL_METADATA_COMMIT
    assert result["measurement_code_git_commit"] == measurement_commit
    assert result["measurement_code_git_commit"] != result["execution_git_commit"]
    assert provenance["legacy_serial_metadata_recovered"] is True
    assert provenance["performance_candidate_source"] == "runtime-manifest.phase4a1"
    assert provenance["final_sync_source"] == "runtime-manifest.phase4a1"
    assert provenance["recovery_validation_errors"] == []


@pytest.mark.parametrize(
    ("missing", "replacement"),
    [
        ("phase4b2_final_sync", {"phase4b2_performance_candidate": False}),
        (None, {"phase4b2_performance_candidate": False}),
        (None, {"phase4b2_final_sync": [{"malformed": True}]}),
    ],
)
def test_raw_serial_contradiction_never_uses_runtime_fallback(
    tmp_path: Path, missing: Optional[str], replacement: dict[str, object]
) -> None:
    root, paths = _make_run(tmp_path, "serial")
    raw_path = root / "resident-serial.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if missing is not None:
        raw.pop(missing)
    raw.update(replacement)
    _json(raw_path, raw)
    result = _build_result(root, paths, "serial")
    assert result["valid"] is False
    assert result["phase4b2_metadata_provenance"][
        "legacy_serial_metadata_recovered"
    ] is False


@pytest.mark.parametrize(
    "failure",
    (
        "candidate-false",
        "wrong-runtime-schema",
        "missing-phase4a1",
        "malformed-sync",
        "wrong-ranks",
        "wrong-gpus",
        "wrong-world-size",
    ),
)
def test_legacy_serial_runtime_metadata_must_be_exact(
    tmp_path: Path, failure: str
) -> None:
    root, paths = _legacy_serial_run(tmp_path)
    runtime_path = root / "runtime-manifest.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    phase = runtime["phase4a1"]
    if failure == "candidate-false":
        phase["phase4b2_performance_candidate"] = False
    elif failure == "wrong-runtime-schema":
        runtime["schema_version"] = "wrong"
    elif failure == "missing-phase4a1":
        runtime.pop("phase4a1")
    elif failure == "malformed-sync":
        phase["phase4b2_final_sync"] = {}
    elif failure == "wrong-ranks":
        phase["phase4b2_final_sync"][1]["global_rank"] = 0
    elif failure == "wrong-gpus":
        phase["phase4b2_final_sync"][1]["physical_gpu_id"] = 0
    else:
        phase["phase4b2_final_sync"][1]["world_size"] = 1
    _json(runtime_path, runtime)
    result = _build_result(root, paths, "serial")
    assert result["valid"] is False
    assert result["phase4b2_metadata_provenance"][
        "legacy_serial_metadata_recovered"
    ] is False


def test_legacy_serial_recovery_requires_runtime_manifest(tmp_path: Path) -> None:
    root, paths = _legacy_serial_run(tmp_path)
    (root / "runtime-manifest.json").unlink()
    result = _build_result(root, paths, "serial")
    assert result["valid"] is False
    assert any("runtime-manifest.json" in error for error in result["errors"])


@pytest.mark.parametrize(
    "failure",
    (
        "commit",
        "workload",
        "config",
        "patch",
        "request-count",
        "topology",
        "placement",
        "tp-size",
    ),
)
def test_legacy_serial_recovery_rejects_provenance_mismatch(
    tmp_path: Path, failure: str
) -> None:
    root, paths = _legacy_serial_run(tmp_path)
    raw_path = root / "resident-serial.json"
    runtime_path = root / "runtime-manifest.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    role = runtime["roles"]["target"]
    if failure == "commit":
        role["git_commit"] = "e" * 40
    elif failure == "workload":
        role["inputs"]["workload_sha256"] = "e" * 64
    elif failure == "config":
        role["inputs"]["config_sha256"] = "e" * 64
    elif failure == "patch":
        runtime["phase4a1"]["patch_manifest_sha256"] = "e" * 64
    elif failure == "request-count":
        raw["request_count"] = 3
    elif failure == "topology":
        role["inputs"]["topology_sha256"] = "e" * 64
    elif failure == "placement":
        role["engine"]["physical_gpu_ids"] = [0, 1]
    else:
        role["engine"]["tensor_parallel_size"] = 1
    _json(raw_path, raw)
    _json(runtime_path, runtime)
    result = _build_result(root, paths, "serial")
    assert result["valid"] is False
    assert result["phase4b2_metadata_provenance"][
        "legacy_serial_metadata_recovered"
    ] is False


def test_recovered_sync_still_passes_through_authoritative_validator(
    tmp_path: Path,
) -> None:
    root, paths = _legacy_serial_run(tmp_path)
    runtime_path = root / "runtime-manifest.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    for row in runtime["phase4a1"]["phase4b2_final_sync"]:
        row["final_cuda_synchronize_complete_ns"] = START + 1
    _json(runtime_path, runtime)
    result = _build_result(root, paths, "serial")
    assert result["valid"] is False
    assert result["phase4b2_metadata_provenance"][
        "legacy_serial_metadata_recovered"
    ] is True
    assert any("predates a measured commit" in error for error in result["errors"])


@pytest.mark.parametrize("mode", ("target", "dual-batch"))
def test_non_serial_modes_never_recover_missing_raw_metadata(
    tmp_path: Path, mode: str
) -> None:
    root, paths = _make_run(tmp_path, mode)
    raw_path = root / {
        "target": "resident-target.json",
        "dual-batch": "resident-dual.json",
    }[mode]
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw.pop("phase4b2_performance_candidate")
    raw.pop("phase4b2_final_sync")
    _json(raw_path, raw)
    _json(
        root / "runtime-manifest.json",
        {"phase4a1": {"phase4b2_performance_candidate": True}},
    )
    result = _build_result(root, paths, mode)
    assert result["valid"] is False
    assert result["phase4b2_metadata_provenance"][
        "legacy_serial_metadata_recovered"
    ] is False


def test_boundary_excludes_setup_and_counts_first_post_bootstrap_token(tmp_path: Path) -> None:
    result = _measure(tmp_path, "target")
    assert result["valid"] is True
    request = result["requests"][0]
    assert request["setup_committed_output_tokens"] == 1
    assert request["measured_committed_output_token_ids"] == [200, 201]
    assert request["total_generated_token_ids"] == [100, 200, 201]
    assert request["token_accounting_valid"] is True
    assert result["measurement"]["setup_excluded"] is True


def test_metrics_tpot_throughput_makespan_and_quantiles() -> None:
    rows = [
        {
            "final_measured_commit_ns": 2_000_000_000,
            "decode_latency_ms": 1000.0,
            "tpot_ms": 2.0,
            "measured_committed_output_token_count": 3,
        },
        {
            "final_measured_commit_ns": 3_000_000_000,
            "decode_latency_ms": 2000.0,
            "tpot_ms": None,
            "measured_committed_output_token_count": 1,
        },
    ]
    summary = summarize_metrics(rows, 1_000_000_000)
    assert summary["decode_makespan_ms"] == 2000.0
    assert summary["aggregate_throughput_tokens_per_second"] == 2.0
    assert summary["tpot_ms"]["mean"] == 2.0
    assert summary["tpot_ms"]["undefined_one_token_request_count"] == 1
    assert percentile([1.0, 3.0], 0.50) == 2.0
    assert summary["decode_latency_ms"]["p90"] == pytest.approx(1900.0)


def test_one_measured_token_has_null_tpot(tmp_path: Path) -> None:
    result = _measure(tmp_path, "target", one_token=True)
    assert all(row["tpot_ms"] is None for row in result["requests"])


def test_target_draft_inactivity_fails_closed(tmp_path: Path) -> None:
    root, paths = _make_run(tmp_path, "target")
    plugin = json.loads((root / "plugin-report.json").read_text(encoding="utf-8"))
    plugin["proposal_generation"] = True
    _json(root / "plugin-report.json", plugin)
    result = build_decode_performance_result(
        mode="target",
        run_root=root,
        workload_path=paths["workload"],
        config_path=paths["config"],
        topology_path=paths["topology"],
        patch_manifest_path=paths["patch"],
        output_path=root / "decode-performance.json",
    )
    assert result["valid"] is False
    assert any("Draft proposal" in error for error in result["errors"])


def test_serial_initial_proposal_and_dual_overlap_contracts(tmp_path: Path) -> None:
    serial = _measure(tmp_path, "serial")
    dual = _measure(tmp_path, "dual-batch")
    assert serial["valid"] is True
    assert dual["valid"] is True
    assert serial["mode_semantics"]["draft_target_overlap"] is False
    assert dual["mode_semantics"]["per_round_global_cuda_synchronize"] is False


def test_setup_leakage_invalidates_performance(tmp_path: Path) -> None:
    root, paths = _make_run(tmp_path, "serial")
    rows = CheckpointJsonl(root / "round-events.jsonl").read()
    (root / "round-events.jsonl").unlink()
    rows[0]["timeline"]["draft_start_ns"] = START - 1
    for row in rows:
        row.pop("record_sha256", None)
    _checkpoint(root / "round-events.jsonl", rows)
    result = build_decode_performance_result(
        mode="serial",
        run_root=root,
        workload_path=paths["workload"],
        config_path=paths["config"],
        topology_path=paths["topology"],
        patch_manifest_path=paths["patch"],
        output_path=root / "decode-performance.json",
    )
    assert result["valid"] is False
    assert any("initial proposal" in error for error in result["errors"])


def test_unknown_commit_request_invalidates_accounting(tmp_path: Path) -> None:
    root, paths = _make_run(tmp_path, "target")
    CheckpointJsonl(root / "timing-events.jsonl").append(
        {
            "schema_version": PERFORMANCE_COMMIT_SCHEMA,
            "event": "measured-token-commit",
            "timestamp_ns": START + 1,
            "request_id": "unknown",
            "token_ids": [999],
            "source": "test",
            "per_token_cuda_synchronize": False,
        }
    )
    result = build_decode_performance_result(
        mode="target",
        run_root=root,
        workload_path=paths["workload"],
        config_path=paths["config"],
        topology_path=paths["topology"],
        patch_manifest_path=paths["patch"],
        output_path=root / "decode-performance.json",
    )
    assert result["valid"] is False
    assert any("unknown request ID" in error for error in result["errors"])


def test_cleanup_failure_invalidates_performance(tmp_path: Path) -> None:
    root, paths = _make_run(tmp_path, "target")
    lifecycle = json.loads((root / "process-lifecycle.json").read_text(encoding="utf-8"))
    lifecycle["cleanup_valid"] = False
    lifecycle["run_valid"] = False
    _json(root / "process-lifecycle.json", lifecycle)
    result = build_decode_performance_result(
        mode="target",
        run_root=root,
        workload_path=paths["workload"],
        config_path=paths["config"],
        topology_path=paths["topology"],
        patch_manifest_path=paths["patch"],
        output_path=root / "decode-performance.json",
    )
    assert result["valid"] is False
    assert result["cleanup_valid"] is False


def test_warmup_jit_is_provenance_not_silently_removed(tmp_path: Path) -> None:
    root, paths = _make_run(tmp_path, "target")
    (root / "timestamped-target-log.jsonl").write_text(
        json.dumps({"timestamp_ns": START + 1, "line": "Triton JIT compiling"}) + "\n",
        encoding="utf-8",
    )
    result = build_decode_performance_result(
        mode="target",
        run_root=root,
        workload_path=paths["workload"],
        config_path=paths["config"],
        topology_path=paths["topology"],
        patch_manifest_path=paths["patch"],
        output_path=root / "decode-performance.json",
    )
    assert result["valid"] is True
    assert result["warmup_clean"] is False
    assert result["jit_observation"]["post_measurement_jit_event_count"] == 1


def test_matched_pair_then_full_triangle_enables_speedups(tmp_path: Path) -> None:
    paths = {}
    for mode in ("target", "serial", "dual-batch"):
        result = _measure(tmp_path, mode)
        paths[mode] = tmp_path / mode / "decode-performance.json"
        assert result["valid"] is True
    pair = compare_decode_performance_results(
        target_path=paths["target"],
        serial_path=paths["serial"],
        output_path=tmp_path / "pair.json",
        markdown_path=tmp_path / "pair.md",
    )
    assert pair["valid"] is True
    assert pair["performance_valid"] is False
    assert pair["performance_valid_for_pair"] is True
    assert pair["speedups"]["target_vs_serial"]["makespan_speedup"] == 1.0
    comparison = compare_decode_performance_results(
        target_path=paths["target"],
        serial_path=paths["serial"],
        dual_path=paths["dual-batch"],
        output_path=tmp_path / "comparison.json",
        markdown_path=tmp_path / "comparison.md",
    )
    assert comparison["performance_valid"] is True
    assert comparison["exact_correctness_triangle"]["valid"] is True
    assert comparison["speedups"] is not None


@pytest.mark.parametrize(
    "field", ["measured_committed_output_token_ids", "finish_reason", "termination_reason"]
)
def test_exact_cross_mode_mismatch_is_diagnostic(tmp_path: Path, field: str) -> None:
    for mode in ("target", "serial", "dual-batch"):
        _measure(tmp_path, mode)
    serial_path = tmp_path / "serial" / "decode-performance.json"
    serial = json.loads(serial_path.read_text(encoding="utf-8"))
    row = serial["requests"][0]
    if field.endswith("token_ids"):
        row[field][0] = 999
        row["total_generated_token_ids"][1] = 999
    else:
        row[field] = "stop"
    _json(serial_path, serial)
    comparison = compare_decode_performance_results(
        target_path=tmp_path / "target" / "decode-performance.json",
        serial_path=serial_path,
        dual_path=tmp_path / "dual-batch" / "decode-performance.json",
        output_path=tmp_path / f"bad-{field}.json",
        markdown_path=tmp_path / f"bad-{field}.md",
    )
    assert comparison["performance_valid"] is True
    assert comparison["matched_work_comparability"]["valid"] is True
    assert comparison["speedups"]["target_vs_serial"]["makespan_speedup"] == 1.0
    assert comparison["exact_correctness_triangle"]["valid"] is False
    diagnostic = comparison["exact_sequence_diagnostic"]
    assert diagnostic["all_equal"] is (not field.endswith("token_ids"))
    if field.endswith("token_ids"):
        assert diagnostic["divergent_request_count"] == 1
        assert diagnostic["matching_request_count"] == 1
        assert diagnostic["first_mismatches"][0]["request_id"] == "a"
    else:
        differences = comparison["matched_work_comparability"]["termination_differences"]
        assert differences[0]["field"] == field
        assert differences[0]["fixed_length_completed"] is True


@pytest.mark.parametrize("kind", ["topology", "workload"])
def test_provenance_mismatch_fails_closed(tmp_path: Path, kind: str) -> None:
    for mode in ("target", "serial", "dual-batch"):
        _measure(tmp_path, mode)
    dual_path = tmp_path / "dual-batch" / "decode-performance.json"
    dual = json.loads(dual_path.read_text(encoding="utf-8"))
    if kind == "topology":
        dual["artifact_sha256"]["topology"] = "0" * 64
    else:
        dual["workload_sha256"] = "0" * 64
    _json(dual_path, dual)
    comparison = compare_decode_performance_results(
        target_path=tmp_path / "target" / "decode-performance.json",
        serial_path=tmp_path / "serial" / "decode-performance.json",
        dual_path=dual_path,
        output_path=tmp_path / f"bad-{kind}.json",
        markdown_path=tmp_path / f"bad-{kind}.md",
    )
    assert comparison["performance_valid"] is False
    assert comparison["speedups"] is None


def _compare_fixture(tmp_path: Path, *, dual: bool = True) -> dict:
    return compare_decode_performance_results(
        target_path=tmp_path / "target/decode-performance.json",
        serial_path=tmp_path / "serial/decode-performance.json",
        dual_path=tmp_path / "dual-batch/decode-performance.json" if dual else None,
        output_path=tmp_path / "comparison.json",
        markdown_path=tmp_path / "comparison.md",
    )


@pytest.mark.parametrize(
    "kind,check",
    [
        ("bootstrap", "bootstrap_equal"),
        ("prompt_sha", "prompt_provenance_equal"),
        ("prompt_count", "prompt_provenance_equal"),
        ("count", "per_request_measured_token_counts_equal"),
        ("total", "total_measured_token_counts_equal"),
        ("request_set", "request_set_equal"),
        ("request_count", "request_count_equal"),
        ("duplicate", "canonical_request_mapping_valid"),
        ("maximum", "requested_workload_semantics_equal"),
        ("workload", "workload_equal"),
        ("config", "config_equal"),
        ("topology", "topology_equal"),
        ("placement", "topology_equal"),
        ("patch", "execution_provenance_equal"),
        ("model", "execution_provenance_equal"),
        ("vllm", "execution_provenance_equal"),
        ("execution", "execution_provenance_equal"),
        ("execution_alias", "execution_provenance_equal"),
        ("accounting", "token_accounting_valid"),
        ("lying_accounting", "token_accounting_valid"),
        ("cleanup", "cleanup_valid"),
        ("failed", "requests_complete"),
        ("incomplete", "requests_complete"),
        ("artifact_invalid", "per_mode_artifacts_valid"),
        ("overlap", "dual_overlap_valid"),
        ("round_sync", "dual_overlap_valid"),
        ("invalid_metrics", "metrics_valid"),
    ],
)
def test_matched_work_rejects_incompatible_evidence(tmp_path: Path, kind: str, check: str) -> None:
    for mode in ("target", "serial", "dual-batch"):
        _measure(tmp_path, mode)
    path = tmp_path / "dual-batch/decode-performance.json"
    value = json.loads(path.read_text())
    row = value["requests"][0]
    if kind == "bootstrap":
        row["bootstrap_token_id"] = row["total_generated_token_ids"][0] = 999
    elif kind == "prompt_sha":
        row["prompt_token_ids_sha256"] = "0" * 64
    elif kind == "prompt_count":
        row["prompt_token_count"] += 1
    elif kind == "count":
        row["measured_committed_output_token_ids"].pop()
        row["total_generated_token_ids"].pop()
        row["measured_committed_output_token_count"] -= 1
        row["finish_reason"] = "stop"
        value["metrics"]["total_measured_committed_output_tokens"] -= 1
    elif kind == "total":
        value["metrics"]["total_measured_committed_output_tokens"] += 1
    elif kind == "request_set":
        row["request_id"] = "unexpected"
    elif kind == "request_count":
        value["request_count"] += 1
    elif kind == "duplicate":
        value["requests"].append(deepcopy(row))
    elif kind == "maximum":
        row["maximum_new_tokens"] += 1
    elif kind in ("workload", "config", "topology"):
        value["artifact_sha256"][kind] = "0" * 64
    elif kind == "placement":
        value["placement"]["target_physical_gpu_ids"] = [0, 1]
    elif kind == "patch":
        value["patch_hashes"][0] = "0" * 64
    elif kind == "model":
        value["models"]["target"]["revision"] = "other-model"
    elif kind == "vllm":
        value["vllm_commit"] = "0" * 40
    elif kind == "execution":
        value["execution_git_commit"] = value["git_commit"] = "0" * 40
    elif kind == "execution_alias":
        value["execution_git_commit"] = "0" * 40
    elif kind == "accounting":
        row["token_accounting_valid"] = False
    elif kind == "lying_accounting":
        row["total_generated_token_ids"][1] = 999
    elif kind == "cleanup":
        value["cleanup_valid"] = False
    elif kind in ("failed", "incomplete"):
        row["finish_reason"] = "abort" if kind == "failed" else None
    elif kind == "artifact_invalid":
        value["valid"] = False
    elif kind == "overlap":
        value["mode_semantics"]["natural_draft_target_overlap"] = False
    elif kind == "round_sync":
        value["mode_semantics"]["per_round_global_cuda_synchronize"] = True
    elif kind == "invalid_metrics":
        value["metrics"]["decode_makespan_ms"] = 0
    _json(path, value)
    report = _compare_fixture(tmp_path)
    assert report["matched_work_comparability"][check] is False
    assert report["performance_valid"] is False
    assert report["speedups"] is None
    assert report["errors"]


@pytest.mark.parametrize(
    "field",
    [
        "clock",
        "setup_excluded",
        "bootstrap_excluded_from_measured_token_count",
        "first_post_bootstrap_token_counted",
        "per_token_cuda_synchronize",
        "first_measured_target_forward_consumes_pending_bootstrap",
        "pre_measurement_tp_barrier",
        "pre_measurement_target_cuda_synchronize",
        "final_all_target_rank_cuda_synchronize",
    ],
)
def test_matched_work_requires_every_boundary_contract_field(tmp_path: Path, field: str) -> None:
    for mode in ("target", "serial"):
        _measure(tmp_path, mode)
    path = tmp_path / "serial/decode-performance.json"
    value = json.loads(path.read_text())
    value["measurement"].pop(field)
    _json(path, value)
    report = _compare_fixture(tmp_path, dual=False)
    assert report["matched_work_comparability"]["measurement_boundary_equivalent"] is False
    assert report["performance_valid_for_pair"] is False
    assert report["speedups"] is None


@pytest.mark.parametrize("legacy_target", [True, False])
def test_legacy_execution_commit_and_different_measurement_commits_are_comparable(
    tmp_path: Path, legacy_target: bool,
) -> None:
    for mode in ("target", "serial"):
        _measure(tmp_path, mode)
    target_path = tmp_path / "target/decode-performance.json"
    target = json.loads(target_path.read_text())
    if legacy_target:
        target.pop("execution_git_commit")
        target.pop("measurement_code_git_commit")
    else:
        target["measurement_code_git_commit"] = "2" * 40
    target["requires_exact_cross_mode_comparison"] = True  # Immutable historical v1 policy.
    _json(target_path, target)
    serial_path = tmp_path / "serial/decode-performance.json"
    serial = json.loads(serial_path.read_text())
    serial["measurement_code_git_commit"] = "1" * 40
    serial["requests"].reverse()  # Canonical ID mapping, not artifact row order.
    serial["warmup_clean"] = False
    serial["jit_observation"]["post_measurement_jit_event_count"] = 1
    _json(serial_path, serial)
    before = {path: path.read_bytes() for path in (target_path, serial_path)}
    report = _compare_fixture(tmp_path, dual=False)
    assert report["performance_valid_for_pair"] is True
    assert report["metrics"] == {"target": target["metrics"], "serial": serial["metrics"]}
    assert report["warmup"]["serial"] == {
        "warmup_clean": False,
        "post_measurement_jit_event_count": 1,
    }
    assert all(path.read_bytes() == contents for path, contents in before.items())


def test_pair_approval_helper_checks_live_artifact_hashes(tmp_path: Path) -> None:
    for mode in ("target", "serial"):
        _measure(tmp_path, mode)
    serial_path = tmp_path / "serial/decode-performance.json"
    serial = json.loads(serial_path.read_text())
    serial["requests"][0]["measured_committed_output_token_ids"][0] = 999
    serial["requests"][0]["total_generated_token_ids"][1] = 999
    _json(serial_path, serial)
    from specrhythm.cli import main

    assert main([
        "phase4b2-decode-compare",
        "--target", str(tmp_path / "target/decode-performance.json"),
        "--serial", str(serial_path),
        "--output", str(tmp_path / "target-serial-matched-work.json"),
        "--markdown-output", str(tmp_path / "target-serial-matched-work.md"),
    ]) == 0
    helper = Path(__file__).resolve().parents[1] / "integrations/vllm/phase4b2_run_helpers.sh"
    command = ["bash", "-c", 'source "$1"; phase4b2_require_matched_work_pair "$2" 2',
               "bash", str(helper), str(tmp_path)]
    env = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])
    approved = subprocess.run(command, capture_output=True, text=True, env=env)
    assert approved.returncode == 0, approved.stderr
    assert approved.stdout.splitlines() == [
        "MATCHED WORK TARGET/SERIAL PASS",
        "exact_sequence_equal = false",
        "performance_comparable = true",
    ]
    serial_path.write_text(serial_path.read_text() + "\n")
    rejected = subprocess.run(command, capture_output=True, text=True, env=env)
    assert rejected.returncode != 0  # Even a stale otherwise valid report cannot approve Dual.


def test_full_length_reason_spelling_is_only_diagnostic(tmp_path: Path) -> None:
    for mode in ("target", "serial"):
        _measure(tmp_path, mode)
    path = tmp_path / "serial/decode-performance.json"
    serial = json.loads(path.read_text())
    serial["requests"][0]["finish_reason"] = "max_tokens"
    _json(path, serial)
    report = _compare_fixture(tmp_path, dual=False)
    assert report["performance_valid_for_pair"] is True
    assert report["matched_work_comparability"]["termination_differences"]
    assert report["exact_correctness_triangle"]["valid"] is False


@pytest.mark.parametrize("legacy_null_metadata", [False, True])
def test_synthetic_corrected_100_with_nine_divergences_preserves_reported_metrics(
    tmp_path: Path, legacy_null_metadata: bool,
) -> None:
    # CPU fixture only, not a copy or revalidation of the server's GPU artifacts.
    expected = {
        "target": (5813.059543, 255.8033319632212, 382.881551485),
        "serial": (50394.65011, 29.50710039169275, 2345.39918652),
    }
    for mode, (makespan, throughput, tpot) in expected.items():
        value = _measure(tmp_path, mode)
        if legacy_null_metadata:
            for role in ("target", "draft"):
                value["models"][role]["revision"] = None
        template = value["requests"][0]
        value["requests"] = []
        for index in range(100):
            row = deepcopy(template)
            row["request_id"] = f"synthetic-{index:03}"
            count = 15 if index < 87 else 14
            tokens = list(range(count))
            if mode == "serial" and index < 9:
                tokens[0] = 999
            row.update(
                {
                    "maximum_new_tokens": None if legacy_null_metadata else count + 1,
                    "measured_committed_output_token_count": count,
                    "measured_committed_output_token_ids": tokens,
                    "total_generated_token_ids": [row["bootstrap_token_id"], *tokens],
                }
            )
            value["requests"].append(row)
        value["request_count"] = value["metrics"]["completed_requests"] = 100
        value["metrics"].update(
            {
                "total_measured_committed_output_tokens": 1487,
                "decode_makespan_ms": makespan,
                "aggregate_throughput_tokens_per_second": throughput,
            }
        )
        value["metrics"]["tpot_ms"]["mean"] = tpot
        _json(tmp_path / mode / "decode-performance.json", value)
    report = _compare_fixture(tmp_path, dual=False)
    assert report["performance_valid_for_pair"] is True
    assert report["exact_sequence_diagnostic"]["all_equal"] is False
    assert report["exact_sequence_diagnostic"]["divergent_request_count"] == 9
    assert report["exact_sequence_diagnostic"]["matching_request_count"] == 91
    assert len(report["exact_sequence_diagnostic"]["first_mismatches"]) == 9
    for mode, (makespan, throughput, tpot) in expected.items():
        assert report["metrics"][mode]["decode_makespan_ms"] == makespan
        assert report["metrics"][mode]["aggregate_throughput_tokens_per_second"] == throughput
        assert report["metrics"][mode]["tpot_ms"]["mean"] == tpot
    speedup = report["speedups"]["target_vs_serial"]
    assert speedup["makespan_speedup"] == pytest.approx(0.11535072445176113)
    assert speedup["throughput_ratio"] == pytest.approx(speedup["makespan_speedup"])


@pytest.mark.parametrize("roles", [("target",), ("draft",), ("target", "draft")])
def test_local_model_null_revisions_are_valid(tmp_path: Path, roles: tuple) -> None:
    for mode in ("target", "serial"):
        value = _measure(tmp_path, mode)
        for role in roles:
            value["models"][role]["revision"] = None
            value["models"][role]["tokenizer_revision"] = None
        value["models"]["tokenizer_revision"] = None
        _json(tmp_path / mode / "decode-performance.json", value)
    report = _compare_fixture(tmp_path, dual=False)
    assert report["performance_valid_for_pair"] is True
    assert report["matched_work_comparability"]["execution_provenance_equal"] is True


@pytest.mark.parametrize(
    "kind",
    ["different", "missing_revision", "bool_revision", "int_revision", "object_revision",
     "empty_path", "whitespace_path", "non_string_path", "missing_model",
     "tokenizer_revision_type", "nested_tokenizer_revision_type", "tokenizer_revision_diff"],
)
def test_model_revision_validation_stays_strict(tmp_path: Path, kind: str) -> None:
    for mode in ("target", "serial"):
        value = _measure(tmp_path, mode)
        for model in value["models"].values():
            model["revision"] = None
        value["models"]["tokenizer_revision"] = None
        if kind != "different" or mode == "serial":
            model = value["models"]["target"]
            if kind == "different":
                model["revision"] = "pinned-explicit-revision"
            elif kind == "missing_revision":
                model.pop("revision")
            elif kind in ("bool_revision", "int_revision", "object_revision"):
                model["revision"] = {"bool_revision": False, "int_revision": 7,
                                     "object_revision": {}}[kind]
            elif kind in ("empty_path", "whitespace_path", "non_string_path"):
                model["path"] = {"empty_path": "", "whitespace_path": " ",
                                 "non_string_path": 7}[kind]
            elif kind == "missing_model":
                value["models"].pop("target")
            elif kind == "tokenizer_revision_type":
                value["models"]["tokenizer_revision"] = []
            elif kind == "nested_tokenizer_revision_type":
                model["tokenizer_revision"] = False
            elif kind == "tokenizer_revision_diff" and mode == "serial":
                value["models"]["tokenizer_revision"] = "different"
        _json(tmp_path / mode / "decode-performance.json", value)
    report = _compare_fixture(tmp_path, dual=False)
    assert report["matched_work_comparability"]["execution_provenance_equal"] is False
    assert report["performance_valid_for_pair"] is False
    assert report["speedups"] is None


def test_native_workload_audit_reproduces_legacy_null_output_limit(tmp_path: Path) -> None:
    for mode in ("target", "serial"):
        root, paths = _make_run(tmp_path, mode, native_workload=True)
        workload = [json.loads(line) for line in paths["workload"].read_text().splitlines()]
        requested = {row["request_id"]: SmokeRequest.from_dict(row) for row in workload}
        assert all("output_tokens" not in row for row in workload)
        result = _build_result(root, paths, mode)
        assert result["valid"] is True
        for row in result["requests"]:
            # Real runner uses this SmokeRequest value for SamplingParams.max_tokens.
            total_limit = requested[row["request_id"]].maximum_new_tokens
            measured_limit = total_limit - row["setup_committed_output_tokens"]
            assert total_limit == len(row["total_generated_token_ids"]) == 3
            assert measured_limit == row["measured_committed_output_token_count"] == 2
            assert row["maximum_new_tokens"] is None  # Historical builder reads wrong key.
            assert row["finish_reason"] == "length"
            assert row["termination_reason"] is None
            assert row["setup_committed_output_tokens"] == 1
    report = _compare_fixture(tmp_path, dual=False)
    assert report["valid"] is True and report["errors"] == []
    assert report["matched_work_comparability"]["requests_complete"] is True
    assert report["matched_work_comparability"]["completion_evidence"][
        "null_output_limit_request_counts"
    ] == {"target": 2, "serial": 2}


@pytest.mark.parametrize("known_limit", [False, True])
def test_valid_early_stop_and_different_labels_remain_comparable(
    tmp_path: Path, known_limit: bool,
) -> None:
    for mode in ("target", "serial"):
        value = _measure(tmp_path, mode)
        for row in value["requests"]:
            row["maximum_new_tokens"] = 8 if known_limit else None
            row["finish_reason"] = "stop" if mode == "target" else "eos"
            row["termination_reason"] = "eos_token" if mode == "target" else 42
        _json(tmp_path / mode / "decode-performance.json", value)
    report = _compare_fixture(tmp_path, dual=False)
    assert report["performance_valid_for_pair"] is True
    assert report["matched_work_comparability"]["termination_differences"]
    assert report["exact_correctness_triangle"]["valid"] is False


@pytest.mark.parametrize("maximum", [None, 3])
@pytest.mark.parametrize(
    "failure",
    ["abort", "aborted", "error", "failed", "cancel", "cancelled", "canceled", "incomplete",
     "no_finish", "unfinished", "not_completed", "no_output", "bad_accounting",
     "missing_limit", "malformed_limit", "short_length", "over_limit"],
)
def test_completion_rejects_contradictory_evidence(
    tmp_path: Path, maximum: Optional[int], failure: str,
) -> None:
    for mode in ("target", "serial"):
        value = _measure(tmp_path, mode)
        for row in value["requests"]:
            row["maximum_new_tokens"] = maximum
        if mode == "serial":
            row = value["requests"][0]
            if failure == "no_finish":
                row["finish_reason"] = None
            elif failure == "unfinished":
                row["finished"] = False
            elif failure == "not_completed":
                row["completed"] = False
            elif failure == "no_output":
                row["total_generated_token_ids"] = []
            elif failure == "bad_accounting":
                row["token_accounting_valid"] = False
            elif failure == "missing_limit":
                row.pop("maximum_new_tokens")
            elif failure == "malformed_limit":
                row["maximum_new_tokens"] = "3"
            elif failure == "short_length":
                row["maximum_new_tokens"] = 8  # Length finish below a known total limit.
            elif failure == "over_limit":
                row["maximum_new_tokens"] = 2
            else:
                row["finish_reason"] = failure
        _json(tmp_path / mode / "decode-performance.json", value)
    report = _compare_fixture(tmp_path, dual=False)
    assert report["matched_work_comparability"]["requests_complete"] is False
    assert report["performance_valid_for_pair"] is False
    assert report["speedups"] is None


def test_performance_boundary_requires_unique_post_ready_event() -> None:
    row = {
        "schema_version": PERFORMANCE_EVENT_SCHEMA,
        "event": PERFORMANCE_EVENT,
        "timestamp_ns": 20,
        "consumer": "serial",
        "setup_ready_published_ns": 10,
        "pre_measurement_tp_barrier": True,
        "pre_measurement_target_cuda_synchronize": True,
        "setup_excluded": True,
    }
    boundary, errors = extract_performance_boundary([row], consumer="serial")
    assert boundary == 20
    assert errors == []
    assert extract_performance_boundary([row, row], consumer="serial")[1]


def test_boundary_publisher_barriers_and_synchronizes_without_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    class Group:
        def barrier(self) -> None:
            calls.append("barrier")

        def broadcast_object(self, value: object, src: int) -> object:
            calls.append(("broadcast", src))
            return value

    class Cuda:
        @staticmethod
        def synchronize() -> None:
            calls.append("synchronize")

    class Torch:
        cuda = Cuda()

    monkeypatch.setattr(
        "specrhythm.phase4.performance_boundary.time.monotonic_ns", lambda: 20
    )
    log = CheckpointJsonl(tmp_path / "timing.jsonl")
    boundary = publish_performance_boundary(
        tp_group=Group(),
        torch_module=Torch(),
        tp_rank=0,
        timing_log=log,
        consumer="serial",
        ready_published_ns=10,
    )
    assert boundary == 20
    assert calls == ["barrier", "synchronize", ("broadcast", 0)]
    row = log.read()[0]
    assert row["event"] == PERFORMANCE_EVENT
    assert row["pre_measurement_target_cuda_synchronize"] is True


def test_serial_setup_ready_can_defer_only_until_performance_boundary(
    tmp_path: Path,
) -> None:
    root, _ = _make_run(tmp_path, "serial")
    manifest_path = root / "decode-ready-manifest.json"
    from specrhythm.phase4.decode_ready import load_decode_ready_manifest

    manifest = load_decode_ready_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    value = build_setup_ready(
        manifest,
        consumer="serial",
        manifest_path=manifest_path,
        ready_published_ns=START - 10,
        initial_proposals_deferred=True,
    )
    assert value["initial_proposals"] == []
    assert value["initial_proposals_deferred_until_performance_boundary"] is True
    assert not validate_setup_ready(
        value,
        manifest_path=manifest_path,
        consumer="serial",
        expected_request_ids=list(REQUESTS),
    )

    manifest_by_id = {row.request_id: row for row in manifest.requests}
    proposals = tuple(
        Proposal(
            protocol_version=PROTOCOL_VERSION,
            request_id=request_id,
            round_id=0,
            parent_prefix_len=3,
            parent_prefix_hash=manifest_by_id[
                request_id
            ].logical_committed_prefix_sha256,
            proposal_token_ids=(300 + index,),
            proposal_eos=False,
            draft_start_ns=START + 1,
            draft_end_ns=START + 2,
            transport_payload_bytes=1,
            model_provenance={"model": "draft"},
            runtime_provenance={"runtime": "test"},
        )
        for index, request_id in enumerate(REQUESTS)
    )
    deferred_path = root / "deferred-test.json"
    _json(
        deferred_path,
        build_deferred_initial_proposals_ready(
            manifest,
            proposals=proposals,
            performance_measurement_start_ns=START,
            published_ns=START + 3,
        ),
    )
    restored = load_deferred_initial_proposals_ready(
        deferred_path,
        manifest_path=manifest_path,
        expected_request_ids=REQUESTS,
    )
    assert restored == proposals


def test_dual_without_physical_overlap_is_invalid(tmp_path: Path) -> None:
    root, paths = _make_run(tmp_path, "dual-batch")
    (root / "overlap-events.jsonl").unlink()
    _checkpoint(root / "overlap-events.jsonl", [{"overlap_duration_ns": 0}])
    result = build_decode_performance_result(
        mode="dual-batch",
        run_root=root,
        workload_path=paths["workload"],
        config_path=paths["config"],
        topology_path=paths["topology"],
        patch_manifest_path=paths["patch"],
        output_path=root / "decode-performance.json",
    )
    assert result["valid"] is False
    assert any("physical Draft/Target overlap" in error for error in result["errors"])


def test_phase4b2_source_has_no_per_token_cuda_synchronization() -> None:
    root = Path(__file__).resolve().parents[1]
    boundary_source = (root / "src/specrhythm/phase4/performance_boundary.py").read_text(
        encoding="utf-8"
    )
    assert '"per_token_cuda_synchronize": False' in boundary_source
    for name in ("resident_vllm.py", "vllm_remote.py", "vllm_dual.py"):
        source = (root / "src/specrhythm/phase4" / name).read_text(encoding="utf-8")
        assert "record_performance_commit" in source


def test_historical_gate3_roots_are_documented_as_immutable() -> None:
    root = Path(__file__).resolve().parents[1]
    runbook = (root / "docs/phase4b2-decode-performance-runbook.md").read_text(
        encoding="utf-8"
    )
    assert "8773a611a555c9c6efcbce146bb722124d0ee513" in runbook
    assert "efea5c8884e93b39114c320a724dc2c768ec1c8d" in runbook
    assert "historical-bootstrap-before.sha256" in runbook
    assert "historical-bootstrap-after.sha256" in runbook


def test_historical_gate3_decision_permits_progress_without_claiming_exactness() -> None:
    assert GATE3_QUALIFICATION["gate3_exact_stock_equivalence"] is False
    assert GATE3_QUALIFICATION["exact_stock_trajectory"] == "96/100"
    assert GATE3_QUALIFICATION["phase4b2_progression_permitted"] is True
    assert GATE3_QUALIFICATION["further_micro_diagnostics"] == "deferred"


def test_phase4b2_cli_exposes_only_baseline_modes() -> None:
    parser = build_parser()
    run = parser.parse_args(
        [
            "phase4b2-decode-run",
            "--mode",
            "dual-batch",
            "--run-root",
            "run",
            "--workload",
            "workload.jsonl",
            "--config",
            "config.yaml",
            "--topology",
            "topology.json",
            "--patch-manifest",
            "patch.json",
            "--output",
            "result.json",
        ]
    )
    assert run.mode == "dual-batch"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "phase4b2-decode-run",
                "--mode",
                "dual-eager",
                "--run-root",
                "run",
                "--workload",
                "workload.jsonl",
                "--config",
                "config.yaml",
                "--topology",
                "topology.json",
                "--patch-manifest",
                "patch.json",
                "--output",
                "result.json",
            ]
        )
