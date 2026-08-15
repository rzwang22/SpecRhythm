import copy
import json
from dataclasses import asdict

import pytest

from specrhythm.cli import main
from specrhythm.phase3.benchmark import (
    BACKEND_SEMANTICS,
    aggregate_rank_samples,
    timing_statistics,
)
from specrhythm.phase3.benchmark_validation import (
    compare_benchmark_reports,
    validate_benchmark_report,
)
from specrhythm.phase3.config import BenchmarkConfig
from specrhythm.phase3.hardware import capture_hardware_state
from specrhythm.phase3.selector_benchmark import SELECTOR_STAGES, run_selector_dry_run


def _samples(offset):
    return [offset + index / 100 for index in range(30)]


def _rank(rank, cuda_samples, world_size=2, model=True):
    host_samples = [value + 0.5 for value in cuda_samples]
    result = {
        "global_rank": rank,
        "local_rank": rank,
        "world_size": world_size,
        "logical_cuda_index": rank,
        "physical_gpu_id": str(rank + 1),
        "cuda_visible_devices": "1,2",
        "cuda_visible_devices_mapping": [
            {"logical_cuda_index": 0, "physical_gpu_id": "1"},
            {"logical_cuda_index": 1, "physical_gpu_id": "2"},
        ],
        "gpu_uuid": f"GPU-{rank}",
        "model_parameter_count": 100 if model else None,
        "parameter_bytes": 200 if model else None,
        "parameter_devices": [f"cuda:{rank}"] if model else [],
        "expected_parameter_device": f"cuda:{rank}" if model else None,
        "model_parameters_on_expected_device": True if model else None,
        "allocated_memory_bytes": 1000,
        "reserved_memory_bytes": 1200,
        "max_allocated_memory_bytes": 1400,
        "max_reserved_memory_bytes": 1600,
        "forward_input_shape": [1, 128] if model else [],
        "forward_output_shape": [1, 128, 1000] if model else [],
        "output_checksum": "shared-checksum" if model else None,
        "forward_invocations": 30 if model else 0,
        "cuda_samples_ms": cuda_samples,
        "host_samples_ms": host_samples,
        "device_memory": {
            str(rank): {
                "allocated_memory_bytes": 1000,
                "reserved_memory_bytes": 1200,
                "max_allocated_memory_bytes": 1400,
                "max_reserved_memory_bytes": 1600,
            }
        },
    }
    return result


def _protocol():
    return {
        "distributed_barrier_before_iteration": True,
        "cuda_synchronize_before_iteration": True,
        "cuda_event_per_rank": True,
        "host_clock_per_rank": True,
        "cuda_event_synchronize_before_host_stop": True,
        "distributed_barrier_after_iteration": True,
        "raw_samples_retained": True,
        "outliers_removed": False,
    }


def _hardware():
    def gpu(physical_gpu_id):
        return {
            "physical_gpu_id": physical_gpu_id,
            "gpu_name": "A800",
            "gpu_uuid": f"GPU-{physical_gpu_id}",
            "temperature_c": 40.0,
            "power_draw_w": 100.0,
            "power_limit_w": 400.0,
            "sm_clock_mhz": 1000.0,
            "memory_clock_mhz": 1500.0,
            "p_state": "P0",
            "memory_used_mib": 1000.0,
            "ecc_status": "Enabled",
            "pcie_generation_current": 4.0,
            "pcie_generation_max": 4.0,
            "pcie_width_current": 16.0,
            "pcie_width_max": 16.0,
        }

    return {
        "clock_locked": False,
        "gpus": [gpu("1"), gpu("2")],
        "peer_access": [],
        "nvlink_pcie_topology": "GPU0 NV8 GPU1",
        "errors": [],
    }


def _verify_report(commit="a" * 40):
    rank0 = _rank(0, _samples(10.0))
    rank1 = _rank(1, _samples(12.0))
    ranks = [rank0, rank1]
    cuda = aggregate_rank_samples(ranks, "cuda_samples_ms")
    host = aggregate_rank_samples(ranks, "host_samples_ms")
    row = {
        "operation": "T_verify",
        "dimensions": {"B_req": 1, "B_cand": 4, "context_length": 128, "TP": 2},
        "warmup_iterations": 5,
        "measured_iterations": 30,
        "cuda_event": asdict(timing_statistics(cuda)),
        "host_wall": asdict(timing_statistics(host)),
        "rank_measurements": ranks,
        "requires_model_rank_evidence": True,
        "global_latency_definition": (
            "per-iteration maximum latency across all participating ranks"
        ),
        "timing_protocol": _protocol(),
        "actual_request_roots": 1,
        "actual_search_pool_nodes": 0,
        "actual_verified_candidate_nodes": 4,
        "actual_target_input_positions": 5,
        "implementation": "HF correctness verifier; serial full-context replay",
        "operation_semantics": {
            "verify_implementation": "serial_full_context_replay",
            "number_of_target_forwards": 5,
            "B_cand_definition": "non-root candidate steps",
            "target_input_positions": 5,
        },
    }
    return {
        "schema_version": "specrhythm.gpu-latency.v2",
        "git_commit": commit,
        "config_sha256": "b" * 64,
        "backend_semantics": copy.deepcopy(BACKEND_SEMANTICS),
        "simulator_latency_surface_compatible": False,
        "draft_model": {"model_path": "draft", "revision": "r1", "tp_size": 1},
        "target_model": {"model_path": "target", "revision": "r1", "tp_size": 2},
        "model_identity": {
            "draft": {
                "model_path": "draft",
                "configured_revision": "r1",
                "config_sha256": "d" * 64,
            },
            "target": {
                "model_path": "target",
                "configured_revision": "r1",
                "config_sha256": "e" * 64,
            },
        },
        "runtime_versions": {
            "pytorch": "2.7.1",
            "transformers": "4.56.1",
            "cuda_runtime": "12.8",
            "nccl": "2.26.2",
        },
        "benchmark_config": {"warmup_iterations": 5, "measured_iterations": 30},
        "hardware_state_before": _hardware(),
        "hardware_state_after": _hardware(),
        "measurements": [row],
    }


def _transfer_report():
    report = _verify_report()
    rank = _rank(0, _samples(0.1), world_size=1, model=False)
    cuda = rank["cuda_samples_ms"]
    host = rank["host_samples_ms"]
    report["measurements"] = [
        {
            "operation": "T_transfer",
            "dimensions": {"payload_bytes": 4096, "direction": "draft_to_target_leader"},
            "warmup_iterations": 5,
            "measured_iterations": 30,
            "cuda_event": asdict(timing_statistics(cuda)),
            "host_wall": asdict(timing_statistics(host)),
            "rank_measurements": [rank],
            "requires_model_rank_evidence": False,
            "global_latency_definition": (
                "per-iteration maximum latency across all participating ranks"
            ),
            "timing_protocol": _protocol(),
            "actual_request_roots": 0,
            "actual_search_pool_nodes": 0,
            "actual_verified_candidate_nodes": 0,
            "actual_target_input_positions": 0,
            "implementation": "bare direct CUDA tensor copy primitive",
            "operation_semantics": {
                "transport_scope": "bare_device_copy_only",
                "complete_draft_to_verify_transport": False,
            },
            "transfer_metadata": {
                "copy_direction": "cuda:0->cuda:1",
                "source_logical_cuda_index": 0,
                "source_physical_gpu_id": "0",
                "destination_logical_cuda_index": 1,
                "destination_physical_gpu_id": "1",
                "cuda_device_can_access_peer": True,
                "p2p_enabled": True,
                "host_staging": False,
                "effective_bandwidth_gbps": 10.0,
                "topology_source": "hardware_state_before.nvlink_pcie_topology",
            },
        }
    ]
    return report


def test_valid_tp_report_and_per_iteration_max_rank_aggregation():
    report = _verify_report()
    assert validate_benchmark_report(report)["valid"]
    row = report["measurements"][0]
    expected = row["rank_measurements"][1]["cuda_samples_ms"]
    assert list(row["cuda_event"]["raw_samples_ms"]) == expected
    assert aggregate_rank_samples(row["rank_measurements"], "cuda_samples_ms") == expected


def test_tp_rank_missing_fails_validation():
    report = _verify_report()
    report["measurements"][0]["rank_measurements"].pop()
    validation = validate_benchmark_report(report)
    assert not validation["valid"]
    assert any("complete ranks" in error for error in validation["errors"])


def test_tp_zero_memory_fails_validation():
    report = _verify_report()
    report["measurements"][0]["rank_measurements"][1][
        "max_allocated_memory_bytes"
    ] = 0
    validation = validate_benchmark_report(report)
    assert not validation["valid"]
    assert any("max_allocated_memory_bytes" in error for error in validation["errors"])


def test_parameter_device_evidence_is_required_for_every_rank():
    report = _verify_report()
    report["measurements"][0]["rank_measurements"][1][
        "model_parameters_on_expected_device"
    ] = False
    validation = validate_benchmark_report(report)
    assert not validation["valid"]
    assert any("expected device" in error for error in validation["errors"])


def test_rank_mapping_and_output_checksum_must_agree():
    report = _verify_report()
    rank = report["measurements"][0]["rank_measurements"][1]
    rank["physical_gpu_id"] = "1"
    rank["output_checksum"] = "different"
    validation = validate_benchmark_report(report)
    assert not validation["valid"]
    assert any("distinct physical GPUs" in error for error in validation["errors"])
    assert any("checksums disagree" in error for error in validation["errors"])


def test_raw_samples_quantiles_and_barrier_protocol_are_strict():
    report = _verify_report()
    row = report["measurements"][0]
    row["cuda_event"]["p90_ms"] = row["cuda_event"]["p50_ms"] - 1
    row["timing_protocol"]["distributed_barrier_before_iteration"] = False
    validation = validate_benchmark_report(report)
    assert not validation["valid"]
    assert any("p90_ms does not match" in error for error in validation["errors"])
    assert any("timing protocol" in error for error in validation["errors"])


def test_outlier_is_retained_and_mean_above_p90_is_allowed():
    samples = [1.0] * 29 + [100.0]
    stats = timing_statistics(samples)
    assert stats.mean_ms > stats.p90_ms
    assert stats.raw_samples_ms[-1] == 100.0
    assert stats.outlier_indices == (29,)


def test_comparison_rejects_different_commit_and_backend():
    first = _verify_report("a" * 40)
    second = _verify_report("c" * 40)
    comparison = compare_benchmark_reports([first, second])
    assert not comparison["valid"]
    assert any("different git_commit" in error for error in comparison["errors"])

    second = _verify_report("a" * 40)
    second["backend_semantics"]["backend"] = "serving_engine"
    comparison = compare_benchmark_reports([first, second])
    assert not comparison["valid"]
    assert any("backend" in error for error in comparison["errors"])


def test_same_run_comparison_reports_variation_without_pooling_samples():
    first = _verify_report()
    second = _verify_report()
    for sample in second["measurements"][0]["rank_measurements"][1][
        "cuda_samples_ms"
    ]:
        assert sample > 0
    comparison = compare_benchmark_reports([first, second], ["one", "two"])
    assert comparison["valid"]
    assert comparison["run_count"] == 2
    assert comparison["cells"][0]["raw_samples_combined"] is False


def test_backend_and_transfer_semantics_are_complete():
    report = _transfer_report()
    assert validate_benchmark_report(report)["valid"]
    del report["measurements"][0]["transfer_metadata"]["p2p_enabled"]
    validation = validate_benchmark_report(report)
    assert not validation["valid"]
    assert any("p2p_enabled" in error for error in validation["errors"])

    report = _verify_report()
    del report["backend_semantics"]["packed_tree_verification"]
    validation = validate_benchmark_report(report)
    assert not validation["valid"]
    assert any("packed_tree_verification" in error for error in validation["errors"])


def test_selector_stage_interface_has_dependency_free_dry_run(tmp_path):
    report = run_selector_dry_run(
        request_count=2, search_pool_size=8, candidate_budget=7
    )
    assert report["prefix_closed"]
    assert report["gpu_measurement"] is False
    assert report["latency_samples_recorded"] is False
    assert [stage["name"] for stage in report["stages"]] == list(SELECTOR_STAGES)

    output = tmp_path / "selector.json"
    assert (
        main(
            [
                "phase3-selector-dry-run",
                "--request-count",
                "2",
                "--search-pool-size",
                "8",
                "--candidate-budget",
                "7",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text())["synthetic_latency_used"] is False


def test_validation_and_comparison_cli_round_trip(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(_verify_report()))
    second.write_text(json.dumps(_verify_report()))
    validation = tmp_path / "validation.json"
    assert (
        main(
            [
                "phase3-benchmark-validate",
                "--input",
                str(first),
                "--output",
                str(validation),
            ]
        )
        == 0
    )
    assert json.loads(validation.read_text())["valid"]
    comparison = tmp_path / "comparison.json"
    markdown = tmp_path / "comparison.md"
    assert (
        main(
            [
                "phase3-benchmark-compare",
                "--input",
                str(first),
                "--input",
                str(second),
                "--output",
                str(comparison),
                "--markdown-output",
                str(markdown),
            ]
        )
        == 0
    )
    assert json.loads(comparison.read_text())["valid"]
    assert "repeated-run comparison" in markdown.read_text()


def test_benchmark_iterations_enforce_hardened_minimums():
    with pytest.raises(ValueError, match="warmup_iterations must be at least 5"):
        BenchmarkConfig.from_dict(
            {"warmup_iterations": 4, "measured_iterations": 30}
        )
    with pytest.raises(ValueError, match="measured_iterations must be at least 30"):
        BenchmarkConfig.from_dict(
            {"warmup_iterations": 5, "measured_iterations": 29}
        )


def test_hardware_snapshot_preserves_unavailable_fields(monkeypatch):
    from specrhythm.phase3 import hardware

    monkeypatch.setattr(
        hardware,
        "_command",
        lambda argv: (None, f"unavailable: {argv[0]}"),
    )
    monkeypatch.setattr(
        hardware,
        "_peer_access",
        lambda: ([], ["CUDA peer access unavailable"]),
    )
    report = capture_hardware_state((0, 1))
    assert report["clock_locked"] is False
    assert report["gpus"] == []
    assert report["peer_access"] == []
    assert any("requested physical GPU IDs" in error for error in report["errors"])
    assert any("CUDA peer access unavailable" in error for error in report["errors"])
