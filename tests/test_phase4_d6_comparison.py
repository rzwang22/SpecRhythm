from __future__ import annotations

import json
import shutil

import pytest
from test_phase4_batch_invariant import diagnostic
from test_phase4_draft_production_comparison import append_rows, pair, qualified_root, write

from specrhythm.phase4.draft_dual_comparison import compare_three, render_summary, summarize_run
from specrhythm.phase4.manifest import sha256_file
from specrhythm.phase4.transport import CheckpointJsonl


@pytest.fixture
def evidence(tmp_path):
    qualification = qualified_root(tmp_path)
    pair(tmp_path, "D6-C", 100, qualification)
    root = tmp_path / "D6-C"
    workload = tmp_path / "corrected-100.jsonl"
    workload.write_text(
        "\n".join(
            json.dumps(
                {
                    "request_id": f"r{i}",
                    "prompt_token_ids": [i + 1],
                    "prompt_length": 1,
                    "task_class": "code" if i < 60 else "chat" if i < 80 else "summarization",
                    "prompt_text": "<|im_start|>user\ntest<|im_start|>assistant",
                    "maximum_new_tokens": 4,
                    "sampling_seed": 1664,
                    "tokenizer_fingerprint": "frozen",
                }
            )
            for i in range(100)
        )
    )
    for mode, source in (("target", "hf"), ("serial", "vllm"), ("dual", "vllm")):
        directory = root / mode
        shutil.copytree(root / source, directory)
        performance = json.loads((directory / "decode-performance.json").read_text())
        raw = json.loads((directory / "resident-serial.json").read_text())
        raw.update(
            request_count=100, outputs=[{"request_id": f"r{i}"} for i in range(100)], errors=None
        )
        performance.update(
            mode="dual-batch" if mode == "dual" else mode,
            workload_sha256=sha256_file(workload),
            errors=None,
        )
        performance["artifact_sha256"]["workload"] = sha256_file(workload)
        performance["metrics"]["tpot_ms"].update(p90=3.0)
        ready = json.loads((directory / "draft-service-ready.json").read_text())
        backend = json.loads((directory / "draft-backend-report.json").read_text())
        backend["draft_retired_request_count"] = 100
        backend["provenance"]["backend"] = backend["backend_name"]
        ready["provenance"] = backend["provenance"]
        write(directory / "draft-service-ready.json", ready)
        write(directory / "draft-backend-report.json", backend)
        raw["draft_shutdown"]["draft_backend_report_sha256"] = sha256_file(
            directory / "draft-backend-report.json"
        )
        if mode == "target":
            write(directory / "plugin-report.json", {"proposal_generation": False})
            append_rows(
                directory / "target-diagnostics.jsonl",
                [
                    {**diagnostic(proposal=(), prefix=(i + 1, 10)), "request_id": f"r{i}"}
                    for i in range(100)
                ],
            )
        if mode == "dual":
            shutil.copyfile(directory / "round-events.jsonl", directory / "proposal-events.jsonl")
            raw["overlap_gate"] = {"valid": True}
            write(directory / "plugin-report.json", {"sampled_row_tp_consensus": True})
            append_rows(
                directory / "overlap-events.jsonl",
                [
                    {"overlap_duration_ns": 10, "host_interval": [20, 30]},
                    {"overlap_duration_ns": 10, "host_interval": [20, 30]},
                    {"overlap_duration_ns": 10, "host_interval": [25, 35]},
                ],
            )
            append_rows(
                directory / "verification-events.jsonl",
                [
                    {"verify_microbatch_id": str(i), "verify_request_ids": [f"r{i}", f"r{i + 1}"]}
                    for i in range(0, 100, 2)
                ],
            )
        raw_name = {
            "target": "resident-target.json",
            "serial": "resident-serial.json",
            "dual": "resident-dual.json",
        }[mode]
        write(directory / raw_name, raw)
        for key, name in (
            ("raw_run", raw_name),
            ("target_diagnostics", "target-diagnostics.jsonl"),
        ):
            performance["artifact_sha256"][key] = sha256_file(directory / name)
        write(directory / "decode-performance.json", performance)
    return root, workload


def test_three_mode_summary_and_overlap_union(evidence):
    root, workload = evidence
    for mode in ("target", "serial", "dual"):
        result = summarize_run(root / mode, mode, 100, workload)
        assert result["valid"], result["errors"]
        write(root / mode / "qualification.json", result)
    report = compare_three(root)
    assert report["valid"], report["errors"]
    assert len(report["ratios"]) == 15
    dual = report["modes"]["dual"]["metrics"]
    assert dual["observed_overlap_ms"] == 15 / 1e6
    assert dual["verification_batch_size"]["p50"] == 2
    assert not dual["overlap_is_critical_path_time_saved"]
    assert "Serial-vLLM" in render_summary(report)
    assert "historical" not in report["ratios"]


@pytest.mark.parametrize("mode", ["target", "serial", "dual"])
def test_first_runtime_failure_has_no_cascading_checks(evidence, mode):
    root, workload = evidence
    name = {
        "target": "resident-target.json",
        "serial": "resident-serial.json",
        "dual": "resident-dual.json",
    }[mode]
    path = root / mode / name
    raw = json.loads(path.read_text())
    raw.update(valid=False, errors=["first execution failure"])
    write(path, raw)
    (root / mode / "decode-performance.json").unlink()
    (root / mode / "process-lifecycle.json").unlink()
    result = summarize_run(root / mode, mode, 100, workload)
    assert not result["valid"] and result["errors"] == [
        "runtime failed: ['first execution failure']"
    ]
    assert result["metrics"] == {}


@pytest.mark.parametrize("failure", ["accounting", "target", "overlap", "batch", "cleanup", "nan"])
def test_material_validity_still_blocks(evidence, failure):
    root, workload = evidence
    directory = root / "dual"
    performance = json.loads((directory / "decode-performance.json").read_text())
    if failure == "accounting":
        performance["metrics"]["total_measured_committed_output_tokens"] += 1
    elif failure == "nan":
        performance["metrics"]["decode_makespan_ms"] = float("nan")
    elif failure == "target":
        path = directory / "target-diagnostics.jsonl"
        rows = CheckpointJsonl(path).read()
        rows[0]["causal_attention"] = False
        append_rows(path, rows)
        performance["artifact_sha256"]["target_diagnostics"] = sha256_file(path)
    elif failure == "overlap":
        raw = json.loads((directory / "resident-dual.json").read_text())
        raw["overlap_gate"]["valid"] = False
        write(directory / "resident-dual.json", raw)
        performance["artifact_sha256"]["raw_run"] = sha256_file(directory / "resident-dual.json")
    else:
        backend = json.loads((directory / "draft-backend-report.json").read_text())
        if failure == "batch":
            backend["draft_batch_statistics_by_purpose"]["proposal"]["histogram"] = {"1": 3}
        else:
            backend["draft_retired_request_count"] = 99
        write(directory / "draft-backend-report.json", backend)
        raw = json.loads((directory / "resident-dual.json").read_text())
        raw["draft_shutdown"]["draft_backend_report_sha256"] = sha256_file(
            directory / "draft-backend-report.json"
        )
        write(directory / "resident-dual.json", raw)
        performance["artifact_sha256"]["raw_run"] = sha256_file(directory / "resident-dual.json")
    write(directory / "decode-performance.json", performance)
    result = summarize_run(directory, "dual", 100, workload)
    assert not result["valid"] and len(result["errors"]) == 1


@pytest.mark.parametrize(
    "rates,case", [((1, 2, 3), "A"), ((3, 1, 2), "B"), ((3, 2, 1), "C"), ((1, 3, 2), "D")]
)
def test_interpretation_is_selected_from_results(evidence, rates, case):
    root, workload = evidence
    for mode, rate in zip(("target", "serial", "dual"), rates):
        result = summarize_run(root / mode, mode, 100, workload)
        assert result["valid"], result["errors"]
        result["metrics"]["throughput_tokens_per_second"] = rate
        write(root / mode / "qualification.json", result)
    assert compare_three(root)["interpretation_case"] == case


def test_d6_smoke_is_not_performance(evidence):
    root, workload = evidence
    (root / "dual/decode-performance.json").unlink()
    report = summarize_run(root / "dual", "dual", 100, workload, smoke=True)
    assert report["valid"] and not report["performance_result"]
