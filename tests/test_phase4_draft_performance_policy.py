from __future__ import annotations

import json
import shutil
from types import SimpleNamespace

import pytest
from test_phase4_draft_production_comparison import compare, pair, qualified_root, write

from specrhythm.phase4 import draft_performance_policy as policy
from specrhythm.phase4.manifest import sha256_file


@pytest.fixture
def d4(tmp_path):
    qualification = qualified_root(tmp_path)
    pair(tmp_path, "D4", 5, qualification)
    return tmp_path, qualification


def mutate(root, filename, change, label="vllm"):
    directory = root / "D4" / label
    path = directory / filename
    value = json.loads(path.read_text())
    change(value)
    write(path, value)
    if filename == "draft-backend-report.json":
        mutate(
            root,
            "resident-serial.json",
            lambda v: v["draft_shutdown"].update(draft_backend_report_sha256=sha256_file(path)),
            label,
        )
    if filename in ("resident-serial.json", "process-lifecycle.json"):
        key = "raw_run" if filename == "resident-serial.json" else "process_lifecycle"
        mutate(
            root,
            "decode-performance.json",
            lambda v: v["artifact_sha256"].update({key: sha256_file(path)}),
            label,
        )


@pytest.mark.parametrize("errors", [None, [], {}, "", ["execution failed"]])
@pytest.mark.parametrize("filename", ["resident-serial.json", "decode-performance.json"])
def test_empty_errors_are_semantic_and_nonempty_errors_block(d4, filename, errors):
    root, qualification = d4
    for label in ("hf", "vllm"):
        mutate(root, filename, lambda v: v.update(errors=errors), label)
    result = compare(root, "D4", qualification)
    assert result["stage_qualified"] is (not bool(errors)), result["errors"]


@pytest.mark.parametrize(
    "filename,change,expected",
    [
        ("resident-serial.json", lambda v: v.update(valid=False), "raw Serial invalid"),
        ("resident-serial.json", lambda v: v["accounting"].update(valid=False), "accounting"),
        (
            "resident-serial.json",
            lambda v: v["kv_monotonicity"].update(valid=False),
            "kv_monotonicity",
        ),
        (
            "resident-serial.json",
            lambda v: v["batch_invariant_validation"].update(valid=False),
            "batch_invariant_validation",
        ),
        ("process-lifecycle.json", lambda v: v.update(cleanup_valid=False), "cleanup"),
        ("process-lifecycle.json", lambda v: v.update(remaining_owned_pids=[42]), "remain alive"),
        ("process-lifecycle.json", lambda v: v.update(run_valid=False), "return path"),
        ("process-lifecycle.json", lambda v: v.update(target_exit_status=1), "return path"),
        ("process-lifecycle.json", lambda v: v.update(effective_exit_status=1), "return path"),
        ("decode-performance.json", lambda v: v["metrics"].update(completed_requests=4), "count"),
        (
            "decode-performance.json",
            lambda v: v["metrics"].update(total_measured_committed_output_tokens=99),
            "accounting",
        ),
        (
            "decode-performance.json",
            lambda v: v["requests"][0].update(token_accounting_valid=False),
            "accounting",
        ),
        (
            "decode-performance.json",
            lambda v: v["metrics"].update(decode_makespan_ms=0),
            "positive",
        ),
        (
            "decode-performance.json",
            lambda v: v["metrics"]["tpot_ms"].update(mean=float("nan")),
            "TPOT",
        ),
        ("draft-backend-report.json", lambda v: v.update(execution_failed=True), "cleanup"),
        (
            "draft-backend-report.json",
            lambda v: v["provenance"]["model"].update(path="/wrong"),
            "identity",
        ),
    ],
)
def test_material_failures_still_block(d4, filename, change, expected):
    root, qualification = d4
    mutate(root, filename, change)
    result = compare(root, "D4", qualification)
    assert not result["stage_qualified"] and not result["performance_interpretation_allowed"]
    assert expected in str(result["blocking_errors"])


def test_historical_and_nonessential_metadata_do_not_requalify_d3(d4):
    root, qualification = d4
    for directory in root.glob("D3-*"):
        shutil.rmtree(directory)
    for label in ("hf", "vllm"):
        mutate(
            root,
            "decode-performance.json",
            lambda v: v.update(
                execution_git_commit="new-comparator-commit", schema_version="future-metadata"
            ),
            label,
        )
        path = root / "D4" / f"{label}-admission.json"
        admission = json.loads(path.read_text())
        admission.update(d3_qualification_sha256="historical-envelope", schema_version="new")
        admission["run_identity"]["execution_commit"] = "older-history"
        write(path, admission)
    mutate(
        root,
        "draft-backend-report.json",
        lambda v: v["provenance"].update(startup_warmup_ns=1234, informational_warning="JIT"),
    )
    mutate(root, "decode-performance.json", lambda v: v["measurement"].update(note="new"))
    result = compare(root, "D4", qualification)
    assert result["stage_qualified"], result["errors"]
    assert result["diagnostics"] and not result["historical_qualification_revalidated"]
    assert result["hf_proposal_equality_is_blocking"] is False
    write(root / "D4/comparison.json", result)
    # Only completed certificates are required; no recursive D3/D4 raw revalidation.
    shutil.rmtree(root / "D4/hf")
    shutil.rmtree(root / "D4/vllm")
    assert policy.require_performance_stage(qualification, "D5", root / "D4/comparison.json")


def test_d5_still_requires_qualified_d4(d4):
    root, qualification = d4
    result = compare(root, "D4", qualification)
    result["d4_qualified"] = False
    write(root / "D4/comparison.json", result)
    with pytest.raises(ValueError, match="qualified D4"):
        policy.require_performance_stage(qualification, "D5", root / "D4/comparison.json")


def test_offline_cli_writes_comparison_without_changing_execution_artifacts(d4, monkeypatch):
    from specrhythm.phase4 import draft_comparison

    root, qualification = d4
    for label in ("hf", "vllm"):
        mutate(root, "resident-serial.json", lambda v: v.update(errors=None), label)
    inputs = {p: sha256_file(p) for p in (root / "D4").rglob("*") if p.is_file()}
    output = root / "D4/comparison.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare",
            "--hf",
            str(root / "D4/hf"),
            "--vllm",
            str(root / "D4/vllm"),
            "--request-count",
            "5",
            "--stage",
            "D4",
            "--qualification",
            str(qualification),
            "--output",
            str(output),
        ],
    )
    assert draft_comparison.main() == 0
    result = json.loads(output.read_text())
    assert result["d4_qualified"] and result["error_serialization"]["hf"]["raw"] is None
    assert all(sha256_file(p) == digest for p, digest in inputs.items())
    with pytest.raises(FileExistsError):
        draft_comparison.main()


def test_real_multirequest_batching_does_not_require_p50_above_one(d4):
    root, qualification = d4

    def mostly_singletons(value):
        value.update(
            draft_model_forward_count=101,
            draft_batch_size_histogram={"1": 100, "5": 1},
            draft_model_forward_count_by_purpose={"proposal": 1, "commit": 100},
        )
        from collections import Counter

        from specrhythm.phase4.draft_metrics import batch_statistics

        for key, number in batch_statistics(Counter({1: 100, 5: 1})).items():
            if key in ("min", "p10", "p50", "p90", "max", "mean"):
                value[f"draft_batch_size_{key}"] = number

    mutate(root, "draft-backend-report.json", mostly_singletons)
    result = compare(root, "D4", qualification)
    assert result["stage_qualified"], result["errors"]
    assert not result["draft_batch_p50_greater_than_one"]


def test_cpu_admission_accepts_new_commit_and_retains_workload_contracts(tmp_path, monkeypatch):
    config = tmp_path / "config"
    config.write_text("frozen config")
    qualification = tmp_path / "qualification.json"
    write(
        qualification,
        {
            "valid": True,
            "errors": None,
            "run_identity": {
                "execution_commit": "old",
                "config_sha256": sha256_file(config),
                "model_path": "/model",
            },
        },
    )
    workload = tmp_path / "workload.jsonl"
    workload.write_text(
        "\n".join(
            json.dumps(
                {
                    "request_id": f"r{i}",
                    "prompt_token_ids": [5],
                    "prompt_length": 1,
                    "maximum_new_tokens": 4,
                    "task_class": ("code", "code", "code", "chat", "summarization")[i],
                    "prompt_text": "<|im_start|>user\ntest\n<|im_start|>assistant",
                    "sampling_seed": 1664,
                    "tokenizer_fingerprint": "frozen-tokenizer",
                }
            )
            for i in range(5)
        )
    )
    reference = tmp_path / "reference.json"
    reference.write_text("{}")
    monkeypatch.setattr(
        policy,
        "load_phase4_config",
        lambda p: SimpleNamespace(draft=SimpleNamespace(resolved_model_path="/model")),
    )
    value = policy.prepare_serial(
        qualification, "D4", None, "hf", config, workload, reference, "new"
    )
    assert value["run_identity"]["execution_commit"] == "new"
    assert value["request_count"] == 5 and value["requests"]["r0"]["maximum_new_tokens"] == 4
    assert not value["historical_identity_revalidated"]
    config.write_text("changed experiment")
    with pytest.raises(ValueError, match="config"):
        policy.prepare_serial(qualification, "D4", None, "hf", config, workload, reference, "new")
