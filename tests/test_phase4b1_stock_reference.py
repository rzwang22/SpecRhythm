from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from integrations.vllm import phase4b1_stock_reference as stock_helper
from integrations.vllm.phase4b1_stock_reference import (
    DIAGNOSTIC_SCHEMA,
    build_stock_determinism_diagnostic,
)


def _smoke():
    run = [
        {
            "request_id": "A",
            "generated_token_ids": [1, 2, 3],
            "text": "a",
            "finish_reason": "length",
            "stop_reason": None,
        },
        {
            "request_id": "B",
            "generated_token_ids": [4, 5],
            "text": "b",
            "finish_reason": "stop",
            "stop_reason": 5,
        },
    ]
    return {
        "runs": [run, json.loads(json.dumps(run))],
        "repeated_run_deterministic": True,
        "correctness_mode": "batch-invariant",
        "batch_invariant_requested": True,
        "batch_invariant_effective": True,
        "batch_invariant_validation": {"valid": True, "errors": []},
        "worker_ranks": [{"global_rank": 0}, {"global_rank": 1}],
    }


def _diagnostic(smoke):
    return build_stock_determinism_diagnostic(
        smoke,
        git_commit="a" * 40,
        config_sha256="b" * 64,
        workload_sha256="c" * 64,
        installed_runner_sha256="d" * 64,
    )


def test_deterministic_pair_is_frozen_without_retry_semantics():
    report = _diagnostic(_smoke())
    assert report["schema_version"] == DIAGNOSTIC_SCHEMA
    assert report["valid"] is True
    assert report["outcome"] == "deterministic"
    assert report["retry_count"] == 0
    assert report["retry_until_success"] is False
    assert report["reference_freeze_eligible"] is True
    assert report["divergent_request_count"] == 0
    assert len(report["runs"]) == 2


def test_nondeterministic_pair_records_exact_first_token_divergence():
    smoke = _smoke()
    smoke["runs"][1][0]["generated_token_ids"] = [1, 9, 3]
    smoke["repeated_run_deterministic"] = False
    report = _diagnostic(smoke)
    assert report["valid"] is False
    assert report["outcome"] == "nondeterministic"
    assert report["reference_freeze_eligible"] is False
    assert report["first_divergent_request_id"] == "A"
    comparison = report["per_request_comparisons"][0]
    assert comparison["first_divergence_position"] == 1
    assert comparison["run_1_token_id"] == 2
    assert comparison["run_2_token_id"] == 9
    assert report["runs"][0][0]["generated_token_ids"] == [1, 2, 3]
    assert report["runs"][1][0]["generated_token_ids"] == [1, 9, 3]


def test_length_and_termination_divergences_fail_closed_per_request():
    smoke = _smoke()
    smoke["runs"][1][0]["generated_token_ids"] = [1, 2]
    smoke["runs"][1][1]["finish_reason"] = "length"
    smoke["repeated_run_deterministic"] = False
    report = _diagnostic(smoke)
    by_id = {
        row["request_id"]: row for row in report["per_request_comparisons"]
    }
    assert by_id["A"]["first_divergence_position"] == 2
    assert by_id["A"]["run_1_token_id"] == 3
    assert by_id["A"]["run_2_token_id"] is None
    assert by_id["B"]["semantic_mismatches"] == ["finish_reason"]
    assert report["divergent_request_count"] == 2


def test_reported_true_cannot_hide_raw_nondeterminism():
    smoke = _smoke()
    smoke["runs"][1][0]["generated_token_ids"][0] = 999
    report = _diagnostic(smoke)
    assert report["valid"] is False
    assert any("flag disagrees" in error for error in report["errors"])
    assert report["retry_until_success"] is False


def test_failed_freeze_preserves_pair_diagnostic_without_reference(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.json"
    workload_path = tmp_path / "workload.jsonl"
    environment_path = tmp_path / "environment.json"
    topology_path = tmp_path / "topology.json"
    for path in (config_path, workload_path, environment_path, topology_path):
        path.write_text("{}\n", encoding="utf-8")
    config = SimpleNamespace(target_model_runner="v1", path=config_path)
    smoke = _smoke()
    smoke["runs"][1][0]["generated_token_ids"][1] = 999
    smoke["repeated_run_deterministic"] = False
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setattr(stock_helper, "load_phase4_config", lambda _: config)
    monkeypatch.setattr(stock_helper, "require_stock_vllm_runner", lambda: "d" * 64)
    monkeypatch.setattr(stock_helper, "run_stock_smoke", lambda *args, **kwargs: smoke)
    monkeypatch.setattr(stock_helper, "_git_commit", lambda: "a" * 40)
    output = tmp_path / "reference.json"
    diagnostic = tmp_path / "diagnostic.json"
    args = SimpleNamespace(
        output=str(output),
        determinism_diagnostic=str(diagnostic),
        config=str(config_path),
        workload=str(workload_path),
        environment=str(environment_path),
        topology=str(topology_path),
        runtime_manifest=str(tmp_path / "runtime.json"),
        correctness_mode="batch-invariant",
        request_count=2,
    )
    with pytest.raises(RuntimeError, match="not deterministic"):
        stock_helper.freeze_with_diagnostic(args)
    assert diagnostic.is_file()
    assert output.exists() is False
    value = json.loads(diagnostic.read_text(encoding="utf-8"))
    assert value["runs"] == smoke["runs"]
    assert value["first_divergent_request_id"] == "A"
    assert value["per_request_comparisons"][0]["first_divergence_position"] == 1
