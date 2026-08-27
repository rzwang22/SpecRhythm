from __future__ import annotations

import json
from pathlib import Path

import pytest

from specrhythm.phase4.numerical_diagnostics import (
    compare_numerical_diagnostics,
    configure_numerical_diagnostic,
    load_numerical_plan,
    token_sha256,
    validate_numerical_records,
)

ROOT = Path(__file__).parents[1]
PLAN = ROOT / "configs/phase4b1_gate3_numerical_diagnostic.json"


def _workload(tmp_path: Path) -> Path:
    configured = json.loads(PLAN.read_text(encoding="utf-8"))["requests"]
    planned_ids = {row["request_id"] for row in configured}
    rows = []
    for index, request_id in enumerate(sorted(planned_ids)):
        rows.append(
            {
                "request_id": request_id,
                "prompt_token_ids": [100 + index, 200 + index],
                "maximum_new_tokens": 32,
            }
        )
    for index in range(96):
        rows.append(
            {
                "request_id": f"other-{index:03d}",
                "prompt_token_ids": [1000 + index],
                "maximum_new_tokens": 2,
            }
        )
    path = tmp_path / "corrected-100.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _summary(seed: str) -> dict:
    digest = (seed.encode().hex() * 64)[:64].ljust(64, "0")
    return {
        "dtype": "torch.bfloat16",
        "shape": [4],
        "raw_sha256": digest,
        "fp32_sha256": digest[::-1],
        "min": -1.0,
        "max": 1.0,
        "norm": 2.0,
        "selected_coordinates": [{"flat_index": 0, "fp32_value": 0.0}],
    }


def _record(item: dict, mode: str, *, plan: dict, stage_seed: str = "a") -> dict:
    prefix = list(item["prompt_token_ids"]) + [91] * int(item["output_position"])
    stock_token = int(item["stock_selected_token_id"])
    resident_token = int(item["resident_selected_token_id"])
    selected = stock_token if mode == "stock-style" else resident_token
    summaries = {name: _summary(stage_seed) for name in (
        "decoder_input_hidden_state",
        "last_layer_branch_output",
        "final_norm_input",
        "final_normalized_hidden_state",
        "lm_head_input",
    )}
    rank_evidence = [
        {
            "tp_rank": rank,
            "tensor_stages": summaries,
            "model_module_paths": {"model": "Qwen3ForCausalLM"},
            "lm_head_input_source_shape": [50, 5120],
            "competing_tokens": [],
            "raw_argmax_token_id": selected,
        }
        for rank in range(2)
    ]
    rank_kv = [
        {
            "key": [item["request_id"], item["output_position"]],
            "tp_rank": rank,
            "kv_cache_before_forward": {
                "aggregate_raw_sha256": _summary(stage_seed)["raw_sha256"],
                "layer_count": 64,
                "logical_token_count": len(prefix) - 1,
            },
            "physical_block_ids": [7],
        }
        for rank in range(2)
    ]
    return {
        "schema_version": "specrhythm.phase4b1-gate3-numerical-record.v1",
        "diagnostic_only": True,
        "execution_mode": mode,
        "plan_sha256": plan["plan_sha256"],
        "workload_sha256": plan["workload_sha256"],
        "tp_world_size": 2,
        "request_id": item["request_id"],
        "output_position": item["output_position"],
        "logical_committed_prefix_token_ids": prefix,
        "logical_committed_prefix_sha256": token_sha256(prefix),
        "num_computed_tokens": len(prefix) - 1,
        "target_input_token_position": len(prefix) - 1,
        "target_input_token_id": prefix[-1],
        "previous_committed_token_id": prefix[-1],
        "tensor_stages": summaries,
        "kv_cache_before_forward": {
            "aggregate_raw_sha256": _summary(stage_seed)["raw_sha256"],
            "layer_count": 64,
        },
        "tp_rank_kv_cache_before_forward": rank_kv,
        "kv_position_mapping": {
            "block_size": 16,
            "logical_positions": list(range(len(prefix) - 1)),
            "physical_block_ids": [7],
        },
        "raw_pre_softmax_logits": {
            "dtype": "torch.bfloat16",
            "competing_tokens": [
                {"token_id": stock_token, "raw_logit": 1.0 if mode == "stock-style" else 0.5},
                {"token_id": resident_token, "raw_logit": 0.875 if mode == "stock-style" else 0.5},
            ],
        },
        "raw_argmax_token_id": selected,
        "tp_rank_evidence": rank_evidence,
        "model_module_paths": {
            "model": "Qwen3ForCausalLM",
            "decoder_input": "model.embed_tokens",
            "final_norm": "model.norm",
            "lm_head": "lm_head",
            "logits_processor": "logits_processor",
        },
        "lm_head": {
            "module_class": "ParallelLMHead",
            "quant_method_class": "UnquantizedLinearMethod",
            "input_dtype": "torch.bfloat16",
            "output_dtype": "torch.bfloat16",
            "path": "LogitsProcessor -> lm_head",
            "batch_invariant_kernel": "matmul_persistent",
        },
        "visible_to_draft": False,
        "execution_shape": {"active_request_count": 50, "lm_head_m": 50},
    }


def _outputs(plan: dict, mode: str) -> list[dict]:
    result = []
    for item in plan["requests"]:
        tokens = [91] * (int(item["output_position"]) + 1)
        tokens[-1] = int(
            item[
                "stock_selected_token_id"
                if mode == "stock-style"
                else "resident_selected_token_id"
            ]
        )
        result.append({"request_id": item["request_id"], "generated_token_ids": tokens})
    for index in range(96):
        result.append(
            {"request_id": f"other-{index:03d}", "generated_token_ids": [index]}
        )
    return result


def test_plan_resolves_exact_four_points_against_frozen_shape(tmp_path):
    workload = _workload(tmp_path)
    plan = load_numerical_plan(PLAN, workload)
    assert plan["expected_request_count"] == 100
    assert len(plan["requests"]) == 4
    assert all(item["prompt_token_ids"] for item in plan["requests"])
    assert plan["workload_sha256"]


def test_configuration_is_all_or_nothing_and_refuses_overwrite(tmp_path, monkeypatch):
    workload = _workload(tmp_path)
    output = tmp_path / "numerical.jsonl"
    with pytest.raises(ValueError, match="all required"):
        configure_numerical_diagnostic(
            plan_path=PLAN,
            output_path=None,
            workload_path=workload,
            execution_mode="stock-style",
        )
    report = configure_numerical_diagnostic(
        plan_path=PLAN,
        output_path=output,
        workload_path=workload,
        execution_mode="stock-style",
    )
    assert report is not None
    output.write_text("immutable\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        configure_numerical_diagnostic(
            plan_path=PLAN,
            output_path=output,
            workload_path=workload,
            execution_mode="stock-style",
        )
    monkeypatch.delenv("SR_PHASE4_NUMERICAL_DIAGNOSTIC_PLAN", raising=False)
    monkeypatch.delenv("SR_PHASE4_NUMERICAL_DIAGNOSTIC_OUTPUT", raising=False)
    monkeypatch.delenv("SR_PHASE4_NUMERICAL_DIAGNOSTIC_MODE", raising=False)


def test_records_fail_closed_on_missing_duplicate_or_draft_visible(tmp_path):
    plan = load_numerical_plan(PLAN, _workload(tmp_path))
    rows = [
        _record(item, "stock-style", plan=plan) for item in plan["requests"]
    ]
    assert validate_numerical_records(rows, plan, execution_mode="stock-style") == []
    assert any("missing" in error for error in validate_numerical_records(
        rows[:-1], plan, execution_mode="stock-style"
    ))
    assert any("duplicate" in error for error in validate_numerical_records(
        rows + [rows[0]], plan, execution_mode="stock-style"
    ))
    rows[0]["visible_to_draft"] = True
    assert any("Draft-visible" in error for error in validate_numerical_records(
        rows, plan, execution_mode="stock-style"
    ))


def test_records_fail_closed_on_missing_tp_rank_evidence(tmp_path):
    plan = load_numerical_plan(PLAN, _workload(tmp_path))
    rows = [
        _record(item, "stock-style", plan=plan) for item in plan["requests"]
    ]
    rows[0]["tp_rank_evidence"] = rows[0]["tp_rank_evidence"][:1]
    errors = validate_numerical_records(rows, plan, execution_mode="stock-style")
    assert any("per-rank tensor evidence" in error for error in errors)

    rows = [
        _record(item, "stock-style", plan=plan) for item in plan["requests"]
    ]
    rows[0]["tp_rank_kv_cache_before_forward"][1][
        "kv_cache_before_forward"
    ].pop("aggregate_raw_sha256")
    errors = validate_numerical_records(rows, plan, execution_mode="stock-style")
    assert any("per-rank KV checksum" in error for error in errors)


def test_exact_comparison_localizes_first_raw_stage_and_proves_sampler(tmp_path):
    plan = load_numerical_plan(PLAN, _workload(tmp_path))
    stock = [
        _record(item, "stock-style", plan=plan, stage_seed="a")
        for item in plan["requests"]
    ]
    resident = [
        _record(item, "resident-target", plan=plan, stage_seed="a")
        for item in plan["requests"]
    ]
    resident[0]["tensor_stages"]["final_norm_input"] = _summary("b")
    resident[0]["tp_rank_evidence"][0]["tensor_stages"][
        "final_norm_input"
    ] = _summary("b")
    resident[1]["kv_cache_before_forward"]["aggregate_raw_sha256"] = "f" * 64
    report = compare_numerical_diagnostics(
        plan=plan,
        stock_rows=stock,
        resident_rows=resident,
        stock_outputs=_outputs(plan, "stock-style"),
        resident_outputs=_outputs(plan, "resident-target"),
    )
    assert report["valid"] is True
    assert report["tolerant_correctness_policy"] is False
    assert report["tie_equivalent_tokens_accepted"] is False
    by_id = {row["request_id"]: row for row in report["comparisons"]}
    assert (
        by_id[plan["requests"][0]["request_id"]]["first_observed_difference"]
        == "final_norm_input"
    )
    assert (
        by_id[plan["requests"][1]["request_id"]]["first_observed_difference"]
        == "kv-cache-raw-values"
    )
    assert all(
        all(row["raw_argmax_matches_output"].values())
        for row in report["comparisons"]
    )


def test_runner_patch_is_observational_and_installed_before_forward():
    patch = (ROOT / "integrations/vllm/patches/0004-gate3-numerical-observer.patch").read_text(
        encoding="utf-8"
    )
    assert "prepare_target_numerical_diagnostic(" in patch
    assert patch.index("prepare_target_numerical_diagnostic(") < patch.index(
        "target_forward_start_ns = time.monotonic_ns()"
    )
    forbidden = ("scheduler.schedule", "sampled_token_ids =", "spec_token_ids =")
    assert all(value not in patch for value in forbidden)


def test_cli_exposes_only_explicit_diagnostic_contract():
    cli = (ROOT / "src/specrhythm/cli.py").read_text(encoding="utf-8")
    assert "--diagnostic-single-run" in cli
    assert "--numerical-diagnostic-plan" in cli
    assert "phase4b1-gate3-numerical-compare" in cli
    assert "tie-equivalent tokens" in cli
