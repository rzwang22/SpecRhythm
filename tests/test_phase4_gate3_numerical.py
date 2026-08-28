from __future__ import annotations

import copy
import inspect
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from specrhythm.phase4 import numerical_diagnostics as numerical
from specrhythm.phase4.numerical_diagnostics import (
    _generic_kv_ownership,
    compare_numerical_diagnostics,
    configure_numerical_diagnostic,
    load_numerical_plan,
    prepare_target_numerical_diagnostic,
    token_sha256,
    validate_numerical_records,
)

ROOT = Path(__file__).parents[1]
PLAN = ROOT / "configs/phase4b1_gate3_numerical_diagnostic.json"


class _Vector:
    def __init__(self, values):
        self.values = list(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)


class _TokenMatrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]

    def __getitem__(self, key):
        row, column = key
        return _Vector(self.rows[row][column])


class _NumpyRows:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]

    def __getitem__(self, key):
        row, column = key
        return _Vector(self.rows[row][column])


class _BlockTable:
    def __init__(self, block_size, rows):
        self.block_size = block_size
        self.rows = [list(row) for row in rows]
        self.num_blocks_per_row = [len(row) for row in self.rows]

    def get_numpy_array(self):
        return _NumpyRows(self.rows)


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


def _runner_for_plan(plan: dict, *, speculative: bool, with_ownership: bool = True):
    item = plan["requests"][0]
    prefix = list(item["prompt_token_ids"]) + [91] * item["output_position"]
    block_table = SimpleNamespace(block_tables=[_BlockTable(16, [[7]])])
    input_batch = SimpleNamespace(
        req_ids=["internal-0"],
        num_tokens_no_spec=[len(prefix)],
        token_ids_cpu=_TokenMatrix([prefix]),
        num_computed_tokens_cpu=[len(prefix) - 1],
        block_table=(block_table if with_ownership else None),
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=["model.layers.0.self_attn.attn"])
        ]
    )
    return SimpleNamespace(
        input_batch=input_batch,
        kv_cache_config=(kv_cache_config if with_ownership else None),
        speculative_config=(object() if speculative else None),
    )


def _install_fake_tp(monkeypatch):
    parallel = ModuleType("vllm.distributed.parallel_state")
    parallel.get_tp_group = lambda: SimpleNamespace(
        rank_in_group=0, world_size=2, cpu_group=object()
    )
    distributed = ModuleType("vllm.distributed")
    distributed.parallel_state = parallel
    vllm = ModuleType("vllm")
    vllm.distributed = distributed
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)
    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", parallel)


def _run_prepare(monkeypatch, plan: dict, runner, *, planned: bool = True):
    _install_fake_tp(monkeypatch)
    state = {
        "plan_by_id": {
            row["request_id"]: row for row in plan["requests"]
        },
        "captured": set(),
        "active": {},
    }
    if not planned:
        state["plan_by_id"] = {
            "not-this-request": {
                **plan["requests"][0],
                "request_id": "not-this-request",
                "prompt_token_ids": [999999],
            }
        }
    monkeypatch.setattr(numerical, "_numerical_state", lambda _runner: state)
    monkeypatch.setattr(
        numerical,
        "_kv_cache_summary",
        lambda _runner, _active: {
            "layout": "flash-attention-paged-kv[num_blocks,2,block_size,...]",
            "logical_token_count": _active["num_computed_tokens"],
            "kv_cache_group_count": 1,
            "layer_count": 1,
            "aggregate_raw_sha256": "a" * 64,
            "layers": [],
        },
    )

    def gather(rows, _group):
        peer = copy.deepcopy(rows)
        for row in peer:
            row["tp_rank"] = 1
        return [rows, peer]

    monkeypatch.setattr(numerical, "_gather_tp_objects", gather)
    for name, value in (
        (numerical.PLAN_ENV, "/tmp/plan.json"),
        (numerical.OUTPUT_ENV, "/tmp/numerical.jsonl"),
        (numerical.MODE_ENV, "stock-style"),
    ):
        monkeypatch.setenv(name, value)
    item = plan["requests"][0]
    prefix_length = len(item["prompt_token_ids"]) + item["output_position"]
    prepare_target_numerical_diagnostic(
        runner,
        scheduler_output=SimpleNamespace(scheduled_spec_decode_tokens={}),
        logits_indices=_Vector([0]),
        positions=_Vector([prefix_length - 1]),
        num_scheduled_tokens=[1],
        slot_mappings_by_group=(
            {0: _Vector([113])}
            if getattr(runner.input_batch, "block_table", None) is not None
            else None
        ),
    )
    return state


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


def _ownership(prefix: list[int], *, group_count: int = 1) -> dict:
    computed = len(prefix) - 1
    groups = []
    for group_id in range(group_count):
        groups.append(
            {
                "kv_cache_group_id": group_id,
                "block_size": 16,
                "num_blocks_per_row": 1,
                "logical_used_block_count": 1,
                "physical_block_ids": [7 + group_id],
                "current_query_slot_mapping": [113 + group_id],
                "layer_names": [f"model.layers.{group_id}.self_attn.attn"],
            }
        )
    return {
        "authority": (
            "InputBatch.MultiGroupBlockTable + "
            "GPUModelRunner._get_slot_mappings.slot_mappings_by_group"
        ),
        "kv_cache_group_count": group_count,
        "logical_positions": list(range(computed)),
        "groups": groups,
    }


def _record(item: dict, mode: str, *, plan: dict, stage_seed: str = "a") -> dict:
    prefix = list(item["prompt_token_ids"]) + [91] * int(item["output_position"])
    stock_token = int(item["stock_selected_token_id"])
    resident_token = int(item["resident_selected_token_id"])
    selected = stock_token if mode == "stock-style" else resident_token
    ownership = _ownership(prefix)
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
                "kv_cache_group_count": 1,
            },
            "kv_ownership": ownership,
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
            "logical_token_count": len(prefix) - 1,
            "kv_cache_group_count": 1,
        },
        "tp_rank_kv_cache_before_forward": rank_kv,
        "kv_position_mapping": ownership,
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


@pytest.mark.parametrize("speculative", (False, True))
def test_observer_accepts_none_spec_common_and_resident_uses_generic_authority(
    tmp_path, monkeypatch, speculative
):
    plan = load_numerical_plan(PLAN, _workload(tmp_path))
    runner = _runner_for_plan(plan, speculative=speculative)
    state = _run_prepare(monkeypatch, plan, runner)
    active = next(iter(state["active"].values()))
    assert active["kv_ownership"]["kv_cache_group_count"] == 1
    assert active["kv_ownership"]["groups"][0]["physical_block_ids"] == [7]
    assert active["kv_ownership"]["groups"][0][
        "current_query_slot_mapping"
    ] == [113]
    signature = inspect.signature(prepare_target_numerical_diagnostic)
    assert "common_attention_metadata" not in signature.parameters
    assert "slot_mappings_by_group" in signature.parameters


def test_stock_and_resident_logical_ownership_schema_is_identical(
    tmp_path, monkeypatch
):
    plan = load_numerical_plan(PLAN, _workload(tmp_path))
    stock = _run_prepare(
        monkeypatch,
        plan,
        _runner_for_plan(plan, speculative=False),
    )
    stock_ownership = copy.deepcopy(
        next(iter(stock["active"].values()))["kv_ownership"]
    )
    resident = _run_prepare(
        monkeypatch,
        plan,
        _runner_for_plan(plan, speculative=True),
    )
    resident_ownership = next(iter(resident["active"].values()))["kv_ownership"]
    assert stock_ownership == resident_ownership
    assert set(stock_ownership) == set(resident_ownership)


def test_irrelevant_forward_does_not_require_generic_ownership(tmp_path, monkeypatch):
    plan = load_numerical_plan(PLAN, _workload(tmp_path))
    runner = _runner_for_plan(plan, speculative=False, with_ownership=False)
    state = _run_prepare(monkeypatch, plan, runner, planned=False)
    assert state["active"] == {}


def test_planned_checkpoint_missing_generic_ownership_fails_closed(
    tmp_path, monkeypatch
):
    plan = load_numerical_plan(PLAN, _workload(tmp_path))
    runner = _runner_for_plan(plan, speculative=False, with_ownership=False)
    with pytest.raises(RuntimeError, match="MultiGroupBlockTable is missing"):
        _run_prepare(monkeypatch, plan, runner)


def test_generic_multigroup_ownership_is_explicit_and_schema_stable(tmp_path):
    plan = load_numerical_plan(PLAN, _workload(tmp_path))
    runner = _runner_for_plan(plan, speculative=False)
    runner.input_batch.block_table.block_tables[0] = _BlockTable(16, [[7, 8]])
    runner.input_batch.block_table.block_tables.append(
        _BlockTable(8, [[11, 12, 13]])
    )
    runner.kv_cache_config.kv_cache_groups.append(
        SimpleNamespace(layer_names=["model.layers.1.self_attn.attn"])
    )
    ownership = _generic_kv_ownership(
        runner,
        request_index=0,
        num_computed_tokens=17,
        flat_query_start=0,
        query_length=1,
        slot_mappings_by_group={0: _Vector([113]), 1: _Vector([201])},
    )
    assert ownership["kv_cache_group_count"] == 2
    assert [group["block_size"] for group in ownership["groups"]] == [16, 8]
    assert ownership["groups"][1]["physical_block_ids"] == [11, 12, 13]
    assert set(ownership) == {
        "authority",
        "kv_cache_group_count",
        "logical_positions",
        "groups",
    }


def test_generic_ownership_rejects_unsupported_group_layout(tmp_path):
    plan = load_numerical_plan(PLAN, _workload(tmp_path))
    runner = _runner_for_plan(plan, speculative=False)
    with pytest.raises(RuntimeError, match="slot mappings are incomplete"):
        _generic_kv_ownership(
            runner,
            request_index=0,
            num_computed_tokens=1,
            flat_query_start=0,
            query_length=1,
            slot_mappings_by_group={},
        )
    runner.input_batch.block_table.block_tables[0].block_size = 0
    with pytest.raises(RuntimeError, match="block size is invalid"):
        _generic_kv_ownership(
            runner,
            request_index=0,
            num_computed_tokens=1,
            flat_query_start=0,
            query_length=1,
            slot_mappings_by_group={0: _Vector([113])},
        )


def test_exact_gate3_validator_rejects_unproven_multigroup_runtime(tmp_path):
    plan = load_numerical_plan(PLAN, _workload(tmp_path))
    rows = [
        _record(item, "stock-style", plan=plan) for item in plan["requests"]
    ]
    rows[0]["kv_position_mapping"] = _ownership(
        rows[0]["logical_committed_prefix_token_ids"], group_count=2
    )
    errors = validate_numerical_records(rows, plan, execution_mode="stock-style")
    assert any("generic KV ownership is invalid" in error for error in errors)


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
    assert any("per-rank KV evidence" in error for error in errors)


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
    assert "slot_mappings_by_group=slot_mappings_by_group" in patch
    assert "common_attention_metadata=spec_decode_common_attn_metadata" not in patch
    forbidden = ("scheduler.schedule", "sampled_token_ids =", "spec_token_ids =")
    assert all(value not in patch for value in forbidden)


def test_cli_exposes_only_explicit_diagnostic_contract():
    cli = (ROOT / "src/specrhythm/cli.py").read_text(encoding="utf-8")
    assert "--diagnostic-single-run" in cli
    assert "--numerical-diagnostic-plan" in cli
    assert "phase4b1-gate3-numerical-compare" in cli
    assert "tie-equivalent tokens" in cli
