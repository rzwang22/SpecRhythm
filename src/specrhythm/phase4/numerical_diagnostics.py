"""Diagnostic-only Gate3 numerical localization.

The GPU entry points in this module are inert unless an explicit immutable
plan and output path are configured before vLLM worker creation.  Captured
Target tensors never enter the Draft transport or any correctness decision.
Pure plan/record validation remains dependency-free for Python 3.9 CPU CI.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from specrhythm.phase4.manifest import sha256_file
from specrhythm.phase4.transport import CheckpointJsonl

PLAN_SCHEMA = "specrhythm.phase4b1-gate3-numerical-plan.v1"
RECORD_SCHEMA = "specrhythm.phase4b1-gate3-numerical-record.v1"
COMPARISON_SCHEMA = "specrhythm.phase4b1-gate3-numerical-comparison.v1"
PER_TOKEN_PLAN_SCHEMA = "specrhythm.phase4b1-gate3-per-token-kv-plan.v1"
PER_TOKEN_RECORD_SCHEMA = "specrhythm.phase4b1-gate3-per-token-kv-record.v1"
PER_TOKEN_COMPARISON_SCHEMA = (
    "specrhythm.phase4b1-gate3-per-token-kv-comparison.v1"
)
PLAN_ENV = "SR_PHASE4_NUMERICAL_DIAGNOSTIC_PLAN"
OUTPUT_ENV = "SR_PHASE4_NUMERICAL_DIAGNOSTIC_OUTPUT"
MODE_ENV = "SR_PHASE4_NUMERICAL_DIAGNOSTIC_MODE"
ALLOWED_MODES = {"stock-style", "matched-stock-async-off", "resident-target"}
LOGICAL_RAW_ORDERING = (
    "contiguous logical tensor [K_or_V,logical_position,...] in C order; "
    "complete K plane followed by complete V plane"
)
TOKEN_DIGEST_ORDERING = (
    "domain-separated SHA256 over ascending logical_position, then K digest, "
    "then V digest"
)
REQUIRED_STAGES = (
    "decoder_input_hidden_state",
    "last_layer_branch_output",
    "final_norm_input",
    "final_normalized_hidden_state",
    "lm_head_input",
)
EXPECTED_PER_TOKEN_CHECKPOINTS = {
    "r3-c7ee1a73ee79dd6dc21cb8dc": (3, 3, 4, 600, 296),
    "r3-646c340a0281105c1c20de27": (12, 20, 21, 448, 323),
    "r3-32ae44a69fffd76f0dd4b787": (4, 23, 24, 3435, 15),
    "r3-e00f5312321ec537a9c716cd": (2, 55, 56, 23826, 1674),
}
EXPECTED_PER_TOKEN_LAYERS = {
    request_id: (values[1], values[2])
    for request_id, values in EXPECTED_PER_TOKEN_CHECKPOINTS.items()
}
SEMANTIC_PREFIX_AUTHORITY = "final-run-output-vs-immutable-stock-reference"
SOURCE_GATE3_COMMIT = "32b09a6749dc44200fffe37411002d862ca1098a"
SOURCE_COARSE_DIAGNOSTIC_COMMIT = "e73e8848904eee7f18e7beba80d1ec2da94e8267"
SOURCE_COARSE_DIAGNOSTIC_ROOT = "phase4b1-gate3-numerical-20260904T070021Z"


def load_numerical_plan(path: Path, workload_path: Path) -> dict[str, Any]:
    """Load and resolve the four-point diagnostic plan against one workload."""

    value = json.loads(path.read_text(encoding="utf-8"))
    schema_version = value.get("schema_version") if isinstance(value, Mapping) else None
    if schema_version not in {PLAN_SCHEMA, PER_TOKEN_PLAN_SCHEMA}:
        raise ValueError("unsupported Gate3 numerical diagnostic plan")
    if value.get("diagnostic_only") is not True:
        raise ValueError("Gate3 numerical plan must be diagnostic-only")
    expected_count = value.get("expected_request_count")
    if expected_count != 100:
        raise ValueError("Gate3 numerical plan requires the frozen 100-request shape")
    if value.get("position_indexing") != "zero-based-generated-token-index":
        raise ValueError("Gate3 numerical output positions must be explicitly zero-based")
    if schema_version == PER_TOKEN_PLAN_SCHEMA and (
        value.get("materialized_kv_position_range") != "[0,num_computed_tokens)"
        or value.get("source_gate3_commit") != SOURCE_GATE3_COMMIT
        or value.get("source_coarse_diagnostic_commit")
        != SOURCE_COARSE_DIAGNOSTIC_COMMIT
        or value.get("source_coarse_diagnostic_root")
        != SOURCE_COARSE_DIAGNOSTIC_ROOT
    ):
        raise ValueError("Gate3 per-token plan provenance or KV boundary differs")
    workload_rows = _load_jsonl(workload_path)
    if len(workload_rows) != expected_count:
        raise ValueError("Gate3 numerical workload must contain exactly 100 requests")
    workload_by_id = {}
    for row in workload_rows:
        request_id = str(row.get("request_id", ""))
        if not request_id or request_id in workload_by_id:
            raise ValueError("Gate3 numerical workload request IDs are invalid")
        workload_by_id[request_id] = row

    raw_requests = value.get("requests")
    if not isinstance(raw_requests, list) or len(raw_requests) != 4:
        raise ValueError("Gate3 numerical plan must contain exactly four requests")
    resolved = []
    seen = set()
    for row in raw_requests:
        if not isinstance(row, Mapping):
            raise ValueError("Gate3 numerical request entry is not an object")
        request_id = str(row.get("request_id", ""))
        position = row.get("output_position")
        stock_token = row.get("stock_selected_token_id")
        resident_token = row.get("resident_selected_token_id")
        if not request_id or request_id in seen or request_id not in workload_by_id:
            raise ValueError("Gate3 numerical request identity is invalid")
        if not _nonnegative_int(position):
            raise ValueError("Gate3 numerical output position is invalid")
        if not _nonnegative_int(stock_token) or not _nonnegative_int(resident_token):
            raise ValueError("Gate3 numerical competing token ID is invalid")
        if stock_token == resident_token:
            raise ValueError("Gate3 numerical competing tokens must differ")
        maximum = workload_by_id[request_id].get("maximum_new_tokens")
        if not _positive_int(maximum) or position >= maximum:
            raise ValueError("Gate3 numerical output position exceeds the frozen request")
        prompt = workload_by_id[request_id].get("prompt_token_ids")
        if not _token_list(prompt):
            raise ValueError("Gate3 numerical request has invalid frozen prompt tokens")
        layer_fields = {}
        if schema_version == PER_TOKEN_PLAN_SCHEMA:
            control_layer = row.get("control_layer_index")
            first_layer = row.get("first_different_layer_index")
            if (
                not _nonnegative_int(control_layer)
                or not _positive_int(first_layer)
                or first_layer != control_layer + 1
                or EXPECTED_PER_TOKEN_CHECKPOINTS.get(request_id)
                != (
                    position,
                    control_layer,
                    first_layer,
                    stock_token,
                    resident_token,
                )
            ):
                raise ValueError(
                    "Gate3 per-token plan has an unverified control/first layer pair"
                )
            layer_fields = {
                "control_layer_index": int(control_layer),
                "control_layer_name": (
                    f"model.layers.{control_layer}.self_attn.attn"
                ),
                "first_different_layer_index": int(first_layer),
                "first_different_layer_name": (
                    f"model.layers.{first_layer}.self_attn.attn"
                ),
            }
        seen.add(request_id)
        resolved.append(
            {
                "request_id": request_id,
                "output_position": int(position),
                "stock_selected_token_id": int(stock_token),
                "resident_selected_token_id": int(resident_token),
                "prompt_token_ids": list(prompt),
                "prompt_sha256": token_sha256(prompt),
                "prompt_length": len(prompt),
                "maximum_new_tokens": int(maximum),
                **layer_fields,
            }
        )
    if schema_version == PER_TOKEN_PLAN_SCHEMA and seen != set(
        EXPECTED_PER_TOKEN_CHECKPOINTS
    ):
        raise ValueError("Gate3 per-token plan does not contain the exact four requests")
    return {
        "schema_version": schema_version,
        "diagnostic_only": True,
        "diagnostic_kind": (
            "per-logical-token-kv"
            if schema_version == PER_TOKEN_PLAN_SCHEMA
            else "coarse-numerical-localization"
        ),
        "expected_request_count": 100,
        "position_indexing": "zero-based-generated-token-index",
        "source_gate3_commit": str(value.get("source_gate3_commit", "")),
        "source_failed_root": str(value.get("source_failed_root", "")),
        "source_coarse_diagnostic_root": str(
            value.get("source_coarse_diagnostic_root", "")
        ),
        "source_coarse_diagnostic_commit": str(
            value.get("source_coarse_diagnostic_commit", "")
        ),
        "materialized_kv_position_range": (
            "[0,num_computed_tokens)"
            if schema_version == PER_TOKEN_PLAN_SCHEMA
            else None
        ),
        "workload_file": workload_path.name,
        "workload_sha256": sha256_file(workload_path),
        "plan_file": path.name,
        "plan_sha256": sha256_file(path),
        "requests": resolved,
    }


def configure_numerical_diagnostic(
    *,
    plan_path: Optional[Path],
    output_path: Optional[Path],
    workload_path: Path,
    execution_mode: Optional[str],
) -> Optional[dict[str, Any]]:
    """Validate and export one diagnostic contract before importing vLLM."""

    values = (plan_path, output_path, execution_mode)
    if all(item is None for item in values):
        for name in (PLAN_ENV, OUTPUT_ENV, MODE_ENV):
            os.environ.pop(name, None)
        return None
    if any(item is None for item in values):
        raise ValueError("numerical plan, output, and execution mode are all required")
    assert plan_path is not None and output_path is not None
    assert execution_mode is not None
    if execution_mode not in ALLOWED_MODES:
        raise ValueError("unsupported Gate3 numerical execution mode")
    if output_path.exists():
        raise FileExistsError("refusing to overwrite Gate3 numerical diagnostics")
    plan = load_numerical_plan(plan_path, workload_path)
    os.environ[PLAN_ENV] = str(plan_path.resolve())
    os.environ[OUTPUT_ENV] = str(output_path.resolve())
    os.environ[MODE_ENV] = execution_mode
    return plan


def validate_numerical_records(
    rows: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    *,
    execution_mode: str,
) -> list[str]:
    errors = []
    per_token_plan = plan.get("schema_version") == PER_TOKEN_PLAN_SCHEMA
    expected_record_schema = (
        PER_TOKEN_RECORD_SCHEMA if per_token_plan else RECORD_SCHEMA
    )
    planned = {
        (str(row["request_id"]), int(row["output_position"])): row
        for row in plan.get("requests", ())
        if isinstance(row, Mapping)
    }
    observed = {}
    for row in rows:
        key = (str(row.get("request_id", "")), row.get("output_position"))
        if key in observed:
            errors.append(f"duplicate numerical checkpoint: {key}")
            continue
        observed[key] = row
        if row.get("schema_version") != expected_record_schema:
            errors.append(f"unsupported numerical record schema: {key}")
        if row.get("execution_mode") != execution_mode:
            errors.append(f"numerical execution mode differs: {key}")
        if row.get("diagnostic_only") is not True:
            errors.append(f"numerical record is not diagnostic-only: {key}")
        if row.get("visible_to_draft") is not False:
            errors.append(f"numerical record is Draft-visible: {key}")
        if row.get("plan_sha256") != plan.get("plan_sha256"):
            errors.append(f"numerical plan checksum differs: {key}")
        if row.get("workload_sha256") != plan.get("workload_sha256"):
            errors.append(f"numerical workload checksum differs: {key}")
        if row.get("tp_world_size") != 2:
            errors.append(f"Gate3 numerical record is not TP=2: {key}")
        if execution_mode == "matched-stock-async-off":
            control = row.get("matched_bootstrap_control")
            if not isinstance(control, Mapping) or control != {
                "async_scheduling_requested": False,
                "async_scheduling_effective": False,
                "speculative_config": None,
                "scheduler_class": "vllm.v1.core.sched.scheduler.Scheduler",
                "custom_class_proposer_absent": True,
                "resident_setup_scheduler_absent": True,
                "target_tp_world_size": 2,
                "physical_target_gpu_ids": [1, 2],
            }:
                errors.append(f"matched-bootstrap runtime control is invalid: {key}")
            shape = row.get("execution_shape")
            if not isinstance(shape, Mapping) or (
                not _positive_int(shape.get("active_request_count"))
                or shape.get("number_of_active_requests")
                != shape.get("active_request_count")
                or not _positive_int(shape.get("sampled_logits_rows"))
                or not _positive_int(shape.get("lm_head_m"))
                or not _positive_int(shape.get("total_scheduled_token_count"))
                or shape.get("planned_request_query_length") != 1
                or not _nonnegative_int(shape.get("decode_row_count"))
                or not _nonnegative_int(shape.get("prefill_row_count"))
                or shape.get("decode_row_count") + shape.get("prefill_row_count")
                != shape.get("active_request_count")
                or shape.get("batch_row_kind")
                not in {"decode-only", "mixed-prefill-decode"}
            ):
                errors.append(f"matched-bootstrap execution shape is invalid: {key}")
        planned_row = planned.get(key)
        if per_token_plan:
            placeholder = row.get("async_cpu_placeholder_view")
            tokens = (
                placeholder.get("token_ids")
                if isinstance(placeholder, Mapping)
                else None
            )
            if (
                not isinstance(placeholder, Mapping)
                or not isinstance(tokens, list)
                or not _signed_token_list(tokens)
                or placeholder.get("source") != "InputBatch.token_ids_cpu"
                or placeholder.get("semantic_prefix_authority") is not False
                or placeholder.get("token_ids_sha256") != token_sha256(tokens or [])
                or placeholder.get("contains_negative_placeholder")
                != any(token_id < 0 for token_id in tokens or [])
            ):
                errors.append(f"numerical async CPU placeholder view is invalid: {key}")
            if isinstance(tokens, list) and isinstance(planned_row, Mapping):
                prompt = list(planned_row["prompt_token_ids"])
                position = int(planned_row["output_position"])
                computed = len(prompt) + position - 1
                if tokens[: len(prompt)] != prompt or len(tokens) != len(prompt) + position:
                    errors.append(
                        f"numerical async CPU placeholder boundary is invalid: {key}"
                    )
                if (
                    row.get("prompt_length") != len(prompt)
                    or row.get("num_computed_tokens") != computed
                    or row.get("target_input_token_position") != computed
                ):
                    errors.append(
                        f"numerical materialized Target boundary is inconsistent: {key}"
                    )
                if not _valid_per_token_capture(
                    row.get("per_logical_token_kv"),
                    planned_row=planned_row,
                    num_computed_tokens=computed,
                    expected_world_size=2,
                    require_independent_binding=(
                        execution_mode == "matched-stock-async-off"
                    ),
                ):
                    errors.append(f"numerical per-token KV capture is invalid: {key}")
                mapping = row.get("kv_position_mapping")
                groups = mapping.get("groups") if isinstance(mapping, Mapping) else None
                owned_layers = (
                    list(groups[0].get("layer_names", ()))
                    if isinstance(groups, list)
                    and len(groups) == 1
                    and isinstance(groups[0], Mapping)
                    else []
                )
                if any(
                    owned_layers.count(planned_row[name]) != 1
                    for name in (
                        "control_layer_name",
                        "first_different_layer_name",
                    )
                ):
                    errors.append(f"numerical per-token layer ownership is ambiguous: {key}")
        else:
            prefix = row.get("logical_committed_prefix_token_ids")
            if not isinstance(prefix, list) or row.get(
                "logical_committed_prefix_sha256"
            ) != token_sha256(prefix if isinstance(prefix, list) else []):
                errors.append(f"numerical prefix checksum is invalid: {key}")
            if isinstance(prefix, list) and isinstance(planned_row, Mapping):
                prompt = list(planned_row["prompt_token_ids"])
                position = int(planned_row["output_position"])
                if (
                    prefix[: len(prompt)] != prompt
                    or len(prefix) != len(prompt) + position
                ):
                    errors.append(
                        f"numerical committed prefix is not the planned boundary: {key}"
                    )
                computed = len(prefix) - 1
                if (
                    row.get("num_computed_tokens") != computed
                    or row.get("target_input_token_position") != computed
                    or row.get("target_input_token_id") != prefix[-1]
                    or row.get("previous_committed_token_id") != prefix[-1]
                ):
                    errors.append(
                        f"numerical pending Target input is inconsistent: {key}"
                    )
        stages = row.get("tensor_stages")
        if not isinstance(stages, Mapping):
            errors.append(f"numerical tensor stages are missing: {key}")
        else:
            for name in REQUIRED_STAGES:
                summary = stages.get(name)
                if not isinstance(summary, Mapping) or not _valid_tensor_summary(summary):
                    errors.append(f"numerical tensor stage {name} is invalid: {key}")
        kv = row.get("kv_cache_before_forward")
        if not isinstance(kv, Mapping) or not kv.get("aggregate_raw_sha256"):
            errors.append(f"numerical KV checksum is missing: {key}")
        mapping = row.get("kv_position_mapping")
        if not _valid_kv_ownership(
            mapping,
            num_computed_tokens=row.get("num_computed_tokens"),
            expected_group_count=1,
        ):
            errors.append(f"numerical generic KV ownership is invalid: {key}")
        rank_kv = row.get("tp_rank_kv_cache_before_forward")
        if not _valid_tp_rank_rows(rank_kv, expected_world_size=2):
            errors.append(f"numerical per-rank KV evidence is invalid: {key}")
        elif any(
            not isinstance(rank_row.get("kv_cache_before_forward"), Mapping)
            or not rank_row["kv_cache_before_forward"].get("aggregate_raw_sha256")
            or not _valid_kv_ownership(
                rank_row.get("kv_ownership"),
                num_computed_tokens=row.get("num_computed_tokens"),
                expected_group_count=1,
            )
            for rank_row in rank_kv
        ):
            errors.append(f"numerical per-rank KV evidence is incomplete: {key}")
        if per_token_plan and not _per_token_aggregates_match_rank_kv(
            row.get("per_logical_token_kv"), rank_kv
        ):
            errors.append(
                f"numerical per-token checksums do not match rank KV evidence: {key}"
            )
        rank_evidence = row.get("tp_rank_evidence")
        if not _valid_tp_rank_rows(rank_evidence, expected_world_size=2):
            errors.append(f"numerical per-rank tensor evidence is invalid: {key}")
        else:
            for rank_row in rank_evidence:
                rank_stages = rank_row.get("tensor_stages")
                if not isinstance(rank_stages, Mapping) or any(
                    not isinstance(rank_stages.get(name), Mapping)
                    or not _valid_tensor_summary(rank_stages[name])
                    for name in REQUIRED_STAGES
                ):
                    errors.append(f"numerical TP-rank tensor stage is invalid: {key}")
                    break
        logits = row.get("raw_pre_softmax_logits")
        if not isinstance(logits, Mapping):
            errors.append(f"numerical raw logits are missing: {key}")
        else:
            candidates = logits.get("competing_tokens")
            if not isinstance(candidates, list) or len(candidates) != 2:
                errors.append(f"numerical competing logits are invalid: {key}")
            elif isinstance(planned_row, Mapping) and [
                candidate.get("token_id")
                for candidate in candidates
                if isinstance(candidate, Mapping)
            ] != [
                planned_row["stock_selected_token_id"],
                planned_row["resident_selected_token_id"],
            ]:
                errors.append(f"numerical competing tokens differ from plan: {key}")
        if row.get("raw_argmax_token_id") is None:
            errors.append(f"numerical raw-logit argmax token is missing: {key}")
        model_paths = row.get("model_module_paths")
        if not isinstance(model_paths, Mapping) or any(
            not model_paths.get(name)
            for name in ("model", "decoder_input", "final_norm", "lm_head", "logits_processor")
        ):
            errors.append(f"numerical model module path evidence is invalid: {key}")
        lm_head = row.get("lm_head")
        if not isinstance(lm_head, Mapping) or any(
            not lm_head.get(name)
            for name in (
                "module_class",
                "quant_method_class",
                "input_dtype",
                "output_dtype",
                "path",
                "batch_invariant_kernel",
            )
        ):
            errors.append(f"numerical LM-head path evidence is invalid: {key}")
    missing = set(planned) - set(observed)
    extra = set(observed) - set(planned)
    if missing:
        errors.append(f"missing numerical checkpoints: {sorted(missing)}")
    if extra:
        errors.append(f"unexpected numerical checkpoints: {sorted(extra)}")
    return errors


def compare_numerical_diagnostics(
    *,
    plan: Mapping[str, Any],
    stock_rows: Sequence[Mapping[str, Any]],
    resident_rows: Sequence[Mapping[str, Any]],
    stock_outputs: Sequence[Mapping[str, Any]],
    resident_outputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Localize the first observed stage difference without tolerance."""

    errors = []
    errors.extend(
        f"stock: {item}"
        for item in validate_numerical_records(
            stock_rows, plan, execution_mode="stock-style"
        )
    )
    for label, rows in (
        ("stock", stock_outputs),
        ("resident", resident_outputs),
    ):
        request_ids = [str(row.get("request_id", "")) for row in rows]
        if len(rows) != 100:
            errors.append(f"{label} diagnostic output does not preserve 100-request shape")
        if any(not request_id for request_id in request_ids) or len(set(request_ids)) != len(
            request_ids
        ):
            errors.append(f"{label} diagnostic output request IDs are invalid")
    errors.extend(
        f"resident: {item}"
        for item in validate_numerical_records(
            resident_rows, plan, execution_mode="resident-target"
        )
    )
    stock_by_key = _records_by_key(stock_rows)
    resident_by_key = _records_by_key(resident_rows)
    stock_output_by_id = _outputs_by_id(stock_outputs)
    resident_output_by_id = _outputs_by_id(resident_outputs)
    comparisons = []
    for item in plan.get("requests", ()):
        if not isinstance(item, Mapping):
            continue
        request_id = str(item["request_id"])
        position = int(item["output_position"])
        key = (request_id, position)
        stock = stock_by_key.get(key)
        resident = resident_by_key.get(key)
        if stock is None or resident is None:
            continue
        stock_token = _output_token(stock_output_by_id.get(request_id), position)
        resident_token = _output_token(resident_output_by_id.get(request_id), position)
        expected_stock = int(item["stock_selected_token_id"])
        expected_resident = int(item["resident_selected_token_id"])
        placeholder_fields = (
            "logical_committed_prefix_token_ids",
            "logical_committed_prefix_sha256",
            "num_computed_tokens",
            "target_input_token_position",
            "target_input_token_id",
            "previous_committed_token_id",
        )
        placeholder_equal = all(
            stock.get(name) == resident.get(name) for name in placeholder_fields
        )
        stock_generated = _generated_tokens(stock_output_by_id.get(request_id))
        resident_generated = _generated_tokens(
            resident_output_by_id.get(request_id)
        )
        semantic_equal = (
            stock_generated is not None
            and resident_generated is not None
            and stock_generated[:position] == resident_generated[:position]
            and len(stock_generated) > position
            and len(resident_generated) > position
        )
        ownership_equal = _logical_ownership(stock) == _logical_ownership(resident)
        stage_equal = {
            name: _stage_sha(stock, name) == _stage_sha(resident, name)
            for name in REQUIRED_STAGES
        }
        kv_equal = (
            stock.get("kv_cache_before_forward", {}).get("aggregate_raw_sha256")
            == resident.get("kv_cache_before_forward", {}).get(
                "aggregate_raw_sha256"
            )
        )
        candidate_equal = _candidate_logits(stock) == _candidate_logits(resident)
        raw_argmax_matches_output = {
            "stock": stock.get("raw_argmax_token_id") == stock_token,
            "resident": resident.get("raw_argmax_token_id") == resident_token,
        }
        first_stage = _first_different_stage(
            semantic_equal=semantic_equal,
            ownership_equal=ownership_equal,
            kv_equal=kv_equal,
            stage_equal=stage_equal,
            candidate_logits_equal=candidate_equal,
            sampler_equal=all(raw_argmax_matches_output.values()),
        )
        row_errors = []
        if stock_token != expected_stock:
            row_errors.append("stock diagnostic token differs from the frozen divergence")
        if resident_token != expected_resident:
            row_errors.append("resident diagnostic token differs from the frozen divergence")
        if stock_token == resident_token:
            row_errors.append("diagnostic pair did not reproduce the exact divergence")
        errors.extend(f"{request_id}@{position}: {message}" for message in row_errors)
        comparisons.append(
            {
                "request_id": request_id,
                "output_position": position,
                "stock_output_token_id": stock_token,
                "resident_output_token_id": resident_token,
                "semantic_input_equal": semantic_equal,
                "semantic_prefix_authority": "paired-final-run-output",
                "async_cpu_placeholder_view_equal": placeholder_equal,
                "logical_kv_ownership_equal": ownership_equal,
                "kv_raw_bytes_equal": kv_equal,
                "tensor_stage_raw_bytes_equal": stage_equal,
                "competing_raw_logits_equal": candidate_equal,
                "raw_argmax_matches_output": raw_argmax_matches_output,
                "first_observed_difference": first_stage,
                "stock_execution_shape": stock.get("execution_shape"),
                "resident_execution_shape": resident.get("execution_shape"),
                "stock_competing_logits": _candidate_logits(stock),
                "resident_competing_logits": _candidate_logits(resident),
                "errors": row_errors,
            }
        )
    if len(comparisons) != 4:
        errors.append("numerical comparison did not resolve all four planned requests")
    return {
        "schema_version": COMPARISON_SCHEMA,
        "diagnostic_only": True,
        "valid": not errors,
        "errors": errors,
        "request_count": len(comparisons),
        "comparisons": comparisons,
        "tolerant_correctness_policy": False,
        "tie_equivalent_tokens_accepted": False,
        "replaces_stock_reference": False,
        "correctness_decision": "fail-closed-pending-human-classification",
    }


def comparison_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Gate3 numerical localization",
        "",
        "Diagnostic-only evidence; it does not replace the immutable stock reference.",
        "",
        "| Request | Position | First observed difference | KV bytes equal | "
        "Raw logits equal | Sampler follows logits |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for row in report.get("comparisons", ()):
        sampler = row.get("raw_argmax_matches_output", {})
        lines.append(
            "| {request_id} | {output_position} | {first_observed_difference} | "
            "{kv_raw_bytes_equal} | {competing_raw_logits_equal} | {sampler} |".format(
                **row,
                sampler=bool(sampler) and all(sampler.values()),
            )
        )
    lines.extend(
        [
            "",
            "Exact token equality remains mandatory. No tolerance or "
            "tie-equivalence policy is enabled.",
            "",
        ]
    )
    return "\n".join(lines)


def compare_per_token_kv_diagnostics(
    *,
    plan: Mapping[str, Any],
    stock_rows: Sequence[Mapping[str, Any]],
    resident_rows: Sequence[Mapping[str, Any]],
    stock_outputs: Sequence[Mapping[str, Any]],
    resident_outputs: Sequence[Mapping[str, Any]],
    immutable_reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare exact logical-token K/V bytes under final-output authority."""

    errors = []
    if plan.get("schema_version") != PER_TOKEN_PLAN_SCHEMA:
        errors.append("per-token KV comparison requires the immutable per-token plan")
    errors.extend(
        f"stock: {item}"
        for item in validate_numerical_records(
            stock_rows, plan, execution_mode="stock-style"
        )
    )
    errors.extend(
        f"resident: {item}"
        for item in validate_numerical_records(
            resident_rows, plan, execution_mode="resident-target"
        )
    )
    stock_by_id = _validated_output_map(stock_outputs, "stock", errors)
    resident_by_id = _validated_output_map(resident_outputs, "resident", errors)
    reference_by_id = _validated_reference_map(immutable_reference, plan, errors)
    expected_ids = set(reference_by_id)
    if set(stock_by_id) != expected_ids:
        errors.append("stock final outputs differ from immutable reference request IDs")
    if set(resident_by_id) != expected_ids:
        errors.append("resident final outputs differ from immutable reference request IDs")

    stock_divergences = _output_divergences(stock_by_id, reference_by_id)
    if stock_divergences:
        errors.append(
            "stock final outputs differ from immutable reference: "
            + repr(sorted(stock_divergences.items()))
        )
    resident_divergences = _output_divergences(resident_by_id, reference_by_id)
    expected_divergences = {
        str(row["request_id"]): int(row["output_position"])
        for row in plan.get("requests", ())
        if isinstance(row, Mapping)
    }
    if resident_divergences != expected_divergences:
        errors.append(
            "resident final outputs do not reproduce exactly the four divergences: "
            f"observed={sorted(resident_divergences.items())}"
        )

    stock_by_key = _records_by_key(stock_rows)
    resident_by_key = _records_by_key(resident_rows)
    comparisons = []
    observed_phases = []
    for item in plan.get("requests", ()):
        if not isinstance(item, Mapping):
            continue
        request_id = str(item["request_id"])
        output_position = int(item["output_position"])
        key = (request_id, output_position)
        stock_record = stock_by_key.get(key)
        resident_record = resident_by_key.get(key)
        if stock_record is None or resident_record is None:
            continue
        row_errors = []
        reference_tokens = _generated_tokens(reference_by_id.get(request_id))
        stock_tokens = _generated_tokens(stock_by_id.get(request_id))
        resident_tokens = _generated_tokens(resident_by_id.get(request_id))
        reference_prefix = (
            reference_tokens[:output_position]
            if reference_tokens is not None
            else None
        )
        stock_prefix = (
            stock_tokens[:output_position] if stock_tokens is not None else None
        )
        resident_prefix = (
            resident_tokens[:output_position]
            if resident_tokens is not None
            else None
        )
        stock_prefix_exact = (
            reference_prefix is not None
            and stock_prefix == reference_prefix
            and stock_tokens is not None
            and len(stock_tokens) > output_position
        )
        resident_prefix_exact = (
            reference_prefix is not None
            and resident_prefix == reference_prefix
            and resident_tokens is not None
            and len(resident_tokens) > output_position
        )
        paired_prefix_exact = (
            stock_prefix is not None
            and resident_prefix is not None
            and stock_prefix == resident_prefix
            and len(stock_prefix) == output_position
        )
        if not stock_prefix_exact:
            row_errors.append("stock actual pre-divergence prefix differs from reference")
        if not resident_prefix_exact:
            row_errors.append("resident actual pre-divergence prefix differs from reference")
        if not paired_prefix_exact:
            row_errors.append("stock and resident actual pre-divergence prefixes differ")
        if (
            _output_token(reference_by_id.get(request_id), output_position)
            != item["stock_selected_token_id"]
            or _output_token(stock_by_id.get(request_id), output_position)
            != item["stock_selected_token_id"]
            or _output_token(resident_by_id.get(request_id), output_position)
            != item["resident_selected_token_id"]
        ):
            row_errors.append("the exact planned stock/resident token pair was not reproduced")
        expected_computed = int(item["prompt_length"]) + output_position - 1
        if (
            stock_record.get("num_computed_tokens") != expected_computed
            or resident_record.get("num_computed_tokens") != expected_computed
            or stock_record.get("target_input_token_position") != expected_computed
            or resident_record.get("target_input_token_position") != expected_computed
        ):
            row_errors.append("materialized KV boundary differs from the planned position")
        if _logical_ownership(stock_record) != _logical_ownership(resident_record):
            row_errors.append("logical KV ownership differs")

        rank_comparisons = []
        stock_capture = stock_record.get("per_logical_token_kv", {})
        resident_capture = resident_record.get("per_logical_token_kv", {})
        stock_ranks = _rank_capture_map(stock_capture)
        resident_ranks = _rank_capture_map(resident_capture)
        if set(stock_ranks) != {0, 1} or set(resident_ranks) != {0, 1}:
            row_errors.append("per-token KV TP rank evidence is incomplete")
        for tp_rank in (0, 1):
            stock_rank = stock_ranks.get(tp_rank)
            resident_rank = resident_ranks.get(tp_rank)
            if stock_rank is None or resident_rank is None:
                continue
            control = _compare_selected_token_layer(
                stock_rank,
                resident_rank,
                layer_name=str(item["control_layer_name"]),
                layer_index=int(item["control_layer_index"]),
                layer_role="control",
                prompt_length=int(item["prompt_length"]),
            )
            first = _compare_selected_token_layer(
                stock_rank,
                resident_rank,
                layer_name=str(item["first_different_layer_name"]),
                layer_index=int(item["first_different_layer_index"]),
                layer_role="first-different",
                prompt_length=int(item["prompt_length"]),
            )
            if not control["identity_valid"] or not first["identity_valid"]:
                row_errors.append(f"TP{tp_rank} selected layer identity differs")
            if not control["all_k_token_hashes_equal"] or not control[
                "all_v_token_hashes_equal"
            ]:
                row_errors.append(f"TP{tp_rank} control layer is not bitwise exact")
            if control["aggregate_raw_bytes_equal"] is not True:
                row_errors.append(f"TP{tp_rank} control aggregate layer differs")
            if first["aggregate_raw_bytes_equal"] is not False:
                row_errors.append(
                    f"TP{tp_rank} established first-different aggregate did not differ"
                )
            if first["first_differing_logical_position"] is None:
                row_errors.append(
                    f"TP{tp_rank} aggregate differs but no per-token K/V difference exists"
                )
            else:
                observed_phases.append(str(first["phase"]))
            rank_comparisons.append(
                {
                    "tp_rank": tp_rank,
                    "prompt_length": int(item["prompt_length"]),
                    "num_computed_tokens": expected_computed,
                    "control_layer": control,
                    "first_different_layer": first,
                }
            )
        errors.extend(
            f"{request_id}@{output_position}: {message}" for message in row_errors
        )
        comparisons.append(
            {
                "request_id": request_id,
                "divergent_output_position": output_position,
                "prompt_length": int(item["prompt_length"]),
                "num_computed_tokens": expected_computed,
                "target_input_position": expected_computed,
                "previous_generated_output_position": output_position - 1,
                "stock_previous_actual_token_id": _output_token(
                    stock_by_id.get(request_id), output_position - 1
                ),
                "resident_previous_actual_token_id": _output_token(
                    resident_by_id.get(request_id), output_position - 1
                ),
                "immutable_previous_token_id": _output_token(
                    reference_by_id.get(request_id), output_position - 1
                ),
                "stock_selected_token_id": int(item["stock_selected_token_id"]),
                "resident_selected_token_id": int(
                    item["resident_selected_token_id"]
                ),
                "semantic_prefix_authority": SEMANTIC_PREFIX_AUTHORITY,
                "stock_predivergence_prefix_exact": stock_prefix_exact,
                "resident_predivergence_prefix_exact": resident_prefix_exact,
                "paired_semantic_prefix_exact": paired_prefix_exact,
                "async_cpu_placeholder_view_is_authoritative": False,
                "tp_ranks": rank_comparisons,
                "errors": row_errors,
            }
        )
    if len(comparisons) != 4:
        errors.append("per-token KV comparison did not resolve all four requests")
    classification = _per_token_classification(observed_phases, errors)
    return {
        "schema_version": PER_TOKEN_COMPARISON_SCHEMA,
        "diagnostic_only": True,
        "valid": not errors,
        "errors": errors,
        "request_count": len(comparisons),
        "semantic_prefix_authority": SEMANTIC_PREFIX_AUTHORITY,
        "comparisons": comparisons,
        "classification": classification,
        "gate3_closed": False,
        "phase4b2_blocked": True,
        "tolerant_correctness_policy": False,
        "tie_equivalent_tokens_accepted": False,
        "replaces_stock_reference": False,
        "correctness_decision": "fail-closed-pending-human-classification",
    }


def per_token_comparison_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Gate3 per-logical-token K/V localization",
        "",
        "Diagnostic-only exact byte comparison; Gate3 remains not closed.",
        "",
        f"Classification: **{report.get('classification', 'FAIL-CLOSED')}**",
        "",
        f"Semantic prefix authority: `{report.get('semantic_prefix_authority')}`",
        "",
        "| Request | Output pos | TP | Control layer K/V exact | First layer | "
        "First logical difference | Phase | K differences | V differences |",
        "| --- | ---: | ---: | --- | --- | ---: | --- | ---: | ---: |",
    ]
    for request in report.get("comparisons", ()):
        for rank in request.get("tp_ranks", ()):
            control = rank["control_layer"]
            first = rank["first_different_layer"]
            control_exact = (
                control["all_k_token_hashes_equal"]
                and control["all_v_token_hashes_equal"]
            )
            lines.append(
                "| {request} | {position} | {rank} | {control_name}: {control} | "
                "{first_name} | {first_position} | {phase} | {k_count} | "
                "{v_count} |".format(
                    request=request["request_id"],
                    position=request["divergent_output_position"],
                    rank=rank["tp_rank"],
                    control_name=control["layer_name"],
                    control=control_exact,
                    first_name=first["layer_name"],
                    first_position=first["first_differing_logical_position"],
                    phase=first["phase"],
                    k_count=first["differing_k_logical_position_count"],
                    v_count=first["differing_v_logical_position_count"],
                )
            )
    lines.extend(
        [
            "",
            "The JSON report contains separate first/last K and V positions plus a ±2 "
            "checksum/equality window around each first difference.",
            "",
            "No tolerance or tie-equivalence rule is enabled. Do not run Serial, Dual or "
            "performance work from this diagnostic.",
            "",
        ]
    )
    return "\n".join(lines)


def prepare_target_numerical_diagnostic(
    runner: Any,
    *,
    scheduler_output: Any,
    logits_indices: Any,
    positions: Any,
    num_scheduled_tokens: Sequence[int],
    slot_mappings_by_group: Any,
) -> None:
    """GPU-only pre-forward hook installed by the pinned observational patch."""

    configured = [os.environ.get(name) for name in (PLAN_ENV, OUTPUT_ENV, MODE_ENV)]
    if not any(configured):
        return
    if not all(configured):
        raise RuntimeError("Gate3 numerical worker environment is partially configured")
    from vllm.distributed.parallel_state import get_tp_group

    tp_group = get_tp_group()
    tp_rank = int(tp_group.rank_in_group)
    tp_world_size = int(tp_group.world_size)
    state = _numerical_state(runner)
    state["active"] = {}
    scheduled_spec = scheduler_output.scheduled_spec_decode_tokens
    flat_cursor = 0
    sample_cursor = 0
    positions_cpu = positions.detach().cpu().tolist()
    logits_indices_cpu = logits_indices.detach().cpu().tolist()
    query_lengths = [int(item) for item in num_scheduled_tokens]
    total_scheduled_tokens = sum(query_lengths)
    decode_rows = sum(length == 1 for length in query_lengths)
    prefill_rows = sum(length > 1 for length in query_lengths)
    if decode_rows and prefill_rows:
        batch_row_kind = "mixed-prefill-decode"
    elif decode_rows:
        batch_row_kind = "decode-only"
    else:
        batch_row_kind = "prefill-only"

    for index, internal_id in enumerate(runner.input_batch.req_ids):
        query_length = int(num_scheduled_tokens[index])
        proposal = tuple(int(item) for item in scheduled_spec.get(internal_id, ()))
        count = int(runner.input_batch.num_tokens_no_spec[index])
        tokens = tuple(
            int(item)
            for item in runner.input_batch.token_ids_cpu[index, :count].tolist()
        )
        match = _match_planned_prefix(tokens, state["plan_by_id"])
        if match is not None:
            plan_row, prefix = match
            output_position = len(prefix) - len(plan_row["prompt_token_ids"])
            key = (plan_row["request_id"], output_position)
            if output_position == plan_row["output_position"] and key not in state["captured"]:
                if proposal:
                    raise RuntimeError("Gate3 Target diagnostic forbids speculative proposals")
                if query_length != 1:
                    raise RuntimeError(
                        "Gate3 divergence diagnostic requires one decode input position"
                    )
                sample_index = sample_cursor
                if sample_index >= len(logits_indices_cpu):
                    raise RuntimeError("Gate3 numerical logits index is unavailable")
                flat_index = int(logits_indices_cpu[sample_index])
                computed = int(runner.input_batch.num_computed_tokens_cpu[index])
                if computed != len(prefix) - 1:
                    raise RuntimeError(
                        "Gate3 numerical Target KV does not end before the pending token"
                    )
                active = {
                    "request_id": plan_row["request_id"],
                    "internal_request_id": str(internal_id),
                    "output_position": output_position,
                    "plan": plan_row,
                    "prefix": list(prefix),
                    "num_computed_tokens": computed,
                    "query_length": query_length,
                    "flat_index": flat_index,
                    "sample_index": sample_index,
                    "target_input_token_position": int(positions_cpu[flat_cursor]),
                    "target_input_token_id": int(prefix[-1]),
                    "request_index": index,
                    "flat_query_start": flat_cursor,
                    "tensor_stages": {},
                    "tp_rank": tp_rank,
                    "tp_world_size": tp_world_size,
                    "total_scheduled_token_count": total_scheduled_tokens,
                    "decode_row_count": decode_rows,
                    "prefill_row_count": prefill_rows,
                    "batch_row_kind": batch_row_kind,
                }
                state["active"][key] = active
        flat_cursor += query_length
        sample_cursor += len(proposal) + 1
    if not state["active"]:
        return
    for active in state["active"].values():
        ownership = _generic_kv_ownership(
            runner,
            request_index=active["request_index"],
            num_computed_tokens=active["num_computed_tokens"],
            flat_query_start=active["flat_query_start"],
            query_length=active["query_length"],
            slot_mappings_by_group=slot_mappings_by_group,
        )
        active["kv_ownership"] = ownership
        active["kv_cache_before_forward"] = _kv_cache_summary(
            runner, active
        )
    local_rows = [
        {
            "key": list(key),
            "tp_rank": tp_rank,
            "kv_cache_before_forward": active["kv_cache_before_forward"],
            "kv_ownership": active["kv_ownership"],
        }
        for key, active in sorted(state["active"].items())
    ]
    gathered = _gather_tp_objects(local_rows, tp_group)
    expected_keys = {tuple(row["key"]) for row in local_rows}
    for rank_rows in gathered:
        if {tuple(row["key"]) for row in rank_rows} != expected_keys:
            raise RuntimeError(
                "Gate3 numerical checkpoints differ across TP ranks"
            )
    for key, active in state["active"].items():
        rank_rows = [
            next(row for row in rows if tuple(row["key"]) == key)
            for rows in gathered
        ]
        active["tp_rank_kv_cache_before_forward"] = rank_rows
        active["kv_cache_before_forward"] = _combine_rank_kv(rank_rows)
        if "control_layer_index" in active["plan"]:
            active["per_logical_token_kv"] = _combine_rank_per_token_kv(
                rank_rows,
                plan_row=active["plan"],
                num_computed_tokens=active["num_computed_tokens"],
            )


def finalize_target_numerical_diagnostic(
    runner: Any,
    *,
    logits: Any,
    target_forward_start_ns: int,
    target_forward_end_ns: int,
) -> None:
    """GPU-only post-logits capture for the four planned checkpoints."""

    state = getattr(runner, "_specrhythm_numerical_state", None)
    if not isinstance(state, Mapping) or not state.get("active"):
        return
    from vllm.distributed.parallel_state import get_tp_group

    if logits is None:
        raise RuntimeError("Gate3 numerical diagnostics received no Target logits")
    tp_group = get_tp_group()
    tp_rank = int(tp_group.rank_in_group)
    logits_cpu = logits.detach().float().cpu()
    model = _unwrap_model(runner.model)
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise RuntimeError("Gate3 numerical diagnostic cannot resolve lm_head")
    local_rows = []
    for key, active in sorted(state["active"].items()):
        sample_index = active["sample_index"]
        missing = [name for name in REQUIRED_STAGES if name not in active["tensor_stages"]]
        if missing:
            raise RuntimeError(
                "Gate3 numerical model hooks missed stages: " + ", ".join(missing)
            )
        plan_row = active["plan"]
        token_ids = (
            int(plan_row["stock_selected_token_id"]),
            int(plan_row["resident_selected_token_id"]),
        )
        row_logits = logits_cpu[sample_index]
        competing = [
            {"token_id": token_id, "raw_logit": float(row_logits[token_id].item())}
            for token_id in token_ids
        ]
        local_rows.append(
            {
                "key": list(key),
                "tp_rank": tp_rank,
                "tensor_stages": active["tensor_stages"],
                "model_module_paths": state["model_paths"],
                "lm_head_input_source_shape": active["lm_head_input_source_shape"],
                "competing_tokens": competing,
                "raw_argmax_token_id": int(row_logits.argmax().item()),
            }
        )
    gathered = _gather_tp_objects(local_rows, tp_group)
    if tp_rank != 0:
        state["captured"].update(state["active"])
        state["active"] = {}
        return
    log = CheckpointJsonl(Path(os.environ[OUTPUT_ENV]).resolve())
    expected_keys = {tuple(row["key"]) for row in local_rows}
    for rank_rows in gathered:
        if {tuple(row["key"]) for row in rank_rows} != expected_keys:
            raise RuntimeError("Gate3 tensor checkpoints differ across TP ranks")
    for key, active in sorted(state["active"].items()):
        plan_row = active["plan"]
        sample_index = active["sample_index"]
        row_logits = logits_cpu[sample_index]
        competing = next(
            row["competing_tokens"] for row in local_rows if tuple(row["key"]) == key
        )
        top_count = min(10, int(row_logits.shape[-1]))
        top_values, top_ids = row_logits.topk(top_count)
        tp_rank_evidence = [
            next(row for row in rows if tuple(row["key"]) == key)
            for rows in gathered
        ]
        if [row["tp_rank"] for row in tp_rank_evidence] != list(
            range(int(tp_group.world_size))
        ):
            raise RuntimeError("Gate3 numerical TP-rank evidence is incomplete")
        per_token_plan = state["plan"]["schema_version"] == PER_TOKEN_PLAN_SCHEMA
        record = {
            "schema_version": (
                PER_TOKEN_RECORD_SCHEMA if per_token_plan else RECORD_SCHEMA
            ),
            "diagnostic_only": True,
            "execution_mode": state["mode"],
            "plan_sha256": state["plan"]["plan_sha256"],
            "workload_sha256": state["plan"]["workload_sha256"],
            "tp_world_size": int(tp_group.world_size),
            "request_id": active["request_id"],
            "internal_request_id": active["internal_request_id"],
            "output_position": active["output_position"],
            "position_indexing": "zero-based-generated-token-index",
            "num_computed_tokens": active["num_computed_tokens"],
            "prompt_length": len(plan_row["prompt_token_ids"]),
            "target_input_token_position": active["target_input_token_position"],
            "kv_cache_before_forward": active["kv_cache_before_forward"],
            "tp_rank_kv_cache_before_forward": active[
                "tp_rank_kv_cache_before_forward"
            ],
            "kv_position_mapping": active["kv_ownership"],
            "tensor_stages": active["tensor_stages"],
            "tp_rank_evidence": tp_rank_evidence,
            "model_module_paths": state["model_paths"],
            "raw_pre_softmax_logits": {
                "dtype": str(logits.dtype),
                "shape": list(logits.shape),
                "competing_tokens": competing,
                "top_candidates": [
                    {"token_id": int(token_id), "raw_logit": float(value)}
                    for token_id, value in zip(top_ids.tolist(), top_values.tolist())
                ],
            },
            "raw_argmax_token_id": int(row_logits.argmax().item()),
            "lm_head": {
                "module_class": type(lm_head).__name__,
                "quant_method_class": type(getattr(lm_head, "quant_method", None)).__name__,
                "weight_dtype": str(getattr(getattr(lm_head, "weight", None), "dtype", None)),
                "input_dtype": active["tensor_stages"]["lm_head_input"]["dtype"],
                "output_dtype": str(logits.dtype),
                "path": "LogitsProcessor._get_logits -> lm_head.quant_method.apply -> TP gather",
                "batch_invariant_kernel": "linear_batch_invariant/matmul_persistent",
            },
            "execution_shape": {
                "active_request_count": len(runner.input_batch.req_ids),
                "number_of_active_requests": len(runner.input_batch.req_ids),
                "sampled_logits_rows": int(logits.shape[0]),
                "lm_head_m": int(active["lm_head_input_source_shape"][0]),
                "total_scheduled_token_count": active[
                    "total_scheduled_token_count"
                ],
                "planned_request_query_length": active["query_length"],
                "decode_row_count": active["decode_row_count"],
                "prefill_row_count": active["prefill_row_count"],
                "batch_row_kind": active["batch_row_kind"],
                "hidden_size": int(active["lm_head_input_source_shape"][-1]),
                "vocab_size": int(logits.shape[-1]),
            },
            "target_forward_start_ns": int(target_forward_start_ns),
            "target_forward_end_ns": int(target_forward_end_ns),
            "target_only_artifact": True,
            "visible_to_draft": False,
        }
        if per_token_plan:
            record["async_cpu_placeholder_view"] = {
                "source": "InputBatch.token_ids_cpu",
                "token_ids": active["prefix"],
                "token_ids_sha256": token_sha256(active["prefix"]),
                "contains_negative_placeholder": any(
                    token_id < 0 for token_id in active["prefix"]
                ),
                "semantic_prefix_authority": False,
            }
            record["per_logical_token_kv"] = active["per_logical_token_kv"]
            if state["mode"] == "matched-stock-async-off":
                scheduler = runner.vllm_config.scheduler_config
                scheduler_class = scheduler.get_scheduler_cls()
                speculative = runner.vllm_config.speculative_config
                visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                try:
                    physical_gpu_ids = [
                        int(item.strip()) for item in visible.split(",") if item.strip()
                    ]
                except ValueError as error:
                    raise RuntimeError(
                        "matched-bootstrap Target physical GPU mapping is invalid"
                    ) from error
                record["matched_bootstrap_control"] = {
                    "async_scheduling_requested": False,
                    "async_scheduling_effective": bool(
                        scheduler.async_scheduling
                    ),
                    "speculative_config": None if speculative is None else "non-null",
                    "scheduler_class": (
                        f"{scheduler_class.__module__}."
                        f"{scheduler_class.__qualname__}"
                    ),
                    "custom_class_proposer_absent": speculative is None,
                    "resident_setup_scheduler_absent": (
                        scheduler_class.__name__ != "ResidentSetupScheduler"
                    ),
                    "target_tp_world_size": int(tp_group.world_size),
                    "physical_target_gpu_ids": physical_gpu_ids,
                }
        else:
            record.update(
                {
                    "logical_committed_prefix_token_ids": active["prefix"],
                    "logical_committed_prefix_sha256": token_sha256(
                        active["prefix"]
                    ),
                    "target_input_token_id": active["target_input_token_id"],
                    "previous_committed_token_id": active["prefix"][-1],
                }
            )
        log.append(record)
        state["captured"].add(key)
    state["active"] = {}


def _numerical_state(runner: Any) -> dict[str, Any]:
    existing = getattr(runner, "_specrhythm_numerical_state", None)
    if isinstance(existing, dict):
        return existing
    workload_path = os.environ.get("SR_PHASE4_WORKLOAD")
    mode = os.environ.get(MODE_ENV)
    plan_path = os.environ.get(PLAN_ENV)
    if not workload_path or not plan_path or mode not in ALLOWED_MODES:
        raise RuntimeError("Gate3 numerical worker environment is incomplete")
    plan = load_numerical_plan(Path(plan_path), Path(workload_path))
    state = {
        "plan": plan,
        "plan_by_id": {row["request_id"]: row for row in plan["requests"]},
        "mode": mode,
        "captured": set(),
        "active": {},
        "hook_handles": [],
    }
    runner._specrhythm_numerical_state = state
    _install_model_hooks(runner, state)
    return state


def _install_model_hooks(runner: Any, state: dict[str, Any]) -> None:
    model = _unwrap_model(runner.model)
    named = dict(model.named_modules())
    embed_name, embed = _unique_suffix_module(named, "model.embed_tokens")
    norm_name, norm = _unique_suffix_module(named, "model.norm")
    logits_name, logits_processor = _unique_suffix_module(named, "logits_processor")
    state["model_paths"] = {
        "model": type(model).__name__,
        "decoder_input": embed_name,
        "final_norm": norm_name,
        "lm_head": _unique_suffix_module(named, "lm_head")[0],
        "logits_processor": logits_name,
    }

    def embed_hook(_module: Any, _args: Any, output: Any) -> None:
        _capture_active_stage(state, "decoder_input_hidden_state", output)

    def norm_pre_hook(_module: Any, args: Any) -> None:
        if not args:
            raise RuntimeError("Gate3 final norm hook received no hidden state")
        _capture_active_stage(state, "last_layer_branch_output", args[0])
        state["norm_residual"] = args[1] if len(args) > 1 else None

    def norm_post_hook(_module: Any, _args: Any, output: Any) -> None:
        if isinstance(output, tuple):
            normalized, norm_input = output
        else:
            normalized = output
            residual = state.pop("norm_residual", None)
            norm_input = _args[0] if residual is None else _args[0] + residual
        _capture_active_stage(state, "final_norm_input", norm_input)
        _capture_active_stage(state, "final_normalized_hidden_state", normalized)

    def logits_pre_hook(_module: Any, args: Any) -> None:
        if len(args) < 2:
            raise RuntimeError("Gate3 logits processor hook received no hidden state")
        hidden_states = args[1]
        for active in state.get("active", {}).values():
            stages = active["tensor_stages"]
            if "lm_head_input" in stages:
                raise RuntimeError("Gate3 lm_head input captured twice")
            stages["lm_head_input"] = _tensor_summary(
                hidden_states, active["sample_index"]
            )
            active["lm_head_input_source_shape"] = list(hidden_states.shape)

    state["hook_handles"].extend(
        [
            embed.register_forward_hook(embed_hook),
            norm.register_forward_pre_hook(norm_pre_hook),
            norm.register_forward_hook(norm_post_hook),
            logits_processor.register_forward_pre_hook(logits_pre_hook),
        ]
    )


def _capture_active_stage(state: Mapping[str, Any], name: str, tensor: Any) -> None:
    for active in state.get("active", {}).values():
        stages = active["tensor_stages"]
        if name in stages:
            raise RuntimeError(f"Gate3 numerical stage captured twice: {name}")
        stages[name] = _tensor_summary(tensor, active["flat_index"])


def _tensor_summary(tensor: Any, row_index: int) -> dict[str, Any]:
    import torch

    value = tensor[row_index].detach().contiguous()
    numeric = value.float().cpu().reshape(-1)
    if not bool(torch.isfinite(numeric).all().item()):
        raise RuntimeError("Gate3 numerical tensor contains a non-finite value")
    raw = value.view(torch.uint8).cpu().numpy().tobytes()
    fp32 = numeric.contiguous().view(torch.uint8).numpy().tobytes()
    count = int(numeric.numel())
    coordinates = sorted(
        {0, min(1, count - 1), min(2, count - 1), count // 4, count // 2, count - 1}
    )
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "fp32_sha256": hashlib.sha256(fp32).hexdigest(),
        "min": float(numeric.min().item()),
        "max": float(numeric.max().item()),
        "norm": float(torch.linalg.vector_norm(numeric).item()),
        "selected_coordinates": [
            {"flat_index": index, "fp32_value": float(numeric[index].item())}
            for index in coordinates
        ],
    }


def _generic_kv_ownership(
    runner: Any,
    *,
    request_index: int,
    num_computed_tokens: int,
    flat_query_start: int,
    query_length: int,
    slot_mappings_by_group: Any,
) -> dict[str, Any]:
    """Read generic per-group ownership without speculative metadata."""

    multi_table = getattr(getattr(runner, "input_batch", None), "block_table", None)
    block_tables = getattr(multi_table, "block_tables", None)
    kv_config = getattr(runner, "kv_cache_config", None)
    kv_groups = getattr(kv_config, "kv_cache_groups", None)
    if not isinstance(block_tables, (list, tuple)) or not block_tables:
        raise RuntimeError("Gate3 numerical generic MultiGroupBlockTable is missing")
    if not isinstance(kv_groups, (list, tuple)) or len(kv_groups) != len(block_tables):
        raise RuntimeError("Gate3 numerical KV group configuration differs from block tables")
    if not isinstance(slot_mappings_by_group, Mapping) or set(
        slot_mappings_by_group
    ) != set(range(len(block_tables))):
        raise RuntimeError("Gate3 numerical generic slot mappings are incomplete")
    groups = []
    for group_id, (block_table, kv_group) in enumerate(
        zip(block_tables, kv_groups)
    ):
        block_size = int(getattr(block_table, "block_size", 0))
        if block_size <= 0:
            raise RuntimeError("Gate3 numerical generic block size is invalid")
        row_counts = getattr(block_table, "num_blocks_per_row", None)
        if row_counts is None or request_index >= len(row_counts):
            raise RuntimeError("Gate3 numerical block-table row count is unavailable")
        committed_blocks = int(row_counts[request_index])
        used_blocks = (
            math.ceil(num_computed_tokens / block_size)
            if num_computed_tokens
            else 0
        )
        if committed_blocks < used_blocks:
            raise RuntimeError("Gate3 numerical generic block-table row is truncated")
        get_numpy = getattr(block_table, "get_numpy_array", None)
        if not callable(get_numpy):
            raise RuntimeError("Gate3 numerical generic block-table CPU authority is missing")
        table = get_numpy()
        try:
            physical_blocks = [
                int(item)
                for item in table[request_index, :used_blocks].tolist()
            ]
        except (AttributeError, IndexError, TypeError) as error:
            raise RuntimeError(
                "Gate3 numerical generic block-table layout is unsupported"
            ) from error
        if len(set(physical_blocks)) != len(physical_blocks):
            raise RuntimeError(
                "Gate3 numerical logical blocks map ambiguously to physical storage"
            )
        mapping = slot_mappings_by_group[group_id]
        slots = _to_int_list(mapping, name="generic slot mapping")
        query_end = flat_query_start + query_length
        if query_end > len(slots):
            raise RuntimeError("Gate3 numerical current query slot mapping is truncated")
        current_slots = slots[flat_query_start:query_end]
        if len(current_slots) != query_length or any(slot < 0 for slot in current_slots):
            raise RuntimeError("Gate3 numerical current query slot mapping is invalid")
        layer_names = getattr(kv_group, "layer_names", None)
        if not isinstance(layer_names, (list, tuple)) or not layer_names or any(
            not isinstance(name, str) or not name for name in layer_names
        ):
            raise RuntimeError("Gate3 numerical KV group layer ownership is invalid")
        groups.append(
            {
                "kv_cache_group_id": group_id,
                "block_size": block_size,
                "num_blocks_per_row": committed_blocks,
                "logical_used_block_count": used_blocks,
                "physical_block_ids": physical_blocks,
                "current_query_slot_mapping": current_slots,
                "layer_names": list(layer_names),
            }
        )
    return {
        "authority": (
            "InputBatch.MultiGroupBlockTable + "
            "GPUModelRunner._get_slot_mappings.slot_mappings_by_group"
        ),
        "kv_cache_group_count": len(groups),
        "logical_positions": list(range(num_computed_tokens)),
        "groups": groups,
    }


def _to_int_list(value: Any, *, name: str) -> list[int]:
    current = value
    for operation in ("detach", "cpu"):
        method = getattr(current, operation, None)
        if callable(method):
            current = method()
    method = getattr(current, "tolist", None)
    if callable(method):
        current = method()
    if not isinstance(current, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in current
    ):
        raise RuntimeError(f"Gate3 numerical {name} is not an integer vector")
    return [int(item) for item in current]


def _kv_cache_summary(runner: Any, active: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    model = _unwrap_model(runner.model)
    computed = int(active["num_computed_tokens"])
    ownership = active.get("kv_ownership")
    if not isinstance(ownership, Mapping):
        raise RuntimeError("Gate3 numerical generic KV ownership is missing")
    groups = ownership.get("groups")
    if not isinstance(groups, list) or not groups:
        raise RuntimeError("Gate3 numerical generic KV groups are missing")
    group_by_layer = {}
    for group in groups:
        if not isinstance(group, Mapping):
            raise RuntimeError("Gate3 numerical generic KV group is malformed")
        for layer_name in group.get("layer_names", ()):
            if layer_name in group_by_layer:
                raise RuntimeError("Gate3 numerical KV layer belongs to multiple groups")
            group_by_layer[layer_name] = group
    layers = []
    per_token_layers = []
    plan_row = active.get("plan")
    selected_layers = _selected_per_token_layers(plan_row)
    aggregate = hashlib.sha256()
    observed_layers = set()
    for name, module in model.named_modules():
        cache = getattr(module, "kv_cache", None)
        if cache is None or not hasattr(cache, "shape") or int(cache.numel()) == 0:
            continue
        layer_name = str(getattr(module, "layer_name", name))
        group = group_by_layer.get(layer_name)
        if group is None:
            raise RuntimeError(
                f"Gate3 numerical KV layer has no generic group ownership: {layer_name}"
            )
        block_size = int(group["block_size"])
        blocks = list(group["physical_block_ids"])
        if len(cache.shape) < 3 or int(cache.shape[1]) != 2:
            raise RuntimeError("Gate3 numerical diagnostic supports FlashAttention KV layout only")
        if int(cache.shape[2]) != block_size:
            raise RuntimeError("Gate3 numerical KV cache block size differs from block table")
        if any(block >= int(cache.shape[0]) for block in blocks):
            raise RuntimeError("Gate3 numerical logical position maps outside KV storage")
        raw_hash = hashlib.sha256()
        numeric_min = None
        numeric_max = None
        squared_norm = 0.0
        remaining = computed
        for physical_block in blocks:
            take = min(block_size, remaining)
            if take <= 0:
                break
            value = cache[physical_block, :, :take].detach().contiguous()
            raw_hash.update(value.view(torch.uint8).cpu().numpy().tobytes())
            numeric = value.float()
            if not bool(torch.isfinite(numeric).all().item()):
                raise RuntimeError("Gate3 numerical KV cache contains a non-finite value")
            local_min = float(numeric.min().item())
            local_max = float(numeric.max().item())
            numeric_min = local_min if numeric_min is None else min(numeric_min, local_min)
            numeric_max = local_max if numeric_max is None else max(numeric_max, local_max)
            squared_norm += float((numeric * numeric).sum().item())
            remaining -= take
        if remaining != 0:
            raise RuntimeError("Gate3 numerical KV block table is truncated")
        digest = raw_hash.hexdigest()
        aggregate.update(layer_name.encode("utf-8"))
        aggregate.update(digest.encode("ascii"))
        observed_layers.add(layer_name)
        layers.append(
            {
                "layer": layer_name,
                "kv_cache_group_id": int(group["kv_cache_group_id"]),
                "dtype": str(cache.dtype),
                "cache_shape": list(cache.shape),
                "logical_token_count": computed,
                "raw_sha256": digest,
                "min": numeric_min,
                "max": numeric_max,
                "norm": math.sqrt(squared_norm),
            }
        )
        if layer_name in selected_layers:
            selection = selected_layers[layer_name]
            token_summary = _logical_token_kv_summary(
                cache,
                group=group,
                num_computed_tokens=computed,
                layer_name=layer_name,
                layer_index=selection["layer_index"],
                layer_role=selection["layer_role"],
            )
            token_summary["aggregate_raw_sha256"] = digest
            per_token_layers.append(token_summary)
    if not layers:
        raise RuntimeError("Gate3 numerical diagnostic found no Target KV-cache layers")
    if observed_layers != set(group_by_layer):
        raise RuntimeError("Gate3 numerical KV-cache layers differ from group ownership")
    if set(selected_layers) != {
        row["layer_name"] for row in per_token_layers
    }:
        raise RuntimeError("Gate3 numerical selected per-token KV layer is missing")
    return {
        "layout": "flash-attention-paged-kv[num_blocks,2,block_size,...]",
        "logical_token_count": computed,
        "kv_cache_group_count": len(groups),
        "layer_count": len(layers),
        "aggregate_raw_sha256": aggregate.hexdigest(),
        "layers": layers,
        "per_logical_token_kv_layers": per_token_layers,
    }


def _selected_per_token_layers(plan_row: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(plan_row, Mapping) or "control_layer_index" not in plan_row:
        return {}
    return {
        str(plan_row["control_layer_name"]): {
            "layer_index": int(plan_row["control_layer_index"]),
            "layer_role": "control",
        },
        str(plan_row["first_different_layer_name"]): {
            "layer_index": int(plan_row["first_different_layer_index"]),
            "layer_role": "first-different",
        },
    }


def _logical_kv_locations(
    physical_blocks: Sequence[int], block_size: int, num_tokens: int
) -> list[tuple[int, int, int]]:
    """Map logical tokens to authoritative paged-KV storage locations."""

    if not _positive_int(block_size) or not _nonnegative_int(num_tokens):
        raise RuntimeError("Gate3 per-token KV logical mapping dimensions are invalid")
    required_blocks = math.ceil(num_tokens / block_size) if num_tokens else 0
    if len(physical_blocks) != required_blocks or any(
        not _nonnegative_int(block) for block in physical_blocks
    ) or len(set(physical_blocks)) != len(physical_blocks):
        raise RuntimeError("Gate3 per-token KV block table is truncated or invalid")
    return [
        (
            logical_position,
            int(physical_blocks[logical_position // block_size]),
            logical_position % block_size,
        )
        for logical_position in range(num_tokens)
    ]


def _hash_token_kv_payloads(
    k_payloads: Sequence[bytes],
    v_payloads: Sequence[bytes],
) -> list[dict[str, Any]]:
    """Hash separate K/V byte planes after one bounded layer transfer."""

    if len(k_payloads) != len(v_payloads):
        raise RuntimeError("Gate3 per-token K/V payload counts differ")
    rows = []
    for logical_position, (k_payload, v_payload) in enumerate(
        zip(k_payloads, v_payloads)
    ):
        if not isinstance(k_payload, bytes) or not isinstance(v_payload, bytes):
            raise RuntimeError("Gate3 per-token K/V payload is not raw bytes")
        rows.append(
            {
                "logical_position": logical_position,
                "k_raw_sha256": hashlib.sha256(k_payload).hexdigest(),
                "v_raw_sha256": hashlib.sha256(v_payload).hexdigest(),
            }
        )
    return rows


def _token_digest_sequence_sha256(tokens: Sequence[Mapping[str, Any]]) -> str:
    """Commit to the ordered per-token K/V digests without raw tensor bytes."""

    digest = hashlib.sha256(b"specrhythm.logical-kv-token-digests.v1\0")
    for expected_position, token in enumerate(tokens):
        if (
            token.get("logical_position") != expected_position
            or not _sha256_text(token.get("k_raw_sha256"))
            or not _sha256_text(token.get("v_raw_sha256"))
        ):
            raise RuntimeError("Gate3 per-token K/V digest sequence is invalid")
        digest.update(str(expected_position).encode("ascii"))
        digest.update(b"\0K\0")
        digest.update(str(token["k_raw_sha256"]).encode("ascii"))
        digest.update(b"\0V\0")
        digest.update(str(token["v_raw_sha256"]).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def logical_kv_digest_binding(
    k_payloads: Sequence[bytes], v_payloads: Sequence[bytes]
) -> dict[str, Any]:
    """Bind token digests to the exact reconstructed logical tensor bytes.

    The raw digest uses the byte order produced by a contiguous tensor shaped
    ``[2, logical_tokens, ...]``.  It intentionally differs from the legacy
    block-by-block full-layer digest when a sequence spans multiple blocks.
    """

    tokens = _hash_token_kv_payloads(k_payloads, v_payloads)
    logical_raw = hashlib.sha256()
    for payload in k_payloads:
        logical_raw.update(payload)
    for payload in v_payloads:
        logical_raw.update(payload)
    return {
        "logical_reconstructed_raw_sha256": logical_raw.hexdigest(),
        "logical_reconstructed_raw_ordering": LOGICAL_RAW_ORDERING,
        "token_digest_sequence_sha256": _token_digest_sequence_sha256(tokens),
        "token_digest_sequence_ordering": TOKEN_DIGEST_ORDERING,
        "tokens": tokens,
    }


def _logical_token_kv_summary(
    cache: Any,
    *,
    group: Mapping[str, Any],
    num_computed_tokens: int,
    layer_name: str,
    layer_index: int,
    layer_role: str,
) -> dict[str, Any]:
    """Copy one selected logical layer once, then hash token K/V slices on CPU."""

    import torch

    block_size = int(group["block_size"])
    blocks = list(group["physical_block_ids"])
    locations = _logical_kv_locations(blocks, block_size, num_computed_tokens)
    if any(block >= int(cache.shape[0]) for _, block, _ in locations):
        raise RuntimeError("Gate3 per-token KV logical position is outside storage")
    pieces = []
    remaining = num_computed_tokens
    for physical_block in blocks:
        take = min(block_size, remaining)
        if take:
            pieces.append(cache[physical_block, :, :take])
        remaining -= take
    if remaining != 0 or not pieces:
        raise RuntimeError("Gate3 per-token KV reconstruction is incomplete")
    logical = torch.cat(pieces, dim=1).detach().contiguous().cpu()
    if list(logical.shape[:2]) != [2, num_computed_tokens]:
        raise RuntimeError("Gate3 per-token KV reconstructed shape is invalid")
    k_values = logical[0]
    v_values = logical[1]
    k_payloads = [
        k_values[position].contiguous().view(torch.uint8).numpy().tobytes()
        for position in range(num_computed_tokens)
    ]
    v_payloads = [
        v_values[position].contiguous().view(torch.uint8).numpy().tobytes()
        for position in range(num_computed_tokens)
    ]
    binding = logical_kv_digest_binding(k_payloads, v_payloads)
    return {
        "layer_name": layer_name,
        "layer_index": int(layer_index),
        "layer_role": layer_role,
        "kv_cache_group_id": int(group["kv_cache_group_id"]),
        "dtype": str(cache.dtype),
        "k_shape_per_logical_token": list(k_values.shape[1:]),
        "v_shape_per_logical_token": list(v_values.shape[1:]),
        "materialized_logical_position_start": 0,
        "materialized_logical_position_end_exclusive": num_computed_tokens,
        "gpu_to_cpu_transfer_count": 1,
        **binding,
    }


def _gather_tp_objects(value: Any, tp_group: Any) -> list[Any]:
    import torch.distributed as distributed

    gathered = [None] * int(tp_group.world_size)
    distributed.all_gather_object(gathered, value, group=tp_group.cpu_group)
    if any(item is None for item in gathered):
        raise RuntimeError("Gate3 numerical TP gather returned incomplete evidence")
    return gathered


def _combine_rank_kv(rank_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ranks = [int(row["tp_rank"]) for row in rank_rows]
    if ranks != list(range(len(rank_rows))):
        raise RuntimeError("Gate3 numerical KV rank order is incomplete")
    aggregate = hashlib.sha256()
    layer_counts = []
    logical_counts = []
    group_counts = []
    for row in rank_rows:
        summary = row.get("kv_cache_before_forward")
        if not isinstance(summary, Mapping) or not summary.get("aggregate_raw_sha256"):
            raise RuntimeError("Gate3 numerical TP rank has no KV checksum")
        aggregate.update(str(row["tp_rank"]).encode("ascii"))
        aggregate.update(str(summary["aggregate_raw_sha256"]).encode("ascii"))
        layer_counts.append(int(summary.get("layer_count", -1)))
        logical_counts.append(int(summary.get("logical_token_count", -1)))
        group_counts.append(int(summary.get("kv_cache_group_count", -1)))
    if (
        len(set(layer_counts)) != 1
        or len(set(logical_counts)) != 1
        or len(set(group_counts)) != 1
    ):
        raise RuntimeError("Gate3 numerical KV shapes differ across TP ranks")
    return {
        "layout": "tensor-parallel-rank-sharded-flash-attention-paged-kv",
        "tp_world_size": len(rank_rows),
        "logical_token_count": logical_counts[0],
        "kv_cache_group_count": group_counts[0],
        "layer_count_per_rank": layer_counts[0],
        "aggregate_raw_sha256": aggregate.hexdigest(),
        "rank_aggregate_raw_sha256": [
            {
                "tp_rank": int(row["tp_rank"]),
                "raw_sha256": row["kv_cache_before_forward"][
                    "aggregate_raw_sha256"
                ],
            }
            for row in rank_rows
        ],
    }


def _combine_rank_per_token_kv(
    rank_rows: Sequence[Mapping[str, Any]],
    *,
    plan_row: Mapping[str, Any],
    num_computed_tokens: int,
) -> dict[str, Any]:
    ranks = [int(row["tp_rank"]) for row in rank_rows]
    if ranks != list(range(len(rank_rows))):
        raise RuntimeError("Gate3 per-token KV TP rank evidence is incomplete")
    expected = {
        str(plan_row["control_layer_name"]): (
            int(plan_row["control_layer_index"]),
            "control",
        ),
        str(plan_row["first_different_layer_name"]): (
            int(plan_row["first_different_layer_index"]),
            "first-different",
        ),
    }
    combined_ranks = []
    group_counts = []
    for row in rank_rows:
        summary = row.get("kv_cache_before_forward")
        ownership = row.get("kv_ownership")
        if not isinstance(summary, Mapping) or not isinstance(ownership, Mapping):
            raise RuntimeError("Gate3 per-token KV rank summary is missing")
        layers = summary.get("per_logical_token_kv_layers")
        if not isinstance(layers, list) or {
            layer.get("layer_name") for layer in layers if isinstance(layer, Mapping)
        } != set(expected):
            raise RuntimeError("Gate3 per-token KV selected layers differ across ranks")
        for layer in layers:
            if not isinstance(layer, Mapping):
                raise RuntimeError("Gate3 per-token KV selected layer is malformed")
            name = str(layer["layer_name"])
            layer_index, layer_role = expected[name]
            aggregate_layers = summary.get("layers")
            aggregate_matches = (
                [
                    aggregate_layer
                    for aggregate_layer in aggregate_layers
                    if isinstance(aggregate_layer, Mapping)
                    and aggregate_layer.get("layer") == name
                ]
                if isinstance(aggregate_layers, list)
                else []
            )
            if (
                layer.get("layer_index") != layer_index
                or layer.get("layer_role") != layer_role
                or layer.get("materialized_logical_position_start") != 0
                or layer.get("materialized_logical_position_end_exclusive")
                != num_computed_tokens
                or len(aggregate_matches) != 1
                or aggregate_matches[0].get("raw_sha256")
                != layer.get("aggregate_raw_sha256")
            ):
                raise RuntimeError("Gate3 per-token KV selected layer contract differs")
        group_counts.append(int(ownership.get("kv_cache_group_count", -1)))
        combined_ranks.append(
            {
                "tp_rank": int(row["tp_rank"]),
                "layers": sorted(layers, key=lambda item: int(item["layer_index"])),
            }
        )
    if group_counts != [1] * len(rank_rows):
        raise RuntimeError("Gate3 per-token KV requires the validated one-group layout")
    return {
        "schema_version": "specrhythm.phase4b1-per-logical-token-kv-capture.v1",
        "authority": (
            "InputBatch.MultiGroupBlockTable logical block order + "
            "FlashAttention cache[physical_block,K_or_V,block_offset]"
        ),
        "layout": "flash-attention-paged-kv[num_blocks,2,block_size,...]",
        "materialized_position_definition": "[0,num_computed_tokens)",
        "materialized_logical_position_start": 0,
        "materialized_logical_position_end_exclusive": num_computed_tokens,
        "kv_cache_group_count": 1,
        "tp_world_size": len(rank_rows),
        "tp_ranks": combined_ranks,
    }


def _match_planned_prefix(
    tokens: Sequence[int], plan_by_id: Mapping[str, Mapping[str, Any]]
) -> Optional[tuple[Mapping[str, Any], tuple[int, ...]]]:
    matches = []
    for row in plan_by_id.values():
        prompt = tuple(row["prompt_token_ids"])
        if len(tokens) >= len(prompt) and tuple(tokens[: len(prompt)]) == prompt:
            matches.append((row, tuple(tokens)))
    if len(matches) > 1:
        raise RuntimeError("Gate3 numerical prompt identity is ambiguous")
    return matches[0] if matches else None


def _unwrap_model(model: Any) -> Any:
    current = model
    for _ in range(5):
        if hasattr(current, "lm_head") and hasattr(current, "model"):
            return current
        next_model = getattr(current, "model", None)
        if next_model is None or next_model is current:
            break
        current = next_model
    raise RuntimeError("Gate3 numerical diagnostics require Qwen3ForCausalLM")


def _unique_suffix_module(named: Mapping[str, Any], suffix: str) -> tuple[str, Any]:
    matches = [(name, module) for name, module in named.items() if name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Gate3 numerical module path {suffix} is not unique")
    return matches[0]


def _valid_tensor_summary(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("dtype")
        and isinstance(value.get("shape"), list)
        and value.get("raw_sha256")
        and value.get("fp32_sha256")
        and all(
            isinstance(value.get(name), (int, float))
            and math.isfinite(float(value[name]))
            for name in ("min", "max", "norm")
        )
    )


def _valid_tp_rank_rows(value: Any, *, expected_world_size: int) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) == expected_world_size
        and [row.get("tp_rank") for row in value if isinstance(row, Mapping)]
        == list(range(expected_world_size))
    )


def _valid_kv_ownership(
    value: Any,
    *,
    num_computed_tokens: Any,
    expected_group_count: int,
) -> bool:
    if not _nonnegative_int(num_computed_tokens) or not isinstance(value, Mapping):
        return False
    if value.get("authority") != (
        "InputBatch.MultiGroupBlockTable + "
        "GPUModelRunner._get_slot_mappings.slot_mappings_by_group"
    ):
        return False
    groups = value.get("groups")
    if (
        value.get("kv_cache_group_count") != expected_group_count
        or not isinstance(groups, list)
        or len(groups) != expected_group_count
        or value.get("logical_positions") != list(range(num_computed_tokens))
    ):
        return False
    for group_id, group in enumerate(groups):
        if not isinstance(group, Mapping):
            return False
        block_size = group.get("block_size")
        used = group.get("logical_used_block_count")
        committed = group.get("num_blocks_per_row")
        physical = group.get("physical_block_ids")
        slots = group.get("current_query_slot_mapping")
        layers = group.get("layer_names")
        if (
            group.get("kv_cache_group_id") != group_id
            or not _positive_int(block_size)
            or used != (math.ceil(num_computed_tokens / block_size) if num_computed_tokens else 0)
            or not _nonnegative_int(committed)
            or committed < used
            or not isinstance(physical, list)
            or len(physical) != used
            or any(not _nonnegative_int(item) for item in physical)
            or len(set(physical)) != len(physical)
            or not isinstance(slots, list)
            or len(slots) != 1
            or any(not _nonnegative_int(item) for item in slots)
            or not isinstance(layers, list)
            or not layers
            or any(not isinstance(name, str) or not name for name in layers)
        ):
            return False
    return True


def _valid_per_token_capture(
    value: Any,
    *,
    planned_row: Mapping[str, Any],
    num_computed_tokens: int,
    expected_world_size: int,
    require_independent_binding: bool = False,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    if (
        value.get("schema_version")
        != "specrhythm.phase4b1-per-logical-token-kv-capture.v1"
        or value.get("authority")
        != (
            "InputBatch.MultiGroupBlockTable logical block order + "
            "FlashAttention cache[physical_block,K_or_V,block_offset]"
        )
        or value.get("layout")
        != "flash-attention-paged-kv[num_blocks,2,block_size,...]"
        or value.get("materialized_position_definition")
        != "[0,num_computed_tokens)"
        or value.get("materialized_logical_position_start") != 0
        or value.get("materialized_logical_position_end_exclusive")
        != num_computed_tokens
        or value.get("kv_cache_group_count") != 1
        or value.get("tp_world_size") != expected_world_size
    ):
        return False
    ranks = value.get("tp_ranks")
    if not _valid_tp_rank_rows(ranks, expected_world_size=expected_world_size):
        return False
    expected_layers = {
        str(planned_row["control_layer_name"]): (
            int(planned_row["control_layer_index"]),
            "control",
        ),
        str(planned_row["first_different_layer_name"]): (
            int(planned_row["first_different_layer_index"]),
            "first-different",
        ),
    }
    for rank in ranks:
        layers = rank.get("layers") if isinstance(rank, Mapping) else None
        if not isinstance(layers, list) or len(layers) != 2:
            return False
        by_name = {
            str(layer.get("layer_name")): layer
            for layer in layers
            if isinstance(layer, Mapping)
        }
        if set(by_name) != set(expected_layers):
            return False
        for name, (layer_index, layer_role) in expected_layers.items():
            layer = by_name[name]
            tokens = layer.get("tokens")
            if (
                layer.get("layer_index") != layer_index
                or layer.get("layer_role") != layer_role
                or layer.get("kv_cache_group_id") != 0
                or not layer.get("dtype")
                or not _sha256_text(layer.get("aggregate_raw_sha256"))
                or not _positive_shape(layer.get("k_shape_per_logical_token"))
                or not _positive_shape(layer.get("v_shape_per_logical_token"))
                or layer.get("materialized_logical_position_start") != 0
                or layer.get("materialized_logical_position_end_exclusive")
                != num_computed_tokens
                or layer.get("gpu_to_cpu_transfer_count") != 1
                or not isinstance(tokens, list)
                or len(tokens) != num_computed_tokens
            ):
                return False
            for position, token in enumerate(tokens):
                if (
                    not isinstance(token, Mapping)
                    or token.get("logical_position") != position
                    or not _sha256_text(token.get("k_raw_sha256"))
                    or not _sha256_text(token.get("v_raw_sha256"))
                ):
                    return False
            has_binding = all(
                field in layer
                for field in (
                    "logical_reconstructed_raw_sha256",
                    "logical_reconstructed_raw_ordering",
                    "token_digest_sequence_sha256",
                    "token_digest_sequence_ordering",
                )
            )
            if require_independent_binding and not has_binding:
                return False
            if has_binding and (
                not _sha256_text(layer.get("logical_reconstructed_raw_sha256"))
                or layer.get("logical_reconstructed_raw_ordering")
                != LOGICAL_RAW_ORDERING
                or layer.get("token_digest_sequence_ordering")
                != TOKEN_DIGEST_ORDERING
            ):
                return False
            if has_binding:
                try:
                    sequence_digest = _token_digest_sequence_sha256(tokens)
                except RuntimeError:
                    return False
                if layer.get("token_digest_sequence_sha256") != sequence_digest:
                    return False
    return True


def _per_token_aggregates_match_rank_kv(
    capture: Any, rank_kv: Any
) -> bool:
    """Bind selected token hashes to the existing full-layer rank checksums."""

    token_ranks = _rank_capture_map(capture)
    if not _valid_tp_rank_rows(rank_kv, expected_world_size=2):
        return False
    aggregate_ranks = {
        int(row["tp_rank"]): row for row in rank_kv if isinstance(row, Mapping)
    }
    if set(token_ranks) != {0, 1} or set(aggregate_ranks) != {0, 1}:
        return False
    for rank, token_row in token_ranks.items():
        summary = aggregate_ranks[rank].get("kv_cache_before_forward")
        aggregate_layers = summary.get("layers") if isinstance(summary, Mapping) else None
        token_layers = token_row.get("layers")
        if not isinstance(aggregate_layers, list) or not isinstance(token_layers, list):
            return False
        aggregate_by_name = {}
        for layer in aggregate_layers:
            if not isinstance(layer, Mapping):
                return False
            name = layer.get("layer")
            if not isinstance(name, str) or name in aggregate_by_name:
                return False
            aggregate_by_name[name] = layer.get("raw_sha256")
        for layer in token_layers:
            if (
                not isinstance(layer, Mapping)
                or aggregate_by_name.get(layer.get("layer_name"))
                != layer.get("aggregate_raw_sha256")
            ):
                return False
    return True


def _positive_shape(value: Any) -> bool:
    return bool(
        isinstance(value, list)
        and value
        and all(_positive_int(dimension) for dimension in value)
    )


def _signed_token_list(value: Any) -> bool:
    return bool(
        isinstance(value, list)
        and value
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    )


def _sha256_text(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _logical_ownership(row: Mapping[str, Any]) -> dict[str, Any]:
    mapping = row.get("kv_position_mapping", {})
    return {
        "authority": mapping.get("authority"),
        "kv_cache_group_count": mapping.get("kv_cache_group_count"),
        "logical_positions": mapping.get("logical_positions"),
        "groups": [
            {
                "kv_cache_group_id": group.get("kv_cache_group_id"),
                "block_size": group.get("block_size"),
                "logical_used_block_count": group.get("logical_used_block_count"),
                "layer_names": group.get("layer_names"),
                "logical_block_offsets": [
                    int(position) % int(group.get("block_size", 1))
                    for position in mapping.get("logical_positions", ())
                ],
            }
            for group in mapping.get("groups", ())
            if isinstance(group, Mapping)
        ],
        "current_query_position": row.get("target_input_token_position"),
    }


def _stage_sha(row: Mapping[str, Any], name: str) -> Any:
    rank_evidence = row.get("tp_rank_evidence")
    if isinstance(rank_evidence, list):
        return [
            rank.get("tensor_stages", {}).get(name, {}).get("raw_sha256")
            for rank in rank_evidence
            if isinstance(rank, Mapping)
        ]
    return row.get("tensor_stages", {}).get(name, {}).get("raw_sha256")


def _candidate_logits(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list(row.get("raw_pre_softmax_logits", {}).get("competing_tokens", ()))


def _first_different_stage(
    *,
    semantic_equal: bool,
    ownership_equal: bool,
    kv_equal: bool,
    stage_equal: Mapping[str, bool],
    candidate_logits_equal: bool,
    sampler_equal: bool,
) -> str:
    if not semantic_equal or not ownership_equal:
        return "semantic-or-logical-kv-ownership"
    if not kv_equal:
        return "kv-cache-raw-values"
    for name in REQUIRED_STAGES:
        if not stage_equal.get(name, False):
            return name
    if not candidate_logits_equal:
        return "lm-head-raw-logits"
    if not sampler_equal:
        return "sampler"
    return "none-observed"


def _validated_output_map(
    rows: Sequence[Mapping[str, Any]], label: str, errors: list[str]
) -> dict[str, Mapping[str, Any]]:
    result = {}
    if len(rows) != 100:
        errors.append(f"{label} final output does not preserve the 100-request shape")
    for row in rows:
        request_id = str(row.get("request_id", ""))
        tokens = _generated_tokens(row)
        if not request_id or request_id in result or tokens is None:
            errors.append(f"{label} final output request/token data is invalid")
            continue
        result[request_id] = row
    return result


def _validated_reference_map(
    reference: Mapping[str, Any], plan: Mapping[str, Any], errors: list[str]
) -> dict[str, Mapping[str, Any]]:
    from specrhythm.phase4.reference import validate_stock_reference

    errors.extend(
        f"immutable reference: {error}"
        for error in validate_stock_reference(reference)
    )
    workload = reference.get("workload")
    if (
        not isinstance(workload, Mapping)
        or workload.get("sha256") != plan.get("workload_sha256")
    ):
        errors.append("immutable stock reference workload checksum differs")
    frozen_requests = (
        workload.get("requests") if isinstance(workload, Mapping) else None
    )
    frozen_by_id = {
        str(row.get("request_id", "")): row
        for row in frozen_requests
        if isinstance(row, Mapping)
    } if isinstance(frozen_requests, list) else {}
    for planned in plan.get("requests", ()):
        if not isinstance(planned, Mapping):
            continue
        frozen = frozen_by_id.get(str(planned["request_id"]))
        if (
            not isinstance(frozen, Mapping)
            or frozen.get("prompt_token_ids") != planned.get("prompt_token_ids")
            or frozen.get("maximum_new_tokens") != planned.get("maximum_new_tokens")
        ):
            errors.append(
                "immutable stock reference planned prompt/output limit differs: "
                f"{planned['request_id']}"
            )
    outputs = reference.get("outputs")
    if not isinstance(outputs, list):
        errors.append("immutable stock reference outputs are missing")
        return {}
    return _validated_output_map(outputs, "immutable reference", errors)


def _generated_tokens(row: Optional[Mapping[str, Any]]) -> Optional[list[int]]:
    tokens = row.get("generated_token_ids") if isinstance(row, Mapping) else None
    if not isinstance(tokens, list) or any(
        not _nonnegative_int(token) for token in tokens
    ):
        return None
    return [int(token) for token in tokens]


def _first_output_divergence(
    actual: Sequence[int], expected: Sequence[int]
) -> Optional[int]:
    for position, (actual_token, expected_token) in enumerate(
        zip(actual, expected)
    ):
        if actual_token != expected_token:
            return position
    return min(len(actual), len(expected)) if len(actual) != len(expected) else None


def _output_divergences(
    actual_by_id: Mapping[str, Mapping[str, Any]],
    reference_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    result = {}
    for request_id, reference in reference_by_id.items():
        actual = _generated_tokens(actual_by_id.get(request_id))
        expected = _generated_tokens(reference)
        if actual is None or expected is None:
            continue
        position = _first_output_divergence(actual, expected)
        if position is not None:
            result[request_id] = position
    return result


def _rank_capture_map(value: Any) -> dict[int, Mapping[str, Any]]:
    ranks = value.get("tp_ranks") if isinstance(value, Mapping) else None
    if not isinstance(ranks, list):
        return {}
    result = {}
    for row in ranks:
        if not isinstance(row, Mapping) or not _nonnegative_int(row.get("tp_rank")):
            continue
        rank = int(row["tp_rank"])
        if rank in result:
            return {}
        result[rank] = row
    return result


def _compare_selected_token_layer(
    stock_rank: Mapping[str, Any],
    resident_rank: Mapping[str, Any],
    *,
    layer_name: str,
    layer_index: int,
    layer_role: str,
    prompt_length: int,
) -> dict[str, Any]:
    stock = _selected_layer(stock_rank, layer_name)
    resident = _selected_layer(resident_rank, layer_name)
    identity_valid = bool(
        stock
        and resident
        and stock.get("layer_name") == resident.get("layer_name") == layer_name
        and stock.get("layer_index") == resident.get("layer_index") == layer_index
        and stock.get("layer_role") == resident.get("layer_role") == layer_role
        and stock.get("kv_cache_group_id")
        == resident.get("kv_cache_group_id")
        == 0
        and stock.get("dtype") == resident.get("dtype")
        and stock.get("k_shape_per_logical_token")
        == resident.get("k_shape_per_logical_token")
        and stock.get("v_shape_per_logical_token")
        == resident.get("v_shape_per_logical_token")
    )
    stock_tokens = _token_hash_map(stock)
    resident_tokens = _token_hash_map(resident)
    positions = sorted(set(stock_tokens) | set(resident_tokens))
    k_differences = [
        position
        for position in positions
        if stock_tokens.get(position, {}).get("k_raw_sha256")
        != resident_tokens.get(position, {}).get("k_raw_sha256")
    ]
    v_differences = [
        position
        for position in positions
        if stock_tokens.get(position, {}).get("v_raw_sha256")
        != resident_tokens.get(position, {}).get("v_raw_sha256")
    ]
    all_differences = sorted(set(k_differences) | set(v_differences))
    first = all_differences[0] if all_differences else None
    window = []
    if first is not None:
        for position in range(max(0, first - 2), min(max(positions, default=-1), first + 2) + 1):
            stock_token = stock_tokens.get(position, {})
            resident_token = resident_tokens.get(position, {})
            window.append(
                {
                    "logical_position": position,
                    "k_equal": stock_token.get("k_raw_sha256")
                    == resident_token.get("k_raw_sha256"),
                    "v_equal": stock_token.get("v_raw_sha256")
                    == resident_token.get("v_raw_sha256"),
                    "stock_k_raw_sha256": stock_token.get("k_raw_sha256"),
                    "resident_k_raw_sha256": resident_token.get("k_raw_sha256"),
                    "stock_v_raw_sha256": stock_token.get("v_raw_sha256"),
                    "resident_v_raw_sha256": resident_token.get("v_raw_sha256"),
                }
            )
    return {
        "layer_name": layer_name,
        "layer_index": layer_index,
        "layer_role": layer_role,
        "kv_cache_group_id": 0,
        "identity_valid": identity_valid,
        "dtype": stock.get("dtype") if stock else None,
        "k_shape_per_logical_token": (
            stock.get("k_shape_per_logical_token") if stock else None
        ),
        "v_shape_per_logical_token": (
            stock.get("v_shape_per_logical_token") if stock else None
        ),
        "aggregate_raw_bytes_equal": (
            stock.get("aggregate_raw_sha256")
            == resident.get("aggregate_raw_sha256")
            if stock and resident
            else None
        ),
        "all_k_token_hashes_equal": not k_differences and bool(positions),
        "all_v_token_hashes_equal": not v_differences and bool(positions),
        "differing_k_logical_position_count": len(k_differences),
        "differing_v_logical_position_count": len(v_differences),
        "first_differing_logical_position": first,
        "phase": _logical_token_phase(first, prompt_length),
        "k_equal_at_first_differing_position": (
            first not in k_differences if first is not None else None
        ),
        "v_equal_at_first_differing_position": (
            first not in v_differences if first is not None else None
        ),
        "first_differing_k_position": (
            k_differences[0] if k_differences else None
        ),
        "first_differing_v_position": (
            v_differences[0] if v_differences else None
        ),
        "last_differing_k_position": (
            k_differences[-1] if k_differences else None
        ),
        "last_differing_v_position": (
            v_differences[-1] if v_differences else None
        ),
        "local_window": window,
    }


def _selected_layer(
    rank: Mapping[str, Any], layer_name: str
) -> Mapping[str, Any]:
    layers = rank.get("layers")
    if not isinstance(layers, list):
        return {}
    matches = [
        layer
        for layer in layers
        if isinstance(layer, Mapping) and layer.get("layer_name") == layer_name
    ]
    return matches[0] if len(matches) == 1 else {}


def _token_hash_map(layer: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    tokens = layer.get("tokens") if isinstance(layer, Mapping) else None
    if not isinstance(tokens, list):
        return {}
    result = {}
    for row in tokens:
        if not isinstance(row, Mapping) or not _nonnegative_int(
            row.get("logical_position")
        ):
            return {}
        position = int(row["logical_position"])
        if position in result:
            return {}
        result[position] = row
    return result


def _logical_token_phase(position: Optional[int], prompt_length: int) -> Optional[str]:
    if position is None:
        return None
    if position < prompt_length:
        return "PROMPT_PREFILL"
    if position == prompt_length:
        return "BOOTSTRAP"
    return "DECODE_HISTORY"


def _per_token_classification(phases: Sequence[str], errors: Sequence[str]) -> str:
    if errors or not phases:
        return "FAIL-CLOSED"
    if "PROMPT_PREFILL" in phases:
        return "PROMPT_PREFILL"
    if "BOOTSTRAP" in phases:
        return "BOOTSTRAP"
    if set(phases) == {"DECODE_HISTORY"}:
        return "DECODE_HISTORY"
    return "FAIL-CLOSED"


def _records_by_key(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    result = {}
    for row in rows:
        position = row.get("output_position")
        if isinstance(position, int) and not isinstance(position, bool):
            result[(str(row.get("request_id", "")), position)] = row
    return result


def _outputs_by_id(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("request_id", "")): row for row in rows}


def _output_token(row: Optional[Mapping[str, Any]], position: int) -> Optional[int]:
    tokens = row.get("generated_token_ids") if isinstance(row, Mapping) else None
    if not isinstance(tokens, list) or position < 0 or position >= len(tokens):
        return None
    token = tokens[position]
    return int(token) if _nonnegative_int(token) else None


def token_sha256(token_ids: Sequence[int]) -> str:
    payload = json.dumps(list(token_ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"workload line {line_number} is not an object")
            rows.append(value)
    return rows


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _positive_int(value: Any) -> bool:
    return _nonnegative_int(value) and value > 0


def _token_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(_nonnegative_int(item) for item in value)
    )
