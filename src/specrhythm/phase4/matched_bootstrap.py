"""Exact three-way Gate3 matched-bootstrap diagnostic comparison.

This module compares immutable stock-async-ON and resident endpoints with one
ordinary stock Target run whose only intended control change is
``async_scheduling=False``.  It never changes the serving correctness policy.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from specrhythm.phase4.numerical_diagnostics import (
    PER_TOKEN_COMPARISON_SCHEMA,
    PER_TOKEN_PLAN_SCHEMA,
    SEMANTIC_PREFIX_AUTHORITY,
    _candidate_logits,
    _generated_tokens,
    _logical_ownership,
    _output_divergences,
    _output_token,
    _rank_capture_map,
    _records_by_key,
    _selected_layer,
    _token_hash_map,
    _validated_output_map,
    _validated_reference_map,
    validate_numerical_records,
)
from specrhythm.phase4.stock_vllm import validate_matched_bootstrap_control

COMPARISON_SCHEMA = "specrhythm.phase4b1-gate3-matched-bootstrap-comparison.v1"
CONTROL_MODE = "matched-stock-async-off"


def compare_matched_bootstrap(
    *,
    plan: Mapping[str, Any],
    stock_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    resident_rows: Sequence[Mapping[str, Any]],
    stock_outputs: Sequence[Mapping[str, Any]],
    control_outputs: Sequence[Mapping[str, Any]],
    resident_outputs: Sequence[Mapping[str, Any]],
    immutable_reference: Mapping[str, Any],
    endpoint_comparison: Mapping[str, Any],
    control_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare A=stock async ON, B=stock async OFF, C=resident async OFF."""

    errors: list[str] = []
    if plan.get("schema_version") != PER_TOKEN_PLAN_SCHEMA:
        errors.append("matched-bootstrap comparison requires the immutable per-token plan")
    if (
        endpoint_comparison.get("schema_version") != PER_TOKEN_COMPARISON_SCHEMA
        or endpoint_comparison.get("valid") is not True
        or endpoint_comparison.get("classification") != "BOOTSTRAP"
        or endpoint_comparison.get("gate3_closed") is not False
        or endpoint_comparison.get("phase4b2_blocked") is not True
        or endpoint_comparison.get("tolerant_correctness_policy") is not False
        or endpoint_comparison.get("tie_equivalent_tokens_accepted") is not False
    ):
        errors.append(
            "immutable stock/resident endpoint comparison is not valid "
            "BOOTSTRAP evidence"
        )
    errors.extend(
        f"stock async ON: {item}"
        for item in validate_numerical_records(
            stock_rows, plan, execution_mode="stock-style"
        )
    )
    errors.extend(
        f"matched async OFF: {item}"
        for item in validate_numerical_records(
            control_rows, plan, execution_mode=CONTROL_MODE
        )
    )
    errors.extend(
        f"resident async OFF: {item}"
        for item in validate_numerical_records(
            resident_rows, plan, execution_mode="resident-target"
        )
    )
    errors.extend(
        f"control runtime: {item}"
        for item in validate_matched_bootstrap_control(control_runtime)
    )

    stock_by_id = _validated_output_map(stock_outputs, "stock async ON", errors)
    control_by_id = _validated_output_map(
        control_outputs, "matched stock async OFF", errors
    )
    resident_by_id = _validated_output_map(
        resident_outputs, "resident async OFF", errors
    )
    reference_by_id = _validated_reference_map(immutable_reference, plan, errors)
    expected_ids = set(reference_by_id)
    reference_order = [
        str(row.get("request_id", ""))
        for row in immutable_reference.get("outputs", ())
        if isinstance(row, Mapping)
    ]
    request_order_equal = {}
    for label, rows in (
        ("stock async ON", stock_by_id),
        ("matched stock async OFF", control_by_id),
        ("resident async OFF", resident_by_id),
    ):
        if set(rows) != expected_ids:
            errors.append(f"{label} request IDs differ from the immutable reference")
    for label, rows in (
        ("stock_async_on", stock_outputs),
        ("matched_async_off", control_outputs),
        ("resident_async_off", resident_outputs),
    ):
        order = [
            str(row.get("request_id", ""))
            for row in rows
            if isinstance(row, Mapping)
        ]
        request_order_equal[label] = order == reference_order
        if not request_order_equal[label]:
            errors.append(f"{label} final output request order differs")
    stock_divergences = _output_divergences(stock_by_id, reference_by_id)
    if stock_divergences:
        errors.append("stock async-ON outputs no longer reproduce the immutable reference")
    expected_resident = {
        str(row["request_id"]): int(row["output_position"])
        for row in plan.get("requests", ())
        if isinstance(row, Mapping)
    }
    if _output_divergences(resident_by_id, reference_by_id) != expected_resident:
        errors.append("resident endpoint no longer reproduces exactly the four divergences")
    control_divergences = _output_divergences(control_by_id, reference_by_id)

    stock_by_key = _records_by_key(stock_rows)
    control_by_key = _records_by_key(control_rows)
    resident_by_key = _records_by_key(resident_rows)
    comparisons = []
    request_states = []
    for item in plan.get("requests", ()):
        if not isinstance(item, Mapping):
            continue
        request_id = str(item["request_id"])
        position = int(item["output_position"])
        key = (request_id, position)
        row_errors = []
        records = {
            "stock_async_on": stock_by_key.get(key),
            "matched_async_off": control_by_key.get(key),
            "resident_async_off": resident_by_key.get(key),
        }
        if any(record is None for record in records.values()):
            row_errors.append("one or more three-way numerical checkpoints are missing")
            errors.extend(f"{request_id}@{position}: {value}" for value in row_errors)
            continue
        stock_record = records["stock_async_on"]
        control_record = records["matched_async_off"]
        resident_record = records["resident_async_off"]
        assert stock_record is not None
        assert control_record is not None
        assert resident_record is not None

        reference_tokens = _generated_tokens(reference_by_id.get(request_id))
        prefixes = {
            label: _prefix_at(rows.get(request_id), position)
            for label, rows in (
                ("stock_async_on", stock_by_id),
                ("matched_async_off", control_by_id),
                ("resident_async_off", resident_by_id),
            )
        }
        expected_prefix = (
            reference_tokens[:position] if reference_tokens is not None else None
        )
        prefix_exact = {
            label: prefix == expected_prefix and prefix is not None
            for label, prefix in prefixes.items()
        }
        if not all(prefix_exact.values()):
            row_errors.append("actual generated prefix diverges before the planned checkpoint")
        selected_tokens = {
            "stock_async_on": _output_token(stock_by_id.get(request_id), position),
            "matched_async_off": _output_token(control_by_id.get(request_id), position),
            "resident_async_off": _output_token(
                resident_by_id.get(request_id), position
            ),
        }
        if (
            selected_tokens["stock_async_on"] != item["stock_selected_token_id"]
            or selected_tokens["resident_async_off"]
            != item["resident_selected_token_id"]
            or selected_tokens["matched_async_off"] is None
        ):
            row_errors.append("endpoint/control selected-token evidence is incomplete")

        expected_computed = int(item["prompt_length"]) + position - 1
        if any(
            record.get("num_computed_tokens") != expected_computed
            or record.get("target_input_token_position") != expected_computed
            for record in records.values()
        ):
            row_errors.append("three-way materialized Target boundary differs")
        ownerships = [_logical_ownership(record) for record in records.values()]
        if ownerships[0] != ownerships[1] or ownerships[0] != ownerships[2]:
            row_errors.append("three-way logical KV ownership differs")

        rank_results = []
        per_request_states = []
        captures = {
            label: _rank_capture_map(record.get("per_logical_token_kv"))
            for label, record in records.items()
        }
        if any(set(ranks) != {0, 1} for ranks in captures.values()):
            row_errors.append("three-way per-token TP rank evidence is incomplete")
        for rank in (0, 1):
            if any(rank not in ranks for ranks in captures.values()):
                continue
            rank_values = {label: ranks[rank] for label, ranks in captures.items()}
            layer_results = {}
            for role, name_field, index_field in (
                ("control", "control_layer_name", "control_layer_index"),
                (
                    "first-different",
                    "first_different_layer_name",
                    "first_different_layer_index",
                ),
            ):
                layer_name = str(item[name_field])
                layers = {
                    label: _selected_layer(value, layer_name)
                    for label, value in rank_values.items()
                }
                if not _three_way_layer_identity(
                    layers,
                    layer_name=layer_name,
                    layer_index=int(item[index_field]),
                    layer_role=role,
                ):
                    row_errors.append(f"TP{rank} {role} layer identity differs")
                    continue
                token_maps = {
                    label: _token_hash_map(layer) for label, layer in layers.items()
                }
                if any(not values for values in token_maps.values()):
                    row_errors.append(f"TP{rank} {role} token map is incomplete")
                    continue
                prompt_exact = _range_three_way_exact(
                    token_maps, 0, int(item["prompt_length"])
                )
                if not prompt_exact:
                    row_errors.append(f"TP{rank} {role} prompt K/V differs")
                control_exact = None
                bootstrap = None
                if role == "control":
                    control_exact = _all_three_way_exact(token_maps)
                    if not control_exact:
                        row_errors.append(f"TP{rank} previous control layer differs")
                else:
                    bootstrap = _bootstrap_components(
                        token_maps, int(item["prompt_length"])
                    )
                    if bootstrap["endpoint_different_component_count"] == 0:
                        row_errors.append(
                            f"TP{rank} first layer has no stock/resident bootstrap difference"
                        )
                    per_request_states.append(str(bootstrap["control_state"]))
                layer_results[role] = {
                    "layer_name": layer_name,
                    "layer_index": int(item[index_field]),
                    "prompt_kv_exact": prompt_exact,
                    "all_materialized_kv_exact": control_exact,
                    "bootstrap": bootstrap,
                    "logical_digest_binding": {
                        label: {
                            "logical_reconstructed_raw_sha256": layer.get(
                                "logical_reconstructed_raw_sha256"
                            ),
                            "token_digest_sequence_sha256": layer.get(
                                "token_digest_sequence_sha256"
                            ),
                        }
                        for label, layer in layers.items()
                    },
                }
            rank_results.append({"tp_rank": rank, "layers": layer_results})

        request_state = _request_control_state(per_request_states)
        if request_state == "MIXED_COMPONENTS":
            request_state = "THIRD_STATE"
        request_states.append(request_state)
        logits = {
            label: _logit_map(record) for label, record in records.items()
        }
        expected_competing = {
            int(item["stock_selected_token_id"]),
            int(item["resident_selected_token_id"]),
        }
        if any(set(value) != expected_competing for value in logits.values()):
            row_errors.append("three-way competing raw logits are incomplete")
        raw_argmax = {
            label: record.get("raw_argmax_token_id")
            for label, record in records.items()
        }
        if raw_argmax["matched_async_off"] != selected_tokens["matched_async_off"]:
            row_errors.append("matched-control raw argmax does not equal its output token")
        row_errors = list(dict.fromkeys(row_errors))
        errors.extend(f"{request_id}@{position}: {value}" for value in row_errors)
        comparisons.append(
            {
                "request_id": request_id,
                "output_position": position,
                "prompt_length": int(item["prompt_length"]),
                "semantic_prefix_authority": SEMANTIC_PREFIX_AUTHORITY,
                "pre_checkpoint_prefix_exact": prefix_exact,
                "three_way_semantically_paired": all(prefix_exact.values()),
                "selected_token_ids": selected_tokens,
                "control_token_equals_stock": (
                    selected_tokens["matched_async_off"]
                    == selected_tokens["stock_async_on"]
                ),
                "control_token_equals_resident": (
                    selected_tokens["matched_async_off"]
                    == selected_tokens["resident_async_off"]
                ),
                "raw_competing_logits": logits,
                "raw_argmax_token_ids": raw_argmax,
                "control_logits_equal_stock": logits["matched_async_off"]
                == logits["stock_async_on"],
                "control_logits_equal_resident": logits["matched_async_off"]
                == logits["resident_async_off"],
                "execution_shape": {
                    label: record.get("execution_shape")
                    for label, record in records.items()
                },
                "request_control_state": request_state,
                "tp_ranks": rank_results,
                "errors": row_errors,
            }
        )

    if len(comparisons) != 4:
        errors.append("matched-bootstrap comparison did not resolve all four requests")
    classification = _overall_classification(request_states, errors)
    return {
        "schema_version": COMPARISON_SCHEMA,
        "diagnostic_only": True,
        "valid": not errors,
        "errors": errors,
        "classification": classification,
        "semantic_prefix_authority": SEMANTIC_PREFIX_AUTHORITY,
        "endpoint_modes": {
            "stock_async_on": "ordinary Target-only, speculative_config=None, async ON",
            "matched_async_off": "ordinary Target-only, speculative_config=None, async OFF",
            "resident_async_off": "resident Target, custom proposer, async OFF",
        },
        "control_runtime": dict(control_runtime),
        "request_order_equal_to_immutable_reference": request_order_equal,
        "request_count": len(comparisons),
        "comparisons": comparisons,
        "control_output_divergences_from_immutable_reference": [
            {
                "request_id": request_id,
                "first_divergence_position": position,
                "immutable_token_id": _output_token(
                    reference_by_id.get(request_id), position
                ),
                "control_token_id": _output_token(
                    control_by_id.get(request_id), position
                ),
            }
            for request_id, position in sorted(control_divergences.items())
        ],
        "control_output_divergence_count": len(control_divergences),
        "gate3_closed": False,
        "phase4b2_blocked": True,
        "tolerant_correctness_policy": False,
        "tie_equivalent_tokens_accepted": False,
        "performance_result": False,
        "correctness_decision": "human-classification-required",
    }


def comparison_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Gate3 matched-bootstrap async control",
        "",
        "Target-only diagnostic control; Gate3 remains open.",
        "",
        f"Classification: **{report.get('classification', 'FAIL-CLOSED')}**",
        "",
        "| Request | Position | Control K/V state | Control token = stock | "
        "Control token = resident | Control logits = stock | Control logits = resident |",
        "| --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in report.get("comparisons", ()):
        lines.append(
            "| {request_id} | {output_position} | {request_control_state} | "
            "{control_token_equals_stock} | {control_token_equals_resident} | "
            "{control_logits_equal_stock} | {control_logits_equal_resident} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "The control may differ at the planned token, but its actual prefix strictly "
            "before that token must match the immutable reference. Prompt K/V and the "
            "selected previous control layer remain exact requirements.",
            "",
            "No tolerance, tie-equivalence, Gate3 closure, or performance conclusion is "
            "authorized by this report. Phase 4B.2 remains blocked.",
            "",
        ]
    )
    return "\n".join(lines)


def _prefix_at(row: Optional[Mapping[str, Any]], position: int) -> Optional[list[int]]:
    tokens = _generated_tokens(row)
    if tokens is None or len(tokens) <= position:
        return None
    return tokens[:position]


def _three_way_layer_identity(
    layers: Mapping[str, Mapping[str, Any]],
    *,
    layer_name: str,
    layer_index: int,
    layer_role: str,
) -> bool:
    expected = {
        "layer_name": layer_name,
        "layer_index": layer_index,
        "layer_role": layer_role,
        "kv_cache_group_id": 0,
    }
    rows = list(layers.values())
    return bool(
        len(rows) == 3
        and all(
            row
            and all(row.get(key) == value for key, value in expected.items())
            for row in rows
        )
        and len({str(row.get("dtype")) for row in rows}) == 1
        and len({tuple(row.get("k_shape_per_logical_token", ())) for row in rows}) == 1
        and len({tuple(row.get("v_shape_per_logical_token", ())) for row in rows}) == 1
    )


def _range_three_way_exact(
    maps: Mapping[str, Mapping[int, Mapping[str, Any]]], start: int, end: int
) -> bool:
    for position in range(start, end):
        for component in ("k_raw_sha256", "v_raw_sha256"):
            values = {
                rows.get(position, {}).get(component) for rows in maps.values()
            }
            if len(values) != 1 or None in values:
                return False
    return True


def _all_three_way_exact(
    maps: Mapping[str, Mapping[int, Mapping[str, Any]]]
) -> bool:
    positions = {tuple(sorted(rows)) for rows in maps.values()}
    if len(positions) != 1 or not positions:
        return False
    first = next(iter(positions))
    return _range_three_way_exact(maps, int(first[0]), int(first[-1]) + 1)


def _bootstrap_components(
    maps: Mapping[str, Mapping[int, Mapping[str, Any]]], position: int
) -> dict[str, Any]:
    components = []
    states = []
    for component in ("k_raw_sha256", "v_raw_sha256"):
        values = {
            label: rows.get(position, {}).get(component)
            for label, rows in maps.items()
        }
        stock = values["stock_async_on"]
        control = values["matched_async_off"]
        resident = values["resident_async_off"]
        endpoint_differs = stock != resident
        if not endpoint_differs:
            state = "COMMON" if control == stock and stock is not None else "THIRD"
        elif control == resident:
            state = "RESIDENT"
            states.append(state)
        elif control == stock:
            state = "STOCK"
            states.append(state)
        else:
            state = "THIRD"
            states.append(state)
        components.append(
            {
                "component": component[0].upper(),
                "stock_async_on_raw_sha256": stock,
                "matched_async_off_raw_sha256": control,
                "resident_async_off_raw_sha256": resident,
                "stock_resident_differ": endpoint_differs,
                "control_state": state,
            }
        )
    endpoint_count = sum(row["stock_resident_differ"] for row in components)
    unrelated = any(
        not row["stock_resident_differ"] and row["control_state"] != "COMMON"
        for row in components
    )
    if unrelated or "THIRD" in states:
        state = "THIRD_STATE"
    elif states and set(states) == {"RESIDENT"}:
        state = "RESIDENT"
    elif states and set(states) == {"STOCK"}:
        state = "STOCK"
    else:
        state = "MIXED_COMPONENTS"
    return {
        "logical_position": position,
        "components": components,
        "endpoint_different_component_count": endpoint_count,
        "unrelated_control_difference": unrelated,
        "control_state": state,
    }


def _request_control_state(states: Sequence[str]) -> str:
    if not states:
        return "THIRD_STATE"
    normalized = {"THIRD_STATE" if value == "THIRD_STATE" else value for value in states}
    if normalized == {"RESIDENT"}:
        return "RESIDENT"
    if normalized == {"STOCK"}:
        return "STOCK"
    if "THIRD_STATE" in normalized:
        return "THIRD_STATE"
    return "MIXED_COMPONENTS"


def _overall_classification(states: Sequence[str], errors: Sequence[str]) -> str:
    if errors or len(states) != 4:
        return "FAIL-CLOSED"
    values = set(states)
    if values == {"RESIDENT"}:
        return "ASYNC_OFF_MATCHES_RESIDENT"
    if values == {"STOCK"}:
        return "ASYNC_OFF_MATCHES_STOCK"
    if values == {"THIRD_STATE"}:
        return "ASYNC_OFF_THIRD_STATE"
    return "MIXED_BY_REQUEST"


def _logit_map(record: Mapping[str, Any]) -> dict[int, float]:
    result = {}
    for row in _candidate_logits(record):
        token_id = row.get("token_id")
        raw_logit = row.get("raw_logit")
        if isinstance(token_id, int) and not isinstance(token_id, bool) and isinstance(
            raw_logit, (int, float)
        ):
            result[int(token_id)] = float(raw_logit)
    return result
