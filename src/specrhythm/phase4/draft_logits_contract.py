"""CPU-only frozen-input, raw-logit and four-path classification contracts."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import sys
from array import array
from pathlib import Path

from specrhythm.phase4.serial import token_prefix_hash

FIXTURE_PATH = Path(__file__).with_name("d3_logits_probe_inputs.json")
FIXTURE_SHA256 = "ac834cc997c0b4c87f7a0b22d726816a983def6cfeb9b7dc16f38583479a2424"
PATHS = ("hf-fresh", "vllm-fresh-singleton", "vllm-fresh-b7", "vllm-persistent-b7")
PATH_SCHEMA = "specrhythm.phase4b3-d3-raw-logits-path.v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_probe_fixture(path=FIXTURE_PATH):
    raw = Path(path).read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256, "frozen probe fixture SHA256 mismatch"
    )
    fixture = json.loads(raw)
    order = fixture["request_order"]
    require(order == [f"batch-8-{i}" for i in range(1, 8)], "frozen request order mismatch")
    require([r["request_id"] for r in fixture["rows"]] == order, "frozen rows mismatch")
    require(fixture["focus_request_id"] == "batch-8-7", "frozen focus mismatch")
    require((fixture["round"], fixture["frame"]) == (2, 8), "frozen frame mismatch")
    require(fixture["tokens_of_interest"] == [16, 227], "frozen token interests mismatch")
    prefixes = fixture["history"]["initial"]
    for item in fixture["history"]["rounds"][:2]:
        updated = {}
        for row in item["synchronizations"]:
            prefix = prefixes[row["request_id"]] + row["committed_delta"]
            require(
                token_prefix_hash(prefix) == row["committed_prefix_hash"], "history hash mismatch"
            )
            if not row["terminal"]:
                updated[row["request_id"]] = prefix
        prefixes = updated
    for row in fixture["rows"]:
        require(
            row["committed_prefix"] == prefixes[row["request_id"]], "history/fresh prefix mismatch"
        )
        require(row["append_token"] == 11, "frozen scheduled token mismatch")
        require(
            row["comparison_prefix"] == row["committed_prefix"] + [11],
            "comparison prefix mismatch",
        )
        for key in ("committed_prefix", "comparison_prefix"):
            require(token_prefix_hash(row[key]) == row[key + "_sha256"], f"{key} hash mismatch")
    require(
        len(fixture["rows"][-1]["comparison_prefix"]) == 44, "focus prefix must contain 44 tokens"
    )
    return fixture


def summarize_logits(values, original_dtype):
    """Preserve the full float32 CPU vector and stable, raw-score top-k.

    JSON's normal float serialization preserves float32 values; no formatting or
    display rounding is used. Top-k ties use ascending token ID like argmax.
    """
    require(len(values) > 227, "raw logits omit tokens of interest")
    require(
        all(type(v) in (float, int) and math.isfinite(v) for v in values),
        "nonfinite/raw logits invalid",
    )
    packed = array("f", values)
    require(
        packed.itemsize == 4 and list(packed) == list(values),
        "raw logits are not lossless float32 values",
    )
    if sys.byteorder != "little":
        packed.byteswap()
    top = heapq.nlargest(10, range(len(values)), key=lambda i: (values[i], -i))
    return {
        "original_dtype": original_dtype,
        "storage_dtype": "float32",
        "representation": "full raw vector, lossless JSON float32 values; no softmax",
        "raw_logits_float32": list(values),
        "raw_logits_sha256_le_f32": hashlib.sha256(packed.tobytes()).hexdigest(),
        "vocab_size": len(values),
        "token_logits": {str(i): values[i] for i in (16, 227)},
        "top10_token_ids": top,
        "top10_logits": [values[i] for i in top],
        "top1_token": top[0],
        "margin_16_minus_227": values[16] - values[227],
        "absolute_delta_16_227": abs(values[16] - values[227]),
    }


def without_vector(value):
    return {k: v for k, v in value.items() if k != "raw_logits_float32"}


def classify_paths(fixture, reports):
    """Evidence labels, never an automatic correctness fix or D4 admission."""
    rows = {row["request_id"]: row for row in fixture["rows"]}
    focus = rows[fixture["focus_request_id"]]
    failures, checked = [], {}
    for path in PATHS:
        report = reports.get(path, {})
        try:
            require(isinstance(report, dict), "path report must be an object")
            for field in ("run_identity", "runtime_provenance", "gpu_identity"):
                require(
                    isinstance(report.get(field), dict) and bool(report[field]), f"missing {field}"
                )
            require(bool(report.get("process_lifetime_id")), "missing process lifetime identity")
            require(
                isinstance(report["gpu_identity"].get("gpu_uuid"), str), "missing actual GPU UUID"
            )
            require(report.get("valid") is True, "path execution/capture/cleanup did not succeed")
            require(
                report.get("schema_version") == PATH_SCHEMA and report.get("path") == path,
                "path schema/identity mismatch",
            )
            require(
                report.get("fixture_sha256") == FIXTURE_SHA256, "path fixture identity mismatch"
            )
            require(
                report.get("prefix_sha256") == focus["comparison_prefix_sha256"],
                "path prefix hash mismatch",
            )
            require(
                report.get("prefix_token_ids") == focus["comparison_prefix"],
                "path prefix tokens mismatch",
            )
            require(report.get("model_dtype") == "bfloat16", "path model dtype differs")
            require(report.get("cleanup_complete") is True, "path cleanup incomplete")
            expected_order = (
                [focus["request_id"]] if path in PATHS[:2] else fixture["request_order"]
            )
            require(report.get("request_order") == expected_order, "path request order differs")
            require(report.get("batch_size") == len(expected_order), "path batch size differs")
            require(report.get("initial_live_requests") == 0, "path inherited request history")
            require(
                report.get("history_rounds_replayed") == (2 if path == PATHS[3] else 0),
                "fresh/history control contaminated",
            )
            raw = report["logits"]
            actual = summarize_logits(raw["raw_logits_float32"], raw["original_dtype"])
            require(actual == raw, "raw-logit summary disagrees with stored tensor values")
            require(raw["vocab_size"] == 151936, "unexpected Qwen3 vocabulary size")
            require(
                report["top1_by_request"][focus["request_id"]] == raw["top1_token"],
                "focus row mapping mismatch",
            )
            require(
                set(report["top1_by_request"]) == set(expected_order),
                "top1 request domain mismatch",
            )
            if path != PATHS[0]:
                frame = report["captured_frame"]
                ids = ["sr-draft:" + rid for rid in expected_order]
                require(
                    frame["prepared_logits_domain"]["request_ids"] == ids,
                    "captured row domain differs",
                )
                require(
                    frame["prepared_logits_domain"]["logits_indices"] == list(range(len(ids))),
                    "captured logits indices differ",
                )
                require(
                    frame["prepared_logits_domain"]["query_start_loc"]
                    == list(range(len(ids) + 1)),
                    "captured query spans differ",
                )
                require(
                    [row["internal_id"] for row in frame["before_materialize"]] == ids
                    and [row["request_id"] for row in frame["before_materialize"]]
                    == expected_order,
                    "captured materialization identity set/order differs",
                )
                for row in frame["before_materialize"]:
                    expected = rows[row["request_id"]]
                    require(
                        row["context_sha256"] == expected["comparison_prefix_sha256"],
                        "captured context differs",
                    )
                    require(
                        row["valid_length"] == len(expected["committed_prefix"])
                        and row["suffix"] == [11],
                        "captured frontier/suffix differs",
                    )
                require(
                    frame["checks_passed"] is True and not frame["errors"],
                    "materialization diagnostic failed",
                )
                require(
                    report["runtime_provenance"]["batch_invariance"][
                        "batch_invariant_env_resolved"
                    ]
                    is True,
                    "batch-invariant flag not effective in worker",
                )
            checked[path] = report
        except (ValueError, KeyError, TypeError, OverflowError) as error:
            failures.append(f"{path}: {error}")
    if len(checked) == 4:
        first = checked[PATHS[0]]
        for path, report in checked.items():
            if report["run_identity"] != first["run_identity"]:
                failures.append(f"{path}: model/config/source execution identity differs")
            if report["gpu_identity"]["gpu_uuid"] != first["gpu_identity"]["gpu_uuid"]:
                failures.append(f"{path}: physical GPU UUID differs")
        if len({r["process_lifetime_id"] for r in checked.values()}) != 4:
            failures.append("four independent process lifetimes were not proved")
    pattern = [checked[p]["logits"]["top1_token"] for p in PATHS] if len(checked) == 4 else None
    cases = {
        (16, 227, 227, 227): ("A", "hf_vllm_execution_numerical_divergence"),
        (16, 16, 16, 227): ("B", "persistent_history_kv_rebase_difference"),
        (16, 16, 227, 227): ("C", "batch_shape_dependent_vllm_execution_difference"),
    }
    case, label = cases.get(tuple(pattern or ()), (None, "mixed_or_unreproduced"))
    if failures:
        case, label = None, "incomplete_or_invalid_evidence"
    deltas, followups = {}, []
    if not failures:
        for left, right in ((PATHS[0], PATHS[1]), (PATHS[1], PATHS[2]), (PATHS[2], PATHS[3])):
            a, b = checked[left]["logits"], checked[right]["logits"]
            vector_delta = [
                y - x for x, y in zip(a["raw_logits_float32"], b["raw_logits_float32"])
            ]
            deltas[f"{right}_minus_{left}"] = {
                "token_logits": {
                    t: b["token_logits"][t] - a["token_logits"][t] for t in ("16", "227")
                },
                "margin_16_minus_227": b["margin_16_minus_227"] - a["margin_16_minus_227"],
                "full_vector_equal": a["raw_logits_float32"] == b["raw_logits_float32"],
                "full_vector_max_abs_delta": max(map(abs, vector_delta)),
            }
        if not deltas[f"{PATHS[2]}_minus_{PATHS[1]}"]["full_vector_equal"]:
            followups.append("audit_pinned_batch_invariance_source_and_effective_attention_kernel")
        if pattern[2:] == [16, 227]:
            followups.append(
                "prepare_same_layer_token_kv_content_comparison_C_vs_D; do_not_fix_yet"
            )
        if pattern[0] != 16:
            followups.append(
                "compare_fresh_HF_with_original_HF_history_at_same_prefix_before_vLLM_overrides"
            )
        if pattern[3] != 227:
            followups.append(
                "repeat_D_only_with_original_process_setup; retain_unreproduced_capture"
            )
        if case is None:
            followups.append("review_exact_mixed_outcome; do_not_force_classification")
    return {
        "schema_version": "specrhythm.phase4b3-d3-logits-classification.v1",
        "diagnostic_only": True,
        "performance_result": False,
        "fixture_sha256": FIXTURE_SHA256,
        "source_diagnostic_sha256": fixture["source_diagnostic_sha256"],
        "probe_prefixes": fixture["rows"],
        "path_order": list(PATHS),
        "top1_pattern": pattern,
        "classification_case": case,
        "classification": label,
        "evidence_status": "invalid"
        if failures
        else "strong_evidence"
        if case
        else "inconclusive",
        "mechanistic_root_cause_proven": False,
        "errors": failures,
        "paths": {
            p: {
                **{k: v for k, v in r.items() if k not in ("logits", "diagnostic_frames")},
                "logits": without_vector(r["logits"]),
            }
            for p, r in checked.items()
        },
        "raw_logit_deltas": deltas,
        "batch_invariance_source_basis": {
            "source_pin": "752a3a504485790a2e8491cacbb35c137339ad34",
            "source_sha256": fixture["numerical_source_sha256"],
            "declared_attention_support": (
                "FlashAttentionBackend.supports_batch_invariance returns True"
            ),
            "sm80_linear_path": (
                "UnquantizedLinearMethod.apply -> linear_batch_invariant -> matmul_persistent"
            ),
            "attention_path": (
                "FlashAttentionImpl.forward -> flash_attn_varlen_func -> "
                "_vllm_fa2_C.varlen_fwd when actual FA version is 2"
            ),
            "attention_split_control": "VLLM_BATCH_INVARIANT sets max_num_splits=1",
            "expected_scope": (
                "within supported vLLM execution; no HF-vLLM bitwise equivalence guarantee"
            ),
            "limitation": (
                "source support declaration is not a measured guarantee for streaming rebase; "
                "inspect per-path actual implementations and raw B/C equality"
            ),
        },
        "persistent_failure_reproduced": pattern is not None and pattern[3] == 227,
        "diagnostic_perturbation_possible": pattern is not None and pattern[3] != 227,
        "required_followups": followups,
        "correctness_fix_authorized_by_result": False,
        "d4_d5_allowed": False,
    }
