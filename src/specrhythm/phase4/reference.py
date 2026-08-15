"""Immutable stock-vLLM serving-correctness references for Phase 4."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from specrhythm.phase4.config import VLLM_COMMIT, VLLM_VERSION, Phase4Config
from specrhythm.phase4.manifest import model_revision_manifest, sha256_file
from specrhythm.phase4.stock_vllm import run_stock_smoke
from specrhythm.phase4.transport import payload_sha256

REFERENCE_SCHEMA = "specrhythm.phase4-stock-target-reference.v1"
VLLM_RUNNER_RELATIVE_PATH = Path("vllm/v1/worker/gpu_model_runner.py")
STOCK_VLLM_RUNNER_SHA256 = (
    "6c92ded8468f44d6df863a617ce588f132fa6df7031feecc0cc421702a41610e"
)


def output_token_hash(token_ids: Sequence[int]) -> str:
    payload = json.dumps(list(token_ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _exclusive_freeze(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True).encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def build_stock_reference(
    smoke: Mapping[str, Any],
    config: Phase4Config,
    *,
    workload_path: Path,
    git_commit: str,
    installed_runner_sha256: str = STOCK_VLLM_RUNNER_SHA256,
) -> dict[str, Any]:
    workload_rows = _load_jsonl(workload_path)
    runs = smoke.get("runs")
    if not isinstance(runs, list) or len(runs) != 2:
        raise ValueError("stock reference requires exactly two target-only runs")
    first, second = runs
    if not isinstance(first, list) or not isinstance(second, list):
        raise ValueError("stock target-only run output is invalid")
    if len(first) != len(second) or len(first) != len(workload_rows):
        raise ValueError("stock runs and frozen workload have different request counts")
    if smoke.get("role") != "target" or smoke.get("repeated_run_deterministic") is not True:
        raise ValueError("stock target-only reference is not deterministic")
    outputs = []
    for one, two, workload_row in zip(first, second, workload_rows):
        if one.get("request_id") != two.get("request_id"):
            raise ValueError("stock repeated runs have different request order")
        if one.get("request_id") != workload_row.get("request_id"):
            raise ValueError("stock output order does not match the frozen workload")
        tokens = list(one.get("generated_token_ids", ()))
        if tokens != list(two.get("generated_token_ids", ())):
            raise ValueError("stock repeated runs have different token IDs")
        if one.get("finish_reason") != two.get("finish_reason") or one.get(
            "stop_reason"
        ) != two.get("stop_reason"):
            raise ValueError("stock repeated runs have different termination")
        outputs.append(
            {
                "request_id": one["request_id"],
                "generated_token_ids": tokens,
                "generated_tokens": len(tokens),
                "finish_reason": one.get("finish_reason"),
                "stop_reason": one.get("stop_reason"),
                "eos_reached": one.get("finish_reason") == "stop",
                "output_sha256": output_token_hash(tokens),
                "top_logprobs": one.get("top_logprobs", []),
            }
        )
    value: dict[str, Any] = {
        "schema_version": REFERENCE_SCHEMA,
        "serving_correctness_reference": "stock-vllm-target-only",
        "created_before_serial": True,
        "immutable": True,
        "gpu_correctness_result": True,
        "gpu_performance_result": False,
        "reports_goodput": False,
        "reports_slo_attainment": False,
        "reports_speedup": False,
        "vllm": {
            "version": config.expected_vllm_version,
            "source_commit": config.expected_vllm_commit,
            "patched": False,
            "model_runner": config.target_model_runner,
            "gpu_model_runner_sha256": installed_runner_sha256,
        },
        "specrhythm_commit": git_commit,
        "model": model_revision_manifest(
            config.target.resolved_model_path, config.target.revision
        ),
        "tokenizer": model_revision_manifest(
            config.target.resolved_tokenizer_path, config.target.tokenizer_revision
        ),
        "workload": {
            "file": workload_path.name,
            "sha256": sha256_file(workload_path),
            "request_ids": [row["request_id"] for row in outputs],
            "prompt_token_ids": [
                list(row.get("prompt_token_ids", ())) for row in workload_rows
            ],
            "requests": [
                {
                    "request_id": row["request_id"],
                    "prompt_token_ids": list(row.get("prompt_token_ids", ())),
                    "maximum_new_tokens": int(row["maximum_new_tokens"]),
                    "sampling_seed": int(row["sampling_seed"]),
                }
                for row in workload_rows
            ],
        },
        "sampling_configuration": smoke.get("sampling"),
        "target_runtime_configuration": {
            "physical_gpu_ids": list(config.target.physical_gpu_ids),
            "tensor_parallel_size": config.target.tensor_parallel_size,
            "dtype": config.target.dtype,
            "max_model_len": config.max_model_len,
            "enforce_eager": config.enforce_eager,
            "enable_prefix_caching": config.enable_prefix_caching,
            "vllm_dbo_enabled": False,
            "built_in_speculative_decoding": False,
            "enable_thinking": config.enable_thinking,
        },
        "outputs": outputs,
        "repeated_run_determinism": True,
        "legacy_hf_trajectory": {
            "role": "provenance-and-divergence-diagnosis-only",
            "comparison": smoke.get("frozen_hf_target_comparison"),
            "can_fail_serving_correctness": False,
        },
        "artifact_sha256_definition": (
            "sha256 of canonical JSON payload before artifact_sha256 is inserted"
        ),
    }
    value["artifact_sha256"] = payload_sha256(value)
    return value


def freeze_stock_reference(
    output_path: Path,
    config: Phase4Config,
    *,
    workload_path: Path,
    environment_path: Path,
    topology_path: Path,
    runtime_manifest_path: Path,
    git_commit: str,
    legacy_hf_target_dir: Optional[Path] = None,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("stock target reference already exists and is immutable")
    if config.target_model_runner == "v1" and os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "0":
        raise RuntimeError(
            "Phase 4A.1 stock reference requires VLLM_USE_V2_MODEL_RUNNER=0"
        )
    installed_runner_sha256 = require_stock_vllm_runner()
    smoke = run_stock_smoke(
        config,
        role="target",
        workload_path=workload_path,
        environment_path=environment_path,
        topology_path=topology_path,
        runtime_manifest_path=runtime_manifest_path,
        git_commit=git_commit,
        frozen_target_dir=legacy_hf_target_dir,
    )
    reference = build_stock_reference(
        smoke,
        config,
        workload_path=workload_path,
        git_commit=git_commit,
        installed_runner_sha256=installed_runner_sha256,
    )
    _exclusive_freeze(output_path, reference)
    return reference


def validate_stock_reference(value: Mapping[str, Any]) -> list[str]:
    errors = []
    if value.get("schema_version") != REFERENCE_SCHEMA:
        errors.append("unsupported stock target reference schema")
    if value.get("serving_correctness_reference") != "stock-vllm-target-only":
        errors.append("serving correctness reference is not stock vLLM Target-only")
    vllm = value.get("vllm")
    if (
        not isinstance(vllm, Mapping)
        or vllm.get("version") != VLLM_VERSION
        or vllm.get("source_commit") != VLLM_COMMIT
        or vllm.get("patched") is not False
        or vllm.get("gpu_model_runner_sha256") != STOCK_VLLM_RUNNER_SHA256
    ):
        errors.append("stock reference does not prove an unmodified pinned vLLM runner")
    if not value.get("specrhythm_commit"):
        errors.append("stock reference SpecRhythm commit is missing")
    for key in ("model", "tokenizer", "sampling_configuration", "target_runtime_configuration"):
        if not isinstance(value.get(key), Mapping):
            errors.append(f"stock reference {key} is missing")
    if value.get("created_before_serial") is not True or value.get("immutable") is not True:
        errors.append("stock target reference is not frozen before Serial")
    if value.get("repeated_run_determinism") is not True:
        errors.append("stock target repeated run was not deterministic")
    expected = value.get("artifact_sha256")
    payload = dict(value)
    payload.pop("artifact_sha256", None)
    if expected != payload_sha256(payload):
        errors.append("stock target reference canonical payload checksum mismatch")
    outputs = value.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        errors.append("stock target reference outputs are missing")
        outputs = []
    request_ids = set()
    for row in outputs:
        if not isinstance(row, Mapping) or not row.get("request_id"):
            errors.append("stock target reference contains an invalid request")
            continue
        request_ids.add(row["request_id"])
        tokens = row.get("generated_token_ids")
        if not isinstance(tokens, list) or not tokens:
            errors.append("stock target reference output token IDs are missing")
        elif row.get("output_sha256") != output_token_hash(tokens):
            errors.append("stock target reference output checksum mismatch")
    if len(request_ids) != len(outputs):
        errors.append("stock target reference request IDs are not unique")
    workload = value.get("workload")
    requests = workload.get("requests") if isinstance(workload, Mapping) else None
    if not isinstance(requests, list) or len(requests) != len(outputs):
        errors.append("stock target reference frozen request inputs are incomplete")
    elif [row.get("request_id") for row in requests] != [
        row.get("request_id") for row in outputs
    ]:
        errors.append("stock target reference input/output request order differs")
    elif any(
        not isinstance(row.get("prompt_token_ids"), list)
        or not row["prompt_token_ids"]
        or not isinstance(row.get("maximum_new_tokens"), int)
        or row["maximum_new_tokens"] < 1
        or not isinstance(row.get("sampling_seed"), int)
        for row in requests
    ):
        errors.append("stock target reference frozen request fields are invalid")
    legacy = value.get("legacy_hf_trajectory")
    if not isinstance(legacy, Mapping) or legacy.get("can_fail_serving_correctness") is not False:
        errors.append("legacy HF trajectory is not explicitly advisory")
    return errors


def require_stock_vllm_runner() -> str:
    try:
        import vllm
    except ImportError as error:
        raise RuntimeError("vLLM is unavailable for stock runner verification") from error
    runner_path = Path(vllm.__file__).resolve().parents[1] / VLLM_RUNNER_RELATIVE_PATH
    actual = sha256_file(runner_path)
    if actual != STOCK_VLLM_RUNNER_SHA256:
        raise RuntimeError(
            "stock reference requires the unmodified pinned gpu_model_runner.py; "
            f"found SHA256 {actual}"
        )
    return actual


def compare_outputs_to_reference(
    outputs: Sequence[Mapping[str, Any]], reference: Mapping[str, Any]
) -> dict[str, Any]:
    reference_errors = validate_stock_reference(reference)
    expected = {
        str(row["request_id"]): row
        for row in reference.get("outputs", ())
        if isinstance(row, Mapping) and row.get("request_id")
    }
    comparisons = []
    all_equal = not reference_errors
    for output in outputs:
        request_id = str(output.get("request_id", ""))
        row = expected.get(request_id)
        actual = list(output.get("generated_token_ids", ()))
        wanted = list(row.get("generated_token_ids", ())) if row else []
        common = min(len(actual), len(wanted))
        divergence = next(
            (index for index in range(common) if actual[index] != wanted[index]), None
        )
        if divergence is None and len(actual) != len(wanted):
            divergence = common
        termination_equal = (
            bool(row)
            and output.get("finish_reason") == row.get("finish_reason")
            and output.get("stop_reason") == row.get("stop_reason")
        )
        equal = divergence is None and termination_equal
        all_equal = all_equal and equal
        comparisons.append(
            {
                "request_id": request_id,
                "equal": equal,
                "first_divergence_position": divergence,
                "stock_token_id": (
                    wanted[divergence]
                    if divergence is not None and divergence < len(wanted)
                    else None
                ),
                "actual_token_id": (
                    actual[divergence]
                    if divergence is not None and divergence < len(actual)
                    else None
                ),
                "termination_equal": termination_equal,
                "actual_topk_at_divergence": (
                    output.get("top_logprobs", [])[divergence]
                    if divergence is not None and divergence < len(output.get("top_logprobs", []))
                    else None
                ),
            }
        )
    if set(expected) != {str(row.get("request_id", "")) for row in outputs}:
        all_equal = False
    return {
        "serving_correctness_reference": "stock-vllm-target-only",
        "all_sequences_equal": all_equal,
        "reference_errors": reference_errors,
        "requests": comparisons,
    }


def build_target_regression(
    smoke: Mapping[str, Any], reference: Mapping[str, Any], *, patch_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    runs = smoke.get("runs", ())
    outputs = runs[0] if isinstance(runs, list) and runs else []
    comparison = compare_outputs_to_reference(outputs, reference)
    return {
        "schema_version": "specrhythm.phase4-patched-target-regression.v1",
        "patched_target_only": True,
        "speculative_decoding_enabled": False,
        "comparison": comparison,
        "repeated_run_deterministic": smoke.get("repeated_run_deterministic") is True,
        "valid": comparison["all_sequences_equal"]
        and smoke.get("repeated_run_deterministic") is True,
        "patch_manifest_sha256": payload_sha256(patch_manifest),
        "gpu_correctness_result": True,
        "gpu_performance_result": False,
        "reports_goodput": False,
        "reports_slo_attainment": False,
        "reports_speedup": False,
        "smoke": smoke,
    }


def load_reference(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("stock target reference root must be an object")
    errors = validate_stock_reference(value)
    if errors:
        raise ValueError("invalid stock target reference: " + "; ".join(errors))
    return value


def reference_file_evidence(path: Path) -> dict[str, Any]:
    return {
        "file": path.name,
        "file_sha256": sha256_file(path),
        "writable": os.access(path, os.W_OK),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("workload JSONL row must be an object")
                rows.append(value)
    return rows
