"""Conservative tensor-parallel structural and engine compatibility checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

REQUIRED_MODEL_FIELDS = (
    "num_attention_heads",
    "num_key_value_heads",
    "hidden_size",
    "intermediate_size",
    "vocab_size",
    "model_type",
)

# Architectures listed by the pinned Transformers 4.56 inference documentation.
TRANSFORMERS_NATIVE_TP_MODEL_TYPES = {
    "cohere",
    "cohere2",
    "gemma",
    "gemma2",
    "glm",
    "granite",
    "llama",
    "mistral",
    "mixtral",
    "olmo",
    "olmo2",
    "phi",
    "phi3",
    "qwen2",
    "qwen2_moe",
    "qwen2_vl",
    "starcoder2",
}


def load_model_config(path: str) -> dict[str, Any]:
    source = Path(path)
    if source.is_dir():
        source = source / "config.json"
    if not source.is_file():
        raise ValueError(f"model config does not exist: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"model config is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("model config root must be an object")
    missing = [field for field in REQUIRED_MODEL_FIELDS if field not in value]
    if missing:
        raise ValueError(f"model config is missing required fields: {', '.join(missing)}")
    for field in REQUIRED_MODEL_FIELDS[:-1]:
        item = value[field]
        if not isinstance(item, int) or isinstance(item, bool) or item < 1:
            raise ValueError(f"model config {field} must be a positive integer")
    if not str(value["model_type"]).strip():
        raise ValueError("model config model_type must not be empty")
    try:
        from transformers import AutoConfig  # type: ignore[import-not-found]

        loaded = AutoConfig.from_pretrained(
            str(source.parent), local_files_only=True, trust_remote_code=False
        )
        plan = getattr(loaded, "base_model_tp_plan", None)
        if plan:
            value["base_model_tp_plan"] = plan
            value["tp_plan_source"] = "transformers-config-class"
    except ImportError:
        value["tp_plan_source"] = "pinned-static-model-type-list"
    except (OSError, ValueError) as error:
        value["tp_plan_source"] = "pinned-static-model-type-list"
        value["tp_plan_probe_error"] = str(error)
    return value


def validate_tp_compatibility(
    config: Mapping[str, Any], tp_sizes: Sequence[int] = (1, 2, 3, 4)
) -> dict[str, Any]:
    model_type = str(config["model_type"]).lower().replace("-", "_")
    dimensions = {field: config[field] for field in REQUIRED_MODEL_FIELDS}
    native_plan = bool(config.get("base_model_tp_plan")) or (
        model_type in TRANSFORMERS_NATIVE_TP_MODEL_TYPES
    )
    rows = []
    for tp_size in tp_sizes:
        if not isinstance(tp_size, int) or isinstance(tp_size, bool) or tp_size < 1:
            raise ValueError("TP sizes must be positive integers")
        reasons = []
        if tp_size > 1 and not native_plan:
            reasons.append(
                f"Transformers 4.56 has no declared native TP plan for model_type={model_type}"
            )
        for field in REQUIRED_MODEL_FIELDS[:-1]:
            if int(config[field]) % tp_size:
                reasons.append(f"{field}={config[field]} is not divisible by TP={tp_size}")
        rows.append(
            {
                "tp_size": tp_size,
                "supported": not reasons,
                "structurally_supported": not any(
                    "not divisible" in reason for reason in reasons
                ),
                "engine_native_tp_plan": tp_size == 1 or native_plan,
                "reason": "supported without model surgery" if not reasons else "; ".join(reasons),
            }
        )
    return {
        "schema_version": "specrhythm.tp-compatibility.v1",
        "engine": "transformers",
        "engine_version_constraint": ">=4.56.1,<4.57",
        "model": dimensions,
        "tp_plan_source": config.get(
            "tp_plan_source", "pinned-static-model-type-list"
        ),
        "tp_plan_probe_error": config.get("tp_plan_probe_error"),
        "results": rows,
        "model_surgery_allowed": False,
    }
