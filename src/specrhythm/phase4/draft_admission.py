"""D3 device binding and cross-run regime admission; no model execution changes."""

from __future__ import annotations

import re
import sys

GPU_MODEL = "NVIDIA A800-SXM4-80GB"
MRV1 = "vllm.v1.worker.gpu_model_runner.GPUModelRunner"
FLASH_IMPL = "vllm.v1.attention.backends.flash_attn.FlashAttentionImpl"
RUNTIME_FIELDS = (
    "gpu_name",
    "dtype",
    "world_size",
    "global_rank",
    "tensor_parallel_size",
    "runner_class",
    "enforce_eager",
    "prefix_caching",
    "cache_initialized",
    "model_instance_count",
    "block_size",
    "kv_cache_group_count",
    "max_num_seqs",
    "max_num_batched_tokens",
    "batch_invariance",
    "attention_implementations",
    "model",
    "tokenizer",
)


def runtime_regime(runtime):
    """Omit physical UUID, allocator capacity and startup timings, never hashes."""
    value = {key: runtime.get(key) for key in RUNTIME_FIELDS}
    if isinstance(value["attention_implementations"], list):
        value["attention_implementations"] = sorted(
            value["attention_implementations"], key=lambda row: row["layer"]
        )
    return value


def runtime_errors(runtime):
    errors = []
    for key, expected in (
        ("gpu_name", GPU_MODEL),
        ("dtype", "bfloat16"),
        ("world_size", 1),
        ("global_rank", 0),
        ("tensor_parallel_size", 1),
        ("runner_class", MRV1),
        ("enforce_eager", True),
        ("prefix_caching", False),
        ("cache_initialized", True),
        ("model_instance_count", 1),
    ):
        if type(runtime.get(key)) is not type(expected) or runtime[key] != expected:
            errors.append(f"unsupported Draft execution property: {key}")
    batch = runtime.get("batch_invariance") or {}
    for key, expected in (
        ("compute_capability", "8.0"),
        ("dtype", "torch.bfloat16"),
        ("batch_invariant_env_raw", "1"),
        ("batch_invariant_requested", True),
        ("batch_invariant_env_resolved", True),
        ("batch_invariant_effective", True),
        ("cascade_attention_enabled", False),
        ("vllm_dbo_enabled", False),
    ):
        if type(batch.get(key)) is not type(expected) or batch[key] != expected:
            errors.append(f"unsupported Draft batch-invariant property: {key}")
    if batch.get("batch_invariant_validation", {}).get("valid") is not True:
        errors.append("effective batch-invariant validation failed")
    attention = runtime.get("attention_implementations")
    if not attention or any(
        row.get("implementation") != FLASH_IMPL
        or row.get("flash_attention_version") != 2
        or row.get("batch_invariant_enabled") is not True
        for row in attention
    ):
        errors.append("unsupported or missing effective FlashAttention implementation/version")
    return errors


def collect_runtime(backend):
    """Read initialized worker metadata once; no forward or diagnostic hook."""
    from specrhythm.phase4.batch_invariant import worker_batch_invariant_evidence

    worker = backend.worker
    context = worker.vllm_config.compilation_config.static_forward_context
    return {
        **backend.provenance,
        "python_major_minor": list(sys.version_info[:2]),
        "python_version": sys.version.split()[0],
        "batch_invariance": worker_batch_invariant_evidence(worker.executor.driver_worker.worker),
        "attention_implementations": [
            {
                "layer": name,
                "implementation": f"{type(layer.impl).__module__}.{type(layer.impl).__name__}",
                "flash_attention_version": getattr(layer.impl, "vllm_flash_attn_version", None),
                "batch_invariant_enabled": getattr(layer.impl, "batch_invariant_enabled", None),
            }
            for name, layer in context.items()
            if hasattr(layer, "impl")
        ],
    }


def admit_device(current, identity, regime):
    from specrhythm.phase4.draft_qualification import compatible_identity

    binding_errors = []
    for key, expected in (
        ("cuda_visible_devices", "0"),
        ("logical_cuda_index", 0),
        ("physical_gpu_id", 0),
        ("gpu_name", GPU_MODEL),
        ("world_size", 1),
        ("tensor_parallel_size", 1),
        ("global_rank", 0),
        ("startup_uuid_validation_count", 1),
    ):
        if type(current.get(key)) is not type(expected) or current[key] != expected:
            binding_errors.append(f"current device binding invalid: {key}")
    uuid = current.get("gpu_uuid")
    if (
        not isinstance(uuid, str)
        or re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", uuid) is None
    ):
        binding_errors.append("current device binding requires a real syntactically valid UUID")
    historical = regime.get("gpu_identity", {})
    retained = regime.get("vllm_runtime_regime", {})
    errors = binding_errors + runtime_errors(current)
    if not compatible_identity(identity, regime.get("run_identity", {})):
        errors.append("pinned model/config/version/source/hash execution identity differs")
    if current.get("vllm_api") != identity.get("vllm_api"):
        errors.append("current worker API source differs from preflight")
    if current.get("python_major_minor") != [3, 11]:
        errors.append("current Python differs from the pinned probe Python 3.11 contract")
    if runtime_regime(current) != retained:
        errors.append("current execution regime differs from retained runtime properties")
    device_compatible = (
        current.get("gpu_name") == historical.get("gpu_name") == GPU_MODEL
        and current.get("batch_invariance", {}).get("compute_capability")
        == retained.get("batch_invariance", {}).get("compute_capability")
        == "8.0"
    )
    if not device_compatible:
        errors.append("current GPU model/compute capability is incompatible with retained probe")
    if regime.get("valid") is not True or regime.get("errors") != []:
        errors.append("retained execution-regime qualification is invalid")
    return {
        "schema_version": "specrhythm.phase4b3-draft-admission.v1",
        "valid": not errors,
        "errors": errors,
        "current_device_binding_valid": not binding_errors,
        "historical_probe_gpu_uuid": historical.get("gpu_uuid"),
        "current_gpu_uuid": uuid,
        "same_physical_gpu": uuid == historical.get("gpu_uuid") if uuid else None,
        "execution_regime_device_compatible": device_compatible,
        "execution_regime_compatible": not errors,
        "current_runtime_provenance": current,
        "python_compatibility_basis": (
            "exact Python 3.11 contract enforced by original probe preflight; "
            "historical patch version was not captured"
        ),
    }
