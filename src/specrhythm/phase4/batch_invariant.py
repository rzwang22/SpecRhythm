"""Auditable Phase-4 batch-invariant correctness configuration.

This module deliberately separates a requested environment setting from an
effective, evidence-backed runtime mode.  The pinned vLLM v0.25.1 source says
that ``VLLM_BATCH_INVARIANT`` requires NVIDIA compute capability >= 8.0, but
hardware support alone is not proof that an initialized worker is effective.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Mapping, MutableMapping, Optional, Sequence

CORRECTNESS_MODES = ("default", "batch-invariant")
BATCH_INVARIANT_ENV = "VLLM_BATCH_INVARIANT"
PINNED_VLLM_MIN_COMPUTE_CAPABILITY = (8, 0)
PINNED_VLLM_HARDWARE_CONTRACT = "NVIDIA compute capability >= 8.0"


def normalize_correctness_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in CORRECTNESS_MODES:
        raise ValueError(
            "correctness mode must be one of " + ", ".join(CORRECTNESS_MODES)
        )
    return mode


def requested_for_mode(mode: str) -> bool:
    return normalize_correctness_mode(mode) == "batch-invariant"


def configure_before_worker_creation(
    mode: str,
    *,
    environ: Optional[MutableMapping[str, str]] = None,
    loaded_modules: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Set and prove the vLLM flag before any vLLM module is imported.

    An explicit ``0`` is used for default-mode runs so A/B and C/D manifests
    cannot depend on an inherited shell value.
    """

    normalized = normalize_correctness_mode(mode)
    environment = os.environ if environ is None else environ
    modules = tuple(sys.modules) if loaded_modules is None else tuple(loaded_modules)
    imported = sorted(
        name for name in modules if name == "vllm" or name.startswith("vllm.")
    )
    if imported:
        raise RuntimeError(
            "correctness mode must be configured before importing vLLM; already loaded: "
            + ", ".join(imported[:5])
        )
    expected = "1" if normalized == "batch-invariant" else "0"
    inherited = environment.get(BATCH_INVARIANT_ENV)
    if inherited not in (None, expected):
        raise RuntimeError(
            f"{BATCH_INVARIANT_ENV}={inherited!r} conflicts with --correctness-mode "
            f"{normalized}"
        )
    environment[BATCH_INVARIANT_ENV] = expected
    return {
        "correctness_mode": normalized,
        "batch_invariant_requested": normalized == "batch-invariant",
        "environment_variable": BATCH_INVARIANT_ENV,
        "environment_value": expected,
        "configured_before_vllm_import": True,
        "inherited_environment_value": inherited,
    }


def parse_compute_capability(value: Any) -> Optional[tuple[int, int]]:
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        pieces = value.split(".", 1)
        if len(pieces) == 2:
            try:
                return int(pieces[0]), int(pieces[1])
            except ValueError:
                return None
    return None


def pinned_vllm_hardware_supported(value: Any) -> bool:
    """Return whether ``value`` satisfies the exact pinned vLLM contract."""

    capability = parse_compute_capability(value)
    return bool(
        capability is not None
        and capability >= PINNED_VLLM_MIN_COMPUTE_CAPABILITY
    )


def worker_batch_invariant_evidence(worker: Any) -> dict[str, Any]:
    """Collect the effective configuration inside one initialized TP worker."""

    import torch
    import vllm.envs as vllm_envs
    from vllm.distributed.parallel_state import get_tp_group

    requested = os.environ.get(BATCH_INVARIANT_ENV) == "1"
    resolved = bool(vllm_envs.VLLM_BATCH_INVARIANT)
    parallel = worker.vllm_config.parallel_config
    capability = tuple(int(item) for item in torch.cuda.get_device_capability(worker.device))
    attention_rows = []
    for groups in getattr(worker.model_runner, "attn_groups", ()):
        for group in groups:
            backend = getattr(group, "backend", None)
            if backend is None:
                continue
            get_name = getattr(backend, "get_name", None)
            name = str(get_name()) if callable(get_name) else str(backend.__name__)
            supports = getattr(backend, "supports_batch_invariance", None)
            attention_rows.append(
                {
                    "backend": name,
                    "supports_batch_invariance": (
                        bool(supports()) if callable(supports) else None
                    ),
                }
            )
    communicator = getattr(get_tp_group(), "device_communicator", None)
    all_reduce_backends = []
    for attribute, name in (
        ("qr_comm", "QUICK_REDUCE"),
        ("fi_ar_comm", "FLASHINFER"),
        ("aiter_ar_comm", "AITER_CUSTOM"),
        ("ca_comm", "VLLM_CUSTOM"),
        ("symm_mem_comm", "SYMM_MEM"),
        ("pynccl_comm", "PYNCCL"),
    ):
        candidate = getattr(communicator, attribute, None)
        if candidate is not None and not bool(getattr(candidate, "disabled", False)):
            all_reduce_backends.append(name)
    if not all_reduce_backends:
        all_reduce_backends.append("TORCH_DISTRIBUTED_FALLBACK")
    if resolved:
        pynccl = getattr(communicator, "pynccl_comm", None)
        all_reduce_backends = [
            "PYNCCL"
            if pynccl is not None and not bool(getattr(pynccl, "disabled", False))
            else "TORCH_DISTRIBUTED_FALLBACK"
        ]
    documented_hardware_supported = pinned_vllm_hardware_supported(capability)
    attention_supported = bool(attention_rows) and all(
        row["supports_batch_invariance"] is True for row in attention_rows
    )
    custom_disabled = bool(parallel.disable_custom_all_reduce)
    cascade_enabled = bool(getattr(worker.model_runner, "cascade_attn_enabled", False))
    dbo_enabled = bool(parallel.use_ubatching)
    effective = bool(
        requested
        and resolved
        and documented_hardware_supported
        and attention_supported
        and custom_disabled
        and not cascade_enabled
        and not dbo_enabled
    )
    reasons = []
    if requested and not resolved:
        reasons.append("worker did not resolve VLLM_BATCH_INVARIANT=true")
    if requested and not documented_hardware_supported:
        reasons.append(
            "pinned vLLM documents batch invariance only for compute capability >= 8.0"
        )
    if requested and not attention_supported:
        reasons.append("active attention backend did not prove batch-invariance support")
    if requested and not custom_disabled:
        reasons.append("custom all-reduce was not disabled")
    if requested and cascade_enabled:
        reasons.append("cascade attention remained enabled")
    if requested and dbo_enabled:
        reasons.append("vLLM dual-batch overlap remained enabled")
    return {
        "batch_invariant_env_raw": os.environ.get(BATCH_INVARIANT_ENV),
        "batch_invariant_requested": requested,
        "batch_invariant_env_resolved": resolved,
        "batch_invariant_effective": effective,
        "batch_invariant_validation": {
            "valid": effective if requested else not resolved,
            "reasons": reasons,
        },
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "pinned_vllm_min_compute_capability": "8.0",
        "documented_hardware_supported": documented_hardware_supported,
        "disable_custom_all_reduce": custom_disabled,
        "all_reduce_backends": all_reduce_backends,
        "attention_batch_invariance": attention_rows,
        "cascade_attention_enabled": cascade_enabled,
        "vllm_dbo_enabled": dbo_enabled,
        "dtype": str(worker.vllm_config.model_config.dtype),
    }


def validate_batch_invariant_ranks(
    rows: Sequence[Mapping[str, Any]], *, requested: bool
) -> dict[str, Any]:
    errors = []
    if not rows:
        errors.append("no TP worker evidence was collected")
    fields = (
        "batch_invariant_env_raw",
        "batch_invariant_requested",
        "batch_invariant_env_resolved",
        "disable_custom_all_reduce",
        "all_reduce_backends",
        "attention_batch_invariance",
        "cascade_attention_enabled",
        "vllm_dbo_enabled",
        "dtype",
    )
    for field in fields:
        values = {repr(row.get(field)) for row in rows}
        if len(values) > 1:
            errors.append(f"TP ranks disagree on {field}")
    expected_raw = "1" if requested else "0"
    for row in rows:
        rank = row.get("global_rank")
        if row.get("batch_invariant_env_raw") != expected_raw:
            errors.append(f"rank {rank} did not inherit {BATCH_INVARIANT_ENV}={expected_raw}")
        if row.get("batch_invariant_requested") is not requested:
            errors.append(f"rank {rank} reports the wrong requested mode")
        if requested and row.get("batch_invariant_env_resolved") is not True:
            errors.append(f"rank {rank} did not resolve VLLM_BATCH_INVARIANT=true")
        if requested and row.get("documented_hardware_supported") is not True:
            errors.append(f"rank {rank} does not satisfy the pinned hardware contract")
        if requested and row.get("disable_custom_all_reduce") is not True:
            errors.append(f"rank {rank} did not prove custom all-reduce is disabled")
        attention_rows = row.get("attention_batch_invariance")
        attention_rows = attention_rows if isinstance(attention_rows, list) else []
        if requested and (
            not attention_rows
            or any(
                not isinstance(attention, Mapping)
                or attention.get("supports_batch_invariance") is not True
                for attention in attention_rows
            )
        ):
            errors.append(
                f"rank {rank} did not prove every active attention backend supports "
                "batch invariance"
            )
        if requested and row.get("batch_invariant_effective") is not True:
            reasons = row.get("batch_invariant_validation", {}).get("reasons", ())
            errors.append(f"rank {rank} cannot prove effective batch invariance: {list(reasons)}")
        if requested and row.get("cascade_attention_enabled") is not False:
            errors.append(f"rank {rank} did not prove cascade attention is disabled")
        if requested and row.get("vllm_dbo_enabled") is not False:
            errors.append(f"rank {rank} did not prove vLLM DBO is disabled")
        if not requested and row.get("batch_invariant_env_resolved") is not False:
            errors.append(f"rank {rank} resolved batch invariance in a default-mode run")
    effective = bool(requested and rows and not errors)
    return {
        "batch_invariant_requested": requested,
        "batch_invariant_effective": effective,
        "batch_invariant_validation": {
            "valid": not errors,
            "errors": errors,
            "rank_count": len(rows),
            "fail_closed": True,
        },
    }


def reference_correctness_mode(reference: Mapping[str, Any]) -> str:
    runtime = reference.get("target_runtime_configuration")
    runtime = runtime if isinstance(runtime, Mapping) else {}
    mode = runtime.get("correctness_mode")
    if mode is None:
        return "default"
    return normalize_correctness_mode(str(mode))


def require_matching_reference_mode(reference: Mapping[str, Any], mode: str) -> None:
    expected = normalize_correctness_mode(mode)
    actual = reference_correctness_mode(reference)
    if actual != expected:
        raise ValueError(
            f"reference correctness mode {actual} does not match requested mode {expected}"
        )


def probe_batch_invariant_hardware(mode: str) -> dict[str, Any]:
    """Read-only pre-worker hardware gate; no model or vLLM worker is created."""

    setup = configure_before_worker_creation(mode)
    requested = requested_for_mode(mode)
    try:
        import torch
    except ImportError:
        return {
            "schema_version": "specrhythm.phase4-batch-invariant-preflight.v1",
            **setup,
            "cuda_available": False,
            "devices": [],
            "pinned_vllm_hardware_contract": PINNED_VLLM_HARDWARE_CONTRACT,
            "batch_invariant_effective": False,
            "effective_requires_initialized_worker_evidence": True,
            "valid": False,
            "errors": ["PyTorch is unavailable"],
        }
    available = bool(torch.cuda.is_available())
    devices = []
    if available:
        for logical_index in range(torch.cuda.device_count()):
            capability = tuple(
                int(item) for item in torch.cuda.get_device_capability(logical_index)
            )
            devices.append(
                {
                    "logical_cuda_index": logical_index,
                    "name": torch.cuda.get_device_name(logical_index),
                    "compute_capability": f"{capability[0]}.{capability[1]}",
                    "documented_hardware_supported": (
                        pinned_vllm_hardware_supported(capability)
                    ),
                }
            )
    errors = []
    if not available:
        errors.append("CUDA is unavailable")
    if requested and any(
        row["documented_hardware_supported"] is not True for row in devices
    ):
        errors.append(
            "pinned vLLM v0.25.1 documents VLLM_BATCH_INVARIANT only for NVIDIA "
            "compute capability >= 8.0"
        )
    valid = available and (not requested or bool(devices) and not errors)
    return {
        "schema_version": "specrhythm.phase4-batch-invariant-preflight.v1",
        **setup,
        "cuda_available": available,
        "devices": devices,
        "pinned_vllm_hardware_contract": PINNED_VLLM_HARDWARE_CONTRACT,
        "batch_invariant_effective": False,
        "effective_requires_initialized_worker_evidence": True,
        "valid": valid,
        "errors": errors,
    }
