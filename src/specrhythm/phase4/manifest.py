"""Phase-4 environment, topology, and runtime provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from ctypes import CDLL, byref, c_int
from ctypes.util import find_library
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from specrhythm.phase4.batch_invariant import PINNED_VLLM_HARDWARE_CONTRACT
from specrhythm.phase4.config import Phase4Config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _run(command: Sequence[str], *, cwd: Optional[Path] = None) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            list(command), cwd=cwd, capture_output=True, text=True, check=False
        )
    except OSError as error:
        return 127, "", str(error)
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def _distribution_provenance(name: str) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False, "version": None}
    files = distribution.files or ()
    record = next((item for item in files if str(item).endswith(".dist-info/RECORD")), None)
    direct = next(
        (item for item in files if str(item).endswith(".dist-info/direct_url.json")), None
    )
    record_path = Path(distribution.locate_file(record)) if record is not None else None
    direct_path = Path(distribution.locate_file(direct)) if direct is not None else None
    direct_value = None
    if direct_path is not None and direct_path.exists():
        try:
            direct_value = json.loads(direct_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            direct_value = {"error": "unreadable direct_url.json"}
    return {
        "installed": True,
        "version": distribution.version,
        "distribution_root": str(Path(distribution.locate_file(".")).resolve()),
        "record_sha256": (
            sha256_file(record_path) if record_path is not None and record_path.exists() else None
        ),
        "direct_url": direct_value,
    }


def _source_provenance(source: Path) -> dict[str, Any]:
    code, commit, error = _run(("git", "rev-parse", "HEAD"), cwd=source)
    tag_code, tag, tag_error = _run(
        ("git", "describe", "--tags", "--exact-match", "HEAD"), cwd=source
    )
    status_code, status, status_error = _run(
        ("git", "status", "--porcelain", "--untracked-files=no"), cwd=source
    )
    return {
        "path": str(source.resolve()),
        "commit": commit if code == 0 else None,
        "commit_error": error if code != 0 else None,
        "exact_tag": tag if tag_code == 0 else None,
        "tag_error": tag_error if tag_code != 0 else None,
        "tracked_tree_clean": status_code == 0 and not status,
        "status_error": status_error if status_code != 0 else None,
    }


def collect_environment(vllm_source: Path) -> dict[str, Any]:
    errors = []
    torch_version = None
    torch_cuda_build = None
    torch_cuda_available = False
    nccl_version = None
    cuda_runtime_version = None
    cuda_runtime_error = None
    try:
        import torch

        torch_version = torch.__version__
        torch_cuda_build = torch.version.cuda
        torch_cuda_available = bool(torch.cuda.is_available())
        if torch_cuda_available:
            nccl = torch.cuda.nccl.version()
            if isinstance(nccl, tuple):
                nccl_version = ".".join(str(part) for part in nccl)
            elif nccl is not None:
                nccl_version = str(nccl)
    except (AttributeError, ImportError, RuntimeError) as error:
        errors.append(f"PyTorch probe failed: {error}")
    library = find_library("cudart")
    if library:
        try:
            runtime = c_int()
            status = CDLL(library).cudaRuntimeGetVersion(byref(runtime))
            if status == 0:
                value = runtime.value
                cuda_runtime_version = f"{value // 1000}.{(value % 1000) // 10}"
            else:
                cuda_runtime_error = f"cudaRuntimeGetVersion returned {status}"
        except OSError as error:
            cuda_runtime_error = str(error)
    else:
        cuda_runtime_error = "libcudart not found by the dynamic loader"
    vllm = _distribution_provenance("vllm")
    transformers = _distribution_provenance("transformers")
    source = _source_provenance(vllm_source)
    if not torch_cuda_available:
        errors.append("CUDA is unavailable; Phase-4 stock-engine GPU bring-up cannot run")
    report = {
        "schema_version": "specrhythm.phase4-environment.v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "pytorch_version": torch_version,
        "pytorch_cuda_build": torch_cuda_build,
        "cuda_runtime_version": cuda_runtime_version,
        "cuda_runtime_query_error": cuda_runtime_error,
        "torch_cuda_available": torch_cuda_available,
        "nccl_version": nccl_version,
        "vllm": vllm,
        "vllm_source": source,
        "transformers": transformers,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "attention_backend_env": os.environ.get("VLLM_ATTENTION_BACKEND"),
        "batch_invariant_env": os.environ.get("VLLM_BATCH_INVARIANT"),
        "errors": errors,
    }
    report["available"] = not errors
    return report


def validate_environment(report: Mapping[str, Any], config: Phase4Config) -> dict[str, Any]:
    errors = []
    if report.get("schema_version") != "specrhythm.phase4-environment.v1":
        errors.append("unsupported environment schema")
    python_version = str(report.get("python_version", ""))
    if not python_version.startswith(config.expected_python_series + "."):
        errors.append(f"Python {config.expected_python_series}.x is required")
    torch_version = str(report.get("pytorch_version", "")).split("+")[0]
    if torch_version != config.expected_pytorch_version:
        errors.append(f"PyTorch {config.expected_pytorch_version} is required")
    vllm = report.get("vllm")
    if not isinstance(vllm, Mapping) or vllm.get("version") != config.expected_vllm_version:
        errors.append(f"vLLM {config.expected_vllm_version} is required")
    source = report.get("vllm_source")
    if not isinstance(source, Mapping) or source.get("commit") != config.expected_vllm_commit:
        errors.append(f"vLLM source commit must be {config.expected_vllm_commit}")
    if not isinstance(source, Mapping) or source.get("exact_tag") != (
        f"v{config.expected_vllm_version}"
    ):
        errors.append("vLLM source checkout must be at the exact frozen tag")
    if not isinstance(source, Mapping) or source.get("tracked_tree_clean") is not True:
        errors.append("vLLM source checkout has tracked modifications")
    if not report.get("torch_cuda_available"):
        errors.append("CUDA is unavailable")
    if not isinstance(vllm, Mapping) or not vllm.get("record_sha256"):
        errors.append("installed vLLM distribution RECORD checksum is missing")
    transformers = report.get("transformers")
    if not isinstance(transformers, Mapping) or not transformers.get("version"):
        errors.append("Transformers version is missing")
    if not report.get("nccl_version"):
        errors.append("NCCL version is missing")
    return {
        "schema_version": "specrhythm.phase4-environment-validation.v1",
        "valid": not errors,
        "errors": errors,
    }


def collect_topology() -> dict[str, Any]:
    def optional_float(value: str) -> Optional[float]:
        normalized = value.strip().lower()
        if "n/a" in normalized or "not supported" in normalized or not normalized:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    query = (
        "index,name,uuid,memory.total,driver_version,temperature.gpu,power.draw,pstate,"
        "clocks.sm,clocks.mem"
    )
    code, output, error = _run(
        (
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        )
    )
    gpus = []
    errors = []
    if code != 0:
        errors.append(f"nvidia-smi GPU query failed: {error or output}")
    else:
        for line in output.splitlines():
            fields = [item.strip() for item in line.split(",")]
            if len(fields) != 10:
                errors.append(f"unexpected nvidia-smi row: {line}")
                continue
            gpus.append(
                {
                    "physical_gpu_id": int(fields[0]),
                    "name": fields[1],
                    "uuid": fields[2],
                    "memory_total_mib": int(fields[3]),
                    "driver_version": fields[4],
                    "temperature_c": optional_float(fields[5]),
                    "power_draw_w": optional_float(fields[6]),
                    "pstate": fields[7],
                    "sm_clock_mhz": optional_float(fields[8]),
                    "memory_clock_mhz": optional_float(fields[9]),
                }
            )
    topo_code, topo, topo_error = _run(("nvidia-smi", "topo", "-m"))
    if topo_code != 0:
        errors.append(f"nvidia-smi topology query failed: {topo_error or topo}")
    return {
        "schema_version": "specrhythm.phase4-topology.v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": gpus,
        "nvlink_pcie_topology": topo if topo_code == 0 else None,
        "errors": errors,
        "available": bool(gpus) and not errors,
    }


def validate_topology(report: Mapping[str, Any], config: Phase4Config) -> dict[str, Any]:
    errors = []
    rows = report.get("gpus")
    rows = rows if isinstance(rows, list) else []
    found = {
        row.get("physical_gpu_id")
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("physical_gpu_id"), int)
    }
    draft = set(config.draft.physical_gpu_ids)
    target = set(config.target.physical_gpu_ids)
    if draft & target:
        errors.append("draft and target GPU sets overlap")
    missing = sorted((draft | target) - found)
    if missing:
        errors.append(f"configured physical GPUs are missing: {missing}")
    uuids = [
        row.get("uuid")
        for row in rows
        if isinstance(row, Mapping) and row.get("physical_gpu_id") in draft | target
    ]
    if len(uuids) != len(set(uuids)):
        errors.append("configured physical GPUs do not have unique UUIDs")
    if len(config.target.physical_gpu_ids) != config.target.tensor_parallel_size:
        errors.append("target GPU count does not equal target TP world size")
    return {
        "schema_version": "specrhythm.phase4-topology-validation.v1",
        "valid": not errors,
        "errors": errors,
        "draft_physical_gpu_ids": list(config.draft.physical_gpu_ids),
        "target_physical_gpu_ids": list(config.target.physical_gpu_ids),
    }


def model_revision_manifest(
    model_path: Path, configured_revision: Optional[str]
) -> dict[str, Any]:
    files = (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    )
    checksums = {
        name: sha256_file(model_path / name) for name in files if (model_path / name).is_file()
    }
    if not checksums:
        raise ValueError(f"model/tokenizer metadata is missing under {model_path}")
    digest = hashlib.sha256(
        json.dumps(checksums, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "path": str(model_path),
        "configured_revision": configured_revision,
        "resolved_revision": configured_revision or f"local-metadata-sha256:{digest}",
        "metadata_file_sha256": checksums,
    }


def build_runtime_manifest(
    config: Phase4Config,
    *,
    role: str,
    git_commit: str,
    workload_path: Path,
    environment_path: Path,
    topology_path: Path,
    worker_ranks: Sequence[Mapping[str, Any]],
    attention_backend: Optional[str],
    correctness_mode: str = "default",
    mode_setup: Optional[Mapping[str, Any]] = None,
    batch_invariant_validation: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    engine = config.draft if role == "draft" else config.target
    requested = correctness_mode == "batch-invariant"
    validation = (
        dict(batch_invariant_validation)
        if batch_invariant_validation is not None
        else {
            "batch_invariant_requested": requested,
            "batch_invariant_effective": False,
            "batch_invariant_validation": {
                "valid": not requested,
                "errors": (
                    [] if not requested else ["worker evidence was not supplied"]
                ),
                "fail_closed": True,
            },
        }
    )
    all_reduce_backends = sorted(
        {
            str(backend)
            for row in worker_ranks
            for backend in row.get("all_reduce_backends", ())
        }
    )
    return {
        "schema_version": "specrhythm.phase4-runtime-manifest.v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "stage": "phase4a-stock-vllm-bringup",
        "result_kind": "engine-bringup",
        "gpu_result": True,
        "serving_performance_result": False,
        "role": role,
        "framework": {
            "name": "vllm",
            "version": config.expected_vllm_version,
            "source_commit": config.expected_vllm_commit,
            "model_runner": config.target_model_runner if role == "target" else None,
            "built_in_speculative_decoding": False,
            "vllm_dbo_enabled": False,
            "specrhythm_dual_batch_implemented": False,
            "attention_backend": attention_backend,
            "all_reduce_backends": all_reduce_backends,
        },
        "correctness": {
            "mode": correctness_mode,
            "VLLM_BATCH_INVARIANT": "1" if requested else "0",
            "batch_invariant_requested": validation.get(
                "batch_invariant_requested", requested
            ),
            "batch_invariant_effective": validation.get(
                "batch_invariant_effective", False
            ),
            "batch_invariant_validation": validation.get(
                "batch_invariant_validation"
            ),
            "configured_before_worker_creation": bool(
                mode_setup
                and mode_setup.get("configured_before_vllm_import") is True
            ),
            "pinned_vllm_hardware_contract": PINNED_VLLM_HARDWARE_CONTRACT,
            "deterministic_sampling": config.sampling.do_sample is False,
            "worker_configuration_evidence": [
                {
                    key: row.get(key)
                    for key in (
                        "global_rank",
                        "batch_invariant_env_raw",
                        "batch_invariant_env_resolved",
                        "batch_invariant_effective",
                        "compute_capability",
                        "documented_hardware_supported",
                        "disable_custom_all_reduce",
                        "all_reduce_backends",
                        "attention_batch_invariance",
                        "cascade_attention_enabled",
                        "vllm_dbo_enabled",
                        "dtype",
                    )
                }
                for row in worker_ranks
            ],
        },
        "engine": {
            "physical_gpu_ids": list(engine.physical_gpu_ids),
            "tensor_parallel_size": engine.tensor_parallel_size,
            "dtype": engine.dtype,
            "enforce_eager": config.enforce_eager,
            "enable_prefix_caching": config.enable_prefix_caching,
            "model": model_revision_manifest(engine.resolved_model_path, engine.revision),
            "tokenizer": model_revision_manifest(
                engine.resolved_tokenizer_path, engine.tokenizer_revision
            ),
        },
        "sampling": config.sampling.to_dict(),
        "worker_ranks": [dict(row) for row in worker_ranks],
        "inputs": {
            "config_file": config.path.name,
            "config_sha256": sha256_file(config.path),
            "workload_file": workload_path.name,
            "workload_sha256": sha256_file(workload_path),
            "environment_file": environment_path.name,
            "environment_sha256": sha256_file(environment_path),
            "topology_file": topology_path.name,
            "topology_sha256": sha256_file(topology_path),
        },
    }


def validate_runtime_manifest(value: Mapping[str, Any], config: Phase4Config) -> list[str]:
    errors = []
    if value.get("schema_version") != "specrhythm.phase4-runtime-manifest.v1":
        errors.append("unsupported runtime manifest schema")
    if value.get("stage") != "phase4a-stock-vllm-bringup":
        errors.append("runtime manifest has the wrong stage")
    if value.get("serving_performance_result") is not False:
        errors.append("bring-up must not be marked as a serving performance result")
    framework = value.get("framework")
    if not isinstance(framework, Mapping):
        errors.append("runtime framework metadata is missing")
    else:
        if framework.get("version") != config.expected_vllm_version:
            errors.append("runtime vLLM version does not match the freeze")
        if framework.get("source_commit") != config.expected_vllm_commit:
            errors.append("runtime vLLM commit does not match the freeze")
        if framework.get("built_in_speculative_decoding") is not False:
            errors.append("built-in speculative decoding is forbidden for this bring-up")
        if framework.get("vllm_dbo_enabled") is not False:
            errors.append("vLLM DBO is forbidden for this bring-up")
        if framework.get("specrhythm_dual_batch_implemented") is not False:
            errors.append("Phase-4A.0 must not claim SpecRhythm Dual-Batch")
        if not framework.get("attention_backend"):
            errors.append("runtime attention backend is missing")
    role = value.get("role")
    engine = config.draft if role == "draft" else config.target if role == "target" else None
    runtime_engine = value.get("engine")
    rows = value.get("worker_ranks")
    rows = rows if isinstance(rows, list) else []
    if engine is None:
        errors.append("runtime role must be draft or target")
    elif len(rows) != engine.tensor_parallel_size:
        errors.append("worker rank count does not equal the configured TP size")
    elif not isinstance(runtime_engine, Mapping):
        errors.append("runtime engine metadata is missing")
    else:
        if runtime_engine.get("physical_gpu_ids") != list(engine.physical_gpu_ids):
            errors.append("runtime engine physical GPUs do not match the config")
        if runtime_engine.get("tensor_parallel_size") != engine.tensor_parallel_size:
            errors.append("runtime engine TP size does not match the config")
        for key in ("model", "tokenizer"):
            revision = runtime_engine.get(key)
            if not isinstance(revision, Mapping) or not revision.get("resolved_revision"):
                errors.append(f"runtime {key} revision is missing")
    inputs = value.get("inputs")
    if not isinstance(inputs, Mapping) or inputs.get("config_sha256") != sha256_file(config.path):
        errors.append("runtime config checksum does not match the checked config")
    correctness = value.get("correctness")
    if not isinstance(correctness, Mapping):
        errors.append("runtime correctness configuration is missing")
    else:
        requested = correctness.get("batch_invariant_requested")
        effective = correctness.get("batch_invariant_effective")
        validation = correctness.get("batch_invariant_validation")
        if requested not in (True, False):
            errors.append("runtime batch-invariant requested flag is invalid")
        if effective not in (True, False):
            errors.append("runtime batch-invariant effective flag is invalid")
        if correctness.get("configured_before_worker_creation") is not True:
            errors.append("correctness mode was not configured before worker creation")
        if not isinstance(validation, Mapping) or validation.get("valid") is not True:
            errors.append("runtime batch-invariant validation did not pass")
        if requested is True and effective is not True:
            errors.append("requested batch-invariant mode is not proven effective")
        expected_env = "1" if requested is True else "0"
        if correctness.get("VLLM_BATCH_INVARIANT") != expected_env:
            errors.append("runtime batch-invariant environment setting is inconsistent")
    for row in rows:
        if not isinstance(row, Mapping):
            errors.append("invalid worker rank record")
            continue
        if row.get("parameter_count", 0) <= 0 or row.get("parameter_bytes", 0) <= 0:
            errors.append("worker rank has no model parameters")
        if row.get("allocated_memory_bytes", 0) <= 0:
            errors.append("worker rank has no allocated CUDA memory")
        if not row.get("gpu_uuid"):
            errors.append("worker rank GPU UUID is missing")
        if not row.get("all_parameters_on_expected_device"):
            errors.append("worker rank model parameters are on an unexpected device")
    return errors
