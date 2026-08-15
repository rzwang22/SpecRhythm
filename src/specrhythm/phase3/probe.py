"""Read-only NVIDIA/PyTorch environment probe for reproducible GPU runs."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class GPUProbeError(RuntimeError):
    """Raised when a caller requires CUDA but the probe cannot establish it."""


def _command(argv: list[str], *, cwd: Optional[Path] = None) -> tuple[Optional[str], str]:
    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd) if cwd is not None else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return None, str(error)
    if completed.returncode:
        message = completed.stderr.strip() or completed.stdout.strip()
        return None, message or f"exit status {completed.returncode}"
    return completed.stdout.strip(), ""


def _git_commit(repo: Optional[Path]) -> tuple[Optional[str], Optional[str]]:
    output, error = _command(["git", "rev-parse", "HEAD"], cwd=repo)
    return output, error or None


def _parse_nvidia_rows(text: str) -> list[dict[str, Any]]:
    rows = []
    for visible_index, line in enumerate(text.splitlines()):
        values = [part.strip() for part in line.split(",")]
        if len(values) != 4:
            continue
        uuid, name, memory_mib, driver = values
        try:
            memory_bytes = int(memory_mib) * 1024 * 1024
        except ValueError:
            memory_bytes = None
        rows.append(
            {
                "visible_index": visible_index,
                "name": name,
                "memory_bytes": memory_bytes,
                "memory_mib": int(memory_mib) if memory_mib.isdigit() else memory_mib,
                "uuid": uuid,
                "compute_capability": None,
                "driver_version": driver,
            }
        )
    return rows


def _torch_metadata(errors: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata: dict[str, Any] = {
        "pytorch_version": None,
        "cuda_runtime": None,
        "nccl_version": None,
        "torch_cuda_available": False,
    }
    devices = []
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        errors.append("PyTorch is not installed; CUDA runtime and NCCL could not be queried")
        return metadata, devices
    metadata["pytorch_version"] = torch.__version__
    metadata["cuda_runtime"] = torch.version.cuda
    metadata["torch_cuda_available"] = bool(torch.cuda.is_available())
    try:
        version = torch.cuda.nccl.version() if torch.cuda.is_available() else None
        if isinstance(version, tuple):
            metadata["nccl_version"] = ".".join(str(item) for item in version)
        elif version is not None:
            metadata["nccl_version"] = str(version)
    except (AttributeError, RuntimeError) as error:
        errors.append(f"NCCL version query failed: {error}")
    if not torch.cuda.is_available():
        errors.append("torch.cuda.is_available() is false")
        return metadata, devices
    try:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "visible_index": index,
                    "name": properties.name,
                    "memory_bytes": properties.total_memory,
                    "uuid": str(getattr(properties, "uuid", "")) or None,
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
    except RuntimeError as error:
        errors.append(f"PyTorch CUDA device query failed: {error}")
    return metadata, devices


def probe_gpu_environment(
    *, repo: Optional[Path] = None, require_cuda: bool = False
) -> dict[str, Any]:
    """Return environment metadata without allocating tensors or changing GPU state."""

    errors: list[str] = []
    commit, git_error = _git_commit(repo)
    if git_error:
        errors.append(f"git commit query failed: {git_error}")
    torch_metadata, torch_devices = _torch_metadata(errors)
    query = [
        "nvidia-smi",
        "--query-gpu=uuid,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    rows_text, nvidia_error = _command(query)
    nvidia_devices = _parse_nvidia_rows(rows_text or "")
    if nvidia_error:
        errors.append(f"nvidia-smi GPU query failed: {nvidia_error}")
    capabilities, capability_error = _command(
        [
            "nvidia-smi",
            "--query-gpu=compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if capabilities:
        for device, capability in zip(nvidia_devices, capabilities.splitlines()):
            device["compute_capability"] = capability.strip()
    elif capability_error:
        errors.append(f"nvidia-smi compute-capability query failed: {capability_error}")
    for device, torch_device in zip(nvidia_devices, torch_devices):
        if not device.get("compute_capability"):
            device["compute_capability"] = torch_device.get("compute_capability")
    topology, topology_error = _command(["nvidia-smi", "topo", "-m"])
    if topology_error:
        errors.append(f"nvidia-smi topology query failed: {topology_error}")
    devices = nvidia_devices or torch_devices
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_mapping = []
    if cuda_visible is not None:
        for logical, physical in enumerate(
            item.strip() for item in cuda_visible.split(",") if item.strip()
        ):
            visible_mapping.append({"logical_index": logical, "configured_id": physical})
    driver_versions = sorted(
        {str(item["driver_version"]) for item in nvidia_devices if item.get("driver_version")}
    )
    result = {
        "schema_version": "specrhythm.gpu-environment.v1",
        "available": bool(devices) and bool(torch_metadata["torch_cuda_available"]),
        "hostname": socket.gethostname(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        **torch_metadata,
        "cuda_driver": driver_versions[0] if len(driver_versions) == 1 else driver_versions,
        "gpu_count": len(devices),
        "gpus": devices,
        "gpu_name": [item.get("name") for item in devices],
        "gpu_memory": [item.get("memory_bytes") for item in devices],
        "gpu_uuid": [item.get("uuid") for item in devices],
        "compute_capability": [item.get("compute_capability") for item in devices],
        "cuda_visible_devices": cuda_visible,
        "cuda_visible_devices_mapping": visible_mapping,
        "nvlink_nvswitch_topology": topology,
        "topology_query_error": topology_error or None,
        "errors": errors,
    }
    if require_cuda and not result["available"]:
        raise GPUProbeError(json.dumps(result, indent=2, sort_keys=True))
    return result
