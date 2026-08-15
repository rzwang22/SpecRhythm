"""Read-only hardware snapshots and logical/physical CUDA identity helpers."""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

_GPU_QUERY_FIELDS = (
    "index",
    "name",
    "uuid",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "clocks.sm",
    "clocks.mem",
    "pstate",
    "memory.used",
    "ecc.mode.current",
    "pcie.link.gen.current",
    "pcie.link.gen.max",
    "pcie.link.width.current",
    "pcie.link.width.max",
)


def _command(argv: list[str]) -> tuple[Optional[str], Optional[str]]:
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return None, str(error)
    if completed.returncode:
        return None, completed.stderr.strip() or completed.stdout.strip()
    return completed.stdout.strip(), None


def _number(value: str) -> Optional[float]:
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"n/a", "[n/a]", "not supported"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def cuda_visible_devices_mapping(device_count: Optional[int] = None) -> list[dict[str, Any]]:
    """Return the explicit logical-to-physical CUDA mapping without inventing IDs."""

    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is not None:
        configured = [item.strip() for item in raw.split(",") if item.strip()]
        return [
            {"logical_cuda_index": logical, "physical_gpu_id": physical}
            for logical, physical in enumerate(configured)
        ]
    if device_count is None:
        try:
            import torch  # type: ignore[import-not-found]

            device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except ImportError:
            device_count = 0
    return [
        {"logical_cuda_index": logical, "physical_gpu_id": str(logical)}
        for logical in range(device_count)
    ]


def physical_gpu_id(logical_cuda_index: int, device_count: Optional[int] = None) -> str:
    for row in cuda_visible_devices_mapping(device_count):
        if row["logical_cuda_index"] == logical_cuda_index:
            return str(row["physical_gpu_id"])
    raise ValueError(f"logical CUDA device {logical_cuda_index} is not visible")


def _parse_gpu_rows(text: str) -> list[dict[str, Any]]:
    rows = []
    for line in text.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) != len(_GPU_QUERY_FIELDS):
            continue
        raw = dict(zip(_GPU_QUERY_FIELDS, values))
        try:
            index = int(raw["index"])
        except ValueError:
            continue
        rows.append(
            {
                "physical_gpu_id": str(index),
                "gpu_name": raw["name"] or None,
                "gpu_uuid": raw["uuid"] or None,
                "temperature_c": _number(raw["temperature.gpu"]),
                "power_draw_w": _number(raw["power.draw"]),
                "power_limit_w": _number(raw["power.limit"]),
                "sm_clock_mhz": _number(raw["clocks.sm"]),
                "memory_clock_mhz": _number(raw["clocks.mem"]),
                "p_state": raw["pstate"] or None,
                "memory_used_mib": _number(raw["memory.used"]),
                "ecc_status": raw["ecc.mode.current"] or None,
                "pcie_generation_current": _number(raw["pcie.link.gen.current"]),
                "pcie_generation_max": _number(raw["pcie.link.gen.max"]),
                "pcie_width_current": _number(raw["pcie.link.width.current"]),
                "pcie_width_max": _number(raw["pcie.link.width.max"]),
            }
        )
    return rows


def _fallback_gpu_rows(errors: list[str]) -> list[dict[str, Any]]:
    """Preserve essential identity when one optional nvidia-smi field is unsupported."""

    identity, identity_error = _command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid",
            "--format=csv,noheader,nounits",
        ]
    )
    if identity_error:
        errors.append(f"nvidia-smi identity query failed: {identity_error}")
        return []
    rows: dict[str, dict[str, Any]] = {}
    for line in (identity or "").splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) != 3:
            continue
        index, name, gpu_uuid = values
        rows[index] = {
            "physical_gpu_id": index,
            "gpu_name": name or None,
            "gpu_uuid": gpu_uuid or None,
            "temperature_c": None,
            "power_draw_w": None,
            "power_limit_w": None,
            "sm_clock_mhz": None,
            "memory_clock_mhz": None,
            "p_state": None,
            "memory_used_mib": None,
            "ecc_status": None,
            "pcie_generation_current": None,
            "pcie_generation_max": None,
            "pcie_width_current": None,
            "pcie_width_max": None,
        }
    optional = {
        "temperature.gpu": "temperature_c",
        "power.draw": "power_draw_w",
        "power.limit": "power_limit_w",
        "clocks.sm": "sm_clock_mhz",
        "clocks.mem": "memory_clock_mhz",
        "pstate": "p_state",
        "memory.used": "memory_used_mib",
        "ecc.mode.current": "ecc_status",
        "pcie.link.gen.current": "pcie_generation_current",
        "pcie.link.gen.max": "pcie_generation_max",
        "pcie.link.width.current": "pcie_width_current",
        "pcie.link.width.max": "pcie_width_max",
    }
    text_fields = {"p_state", "ecc_status"}
    for query_field, output_field in optional.items():
        output, error = _command(
            [
                "nvidia-smi",
                f"--query-gpu=index,{query_field}",
                "--format=csv,noheader,nounits",
            ]
        )
        if error:
            errors.append(f"nvidia-smi {query_field} query failed: {error}")
            continue
        for line in (output or "").splitlines():
            values = [item.strip() for item in line.split(",")]
            if len(values) != 2 or values[0] not in rows:
                continue
            rows[values[0]][output_field] = (
                values[1] if output_field in text_fields else _number(values[1])
            )
    return list(rows.values())


def _peer_access() -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return rows, ["PyTorch is unavailable; CUDA peer access was not queried"]
    if not torch.cuda.is_available():
        return rows, ["CUDA is unavailable; CUDA peer access was not queried"]
    mapping = cuda_visible_devices_mapping(torch.cuda.device_count())
    by_logical = {row["logical_cuda_index"]: row for row in mapping}
    for source in range(torch.cuda.device_count()):
        for destination in range(torch.cuda.device_count()):
            if source == destination:
                continue
            try:
                available = bool(torch.cuda.can_device_access_peer(source, destination))
            except (AttributeError, RuntimeError) as error:
                errors.append(f"peer access {source}->{destination} query failed: {error}")
                available = None
            rows.append(
                {
                    "source_logical_cuda_index": source,
                    "source_physical_gpu_id": by_logical.get(source, {}).get(
                        "physical_gpu_id"
                    ),
                    "destination_logical_cuda_index": destination,
                    "destination_physical_gpu_id": by_logical.get(destination, {}).get(
                        "physical_gpu_id"
                    ),
                    "cuda_device_can_access_peer": available,
                }
            )
    return rows, errors


def capture_hardware_state(
    physical_gpu_ids: Iterable[int],
) -> dict[str, Any]:
    """Capture best-effort read-only GPU state; unavailable fields stay null."""

    requested = {str(value) for value in physical_gpu_ids}
    errors: list[str] = []
    query, query_error = _command(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(_GPU_QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]
    )
    if query_error:
        errors.append(f"nvidia-smi state query failed: {query_error}")
    all_gpus = (
        _parse_gpu_rows(query or "") if query is not None else _fallback_gpu_rows(errors)
    )
    gpus = [row for row in all_gpus if row["physical_gpu_id"] in requested]
    missing = sorted(requested - {row["physical_gpu_id"] for row in gpus})
    if missing:
        errors.append(f"requested physical GPU IDs were not reported: {missing}")
    topology, topology_error = _command(["nvidia-smi", "topo", "-m"])
    if topology_error:
        errors.append(f"nvidia-smi topology query failed: {topology_error}")
    peers, peer_errors = _peer_access()
    errors.extend(peer_errors)
    return {
        "schema_version": "specrhythm.gpu-hardware-state.v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "clock_locked": False,
        "clock_lock_note": "clocks were observed only; this benchmark never changes GPU clocks",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_visible_devices_mapping": cuda_visible_devices_mapping(),
        "gpus": gpus,
        "peer_access": peers,
        "nvlink_pcie_topology": topology,
        "errors": errors,
    }
