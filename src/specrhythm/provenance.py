"""Reproducible provenance manifests for generated workloads."""

from __future__ import annotations

import hashlib
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import urlsplit, urlunsplit

from specrhythm import __version__


def sha256_file(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_path(path: Union[str, Path]) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.name


def pin_source_url(source_url: str, source_commit_sha: str) -> str:
    """Return a GitHub trace URL pinned to the manifest's exact commit SHA."""

    commit = source_commit_sha.strip().lower()
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit):
        raise ValueError("source_commit_sha must be a full 40- or 64-character hexadecimal SHA")
    parsed = urlsplit(source_url)
    parts = parsed.path.split("/")
    ref_index: Optional[int] = None
    if parsed.netloc.lower() == "github.com" and len(parts) > 5 and parts[3] == "blob":
        ref_index = 4
    elif parsed.netloc.lower() == "raw.githubusercontent.com" and len(parts) > 4:
        ref_index = 3

    if ref_index is None:
        if commit not in source_url:
            raise ValueError("source_url must be pinned to source_commit_sha")
        return source_url

    current_ref = parts[ref_index]
    if re.fullmatch(r"[0-9a-fA-F]{7,64}", current_ref) and current_ref.lower() != commit:
        raise ValueError("source_url commit does not match source_commit_sha")
    parts[ref_index] = commit
    return urlunsplit(parsed._replace(path="/".join(parts)))


def build_manifest(
    *,
    config: dict[str, Any],
    config_path: Union[str, Path],
    source_trace_path: Union[str, Path],
    output_workload_path: Union[str, Path],
    source_url: str,
    source_commit_sha: str,
    time_scale: float,
    window_start_ms: float,
    window_duration_ms: Optional[float],
    generation_command: str,
    request_count: int,
) -> dict[str, Any]:
    """Build a manifest without relying on machine-specific absolute paths."""

    source_trace = Path(source_trace_path)
    output_workload = Path(output_workload_path)
    normalized_commit = source_commit_sha.strip().lower()
    return {
        "schema_version": "specrhythm.provenance.v1",
        "workload_family": config.get("workload_family", "unspecified"),
        "data_status": config.get("data_status", "unspecified"),
        "payload_status": config.get("payload", {}).get("status", "unspecified"),
        "acceptance_status": config.get("acceptance", {}).get("status", "unspecified"),
        "source_url": pin_source_url(source_url, normalized_commit),
        "source_commit_sha": normalized_commit,
        "source_trace_file": source_trace.name,
        "source_trace_path": _portable_path(source_trace),
        "source_trace_sha256": sha256_file(source_trace),
        "config_file": _portable_path(config_path),
        "config_sha256": sha256_file(config_path),
        "output_workload_file": output_workload.name,
        "output_workload_path": _portable_path(output_workload),
        "output_workload_sha256": sha256_file(output_workload),
        "seed": int(config.get("seed", 0)),
        "time_scale": float(time_scale),
        "window_start_ms": float(window_start_ms),
        "window_duration_ms": (
            float(window_duration_ms) if window_duration_ms is not None else None
        ),
        "generation_command": generation_command,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "python_version": platform.python_version(),
        "specrhythm_version": __version__,
        "request_count": int(request_count),
    }
