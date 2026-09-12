"""Import-free inspection of the installed vLLM distribution."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path


def locate_installed_vllm_file(relative_path: Path) -> Path:
    """Locate one installed vLLM file without importing the ``vllm`` package."""

    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("installed vLLM path must be a safe relative path")
    try:
        distribution = metadata.distribution("vllm")
    except metadata.PackageNotFoundError as error:
        raise RuntimeError("vLLM is unavailable for installed runner verification") from error
    path = Path(distribution.locate_file(relative_path)).resolve()
    if not path.is_file():
        raise RuntimeError(f"installed vLLM file is missing: {relative_path}")
    return path
