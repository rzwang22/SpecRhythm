"""Pure, canonical data helpers shared by S0 build and independent validation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from specrhythm.phase3.trace import sha256_file

TASKS = ("chat", "code", "summarization", "reasoning")
REQUEST_SCHEMA = "specrhythm.serving-request.v1"
CONFIG_SCHEMA = "specrhythm.serving-config.v1"
LOCK_SCHEMA = "specrhythm.serving-sources.v1"
MANIFEST_SCHEMA = "specrhythm.serving-manifest.v1"
ALGORITHM = "sha256-rank-global-dedup-then-quota-v1"


class DataError(ValueError):
    def __init__(self, message: str, **details: Any):
        super().__init__(message)
        self.details = details


def require(condition: Any, message: str, **details: Any) -> None:
    if not condition:
        raise DataError(message, **details)


def canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        + "\n"
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_prompt(text: str) -> str:
    # Only line-ending normalization; preserve indentation, spaces and Unicode.
    return text.replace("\r\n", "\n")


def read_json(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    with Path(path).open("xb") as handle:
        handle.write(canonical(value))


def integer(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def finite(value: Any, minimum: float = 0) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and value >= minimum
    except OverflowError:
        return False


def is_hash(value: Any, size: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == size
        and all(c in "0123456789abcdef" for c in value)
    )


def distribution(values: list) -> dict:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0] if ordered else None,
        **{
            f"p{p}": ordered[max(0, math.ceil(len(ordered) * p / 100) - 1)] if ordered else None
            for p in (50, 90, 99)
        },
        "max": ordered[-1] if ordered else None,
    }


def source_path(root: Path, group: str, filename: str) -> Path:
    relative = Path(filename)
    require(
        not relative.is_absolute() and ".." not in relative.parts,
        "source file path must be relative and contained",
        path=filename,
    )
    return Path(root) / group / relative


def file_inventory(root: Path, names: list[str]) -> dict[str, str]:
    return {name: sha256_file(Path(root) / name) for name in sorted(names)}
