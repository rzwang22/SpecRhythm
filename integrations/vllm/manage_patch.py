#!/usr/bin/env python3
"""Check, apply, or restore the pinned Phase-4A.1 vLLM Python patch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

BASE_COMMIT = "752a3a504485790a2e8491cacbb35c137339ad34"
TARGET_FILE = Path("vllm/v1/worker/gpu_model_runner.py")
BASE_SHA256 = "6c92ded8468f44d6df863a617ce588f132fa6df7031feecc0cc421702a41610e"
PATCHED_SHA256 = "ba307cbfdfa9079c04e1bf9bb6387eb923cbabb1eea811e720a94897ea6483fa"
PATCH = Path(__file__).parent / "patches" / "0001-custom-proposer-request-and-verify-hooks.patch"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_patch(root: Path, *, reverse: bool, dry_run: bool) -> subprocess.CompletedProcess[str]:
    command = ["patch", "--batch", "--fuzz=0", "-p1", "-d", str(root)]
    if dry_run:
        command.append("--dry-run")
    if reverse:
        command.append("--reverse")
    command.extend(("-i", str(PATCH)))
    return subprocess.run(command, capture_output=True, text=True, check=False)


def source_commit(source: Optional[Path]) -> Optional[str]:
    if source is None:
        return None
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=source,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError("vLLM source path is not a Git checkout")
    commit = completed.stdout.strip()
    if commit != BASE_COMMIT:
        raise ValueError(f"vLLM source must be exact base commit {BASE_COMMIT}")
    return commit


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("check", "apply", "restore"))
    parser.add_argument("--vllm-root", required=True)
    parser.add_argument("--source")
    parser.add_argument("--manifest")
    args = parser.parse_args()
    root = Path(args.vllm_root).resolve()
    source = Path(args.source).resolve() if args.source else None
    commit = source_commit(source)
    target = root / TARGET_FILE
    if not target.is_file():
        raise SystemExit(f"missing installed vLLM file: {target}")
    before = sha256(target)
    expected = PATCHED_SHA256 if args.operation == "restore" else BASE_SHA256
    if before != expected:
        raise SystemExit(
            f"refusing {args.operation}: {TARGET_FILE} SHA256 is {before}, expected {expected}"
        )
    reverse = args.operation == "restore"
    check = run_patch(root, reverse=reverse, dry_run=True)
    if check.returncode != 0:
        raise SystemExit(check.stdout + check.stderr)
    if args.operation in {"apply", "restore"}:
        applied = run_patch(root, reverse=reverse, dry_run=False)
        if applied.returncode != 0:
            raise SystemExit(applied.stdout + applied.stderr)
    after = sha256(target)
    expected_after = (
        BASE_SHA256 if reverse else (BASE_SHA256 if args.operation == "check" else PATCHED_SHA256)
    )
    if after != expected_after:
        raise SystemExit(f"post-operation SHA256 is {after}, expected {expected_after}")
    report = {
        "schema_version": "specrhythm.vllm-base-and-patch-manifest.v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": args.operation,
        "vllm_version": "0.25.1",
        "vllm_base_commit": BASE_COMMIT,
        "verified_source_commit": commit,
        "patch_file": PATCH.name,
        "patch_sha256": sha256(PATCH),
        "patch_scope": [str(TARGET_FILE)],
        "python_only": True,
        "cpp_cuda_modified": False,
        "target_only_behavior_change_when_speculation_disabled": False,
        "runtime_security": {
            "VLLM_ALLOW_INSECURE_SERIALIZATION": os.environ.get(
                "VLLM_ALLOW_INSECURE_SERIALIZATION"
            )
            == "1",
            "allowed_scope": "local-trusted-experiment-only",
            "production_safe": False,
        },
        "vllm_root": str(root),
        "target_file_sha256_before": before,
        "target_file_sha256_after": after,
        "patch_applied": after == PATCHED_SHA256,
    }
    if args.manifest:
        atomic_json(Path(args.manifest).resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
