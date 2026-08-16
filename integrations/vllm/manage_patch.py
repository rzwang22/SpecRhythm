#!/usr/bin/env python3
"""Check, apply, or restore the ordered pinned-vLLM Python patch stack."""

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
WORKER_HOOKS_SHA256 = "a99c410cd791f20071bb17b8a619e5b309427b50ed864b8753d066c1dc4b150c"
PATCHED_SHA256 = "5cd618de8826e15ef00ca1735101a29af06029b7ce9d54cede00bf2b401cc257"
PATCH = Path(__file__).parent / "patches" / "0001-custom-proposer-request-and-verify-hooks.patch"
SCHEDULER_FILE = Path("vllm/v1/core/sched/scheduler.py")
SCHEDULER_BASE_SHA256 = (
    "e25d4c9a95abdbe8e516714ed02574d929ca0d5e8c11c4cc73b84d3a3b905443"
)
SCHEDULER_PATCHED_SHA256 = (
    "ffaefd61869589f086e6acdf9a0c4f55f80d5dad145ca3f6fff2379f7a4e2455"
)
SCHEDULER_PATCH = (
    Path(__file__).parent
    / "patches"
    / "0002-scheduler-request-admissibility-hook.patch"
)
TIMING_PATCH = (
    Path(__file__).parent / "patches" / "0003-target-forward-timing-observer.patch"
)
LEGACY_PATCHED_SHA256 = (
    "ba307cbfdfa9079c04e1bf9bb6387eb923cbabb1eea811e720a94897ea6483fa"
)
LEGACY_PATCH = Path(__file__).parent / "patches" / "0000-phase4a1-legacy-hooks.patch"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_patch(
    root: Path, *, patch: Path, reverse: bool, dry_run: bool
) -> subprocess.CompletedProcess[str]:
    command = ["patch", "--batch", "--fuzz=0", "-p1", "-d", str(root)]
    if dry_run:
        command.append("--dry-run")
    if reverse:
        command.append("--reverse")
    command.extend(("-i", str(patch)))
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
    scheduler = root / SCHEDULER_FILE
    for path in (target, scheduler):
        if not path.is_file():
            raise SystemExit(f"missing installed vLLM file: {path}")
    before = {str(TARGET_FILE): sha256(target), str(SCHEDULER_FILE): sha256(scheduler)}
    if args.operation in {"check", "apply"}:
        expected = {
            str(TARGET_FILE): {BASE_SHA256},
            str(SCHEDULER_FILE): {SCHEDULER_BASE_SHA256},
        }
    else:
        expected = {
            str(TARGET_FILE): {
                BASE_SHA256,
                WORKER_HOOKS_SHA256,
                PATCHED_SHA256,
                LEGACY_PATCHED_SHA256,
            },
            str(SCHEDULER_FILE): {SCHEDULER_BASE_SHA256, SCHEDULER_PATCHED_SHA256},
        }
    for relative, allowed in expected.items():
        if before[relative] not in allowed:
            raise SystemExit(
                f"refusing {args.operation}: {relative} SHA256 is {before[relative]}, "
                f"expected one of {sorted(allowed)}"
            )
    active_runner_patch = (
        LEGACY_PATCH if before[str(TARGET_FILE)] == LEGACY_PATCHED_SHA256 else PATCH
    )
    operations = [
        (TARGET_FILE, active_runner_patch),
        (SCHEDULER_FILE, SCHEDULER_PATCH),
        (TARGET_FILE, TIMING_PATCH),
    ]
    if args.operation == "restore":
        operations = []
        if before[str(TARGET_FILE)] == PATCHED_SHA256:
            operations.append((TARGET_FILE, TIMING_PATCH))
        if before[str(SCHEDULER_FILE)] == SCHEDULER_PATCHED_SHA256:
            operations.append((SCHEDULER_FILE, SCHEDULER_PATCH))
        if before[str(TARGET_FILE)] in {PATCHED_SHA256, WORKER_HOOKS_SHA256}:
            operations.append((TARGET_FILE, PATCH))
        elif before[str(TARGET_FILE)] == LEGACY_PATCHED_SHA256:
            operations.append((TARGET_FILE, LEGACY_PATCH))
    for _relative, patch in operations:
        check = run_patch(
            root, patch=patch, reverse=args.operation == "restore", dry_run=True
        )
        if check.returncode != 0:
            raise SystemExit(check.stdout + check.stderr)
        if args.operation in {"apply", "restore"}:
            applied = run_patch(
                root, patch=patch, reverse=args.operation == "restore", dry_run=False
            )
            if applied.returncode != 0:
                raise SystemExit(applied.stdout + applied.stderr)
    after = {str(TARGET_FILE): sha256(target), str(SCHEDULER_FILE): sha256(scheduler)}
    expected_after = {
        str(TARGET_FILE): (
            PATCHED_SHA256 if args.operation == "apply" else BASE_SHA256
        ),
        str(SCHEDULER_FILE): (
            SCHEDULER_PATCHED_SHA256 if args.operation == "apply" else SCHEDULER_BASE_SHA256
        ),
    }
    if args.operation == "check":
        expected_after = {
            str(TARGET_FILE): BASE_SHA256,
            str(SCHEDULER_FILE): SCHEDULER_BASE_SHA256,
        }
    for relative, expected_sha in expected_after.items():
        if after[relative] != expected_sha:
            raise SystemExit(
                f"post-operation {relative} SHA256 is {after[relative]}, "
                f"expected {expected_sha}"
            )
    stack = [
        {
            "order": 1,
            "patch_file": active_runner_patch.name,
            "patch_sha256": sha256(active_runner_patch),
            "target_file": str(TARGET_FILE),
            "original_source_sha256": BASE_SHA256,
            "patched_source_sha256": WORKER_HOOKS_SHA256,
        },
        {
            "order": 2,
            "patch_file": SCHEDULER_PATCH.name,
            "patch_sha256": sha256(SCHEDULER_PATCH),
            "target_file": str(SCHEDULER_FILE),
            "original_source_sha256": SCHEDULER_BASE_SHA256,
            "patched_source_sha256": SCHEDULER_PATCHED_SHA256,
        },
        {
            "order": 3,
            "patch_file": TIMING_PATCH.name,
            "patch_sha256": sha256(TIMING_PATCH),
            "target_file": str(TARGET_FILE),
            "original_source_sha256": WORKER_HOOKS_SHA256,
            "patched_source_sha256": PATCHED_SHA256,
        },
    ]
    report = {
        "schema_version": "specrhythm.vllm-base-and-patch-manifest.v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": args.operation,
        "vllm_version": "0.25.1",
        "vllm_base_commit": BASE_COMMIT,
        "verified_source_commit": commit,
        "patch_file": active_runner_patch.name,
        "patch_sha256": sha256(active_runner_patch),
        "patch_stack": stack,
        "patch_stack_applied": args.operation == "apply",
        "patch_scope": [str(TARGET_FILE), str(SCHEDULER_FILE)],
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
        "target_file_sha256_before": before[str(TARGET_FILE)],
        "target_file_sha256_after": after[str(TARGET_FILE)],
        "scheduler_file_sha256_before": before[str(SCHEDULER_FILE)],
        "scheduler_file_sha256_after": after[str(SCHEDULER_FILE)],
        "source_files": {
            relative: {"before_sha256": before[relative], "after_sha256": after[relative]}
            for relative in sorted(before)
        },
        "patch_applied": after[str(TARGET_FILE)] == PATCHED_SHA256,
    }
    if args.manifest:
        atomic_json(Path(args.manifest).resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
