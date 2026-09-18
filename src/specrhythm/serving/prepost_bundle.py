"""Bounded pre/post export including joint correctness; raw events appear once."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

from specrhythm.serving.common import require
from specrhythm.serving.eager_latency_bundle import bundle


def export(root, output):
    # Main archive's original bounded inventory includes measurement and failed status.
    result = bundle(root, output)
    joint = root / "prepost-joint-gpu-check"
    if not joint.exists():
        result["joint_correctness"] = "not in this point; see first paired root"
        return result
    destination = output.with_name(output.name.removesuffix(".tar.gz") + "-joint.tar.gz")
    require(not destination.exists(), "joint archive already exists")
    names = {
        "result.json",
        "failure.json",
        "runtime.json",
        "draft-backend-report.json",
        "point.json",
        "light-summary.json",
        "process-lifecycle.json",
        "execution-manifest.json",
        "diagnostic-primary-error.json",
        "draft-child-failure.json",
        "child-failure.json",
        "exit-code.json",
        "launcher-failure.json",
        "diagnostic-secondary-errors.json",
        "config.json",
        "patch-manifest.json",
        "environment.json",
        "topology.json",
    }
    files = sorted(
        p
        for p in joint.rglob("*")
        if p.is_file() and (p.name in names or p.name.startswith("fixed-logging-"))
    )
    require(len(files) <= 64, "joint archive file count exceeded")
    require(
        sum(p.stat().st_size for p in files) <= 256 * 1024 * 1024,
        "joint archive byte limit exceeded before export",
    )
    total, inventory = 0, []
    with tarfile.open(destination, "x:gz") as archive:
        for path in files:
            require(not path.is_symlink(), "joint archive refuses symlinks")
            total += path.stat().st_size
            require(total <= 256 * 1024 * 1024, "joint archive byte limit exceeded")
            name = str(path.relative_to(joint))
            before = path.stat()
            data = path.read_bytes()
            after = path.stat()
            require(
                len(data) == before.st_size
                and (before.st_ino, before.st_mtime_ns, before.st_size)
                == (after.st_ino, after.st_mtime_ns, after.st_size),
                "joint source changed during export",
            )
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
            inventory.append(
                dict(
                    path=name,
                    bytes=len(data),
                    status="INCLUDED",
                    sha256=hashlib.sha256(data).hexdigest(),
                )
            )
        from specrhythm.continuation.prepost import MODES

        expected = ["result.json"]
        for mode in ("target", *MODES):
            runs = list((joint / mode / "runs").glob("*/point.json"))
            if not runs:
                expected.append(mode + "/runs/<point>/point.json")
            for point in runs:
                expected.extend(
                    str((point.parent / name).relative_to(joint))
                    for name in (
                        "runtime.json",
                        "draft-backend-report.json",
                        "light-summary.json",
                        "process-lifecycle.json",
                    )
                )
        included = {r["path"] for r in inventory}
        inventory.extend(
            dict(path=name, status="MISSING") for name in expected if name not in included
        )
        payload = json.dumps(
            dict(
                inventory=inventory,
                source_results_unchanged=True,
                missing=sum(r["status"] == "MISSING" for r in inventory),
                truncated=False,
                limits=dict(files=64, bytes=256 * 1024 * 1024),
                semantics="allowlisted files once; missing result may be a "
                "failed/not-started check; archive does not qualify execution",
            ),
            indent=2,
        ).encode()
        info = tarfile.TarInfo("joint-inventory.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    result["joint_correctness"] = dict(
        path=str(destination), files=len(files), source_bytes=total, limit_bytes=256 * 1024 * 1024
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(export(args.root, args.output)))


if __name__ == "__main__":
    main()
