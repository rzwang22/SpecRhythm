#!/usr/bin/env python3
"""Run one command while timestamping merged output for JIT provenance only."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite timestamped log: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as artifact:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            artifact.write(
                json.dumps(
                    {
                        "schema_version": "specrhythm.phase4b2-timestamped-log.v1",
                        "timestamp_clock": "time.monotonic_ns",
                        "timestamp_ns": time.monotonic_ns(),
                        "stream": "merged-stdout-stderr",
                        "line": line.rstrip("\n"),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            artifact.flush()
        status = process.wait()
        artifact.flush()
        os.fsync(artifact.fileno())
    return int(status)


if __name__ == "__main__":
    raise SystemExit(main())
