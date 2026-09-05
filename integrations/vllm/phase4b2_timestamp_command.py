#!/usr/bin/env python3
"""Run one command while timestamping merged output for JIT provenance only."""

from __future__ import annotations

import argparse
import codecs
import json
import os
import selectors
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
        )
        assert process.stdout is not None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pending = ""
        child_exited_at = None

        def record(line: str) -> None:
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
        # EOF belongs to every inherited writer, not just our direct child.
        # Poll the actual child independently and bound the post-exit drain.
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                if process.poll() is not None and child_exited_at is None:
                    child_exited_at = time.monotonic()
                if child_exited_at is not None and time.monotonic() - child_exited_at >= 0.25:
                    break
                events = selector.select(timeout=0.05)
                if not events:
                    continue
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    break
                pending += decoder.decode(chunk)
                while "\n" in pending:
                    line, pending = pending.split("\n", 1)
                    record(line + "\n")
            pending += decoder.decode(b"", final=True)
            if pending:
                record(pending)
        process.stdout.close()
        status = process.wait()
        artifact.flush()
        os.fsync(artifact.fileno())
    return int(status)


if __name__ == "__main__":
    raise SystemExit(main())
