"""Read-only S1 child log mirroring and bounded failure summaries."""

from __future__ import annotations

import codecs
import json
import sys
import threading
from pathlib import Path

from specrhythm.serving.common import read_json

LOGS = {"Target": "target.log", "Draft": "draft-service.log"}


class RunConsole:
    """Children still write directly to raw files; display never owns their stdout pipes."""

    def __init__(self, directory, mode, *, stream=None, poll_seconds=0.05):
        self.directory, self.mode = directory, mode
        self.stream = sys.stdout if stream is None else stream
        self.poll_seconds = poll_seconds
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._follow, name="s1-log-display", daemon=True)
        self.display_available = True

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        # Terminal display cannot hold up owned cleanup or replace a child return code.
        self.thread.join(timeout=2)

    def _emit(self, source, text):
        if not self.display_available:
            return
        try:
            for line in text.splitlines():
                print(f"[S1-P {self.mode} {source}] {line}", file=self.stream, flush=True)
        except (OSError, ValueError):
            # A closed terminal does not invalidate or truncate the original child logs.
            self.display_available = False

    def _follow(self):
        handles = {}
        decoders = {s: codecs.getincrementaldecoder("utf-8")("replace") for s in LOGS}
        try:
            while True:
                final = self.stop.is_set()
                for source, name in LOGS.items():
                    if source not in handles:
                        try:
                            handles[source] = (self.directory / name).open("rb")
                        except FileNotFoundError:
                            continue
                    while True:
                        chunk = handles[source].read(65536)
                        if not chunk:
                            break
                        self._emit(source, decoders[source].decode(chunk))
                        if not final:
                            break
                if final:
                    for source, decoder in decoders.items():
                        self._emit(source, decoder.decode(b"", final=True))
                    return
                self.stop.wait(self.poll_seconds)
        except OSError as error:
            self._emit("console", f"log display unavailable: {error}; original logs retained")
        finally:
            for handle in handles.values():
                handle.close()


def print_failure(error, *, directory=None, mode="gate", stream=None):
    """Emit qualification fields and both original log tails without changing artifacts."""
    stream = sys.stderr if stream is None else stream

    def emit(source, value):
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        print(f"[S1-P {mode} {source}] {text}", file=stream, flush=True)

    try:
        emit("FAILED", {"error": str(error), "details": getattr(error, "details", {})})
        if directory is None:
            return
        directory = Path(directory)
        for name in (
            "result.json",
            "child-failure.json",
            "draft-child-failure.json",
            "launcher-failure.json",
            "exit-code.json",
            "process-lifecycle.json",
        ):
            path = directory / name
            if not path.is_file():
                continue
            try:
                value = read_json(path)
                if name == "result.json":
                    emit(
                        "qualification",
                        {
                            "artifact": str(path),
                            "errors": value.get("errors"),
                            "error_details": value.get("error_details"),
                            "primary_error": value.get("primary_error"),
                        },
                    )
                    for diagnostic in value.get("secondary_diagnostics", []):
                        emit("secondary diagnostic", diagnostic)
                    for check in value.get("draft_backend_checks", {}).values():
                        if check.get("valid") is not True:
                            emit("Draft check", check)
                elif name == "process-lifecycle.json":
                    emit(
                        "lifecycle",
                        {
                            "artifact": str(path),
                            **{
                                k: value.get(k)
                                for k in (
                                    "target_exit_status",
                                    "effective_exit_status",
                                    "cleanup_valid",
                                    "owned_cleanup_completed",
                                    "remaining_owned_pids",
                                    "launch_error",
                                    "failure_detection",
                                )
                            },
                        },
                    )
                else:
                    emit("failure evidence", {"artifact": str(path), **value})
            except (OSError, ValueError, TypeError, AttributeError) as read_error:
                emit("failure evidence", f"cannot read {path}: {read_error}")
        for source, name in LOGS.items():
            path = directory / name
            emit(source, f"last 40 lines (at most 64 KiB): {path}")
            try:
                with path.open("rb") as handle:
                    handle.seek(0, 2)
                    handle.seek(max(0, handle.tell() - 65536))
                    tail = handle.read().decode("utf-8", errors="replace").splitlines()[-40:]
                for line in tail:
                    emit(source, line)
            except OSError as read_error:
                emit(source, f"log unavailable: {read_error}")
    except (OSError, ValueError):
        # Preserve the actual execution failure if the console itself is unavailable.
        return
