"""Publish a final immutable Draft report after owner stop, within its drain deadline."""

import json
import os
import tempfile
import time
from pathlib import Path

from specrhythm.io_context import file_context
from specrhythm.phase4.manifest import atomic_write_json, sha256_file

SCHEMA = "specrhythm.final-report-publication.v1"
STATE_NAME = "draft-report-state.json"


def publish_final_report(path, build, *, deadline_ns, owner_stopped):
    path = Path(path)
    if not owner_stopped:
        raise ValueError("final report requires a stopped owner")
    if path.exists():
        raise FileExistsError(path)

    def bounded():
        if type(deadline_ns) is not int or time.monotonic_ns() >= deadline_ns:
            raise TimeoutError("final Draft report exceeded original drain deadline")

    bounded()
    fd, temporary = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".partial", dir=path.parent
    )
    state = dict(
        schema_version=SCHEMA,
        status="WRITING",
        report_file=path.name,
        temporary_file=Path(temporary).name,
        writer_pid=os.getpid(),
        started_ns=time.monotonic_ns(),
        deadline_ns=deadline_ns,
        owner_stopped=True,
        final_file_published=False,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            atomic_write_json(path.with_name(STATE_NAME), state)
            value = build()
            bounded()
            json.dump(
                {
                    **value,
                    "report_publication": dict(
                        schema_version=SCHEMA, state_file=STATE_NAME, owner_stopped=True
                    ),
                },
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            with file_context(path, physical_path=temporary, write_kind="immutable_json"):
                os.fsync(handle.fileno())
        bounded()
        # Same-filesystem exclusive publication: never overwrite an existing final.
        os.link(temporary, path)
        state["final_file_published"] = True
        bounded()
        state.update(
            status="COMPLETE",
            bytes=path.stat().st_size,
            sha256=sha256_file(path),
            completed_ns=time.monotonic_ns(),
        )
        bounded()
        atomic_write_json(path.with_name(STATE_NAME), state)
        bounded()
        Path(temporary).unlink()
        return state
    except BaseException as error:
        state.update(
            status="FAILED",
            error=str(error),
            exception_type=type(error).__name__,
            failed_ns=time.monotonic_ns(),
        )
        try:
            atomic_write_json(path.with_name(STATE_NAME), state)
        except Exception as secondary:
            # Keep the original exception and any partial file; no fabricated JSON.
            print("secondary final report status write: " + str(secondary), flush=True)
        raise


def qualify_publication_receipt(state, *, report_file, size, sha256):
    """Same completion contract for disk files and inventory-backed offline projections."""
    if (
        not isinstance(state, dict)
        or state.get("schema_version") != SCHEMA
        or state.get("status") != "COMPLETE"
        or state.get("owner_stopped") is not True
        or state.get("final_file_published") is not True
        or state.get("report_file") != report_file
        or type(state.get("bytes")) is not int
        or state["bytes"] != size
        or state.get("sha256") != sha256
        or any(
            type(state.get(k)) is not int for k in ("started_ns", "completed_ns", "deadline_ns")
        )
        or not 0 < state["started_ns"] <= state["completed_ns"] <= state["deadline_ns"]
    ):
        raise ValueError("incomplete/mismatched final Draft report publication: " + report_file)
    return state


def qualify_final_report(path):
    path = Path(path)
    state_path = path.with_name(STATE_NAME)
    with state_path.open() as handle:
        state = json.load(handle)
    if not isinstance(state, dict) or state.get("status") != "COMPLETE":
        raise ValueError("incomplete final Draft report publication: " + str(state_path))
    return qualify_publication_receipt(
        state, report_file=path.name, size=path.stat().st_size, sha256=sha256_file(path)
    )
