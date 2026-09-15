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


def publish_final_report(path, build, *, deadline_ns, owner_stopped, mode="unspecified"):
    from specrhythm.phase4.drain_deadline import deadline_context, validate_deadline

    path = Path(path)
    if path.exists():
        raise FileExistsError(path)  # Never rewrite an existing immutable report/receipt.
    stage = "deadline_validation"
    state = dict(
        schema_version=SCHEMA, status="WRITING", report_file=path.name,
        writer_pid=os.getpid(), started_ns=time.monotonic_ns(),
        deadline_ns=deadline_ns, owner_stopped=owner_stopped,
        final_file_published=False, mode=mode,
    )

    def bounded():
        validate_deadline(deadline_ns, mode=mode, directory=path.parent.resolve(),
                          phase="report:" + stage)

    try:
        bounded()
        if not owner_stopped:
            raise ValueError("final report requires a stopped owner")
        stage = "temporary_create"
        fd, temporary = tempfile.mkstemp(
            prefix="." + path.name + ".", suffix=".partial", dir=path.parent)
        state["temporary_file"] = Path(temporary).name
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            stage = "writing_state"
            atomic_write_json(path.with_name(STATE_NAME), state)
            stage = "build"
            value = build()
            bounded()
            stage = "serialize"
            json.dump(
                {**value, "report_publication": dict(
                    schema_version=SCHEMA, state_file=STATE_NAME, owner_stopped=True)},
                handle, indent=2, sort_keys=True, allow_nan=False,
            )
            handle.write("\n")
            stage = "flush"
            handle.flush()
            stage = "fsync"
            with file_context(path, physical_path=temporary, write_kind="immutable_json"):
                os.fsync(handle.fileno())
        bounded()
        stage = "publish"
        # Same-filesystem exclusive publication: never overwrite an existing final.
        os.link(temporary, path)
        state["final_file_published"] = True
        bounded()
        stage = "verify"
        state.update(status="COMPLETE", bytes=path.stat().st_size, sha256=sha256_file(path),
                     completed_ns=time.monotonic_ns())
        bounded()
        stage = "complete_state"
        atomic_write_json(path.with_name(STATE_NAME), state)
        bounded()
        stage = "remove_temporary"
        Path(temporary).unlink()
        return state
    except BaseException as error:
        context = deadline_context(deadline_ns, mode=mode, directory=path.parent.resolve(),
                                   phase="report:" + stage)
        error.report_publication_context = context
        state.update(status="FAILED", error=str(error), exception_type=type(error).__name__,
                     failed_ns=time.monotonic_ns(), failed_stage=stage,
                     report_publication_context=context)
        if hasattr(error, "deadline_context"):
            state["deadline_context"] = error.deadline_context
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
