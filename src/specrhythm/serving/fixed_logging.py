"""Fixed-only, bounded synchronous batching of explicitly post-run diagnostic records.

Socket/control/ready packets, ownership journals and release/error reports never enter
this buffer. There is no background writer. A final receipt is required for qualification.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import defaultdict
from pathlib import Path

from specrhythm.io_context import IO_CONTEXT as IO_CONTEXT
from specrhythm.io_context import file_context
from specrhythm.phase4.resident_setup import ADMISSION_EVENT_SCHEMA
from specrhythm.phase4.transport import canonical_json_bytes, payload_sha256
from specrhythm.serving.common import read_json, require
from specrhythm.serving.s2_pool import publish

MODES = ("original-live", "buffered-live", "deferred-window")
MAX_RECORDS = 256
MAX_BYTES = 1024 * 1024
# Audited single-process writers. read-your-writes flushes preserve startup readers.
# Initial-proposal/timing records remain synchronous. Resident admission is an
# after-schedule audit, never the in-memory admission predicate or control packet.
POST_RUN_LOGS = frozenset(
    {
        "target-diagnostics.jsonl",
        "round-events.jsonl",
        "transport-events.jsonl",
        "proposal-events.jsonl",
        "verification-events.jsonl",
        "request-state-events.jsonl",
        "scheduler-events.jsonl",
        "proposal-lifecycle-events.jsonl",
        "draft-work-events.jsonl",
        "draft-transport.jsonl",
        "admission-events.jsonl",
    }
)
_CURRENT = None


class DiagnosticLogs:
    def __init__(
        self,
        directory,
        mode="original-live",
        role="unknown",
        *,
        max_records=MAX_RECORDS,
        max_bytes=MAX_BYTES,
        fsync=None,
    ):
        require(mode in MODES, "unknown fixed diagnostic observation", actual=mode)
        require(max_records > 0 and max_bytes > 0, "invalid diagnostic log buffer bounds")
        self.directory, self.mode, self.role = Path(directory), mode, role
        self.pid = os.getpid()
        self.max_records, self.max_bytes = max_records, max_bytes
        self.fsync = fsync or (lambda fd: os.fsync(fd))
        self.lock = threading.RLock()
        self.pending, self.pending_bytes = [], 0
        self.produced = self.written = self.flushes = self.fsyncs = 0
        self.peak_records = self.peak_bytes = self.oversized_records = 0
        self.blocking_ns = self.flush_ns = 0
        self.streams = defaultdict(lambda: {"produced": 0, "written": 0})
        self.hashes = {}
        self.native_streams = set()
        self.flush_reasons = defaultdict(int)
        self.secondary_errors = []
        self.error = None
        self.closed = False
        self.final_flush = None
        self.deadline_ns = None
        self.receipt_path = self.directory / f"fixed-logging-{self.pid}.json"
        self._publish("RUNNING")

    def _budget(self):
        if self.deadline_ns is not None:
            from specrhythm.serving.fixed_settle import remaining

            remaining(self.deadline_ns)

    def _publish(self, status):
        publish(
            self.receipt_path,
            {
                **self.snapshot(),
                "status": status,
                "counts_final": status in ("COMPLETE", "FAILED"),
            },
        )

    def snapshot(self):
        with self.lock:
            return self._snapshot()

    def _snapshot(self):
        return {
            "schema_version": "specrhythm.fixed-logging.v1",
            "pid": self.pid,
            "role": self.role,
            "observation": self.mode,
            "resident_admission_policy": {
                "version": "serial-resident-audit-buffer.v1",
                "schema": ADMISSION_EVENT_SCHEMA,
                "consumer": "serial",
                "persistence": "bounded-batches" if self.mode == "buffered-live" else "per-record",
                "other_records": "native; flush earlier same-file records before append",
            },
            "max_buffer_records": self.max_records,
            "max_buffer_bytes": self.max_bytes,
            "produced_records": self.produced,
            "written_records": self.written,
            "pending_records": len(self.pending),
            "pending_bytes": self.pending_bytes,
            "flush_count": self.flushes,
            "fsync_count": self.fsyncs,
            "peak_buffer_records": self.peak_records,
            "peak_buffer_bytes": self.peak_bytes,
            "oversized_direct_records": self.oversized_records,
            "append_blocking_ms": self.blocking_ns / 1e6,
            "flush_blocking_ms": self.flush_ns / 1e6,
            "streams": {
                k: {
                    **v,
                    "written_bytes_sha256": self.hashes[k].hexdigest()
                    if k in self.hashes and k not in self.native_streams
                    else None,
                    "digest_scope": (
                        "native framing; stream digest not collected"
                        if k in self.native_streams else "buffered bytes after install"),
                }
                for k, v in self.streams.items()
            },
            "final_flush": self.final_flush,
            "error": self.error,
            "secondary_errors": self.secondary_errors[:8],
            "flush_reasons": dict(self.flush_reasons),
            "integrity_complete": self.closed
            and self.error is None
            and self.produced == self.written
            and not self.pending,
            "coverage": "CheckpointJsonl appends after observer installation; "
            "earlier native writes excluded from counters/digests",
            "checksum_semantics": "original checksummed bytes and per-file append order unchanged",
            "uuid_query_mode": "live",
            "background_writer": False,
        }

    def eligible(self, path, value=None):
        if path.parent != self.directory or path.name not in POST_RUN_LOGS:
            return False
        # None is a read-your-writes query. Only this audited producer schema is
        # batched; similarly named control/ownership records stay synchronous.
        return path.name != "admission-events.jsonl" or value is None or (
            value.get("schema_version") == ADMISSION_EVENT_SCHEMA
            and value.get("consumer") == "serial"
        )

    def append(self, log, value, native):
        start = time.monotonic_ns()
        with self.lock:
            try:
                require(
                    not self.closed and self.error is None,
                    "diagnostic logger already finalized or failed",
                )
                key = (log.path.name if log.path.parent == self.directory
                       else "outside-point-directory")
                self.produced += 1
                self.streams[key]["produced"] += 1
                if self.mode == "original-live" or not self.eligible(log.path, value):
                    if any(path == log.path for path, _ in self.pending):
                        self._flush("unbuffered-same-file")
                    with file_context(log.path):
                        native(log, value)  # Preserve per-record write/flush/fsync.
                    self.native_streams.add(key)
                    self.written += 1
                    self.streams[key]["written"] += 1
                    self.fsyncs += 1
                    return
                from specrhythm.phase4.admission_record import PreparedAdmission

                if type(value) is PreparedAdmission:
                    line = value.line
                else:
                    payload = dict(value)
                    require(
                        "record_sha256" not in payload,
                        "record_sha256 is reserved for checkpoint framing",
                    )
                    payload["record_sha256"] = payload_sha256(payload)
                    line = canonical_json_bytes(payload) + b"\n"
                if self.pending and (
                    len(self.pending) >= self.max_records
                    or self.pending_bytes + len(line) > self.max_bytes
                ):
                    self._flush("capacity")
                if len(line) > self.max_bytes:
                    self.oversized_records += 1
                    self._write([(log.path, line)])  # One oversized record, never queued.
                    return
                self.pending.append((log.path, line))
                self.pending_bytes += len(line)
                self.peak_records = max(self.peak_records, len(self.pending))
                self.peak_bytes = max(self.peak_bytes, self.pending_bytes)
                if len(self.pending) == self.max_records or self.pending_bytes == self.max_bytes:
                    self._flush("capacity")
            except BaseException as error:
                self.abort(error)
                raise
            finally:
                self.blocking_ns += time.monotonic_ns() - start

    def _write(self, rows):
        # Per-file bytes retain original order. No concatenated whole-buffer copy.
        handles = {}
        self._budget()
        started = time.monotonic_ns()
        failure = None
        try:
            for path, line in rows:
                if path not in handles:
                    handles[path] = path.open("ab")
                handle = handles[path]
                require(handle.write(line) == len(line), "short diagnostic log write")
            for path, handle in handles.items():
                self._budget()
                handle.flush()
                with file_context(path, write_kind="buffered_checkpoint_jsonl"):
                    self.fsync(handle.fileno())
                self.fsyncs += 1
                self._budget()
            for handle in handles.values():
                handle.close()
            # Counts/digests advance only after physical flush/fsync/close succeeds.
            for path, line in rows:
                self.written += 1
                self.streams[path.name]["written"] += 1
                if path.name not in self.native_streams:
                    self.hashes.setdefault(path.name, hashlib.sha256()).update(line)
            self.flushes += 1
        except BaseException as error:
            failure = error
            raise
        finally:
            for handle in handles.values():
                try:
                    handle.close()
                except Exception as secondary:
                    if failure is None:
                        raise
                    self.secondary_errors.append(str(secondary))
            self.flush_ns += time.monotonic_ns() - started

    def _flush(self, reason):
        if not self.pending:
            return
        self._write(self.pending)
        self.flush_reasons[reason] += 1
        self.pending.clear()
        self.pending_bytes = 0

    def before_read(self, log):
        if self.mode == "buffered-live" and self.eligible(log.path):
            with self.lock:
                try:
                    require(self.error is None, "cannot read a failed diagnostic log")
                    self._flush("read-your-writes")
                except BaseException as error:
                    self.abort(error)
                    raise

    def finish(self, deadline_ns=None):
        with self.lock:
            if self.closed:
                require(self.error is None, "diagnostic logger finalization previously failed")
                return self.snapshot()
            self.deadline_ns = deadline_ns
            start = time.monotonic_ns()
            try:
                require(self.error is None, "cannot finalize a failed diagnostic log")
                if self.mode == "buffered-live":
                    require(
                        type(deadline_ns) is int,
                        "buffered final flush requires shared drain deadline",
                    )
                    self._budget()
                before = self.written
                self._flush("final")
                self._budget()
                self.final_flush = {
                    "start_ns": start,
                    "end_ns": time.monotonic_ns(),
                    "records": self.written - before,
                    "deadline_ns": deadline_ns,
                }
                self.closed = True
                self._publish("COMPLETE")
                return self.snapshot()
            except BaseException as error:
                self.abort(error)
                raise

    def abort(self, error):
        if self.error is None:
            self.error = f"{type(error).__name__}: {error}"
        try:
            self._publish("FAILED")
        except Exception as secondary:
            print(f"[fixed logging secondary] {secondary}; primary={self.error}", flush=True)
        from specrhythm.serving.fixed_artifacts import record_error

        record_error(self.directory, error, "diagnostic_logging")


def current(role=None):
    global _CURRENT
    if not os.environ.get("SR_FIXED_POINT"):
        return None
    if _CURRENT is None or _CURRENT.pid != os.getpid():
        factory = DiagnosticLogs
        if os.environ.get("SR_FIXED_OBSERVATION") == "deferred-window":
            from specrhythm.serving.deferred_logging import DeferredLogs

            factory = DeferredLogs
        _CURRENT = factory(
            Path(os.environ["SR_FIXED_POINT"]).parent,
            os.environ.get("SR_FIXED_OBSERVATION", "original-live"),
            role or "unknown",
        )
    if role and _CURRENT.role != role:
        _CURRENT.role = role
        _CURRENT._publish("RUNNING")
    if _CURRENT.mode == "deferred-window":
        from specrhythm.diagnostic_report import install

        install(_CURRENT.defer_report)
    return _CURRENT


def buffered():
    return os.environ.get("SR_FIXED_OBSERVATION", "original-live") in (
        "buffered-live", "deferred-window")


def finish_current(role=None):
    logs = current(role)
    if logs is None:
        return None
    path = logs.directory / "drain-state.json"
    deadline = read_json(path)["deadline_ns"] if path.exists() else None
    return logs.finish(deadline)


def finish_worker(worker):
    # CPU/file operations only; Target has already been fenced and aborted by drain.
    return finish_current("target-rank-" + str(worker.rank))


def wait_draft_receipt(directory, deadline):
    from specrhythm.serving.fixed_settle import remaining

    while True:
        remaining(deadline)
        for path in directory.glob("fixed-logging-*.json"):
            value = read_json(path)
            if value["role"] != "draft":
                continue
            require(value["status"] != "FAILED", "Draft diagnostic log flush failed", actual=value)
            if value["status"] == "COMPLETE":
                require(
                    value["integrity_complete"], "Draft log integrity incomplete", actual=value
                )
                return value
        time.sleep(min(0.01, remaining(deadline)))  # Drain only; never a model/overlap delay.


def finalize_drain(llm, directory, deadline):
    """All four receipts before arrival-to-drain end, under the existing supervisor deadline."""
    if not buffered():
        return None
    from specrhythm.serving.fixed_settle import remaining

    draft = wait_draft_receipt(directory, deadline)
    workers = llm.collective_rpc(finish_worker, timeout=remaining(deadline))
    coordinator = finish_current("coordinator")
    remaining(deadline)
    return {"draft": draft, "target_workers": workers, "coordinator": coordinator}


def qualify(directory, observation):
    """Small final receipts only. Raw-log checksum audit remains an explicit command."""
    receipts = [read_json(p) for p in sorted(directory.glob("fixed-logging-*.json"))]
    result = {
        "observation": observation,
        "receipts": receipts,
        "finalization_required": observation != "original-live",
    }
    if observation == "original-live":
        return result  # Old immutable artifacts predate receipts; native per-record fsync.
    require(observation in ("buffered-live", "deferred-window"),
            "unknown diagnostic log configuration")
    require(
        sorted(r["role"] for r in receipts)
        == ["coordinator", "draft", "target-rank-0", "target-rank-1"],
        "missing/duplicated diagnostic log completion receipt",
        artifact=str(directory / "fixed-logging-*.json"),
        actual=receipts,
    )
    drain = read_json(directory / "drain-state.json")
    for row in receipts:
        final = row.get("final_flush") or {}
        require(
            row.get("status") == "COMPLETE"
            and row.get("integrity_complete") is True
            and row.get("counts_final") is True
            and row.get("error") is None
            and row["observation"] == observation
            and row["produced_records"] == row["written_records"]
            and row["pending_records"] == row["pending_bytes"] == 0
            and row["peak_buffer_records"] <= row["max_buffer_records"]
            and row["peak_buffer_bytes"] <= row["max_buffer_bytes"]
            and final.get("deadline_ns") == drain["deadline_ns"]
            and drain["start_ns"]
            <= final.get("start_ns", -1)
            <= final.get("end_ns", -1)
            <= drain["end_ns"]
            <= drain["deadline_ns"],
            "diagnostic log integrity/final flush incomplete",
            artifact=str(directory / f"fixed-logging-{row['pid']}.json"),
            actual=row,
        )
    if observation == "deferred-window":
        from specrhythm.phase4.manifest import sha256_file

        for receipt in receipts:
            require(set(receipt["published_files"]) == set(receipt["deferred_streams"]),
                    "deferred stream publication receipt missing")
            for name, raw in receipt["published_files"].items():
                require((directory / name).stat().st_size == raw["bytes"]
                        and sha256_file(directory / name) == raw["sha256"],
                        "deferred stream checksum differs", artifact=name)
            require(receipt["deferred_report_slots"] == len(receipt["reports"]),
                    "deferred report producer lacks final receipt")
            if receipt["deferred_report_requests"]:
                require(set(receipt["reports"]) == {"plugin-report.json"},
                        "deferred plugin report missing")
            for name, stream in receipt["streams"].items():
                digest = stream["written_bytes_sha256"]
                if digest is not None:
                    require(sha256_file(directory / name) == digest,
                            "deferred log physical checksum differs", artifact=name)
            for name, report in receipt["reports"].items():
                require(report["status"] == "COMPLETE"
                        and report["deadline_ns"] == drain["deadline_ns"]
                        and report["end_ns"] <= drain["deadline_ns"]
                        and (directory / name).stat().st_size == report["bytes"]
                        and sha256_file(directory / name) == report["sha256"],
                        "deferred report incomplete", artifact=name)
    return result


def install_checkpoint_logging(role, capture, timers):
    """Same CPU adapter called by the observed real child/worker startup."""
    from specrhythm.phase4.transport import CheckpointJsonl

    logs = current(role)
    append, read = CheckpointJsonl.append, CheckpointJsonl.read
    if getattr(append, "_fixed_checkpoint_logging", False):
        return logs

    def retain(log, row):
        with timers.span("checkpoint_log_write", log_name=log.path.name):
            current().append(log, row, append)
        capture(row)  # Contract validation is immediate, not deferred with persistence.

    def read_visible(log):
        logs = current()
        if logs.mode == "deferred-window":
            return logs.read_snapshot(log, read)
        logs.before_read(log)
        return read(log)

    retain._fixed_checkpoint_logging = True
    CheckpointJsonl.append, CheckpointJsonl.read = retain, read_visible
    return logs


def abort_current(error):
    # Never reconstruct a logger whose initialization itself failed and mask that error.
    if _CURRENT is not None and _CURRENT.pid == os.getpid():
        _CURRENT.abort(error)
