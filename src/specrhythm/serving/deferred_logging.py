"""Opt-in bounded post-run persistence. Overflow fails; it never flushes to make space."""

import hashlib
import json
import time
from collections import defaultdict

from specrhythm.continuation.trace import TRACE
from specrhythm.phase4.drain_deadline import validate_deadline
from specrhythm.phase4.manifest import atomic_write_json, sha256_file
from specrhythm.phase4.transport import canonical_json_bytes, payload_sha256
from specrhythm.serving.common import require
from specrhythm.serving.fixed_logging import DiagnosticLogs

MODE = "deferred-window"
RECORD_LIMIT = 100000
BYTE_LIMIT = 256 * 1024 * 1024
REPORT_LIMIT = 64 * 1024 * 1024


class DeferredLogs(DiagnosticLogs):
    def __init__(self, directory, mode=MODE, role="unknown", *, max_records=RECORD_LIMIT,
                 max_bytes=BYTE_LIMIT, **kwargs):
        self.phases = defaultdict(lambda: dict(produced=0, enqueue_ns=0, encoded_bytes=0,
                                               write_records=0, flushes=0, persistence_ns=0))
        self.report_builders = {}
        self.report_requests = 0
        self.report_receipts = {}
        self.io_phases = {}
        self.emergency_attempted = False
        super().__init__(directory, mode, role, max_records=max_records,
                         max_bytes=max_bytes, **kwargs)

    def _snapshot(self):
        value = super()._snapshot()
        value.update(
            persistence="post-run bounded memory; no capacity flush",
            phase_costs={k: dict(self.phases[k]) for k in
                         ("setup", "warmup", "measurement", "drain")},
            phase_scope="already observed control phase; actual timestamps retained separately",
            deferred_report_requests=self.report_requests,
            deferred_report_slots=len(self.report_builders),
            maximum_report_bytes=REPORT_LIMIT,
            reports=dict(self.report_receipts),
            physical_io_by_phase=dict(self.io_phases),
            memory_scope="encoded pending bytes plus bounded row/path references; "
                         "report builder holds existing proposer, no history copy",
        )
        value["resident_admission_policy"]["persistence"] = "post-run bounded memory"
        return value

    def record_fsync(self, attribution, started, ended):
        phase = "drain" if self.deadline_ns is not None else TRACE.phase
        key = phase + ":" + str(attribution.get("log_name"))
        require(key in self.io_phases or len(self.io_phases) < 256,
                "diagnostic I/O filename budget exhausted")
        row = self.io_phases.setdefault(key, dict(
            phase=phase, **attribution, fsync_calls=0, fsync_ns=0,
            reason="deferred diagnostic finalization" if self.deadline_ns is not None
            else "live control/lifecycle or non-deferred stream; see write_kind"))
        row["fsync_calls"] += 1
        row["fsync_ns"] += ended - started

    def append(self, log, value, native):
        if not self.eligible(log.path, value):
            # Control/ownership/initial readiness retain their existing online behavior.
            return super().append(log, value, native)
        started = time.monotonic_ns()
        phase = self.phases[TRACE.phase]
        with self.lock:
            try:
                require(not self.closed and self.error is None,
                        "diagnostic logger finalized/failed")
                self.produced += 1
                self.streams[log.path.name]["produced"] += 1
                phase["produced"] += 1
                from specrhythm.phase4.admission_record import PreparedAdmission

                if type(value) is PreparedAdmission:
                    line = value.line
                else:
                    payload = dict(value)
                    require("record_sha256" not in payload, "record_sha256 is reserved")
                    payload["record_sha256"] = payload_sha256(payload)
                    line = canonical_json_bytes(payload) + b"\n"
                phase["encoded_bytes"] += len(line)
                require(len(self.pending) < self.max_records
                        and self.pending_bytes + len(line) <= self.max_bytes,
                        "deferred diagnostic buffer exhausted; evidence incomplete")
                self.pending.append((log.path, line))
                self.pending_bytes += len(line)
                self.peak_records = max(self.peak_records, len(self.pending))
                self.peak_bytes = max(self.peak_bytes, self.pending_bytes)
            except BaseException as error:
                self.abort(error)
                raise
            finally:
                elapsed = time.monotonic_ns() - started
                phase["enqueue_ns"] += elapsed
                self.blocking_ns += elapsed

    def read_snapshot(self, log, native):
        # CheckpointJsonl readers retain framing/checksum validation and read-your-writes,
        # without requiring diagnostic persistence. External control readers use other files.
        with self.lock:
            require(self.error is None, "cannot read failed diagnostic log")
            result = native(log)
            for path, line in self.pending:
                if path != log.path:
                    continue
                value = json.loads(line)
                expected = value.pop("record_sha256")
                require(payload_sha256(value) == expected, "pending diagnostic checksum differs")
                result.append({**value, "record_sha256": expected})
            return result

    def defer_report(self, path, build):
        if path.parent != self.directory or path.name != "plugin-report.json":
            return False
        require(not self.closed and self.error is None, "deferred report after finalization")
        require(not self.report_builders or path in self.report_builders,
                "unexpected extra deferred report producer")
        self.report_requests += 1
        self.report_builders[path] = build
        return True

    def _write(self, rows):
        started = time.monotonic_ns()
        phase = self.phases["drain" if self.deadline_ns is not None else TRACE.phase]
        for path, _ in rows:
            if path.name not in self.hashes:
                self.hashes[path.name] = hashlib.sha256()
                if path.exists():
                    with path.open("rb") as previous:
                        for chunk in iter(lambda: previous.read(1024 * 1024), b""):
                            self.hashes[path.name].update(chunk)
        before, flushes = self.written, self.flushes
        try:
            return super()._write(rows)
        finally:
            phase["write_records"] += self.written - before
            phase["flushes"] += self.flushes - flushes
            phase["persistence_ns"] += time.monotonic_ns() - started

    def _flush(self, reason):
        require(reason in ("final", "failure"), "deferred diagnostics cannot flush online")
        super()._flush(reason)
        if reason != "final":
            return
        for path, build in self.report_builders.items():
            start = time.monotonic_ns()
            self._budget()
            value = build()  # Once, after measurement and all Target round callbacks.
            self._budget()
            encoded = canonical_json_bytes(value)
            require(len(encoded) <= REPORT_LIMIT, "deferred final report exceeds byte budget")
            self._budget()
            atomic_write_json(path, value)
            self._budget()
            # Validate the published content as well as its physical bytes.
            require(json.loads(path.read_bytes()) == value, "deferred report reread differs")
            self.report_receipts[path.name] = dict(
                status="COMPLETE", bytes=path.stat().st_size, sha256=sha256_file(path),
                canonical_sha256=hashlib.sha256(encoded).hexdigest(),
                encoded_bytes=len(encoded), start_ns=start, end_ns=time.monotonic_ns(),
                deadline_ns=self.deadline_ns, build_count=1,
            )
            self._budget()

    def finish(self, deadline_ns=None):
        try:
            require(self.deadline_ns in (None, deadline_ns), "diagnostic deadline conflict")
            if not self.closed:
                validate_deadline(deadline_ns, mode=self.role, directory=self.directory,
                                  phase="diagnostic_final_flush")
            return super().finish(deadline_ns)
        except BaseException as error:
            self.abort(error)
            raise

    def abort(self, error):
        # Keep original failure sticky. Best-effort preserve acquired rows only under
        # the existing supervisor deadline. Never retry a partly failed write.
        if not self.emergency_attempted and not self.deadline_ns and self.pending:
            self.emergency_attempted = True
            try:
                from specrhythm.serving.common import read_json

                deadline = read_json(self.directory / "drain-state.json")["deadline_ns"]
                validate_deadline(deadline, mode=self.role, directory=self.directory,
                                  phase="diagnostic_failure_preservation")
                self.deadline_ns = deadline
                self._flush("failure")
            except Exception as secondary:
                self.secondary_errors.append("failure preservation: " + str(secondary))
        super().abort(error)
