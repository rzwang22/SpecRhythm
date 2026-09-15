"""Nonblocking, bounded reads of a single-writer absolute phase deadline.

One attempt per supervisor poll: retries never sleep here or replace the last
deadline. Before the first publication ENOENT is governed by the launcher's
existing overall timeout; after publication, unreadability has a separate bound.
"""

import errno
import json
from pathlib import Path

PHASES = {
    name: i
    for i, name in enumerate(
        (
            "scan_setup_and_warmup",
            "scan_window_and_atomic_step",
            "wait_owner",
            "target_fence",
            "target_evidence",
            "target_abort",
            "draft_settle",
            "draft_shutdown",
            "target_final_evidence",
            "diagnostic_final_flush",
            "await_coordinator_exit",
        )
    )
}
PHASES.update(capacity_shutdown=2, startup_failure_cleanup=2)
FIELDS = (
    "schema_version",
    "mode",
    "run_directory",
    "phase",
    "status",
    "start_ns",
    "deadline_ns",
    "settled_requests",
    "end_ns",
)


def read_snapshot(path):
    with Path(path).open("rb") as handle:  # No exists/open race.
        raw = handle.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("phase snapshot exceeds 8 MiB")
    return json.loads(raw)


class PhaseSnapshotReader:
    def __init__(
        self,
        path,
        *,
        mode=None,
        max_stale_ns=1_000_000_000,
        max_retries=20,
        absolute_limit_ns=None,
    ):
        self.path, self.mode = Path(path), mode
        self.absolute_limit_ns = absolute_limit_ns
        self.max_stale_ns, self.max_retries = max_stale_ns, max_retries
        self.last = None
        self.last_valid_ns = None
        self.error_start_ns = None
        self.consecutive = self.total_errors = self.recoveries = 0
        self.events, self.omitted_events = [], 0

    def _event(self, row):
        if len(self.events) < 128:
            self.events.append(row)
        else:
            self.omitted_events += 1

    def report(self):
        return dict(
            path=str(self.path),
            last_valid_state=self.last,
            last_valid_read_ns=self.last_valid_ns,
            retry_count=self.consecutive,
            total_read_errors=self.total_errors,
            recoveries=self.recoveries,
            max_stale_ns=self.max_stale_ns,
            max_retries=self.max_retries,
            events=self.events,
            omitted_events=self.omitted_events,
        )

    def poll(self, now):
        try:
            value = read_snapshot(self.path)
        except (OSError, ValueError, UnicodeError) as error:
            transient = isinstance(error, (json.JSONDecodeError, UnicodeError)) or (
                isinstance(error, OSError)
                and error.errno in (errno.ENOENT, errno.EINTR, errno.EAGAIN, errno.ESTALE)
            )
            self.consecutive += 1
            self.total_errors += 1
            if self.error_start_ns is None:
                self.error_start_ns = now
            detail = dict(
                exception_type=type(error).__name__,
                error=str(error),
                artifact=str(self.path),
                timestamp_ns=now,
                retry_count=self.consecutive,
                transient=transient,
            )
            self._event(detail)
            # Absence before first publication was always legal (non-scan setup).
            # A global startup/total deadline is still checked on every poll.
            initial_absence = self.last is None and isinstance(error, FileNotFoundError)
            expired = (
                self.consecutive >= self.max_retries
                or now - self.error_start_ns >= self.max_stale_ns
            )
            if initial_absence:
                self.consecutive = 0
                self.error_start_ns = None
            if not transient or (expired and not initial_absence):
                return dict(reason="owned phase snapshot unreadable", **detail)
            return self._deadline(now)
        try:
            self._validate(value)
        except (ValueError, TypeError) as error:
            return dict(
                reason="invalid owned phase deadline",
                exception_type=type(error).__name__,
                error=str(error),
                artifact=str(self.path),
                timestamp_ns=now,
                actual={k: value.get(k) for k in FIELDS} if isinstance(value, dict) else None,
            )
        if self.consecutive:
            self.recoveries += 1
            self._event(
                dict(
                    event="read_recovered",
                    timestamp_ns=now,
                    artifact=str(self.path),
                    retries=self.consecutive,
                )
            )
        self.consecutive = 0
        self.error_start_ns = None
        self.last = {k: value[k] for k in FIELDS if k in value}
        self.last_valid_ns = now
        return self._deadline(now)

    def _deadline(self, now):
        if self.last is not None and now >= self.last["deadline_ns"]:
            return dict(
                reason="owned Target execution timeout",
                phase=self.last["phase"],
                deadline_ns=self.last["deadline_ns"],
                artifact=str(self.path),
                timestamp_ns=now,
            )
        return None

    def _validate(self, value):
        if not isinstance(value, dict):
            raise ValueError("phase snapshot must be an object")
        deadline, phase = value.get("deadline_ns"), value.get("phase")
        if type(deadline) is not int or deadline <= 0 or not isinstance(phase, str) or not phase:
            raise ValueError("phase and positive integer absolute deadline required")
        if self.absolute_limit_ns is not None and deadline > self.absolute_limit_ns:
            raise ValueError("phase deadline exceeds total supervisor deadline")
        if self.mode is not None and (
            phase not in PHASES or value.get("status") not in ("RUNNING", "COMPLETE", "FAILED")
        ):
            raise ValueError("unknown phase/status")
        if "start_ns" in value and (
            type(value["start_ns"]) is not int or not 0 < value["start_ns"] <= deadline
        ):
            raise ValueError("invalid phase start/deadline")
        if self.mode is not None and (
            value.get("mode") != self.mode
            or value.get("run_directory") != str(self.path.parent.resolve())
        ):
            raise ValueError("phase snapshot run/mode identity conflict")
        if self.last is None:
            return
        old = self.last
        for key in ("mode", "run_directory"):
            if key in old and value.get(key) != old[key]:
                raise ValueError("phase snapshot identity changed")
        before, after = PHASES.get(old["phase"]), PHASES.get(phase)
        if phase != old["phase"] and (before is None or after is None or after < before):
            raise ValueError("illegal phase transition")
        # Only setup -> window/drain and window -> drain start new transactions.
        new_transaction = before in (0, 1) and after is not None and after > before
        if deadline != old["deadline_ns"] and not new_transaction:
            raise ValueError("absolute deadline changed inside phase transaction")
        if old.get("status") in ("COMPLETE", "FAILED") and value.get("status") != old["status"]:
            raise ValueError("terminal phase snapshot revived")
