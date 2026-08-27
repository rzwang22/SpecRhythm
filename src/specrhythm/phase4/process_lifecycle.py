"""Owned process-group lifecycle management for Phase-4B server runs."""

from __future__ import annotations

import argparse
import json
import os
import signal
import stat
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

LIFECYCLE_SCHEMA = "specrhythm.phase4b-process-lifecycle.v1"


def process_group_members(pgid: int) -> list[dict[str, Any]]:
    completed = subprocess.run(
        ("ps", "-axo", "pid=,pgid=,sess=,comm="),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("cannot enumerate owned process group: " + completed.stderr)
    rows = []
    for line in completed.stdout.splitlines():
        fields = line.strip().split(None, 3)
        if len(fields) < 3:
            continue
        try:
            pid, row_pgid, sid = (int(value) for value in fields[:3])
        except ValueError:
            continue
        if row_pgid == pgid:
            rows.append(
                {
                    "pid": pid,
                    "pgid": row_pgid,
                    "session_id": sid,
                    "command": fields[3] if len(fields) == 4 else "",
                }
            )
    return sorted(rows, key=lambda row: int(row["pid"]))


def run_owned_target(
    command: Sequence[str],
    *,
    target_log: Path,
    artifact_path: Path,
    guard_path: Optional[Path] = None,
    draft_pid: Optional[int] = None,
    draft_socket: Optional[Path] = None,
    graceful_seconds: float = 5.0,
    kill_seconds: float = 2.0,
    poll_seconds: float = 0.05,
) -> tuple[int, dict[str, Any]]:
    """Run one Target in an owned session and fail on leaked descendants."""

    if not command:
        raise ValueError("Target command is empty")
    for name, value in (
        ("graceful_seconds", graceful_seconds),
        ("kill_seconds", kill_seconds),
        ("poll_seconds", poll_seconds),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    guard = guard_path or artifact_path.with_suffix(artifact_path.suffix + ".active")
    _acquire_guard(guard)
    target_log.parent.mkdir(parents=True, exist_ok=True)
    started_ns = time.monotonic_ns()
    started_at = datetime.now(timezone.utc).isoformat()
    actions: list[dict[str, Any]] = []
    observed: dict[int, dict[str, Any]] = {}
    coordinator_status: Optional[int] = None
    coordinator_reaped = False
    process: Optional[subprocess.Popen[Any]] = None
    pgid: Optional[int] = None
    session_id: Optional[int] = None
    launch_error: Optional[str] = None
    with target_log.open("w", encoding="utf-8") as target_handle:
        try:
            process = subprocess.Popen(
                list(command),
                stdout=target_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            pgid = os.getpgid(process.pid)
            session_id = os.getsid(process.pid)
            if pgid != process.pid or session_id != process.pid:
                raise RuntimeError("Target did not enter its recorded owned session")
            while process.poll() is None:
                _record_members(observed, process_group_members(pgid))
                time.sleep(poll_seconds)
            coordinator_status = process.wait()
            coordinator_reaped = True
            _record_members(observed, process_group_members(pgid))
        except Exception as error:
            launch_error = str(error)
            if process is not None and coordinator_status is None:
                coordinator_status = process.poll()

    remaining = process_group_members(pgid) if pgid is not None else []
    leaked_after_coordinator_exit = bool(remaining)
    if coordinator_status not in {0, None} or remaining or launch_error:
        if pgid is not None and remaining:
            _signal_group(pgid, signal.SIGTERM, actions)
            remaining = _wait_for_group_empty(pgid, graceful_seconds, poll_seconds, observed)
        if pgid is not None and remaining:
            _signal_group(pgid, signal.SIGKILL, actions)
            remaining = _wait_for_group_empty(pgid, kill_seconds, poll_seconds, observed)
        if process is not None and not coordinator_reaped:
            try:
                coordinator_status = process.wait(timeout=kill_seconds)
                coordinator_reaped = True
            except subprocess.TimeoutExpired:
                pass
    draft_shutdown = _cleanup_draft(
        draft_pid,
        draft_socket,
        force=bool(coordinator_status not in {0, None} or leaked_after_coordinator_exit),
        graceful_seconds=graceful_seconds,
        poll_seconds=poll_seconds,
    )
    remaining = process_group_members(pgid) if pgid is not None else []
    cleanup_valid = (
        launch_error is None
        and coordinator_reaped
        and not leaked_after_coordinator_exit
        and not remaining
        and draft_shutdown["valid"]
    )
    run_valid = cleanup_valid and coordinator_status == 0 and not leaked_after_coordinator_exit
    effective_status = coordinator_status if coordinator_status not in {None, 0} else 0
    if effective_status == 0 and not run_valid:
        effective_status = 125
    ended_ns = time.monotonic_ns()
    report = {
        "schema_version": LIFECYCLE_SCHEMA,
        "launch_method": "python-subprocess-start_new_session",
        "setsid_wrapper_used": False,
        "command": list(command),
        "coordinator_pid": process.pid if process is not None else None,
        "pgid": pgid,
        "session_id": session_id,
        "owned_processes_observed": [observed[pid] for pid in sorted(observed)],
        "owned_process_ids": sorted(observed),
        "start_timestamp": started_at,
        "start_monotonic_ns": started_ns,
        "exit_timestamp": datetime.now(timezone.utc).isoformat(),
        "exit_monotonic_ns": ended_ns,
        "term_kill_actions": actions,
        "child_reap_result": {
            "coordinator_reaped": coordinator_reaped,
            "owned_group_empty": not remaining,
            "wrapper_exited_with_descendants_alive": leaked_after_coordinator_exit,
        },
        "target_exit_status": coordinator_status,
        "effective_exit_status": effective_status,
        "draft_shutdown_result": draft_shutdown,
        "remaining_owned_pids": [int(row["pid"]) for row in remaining],
        "launch_error": launch_error,
        "cleanup_valid": cleanup_valid,
        "run_valid": run_valid,
    }
    _atomic_json(artifact_path, report)
    if cleanup_valid:
        guard.unlink(missing_ok=True)
    return int(effective_status), report


def validate_lifecycle_artifact(value: Mapping[str, Any]) -> list[str]:
    errors = []
    if value.get("schema_version") != LIFECYCLE_SCHEMA:
        errors.append("unsupported lifecycle schema")
    if not isinstance(value.get("coordinator_pid"), int):
        errors.append("coordinator PID is missing")
    if value.get("pgid") != value.get("coordinator_pid"):
        errors.append("coordinator does not own the process group")
    if value.get("session_id") != value.get("coordinator_pid"):
        errors.append("coordinator does not own the session")
    reap = value.get("child_reap_result")
    if not isinstance(reap, Mapping) or reap.get("coordinator_reaped") is not True:
        errors.append("coordinator was not reaped")
    if value.get("remaining_owned_pids"):
        errors.append("owned processes remain alive")
    if value.get("cleanup_valid") is not True:
        errors.append("cleanup is invalid")
    return errors


def _record_members(
    observed: dict[int, dict[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> None:
    for row in rows:
        pid = int(row["pid"])
        observed.setdefault(pid, dict(row))


def _signal_group(pgid: int, selected: signal.Signals, actions: list[dict[str, Any]]) -> None:
    try:
        os.killpg(pgid, selected)
        delivered = True
    except ProcessLookupError:
        delivered = False
    actions.append(
        {
            "signal": selected.name,
            "pgid": pgid,
            "timestamp_ns": time.monotonic_ns(),
            "delivered": delivered,
        }
    )


def _wait_for_group_empty(
    pgid: int,
    timeout: float,
    poll_seconds: float,
    observed: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    rows = process_group_members(pgid)
    while rows and time.monotonic() < deadline:
        _record_members(observed, rows)
        time.sleep(poll_seconds)
        rows = process_group_members(pgid)
    return rows


def _cleanup_draft(
    draft_pid: Optional[int],
    draft_socket: Optional[Path],
    *,
    force: bool,
    graceful_seconds: float,
    poll_seconds: float,
) -> dict[str, Any]:
    if draft_pid is None:
        return {"required": False, "valid": True, "pid": None}
    socket_before = _unix_socket_identity(draft_socket)
    was_alive = _pid_alive(draft_pid)
    signaled = False
    if force and was_alive:
        try:
            os.kill(draft_pid, signal.SIGTERM)
            signaled = True
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + graceful_seconds
    while _pid_alive(draft_pid) and time.monotonic() < deadline:
        time.sleep(poll_seconds)
    alive = _pid_alive(draft_pid)
    socket_after_exit = _unix_socket_identity(draft_socket)
    removed_stale_socket = False
    socket_cleanup_error = None
    if not alive and socket_after_exit is not None:
        if socket_before is None:
            socket_cleanup_error = (
                "Draft socket appeared after ownership was first observed"
            )
        elif socket_before != socket_after_exit:
            socket_cleanup_error = "Draft socket identity changed during cleanup"
        elif socket_after_exit["is_socket"] is not True:
            socket_cleanup_error = "owned Draft path is not a Unix socket"
        else:
            try:
                assert draft_socket is not None
                draft_socket.unlink()
                removed_stale_socket = True
            except OSError as error:
                socket_cleanup_error = str(error)
    socket_final = _unix_socket_identity(draft_socket)
    return {
        "required": True,
        "pid": draft_pid,
        "was_alive": was_alive,
        "term_sent": signaled,
        "alive_after_cleanup": alive,
        "socket_path": str(draft_socket) if draft_socket is not None else None,
        "socket_identity_before_cleanup": socket_before,
        "socket_identity_after_process_exit": socket_after_exit,
        "stale_owned_socket_removed": removed_stale_socket,
        "socket_cleanup_error": socket_cleanup_error,
        "socket_exists_after_cleanup": socket_final is not None,
        "reaped_by_calling_shell": not alive,
        "valid": not alive and socket_final is None and socket_cleanup_error is None,
    }


def _unix_socket_identity(path: Optional[Path]) -> Optional[dict[str, Any]]:
    if path is None:
        return None
    try:
        value = path.lstat()
    except FileNotFoundError:
        return None
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "is_socket": stat.S_ISSOCK(value.st_mode),
    }


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    completed = subprocess.run(
        ("ps", "-o", "stat=", "-p", str(pid)),
        capture_output=True,
        text=True,
        check=False,
    )
    state = completed.stdout.strip()
    return bool(state) and not state.startswith("Z")


def _acquire_guard(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(
            f"previous Phase-4B cleanup is incomplete; remove only after audit: {path}"
        ) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(f"owner_pid={os.getpid()}\n")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--target-log", required=True)
    parser.add_argument("--guard")
    parser.add_argument("--draft-pid", type=int)
    parser.add_argument("--draft-socket")
    parser.add_argument("--graceful-seconds", type=float, default=5.0)
    parser.add_argument("--kill-seconds", type=float, default=2.0)
    parser.add_argument("--poll-seconds", type=float, default=0.05)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    status, _ = run_owned_target(
        command,
        target_log=Path(args.target_log).resolve(),
        artifact_path=Path(args.artifact).resolve(),
        guard_path=Path(args.guard).resolve() if args.guard else None,
        draft_pid=args.draft_pid,
        draft_socket=Path(args.draft_socket).resolve() if args.draft_socket else None,
        graceful_seconds=args.graceful_seconds,
        kill_seconds=args.kill_seconds,
        poll_seconds=args.poll_seconds,
    )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
