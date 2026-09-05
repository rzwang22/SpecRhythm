"""PID-identity checked ownership tracking and Linux orphan adoption."""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional


def process_table() -> dict[int, dict[str, Any]]:
    if sys.platform == "linux":
        rows = {}
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                text = (entry / "stat").read_text()
                end = text.rindex(")")
                fields = text[end + 2:].split()
                pid = int(entry.name)
                rows[pid] = {
                    "pid": pid, "ppid": int(fields[1]), "pgid": int(fields[2]),
                    "session_id": int(fields[3]), "state": fields[0],
                    "start_identity": fields[19], "command": text[text.index("(") + 1:end],
                    "exit_code": (
                        os.waitstatus_to_exitcode(int(fields[49]))
                        if len(fields) > 49 and fields[0] == "Z" else None
                    ),
                }
            except (OSError, ValueError, IndexError):
                continue  # Process exited while the snapshot was read.
        return rows
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,pgid=,sess=,stat=,lstart=,comm="],
        capture_output=True, text=True, check=True,
    )
    rows = {}
    for line in result.stdout.splitlines():
        fields = line.split(None, 10)
        if len(fields) < 10:
            continue
        pid, ppid, pgid, sid = map(int, fields[:4])
        rows[pid] = {
            "pid": pid, "ppid": ppid, "pgid": pgid, "session_id": sid,
            "state": fields[4], "start_identity": " ".join(fields[5:10]),
            "command": fields[10] if len(fields) > 10 else "", "exit_code": None,
        }
    return rows


def set_subreaper(enabled: bool) -> Optional[int]:
    if sys.platform != "linux":
        return None
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_GET_CHILD_SUBREAPER failed")
    if libc.prctl(36, int(enabled), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER failed")
    return previous.value


class OwnedProcesses:
    def __init__(self, root_pid: int, *, target_token: Optional[str] = None) -> None:
        self.root_pid = root_pid
        self.target_token = target_token
        self.observed: dict[int, dict[str, Any]] = {}

    def snapshot(self) -> list[dict[str, Any]]:
        table = process_table()
        reused = {pid for pid, previous in self.observed.items() if pid in table
                  and table[pid]["start_identity"] != previous["start_identity"]}
        owned = {
            pid for pid, previous in self.observed.items()
            if pid in table and table[pid]["start_identity"] == previous["start_identity"]
        }
        root = table.get(self.root_pid)
        if root is not None and self.root_pid not in self.observed:
            owned.add(self.root_pid)
        if self.target_token:
            # The owned session outlives its leader: a child can fork between
            # polls and retain that session after the coordinator is reaped.
            # A known reused root cannot establish ownership of its new group.
            if self.root_pid not in reused:
                owned.update(pid for pid, row in table.items()
                             if pid not in reused and (row["pgid"] == self.root_pid
                                                       or row["session_id"] == self.root_pid))
            marker = ("SR_PHASE4_OWNED_TARGET_TOKEN=" + self.target_token).encode()
            for pid, row in table.items():
                if row["ppid"] != os.getpid() or pid in owned or pid in reused:
                    continue
                try:
                    if marker in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
                        owned.add(pid)  # Adopted detached descendant, proven by launch token.
                except OSError:
                    pass
        while True:
            children = {pid for pid, row in table.items()
                        if row["ppid"] in owned and pid not in reused}
            added = children - owned
            if not added:
                break
            owned.update(added)
        rows = [table[pid] for pid in sorted(owned)]
        for row in rows:
            self.observed.setdefault(row["pid"], dict(row))
        return rows

    def reap(self, *, exclude_root: bool = True) -> None:
        table = process_table()
        for pid, previous in tuple(self.observed.items()):
            if exclude_root and pid == self.root_pid:
                continue
            if pid not in table or table[pid]["start_identity"] != previous["start_identity"]:
                continue
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass

    def signal(self, selected: signal.Signals, actions: list[dict[str, Any]]) -> None:
        for row in reversed(self.snapshot()):
            pid = row["pid"]
            if row["state"].startswith("Z"):
                continue
            # Recheck identity immediately before signaling; never kill by name
            # or by the caller's session/process group (Draft is a shell child).
            current = process_table().get(pid)
            if current is None or current["start_identity"] != row["start_identity"]:
                continue
            try:
                os.kill(pid, selected)
                delivered = True
            except ProcessLookupError:
                delivered = False
            actions.append({"pid": pid, "pgid": row["pgid"], "signal": selected.name,
                            "start_identity": row["start_identity"], "delivered": delivered,
                            "timestamp_ns": time.monotonic_ns()})


def socket_owned_by_pid(path: Path, pid: int) -> bool:
    """Prove the supplied Draft PID has the actual Unix socket open."""
    if sys.platform == "linux":
        try:
            inodes = {
                row.split()[6] for row in Path("/proc/net/unix").read_text().splitlines()[1:]
                if len(row.split()) >= 8 and Path(row.split()[7]).resolve() == path.resolve()
            }
            return any(
                os.readlink(fd) in {f"socket:[{inode}]" for inode in inodes}
                for fd in Path(f"/proc/{pid}/fd").iterdir()
            )
        except OSError:
            return False
    result = subprocess.run(
        ["lsof", "-a", "-p", str(pid), "-U", "-Fn"], capture_output=True, text=True,
    )
    return result.returncode == 0 and any(
        row.startswith("n/") and Path(row[1:]).resolve() == path.resolve()
        for row in result.stdout.splitlines()
    )
