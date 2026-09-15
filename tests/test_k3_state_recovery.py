"""Real owned CPU children plus deterministic read faults, never simulated PASS fields."""

import errno
import json
import os
import sys
import time

import pytest

from specrhythm.phase4 import state_snapshot
from specrhythm.phase4.process_lifecycle import run_owned_target
from specrhythm.serving.fixed_artifacts import retained_report
from specrhythm.serving.s2_pool import publish


def snapshot(path, deadline, phase="draft_settle"):
    value = dict(
        phase=phase,
        status="RUNNING",
        deadline_ns=deadline,
        mode="serial-k3",
        run_directory=str(path.parent.resolve()),
    )
    publish(path, value)
    return value


CHILD = """
import json, pathlib, signal, sys, time
from specrhythm.serving.fixed_artifacts import record_error
p=pathlib.Path(sys.argv[1])
def stop(*args):
    decision=json.loads((p/'supervisor-decision.json').read_text())
    (p/'signal-observed.json').write_text(json.dumps(dict(
        signal_ns=time.monotonic_ns(), decision=decision)))
    # Deterministic shared-clock injection for this downstream error only.
    # macOS Python3.9 uses a per-process epoch; Linux production uses shared
    # monotonic time. The real pre-signal decision/action comparison is in parent.
    clock=time.monotonic_ns
    time.monotonic_ns=lambda: decision['decision_ns']+1
    record_error(p, BrokenPipeError('injected downstream Broken pipe'), 'draft_socket')
    time.monotonic_ns=clock
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
(p/'child-ready').touch()
with (p/'release').open() as f:
    command=f.read(1)
if command=='F':
    print('WorkerProc failed: injected real worker log', flush=True)
    signal.pause()
"""


@pytest.mark.parametrize("fault", ["enoent", "json", "missing", "corrupt", "worker"])
def test_live_supervisor_read_recovery_or_bounded_failure(tmp_path, monkeypatch, fault):
    path = tmp_path / "drain-state.json"
    deadline = time.monotonic_ns() + 10_000_000_000
    snapshot(path, deadline)
    os.mkfifo(tmp_path / "release")
    original = state_snapshot.read_snapshot
    count, released = 0, False

    def send(command):
        nonlocal released
        # Child created child-ready before opening the FIFO; blocking rendezvous
        # ensures the command is delivered, with no arbitrary scheduling sleep.
        with (tmp_path / "release").open("w") as out:
            out.write(command)
        released = True

    def read(p):
        nonlocal count
        if not (tmp_path / "child-ready").exists():
            return original(p)
        count += 1
        if count == 1:
            return original(p)  # Establish the last valid deadline before fault.
        if count == 2 or fault in ("missing", "corrupt", "worker"):
            if fault == "worker" and not released:
                send("F")
            if fault in ("json", "corrupt"):
                raise json.JSONDecodeError("injected short read", "{", 1)
            raise FileNotFoundError(errno.ENOENT, "injected transient", str(p))
        result = original(p)
        if not released:
            send("X")
        return result

    monkeypatch.setattr(state_snapshot, "read_snapshot", read)
    code, life = run_owned_target(
        [sys.executable, "-c", CHILD, str(tmp_path)],
        target_log=tmp_path / "target.log",
        artifact_path=tmp_path / "process-lifecycle.json",
        phase_deadline_path=path,
        run_mode="serial-k3",
        timeout_seconds=15,
        poll_seconds=0.01,
    )
    reads = life["phase_state_reads"]
    assert reads["last_valid_state"]["deadline_ns"] == deadline
    assert life["launch_error"] is None and life["remaining_owned_pids"] == []
    if fault in ("enoent", "json"):
        assert code == 0 and life["run_valid"] and life["cleanup_valid"]
        assert life["term_kill_actions"] == [] and reads["recoveries"] == 1
        assert reads["total_read_errors"] == 1
    else:
        assert code == 125 and not life["cleanup_valid"] and life["owned_cleanup_completed"]
        reason = "fatal-runtime-log" if fault == "worker" else "owned phase snapshot unreadable"
        assert life["failure_detection"]["reason"] == reason
        decision = json.loads((tmp_path / "supervisor-decision.json").read_text())
        acknowledgment = json.loads((tmp_path / "signal-observed.json").read_text())
        assert decision["decision_ns"] <= life["term_kill_actions"][0]["timestamp_ns"]
        assert acknowledgment["decision"] == decision  # Child read it in its signal handler.
        assert decision["observed_processes"] and decision["mode"] == "serial-k3"
        report = retained_report(tmp_path, {"valid": False})
        assert report["primary_error"]["phase"] == "supervisor"
        assert "Broken pipe" in report["ordered_failures"][1]["error"]
        assert report["failure_layer"] == "supervisor"


@pytest.mark.parametrize("fault", ["missing", "json"])
def test_absolute_deadline_wins_during_retry_and_never_moves(tmp_path, monkeypatch, fault):
    path = tmp_path / "drain-state.json"
    snapshot(path, 100)
    reader = state_snapshot.PhaseSnapshotReader(path, mode="serial-k3")
    assert reader.poll(50) is None
    if fault == "missing":
        path.unlink()
    else:
        path.write_text('{"deadline_ns":')
    assert reader.poll(60) is None
    result = reader.poll(101)
    assert result["reason"] == "owned Target execution timeout"
    assert result["deadline_ns"] == reader.last["deadline_ns"] == 100


@pytest.mark.parametrize(
    "mutation",
    [
        {"deadline_ns": True},
        {"deadline_ns": 100.0},
        {"deadline_ns": -1},
        {"deadline_ns": 101},
        {"phase": "target_abort"},
        {"mode": "pingpong-k3"},
        {"run_directory": "/other"},
        {"status": "typo"},
    ],
)
def test_invalid_state_is_not_retried(tmp_path, mutation):
    path = tmp_path / "drain-state.json"
    value = snapshot(path, 100)
    reader = state_snapshot.PhaseSnapshotReader(path, mode="serial-k3")
    assert reader.poll(1) is None
    publish(path, {**value, **mutation})
    assert reader.poll(2)["reason"] == "invalid owned phase deadline"
    assert reader.report()["total_read_errors"] == 0


def test_permission_fails_immediately_and_initial_absence_stays_bounded(tmp_path, monkeypatch):
    path = tmp_path / "drain-state.json"
    reader = state_snapshot.PhaseSnapshotReader(path, mode="serial-k3")
    for now in range(30):
        assert reader.poll(now) is None  # Outer supervisor startup/total budget owns this case.
    monkeypatch.setattr(
        state_snapshot,
        "read_snapshot",
        lambda p: (_ for _ in ()).throw(PermissionError(errno.EACCES, "injected denied", str(p))),
    )
    failure = reader.poll(31)
    assert failure["exception_type"] == "PermissionError" and not failure["transient"]
    # Actual missing-file supervisor still reaches its existing total timeout.
    monkeypatch.undo()
    code, life = run_owned_target(
        [sys.executable, "-c", "import signal; signal.pause()"],
        target_log=tmp_path / "target.log",
        artifact_path=tmp_path / "process-lifecycle.json",
        phase_deadline_path=path,
        timeout_seconds=0.15,
        poll_seconds=0.01,
    )
    assert code == 124 and life["phase_state_reads"]["last_valid_state"] is None
    assert life["owned_cleanup_completed"]


def test_state_changes_keep_transaction_deadline_and_terminal_state(tmp_path):
    path = tmp_path / "drain-state.json"
    value = snapshot(path, 100, "scan_setup_and_warmup")
    reader = state_snapshot.PhaseSnapshotReader(path, mode="serial-k3", absolute_limit_ns=200)
    assert reader.poll(1) is None
    value.update(phase="wait_owner", deadline_ns=150)
    publish(path, value)
    assert reader.poll(2) is None
    value.update(phase="await_coordinator_exit", status="COMPLETE")
    publish(path, value)
    assert reader.poll(3) is None
    value.update(status="RUNNING")
    publish(path, value)
    assert "revived" in reader.poll(4)["error"]


def test_clean_target_exit_waits_for_draft_report_on_original_deadline(tmp_path, monkeypatch):
    import subprocess

    from specrhythm.phase4.report_publication import qualify_final_report

    path = tmp_path / "drain-state.json"
    deadline = time.monotonic_ns() + 10_000_000_000
    snapshot(path, deadline, "await_coordinator_exit")
    os.mkfifo(tmp_path / "release")
    draft = subprocess.Popen(
        [
            sys.executable,
            "-c",
            """
import pathlib, sys
from specrhythm.phase4.report_publication import publish_final_report
p=pathlib.Path(sys.argv[1])
with (p/'release').open() as f: f.read(1)
publish_final_report(p/'draft-backend-report.json', lambda: {'CPU': True},
    deadline_ns=int(sys.argv[2]), owner_stopped=True)
""",
            str(tmp_path),
            str(deadline),
        ],
        start_new_session=True,
    )
    read, released = state_snapshot.read_snapshot, False

    def inject(p):
        nonlocal released
        value = read(p)
        if (tmp_path / "target-finished").exists() and not released:
            with (tmp_path / "release").open("w") as out:
                out.write("X")
            released = True
        return value

    monkeypatch.setattr(state_snapshot, "read_snapshot", inject)
    try:
        code, life = run_owned_target(
            [
                sys.executable,
                "-c",
                "import pathlib,sys;pathlib.Path(sys.argv[1]).touch()",
                str(tmp_path / "target-finished"),
            ],
            target_log=tmp_path / "target.log",
            artifact_path=tmp_path / "process-lifecycle.json",
            phase_deadline_path=path,
            run_mode="serial-k3",
            draft_pid=draft.pid,
            timeout_seconds=15,
            poll_seconds=0.01,
        )
        assert code == 0 and released and draft.wait(timeout=5) == 0
        assert (
            qualify_final_report(tmp_path / "draft-backend-report.json")["deadline_ns"] == deadline
        )
        assert life["run_valid"] and life["term_kill_actions"] == []
    finally:
        if draft.poll() is None:
            draft.terminate()
            draft.wait(timeout=5)
