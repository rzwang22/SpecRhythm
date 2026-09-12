"""Real CPU subprocess failures; no vLLM imports or GPU processes."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from specrhythm.phase4.owned_processes import process_table
from specrhythm.phase4.process_lifecycle import run_owned_target

REPO = Path(__file__).resolve().parents[1]
WRAPPER = REPO / "integrations/vllm/phase4b2_timestamp_command.py"
DRAFT = """
import signal, socket, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(sys.argv[1])
server.listen(1)
time.sleep(60)
"""
WORKER = """
import os, pathlib, signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))
time.sleep(60)
"""


def wait_file(path, seconds=3):
    deadline = time.monotonic() + seconds
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.exists()


def assert_gone(pids):
    table = process_table()
    assert not [table[pid] for pid in pids if pid in table]


def actor(tmp_path, reason="fatal", detached=False):
    code = f"""
import json, os, pathlib, signal, subprocess, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
ready = pathlib.Path({str(tmp_path / 'worker.pid')!r})
worker = subprocess.Popen(
    [sys.executable, '-c', {WORKER!r}, str(ready)], start_new_session={detached!r})
while not ready.exists():
    time.sleep(0.01)
pathlib.Path({str(tmp_path / 'actor.pid')!r}).write_text(str(os.getpid()))
if {reason!r} == 'fatal':
    print('EngineCore encountered a fatal error.', flush=True)
elif {reason!r} == 'worker-exit':
    failed = subprocess.Popen([sys.executable, '-c', 'import os; os._exit(17)'])
    pathlib.Path({str(tmp_path / 'failed.pid')!r}).write_text(str(failed.pid))
else:
    os._exit(7)
time.sleep(60)
"""
    return code


@pytest.mark.parametrize("reason", ["fatal", "coordinator-exit", "worker-exit"])
def test_bounded_target_tree_draft_kill_and_owned_socket_cleanup(tmp_path, reason):
    if reason == "worker-exit" and sys.platform != "linux":
        pytest.skip("Linux exit-status and subreaper contract; other cases are portable")
    socket_path = Path(f"/tmp/sr-cleanup-{os.getpid()}-{time.monotonic_ns()}.sock")
    draft = subprocess.Popen([sys.executable, "-c", DRAFT, str(socket_path)])
    wait_file(socket_path)
    started = time.monotonic()
    try:
        command = [sys.executable, str(WRAPPER), "--output", str(tmp_path / "timestamped.jsonl"),
                   "--", sys.executable, "-c", actor(tmp_path, reason, sys.platform == "linux")]
        status, report = run_owned_target(
            command, target_log=tmp_path / "target.log", artifact_path=tmp_path / "lifecycle.json",
            draft_pid=draft.pid, draft_socket=socket_path,
            natural_teardown_grace_seconds=10,
            graceful_seconds=0.15, kill_seconds=1, poll_seconds=0.01,
        )
        assert time.monotonic() - started < 8
        assert status != 0 and report["run_valid"] is False
        assert report["natural_teardown_completed"] is None
        assert report["natural_teardown_started_ns"] is None
        assert report["remaining_owned_pids"] == []
        assert {row["signal"] for row in report["term_kill_actions"]} == {"SIGTERM", "SIGKILL"}
        if reason != "coordinator-exit":
            assert report["failure_detection"]["reason"] == (
                "fatal-runtime-log" if reason == "fatal" else "owned-child-nonzero-exit"
            )
        draft_result = report["draft_shutdown_result"]
        assert draft_result["term_sent"] is True and draft_result["kill_sent"] is True
        assert draft_result["valid"] is True
        assert draft_result["remaining_owned_pids"] == []
        assert draft_result["socket_ownership_proven"] is True
        assert draft_result["stale_owned_socket_removed"] is True
        assert not socket_path.exists()
        draft.wait(timeout=1)
        pids = [int(path.read_text()) for path in tmp_path.glob("*.pid")]
        assert pids
        assert_gone([*pids, draft.pid, report["coordinator_pid"]])
    finally:
        if draft.poll() is None:
            draft.kill()
        draft.wait(timeout=2)
        for path in tmp_path.glob("*.pid"):
            pid = int(path.read_text())
            if pid in process_table():
                os.kill(pid, signal.SIGKILL)
        socket_path.unlink(missing_ok=True)


def test_unrelated_socket_owner_is_never_unlinked_or_signaled(tmp_path):
    socket_path = Path(f"/tmp/sr-unrelated-{os.getpid()}-{time.monotonic_ns()}.sock")
    unrelated = subprocess.Popen([sys.executable, "-c", DRAFT, str(socket_path)])
    draft = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    wait_file(socket_path)
    try:
        status, report = run_owned_target(
            [sys.executable, "-c", "raise SystemExit(7)"],
            target_log=tmp_path / "target.log", artifact_path=tmp_path / "lifecycle.json",
            draft_pid=draft.pid, draft_socket=socket_path,
            graceful_seconds=0.1, kill_seconds=0.5, poll_seconds=0.01,
        )
        assert status != 0 and report["cleanup_valid"] is False
        assert report["draft_shutdown_result"]["socket_ownership_proven"] is False
        assert socket_path.exists() and unrelated.poll() is None
    finally:
        unrelated.kill()
        unrelated.wait(timeout=2)
        draft.wait(timeout=2)
        socket_path.unlink(missing_ok=True)


def test_replaced_socket_is_preserved_even_when_original_owner_is_terminated(tmp_path):
    socket_path = Path(f"/tmp/sr-replaced-{os.getpid()}-{time.monotonic_ns()}.sock")
    draft = subprocess.Popen([sys.executable, "-c", DRAFT, str(socket_path)])
    wait_file(socket_path)
    script = (
        "import os,socket,sys,time; os.unlink(sys.argv[1]); "
        "s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); s.bind(sys.argv[1]); "
        "print('WorkerProc failed.',flush=True); time.sleep(60)"
    )
    try:
        status, report = run_owned_target(
            [sys.executable, "-c", script, str(socket_path)],
            target_log=tmp_path / "target.log", artifact_path=tmp_path / "lifecycle.json",
            draft_pid=draft.pid, draft_socket=socket_path,
            graceful_seconds=0.1, kill_seconds=1, poll_seconds=0.01,
        )
        assert status != 0
        result = report["draft_shutdown_result"]
        assert result["socket_ownership_proven"] is True
        assert result["socket_cleanup_error"] == "Draft socket identity changed during cleanup"
        assert result["valid"] is False and socket_path.exists()
    finally:
        if draft.poll() is None:
            draft.kill()
        draft.wait(timeout=2)
        socket_path.unlink(missing_ok=True)


@pytest.mark.parametrize("target_token", [None, "owned-target-launch"])
def test_reused_pid_is_not_signaled(monkeypatch, target_token):
    from specrhythm.phase4 import owned_processes

    previous = dict(pid=123456, ppid=1, pgid=123456, session_id=123456,
                    start_identity="old", state="S", command="test", exit_code=None)
    owner = owned_processes.OwnedProcesses(previous["pid"], target_token=target_token)
    owner.observed[previous["pid"]] = previous
    monkeypatch.setattr(owned_processes, "process_table", lambda: {
        previous["pid"]: {**previous, "start_identity": "reused"},
        previous["pid"] + 1: {**previous, "pid": previous["pid"] + 1,
                              "ppid": previous["pid"], "start_identity": "unrelated-child"},
    })
    delivered = []
    monkeypatch.setattr(owned_processes.os, "kill", lambda pid, sig: delivered.append((pid, sig)))
    monkeypatch.setattr(owned_processes.os, "waitpid",
                        lambda pid, flags: delivered.append((pid, flags)))
    actions = []
    owner.signal(signal.SIGKILL, actions)
    owner.reap(exclude_root=False)
    assert delivered == actions == []


def test_phase4b2_shell_returns_nonzero_without_killing_interactive_caller(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "specrhythm"
    fake.write_text(f"#!{sys.executable}\n" + actor(tmp_path))
    fake.chmod(0o755)
    draft_script = tmp_path / "draft.py"
    draft_script.write_text(DRAFT)
    environment = dict(os.environ)
    for key in ("SR_PHASE4B_CONFIG", "SR_PHASE4B_ENVIRONMENT", "SR_PHASE4B_TOPOLOGY",
                "SR_PHASE4B_PATCH_MANIFEST"):
        environment[key] = "cpu-fixture"
    environment.update(
        SR_PHASE4B_COMMIT=f"cpu{os.getpid()}", PHASE4B_PYTHON=sys.executable,
        PATH=os.pathsep.join([str(bin_dir), str(Path(sys.executable).parent), os.environ["PATH"]]),
        PYTHONPATH=str(REPO / "src"), PHASE4B_CLEANUP_GRACE_SECONDS="0.15",
        PHASE4B_CLEANUP_KILL_SECONDS="1", PHASE4B_CLEANUP_POLL_SECONDS="0.01",
        CPU_DRAFT_SCRIPT=str(draft_script), CPU_RUN_ROOT=str(tmp_path / "dual"),
    )
    script = """set +e
set +u
set +o pipefail
trap - ERR
source integrations/vllm/phase4b_run_helpers.sh
source integrations/vllm/phase4b1_gate_helpers.sh
source integrations/vllm/phase4b2_run_helpers.sh
phase4b1_start_draft () {
  python "$CPU_DRAFT_SCRIPT" "$3" >"$2/draft-service.log" 2>&1 &
  PHASE4B1_DRAFT_PID="$!"
  for attempt in {1..100}; do
    if test -S "$3"; then return 0; fi
    sleep 0.01
  done
  return 124
}
phase4b2_run_mode dual "$CPU_RUN_ROOT" workload 100 reference
RC="$?"
echo "CPU_FAILURE_RC=$RC"
echo "INTERACTIVE_SHELL_SURVIVED"
"""
    result = subprocess.run(
        ["bash", "-e", "-u", "-o", "pipefail", "-c", script], cwd=REPO,
        env=environment, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "INTERACTIVE_SHELL_SURVIVED" in result.stdout
    assert "CPU_FAILURE_RC=125" in result.stdout
    report = json.loads((tmp_path / "dual/process-lifecycle.json").read_text())
    assert report["remaining_owned_pids"] == []
    assert report["draft_shutdown_result"]["valid"] is True
    assert_gone(report["owned_process_ids"] + [report["draft_shutdown_result"]["pid"]])
