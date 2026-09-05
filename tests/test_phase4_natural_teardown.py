"""Normal post-coordinator teardown versus leaked/fatal owned subprocesses."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from test_phase4_process_cleanup import REPO, WRAPPER, assert_gone

from specrhythm.phase4.owned_processes import process_table
from specrhythm.phase4.process_lifecycle import run_owned_target, validate_lifecycle_artifact


def coordinator(tmp_path, case):
    child = f"""
import os, pathlib, signal, sys, time
root = pathlib.Path({str(tmp_path)!r})
def term(*_):
    (root / 'term-seen').write_text('TERM')
    if {case!r} != 'leak-kill' and {case!r} != 'late-fatal':
        sys.exit(0)
signal.signal(signal.SIGTERM, term)
(root / 'child.pid').write_text(str(os.getpid()))
time.sleep(0.8)
if {case!r} == 'natural':
    (root / 'natural-exit').write_text('clean')
elif {case!r} == 'late-nonzero':
    os._exit(19)
else:
    if {case!r} == 'late-fatal':
        print('WorkerProc failed.', flush=True)
    time.sleep(60)
"""
    return f"""
import os, pathlib, signal, subprocess, sys, time
root = pathlib.Path({str(tmp_path)!r})
subprocess.Popen([sys.executable, '-c', {child!r}],
                 start_new_session={sys.platform == 'linux'!r})
while not (root / 'child.pid').exists():
    time.sleep(0.01)
if 'CPU_DRAFT_PID' in os.environ:
    os.kill(int(os.environ['CPU_DRAFT_PID']), signal.SIGUSR1)
os._exit(0)
"""


def cleanup_fixture(tmp_path):
    for path in tmp_path.glob('*.pid'):
        pid = int(path.read_text())
        if pid in process_table():
            os.kill(pid, signal.SIGKILL)
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass


def test_clean_coordinator_without_descendants_removes_guard(tmp_path):
    guard = tmp_path / 'process-lifecycle.active'
    status, report = run_owned_target(
        [sys.executable, '-c', 'pass'], target_log=tmp_path / 'target.log',
        artifact_path=tmp_path / 'process-lifecycle.json', guard_path=guard,
        natural_teardown_grace_seconds=0.2, poll_seconds=0.01,
    )
    assert status == report['target_exit_status'] == report['effective_exit_status'] == 0
    assert report['post_coordinator_descendants_observed'] is False
    assert report['natural_teardown_completed'] is True
    assert report['leaked_after_coordinator_exit'] is False
    assert report['cleanup_valid'] is report['run_valid'] is True
    assert report['owned_cleanup_completed'] is True
    assert report['term_kill_actions'] == [] and not guard.exists()
    assert validate_lifecycle_artifact(report) == []


@pytest.mark.parametrize('timestamp_wrapper', [False, True])
def test_transient_descendant_exits_naturally_without_signals(tmp_path, timestamp_wrapper):
    command = [sys.executable, '-c', coordinator(tmp_path, 'natural')]
    if timestamp_wrapper:
        command = [sys.executable, str(WRAPPER), '--output', str(tmp_path / 'timestamps.jsonl'),
                   '--', *command]
    guard = tmp_path / 'process-lifecycle.active'
    try:
        status, report = run_owned_target(
            command, target_log=tmp_path / 'target.log',
            artifact_path=tmp_path / 'process-lifecycle.json', guard_path=guard,
            natural_teardown_grace_seconds=2, poll_seconds=0.01,
        )
        assert status == 0 and report['target_exit_status'] == 0
        assert report['failure_detection'] is None
        assert report['post_coordinator_descendants_observed'] is True
        assert report['post_coordinator_owned_pids']
        assert report['natural_teardown_completed'] is True
        assert report['leaked_after_coordinator_exit'] is False
        assert report['child_reap_result']['wrapper_exited_with_descendants_alive'] is False
        assert report['cleanup_valid'] is report['run_valid'] is True
        assert report['owned_cleanup_completed'] is True
        assert report['term_kill_actions'] == []
        assert (tmp_path / 'natural-exit').exists() and not (tmp_path / 'term-seen').exists()
        assert not guard.exists() and validate_lifecycle_artifact(report) == []
        assert_gone(report['owned_process_ids'])
    finally:
        cleanup_fixture(tmp_path)


@pytest.mark.parametrize('case', ['leak-term', 'leak-kill'])
def test_real_leak_remains_invalid_even_after_successful_cleanup(tmp_path, case):
    guard = tmp_path / 'process-lifecycle.active'
    try:
        status, report = run_owned_target(
            [sys.executable, '-c', coordinator(tmp_path, case)],
            target_log=tmp_path / 'target.log', artifact_path=tmp_path / 'process-lifecycle.json',
            guard_path=guard, natural_teardown_grace_seconds=0.15,
            graceful_seconds=0.15, kill_seconds=1, poll_seconds=0.01,
        )
        assert status == 125 and report['target_exit_status'] == 0
        assert report['failure_detection'] is None
        assert report['post_coordinator_descendants_observed'] is True
        assert report['natural_teardown_completed'] is False
        assert report['leaked_after_coordinator_exit'] is True
        assert report['owned_cleanup_completed'] is True
        assert report['cleanup_valid'] is report['run_valid'] is False
        assert guard.exists() and (tmp_path / 'term-seen').exists()
        elapsed = report['natural_teardown_ended_ns'] - report['natural_teardown_started_ns']
        assert elapsed >= 150_000_000
        actions = report['term_kill_actions']
        assert all(row['timestamp_ns'] >= report['natural_teardown_ended_ns'] for row in actions)
        assert {row['signal'] for row in actions} == (
            {'SIGTERM'} if case == 'leak-term' else {'SIGTERM', 'SIGKILL'}
        )
        assert report['remaining_owned_pids'] == []
        assert_gone(report['owned_process_ids'])
    finally:
        cleanup_fixture(tmp_path)


@pytest.mark.parametrize('case', ['late-fatal', 'late-nonzero'])
def test_fatal_during_natural_teardown_does_not_wait_for_grace(tmp_path, case):
    if case == 'late-nonzero' and sys.platform != 'linux':
        pytest.skip('Linux owned-child zombie exit status')
    started = time.monotonic()
    try:
        status, report = run_owned_target(
            [sys.executable, '-c', coordinator(tmp_path, case)],
            target_log=tmp_path / 'target.log', artifact_path=tmp_path / 'process-lifecycle.json',
            natural_teardown_grace_seconds=10, graceful_seconds=0.15,
            kill_seconds=1, poll_seconds=0.01,
        )
        assert time.monotonic() - started < 5
        assert status == 125 and report['target_exit_status'] == 0
        assert report['natural_teardown_completed'] is False
        assert report['leaked_after_coordinator_exit'] is False
        assert report['failure_detection']['reason'] == (
            'fatal-runtime-log' if case == 'late-fatal' else 'owned-child-nonzero-exit'
        )
        assert report['failed_coordinator_descendants'] is True
        assert report['run_valid'] is report['cleanup_valid'] is False
        assert report['owned_cleanup_completed'] is True
        assert (tmp_path / 'process-lifecycle.json.active').exists()
        assert_gone(report['owned_process_ids'])
    finally:
        cleanup_fixture(tmp_path)


@pytest.mark.parametrize('value', [0, -1, float('inf'), float('nan')])
def test_natural_teardown_deadline_must_be_finite_and_positive(tmp_path, value):
    with pytest.raises(ValueError, match='natural_teardown_grace_seconds'):
        run_owned_target(
            [sys.executable, '-c', 'pass'], target_log=tmp_path / 'target.log',
            artifact_path=tmp_path / 'lifecycle.json', natural_teardown_grace_seconds=value,
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('exit_code', [0, 19, -signal.SIGTERM])
def test_exit_between_live_snapshot_and_waitpid_preserves_real_status(
    tmp_path, monkeypatch, exit_code,
):
    from specrhythm.phase4 import owned_processes, process_lifecycle

    owner = owned_processes.OwnedProcesses(12340)
    child = dict(pid=12341, ppid=12340, pgid=12340, session_id=12340,
                 start_identity='owned-child', state='S', command='fixture', exit_code=None)
    owner.observed[child['pid']] = child
    reaped = False

    def table():
        return {} if reaped else {child['pid']: child}

    def waitpid(pid, options):
        nonlocal reaped
        assert pid == child['pid'] and options == os.WNOHANG
        reaped = True
        return pid, exit_code << 8 if exit_code >= 0 else -exit_code

    monkeypatch.setattr(owned_processes, 'process_table', table)
    monkeypatch.setattr(owned_processes.os, 'waitpid', waitpid)
    log = tmp_path / 'target.log'
    log.write_text('')
    remaining, failure = process_lifecycle._wait_for_natural_teardown(
        owner, process_lifecycle._RuntimeFailureMonitor(log), 0.2, 0.01,
    )
    assert reaped and remaining == []
    if exit_code == 0:
        assert failure is None
    else:
        assert failure['reason'] == 'owned-child-nonzero-exit'
        assert failure['child']['exit_code'] == exit_code
        assert failure['child']['state'] == 'reaped'


@pytest.mark.parametrize('mode', ['target', 'serial', 'dual'])
def test_shared_shell_accepts_natural_teardown_and_removes_guard(tmp_path, mode):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    fake = bin_dir / 'specrhythm'
    fake.write_text(f'#!{sys.executable}\n' + coordinator(tmp_path, 'natural'))
    fake.chmod(0o755)
    draft = tmp_path / 'draft.py'
    draft.write_text('''import signal, socket, sys, time
signal.signal(signal.SIGUSR1, lambda *_: sys.exit(0))
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(sys.argv[1])
server.listen(1)
time.sleep(60)
''')
    environment = dict(os.environ, CPU_DRAFT_SCRIPT=str(draft), CPU_MODE=mode,
                       CPU_RUN_ROOT=str(tmp_path / mode), PYTHONPATH=str(REPO / 'src'))
    for key in ('SR_PHASE4B_CONFIG', 'SR_PHASE4B_ENVIRONMENT', 'SR_PHASE4B_TOPOLOGY',
                'SR_PHASE4B_PATCH_MANIFEST'):
        environment[key] = 'cpu-fixture'
    environment.update(
        SR_PHASE4B_COMMIT=f'cpu{os.getpid():x}', PHASE4B_PYTHON=sys.executable,
        PATH=os.pathsep.join([str(bin_dir), str(Path(sys.executable).parent), os.environ['PATH']]),
        PHASE4B_NATURAL_TEARDOWN_GRACE_SECONDS='2', PHASE4B_CLEANUP_POLL_SECONDS='0.01',
    )
    script = '''set +e
set +u
set +o pipefail
trap - ERR
source integrations/vllm/phase4b_run_helpers.sh
source integrations/vllm/phase4b1_gate_helpers.sh
source integrations/vllm/phase4b2_run_helpers.sh
phase4b1_start_draft () {
  python "$CPU_DRAFT_SCRIPT" "$3" >"$2/draft-service.log" 2>&1 &
  PHASE4B1_DRAFT_PID="$!"
  export CPU_DRAFT_PID="$!"
  for attempt in {1..100}; do
    if test -S "$3"; then return 0; fi
    sleep 0.01
  done
  return 124
}
phase4b2_run_mode "$CPU_MODE" "$CPU_RUN_ROOT" workload 100 reference
RC="$?"
echo "CPU_RUN_RC=$RC"
echo "INTERACTIVE_SHELL_SURVIVED"
'''
    try:
        result = subprocess.run(
            ['bash', '-e', '-u', '-o', 'pipefail', '-c', script], cwd=REPO, env=environment,
            capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0, result.stderr
        assert 'CPU_RUN_RC=0' in result.stdout, result.stdout + result.stderr
        assert 'INTERACTIVE_SHELL_SURVIVED' in result.stdout
        assert 'cleanup failed: lifecycle guard remains' not in result.stderr
        report = json.loads((tmp_path / mode / 'process-lifecycle.json').read_text())
        assert report['post_coordinator_descendants_observed'] is True
        assert report['natural_teardown_completed'] is True
        assert report['cleanup_valid'] is report['run_valid'] is True
        assert report['term_kill_actions'] == []
        assert not (tmp_path / mode / 'process-lifecycle.active').exists()
        assert_gone(report['owned_process_ids'] + [report['draft_shutdown_result']['pid']])
    finally:
        cleanup_fixture(tmp_path)
