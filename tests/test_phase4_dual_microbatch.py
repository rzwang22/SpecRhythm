from __future__ import annotations

import inspect
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_phase4_dual_batched_draft import commit_row, initial
from test_phase4_dual_batched_draft import machine as _machine
from test_phase4_dual_scheduler import Request, proposal_result
from test_phase4_dual_scheduler import scheduler as _scheduler
from test_phase4_serial import phase4_config as _config

from specrhythm.phase4.dual_microbatch import (
    FIELDS,
    positive_size,
    scheduler_evidence,
    selected_size,
)
from specrhythm.phase4.dual_runner import _configure_environment, run_resident_dual_batch
from specrhythm.phase4.request_identity import FrozenPromptIdentityMap

scheduler = _scheduler
machine = _machine
phase4_config = _config
HELPER = Path('integrations/vllm/phase4b1_gate_helpers.sh').resolve()


def helper(tmp_path, mode, value):
    env = {**os.environ, 'PYTHON': sys.executable, 'HELPER': str(HELPER), 'MODE': mode,
           'RUN_DIR': str(tmp_path / 'run'), 'CAPTURE': str(tmp_path / 'capture'),
           'SR_PHASE4B_COMMIT': 'a' * 40, 'SR_PHASE4B_DUAL_MICROBATCH_SIZE': value}
    if value is None:
        env.pop('SR_PHASE4B_DUAL_MICROBATCH_SIZE')
    script = r'''
source "$HELPER"
python () { "$PYTHON" "$@"; }
phase4b1_require_environment () { return 0; }
phase4b1_start_draft () { PHASE4B1_DRAFT_PID=123; }
phase4b_run_target_with_cleanup () { printf '%s\n' "$@" > "$CAPTURE"; }
phase4b1_run_mode "$MODE" "$RUN_DIR" workload 100 reference
'''
    result = subprocess.run(['bash'], input=script, env=env, text=True, capture_output=True)
    path = tmp_path / 'capture'
    return result, path.read_text().splitlines() if path.exists() else []


def test_default_is_two():
    assert selected_size({}) == 2
    assert inspect.signature(run_resident_dual_batch).parameters['microbatch_size'].default == 2


@pytest.mark.parametrize('value', [None, '2', '4', '8', '16', '32', '64', '100', '101', '7'])
@pytest.mark.parametrize('backend', ['hf-persistent', 'vllm-batched'])
def test_dual_helper_default_and_explicit_values(tmp_path, monkeypatch, value, backend):
    monkeypatch.setenv('SR_PHASE4_DRAFT_BACKEND', backend)
    result, args = helper(tmp_path, 'dual', value)
    assert result.returncode == 0, result.stderr
    assert args[args.index('--microbatch-size') + 1] == str(int(value or 2))
    assert args[args.index('--test-coordination') + 1] == 'none'


@pytest.mark.parametrize('value', ['0', '-1', '1.5', 'x', '', ' 4', '+4'])
def test_invalid_value_fails_before_directory_or_draft_start(tmp_path, value):
    result, args = helper(tmp_path, 'dual', value)
    assert result.returncode != 0 and not args
    assert not (tmp_path / 'run').exists()


@pytest.mark.parametrize('value', [0, -1, 0.5, True, None, [], {}])
def test_positive_integer_only(value):
    with pytest.raises(ValueError, match='positive integer'):
        positive_size(value)


@pytest.mark.parametrize('mode', ['target', 'serial'])
def test_nondual_ignores_invalid_control(tmp_path, mode):
    result, args = helper(tmp_path, mode, 'invalid')
    assert result.returncode == 0, result.stderr
    assert '--microbatch-size' not in args


@pytest.mark.parametrize('n,ready_count', [
    *[(n, r) for n in (2, 4, 8, 16, 32, 64, 100) for r in (1, 100)], (100, 101),
])
def test_runner_to_scheduler_upper_bound_without_waiting(
    scheduler, machine, monkeypatch, tmp_path, n, ready_count
):
    # Exercise the unchanged runner transport and actual Scheduler constructor.
    with monkeypatch.context() as env:
        env.setattr(os, 'environ', dict(os.environ))
        paths = defaultdict(lambda: tmp_path / 'unused')
        paths.update(microbatch_size=n, request_count=100)
        _configure_environment(**{k: paths[k] for k in (
            'workload_path', 'draft_socket_path', 'scheduler_events_path',
            'request_state_events_path', 'proposal_events_path', 'verification_events_path',
            'transport_events_path', 'target_diagnostics_path', 'plugin_report_path',
            'microbatch_size', 'request_count')})
        instance = type(scheduler)()
    prompts = {f'r{i}': (i + 1, 200) for i in range(max(100, ready_count))}
    instance.requests = {f'opaque-{i}': Request(f'opaque-{i}', p)
                         for i, p in enumerate(prompts.values())}
    instance.running = list(instance.requests.values())
    instance._dual_identity = FrozenPromptIdentityMap(prompts)
    instance._dual_expected_ids = tuple(prompts)
    instance._bind_vllm_requests()
    instance.max_num_running_reqs = 256
    instance.max_num_scheduled_tokens = 8192
    polls = []
    ready_ids = list(prompts)[:ready_count]

    def poll(command, payload):
        assert command == 'poll_ready'  # no status check or wait to fill N
        polls.append(payload['limit'])
        ids = ready_ids[:payload['limit']]
        return {'ready': [proposal_result(r, prefix=prompts[r] + (10,)) for r in ids],
                'pending_request_ids': []}

    instance._dual_client = SimpleNamespace(call=poll)
    output = instance.schedule()
    selected = [instance._dual_identity.stable_id(k)
                for k in output.scheduled_spec_decode_tokens]
    assert selected == ready_ids[:n] and polls == [n]
    rows = instance._dual_events.read()
    assert scheduler_evidence(rows, n) == dict.fromkeys(FIELDS, n)
    assert rows[-1]['verify_request_ids'] == selected
    assert rows[-1]['dual_scheduler_constraints']['max_num_seqs'] == 256
    # Feed the chosen cohort to the real production Draft machine and preserve
    # IDs/order through commit_many and propose_many, including one retirement.
    initial_rows = [initial(r, prompts[r] + (10,)) for r in selected]
    for row in initial_rows:
        machine.initialize(row)
    proposals = machine.execute_batch('propose_only', initial_rows)
    commits = [commit_row(machine, r['proposal'], [999], terminal=i == 0)
               for i, r in enumerate(proposals)]
    results = machine.execute_batch('commit_and_propose', commits)
    assert [r['request_id'] for r in results] == selected
    survivors = [r for r in results if r.get('proposal')]
    assert [r['request_id'] for r in survivors] == selected[1:]
    if survivors:
        assert all(r['draft_cohort_request_ids'] == selected[1:] for r in survivors)


def test_mismatched_or_missing_scheduler_evidence_rejected():
    for rows in ([], [dict.fromkeys(FIELDS, 2)],
                 [{**dict.fromkeys(FIELDS, 4), 'verify_request_ids': list('abcde')}]):
        with pytest.raises(ValueError):
            scheduler_evidence(rows, 4)
