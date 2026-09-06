from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from test_phase4 import WORKLOAD
from test_phase4_serial import phase4_config as _config

phase4_config = _config
RUNBOOK = Path('docs/phase4b4a-microbatch-runbook.md')
HELPER = Path('integrations/vllm/phase4b4_microbatch_helpers.sh').resolve()


def test_runbook_complete_safe_shell_and_embedded_python():
    text = RUNBOOK.read_text()
    blocks = re.findall(r'```bash\n(.*?)```', text, re.S)
    assert len(blocks) == 19
    for block in blocks:
        p = subprocess.run(['bash', '-n'], input=block, text=True, capture_output=True)
        assert p.returncode == 0, p.stderr
        assert 'RC="$?"' in block and 'rc=$RC' in block
        assert not re.search(r'\bexit\b|set -e', block)
        for _, code in re.findall(r"<<'(PY(?:CODE)?)'\n(.*?)\n\1", block, re.S):
            compile(code, 'runbook-inline', 'exec')
    assert '@DELIVERED_COMMIT@' in text
    assert '--expect-state patched' in text and 'manage_patch.py apply' not in text
    assert 'SR_PHASE4_DUAL_UUID_QUERY_MODE=live' in text
    assert 'VLLM_BATCH_INVARIANT=1' in text and 'WORLD_SIZE=1' in text
    assert re.findall(r'^phase4b4_run (.*)$', text, re.M) == [
        'target', 'serial', 'dual 2', 'dual 4', 'dual 8', 'dual 16', 'dual 32',
        'dual 64', 'dual 100']
    assert 'SR_D6_' not in text


@pytest.mark.parametrize('mode,size', [('target', ''), ('serial', ''),
                                     *[('dual', str(n)) for n in (2, 4, 8, 16, 32, 64, 100)]])
def test_sweep_real_runner_mode_and_measurement_mapping(tmp_path, mode, size):
    (tmp_path / 'preflight.json').write_text(json.dumps({
        'valid': True, 'errors': None, 'execution_git_commit': 'fixed'}))
    env = {**os.environ, 'HELPER': str(HELPER), 'PYTHON': sys.executable,
           'SR_PHASE4B4_ROOT': str(tmp_path), 'SR_PHASE4B_COMMIT': 'fixed',
           'SR_PHASE4B_WORKLOAD': 'hundred', 'SR_PHASE4B_REFERENCE': 'ref100',
           'MODE': mode, 'SIZE': size, 'CALLS': str(tmp_path / 'calls')}
    script = r'''
source "$HELPER"
git () { echo fixed; }
python () {
  if test "$2" = specrhythm.phase4.dual_microbatch_sweep; then
    printf 'validate %s\n' "$*" >> "$CALLS"
  else
    "$PYTHON" "$@"
  fi
}
phase4b2_run_mode () {
  printf 'run %s %s %s %s %s\n' "$1" "$4" "$SR_PHASE4_DRAFT_BACKEND" \
    "${SR_PHASE4B_DUAL_MICROBATCH_SIZE:-unset}" \
    "${PHASE4B1_OVERLAP_REQUIREMENT:-default}" >> "$CALLS"
}
phase4b2_measure_mode () { printf 'measure %s\n' "$1" >> "$CALLS"; }
phase4b4_run "$MODE" "$SIZE"
'''
    p = subprocess.run(['bash'], input=script, text=True, capture_output=True, env=env)
    assert p.returncode == 0, p.stderr
    calls = (tmp_path / 'calls').read_text().splitlines()
    backend = 'hf-persistent' if mode == 'target' else 'vllm-batched'
    bound, overlap = (size, 'characterization') if mode == 'dual' else ('unset', 'default')
    assert calls[0] == f'run {mode} 100 {backend} {bound} {overlap}'
    assert calls[1] == f"measure {'dual-batch' if mode == 'dual' else mode}"
    assert f'--mode {mode}' in calls[2]
    assert ('--microbatch-size' in calls[2]) == (mode == 'dual')


def test_runbook_preflight_executes_cpu(tmp_path, monkeypatch, phase4_config):
    root = tmp_path / 'preflight'
    root.mkdir()
    (root / 'patch-stage').mkdir()
    for name in ('probe-validation.json', 'environment.json', 'topology.json',
                 'installed-patch-check.json', 'patch-stage/vllm-patch-stack.json'):
        (root / name).write_text('{"valid":true}')
    five = [json.loads(row) for row in WORKLOAD.read_text().splitlines()]
    workload = root / '100.jsonl'
    workload.write_text('\n'.join(json.dumps(dict(r, request_id=f'r{i}', maximum_new_tokens=16))
                                  for i, r in enumerate(five * 20)))
    monkeypatch.setenv('SR_PHASE4B_WORKLOAD', str(workload))
    monkeypatch.setenv('SR_PHASE4B_CONFIG', str(Path('configs/phase4b_dual_batch_1d2v.yaml')))
    monkeypatch.setenv('SR_PHASE4B4_ROOT', str(root))
    monkeypatch.setenv('SR_PHASE4B_COMMIT', subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip())
    codes = re.findall(r"<<'PYCODE'\n(.*?)\nPYCODE", RUNBOOK.read_text(), re.S)
    preflight = next(code for code in codes if 'model_revision_manifest' in code)
    p = subprocess.run([sys.executable, '-c', preflight], text=True, capture_output=True)
    assert p.returncode == 0, p.stderr
    report = json.loads((root / 'preflight.json').read_text())
    assert report['valid'] and report['expected_measured_committed_tokens'] == 1487
    assert report['microbatch_sweep'] == [2, 4, 8, 16, 32, 64, 100]


def test_mb101_outside_predefined_sweep_before_launch(tmp_path):
    from specrhythm.phase4.dual_microbatch import positive_size

    assert positive_size('101') == 101
    env = {**os.environ, 'HELPER': str(HELPER), 'PYTHON': sys.executable}
    script = r'''
source "$HELPER"
python () { "$PYTHON" "$@"; }
phase4b2_run_mode () { echo unexpected-runtime; return 90; }
phase4b4_run dual 101
'''
    p = subprocess.run(['bash'], input=script, text=True, capture_output=True, env=env)
    assert p.returncode == 2 and '2/4/8/16/32/64/100' in p.stderr
    assert 'unexpected-runtime' not in p.stdout
