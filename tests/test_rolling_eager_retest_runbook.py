"""Execute the delivered shell control flow with local command substitutes only."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

PATH = Path('docs/rolling-eager-retest-runbook.md')
A_SHA = 'bbf12170118961a88244ae97af949eeee7d6028a'


def executable(path, source):
    path.write_text(source)
    path.chmod(0o755)


@pytest.mark.parametrize('failure', ['correctness', 'serial_gate', 'eager_gate', 'report', 'none'])
def test_first_failure_stops_later_point_and_commit_but_preserves_parent(tmp_path, failure):
    repo, binary, results = (tmp_path / name for name in ('repo', 'bin', 'results'))
    for p in (repo / 'scripts', binary, results):
        p.mkdir(parents=True)
    log = tmp_path / 'calls.txt'
    executable(binary / 'git', '#!/bin/bash\nif [[ "$1" == rev-parse ]]; then '
               'printf "%s\\n" "$SR_FIXED_COMMIT"; fi\n')
    executable(binary / 'sha256sum', '#!/bin/bash\nexit 0\n')
    executable(repo / 'scripts/run_decode_scan.sh', '''#!/bin/bash
printf '%s cli %s\n' "$SR_FIXED_COMMIT" "$*" >> "$CALLS"
if [[ "$1" == prepare ]]; then mkdir -p "$SR_FIXED_ROOT"; fi
if [[ "$1" == bundle ]]; then printf '{}' > "$3"; fi
''')
    fake = binary / 'python'
    executable(fake, '#!'+sys.executable+'\n'+'''import os, pathlib, sys
args=sys.argv[1:]
body=sys.stdin.read() if args==['-'] else ''
kind=('correctness' if args[:3]==['-m','specrhythm.continuation.gpu_check','run'] else
      'report' if args[:2]==['-m','specrhythm.serving.eager_retest_report'] else
      'serial_gate' if "Serial B16 execution, measurement" in body else
      'eager_gate' if "life = eager['lifetime_counters']" in body else 'python')
with open(os.environ['CALLS'],'a') as f: f.write(os.environ['SR_FIXED_COMMIT']+' '+kind+'\\n')
if kind==os.environ['FAIL_STAGE']: sys.exit(23)
if kind=='report': pathlib.Path(args[args.index('--output')+1]).write_text('{}')
''')
    block, = re.findall(r'```bash\n(.*?)```', PATH.read_text(), re.S)
    block = block.replace('__AB_FULL_SHA__', 'c'*40)
    block = block.replace('/root/autodl-tmp/src/SpecRhythm', str(repo))
    block = block.replace('/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11',
                          str(fake))
    block = block.replace('/root/autodl-tmp/SpecRhythm-data/results/rolling-eager', str(results))
    parsed = subprocess.run(['bash', '-n'], input=block, text=True, capture_output=True)
    assert parsed.returncode == 0, parsed.stderr
    env = {**os.environ, 'PATH': str(binary)+os.pathsep+os.environ['PATH'],
           'CALLS': str(log), 'FAIL_STAGE': failure}
    result = subprocess.run(['bash'], input=block+'\nprintf "PARENT_ALIVE\\n"\n',
                            env=env, text=True, capture_output=True, timeout=20)
    assert result.returncode == 0 and 'PARENT_ALIVE' in result.stdout, result.stderr
    lines = log.read_text().splitlines()
    measured = [r for r in lines if 'cli run --single-point' in r]
    if failure == 'none':
        assert len(measured) == 4
        assert len(list(results.glob('*-complete-with-attribution.tar.gz'))) == 2
        assert {r.split()[0] for r in measured} == {A_SHA, 'c'*40}
    else:
        assert 'Stopped (rc=23)' in result.stdout
        assert all(r.startswith(A_SHA) for r in lines)
        assert len(measured) == {'correctness': 0, 'serial_gate': 1,
                                 'eager_gate': 2, 'report': 2}[failure]
        assert len(list(results.glob('*-failure-small.tar.gz'))) == 1
        assert not list(results.glob('*-complete-with-attribution.tar.gz'))


def test_frozen_workload_and_observation_are_not_changed_between_commits():
    source = PATH.read_text()
    assert 'for SR_FIXED_COMMIT in "$SR_EAGER_A_ONLY" "$SR_EAGER_AB"' in source
    assert '--warmup-steps 2 --window-seconds 30 --repeats 1' in source
    assert '--observation buffered-live --identity-matching bound-prefix' in source
    assert '--setup-timeout 900 --drain-timeout 60' in source
    assert "scan['options']['samples'] is None" in source
    assert 'cdaf71adace15d229f5087b98f9fd162a958456226a660184fe03f5d6ebd8ff4' in source
    assert source.index('gpu_check run') < source.index('run --single-point --batch 16')
    assert 'mode pingpong' not in source and 'gpu_uuid' not in source
