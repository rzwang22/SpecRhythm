"""Execute the delivered foreground launcher, replacing checkout/GPU commands only."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ENTRY = Path(__file__).resolve().parents[1] / "scripts/run_ordinary_cpu_pinned.sh"
SHA = "f83edb43d22dcbb605468b8518bcca7c1bc36f45"


@pytest.mark.parametrize("code", [0, 23])
def test_fixed_entry_exact_execution_and_single_upload(tmp_path, code):
    binary = tmp_path / "bin"
    binary.mkdir()
    git = binary / "git"
    gpu_stub = '''#!/bin/bash
[[ "$1" == "$SHA" ]] || exit 81
[[ -z "${SR_K3_TARGET_DISPATCH:-}" ]] || exit 82
[[ -z "${SR_K3_TARGET_CPU:-}" ]] || exit 88
[[ -z "${SR_K3_MANAGED_LOCAL:-}" ]] || exit 83
[[ -z "${SR_K3_SKIP_CAPACITY:-}" ]] || exit 84
[[ -z "${SR_FIXED_TARGET_DIAGNOSTICS:-}" ]] || exit 85
[[ "$SR_K3_LOCAL_BASE" == /tmp/specrhythm-runs ]] || exit 86
[[ "$SR_PING_RESULTS" == /root/autodl-tmp/SpecRhythm-data/results/rolling-eager ]] || exit 87
printf 'UPLOAD ONLY: /tmp/cpu-fixture.tar.gz\\n'
exit "$CODE"
'''
    git.write_text("#!" + sys.executable + "\n" + '''import json,os,pathlib,sys
args=sys.argv[1:]
with open(os.environ['CALLS'],'a') as f: f.write(json.dumps(args)+'\\n')
if 'cat-file' in args: assert args[-1]==os.environ['SHA']+'^{commit}'
if 'worktree' in args:
 assert args[-1]==os.environ['SHA'] and '--detach' in args
 root=pathlib.Path(args[-2]); (root/'scripts').mkdir(parents=True)
 (root/'scripts/run_ordinary_cpu.sh').write_text(''' + repr(gpu_stub) + ")\n")
    git.chmod(0o755)
    env = {**os.environ, "PATH": str(binary) + os.pathsep + os.environ["PATH"],
           "CALLS": str(tmp_path / "calls"), "SHA": SHA, "CODE": str(code),
           "SR_K3_REPO": str(tmp_path / "repo"), "SR_K3_MANAGED_LOCAL": "1",
           "SR_K3_SKIP_CAPACITY": "1", "SR_K3_TARGET_DISPATCH": "encode-once",
           "SR_FIXED_TARGET_DIAGNOSTICS": "full", "SR_K3_TARGET_CPU": "block-sets"}
    result = subprocess.run(["bash", str(ENTRY)], env=env, capture_output=True,
                            text=True, timeout=20)
    assert result.returncode == code, result.stderr
    assert result.stdout.count("UPLOAD ONLY:") == 1
    calls = [json.loads(s) for s in (tmp_path / "calls").read_text().splitlines()]
    assert any(c[-3:] == ["fetch", "origin", "codex/rolling-eager-v0.1"] for c in calls)
    assert sum('worktree' in c for c in calls) == 1
    if code:
        assert 'first error (rc=23)' in result.stderr
