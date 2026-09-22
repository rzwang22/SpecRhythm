"""Execute the fixed launcher; replace git checkout and the external GPU runner only."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ENTRY = Path(__file__).resolve().parents[1] / "scripts/run_k3_dispatch_pinned.sh"
SHA = "24a5042d8bce503699bb040f293541f853e1e6b1"


@pytest.mark.parametrize(
    "configuration", ["lean-reference", "lean-dispatch-opt"]
)
@pytest.mark.parametrize("failure", [0, 23])
def test_fixed_launcher_passes_exact_commit_and_config(tmp_path, configuration, failure):
    binary = tmp_path / "bin"
    binary.mkdir()
    git = binary / "git"
    git.write_text("#!" + sys.executable + "\n" + '''import json,os,pathlib,sys
args=sys.argv[1:]
with open(os.environ['CALLS'],'a') as f: f.write(json.dumps(args)+'\\n')
if 'cat-file' in args:
 assert args[-1]==os.environ['SHA']+'^{commit}'
if 'worktree' in args:
 assert args[-1]==os.environ['SHA'] and '--detach' in args
 root=pathlib.Path(args[-2]); (root/'scripts').mkdir(parents=True)
 (root/'scripts/run_k3_b128.sh').write_text(''' + repr('''#!/bin/bash
[[ "$1" == "$SHA" ]] || exit 81
[[ "$SR_K3_VALIDATION_PROFILE" == performance-exploration ]] || exit 82
[[ -z "${SR_K3_REPEAT_CHILD:-}" ]] || exit 83
[[ -z "${SR_K3_SKIP_CAPACITY:-}" ]] || exit 84
[[ -z "${SR_K3_DRAFT_DISPATCH:-}" ]] || exit 85
[[ -z "${SR_K3_TARGET_DIAGNOSTICS:-}" ]] || exit 88
[[ -z "${SR_FIXED_TARGET_DIAGNOSTICS:-}" ]] || exit 89
[[ -z "${SR_K3_TARGET_DISPATCH:-}" ]] || exit 90
[[ "$SR_K3_LOCAL_BASE" == /tmp/specrhythm-runs ]] || exit 86
[[ "$SR_PING_RESULTS" == /root/autodl-tmp/SpecRhythm-data/results/rolling-eager ]] || exit 87
printf '%s %s\\n' "$SR_K3_EXPERIMENT" "$1" > "$RESULT"
printf 'UPLOAD ONLY: /tmp/cpu-fixture.tar.gz\\n'
exit "$FAILURE"
''') + ''')
''')
    git.chmod(0o755)
    env = {**os.environ, "PATH": str(binary) + os.pathsep + os.environ["PATH"],
           "CALLS": str(tmp_path / "calls"), "SHA": SHA, "RESULT": str(tmp_path / "result"),
           "SR_K3_REPO": str(tmp_path / "repo"), "FAILURE": str(failure),
           "SR_K3_REPEAT_CHILD": "stale-inherited-flag", "SR_K3_SKIP_CAPACITY": "1",
           "SR_K3_DRAFT_DISPATCH": "legacy", "SR_K3_TARGET_DISPATCH": "encode-once",
           "SR_K3_TARGET_DIAGNOSTICS": "full", "SR_FIXED_TARGET_DIAGNOSTICS": "full"}
    env.pop("SR_PHASE4_NUMERICAL_DIAGNOSTIC_PLAN", None)
    args = [] if configuration == "lean-dispatch-opt" else [configuration]
    result = subprocess.run(["bash", str(ENTRY), *args], env=env,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == failure, result.stderr
    assert result.stdout.count("UPLOAD ONLY:") == 1
    assert (tmp_path / "result").read_text().strip() == configuration + " " + SHA
    calls = [json.loads(line) for line in (tmp_path / "calls").read_text().splitlines()]
    assert any(c[-3:] == ["fetch", "origin", "codex/rolling-eager-v0.1"] for c in calls)
    assert "strict-output" not in result.stdout


@pytest.mark.parametrize("args", [["baseline"], ["lean-target"], ["lean-reference", "b128"]])
def test_invalid_configuration_stops_before_checkout(args):
    result = subprocess.run(["bash", str(ENTRY), *args], capture_output=True, text=True,
                            timeout=20)
    assert result.returncode == 2 and "UPLOAD ONLY:" not in result.stdout
