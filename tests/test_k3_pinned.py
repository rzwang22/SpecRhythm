"""Run the pinned child launcher; substitute git/GPU runner, preserve original exit."""

import os
import re
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("failure", ["fetch", "runner", "none"])
def test_k3_pinned_foreground_child(tmp_path, failure):
    script = Path("scripts/run_k3_b16_pinned.sh").resolve()
    sha = re.search(r"^FINAL_SHA=([0-9a-f]{40})$", script.read_text(), re.M)[1]
    assert sha in Path("docs/k3-runbook.md").read_text()
    git = tmp_path / "git"
    git.write_text('''#!/bin/bash
printf '%s\\n' "$*" >> "$CALLS"
if [[ "$3" == fetch && "$FAIL_AT" == fetch ]]; then exit 27; fi
if [[ "$3" == worktree ]]; then
  mkdir -p "$6/scripts"
  cat > "$6/scripts/run_k3_b16.sh" <<'RUNNER'
[[ "$1" == "$EXPECTED_SHA" ]] || exit 99
[[ "$SR_EXEC_REPO" == *-k3-* ]] || exit 98
if [[ "$FAIL_AT" == runner ]]; then exit 23; fi
RUNNER
fi
''')
    git.chmod(0o755)
    log = tmp_path / "calls"
    result = subprocess.run(
        ["bash", "-c", 'if bash "$SCRIPT"; then rc=0; else rc=$?; fi; echo "PARENT:$rc"'],
        env={
            **os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "SCRIPT": str(script), "SR_K3_REPO": str(tmp_path / "repo"),
            "EXPECTED_SHA": sha, "CALLS": str(log), "FAIL_AT": failure,
        },
        text=True, capture_output=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "PARENT:" + {"fetch": "27", "runner": "23", "none": "0"}[failure] in result.stdout
    calls = log.read_text()
    assert ("worktree add --detach" in calls) == (failure != "fetch")
    assert "reset" not in calls and "push" not in calls
