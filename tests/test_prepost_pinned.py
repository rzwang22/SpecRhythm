"""Pinned operator launcher: real child-shell failure contract without GPU execution."""

import os
import re
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_prepost_b16_pinned.sh"


@pytest.mark.parametrize("failure", ["fetch", "runner", "none"])
def test_pinned_child_keeps_parent_and_preserves_first_code(tmp_path, failure):
    git = tmp_path / "git"
    git.write_text("""#!/bin/bash
printf '%s\\n' "$*" >> "$CALLS"
if [[ "$3" == fetch && "$FAIL_AT" == fetch ]]; then exit 27; fi
if [[ "$3" == worktree ]]; then
  mkdir -p "$6/scripts"
  cat > "$6/scripts/run_prepost_b16.sh" <<'RUNNER'
[[ "$1" == f01e8d037007540209999179357c3dd2ff2335a7 ]] || exit 99
if [[ "$FAIL_AT" == runner ]]; then exit 23; fi
RUNNER
fi
""")
    git.chmod(0o755)
    log = tmp_path / "calls"
    result = subprocess.run(
        ["bash", "-c", 'if bash "$SCRIPT"; then rc=0; else rc=$?; fi; echo "PARENT:$rc"'],
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "SCRIPT": str(SCRIPT),
            "SR_PREPOST_REPO": str(tmp_path / "repo"),
            "CALLS": str(log),
            "FAIL_AT": failure,
        },
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "PARENT:" + {"fetch": "27", "runner": "23", "none": "0"}[failure] in result.stdout
    calls = log.read_text()
    assert ("worktree add" in calls) == (failure != "fetch")
    assert "reset" not in calls and "push" not in calls
    sha = re.search(r"FINAL_SHA=([0-9a-f]{40})", SCRIPT.read_text())[1]
    runbook = (SCRIPT.parents[1] / "docs/prepost3-runbook.md").read_text()
    assert sha in runbook and 'git -C "$REPO" worktree add --detach' in runbook
