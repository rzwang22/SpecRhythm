"""Independent Bash orchestration keeps the source checkout and stops the next version."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_eager_latency_runbook import executable


@pytest.mark.parametrize("failure", ["none", "control", "optimized"])
@pytest.mark.parametrize("order", ["control-first", "optimized-first"])
def test_version_worktrees_and_original_error_reach_live_parent(tmp_path, failure, order):
    repo, binary, results = (tmp_path / n for n in ("repo", "bin", "results"))
    for path in (repo, binary, results):
        path.mkdir()
    (repo / "user-checkout").write_text("preserved")
    control, optimized = "a" * 40, "b" * 40
    log = tmp_path / "calls"
    executable(
        binary / "git",
        "#!"
        + sys.executable
        + "\n"
        + """
import os,pathlib,sys
a=sys.argv[1:]
if a[:2]==['worktree','add']:
 p=pathlib.Path(a[3])/'scripts';p.mkdir(parents=True)
 (p/'run_execution_pair_b16.sh').write_text(os.environ['PAIR_STUB'])
""",
    )
    fake_python = binary / "report-python"
    executable(
        fake_python,
        "#!"
        + sys.executable
        + "\n"
        + """
import pathlib,sys
pathlib.Path(sys.argv[sys.argv.index('--output')+1]).write_text('{}')
""",
    )
    pair = """#!/bin/bash
printf '%s\n' "$1" >> "$CALLS"
if [[ "$1" == "$FAIL_SHA" ]]; then
  printf 'POINT exported diagnostic_evidence failure rc=23\n'
  exit 23
fi
printf '%s\n' "$SR_EXEC_REPO/serial.json" "$SR_EXEC_REPO/eager.json" > "$SR_EXEC_REPORT_LIST"
"""
    script = Path("scripts/run_eager_execution_b16.sh").resolve()
    shell = f"""if bash '{script}' '{control}' '{optimized}'; then
printf 'COMPLETE\n'
else printf 'STOP rc=%s\n' "$?"; fi
printf 'PARENT_ALIVE\n'
"""
    result = subprocess.run(
        ["bash"],
        input=shell,
        text=True,
        capture_output=True,
        timeout=20,
        env={
            **os.environ,
            "PATH": str(binary) + ":" + os.environ["PATH"],
            "SR_EXEC_SOURCE_REPO": str(repo),
            "SR_EXEC_RESULTS": str(results),
            "SR_FIXED_PYTHON": str(fake_python),
            "CALLS": str(log),
            "PAIR_STUB": pair,
            "SR_EXEC_VERSION_ORDER": order,
            "FAIL_SHA": {"control": control, "optimized": optimized, "none": ""}[failure],
        },
    )
    assert result.returncode == 0 and "PARENT_ALIVE" in result.stdout, result.stderr
    calls = log.read_text().splitlines()
    expected = [control, optimized] if order == "control-first" else [optimized, control]
    if failure != "none":
        sha = control if failure == "control" else optimized
        assert calls == expected[: expected.index(sha) + 1]
        assert "STOP rc=23" in result.stdout
        assert not list(results.glob("execution-two-version-*.json"))
    else:
        assert calls == expected
        assert len(list(results.glob("execution-two-version-*.json"))) == 1
    assert (repo / "user-checkout").read_text() == "preserved"


def test_only_detached_versions_no_reset_merge_or_grid():
    source = Path("scripts/run_eager_execution_b16.sh").read_text()
    assert "worktree add --detach" in source
    assert "git merge-base --is-ancestor" in source
    assert "git reset" not in source and "git checkout" not in source
    assert "CONTROL_SHA" in source and "OPTIMIZED_SHA" in source
    assert "ORDER=(control optimized)" in source
