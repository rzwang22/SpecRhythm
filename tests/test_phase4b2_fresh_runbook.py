"""Validate operator shell survival and immutable historical evidence checks."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUNBOOK = REPO / "docs/phase4b2-fresh-three-mode-runbook.md"
BLOCKS = re.findall(r"```bash\n(.*?)\n```", RUNBOOK.read_text(), flags=re.DOTALL)
PYTHON_SNIPPETS = re.findall(
    r"<<'PY'\n(.*?)\nPY", "\n".join(BLOCKS), flags=re.DOTALL,
)


def test_fresh_runbook_bash_and_embedded_python_syntax():
    assert BLOCKS and PYTHON_SNIPPETS
    result = subprocess.run(
        ["bash", "-n"], input="\n".join(BLOCKS), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    for index, snippet in enumerate(PYTHON_SNIPPETS):
        compile(snippet, f"runbook-python-{index}", "exec")


def test_fresh_runbook_contains_no_failure_termination_commands():
    commands = "\n".join(BLOCKS)
    assert not re.search(r"\bexit\b|\bexec\b|set\s+-[^\n]*e|trap\s+[^-]", commands)
    assert BLOCKS[0].startswith("set +e\nset +u\nset +o pipefail\ntrap - ERR\n")
    assert all('RC="$?"' in block and 'rc=$RC"' in block for block in BLOCKS)


@pytest.mark.parametrize("failure", ["immutable-directory", "draft-start", "measurement"])
def test_failed_helpers_return_to_shell_with_return_code(tmp_path, failure):
    environment = dict(os.environ)
    for name in (
        "SR_PHASE4B_FIX_COMMIT", "SR_PHASE4B_COMMIT", "SR_PHASE4B_CONFIG",
        "SR_PHASE4B_ENVIRONMENT", "SR_PHASE4B_TOPOLOGY", "SR_PHASE4B_PATCH_MANIFEST",
    ):
        environment[name] = "cpu-test-only"
    environment["SR_PHASE4B2_ROOT"] = str(tmp_path)
    if failure == "immutable-directory":
        (tmp_path / "dual").mkdir()
    expected = {"immutable-directory": 2, "draft-start": 19, "measurement": 23}[failure]
    setup = "\n".join(
        f'source "{REPO / "integrations/vllm" / name}"'
        for name in (
            "phase4b_run_helpers.sh", "phase4b1_gate_helpers.sh", "phase4b2_run_helpers.sh",
        )
    )
    # Stubs ensure no GPU service or CLI is started even if a guard regresses.
    setup += '\nphase4b1_start_draft () { return 19; }\nspecrhythm () { return 23; }\n'
    command = (
        'phase4b2_measure_mode dual-batch "$SR_PHASE4B2_ROOT/dual" workload'
        if failure == "measurement" else
        'phase4b2_run_mode dual "$SR_PHASE4B2_ROOT/dual" workload 100 reference'
    )
    script = BLOCKS[0] + "\n" + setup + command + '''
RC="$?"
echo "intentional failure rc=$RC"
echo "INTERACTIVE_SHELL_SURVIVED"
'''
    # Start with inherited failure/unset-variable termination enabled; the actual
    # documented preamble must neutralize it before calling a failing helper.
    result = subprocess.run(
        ["bash", "-e", "-u", "-o", "pipefail", "-c", script],
        env=environment, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert f"intentional failure rc={expected}" in result.stdout
    assert "INTERACTIVE_SHELL_SURVIVED" in result.stdout


def test_runbook_inventory_is_external_and_detects_failed_evidence_changes(tmp_path):
    failed = tmp_path / "historical" / "target"
    failed.mkdir(parents=True)
    evidence = failed / "target.log"
    evidence.write_text("original EngineCore failure\n")
    audit = tmp_path / "audits" / "new"
    environment = dict(os.environ, SR_PHASE4B2_FAILED_ROOT=str(failed.parent),
                       SR_PHASE4B2_AUDIT_DIR=str(audit), SR_PHASE4B2_ROOT=str(tmp_path / "fresh"))
    before = next(snippet for snippet in PYTHON_SNIPPETS if "audit.mkdir(" in snippet)
    after = next(snippet for snippet in PYTHON_SNIPPETS if 'rows == before["entries"]' in snippet)
    original = evidence.read_bytes()
    result = subprocess.run([sys.executable, "-c", before], env=environment, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in failed.iterdir()) == ["target.log"]
    assert evidence.read_bytes() == original
    result = subprocess.run([sys.executable, "-c", after], env=environment, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert evidence.read_bytes() == original
    evidence.write_text("changed evidence\n")
    result = subprocess.run([sys.executable, "-c", after], env=environment, capture_output=True)
    assert result.returncode != 0
    assert b"historical failed Target evidence changed" in result.stderr


def test_current_runbook_preserves_aa80ed_and_keeps_five_patch_stack():
    commands = "\n".join(BLOCKS)
    failed_root = ('aa80ed22dfb13c7969e0944c4b56025ae3f42ca5/'
                   'phase4b2-row-mapping-20260905T115441Z-1467')
    assert failed_root in commands
    assert 'manage_patch.py restore' not in commands
    assert 'manage_patch.py apply' not in commands
    assert '--expect-state patched' in commands
    assert 'old / "target/decode-performance.json"' not in commands
    assert 'natural_teardown_completed' in commands


@pytest.mark.parametrize('recorded_status', [0, 7])
def test_old_target_audit_preserves_raw_files_and_rejects_a_different_cause(
    tmp_path, recorded_status,
):
    # Synthetic old-format evidence: this does not substitute for the A800 files.
    root = tmp_path / 'historical'
    target = root / 'target'
    target.mkdir(parents=True)
    audit = tmp_path / 'audit'
    audit.mkdir()
    value = {
        'target_exit_status': recorded_status, 'effective_exit_status': 125,
        'failure_detection': None, 'launch_error': None,
        'child_reap_result': {'wrapper_exited_with_descendants_alive': True},
        'term_kill_actions': [{'signal': 'SIGTERM'}], 'run_valid': False,
    }
    (target / 'process-lifecycle.json').write_text(json.dumps(value))
    (target / 'target.log').write_text('Running: 100 reqs\nEngineCore: SIGTERM\n')
    before = {path.name: path.read_bytes() for path in target.iterdir()}
    snippet = next(value for value in PYTHON_SNIPPETS if 'review["source_sha256"]' in value)
    result = subprocess.run(
        [sys.executable, '-c', snippet], capture_output=True,
        env=dict(os.environ, SR_PHASE4B2_FAILED_ROOT=str(root), SR_PHASE4B2_AUDIT_DIR=str(audit)),
    )
    assert (result.returncode == 0) == (recorded_status == 0)
    assert {path.name: path.read_bytes() for path in target.iterdir()} == before
    report = json.loads((audit / 'failed-target-lifecycle-review.json').read_text())
    assert report['target_exit_status'] == recorded_status
    assert len(report['source_sha256']) == 2
    assert report['log_evidence'][1]['line'] == 2
