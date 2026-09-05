"""Validate operator shell survival and immutable historical evidence checks."""

from __future__ import annotations

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
    failed = tmp_path / "historical" / "dual"
    failed.mkdir(parents=True)
    evidence = failed / "target.log"
    evidence.write_text("original EngineCore failure\n")
    audit = tmp_path / "audits" / "new"
    environment = dict(os.environ, SR_PHASE4B2_FAILED_DUAL=str(failed),
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
    assert b"historical failed Dual evidence changed" in result.stderr
