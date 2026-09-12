"""The recovery runbook stays offline and leaves interactive Bash alive."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

RUNBOOK = Path(__file__).resolve().parents[1] / "docs/phase4b2-terminal-state-recovery-runbook.md"
BLOCKS = re.findall(r"```bash\n(.*?)\n```", RUNBOOK.read_text(), flags=re.DOTALL)


def test_terminal_recovery_runbook_syntax_and_return_code_contract():
    script = "\n".join(BLOCKS)
    result = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    snippets = re.findall(r"<<'PY'\n(.*?)\nPY", script, flags=re.DOTALL)
    for index, snippet in enumerate(snippets):
        compile(snippet, f"recovery-runbook-{index}", "exec")
    assert not re.search(r"\bexit\b|\bexec\b|set\s+-[^\n]*e|trap\s+[^-]", script)
    assert all('RC="$?"' in block and 'rc=$RC"' in block for block in BLOCKS)
    assert "phase4b2_run_mode" not in script
    assert "specrhythm phase4b1-resident-dual-run" not in script
    assert "--terminal-revalidation" in script


def test_failed_offline_recovery_command_leaves_shell_alive():
    block = next(row for row in BLOCKS if row.startswith("specrhythm phase4b2-reconcile"))
    script = BLOCKS[0] + "\nspecrhythm () { return 17; }\n" + block
    script += '\necho "RECOVERY_FAILURE_RETURNED_TO_OPERATOR"\n'
    result = subprocess.run(
        ["bash", "-e", "-u", "-o", "pipefail", "-c", script],
        env=dict(os.environ, SR_PHASE4B2_REVALIDATION_COMMIT="cpu-test"),
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "strict completed Dual terminal-state recovery rc=17" in result.stdout
    assert "RECOVERY_FAILURE_RETURNED_TO_OPERATOR" in result.stdout
