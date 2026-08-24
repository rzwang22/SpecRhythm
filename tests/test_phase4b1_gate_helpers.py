from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

HELPER = (
    Path(__file__).resolve().parents[1]
    / "integrations"
    / "vllm"
    / "phase4b1_gate_helpers.sh"
)


@pytest.mark.parametrize(
    ("run_status", "leave_guard", "expected_status"),
    ((7, False, 7), (0, True, 1), (0, False, 0)),
)
def test_run_mode_preserves_run_and_cleanup_status(
    tmp_path, run_status, leave_guard, expected_status
):
    environment = dict(os.environ)
    environment.update(
        {
            "PHASE4B1_HELPER": str(HELPER),
            "RUN_DIR": str(tmp_path / "run"),
            "RUN_STATUS": str(run_status),
            "LEAVE_GUARD": "1" if leave_guard else "0",
            "EXPECTED_STATUS": str(expected_status),
            "CALL_MARKER": str(tmp_path / "called"),
            "SR_PHASE4B_CONFIG": "config",
            "SR_PHASE4B_ENVIRONMENT": "environment",
            "SR_PHASE4B_TOPOLOGY": "topology",
            "SR_PHASE4B_PATCH_MANIFEST": "patch-manifest",
            "SR_PHASE4B_COMMIT": "a" * 40,
        }
    )
    completed = subprocess.run(
        [
            "bash",
            "-c",
            r'''
source "$PHASE4B1_HELPER"
phase4b1_start_draft () {
  PHASE4B1_DRAFT_PID=123
  export PHASE4B1_DRAFT_PID
  return 0
}
phase4b_run_target_with_cleanup () {
  : >"$CALL_MARKER"
  if test "$LEAVE_GUARD" = 1; then
    : >"$PHASE4B_RUN_GUARD"
  fi
  return "$RUN_STATUS"
}
if phase4b1_run_mode target "$RUN_DIR" workload 2 reference; then
  observed_status=0
else
  observed_status="$?"
fi
test "$observed_status" -eq "$EXPECTED_STATUS"
test -f "$CALL_MARKER"
''',
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_gate_validator_preserves_original_cli_failure(tmp_path):
    environment = dict(os.environ)
    environment.update(
        {
            "PHASE4B1_HELPER": str(HELPER),
            "GATE": str(tmp_path / "gate"),
            "DUAL": str(tmp_path / "dual"),
        }
    )
    completed = subprocess.run(
        [
            "bash",
            "-c",
            r'''
source "$PHASE4B1_HELPER"
specrhythm () { return 9; }
python () {
  echo "unexpected Python assertion" >&2
  return 1
}
if phase4b1_validate_gate "$GATE" "$DUAL"; then
  observed_status=0
else
  observed_status="$?"
fi
test "$observed_status" -eq 9
''',
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Gate validator failed with status 9" in completed.stderr
    assert "unexpected Python assertion" not in completed.stderr
