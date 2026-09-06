from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from test_phase4 import WORKLOAD
from test_phase4_serial import phase4_config as _config_fixture

HELPER = Path("integrations/vllm/phase4b3_d6_helpers.sh").resolve()
RUNBOOK = Path("docs/phase4b3-d6-runbook.md")
phase4_config = _config_fixture


def test_d6_runbook_complete_safe_shell_and_embedded_python():
    text = RUNBOOK.read_text()
    blocks = re.findall(r"```bash\n(.*?)```", text, re.S)
    assert len(blocks) == 16
    for block in blocks:
        result = subprocess.run(["bash", "-n"], input=block, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        assert 'RC="$?"' in block and "rc=$RC" in block
        assert not re.search(r"\bexit\b|set -e", block)
        for python in re.findall(r"<<'(PY(?:CODE)?)'\n(.*?)\n\1", block, re.S):
            compile(python[1], "runbook-inline", "exec")
    assert "@DELIVERED_COMMIT@" in text
    assert "--expect-state patched" in text and "manage_patch.py apply" not in text
    assert "SR_PHASE4_DUAL_UUID_QUERY_MODE=live" in text
    assert "VLLM_BATCH_INVARIANT=1" in text and "WORLD_SIZE=1" in text
    assert "Repeat A/B" not in text
    stages = re.findall(r"^phase4b3_d6_run ([ABC] \w+)$", text, re.M)
    assert stages == ["A dual", "B dual", "C target", "C serial", "C dual"]


def helper_run(tmp_path, stage, mode, *, prerequisites=True):
    root = tmp_path / "root"
    root.mkdir()
    for path in (
        "preflight.json",
        "D6-A/dual/qualification.json",
        "D6-B/dual/qualification.json",
        "C/target/qualification.json",
        "C/serial/qualification.json",
    ):
        file = root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps({"valid": prerequisites, "errors": None}))
    env = {
        **os.environ,
        "HELPER": str(HELPER),
        "PYTHON": sys.executable,
        "SR_D6_ROOT": str(root),
        "SR_D6_C_ROOT": str(root / "C"),
        "SR_PHASE4B_COMMIT": "fixed",
        "SR_D6_SMOKE_WORKLOAD": "smoke",
        "SR_PHASE4B3_WORKLOAD5": "five",
        "SR_PHASE4B3_REFERENCE5": "ref5",
        "SR_PHASE4B_WORKLOAD": "hundred",
        "SR_PHASE4B_REFERENCE": "ref100",
        "CALLS": str(tmp_path / "calls"),
        "STAGE": stage,
        "MODE": mode,
    }
    script = r"""
source "$HELPER"
git () { echo fixed; }
python () {
  if test "$1" = -m; then
    printf 'validate %s\n' "$*" >> "$CALLS"
  else
    "$PYTHON" "$@"
  fi
}
phase4b2_run_mode () {
  printf 'run %s %s %s %s %s\n' "$1" "$3" "$4" "$SR_PHASE4_DRAFT_BACKEND" \
    "${PHASE4B1_OVERLAP_REQUIREMENT:-default}" >> "$CALLS"
}
phase4b2_measure_mode () { printf 'measure %s\n' "$1" >> "$CALLS"; }
phase4b3_d6_run "$STAGE" "$MODE"
"""
    completed = subprocess.run(["bash"], input=script, text=True, capture_output=True, env=env)
    path = tmp_path / "calls"
    return completed, path.read_text().splitlines() if path.exists() else []


@pytest.mark.parametrize(
    "stage,mode,workload,count,backend,overlap",
    [
        ("A", "dual", "smoke", 2, "vllm-batched", "separate-gate"),
        ("B", "dual", "five", 5, "vllm-batched", "required"),
        ("C", "target", "hundred", 100, "hf-persistent", "default"),
        ("C", "serial", "hundred", 100, "vllm-batched", "required"),
        ("C", "dual", "hundred", 100, "vllm-batched", "required"),
    ],
)
def test_d6_reuses_existing_runner_with_exact_stage_inputs(
    tmp_path, stage, mode, workload, count, backend, overlap
):
    completed, calls = helper_run(tmp_path, stage, mode)
    assert completed.returncode == 0, completed.stderr
    assert calls[0] == f"run {mode} {workload} {count} {backend} {overlap}"
    assert any("--smoke" in row for row in calls) == (stage == "A")
    assert [row for row in calls if row.startswith("measure")] == (
        [] if stage == "A" else [f"measure {mode}"]
    )


@pytest.mark.parametrize("stage,mode", [("A", "dual"), ("B", "dual"), ("C", "target")])
def test_failed_preflight_stops_before_any_runtime(tmp_path, stage, mode):
    completed, calls = helper_run(tmp_path, stage, mode, prerequisites=False)
    assert completed.returncode != 0
    assert calls == []


def test_runbook_preflight_executes_on_cpu_including_two_request_smoke(
    tmp_path, monkeypatch, phase4_config
):
    root = tmp_path / "preflight"
    root.mkdir()
    (root / "patch-stage").mkdir()
    for name in (
        "probe-validation.json",
        "environment.json",
        "topology.json",
        "installed-patch-check.json",
        "patch-stage/vllm-patch-stack.json",
    ):
        (root / name).write_text('{"valid":true}')
    five = [json.loads(row) for row in WORKLOAD.read_text().splitlines()]
    for key, rows in (
        ("SR_PHASE4B3_WORKLOAD5", five),
        ("SR_D6_SMOKE_WORKLOAD", five[:2]),
        ("SR_PHASE4B_WORKLOAD", [dict(r, request_id=f"r{i}") for i, r in enumerate(five * 20)]),
    ):
        path = root / (key + ".jsonl")
        path.write_text("\n".join(json.dumps(row) for row in rows))
        monkeypatch.setenv(key, str(path))
    monkeypatch.setenv("SR_PHASE4B_CONFIG", str(Path("configs/phase4b_dual_batch_1d2v.yaml")))
    monkeypatch.setenv("SR_D6_ROOT", str(root))
    monkeypatch.setenv(
        "SR_PHASE4B_COMMIT",
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    codes = re.findall(r"<<'PYCODE'\n(.*?)\nPYCODE", RUNBOOK.read_text(), re.S)
    preflight = next(code for code in codes if "model_revision_manifest" in code)
    completed = subprocess.run([sys.executable, "-c", preflight], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    report = json.loads((root / "preflight.json").read_text())
    assert report["valid"] and report["inputs"]["SR_D6_SMOKE_WORKLOAD"]["request_count"] == 2
    assert report["dual_uuid_query_mode"] == "live"
