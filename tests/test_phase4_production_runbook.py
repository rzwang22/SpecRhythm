from __future__ import annotations

import ast
import hashlib
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
HELPER = ROOT / "integrations/vllm/phase4b3_qualification_helpers.sh"
RUNBOOK = ROOT / "docs/phase4b3-production-qualification-runbook.md"


def test_fresh_server_runbook_is_complete_and_non_exiting():
    text = RUNBOOK.read_text()
    blocks = re.findall(r"```bash\n(.*?)```", text, re.S)
    assert len(blocks) >= 20
    for block in blocks:
        assert not re.search(r"(?m)^\s*set\s+-e|\bexit\b", block)
        subprocess.run(["bash", "-n"], input=block, text=True, check=True)
        for code in re.findall(r"<<'PY'\n(.*?)\nPY", block, re.S):
            ast.parse(code)
    for required in (
        "conda.sh",
        "conda activate",
        "@DELIVERED_COMMIT@",
        "git switch --detach",
        "--no-deps --no-build-isolation",
        "SR_VLLM_SOURCE",
        "SR_VLLM_ROOT",
        "SR_DRAFT_MODEL",
        "SR_TARGET_MODEL",
        "SR_PHASE4B_CONFIG",
        "SR_PHASE4B_ENVIRONMENT",
        "SR_PHASE4B_TOPOLOGY",
        "SR_PHASE4B_PATCH_MANIFEST",
        "SR_PHASE4B3_WORKLOAD5",
        "SR_PHASE4B3_REFERENCE5",
        "SR_PHASE4B_WORKLOAD",
        "SR_PHASE4B_REFERENCE",
        "--expect-state patched",
        "--request-count 8",
        "draft_qualification aggregate",
        "phase4b3_run_serial D4 hf",
        "phase4b3_run_serial D4 vllm",
        "phase4b3_compare_serial D4",
        "phase4b3_run_serial D5 hf",
        "phase4b3_run_serial D5 vllm",
        "phase4b3_compare_serial D5",
        "tar -czf",
    ):
        assert required in text
    assert "--diagnostic" not in "\n".join(blocks)
    subprocess.run(["bash", "-n", str(HELPER)], check=True)


def test_admission_rerunbook_bootstraps_new_root_and_stops_after_d3():
    text = (ROOT / "docs/phase4b3-device-admission-runbook.md").read_text()
    blocks = re.findall(r"```bash\n(.*?)```", text, re.S)
    assert len(blocks) >= 12
    commands = "\n".join(blocks)
    assert not re.search(r"(?m)^\s*set\s+-e|\bexit\b", commands)
    for block in blocks:
        subprocess.run(["bash", "-n"], input=block, text=True, check=True)
        for code in re.findall(r"<<'PY'\n(.*?)\nPY", block, re.S):
            ast.parse(code)
    for required in (
        "conda activate",
        "git switch --detach",
        "@DELIVERED_COMMIT@",
        "--no-deps --no-build-isolation",
        "SR_VLLM_ROOT",
        "SR_VLLM_SOURCE",
        "SR_DRAFT_MODEL",
        "SR_TARGET_MODEL",
        "SR_PHASE4B_CONFIG",
        "--expect-state patched",
        "SR_PHASE4B_ENVIRONMENT",
        "SR_PHASE4B_TOPOLOGY",
        "SR_PHASE4B3_PROBE_SOURCE",
        "SR_PHASE4B3_BASELINE_ROOT",
        "--gate D1",
        "--gate D2",
        "--request-count 2",
        "--request-count 4",
        "--request-count 8",
        "draft_qualification aggregate",
        "tar -czf",
        "draft-admission-$(date",
        "same_physical_gpu",
        "structural_checks_executed",
    ):
        assert required in commands
    assert "--diagnostic" not in commands
    assert "phase4b3_run_serial" not in commands and "draft_comparison" not in commands
    assert "--stage D4" not in commands and "--stage D5" not in commands


@pytest.mark.parametrize("stage", ["D4", "D5"])
def test_operator_helper_never_starts_gpu_after_failed_admission(tmp_path, stage):
    script = f'''
set +e
set +u
source "{HELPER}"
export SR_PHASE4B3_ROOT="{tmp_path}"
export SR_PHASE4B_COMMIT=cpu-test
export SR_PHASE4B_CONFIG=cpu-config
export SR_PHASE4B3_WORKLOAD5=cpu-five
export SR_PHASE4B3_REFERENCE5=cpu-five-reference
export SR_PHASE4B_WORKLOAD=cpu-hundred
export SR_PHASE4B_REFERENCE=cpu-hundred-reference
python () {{ return 31; }}
phase4b2_run_mode () {{ touch "$SR_PHASE4B3_ROOT/GPU-started"; }}
phase4b2_measure_mode () {{ touch "$SR_PHASE4B3_ROOT/measured"; }}
phase4b3_run_serial {stage} vllm
sr3_status=$?
test "$sr3_status" = 31 && test ! -e "$SR_PHASE4B3_ROOT/GPU-started" &&
  test ! -e "$SR_PHASE4B3_ROOT/measured"
'''
    subprocess.run(["bash"], input=script, text=True, check=True)


def test_new_d3_entry_requires_operator_opt_in_before_model_loading(monkeypatch):
    from specrhythm.phase4 import draft_qualification_gate

    monkeypatch.setattr(
        "sys.argv",
        [
            "gate",
            "--config",
            "absent",
            "--expected-commit",
            "absent",
            "--probe-root",
            "absent",
            "--request-count",
            "8",
            "--output",
            "absent",
        ],
    )
    with pytest.raises(SystemExit) as error:
        draft_qualification_gate.main()
    assert error.value.code == 2


def test_legacy_gate_semantics_and_vllm_patch_stack_are_not_changed():
    from specrhythm.phase4.draft_logits_contract import load_probe_fixture

    for name, digest in load_probe_fixture()["patch_sha256"].items():
        assert (
            hashlib.sha256((ROOT / "integrations/vllm/patches" / name).read_bytes()).hexdigest()
            == digest
        )
    source = (ROOT / "src/specrhythm/phase4/draft_gate.py").read_text()
    assert '"schema_version": "specrhythm.phase4b3-draft-gate.v1"' in source
    assert 'raise RuntimeError("Draft proposal differs from HF oracle; gate stopped")' in source
