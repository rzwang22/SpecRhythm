from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

HELPER = Path("integrations/vllm/phase4b4b_pingpong_helpers.sh").resolve()
RUNBOOK = Path("docs/phase4b4b-pingpong-runbook.md")


def test_complete_runbook_bash_blocks_parse_and_preserve_five_patch_stack():
    text = RUNBOOK.read_text()
    blocks = re.findall(r"```bash\n(.*?)\n```", text, re.S)
    assert len(blocks) == 19
    for block in blocks:
        result = subprocess.run(["bash", "-n"], input=block, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        assert not re.search(r"\bexit\b|set -e|set -eo|set -euo", block)
        assert 'RC="$?"' in block
    assert "@DELIVERED_COMMIT@" in text
    assert "SR_PHASE4_DUAL_UUID_QUERY_MODE=live" in text
    assert "SR_PHASE4B_DUAL_RHYTHM=legacy" in text
    assert "--expect-state patched" in text
    assert "manage_patch.py apply" not in text
    assert "repeat **all of sections 1–7**" in text
    assert "corrected5-prerequisite.json" in text


@pytest.mark.parametrize(
    "stage,cell,mode,perf,count,rhythm,size",
    [
        ("smoke", "pingpong", "dual", None, 2, "pingpong", 100),
        ("five", "pingpong", "dual", "dual-batch", 5, "pingpong", 100),
        ("full", "target", "target", "target", 100, "legacy", None),
        ("full", "serial", "serial", "serial", 100, "legacy", 100),
        ("full", "dual-mb2", "dual", "dual-batch", 100, "legacy", 2),
        ("full", "dual-mb100", "dual", "dual-batch", 100, "legacy", 100),
        ("full", "pingpong", "dual", "dual-batch", 100, "pingpong", 100),
    ],
)
def test_helper_exact_runtime_measurement_and_policy_transport(
    tmp_path, stage, cell, mode, perf, count, rhythm, size
):
    env = {
        **os.environ,
        "HELPER": str(HELPER),
        "STAGE": stage,
        "CELL": cell,
        "SR_PP_ROOT": str(tmp_path),
        "SR_PHASE4B_COMMIT": "a" * 40,
        "SR_PP_SMOKE_WORKLOAD": "smoke",
        "SR_PHASE4B3_WORKLOAD5": "five",
        "SR_PHASE4B3_REFERENCE5": "ref5",
        "SR_PHASE4B_WORKLOAD": "hundred",
        "SR_PHASE4B_REFERENCE": "ref100",
        "SR_PHASE4B_DUAL_RHYTHM": "legacy",
    }
    env.pop("SR_PHASE4B_DUAL_MICROBATCH_SIZE", None)
    script = r"""
source "$HELPER"
phase4b4b_require () { return 0; }
git () { printf '%s' "$SR_PHASE4B_COMMIT"; }
phase4b2_run_mode () {
  printf 'runtime %s %s %s %s %s\n' "$1" "$4" "$SR_PHASE4B_DUAL_RHYTHM" \
    "${SR_PHASE4B_DUAL_MICROBATCH_SIZE:-none}" "$SR_PHASE4_DRAFT_BACKEND"
}
phase4b2_measure_mode () { printf 'measure %s\n' "$1"; }
python () { printf 'report %s\n' "$*"; }
phase4b4b_run "$STAGE" "$CELL"
"""
    result = subprocess.run(["bash"], input=script, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    backend = "hf-persistent" if mode == "target" else "vllm-batched"
    assert f"runtime {mode} {count} {rhythm} {size or 'none'} {backend}" in result.stdout
    if perf:
        assert f"measure {perf}\n" in result.stdout
    else:
        assert "measure " not in result.stdout
        assert "--smoke" in result.stdout
    assert "pingpong_comparison" in result.stdout


def test_missing_prerequisite_stops_before_runtime(tmp_path):
    env = {
        **os.environ,
        "HELPER": str(HELPER),
        "SR_PP_ROOT": str(tmp_path),
        "SR_PHASE4B_COMMIT": "a" * 40,
        "PYTHON": sys.executable,
    }
    script = r"""
source "$HELPER"
python () { "$PYTHON" "$@"; }
phase4b2_run_mode () { echo 'must-not-run'; }
phase4b4b_run full pingpong
"""
    result = subprocess.run(["bash"], input=script, env=env, text=True, capture_output=True)
    assert result.returncode != 0 and "must-not-run" not in result.stdout
    assert not (tmp_path / "pingpong").exists()


@pytest.mark.parametrize("count", [2, 5, 100])
def test_resident_helper_creates_shared_assignment_and_uses_burst_capacity(tmp_path, count):
    workload = tmp_path / "workload.jsonl"
    workload.write_text(
        "\n".join(
            json.dumps(
                {
                    "request_id": f"r{i}",
                    "prompt_token_ids": [i + 1],
                    "prompt_length": 1,
                    "task_class": "code"
                    if i < count * 0.6
                    else "chat"
                    if i < count * 0.8
                    else "summarization",
                    "prompt_text": "<|im_start|>user\ntest<|im_start|>assistant",
                    "maximum_new_tokens": 16,
                    "sampling_seed": 1664,
                    "tokenizer_fingerprint": "frozen",
                }
            )
            for i in range(count)
        )
        + "\n"
    )
    env = {
        **os.environ,
        "HELPER": str(Path("integrations/vllm/phase4b1_gate_helpers.sh").resolve()),
        "PYTHON": sys.executable,
        "WORKLOAD": str(workload),
        "COUNT": str(count),
        "RUN_DIR": str(tmp_path / "run"),
        "CAPTURE": str(tmp_path / "capture"),
        "SR_PHASE4B_COMMIT": "a" * 40,
        "SR_PHASE4B_DUAL_RHYTHM": "pingpong",
        "SR_PHASE4_DRAFT_BACKEND": "vllm-batched",
        "SR_PHASE4B_DUAL_MICROBATCH_SIZE": "invalid",
    }
    script = r"""
source "$HELPER"
python () { "$PYTHON" "$@"; }
phase4b1_require_environment () { return 0; }
phase4b1_start_draft () {
  test -f "$SR_PHASE4_DUAL_RHYTHM_MANIFEST" || return 2
  PHASE4B1_DRAFT_PID=123
}
phase4b_run_target_with_cleanup () { printf '%s\n' "$@" > "$CAPTURE"; }
phase4b1_run_mode dual "$RUN_DIR" "$WORKLOAD" "$COUNT" reference
"""
    result = subprocess.run(["bash"], input=script, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    args = (tmp_path / "capture").read_text().splitlines()
    assert args[args.index("--microbatch-size") + 1] == str(count)
    manifest = json.loads((tmp_path / "run/dual-rhythm.json").read_text())
    assert manifest["readiness_capacity"] == count
    assert manifest["initial_sizes"] == {"A": (count + 1) // 2, "B": count // 2}
