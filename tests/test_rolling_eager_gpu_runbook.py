"""The delivered operator sequence is bounded, parseable and explicit about GPU evidence."""

import re
import subprocess
from pathlib import Path

PATH = Path("docs/rolling-eager-gpu-runbook.md")


def test_runbook_shell_blocks_parse_and_keep_parent_terminal_open():
    source = PATH.read_text()
    blocks = re.findall(r"```bash\n(.*?)```", source, flags=re.S)
    assert len(blocks) == 2
    for block in blocks:
        parsed = subprocess.run(["bash", "-n"], input=block, capture_output=True, text=True)
        assert parsed.returncode == 0, parsed.stderr
    main = blocks[0]
    assert "if bash <<'BASH'" in main
    assert "set -Eeuo pipefail" in main and "trap on_failure ERR" in main
    assert "rc=$?; trap - ERR; set +e" in main
    assert 'exit "$rc"' in main
    assert re.search(r"BASH\nthen\n.*?else\n.*?fi", main, re.S)


def test_default_sequence_only_capacity_correctness_serial_then_eager_b16():
    source = PATH.read_text()
    commands = re.findall(r"^bash scripts/run_decode_scan.sh (capacity|run)([^\n]+)",
                          source, flags=re.M)
    assert commands == [
        ("capacity", " --single-point --batch 16 --mode serial"),
        ("capacity", " --single-point --batch 16 --mode serial-eager"),
        ("run", " --single-point --batch 16 --mode serial"),
        ("run", " --single-point --batch 16 --mode serial-eager"),
    ]
    assert source.index("gpu_check run") < source.index(
        "run --single-point --batch 16 --mode serial\n"
    )
    assert "--remaining" not in source
    assert "--warmup-steps 2 --window-seconds 30 --repeats 1" in source
    assert "--setup-timeout 900 --drain-timeout 60" in source
    assert "samples=None" in source
    assert "--observation buffered-live --identity-matching bound-prefix" in source


def test_source_paths_final_sha_template_and_separate_evidence_acceptance():
    source = PATH.read_text()
    assert "export SR_FIXED_COMMIT=__FINAL_FULL_SHA__" in source
    assert "git fetch origin codex/rolling-eager-v0.1" in source
    assert 'git checkout --detach "$SR_FIXED_COMMIT"' in source
    assert "/root/autodl-tmp/src/SpecRhythm" in source
    assert "/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11" in source
    assert "s1p-5a00049-20260909T144802Z-1469" in source
    assert "INJECTED_DIAGNOSTIC" in source and "NOT_OBSERVED" in source
    assert "UNKNOWN" in source and "native CUDA clock bounds" in source
    assert "life['accepted_promoted_candidates'] <= life['verified_promoted_candidates']" in source
    assert "c['started'] > 0" in source
    assert "gpu_check stop --root" in source
    assert "run_decode_scan.sh stop --wait-seconds 65" in source
    assert 'bundle --output "${SR_FIXED_ROOT}-failure-small.tar.gz"' in source
