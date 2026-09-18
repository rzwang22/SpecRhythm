"""Execute the exact four-point shell flow with CPU command substitutes."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_eager_latency_runbook import executable

SCRIPT = Path("scripts/run_audit_layers_b16.sh")
SHA = "a" * 40
POINTS = ["serial:full", "serial-eager:full", "serial:runtime", "serial-eager:runtime"]


@pytest.mark.parametrize(
    "failure,point",
    [
        ("capacity", POINTS[0]),
        ("correctness", POINTS[0]),
        ("correctness_gate", POINTS[1]),
        ("measurement", POINTS[1]),
        ("report", POINTS[2]),
        ("evidence", POINTS[2]),
        ("correctness", POINTS[3]),
        ("measurement", POINTS[3]),
        ("none", ""),
    ],
)
def test_four_points_preserve_first_error_stop_and_parent_shell(tmp_path, failure, point):
    repo, binary, results = (tmp_path / n for n in ("repo", "bin", "results"))
    for p in (repo / "scripts", binary, results):
        p.mkdir(parents=True)
    log = tmp_path / "calls.txt"
    executable(
        binary / "git",
        f'#!/bin/bash\nif [[ "$1" == rev-parse ]]; then printf "%s\\n" "{SHA}"; fi\n',
    )
    executable(binary / "sha256sum", "#!/bin/bash\nexit 0\n")
    executable(
        repo / "scripts/run_decode_scan.sh",
        """#!/bin/bash
point="$SR_AUDIT_SERVING_MODE:$SR_AUDIT_MODE"
printf '%s cli %s\n' "$point" "$*" >> "$CALLS"
if [[ "$1" == prepare ]]; then mkdir -p "$SR_FIXED_ROOT"; fi
if [[ "$1" == capacity && "$FAIL_STAGE" == capacity && "$FAIL_POINT" == "$point" ]]; then
  exit 23
fi
if [[ "$1" == bundle ]]; then printf '{}' > "$3"; fi
""",
    )
    fake = binary / "python"
    executable(
        fake,
        "#!"
        + sys.executable
        + "\n"
        + """import os, pathlib, sys
args=sys.argv[1:]
body=sys.stdin.read() if args and args[0]=='-' else ''
point=os.environ.get('SR_AUDIT_SERVING_MODE','')+':'+os.environ.get('SR_AUDIT_MODE','')
kind=('correctness' if args[:3]==['-m','specrhythm.continuation.gpu_check','run'] else
 'report' if args[:2]==['-m','specrhythm.serving.audit_layer_report'] else
 'correctness_gate' if "physical_gpu_correctness" in body else
 'measurement' if "formal_comparison_eligible" in body else
 'evidence' if "trace_coverage" in body else 'python')
with open(os.environ['CALLS'],'a') as f: f.write(point+' '+kind+' '+str(args)+'\\n')
if kind==os.environ['FAIL_STAGE'] and point==os.environ['FAIL_POINT']: sys.exit(23)
if '--output' in args: pathlib.Path(args[args.index('--output')+1]).write_text('{}')
""",
    )
    block = SCRIPT.read_text().replace('main "$@"', "main " + SHA)
    block = block.replace("/root/autodl-tmp/src/SpecRhythm", str(repo))
    block = block.replace(
        "/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11", str(fake)
    )
    block = block.replace("/root/autodl-tmp/SpecRhythm-data/results/rolling-eager", str(results))
    block = (
        "if bash <<'INNER'\n"
        + block
        + "\nINNER\nthen :; else printf 'Stopped rc=%s\\n' \"$?\"; fi\nprintf 'PARENT_ALIVE\\n'\n"
    )
    env = {
        **os.environ,
        "PATH": str(binary) + os.pathsep + os.environ["PATH"],
        "CALLS": str(log),
        "FAIL_STAGE": failure,
        "FAIL_POINT": point,
    }
    result = subprocess.run(
        ["bash"], input=block, env=env, text=True, capture_output=True, timeout=20
    )
    assert result.returncode == 0 and "PARENT_ALIVE" in result.stdout, result.stderr
    rows = log.read_text().splitlines()
    prepared = [r.split()[0] for r in rows if "cli prepare " in r]
    if failure == "none":
        assert prepared == POINTS
        assert len(list(results.glob("*-complete-evidence.tar.gz"))) == 4
        assert len(list(results.glob("audit-four-point-*.json"))) == 1
    else:
        assert "Stopped rc=23" in result.stdout
        assert prepared == POINTS[: POINTS.index(point) + 1]
        assert len(list(results.glob("*-failure-evidence.tar.gz"))) == 1
        assert not list(results.glob("audit-four-point-*.json"))
    for row in rows:
        if "cli prepare " in row:
            assert "--draft-audit " + row.split()[0].split(":")[1] in row
        if "cli run " in row or "cli capacity " in row:
            assert "--single-point --batch 16 --mode " + row.split()[0].split(":")[0] in row


def test_four_point_fixed_semantics_and_no_cross_run_identity_gate():
    source = SCRIPT.read_text()
    for text in (
        "--warmup-steps 2 --window-seconds 30 --repeats 1",
        "--setup-timeout 900 --drain-timeout 60",
        "--observation buffered-live --identity-matching bound-prefix",
        "opts['samples'] is None",
        "pool_size",
        "==360",
    ):
        assert text in source
    assert source.index("gpu_check run") < source.index("run --single-point --batch 16")
    assert "gpu_uuid" not in source and "pingpong" not in source
