"""Execute the exact foreground pair flow; substitute external GPU commands only."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_eager_latency_runbook import executable

SCRIPT = Path("scripts/run_ping_prepost_b16.sh")
SHA = "a" * 40
MODES = ("pingpong-prepost3", "pingpong-eager-prepost3")


@pytest.mark.parametrize(
    "failure,point,export_error",
    [
        ("capacity", MODES[0], False),
        ("capacity", MODES[1], False),
        ("correctness", MODES[1], False),
        ("execution", MODES[0], False),
        ("measurement", MODES[1], False),
        ("evidence", MODES[0], False),
        ("evidence", MODES[0], True),
        ("none", "", True),
        ("none", "", False),
    ],
)
def test_one_bundle_first_error_stops_points_and_parent_remains_open(
    tmp_path, failure, point, export_error
):
    repo, binary, results = (tmp_path / n for n in ("repo", "bin", "results"))
    for path in (repo / "scripts", binary, results):
        path.mkdir(parents=True)
    log = tmp_path / "calls.txt"
    executable(
        binary / "git", '#!/bin/bash\nif [[ "$1" == rev-parse ]]; then echo ' + SHA + "; fi\n"
    )
    executable(
        repo / "scripts/run_decode_scan.sh",
        """#!/bin/bash
printf '%s %s\n' "$SR_AUDIT_SERVING_MODE" "$1" >> "$CALLS"
if [[ "$1" == prepare ]]; then mkdir -p "$SR_FIXED_ROOT"; fi
stage="$1"; if [[ "$stage" == run ]]; then stage=execution; fi
if [[ "$stage" == "$FAIL_STAGE" && "$SR_AUDIT_SERVING_MODE" == "$FAIL_POINT" ]]; then exit 23; fi
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
mode=os.environ.get('SR_AUDIT_SERVING_MODE','')
kind=('export' if 'specrhythm.serving.ping_prepost_delivery' in args else
 'correctness' if 'specrhythm.serving.ping_prepost_gpu_check' in args else
 'evidence' if 'specrhythm.serving.ping_prepost_evidence' in args else
 'measurement' if 'formal_comparison_eligible' in body else
 'failure' if 'summarize(root' in body else 'other')
with open(os.environ['CALLS'],'a') as f: f.write(mode+' '+kind+'\\n')
if kind=='export':
 pathlib.Path(args[args.index('--output')+1]).write_bytes(b'bounded package stand-in')
 sys.exit(41 if os.environ['EXPORT_ERROR']=='yes' else 0)
if kind==os.environ['FAIL_STAGE'] and mode==os.environ['FAIL_POINT']: sys.exit(23)
""",
    )
    script = tmp_path / "run.sh"
    script.write_text(SCRIPT.read_text())
    block = (
        'if bash "' + str(script) + '" ' + SHA + "; then echo PASSED; "
        'else echo "STOPPED=$?"; fi\necho PARENT_ALIVE\n'
    )
    env = {
        **os.environ,
        "PATH": str(binary) + os.pathsep + os.environ["PATH"],
        "SR_EXEC_REPO": str(repo),
        "SR_FIXED_PYTHON": str(fake),
        "SR_PING_RESULTS": str(results),
        "SR_PING_RUN_TAG": "fixture",
        "CALLS": str(log),
        "FAIL_STAGE": failure,
        "FAIL_POINT": point,
        "EXPORT_ERROR": "yes" if export_error else "no",
    }
    result = subprocess.run(
        ["bash"], input=block, env=env, text=True, capture_output=True, timeout=20
    )
    assert result.returncode == 0 and "PARENT_ALIVE" in result.stdout, result.stderr
    calls = log.read_text().splitlines()
    assert sum(r.endswith(" export") for r in calls) == 1
    assert result.stdout.count("UPLOAD ONLY:") == 1
    assert len(list(results.glob("*.tar.gz"))) == 1
    assert not list(results.rglob("*-evidence.tar.gz"))
    if failure == "none":
        assert [r for r in calls if r.endswith(" run")] == [m + " run" for m in MODES]
        assert ("STOPPED=41" if export_error else "PASSED") in result.stdout
    else:
        assert "STOPPED=23" in result.stdout
        assert "FIRST FAILURE: rc=23" in result.stdout
        failing = point + " " + ("run" if failure == "execution" else failure)
        assert calls.index(failing) < next(i for i, r in enumerate(calls) if r.endswith(" export"))
        assert not any(
            r.endswith((" run", " capacity", " correctness", " evidence"))
            for r in calls[calls.index(failing) + 1 :]
        )
        if export_error:
            assert "Export rc=41; original rc=23" in result.stdout
    assert not any(" status" in r or " errors" in r for r in calls)


def test_fixed_pair_config_no_grid_or_cross_run_uuid():
    s = SCRIPT.read_text()
    for required in (
        "--single-point --batch 16",
        "--draft-audit runtime",
        "--warmup-steps 2 --window-seconds 30 --repeats 1",
        "--setup-timeout 900 --drain-timeout 60",
        "==360",
        "==(16,8,8)",
        "--observation buffered-live --identity-matching bound-prefix",
        "ping_prepost_gpu_check",
        "ping_prepost_evidence",
        "ping_prepost_delivery",
    ):
        assert required in s
    assert "MODES=(pingpong-prepost3 pingpong-eager-prepost3)" in s
    assert "trap finish EXIT" in s and "trap - EXIT ERR" in s
    assert "cross_run_uuid" not in s and "sleep" not in s
