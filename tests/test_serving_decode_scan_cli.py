"""CPU scan orchestration: frozen twelve points, no reruns, failure retention/inspection."""

import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from specrhythm.serving import decode_scan_cli as cli
from specrhythm.serving.common import DataError, read_json
from specrhythm.serving.decode_scan_plan import BATCHES, MODES, SCHEMA, options, selected_point
from specrhythm.serving.s2_plan import sealed
from specrhythm.serving.s2_pool import publish


@pytest.fixture
def root(tmp_path):
    publish(
        tmp_path / "scan-config.json",
        sealed(
            dict(
                schema_version=SCHEMA,
                execution={"git_commit": "a" * 40},
                workload_sha256="work",
                pool_size=360,
                options=options(),
                points=[selected_point(m, b) for b in BATCHES for m in MODES],
            )
        ),
    )
    return tmp_path


def report(root, p, *, valid=True, measurement="PASS", probe=False):
    directory = root / "runs" / f"{p['mode']}-{p['batch']}-{'probe' if probe else 'decode'}"
    directory.mkdir(parents=True, exist_ok=True)
    value = dict(
        point={**p, "probe": probe},
        probe=probe,
        mode=p["mode"],
        batch=p["batch"],
        git_commit="a" * 40,
        workload_sha256="work",
        valid=valid,
        errors=[] if valid else ["primary"],
        measurement_status=measurement,
        execution_status="PASS" if valid else "FAILED",
        effective_exit_code=0 if valid else 23,
        formal_comparison_eligible=valid and measurement == "PASS" and not probe,
    )
    publish(directory / "process-lifecycle.json", {"cleanup_valid": True})
    publish(directory / "light-summary.json", value)
    return directory, {**value, "cleanup_status": "PASS"}


def test_fresh_engines_each_point_small_three_first_and_no_success_rerun(root, monkeypatch):
    invoked = []
    monkeypatch.setattr(cli, "point_manifest", lambda root, b: root / f"execution-B{b}.json")

    def run(root, p, *, probe, manifest_path):
        assert str(p["batch"]) in manifest_path.name
        invoked.append((p["mode"], p["batch"], probe))
        return report(root, p, probe=probe)

    monkeypatch.setattr(cli, "run_point", run)
    with pytest.raises(DataError, match="first finish"):
        cli.run(root, batch=128)
    cli.run(root, batch=16, probe=True)
    cli.run(root, batch=16)
    cli.run(root, remaining=True)
    cli.run(root, remaining=True)
    assert len(invoked) == 15 and len(set(invoked)) == 15
    result = cli.summary(root)
    assert result["valid_comparison_points"] == 12
    assert Path(result["artifact"]).with_suffix(".csv").exists()
    assert "TPOT" not in json.dumps(result) and result["full_offline_audit"] == "NOT_RUN"


@pytest.mark.parametrize(
    "measurement,valid", [("INVALID", False), ("INSUFFICIENT", True), ("STOPPED", True)]
)
def test_bad_point_stops_next_gpu_and_remains_excluded_and_bundleable(
    root, monkeypatch, measurement, valid
):
    invoked = []
    monkeypatch.setattr(cli, "point_manifest", lambda *a: root / "manifest.json")

    def run(root, p, **kw):
        invoked.append(p)
        return report(root, p, valid=valid, measurement=measurement)

    monkeypatch.setattr(cli, "run_point", run)
    with pytest.raises(DataError, match="subsequent GPU"):
        cli.run(root, batch=16)
    with pytest.raises(DataError, match="do not auto"):
        cli.run(root, batch=16)
    assert len(invoked) == 1
    result = cli.summary(root)
    assert result["valid_comparison_points"] == 0
    assert len(result["points"]) == 12
    directory = root / "runs/target-16-decode"
    from test_serving_decode_scan_results import evidence

    boundary = evidence("pingpong", "ABAAB")[0]["decode_scan"]["warmup_boundary"]
    publish(
        directory / "measurement-snapshot.json",
        {
            "committed_window_tokens": 321,
            "scan_warmup_boundary": boundary,
            "measurement_complete": False,
            "formal_comparison_eligible": False,
        },
    )
    publish(directory / "diagnostic-primary-error.json", {"error": "primary", "phase": "worker"})
    (directory / "target.log").write_text("raw trace not packed")
    out = root / "small.tar.gz"
    cli.bundle(root, out)
    with tarfile.open(out) as tar:
        names = tar.getnames()
        assert "runs/target-16-decode/measurement-snapshot.json" in names
        assert "runs/target-16-decode/diagnostic-primary-error.json" in names
        assert not any(n.endswith("target.log") for n in names)
        snapshot = json.load(tar.extractfile("runs/target-16-decode/measurement-snapshot.json"))
        assert snapshot["scan_warmup_boundary"] == boundary
    assert out.stat().st_size < 10 * 1024**2


def test_failed_capacity_is_not_ignored_before_decode(root, monkeypatch):
    report(root, selected_point("serial", 16), valid=False, measurement="INVALID", probe=True)
    monkeypatch.setattr(cli, "run_point", lambda *a, **kw: pytest.fail("GPU called"))
    with pytest.raises(DataError, match="capacity probe failed"):
        cli.run(root, batch=16)


def test_independent_single_point_waives_order_only_and_retains_failure_gates(root, monkeypatch):
    invoked = []
    monkeypatch.setattr(cli, "point_manifest", lambda root, b: root / f"execution-B{b}.json")

    def run(root, p, *, probe, manifest_path):
        assert p["test_order"] == "independent-single-point"
        assert str(p["batch"]) in manifest_path.name
        invoked.append((p["mode"], p["batch"], probe))
        return report(root, p, probe=probe)

    monkeypatch.setattr(cli, "run_point", run)
    with pytest.raises(DataError, match="first finish"):
        cli.run(root, batch=64, mode="pingpong")
    with pytest.raises(DataError, match="explicit mode"):
        cli.run(root, batch=64, single_point=True)
    for probe in (True, False):
        cli.run(root, batch=64, mode="pingpong", probe=probe, single_point=True)
    for mode in MODES:
        cli.run(root, batch=128, mode=mode, single_point=True)
    assert invoked == [("pingpong", 64, True), ("pingpong", 64, False),
                       *[(m, 128, False) for m in MODES]]
    assert cli.summary(root)["valid_comparison_points"] == 4
    report(root, selected_point("pingpong", 64), measurement="INSUFFICIENT")
    with pytest.raises(DataError, match="do not auto-continue"):
        cli.run(root, batch=128, mode="target", single_point=True)
    assert len(invoked) == 5


def test_inspection_imports_neither_gpu_framework_nor_full_audit(root):
    source = "import sys; from specrhythm.serving import decode_scan_cli as c; "
    source += f"from pathlib import Path; c.summary(Path({str(root)!r})); "
    source += "assert 'torch' not in sys.modules and 'vllm' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", source], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
    assert cli.main(["status", "--root", str(root)]) == 0
    assert cli.main(["errors", "--root", str(root)]) == 0
    assert read_json(root / "scan-config.json")["options"]["samples"] is None


def test_documented_failure_trap_preserves_first_rc_and_interactive_parent(tmp_path):
    text = (Path(__file__).resolve().parents[1] / "docs/decode-scan-runbook.md").read_text()
    code = re.findall(r"```bash\n(.*?)```", text, re.S)[0]
    assert code.startswith("if bash <<'BASH'\n") and code.rstrip().endswith("fi")
    trap = code[code.index("on_failure() {"):code.index("trap on_failure ERR")]
    child = (
        "set -Eeuo pipefail\nexport SR_FIXED_ROOT=/unused-cpu-fixture\n"
        # Replace inspection commands only; execute the actual documented trap body.
        'bash() { printf "inspection:%s\\n" "$*"; return 44; }\n'
        # Match the real CLI's external process exit (macOS Bash 3.2 treats
        # a bare `(exit 3)` subshell differently from a foreground command).
        + trap + "trap on_failure ERR\nsh -c 'exit 3'\necho GPU-NEXT-POINT\n"
    )
    # Exercise the documented outer conditional under an already strict parent.
    outer = code[code.index("then\n"):]
    parent = ("set -e\nif bash <<'CHILD'\n" + child + "CHILD\n" + outer
              + "printf 'parent-alive:%s\\n' \"$rc\"\n")
    out = subprocess.run(["bash"], input=parent, text=True, capture_output=True, timeout=5)
    assert out.returncode == 0 and "parent-alive:3" in out.stdout
    assert "GPU-NEXT-POINT" not in out.stdout
    assert all("run_decode_scan.sh " + c in out.stdout
               for c in ("status", "errors", "summary", "bundle"))
