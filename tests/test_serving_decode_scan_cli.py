"""CPU scan orchestration: frozen twelve points, no reruns, failure retention/inspection."""

import json
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
    publish(
        directory / "measurement-snapshot.json",
        {
            "committed_window_tokens": 321,
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
    assert out.stat().st_size < 10 * 1024**2


def test_failed_capacity_is_not_ignored_before_decode(root, monkeypatch):
    report(root, selected_point("serial", 16), valid=False, measurement="INVALID", probe=True)
    monkeypatch.setattr(cli, "run_point", lambda *a, **kw: pytest.fail("GPU called"))
    with pytest.raises(DataError, match="capacity probe failed"):
        cli.run(root, batch=16)


def test_inspection_imports_neither_gpu_framework_nor_full_audit(root):
    source = "import sys; from specrhythm.serving import decode_scan_cli as c; "
    source += f"from pathlib import Path; c.summary(Path({str(root)!r})); "
    source += "assert 'torch' not in sys.modules and 'vllm' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", source], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
    assert cli.main(["status", "--root", str(root)]) == 0
    assert cli.main(["errors", "--root", str(root)]) == 0
    assert read_json(root / "scan-config.json")["options"]["samples"] is None
