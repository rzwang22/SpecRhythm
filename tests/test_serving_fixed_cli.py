"""Independent foreground launcher, bounded stop and CPU-only light paths."""

import json
import sys
import tarfile
import tempfile
from pathlib import Path

import pytest
from test_serving_s2 import profile

from specrhythm.serving import fixed_cli, s2_cli
from specrhythm.serving.common import DataError, read_json
from specrhythm.serving.fixed_plan import point, settings
from specrhythm.serving.s1_preflight import clean_environment
from specrhythm.serving.s1_workload import write_once
from specrhythm.serving.s2_plan import sealed


def test_default_environment_preserves_live_and_strips_diagnostic_leak(monkeypatch):
    monkeypatch.setenv("SR_FIXED_POINT", "/old/point.json")
    monkeypatch.setenv("SR_FIXED_OBSERVATION", "buffered-live")
    for mode in ("target", "serial", "pingpong"):
        env = clean_environment(mode)
        assert "SR_FIXED_POINT" not in env
        assert "SR_FIXED_OBSERVATION" not in env
        assert env["SR_PHASE4_DUAL_UUID_QUERY_MODE"] == "live"


def test_light_summary_and_bundle_never_load_heavy_audit_or_execution(tmp_path, monkeypatch):
    from specrhythm.serving import fixed_results

    monkeypatch.setattr(fixed_cli, "run_point", lambda *a, **kw: pytest.fail("runtime launched"))
    monkeypatch.setattr(
        fixed_results, "offline_audit", lambda *a: pytest.fail("heavy audit called")
    )
    directory = tmp_path / "runs/one"
    directory.mkdir(parents=True)
    write_once(directory / "light-summary.json", {"valid": True})
    (directory / "target-diagnostics.jsonl").write_text("huge raw data excluded\n" * 1000)
    output = tmp_path / "small.tar.gz"
    assert fixed_cli.main(["bundle", "--root", str(tmp_path), "--output", str(output)]) == 0
    with tarfile.open(output) as tar:
        assert tar.getnames() == ["runs/one/light-summary.json"]
    assert fixed_cli.main(["summary", "--root", str(tmp_path)]) == 0


def test_stop_is_bounded_owned_cleanup_without_foreign_pid_access(tmp_path, monkeypatch):
    from specrhythm.serving import s1_cli

    directory = tmp_path / "runs/one"
    directory.mkdir(parents=True)
    write_once(tmp_path / "stage.json", {"state": "RUNNING", "directory": str(directory)})
    calls = []
    monkeypatch.setattr(
        s1_cli, "cleanup_attempt", lambda path: calls.append(path) or {"valid": True}
    )
    result = fixed_cli.stop(tmp_path, 0)
    assert calls == [directory] and result["state"] == "FORCED_OWNED_CLEANUP"
    assert read_json(directory / "stop-request.json")["reason"] == "operator_stop"


def test_short_runs_only_four_points_without_s2_grid(tmp_path, monkeypatch):
    write_once(tmp_path / "diagnostic-config.json", {"options": settings()})
    monkeypatch.setattr(fixed_cli, "capacity_passed", lambda root: None)
    modes = []

    def run(root, unit, **kw):
        modes.append(unit["mode"])
        return tmp_path, {}

    monkeypatch.setattr(fixed_cli, "run_point", run)
    assert fixed_cli.main(["short", "--root", str(tmp_path)]) == 0
    assert modes == ["target", "serial", "serial-split", "pingpong"]


def test_diagnostic_launcher_preserves_real_rc_and_source_logs(tmp_path, monkeypatch, capsys):
    path, manifest, _ = profile(
        tmp_path / "inputs",
        execution={"git_commit": "a" * 40, "eos_token_ids": [999], "vllm_source": str(tmp_path)},
    )
    manifest.pop("sha256")
    manifest["fixed_diagnostic"] = {"options": settings()}
    path.write_text(json.dumps(sealed(manifest)))
    monkeypatch.setattr(s2_cli, "validate_execution_files", lambda *a: None)
    monkeypatch.setattr(s2_cli, "qualify", lambda *a: pytest.fail("heavy qualifier called"))
    with tempfile.TemporaryDirectory(prefix="sr-fixed-", dir="/tmp") as short:
        directory = Path(short)
        selected = point("serial-split")
        write_once(directory / "point.json", selected)

        def child(kind, root, mode, manifest, out, probe=False, *, diagnostic=False):
            assert diagnostic is True and mode == "pingpong"
            if kind == "draft-child":
                script = """import os,socket,time
from pathlib import Path
s=socket.socket(socket.AF_UNIX)
s.bind(os.environ['SR_S2_DRAFT_SOCKET']); s.listen()
Path(os.environ['SR_S2_RUN_DIRECTORY'],'draft-service-ready.json').write_text('{}')
print('real CPU Draft stdout',flush=True)
time.sleep(60)
"""
            else:
                script = """import os,json,sys
from pathlib import Path
Path(os.environ['SR_S2_RUN_DIRECTORY'],'child-failure.json').write_text(
json.dumps({'error':'first worker failure','field':'worker start'}))
print('real CPU Target stderr',file=sys.stderr,flush=True)
raise SystemExit(23)
"""
            return [sys.executable, "-c", script]

        monkeypatch.setattr(s2_cli, "child_command", child)
        with pytest.raises(DataError) as error:
            s2_cli.execute(tmp_path, "fixed", "pingpong", path, directory, diagnostic=selected)
        assert error.value.details["returncode"] == 23
        report = read_json(directory / "light-summary.json")
        assert report["effective_exit_code"] == 23 and report["mode"] == "serial-split"
        assert report["primary_error"]["error"] == "first worker failure"
        assert report["secondary_diagnostics"]
        output = capsys.readouterr()
        assert "serial-split Target" in output.out and "serial-split Draft" in output.out
        assert "first worker failure" in output.err
        assert read_json(directory / "process-lifecycle.json")["cleanup_valid"]
        assert (directory / "light-summary.csv").exists()
