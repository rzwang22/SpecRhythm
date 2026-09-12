"""Foreground S2 failure propagation with real owned CPU processes; no inference imports."""

import json
import sys
import tempfile
from pathlib import Path

import pytest
from test_serving_s2 import profile

from specrhythm.serving.common import DataError, read_json
from specrhythm.serving.s2_cli import execute, main


def test_foreground_dual_source_logs_primary_error_and_actual_exit(tmp_path, monkeypatch, capsys):
    from specrhythm.serving import s2_cli

    path, _, _ = profile(
        tmp_path / "inputs",
        execution={"git_commit": "a" * 40, "eos_token_ids": [999], "vllm_source": str(tmp_path)},
    )
    monkeypatch.setattr(s2_cli, "validate_execution_files", lambda *a: None)
    with tempfile.TemporaryDirectory(prefix="sr-s2-cli-", dir="/tmp") as short:
        directory = Path(short) / "run"
        directory.mkdir()

        def child(kind, root, mode, manifest, out, probe=False):
            if kind == "draft-child":
                source = """import os, socket, json, time
from pathlib import Path
s=socket.socket(socket.AF_UNIX)
s.bind(os.environ["SR_S2_DRAFT_SOCKET"])
s.listen()
Path(os.environ["SR_S2_RUN_DIRECTORY"],"draft-service-ready.json").write_text('{}')
print("CPU Draft ready",flush=True)
time.sleep(600)
"""
            else:
                source = """import os, sys, json
from pathlib import Path
failure = {"error":"original startup failure", "field":"LLM construction",
           "expected":"worker created", "actual":"CPU injected exception"}
Path(os.environ["SR_S2_RUN_DIRECTORY"],"child-failure.json").write_text(json.dumps(failure))
print("CPU Target stderr evidence",file=sys.stderr,flush=True)
raise SystemExit(23)
"""
            return [sys.executable, "-c", source]

        monkeypatch.setattr(s2_cli, "child_command", child)
        with pytest.raises(DataError) as error:
            execute(tmp_path, "G1", "serial", path, directory)
        assert error.value.details["returncode"] == 23
        assert read_json(directory / "exit-code.json")["coordinator_exit_code"] == 23
        report = read_json(directory / "result.json")
        assert report["primary_error"]["error"] == "original startup failure"
        assert report["secondary_diagnostics"]
        assert "CPU Target stderr evidence" in (directory / "target.log").read_text()
        output = capsys.readouterr()
        assert "[S2 G1 serial Target]" in output.out
        assert "[S2 G1 serial Draft]" in output.out
        assert "original startup failure" in output.err
        assert read_json(directory / "process-lifecycle.json")["cleanup_valid"]


def test_offline_summary_cannot_launch_children(tmp_path, monkeypatch):
    from specrhythm.serving import s2_cli

    monkeypatch.setattr(
        s2_cli, "attempt", lambda *a, **kw: pytest.fail("offline path launched runtime")
    )
    output = tmp_path / "light.json"
    assert main(["summary", "--root", str(tmp_path), "--output", str(output)]) == 0
    assert read_json(output)["results"] == []
    assert "torch" not in json.dumps(read_json(output))


def test_resume_cleans_newer_interruption_before_reusing_sealed_success(tmp_path, monkeypatch):
    from specrhythm.serving import s2_cli
    from specrhythm.serving.s1_workload import write_once
    from specrhythm.serving.s2_results import seal_result

    path, manifest, _ = profile(tmp_path / "inputs")
    base = tmp_path / "runs" / "G1" / path.parent.name / "serial"
    first, interrupted = base / "attempt-001", base / "attempt-002"
    first.mkdir(parents=True)
    interrupted.mkdir()
    report = {"valid": True, "execution_sha256": manifest["sha256"], "mode": "serial"}
    seal_result(first, report)
    write_once(interrupted / "draft-owner.json", {"root_pid": 123})
    # Real cleanup writes sidecars with the same glob prefix; these are not attempts.
    write_once(base / "attempt-001-cleanup-abcd.json", {"valid": True})
    cleaned = []
    monkeypatch.setattr(s2_cli, "cleanup_attempt", lambda directory: cleaned.append(directory))
    monkeypatch.setattr(s2_cli, "execute", lambda *a, **kw: pytest.fail("restarted valid run"))
    directory, actual = s2_cli.attempt(tmp_path, "G1", "serial", path)
    assert cleaned == [interrupted]
    assert directory == first and actual == report


def test_failed_attempt_reports_failed_stage_and_keeps_real_exit(tmp_path, monkeypatch):
    from specrhythm.serving import s2_cli

    path, _, _ = profile(tmp_path / "inputs")

    def fail(*args, **kwargs):
        raise DataError("original failure", returncode=23)

    monkeypatch.setattr(s2_cli, "execute", fail)
    with pytest.raises(DataError) as error:
        s2_cli.attempt(tmp_path, "G1", "serial", path)
    assert error.value.details["returncode"] == 23
    assert read_json(tmp_path / "stage.json")["state"] == "FAILED"


def test_startup_exception_has_explicit_primary_fields_and_secondary_reports(tmp_path):
    from specrhythm.serving.s1_workload import write_once
    from specrhythm.serving.s2_cli import failed_report

    path = tmp_path / "child-failure.json"
    write_once(path, {"error": "missing worker input", "type": "FileNotFoundError", "details": {}})
    value = failed_report(tmp_path, "serial", {"sha256": "a" * 64}, 23)
    assert value["effective_exit_code"] == 23
    assert value["primary_error"]["expected"] == "successful execution"
    assert value["primary_error"]["actual"] == "missing worker input"
    assert value["primary_error"]["artifact"] == str(path)
    assert value["primary_error"]["field"] == "child-failure.execution"
    assert len(value["secondary_diagnostics"]) == 2
