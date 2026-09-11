"""Failure snapshots and real bounded owned-process exit, independent of GPU/audit."""

import json
import sys
import tarfile
import time

from specrhythm.phase4.process_lifecycle import run_owned_target
from specrhythm.serving import fixed_cli, fixed_results
from specrhythm.serving.common import read_json
from specrhythm.serving.fixed_artifacts import record_error
from specrhythm.serving.fixed_plan import point, settings
from specrhythm.serving.s2_pool import publish


def test_existing_supervisor_bounds_blocked_drain_and_preserves_partial_point(
    tmp_path, monkeypatch
):
    directory = tmp_path / "runs/serial"
    directory.mkdir(parents=True)
    publish(directory / "point.json", point("serial"))
    script = """
import pathlib, signal, sys, time
from specrhythm.serving.s2_pool import publish
p = pathlib.Path(sys.argv[1])
signal.signal(signal.SIGTERM, signal.SIG_IGN)
publish(p / "measurement-snapshot.json", {
    "mode": "serial", "sample_count": 1, "committed_window_tokens": 64,
    "measurement_availability": "UNQUALIFIED_PARTIAL", "measurement_complete": True,
    "drain_complete": False, "formal_comparison_eligible": False})
publish(p / "drain-state.json", {"status": "RUNNING", "phase": "blocked_worker_RPC",
    "deadline_ns": time.monotonic_ns() + 150_000_000, "settled_requests": 0})
time.sleep(60)
"""
    started = time.monotonic()
    rc, lifecycle = run_owned_target(
        [sys.executable, "-c", script, str(directory)],
        target_log=directory / "target.log",
        artifact_path=directory / "process-lifecycle.json",
        phase_deadline_path=directory / "drain-state.json",
        timeout_seconds=30,
        graceful_seconds=0.1,
        kill_seconds=1,
        poll_seconds=0.01,
    )
    assert time.monotonic() - started < 5 and rc == 124
    assert not lifecycle["run_valid"] and lifecycle["owned_cleanup_completed"]
    assert not lifecycle["remaining_owned_pids"]
    assert lifecycle["failure_detection"]["phase"] == "blocked_worker_RPC"
    assert {r["signal"] for r in lifecycle["term_kill_actions"]} == {"SIGTERM", "SIGKILL"}
    publish(directory / "exit-code.json", {"effective_exit_code": rc})
    assert not (directory / "runtime.json").exists()
    assert not (directory / "draft-backend-report.json").exists()
    monkeypatch.setattr(
        fixed_results,
        "offline_audit",
        lambda *a: (_ for _ in ()).throw(AssertionError("audit launched")),
    )
    summary = fixed_results.comparisons(tmp_path)
    assert summary["partial_measurements"][0]["measurement"]["committed_window_tokens"] == 64
    assert summary["points"][0]["measurement_status"] == "INVALID"
    assert all(v["status"] == "PENDING" for v in summary["measured_comparisons"].values())
    status = fixed_cli.status(tmp_path)
    assert status["results"][0]["effective_exit_code"] == 124
    output = tmp_path / "small.tar.gz"
    fixed_cli.bundle(tmp_path, output)
    with tarfile.open(output) as tar:
        names = tar.getnames()
        for name in (
            "measurement-snapshot.json",
            "drain-state.json",
            "exit-code.json",
            "process-lifecycle.json",
        ):
            assert "runs/serial/" + name in names


def test_first_failure_survives_secondary_and_missing_final_reports(tmp_path, capsys):
    record_error(tmp_path, ValueError("first real release failure"), "draft_release")
    record_error(tmp_path, RuntimeError("secondary engine shutdown failure"), "engine_cleanup")
    result = fixed_results.emit_result(
        tmp_path,
        {
            "valid": False,
            "effective_exit_code": 17,
            "errors": ["runtime.json missing"],
            "secondary_diagnostics": ["runtime.json missing"],
        },
        point("serial"),
    )
    assert result["primary_error"]["error"] == "first real release failure"
    assert result["cleanup_diagnostics"][0]["error"] == "secondary engine shutdown failure"
    assert result["effective_exit_code"] == 17 and not result["valid"]
    assert "first real release failure" in capsys.readouterr().out
    assert read_json(tmp_path / "light-summary.json")["measurement_status"] == "INVALID"


def test_operator_stop_skips_following_points_and_repeat_stop_is_noop(tmp_path, monkeypatch):
    publish(tmp_path / "diagnostic-config.json", {"options": settings()})
    monkeypatch.setattr(fixed_cli, "capacity_passed", lambda *a: None)
    called = []

    def run(root, selected, **kw):
        called.append(selected["mode"])
        publish(root / "stop-request.json", {"reason": "operator_stop"})
        return root, {}

    monkeypatch.setattr(fixed_cli, "run_point", run)
    assert fixed_cli.main(["short", "--root", str(tmp_path)]) == 0
    assert called == ["target"]
    directory = tmp_path / "runs/one"
    directory.mkdir(parents=True)
    publish(tmp_path / "stage.json", {"state": "COMPLETE", "directory": str(directory)})
    assert fixed_cli.stop(tmp_path) == fixed_cli.stop(tmp_path)
    assert not (directory / "stop-request.json").exists()


def test_snapshot_primary_survives_cleanup_report_write_failure(tmp_path, monkeypatch):
    from specrhythm.serving import fixed_artifacts

    record_error(tmp_path, RuntimeError("first"), "drain")
    original = (tmp_path / "diagnostic-primary-error.json").read_bytes()
    monkeypatch.setattr(
        fixed_artifacts, "publish", lambda *a: (_ for _ in ()).throw(OSError("disk full"))
    )
    record_error(tmp_path, ValueError("secondary"), "cleanup")
    assert (tmp_path / "diagnostic-primary-error.json").read_bytes() == original
    assert json.loads(original)["error"] == "first"


def test_errors_displays_retained_snapshot_without_final_reports(tmp_path, capsys):
    directory = tmp_path / "runs/failed"
    directory.mkdir(parents=True)
    publish(tmp_path / "stage.json", {"directory": str(directory), "state": "FAILED"})
    publish(
        directory / "measurement-snapshot.json",
        {
            "committed_window_tokens": 64,
            "sample_count": 1,
            "measurement_complete": True,
            "drain_complete": False,
            "measurement_availability": "UNQUALIFIED_PARTIAL",
        },
    )
    record_error(directory, RuntimeError("first drain error"), "draft_release")
    assert fixed_cli.main(["errors", "--root", str(tmp_path)]) == 0
    shown = capsys.readouterr().out
    assert "first drain error" in shown and '"committed_window_tokens": 64' in shown
    assert "runtime.json" in shown and "missing after execution failure" in shown


def test_concurrent_worker_coordinator_failures_cannot_replace_primary(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    record_error(tmp_path, ValueError("first"), "measurement")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda n: record_error(tmp_path, RuntimeError(str(n)), "cleanup"), range(8)))
    assert read_json(tmp_path / "diagnostic-primary-error.json")["error"] == "first"
    assert {r["error"] for r in read_json(tmp_path / "diagnostic-secondary-errors.json")} == {
        str(n) for n in range(8)
    }
