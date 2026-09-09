"""CPU contracts with real backend report production and real owned subprocesses; no GPU."""

import io
import json
import sys
import threading

import pytest
from test_serving_s1 import execution, native_artifacts

from specrhythm.phase4.manifest import atomic_write_json
from specrhythm.serving.common import DataError, digest, read_json
from specrhythm.serving.s1_console import RunConsole, print_failure
from specrhythm.serving.s1_policy import G0_SCHEMA, policy_fields
from specrhythm.serving.s1_results import draft_backend_checks, inspect_run, seal_run
from specrhythm.serving.s1_workload import load_execution, write_once


@pytest.mark.parametrize(
    "field,actual",
    [
        (None, None),
        ("backend_name", "vllm-batched"),
        ("backend_name", "hf-persistent"),
        ("backend_shutdown_complete", False),
        ("draft_live_requests_final", 2),
        ("draft_live_requests_final", False),
        ("execution_failed", True),
        ("execution_failed", None),
    ],
)
def test_actual_backend_report_and_independent_field_failures(
    tmp_path, monkeypatch, field, actual
):
    path, directory = native_artifacts(tmp_path, monkeypatch, "serial", speculative=True)
    artifact = directory / "draft-backend-report.json"
    backend = read_json(artifact)
    # native_artifacts now calls VllmBatchedDraftBackend.report(), with a CPU worker only.
    assert backend["schema_version"] == "specrhythm.phase4b3-draft-backend.v1"
    assert backend["backend_name"] == "vllm-batched-paged-kv-draft"
    assert backend["provenance"]["backend"] == backend["backend_name"]
    assert backend["worker_resources"]["closed"] is True
    if field:
        if actual is None:
            backend.pop(field)
        else:
            backend[field] = actual
        artifact.write_text(json.dumps(backend))
    report = inspect_run(path, directory, "serial")
    checks = report["draft_backend_checks"]
    assert len(checks) == 4
    assert report["valid"] is (field is None), report
    assert report["cross_run_output_comparison"]["status"] == "NOT_REQUIRED"
    assert all(check["valid"] for key, check in checks.items() if key != field)
    if field:
        failed = checks[field]
        assert failed["valid"] is False and failed["actual"] == actual
        assert failed["artifact"] == str(artifact.resolve())
        assert field in report["errors"][0] and "expected=" in report["errors"][0]
        assert "actual=" in report["errors"][0] and str(artifact) in report["errors"][0]
        assert report["error_details"]["checks"] == [failed]


def test_multiple_backend_errors_print_all_fields_and_log_tails(tmp_path, monkeypatch, capsys):
    from specrhythm.serving import s1_cli

    path, directory = native_artifacts(tmp_path, monkeypatch)
    artifact = directory / "draft-backend-report.json"
    backend = read_json(artifact)
    backend.update(
        backend_name="vllm-batched",
        backend_shutdown_complete=False,
        draft_live_requests_final=2,
        execution_failed=True,
    )
    artifact.write_text(json.dumps(backend))
    (directory / "target.log").write_text("Target retained failure tail\n")
    (directory / "draft-service.log").write_text("Draft retained failure tail\n")
    report = inspect_run(path, directory, "target")
    seal_run(directory, report)
    before = {p: p.read_bytes() for p in directory.iterdir() if p.is_file()}

    def failed_gate(root, selected):
        atomic_write_json(root / "stage.json", dict(mode="target", directory=str(directory)))
        raise DataError("S1 run qualification failed", errors=report["errors"])

    monkeypatch.setattr(s1_cli, "gate", failed_gate)
    assert s1_cli.supervise(tmp_path, "G1", "cpu-qualification") == 1
    output = capsys.readouterr().err
    for field in draft_backend_checks(backend, artifact):
        assert f'"field": "{field}"' in output
    assert '"expected": "vllm-batched-paged-kv-draft"' in output
    assert '"actual": "vllm-batched"' in output
    assert str(artifact) in output
    assert "[S1-P target Target] Target retained failure tail" in output
    assert "[S1-P target Draft] Draft retained failure tail" in output
    assert before == {p: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    assert read_json(tmp_path / "exit-code.json")["exit_code"] == 1


class ObservedConsole(io.StringIO):
    def __init__(self):
        super().__init__()
        self.live = threading.Event()

    def write(self, text):
        count = super().write(text)
        if all(
            message in self.getvalue()
            for message in (
                "[S1-P target Target] Target stdout",
                "[S1-P target Target] Target stderr",
                "[S1-P target Draft] Draft stdout",
                "[S1-P target Draft] Draft stderr",
            )
        ):
            self.live.set()
        return count


@pytest.mark.parametrize("failure", ["target", "draft"])
def test_foreground_cli_real_exit_and_owned_cleanup(tmp_path, monkeypatch, capsys, failure):
    from specrhythm.serving import s1_cli

    root = tmp_path / "root"
    path, manifest, _ = execution(root / "s1-smoke4")
    manifest["execution"]["vllm_source"] = str(tmp_path / "cpu-source-only")
    manifest["manifest_sha256"] = digest(
        {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    )
    path.write_text(json.dumps(manifest))
    write_once(
        root / "g0.json",
        dict(
            schema_version=G0_SCHEMA,
            **policy_fields(),
            valid=True,
            execution=manifest["execution"],
        ),
    )
    monkeypatch.setattr(s1_cli, "load_execution", lambda path, **kw: load_execution(path))
    monkeypatch.setattr(s1_cli, "validate_execution_files", lambda *a: None)
    monkeypatch.setattr(s1_cli, "gate_capacity", lambda *a: None)
    console = ObservedConsole()
    monkeypatch.setattr(
        s1_cli,
        "RunConsole",
        lambda directory, mode: RunConsole(
            directory,
            mode,
            stream=console,
        ),
    )
    release = tmp_path / "release-target-after-live-output"
    commands = []

    def command(kind, *args):
        commands.append(kind)
        directory = args[args.index("--directory") + 1]
        if kind == "draft-child":
            script = """
import os, signal, socket, sys, time
from pathlib import Path
root = Path(sys.argv[1])
while not (root / "draft-owner.json").is_file():
    time.sleep(0.005)
print("Draft stdout", flush=True)
print("Draft stderr", file=sys.stderr, flush=True)
if sys.argv[2] == "draft":
    raise SystemExit(9)
signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))
sock = socket.socket(socket.AF_UNIX)
sock.bind(os.environ["SR_S1_DRAFT_SOCKET"])
sock.listen()
(root / "draft-service-ready.json").write_text("{}")
while True:
    time.sleep(0.01)
"""
            return [sys.executable, "-u", "-c", script, str(directory), failure]
        assert kind == "child"
        script = """
import sys, time
from pathlib import Path
print("Target stdout", flush=True)
print("Target stderr", file=sys.stderr, flush=True)
end = time.monotonic() + 5
while not Path(sys.argv[1]).exists() and time.monotonic() < end:
    time.sleep(0.01)
raise SystemExit(7 if Path(sys.argv[1]).exists() else 90)
"""
        return [sys.executable, "-u", "-c", script, str(release)]

    monkeypatch.setattr(s1_cli, "child_command", command)

    # The Target can exit 7 only after both live streams were displayed, before process exit.
    def release_after_display():
        if console.live.wait(timeout=8):
            release.touch()

    controller = threading.Thread(target=release_after_display, daemon=True)
    if failure == "target":
        controller.start()
    expected = 7 if failure == "target" else 9
    assert s1_cli.main(["gate", "--root", str(root), "--gate", "G1"]) == expected
    if failure == "target":
        controller.join(timeout=1)
        assert release.exists()
    directory = root / "G1/0-target/attempt-001"
    assert read_json(root / "exit-code.json")["exit_code"] == expected
    assert read_json(directory / "exit-code.json")["effective_exit_code"] == expected
    lifecycle = read_json(directory / "process-lifecycle.json")
    assert lifecycle["owned_cleanup_completed"] is True
    assert lifecycle["remaining_owned_pids"] == []
    assert lifecycle["draft_shutdown_result"]["valid"] is True
    assert commands == (["draft-child", "child"] if failure == "target" else ["draft-child"])
    assert (directory / "draft-service.log").read_text() == "Draft stdout\nDraft stderr\n"
    if failure == "target":
        assert (directory / "target.log").read_text() == "Target stdout\nTarget stderr\n"
        assert lifecycle["target_exit_status"] == 7
    else:
        assert read_json(directory / "exit-code.json")["draft_exit_code"] == 9
    error = capsys.readouterr().err
    assert "last 40 lines" in error and "Draft stderr" in error
    assert f'"effective_exit_code": {expected}' in error


def test_console_final_fragments_and_closed_display_keep_original_logs(tmp_path):
    target, draft = tmp_path / "target.log", tmp_path / "draft-service.log"
    target.write_bytes("no trailing newline 完整".encode())
    draft.write_bytes(b"stderr last fragment")
    before = (target.read_bytes(), draft.read_bytes())
    stream = io.StringIO()
    with RunConsole(tmp_path, "serial", stream=stream):
        pass
    assert "[S1-P serial Target] no trailing newline 完整" in stream.getvalue()
    assert "[S1-P serial Draft] stderr last fragment" in stream.getvalue()
    stream.close()
    with RunConsole(tmp_path, "serial", stream=stream):
        pass
    print_failure(DataError("original failure", returncode=7), directory=tmp_path, stream=stream)
    assert (target.read_bytes(), draft.read_bytes()) == before


def test_malformed_failure_evidence_still_prints_original_error_and_tails(tmp_path):
    (tmp_path / "result.json").write_text("null")
    (tmp_path / "process-lifecycle.json").write_text("[]")
    (tmp_path / "target.log").write_text("retained Target failure\n")
    stream = io.StringIO()
    print_failure(DataError("original failure", returncode=7), directory=tmp_path, stream=stream)
    output = stream.getvalue()
    assert '"returncode": 7' in output
    assert "original failure" in output and "retained Target failure" in output
    assert f"cannot read {tmp_path / 'result.json'}" in output
    assert f"cannot read {tmp_path / 'process-lifecycle.json'}" in output
