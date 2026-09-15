"""Actual report publication/socket/export paths, including interrupted byte streams."""

import json
import socket
import tarfile
import time
from types import SimpleNamespace

import pytest

from specrhythm.phase4 import draft_service, report_publication
from specrhythm.phase4.manifest import sha256_file
from specrhythm.serving.k3 import MODES
from specrhythm.serving.k3_local_run import validate_archive
from specrhythm.serving.ping_prepost_delivery import export


def payload(archive, name):
    with tarfile.open(archive) as bundle:
        inventory = json.load(bundle.extractfile("inventory.json"))
        return bundle.extractfile(inventory["logical_paths"][name]).read(), inventory


@pytest.mark.parametrize("fault", ["write", "fsync", "deadline", "none"])
def test_publication_partial_and_export_raw_bytes(tmp_path, monkeypatch, fault):
    directory = tmp_path / "delivery"
    directory.mkdir()
    path = directory / "draft-backend-report.json"
    deadline = time.monotonic_ns() + 10_000_000_000
    (directory / "target.log").write_bytes(b"raw GPU-process log\xff\n")
    original_dump, original_fsync = json.dump, report_publication.os.fsync

    def dump(value, handle, **kwargs):
        if "report_publication" in value and fault == "write":
            handle.write('{"provenance":')
            raise OSError("injected report write interruption")
        return original_dump(value, handle, **kwargs)

    def sync(fd):
        if fault == "fsync":
            # Only the report fsync, after WRITING marker was atomically published.
            from specrhythm.io_context import IO_CONTEXT

            if getattr(IO_CONTEXT, "name", None) == path.name:
                raise OSError("injected report fsync interruption")
        return original_fsync(fd)

    def build():
        assert not path.exists()
        marker = json.loads((directory / "draft-report-state.json").read_text())
        assert marker["status"] == "WRITING"
        with pytest.raises(ValueError, match="incomplete"):
            report_publication.qualify_final_report(path)
        if fault == "deadline":
            monkeypatch.setattr(report_publication.time, "monotonic_ns", lambda: deadline + 1)
        return {"provenance": {"fixture": "CPU only"}}

    monkeypatch.setattr(report_publication.json, "dump", dump)
    monkeypatch.setattr(report_publication.os, "fsync", sync)
    if fault == "none":
        state = report_publication.publish_final_report(
            path, build, deadline_ns=deadline, owner_stopped=True
        )
        assert state == report_publication.qualify_final_report(path)
        assert state["sha256"] == sha256_file(path)
        assert not list(directory.glob("*.partial"))
    else:
        with pytest.raises((OSError, TimeoutError), match="interruption|deadline"):
            report_publication.publish_final_report(
                path, build, deadline_ns=deadline, owner_stopped=True
            )
        assert not path.exists()
        state = json.loads((directory / "draft-report-state.json").read_text())
        assert state["status"] == "FAILED" and not state["final_file_published"]
        partial = directory / state["temporary_file"]
        assert partial.exists()
        with pytest.raises(ValueError, match="incomplete"):
            report_publication.qualify_final_report(path)
    # A malformed pre-existing final report is independent from the injected partial.
    corrupt = directory / "points/serial-k3/runs/broken/draft-backend-report.json"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b'{"report":\xff')
    archive = tmp_path / "delivery.tar.gz"
    result = export(directory, archive, first_code=125, stage="capacity", modes=MODES)
    assert result["first_exit_code"] == 125 and result["export_validation_exit_code"] == 41
    assert (
        result["archive_integrity"] == "COMPLETE" and result["evidence_integrity"] == "INCOMPLETE"
    )
    assert validate_archive(archive)["archive_integrity"] == "VERIFIED"
    assert payload(archive, str(corrupt.relative_to(directory)))[0] == corrupt.read_bytes()
    assert payload(archive, "target.log")[0] == b"raw GPU-process log\xff\n"
    if fault != "none":
        assert payload(archive, partial.name)[0] == partial.read_bytes()


def test_complete_marker_cannot_hide_corrupt_report(tmp_path):
    path = tmp_path / "draft-backend-report.json"
    report_publication.publish_final_report(
        path, lambda: {}, deadline_ns=time.monotonic_ns() + 10_000_000_000, owner_stopped=True
    )
    path.write_bytes(b"{")
    with pytest.raises(ValueError, match="mismatched"):
        report_publication.qualify_final_report(path)


@pytest.mark.parametrize("invalid_request", [False, True])
def test_socket_failure_never_sends_twice_or_replaces_original(
    tmp_path, monkeypatch, invalid_request
):
    server = draft_service.DraftUnixServer(
        tmp_path / "unused", None, event_log=SimpleNamespace(append=lambda r: None)
    )
    server._dispatch = lambda *a: {}
    left, right = socket.socketpair()
    draft_service.send_message(
        right,
        dict(
            protocol_version=("bad" if invalid_request else draft_service.PROTOCOL_VERSION),
            operation="info",
            payload={},
        ),
    )
    right.close()
    send, attempts = draft_service.send_message, []

    def observed(*args):
        attempts.append(args[1])
        return send(*args)

    monkeypatch.setattr(draft_service, "send_message", observed)
    try:
        with pytest.raises(ValueError if invalid_request else BrokenPipeError) as error:
            server._handle(left)
    finally:
        left.close()
    assert len(attempts) == 1
    if invalid_request:
        assert (
            "incompatible" in str(error.value) and "BrokenPipeError" in error.value.response_error
        )


def test_target_only_real_service_factory_uses_publication(tmp_path, monkeypatch):
    from test_serial_eager_owner import OwnerWorker

    from specrhythm.phase4.vllm_draft_backend import VllmBatchedDraftBackend
    from specrhythm.serving import fixed_draft, fixed_logging
    from specrhythm.serving.fixed_settle import (
        DiagnosticSerialServer,
        PublishedDiagnosticSerialMachine,
    )

    class Hardware(VllmBatchedDraftBackend):
        def __init__(self, config):
            super().__init__(config, worker=OwnerWorker())

    seen = []

    def serve(server, ready):
        assert isinstance(server.machine, PublishedDiagnosticSerialMachine)
        result = server._dispatch("shutdown", {"deadline_ns": time.monotonic_ns() + 10**10})
        seen.append(result)

    monkeypatch.setenv("SR_K3_MANAGED_LOCAL", "1")
    monkeypatch.setattr(DiagnosticSerialServer, "serve", serve)
    monkeypatch.setattr(fixed_logging, "_CURRENT", None)
    fixed_draft.serve(
        SimpleNamespace(max_model_len=4096),
        tmp_path,
        tmp_path / "unused.sock",
        "target",
        backend_class=Hardware,
    )
    report = tmp_path / "draft-backend-report.json"
    assert report_publication.qualify_final_report(report)["status"] == "COMPLETE"
    assert seen[0]["shutdown"] and seen[0]["draft_backend_report_sha256"] == sha256_file(report)
    archive = tmp_path.parent / (tmp_path.name + ".tar.gz")
    export(tmp_path, archive, first_code=23, modes=MODES)
    _, inventory = payload(archive, "draft-backend-report.json")
    row = next(r for r in inventory["inventory"] if r["path"] == "draft-backend-report.json")
    state = json.loads(payload(archive, "draft-report-state.json")[0])
    # Export intentionally projects the report; original digest/size is explicit.
    assert (
        report_publication.qualify_publication_receipt(
            state, report_file=report.name, size=row["source_bytes"], sha256=row["source_sha256"]
        )["status"]
        == "COMPLETE"
    )
    with pytest.raises(ValueError, match="mismatched"):
        report_publication.qualify_publication_receipt(
            state,
            report_file=report.name,
            size=row["source_bytes"] + 1,
            sha256=row["source_sha256"],
        )
