"""Collected owner joins -> bounded publication -> archive -> exact restored evidence."""

import copy
import json
import tarfile
from pathlib import Path

import pytest
import test_ping_prepost_evidence as collector
from test_k3 import machine

from specrhythm.serving import audit_details
from specrhythm.serving.audit_layer_report import MAX_OUTPUT, write
from specrhythm.serving.common import DataError
from specrhythm.serving.execution_evidence import qualify
from specrhythm.serving.ping_prepost_delivery import export
from specrhythm.serving.ping_prepost_evidence import mechanism, publish_qualified


@pytest.fixture
def generated(monkeypatch):
    monkeypatch.setattr(collector, "machine", machine)
    r, b = collector.collected(False, "pingpong-k3")
    ping = mechanism(r, b)
    assert ping["status"] == "COMPLETE", ping["errors"]
    return dict(mode="pingpong-k3", pingpong=ping, source_commit="cpu-produced",
                measurement_start_ns=r["measurement_start_ns"],
                measurement_end_ns=r["measurement_end_ns"])


def test_collected_report_exceeds_old_bound_then_lossless_single_archive(generated, tmp_path):
    # Scale the actual generated join rows to exercise storage, not pretend that
    # duplicated CPU fixtures are a new valid GPU run. Qualifier results stay equal.
    value = copy.deepcopy(generated)
    original = copy.deepcopy(value["pingpong"]["cycles"])
    copies = 2 * MAX_OUTPUT // len(json.dumps(original).encode()) + 1
    value["pingpong"]["cycles"] = original * copies
    assert len(json.dumps(value).encode()) > MAX_OUTPUT
    with pytest.raises(DataError, match="8 MiB"):
        write(value, tmp_path / "old.json")
    directory = tmp_path / "delivery"
    directory.mkdir()
    output = directory / "pingpong-k3-audit-report.json"
    audit_details.publish(value, output)
    compact = json.loads(output.read_text())
    assert output.stat().st_size < MAX_OUTPUT
    assert len(compact["detail_tables"]["files"]) >= 2
    restored = audit_details.restore_file(compact, directory)
    assert restored == value and qualify(restored) == qualify(value)
    archive = tmp_path / "single.tar.gz"
    export(directory, archive, first_code=23, stage="diagnostic_evidence")
    with tarfile.open(archive) as t:
        inv = json.load(t.extractfile("inventory.json"))
        assert inv["first_exit_code"] == 23
        paths = inv["logical_paths"]
        result = json.load(t.extractfile(paths[output.name]))
        roundtrip = audit_details.restore(result, lambda name: t.extractfile(paths[name]).read())
    assert roundtrip == value
    assert qualify(roundtrip) == qualify(value)


@pytest.mark.parametrize("fault", ["missing", "bytes", "order", "count", "binding", "shards"])
def test_external_evidence_faults_rejected(generated, tmp_path, fault):
    output = tmp_path / "pingpong-k3-audit-report.json"
    audit_details.publish(generated, output)
    summary = json.loads(output.read_text())
    entry = summary["detail_tables"]["files"][0]
    if fault == "missing":
        (tmp_path / entry["name"]).unlink()
    elif fault == "bytes":
        (tmp_path / entry["name"]).write_bytes(b"truncated")
    elif fault == "count":
        entry["rows"] += 1
    elif fault == "binding":
        summary["mode"] = "serial-k3"
    elif fault == "shards":
        summary["detail_tables"]["files"] *= 9
    else:
        import hashlib

        path = tmp_path / entry["name"]
        rows = path.read_text().splitlines()
        row = json.loads(rows[0])
        row["index"] = 1
        rows[0] = json.dumps(row)
        raw = ("\n".join(rows) + "\n").encode()
        path.write_bytes(raw)
        entry.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    with pytest.raises((DataError, OSError)):
        audit_details.restore_file(summary, tmp_path)


def test_export_rejects_missing_required_shard(generated, tmp_path):
    directory = tmp_path / "delivery"
    directory.mkdir()
    output = directory / "pingpong-k3-audit-report.json"
    audit_details.publish(generated, output)
    summary = json.loads(output.read_text())
    (directory / summary["detail_tables"]["files"][0]["name"]).unlink()
    result = export(directory, tmp_path / "failure.tar.gz", first_code=23)
    assert result["first_exit_code"] == 23
    assert result["export_status"] == "INCOMPLETE"
    assert any("audit details" in e for e in result["export_errors"])


@pytest.mark.parametrize("fault", ["summary_size", "detail_size", "write", "secondary"])
def test_publication_failure_records_original_qualification(
    generated, tmp_path, monkeypatch, fault
):
    value = copy.deepcopy(generated)
    output, status = tmp_path / "pingpong-k3-audit-report.json", tmp_path / "evidence-status.json"
    if fault == "summary_size":
        value["unexpected_large_field"] = "x" * MAX_OUTPUT
    elif fault == "detail_size":
        value["pingpong"]["cycles"][0]["oversized"] = "x" * MAX_OUTPUT
    else:
        original = Path.open

        def broken(path, mode="r", *args, **kwargs):
            if ("-details." in path.name or fault == "secondary") and "x" in mode:
                raise OSError("injected publication failure")
            return original(path, mode, *args, **kwargs)

        monkeypatch.setattr(Path, "open", broken)
    before = qualify(value)
    with pytest.raises((DataError, OSError)):
        publish_qualified(value, {}, {}, output, status)
    assert not output.exists()
    if fault != "secondary":
        failed = json.loads(status.read_text())
        assert failed["diagnostic_integrity"] == "FAILED"
        assert failed["original_qualification"] == before["original_qualification"]
        assert failed["report_publication"]["status"] == "FAILED"
        assert "report publication" in failed["errors"][-1]


def test_missing_table_is_not_synthesized(generated, tmp_path):
    del generated["pingpong"]["cycles"]
    output = tmp_path / "pingpong-k3-audit-report.json"
    audit_details.publish(generated, output)
    assert audit_details.restore_file(json.loads(output.read_text()), tmp_path) == generated


def test_first_publication_error_survives_summary_export_and_reread(generated, tmp_path):
    from specrhythm.serving.execution_failure import summarize

    directory = tmp_path / "delivery"
    root = directory / "points" / "pingpong-k3"
    root.mkdir(parents=True)
    output = root.with_name("pingpong-k3-audit-report.json")
    status = root.with_name("pingpong-k3-evidence-status.json")
    value = {**generated, "oversized_summary": "x" * MAX_OUTPUT}
    with pytest.raises(DataError, match="8 MiB"):
        publish_qualified(value, {}, {}, output, status)
    first = summarize(root, 23, "diagnostic_evidence")
    assert first["report_publication"]["type"] == "DataError"
    assert first["report_publication"]["details"]["actual_bytes"] > MAX_OUTPUT
    assert first["diagnostic_integrity"] == "FAILED"
    write(first, directory / "first-failure.json")
    archive = tmp_path / "failure.tar.gz"
    export(directory, archive, first_code=23)
    with tarfile.open(archive) as t:
        index = json.load(t.extractfile("inventory.json"))
        retained = json.load(t.extractfile(index["logical_paths"]["first-failure.json"]))
    assert retained == first and index["first_exit_code"] == 23


def test_history_is_not_overwritten(generated, tmp_path):
    output = tmp_path / "pingpong-k3-audit-report.json"
    audit_details.publish(generated, output)
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        audit_details.publish(generated, output)
    assert output.read_bytes() == original
