"""Real archive limits/hash replay and read-only local/durable reexport; no GPU claims."""

import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from specrhythm.serving import delivery_budget, ordinary_reexport, ping_prepost_delivery
from specrhythm.serving.dual_batch_run import CPU_CASES, CPU_ORDER
from specrhythm.serving.k3 import B128
from specrhythm.serving.k3_capacity import preflight
from specrhythm.serving.k3_local_run import validate_archive
from specrhythm.serving.k3_validation import plan

EXECUTION = "944328263b02a915f397bcb03c8c743c71e9a56c"


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def source(tmp_path):
    """13-run archive layout, independent of performance/output qualification."""
    root = tmp_path / "original" / "delivery"
    cases = {k: dict(mode=v[0], target_dispatch=v[1], target_cpu=v[2])
             for k, v in CPU_CASES.items()}
    put(root, "dual-batch-plan.json", dict(cpu_comparison=True, cases=cases,
        order=list(CPU_ORDER), smoke_cases=[*cases, "S0"], k3_configuration=B128,
        execution_commit=EXECUTION, validation_profile="performance-exploration"))
    put(root, "validation-plan.json", plan("performance-exploration", B128))
    put(root, "k3-capacity-contract.json", preflight(B128))
    put(root, "comparison.json", dict(valid=True, scope="archive fixture only; no GPU result"))
    put(root, "smoke-comparison.json", dict(scope="archive fixture only"))
    put(root, "runner-outcome.json", dict(first_exit_code=0, stage="complete"))
    roots = [f"capacity/{c}" for c in cases] + [f"smoke/{c}" for c in [*cases, "S0"]]
    for i, c in enumerate(CPU_ORDER):
        case = f"windows/{i}-{c}"
        mode = cases[c]["mode"]
        roots.append(f"{case}/points/{mode}")
        for name in ("comparison.json", f"points/{mode}-audit-report.json",
                     f"points/{mode}-evidence-status.json"):
            put(root, f"{case}/{name}", dict(fixture=f"{case}/{name}"))
    for run in roots:
        for name in ("point.json", "runtime.json", "draft-backend-report.json",
                     "light-summary.json", "process-lifecycle.json"):
            put(root, f"{run}/runs/fixture/{name}", dict(provenance=run, target_steps=[run]))
    # Same 545 candidate count as the observed failure; unique payloads also >512.
    count = len(list(root.rglob("*.json")))
    for i in range(545 - count):
        put(root, f"commands/{i:04}/command.json", dict(command=i))
    return root


def snapshots(root):
    return {str(p.relative_to(root)): (p.stat().st_mtime_ns, p.read_bytes())
            for p in root.rglob("*") if p.is_file()}


def inspect(path):
    with tarfile.open(path) as t:
        inv = json.load(t.extractfile("inventory.json"))
        for r in inv["inventory"]:
            if r["status"] == "INCLUDED":
                data = t.extractfile(r["object"]).read()
                assert (len(data), hashlib.sha256(data).hexdigest()) == (r["bytes"], r["sha256"])
        return inv


def test_real_545_file_export_old_failure_new_complete(source, tmp_path, monkeypatch):
    before = snapshots(source)
    export = ping_prepost_delivery.export
    with monkeypatch.context() as m:
        m.setattr(delivery_budget, "count_budget", lambda *a, **kw:
                  dict(logical_files=512, unique_files=512, profile="old-limit"))
        old = export(source, tmp_path / "old.tar.gz", modes=())
    assert old["export_validation_exit_code"] == 41
    assert sum(r["status"] == "OMITTED_LIMIT" for r in old["inventory"]) == 33
    old_digest = hashlib.sha256((tmp_path / "old.tar.gz").read_bytes()).hexdigest()
    current = export(source, tmp_path / "new.tar.gz", modes=(), source_read_only=True)
    assert current["export_validation_exit_code"] == 0
    assert len(current["logical_paths"]) == 545
    assert current["observed_counts"]["unique_files"] > 512
    assert current["limits"]["logical_files"] == current["limits"]["unique_files"] == 896
    assert current["limits"]["planned_runs"] == 13
    assert current["limits"]["file_bytes"] == 2 * 1024**3
    assert current["limits"]["unique_payload_bytes"] == 4 * 1024**3
    assert inspect(tmp_path / "new.tar.gz")["missing"] == 0
    assert snapshots(source) == before
    assert hashlib.sha256((tmp_path / "old.tar.gz").read_bytes()).hexdigest() == old_digest


@pytest.mark.parametrize("fault", ["missing", "malformed", "count", "file", "total"])
def test_real_required_missing_corrupt_and_bounded_overflow_still_fail(
    source, tmp_path, monkeypatch, fault
):
    last = source / "windows/5-P0/points/pingpong-k3/runs/fixture/runtime.json"
    if fault == "missing":
        last.unlink()
    elif fault == "malformed":
        last.write_bytes(b'{"interrupted":')
    elif fault == "count":
        for i in range(352):
            put(source, f"extra/{i:04}/command.json", dict(extra=i))
    else:
        monkeypatch.setattr(ping_prepost_delivery, "byte_limits", lambda p:
                            (1, 4 * 1024**3) if fault == "file" else (1024**3, 1))
    result = ping_prepost_delivery.export(source, tmp_path / "failed.tar.gz", modes=())
    assert result["export_validation_exit_code"] == 41
    assert result["evidence_integrity"] == "INCOMPLETE"
    inv = inspect(tmp_path / "failed.tar.gz")
    if fault == "malformed":
        row = next(r for r in inv["inventory"] if r["path"] == str(last.relative_to(source)))
        assert row["raw_bytes_retained"] and row["status"] == "INCLUDED"
    elif fault == "count":
        assert inv["observed_counts"]["candidate_logical_files"] == 897
        omitted = [r for r in inv["inventory"] if r["status"] == "OMITTED_LIMIT"]
        assert len(omitted) == 1 and omitted[0]["limit"]["kind"] == "logical_files"


@pytest.mark.parametrize("field,value", [
    ("cpu_comparison", "true"), ("cases", {}), ("order", ["P0"] * 99),
    ("smoke_cases", ["P0"]), ("k3_configuration", "k3-b16-v1"),
])
def test_invalid_plan_does_not_grant_expansion(source, tmp_path, field, value):
    path = source / "dual-batch-plan.json"
    raw = json.loads(path.read_text())
    raw[field] = value
    path.write_text(json.dumps(raw))
    result = ping_prepost_delivery.export(source, tmp_path / "invalid.tar.gz", modes=())
    assert result["limits"]["logical_files"] == 512
    assert result["export_validation_exit_code"] == 41
    assert any("declaration" in e for e in result["evidence_errors"])


def test_legacy_count_budget_unchanged(tmp_path):
    assert delivery_budget.count_budget(tmp_path)["logical_files"] == 512
    put(tmp_path, "dual-batch-plan.json", dict(cpu_comparison=False))
    assert delivery_budget.count_budget(tmp_path)["unique_files"] == 512


def test_nonobject_plan_retained_as_failed_evidence(source, tmp_path):
    (source / "dual-batch-plan.json").write_text("null")
    result = ping_prepost_delivery.export(source, tmp_path / "null-plan.tar.gz", modes=())
    assert result["export_validation_exit_code"] == 41
    assert result["limits"]["logical_files"] == 512
    assert "dual-batch-plan.json" in result["logical_paths"]


@pytest.mark.parametrize("fault", ["none", "copy", "digest", "missing", "first_then_copy"])
def test_reexport_actual_archive_copy_fallback_is_read_only(
    source, tmp_path, monkeypatch, capsys, fault
):
    from specrhythm.serving import k3_local_run

    put(source.parent, "delivery-status.json", dict(final_exit_code=41,
        execution_first_exit_code=0, export_exit_code=41, status="DELIVERED"))
    (source.parent / "old.tar.gz").write_bytes(b"old failure archive must never change")
    if fault == "missing":
        (source / "windows/5-P0/points/pingpong-k3/runs/fixture/runtime.json").unlink()
    if fault == "first_then_copy":
        put(source, "runner-outcome.json", dict(first_exit_code=23, stage="performance"))
    if fault in ("copy", "first_then_copy"):
        def fail_copy(a, b, size):
            b.write(a.read(10))
            raise OSError("injected durable copy failure")
        monkeypatch.setattr(k3_local_run.shutil, "copyfileobj", fail_copy)
    if fault == "digest":
        original = k3_local_run.sha256_file
        monkeypatch.setattr(k3_local_run, "sha256_file", lambda p:
                            "0" * 64 if str(p).endswith(".copying") else original(p))
    before = snapshots(source.parent)
    code = ordinary_reexport.run(source, expected_execution=EXECUTION, exporter_commit="b" * 40,
        local_base=tmp_path / "local", persistent=tmp_path / "durable", tag="reexport",
        minimum_free_bytes=1)
    assert code == {"none": 0, "copy": 43, "digest": 43,
                    "missing": 41, "first_then_copy": 23}[fault]
    text = capsys.readouterr().out
    assert text.count("UPLOAD ONLY:") == 1
    upload = Path(text.split("UPLOAD ONLY: ")[1].strip())
    assert validate_archive(upload)["archive_integrity"] == "VERIFIED"
    inv = inspect(upload)
    assert inv["reexport_provenance"]["source_execution_commit"] == EXECUTION
    assert inv["reexport_provenance"]["previous_delivery"]["value"]["final_exit_code"] == 41
    if fault in ("copy", "digest", "first_then_copy"):
        assert upload.parent == tmp_path / "local/reexport"
        assert inv["reexport_provenance"]["delivery_failure"]["delivery_exit_code"] == 43
    else:
        assert upload.parent == tmp_path / "durable"
    assert snapshots(source.parent) == before


def test_readonly_missing_comparison_is_not_created(source, tmp_path):
    (source / "comparison.json").unlink()
    result = ping_prepost_delivery.export(source, tmp_path / "no-comparison.tar.gz",
                                          modes=(), source_read_only=True)
    assert result["export_validation_exit_code"] == 41
    assert not (source / "comparison.json").exists()


@pytest.mark.parametrize("first", [0, 23])
def test_archive_write_failure_no_invented_upload_or_lost_first_error(
    source, tmp_path, monkeypatch, capsys, first
):
    put(source, "runner-outcome.json", dict(first_exit_code=first, stage="complete"))
    before = snapshots(source)

    def broken_add(*args, **kwargs):
        raise OSError("injected archive write failure")

    monkeypatch.setattr(tarfile.TarFile, "addfile", broken_add)
    code = ordinary_reexport.run(source, expected_execution=EXECUTION, exporter_commit="b" * 40,
        local_base=tmp_path / "local", persistent=tmp_path / "durable", tag="failed",
        minimum_free_bytes=1)
    text = capsys.readouterr().out
    assert code == (first or 41)
    assert "UPLOAD ONLY:" not in text and "NO ARCHIVE:" in text
    assert snapshots(source) == before


def test_wrong_execution_and_source_overlap_rejected_without_source_changes(source, tmp_path):
    before = snapshots(source)
    options = dict(expected_execution=EXECUTION, exporter_commit="b" * 40,
        local_base=tmp_path / "local", persistent=tmp_path / "durable", tag="new",
        minimum_free_bytes=1)
    with pytest.raises(ValueError, match="execution SHA"):
        ordinary_reexport.run(source, **{**options, "expected_execution": "c" * 40})
    with pytest.raises(ValueError, match="separate"):
        ordinary_reexport.run(source, **{**options, "local_base": source})
    assert snapshots(source) == before


def test_actual_cli_reads_original_outcome_without_launching_models(
    source, tmp_path, monkeypatch, capsys
):
    import sys

    monkeypatch.setenv("SR_K3_LOCAL_BASE", str(tmp_path / "cli-local"))
    monkeypatch.setenv("SR_PING_RESULTS", str(tmp_path / "cli-durable"))
    monkeypatch.setattr(sys, "argv", ["reexport", "--source", str(source),
        "--expected-execution", EXECUTION, "--exporter-commit", "b" * 40, "--tag", "cli"])
    calls = []

    def checkout(args, **kw):
        calls.append(args)
        assert args[0] == "git" and args[1] in ("rev-parse", "status")
        return "b" * 40 if args[1] == "rev-parse" else b""

    monkeypatch.setattr(ordinary_reexport.subprocess, "check_output", checkout)
    with pytest.raises(SystemExit) as stop:
        ordinary_reexport.main()
    assert stop.value.code == 0 and len(calls) == 2
    assert capsys.readouterr().out.count("UPLOAD ONLY:") == 1
