"""Actual runtime/report/export and short fixture production; GPU bodies substituted."""

import json
import subprocess
import tarfile
from pathlib import Path

import pytest
from test_prepost_runtime_kind import driven as _driven
from test_prepost_runtime_kind import hardware as _hardware
from test_prepost_runtime_kind import produced as _produced
from test_serving_s1 import request

from specrhythm.serving import decode_scan_results, dual_batch_run
from specrhythm.serving.common import read_json
from specrhythm.serving.decode_scan_plan import manifest, options
from specrhythm.serving.k3 import B16, B128, configuration_of
from specrhythm.serving.k3_acceptance import native_geometry
from specrhythm.serving.k3_capacity import qualify
from specrhythm.serving.ping_prepost_delivery import export
from specrhythm.serving.prepost_gpu_check import prepare
from specrhythm.serving.s2_plan import check_seal

hardware, produced, driven = _hardware, _produced, _driven


@pytest.mark.parametrize("stage", ["capacity_probe", "correctness", "performance"])
def test_dual_runtime_report_export_replay(driven, monkeypatch, tmp_path, stage):
    monkeypatch.setenv("SR_K3_TARGET_DISPATCH", "dual-batch")
    monkeypatch.setenv("SR_S2_MODE", "pingpong-k3")
    h = driven("pingpong-k3", stage, full_run=True, configuration=B128,
               target_dispatch="dual-batch")
    result = (decode_scan_results.summarize if stage != "correctness" else
              __import__("specrhythm.serving.fixed_results", fromlist=["summarize"]).summarize)(
                  h.path, h.directory, h.point, probe=h.probe)
    assert result["valid"], result
    assert h.runtime["diagnostic_configuration"]["target_dispatch"] == "dual-batch"
    assert qualify(h.actual, "pingpong-k3")["status"] == "PASS"
    archive = tmp_path / "delivery.tar.gz"
    export(h.directory, archive, first_code=1, modes=("pingpong-k3",))
    with tarfile.open(archive) as t:
        paths = json.load(t.extractfile("inventory.json"))["logical_paths"]
        runtime = json.load(t.extractfile(paths["runtime.json"]))
    assert runtime["probe"] is (stage == "capacity_probe")
    assert runtime["diagnostic_configuration"] == h.runtime["diagnostic_configuration"]
    if stage == "performance":
        assert native_geometry(runtime, "pingpong-k3", configuration=B128) == native_geometry(
            h.runtime, "pingpong-k3", configuration=B128)


def test_bounded_smoke_fixture_does_not_mutate_performance_workload(tmp_path):
    source = tmp_path / "source"
    (source / "inputs").mkdir(parents=True)
    for name in ("config.json", "patch-manifest.json", "environment.json", "topology.json"):
        (source / name).write_text("{}")
    rows = [request(i, 512).to_dict() for i in range(360)]
    raw = "\n".join(json.dumps(r) for r in rows)
    (source / "inputs/requests.jsonl").write_text(raw)
    base = manifest({"capacity": {}}, [r["request_id"] for r in rows], "f" * 64,
                    options(target_dispatch="dual-batch"), 128, k3_configuration=B128,
                    validation_profile="performance-exploration")
    (source / "inputs/execution-B128.json").write_text(json.dumps(base))
    path = prepare(source, tmp_path / "smoke", "pingpong-k3", modes=("pingpong-k3",),
                   configuration=B16, source_configuration=B128, max_output_tokens=8)
    value = read_json(path)
    check_seal(value)
    assert configuration_of(value) == B16 and value["active_limit"] == 16
    assert value["request_ids"] == [r["request_id"] for r in rows[:16]]
    actual = [json.loads(r) for r in path.with_name("requests.jsonl").read_text().splitlines()]
    assert len(actual) == 16 and {r["maximum_new_tokens"] for r in actual} == {8}
    assert (source / "inputs/requests.jsonl").read_text() == raw


def test_runner_first_error_stops_smoke_and_windows_and_exports_raw(tmp_path, monkeypatch):
    directory = tmp_path / "delivery"
    calls = []
    monkeypatch.setenv("SR_FIXED_S1", "/frozen/s1")
    monkeypatch.setattr(subprocess, "check_output", lambda args, **kw:
                        "a" * 40 if args[1] == "rev-parse" else b"")

    def fail(args, **kwargs):
        calls.append(args)
        root = Path(args[args.index("--root") + 1])
        root.mkdir(parents=True)
        (root / "draft.log").write_text("original device failure, retained\n")
        raise subprocess.CalledProcessError(23, args)

    monkeypatch.setattr(subprocess, "run", fail)
    assert dual_batch_run.run(tmp_path, directory, "a" * 40) == 23
    assert len(calls) == 1
    outcome = read_json(directory / "runner-outcome.json")
    assert outcome["first_exit_code"] == 23 and outcome["stage"] == "prepare_capacity"
    archive = tmp_path / "delivery.tar.gz"
    export(directory, archive, first_code=23, stage=outcome["stage"], modes=())
    with tarfile.open(archive) as t:
        inventory = json.load(t.extractfile("inventory.json"))
        assert inventory["first_exit_code"] == 23
        paths = inventory["logical_paths"]
        assert "dual-batch-plan.json" in paths and "first-failure.json" in paths
        assert t.extractfile(paths["capacity/reference/draft.log"]).read().startswith(b"original")
    assert not (directory / "smoke").exists() and not (directory / "windows").exists()


def test_real_local_shell_preserves_first_error_and_only_one_upload(tmp_path, monkeypatch, capsys):
    import os
    import shutil
    import sys

    from specrhythm.serving import k3_local_run as local

    source = Path(__file__).resolve().parents[1]
    repo, binary = tmp_path / "repo", tmp_path / "bin"
    (repo / "scripts").mkdir(parents=True)
    binary.mkdir()
    (repo / "src").symlink_to(source / "src", target_is_directory=True)
    shutil.copyfile(source / "scripts/run_dual_batch.sh", repo / "scripts/run_dual_batch.sh")
    (binary / "git").write_text(
        '#!/bin/bash\nif [[ "$1" == rev-parse ]]; then echo ' + "a" * 40 + '; fi\n')
    (binary / "git").chmod(0o755)
    monkeypatch.setenv("PATH", str(binary) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("SR_FIXED_PYTHON", sys.executable)
    # Real prepare fails before environment/GPU access: no sealed S1 input exists.
    monkeypatch.setenv("SR_FIXED_S1", str(tmp_path / "missing-sealed-s1"))
    code = local.run(repo, "a" * 40, local_base=tmp_path / "local",
        persistent=tmp_path / "durable", tag="dual", minimum_free_bytes=1,
        configuration=B128, validation_profile="performance-exploration", dual_batch=True)
    output = capsys.readouterr().out
    assert code != 0 and output.count("UPLOAD ONLY:") == 1
    archive = Path(output.split("UPLOAD ONLY: ")[1].strip())
    assert local.validate_archive(archive)["archive_integrity"] == "VERIFIED"
    with tarfile.open(archive) as t:
        inv = json.load(t.extractfile("inventory.json"))
        paths = inv["logical_paths"]
        outcome = json.load(t.extractfile(paths["runner-outcome.json"]))
        failure = json.load(t.extractfile(paths["first-failure.json"]))
        assert outcome["first_exit_code"] == inv["first_exit_code"] == code
        assert outcome["stage"] == failure["failed_stage"] == "prepare_capacity"
        assert "missing-sealed-s1" in t.extractfile(paths["commands/0/runner.log"]).read().decode()
    root = tmp_path / "local/dual/pingpong-k3-delivery-dual"
    assert not (root / "smoke").exists() and not (root / "windows").exists()


def test_secondary_report_failure_cannot_replace_execution_code(tmp_path, monkeypatch):
    from specrhythm.serving import audit_layer_report

    monkeypatch.setenv("SR_FIXED_S1", "/missing")
    monkeypatch.setattr(subprocess, "check_output", lambda args, **kw:
                        "a" * 40 if args[1] == "rev-parse" else b"")

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(23, args)

    monkeypatch.setattr(subprocess, "run", fail)
    write = audit_layer_report.write

    def broken_comparison(value, path):
        if path.name == "comparison.json":
            raise OSError("injected report disk error")
        return write(value, path)

    monkeypatch.setattr(audit_layer_report, "write", broken_comparison)
    directory = tmp_path / "delivery"
    assert dual_batch_run.run(tmp_path, directory, "a" * 40) == 23
    outcome = read_json(directory / "runner-outcome.json")
    assert outcome["first_exit_code"] == 23
    assert "injected report disk error" in outcome["secondary_report_errors"][0]["error"]


def test_missing_runner_comparison_cannot_be_replaced_by_empty_mode_success(tmp_path):
    directory = tmp_path / "delivery"
    directory.mkdir()
    (directory / "dual-batch-plan.json").write_text('{"order": ["reference", "dual-batch"]}')
    archive = tmp_path / "delivery.tar.gz"
    export(directory, archive, first_code=41, modes=())
    with tarfile.open(archive) as t:
        inv = json.load(t.extractfile("inventory.json"))
        value = json.load(t.extractfile(inv["logical_paths"]["comparison.json"]))
        assert inv["first_exit_code"] == 41
        assert value["valid"] is False and "missing" in value["comparison_error"]
