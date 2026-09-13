"""Actual joint loop + run_point/stage publication + real report rejection + single export."""

import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
from test_prepost_device_contract import write
from test_prepost_runtime_kind import driven as _driven
from test_prepost_runtime_kind import hardware as _hardware
from test_prepost_runtime_kind import produced as _produced

from specrhythm.serving import fixed_results, ping_prepost_gpu_check, s2_cli
from specrhythm.serving.common import DataError, read_json
from specrhythm.serving.execution_failure import summarize_joint
from specrhythm.serving.ping_prepost import MODES
from specrhythm.serving.ping_prepost_delivery import export

hardware, produced, driven = _hardware, _produced, _driven
RUN = subprocess.run  # Hardware fixture intercepts nvidia-smi only during the parent test.


@pytest.mark.parametrize("failed_mode", ["target", *MODES])
@pytest.mark.parametrize("process_failure", [False, True])
def test_joint_real_failure_mode_run_and_exit_codes(
    failed_mode, process_failure, driven, tmp_path, monkeypatch
):
    called = []

    def prepare(source, root, mode, **kw):
        root.mkdir(parents=True)
        return root / "fixture-manifest.json"

    monkeypatch.setattr(ping_prepost_gpu_check, "prepare", prepare)

    def execute(root, gate, mode, path, directory, *, diagnostic, **kw):
        called.append(mode)
        h = driven(mode if mode != "target" else MODES[0], "correctness")
        # GPU child boundary: use its serialized real drive report and resident CPU model.
        shutil.copytree(
            h.directory,
            directory,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("result.json", "light-summary*", "point.json"),
        )
        h.runtime["point"] = diagnostic
        if mode == failed_mode and not process_failure:
            del h.runtime["capacity" if mode == "target" else "probe"]
        write(directory / "runtime.json", h.runtime)
        if mode == failed_mode and process_failure:
            result = s2_cli.failed_report(
                directory,
                mode,
                read_json(h.path),
                23,
                DataError("CPU model execution failure", returncode=23),
            )
        else:
            result = fixed_results.summarize(h.path, directory, diagnostic)
        result = fixed_results.emit_result(directory, result, diagnostic)
        if not result["valid"]:
            raise DataError(
                result["errors"][0],
                returncode=23 if process_failure else 1,
                directory=str(directory),
                mode=mode,
            )
        return result

    monkeypatch.setattr(s2_cli, "execute", execute)
    delivery = tmp_path / "delivery"
    joint = delivery / "joint"
    with pytest.raises(DataError):
        ping_prepost_gpu_check.run(tmp_path / "source", joint)
    failure = read_json(joint / "failure.json")
    code = 23 if process_failure else 1
    assert called == ["target", *MODES][: ["target", *MODES].index(failed_mode) + 1]
    assert failure["mode"] == failed_mode
    run = failure["run_directory"]
    assert "/joint/" + failed_mode + "/runs/joint-correctness-" in run
    assert failure["effective_exit_code"] == (23 if process_failure else 0)
    assert failure["command_exit_code"] == failure["original_returncode"] == code
    if not process_failure:
        assert failure["failure_layer"] == "report_qualification"
        assert failure["primary_error"]["field"] == (
            "'capacity'" if failed_mode == "target" else "probe"
        )
        assert failure["cleanup_status"] == "PASS"
    first = summarize_joint(joint, code)
    assert first["outer_stage"] == "joint_gpu_correctness"
    assert first["point"] == failed_mode and first["run_directory"] == run
    assert first["first_exit_code"] == code and first["primary_error"] == failure["primary_error"]
    # Execute the runner's exact failure heredoc with deliberately stale shell context.
    body = re.search(
        r"<<'PY_FAILURE'\n(.*?)\nPY_FAILURE",
        Path("scripts/run_ping_prepost_b16.sh").read_text(),
        re.S,
    ).group(1)
    printed = RUN(
        [sys.executable, "-", str(code), "joint_gpu_correctness", "stale-capacity"],
        input=body,
        text=True,
        capture_output=True,
        timeout=20,
        env={
            **os.environ,
            "PYTHONPATH": str(Path("src").resolve()),
            "SR_PING_DELIVERY": str(delivery),
            "SR_FIXED_ROOT": "/stale/capacity",
        },
    )
    assert printed.returncode == 0, printed.stderr
    first = json.loads(printed.stdout)
    assert first["point"] == failed_mode and first["run_directory"] == run
    assert first["effective_exit_code"] == (23 if process_failure else 0)
    assert first["evidence_packages"] == [str(delivery) + ".tar.gz"]
    archive = tmp_path / "delivery.tar.gz"
    inv = export(delivery, archive, first_code=code, stage="joint_gpu_correctness")
    assert inv["export_status"] == "COMPLETE" and inv["export_validation_exit_code"] == 0
    with tarfile.open(archive) as t:
        paths = json.load(t.extractfile("inventory.json"))["logical_paths"]
        assert json.load(t.extractfile(paths["joint/failure.json"])) == failure
        assert json.load(t.extractfile(paths["first-failure.json"])) == first
        assert "joint/" + failed_mode + "/stage.json" in paths
    # A later bounded-export rejection cannot replace the first command/process codes.
    monkeypatch.setattr("specrhythm.serving.ping_prepost_delivery.FILE_LIMIT", 1)
    bad = export(
        delivery, tmp_path / "limited.tar.gz", first_code=code, stage="joint_gpu_correctness"
    )
    assert bad["export_validation_exit_code"] == 41 and bad["first_exit_code"] == code
    assert read_json(delivery / "first-failure.json") == first
