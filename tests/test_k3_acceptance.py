"""Execute the runner's exact gate on real drive -> serializer -> summarize output."""

import copy
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
from test_prepost_runtime_kind import driven as _driven
from test_prepost_runtime_kind import hardware as _hardware
from test_prepost_runtime_kind import produced as _produced

from specrhythm.serving.common import DataError, read_json
from specrhythm.serving.decode_scan_results import emit_result, summarize
from specrhythm.serving.k3 import MODES, geometry
from specrhythm.serving.k3_acceptance import full_batch_receipt, measurement, native_geometry
from specrhythm.serving.ping_prepost_delivery import export

hardware, produced, driven = _hardware, _produced, _driven
RUN = subprocess.run


def entry(root, mode, body=None):
    if body is None:
        body = re.search(r"<<'PY_MEASUREMENT'\n(.*?)\nPY_MEASUREMENT",
                         Path("scripts/run_k3_b16.sh").read_text(), re.S).group(1)
    return RUN([sys.executable, "-"], input=body, text=True, capture_output=True,
               timeout=20, env={**os.environ, "PYTHONPATH": str(Path("src").resolve()),
                               "SR_FIXED_ROOT": str(root),
                               "SR_AUDIT_SERVING_MODE": mode})


def retain(h):
    # point_reports discovers runs; retain the actual serialized producer files.
    import shutil

    result = summarize(h.path, h.directory, h.point)
    assert result["valid"], result.get("primary_error")
    result = emit_result(h.directory, result, h.point)
    root = h.directory.parent / "entry"
    directory = root / "runs" / "actual-drive"
    shutil.copytree(h.directory, directory)
    h.directory = directory
    return root, result


@pytest.mark.parametrize("mode", MODES)
def test_real_measurement_entry_accepts_each_actual_geometry(driven, mode):
    h = driven(mode, "performance", full_run=True)
    root, result = retain(h)
    assert result["actual_target_batch"]["max"] == geometry(mode)["target_request_ceiling"]
    p = entry(root, mode)
    assert p.returncode == 0, p.stderr
    assert "Target ceiling" + str(geometry(mode)["target_request_ceiling"]) in p.stdout


@pytest.mark.parametrize("mode", MODES[:2])
def test_original_fca_entry_rejects_real_serial_B16(driven, mode):
    h = driven(mode, "performance", full_run=True)
    root, result = retain(h)
    assert result["formal_comparison_eligible"] is True
    body = Path("tests/fixtures/k3_fca_measurement.py.txt").read_text()
    p = entry(root, mode, body)
    assert p.returncode == 1 and "AssertionError" in p.stderr
    assert result["actual_target_batch"]["min"] == 16


@pytest.mark.parametrize("mode", MODES)
def test_fixed_native_full_batch_and_single_archive(driven, tmp_path, mode):
    h = driven(mode, "correctness", full_run=True)
    from specrhythm.serving.fixed_results import summarize as correct_summary

    result = correct_summary(h.path, h.directory, h.point)
    assert result["valid"], result.get("primary_error")
    proof = native_geometry(h.runtime, mode, full_fixture=True)
    n = geometry(mode)["target_request_ceiling"]
    assert len(proof["first_full_batch"]["request_ids"]) == n
    assert set(proof["first_full_batch"]["ranks"]) == {"0", "1"}
    assert all(len(set(f["internal_request_ids"])) == f["B"] == n
               for f in proof["first_full_batch"]["ranks"].values())
    assert full_batch_receipt(proof, mode)
    assert not full_batch_receipt(None, mode)
    assert not full_batch_receipt(proof, MODES[(MODES.index(mode) + 1) % len(MODES)])
    bad = copy.deepcopy(proof)
    del bad["first_full_batch"]["ranks"]["1"]
    assert not full_batch_receipt(bad, mode)
    archive = tmp_path / "single.tar.gz"
    export(h.directory, archive, modes=MODES)
    with tarfile.open(archive) as t:
        paths = json.load(t.extractfile("inventory.json"))["logical_paths"]
        runtime = json.load(t.extractfile(paths["runtime.json"]))
    assert native_geometry(runtime, mode, full_fixture=True) == proof


@pytest.mark.parametrize("mode", MODES[:2])
def test_native_serial_hidden_B8_cannot_pass_full_fixture(driven, mode):
    h = driven(mode, "correctness", full_run=True, admission_limit=8)
    with pytest.raises(DataError, match="lacks one native Target forward"):
        native_geometry(h.runtime, mode, full_fixture=True)


@pytest.mark.parametrize("mode", MODES[2:])
def test_native_ping_B16_is_rejected_by_actual_entry(driven, mode):
    h = driven(mode, "performance", full_run=True)
    root, _ = retain(h)
    # Simulate wrong GPU/IPC batch with real producer hooks; keep the good summary
    # to ensure the final entry checks raw execution instead of trusting that label.
    bad = driven(mode, "correctness", full_run=True, admission_limit=16)
    path = root / "runs/actual-drive/runtime.json"
    r = read_json(path)
    for key in ("target_steps", "target_devices"):
        r[key] = bad.runtime[key]
    path.write_text(json.dumps(r))
    p = entry(root, mode)
    assert p.returncode == 1 and "actual Target batch exceeds geometry" in p.stderr


@pytest.mark.parametrize("fault", ["mode", "runtime_mode", "geometry", "geometry_type",
    "missing_geometry", "runtime_geometry", "ceiling", "batch", "batch_bool", "actual_min",
    "actual_max", "actual_type", "actual_missing", "actual_count", "rank_missing",
    "native_duplicate", "native_batch", "native_split"])
def test_actual_entry_rejects_inconsistent_reports(driven, fault):
    mode = MODES[0]
    h = driven(mode, "performance", full_run=True)
    root, _ = retain(h)
    folder = root / "runs/actual-drive"
    p, r = read_json(folder / "result.json"), read_json(folder / "runtime.json")
    if fault == "mode":
        p["mode"] = MODES[1]
    elif fault == "runtime_mode":
        r["point"]["runtime_mode"] = MODES[1]
    elif fault == "geometry":
        p["execution_geometry"] = geometry(MODES[2])
    elif fault == "geometry_type":
        p["execution_geometry"]["target_request_ceiling"] = 16.0
    elif fault == "missing_geometry":
        del p["execution_geometry"]
    elif fault == "runtime_geometry":
        r["capacity"]["execution_geometry"] = geometry(MODES[2])
    elif fault == "ceiling":
        p["sub_batch"] = 8
    elif fault in ("batch", "batch_bool"):
        p["batch"] = 8 if fault == "batch" else True
    elif fault.startswith("actual_"):
        field = fault[7:]
        if field == "missing":
            del p["actual_target_batch"]
        else:
            p["actual_target_batch"]["max" if field == "type" else field] = (
                16.0 if field == "type" else 8)
    elif fault == "rank_missing":
        r["target_devices"][1]["device"]["forwards"].clear()
    else:
        native = r["target_devices"][0]["device"]["forwards"]
        f = native[0]
        if fault == "native_duplicate":
            f["internal_request_ids"].append(f["internal_request_ids"][0])
        elif fault == "native_batch":
            f["B"] = 8
        else:
            native.insert(1, copy.deepcopy(f))
    (folder / "result.json").write_text(json.dumps(p))
    (folder / "runtime.json").write_text(json.dumps(r))
    failed = entry(root, mode)
    assert failed.returncode == 1


def test_partial_request_batch_retains_population_and_does_not_require_full_every_step(driven):
    h = driven(MODES[0], "performance", full_run=True, admission_limit=8)
    _, result = retain(h)
    proof = measurement(result, h.runtime, MODES[0])
    assert proof["full_batch_steps"] == 0
    assert proof["partial_batches"][0]["population"] == h.runtime["target_steps"][0]["population"]
    assert "unfilled_reason/deferred" in proof["partial_reason_source"]
