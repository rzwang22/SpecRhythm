"""Real run -> capacity_for -> drive -> summarize/device qualifier -> archive replay."""

import copy
import json
import tarfile

import pytest
from test_prepost_runtime_kind import driven as _driven
from test_prepost_runtime_kind import hardware as _hardware
from test_prepost_runtime_kind import produced as _produced

from specrhythm.serving import decode_scan_results, fixed_results
from specrhythm.serving.common import read_json
from specrhythm.serving.device_contract import qualify_prepost
from specrhythm.serving.k3 import MODES
from specrhythm.serving.k3_capacity import qualify
from specrhythm.serving.ping_prepost_delivery import export

hardware, produced, driven = _hardware, _produced, _driven


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("stage", ["capacity_probe", "correctness", "performance"])
def test_real_run_capacity_report_and_single_archive(mode, stage, driven, tmp_path):
    h = driven(mode, stage, full_run=True)
    capacity = qualify(h.actual, mode)
    assert capacity["status"] == "PASS"
    summarize = (fixed_results.summarize if stage == "correctness"
                 else decode_scan_results.summarize)
    result = summarize(h.path, h.directory, h.point, probe=h.probe)
    assert result["capacity_reservation_qualification"] == capacity
    assert result["valid"], result.get("primary_error")
    contract = qualify_prepost(h.runtime, h.backend, h.actual, mode, probe=h.probe, stage=stage)
    archive = tmp_path / "capacity-chain.tar.gz"
    export(h.directory, archive, first_code=1, modes=MODES)
    with tarfile.open(archive) as t:
        paths = json.load(t.extractfile("inventory.json"))["logical_paths"]
        actual, runtime, backend = [json.load(t.extractfile(paths[n])) for n in (
            "actual-capacity.json", "runtime.json", "draft-backend-report.json")]
    assert qualify(actual, mode) == capacity
    assert qualify_prepost(runtime, backend, actual, mode, probe=h.probe, stage=stage) == contract
    assert read_json(h.directory / "actual-capacity.json") == h.actual  # exporter stays read-only


@pytest.mark.parametrize("fault", ["missing", "bool", "float", "negative", "block", "budget",
                                   "role", "duplicate", "extra", "metadata"])
def test_capacity_report_rejects_missing_or_inconsistent_raw_evidence(fault, driven):
    h = driven(MODES[-1], "capacity_probe", full_run=True)
    a = copy.deepcopy(h.actual)
    if fault == "missing":
        del a["checks"][0]["reserved_speculative_positions"]
    elif fault in ("bool", "float", "negative"):
        a["checks"][0]["candidate_length"] = {"bool": True, "float": 3.0, "negative": -3}[fault]
    elif fault == "block":
        a["checks"][0]["required_blocks"] -= 1
    elif fault == "budget":
        del a["capacity_request_budgets"][0]["maximum_new_tokens"]
    elif fault == "role":
        a["ranks"][0]["role"] = "draft"
    elif fault == "duplicate":
        a["ranks"][1] = a["ranks"][0]
    elif fault == "extra":
        a["checks"][0]["extra_speculative_tokens"] = -1
    else:
        a["metadata"]["speculative_reservations"]["target"]["reserved_speculative_positions"] = 3
    with pytest.raises(ValueError):
        qualify_prepost(h.runtime, h.backend, a, MODES[-1], probe=True, stage="capacity_probe")
