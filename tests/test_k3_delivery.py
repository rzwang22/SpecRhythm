"""Reuse exact production joint failure and single-archive regressions for K3."""

from functools import partial

import pytest
import test_ping_joint_failure as prior
from test_prepost_runtime_kind import driven as _driven
from test_prepost_runtime_kind import hardware as _hardware
from test_prepost_runtime_kind import produced as _produced

from specrhythm.serving.common import read_json
from specrhythm.serving.k3 import MODES, PROTOCOL
from specrhythm.serving.k3_acceptance import full_batch_receipt, native_geometry
from specrhythm.serving.k3_gpu_check import coverage

hardware, produced, driven = _hardware, _produced, _driven


@pytest.mark.parametrize("failed_mode", ["target", *MODES])
@pytest.mark.parametrize("process_failure", [False, True])
def test_k3_actual_joint_first_failure_and_export(
    failed_mode, process_failure, driven, tmp_path, monkeypatch
):
    monkeypatch.setattr(prior, "MODES", MODES)
    monkeypatch.setattr(
        prior.ping_prepost_gpu_check,
        "run",
        partial(
            prior.ping_prepost_gpu_check.run,
            modes=MODES,
            protocol=PROTOCOL,
            coverage_check=coverage,
            require_mixed=False,
            native_check=native_geometry,
        ),
    )
    prior.test_joint_real_failure_mode_run_and_exit_codes(
        failed_mode, process_failure, driven, tmp_path, monkeypatch
    )
    failure = read_json(tmp_path / "delivery/joint/failure.json")
    for receipt in failure["completed_runs"]:
        if receipt["mode"] != "target":
            assert full_batch_receipt(receipt["native_target_geometry"], receipt["mode"])
