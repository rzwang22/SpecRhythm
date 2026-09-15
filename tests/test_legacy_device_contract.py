"""Legacy Dual producer retains live query semantics, including zero-access probes."""

import copy

import pytest
from test_serving_fixed_startup import hardware as _hardware
from test_serving_fixed_startup import observed_startup as _observed
from test_serving_fixed_startup import phase4_config as _config
from test_serving_fixed_startup import startup as _startup

from specrhythm.serving.common import DataError
from specrhythm.serving.device_contract import legacy_dual

hardware, observed_startup, phase4_config, startup = _hardware, _observed, _config, _startup


@pytest.mark.parametrize("probe", [True, False])
def test_actual_legacy_startup_query_counts_and_strict_validator(observed_startup, probe):
    h = observed_startup
    runtime = h.run("pingpong", probe=probe)
    runtime["target_steps"] = [] if probe else [{"rows": [{"candidate_positions": 1}]}]
    ranks = legacy_dual(runtime, "pingpong", "capacity_probe" if probe else "performance")
    assert all(r["uuid_verification_access_count"] == int(not probe) for r in ranks)
    for fault in ("missing", "queries", "cache", "initial"):
        broken = copy.deepcopy(runtime)
        r = broken["target_final_memory"][1]
        if fault == "missing":
            del r["dual_uuid_query"]
        else:
            key = {
                "queries": "uuid_verification_subprocess_query_count",
                "cache": "uuid_cache_hit_count",
                "initial": "uuid_initial_validation_count",
            }[fault]
            r["dual_uuid_query"][key] += 1
        with pytest.raises(DataError, match="mode=pingpong.*rank=1.*dual_uuid_query"):
            legacy_dual(broken, "pingpong", "capacity_probe" if probe else "performance")
