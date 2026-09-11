"""Purpose costs, projection rounding proof and bounded bad-row reporting; no GPU."""

import copy

import pytest
from test_serving_fixed_attribution import artifacts, device, forward

from specrhythm.serving.fixed_attribution import analyze_raw, clock_failures, gpu_bounds
from specrhythm.serving.fixed_timing import project_anchor, recorded_costs


def test_real_projection_loses_integer_bits_but_new_bracket_is_exact():
    anchor, uncertainty, a, b = 17_357_310_142_137_541, 1, 1.0, 10_001.25
    old_width_error = round(anchor + uncertainty + a) - round(anchor + a) - uncertainty
    assert old_width_error == 3  # Exact reproducible double rounding; legacy tolerance is 2.
    row = project_anchor(anchor, anchor + uncertainty, a, b)
    assert row["start_upper_ns"] - row["start_lower_ns"] == uncertainty
    assert row["end_upper_ns"] - row["end_lower_ns"] == uncertainty
    assert abs(row["end_lower_ns"] - row["start_lower_ns"] - (b - a)) <= 1
    row["gpu_event_ms"] = (b - a) / 1e6
    d = device(0, "cpu-identity", [row])
    d.update(
        projection_version="integer-anchor-v2",
        anchor_before_ns=anchor,
        anchor_after_ns=anchor + uncertainty,
    )
    assert gpu_bounds(d, anchor, anchor + 100_000)
    broken = copy.deepcopy(d)
    broken["forwards"][0]["start_upper_ns"] += 1
    with pytest.raises(ValueError, match="anchor uncertainty"):
        gpu_bounds(broken, anchor, anchor + 100_000)


@pytest.mark.parametrize("a,b", [(0.5, 10.5), (1.5, 20.25), (2**32 + 0.75, 2**32 + 100.25)])
def test_integer_projection_rounding_bound_and_no_arbitrary_tolerance(a, b):
    row = project_anchor(2**54 + 7, 2**54 + 18, a, b)
    assert row["start_upper_ns"] - row["start_lower_ns"] == 11
    assert abs(row["end_lower_ns"] - row["start_lower_ns"] - (b - a)) <= 1
    with pytest.raises(ValueError):
        project_anchor(0, 1, b, a)


def test_purpose_separation_boundary_clipping_and_tp_max_not_sum():
    start, end = 100_000_000, 200_000_000
    d = device(
        0,
        "d",
        [
            forward(["x"], 95, 110, "commit"),
            forward(["x"], 130, 140, "proposal"),
            forward(["x"], 170, 175, "commit"),
            forward(["x"], 195, 210, "prefill"),
        ],
    )
    targets = [device(rank, str(rank), [forward(["x"], 150, 160)]) for rank in (0, 1)]
    r = recorded_costs(targets, d, [{"gpu_event_ms": 10}], start, end, 1)
    assert r["window_launch_selected_sum_ms"] == {
        "D_proposal": 10,
        "D_commit_or_prefix_sync": 5,
        "V_target": 10,
        "other_recorded_Draft_GPU": 15,
    }
    # Out-of-window launches still contribute clipped physical intervals, never whole duration.
    assert r["clipped_all_recorded_gpu_union_lower_ms"] == pytest.approx(40, abs=0.00001)
    assert r["target_forward_sum_ms_by_rank"] == {"0": 10, "1": 10}
    assert not r["all_gpu_work_covered"]
    assert (
        recorded_costs(targets, d, [], start, end, 0)["approximate_ms_per_64_rotation"][
            "D_proposal"
        ]
        is None
    )


def test_real_join_costs_and_legacy_corrupt_clock_remains_unknown_with_small_failure_rows():
    runtime, backend = artifacts()
    good = analyze_raw(runtime, backend)
    assert good["recorded_gpu_costs"]["complete_rotation_count"] == 2
    assert good["clock_failure_details"]["failure_count"] == 0
    bad = runtime["target_devices"][0]["device"]["forwards"][0]
    bad["start_upper_ns"] += 5
    raw = analyze_raw(runtime, backend)
    assert raw["overlap"]["status"] == "UNKNOWN"
    details = raw["clock_failure_details"]
    assert details["failure_count"] == 1
    assert details["rows"][0]["start_width_minus_uncertainty_ns"] == 5
    d = runtime["target_devices"][0]["device"]
    d["forwards"] = [copy.deepcopy(bad) for _ in range(10_000)]
    r = clock_failures(runtime, backend, 0, 10**12)
    assert r["failure_count"] == 10_000 and len(r["rows"]) == 16 and r["truncated"]
