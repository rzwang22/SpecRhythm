"""Scan selection, real fixed scheduler limits, full-window accounting and safe stop."""

import copy
from collections import Counter

import pytest
from test_serving_fixed_identity import fixed_schedulers as _fixed
from test_serving_fixed_identity import s2_schedulers as _s2
from test_serving_s2 import main_rows, preferred

from specrhythm.serving.common import DataError, digest
from specrhythm.serving.decode_scan_plan import BATCHES, MODES, manifest, options, select
from specrhythm.serving.decode_scan_results import full_load_fraction
from specrhythm.serving.decode_scan_window import ScanShapeStop, ScanWindow
from specrhythm.serving.fixed_plan import settings
from specrhythm.serving.s2_pool import publish

s2_schedulers = _s2
fixed_schedulers = _fixed


def test_pool360_uses_unique_original_rows_and_same_frozen_order_across_points():
    rows = main_rows()
    small = preferred(rows)
    chosen, selection = select(rows, small)
    again, other = select(rows, small)
    assert chosen == again and selection == other
    assert len(chosen) == len({r.request_id for r in chosen}) == 360
    assert Counter(r.task_class for r in chosen) == {
        "chat": 108,
        "code": 108,
        "reasoning": 72,
        "summarization": 72,
    }
    by_id = {r.request_id: r for r in rows}
    assert all(r is by_id[r.request_id] for r in chosen)
    changed, _ = select(rows, small, seed=1667)
    assert {r.request_id for r in changed} != {r.request_id for r in chosen}
    hashes, order = set(), set()
    for b in BATCHES:
        m = manifest(
            {}, selection["request_ids"], digest([r.to_dict() for r in chosen]), options(), b
        )
        hashes.add(m["workload_sha256"])
        order.add(m["fixed_diagnostic"]["request_order_sha256"])
        assert m["actual_N"] == 360 and m["active_limit"] == b
        for mode, cap in m["fixed_diagnostic"]["capacity"].items():
            assert cap["max_requests_per_target_forward"] == (b // 2 if mode == "pingpong" else b)
            assert cap["target_query_token_limit"] >= 5 * b
            assert cap["target_sequence_limit"] >= 360
            assert cap["draft_sequence_limit"] >= b
    assert len(hashes) == len(order) == 1
    with pytest.raises(DataError, match="main1000"):
        select(rows[:100] * 10, small)
    assert options()["samples"] is None and settings()["samples"] == 12
    assert settings()["observation"] == "original-live"
    assert settings()["identity_matching"] == "linear"


@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("mode", MODES)
def test_actual_scheduler_all360_limits_and_partial_stop_before_forward(
    fixed_schedulers, batch, mode
):
    build, path = fixed_schedulers
    s, packet = build(mode, "bound-prefix", batch=batch, n=360)
    expected = batch // 2 if mode == "pingpong" else batch
    packet["decode_scan_full_batch"] = expected
    publish(path, packet)
    out = s.schedule()
    assert len(out.num_scheduled_tokens) == expected
    assert len(s._identity().internal_to_stable) == 360
    if mode == "pingpong":
        assert all(packet["requests"][r]["cohort"] == "A" for r in out.num_scheduled_tokens)
    # Fresh independent stock scheduler. Withdraw one admissible request but keep
    # the prepared pool intact, forcing the actual selection to be partial.
    s, packet = build(mode, "bound-prefix", batch=batch, n=360)
    packet["decode_scan_full_batch"] = expected
    packet["requests"]["0"]["state"] = "QUEUED"
    publish(path, packet)
    forwarded = []
    with pytest.raises(ScanShapeStop) as caught:
        output = s.schedule()
        forwarded.append(output)  # Pinned EngineCore dispatch occurs only after schedule returns.
    assert not forwarded and not caught.value.evidence["model_forward_issued"]
    assert caught.value.evidence["scheduled_batch"] == expected - 1
    assert s.s2_steps[-1]["B"] == expected - 1


def step(batch, i, *, grouped=False):
    half = batch // 2
    cohort = "AB"[i % 2] if grouped else None
    ids = range(half) if cohort == "A" else range(half, batch) if cohort == "B" else range(batch)
    return {
        "B": half if grouped else batch,
        "cohort": cohort,
        "request_ids": [str(r) for r in ids],
        "start_ns": i * 1_000_000_000 + 1,
    }


@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("grouped", [False, True])
def test_warmup_rotations_time_primary_and_partial_last_rotation(batch, grouped):
    opts = options(window_seconds=30)
    w = ScanWindow(opts, batch, grouped)
    for i in range(4 if grouped else 2):
        assert not w.ready(i + 1, population={"active_requests": batch})
        assert not w.step_completed(step(batch, i, grouped=grouped), (i + 1) * 1_000_000_000)
    assert w.warmup_rotations == 2 and w.samples == 0
    assert not w.ready(9_000_000_000, population={"active_requests": batch - 1})
    assert w.ready(10_000_000_000, population={"active_requests": batch})
    for i in range(29):
        assert not w.step_completed(step(batch, i, grouped=grouped), (11 + i) * 1_000_000_000)
    assert w.samples == 29  # Old 12-sample limit never applies.
    assert w.time_expired(40_100_000_000)
    assert w.reason == "time_budget"
    assert len(w.evidence()["complete_rotations"]) == (14 if grouped else 29)
    assert len(w.evidence()["partial_rotations"]) == int(grouped)


def test_actual_time_overshoot_and_refill_gaps_are_not_removed():
    w = ScanWindow(options(warmup_steps=0, window_seconds=30), 16, False)
    assert w.ready(1_000_000_000, population={"active_requests": 16})
    assert not w.time_expired(30_900_000_000)  # Last atomic step may issue.
    assert w.step_completed(step(16, 0), 31_400_000_000)
    assert w.samples == 1 and w.reason == "time_budget"
    rows = [{"admission_ns": 1, "completion_ns": 11} for _ in range(16)]
    rows += [{"admission_ns": 21, "completion_ns": 31} for _ in range(16)]
    assert full_load_fraction(rows, 1, 31, 16) == pytest.approx(2 / 3)
    damaged = copy.deepcopy(rows)
    damaged.append({"admission_ns": 1, "completion_ns": 2})
    with pytest.raises(DataError, match="ceiling"):
        full_load_fraction(damaged, 1, 31, 16)
