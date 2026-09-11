"""Fixed64 capacity/admission, window and interval contracts; no inference frameworks."""

from collections import Counter

import pytest
from test_serving_s1 import request

from specrhythm.serving.common import DataError
from specrhythm.serving.fixed_plan import (
    MODES,
    build_manifest,
    capacity_metadata,
    settings,
    stage_points,
)
from specrhythm.serving.fixed_results import (
    intersections,
    overlap_metrics,
    population_metrics,
    positions,
    rounds_by_prefix,
    stage_shape,
    union,
)
from specrhythm.serving.fixed_runtime import Window, population
from specrhythm.serving.s1_workload import ResidentServingRequest
from specrhythm.serving.s2_clock import ServingClock
from specrhythm.serving.s2_plan import sealed


def clock_fixture(n=100, *, future=False, fixed=True):
    definitions = [ResidentServingRequest(request(i, 4)) for i in range(n)]
    ids = [r.request_id for r in definitions]
    trace = sealed(
        {
            "rows": [
                {"request_id": rid, "arrival_offset_seconds": 1.0 if future and i >= 10 else 0.0}
                for i, rid in enumerate(ids)
            ]
        }
    )
    assignment = {r: "A" if i < 32 else "B" for i, r in enumerate(ids[:64])} if fixed else {}
    clock = ServingClock(
        definitions,
        trace,
        {r: {"token": 10, "terminal": False} for r in ids},
        mode="pingpong",
        active_limit=64,
        per_cohort_capacity=32,
        fixed_assignment=assignment,
    )
    clock.start(100, threaded=False)
    clock.observe(100)
    return clock, ids


def test_fixed_initial_population_is_32_plus_32_and_total_64():
    c, ids = clock_fixture()
    assert c.admit(100) == ids[:64]
    assert Counter(c.rows[r]["cohort"] for r in ids[:64]) == {"A": 32, "B": 32}
    assert c.admit(101) == []
    assert population(c)["held_slots"] == 64


def test_busy_cohort_cannot_overfill_other_cohort():
    c, ids = clock_fixture(fixed=False)
    admitted = c.admit(100, busy_cohorts={"B"})
    assert len(admitted) == 32
    assert all(c.rows[r]["cohort"] == "A" for r in admitted)
    assert c.admit(101, busy_cohorts={"B"}) == []
    assert len(c.queue) == 68
    assert len(c.admit(102)) == 32
    assert population(c)["cohort_held"] == {"A": 32, "B": 32}


def test_terminal_inflight_work_retains_slot_until_actual_release():
    from specrhythm.serving.s2_runtime import release_finished

    c, ids = clock_fixture()
    c.admit(100)
    c.commit(ids[0], [11], 101, finished=True, finish_reason="stop")
    assert population(c)["draining"] == 1
    assert c.admit(102) == []
    release_finished(c, {ids[0]})
    assert c.admit(103) == []
    release_finished(c, set())
    assert c.admit(104, busy_cohorts={"A"}) == []
    assert c.admit(105) == [ids[64]]
    assert c.rows[ids[64]]["cohort"] == "A"
    assert len({r["request_id"] for r in c.rows.values() if r["state"] == "ACTIVE"}) == 64


def test_ready_partial_batch_never_admits_future_trace_rows():
    c, ids = clock_fixture(future=True, fixed=False)
    assert c.admit(100) == ids[:10]
    assert population(c)["held_slots"] == 10
    assert c.admit(101) == []
    c.observe(1_000_000_100)
    assert c.admit(1_000_000_100) == ids[10:64]


def test_default_s2_capacity_behavior_is_unchanged():
    c, ids = clock_fixture()
    c = ServingClock(c.definitions.values(), c.trace, c.bootstraps, mode="pingpong")
    c.start(100, threaded=False)
    c.observe(100)
    assert c.per_cohort_capacity is None and c.active_limit == 128
    assert len(c.admit(100, busy_cohorts={"B"})) == 100


def test_supply_phase_covers_inter_step_waits_and_held_terminal_time():
    c, ids = clock_fixture()
    c.admit(1_000_100)
    c.commit(ids[0], [11], 5_000_100, finished=True, finish_reason="stop")
    for rid in ids[1:64]:
        c.commit(rid, [11], 9_000_100, finished=True, finish_reason="stop")
    c.released(ids[:64], 11_000_100)
    runtime = {"start_ns": 100, "requests": list(c.rows.values()), "events": c.events}
    value = population_metrics(runtime, 100, 12_000_100)
    assert value["supply_phase_ms"] == {"fill": 1, "full-load": 4, "tail": 7}
    assert value["full_active_load_fraction"] == pytest.approx(4 / 12)
    assert value["full_held_load_fraction"] == pytest.approx(10 / 12)
    # No Target-step table exists: gaps and real terminal holding are still charged.
    released = next(e for e in c.events if e["event"] == "resources-released")
    released["timestamp_ns"] = 4_000_100
    with pytest.raises(DataError, match="slot lifetime"):
        population_metrics(runtime, 100, 12_000_100)


def test_capacity_metadata_keeps_pool_active_forward_query_separate():
    for mode in MODES:
        m = capacity_metadata(mode)
        assert (m["resident_request_requirement"], m["active_request_limit"]) == (100, 64)
        assert m["resident_request_count"] is None  # No prefill has happened yet.
        assert capacity_metadata(mode, 98)["resident_request_count"] == 98
        assert m["target_sequence_limit"] == 128 and m["target_query_token_limit"] == 4096
        assert m["max_requests_per_target_forward"] == (
            32 if mode in ("serial-split", "pingpong") else 64
        )


def test_frozen_shape_union_and_stage_samples_are_not_continuation():
    ids = [str(i) for i in range(100)]
    manifest = build_manifest({}, ids, "sha", settings())
    d = manifest["fixed_diagnostic"]
    assert d["cohorts"]["A"] + d["cohorts"]["B"] == d["initial_request_ids"] == ids[:64]
    assert all(r["arrival_offset_seconds"] == 0 for r in manifest["trace"]["rows"])
    samples = list(stage_points(2, 1))
    assert len(samples) == 21 and sum(p["discard_warmup"] for p in samples) == 7
    assert all(p["kind"] == "initial-state" for p in samples)
    assert d["cross_run_token_equality"] == "NOT_REQUIRED"


def test_window_warmup_and_sample_budget_use_actual_nonempty_steps():
    w = Window(settings(warmup_steps=2, samples=2))
    assert not w.ready(100)
    assert not w.step_completed(0, 101)
    assert w.warmup_steps == 0
    for now in (102, 103):
        w.step_completed(32, now)
    assert w.ready(104) and w.start_ns == 104
    assert not w.step_completed(32, 105)
    assert w.step_completed(32, 106) and w.reason == "sample_budget"
    assert w.samples == 2 and w.warmup_steps == 2


def test_window_time_budget_and_initial_state_one_step():
    w = Window(settings(warmup_steps=0, window_seconds=0.001))
    w.ready(100)
    assert w.step_completed(0, 1_000_100) and w.reason == "time_budget"
    w = Window(settings(warmup_steps=999), initial_state=True)
    assert w.ready(1)
    assert w.step_completed(64, 2) and w.samples == 1


@pytest.mark.parametrize("q,k,root", [(5, 4, 1), (1, 0, 1), (3, 2, 1)])
def test_query_root_counted_exactly_once(q, k, root):
    assert positions(
        dict(
            query_positions=q,
            candidate_positions=k,
            base_root_positions=root,
            position_start=20,
            position_end_exclusive=20 + q,
        )
    ) == (q, k, root)


@pytest.mark.parametrize("q,k,root", [(6, 4, 2), (4, 4, 0), (5, 5, 0), (6, 5, 1)])
def test_bad_position_accounting_still_fails(q, k, root):
    with pytest.raises(DataError):
        positions(
            dict(
                query_positions=q,
                candidate_positions=k,
                base_root_positions=root,
                position_start=20,
                position_end_exclusive=20 + q,
            )
        )


def test_round_conservation_and_unique_prefix():
    row = dict(
        request_id="r",
        parent_prefix_len=12,
        proposal_token_ids=[1, 2, 3, 4],
        accepted_draft_token_ids=[1, 2],
        rejected_draft_token_ids=[3, 4],
        target_correction_token_ids=[7],
        target_bonus_token_ids=[],
        committed_token_ids=[1, 2, 7],
        accepted_draft_tokens=2,
        rejected_draft_tokens=2,
    )
    assert len(rounds_by_prefix([row])) == 1
    with pytest.raises(DataError, match="duplicate"):
        rounds_by_prefix([row, row])
    with pytest.raises(DataError, match="accounting"):
        rounds_by_prefix([{**row, "committed_token_ids": [1, 2, 3]}])


def test_union_not_pairwise_intersection_sum():
    assert union([(1, 8), (2, 6), (8, 10)]) == [[1, 10]]
    assert intersections([(1, 8), (2, 6)], [(3, 9), (5, 10)]) == [(3, 8)]
    assert intersections([], [(1, 3)]) == []
    assert union([]) == []


def test_device_overlap_union_retains_clock_uncertainty_and_zero():
    def f(a, b):
        return dict(start_lower_ns=a, start_upper_ns=a + 1, end_lower_ns=b - 1, end_upper_ns=b)

    targets = [{"device": {"forwards": [f(10, 30)]}}] * 2
    draft = {"forwards": [{**f(20, 40), "purpose": "proposal"}]}
    o = overlap_metrics(targets, draft, 0, 100)
    assert o["event_overlap_lower_ms"] == 8 / 1e6
    assert o["event_overlap_upper_ms"] == 10 / 1e6
    assert o["exact_kernel_overlap"] is False and o["critical_path_time_saved"] is None
    assert overlap_metrics(targets, draft, 50, 100)["physical_overlap_status"] == "ZERO"


def test_shape_mismatch_or_short_EOS_is_missing_sample_not_fabricated_time():
    m = build_manifest({}, list(map(str, range(100))), "sha", settings())
    p = dict(kind="initial-state", batch=32, half="A", mode="serial")
    shape = stage_shape(m, p, [], [])
    assert shape["status"] == "INSUFFICIENT" and shape["target_gpu_event_ms"] is None
    row = dict(
        request_ids=m["fixed_diagnostic"]["cohorts"]["A"],
        candidate_lengths=[4] * 31 + [1],
        gpu_event_ms=5,
        context_lengths=[16] * 32,
    )
    assert stage_shape(m, p, [row], [])["target_gpu_event_ms"] is None
    # vLLM may reorder rows: context matching must follow actual IDs, not row index.
    row.update(
        request_ids=list(reversed(row["request_ids"])),
        context_lengths=list(reversed(range(32))),
        candidate_lengths=[4] * 32,
    )
    shape = stage_shape(m, p, [row], [])
    assert shape["status"] == "PASS"
    assert shape["context_lengths"] == list(range(32))
