"""S2 CPU contracts; none of these tests qualify real GPU residency or performance."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import threading
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from test_phase4_dual_scheduler import DraftClientStub, StockSchedulerStub, proposal_result
from test_phase4_serial import phase4_config as _config_fixture
from test_serving_s1 import execution as s1_execution
from test_serving_s1 import freeze_serial_inputs, request

from specrhythm.phase4.decode_ready import DecodeReadyProvenance
from specrhythm.phase4.manifest import sha256_file
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.serving.common import DataError, read_json
from specrhythm.serving.runtime_profile import PROFILE_ENV, load_s2
from specrhythm.serving.s1_workload import write_once
from specrhythm.serving.s2_clock import ServingClock
from specrhythm.serving.s2_plan import (
    MODES,
    RATIO,
    capacity_for,
    freeze_sizes,
    poisson_trace,
    sealed,
    selection_order,
    slo_policy,
)
from specrhythm.serving.s2_pool import ResidentPoolAudit, prefix_record, publish
from specrhythm.serving.s2_results import observation_metrics, request_metrics, validate_population

phase4_config = _config_fixture


def profile(directory, rows=None, execution=None):
    rows = rows or [request(i, 5) for i in range(4)]
    directory.mkdir(parents=True, exist_ok=True)
    work = directory / "requests.jsonl"
    work.write_text("".join(json.dumps(r.to_dict()) + "\n" for r in rows))
    ids = [r.request_id for r in rows]
    value = sealed(
        {
            "schema_version": "specrhythm.s2-execution.v1",
            "workload_file": work.name,
            "workload_sha256": sha256_file(work),
            "request_ids": ids,
            "requested_N": len(ids),
            "active_limit": 128,
            "trace": poisson_trace(ids, 0.25),
            "execution": execution or {"eos_token_ids": [999], "git_commit": "a" * 40},
        }
    )
    path = directory / "execution-manifest.json"
    write_once(path, value)
    return path, value, rows


def main_rows():
    return [
        replace(
            request(i, 64),
            task_class=task,
            slo_class=task,
            prompt_length=512,
            prompt_token_ids=list(range(512)),
        )
        for i, task in enumerate(t for t, q in RATIO.items() for _ in range(q * 100))
    ]


def preferred(main):
    return [
        r.request_id
        for task, q in RATIO.items()
        for r in [r for r in main if r.task_class == task][: q * 10]
    ]


def ranks(blocks=100000):
    return [
        dict(
            mode=m,
            role=role,
            physical_gpu_id=gpu,
            gpu_uuid=f"GPU-{gpu}",
            block_size=16,
            num_gpu_blocks=blocks,
            free_memory_bytes=8 * 1024**3,
            vocab_size=151936,
        )
        for m in MODES
        for role, gpu in (("draft", 0), ("target", 1), ("target", 2))
    ]


def test_nested_stratification_preserves_original_S1_ids_and_each_ten():
    main = main_rows()
    ids = preferred(main)
    selected = selection_order(main, ids)
    assert set(selected["request_ids"][:100]) == set(ids)
    by_id = {r.request_id: r for r in main}
    for n in range(10, 501, 10):
        assert Counter(by_id[r].task_class for r in selected["request_ids"][:n]) == {
            t: q * n // 10 for t, q in RATIO.items()
        }
    assert selected == selection_order(main, ids)
    assert selection_order(main, ids, 234)["request_ids"][:100] == selected["request_ids"][:100]
    assert selection_order(main, ids, 234)["request_ids"][100:] != selected["request_ids"][100:]


def test_capacity_shrinks_common_nested_subset_on_draft_rank_only():
    main = main_rows()
    selected = selection_order(main, preferred(main))
    limits = ranks()
    limits[0]["num_gpu_blocks"] = 2400
    plan = freeze_sizes(main, selected, limits)
    small, large = (plan["sizes"][k] for k in ("small", "large"))
    assert 0 < small["actual_N"] < 100 and small["actual_N"] % 10 == 0
    assert large["actual_N"] == small["actual_N"] and not large["distinct_scale"]
    assert large["request_ids"] == small["request_ids"]
    assert small["shrink_reason"] and plan["all_modes_share_actual_N"]
    assert all(r["valid"] for r in small["capacity_attempts"][-1]["ranks"])


@pytest.mark.parametrize("role", ["draft", "target"])
def test_capacity_covers_full_caps_worst_128_partial_and_no_snapshot(role):
    rows = main_rows()[:500]
    rank = next(r for r in ranks() if r["role"] == role)
    result = capacity_for(rows, rank)
    assert result["pool_size"] == 500 and result["active_limit"] == 128
    initial = (512 + (role == "draft") + 15) // 16
    growth = (512 + 64 + 4 + 15) // 16 - initial
    assert result["prefix_blocks"] == initial * 500
    assert result["worst_active_growth_blocks"] == growth * 128
    assert result["partial_block_copy_margin"] == 128 and result["snapshot_copy_blocks"] == 0
    assert result["early_EOS_assumed"] is False
    assert not capacity_for(rows, {**rank, "free_memory_bytes": 0})["valid"]


def test_capacity_requires_all_modes_and_ranks():
    main = main_rows()
    with pytest.raises(DataError, match="all three modes"):
        freeze_sizes(main, selection_order(main, preferred(main)), ranks()[:-1])


def test_trace_independent_streams_scaling_and_mode_reuse():
    ids = [f"r{i}" for i in range(100)]
    a, b, c = (poisson_trace(ids, r, 101, 202) for r in (0.25, 0.5, 1))
    assert a["rows"][0]["arrival_offset_seconds"] == 0
    assert a["unit_samples_sha256"] == b["unit_samples_sha256"] == c["unit_samples_sha256"]
    assert a["order_sha256"] == b["order_sha256"] == c["order_sha256"]
    assert [r["arrival_offset_seconds"] for r in a["rows"]] == [
        4 * r["arrival_offset_seconds"] for r in c["rows"]
    ]
    other = poisson_trace(ids, 0.25, 999, 202)
    assert (
        other["order_sha256"] == a["order_sha256"]
        and other["unit_samples_sha256"] != a["unit_samples_sha256"]
    )
    other = poisson_trace(ids, 0.25, 101, 999)
    assert (
        other["unit_samples_sha256"] == a["unit_samples_sha256"]
        and other["order_sha256"] != a["order_sha256"]
    )
    assert len({poisson_trace(ids, 0.25, 101, 202)["sha256"] for _ in MODES}) == 1
    assert any(r["arrival_offset_seconds"] % 1 for r in a["rows"][1:])


def clock_fixture(n=3, limit=1, mode="pingpong", terminal=False):
    definitions = [request(i, 3) for i in range(n)]
    trace = poisson_trace([r.request_id for r in definitions], 1, 1, 1)
    # Tests use fixed arrivals, with the same sealed shape; production uses exponential intervals.
    trace = sealed(
        {
            **{k: v for k, v in trace.items() if k != "sha256"},
            "rows": [
                {**row, "arrival_offset_seconds": i / 100} for i, row in enumerate(trace["rows"])
            ],
        }
    )
    boot = {
        r.request_id: {
            "token": 999 if terminal else 50,
            "terminal": terminal,
            "finish_reason": "stop" if terminal else None,
        }
        for r in definitions
    }
    return ServingClock(definitions, trace, boot, active_limit=limit, mode=mode)


def test_before_arrival_cannot_commit_and_finished_sync_holds_slot():
    c = clock_fixture()
    c.start(100, threaded=False)
    rid = c.trace["rows"][0]["request_id"]
    with pytest.raises(DataError, match="before arrival"):
        c.commit(rid, [51], 101)
    c.observe(30_000_100)
    assert c.admit(30_000_101) == [rid]
    c.commit(rid, [51, 999], 30_000_102, finished=True, finish_reason="stop")
    assert c.admit(30_000_103) == []
    c.released([rid], 30_000_104)
    assert len(c.admit(30_000_105, busy_cohorts=("A",))) == 1
    next_row = [r for r in c.rows.values() if r["state"] == "ACTIVE"][0]
    assert next_row["cohort"] == "B"


def test_arrival_observer_progresses_while_target_is_busy():
    c = clock_fixture(n=3)
    c.start(time.monotonic_ns())
    busy = threading.Event()
    # The Target thread remains blocked, but the independent observer queues future arrivals.
    busy.wait(0.08)
    c.close()
    assert len(c.queue) == 3
    assert all(r["state"] == "QUEUED" for r in c.rows.values())
    assert all(r["observed_arrival_ns"] >= r["arrival_ns"] for r in c.rows.values())
    assert not any(e["event"] == "commit" for e in c.events)


def test_bootstrap_terminal_waits_for_arrival_with_zero_timed_tokens():
    c = clock_fixture(n=2, terminal=True)
    c.start(100, threaded=False)
    c.observe(100)
    assert not c.complete
    c.observe(10_000_100)
    assert c.complete and not c.admit(10_000_101)
    runtime = dict(
        start_ns=100, end_ns=10_000_200, requests=list(c.rows.values()), events=c.events
    )
    rows = request_metrics(
        list(c.definitions.values()), runtime, c.trace, [999], slo_policy(dict.fromkeys(RATIO, 10))
    )
    assert all(r["timed_tokens"] == 0 and r["bootstrap_tokens"] == 1 for r in rows)
    assert all(r["request_tpot_ms"] is None and r["slo_good"] is None for r in rows)
    validate_population(runtime, list(c.definitions.values()), 1)


def completed_clock():
    c = clock_fixture(n=1, mode="serial")
    c.start(100, threaded=False)
    c.observe(1_000_100)
    rid = c.admit(10_000_100)[0]
    c.commit(rid, [51, 999], 12_000_100, finished=True, finish_reason="stop")
    c.released([rid], 12_000_101)
    runtime = dict(
        start_ns=100, end_ns=13_000_100, requests=list(c.rows.values()), events=c.events
    )
    return c, runtime


def test_queue_is_inside_slo_and_batch_commits_are_not_spread():
    c, runtime = completed_clock()
    policy = slo_policy(dict.fromkeys(RATIO, 2))
    rows = request_metrics(list(c.definitions.values()), runtime, c.trace, [999], policy)
    assert rows[0]["queue_ms"] == 10 and rows[0]["arrival_handling_lag_ms"] == 1
    assert rows[0]["queue_inclusive_decode_avg_ms_per_token"] == 6
    assert rows[0]["slo_good"] is False
    assert rows[0]["token_intervals_ms"] == [0] and rows[0]["request_tpot_ms"] == 0
    metrics = observation_metrics(rows, 100, 13_000_100)
    assert metrics["by_task"]["overall"]["token_goodput_tok_s"] == 0
    assert metrics["by_task"]["overall"]["throughput_tok_s"] == pytest.approx(2 / 0.013)


@pytest.mark.parametrize(
    "field", ["generated_token_ids", "resources_released", "completion_ns", "finish_reason"]
)
def test_internal_count_cleanup_time_and_eos_failures_remain_blocking(field):
    c, runtime = completed_clock()
    row = runtime["requests"][0]
    row[field] = {
        "generated_token_ids": [50, 51],
        "resources_released": False,
        "completion_ns": 100,
        "finish_reason": "length",
    }[field]
    with pytest.raises(DataError):
        request_metrics(list(c.definitions.values()), runtime, c.trace, [999], None)


def test_different_valid_outputs_are_independently_accepted():
    c, runtime = completed_clock()
    request_metrics(list(c.definitions.values()), runtime, c.trace, [999], None)
    runtime["requests"][0]["generated_token_ids"] = [50, 999]
    runtime["requests"][0]["commits"][0]["token_ids"] = [999]
    other = request_metrics(list(c.definitions.values()), runtime, c.trace, [999], None)
    assert other[0]["timed_tokens"] == 1


def test_pool_private_staged_identity_and_fresh_restore():
    audit = ResidentPoolAudit("draft")
    rows = {"a": prefix_record([1, 2, 3], 3, [[1]]), "b": prefix_record([4, 5, 6], 3, [[2]])}
    states = {r: {"state": "STAGED"} for r in rows}
    audit.check(rows, states, freeze=True)
    changed = copy.deepcopy(rows)
    changed["b"]["block_ids"] = [[1]]
    with pytest.raises(DataError, match="shared"):
        audit.check(changed, states)
    changed = copy.deepcopy(rows)
    changed["b"]["prefix_sha256"] = token_prefix_hash([99])
    with pytest.raises(DataError, match="changed or evicted"):
        audit.check(changed, states)
    with pytest.raises(DataError, match="changed or evicted"):
        audit.check({"a": rows["a"]}, states)
    with pytest.raises(DataError, match="continuation"):
        audit.check(rows, states, freeze=True)
    fresh = ResidentPoolAudit("draft")
    fresh.check(rows, states, freeze=True)
    assert fresh.report()["initial"] == audit.report()["initial"]


def test_profile_dispatch_preserves_S1_and_is_lossless_S2(tmp_path, monkeypatch):
    from specrhythm.serving import runtime_profile as profile_module
    from specrhythm.serving import s1_workload

    path, _, rows = profile(tmp_path / "s2")
    monkeypatch.setenv(PROFILE_ENV, str(path))
    loaded = profile_module.load_runtime_requests(path.parent / "requests.jsonl", len(rows))
    assert [r.source.to_dict() for r in loaded] == [r.to_dict() for r in rows]
    assert profile_module.target_options() == {"max_num_seqs": 512, "max_num_batched_tokens": 4096}
    monkeypatch.delenv(PROFILE_ENV)
    old, _, original = s1_execution(tmp_path / "s1")
    monkeypatch.setenv(s1_workload.PROFILE_ENV, str(old))
    assert profile_module.target_options() == s1_workload.target_options()
    assert profile_module.load_runtime_requests(
        old.parent / "requests.jsonl", 4
    ) == s1_workload.load_runtime_requests(old.parent / "requests.jsonl", 4)
    assert len(original) == 4


def test_load_rechecks_bytes_after_first_read(tmp_path):
    path, _, _ = profile(tmp_path)
    load_s2(str(path))
    (tmp_path / "requests.jsonl").write_text("tampered")
    with pytest.raises(DataError, match="workload hash"):
        load_s2(str(path))


@pytest.mark.parametrize("mode", MODES)
def test_real_startup_writes_valid_context_before_LLM(tmp_path, monkeypatch, mode, phase4_config):
    from specrhythm.serving import s2_runtime

    old, manifest, _ = s1_execution(tmp_path / "s1")
    freeze_serial_inputs(tmp_path, old, manifest)
    path, value, _ = profile(tmp_path / "s2", execution=manifest["execution"])
    directory = tmp_path / "attempt"
    directory.mkdir()
    monkeypatch.setenv(PROFILE_ENV, str(path))
    monkeypatch.setenv("SR_S2_DRAFT_SOCKET", str(tmp_path / "draft.sock"))
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setattr(s2_runtime, "validate_installed_patch_stack", lambda *a: {})

    class LLM:
        def __init__(self, **kwargs):
            context = read_json(directory / "decode-ready-context.json")
            parsed = DecodeReadyProvenance.from_dict(context)
            assert parsed.workload_sha256 == value["workload_sha256"]
            assert (
                parsed.target_tensor_parallel_size == 2 and parsed.draft_tensor_parallel_size == 1
            )
            assert context["s2_execution_binding"]["sha256"] == value["sha256"]
            assert kwargs["max_num_seqs"] == 512 and kwargs["async_scheduling"] is False
            assert kwargs["speculative_config"]["num_speculative_tokens"] == 4
            raise RuntimeError("LLM-entry reached after real context validation")

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=LLM))
    config, _, _ = s2_runtime.configure(tmp_path, path, directory, mode)
    with pytest.raises(RuntimeError, match="LLM-entry reached"):
        s2_runtime.make_engine(config, mode)


@pytest.fixture
def s2_schedulers(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    stub = ModuleType("vllm.v1.core.sched.scheduler")

    class Stock(StockSchedulerStub):
        def schedule(self):
            result = super().schedule()
            result.preempted_req_ids = set()
            return result

    stub.Scheduler = Stock
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    for name, file in (
        ("specrhythm.phase4.resident_scheduler", "resident_scheduler.py"),
        ("specrhythm.phase4.vllm_dual_scheduler", "vllm_dual_scheduler.py"),
    ):
        spec = importlib.util.spec_from_file_location(name, root / "src/specrhythm/phase4" / file)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    name = "specrhythm.serving.s2_scheduler"
    spec = importlib.util.spec_from_file_location(
        name, root / "src/specrhythm/serving/s2_scheduler.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    path = tmp_path / "control.json"
    monkeypatch.setenv("SR_S2_CONTROL", str(path))
    return module, path


def test_pingpong_real_ready_gate_allows_single_cohort_and_stages_future(
    s2_schedulers, monkeypatch, tmp_path
):
    module, path = s2_schedulers
    dual = sys.modules["specrhythm.phase4.vllm_dual_scheduler"]
    monkeypatch.setattr(
        dual,
        "load_smoke_requests",
        lambda *a, **kw: [
            SimpleNamespace(request_id="a", prompt_token_ids=(1, 2)),
            SimpleNamespace(request_id="b", prompt_token_ids=(3, 4)),
        ],
    )
    client = DraftClientStub()
    monkeypatch.setattr(dual, "DualDraftClient", lambda *a, **kw: client)
    for key, value in {
        "SR_PHASE4_DUAL_BATCH": "1",
        "SR_PHASE4_DUAL_DRAFT_SOCKET": str(tmp_path / "d.sock"),
        "SR_PHASE4_DUAL_SCHEDULER_EVENTS": str(tmp_path / "events.jsonl"),
        "SR_PHASE4_WORKLOAD": str(tmp_path / "work"),
        "SR_PHASE4_DUAL_RESIDENT": "0",
        "SR_PHASE4_REQUEST_COUNT": "2",
        "SR_PHASE4_DUAL_MICROBATCH_SIZE": "128",
        "SR_PHASE4_DUAL_TEST_COORDINATION": "none",
    }.items():
        monkeypatch.setenv(key, value)
    packet = {
        "barrier_ns": 100,
        "active_limit": 128,
        "requests": {
            "a": {"state": "ACTIVE", "cohort": "A"},
            "b": {"state": "STAGED", "cohort": None},
        },
    }
    publish(path, packet)
    instance = module.S2PingScheduler()
    instance._bind_vllm_requests()
    instance.kv_cache_manager = SimpleNamespace(
        get_block_ids=lambda i: [[1 if i == "opaque-a" else 2]]
    )
    instance.freeze_pool()
    instance._accept_ready_result(proposal_result())
    first = instance.schedule()
    assert set(first.num_scheduled_tokens) == {"opaque-a"}
    assert instance.selected_cohort == "B"
    # A genuinely new ready proposal on A; empty B must not hold it until a future arrival.
    req = instance.requests["opaque-a"]
    req.all_token_ids.extend([11, 12, 13])
    req.spec_token_ids = []
    req.num_output_tokens = 4
    req.num_computed_tokens = 5
    instance._accept_ready_result(
        proposal_result("a", prefix=tuple(req.all_token_ids), version=2, round_id=1)
    )
    assert set(instance.schedule().num_scheduled_tokens) == {"opaque-a"}
    with pytest.raises(DataError, match="preemption forbidden"):
        instance._preempt_request(req, 1)


@pytest.mark.parametrize("mode", MODES)
def test_all_modes_can_admit_128_from_a_larger_pool(mode):
    c = clock_fixture(n=130, limit=128, mode=mode)
    c.start(100, threaded=False)
    c.observe(2_000_000_100)
    admitted = c.admit(2_000_000_101)
    assert len(admitted) == 128 and len(c.queue) == 2
    assert len(c.rows) == 130
    if mode == "pingpong":
        assert Counter(c.rows[r]["cohort"] for r in admitted) == {"A": 64, "B": 64}
    c.commit(admitted[0], [51, 999], 2_000_000_102, finished=True, finish_reason="stop")
    assert c.admit(2_000_000_103) == []
    c.released(admitted[:1], 2_000_000_104)
    assert len(c.admit(2_000_000_105)) == 1


def test_frozen_gates_share_actual_N_and_SLO_across_explicit_seeds(tmp_path, monkeypatch):
    from specrhythm.serving import s2_preflight

    main = main_rows()
    selected = selection_order(main, preferred(main))
    plan = freeze_sizes(main, selected, ranks())
    policy = slo_policy(dict.fromkeys(RATIO, 100))
    write_once(tmp_path / "capacity-plan.json", plan)
    write_once(tmp_path / "slo-policy.json", policy)
    prep = {
        "execution": {"git_commit": "a" * 40, "eos_token_ids": [999]},
        "arrival_seed": 10,
        "order_seed": 20,
        "seed_pairs": [
            {"arrival_seed": 10, "order_seed": 20},
            {"arrival_seed": 11, "order_seed": 21},
        ],
    }
    monkeypatch.setattr(s2_preflight, "preparation", lambda root: (prep, selected))
    monkeypatch.setattr(s2_preflight, "source_rows", lambda *a: main)
    monkeypatch.setattr(s2_preflight, "validate_execution_files", lambda *a: None)
    value = s2_preflight.freeze(tmp_path)
    assert value["slo_sha256"] == policy["sha256"]
    assert len(value["gates"]["G2"]) == 6
    assert s2_preflight.freeze(tmp_path) == value  # safe resume of completed G0 inputs
    for gate, count in (("G1", 10), ("G2", 100), ("G3", 500)):
        sets = []
        for unit in value["gates"][gate]:
            manifest, _ = load_s2(str(tmp_path / unit["manifest"]))
            assert unit["modes"] == list(MODES)
            assert manifest["actual_N"] == count and manifest["active_limit"] == 128
            sets.append(manifest["request_set_sha256"])
        assert len(set(sets)) == 1
