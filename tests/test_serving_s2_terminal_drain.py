"""Real S2 Draft dispatch/materialization/release feeds the full offline qualifier."""

import copy
import json
import threading
import time
from types import SimpleNamespace

import pytest
from test_phase4_dual_batched_draft import commit_row, initial
from test_serving_s2 import clock_fixture
from test_serving_s2_results import fixture
from test_serving_s2_runtime import PoolWorker

from specrhythm.phase4.dual_batched_draft import BatchedDualDraftController
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.transport import CheckpointJsonl
from specrhythm.phase4.vllm_draft_backend import VllmBatchedDraftBackend
from specrhythm.serving import s2_draft
from specrhythm.serving.common import read_json
from specrhythm.serving.runtime_profile import load_s2
from specrhythm.serving.s2_pool import publish
from specrhythm.serving.s2_results import qualify
from specrhythm.serving.s2_runtime import release_finished


def dump(path, value):
    path.write_text(json.dumps(value))


def jsonl(path, rows):
    path.unlink(missing_ok=True)
    log = CheckpointJsonl(path)
    for row in rows:
        log.append({k: v for k, v in row.items() if k != "record_sha256"})


@pytest.fixture
def execution(tmp_path, monkeypatch, request):
    path, directory, policy = fixture(tmp_path, monkeypatch, "pingpong")
    _, definitions = load_s2(str(path))
    runtime = read_json(directory / "runtime.json")
    ids = [r.request_id for r in definitions]
    raw = {r["request_id"]: r for r in runtime["requests"]}
    # Three A/B/A resident requests suffice for the conflict; fourth preserves the
    # existing four-task native fixture. All Target input/commit validators run.
    cohorts = {rid: raw[rid]["cohort"] for rid in ids}
    tail = next(r for r in ids if cohorts[r] == "A")
    verify = next(r for r in ids if r != tail and cohorts[r] == cohorts[tail])
    variant = getattr(request, "param", "same")
    if variant == "cross":
        cohorts[verify] = "B" if cohorts[tail] == "A" else "A"
        raw[verify]["cohort"] = cohorts[verify]
        for event in runtime["events"]:
            if event.get("request_id") == verify and event["event"] == "admitted":
                event["cohort"] = cohorts[verify]
    prefixes = {
        d.request_id: tuple(d.prompt_token_ids) + (raw[d.request_id]["generated_token_ids"][0],)
        for d in definitions
    }
    nano = SimpleNamespace(value=1)

    def tick():
        nano.value += 1
        return nano.value

    monkeypatch.setattr(time, "monotonic_ns", tick)
    monkeypatch.setenv("SR_S2_CONTROL", str(directory / "s2-control.json"))
    monkeypatch.setenv("SR_S2_RUN_DIRECTORY", str(directory))
    monkeypatch.setenv("SR_S2_MODE", "pingpong")
    monkeypatch.setattr(s2_draft, "cuda_memory", lambda torch: {"free_memory_bytes": 10**9})
    state = {r: {"state": "STAGED", "cohort": cohorts[r]} for r in ids}
    publish(directory / "s2-control.json", {"barrier_ns": None, "requests": state})

    class Worker(PoolWorker):
        torch = SimpleNamespace(
            cuda=SimpleNamespace(
                synchronize=lambda: None,
                Event=lambda **kw: SimpleNamespace(
                    record=lambda: None,
                    elapsed_time=lambda other: 0.001,
                ),
            )
        )

        def materialize(self, rows, purpose):
            started = tick()
            result = super().materialize(rows, purpose)
            self.metrics.gpu_ms[purpose] += 0.001
            self.metrics.s1_forward_records.append(
                {
                    "purpose": purpose,
                    "B": len(rows),
                    "Q": sum(len(r.suffix) for r in rows),
                    "host_start_ns": started,
                    "host_launch_end_ns": tick(),
                    "cuda_completion_observed_ns": tick(),
                    "gpu_event_ms": 0.001,
                }
            )
            return result

        def request_evidence(self, rid):
            return {
                **super().request_evidence(rid),
                "draft_physical_request_block_identity": copy.deepcopy(self.kv.get_block_ids(rid)),
                "draft_physical_request_block_observable": True,
            }

    backend = s2_draft.S2DraftBackend(SimpleNamespace(max_model_len=4096), worker=Worker())
    machine = s2_draft.S2DualMachine(backend)
    for rid in ids:
        machine.initialize(initial(rid, prefixes[rid], remaining=1))
    for r in state.values():
        r["state"] = "ACTIVE"
    publish(directory / "s2-control.json", {"barrier_ns": 100, "requests": state})
    work_events = []

    def dispatch(operation, rows, at):
        nano.value = at
        started = tick()
        results = machine.execute_batch(operation, rows)
        for row, result in zip(rows, results):
            work_events.append(
                {
                    "operation": operation,
                    "request_id": row["request_id"],
                    "success": True,
                    "start_ns": started,
                    "end_ns": tick(),
                    "result": copy.deepcopy(result),
                }
            )
        return results

    proposed = dispatch(
        "propose_only",
        [
            {
                **initial(verify, prefixes[verify], remaining=2),
                "logical_cohort": cohorts[verify],
            }
        ],
        150,
    )[0]
    proposal = proposed["proposal"]
    assert raw[verify]["generated_token_ids"][-1] not in proposal["proposal_token_ids"]

    def finish(rid, at):
        return dispatch(
            "finish_tail",
            [
                {
                    "request_id": rid,
                    "logical_cohort": cohorts[rid],
                    "terminal": True,
                    "prefix_version": 2,
                    "committed_delta": raw[rid]["generated_token_ids"][1:],
                    "prefix_token_sha256": token_prefix_hash(
                        prefixes[rid] + tuple(raw[rid]["generated_token_ids"][1:])
                    ),
                }
            ],
            at,
        )

    finish(tail, 250)
    tail_work = copy.deepcopy(backend.history[-1])
    vstart, vend = tail_work["host_start_ns"] + 2, 310
    if variant == "zero":
        vstart = tail_work["host_end_ns"] + 1
    committed = raw[verify]["generated_token_ids"][1:]
    dispatch(
        "commit_and_propose",
        [
            {
                **commit_row(machine, proposal, committed, terminal=True),
                "logical_cohort": cohorts[verify],
            }
        ],
        400,
    )
    for i, rid in enumerate(r for r in ids if r not in (tail, verify)):
        finish(rid, 450 + i * 30)
    backend.shutdown()
    old_backend = read_json(directory / "draft-backend-report.json")
    old_backend.update(backend.report())
    dump(directory / "draft-backend-report.json", old_backend)
    workers = [
        {
            "global_rank": i,
            "tp_rank": i,
            "local_rank": i,
            "logical_cuda_index": 0,
            "physical_gpu_id": i + 1,
            "gpu_uuid": f"GPU-target-{i}",
        }
        for i in (0, 1)
    ]
    dump(directory / "actual-capacity.json", {"target_worker_ranks": workers})
    jsonl(directory / "draft-work-events.jsonl", work_events)

    times = {r: 350 if r != tail else 240 for r in ids}
    for r in runtime["requests"]:
        rid = r["request_id"]
        r["completion_ns"] = times[rid]
        r["commits"][0]["timestamp_ns"] = times[rid]
    for e in runtime["events"]:
        if e["event"] in ("commit", "finished"):
            e["timestamp_ns"] = times[e["request_id"]]
        if e["event"] == "resources-released":
            e["timestamp_ns"] = 600
    runtime["end_ns"] = 800
    runtime["target_final_sync"] = [{"rank": i, "timestamp_ns": 700 + i} for i in (0, 1)]
    dump(directory / "runtime.json", runtime)
    dump(
        directory / "arrival-output-events.json",
        {
            "events": runtime["events"],
            "requests": runtime["requests"],
            "failure": None,
        },
    )
    timing = [json.loads(r) for r in (directory / "timing-events.jsonl").read_text().splitlines()]
    timing = [e for e in timing if e.get("request_id") != verify]
    for e in timing:
        if "request_id" in e:
            e["timestamp_ns"] = times[e["request_id"]] - 1
    jsonl(directory / "timing-events.jsonl", timing)
    rounds = [
        {
            **proposal,
            "accepted_draft_token_ids": [],
            "rejected_draft_token_ids": proposal["proposal_token_ids"],
            "accepted_draft_tokens": 0,
            "rejected_draft_tokens": 1,
            "target_correction_token_ids": committed,
            "target_bonus_token_ids": [],
            "committed_token_ids": committed,
            "terminal": True,
            "commit_start_ns": 348,
            "commit_end_ns": 349,
            "verify_microbatch_id": "v0",
        }
    ]
    jsonl(directory / "proposal-events.jsonl", rounds)
    diagnostics = [
        json.loads(r) for r in (directory / "target-diagnostics.jsonl").read_text().splitlines()
    ]
    for d in diagnostics:
        rid = d["request_id"]
        d.update(
            target_forward_start_ns=200 if rid == tail else 320,
            target_forward_end_ns=220 if rid == tail else 340,
        )
        if rid == verify:
            prefix = list(prefixes[rid])
            d.update(
                proposal_id=proposal["proposal_id"],
                round_id=0,
                proposal_token_ids=proposal["proposal_token_ids"],
                query_length=2,
                scheduled_token_count=2,
                target_forward_start_ns=vstart,
                target_forward_end_ns=vend,
                target_input_token_ids=[prefix[-1], *proposal["proposal_token_ids"]],
                position_ids=[len(prefix) - 1, len(prefix)],
                physical_kv_num_computed_tokens=len(prefix) - 1,
                logical_committed_prefix_count=len(prefix),
                target_pending_input_token_id=prefix[-1],
                logits_position_mapping=[
                    {
                        "proposal_index": 0,
                        "proposal_token_id": proposal["proposal_token_ids"][0],
                        "target_logits_row": 0,
                        "predicts_flattened_input_position": 1,
                    }
                ],
                attention_mask_proof={
                    "causal": True,
                    "query_start_range": [0, 2],
                    "query_length_matches_scheduler": True,
                    "positions_contiguous": True,
                },
            )
    jsonl(directory / "target-diagnostics.jsonl", diagnostics)
    jsonl(
        directory / "verification-events.jsonl",
        [
            {
                **proposal,
                "verify_request_ids": [verify],
                "verify_sequence": 0,
                "verify_microbatch_id": "v0",
                "verify_host_start_ns": vstart,
                "verify_host_end_ns": vend,
                "target_physical_gpu_ids": [1, 2],
                "target_rank_intervals": [
                    {
                        **w,
                        "host_start_ns": vstart,
                        "host_end_ns": vend,
                        "cuda_events": True,
                        "cuda_synchronized": True,
                        "cuda_elapsed_ns": 1,
                    }
                    for w in workers
                ],
            }
        ],
    )
    states = []
    for rid in ids:
        flow = (
            ["BOOTSTRAP", "DRAFT_READY"]
            + (
                ["DRAFTING", "PROPOSAL_READY", "VERIFY_READY"]
                if rid == verify
                else ["TARGET_TAIL_READY"]
            )
            + ["VERIFYING", "COMMITTING", "TERMINAL"]
        )
        for i, (a, b) in enumerate(zip(flow, flow[1:])):
            final = b in ("COMMITTING", "TERMINAL")
            prefix = prefixes[rid] + (tuple(raw[rid]["generated_token_ids"][1:]) if final else ())
            states.append(
                {
                    "request_id": rid,
                    "source_state": a,
                    "destination_state": b,
                    "internal_request_id": "opaque-" + rid,
                    "reason": "CPU Target output fixture",
                    "timestamp_ns": (224 if rid == tail else 340) + i if final else 125 + i,
                    "prefix_version": 2 if final else 1,
                    "committed_prefix_length": len(prefix),
                    "committed_prefix_sha256": token_prefix_hash(prefix),
                }
            )
    jsonl(directory / "request-state-events.jsonl", states)
    jsonl(
        directory / "proposal-lifecycle-events.jsonl",
        [
            {
                **proposal,
                "lifecycle_state": s,
                "timestamp_ns": 190 + i,
            }
            for i, s in enumerate(("CREATED", "PUBLISHED", "INSTALLED", "CONSUMED"))
        ],
    )
    return SimpleNamespace(
        path=path,
        directory=directory,
        policy=policy,
        tail=tail,
        verify=verify,
        backend=old_backend,
        tail_work=tail_work,
        runtime=runtime,
        dispatch=dispatch,
    )


def test_real_terminal_dispatch_same_cohort_other_request_qualifies(execution):
    e = execution
    result = qualify(e.path, e.directory, "pingpong", e.policy)
    assert result["valid"], (result["errors"], result.get("primary_error"))
    assert result["terminal_drain"]["same_cohort_host_overlap_ms"] > 0
    assert result["terminal_drain"]["work_count"] == 3
    assert result["overlap_ms"] == result["stage_host_overlap_ms"] == 0
    assert result["draft_forward_count"] == 4  # includes all actual terminal materializations
    assert result["draft_gpu_event_ms"] == 0.004
    assert result["metrics"]["observation_makespan_ms"] == pytest.approx(700 / 1e6)
    assert result["cross_run_token_length_EOS_round_equality"] == "NOT_REQUIRED"


@pytest.mark.parametrize("execution", ["cross", "zero"], indirect=True)
def test_legal_cross_cohort_and_zero_overlap(execution, request):
    e = execution
    result = qualify(e.path, e.directory, "pingpong", e.policy)
    assert result["valid"], (result["errors"], result.get("primary_error"))
    assert result["terminal_drain"]["same_cohort_host_overlap_ms"] == 0
    if request.node.callspec.params["execution"] == "cross":
        assert result["stage_host_overlap_ms"] == result["terminal_drain"]["host_overlap_ms"] > 0
    else:
        assert result["terminal_drain"]["pair_count"] == 0


@pytest.mark.parametrize(
    "fault,expected",
    [
        ("same-request", "ownership collision"),
        ("nonterminal", "nonterminal or generates"),
        ("proposal", "nonterminal or generates"),
        ("proposal-forward", "nonterminal or generates"),
        ("active-operation", "same-cohort stages"),
        ("active-drafting", "same-cohort stages"),
        ("missing-receipt", "same-cohort stages"),
        ("changed-kv", "ownership/isolation"),
        ("same-device", "independent Draft/Target"),
        ("early-release", "slot released before"),
        ("early-drain", "slot released before"),
        ("missing-owner-result", "matching successful owner"),
    ],
)
def test_real_dispatch_evidence_cannot_bypass_material_checks(execution, fault, expected):
    e = execution
    backend = copy.deepcopy(e.backend)
    work = next(w for w in backend["s2_work_records"] if w["operation"] == "finish_tail")
    receipt = work["terminal_drain"]
    if fault == "same-request":
        path = e.directory / "target-diagnostics.jsonl"
        rows = CheckpointJsonl(path).read()
        verify = next(r for r in rows if r["request_id"] == e.verify)
        own = next(r for r in rows if r["request_id"] == e.tail)
        for key in ("target_forward_start_ns", "target_forward_end_ns"):
            own[key] = verify[key]
        jsonl(path, rows)
    elif fault == "nonterminal":
        receipt["rows"][0]["terminal"] = False
    elif fault == "proposal":
        receipt["results"][0]["proposal"] = {"proposal_token_ids": [10]}
    elif fault == "proposal-forward":
        receipt["proposal_forward_count"] = 1
    elif fault == "active-operation":
        work["operation"] = "commit_and_propose"
    elif fault == "active-drafting":
        # Extend actual active proposal work into another A request's earlier tail.
        # It remains disjoint from its own later verification.
        backend["s2_work_records"][0]["host_end_ns"] = 205
    elif fault == "missing-receipt":
        work.pop("terminal_drain")
    elif fault == "changed-kv":
        receipt["unrelated_after_sha256"] = "bad"
    elif fault == "same-device":
        receipt["gpu_uuid"] = "GPU-target-0"
    elif fault in ("early-release", "early-drain"):
        runtime = copy.deepcopy(e.runtime)
        if fault == "early-release":
            for event in runtime["events"]:
                if event["event"] == "resources-released" and event["request_id"] == e.tail:
                    event["timestamp_ns"] = work["host_start_ns"] + 1
        else:
            runtime["end_ns"] = 580
            runtime["target_final_sync"] = [{"timestamp_ns": 570}, {"timestamp_ns": 571}]
        dump(e.directory / "runtime.json", runtime)
        dump(
            e.directory / "arrival-output-events.json",
            {
                "events": runtime["events"],
                "requests": runtime["requests"],
                "failure": None,
            },
        )
    elif fault == "missing-owner-result":
        path = e.directory / "draft-work-events.jsonl"
        rows = CheckpointJsonl(path).read()
        for row in rows:
            if row["request_id"] == e.tail:
                row.pop("start_ns", None)
        jsonl(path, rows)
    dump(e.directory / "draft-backend-report.json", backend)
    result = qualify(e.path, e.directory, "pingpong", e.policy)
    assert not result["valid"] and expected in result["errors"][0], result
    error = result["primary_error"]
    assert error["field"] == "s2_stage_dependency"
    assert error["artifact"].endswith(
        ("draft-backend-report.json", "draft-work-events.jsonl", "runtime.json")
    )
    assert error["actual"]["operation"]
    assert error["actual"]["draft_request_ids"]
    assert "terminal_by_request" in error["actual"]
    assert error["actual"]["target_pairs"]
    assert "same_request_ids" in error["actual"]["target_pairs"][0]
    assert "target-diagnostics.jsonl" in error["artifacts"]


@pytest.fixture
def live_machine(tmp_path, monkeypatch):
    monkeypatch.setenv("SR_S2_CONTROL", str(tmp_path / "control.json"))
    monkeypatch.setenv("SR_S2_RUN_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("SR_S2_MODE", "pingpong")
    monkeypatch.setattr(s2_draft, "cuda_memory", lambda torch: {"free_memory_bytes": 10**9})
    states = {rid: {"state": "STAGED", "cohort": "A"} for rid in ("a", "b")}
    publish(tmp_path / "control.json", {"barrier_ns": None, "requests": states})
    worker = PoolWorker()
    worker.torch = None
    backend = s2_draft.S2DraftBackend(SimpleNamespace(max_model_len=4096), worker=worker)
    machine = s2_draft.S2DualMachine(backend)
    for rid in states:
        machine.initialize(initial(rid))
    for row in states.values():
        row["state"] = "ACTIVE"
    publish(tmp_path / "control.json", {"barrier_ns": 1, "requests": states})
    yield machine
    # Fault-injection cases may deliberately leave a corrupt pool. Always close
    # the fake worker without repeating the very S2 audit under test at teardown.
    VllmBatchedDraftBackend.shutdown(backend)


def tail_row(rid="a"):
    return {
        "request_id": rid,
        "logical_cohort": "A",
        "terminal": True,
        "prefix_version": 2,
        "committed_delta": [99],
        "prefix_token_sha256": token_prefix_hash((1, 2, 3, 99)),
    }


@pytest.mark.parametrize(
    "fault", ["nonterminal", "pending-proposal", "version", "hash", "shared-kv"]
)
def test_actual_tail_dispatch_rejects_invalid_state_before_mutation(live_machine, fault):
    m = live_machine
    row = tail_row()
    if fault == "nonterminal":
        row["terminal"] = False
    elif fault == "pending-proposal":
        m.execute_batch("propose_only", [{**initial("a", remaining=2), "logical_cohort": "A"}])
    elif fault == "version":
        row["prefix_version"] = 1
    elif fault == "hash":
        row["prefix_token_sha256"] = "bad"
    else:
        m.backend.worker.blocks["sr-draft:b"] = m.backend.worker.blocks["sr-draft:a"]
    before = len(m.backend.worker.calls)
    with pytest.raises(ValueError):
        m.execute_batch("finish_tail", [row])
    assert len(m.backend.worker.calls) == before
    assert not m.backend.retired
    assert m.backend.terminal_release is None


def test_actual_release_cannot_change_unrelated_prefix(live_machine, monkeypatch):
    m = live_machine
    original = m.backend.worker.release

    def wrong_scope(ids):
        original(ids)
        m.backend.states["b"].prefix += (10,)

    monkeypatch.setattr(m.backend.worker, "release", wrong_scope)
    with pytest.raises(ValueError, match="unrelated KV"):
        m.execute_batch("finish_tail", [tail_row()])
    assert not m.backend.history  # No successful receipt is fabricated.


def test_owner_inflight_holds_real_coordinator_slot_until_tail_release(tmp_path, monkeypatch):
    clock = clock_fixture(n=2)
    control_path = tmp_path / "control.json"
    monkeypatch.setenv("SR_S2_CONTROL", str(control_path))
    monkeypatch.setenv("SR_S2_RUN_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("SR_S2_MODE", "pingpong")
    monkeypatch.setattr(s2_draft, "cuda_memory", lambda torch: {"free_memory_bytes": 10**9})
    publish(control_path, clock.control())
    entered, unblock = threading.Event(), threading.Event()

    class BlockingRelease(PoolWorker):
        torch = None

        def release(self, ids):
            entered.set()
            assert unblock.wait(5)
            super().release(ids)

    def factory():
        backend = s2_draft.S2DraftBackend(
            SimpleNamespace(max_model_len=4096), worker=BlockingRelease()
        )
        return s2_draft.S2DualMachine(backend)

    controller = BatchedDualDraftController(factory, CheckpointJsonl(tmp_path / "work.jsonl"))
    try:
        for rid, definition in clock.definitions.items():
            controller.execute(
                "initialize", initial(rid, tuple(definition.prompt_token_ids) + (50,))
            )
        clock.start(time.monotonic_ns() - 20_000_000, threaded=False)
        clock.observe(time.monotonic_ns())
        (rid,) = clock.admit(time.monotonic_ns())
        clock.commit(rid, [999], time.monotonic_ns(), finished=True, finish_reason="stop")
        publish(control_path, clock.control())
        controller.enqueue(
            "finish_tail",
            [
                {
                    **tail_row(rid),
                    "committed_delta": [999],
                    "logical_cohort": clock.rows[rid]["cohort"],
                    "prefix_token_sha256": token_prefix_hash(
                        tuple(clock.definitions[rid].prompt_token_ids) + (50, 999)
                    ),
                }
            ],
        )
        assert entered.wait(5)
        release_finished(clock, set(controller.status()["inflight_request_ids"]))
        assert not clock.rows[rid]["resources_released"]
        assert not clock.complete and clock.admit(time.monotonic_ns()) == []
        assert rid in controller.machine.backend.states  # actual KV is still held
        unblock.set()
        deadline = time.monotonic() + 5
        while controller.status()["inflight_request_ids"] and time.monotonic() < deadline:
            time.sleep(0.001)
        assert not controller.status()["inflight_request_ids"]
        assert not controller.status()["failures"]
        release_finished(clock, set(controller.status()["inflight_request_ids"]))
        assert clock.rows[rid]["resources_released"]
        work = controller.machine.backend.history[-1]
        released = work["terminal_drain"]["release"]["resources_released_ns"]
        slot_time = next(
            e["timestamp_ns"] for e in clock.events if e["event"] == "resources-released"
        )
        assert clock.rows[rid]["completion_ns"] <= released <= work["host_end_ns"] <= slot_time
        assert len(clock.admit(time.monotonic_ns())) == 1
    finally:
        unblock.set()
        controller.shutdown()
        clock.close()


def test_foreground_failure_prints_both_stages_and_actual_artifacts(execution, capsys):
    from specrhythm.serving.s1_console import print_failure

    e = execution
    backend = copy.deepcopy(e.backend)
    work = next(w for w in backend["s2_work_records"] if w["operation"] == "finish_tail")
    work["terminal_drain"]["rows"][0]["terminal"] = False
    dump(e.directory / "draft-backend-report.json", backend)
    report = qualify(e.path, e.directory, "pingpong", e.policy)
    assert not report["valid"]
    dump(e.directory / "result.json", report)
    print_failure(
        ValueError(report["errors"][0]), directory=e.directory, mode="pingpong", label="S2 G1"
    )
    text = capsys.readouterr().err
    for expected in (
        "finish_tail",
        '"terminal": false',
        e.tail,
        e.verify,
        '"draft_cohort": "A"',
        "draft_host_interval_ns",
        "target_host_interval_ns",
        '"same_request_ids": []',
        str(e.directory / "draft-backend-report.json"),
        str(e.directory / "runtime.json"),
    ):
        assert expected in text
