from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from test_phase4_dual_batched_draft import commit_row, initial, wait
from test_phase4_dual_microbatch import helper
from test_phase4_dual_scheduler import Request, proposal_result, retire, tail_result
from test_phase4_dual_scheduler import scheduler as _scheduler
from test_phase4_serial import phase4_config as _config
from test_phase4_vllm_draft import FakeWorker

from specrhythm.phase4.dual_batched_draft import (
    BatchedDualDraftController,
    BatchedDualDraftMachine,
)
from specrhythm.phase4.dual_pingpong_draft import PingPongDraftMachine
from specrhythm.phase4.dual_rhythm import (
    CohortDraftClient,
    balanced_assignment,
    cohort_for,
    load_assignment,
    selected_rhythm,
    write_assignment,
)
from specrhythm.phase4.dual_service import run_dual_draft_service
from specrhythm.phase4.transport import CheckpointJsonl
from specrhythm.phase4.vllm_draft_backend import VllmBatchedDraftBackend
from specrhythm.phase4.vllm_dual import DualBatchRemoteProposer
from specrhythm.phase4.vllm_pingpong import PingPongRemoteProposer

scheduler = _scheduler
phase4_config = _config


def test_selector_default_and_target_methods_inherited():
    assert selected_rhythm({}) == "legacy"
    assert selected_rhythm({"SR_PHASE4B_DUAL_RHYTHM": "pingpong"}) == "pingpong"
    for method in (
        "propose",
        "on_target_verify_start",
        "on_target_verify_end",
        "_rank_zero_update",
    ):
        assert getattr(PingPongRemoteProposer, method) is getattr(DualBatchRemoteProposer, method)
    assert PingPongDraftMachine._commit_many is BatchedDualDraftMachine._commit_many
    assert PingPongDraftMachine._propose_many is BatchedDualDraftMachine._propose_many


@pytest.mark.parametrize("value", ["", "auto", "PINGPONG", "pingpong "])
def test_selector_fail_closed(value):
    with pytest.raises(ValueError):
        selected_rhythm({"SR_PHASE4B_DUAL_RHYTHM": value})


@pytest.mark.parametrize("count", [2, 5, 99, 100])
def test_assignment_balanced_stable_immutable(count):
    ids = [f"r-{i}" for i in range(count)]
    assignment = balanced_assignment(ids)
    assert set(assignment) == set(ids)
    assert abs(list(assignment.values()).count("A") - list(assignment.values()).count("B")) <= 1
    before = dict(assignment)
    remaining = ids[1:]
    assert all(assignment[r] == before[r] for r in remaining)
    with pytest.raises(TypeError):
        assignment[ids[0]] = "B"
    with pytest.raises(ValueError, match="mix"):
        cohort_for(assignment, ids)
    with pytest.raises(ValueError, match="unknown"):
        cohort_for(assignment, ["new-arrival"])


def test_manifest_binds_unsorted_workload_and_detects_changes(tmp_path, monkeypatch):
    from specrhythm.phase4 import dual_rhythm

    workload, output = tmp_path / "w", tmp_path / "assignment.json"
    workload.write_text("frozen input")
    rows = [SimpleNamespace(request_id=i) for i in ("z", "a", "q", "b", "c")]
    monkeypatch.setattr(dual_rhythm, "load_smoke_requests", lambda *a, **kw: rows)
    write_assignment(output, workload, 5)
    assert dict(load_assignment(output, workload=workload, count=5)) == {
        "z": "A",
        "a": "B",
        "q": "A",
        "b": "B",
        "c": "A",
    }
    with pytest.raises(FileExistsError):
        write_assignment(output, workload, 5)
    workload.write_text("different workload")
    with pytest.raises(ValueError, match="frozen workload"):
        load_assignment(output, workload=workload)
    value = json.loads(output.read_text())
    value["assignment"]["z"] = "B"
    output.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="assignment"):
        load_assignment(output)


@pytest.fixture
def pingpong_factory(scheduler, monkeypatch, tmp_path):
    base_module = ModuleType("specrhythm.phase4.vllm_dual_scheduler")
    base_module.DualBatchScheduler = type(scheduler)
    monkeypatch.setitem(sys.modules, base_module.__name__, base_module)
    path = Path("src/specrhythm/phase4/vllm_pingpong_scheduler.py")
    spec = importlib.util.spec_from_file_location("cpu_pingpong_scheduler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def factory(count=2):
        ids = list("ab") if count == 2 else [f"r{i}" for i in range(count)]
        definitions = [
            SimpleNamespace(request_id=rid, prompt_token_ids=(i * 2 + 1, i * 2 + 2))
            for i, rid in enumerate(ids)
        ]
        monkeypatch.setitem(
            type(scheduler).__init__.__globals__,
            "load_smoke_requests",
            lambda *a, **kw: definitions,
        )
        monkeypatch.setattr(module, "load_assignment", lambda **kw: balanced_assignment(ids))
        for key, value in {
            "SR_PHASE4_DUAL_RESIDENT": "1",
            "SR_PHASE4_DUAL_MICROBATCH_SIZE": str(count),
            "SR_PHASE4_RESIDENT_SETUP_READY": str(tmp_path / "setup-ready.json"),
            "SR_PHASE4_DECODE_READY_MANIFEST": str(tmp_path / "decode-ready.json"),
        }.items():
            monkeypatch.setenv(key, value)
        value = module.PingPongScheduler()
        value.requests = {
            f"opaque-{r.request_id}": Request(f"opaque-{r.request_id}", r.prompt_token_ids)
            for r in definitions
        }
        value.running = list(value.requests.values())
        value._bind_vllm_requests()
        value._dual_setup_ready = {"valid": True}
        return value

    return factory


def consume(value, output):
    # Stock vLLM clears spec IDs only for scheduled requests and returns their rows.
    for internal_id in output.num_scheduled_tokens:
        request = value.requests[internal_id]
        request.spec_token_ids = []
        request.all_token_ids += [99]
        request.num_output_tokens += 1
        request.num_computed_tokens = len(request.all_token_ids) - 1


def test_fill_then_alternation_holds_other_ready_proposal(pingpong_factory):
    s = pingpong_factory()
    s._accept_ready_result(proposal_result("a"))
    assert not s.schedule().num_scheduled_tokens  # Both initial groups must be ready.
    s._accept_ready_result(proposal_result("b"))
    first = s.schedule()
    assert list(first.num_scheduled_tokens) == ["opaque-a"]
    consume(s, first)
    assert s.requests["opaque-b"].spec_token_ids == [11, 12]
    second = s.schedule()
    assert list(second.num_scheduled_tokens) == ["opaque-b"]
    consume(s, second)
    assert not s.schedule().num_scheduled_tokens
    assert s.selected_cohort == "A"  # Empty polls do not opportunistically flip.
    s._accept_ready_result(proposal_result("a", prefix=(1, 2, 10, 99), version=2, round_id=1))
    assert list(s.schedule().num_scheduled_tokens) == ["opaque-a"]
    rows = [r for r in s._dual_events.read() if r.get("attained_cohort_request_ids")]
    assert [r["logical_cohort"] for r in rows] == ["A", "B", "A"]


@pytest.mark.parametrize("count", [5, 100])
def test_full_eligible_large_cohort_without_mb2_fragmentation(pingpong_factory, count):
    s = pingpong_factory(count)
    for rid in s.assignment:
        r = s.requests[s._dual_identity.internal_id(rid)]
        s._accept_ready_result(proposal_result(rid, prefix=tuple(r.all_token_ids)))
    output = s.schedule()
    assert len(output.num_scheduled_tokens) == (count + 1) // 2
    assert all(
        s.assignment[s._dual_identity.stable_id(r)] == "A" for r in output.num_scheduled_tokens
    )
    consume(s, output)
    assert len(s.schedule().num_scheduled_tokens) == count // 2
    assert not any(r.get("capacity_clipped") for r in s._dual_events.read())


def test_drain_retirement_and_terminal_tail(pingpong_factory):
    s = pingpong_factory()
    s._accept_ready_result(proposal_result("a"))
    s._accept_ready_result(tail_result("b"))
    consume(s, s.schedule())
    retire(s, "a")
    output = s.schedule()
    assert output.num_scheduled_tokens == {"opaque-b": 1}
    assert not output.scheduled_spec_decode_tokens
    assert s._dual_events.read()[-1]["pipeline_phase"] == "drain"
    retire(s, "b")
    assert not s.schedule().num_scheduled_tokens
    assert s.assignment == {"a": "A", "b": "B"}


def test_steady_state_does_not_wait_to_fill_entire_cohort(pingpong_factory):
    s = pingpong_factory(5)
    for rid in s.assignment:
        r = s.requests[s._dual_identity.internal_id(rid)]
        s._accept_ready_result(proposal_result(rid, prefix=tuple(r.all_token_ids)))
    consume(s, s.schedule())
    consume(s, s.schedule())
    r = s.requests["opaque-r0"]
    s._accept_ready_result(
        proposal_result("r0", prefix=tuple(r.all_token_ids), version=2, round_id=1)
    )
    assert list(s.schedule().num_scheduled_tokens) == ["opaque-r0"]


def test_policy_wait_does_not_trigger_legacy_ready_deferral_failure(pingpong_factory):
    from specrhythm.phase4.dual_correctness import validate_scheduler_cycles

    s = pingpong_factory()
    s._accept_ready_result(proposal_result("a"))
    for _ in range(20):
        assert not s.schedule().num_scheduled_tokens
    assert not validate_scheduler_cycles(s._dual_events.read())
    assert all(
        r["request_admissibility"][0]["inadmissible_reason"] == "pingpong initial pipeline fill"
        for r in s._dual_events.read()
    )


def test_clipping_retains_stock_budget_and_records_attained_size(pingpong_factory, monkeypatch):
    s = pingpong_factory(100)
    for rid in s.assignment:
        r = s.requests[s._dual_identity.internal_id(rid)]
        s._accept_ready_result(proposal_result(rid, prefix=tuple(r.all_token_ids)))
    stock = type(s).__mro__[2]
    original = stock.schedule

    def clipped(self):
        result = original(self)
        ids = list(result.num_scheduled_tokens)[:7]
        result.num_scheduled_tokens = {r: result.num_scheduled_tokens[r] for r in ids}
        result.scheduled_spec_decode_tokens = {
            r: result.scheduled_spec_decode_tokens[r] for r in ids
        }
        return result

    monkeypatch.setattr(stock, "schedule", clipped)
    s.max_num_scheduled_tokens = 35
    output = s.schedule()
    row = s._dual_events.read()[-1]
    assert len(output.num_scheduled_tokens) == 7
    assert len(row["eligible_cohort_request_ids"]) == 50
    assert row["capacity_clipped"]
    assert row["dual_scheduler_constraints"]["max_num_scheduled_tokens"] == 35


def test_pending_same_cohort_draft_blocks_held_ready_members(pingpong_factory):
    s = pingpong_factory(5)
    s.fill_complete = True
    s.initial_ready = set(s.assignment)
    rid = "r0"
    r = s.requests["opaque-r0"]
    s._accept_ready_result(proposal_result(rid, prefix=tuple(r.all_token_ids)))
    s._dual_client.responses.append({"ready": [], "pending_request_ids": ["r2"]})
    assert not s.schedule().num_scheduled_tokens
    assert s._dual_events.read()[-1]["target_waiting_for_draft"]
    s._dual_client.responses.append({"ready": [], "pending_request_ids": []})
    assert list(s.schedule().num_scheduled_tokens) == ["opaque-r0"]


def test_initial_submit_split_commit_not_split_or_synchronously_waited():
    calls = []
    client = CohortDraftClient(
        SimpleNamespace(call=lambda op, p: calls.append((op, p))), balanced_assignment("abcde")
    )
    client.call(
        "enqueue", {"work_operation": "propose_only", "rows": [{"request_id": r} for r in "abcde"]}
    )
    assert [[r["request_id"] for r in p["rows"]] for _, p in calls] == [list("ace"), list("bd")]
    assert [p["rows"][0]["logical_cohort"] for _, p in calls] == ["A", "B"]
    with pytest.raises(ValueError, match="mix"):
        client.call(
            "enqueue",
            {"work_operation": "commit_and_propose", "rows": [{"request_id": r} for r in "ab"]},
        )
    assert len(calls) == 2


def test_production_machine_eos_tail_prefix_checks_and_final_report(phase4_config, tmp_path):
    backend = VllmBatchedDraftBackend(phase4_config, worker=FakeWorker())
    machine = PingPongDraftMachine(
        backend, assignment=balanced_assignment("abc"), report_path=tmp_path / "report.json"
    )
    rows = [initial("a"), initial("b", remaining=1), initial("c")]
    for row in rows:
        machine.initialize(row)
    a_rows = [{**r, "logical_cohort": "A"} for r in rows if r["request_id"] != "b"]
    first = machine.execute_batch("propose_only", a_rows)
    commits = [
        {
            **commit_row(
                machine, r["proposal"], r["proposal"]["proposal_token_ids"][:1], terminal=True
            ),
            "logical_cohort": "A",
        }
        for r in first
    ]
    bad = [{**commits[0], "prefix_token_sha256": "f" * 64}, commits[1]]
    with pytest.raises(ValueError, match="prefix"):
        machine.execute_batch("commit_and_propose", bad)
    assert not machine.requests["a"].finished
    machine.execute_batch("commit_and_propose", commits)
    tail = {**rows[1], "logical_cohort": "B"}
    assert machine.execute_batch("propose_only", [tail])[0]["target_tail"]
    from specrhythm.phase4.serial import token_prefix_hash

    machine.execute_batch(
        "finish_tail",
        [
            {
                "request_id": "b",
                "logical_cohort": "B",
                "terminal": True,
                "committed_delta": [99],
                "prefix_version": 2,
                "prefix_token_sha256": token_prefix_hash((1, 2, 3, 99)),
            }
        ],
    )
    machine.shutdown()
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["draft_live_requests_final"] == 0
    assert report["backend_shutdown_complete"]
    assert report["draft_retired_request_count"] == 3
    assert {r["logical_cohort"] for r in report["pingpong_work_records"]} == {"A", "B"}
    assert (
        sum(
            sum(r["forward_batch_histograms"]["proposal"].values())
            for r in report["pingpong_work_records"]
        )
        == report["draft_model_forward_count_by_purpose"]["proposal"]
    )


def test_owner_allows_opposite_target_while_committed_draft_is_inflight(phase4_config, tmp_path):
    entered, release = threading.Event(), threading.Event()

    def factory():
        backend = VllmBatchedDraftBackend(phase4_config, worker=FakeWorker())
        original = backend.commit_many

        def gated(plans):
            entered.set()
            assert release.wait(5)
            return original(plans)

        backend.commit_many = gated
        return PingPongDraftMachine(backend, assignment=balanced_assignment("abcde"))

    controller = BatchedDualDraftController(factory, CheckpointJsonl(tmp_path / "work.jsonl"))
    machine = controller.machine
    try:
        for rid in "abcde":
            controller.execute("initialize", initial(rid))
        for group, ids in (("A", "ace"), ("B", "bd")):
            controller.enqueue(
                "propose_only", [{**initial(r), "logical_cohort": group} for r in ids]
            )
        ready = {r["request_id"]: r for r in wait(controller, 5)}
        commits = [
            {**commit_row(machine, ready[r]["proposal"], [999]), "logical_cohort": "A"}
            for r in "ace"
        ]
        controller.enqueue("commit_and_propose", commits)  # Returns before owner completes.
        assert entered.wait(5)
        assert set(controller.status()["inflight_request_ids"]) == set("ace")
        assert {r["request_id"] for r in controller.claimed(list("bd"))["claimed"]} == set("bd")
        assert not controller.poll_ready(5)["ready"]
        with pytest.raises(ValueError, match="in flight"):
            controller.enqueue("commit_and_propose", commits)
    finally:
        release.set()
        controller.shutdown()


@pytest.mark.parametrize("mode", ["target", "serial"])
def test_nondual_ignores_invalid_rhythm(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("SR_PHASE4B_DUAL_RHYTHM", "invalid")
    result, args = helper(tmp_path, mode, "invalid")
    assert result.returncode == 0, result.stderr
    assert "--microbatch-size" not in args


def test_hf_pingpong_rejected_before_model_load(phase4_config, monkeypatch, tmp_path):
    monkeypatch.setenv("SR_PHASE4B_DUAL_RHYTHM", "pingpong")
    monkeypatch.setenv("SR_PHASE4_DRAFT_BACKEND", "hf-persistent")
    with pytest.raises(ValueError, match="production"):
        run_dual_draft_service(
            phase4_config,
            socket_path=tmp_path / "s",
            event_log_path=tmp_path / "e",
            transport_log_path=tmp_path / "t",
            ready_path=tmp_path / "r",
        )


def test_fixed_sweep_explicitly_legacy_and_mb100_remains_supported():
    source = Path("integrations/vllm/phase4b4_microbatch_helpers.sh").read_text()
    assert "SR_PHASE4B_DUAL_RHYTHM=legacy" in source
    assert "2|4|8|16|32|64|100" in source
    assert "microbatch_size: int = 2" in inspect.getsource(
        __import__(
            "specrhythm.phase4.dual_runner", fromlist=["run_resident_dual_batch"]
        ).run_resident_dual_batch
    )
