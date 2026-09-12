"""Exercise the actual Serial-eager owner, machine, physical backend and IPC."""

from __future__ import annotations

import json
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_rolling_eager_gpu_backend import FencedWorker, greedy_tokens

from specrhythm.continuation.gpu_backend import RollingVllmDraftBackend
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.transport import CheckpointJsonl, UnixDraftClient
from specrhythm.serving import eager_draft
from specrhythm.serving.eager_machine import EagerSerialMachine
from specrhythm.serving.eager_owner import EagerOwner
from specrhythm.serving.eager_proposer import EagerSerialProposer
from specrhythm.serving.eager_results import summarize_eager
from specrhythm.serving.s2_pool import prefix_record
from specrhythm.serving.s2_proposer import S2SerialProposer


class OwnerWorker(FencedWorker):
    provenance = {"backend": "CPU-physical-substitute", "physical_gpu_id": 0,
                  "gpu_uuid": "GPU-owner-test"}

    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.resume = threading.Event()
        self.block_eager = False

    def materialize(self, rows, purpose):
        if purpose == "eager" and self.block_eager:
            self.entered.set()
            if not self.resume.wait(5):
                raise TimeoutError("test did not release the bounded physical forward")
            self.block_eager = False
        return super().materialize(rows, purpose)


class OwnerBackend(RollingVllmDraftBackend):
    def physical_rows(self):
        return {rid: prefix_record(state.prefix, state.materialized, [[id(state)]])
                for rid, state in self.states.items()}


def make_machine(worker=None):
    backend = OwnerBackend(SimpleNamespace(max_model_len=4096), worker=worker or OwnerWorker())
    return EagerSerialMachine(backend, request_ids=("a", "b", "c"))


def proposal_row(rid="a", prefix=(10, 20), round_id=0, remaining=100):
    return {"request_id": rid, "round_id": round_id, "committed_prefix_len": len(prefix),
            "committed_prefix_hash": token_prefix_hash(prefix),
            "remaining_output_budget": remaining, "eos_token_ids": []}


def verify_row(proposal):
    return {"request_id": proposal["request_id"], "round_id": proposal["round_id"],
            "proposal_id": proposal["runtime_provenance"]["rolling_proposal_id"],
            "parent_prefix_hash": proposal["parent_prefix_hash"],
            "parent_prefix_len": proposal["parent_prefix_len"],
            "proposal_tokens": list(proposal["proposal_token_ids"])}


def sync_row(proposal, prefix, *, reject=False, terminal=False):
    tokens = tuple(proposal["proposal_token_ids"])
    delta = tokens[:2] + (900,) if reject else tokens + greedy_tokens(prefix + tokens, 1)
    final = prefix + delta
    return {"request_id": proposal["request_id"], "round_id": proposal["round_id"],
            "committed_delta": list(delta), "committed_prefix_hash": token_prefix_hash(final),
            "terminal": terminal}, final


def settle_row(rid="a", prefix=(10, 20), round_id=0, *, terminal=False, deadline=None):
    return {"request_id": rid, "committed_prefix": list(prefix),
            "committed_prefix_hash": token_prefix_hash(prefix), "natural_terminal": terminal,
            "admitted": True, "runtime_mode": "serial-eager", "next_round_id": round_id,
            "prefix_version": round_id, "bootstrap_prefix_hash": token_prefix_hash((10, 20)),
            "deadline_ns": deadline or time.monotonic_ns() + 3_000_000_000}


@pytest.fixture
def owner_factory():
    owners = []

    def create(worker=None):
        owner = EagerOwner(lambda: make_machine(worker), timeout_seconds=3)
        owners.append(owner)
        return owner

    yield create
    for owner in owners:
        owner.machine.backend.worker.resume.set()
        if not owner.closed and owner.failure is None:
            owner.call("status", {})
            for rid, state in tuple(owner.machine.requests.items()):
                if rid not in owner.machine.backend.retired:
                    owner.call("diagnostic_settle", settle_row(
                        rid, state.committed_token_ids, state.next_round_id,
                        terminal=state.finished,
                    ))
            owner.call("shutdown", {"deadline_ns": time.monotonic_ns() + 3_000_000_000})
        owner._thread.join(timeout=3)
        assert not owner._thread.is_alive(), "test cleanup did not physically retire owner"


def initialize(client, *, rid="a", prefix=(10, 20), remaining=100):
    client.call("initialize", {"request_id": rid, "committed_token_ids": list(prefix),
                              "committed_prefix_hash": token_prefix_hash(prefix)})
    result = client.call("batch_propose", {"requests": [proposal_row(rid, prefix, 0, remaining)]})
    return result["proposals"][0] if result["proposals"] else None


@pytest.fixture
def short_socket():
    # macOS AF_UNIX paths cannot fit pytest's long system temporary directory.
    with tempfile.TemporaryDirectory(prefix="sr-eager-", dir="/tmp") as directory:
        yield Path(directory) / "eager.sock"


def test_enqueue_returns_while_forward_blocked_and_feedback_preempts_next_step(owner_factory):
    worker = OwnerWorker()
    worker.block_eager = True
    owner = owner_factory(worker)
    proposal = initialize(owner)
    assert owner.call("eager_enqueue", {"requests": [verify_row(proposal)]})["enqueued"]
    assert worker.entered.wait(2)
    sync, final = sync_row(proposal, (10, 20), reject=True)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(owner.call, "synchronize_and_batch_propose", {
            "synchronizations": [sync],
            "proposals": [proposal_row(prefix=final, round_id=1, remaining=97)],
        })
        deadline = time.monotonic() + 2
        while owner.commands.empty() and time.monotonic() < deadline:
            threading.Event().wait(0.001)
        assert not owner.commands.empty() and not future.done()
        worker.resume.set()
        result = future.result(timeout=2)
    assert tuple(result["proposals"][0]["proposal_token_ids"]) == greedy_tokens(final, 4)
    assert len([1 for purpose, _ in worker.calls if purpose == "eager"]) == 1
    assert owner.machine.counters["recovery_jobs"] == 1
    assert owner.machine.counters["early_generated_tokens"] == 1


@pytest.mark.parametrize("target_first", [False, True])
def test_machine_repeated_success_reject_recovery_success_uses_real_backend(target_first):
    machine = make_machine()
    prefix = (10, 20)
    machine.initialize("a", prefix, token_prefix_hash(prefix))
    proposal = machine.batch_propose([proposal_row()])["proposals"][0]
    remaining = 100
    try:
        for round_id in range(6):
            machine.verify_start([verify_row(proposal)])
            sync, final = sync_row(proposal, prefix, reject=round_id == 2)
            if target_first:
                machine.prepare_synchronizations([sync])
            while machine.step():
                pass
            if not target_first:
                machine.prepare_synchronizations([sync])
            assert machine.finish_synchronizations([sync]) is not None
            remaining -= len(sync["committed_delta"])
            proposal = machine.batch_propose([
                proposal_row(prefix=final, round_id=round_id + 1, remaining=remaining)
            ])["proposals"][0]
            assert tuple(proposal["proposal_token_ids"]) == greedy_tokens(final, 4)
            assert machine.backend.states["a"].prefix == final
            assert machine.core.state("a").eager_decision.eligible
            prefix = final
        assert machine.counters["promotions"] == 5
        assert machine.counters["parent_rejections"] == 1
        assert machine.counters["recovery_jobs"] == 1
        assert machine.counters["committed_tokens"] == 28
        assert machine.counters["verified_promoted_candidates"] == 16
        result = machine.diagnostic_settle(settle_row(prefix=prefix, round_id=6))
        assert result["released"] and result["new_proposals_generated"] == 0
        assert machine.shutdown()["shutdown"]
    finally:
        machine.backend.shutdown()


def test_machine_mixed_batch_promotes_recovers_and_finishes_terminal():
    machine = make_machine()
    prefixes = {"a": (10, 20), "b": (10, 21), "c": (10, 22)}
    try:
        for rid, prefix in prefixes.items():
            machine.initialize(rid, prefix, token_prefix_hash(prefix))
        proposals = machine.batch_propose([
            proposal_row(rid, prefix, remaining=5 if rid == "c" else 100)
            for rid, prefix in prefixes.items()
        ])["proposals"]
        machine.verify_start([verify_row(p) for p in proposals])
        while machine.step():
            pass
        rows = [sync_row(p, prefixes[p["request_id"]], reject=p["request_id"] == "b",
                         terminal=p["request_id"] == "c") for p in proposals]
        machine.prepare_synchronizations([row for row, _ in rows])
        assert len(machine.finish_synchronizations([row for row, _ in rows])) == 3
        assert machine.core.state("a").current_proposal_id is not None
        assert machine.core.state("b").recovery_required
        assert machine.core.state("c").finished and "c" in machine.backend.retired
        for row, final in rows[:2]:
            assert machine.diagnostic_settle(settle_row(
                row["request_id"], final, round_id=1
            ))["released"]
        assert machine.shutdown()["shutdown"]
    finally:
        machine.backend.shutdown()


@pytest.mark.parametrize("tail_only", [False, True])
def test_finish_authoritative_exact_prefix_and_target_only_tail(owner_factory, tail_only):
    owner = owner_factory()
    proposal = initialize(owner, remaining=1 if tail_only else 100)
    prefix = (10, 20)
    if not tail_only:
        owner.call("eager_enqueue", {"requests": [verify_row(proposal)]})
    else:
        prefix += greedy_tokens(prefix, 1)
    payload = {"request_id": "a", "committed_prefix": list(prefix),
               "committed_prefix_hash": token_prefix_hash(prefix)}
    assert owner.call("finish_request", payload)["finished"]
    assert owner.call("finish_request", payload)["finished"]
    assert owner.call("shutdown", {"deadline_ns": time.monotonic_ns() + 2_000_000_000})["shutdown"]
    assert not owner.machine.backend.worker.memory


def test_drain_settles_feedback_and_retires_partial_work(owner_factory):
    worker = OwnerWorker()
    worker.block_eager = True
    owner = owner_factory(worker)
    proposal = initialize(owner)
    owner.call("eager_enqueue", {"requests": [verify_row(proposal)]})
    assert worker.entered.wait(2)
    _, final = sync_row(proposal, (10, 20))
    row = settle_row(prefix=final, round_id=1)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(owner.call, "diagnostic_settle", row)
        worker.resume.set()
        result = future.result(timeout=2)
    assert result["released"] and result["new_proposals_generated"] == 0
    assert not owner.machine.backend.worker.memory
    assert owner.call("diagnostic_settle", row) == result
    assert owner.call("shutdown", {"deadline_ns": row["deadline_ns"]})["shutdown"]


def test_owner_deadline_timeout_preserves_physical_work(owner_factory):
    worker = OwnerWorker()
    worker.block_eager = True
    owner = owner_factory(worker)
    proposal = initialize(owner)
    owner.call("eager_enqueue", {"requests": [verify_row(proposal)]})
    assert worker.entered.wait(2)
    with pytest.raises(TimeoutError, match="physical work retained"):
        owner.call("status", {"deadline_ns": time.monotonic_ns() + 20_000_000})
    assert owner._thread.is_alive() and worker.memory and not owner.closed
    worker.resume.set()
    assert owner.call("status", {})["failures"] == []


def test_owner_waits_for_unfinished_valid_continuation_and_duplicate_feedback_is_idempotent(
    owner_factory,
):
    worker = OwnerWorker()
    worker.block_eager = True
    owner = owner_factory(worker)
    proposal = initialize(owner)
    owner.call("eager_enqueue", {"requests": [verify_row(proposal)]})
    assert worker.entered.wait(2)
    sync, final = sync_row(proposal, (10, 20))
    payload = {"synchronizations": [sync],
               "proposals": [proposal_row(prefix=final, round_id=1, remaining=95)]}
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(owner.call, "synchronize_and_batch_propose", payload)
        worker.resume.set()
        result = future.result(timeout=2)
    next_proposal = result["proposals"][0]
    assert next_proposal["runtime_provenance"]["source_continuation_id"] is not None
    assert len([1 for purpose, _ in worker.calls if purpose == "eager"]) == 5
    assert owner.machine.counters["unhidden_wait_ns"] > 0
    assert owner.machine.counters["committed_tokens"] == 5
    duplicate = owner.call("synchronize_and_batch_propose", payload)
    assert duplicate["proposals"][0] == next_proposal
    assert owner.machine.counters["committed_tokens"] == 5
    changed = {**sync, "committed_delta": [999]}
    with pytest.raises(RuntimeError, match="conflicting duplicate Target feedback"):
        owner.call(
            "synchronize_and_batch_propose", {"synchronizations": [changed], "proposals": []}
        )
    assert owner.call("status", {})["failures"] == []


def test_machine_switch_off_aborts_partial_then_switch_on_rolls_from_legal_prefix():
    machine = make_machine()
    prefix = (10, 20)
    try:
        machine.initialize("a", prefix, token_prefix_hash(prefix))
        proposal = machine.batch_propose([proposal_row()])["proposals"][0]
        machine.verify_start([verify_row(proposal)])
        machine.step()
        assert machine.set_eager(False, 1)["enabled"] is False
        assert not machine.step()
        sync, final = sync_row(proposal, prefix)
        machine.prepare_synchronizations([sync])
        machine.finish_synchronizations([sync])
        assert machine.core.state("a").committed_prefix == final
        assert machine.counters["promotions"] == 0
        recovered = machine.batch_propose([
            proposal_row(prefix=final, round_id=1, remaining=95)
        ])["proposals"][0]
        machine.set_eager(True, 2)
        machine.verify_start([verify_row(recovered)])
        while machine.step():
            pass
        sync, final = sync_row(recovered, final)
        machine.prepare_synchronizations([sync])
        machine.finish_synchronizations([sync])
        assert machine.counters["promotions"] == 1
        assert machine.core.state("a").eager_decision.eligible
        assert machine.diagnostic_settle(settle_row(prefix=final, round_id=2))["released"]
        # A replacement gets its own stable ID, version and first proposal.
        machine.initialize("b", prefix, token_prefix_hash(prefix))
        refill = machine.batch_propose([proposal_row("b")])["proposals"][0]
        assert refill["round_id"] == 0
        assert refill["runtime_provenance"]["source_continuation_id"] is None
        assert machine.core.state("b").active_continuation_id is None
        assert machine.diagnostic_settle(settle_row("b"))["released"]
        assert machine.shutdown()["shutdown"]
    finally:
        machine.backend.shutdown()


def test_parent_rejection_before_first_step_is_admission_without_gpu_start():
    machine = make_machine()
    prefix = (10, 20)
    try:
        machine.initialize("a", prefix, token_prefix_hash(prefix))
        proposal = machine.batch_propose([proposal_row()])["proposals"][0]
        machine.verify_start([verify_row(proposal)])
        assert machine.counters["admissions"] == 1
        assert machine.counters["started"] == 0
        assert machine.backend.metrics.counters["eager_enrolled"] == 1
        assert machine.backend.metrics.counters["eager_started"] == 0
        sync, final = sync_row(proposal, prefix, reject=True)
        machine.prepare_synchronizations([sync])
        assert machine.finish_synchronizations([sync]) is not None
        assert not machine.step()
        assert machine.counters["started"] == 0
        assert machine.counters["early_generated_tokens"] == 0
        assert machine.backend.metrics.forwards["eager"] == 0
        assert machine.diagnostic_settle(settle_row(prefix=final, round_id=1))["released"]
        assert machine.shutdown()["shutdown"]
    finally:
        machine.backend.shutdown()


def test_actual_owner_report_flows_into_window_summary_with_authoritative_output(owner_factory):
    owner = owner_factory()
    measured_start = time.monotonic_ns()
    proposal = initialize(owner)
    prefix = (10, 20)
    remaining = 100
    committed = []
    for round_id in range(2):
        owner.call("eager_enqueue", {"requests": [verify_row(proposal)]})
        sync, final = sync_row(proposal, prefix, reject=round_id == 1)
        committed.append({"token_ids": sync["committed_delta"]})
        remaining -= len(sync["committed_delta"])
        result = owner.call("synchronize_and_batch_propose", {
            "synchronizations": [sync],
            "proposals": [proposal_row(prefix=final, round_id=round_id + 1, remaining=remaining)],
        })
        proposal = result["proposals"][0]
        prefix = final
    owner.call("diagnostic_settle", settle_row(prefix=prefix, round_id=2))
    owner.call("shutdown", {"deadline_ns": time.monotonic_ns() + 2_000_000_000})
    measured_end = time.monotonic_ns()
    actual_backend = {
        **owner.machine.backend.report(), "rolling_eager": owner.machine.eager_report()
    }
    summary = summarize_eager(actual_backend, {
        "measurement_start_ns": measured_start, "measurement_end_ns": measured_end,
        "requests": [{"request_id": "a", "commits": committed}], "target_devices": [],
    })
    counts = summary["lifetime_counters"]
    assert summary["window_counters"] == counts
    assert counts["committed_tokens"] == 8
    assert counts["parent_accepted_tokens"] == 6
    assert counts["correction_tokens"] == counts["bonus_tokens"] == 1
    assert counts["promotions"] == 1
    assert counts["verified_promoted_candidates"] == 4
    assert counts["accepted_promoted_candidates"] == 2
    assert summary["window_unhidden_wait_ns"] == counts["unhidden_wait_ns"]
    assert summary["event_count"] == len(actual_backend["rolling_eager"]["events"])
    assert summary["cleanup_status"] == "PASS"
    assert summary["GPU_overlap"]["status"] == "UNKNOWN"
    assert not summary["GPU_overlap"]["host_overlap_is_gpu_evidence"]


@pytest.mark.parametrize("steps", [1, 5])
def test_repeated_gpu_retirement_requires_exact_partial_or_completed_tokens(steps):
    machine = make_machine()
    prefix = (10, 20)
    try:
        machine.initialize("a", prefix, token_prefix_hash(prefix))
        proposal = machine.batch_propose([proposal_row()])["proposals"][0]
        machine.verify_start([verify_row(proposal)])
        work = machine.works["a"]
        for _ in range(steps):
            assert machine.step()
        sync, final = sync_row(proposal, prefix, reject=True)
        machine.prepare_synchronizations([sync])
        evidence = machine.backend.gpu_continuation_snapshot(work)
        tokens = evidence["generated_tokens"]
        frontier = evidence["materialized_kv_frontier"]
        assert len(tokens) == steps
        before = machine.core.state("a")
        assert machine.core.record_continuation_abort(work, tokens, frontier) == "invalidated"
        assert machine.core.state("a") == before
        with pytest.raises(ValueError, match="conflicting already completed GPU retirement"):
            machine.core.record_continuation_abort(
                work, tokens[:-1] + (tokens[-1] + 1000,), frontier
            )
        assert machine.core.state("a") == before
        assert before.accounting.early_generated_tokens == steps
        assert before.accounting.discarded_early_tokens == steps
        assert machine.finish_synchronizations([sync]) is not None
        assert machine.diagnostic_settle(settle_row(prefix=final, round_id=1))["released"]
        assert machine.shutdown()["shutdown"]
        assert not machine.backend.worker.memory
    finally:
        machine.backend.shutdown()


def test_duplicate_admission_submits_once_and_conflict_keeps_error(owner_factory):
    owner = owner_factory()
    proposal = initialize(owner)
    row = verify_row(proposal)
    owner.call("eager_enqueue", {"requests": [row]})
    owner.call("eager_enqueue", {"requests": [row]})
    owner.call("status", {})
    assert owner.machine.counters["admissions"] == 1
    assert owner.machine.backend.metrics.counters["eager_enrolled"] == 1
    row["proposal_tokens"] = [999]
    owner.call("eager_enqueue", {"requests": [row]})
    owner._thread.join(timeout=2)
    assert not owner._thread.is_alive()
    assert "conflicting duplicate" in str(owner.failure)
    assert owner.machine.backend.closed


@pytest.mark.parametrize("rank", [0, 1])
def test_target_hook_baseline_first_and_rank_zero_only(monkeypatch, rank):
    calls = []
    proposal = SimpleNamespace(round_id=3, proposal_token_ids=(1, 2, 3, 4),
                               parent_prefix_len=2, parent_prefix_hash="parent-hash",
                               runtime_provenance={"rolling_proposal_id": "proposal-3"})
    proposer = object.__new__(EagerSerialProposer)
    proposer.tp_rank = rank
    proposer.identity = SimpleNamespace(stable_id=lambda rid: "a")
    proposer.requests = {"a": SimpleNamespace(pending_proposal=None)}
    proposer.tp_group = SimpleNamespace(barrier=lambda: calls.append("enqueue-barrier"))

    def baseline(self, **kwargs):
        calls.append("baseline-barrier")
        self.requests["a"].pending_proposal = proposal

    def submit(operation, payload):
        calls.append((operation, payload))
        return {"enqueued": True}

    monkeypatch.setattr(S2SerialProposer, "on_target_verify_start", baseline)
    proposer.client = SimpleNamespace(call=submit)
    proposer.on_target_verify_start(request_ids=["internal-a"],
                                    scheduled_spec_token_ids={"internal-a": [1, 2, 3, 4]})
    calls.append("target-forward")
    assert calls[0] == "baseline-barrier" and calls[-1] == "target-forward"
    assert calls[-2] == "enqueue-barrier"
    assert len(calls) == (4 if rank == 0 else 3)
    if rank == 0:
        assert calls[1][0] == "eager_enqueue"
        assert calls[1][1]["requests"][0]["proposal_id"] == "proposal-3"


def test_real_unix_server_routes_commands_and_preserves_drain_deadline(
    owner_factory, tmp_path, monkeypatch, short_socket,
):
    owner = owner_factory()
    ready = threading.Event()
    writer = eager_draft.atomic_write_json

    def write_ready(path, value):
        writer(path, value)
        ready.set()

    monkeypatch.setattr(eager_draft, "atomic_write_json", write_ready)
    socket_path = short_socket
    server = eager_draft.EagerSerialServer(
        socket_path, owner, event_log=CheckpointJsonl(tmp_path / "events.jsonl")
    )
    thread = threading.Thread(target=server.serve, args=(tmp_path / "ready.json",))
    thread.start()
    assert ready.wait(2)
    client = UnixDraftClient(socket_path, timeout_seconds=2)
    try:
        proposal = initialize(client)
        assert client.call("eager_enqueue", {"requests": [verify_row(proposal)]})["enqueued"]
        deadline = time.monotonic_ns() + 2_000_000_000
        assert client.call("diagnostic_settle", settle_row(deadline=deadline))["released"]
        with pytest.raises(RuntimeError, match="original absolute deadline"):
            client.call("status", {"deadline_ns": deadline + 1})
        assert client.call("shutdown", {"deadline_ns": deadline})["shutdown"]
    finally:
        if server.running:
            client.call("shutdown", {"deadline_ns": server.deadline_ns})
        thread.join(timeout=2)
    assert not thread.is_alive() and not socket_path.exists()
    ready_state = json.loads((tmp_path / "ready.json").read_text())
    assert ready_state["rolling_eager_qualification"] == "PENDING"
    events = CheckpointJsonl(tmp_path / "events.jsonl").read()
    assert {row["operation"] for row in events} >= {
        "eager_enqueue", "diagnostic_settle", "shutdown"
    }
