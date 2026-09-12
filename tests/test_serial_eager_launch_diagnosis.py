"""Characterize the existing late-launch path without optimizing its behavior.

These are deterministic interleavings, not latency or GPU-overlap measurements.
The real owner, continuation machine, S2 pool audit and physical worker adapter
execute; test latches control when authoritative feedback reaches the mailbox.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from test_serial_eager_owner import proposal_row, settle_row, sync_row, verify_row
from test_serving_s2_runtime import PoolWorker

from specrhythm.continuation.gpu_backend import GPUContinuationBackendMixin
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.serving import s2_draft
from specrhythm.serving.eager_machine import EagerSerialMachine
from specrhythm.serving.eager_owner import EagerOwner
from specrhythm.serving.s2_pool import publish


class ObservedPoolWorker(PoolWorker):
    def __init__(self):
        super().__init__()
        self.hold_first_eager = False
        self.first_eager_entered = threading.Event()
        self.allow_first_eager = threading.Event()

    def materialize(self, rows, purpose):
        if purpose == "eager" and self.hold_first_eager:
            self.hold_first_eager = False
            self.first_eager_entered.set()
            if not self.allow_first_eager.wait(3):
                raise TimeoutError("diagnosis test did not release first-forward latch")
        return super().materialize(rows, purpose)


class ObservedS2Backend(GPUContinuationBackendMixin, s2_draft.S2DraftBackend):
    def __init__(self, *args, **kwargs):
        self.audit_calls = 0
        self.prefix_visits = 0
        self.hold_next_audit = False
        self.audit_entered = threading.Event()
        self.allow_audit = threading.Event()
        super().__init__(*args, **kwargs)

    def _audit(self):
        self.audit_calls += 1
        result = super()._audit()
        if self.hold_next_audit:
            self.hold_next_audit = False
            self.audit_entered.set()
            if not self.allow_audit.wait(3):
                raise TimeoutError("diagnosis test did not release real-audit latch")
        return result

    def physical_rows(self):
        self.prefix_visits += len(self.states)
        return super().physical_rows()


@pytest.fixture
def resident_owner(tmp_path, monkeypatch):
    pool_ids = tuple(f"request-{index:03d}" for index in range(360))
    active_ids = pool_ids[:16]
    prefixes = {rid: (10, index + 20) for index, rid in enumerate(pool_ids)}
    states = {rid: {"state": "STAGED", "cohort": None} for rid in pool_ids}
    control_path = tmp_path / "control.json"
    monkeypatch.setenv("SR_S2_CONTROL", str(control_path))
    monkeypatch.setenv("SR_S2_RUN_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("SR_S2_MODE", "serial-eager")
    monkeypatch.setattr(s2_draft, "cuda_memory", lambda _: {"free_memory_bytes": 10**9})
    publish(control_path, {"barrier_ns": None, "requests": states})
    worker = ObservedPoolWorker()

    def factory():
        backend = ObservedS2Backend(SimpleNamespace(max_model_len=4096), worker=worker)
        machine = EagerSerialMachine(backend, request_ids=pool_ids)
        for rid in active_ids:
            machine.initialize(rid, prefixes[rid], token_prefix_hash(prefixes[rid]))
        # Untouched standby KV uses the same actual persistent allocator; its
        # independent logical initialization is irrelevant to an active admission.
        backend.initialize_many(tuple((rid, prefixes[rid]) for rid in pool_ids[16:]))
        return machine

    owner = EagerOwner(factory, timeout_seconds=3)
    for rid in active_ids:
        states[rid]["state"] = "ACTIVE"
    publish(control_path, {"barrier_ns": 1, "requests": states})
    proposals = owner.call("batch_propose", {
        "requests": [proposal_row(rid, prefixes[rid]) for rid in active_ids]
    })["proposals"]
    try:
        yield SimpleNamespace(owner=owner, backend=owner.machine.backend, worker=worker,
                              proposals=proposals, prefixes=prefixes, active_ids=active_ids)
    finally:
        owner.machine.backend.allow_audit.set()
        worker.allow_first_eager.set()
        if owner.failure is None and not owner.closed:
            owner.call("status", {})
            for rid, state in tuple(owner.machine.requests.items()):
                if rid not in owner.machine.backend.retired:
                    owner.call("diagnostic_settle", settle_row(
                        rid, state.committed_token_ids, state.next_round_id,
                        terminal=state.finished,
                    ))
            owner.call("shutdown", {"deadline_ns": time.monotonic_ns() + 3_000_000_000})
        owner._thread.join(timeout=3)
        assert not owner._thread.is_alive()


def test_b16_admission_visits_the_360_request_pool_seventeen_times_before_first_forward(
    resident_owner,
):
    case = resident_owner
    before_audits = case.backend.audit_calls
    before_visits = case.backend.prefix_visits
    case.worker.hold_first_eager = True
    receipt = case.owner.call("eager_enqueue", {
        "requests": [verify_row(proposal) for proposal in case.proposals]
    })
    assert receipt["enqueued"]
    assert case.worker.first_eager_entered.wait(2)
    assert case.backend.audit_calls - before_audits == 16 + 1
    assert case.backend.prefix_visits - before_visits == (16 + 1) * 360
    assert case.owner.machine.counters["admissions"] == 16
    assert case.owner.machine.counters["started"] == 0
    assert not [rows for purpose, rows in case.worker.calls if purpose == "eager"]
    case.worker.allow_first_eager.set()


@pytest.mark.parametrize("accepted_requests", [0, 5, 16])
def test_feedback_queued_during_admission_starts_only_fully_accepted_parents_after_feedback(
    resident_owner, accepted_requests, monkeypatch,
):
    case = resident_owner
    case.backend.hold_next_audit = True
    receipt = case.owner.call("eager_enqueue", {
        "requests": [verify_row(proposal) for proposal in case.proposals]
    })
    assert receipt["enqueued"]
    assert case.backend.audit_entered.wait(2)
    assert not [rows for purpose, rows in case.worker.calls if purpose == "eager"]
    synchronizations = [sync_row(
        proposal, case.prefixes[proposal["request_id"]], reject=index >= accepted_requests
    )[0] for index, proposal in enumerate(case.proposals)]
    feedback_queued = threading.Event()
    original_put = case.owner.commands.put

    def observed_put(command, *args, **kwargs):
        original_put(command, *args, **kwargs)
        if command[0] == "synchronize_and_batch_propose":
            feedback_queued.set()

    monkeypatch.setattr(case.owner.commands, "put", observed_put)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(case.owner.call, "synchronize_and_batch_propose", {
            "synchronizations": synchronizations, "proposals": [],
        })
        assert feedback_queued.wait(2)
        feedback_queued_ns = time.monotonic_ns()
        case.backend.allow_audit.set()
        assert len(result.result(timeout=2)["synchronizations"]) == 16
    counters = case.owner.machine.counters
    assert counters["admissions"] == 16
    assert counters["parent_full_accepts"] == accepted_requests
    assert counters["started"] == counters["completed"] == accepted_requests
    assert counters["parent_rejections"] == 16 - accepted_requests
    assert counters["early_generated_tokens"] == accepted_requests * 5
    events = case.owner.machine.events
    feedback_end = max(event["end_ns"] for event in events if event["phase"] == "parent_result")
    starts = [event["start_ns"] for event in events if event["phase"] == "eager_start"]
    assert all(timestamp >= feedback_end >= feedback_queued_ns for timestamp in starts)
    actual_rows = [rows for purpose, rows in case.worker.calls if purpose == "eager"]
    assert len(actual_rows) == (5 if accepted_requests else 0)
    assert all(len(rows) == accepted_requests for rows in actual_rows)
