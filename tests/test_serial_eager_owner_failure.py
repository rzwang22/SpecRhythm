"""Abandoned caller responses cannot block physical owner failure retirement."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_rolling_eager_gpu_backend import greedy_tokens
from test_serial_eager_owner import (
    OwnerWorker,
    initialize,
    proposal_row,
    settle_row,
    sync_row,
    verify_row,
)
from test_serial_eager_owner import owner_factory as _owner_factory

from specrhythm.serving.eager_owner import EagerOwner
from specrhythm.serving.eager_results import summarize_eager

owner_factory = _owner_factory


def test_abandoned_waiter_device_error_does_not_block_backend_shutdown():
    entered = threading.Event()
    release = threading.Event()
    cleaned = threading.Event()

    class Backend:
        backend_name = "cpu-failure-review"
        provenance = {}
        failed = False

        def _fail(self):
            self.failed = True

        def shutdown(self):
            cleaned.set()

    class Machine:
        def __init__(self):
            self.backend = Backend()
            self.works = {}
            self.owner_stopped = False

        def prepare_synchronizations(self, rows):
            pass

        def finish_synchronizations(self, rows):
            entered.set()
            assert release.wait(2), "test did not release the controlled failing device operation"
            self.backend.failed = True
            raise RuntimeError("controlled device failure after caller timeout")

    owner = EagerOwner(Machine, timeout_seconds=1)
    responses = []
    original_put = owner.commands.put

    def capture(command):
        responses.append(command[2])
        original_put(command)

    owner.commands.put = capture
    try:
        with pytest.raises(TimeoutError, match="physical work retained"):
            owner.call("synchronize_and_batch_propose", {
                "synchronizations": [], "proposals": [],
                "deadline_ns": time.monotonic_ns() + 20_000_000,
            })
        assert entered.is_set()
        release.set()
        owner._thread.join(timeout=1)
        assert not owner._thread.is_alive(), "owner deadlocked notifying an abandoned caller twice"
        assert cleaned.is_set(), "device cleanup did not run after the failure"
        assert owner.machine.owner_stopped
        assert str(owner.failure) == "controlled device failure after caller timeout"
        assert responses[0].qsize() == 1
        ok, failure = responses[0].get_nowait()
        assert not ok and failure is owner.failure
    finally:
        release.set()
        # Preserve test-process cleanup even if a regression reintroduces the bug.
        # The assertions above already require autonomous exit before this fallback.
        if owner._thread.is_alive() and responses and not responses[0].empty():
            responses[0].get_nowait()
        owner._thread.join(timeout=2)


def test_waiting_full_accept_then_switch_off_recovers_at_actual_abort_boundary(owner_factory):
    worker = OwnerWorker()
    worker.block_eager = True
    owner = owner_factory(worker)
    proposal = initialize(owner)
    owner.call("eager_enqueue", {"requests": [verify_row(proposal)]})
    assert worker.entered.wait(2)
    work_id = owner.machine.works["a"].work_id
    sync, final = sync_row(proposal, (10, 20))
    feedback_queued, switch_queued = threading.Event(), threading.Event()
    original_put = owner.commands.put

    def observed_put(command):
        original_put(command)
        if command[0] == "synchronize_and_batch_propose":
            feedback_queued.set()
        elif command[0] == "eager_switch":
            switch_queued.set()

    owner.commands.put = observed_put
    with ThreadPoolExecutor(max_workers=2) as pool:
        feedback = pool.submit(owner.call, "synchronize_and_batch_propose", {
            "synchronizations": [sync],
            "proposals": [proposal_row(prefix=final, round_id=1, remaining=95)],
        })
        assert feedback_queued.wait(2)
        switch = pool.submit(owner.call, "eager_switch", {"enabled": False, "decision_version": 1})
        assert switch_queued.wait(2)
        # Both messages arrive during an actual in-flight worker write. FIFO
        # processes the full-accept feedback, then the switch, before token two.
        worker.resume.set()
        assert switch.result(timeout=2) == {"enabled": False, "decision_version": 1}
        recovered = feedback.result(timeout=2)["proposals"][0]
    assert owner.failure is None
    assert tuple(recovered["proposal_token_ids"]) == greedy_tokens(final, 4)
    assert recovered["runtime_provenance"].get("source_continuation_id") is None
    assert owner.machine.counters["committed_tokens"] == 5
    assert owner.machine.counters["promotions"] == 0
    assert owner.machine.counters["completed"] == 0
    assert owner.machine.counters["recovery_jobs"] == 1
    assert owner.machine.counters["recovery_generated_tokens"] == 4
    assert owner.machine.counters["discarded_early_tokens"] == 1
    assert sum(purpose == "eager" for purpose, _ in worker.calls) == 1
    waits = [event for event in owner.machine.events if event["phase"] == "unhidden_wait"]
    assert len(waits) == 1
    wait = waits[0]
    assert wait["end_ns"] == owner.machine.work_times[work_id][1]
    assert wait["end_ns"] > wait["start_ns"]
    assert wait["counter_delta"]["unhidden_wait_ns"] == wait["end_ns"] - wait["start_ns"]
    deadline = time.monotonic_ns() + 2_000_000_000
    row = settle_row(prefix=final, round_id=1, deadline=deadline)
    settled = owner.call("diagnostic_settle", row)
    assert settled["released"] and settled["new_proposals_generated"] == 0
    assert owner.call("shutdown", {"deadline_ns": deadline})["shutdown"]
    assert not owner._thread.is_alive() and not worker.memory
    events = owner.machine.events
    summary = summarize_eager({"rolling_eager": owner.machine.eager_report()}, {
        "measurement_start_ns": min(event["start_ns"] for event in events),
        "measurement_end_ns": max(event["end_ns"] for event in events),
        "requests": [{"commits": [{"token_ids": sync["committed_delta"]}]}],
    })
    assert summary["window_unhidden_wait_ns"] == summary["lifetime_counters"]["unhidden_wait_ns"]
    assert summary["cleanup_status"] == "PASS"
    assert summary["pending_work"] == [] and summary["owner_stopped"]
