"""Token-boundary event coordination; no wall-time speed or CPU/GPU overlap claim."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_k3 import machine
from test_prepost_protocol import feedback
from test_serial_eager_owner import OwnerWorker, settle_row, verify_row

from specrhythm.serving.k3_owner import K3Owner
from specrhythm.serving.ping_prepost_owner import PingPrePostOwner


@pytest.mark.parametrize("owner_type", [PingPrePostOwner, K3Owner])
def test_real_owner_status_queue_blocks_old_but_k3_snapshot_does_not(owner_type):
    entered, resume, queued = threading.Event(), threading.Event(), threading.Event()

    class Worker(OwnerWorker):
        armed = False

        def materialize(self, rows, purpose):
            if self.armed and purpose == "prepost_post":
                entered.set()
                assert resume.wait(3)
            return super().materialize(rows, purpose)

    worker = Worker()
    owner = owner_type(lambda: machine(False, worker=worker), timeout_seconds=3)
    put = owner.commands.put

    def observe(command, **kwargs):
        put(command, **kwargs)
        if command[0] == "status":
            queued.set()

    owner.commands.put = observe
    try:
        p = owner.call(
            "pp_admit",
            dict(opportunity=0, normal_cohort="A", capacity=1, active_request_ids=["a", "b"]),
        )["claims"][0]["proposal"]
        owner.call("eager_enqueue", {"requests": [verify_row(p)]})
        row, _ = feedback(p, (10, 20), reject=True)
        worker.armed = True
        owner.call("pp_feedback", {"synchronizations": [row]})
        assert entered.wait(3)
        with ThreadPoolExecutor(1) as executor:
            status = executor.submit(owner.call, "status", {})
            if owner_type is K3Owner:
                assert "b" in status.result(timeout=3)["ready_request_ids"]
                assert not queued.is_set() and not resume.is_set()
            else:
                assert queued.wait(3)
                assert not status.done()  # Exact same production call waits for active write.
            resume.set()
            status.result(timeout=3)
    finally:
        resume.set()
        close(owner)


def close(owner):
    owner.call("pp_stop", {})
    for rid, s in list(owner.machine.requests.items()):
        owner.call(
            "diagnostic_settle",
            settle_row(rid, s.committed_token_ids, s.next_round_id, terminal=s.finished),
        )
    owner.call("shutdown", {"deadline_ns": time.monotonic_ns() + 3_000_000_000})
    assert not owner._thread.is_alive() and not owner.machine.backend.worker.memory


@pytest.mark.parametrize("eager", [False, True])
def test_ready_b_claim_precedes_a_complete_recovery_and_a_progresses_during_b(eager):
    entered, resume, second, resume_second = [threading.Event() for _ in range(4)]
    claimed, ready_a = threading.Event(), threading.Event()

    class Worker(OwnerWorker):
        armed = False

        def materialize(self, rows, purpose):
            if self.armed and purpose == "prepost_post":
                entered.set()
                assert resume.wait(3)
            elif self.armed and purpose == "prepost_extension" and not second.is_set():
                second.set()
                assert resume_second.wait(3)
            return super().materialize(rows, purpose)

    worker = Worker()

    def factory():
        m = machine(eager, worker=worker)
        ping = m._ping

        def observe(event, **values):
            ping(event, **values)
            if event == "ready" and values["request_id"] == "a" and values["prefix_version"] == 1:
                ready_a.set()

        m._ping = observe
        return m

    owner = K3Owner(factory, timeout_seconds=3)
    put = owner.commands.put

    def enqueued(command, **kwargs):
        put(command, **kwargs)
        if command[0] == "pp_admit" and command[1]["opportunity"] == 1:
            claimed.set()

    owner.commands.put = enqueued
    try:
        a = owner.call(
            "pp_admit",
            dict(opportunity=0, normal_cohort="A", capacity=1, active_request_ids=["a", "b"]),
        )["claims"][0]["proposal"]
        owner.call("eager_enqueue", {"requests": [verify_row(a)]})
        row, _ = feedback(a, (10, 20), reject=True)
        worker.armed = True
        owner.call("pp_feedback", {"synchronizations": [row]})
        assert entered.wait(3)
        with ThreadPoolExecutor(1) as executor:
            result = executor.submit(
                owner.call,
                "pp_admit",
                dict(opportunity=1, normal_cohort="B", capacity=1, active_request_ids=["a", "b"]),
            )
            assert claimed.wait(3)
            resume.set()
            b = result.result(timeout=3)["claims"][0]["proposal"]
        assert b["request_id"] == "b"
        assert second.wait(3) and not ready_a.is_set()
        # Target B may enter now. Enqueue does not wait for A's active extension.
        owner.call("eager_enqueue", {"requests": [verify_row(b)]})
        resume_second.set()
        assert ready_a.wait(3)
        assert ("b", 0) not in owner.machine.sync_bindings  # B feedback has not arrived.
        assert len(owner.machine.ready["a"]["proposal"]["proposal_token_ids"]) == 3
        row, _ = feedback(b, (10, 20))
        owner.call("pp_feedback", {"synchronizations": [row]})
    finally:
        resume.set()
        resume_second.set()
        close(owner)
