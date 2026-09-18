"""Deterministic mailbox/physical-write ordering; CPU concurrency is not GPU overlap."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_ping_prepost import machine
from test_prepost_protocol import feedback
from test_serial_eager_owner import OwnerWorker, settle_row, verify_row

from specrhythm.serving.ping_prepost_owner import PingPrePostOwner


@pytest.mark.parametrize("reject", [False, True])
def test_queued_feedback_precedes_next_physical_token_and_other_home_can_claim(reject):
    class Controlled(OwnerWorker):
        def materialize(self, rows, purpose):
            if purpose == "prepost_lookahead" and not self.entered.is_set():
                self.entered.set()
                assert self.resume.wait(3), "test did not release physical write"
            return super().materialize(rows, purpose)

    worker = Controlled()
    owner = PingPrePostOwner(lambda: machine(worker=worker), timeout_seconds=3)
    queued = threading.Event()
    put = owner.commands.put

    def observed(command, **kwargs):
        put(command, **kwargs)
        if command[0] == "pp_feedback":
            queued.set()

    owner.commands.put = observed
    try:
        a = owner.call(
            "pp_admit",
            dict(opportunity=0, normal_cohort="A", capacity=1, active_request_ids=["a", "b"]),
        )["claims"][0]
        owner.call("eager_enqueue", {"requests": [verify_row(a["proposal"])]})
        assert worker.entered.wait(3)
        row, final = feedback(a["proposal"], (10, 20), reject=reject)
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(owner.call, "pp_feedback", {"synchronizations": [row]})
            assert queued.wait(3)
            worker.resume.set()
            ack = result.result(timeout=3)
            assert ack["proposals"] == []
        b = owner.call(
            "pp_admit",
            dict(opportunity=1, normal_cohort="B", capacity=1, active_request_ids=["b"]),
        )["claims"]
        assert b[0]["request_id"] == "b"
        owner.call("pp_stop", {})
        owner.call("status", {})
        # Inspect the completed physical records, not wall-time thresholds.
        records = owner.machine.backend.prepost_forwards.rows()
        look = [r for r in records if r["purpose"] == "prepost_lookahead"]
        if reject:
            assert len(look) == 1  # feedback already queued invalidates remaining two.
        assert owner.machine.requests["a"].committed_token_ids == final
    finally:
        worker.resume.set()
        owner.call("pp_stop", {})
        for rid, state in owner.machine.requests.items():
            owner.call(
                "diagnostic_settle",
                settle_row(
                    rid, state.committed_token_ids, state.next_round_id, terminal=state.finished
                ),
            )
        owner.call("shutdown", {"deadline_ns": time.monotonic_ns() + 3_000_000_000})
        assert not owner._thread.is_alive() and not worker.memory
