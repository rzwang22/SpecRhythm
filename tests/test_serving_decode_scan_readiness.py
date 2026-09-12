"""Real fixed/S2/Dual selection and FIFO claim contract, CPU stock allocation only."""

import threading

import pytest
from test_phase4_dual_scheduler import proposal_result
from test_serving_fixed_identity import fixed_schedulers as _fixed
from test_serving_fixed_identity import s2_schedulers as _s2

from specrhythm.phase4.dual_service import AsyncDualDraftController
from specrhythm.serving.common import DataError
from specrhythm.serving.decode_scan_readiness import ReadinessEvidence, ScanBatchWait, coalesce
from specrhythm.serving.decode_scan_window import ScanShapeStop
from specrhythm.serving.s2_pool import publish

s2_schedulers = _s2
fixed_schedulers = _fixed


class ReadyTransport:
    """Supply owner results through the actual locked FIFO/claim implementation."""

    def __init__(self, results, pending=()):
        self.owner = object.__new__(AsyncDualDraftController)
        self.owner._lock = threading.Lock()
        self.owner._ready = {r["request_id"]: r for r in results}
        self.owner._ready_order = list(self.owner._ready)
        self.owner._claimed = {}
        self.owner._inflight = set(pending)
        self.owner._failures = {}
        self.limits = []

    def call(self, command, payload):
        assert command == "poll_ready"
        self.limits.append(payload["limit"])
        return self.owner.poll_ready(payload["limit"])


def test_other_cohort_ready_cannot_starve_selected_cohort(fixed_schedulers):
    build, path = fixed_schedulers
    # Legal stable prefixes: 31 B proposals already installed, A's 32 published
    # after real Draft work. This reproduces the code defect, NOT a claim that
    # the retained GPU failure had this exact distribution (its small pack lacks it).
    s, packet = build("pingpong", "bound-prefix", batch=64, n=360,
                      ready_ids=range(32, 63))
    packet["decode_scan_full_batch"] = 32
    publish(path, packet)
    transport = ReadyTransport([
        proposal_result(str(i), prefix=tuple(s.requests[str(i)].all_token_ids),
                        tokens=(11, 12, 13, 14)) for i in range(32)
    ], pending=["63"])
    s._dual_client = transport
    output = s.schedule()
    assert list(output.num_scheduled_tokens) == [str(i) for i in range(32)]
    assert len(transport.owner._claimed) == 32
    assert not transport.owner._ready_order
    assert len(s._dual_consumed_proposals) == 32
    assert all(r["base_root_positions"] == 1 and r["candidate_positions"] == 4
               for r in s.s2_steps[-1]["rows"])


def test_wait_does_not_allocate_and_repoll_reconsiders_cohort(fixed_schedulers):
    build, path = fixed_schedulers
    s, packet = build("pingpong", "bound-prefix", ready_ids=range(1, 32))
    packet["decode_scan_full_batch"] = 32
    publish(path, packet)
    s._dual_client = ReadyTransport([], pending=["0", *map(str, range(32, 64))])
    for _ in range(3):
        with pytest.raises(ScanBatchWait) as caught:
            s.schedule()
        assert caught.value.evidence["reason"] == "draft_inflight"
        assert s.current_step == 0 and not s.s2_steps and not s._dual_consumed_proposals
    # A becomes ready while the whole B cohort remains in flight: no global wait.
    s._dual_client = ReadyTransport([
        proposal_result("0", prefix=tuple(s.requests["0"].all_token_ids),
                        tokens=(11, 12, 13, 14))], pending=map(str, range(32, 64)))
    out = s.schedule()
    assert len(out.num_scheduled_tokens) == 32 and s.current_step == 1
    assert len(s._dual_consumed_proposals) == 32
    assert s.scan_readiness["cohorts"]["B"]["drafting"] == 32


def test_scan_deadline_after_poll_prevents_stock_dispatch(fixed_schedulers):
    build, path = fixed_schedulers
    s, packet = build("pingpong", "bound-prefix")
    packet.update(decode_scan_full_batch=32, decode_scan_deadline_ns=1)
    publish(path, packet)
    with pytest.raises(ScanBatchWait) as caught:
        s.schedule()
    assert caught.value.evidence["reason"] == "time_budget"
    assert s.current_step == 0 and not s._dual_consumed_proposals


def test_full_idle_cohort_missing_proposal_is_material_error(fixed_schedulers):
    build, path = fixed_schedulers
    s, packet = build("pingpong", "bound-prefix", ready_ids=range(1, 32))
    packet["decode_scan_full_batch"] = 32
    publish(path, packet)
    s._dual_client = ReadyTransport([])
    with pytest.raises(DataError, match="full idle cohort lacks verifiable proposals"):
        s.schedule()
    assert s.current_step == 0 and not s._dual_consumed_proposals


def test_unexpected_stock_partial_still_trips_final_guard(fixed_schedulers, monkeypatch):
    from test_phase4_dual_scheduler import StockSchedulerStub

    build, path = fixed_schedulers
    s, packet = build("pingpong", "bound-prefix")
    packet["decode_scan_full_batch"] = 32
    publish(path, packet)
    stock = StockSchedulerStub.schedule

    def insufficient_allocation(self):
        out = stock(self)
        out.num_scheduled_tokens.pop("0")
        out.scheduled_spec_decode_tokens.pop("0")
        return out

    monkeypatch.setattr(StockSchedulerStub, "schedule", insufficient_allocation)
    with pytest.raises(ScanShapeStop) as caught:
        s.schedule()
    assert caught.value.evidence["scheduled_batch"] == 31
    assert s.current_step == 1  # This exceptional path is drained, never retried.


def test_wait_compaction_bounds_history_without_poll_limit_or_time_loss():
    evidence = ReadinessEvidence()
    for i in range(20000):
        evidence.waiting({"timestamp_ns": i + 10, "reason": "draft_inflight"})
    assert len(evidence.rows) == 2 and evidence.poll_count == 20000
    evidence.close(30010, "wait-stop:time_budget")
    assert evidence.report()["closed_wait_ns"] == 30000
    assert evidence.report()["history_complete"]
    rows = []
    for i in range(20000):
        coalesce(rows, {"timestamp_ns": i, "request_id": str(i)})
    assert len(rows) == 512 and rows[-1]["omitted_transitions"] == 20000 - 512
    assert sum(r["observations"] for r in rows) == 20000
