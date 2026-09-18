"""Lossless production control/RPC, scoped readers and feedback contracts."""

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS

import pytest
from test_k3 import machine
from test_k3_pipeline import close
from test_k3_prompt_proof import fixed_schedulers as _fixed
from test_k3_prompt_proof import s2_schedulers as _s2
from test_k3_prompt_proof import target_pool as _pool
from test_k3_resident_schedule import prepare

from specrhythm.serving import s2_pool
from specrhythm.serving.dual_batch import (
    CommandPublisher,
    control_scope,
    pack,
    run_target,
    unpack,
)
from specrhythm.serving.k3_owner import K3Owner
from specrhythm.serving.ping_prepost_controller import PingPrePostController

fixed_schedulers, s2_schedulers, target_pool = _fixed, _s2, _pool


@pytest.fixture
def dual(monkeypatch):
    monkeypatch.setenv("SR_K3_TARGET_DISPATCH", "dual-batch")
    monkeypatch.setenv("SR_S2_MODE", "pingpong-k3")


def test_real_owner_wire_roundtrip_and_duplicate_claim(dual):
    owner = K3Owner(lambda: machine(False), timeout_seconds=3)
    try:
        calls = []

        def rpc(operation, payload):
            calls.append(operation)
            # Actual wire boundary, not an in-process dictionary shortcut.
            return json.loads(json.dumps(owner.call(operation, json.loads(json.dumps(payload)))))

        controller = PingPrePostController("pingpong-k3")
        clock = NS(rows={r: {"state": "ACTIVE"} for r in ("a", "b")})
        admitted = controller.select(clock, NS(call=rpc))
        assert calls == ["pp_admit_command"]
        assert len(admitted["claims"]) == 1
        claim = admitted["claims"][0]
        assert claim == owner.machine.claims[claim["request_id"]]
        assert unpack(pack(admitted)) == admitted
        with pytest.raises(RuntimeError, match="stale/duplicate"):
            rpc("pp_admit_command", dict(opportunity=0, capacity=8,
                                        normal_cohort="A", active_request_ids=["a", "b"]))
        other = controller.select(clock, NS(call=rpc))
        assert other["claims"][0]["request_id"] != claim["request_id"]
        assert controller.select(clock, NS(call=rpc))["claims"] == []
        assert not owner.machine.works  # Ordinary means no conditional continuation.
    finally:
        close(owner)


def test_real_scheduler_uses_one_current_read_and_two_live_audits(target_pool, dual, monkeypatch):
    scheduler, packet, path = target_pool
    prepare(scheduler, packet, path)
    original, reads = s2_pool.read_json, []

    def read(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(s2_pool, "read_json", read)
    checks = scheduler.s2_pool.checks
    output = run_target(NS(step=scheduler.schedule))
    assert len(output.num_scheduled_tokens) == 8
    assert len(reads) == 1  # Reference's nested scheduler reads this twice.
    assert scheduler.s2_pool.checks == checks + 2
    assert len(scheduler._resident_events.read()) >= 360
    assert "dual_batch_command" in original(path)
    assert s2_pool.control()["pp_admission"] == packet["pp_admission"]
    assert len(reads) == 2  # No cross-call cache.
    with pytest.raises(ValueError, match="duplicate Target claim"):
        run_target(NS(step=scheduler.schedule))


def test_scoped_control_thread_exception_invalidation_and_live_updates(
    tmp_path, dual, monkeypatch
):
    path = tmp_path / "s2-control.json"
    monkeypatch.setenv("SR_S2_CONTROL", str(path))
    s2_pool.publish(path, {"version": 1})
    with pytest.raises(RuntimeError, match="worker"):
        with control_scope():
            s2_pool.publish(path, {"version": 2})
            with control_scope():
                assert s2_pool.control() == {"version": 1}
            with ThreadPoolExecutor(1) as pool:
                assert pool.submit(s2_pool.control).result() == {"version": 2}
            raise RuntimeError("worker")
    assert s2_pool.control() == {"version": 2}


def test_content_interning_preserves_all_fields_and_rejects_corruption():
    a = dict(claims=[dict(request_id=str(i), proposal=dict(
        model_provenance={"large": list(range(400)), "device": i % 2},
        proposal_token_ids=[1, 2, i], runtime_provenance={"prefix_version": i}))
        for i in range(64)], opportunity_cohort="A")
    before = copy.deepcopy(a)
    compact = pack(a)
    assert len(compact["models"]) == 2 and unpack(json.loads(json.dumps(compact))) == a == before
    assert len(json.dumps(compact)) < len(json.dumps(a)) // 2
    compact["admission"]["claims"][0]["proposal"]["model_reference"] = True
    with pytest.raises(ValueError, match="model reference"):
        unpack(compact)


def test_publisher_owns_lifecycle_and_does_not_advance_after_failed_publish(
    tmp_path, dual, monkeypatch
):
    path = tmp_path / "s2-control.json"
    monkeypatch.setenv("SR_S2_CONTROL", str(path))
    publisher = CommandPublisher(path)
    calls, original = [], s2_pool.publish

    def publish(path, value):
        calls.append(value)
        original(path, value)

    monkeypatch.setattr(s2_pool, "publish", publish)
    packet = dict(initial_proposals={}, initial_enqueues={}, requests={"r": {"state": "ACTIVE"}})
    publisher.publish(copy.deepcopy(packet))
    publisher.publish(copy.deepcopy(packet))
    assert len(calls) == 1
    packet["requests"]["r"]["state"] = "FINISHED"
    monkeypatch.setattr(s2_pool, "publish", lambda *a: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError, match="disk"):
        publisher.publish(copy.deepcopy(packet))
    assert s2_pool.control()["requests"]["r"]["state"] == "ACTIVE"
    monkeypatch.setattr(s2_pool, "publish", publish)
    publisher.publish(copy.deepcopy(packet))
    assert s2_pool.control()["requests"]["r"]["state"] == "FINISHED"


def test_compact_owner_rejects_eager_path(dual):
    owner = K3Owner(lambda: machine(True), timeout_seconds=3)
    try:
        with pytest.raises(RuntimeError, match="ordinary"):
            owner.call("pp_admit_command", {})
    finally:
        close(owner)
