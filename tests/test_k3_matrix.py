"""Four geometries through the owner queue and actual resident Target scheduler."""

from types import SimpleNamespace as NS

import pytest
from test_k3 import Backend
from test_k3_pipeline import close
from test_k3_prompt_proof import fixed_schedulers as _fixed
from test_k3_prompt_proof import s2_schedulers as _s2
from test_k3_prompt_proof import target_pool as _pool
from test_ping_prepost import run_work
from test_serial_eager_owner import OwnerWorker, proposal_row

from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.serving.k3 import MODES, geometry
from specrhythm.serving.k3_machine import K3Machine
from specrhythm.serving.k3_owner import K3Owner
from specrhythm.serving.k3_window import K3Window
from specrhythm.serving.ping_prepost_controller import PingPrePostController
from specrhythm.serving.s2_pool import publish

fixed_schedulers, s2_schedulers, target_pool = _fixed, _s2, _pool


@pytest.mark.parametrize("mode", MODES)
def test_real_owner_claim_single_target_schedule_geometry(target_pool, monkeypatch, mode):
    s, packet, path = target_pool
    g = geometry(mode)
    monkeypatch.setenv("SR_S2_MODE", mode)
    ids = list(s.requests)[:16]
    packet["max_requests_per_target_forward"] = g["target_request_ceiling"]
    for i, (rid, r) in enumerate(s.requests.items()):
        r.sampling_params = NS(max_tokens=100)
        packet["requests"][rid]["cohort"] = "A" if g["serial_idle_gate"] or i < 8 else "B"
    publish(path, packet)
    s.freeze_pool()

    def factory():
        m = K3Machine(Backend(NS(max_model_len=4096), worker=OwnerWorker()),
                      request_ids=ids, eager=g["eager"], mode=mode)
        rows = []
        for rid in ids:
            prefix = tuple(s.requests[rid].all_token_ids)
            m.initialize(rid, prefix, token_prefix_hash(prefix))
            rows.append(dict(proposal_row(rid, prefix),
                             home_cohort=packet["requests"][rid]["cohort"]))
        m.register(rows)
        run_work(m)
        return m

    owner = K3Owner(factory, timeout_seconds=3)
    try:
        packet["pp_admission"] = PingPrePostController(mode).select(
            NS(rows={r: packet["requests"][r] for r in ids}), NS(call=owner.call))
        publish(path, packet)
        audits = s.s2_pool.checks
        result = s.schedule()  # One stock scheduler output, consumed by one Target forward.
        assert len(result.num_scheduled_tokens) == g["target_request_ceiling"]
        assert set(map(len, result.scheduled_spec_decode_tokens.values())) == {3}
        assert set(result.num_scheduled_tokens.values()) == {4}  # root + K3
        assert s.s2_pool.checks == audits + 2
        records = owner.machine.backend.prepost_forwards.rows()
        assert len(records) == 2 and {r["B"] for r in records} == {16}
        assert not owner.machine.works  # no conditional continuation before verify_start
    finally:
        close(owner)


@pytest.mark.parametrize("mode", MODES)
def test_warmup_matches_request_opportunities_not_target_steps(mode):
    g = geometry(mode)
    w = K3Window(dict(warmup_steps=2, window_seconds=30), 16, mode)
    n = g["target_request_ceiling"]
    for i in range(32 // n):
        ids = (
            [str(j) for j in range((i % 2) * 8, (i % 2) * 8 + 8)]
            if n == 8
            else [str(j) for j in range(16)]
        )
        w.step_completed(dict(B=n, request_ids=ids, cohort="A" if i % 2 == 0 else "B",
                              start_ns=10*i+1, committed_tokens=n, rows=[]), 10*i+2)
    boundary = w.warmup_boundary()
    assert boundary["request_opportunities"] == 32
    assert boundary["completed_steps"] == (2 if n == 16 else 4)
    assert set(boundary["opportunities_by_request"].values()) == {2}
