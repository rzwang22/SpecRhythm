"""B64 production owner queue -> controller claim -> actual resident scheduler."""

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
from specrhythm.serving.k3 import B64, B128, MODES, geometry
from specrhythm.serving.k3_machine import K3Machine
from specrhythm.serving.k3_owner import K3Owner
from specrhythm.serving.ping_prepost_controller import PingPrePostController
from specrhythm.serving.s2_pool import publish

fixed_schedulers, s2_schedulers, target_pool = _fixed, _s2, _pool


@pytest.mark.parametrize("configuration", [B64, B128])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("draft_dispatch", [None, "legacy", "unified"])
def test_B64_owner_target_no_hidden_B16_limit(
    target_pool, monkeypatch, mode, draft_dispatch, configuration
):
    s, packet, path = target_pool
    g = geometry(mode, configuration)
    active = g["active_limit"]
    monkeypatch.setenv("SR_S2_MODE", mode)
    ids = list(s.requests)[:active]
    packet.update(
        k3_configuration=configuration,
        active_limit=active,
        max_requests_per_target_forward=g["target_request_ceiling"],
    )
    for i, (rid, r) in enumerate(s.requests.items()):
        r.sampling_params = NS(max_tokens=100)
        packet["requests"][rid]["cohort"] = (
            "A" if g["serial_idle_gate"] or i < active // 2 else "B"
        )
        packet["requests"][rid]["state"] = "ACTIVE" if rid in ids else "STAGED"
    publish(path, packet)
    s.freeze_pool()

    def factory():
        m = K3Machine(
            Backend(NS(max_model_len=4096), worker=OwnerWorker()),
            request_ids=ids,
            mode=mode,
            eager=g["eager"],
            configuration=configuration,
            draft_dispatch=draft_dispatch,
        )
        rows = []
        for rid in ids:
            prefix = tuple(s.requests[rid].all_token_ids)
            m.initialize(rid, prefix, token_prefix_hash(prefix))
            rows.append(
                dict(proposal_row(rid, prefix), home_cohort=packet["requests"][rid]["cohort"])
            )
        m.register(rows)
        run_work(m)
        return m

    owner = K3Owner(factory, timeout_seconds=3)
    try:
        packet["pp_admission"] = PingPrePostController(mode, configuration).select(
            NS(rows={r: packet["requests"][r] for r in ids}), NS(call=owner.call)
        )
        publish(path, packet)
        audits = s.s2_pool.checks
        result = s.schedule()
        assert len(result.num_scheduled_tokens) == g["target_request_ceiling"]
        assert set(result.num_scheduled_tokens.values()) == {4}  # actual root + K3
        assert {len(v) for v in result.scheduled_spec_decode_tokens.values()} == {3}
        assert s.s2_pool.checks == audits + 2
        forwards = owner.machine.backend.prepost_forwards.rows()
        assert len(forwards) == 2 and {f["B"] for f in forwards} == {active}
        assert owner.machine.backend.physical_batch_ceiling == active
        assert not owner.machine.works
    finally:
        close(owner)
