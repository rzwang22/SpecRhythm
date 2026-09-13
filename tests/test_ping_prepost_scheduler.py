"""Real resident/FixedBatch/Ping gates; only the stock allocator is a CPU stand-in."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from test_ping_prepost import Backend, cleanup, run_work
from test_prepost_protocol import feedback
from test_serial_eager_owner import OwnerWorker, proposal_row, verify_row
from test_serving_fixed_identity import fixed_schedulers as _fixed_schedulers
from test_serving_s2 import s2_schedulers as _s2_schedulers

from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.serving.ping_prepost_controller import PingPrePostController
from specrhythm.serving.ping_prepost_machine import PingPrePostMachine
from specrhythm.serving.s2_pool import publish

fixed_schedulers, s2_schedulers = _fixed_schedulers, _s2_schedulers


def test_actual_scheduler_claimed_ragged_single_batch_and_replay_guard(
    fixed_schedulers, monkeypatch
):
    build, path = fixed_schedulers
    base, packet = build("serial", "bound-prefix", batch=16, n=16)
    fixed_batch = next(c for c in base.__class__.__mro__ if c.__name__ == "FixedBatch")
    monkeypatch.setitem(
        sys.modules, "specrhythm.serving.fixed_scheduler", NS(FixedBatch=fixed_batch)
    )
    spec = importlib.util.spec_from_file_location(
        "cpu_ping_prepost_scheduler",
        Path(__file__).resolve().parents[1] / "src/specrhythm/serving/ping_prepost_scheduler.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scheduler = module.PingPrePostScheduler()
    scheduler.requests, scheduler.running = base.requests, base.running
    scheduler.kv_cache_manager = base.kv_cache_manager
    scheduler._resident_ready = base._resident_ready
    scheduler._bind_requests()
    packet["max_requests_per_target_forward"] = 8
    for i, r in enumerate(packet["requests"].values()):
        r["cohort"] = "A" if i < 8 else "B"
    publish(path, packet)
    scheduler.freeze_pool()
    ids = list(scheduler.requests)
    m = PingPrePostMachine(Backend(NS(max_model_len=4096), worker=OwnerWorker()), request_ids=ids)
    for rid, r in scheduler.requests.items():
        m.initialize(rid, tuple(r.all_token_ids), token_prefix_hash(r.all_token_ids))
    m.register(
        [
            {
                **proposal_row(rid, tuple(r.all_token_ids)),
                "home_cohort": packet["requests"][rid]["cohort"],
            }
            for rid, r in scheduler.requests.items()
        ]
    )
    controller = PingPrePostController()
    try:
        for cycle in range(4):
            packet["pp_admission"] = controller.select(
                NS(rows=packet["requests"]), NS(call=lambda op, p: m.admit(p))
            )
            publish(path, packet)
            result = scheduler.schedule()
            claims = packet["pp_admission"]["claims"]
            assert len(result.num_scheduled_tokens) == len(claims) == 8
            assert set(result.num_scheduled_tokens) == {c["request_id"] for c in claims}
            if cycle in (2, 3):
                assert {len(v) for v in result.scheduled_spec_decode_tokens.values()} == {1, 4}
            if cycle == 3:
                assert {c["home_cohort"] for c in claims} == {"A", "B"}
            with pytest.raises(Exception, match="duplicate Target claim"):
                scheduler.schedule()
            m.verify_start([verify_row(c["proposal"]) for c in claims])
            rows = []
            for i, c in enumerate(claims):
                rid = c["request_id"]
                row, final = feedback(
                    c["proposal"],
                    m.requests[rid].committed_token_ids,
                    reject=cycle in (1, 2) and i == 0,
                )
                rows.append(row)
                request = scheduler.requests[rid]
                request.all_token_ids[:] = final
                request.num_output_tokens = len(final) - request.num_prompt_tokens
                request.num_computed_tokens = len(final) - 1
            run_work(m)
            m.feedback(rows)
            run_work(m)
        events = [r for r in m.ping_events.rows() if r["event"] == "admission"]
        assert [r["opportunity_cohort"] for r in events[:3]] == ["A", "B", "A"]
        assert events[1]["claims"][0]["home_cohort"] == "A"
        assert any(r["deferred"] for r in events)
    finally:
        cleanup(m)
