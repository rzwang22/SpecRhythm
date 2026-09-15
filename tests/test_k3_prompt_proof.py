"""Real Target pool producer/auditor; hash reuse cannot memoize mutable KV."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from test_serving_fixed_identity import fixed_schedulers as _fixed
from test_serving_s2 import s2_schedulers as _s2

from specrhythm.serving import k3_prompt_proof
from specrhythm.serving.k3 import MODES

fixed_schedulers, s2_schedulers = _fixed, _s2


@pytest.fixture
def target_pool(fixed_schedulers, monkeypatch):
    build, path = fixed_schedulers
    base, packet = build("serial", "bound-prefix", batch=16, n=360)
    fixed = next(c for c in type(base).__mro__ if c.__name__ == "FixedBatch")
    monkeypatch.setitem(sys.modules, "specrhythm.serving.fixed_scheduler", NS(FixedBatch=fixed))
    spec = importlib.util.spec_from_file_location(
        "cpu_k3_scheduler", Path("src/specrhythm/serving/ping_prepost_scheduler.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scheduler = module.PingPrePostScheduler()
    scheduler.requests, scheduler.running = base.requests, base.running
    scheduler.kv_cache_manager = base.kv_cache_manager
    scheduler._resident_ready = base._resident_ready
    scheduler._bind_requests()
    return scheduler, packet, path


@pytest.mark.parametrize("mode", MODES)
def test_target_current_pool_reuses_hash_but_reads_every_live_frontier_and_block(
    target_pool, monkeypatch, mode
):
    s, packet, _ = target_pool
    monkeypatch.setenv("SR_S2_MODE", mode)
    hashes = []
    original = k3_prompt_proof.prefix_record
    monkeypatch.setattr(
        k3_prompt_proof, "prefix_record", lambda *a: (hashes.append(a[0]), original(*a))[1]
    )
    s.freeze_pool()
    assert len(hashes) == 360
    old_method = next(c for c in type(s).__mro__ if c.__name__ == "PoolScheduler").physical_rows
    for _ in range(2):
        rows = s.physical_rows()
        assert rows == old_method(s)
        s.s2_pool.check(rows, packet["requests"])
    assert len(hashes) == 360  # Old producer rehashed every resident twice per schedule.
    assert s.s2_pool.checks == 3 and len(s._k3_prompt_proofs) == 360
    s.requests["0"].num_computed_tokens += 2
    assert s.physical_rows()["0"]["materialized_tokens"] == s.requests["0"].num_computed_tokens
    s.kv_cache_manager = NS(get_block_ids=lambda i: [[int(i) + 1, 1000 + int(i)]])
    assert s.physical_rows()["0"]["block_ids"] == [[1, 1000]]
    assert len(hashes) == 360


@pytest.mark.parametrize("fault", ["prompt", "bool", "float", "block", "frontier", "eviction"])
def test_target_proof_cannot_hide_corruption(target_pool, monkeypatch, fault):
    s, packet, _ = target_pool
    monkeypatch.setenv("SR_S2_MODE", "pingpong-k3")
    s.freeze_pool()
    r = s.requests["0"]
    if fault in ("prompt", "bool", "float"):
        r.prompt_token_ids = [dict(prompt=99, bool=True, float=1.0)[fault], 1]
    elif fault == "block":
        s.kv_cache_manager = NS(get_block_ids=lambda i: [[1]])
    elif fault == "frontier":
        r.num_computed_tokens = 0
    else:
        del s.requests["359"]  # Unarrived immutable prefix cannot disappear.
    with pytest.raises(ValueError):
        s.s2_pool.check(s.physical_rows(), packet["requests"])


def test_finished_proof_is_retired_and_no_old_mode_is_optimized(target_pool, monkeypatch):
    s, _, _ = target_pool
    monkeypatch.setenv("SR_S2_MODE", "pingpong-k3")
    s.physical_rows()
    del s.requests["0"]
    s.physical_rows()
    assert "0" not in s._k3_prompt_proofs
    s._k3_prompt_proofs.clear()
    monkeypatch.setenv("SR_S2_MODE", "pingpong-prepost3")
    s.physical_rows()
    assert s._k3_prompt_proofs == {}


def test_prompt_proof_captures_bounded_real_snapshot_scope(target_pool, monkeypatch):
    from specrhythm.continuation.trace import CausalTrace

    s, _, _ = target_pool
    monkeypatch.setenv("SR_S2_MODE", "pingpong-k3")
    trace = CausalTrace(True, layout="phased")
    monkeypatch.setattr(k3_prompt_proof, "TRACE", trace)
    s.freeze_pool()
    trace.follow_control({"diagnostic_phase": "measurement"})
    s.physical_rows()
    records = trace.report()
    rows = [r for r in records["rows"] if r["category"] == "target_pool_prompt_proof"]
    assert [r["prompt_hashes"] for r in rows] == [360, 0]
    assert rows[-1]["prompt_proofs_reused"] == rows[-1]["resident_rows"] == 360
    assert rows[-1]["current_prompt_tokens_compared"] == 720
    assert rows[-1]["policy"] == k3_prompt_proof.POLICY
    assert records["phases"]["measurement"]["retained_rows"] == 2
    assert records["dropped_rows"] == 0
