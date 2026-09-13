"""Same actual allocator lifecycle under full/runtime checks for the new backend."""

from types import SimpleNamespace

import pytest
from test_ping_prepost import admit, run_work
from test_prepost_protocol import feedback
from test_rolling_eager_gpu_backend import greedy_tokens
from test_serial_eager_owner import proposal_row, settle_row, verify_row

from specrhythm.continuation.ping_prepost_backend import PingPrePostBackendMixin
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.serving.ping_prepost_machine import PingPrePostMachine


@pytest.mark.parametrize("mode", ["full", "runtime"])
def test_real_runtime_allocator_mixed_post_fence_and_release(mode, tmp_path, monkeypatch):
    from test_fixed_runtime_audit import LifecycleWorker

    from specrhythm.serving import s2_draft
    from specrhythm.serving.fixed_audit import FixedAuditMixin
    from specrhythm.serving.s2_pool import publish

    monkeypatch.setattr(s2_draft, "cuda_memory", lambda _: {"free_memory_bytes": 10**9})
    monkeypatch.setenv("SR_FIXED_DRAFT_AUDIT", mode)
    monkeypatch.setenv("SR_S2_MODE", "pingpong-eager-prepost3")
    path = tmp_path / "control.json"
    monkeypatch.setenv("SR_S2_CONTROL", str(path))
    monkeypatch.setenv("SR_S2_RUN_DIRECTORY", str(tmp_path))
    states = {str(i): {"state": "STAGED"} for i in range(360)}
    publish(path, {"barrier_ns": None, "requests": states})

    class Audited(PingPrePostBackendMixin, FixedAuditMixin, s2_draft.S2DraftBackend):
        pass

    b = Audited(SimpleNamespace(max_model_len=4096), worker=LifecycleWorker())
    m = PingPrePostMachine(b, request_ids=states)
    for rid in ("0", "1"):
        m.initialize(rid, (10, 20), token_prefix_hash((10, 20)))
    b.initialize_many([(rid, (10, 20)) for rid in states if rid not in ("0", "1")])
    for state in states.values():
        state["state"] = "ACTIVE"
    publish(path, {"barrier_ns": 1, "requests": states})
    b.audit_requests(tuple(states))
    m.register([{**proposal_row(rid), "home_cohort": "A"} for rid in ("0", "1")])
    proposals = [c["proposal"] for c in admit(m, active=["0", "1"])["claims"]]
    m.verify_start([verify_row(p) for p in proposals])
    visits = b.audit_full_visits
    for _ in range(3):
        m.step()
    if mode == "runtime":
        assert b.audit_full_visits == visits
    rows = [feedback(p, (10, 20), reject=p["request_id"] == "1")[0] for p in proposals]
    m.feedback(rows)
    assert m.step()
    assert b.states["0"].proposal == greedy_tokens(b.states["0"].prefix, 4)
    assert len(b.states["1"].proposal) == 1
    # A second parent batch: common rejection recovery merges with the other
    # parent's still-running lookahead through the actual allocator callbacks.
    proposals = [c["proposal"] for c in admit(m, active=["0", "1"])["claims"]]
    m.verify_start([verify_row(p) for p in proposals])
    m.step()
    rejected = next(p for p in proposals if p["request_id"] == "1")
    accepted = next(p for p in proposals if p["request_id"] == "0")
    m.feedback([feedback(rejected, m.requests["1"].committed_token_ids, reject=True)[0]])
    m.step()
    assert b.prepost_forwards.rows()[-1]["purpose"] == "prepost_mixed"
    assert b.prepost_forwards.rows()[-1]["B"] == 2
    m.feedback([feedback(accepted, m.requests["0"].committed_token_ids)[0]])
    run_work(m)
    b._audit()  # Explicit full reconciliation agrees with incrementally maintained evidence.
    for rid, state in list(m.requests.items()):
        m.diagnostic_settle(settle_row(rid, state.committed_token_ids, state.next_round_id))
    b.finish_many(tuple(b.states))
    m.shutdown()
    assert not b.worker.blocks
    if mode == "runtime":
        assert b.audit_guard.allocations == b.audit_guard.releases
        assert not b.audit_guard.pending and not b.audit_guard.owners
