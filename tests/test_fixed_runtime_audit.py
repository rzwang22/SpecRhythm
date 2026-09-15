"""Actual owner/backend full/runtime equivalence and allocator boundary faults; CPU only."""

from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from test_phase4_vllm_draft import FakeWorker, commit, next_token
from test_serial_eager_owner import proposal_row, settle_row, sync_row, verify_row
from test_serving_s2_runtime import PoolWorker

from specrhythm.continuation.gpu_backend import GPUContinuationBackendMixin
from specrhythm.phase4.draft_batch import DraftMaterialization, DraftProposalPlan
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.serving import s2_draft
from specrhythm.serving.eager_machine import EagerSerialMachine
from specrhythm.serving.eager_owner import EagerOwner
from specrhythm.serving.fixed_audit import FixedAuditMixin
from specrhythm.serving.s2_pool import publish


class LifecycleWorker(PoolWorker):
    """Same allocate_slots/free path as the real private full-attention worker."""

    def __init__(self):
        super().__init__()
        self.kv = SimpleNamespace(
            allocate_slots=self.allocate_slots,
            free=self.free,
            get_block_ids=lambda rid: [list(self.blocks[rid])],
        )

    def allocate_slots(self, request, *, num_new_tokens):
        rid = request.request_id
        blocks = self.blocks.setdefault(rid, [])
        count = math.ceil((request.num_computed_tokens + num_new_tokens) / 16) - len(blocks)
        added = list(range(self.next_block, self.next_block + max(count, 0)))
        self.next_block += len(added)
        blocks.extend(added)
        return SimpleNamespace(get_block_ids=lambda: [added])

    def free(self, request):
        del self.blocks[request.request_id]

    def materialize(self, rows, purpose):
        for row in rows:
            self.kv.allocate_slots(
                SimpleNamespace(request_id=row.request_id, num_computed_tokens=row.valid_length),
                num_new_tokens=len(row.suffix),
            )
        return FakeWorker.materialize(self, rows, purpose)

    def release(self, ids):
        self.fence("release")
        FakeWorker.release(self, ids)
        for rid in ids:
            self.kv.free(SimpleNamespace(request_id=rid))

    def shutdown(self):
        if self.memory:
            self.release(tuple(self.memory))
        self.fence("shutdown")
        return super().shutdown()


class Backend(GPUContinuationBackendMixin, FixedAuditMixin, s2_draft.S2DraftBackend):
    def step_gpu_continuations(self, works):
        value = super().step_gpu_continuations(works)
        if value and all(v is not None for v in value.values()):
            self.completed.set()
        return value


@pytest.fixture
def make_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(s2_draft, "cuda_memory", lambda _: {"free_memory_bytes": 10**9})
    instances = []

    def build(mode, n=360):
        folder = tmp_path / str(len(instances))
        folder.mkdir()
        monkeypatch.setenv("SR_FIXED_DRAFT_AUDIT", mode)
        monkeypatch.setenv("SR_S2_MODE", "serial-eager")
        monkeypatch.setenv("SR_S2_CONTROL", str(folder / "control.json"))
        monkeypatch.setenv("SR_S2_RUN_DIRECTORY", str(folder))
        states = {f"r{i}": {"state": "STAGED"} for i in range(n)}
        publish(folder / "control.json", {"barrier_ns": None, "requests": states})
        backend = Backend(SimpleNamespace(max_model_len=4096), worker=LifecycleWorker())
        backend.completed = threading.Event()
        prefixes = {rid: (10, i + 20) for i, rid in enumerate(states)}
        backend.initialize_many(tuple(prefixes.items()))
        for row in states.values():
            row["state"] = "ACTIVE"
        publish(folder / "control.json", {"barrier_ns": 1, "requests": states})
        backend.audit_requests(tuple(prefixes))
        instances.append(backend)
        return backend, prefixes, states, folder / "control.json"

    yield build
    for backend in instances:
        if not backend.closed:
            backend.worker.fence("test_cleanup")
            # Fault tests restore corrupt state before cleanup; never waive a check.
            backend.finish_many(tuple(backend.states))
            backend.shutdown()


def test_runtime_stable_token_checks_do_not_scan_unrelated_resident_requests(make_backend):
    b, prefixes, _, _ = make_backend("runtime")
    before = b.audit_full_visits
    checks = b.audit_guard.visits
    plans = [DraftProposalPlan(rid, 0, prefixes[rid], 4, ()) for rid in list(prefixes)[:16]]
    b.propose_many(plans)
    assert b.audit_full_visits == before
    assert b.audit_guard.visits - checks == 16 * (2 + 3)  # audit pair and three write checks
    assert len(b.audit_guard.blocks) == 360
    b.audit_ids = None
    b._audit()  # Explicit complete reconciliation at a test state boundary.
    assert b.audit_full_visits > before


@pytest.mark.parametrize(
    "fault", ["state", "frontier", "version", "ownership", "duplicate", "pending", "identity"]
)
def test_runtime_negative_invariants(make_backend, fault):
    b, _, states, path = make_backend("runtime", 2)
    state = b.states["r0"]
    old_frontier = state.materialized
    old_internal = state.internal_id
    blocks = list(b.worker.blocks[state.internal_id])
    if fault == "state":
        states["r0"]["state"] = "INVALID"
        publish(path, {"barrier_ns": 1, "requests": states})
    elif fault == "frontier":
        state.materialized = 99
    elif fault == "version":
        state.next_round = -1
    elif fault == "ownership":
        b.worker.blocks[state.internal_id] = list(b.worker.blocks["sr-draft:r1"])
    elif fault == "duplicate":
        b.worker.blocks[state.internal_id] = blocks * 2
    elif fault == "identity":
        state.internal_id = "sr-draft:r1"
    else:
        b.audit_guard.pending = True
    try:
        with pytest.raises((ValueError, RuntimeError)):
            if fault == "pending":
                b.finish_many(["r0"])
            else:
                b.audit_requests(["r0"])
    finally:
        state.materialized = old_frontier
        state.next_round = 0
        state.internal_id = old_internal
        b.worker.blocks[state.internal_id] = blocks
        b.audit_guard.pending = False
        states["r0"]["state"] = "ACTIVE"
        publish(path, {"barrier_ns": 1, "requests": states})


def test_runtime_failed_write_requires_fence_then_real_allocator_teardown(make_backend):
    b, p, _, _ = make_backend("runtime", 2)
    b.worker.inject_failure = True
    with pytest.raises(RuntimeError, match="injected forward failure"):
        b.propose_many([DraftProposalPlan(r, 0, p[r], 4, ()) for r in p])
    assert b.failed and b.audit_guard.pending and b.audit_guard.owners
    with pytest.raises(RuntimeError, match="failed"):
        b.finish_many(tuple(p))
    b.shutdown()
    assert b.closed and not b.worker.blocks and not b.audit_guard.owners
    assert not b.audit_guard.pending
    assert b.audit_guard.allocations == b.audit_guard.releases


@pytest.mark.parametrize("mode", ["full", "runtime"])
def test_finished_control_race_keeps_admitted_write_and_terminal_feedback_legal(
    make_backend, mode
):
    from test_rolling_eager_gpu_backend import begin, normal, plan_and_receipt

    from specrhythm.continuation.core import RollingContinuation
    from specrhythm.continuation.policy import StaticEagerEligibility

    b, p, states, path = make_backend(mode, 1)
    core = RollingContinuation("owner", StaticEagerEligibility(set(p)))
    core.register("r0", p["r0"], max_output_tokens=100)
    proposal = normal(b, core, "r0")
    work = begin(b, core, proposal, "r0")
    states["r0"]["state"] = "FINISHED"
    publish(path, {"barrier_ns": 1, "requests": states})
    # Control publication can precede the authoritative socket feedback. Keep
    # the existing admitted token/fence semantics; never enroll a new task here.
    b.step_gpu_continuation(work)
    plan, _ = plan_and_receipt(proposal, accepted=1, tail=(999,), terminal=True)
    b.rebase_gpu_parent(plan, work=work)
    assert "r0" in b.retired and not b.worker.memory
    with pytest.raises(ValueError, match="retired"):
        b.step_gpu_continuation(work)
    b.shutdown()


def test_allocator_rejects_unscoped_free_and_cross_request_allocation(make_backend):
    b, _, _, _ = make_backend("runtime", 2)
    b.worker.kv.original.unreviewed_mutation = lambda: None
    with pytest.raises(AttributeError, match="does not cover allocator API"):
        b.worker.kv.unreviewed_mutation()
    with pytest.raises(ValueError, match="outside fenced"):
        b.worker.kv.free(SimpleNamespace(request_id="sr-draft:r1"))
    original = b.worker.kv.original.allocate_slots

    def conflicting(request, **kw):
        result = original(request, **kw)
        b.worker.blocks[request.request_id].append(b.worker.blocks["sr-draft:r1"][0])
        return result

    b.worker.kv.original.allocate_slots = conflicting
    row = DraftMaterialization("sr-draft:r0", (10, 20, 30), 2)
    try:
        with pytest.raises(ValueError, match="ownership conflict"):
            b.worker.materialize([row], "proposal")
    finally:
        b.worker.blocks["sr-draft:r0"].pop()
        b.worker.kv.original.allocate_slots = original
        b.worker.fence("failed_test_write")


def test_full_runtime_ordinary_outputs_and_rollback_match(make_backend):
    results = []
    for mode in ("full", "runtime"):
        b, p, _, _ = make_backend(mode, 3)
        committed = []
        for round_id in range(4):
            proposals = b.propose_many([DraftProposalPlan(r, round_id, p[r], 4, ()) for r in p])
            plans = [
                commit(
                    r,
                    p[r],
                    proposals[r],
                    accepted=(i + round_id) % 5,
                    tail=[77],
                    round_id=round_id,
                )
                for i, r in enumerate(p)
            ]
            b.commit_many(plans)
            p = {r: b.states[r].prefix for r in p}
            committed.append(p)
            b.audit_ids = None
            b._audit()
        results.append((committed, b.metrics.counters.copy(), b.physical_rows()))
        b.finish_many(tuple(p))
        b.shutdown()
    assert results[0] == results[1]


def test_real_owner_full_runtime_rolling_mixed_outcomes(tmp_path, monkeypatch):
    monkeypatch.setattr(s2_draft, "cuda_memory", lambda _: {"free_memory_bytes": 10**9})
    results = []
    for mode in ("full", "runtime"):
        folder = tmp_path / mode
        folder.mkdir()
        monkeypatch.setenv("SR_FIXED_DRAFT_AUDIT", mode)
        monkeypatch.setenv("SR_S2_MODE", "serial-eager")
        monkeypatch.setenv("SR_S2_CONTROL", str(folder / "control.json"))
        monkeypatch.setenv("SR_S2_RUN_DIRECTORY", str(folder))
        ids = [f"r{i}" for i in range(3)]
        prefixes = {r: (10, i + 20) for i, r in enumerate(ids)}
        control = {"barrier_ns": None, "requests": {r: {"state": "STAGED"} for r in ids}}
        publish(folder / "control.json", control)

        def factory(ids=ids, prefixes=prefixes):
            b = Backend(SimpleNamespace(max_model_len=4096), worker=LifecycleWorker())
            b.completed = threading.Event()
            machine = EagerSerialMachine(b, request_ids=ids)
            for r in ids:
                machine.initialize(r, prefixes[r], token_prefix_hash(prefixes[r]))
            return machine

        owner = EagerOwner(factory)
        control["barrier_ns"] = 1
        for r in ids:
            control["requests"][r]["state"] = "ACTIVE"
        publish(folder / "control.json", control)
        proposals = owner.call(
            "batch_propose", {"requests": [proposal_row(r, prefixes[r]) for r in ids]}
        )["proposals"]
        try:
            outputs = []
            for turn in range(7):
                b = owner.machine.backend
                b.completed.clear()
                owner.call("eager_enqueue", {"requests": [verify_row(p) for p in proposals]})
                assert b.completed.wait(5)
                sync = []
                for i, p in enumerate(proposals):
                    rid = p["request_id"]
                    row, new_prefix = sync_row(p, prefixes[rid], reject=(turn + i) % 7 == 2)
                    if (turn + i) % 7 != 2:
                        tokens = tuple(p["proposal_token_ids"])
                        bridge = next_token(prefixes[rid] + tokens)
                        delta = tokens + (bridge + int((turn + i) % 7 == 5),)
                        new_prefix = prefixes[rid] + delta
                        row.update(
                            committed_delta=list(delta),
                            committed_prefix_hash=token_prefix_hash(new_prefix),
                        )
                    sync.append(row)
                    prefixes[rid] = new_prefix
                reply = owner.call(
                    "synchronize_and_batch_propose",
                    {
                        "synchronizations": sync,
                        "proposals": [
                            proposal_row(r, prefixes[r], turn + 1, 102 - len(prefixes[r]))
                            for r in ids
                        ],
                    },
                )
                proposals = reply["proposals"]
                outputs.append(dict(prefixes))
            results.append((outputs, dict(owner.machine.counters)))
        finally:
            for r in ids:
                state = owner.machine.requests[r]
                owner.call(
                    "diagnostic_settle",
                    settle_row(r, state.committed_token_ids, state.next_round_id),
                )
            owner.call("shutdown", {})
            owner._thread.join(5)
            assert not owner._thread.is_alive()
            assert not b.worker.memory and not b.worker.blocks
    assert results[0] == results[1]


def test_full_runtime_feedback_first_switch_eos_refill_and_late_result(tmp_path, monkeypatch):
    """Hold an unrelated owner command until both immutable messages are queued."""
    monkeypatch.setattr(s2_draft, "cuda_memory", lambda _: {"free_memory_bytes": 10**9})
    results = []
    for mode in ("full", "runtime"):
        entered, resume, feedback_queued = (threading.Event() for _ in range(3))

        class ControlledOwner(EagerOwner):
            def _dispatch(self, operation, payload, _entered=entered, _resume=resume):
                if operation == "test_gate":
                    _entered.set()
                    assert _resume.wait(5)
                    return {}
                if operation == "test_reconcile":
                    self.machine.backend._audit()
                    return self.machine.backend.physical_rows()
                if operation == "test_late":
                    return self.machine.backend.step_gpu_continuation(payload["work"])
                return super()._dispatch(operation, payload)

        folder = tmp_path / mode
        folder.mkdir()
        for name, value in dict(
            SR_FIXED_DRAFT_AUDIT=mode,
            SR_S2_MODE="serial-eager",
            SR_S2_CONTROL=str(folder / "control.json"),
            SR_S2_RUN_DIRECTORY=str(folder),
        ).items():
            monkeypatch.setenv(name, value)
        packet = dict(barrier_ns=None, requests={r: dict(state="STAGED") for r in ("r0", "r1")})
        path = folder / "control.json"
        publish(path, packet)

        def factory():
            b = Backend(SimpleNamespace(max_model_len=4096), worker=LifecycleWorker())
            b.completed = threading.Event()
            m = EagerSerialMachine(b, request_ids=("r0", "r1"))
            for rid in ("r0", "r1"):
                m.initialize(rid, (10, 20), token_prefix_hash((10, 20)))
            return m

        owner = ControlledOwner(factory)
        b = owner.machine.backend
        packet.update(barrier_ns=1)
        packet["requests"]["r0"]["state"] = "ACTIVE"
        packet["requests"]["r1"]["state"] = "QUEUED"
        publish(path, packet)
        original_put = owner.commands.put

        def put(command, *args, _put=original_put, _queued=feedback_queued, **kwargs):
            result = _put(command, *args, **kwargs)
            if command[0] == "synchronize_and_batch_propose":
                _queued.set()
            return result

        owner.commands.put = put
        prefix = (10, 20)
        row = proposal_row("r0", prefix)
        row["eos_token_ids"] = [999]
        proposal = owner.call("batch_propose", {"requests": [row]})["proposals"][0]
        try:
            sync, prefix = sync_row(proposal, prefix, reject=True)
            with ThreadPoolExecutor(max_workers=2) as executor:
                gate = executor.submit(owner.call, "test_gate", {})
                assert entered.wait(5)
                owner.call("eager_enqueue", {"requests": [verify_row(proposal)]})
                feedback = executor.submit(
                    owner.call,
                    "synchronize_and_batch_propose",
                    {
                        "synchronizations": [sync],
                        "proposals": [proposal_row("r0", prefix, 1, 102 - len(prefix))],
                    },
                )
                assert feedback_queued.wait(5)
                resume.set()
                gate.result(5)
                proposal = feedback.result(5)["proposals"][0]
            assert owner.machine.counters["admissions"] == 1
            assert owner.machine.counters["started"] == 0
            assert b.metrics.forwards["eager"] == 0
            owner.call("test_reconcile", {})

            # The same request keeps its static membership across disable/re-enable.
            owner.call("eager_switch", dict(enabled=False, decision_version=1))
            owner.call("eager_enqueue", {"requests": [verify_row(proposal)]})
            tokens = tuple(proposal["proposal_token_ids"])
            delta = tokens + (next_token(prefix + tokens),)
            prefix += delta
            sync = dict(
                request_id="r0",
                round_id=1,
                committed_delta=list(delta),
                committed_prefix_hash=token_prefix_hash(prefix),
                terminal=False,
            )
            proposal = owner.call(
                "synchronize_and_batch_propose",
                {
                    "synchronizations": [sync],
                    "proposals": [proposal_row("r0", prefix, 2, 102 - len(prefix))],
                },
            )["proposals"][0]
            assert b.metrics.forwards["eager"] == 0
            owner.call("eager_switch", dict(enabled=True, decision_version=2))
            b.completed.clear()
            owner.call("eager_enqueue", {"requests": [verify_row(proposal)]})
            assert b.completed.wait(5)  # Real CPU worker path, before any Target feedback.
            work = owner.machine.works["r0"]
            tokens = tuple(proposal["proposal_token_ids"])
            delta = tokens + (next_token(prefix + tokens),)
            prefix += delta
            sync = dict(
                request_id="r0",
                round_id=2,
                committed_delta=list(delta),
                committed_prefix_hash=token_prefix_hash(prefix),
                terminal=False,
            )
            proposal = owner.call(
                "synchronize_and_batch_propose",
                {
                    "synchronizations": [sync],
                    "proposals": [proposal_row("r0", prefix, 3, 102 - len(prefix))],
                },
            )["proposals"][0]
            assert owner.machine.counters["promotions"] == 1
            with pytest.raises(RuntimeError, match="retired"):
                owner.call("test_late", {"work": work})
            owner.call("test_reconcile", {})

            # Authoritative EOS consumes one accepted candidate and one correction.
            b.completed.clear()
            owner.call("eager_enqueue", {"requests": [verify_row(proposal)]})
            assert b.completed.wait(5)
            delta = (proposal["proposal_token_ids"][0], 999)
            prefix += delta
            packet["requests"]["r0"]["state"] = "FINISHED"
            publish(path, packet)
            terminal = dict(
                request_id="r0",
                round_id=3,
                committed_delta=list(delta),
                committed_prefix_hash=token_prefix_hash(prefix),
                terminal=True,
            )
            payload = {"synchronizations": [terminal], "proposals": []}
            owner.call("synchronize_and_batch_propose", payload)
            before = len(b.worker.calls)
            owner.call("synchronize_and_batch_propose", payload)  # Late identical feedback.
            assert len(b.worker.calls) == before and "r0" in b.retired
            owner.call("test_reconcile", {})

            # Refill uses an already-prefilled queued request, never timed initialize.
            packet["requests"]["r1"]["state"] = "ACTIVE"
            publish(path, packet)
            refill = owner.call("batch_propose", {"requests": [proposal_row("r1")]})
            assert refill["proposals"][0]["round_id"] == 0
            assert len([p for p, _ in b.worker.calls if p == "setup"]) == 2
            owner.call("diagnostic_settle", settle_row("r1"))  # Cancel with authoritative prefix.
            assert not b.worker.blocks and not b.worker.memory
            owner.call("test_reconcile", {})
            c = dict(owner.machine.counters)
            assert c["committed_tokens"] == len(prefix) - 2
            assert (
                c["committed_tokens"]
                == c["parent_accepted_tokens"] + c["correction_tokens"] + c["bonus_tokens"]
            )
            results.append((prefix, c))
        finally:
            resume.set()
            for rid, state in owner.machine.requests.items():
                if rid not in b.retired:
                    owner.call(
                        "diagnostic_settle",
                        settle_row(
                            rid,
                            state.committed_token_ids,
                            state.next_round_id,
                            terminal=state.finished,
                        ),
                    )
            owner.call("shutdown", {})
            assert not owner._thread.is_alive() and not b.worker.blocks
    assert results[0] == results[1]


@pytest.mark.parametrize("mode", ["full", "runtime"])
def test_gpu_correctness_entry_uses_same_audit_layer_with_isolated_references(
    tmp_path,
    monkeypatch,
    mode,
):
    from specrhythm.continuation.audit_gpu_check import initialized, prepare
    from specrhythm.continuation.gpu_check import run_checks

    monkeypatch.setattr(s2_draft, "cuda_memory", lambda _: {"free_memory_bytes": 10**9})
    for name in ("SR_FIXED_DRAFT_AUDIT", "SR_S2_MODE", "SR_S2_RUN_DIRECTORY", "SR_S2_CONTROL"):
        monkeypatch.setenv(name, "")  # Restore adapter env changes after this test.
    requests = [
        dict(request_id=f"live{i}", prefix=(10, i + 20), remaining_output_tokens=64)
        for i in range(3)
    ]
    b = prepare(
        SimpleNamespace(max_model_len=4096), tmp_path, requests, mode, worker=LifecycleWorker()
    )
    try:
        result = run_checks(
            b,
            requests,
            eos_token_ids=(),
            vocab_size=1000,
            snapshot=lambda b: b.physical_rows(),
            checkpoint=initialized,
        )
        assert result["reference_comparisons"]
        assert all(row["exact"] for row in result["reference_comparisons"])
        assert b.pool.initial is not None
        if mode == "runtime":
            assert b.audit_guard.allocations == b.audit_guard.releases > 0
            assert not b.audit_guard.handles
    finally:
        b.shutdown()
    if mode == "runtime":
        assert not b.audit_guard.owners and not b.audit_guard.pending
        assert b.audit_guard.allocations == b.audit_guard.releases
