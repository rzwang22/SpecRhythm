"""Real Draft owner/state/KV stop protocol, with independent CPU KV arrays."""

import time
from types import SimpleNamespace

import pytest
from test_phase4_dual_batched_draft import initial, wait
from test_serving_s2_runtime import PoolWorker

from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.transport import CheckpointJsonl
from specrhythm.serving import fixed_runtime, s2_draft
from specrhythm.serving.common import DataError, read_json
from specrhythm.serving.fixed_plan import point, settings
from specrhythm.serving.fixed_settle import (
    DiagnosticDualController,
    DiagnosticDualMachine,
    DiagnosticSerialMachine,
    DiagnosticSerialServer,
)
from specrhythm.serving.s2_pool import publish


@pytest.fixture
def owner(tmp_path, monkeypatch):
    monkeypatch.setenv("SR_S2_CONTROL", str(tmp_path / "control.json"))
    monkeypatch.setenv("SR_S2_RUN_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("SR_S2_MODE", "pingpong")
    monkeypatch.setattr(s2_draft, "cuda_memory", lambda *a: {"free_memory_bytes": 10**9})
    states = {rid: {"state": "STAGED", "cohort": "A"} for rid in ("a", "b")}
    publish(tmp_path / "control.json", {"barrier_ns": None, "requests": states})
    controllers, backends = [], []

    def make(dual=False):
        def factory():
            backend = s2_draft.S2DraftBackend(
                SimpleNamespace(max_model_len=4096), worker=PoolWorker()
            )
            backends.append(backend)
            return (DiagnosticDualMachine if dual else DiagnosticSerialMachine)(backend)

        if dual:
            value = DiagnosticDualController(factory, CheckpointJsonl(tmp_path / "work.jsonl"))
            controllers.append(value)
            return value
        return factory()

    yield SimpleNamespace(make=make, states=states, path=tmp_path / "control.json")
    # Failures are asserted BEFORE fixture cleanup; never counted as successful stop.
    for controller in controllers:
        if controller._thread.is_alive():
            controller.shutdown(failed=True)
    for backend in backends:
        if not backend.closed:
            backend.shutdown()


def payload(rid="a", final=(1, 2, 3), *, round_id=0, version=1, terminal=False, admitted=True):
    return dict(
        request_id=rid,
        committed_prefix=list(final),
        committed_prefix_hash=token_prefix_hash(final),
        natural_terminal=terminal,
        admitted=admitted,
        runtime_mode="serial",
        next_round_id=round_id,
        prefix_version=version,
        bootstrap_prefix_hash=token_prefix_hash((1, 2, 3)),
        deadline_ns=time.monotonic_ns() + 5_000_000_000,
    )


def activate(owner):
    owner.states["a"]["state"] = "ACTIVE"
    publish(owner.path, {"barrier_ns": 1, "requests": owner.states})


def test_physical_only_unactivated_resident_is_not_logically_initialized_or_cancelled(owner):
    machine = owner.make()
    machine.backend.initialize_many((("a", (1, 2, 3)),))
    receipt = machine.diagnostic_settle(payload(admitted=False))
    assert not receipt["logical_initialized"] and not machine.requests
    assert receipt["released"] and not machine.backend.worker.memory
    assert machine.shutdown()["shutdown"]


@pytest.mark.parametrize(
    "bad", ["unknown", "active_uninitialized", "wrong_prefix", "wrong_round", "shared_kv"]
)
def test_stop_cannot_guess_state_or_release_unproven_KV(owner, bad):
    machine = owner.make()
    if bad != "unknown":
        if bad == "active_uninitialized":
            machine.backend.initialize_many((("a", (1, 2, 3)),))
        else:
            machine.initialize("a", (1, 2, 3), token_prefix_hash((1, 2, 3)))
    p = payload()
    if bad == "wrong_prefix":
        p.update(committed_prefix=[9, 2, 3], committed_prefix_hash=token_prefix_hash((9, 2, 3)))
    elif bad == "wrong_round":
        p["next_round_id"] = 4
    elif bad == "shared_kv":
        machine.initialize("b", (1, 2, 3), token_prefix_hash((1, 2, 3)))
        machine.backend.worker.blocks["sr-draft:b"] = machine.backend.worker.blocks["sr-draft:a"]
    with pytest.raises(DataError):
        machine.diagnostic_settle(p)
    assert not machine.backend.retired and not machine.diagnostic_receipts


def test_normal_serial_shutdown_still_refuses_pending_and_cannot_fake_cancel(owner):
    machine = owner.make()
    machine.initialize("a", (1, 2, 3), token_prefix_hash((1, 2, 3)))
    activate(owner)
    machine.batch_propose(
        [
            dict(
                request_id="a",
                round_id=0,
                committed_prefix_len=3,
                committed_prefix_hash=token_prefix_hash((1, 2, 3)),
                remaining_output_budget=8,
                eos_token_ids=[],
            )
        ]
    )
    with pytest.raises(ValueError, match="unresolved proposals"):
        machine.shutdown()
    assert not machine.backend.closed
    result = machine.diagnostic_settle(payload())
    assert result["discarded_proposal_tokens"] == 4 and not machine.requests["a"].finished
    assert machine.shutdown()["shutdown"]


def test_terminal_tail_completes_on_real_owner_before_idempotent_stop(owner):
    controller = owner.make(dual=True)
    controller.execute("initialize", initial("a"))
    activate(owner)
    start = {**initial("a", remaining=1), "logical_cohort": "A"}
    controller.enqueue("propose_only", [start])
    assert wait(controller, 1)[0]["target_tail"]
    owner.states["a"]["state"] = "FINISHED"
    publish(owner.path, {"barrier_ns": 1, "requests": owner.states})
    controller.enqueue(
        "finish_tail",
        [
            dict(
                request_id="a",
                logical_cohort="A",
                round_id=0,
                proposal_id=None,
                committed_delta=[999],
                prefix_version=2,
                prefix_token_sha256=token_prefix_hash((1, 2, 3, 999)),
                terminal=True,
                remaining_output_budget=0,
                eos_token_ids=[999],
            )
        ],
    )
    # Actual stop waits for the outstanding tail; idle alone does not settle ready proposals.
    client = SimpleNamespace(call=lambda *a: controller.status())
    fixed_runtime.wait_draft(client, 5)
    p = payload(final=(1, 2, 3, 999), terminal=True, version=2)
    p["runtime_mode"] = "pingpong"
    original_release = controller.machine.release_receipts["a"]["resources_released_ns"]
    # Pure authority validation of already-freed state must still enforce version.
    with pytest.raises(DataError, match="final prefix/version"):
        controller.machine._settle({**p, "prefix_version": 3})
    result = controller.execute("diagnostic_settle", p)
    assert result["release_kind"] == "already_naturally_released"
    assert result["resources_released_ns"] == original_release < result["start_ns"]
    assert result["new_proposals_generated"] == 0 and controller.machine.requests["a"].finished
    assert controller.execute("diagnostic_settle", p) == result
    assert controller.shutdown()["shutdown"]
    assert not controller.machine.backend.worker.memory


def test_single_deadline_wait_budget_shrinks_and_expired_stop_does_not_release(owner):
    machine = owner.make()
    machine.initialize("a", (1, 2, 3), token_prefix_hash((1, 2, 3)))
    p = payload()
    p["deadline_ns"] = time.monotonic_ns() - 1
    with pytest.raises(DataError, match="drain deadline"):
        machine.diagnostic_settle(p)
    assert machine.backend.worker.memory and not machine.backend.retired
    timeouts = []
    client = SimpleNamespace()

    def status(*a):
        timeouts.append(client.timeout_seconds)
        time.sleep(0.002)
        return {"inflight_request_ids": ["a"] if len(timeouts) < 3 else [], "failures": {}}

    client.call = status
    fixed_runtime.wait_draft(client, 1)
    assert timeouts[0] > timeouts[1] > timeouts[2]
    assert machine.diagnostic_settle(payload())["released"]
    assert machine.shutdown()["shutdown"]


def test_capacity_runtime_zero_verification_uses_actual_empty_machine_shutdown(
    owner, tmp_path, monkeypatch
):
    machine = owner.make()
    server = DiagnosticSerialServer(
        tmp_path / "unused.sock", machine, event_log=CheckpointJsonl(tmp_path / "events.jsonl")
    )
    calls = []

    def call(operation, row):
        calls.append(operation)
        return server._dispatch(operation, row)

    manifest = {
        "fixed_diagnostic": {"options": settings()},
        "sha256": "sha",
        "execution": {"git_commit": "a" * 40},
    }
    monkeypatch.setattr(
        fixed_runtime, "configure", lambda *a: (SimpleNamespace(target=None), manifest, [])
    )
    monkeypatch.setattr(fixed_runtime, "validate_worker_ranks", lambda *a: [])
    monkeypatch.setattr(
        fixed_runtime,
        "capacity_for",
        lambda *a, **kw: dict(
            valid=True,
            required_blocks=0,
            num_gpu_blocks=1,
            additional_workspace_reserve_bytes=0,
            resident_draft_logits_bytes=0,
            free_memory_bytes_after_engine_init=1,
        ),
    )
    monkeypatch.setattr(fixed_runtime, "client_for", lambda *a: SimpleNamespace(call=call))
    cfg = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            async_scheduling=False, max_num_seqs=128, max_num_batched_tokens=4096
        ),
        model_config=SimpleNamespace(max_model_len=4096),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
    )
    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(
            vllm_config=cfg, engine_core=SimpleNamespace(
                shutdown=lambda: None, engine_core=SimpleNamespace(scheduler=SimpleNamespace())
            )
        ),
        collective_rpc=lambda callback, **kw: (
            [{"batch_invariant_effective": True, "s2_capacity": {}, "s1_effective_capacity": {}}]
            * 2
            if callback is fixed_runtime.target_startup
            else []
        ),
    )
    monkeypatch.setattr(fixed_runtime, "make_engine", lambda *a, **kw: llm)
    publish(tmp_path / "draft-startup.json", {"s2_capacity": {}})
    result = fixed_runtime.run(tmp_path, None, tmp_path, point("target"), probe=True)
    assert calls == ["shutdown"] and result["draft_shutdown"]["shutdown"]
    assert read_json(tmp_path / "measurement-snapshot.json")["sample_count"] == 0
    assert read_json(tmp_path / "drain-state.json")["status"] == "COMPLETE"


def test_final_committed_terminal_sync_has_no_continuation_or_duplicate_release(owner):
    machine = owner.make()
    machine.initialize("a", (1, 2, 3), token_prefix_hash((1, 2, 3)))
    activate(owner)
    machine.batch_propose(
        [
            dict(
                request_id="a",
                round_id=0,
                committed_prefix_len=3,
                committed_prefix_hash=token_prefix_hash((1, 2, 3)),
                remaining_output_budget=8,
                eos_token_ids=[999],
            )
        ]
    )
    proposal_forwards = machine.backend.metrics.forwards["proposal"]
    owner.states["a"]["state"] = "FINISHED"
    publish(owner.path, {"barrier_ns": 1, "requests": owner.states})
    p = payload(final=(1, 2, 3, 999), round_id=1, version=2, terminal=True)
    result = machine.diagnostic_settle(p)
    assert result["committed_sync_tokens"] == 1 and result["disposition"] == "NATURAL_TERMINAL"
    assert machine.requests["a"].committed_token_ids == (1, 2, 3, 999)
    assert machine.requests["a"].finished
    assert machine.backend.metrics.forwards["proposal"] == proposal_forwards
    assert len(machine.release_receipts) == 1 and not machine.backend.worker.memory
    assert machine.diagnostic_settle(p) == result
    p["committed_prefix"] = [1, 2, 3, 999, 999]
    with pytest.raises(DataError, match="changed repeated"):
        machine.diagnostic_settle(p)
    assert machine.shutdown()["shutdown"]
