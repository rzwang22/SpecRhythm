from __future__ import annotations

import json
import random
from collections import Counter, namedtuple
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_phase4_serial import phase4_config as _config_fixture

from specrhythm.phase4.batched_draft_service import (
    BatchedDraftStateMachine,
    write_immutable_report,
)
from specrhythm.phase4.draft_batch import (
    DraftCommitPlan,
    DraftMaterialization,
    DraftProposalPlan,
    commit_frontier,
    mapped_rows,
)
from specrhythm.phase4.draft_metrics import DraftMetrics, batch_statistics
from specrhythm.phase4.draft_service import HFPersistentDraftBackend, _HFRequest
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.vllm_draft_backend import VllmBatchedDraftBackend, selected_draft_backend
from specrhythm.phase4.vllm_draft_worker import (
    API_PATH,
    EXECUTE_FIELDS,
    DraftOnlyForwardAdapter,
    audit_source_files,
    complete_draft_forward,
    validate_placement,
)

phase4_config = _config_fixture


def next_token(prefix):
    return (sum((i + 1) * token for i, token in enumerate(prefix)) + len(prefix) ** 2) % 79 + 10


class FakeWorker:
    """Independent physical KV array with stale tail retained after rejection."""

    provenance = {"backend": "CPU-fake", "physical_gpu_id": 0}

    def __init__(self):
        self.memory = {}
        self.calls = []
        self.closed = False
        self.inject_failure = False

    def materialize(self, rows, purpose):
        if self.inject_failure:
            raise RuntimeError("injected forward failure")
        self.calls.append((purpose, tuple(rows)))
        output = {}
        for row in reversed(rows):  # Deliberately different model output order.
            memory = self.memory.setdefault(row.request_id, [])
            assert tuple(memory[: row.valid_length]) == row.context[: row.valid_length]
            memory.extend([-999] * max(0, len(row.context) - len(memory)))
            memory[row.valid_length : len(row.context)] = row.suffix
            output[row.request_id] = next_token(memory[: len(row.context)])
        self.metrics.forward(purpose, len(rows), sum(len(row.suffix) for row in rows))
        return output

    def greedy(self, logits):
        self.metrics.syncs["bulk_token_d2h"] += 1
        return tuple(logits)

    def fence(self, reason):
        self.metrics.syncs[reason] += 1

    def request_evidence(self, request_id):
        return {"draft_internal_request_id": request_id, "physical_observable": True}

    def release(self, request_ids):
        for rid in request_ids:
            del self.memory[rid]

    def resource_evidence(self):
        return {"live_allocator_requests": len(self.memory), "closed": self.closed}

    def shutdown(self):
        self.memory.clear()
        self.closed = True


def backend(config, count=1):
    worker = FakeWorker()
    value = VllmBatchedDraftBackend(config, worker=worker)
    prefixes = {f"req-{i}": tuple(range(1, i + 3)) for i in range(count)}
    value.initialize_many(tuple(prefixes.items()))
    return value, worker, prefixes


def plans(prefixes, *, round_id=0, budget=4):
    return [
        DraftProposalPlan(rid, round_id, prefix, budget, ()) for rid, prefix in prefixes.items()
    ]


def commit(rid, prefix, proposal, accepted, tail, *, round_id=0, terminal=False):
    final = prefix + proposal[:accepted] + tuple(tail)
    correction = int(bool(tail) and accepted < len(proposal))
    return DraftCommitPlan(
        rid,
        round_id,
        prefix,
        proposal,
        accepted,
        tuple(tail),
        token_prefix_hash(final),
        terminal,
        correction,
        int(bool(tail)) - correction,
    )


@pytest.mark.parametrize("value", ["", "VLLM", "auto", "cached", "hf"])
def test_unknown_backend_rejected(value):
    with pytest.raises(ValueError, match="hf-persistent or vllm-batched"):
        selected_draft_backend({"SR_PHASE4_DRAFT_BACKEND": value})


def test_backend_selection_is_explicit():
    assert selected_draft_backend({}) == "hf-persistent"
    assert selected_draft_backend({"SR_PHASE4_DRAFT_BACKEND": "vllm-batched"}) == "vllm-batched"


@pytest.mark.parametrize("mode", ["hf-persistent", "vllm-batched"])
def test_serial_factory_routes_only_explicit_backend(monkeypatch, phase4_config, tmp_path, mode):
    import specrhythm.phase4.batched_draft_service as batched
    import specrhythm.phase4.draft_service as service

    calls = []
    monkeypatch.setenv("SR_PHASE4_DRAFT_BACKEND", mode)
    monkeypatch.setattr(service, "HFPersistentDraftBackend", lambda cfg: calls.append("hf"))
    monkeypatch.setattr(service.DraftUnixServer, "serve", lambda self, path: calls.append("serve"))
    monkeypatch.setattr(batched, "serve_batched_draft", lambda cfg, **kw: calls.append("vllm"))
    service.run_draft_service(
        phase4_config,
        socket_path=tmp_path / "socket",
        event_log_path=tmp_path / "events",
        ready_path=tmp_path / "ready",
    )
    assert calls == (["hf", "serve"] if mode == "hf-persistent" else ["vllm"])


def test_service_failure_releases_worker_and_retains_immutable_startup(
    monkeypatch, phase4_config, tmp_path
):
    import specrhythm.phase4.draft_service as service
    import specrhythm.phase4.vllm_draft_backend as backend_module
    from specrhythm.phase4.batched_draft_service import serve_batched_draft

    value, worker, _ = backend(phase4_config)
    monkeypatch.setattr(backend_module, "VllmBatchedDraftBackend", lambda cfg: value)

    def fail(self, path):
        raise RuntimeError("bind/serve failure")

    monkeypatch.setattr(service.DraftUnixServer, "serve", fail)
    kwargs = dict(
        socket_path=tmp_path / "socket",
        event_log_path=tmp_path / "events",
        ready_path=tmp_path / "ready",
    )
    with pytest.raises(RuntimeError, match="bind/serve"):
        serve_batched_draft(phase4_config, **kwargs)
    assert worker.closed and value.closed
    report = json.loads((tmp_path / "draft-backend-report.json").read_text())
    assert report["execution_failed"] is True
    startup = (tmp_path / "draft-startup.json").read_bytes()
    with pytest.raises(FileExistsError):
        serve_batched_draft(phase4_config, **kwargs)
    assert (tmp_path / "draft-startup.json").read_bytes() == startup


@pytest.mark.parametrize("count", [2, 4, 8, 32, 100])
def test_real_step_shape_not_per_request_loop(phase4_config, count):
    value, worker, prefixes = backend(phase4_config, count)
    result = value.propose_many(plans(prefixes))
    calls = [rows for purpose, rows in worker.calls if purpose == "proposal"]
    assert [len(rows) for rows in calls] == [count] * 3  # first token uses cached logits
    assert sum(len(tokens) for tokens in result.values()) == 4 * count
    report = value.report()
    assert report["draft_model_forward_count"] == 3
    assert report["draft_batch_size_p50"] == count
    assert report["draft_autoregressive_step_count"] == 4
    assert report["draft_host_sync_count_by_reason"]["bulk_token_d2h"] == 4
    assert report["draft_scalar_item_count"] == 0
    assert all(
        tuple(row.context[: row.valid_length]) == prefixes[rid] + result[rid][:step]
        for step, rows in enumerate(calls)
        for rid, row in zip(prefixes, rows)
    )


def test_shrinking_eos_and_budget_masks(phase4_config):
    value, worker, prefixes = backend(phase4_config, 4)
    ids = tuple(prefixes)
    work = [
        DraftProposalPlan(
            rid, 0, prefixes[rid], budget, (next_token(prefixes[rid]),) if i == 0 else ()
        )
        for i, (rid, budget) in enumerate(zip(ids, [4, 2, 4, 0]))
    ]
    result = value.propose_many(work)
    assert [len(result[rid]) for rid in ids] == [1, 2, 4, 0]
    assert [len(rows) for purpose, rows in worker.calls if purpose == "proposal"] == [2, 1, 1]
    assert value.states[ids[3]].proposal is None
    assert value.report()["draft_active_rows_per_step"]["histogram"] == {"1": 2, "2": 1, "3": 1}


@pytest.mark.parametrize("accepted", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("terminal", [False, True])
def test_commit_valid_frontier_and_only_missing_tail(phase4_config, accepted, terminal):
    value, worker, prefixes = backend(phase4_config)
    rid, prefix = next(iter(prefixes.items()))
    proposal = value.propose_many(plans(prefixes))[rid]
    tail = () if terminal and accepted == 4 else (111,)
    plan = commit(rid, prefix, proposal, accepted, tail, terminal=terminal)
    evidence = value.commit_many((plan,))[rid]
    expected_r = min(len(prefix) + 3, len(prefix) + accepted)
    rows = worker.calls[-1][1]
    assert rows[0].valid_length == expected_r
    assert rows[0].suffix == plan.final_prefix[expected_r:]
    assert evidence["materialized_kv_length"] == len(plan.final_prefix)
    if terminal:
        assert not value.states and not worker.memory
    else:
        assert value.states[rid].prefix == plan.final_prefix
        assert value.states[rid].next_logits == next_token(plan.final_prefix)


def test_ragged_commit_is_one_model_call(phase4_config):
    value, worker, prefixes = backend(phase4_config, 4)
    proposals = value.propose_many(plans(prefixes))
    work = [
        commit(rid, prefix, proposals[rid], a, (111,))
        for (rid, prefix), a in zip(prefixes.items(), [0, 1, 3, 4])
    ]
    before = len(worker.calls)
    value.commit_many(work)
    assert len(worker.calls) == before + 1
    assert [len(r.suffix) for r in worker.calls[-1][1]] == [1, 1, 1, 2]
    assert all(value.states[p.request_id].next_logits == next_token(p.final_prefix) for p in work)


def test_empty_tail_refreshes_shortened_frontier(phase4_config):
    value, worker, prefixes = backend(phase4_config)
    rid, prefix = next(iter(prefixes.items()))
    proposal = value.propose_many(plans(prefixes))[rid]
    plan = commit(rid, prefix, proposal, 1, ())
    value.commit_many((plan,))
    assert worker.calls[-1][1][0].suffix == (proposal[0],)
    assert value.states[rid].next_logits == next_token(plan.final_prefix)


def test_hf_algorithm_is_reference_across_random_rounds(phase4_config):
    class Cache:
        def __init__(self, prefix):
            self.tokens = list(prefix)

        def crop(self, size):
            self.tokens = self.tokens[:size]

    value, _worker, prefixes = backend(phase4_config, 8)
    hf = HFPersistentDraftBackend.__new__(HFPersistentDraftBackend)
    hf.states = {
        rid: _HFRequest(Cache(prefix), next_token(prefix), len(prefix))
        for rid, prefix in prefixes.items()
    }
    hf.torch = SimpleNamespace(argmax=lambda x, dim: SimpleNamespace(item=lambda: x))

    def append(state, token):
        state.cache.tokens.append(token)
        return next_token(state.cache.tokens)

    hf._append_token = append
    random_source = random.Random(445)
    for round_id in range(12):
        proposed = value.propose_many(plans(prefixes, round_id=round_id))
        work = []
        for rid, prefix in prefixes.items():
            reference, forwards = hf.propose(rid, 4, ())
            assert reference == proposed[rid] and forwards == 3
            accepted = random_source.randrange(5)
            plan = commit(rid, prefix, reference, accepted, (111,), round_id=round_id)
            hf.rollback(rid, accepted)
            hf.append_target_token(rid, 111)
            assert tuple(hf.states[rid].cache.tokens) == plan.final_prefix
            work.append(plan)
        value.commit_many(work)
        prefixes = {p.request_id: p.final_prefix for p in work}


def test_invalid_whole_cohort_does_not_mutate(phase4_config):
    value, worker, prefixes = backend(phase4_config, 2)
    proposed = value.propose_many(plans(prefixes))
    work = [
        commit(rid, prefix, proposed[rid], 1, (111,), round_id=i)
        for i, (rid, prefix) in enumerate(prefixes.items())
    ]
    before = len(worker.calls)
    with pytest.raises(ValueError, match="stale"):
        value.commit_many(work)
    assert len(worker.calls) == before
    assert all(value.states[rid].prefix == prefix for rid, prefix in prefixes.items())


def test_worker_failure_poisoned_no_successful_commit(phase4_config):
    value, worker, prefixes = backend(phase4_config)
    proposal = value.propose_many(plans(prefixes))["req-0"]
    worker.inject_failure = True
    with pytest.raises(RuntimeError, match="injected"):
        value.commit_many([commit("req-0", prefixes["req-0"], proposal, 0, (111,))])
    assert value.states["req-0"].prefix == prefixes["req-0"]
    with pytest.raises(RuntimeError, match="failed"):
        value.propose_many(plans(prefixes))
    value.shutdown()
    assert not worker.memory


def proposal_row(rid, prefix, round_id=0, remaining=9, eos=()):
    return {
        "request_id": rid,
        "round_id": round_id,
        "committed_prefix_len": len(prefix),
        "committed_prefix_hash": token_prefix_hash(prefix),
        "remaining_output_budget": remaining,
        "eos_token_ids": eos,
    }


def test_serial_control_contract_batched_commits_and_immutable_report(phase4_config, tmp_path):
    worker = FakeWorker()
    value = VllmBatchedDraftBackend(phase4_config, worker=worker)
    report_path = tmp_path / "draft-backend-report.json"
    machine = BatchedDraftStateMachine(value, report_path=report_path)
    prefixes = {"a": (1, 2), "b": (4, 6, 8)}
    for rid, prefix in prefixes.items():
        machine.initialize(rid, prefix, token_prefix_hash(prefix))
    proposed = machine.batch_propose([proposal_row(rid, p) for rid, p in prefixes.items()])
    sync = []
    for row in proposed["proposals"]:
        rid = row["request_id"]
        delta = tuple(row["proposal_token_ids"]) + (111,)
        sync.append(
            {
                "request_id": rid,
                "round_id": 0,
                "committed_delta": delta,
                "committed_prefix_hash": token_prefix_hash(prefixes[rid] + delta),
                "terminal": True,
            }
        )
    response = machine.synchronize_and_batch_propose(sync, [])
    assert len(response["synchronizations"]) == 2
    assert [len(r.suffix) for r in worker.calls[-1][1]] == [2, 2]
    assert len(worker.calls[-1][1]) == 2
    with pytest.raises(ValueError, match="live proposal"):
        machine.synchronize_and_batch_propose(sync, [])
    machine.shutdown()
    report = json.loads(report_path.read_text())
    assert report["draft_model_forward_count"] == 4  # three proposal + one commit
    assert report["draft_live_requests_final"] == 0
    assert report["backend_shutdown_complete"]
    with pytest.raises(FileExistsError):
        write_immutable_report(report_path, {})


def test_scalar_wire_lifecycle_still_supported(phase4_config):
    worker = FakeWorker()
    value = VllmBatchedDraftBackend(phase4_config, worker=worker)
    machine = BatchedDraftStateMachine(value)
    prefix = (1, 2)
    machine.initialize("a", prefix, token_prefix_hash(prefix))
    proposed = machine.batch_propose([proposal_row("a", prefix)])["proposals"][0]
    delta = tuple(proposed["proposal_token_ids"]) + (111,)
    machine.synchronize_committed_prefix(
        "a", 0, delta, token_prefix_hash(prefix + delta), terminal=False
    )
    machine.rollback_rejected_suffix("a", 0)
    result = machine.append_target_correction_or_bonus("a", 0)
    assert result["materialized_kv_length"] == len(prefix + delta)
    assert value.states["a"].next_round == 1


def test_output_budget_one_produces_no_round(phase4_config):
    worker = FakeWorker()
    value = VllmBatchedDraftBackend(phase4_config, worker=worker)
    machine = BatchedDraftStateMachine(value)
    machine.initialize("a", (1, 2), token_prefix_hash((1, 2)))
    assert not machine.batch_propose([proposal_row("a", (1, 2), remaining=1)])["proposals"]
    assert len(worker.calls) == 1
    machine.finish("a")
    machine.shutdown()


@pytest.mark.parametrize("bad", [[], ["a", "a"], ["a", "missing"], ["a", "b", "capacity"]])
def test_runner_row_map_fails_closed(bad):
    with pytest.raises((ValueError, RuntimeError)):
        mapped_rows(["a", "b"], bad, list(range(len(bad))))


def test_runner_permuted_rows_map_by_identity():
    assert mapped_rows(["a", "b"], ["b", "a"], [2, 1]) == {"a": 1, "b": 2}


def test_metrics_do_not_claim_batch_from_setup():
    metrics = DraftMetrics()
    metrics.forward("setup", 100, 3000)
    metrics.forward("proposal", 1, 1)
    report = metrics.snapshot("test")
    assert report["draft_batch_size_p50"] == 1
    assert not report["true_batching_observed"]
    assert report["draft_model_forward_count"] == 1
    assert batch_statistics(Counter({1: 1, 2: 2, 8: 1}))["p50"] == 2
    assert batch_statistics(Counter())["p50"] is None


def fake_runner():
    class Logits:
        ndim = 2
        shape = (2, 3)

        def clone(self):
            return self

        def unbind(self, dim):
            return (20, 10)

    state_cls = namedtuple("ExecuteModelState", EXECUTE_FIELDS)
    state = state_cls(None, Logits(), None, None, None, None, None, None, None, None)
    return SimpleNamespace(
        use_async_scheduling=False,
        is_pooling_model=False,
        use_aux_hidden_state_outputs=False,
        routed_experts_initialized=False,
        uses_mrope=False,
        speculative_config=None,
        lora_config=None,
        execute_model_state=state,
        kv_connector_output=None,
        input_batch=SimpleNamespace(req_ids=["b", "a"]),
    )


def test_forward_completion_consumes_once_and_retains_row_map():
    runner = fake_runner()
    assert complete_draft_forward(runner, None, ["a", "b"]) == {"a": 10, "b": 20}
    assert runner.execute_model_state is None
    with pytest.raises(RuntimeError, match="completion state"):
        complete_draft_forward(runner, None, ["a", "b"])


@pytest.mark.parametrize(
    "field",
    [
        "use_async_scheduling",
        "routed_experts_initialized",
        "is_pooling_model",
        "uses_mrope",
        "lora_config",
    ],
)
def test_completion_rejects_unsupported_obligations(field):
    runner = fake_runner()
    setattr(runner, field, True)
    with pytest.raises(RuntimeError, match="unsupported"):
        complete_draft_forward(runner, None, ["a", "b"])
    assert runner.execute_model_state is not None


def test_completion_rejects_connector_and_unfinished_execute():
    runner = fake_runner()
    runner.kv_connector_output = object()
    with pytest.raises(RuntimeError, match="deferred"):
        complete_draft_forward(runner, None, ["a", "b"])
    worker = SimpleNamespace(model_runner=runner)
    with pytest.raises(RuntimeError, match="previous completion"):
        DraftOnlyForwardAdapter.sr_draft_materialize(worker, None)


@pytest.mark.parametrize("uuid", [None, "", "GPU-0", "rank-0", "GPU-00000000-0000-0000-0000"])
def test_no_guessed_uuid_allowed(phase4_config, uuid):
    with pytest.raises(RuntimeError, match="UUID/device-binding"):
        validate_placement(
            {
                "physical_gpu_id": 0,
                "logical_cuda_index": 0,
                "cuda_visible_devices": "0",
                "gpu_uuid": uuid,
            },
            phase4_config,
        )


def test_correct_real_identity_contract(phase4_config):
    validate_placement(
        {
            "physical_gpu_id": 0,
            "logical_cuda_index": 0,
            "cuda_visible_devices": "0",
            "gpu_uuid": "GPU-12345678-abcd-abcd-abcd-123456789abc",
        },
        phase4_config,
    )


def test_dual_rejects_new_backend_before_loading_model(monkeypatch, phase4_config, tmp_path):
    from specrhythm.phase4.dual_service import run_dual_draft_service

    monkeypatch.setenv("SR_PHASE4_DRAFT_BACKEND", "vllm-batched")
    with pytest.raises(ValueError, match="Serial-only"):
        run_dual_draft_service(
            phase4_config,
            socket_path=tmp_path / "socket",
            event_log_path=tmp_path / "events",
            transport_log_path=tmp_path / "t",
            ready_path=tmp_path / "ready",
        )


def test_impossible_frontier_and_hash_rejected():
    plan = commit("a", (1, 2), (3, 4), 1, (5,))
    with pytest.raises(ValueError, match="frontier"):
        commit_frontier(plan, 4)
    with pytest.raises(ValueError, match="hash"):
        DraftCommitPlan("a", 0, (1, 2), (3,), 1, (), "fake", True)
    with pytest.raises(ValueError, match="nonempty missing suffix"):
        DraftMaterialization("a", (1, 2), 2)


def test_api_inventory_is_packaged_and_includes_internal_seams():
    api = json.loads(API_PATH.read_text())
    assert api["vllm_commit"] == "752a3a504485790a2e8491cacbb35c137339ad34"
    runner = next(f for f in api["files"] if f["path"].endswith("/gpu_model_runner.py"))
    assert runner["base_sha256"] != runner["required_installed_sha256"]
    assert "_update_streaming_request" in json.dumps(runner["symbols"])
    assert "ExecuteModelState" in json.dumps(runner["symbols"])
    assert all(len(f["required_installed_sha256"]) == 64 for f in api["files"])


def test_missing_pinned_source_fails_closed(tmp_path):
    with pytest.raises(RuntimeError, match="source mismatch"):
        audit_source_files(tmp_path)


def test_importing_backend_never_imports_gpu_packages():
    import subprocess
    import sys

    code = (
        "import sys; import specrhythm.phase4.vllm_draft_backend; "
        "import specrhythm.phase4.vllm_draft_worker; "
        "assert 'torch' not in sys.modules and 'vllm' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_no_new_target_patch_files():
    api = json.loads(API_PATH.read_text())
    assert api["required_patch_state"] == "existing-five-patch-stack"
    assert not list(Path("integrations/vllm/patches").glob("0006*"))
