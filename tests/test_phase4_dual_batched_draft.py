from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from test_phase4_serial import phase4_config as _config_fixture
from test_phase4_vllm_draft import FakeWorker, next_token

from specrhythm.phase4 import dual_batched_draft as integration
from specrhythm.phase4.draft_batch import DraftCommitPlan
from specrhythm.phase4.dual_service import DualDraftClient, run_dual_draft_service
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.transport import CheckpointJsonl
from specrhythm.phase4.vllm_draft_backend import VllmBatchedDraftBackend

phase4_config = _config_fixture


def initial(rid, prefix=(1, 2, 3), remaining=10, eos=()):
    return {
        "request_id": rid,
        "committed_token_ids": list(prefix),
        "prefix_version": 1,
        "prefix_token_sha256": token_prefix_hash(prefix),
        "remaining_output_budget": remaining,
        "eos_token_ids": list(eos),
        "terminal": False,
        "measurement_start_ns": 1,
    }


def wait(controller, count):
    deadline = time.monotonic() + 5
    ready = []
    while len(ready) < count and time.monotonic() < deadline:
        response = controller.poll_ready(count - len(ready))
        assert not response["failures"], response["failures"]
        ready.extend(response["ready"])
        time.sleep(0.001)
    assert len(ready) == count
    return ready


def commit_row(machine, proposal, delta, *, terminal=False, remaining=5):
    rid = proposal["request_id"]
    state = machine.requests[rid]
    return {
        "request_id": rid,
        "round_id": proposal["round_id"],
        "proposal_id": proposal["proposal_id"],
        "committed_delta": list(delta),
        "prefix_version": state.prefix_version + 1,
        "prefix_token_sha256": token_prefix_hash(state.committed_token_ids + tuple(delta)),
        "terminal": terminal,
        "remaining_output_budget": remaining,
        "eos_token_ids": [],
    }


@pytest.fixture
def machine(phase4_config):
    backend = VllmBatchedDraftBackend(phase4_config, worker=FakeWorker())
    value = integration.BatchedDualDraftMachine(backend)
    yield value
    backend.shutdown()


def test_dual_dispatch_uses_production_backend_without_loading_hf(
    monkeypatch, phase4_config, tmp_path
):
    monkeypatch.setenv("SR_PHASE4_DRAFT_BACKEND", "vllm-batched")
    calls = []
    monkeypatch.setattr(integration, "serve_vllm_dual", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(
        "specrhythm.phase4.dual_service.HFPersistentDraftBackend",
        lambda *a: pytest.fail("second HF model was constructed"),
    )
    run_dual_draft_service(
        phase4_config,
        socket_path=tmp_path / "s",
        event_log_path=tmp_path / "e",
        transport_log_path=tmp_path / "t",
        ready_path=tmp_path / "r",
    )
    assert len(calls) == 1 and calls[0][0][0] is phase4_config


def test_batch_proposals_preserve_multiple_ids_and_independent_tokens(machine):
    rows = [initial("a"), initial("b", (7, 8)), initial("c", (1, 9, 8, 7))]
    for row in rows:
        machine.initialize(row)
    results = machine.execute_batch("propose_only", rows)
    assert [r["request_id"] for r in results] == ["a", "b", "c"]
    for row, result in zip(rows, results):
        prefix, expected = tuple(row["committed_token_ids"]), []
        for _ in range(4):
            expected.append(next_token(prefix + tuple(expected)))
        assert result["proposal"]["proposal_token_ids"] == expected
    calls = [rows for purpose, rows in machine.backend.worker.calls if purpose == "proposal"]
    assert len(calls) == 3 and all(len(rows) == 3 for rows in calls)
    assert len({r["draft_cohort_id"] for r in results}) == 1


def test_commit_mapping_correction_bonus_and_terminal_truncation(machine):
    rows = [initial("a"), initial("b", (4, 5)), initial("c", (7, 8))]
    for row in rows:
        machine.initialize(row)
    first = machine.execute_batch("propose_only", rows)
    proposals = {r["request_id"]: r["proposal"] for r in first}
    a, b, c = [proposals[r] for r in ("a", "b", "c")]
    commits = [
        commit_row(machine, a, a["proposal_token_ids"][:1] + [999]),
        commit_row(machine, b, b["proposal_token_ids"] + [999]),
        commit_row(machine, c, c["proposal_token_ids"][:2], terminal=True),
    ]
    second = machine.execute_batch("commit_and_propose", commits)
    assert second[0]["correction_length"] == 1 and second[1]["bonus_length"] == 1
    assert second[2]["terminal"] and second[2]["proposal"] is None
    assert "c" not in machine.backend.states and "c" in machine.backend.retired
    for row, result in zip(commits[:2], second[:2]):
        assert result["draft_sync_complete_ns"] <= result["proposal"]["draft_start_ns"]
        assert result["prefix_token_sha256"] == row["prefix_token_sha256"]
    assert len([r for p, r in machine.backend.worker.calls if p == "commit"][-1]) == 2


def test_eos_and_proposal_free_terminal_tail_release_kv(machine):
    eos = next_token((1, 2, 3))
    rows = [initial("eos", eos=(eos,)), initial("tail", remaining=1)]
    for row in rows:
        machine.initialize(row)
    results = machine.execute_batch("propose_only", rows)
    assert results[0]["proposal"]["proposal_token_ids"] == [eos]
    assert results[1]["target_tail"] and results[1]["proposal"] is None
    machine.execute_batch(
        "commit_and_propose", [commit_row(machine, results[0]["proposal"], [eos], terminal=True)]
    )
    row = {
        "request_id": "tail",
        "committed_delta": [99],
        "prefix_version": 2,
        "prefix_token_sha256": token_prefix_hash((1, 2, 3, 99)),
        "terminal": True,
    }
    result = machine.execute_batch("finish_tail", [row])[0]
    assert result["logical_draft_kv_length"] == 4 and result["terminal"]
    assert not machine.backend.states and not machine.backend.worker.memory
    assert machine.backend.metrics.forwards["commit"] == 2


def test_commit_cohort_validated_before_any_mutation(machine):
    rows = [initial("a"), initial("b")]
    for row in rows:
        machine.initialize(row)
    proposals = machine.execute_batch("propose_only", rows)
    commits = [commit_row(machine, r["proposal"], [999], terminal=True) for r in proposals]
    commits[1]["prefix_token_sha256"] = "wrong"
    before = len(machine.backend.worker.calls)
    with pytest.raises(ValueError, match="hash"):
        machine.execute_batch("commit_and_propose", commits)
    assert len(machine.backend.worker.calls) == before
    assert all(s.next_round_id == 0 for s in machine.requests.values())


def test_owner_thread_single_model_original_cohorts_and_shutdown(phase4_config, tmp_path):
    constructed, threads = [], []
    entered, release = threading.Event(), threading.Event()

    class Worker(FakeWorker):
        def __init__(self):
            super().__init__()
            self.owner = threading.get_ident()

        def materialize(self, rows, purpose):
            threads.append(threading.get_ident())
            assert threads[-1] == self.owner
            if purpose == "proposal" and not entered.is_set():
                entered.set()
                assert release.wait(5)
            return super().materialize(rows, purpose)

        def shutdown(self):
            threads.append(threading.get_ident())
            assert threads[-1] == self.owner
            super().shutdown()

    def factory():
        backend = VllmBatchedDraftBackend(phase4_config, worker=Worker())
        constructed.append(backend)
        return integration.BatchedDualDraftMachine(backend, report_path=tmp_path / "backend.json")

    controller = integration.BatchedDualDraftController(
        factory, CheckpointJsonl(tmp_path / "events")
    )
    try:
        rows = [initial(str(i), (i + 1, 9)) for i in range(4)]
        for row in rows:
            controller.execute("initialize", row)
        controller.enqueue("propose_only", rows[:2])
        assert entered.wait(5)
        # A later message remains a distinct batch even while the worker is busy.
        controller.enqueue("propose_only", rows[2:])
        started = time.monotonic()
        assert controller.poll_ready(2)["blocking_on_draft_gpu"] is False
        assert time.monotonic() - started < 0.5
        with pytest.raises(ValueError, match="in flight"):
            controller.enqueue("propose_only", rows[:1])
        release.set()
        ready = wait(controller, 4)
        cohorts = {tuple(r["draft_cohort_request_ids"]) for r in ready}
        assert cohorts == {("0", "1"), ("2", "3")}
        commits = [
            commit_row(controller.machine, r["proposal"], [999], terminal=True) for r in ready
        ]
        controller.enqueue("commit_and_propose", commits)
    finally:
        release.set()
        result = controller.shutdown()
    assert len(constructed) == 1 and len(set(threads)) == 1
    assert threads[0] != threading.get_ident()
    assert result["shutdown"] and not result["inflight_request_ids"]
    assert not result["ready_request_ids"] and not result["claimed_request_ids"]
    report = json.loads((tmp_path / "backend.json").read_text())
    assert report["backend_shutdown_complete"] and report["draft_live_requests_final"] == 0
    assert report["execution_failed"] is False
    assert report["draft_batch_size_max"] >= 2


def test_failed_service_bind_still_closes_owner_backend(monkeypatch, phase4_config, tmp_path):
    backends = []

    def factory(config):
        backend = VllmBatchedDraftBackend(config, worker=FakeWorker())
        backends.append(backend)
        return backend

    monkeypatch.setattr(integration, "VllmBatchedDraftBackend", factory)
    monkeypatch.setattr(
        integration.DualDraftUnixServer,
        "serve",
        lambda self: (_ for _ in ()).throw(RuntimeError("bind failed")),
    )
    with pytest.raises(RuntimeError, match="bind failed"):
        integration.serve_vllm_dual(
            phase4_config,
            socket_path=tmp_path / "s",
            event_log_path=tmp_path / "e",
            transport_log_path=tmp_path / "t",
            ready_path=tmp_path / "ready.json",
        )
    assert len(backends) == 1 and backends[0].closed
    report = json.loads((tmp_path / "draft-backend-report.json").read_text())
    assert report["execution_failed"] and report["backend_shutdown_complete"]


@pytest.mark.parametrize("selected", [None, "hf-persistent"])
def test_dual_hf_dispatch_compatibility(monkeypatch, phase4_config, tmp_path, selected):
    import specrhythm.phase4.dual_service as service

    calls = []
    monkeypatch.delenv("SR_PHASE4_DRAFT_BACKEND", raising=False)
    if selected:
        monkeypatch.setenv("SR_PHASE4_DRAFT_BACKEND", selected)
    monkeypatch.setattr(service, "HFPersistentDraftBackend", lambda config: calls.append("hf"))
    monkeypatch.setattr(service, "AsyncDualDraftController", lambda *args: calls.append("scalar"))
    monkeypatch.setattr(service.DualDraftUnixServer, "serve", lambda self: calls.append("serve"))
    monkeypatch.setattr(integration, "serve_vllm_dual", lambda *a, **kw: pytest.fail("new path"))
    run_dual_draft_service(
        phase4_config,
        socket_path=tmp_path / "s",
        event_log_path=tmp_path / "e",
        transport_log_path=tmp_path / "t",
        ready_path=tmp_path / "r",
    )
    assert calls == ["hf", "scalar", "serve"]


@pytest.mark.parametrize("terminal,tail", [(False, (99,)), (True, ()), (False, ())])
def test_empty_proposal_extension_cannot_allow_nonterminal_or_empty_tail(machine, terminal, tail):
    machine.initialize(initial("tail"))
    plan = DraftCommitPlan(
        "tail",
        0,
        (1, 2, 3),
        (),
        0,
        tail,
        token_prefix_hash((1, 2, 3) + tail),
        terminal,
        len(tail),
        0,
    )
    before = len(machine.backend.worker.calls)
    with pytest.raises(ValueError, match="stale"):
        machine.backend.commit_many([plan])
    assert len(machine.backend.worker.calls) == before


def test_checkpoint_failure_still_allows_owner_cleanup(phase4_config, tmp_path):
    class BrokenLog:
        def append(self, row):
            raise OSError("checkpoint unavailable")

    def factory():
        backend = VllmBatchedDraftBackend(phase4_config, worker=FakeWorker())
        return integration.BatchedDualDraftMachine(backend, report_path=tmp_path / "backend.json")

    controller = integration.BatchedDualDraftController(factory, BrokenLog())
    with pytest.raises(RuntimeError, match="checkpoint unavailable"):
        controller.execute("initialize", initial("r"))
    result = controller.shutdown()
    assert result["failures"] and result["shutdown"]
    report = json.loads((tmp_path / "backend.json").read_text())
    assert report["execution_failed"] and report["backend_shutdown_complete"]
    assert not controller._thread.is_alive()


def test_existing_unix_rpc_contract_runs_batched_dual_to_natural_shutdown(
    monkeypatch, phase4_config
):
    # Keep Unix path below macOS's small sockaddr_un limit as well as Linux's.
    with TemporaryDirectory(prefix="sr6-", dir="/tmp") as temporary:
        root = Path(temporary)
        failures = []
        monkeypatch.setattr(
            integration,
            "VllmBatchedDraftBackend",
            lambda config: VllmBatchedDraftBackend(config, worker=FakeWorker()),
        )

        def serve():
            try:
                integration.serve_vllm_dual(
                    phase4_config,
                    socket_path=root / "s",
                    event_log_path=root / "events",
                    transport_log_path=root / "transport",
                    ready_path=root / "ready.json",
                )
            except Exception as error:
                failures.append(error)

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        deadline = time.monotonic() + 5
        while not (root / "ready.json").exists() and time.monotonic() < deadline:
            assert not failures
            time.sleep(0.001)
        client = DualDraftClient(root / "s")
        try:
            rows = [initial("a"), initial("b", (4, 5))]
            for row in rows:
                client.call("execute", {"work_operation": "initialize", "row": row})
            client.call("enqueue", {"work_operation": "propose_only", "rows": rows})
            ready = []
            while len(ready) < 2 and time.monotonic() < deadline:
                response = client.call("poll_ready", {"limit": 2 - len(ready)})
                assert not response["failures"]
                ready.extend(response["ready"])
                time.sleep(0.001)
            assert len(ready) == 2
            assert ready[0]["draft_cohort_id"] == ready[1]["draft_cohort_id"]
            prefixes = {r["request_id"]: tuple(r["committed_token_ids"]) for r in rows}
            commits = []
            for result in ready:
                proposal = result["proposal"]
                rid = result["request_id"]
                commits.append(
                    {
                        "request_id": rid,
                        "proposal_id": proposal["proposal_id"],
                        "round_id": proposal["round_id"],
                        "prefix_version": 2,
                        "prefix_token_sha256": token_prefix_hash(prefixes[rid] + (999,)),
                        "committed_delta": [999],
                        "terminal": True,
                    }
                )
            client.call("enqueue", {"work_operation": "commit_and_propose", "rows": commits})
        finally:
            shutdown = client.call("shutdown", {})
            thread.join(timeout=5)
        assert not thread.is_alive() and not failures
        assert shutdown["shutdown"] and not shutdown["failures"]
        report = json.loads((root / "draft-backend-report.json").read_text())
        assert report["draft_retired_request_count"] == 2
        assert report["draft_live_requests_final"] == 0
        assert report["draft_batch_size_max"] == 2
        assert not (root / "s").exists()
