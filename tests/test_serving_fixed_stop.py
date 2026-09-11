"""Real fixed coordinator -> Serial server -> state-machine shutdown regression.

The fallback factories keep the sample reproducer executable on 89a9621: its
actual shutdown rejects 64 pending proposals. Only GPU computation/Target engine
and transport are substituted; shutdown is never a successful stub.
"""

import time
from types import SimpleNamespace

import pytest
from test_serving_fixed import clock_fixture
from test_serving_s2_runtime import PoolWorker

from specrhythm.phase4.batched_draft_service import BatchedDraftStateMachine
from specrhythm.phase4.draft_service import DraftUnixServer
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.transport import CheckpointJsonl
from specrhythm.serving import fixed_draft, fixed_runtime, s2_draft
from specrhythm.serving.common import read_json
from specrhythm.serving.fixed_plan import build_manifest, point, settings
from specrhythm.serving.s2_pool import publish


@pytest.mark.parametrize(
    "case",
    [
        "sample",
        "time_before_verify",
        "time_after_step",
        "unsynced",
        "operator",
        "operator_after_step",
        "release_failure",
        "rpc_failure",
        "commit_failure",
        "initial",
        "target",
    ],
)
def test_serial_sample_stop_settles_real_pending_proposals(tmp_path, monkeypatch, case):
    clock, ids = clock_fixture()
    definitions = list(clock.definitions.values())
    options = settings(warmup_steps=0, samples=1, window_seconds=5, drain_timeout=5)
    if case in ("operator_after_step", "time_after_step"):
        options["samples"] = 2
    mode = "target" if case == "target" else "serial"
    selected = point(mode)
    active = ids[:64]
    if case == "initial":
        selected.update(kind="initial-state", batch=32)
        active = ids[:32]
    manifest = build_manifest(
        {"eos_token_ids": [999], "git_commit": "a" * 40}, ids, "sha", options
    )
    monkeypatch.setenv("SR_S2_CONTROL", str(tmp_path / "s2-control.json"))
    monkeypatch.setenv("SR_S2_RUN_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("SR_S2_MODE", mode)
    monkeypatch.setattr(s2_draft, "cuda_memory", lambda *a: {"free_memory_bytes": 10**9})
    publish(
        tmp_path / "s2-control.json",
        {
            "barrier_ns": None,
            "requests": {rid: {"state": "STAGED", "cohort": None} for rid in ids},
        },
    )
    releases, settlement_calls = [], []

    class Worker(PoolWorker):
        def release(self, released):
            if case == "release_failure":
                raise RuntimeError("primary physical release failed")
            assert not set(released) & set(releases), "double physical release"
            releases.extend(released)
            return super().release(released)

    worker = Worker()
    backend = s2_draft.S2DraftBackend(SimpleNamespace(max_model_len=4096), worker=worker)
    factory = getattr(fixed_draft, "diagnostic_serial_machine", BatchedDraftStateMachine)
    machine = factory(backend, report_path=tmp_path / "draft-backend-report.json")
    server_class = getattr(fixed_draft, "DiagnosticSerialServer", DraftUnixServer)
    server = server_class(
        tmp_path / "unused.sock", machine, event_log=CheckpointJsonl(tmp_path / "work.jsonl")
    )
    for r in definitions:
        prefix = r.prompt_token_ids + (10,)
        machine.initialize(r.request_id, prefix, token_prefix_hash(prefix))
    warm = {
        r.request_id: SimpleNamespace(
            prefix_version=1,
            logical_committed_prefix_token_ids=r.prompt_token_ids + (10,),
            logical_committed_prefix_count=len(r.prompt_token_ids) + 1,
            logical_committed_prefix_sha256=token_prefix_hash(r.prompt_token_ids + (10,)),
        )
        for r in definitions
    }

    class Client:
        timeout_seconds = 5

        def call(self, operation, payload):
            if operation == "diagnostic_settle":
                # Must already exist BEFORE even the first risky drain RPC.
                snapshot = read_json(tmp_path / "measurement-snapshot.json")
                assert (
                    not snapshot["drain_complete"] and not snapshot["formal_comparison_eligible"]
                )
                settlement_calls.append(payload)
                if case == "rpc_failure":
                    raise RuntimeError("primary settlement RPC failed")
            result = server._dispatch(operation, payload)
            now = time.monotonic_ns()
            return {**result, "service_send_ns": now, "transport_end_ns": now}

    scheduler = SimpleNamespace(
        s2_steps=[],
        requests={r: object() for r in ids},
        s2_pool=SimpleNamespace(report=lambda: {}),
    )
    proposal_forwards_at_abort = []

    class Engine:
        def step(self):
            assert case not in ("operator", "time_before_verify")
            if mode == "serial":
                assert sum(
                    s.pending_proposal is not None for s in machine.requests.values()
                ) == len(active)
                synchronizations, next_rows = [], []
                for rid in active:
                    state = machine.requests[rid]
                    prefix = state.committed_token_ids + (777,)
                    synchronizations.append(
                        dict(
                            request_id=rid,
                            round_id=0,
                            committed_delta=[777],
                            committed_prefix_hash=token_prefix_hash(prefix),
                            terminal=False,
                        )
                    )
                    next_rows.append(
                        dict(
                            request_id=rid,
                            round_id=1,
                            committed_prefix_len=len(prefix),
                            committed_prefix_hash=token_prefix_hash(prefix),
                            remaining_output_budget=2,
                            eos_token_ids=[999],
                        )
                    )
                if case != "unsynced":
                    machine.synchronize_and_batch_propose(synchronizations, next_rows)
            scheduler.s2_steps.append(
                dict(
                    request_ids=active,
                    rows=[
                        dict(request_id=r, candidate_positions=0 if mode == "target" else 3)
                        for r in active
                    ],
                    B=len(active),
                    cohort=None,
                )
            )
            if case == "operator_after_step":
                publish(tmp_path / "stop-request.json", {"reason": "operator_stop"})
            return [
                SimpleNamespace(
                    request_id=r,
                    finished=False,
                    outputs=[SimpleNamespace(token_ids=[10, 777], finish_reason=None)],
                )
                for r in active
            ]

        def abort_request(self, cancelled):
            assert set(cancelled) == set(ids)
            proposal_forwards_at_abort.append(backend.metrics.forwards["proposal"])
            scheduler.requests.clear()

        def has_unfinished_requests(self):
            return bool(scheduler.requests)

    if case == "operator":
        publish(tmp_path / "stop-request.json", {"reason": "operator_stop"})
    if case.startswith("time_"):
        checks = []

        def expired(window, now):
            checks.append(now)
            if len(checks) >= (2 if case == "time_before_verify" else 3):
                window.reason = "time_budget"
                return True
            return False

        monkeypatch.setattr(fixed_runtime.Window, "time_expired", expired)
    if case == "commit_failure":
        commit = fixed_runtime.ServingClock.commit

        def failing_commit(self, rid, *args, **kw):
            if rid == active[1]:
                raise RuntimeError("primary output publication failed")
            return commit(self, rid, *args, **kw)

        monkeypatch.setattr(fixed_runtime.ServingClock, "commit", failing_commit)
    monkeypatch.setattr(
        fixed_runtime,
        "prepare_resident",
        lambda *a, **kw: (
            Engine(),
            scheduler,
            Client(),
            warm,
            {r: {"token": 10, "terminal": False} for r in ids},
            None,
        ),
    )
    llm = SimpleNamespace(collective_rpc=lambda callback, **kw: [])
    failed = case in ("release_failure", "rpc_failure", "commit_failure")
    try:
        if failed:
            with pytest.raises(RuntimeError, match="primary"):
                fixed_runtime.drive(llm, manifest, definitions, tmp_path, selected, options)
            snapshot = read_json(tmp_path / "measurement-snapshot.json")
            assert snapshot["committed_window_tokens"] == (1 if case == "commit_failure" else 64)
            assert snapshot["completed_steps"] == 1
            assert not snapshot["drain_complete"]
            assert "primary" in read_json(tmp_path / "diagnostic-primary-error.json")["error"]
            rows = read_json(tmp_path / "arrival-output-events.json")["requests"]
            assert not any(r["resources_released"] for r in rows)
            assert worker.memory and not backend.retired and not releases
            assert any(s.pending_proposal for s in machine.requests.values())
            return
        result = fixed_runtime.drive(llm, manifest, definitions, tmp_path, selected, options)
        expected_reason = (
            "operator_stop"
            if case.startswith("operator")
            else "time_budget"
            if case.startswith("time_")
            else "sample_budget"
        )
        assert result["stop_reason"] == expected_reason
        tokens = 0 if case in ("operator", "time_before_verify") else len(active)
        assert result["sample_count"] == int(tokens > 0)
        assert all(r["state"] == "DIAGNOSTIC_CANCELLED" for r in result["requests"])
        assert not any("completion_ns" in r or r.get("finish_reason") for r in result["requests"])
        assert all(r["resources_released"] for r in result["requests"])
        assert sum(len(r["generated_token_ids"]) - 1 for r in result["requests"]) == tokens
        assert not any(s.pending_proposal for s in machine.requests.values())
        assert not any(s.finished for s in machine.requests.values()), "cancellation is not EOS"
        assert backend.metrics.forwards["proposal"] == proposal_forwards_at_abort[-1]
        assert not worker.memory and not backend.states and len(releases) == 100
        assert read_json(tmp_path / "draft-backend-report.json")["backend_shutdown_complete"]
        receipt = result["diagnostic_drain"]
        assert receipt["settled_requests"] == 100
        assert sum(r["committed_sync_tokens"] for r in receipt["receipts"]) == (
            64 if case == "unsynced" else 0
        )
        assert result["end_ns"] >= receipt["end_ns"] > result["measurement_end_ns"]
        snapshot = read_json(tmp_path / "measurement-snapshot.json")
        assert snapshot["committed_window_tokens"] == tokens and snapshot["drain_complete"]
        assert snapshot["peak_held_slots"] <= len(active)
        from specrhythm.serving.fixed_results import settlement_checks

        logical_releases = {
            e["request_id"]: e["timestamp_ns"]
            for e in result["events"]
            if e["event"] == "cancelled-resources-released"
        }
        settlement_checks(
            result, {r.request_id: r for r in definitions}, backend.report(), logical_releases
        )
        # Repeat the actual protocol with exactly the same authority, after real shutdown.
        for payload in settlement_calls:
            server._dispatch("diagnostic_settle", payload)
        assert len(releases) == 100
    finally:
        if not backend.closed:
            backend.shutdown()  # Emergency fixture teardown is never asserted as a valid drain.
