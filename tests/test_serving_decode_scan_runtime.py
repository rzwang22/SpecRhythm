"""Scan coordinator -> actual Draft state machine -> bounded settlement/shutdown.

GPU model/engine/transport and monotonic time are substitutes. Identity, proposal,
commit, cancellation, pool ownership and shutdown rejection are real code.
"""

import time
from types import SimpleNamespace

import pytest
from test_serving_fixed import clock_fixture
from test_serving_s2_runtime import PoolWorker

from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.transport import CheckpointJsonl
from specrhythm.serving import fixed_draft, fixed_runtime, s2_draft
from specrhythm.serving.common import read_json
from specrhythm.serving.decode_scan_plan import manifest, options, selected_point
from specrhythm.serving.fixed_artifacts import record_error
from specrhythm.serving.fixed_results import settlement_checks
from specrhythm.serving.s2_pool import publish


@pytest.mark.parametrize(
    "case",
    ["serial", "target", "probe", "refill", "exhausted", "stop", "release_failure", "partial"],
)
def test_scan_time_stop_real_proposal_settlement_and_prepared_pool(tmp_path, monkeypatch, case):
    clock, ids = clock_fixture(360)
    definitions = list(clock.definitions.values())
    # Use the original request constructor with a larger legal CPU test output budget.
    from dataclasses import replace

    from specrhythm.serving.s1_workload import ResidentServingRequest

    definitions = [
        ResidentServingRequest(replace(r.source, maximum_new_tokens=64)) for r in definitions
    ]
    opts = options(warmup_steps=2, window_seconds=30, drain_timeout=60)
    mode = "target" if case == "target" else "serial"
    point = selected_point(mode, 16)
    m = manifest({"eos_token_ids": [999], "git_commit": "a" * 40}, ids, "sha", opts, 16)
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
    released = []

    class Worker(PoolWorker):
        def release(self, rows):
            if case == "release_failure":
                raise RuntimeError("primary release failure")
            assert not set(rows) & set(released)
            released.extend(rows)
            return super().release(rows)

    worker = Worker()
    backend = s2_draft.S2DraftBackend(SimpleNamespace(max_model_len=4096), worker=worker)
    machine = fixed_draft.diagnostic_serial_machine(
        backend, report_path=tmp_path / "draft-backend-report.json"
    )
    server = fixed_draft.DiagnosticSerialServer(
        tmp_path / "unused.sock", machine, event_log=CheckpointJsonl(tmp_path / "work.jsonl")
    )
    warm = {}
    for r in definitions:
        prefix = r.prompt_token_ids + (10,)
        machine.initialize(r.request_id, prefix, token_prefix_hash(prefix))
        warm[r.request_id] = SimpleNamespace(
            prefix_version=1,
            logical_committed_prefix_token_ids=prefix,
            logical_committed_prefix_count=len(prefix),
            logical_committed_prefix_sha256=token_prefix_hash(prefix),
        )
    prepared_rows = backend.physical_rows()
    ticks = [time.monotonic_ns()]

    def now():
        ticks[0] += 1000
        return ticks[0]

    monkeypatch.setattr(time, "monotonic_ns", now)
    monkeypatch.setattr(time, "monotonic", lambda: ticks[0] / 1e9)
    calls, settled = [], []

    class Client:
        timeout_seconds = 60

        def call(self, operation, payload):
            calls.append(operation)
            if operation == "diagnostic_settle":
                assert (tmp_path / "measurement-snapshot.json").exists()
                settled.append(payload)
            if operation == "synchronize_and_batch_propose" and len(calls) > 1:
                # Refill management is within the real measurement denominator.
                ticks[0] += 100_000_000
            result = server._dispatch(operation, payload)
            return {**result, "service_send_ns": ticks[0], "transport_end_ns": ticks[0]}

    scheduler = SimpleNamespace(
        s2_steps=[], requests=dict.fromkeys(ids), s2_pool=SimpleNamespace(report=lambda: {})
    )
    generated = {rid: [10] for rid in ids}
    admissions, executed = [], []

    class Engine:
        def step(self):
            packet = read_json(tmp_path / "s2-control.json")
            active = [rid for rid, r in packet["requests"].items() if r["state"] == "ACTIVE"]
            assert len(active) == 16
            for rid in active:
                if rid not in admissions:
                    # Refill retains original private physical KV, no fresh setup call.
                    assert (
                        backend.physical_rows()[rid]["block_ids"]
                        == prepared_rows[rid]["block_ids"]
                    )
                    assert (
                        machine.requests[rid].committed_token_ids
                        == warm[rid].logical_committed_prefix_token_ids
                    )
                    admissions.append(rid)
            if case == "partial" and len(executed) == 3:
                from specrhythm.serving.decode_scan_window import ScanShapeStop

                raise ScanShapeStop(16, {"B": 15, "cohort": None, "request_ids": active[:-1]})
            terminal = case in ("refill", "exhausted") and (
                len(executed) == 2 or case == "exhausted"
            )
            sync, proposals = [], []
            for rid in active:
                if mode == "serial":
                    state = machine.requests[rid]
                    assert state.pending_proposal is not None
                    prefix = state.committed_token_ids + ((999,) if terminal else (777,))
                    sync.append(
                        dict(
                            request_id=rid,
                            round_id=state.pending_proposal.round_id,
                            committed_delta=[prefix[-1]],
                            committed_prefix_hash=token_prefix_hash(prefix),
                            terminal=terminal,
                        )
                    )
                    if not terminal:
                        proposals.append(
                            dict(
                                request_id=rid,
                                round_id=state.pending_proposal.round_id + 1,
                                committed_prefix_len=len(prefix),
                                committed_prefix_hash=token_prefix_hash(prefix),
                                remaining_output_budget=50,
                                eos_token_ids=[999],
                            )
                        )
                generated[rid].append(999 if terminal else 777)
            if mode == "serial":
                machine.synchronize_and_batch_propose(sync, proposals)
            ticks[0] += 2_000_000_000 if case != "exhausted" else 100_000_000
            scheduler.s2_steps.append(
                dict(
                    B=16,
                    request_ids=active,
                    cohort=None,
                    rows=[
                        dict(request_id=rid, candidate_positions=4 if mode == "serial" else 0)
                        for rid in active
                    ],
                )
            )
            executed.append(active)
            if terminal:
                for rid in active:
                    del scheduler.requests[rid]
            if case == "stop" and len(executed) == 3:
                publish(tmp_path / "stop-request.json", {"reason": "operator_stop"})
            return [
                SimpleNamespace(
                    request_id=rid,
                    finished=terminal,
                    outputs=[
                        SimpleNamespace(
                            token_ids=generated[rid], finish_reason="stop" if terminal else None
                        )
                    ],
                )
                for rid in active
            ]

        def abort_request(self, rows):
            assert set(rows) == set(scheduler.requests)
            scheduler.requests.clear()

        def has_unfinished_requests(self):
            return bool(scheduler.requests)

    monkeypatch.setattr(
        fixed_runtime,
        "prepare_resident",
        lambda *a, **kw: (
            Engine(),
            scheduler,
            Client(),
            warm,
            {rid: {"token": 10, "terminal": False} for rid in ids},
            None,
        ),
    )
    llm = SimpleNamespace(collective_rpc=lambda *a, **kw: [])
    try:
        if case == "release_failure":
            with pytest.raises(RuntimeError, match="primary release"):
                fixed_runtime.drive(llm, m, definitions, tmp_path, point, opts)
            record_error(tmp_path, RuntimeError("secondary cleanup"), "cleanup")
            assert (
                read_json(tmp_path / "diagnostic-primary-error.json")["error"]
                == "primary release failure"
            )
            snap = read_json(tmp_path / "measurement-snapshot.json")
            assert snap["committed_window_tokens"] > 12 * 16
            assert not snap["formal_comparison_eligible"] and not snap["drain_complete"]
            return
        result = fixed_runtime.drive(
            llm, m, definitions, tmp_path, point, opts, probe=case == "probe"
        )
        expected = {
            "stop": "operator_stop",
            "exhausted": "pool_exhausted_before_window",
            "partial": "partial_batch_prevented",
            "probe": "capacity_probe",
        }.get(case, "time_budget")
        assert result["stop_reason"] == expected
        snap = read_json(tmp_path / "measurement-snapshot.json")
        assert snap["committed_window_tokens"] == 16 * result["sample_count"]
        assert len(released) == 360 and not worker.memory and not backend.states
        assert read_json(tmp_path / "draft-backend-report.json")["backend_shutdown_complete"]
        releases = {
            e["request_id"]: e["timestamp_ns"]
            for e in result["events"]
            if e["event"] in ("resources-released", "cancelled-resources-released")
        }
        # Final report overhead occurs after drain, with real monotonic ordering.
        settlement_checks(
            result, {r.request_id: r for r in definitions}, backend.report(), releases
        )
        if case in ("serial", "target", "refill"):
            assert result["sample_count"] > 12 and result["warmup_steps"] == 2
            assert snap["window_ms"] >= 30000
            assert sum(len(r["generated_token_ids"]) - 1 for r in result["requests"]) == 16 * (
                result["sample_count"] + 2
            )
            if case == "refill":
                assert admissions == ids[:32]
                assert snap["window_ms"] > result["sample_count"] * 2000
        if case == "probe":
            assert not executed and result["measurement_start_ns"] is None
            assert not any(r.get("admission_ns") is not None for r in result["requests"])
        if case == "partial":
            assert result["decode_scan"]["rejected_step"]["model_forward_issued"] is False
        for payload in settled:
            server._dispatch("diagnostic_settle", payload)
        assert len(released) == 360
        assert not any(s.pending_proposal for s in machine.requests.values())
    finally:
        if not backend.closed:
            backend.shutdown()
