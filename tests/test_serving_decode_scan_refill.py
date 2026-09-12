"""Coordinator -> actual fixed/S2/Dual scheduler -> owner -> release/refill -> drain.

Only stock GPU allocation/forward, transport, setup hardware and time are replaced.
Owner batching, private KV materialization, proposal versions/FIFO claims and
normal shutdown's unresolved-proposal check all execute production code.
"""

import time
from types import SimpleNamespace

import pytest
from test_phase4_dual_batched_draft import commit_row, initial
from test_serving_fixed_identity import fixed_schedulers as _fixed
from test_serving_fixed_identity import s2_schedulers as _s2
from test_serving_s2_runtime import PoolWorker

from specrhythm.phase4.transport import CheckpointJsonl
from specrhythm.serving import fixed_runtime, s2_draft
from specrhythm.serving.common import read_json
from specrhythm.serving.decode_scan_plan import manifest, options, selected_point
from specrhythm.serving.fixed_settle import DiagnosticDualController, DiagnosticDualMachine
from specrhythm.serving.s2_pool import publish

s2_schedulers = _s2
fixed_schedulers = _fixed


@pytest.mark.parametrize(
    "case", ["refill", "warmup_refill", "wait_expiry", "operator_stop", "rpc_failure"])
def test_actual_scan_refill_owner_and_full_batch_wait(
    fixed_schedulers, tmp_path, monkeypatch, case
):
    import threading

    build, path = fixed_schedulers
    scheduler, _ = build("pingpong", "bound-prefix", batch=64, n=360,
                         ready_ids=[], bootstrap_only=True)
    ids = list(scheduler.requests)
    definitions = [SimpleNamespace(request_id=rid, prompt_token_ids=(1, int(rid) + 1),
                                   prompt_length=2, maximum_new_tokens=256) for rid in ids]
    refilling = case in ("refill", "warmup_refill")
    opts = options(warmup_steps=2 if refilling else 0, window_seconds=30,
                   drain_timeout=60)
    m = manifest({"eos_token_ids": [999], "git_commit": "a" * 40}, ids, "sha", opts, 64)
    path = tmp_path / "s2-control.json"
    monkeypatch.setenv("SR_S2_CONTROL", str(path))
    monkeypatch.setenv("SR_S2_RUN_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("SR_S2_MODE", "pingpong")
    monkeypatch.setattr(s2_draft, "cuda_memory", lambda *a: {"free_memory_bytes": 10**9})
    publish(path, {"barrier_ns": None,
                   "requests": {rid: {"state": "STAGED", "cohort": None} for rid in ids}})
    blocked, release = threading.Event(), threading.Event()
    release_terminal = threading.Event()
    released, executed, admits, progress_while_refill = [], [], [], []
    real_sleep = time.sleep
    ticks = [time.monotonic_ns()]

    def now():
        ticks[0] += 1000
        return ticks[0]

    def sleep(seconds):
        # Let the real owner thread progress; simulate a costly proposal wait.
        ticks[0] += 250_000_000
        real_sleep(0.001)

    monkeypatch.setattr(time, "monotonic_ns", now)
    monkeypatch.setattr(time, "monotonic", lambda: ticks[0] / 1e9)
    monkeypatch.setattr(time, "sleep", sleep)

    class Worker(PoolWorker):
        torch = SimpleNamespace(cuda=SimpleNamespace(
            synchronize=lambda: None,
            Event=lambda **kw: SimpleNamespace(record=lambda: None, elapsed_time=lambda e: .1)))

        def materialize(self, rows, purpose):
            if purpose == "proposal" and rows[0].request_id == (
                "sr-draft:64" if refilling else "sr-draft:32"
            ):
                blocked.set()
                assert release.wait(10), "test failed to release CPU model"
            return super().materialize(rows, purpose)

        def release(self, rows):
            if case == "warmup_refill" and "sr-draft:32" in rows:
                assert release_terminal.wait(10), "third A step must precede terminal release"
            assert not set(rows) & set(released)
            released.extend(rows)
            return super().release(rows)

    controller = DiagnosticDualController(
        lambda: DiagnosticDualMachine(s2_draft.S2DraftBackend(
            SimpleNamespace(max_model_len=4096), worker=Worker())),
        CheckpointJsonl(tmp_path / "draft-work.jsonl"))
    warm = {}
    for d in definitions:
        row = initial(d.request_id, d.prompt_token_ids + (10,), remaining=255)
        controller.execute("initialize", row)
        warm[d.request_id] = SimpleNamespace(
            prefix_version=1, logical_committed_prefix_token_ids=tuple(row["committed_token_ids"]),
            logical_committed_prefix_sha256=row["prefix_token_sha256"])
    prepared = controller.machine.backend.physical_rows()

    class Client:
        timeout_seconds = 60

        def call(self, operation, payload):
            if operation == "status":
                drain = tmp_path / "drain-state.json"
                if drain.exists() and read_json(drain)["phase"] == "wait_owner":
                    release.set()
                if blocked.is_set() and executed and not refilling:
                    if case == "operator_stop":
                        publish(tmp_path / "stop-request.json", {"reason": "operator_stop"})
                    if case == "rpc_failure":
                        raise RuntimeError("primary status RPC failure")
                return controller.status()
            if operation == "poll_ready":
                return controller.poll_ready(payload["limit"])
            if operation == "enqueue":
                for row in payload["rows"]:
                    if payload["work_operation"] == "propose_only":
                        rid = row["request_id"]
                        assert rid not in admits
                        if rid == "64":
                            assert "sr-draft:32" in released
                            physical = controller.machine.backend.physical_rows()[rid]
                            assert physical["block_ids"] == prepared[rid]["block_ids"]
                        admits.append(rid)
                return controller.enqueue(payload["work_operation"], payload["rows"])
            if operation == "execute":
                return controller.execute(payload["work_operation"], payload["row"])
            assert operation == "shutdown"
            return controller.shutdown()

    client = Client()
    scheduler._dual_client = client

    class Engine:
        def step(self):
            out = scheduler.schedule()  # Wait exception must precede this stock return.
            selected = list(out.num_scheduled_tokens)
            assert len(selected) == 32
            packet = read_json(path)
            cohort = packet["requests"][selected[0]]["cohort"]
            if "64" in admits and not release.is_set():
                assert cohort == "A" and "64" in controller.status()["inflight_request_ids"]
                progress_while_refill.append(True)
                release.set()
            assert (packet["decode_scan_deadline_ns"] is None
                    or now() < packet["decode_scan_deadline_ns"])
            commits, outputs = [], []
            for rid in selected:
                r = scheduler.requests[rid]
                proposal = controller.claimed([rid])["claimed"][0]["proposal"]
                terminal = (refilling and len(executed) ==
                            (1 if case == "warmup_refill" else 5) and rid == "32")
                # CPU Target correction; stop never fabricates natural EOS.
                delta = [999 if terminal else 777]
                commits.append({**commit_row(controller.machine, proposal, delta,
                                             terminal=terminal, remaining=200),
                                "logical_cohort": cohort})
                r.all_token_ids.extend(delta)
                r.num_output_tokens += 1
                r.num_computed_tokens = len(r.all_token_ids) - 1
                r.spec_token_ids = []
                if terminal:
                    r.finished = True
                    del scheduler.requests[rid]
                    scheduler.running.remove(r)
                outputs.append(SimpleNamespace(request_id=rid, finished=terminal, outputs=[
                    SimpleNamespace(token_ids=r.all_token_ids[2:],
                                    finish_reason="stop" if terminal else None)]))
            controller.enqueue("commit_and_propose", commits)
            ticks[0] += 2_000_000_000
            executed.append(selected)
            if case == "warmup_refill" and len(executed) == 3:
                release_terminal.set()
            return outputs

        def abort_request(self, cancelled):
            assert set(cancelled) == set(scheduler.requests)
            scheduler.requests.clear()
            scheduler.running.clear()

        def has_unfinished_requests(self):
            return bool(scheduler.requests)

    monkeypatch.setattr(fixed_runtime, "prepare_resident", lambda *a, **kw: (
        Engine(), scheduler, client, warm,
        {rid: {"token": 10, "terminal": False} for rid in ids}, None))
    result = None
    try:
        if case == "rpc_failure":
            with pytest.raises(RuntimeError, match="primary status RPC failure"):
                fixed_runtime.drive(SimpleNamespace(collective_rpc=lambda *a, **kw: []),
                                    m, definitions, tmp_path, selected_point("pingpong", 64), opts)
            assert read_json(tmp_path / "measurement-snapshot.json")["sample_count"] == 1
            assert read_json(tmp_path / "diagnostic-primary-error.json")["error"] == (
                "primary status RPC failure")
        else:
            result = fixed_runtime.drive(SimpleNamespace(collective_rpc=lambda *a, **kw: []),
                                         m, definitions, tmp_path,
                                         selected_point("pingpong", 64), opts)
    finally:
        release_terminal.set()
        release.set()
        # An injected RPC failure is intentionally failed, not a fabricated clean shutdown.
        if result is not None:
            controller.shutdown()
        else:
            with pytest.raises(RuntimeError, match="unresolved proposals"):
                controller.shutdown(failed=True)
            assert not controller._thread.is_alive()
    if result is None:
        return
    assert result["stop_reason"] == ("operator_stop" if case == "operator_stop" else "time_budget")
    assert result["diagnostic_drain"]["settled_requests"] == 360
    assert len(released) == 360 and not controller.machine.backend.worker.memory
    assert all(r["resources_released"] for r in result["requests"])
    assert sum(len(r["generated_token_ids"]) - 1 for r in result["requests"]) == len(executed) * 32
    assert all(len(batch) == 32 for batch in executed)
    if refilling:
        assert admits == ids[:65] and progress_while_refill
        assert sum(r["state"] == "FINISHED" for r in result["requests"]) == 1
        terminal = next(r for r in result["requests"] if r["request_id"] == "32")
        refill = next(r for r in result["requests"] if r["request_id"] == "64")
        assert terminal["completion_ns"] < refill["admission_ns"]
        assert controller.machine.release_receipts["32"]["resources_released_ns"] < (
            refill["admission_ns"])
    else:
        assert len(executed) == 1 and not any("completion_ns" in r for r in result["requests"])
        if case == "wait_expiry":
            assert result["measurement_end_ns"] - result["measurement_start_ns"] >= 30 * 10**9
            evidence = result["decode_scan"]["readiness"]
            assert evidence["closed_wait_ns"] >= 27 * 10**9
            assert evidence["wait_poll_count"] > 1

    if case == "warmup_refill":
        from specrhythm.serving.decode_scan_results import warmup_boundary

        boundary = warmup_boundary(result, selected_point("pingpong", 64), opts)
        sequence = "".join(s["cohort"] for s in boundary["steps"])
        assert sequence.startswith("ABAA") and sequence.endswith("B")
        assert boundary["completed_rotations"] == 2
        assert boundary["historical_unpaired_steps"] and boundary["pending_step"] is None
        assert boundary["committed_tokens_excluded"] == boundary["completed_steps"] * 32
        assert all(s["end_ns"] <= result["measurement_start_ns"] for s in boundary["steps"])
        snapshot = read_json(tmp_path / "measurement-snapshot.json")
        assert snapshot["scan_warmup_boundary"] == boundary
        assert snapshot["committed_window_tokens"] == snapshot["sample_count"] * 32
        assert snapshot["sample_count"] + boundary["completed_steps"] == len(executed)
        assert snapshot["window_ms"] >= 30000 and snapshot["drain_complete"]
