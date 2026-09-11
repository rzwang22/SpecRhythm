"""Real owner dispatch and fixed coordinator windows with CPU model/engine substitutes."""

import sys
import threading
from types import SimpleNamespace

import pytest
from test_phase4_dual_batched_draft import initial
from test_serving_fixed import clock_fixture
from test_serving_s2_runtime import PoolWorker

from specrhythm.phase4.transport import CheckpointJsonl
from specrhythm.serving import fixed_runtime, s2_draft
from specrhythm.serving.common import DataError, read_json
from specrhythm.serving.fixed_observe import DeviceTimeline
from specrhythm.serving.fixed_plan import build_manifest, point, settings
from specrhythm.serving.fixed_settle import DiagnosticDualController, DiagnosticDualMachine
from specrhythm.serving.s2_pool import publish


@pytest.mark.parametrize("mode", ["serial-split", "pingpong"])
def test_real_owner_dispatch_initial_work_wait_and_window_drain(tmp_path, monkeypatch, mode):
    clock, ids = clock_fixture()
    definitions = list(clock.definitions.values())
    options = settings(warmup_steps=0, samples=1, window_seconds=5, drain_timeout=5)
    manifest = build_manifest({"eos_token_ids": [999]}, ids, "sha", options)
    monkeypatch.setenv("SR_S2_CONTROL", str(tmp_path / "s2-control.json"))
    monkeypatch.setenv("SR_S2_RUN_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("SR_S2_MODE", "pingpong")
    monkeypatch.setattr(s2_draft, "cuda_memory", lambda *a: {"free_memory_bytes": 10**9})
    publish(
        tmp_path / "s2-control.json",
        {
            "barrier_ns": None,
            "requests": {rid: {"state": "STAGED", "cohort": None} for rid in ids},
        },
    )
    events = []
    gate = threading.Event()
    entered_b = threading.Event()

    class Worker(PoolWorker):
        torch = SimpleNamespace(
            cuda=SimpleNamespace(
                synchronize=lambda: None,
                Event=lambda **kw: SimpleNamespace(
                    record=lambda: None, elapsed_time=lambda e: 0.1
                ),
            )
        )

        def materialize(self, rows, purpose):
            if purpose == "proposal":
                events.append(("Draft", rows[0].request_id))
                if rows[0].request_id == "sr-draft:" + ids[32]:
                    entered_b.set()
                    # A real blocked CPU model simulates GPU ownership. The test
                    # releases this operation; runtime must not fabricate completion.
                    assert gate.wait(5)
            return super().materialize(rows, purpose)

    def factory():
        return DiagnosticDualMachine(
            s2_draft.S2DraftBackend(SimpleNamespace(max_model_len=4096), worker=Worker())
        )

    controller = DiagnosticDualController(factory, CheckpointJsonl(tmp_path / "work.jsonl"))
    for r in definitions:
        controller.execute("initialize", initial(r.request_id, r.prompt_token_ids + (10,)))
    warm = {
        r.request_id: SimpleNamespace(
            prefix_version=1,
            logical_committed_prefix_token_ids=r.prompt_token_ids + (10,),
            logical_committed_prefix_sha256=initial(r.request_id, r.prompt_token_ids + (10,))[
                "prefix_token_sha256"
            ],
        )
        for r in definitions
    }

    class Client:
        def call(self, operation, payload):
            if operation == "status":
                return controller.status()
            if operation == "enqueue":
                value = controller.enqueue(payload["work_operation"], payload["rows"])
                if payload["rows"][0]["logical_cohort"] == "B":
                    assert entered_b.wait(5)
                    if mode == "serial-split":
                        gate.set()  # Completion precedes Target only if real wait is honored.
                return value
            if operation == "execute":
                return controller.execute(payload["work_operation"], payload["row"])
            assert operation == "shutdown"
            events.append(("shutdown", None))
            return controller.shutdown()

    scheduler = SimpleNamespace(
        s2_steps=[],
        requests={rid: object() for rid in ids},
        s2_pool=SimpleNamespace(report=lambda: {}),
    )

    class Engine:
        def step(self):
            if mode == "serial-split":
                assert not controller.status()["inflight_request_ids"]
            else:
                assert set(controller.status()["inflight_request_ids"]) == set(ids[32:64])
            events.append(("Target", None))
            gate.set()
            # Real owner has already produced A; take its actual candidates.
            ready = controller.poll_ready(32)["ready"]
            assert len(ready) == 32
            scheduled = [r["request_id"] for r in ready]
            assert scheduled == ids[:32]
            scheduler.s2_steps.append(
                {
                    "request_ids": scheduled,
                    "B": 32,
                    "cohort": "A",
                    "rows": [{"request_id": rid, "candidate_positions": 4} for rid in scheduled],
                }
            )
            return [
                SimpleNamespace(
                    request_id=r["request_id"],
                    finished=False,
                    outputs=[
                        SimpleNamespace(
                            token_ids=[10, 777],
                            finish_reason=None,
                        )
                    ],
                )
                for r in ready
            ]

        def abort_request(self, cancelled):
            assert not controller.status()["inflight_request_ids"]
            assert set(cancelled) == set(ids)
            events.append(("abort", None))
            scheduler.requests.clear()

        def has_unfinished_requests(self):
            return bool(scheduler.requests)

    engine = Engine()
    bootstrap = {r: {"token": 10, "terminal": False} for r in ids}
    monkeypatch.setattr(
        fixed_runtime,
        "prepare_resident",
        lambda *a, **kw: (engine, scheduler, Client(), warm, bootstrap, None),
    )
    llm = SimpleNamespace(collective_rpc=lambda callback, **kw: [])
    try:
        result = fixed_runtime.drive(
            llm,
            manifest,
            definitions,
            tmp_path,
            {**point(mode), "kind": "initial-state", "batch": 32},
            options,
        )
    finally:
        gate.set()
        controller.shutdown()
    assert result["sample_count"] == 1 and result["stop_reason"] == "sample_budget"
    assert result["capacity"]["resident_request_count"] == 100
    assert result["capacity"]["active_request_limit"] == 64
    assert result["capacity"]["max_requests_per_target_forward"] == 32
    assert result["end_ns"] > result["measurement_end_ns"]
    assert all(r["state"] == "DIAGNOSTIC_CANCELLED" for r in result["requests"])
    assert all(r["resources_released"] for r in result["requests"])
    assert not any("completion_ns" in r for r in result["requests"])
    assert sum(len(r["generated_token_ids"]) - 1 for r in result["requests"]) == 32
    assert [k for k, _ in events].index("Target") < [k for k, _ in events].index("abort")
    assert [k for k, _ in events].index("abort") < [k for k, _ in events].index("shutdown")
    history = controller.machine.backend.history
    assert [r["logical_cohort"] for r in history] == ["A", "B"]
    assert all(len(r["request_ids"]) == 32 for r in history)
    assert not controller.machine.backend.worker.memory


def test_draft_wait_propagates_real_failure_and_timeout():
    with pytest.raises(DataError, match="Draft failed"):
        fixed_runtime.wait_draft(SimpleNamespace(call=lambda *a: {"failures": {"r": "bad KV"}}), 1)
    with pytest.raises(DataError, match="drain deadline"):
        fixed_runtime.wait_draft(
            SimpleNamespace(call=lambda *a: {"failures": {}, "inflight_request_ids": ["r"]}), 0
        )


def test_primary_exception_survives_engine_cleanup_failure(tmp_path, monkeypatch):
    options = settings()
    manifest = {"fixed_diagnostic": {"options": options}}
    monkeypatch.setattr(fixed_runtime, "configure", lambda *a: (None, manifest, []))

    def fail():
        raise RuntimeError("secondary shutdown")

    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(engine_core=SimpleNamespace(shutdown=fail)),
        collective_rpc=lambda *a: (_ for _ in ()).throw(ValueError("primary worker")),
    )
    monkeypatch.setattr(fixed_runtime, "make_engine", lambda *a, **kw: llm)
    with pytest.raises(ValueError, match="primary worker"):
        fixed_runtime.run(tmp_path, tmp_path / "manifest", tmp_path, point("target"))
    assert read_json(tmp_path / "diagnostic-primary-error.json")["error"] == "primary worker"


def test_cuda_events_are_read_at_final_fence_without_per_forward_sync():
    events, calls = [], []

    class Event:
        def __init__(self, **kwargs):
            self.value = len(events)
            events.append(self)

        def record(self):
            calls.append("record")

        def synchronize(self):
            calls.append("sync")

        def query(self):
            return True

        def elapsed_time(self, other):
            return other.value - self.value

    model = SimpleNamespace(
        register_forward_pre_hook=lambda f: f, register_forward_hook=lambda f: f
    )
    timeline = DeviceTimeline(
        SimpleNamespace(cuda=SimpleNamespace(Event=Event)),
        model,
        lambda: {"B": 32},
        identity={"gpu_uuid": "real-startup-evidence"},
    )
    for _ in range(4):
        timeline.before(None, None)
        timeline.after(None, None, None)
    assert calls.count("sync") == 1  # The startup anchor only.
    report = timeline.report()
    assert len(report["forwards"]) == 4
    assert all(r["gpu_event_ms"] == 1 for r in report["forwards"])
    assert calls.count("sync") == 1
    assert report["extra_per_round_synchronization"] is False


def test_entry_modules_import_without_vllm_or_torch():
    # This interpreter may have CPU substitutes from unrelated tests; separate import
    # verifies the offline entry cannot eagerly import either real GPU framework.
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "import specrhythm.serving.fixed_cli; import specrhythm.serving.fixed_results; "
            "assert 'torch' not in sys.modules and 'vllm' not in sys.modules",
        ],
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()
