"""CPU owner/data-plane and manual coordinator contracts using real S2 adapters."""

from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import pytest
from test_phase4_vllm_draft import FakeWorker
from test_serving_s1 import ready_manifest, request
from test_serving_s2 import profile

from specrhythm.phase4.draft_batch import DraftProposalPlan
from specrhythm.phase4.serial import PROTOCOL_VERSION, Proposal
from specrhythm.serving.common import DataError, read_json
from specrhythm.serving.s1_workload import ResidentServingRequest, write_once
from specrhythm.serving.s2_plan import MODES, poisson_trace
from specrhythm.serving.s2_pool import prefix_record, publish


class PoolWorker(FakeWorker):
    provenance = {
        "physical_gpu_id": 0,
        "gpu_uuid": "GPU-real-fixture",
        "block_size": 16,
        "kv_cache_num_blocks": 10000,
    }
    torch = object()

    def __init__(self):
        super().__init__()
        self.blocks = {}
        self.next_block = 1
        self.kv = SimpleNamespace(get_block_ids=lambda rid: [self.blocks[rid]])
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(get_vocab_size=lambda: 100)
        )

    def materialize(self, rows, purpose):
        for row in rows:
            if row.request_id not in self.blocks:
                self.blocks[row.request_id] = [self.next_block]
                self.next_block += 1
        return super().materialize(rows, purpose)

    def release(self, ids):
        super().release(ids)
        for rid in ids:
            del self.blocks[rid]


def test_actual_backend_retains_staged_KV_and_new_instance_rebuilds(tmp_path, monkeypatch):
    from specrhythm.serving import s2_draft

    path = tmp_path / "s2-control.json"
    monkeypatch.setenv("SR_S2_CONTROL", str(path))
    monkeypatch.setenv("SR_S2_RUN_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("SR_S2_MODE", "serial")
    monkeypatch.setattr(s2_draft, "cuda_memory", lambda torch: {"free_memory_bytes": 1000000000})
    states = {rid: {"state": "STAGED", "cohort": None} for rid in ("a", "b")}
    publish(path, {"barrier_ns": None, "requests": states})
    backend = s2_draft.S2DraftBackend(SimpleNamespace(max_model_len=4096), worker=PoolWorker())
    backend.initialize_many((("a", (1, 2, 3)), ("b", (4, 5, 6))))
    initial = backend.physical_rows()
    plan = DraftProposalPlan("a", 0, (1, 2, 3), 2, ())
    with pytest.raises(DataError, match="before arrival"):
        backend.propose_many([plan])
    states["a"]["state"] = "ACTIVE"
    publish(path, {"barrier_ns": 1, "requests": states})
    proposed = backend.propose_many([plan])
    assert len(proposed["a"]) == 2
    assert backend.physical_rows()["b"] == initial["b"]
    assert backend.worker.memory["sr-draft:b"] == [4, 5, 6]
    assert sum(purpose == "setup" for purpose, _ in backend.worker.calls) == 1
    with pytest.raises(DataError, match="pre-initialization|re-initialization"):
        backend.initialize("c", (7, 8, 9))
    backend.finish_many(["a", "b"])
    backend.shutdown()
    # A distinct worker/allocator gets the original prefixes, never the old continuation.
    states = {rid: {"state": "STAGED", "cohort": None} for rid in ("a", "b")}
    publish(path, {"barrier_ns": None, "requests": states})
    fresh = s2_draft.S2DraftBackend(SimpleNamespace(max_model_len=4096), worker=PoolWorker())
    fresh.initialize_many((("a", (1, 2, 3)), ("b", (4, 5, 6))))
    assert fresh.physical_rows() == initial
    fresh.finish_many(["a", "b"])
    fresh.shutdown()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("pool_size", [2, 360])
def test_real_drive_prefills_all_before_clock_and_admits_dynamically(
    tmp_path, monkeypatch, mode, pool_size
):
    from specrhythm.serving import s2_runtime

    path, manifest, raw = profile(
        tmp_path / "inputs", rows=[request(i, 3) for i in range(pool_size)])
    rows = [ResidentServingRequest(r) for r in raw]
    directory = tmp_path / "attempt"
    directory.mkdir()
    control_path = directory / "s2-control.json"
    monkeypatch.setenv("SR_S2_CONTROL", str(control_path))
    monkeypatch.setenv("SR_S2_DRAFT_SOCKET", str(tmp_path / "draft.sock"))
    publish(
        control_path,
        {"barrier_ns": None, "requests": {r.request_id: {"state": "STAGED"} for r in rows}},
    )
    trace = poisson_trace([r.request_id for r in rows], 100, 1, 1)
    boot = [100 + i for i in range(pool_size)]
    warm = ready_manifest(raw, {"logical": {"workload_sha256": manifest["workload_sha256"]}}, boot)
    calls = []
    finished = set()

    class Client:
        def call(self, operation, payload):
            calls.append((operation, payload))
            if operation == "status":
                return {"failures": {}, "inflight_request_ids": []}
            if operation == "synchronize_and_batch_propose":
                now = time.monotonic_ns()
                proposals = [
                    Proposal(
                        PROTOCOL_VERSION,
                        r["request_id"],
                        0,
                        r["committed_prefix_len"],
                        r["committed_prefix_hash"],
                        (200,),
                        False,
                        now,
                        now,
                        0,
                        {},
                        {},
                    ).to_dict()
                    for r in payload["proposals"]
                ]
                return {"proposals": proposals, "service_send_ns": now, "transport_end_ns": now}
            return {}

    monkeypatch.setattr(s2_runtime, "client_for", lambda m: Client())

    class Params:
        def __init__(self, **kwargs):
            assert kwargs["logprobs"] == 5

        def update_from_generation_config(self, *a):
            self.eos_token_id = 999
            self.stop_token_ids = []

        def update_from_tokenizer(self, *a):
            pass

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=Params))

    class Inproc:
        pass

    monkeypatch.setitem(
        sys.modules, "vllm.v1.engine.core_client", SimpleNamespace(InprocClient=Inproc)
    )
    initial = {
        r.request_id: prefix_record(r.prompt_token_ids, r.prompt_length, [[i + 1]])
        for i, r in enumerate(rows)
    }
    draft_initial = {
        r.request_id: prefix_record([*r.prompt_token_ids, boot[i]], r.prompt_length + 1, [[i + 1]])
        for i, r in enumerate(rows)
    }
    scheduler = SimpleNamespace(
        s2_steps=[],
        freeze_pool=lambda: {"initial": initial},
        s2_pool=SimpleNamespace(report=lambda: {"initial": initial}),
    )
    core = Inproc()
    core.engine_core = SimpleNamespace(scheduler=scheduler)
    added = []

    class Engine:
        engine_core = core
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(try_get_generation_config=lambda: {})
        )
        setup = True

        def add_request(self, rid, prompt, params, **kw):
            added.append(rid)

        def step(self):
            packet = read_json(control_path)
            if self.setup:
                assert packet["barrier_ns"] is None and len(added) == pool_size
                self.setup = False
                write_once(directory / "setup-ready.json", {"global_decode_ready": True})
                write_once(directory / "decode-ready-manifest.json", warm.to_dict())
                publish(directory / "draft-pool.json", {"rows": draft_initial})
                return [
                    SimpleNamespace(
                        request_id=r.request_id,
                        finished=False,
                        outputs=[SimpleNamespace(token_ids=[boot[i]], finish_reason=None)],
                    )
                    for i, r in enumerate(rows)
                ]
            active = [rid for rid, r in packet["requests"].items() if r["state"] == "ACTIVE"]
            assert len(active) == 1  # one real coordinator slot, others retain staged state
            rid = active[0]
            assert rid not in finished
            if mode == "serial":
                assert rid in packet["initial_proposals"]
            if mode == "pingpong":
                assert rid in packet["initial_enqueues"]
            # A blocked Target call cannot shift the frozen arrivals.
            time.sleep(0.015)
            finished.add(rid)
            i = [r.request_id for r in rows].index(rid)
            return [
                SimpleNamespace(
                    request_id=rid,
                    finished=True,
                    outputs=[SimpleNamespace(token_ids=[boot[i], 200, 999], finish_reason="stop")],
                )
            ]

        def has_unfinished_requests(self):
            return len(finished) < 2

    tokenizer = SimpleNamespace(
        eos_token_id=999,
        encode=lambda text, **kw: list(
            next(r.prompt_token_ids for r in rows if r.prompt_text == text)
        ),
    )
    llm = SimpleNamespace(
        llm_engine=Engine(),
        get_tokenizer=lambda: tokenizer,
        collective_rpc=lambda callback: [
            {"rank": i, "timestamp_ns": time.monotonic_ns()} for i in (0, 1)
        ],
    )
    if pool_size == 360:
        prepared = s2_runtime.prepare_resident(llm, rows, directory, mode, [999])
        assert len(added) == len(set(added)) == len(prepared[3]) == len(prepared[4]) == 360
        assert not finished and not calls, "no timed Draft work during setup"
        pool = read_json(directory / "resident-pool.json")
        assert pool["selected_requests"] == pool["resident_requests"] == 360
        assert set(pool["target"]["initial"]) == set(pool["draft"]["rows"]) == set(added)
        return
    result = s2_runtime.drive(llm, rows, trace, directory, mode, [999], active_limit=1)
    assert len(finished) == 2
    assert result["end_ns"] > result["start_ns"]
    assert all(
        len(r["generated_token_ids"]) == 3 and len(r["commits"]) == 1 for r in result["requests"]
    )
    assert all(r["resources_released"] for r in result["requests"])
    events = result["events"]
    arrivals = [e for e in events if e["event"] == "arrived"]
    assert len(arrivals) == 2
    assert all(e["planned_arrival_ns"] >= result["start_ns"] for e in arrivals)
    assert read_json(directory / "resident-pool.json")["target_bootstrap_materialized"] is False
