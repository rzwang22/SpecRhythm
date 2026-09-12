"""S2 run -> real proposer construction/RPC -> inherited verification, without CUDA.

Unlike the earlier coordinator test, collective_rpc executes its callback on each
worker. Neither fixture nor test assigns uuid_queries: the S2 startup must own it.
"""

from __future__ import annotations

import copy
import importlib
import os
import sys
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from test_phase4_dual_uuid import UUIDS
from test_phase4_dual_uuid import harness as _hardware
from test_phase4_serial import phase4_config as _config
from test_serving_s1 import execution, freeze_serial_inputs
from test_serving_s2 import profile, ranks

from specrhythm.phase4 import dual_commit, stock_vllm, vllm_dual
from specrhythm.phase4.dual import DualProposal, proposal_identity
from specrhythm.phase4.dual_uuid import build_dual_uuid_query_report, worker_dual_uuid_evidence
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.serving import s2_runtime
from specrhythm.serving.common import read_json
from specrhythm.serving.runtime_profile import PROFILE_ENV
from specrhythm.serving.s1_workload import write_once
from specrhythm.serving.s2_pool import publish

hardware = _hardware
phase4_config = _config


@pytest.fixture
def startup(tmp_path, monkeypatch, hardware, phase4_config):
    old, previous, _ = execution(tmp_path / "s1")
    freeze_serial_inputs(tmp_path, old, previous)
    path, manifest, definitions = profile(tmp_path / "s2", execution=previous["execution"])
    directory = tmp_path / "attempt"
    directory.mkdir()
    monkeypatch.setattr(os, "environ", os.environ.copy())
    for key, value in {
        PROFILE_ENV: str(path),
        "SR_S2_DRAFT_SOCKET": str(tmp_path / "draft.sock"),
        "SR_S2_CONTROL": str(directory / "s2-control.json"),
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "SR_PHASE4_DUAL_UUID_QUERY_MODE": "live",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(s2_runtime, "validate_installed_patch_stack", lambda *a: {})
    monkeypatch.setattr(
        stock_vllm,
        "worker_batch_invariant_evidence",
        lambda worker: {"batch_invariant_effective": True},
    )
    policies = {
        r.request_id: dual_commit.DualStopPolicy(r.maximum_new_tokens, 999) for r in definitions
    }
    # GPU tokenizer/model sampling setup is outside this startup-dependency contract.
    monkeypatch.setattr(dual_commit, "load_dual_stop_policies", lambda *a: policies)
    monkeypatch.setattr(vllm_dual, "load_dual_stop_policies", lambda *a: policies)
    h = SimpleNamespace(
        mode=None,
        rank=0,
        workers=[],
        rpcs=[],
        broadcast=None,
        gathered={},
        shutdown=False,
        draft_shutdown=False,
        path=path,
        directory=directory,
        root=tmp_path,
        hardware=hardware,
    )

    @contextmanager
    def on_rank(rank):
        h.rank = rank
        # Different processes expose one physical GPU each; both use logical cuda:0.
        with monkeypatch.context() as scope:
            scope.setenv("CUDA_VISIBLE_DEVICES", str(rank + 1))
            scope.setenv("LOCAL_RANK", str(rank))
            yield

    h.on_rank = on_rank

    def broadcast(value, src):
        if h.rank == src:
            h.broadcast = copy.deepcopy(value)
        return copy.deepcopy(h.broadcast)

    def gather(output, local, group):
        h.gathered[h.rank] = copy.deepcopy(local)
        if h.rank == 0:
            output[:] = [h.gathered[r] for r in (0, 1)]

    groups = [
        SimpleNamespace(
            rank_in_group=r,
            world_size=2,
            cpu_group=None,
            barrier=lambda: None,
            broadcast_object=broadcast,
        )
        for r in (0, 1)
    ]
    hardware.torch.distributed = SimpleNamespace(get_rank=lambda: h.rank, all_gather_object=gather)
    monkeypatch.setitem(sys.modules, "torch.distributed", hardware.torch.distributed)
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.parallel_state",
        SimpleNamespace(get_tp_group=lambda: groups[h.rank]),
    )
    hardware.cuda.Event = lambda **kw: SimpleNamespace(
        record=lambda: None,
        synchronize=lambda: None,
        elapsed_time=lambda other: 0.1,
    )
    hardware.cuda.nvtx = SimpleNamespace(range_push=lambda *a: None, range_pop=lambda: None)
    for name in (
        "memory_allocated",
        "memory_reserved",
        "max_memory_allocated",
        "max_memory_reserved",
    ):
        setattr(hardware.cuda, name, lambda device=None: 1024)
    hardware.cuda.mem_get_info = lambda: (8 * 1024**3, 80 * 1024**3)

    class LLM:
        def __init__(self, **kwargs):
            # Real configure created and parsed this before the simulated LLM worker entry.
            assert (directory / "decode-ready-context.json").is_file()
            cls_path = kwargs["speculative_config"]["model"]
            module, name = cls_path.rsplit(".", 1)
            proposer_cls = getattr(importlib.import_module(module), name)
            for rank in (0, 1):
                with on_rank(rank):
                    worker = hardware.worker(rank=rank)
                    cfg = worker.vllm_config
                    cfg.model_config = SimpleNamespace(
                        enforce_eager=True,
                        max_model_len=4096,
                        get_vocab_size=lambda: 151936,
                        hf_config=SimpleNamespace(eos_token_id=999),
                    )
                    cfg.scheduler_config.max_num_seqs = 512
                    cfg.scheduler_config.max_num_batched_tokens = 4096
                    cfg.cache_config.block_size = 16
                    worker.model_runner.kv_cache_config = SimpleNamespace(
                        num_blocks=100000,
                        kv_cache_groups=[object()],
                    )
                    worker.model_runner.attn_groups = [
                        [SimpleNamespace(backend=SimpleNamespace(get_name=lambda: "CPU-fixture"))]
                    ]
                    worker.model_runner.drafter = proposer_cls(cfg)
                    assert not hasattr(worker.model_runner.drafter, "uuid_queries")
                    h.workers.append(worker)
            self.llm_engine = SimpleNamespace(
                vllm_config=cfg,
                engine_core=SimpleNamespace(shutdown=lambda: setattr(h, "shutdown", True)),
            )

        def collective_rpc(self, callback):
            h.rpcs.append(callback.__name__)
            rows = []
            for rank, worker in enumerate(h.workers):
                with on_rank(rank):
                    rows.append(callback(worker))
            return rows

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=LLM))

    def shutdown(operation, payload):
        assert operation == "shutdown"
        h.draft_shutdown = True
        return {"shutdown": True}

    monkeypatch.setattr(s2_runtime, "client_for", lambda mode: SimpleNamespace(call=shutdown))

    def drive(llm, rows, trace, out, mode, eos, **kwargs):
        # Exercise real read-only prefill/capacity snapshots before verification.
        h.prefill_snapshots = llm.collective_rpc(s2_runtime.target_snapshot)
        h.repeat_snapshots = llm.collective_rpc(s2_runtime.target_snapshot)
        if mode == "pingpong":
            definition = rows[0]
            rid, internal = definition.request_id, "opaque-request"
            prefix = tuple(definition.prompt_token_ids) + (100,)
            now = time.monotonic_ns()
            proposal = DualProposal(
                request_id=rid,
                round_id=0,
                proposal_id=proposal_identity(rid, 0, 1, (200,)),
                prefix_version=1,
                prefix_token_count=len(prefix),
                prefix_token_sha256=token_prefix_hash(prefix),
                draft_kv_length_before=len(prefix),
                draft_kv_length_after=len(prefix) + 1,
                proposal_token_ids=(200,),
                created_timestamp_ns=now,
                draft_start_ns=now - 2,
                draft_end_ns=now - 1,
            )
            publish(
                directory / "s2-control.json",
                {
                    "requests": {rid: {"state": "ACTIVE", "cohort": "A"}},
                    "initial_enqueues": {rid: {"enqueue_start_ns": now - 3}},
                },
            )
            leader = h.workers[0].model_runner.drafter
            leader.identity.bind(internal, tuple(definition.prompt_token_ids))
            leader.requests[rid] = vllm_dual._Request(
                tuple(definition.prompt_token_ids),
                definition.maximum_new_tokens,
                prefix,
                (100,),
                lifecycle="DRAFT_READY",
            )
            leader.client.client = SimpleNamespace(
                call=lambda *a: {
                    "claimed": [{"request_id": rid, "proposal": proposal.to_dict()}],
                }
            )
            for rank in (0, 1):
                with on_rank(rank):
                    h.workers[rank].model_runner.drafter.on_target_verify_start(
                        request_ids=[internal],
                        scheduled_spec_token_ids={internal: [200]},
                    )
            # The simulated gather delivers the actual rank-1 row to the actual rank-0 hook.
            for rank in (1, 0):
                with on_rank(rank):
                    h.workers[rank].model_runner.drafter.on_target_verify_end(
                        request_ids=[internal],
                        sampled_token_ids=[[200, 201]],
                        scheduled_spec_token_ids={internal: [200]},
                    )
            leader._write_report()
        return {"test_drive_completed": True}

    monkeypatch.setattr(s2_runtime, "drive", drive)

    def run(mode="pingpong", *, probe=False):
        h.mode = mode
        monkeypatch.setenv("SR_S2_MODE", mode)
        write_once(
            directory / "draft-startup.json",
            {
                "s2_capacity": next(
                    r for r in ranks() if r["mode"] == mode and r["role"] == "draft"
                ),
            },
        )
        return s2_runtime.run(tmp_path, path, directory, mode, probe=probe)

    h.run = run
    return h


def test_s2_startup_rpc_reaches_real_pingpong_verify_end(startup):
    h = startup
    result = h.run()
    assert h.shutdown and h.draft_shutdown
    assert h.rpcs == ["initialize_pingpong_worker"] + ["target_snapshot"] * 3
    assert read_json(h.directory / "runtime.json") == result
    initial = read_json(h.directory / "actual-capacity.json")["target_worker_ranks"]
    final = result["target_final_memory"]
    for rank, worker in enumerate(h.workers):
        with h.on_rank(rank):
            q = worker.model_runner.drafter.uuid_queries
            evidence = worker_dual_uuid_evidence(worker)
            assert q.torch is h.hardware.torch
            assert evidence["logical_cuda_index"] == 0
            assert evidence["tp_rank"] == rank
            assert evidence["gpu_uuid"] == UUIDS[rank + 1]
            assert evidence["uuid_initial_validation_count"] == 1
            assert evidence["uuid_verification_subprocess_query_count"] == 1
            assert evidence["uuid_cache_hit_count"] == 0
            assert evidence == final[rank]["dual_uuid_query"]
            assert initial[rank]["dual_uuid_query"]["uuid_verification_access_count"] == 0
            assert (
                h.repeat_snapshots[rank]["dual_uuid_query"]["uuid_initial_validation_count"] == 1
            )
            for _ in range(2):
                s2_runtime.target_snapshot(worker)
            assert worker.model_runner.drafter.uuid_queries is q
            assert worker_dual_uuid_evidence(worker) == evidence
            calls = len(h.hardware.calls)
            with pytest.raises(RuntimeError, match="already initialized"):
                s2_runtime.initialize_pingpong_worker(worker)
            assert len(h.hardware.calls) == calls
    logs = h.workers[0].model_runner.drafter.verification_log.read()
    evidence = [r["dual_uuid_query"] for r in final]
    assert build_dual_uuid_query_report(evidence, initial, logs)["valid"]
    assert read_json(h.directory / "plugin-report.json")["sampled_row_tp_consensus"] is True


def test_s2_capacity_probe_has_initialized_workers_and_zero_verification(startup):
    h = startup
    result = h.run(probe=True)
    assert result["probe"] is True and h.shutdown and h.draft_shutdown
    assert h.rpcs == ["initialize_pingpong_worker", "target_snapshot"]
    assert not hasattr(h, "prefill_snapshots")
    assert not (h.directory / "verification-events.jsonl").exists()
    for row in result["target_final_memory"]:
        evidence = row["dual_uuid_query"]
        assert evidence["uuid_query_mode"] == "live"
        assert evidence["uuid_initial_validation_count"] == 1
        assert evidence["uuid_verification_access_count"] == 0
        assert evidence["uuid_verification_subprocess_query_count"] == 0
        assert evidence["uuid_cache_hit_count"] == 0


@pytest.mark.parametrize("mode", ("target", "serial"))
def test_s2_other_modes_keep_real_proposer_and_plain_snapshot_paths(startup, mode):
    h = startup
    result = h.run(mode)
    assert h.shutdown and h.draft_shutdown
    assert h.rpcs == ["target_snapshot"] * 4
    for worker, row in zip(h.workers, result["target_final_memory"]):
        assert type(worker.model_runner.drafter).__name__ == (
            "S2TargetProposer" if mode == "target" else "S2SerialProposer"
        )
        assert not hasattr(worker.model_runner.drafter, "uuid_queries")
        assert "dual_uuid_query" not in row
    assert h.hardware.calls  # The unchanged plain snapshots still perform actual identity lookup.


def test_s2_startup_missing_real_uuid_fails_before_verification(startup, monkeypatch):
    h = startup
    monkeypatch.setattr(
        stock_vllm.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(stdout="", returncode=0),
    )
    with pytest.raises(RuntimeError, match="UUID"):
        h.run()
    assert h.shutdown and not hasattr(h, "prefill_snapshots")
    assert all(not hasattr(w.model_runner.drafter, "uuid_queries") for w in h.workers)
    assert not (h.directory / "runtime.json").exists()
