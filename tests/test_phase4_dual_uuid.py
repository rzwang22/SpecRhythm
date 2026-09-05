"""CPU-only UUID A/B tests: intercept subprocesses, never import real CUDA."""

from __future__ import annotations

import copy
import sys
from types import SimpleNamespace

import pytest

from specrhythm.phase4 import stock_vllm
from specrhythm.phase4.dual_uuid import (
    UUID_QUERY_MODE_ENV,
    build_dual_uuid_query_report,
    dual_uuid_query_mode,
    worker_dual_runtime_snapshot,
    worker_dual_uuid_evidence,
)
from specrhythm.phase4.vllm_dual import DualBatchRemoteProposer
from specrhythm.phase4.vllm_remote import RemoteDraftProposer

UUIDS = {
    1: "GPU-529b6239-6138-7741-e9ca-824668c035c1",
    2: "GPU-c6ccf586-fd34-207a-8793-20726626c44e",
}


@pytest.fixture
def harness(monkeypatch):
    calls = []
    cuda = SimpleNamespace(
        logical=0, name="A800",
        current_device=lambda: cuda.logical,
        get_device_properties=lambda device: SimpleNamespace(name=cuda.name),
        synchronize=lambda device=None: None,
        memory_allocated=lambda device: 10,
        memory_reserved=lambda device: 20,
        max_memory_allocated=lambda device: 10,
        max_memory_reserved=lambda device: 20,
    )
    torch = SimpleNamespace(cuda=cuda)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.delenv(UUID_QUERY_MODE_ENV, raising=False)
    monkeypatch.setattr(stock_vllm, "worker_batch_invariant_evidence", lambda worker: {})

    def query(command, **kwargs):
        assert command[:2] == ["nvidia-smi", "-i"]
        assert command[3:] == ["--query-gpu=uuid", "--format=csv,noheader,nounits"]
        assert kwargs == {"capture_output": True, "text": True, "check": False}
        calls.append(command)
        return SimpleNamespace(stdout=UUIDS[int(command[2])] + "\n", returncode=0)

    monkeypatch.setattr(stock_vllm.subprocess, "run", query)

    def worker(rank=1, logical=0):
        cuda.logical = logical
        parameter = SimpleNamespace(
            numel=lambda: 10, element_size=lambda: 2, device=f"cuda:{logical}",
        )
        return SimpleNamespace(
            rank=rank, local_rank=rank, device=logical,
            get_model=lambda: SimpleNamespace(parameters=lambda: [parameter]),
            model_runner=SimpleNamespace(
                drafter=SimpleNamespace(torch=torch, tp_rank=rank),
            ),
            vllm_config=SimpleNamespace(
                parallel_config=SimpleNamespace(world_size=2),
                scheduler_config=SimpleNamespace(
                    get_scheduler_cls=lambda: type, scheduler_cls="DualBatchScheduler",
                    async_scheduling=False, enable_chunked_prefill=False,
                ),
                speculative_config=object(),
                cache_config=SimpleNamespace(enable_prefix_caching=False),
                model_config=SimpleNamespace(enforce_eager=False),
            ),
        )

    return SimpleNamespace(worker=worker, cuda=cuda, torch=torch, calls=calls)


@pytest.mark.parametrize("mode", [None, "live", "cached"])
def test_real_startup_and_repeated_verification_queries(harness, monkeypatch, mode):
    if mode is not None:
        monkeypatch.setenv(UUID_QUERY_MODE_ENV, mode)
    worker = harness.worker()
    snapshot = worker_dual_runtime_snapshot(worker)
    assert [int(call[2]) for call in harness.calls] == [2]
    queries = worker.model_runner.drafter.uuid_queries
    assert queries.evidence()["uuid_initial_validation_count"] == 1
    for _ in range(4):
        identity = queries.for_verification()
        assert identity == {key: snapshot[key] for key in identity}
    cached = mode == "cached"
    assert len(harness.calls) == (1 if cached else 5)
    evidence = worker_dual_uuid_evidence(worker)
    assert evidence["uuid_query_mode"] == (mode or "live")
    assert evidence["uuid_verification_access_count"] == 4
    assert evidence["uuid_verification_subprocess_query_count"] == (0 if cached else 4)
    assert evidence["uuid_cache_hit_count"] == (4 if cached else 0)


def test_cached_never_launches_verification_subprocess_and_cache_is_immutable(
    harness, monkeypatch,
):
    monkeypatch.setenv(UUID_QUERY_MODE_ENV, "cached")
    worker = harness.worker()
    snapshot = worker_dual_runtime_snapshot(worker)
    queries = worker.model_runner.drafter.uuid_queries

    def forbidden(*args, **kwargs):
        pytest.fail("verification launched a subprocess in cached mode")

    monkeypatch.setattr(stock_vllm.subprocess, "run", forbidden)
    snapshot["gpu_uuid"] = "guessed"
    first = queries.for_verification()
    first["gpu_uuid"] = "mutated"
    monkeypatch.setenv(UUID_QUERY_MODE_ENV, "live")  # Mode is latched for this worker.
    for _ in range(5):
        assert queries.for_verification()["gpu_uuid"] == UUIDS[2]
    with pytest.raises(TypeError):
        queries._identity["gpu_uuid"] = "guessed"
    with pytest.raises(RuntimeError, match="already initialized"):
        worker_dual_runtime_snapshot(worker)


@pytest.mark.parametrize("visible,logical,physical", [("2", 0, 2), ("1,2", 1, 2), ("2,1", 1, 1)])
def test_rank_is_never_a_device_index(harness, monkeypatch, visible, logical, physical):
    monkeypatch.setenv(UUID_QUERY_MODE_ENV, "cached")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    worker = harness.worker(rank=1, logical=logical)
    worker_dual_runtime_snapshot(worker)
    identity = worker.model_runner.drafter.uuid_queries.for_verification()
    assert identity["logical_cuda_index"] == logical
    assert identity["physical_gpu_id"] == physical
    assert identity["gpu_uuid"] == UUIDS[physical]
    assert harness.calls[0][2] == str(physical)


@pytest.mark.parametrize("stdout,returncode", [
    ("", 0), ("  \n", 0), ("GPU-1", 0), ("1", 0), ("unknown", 0),
    ("GPU-529b6239-6138-7741-e9ca-824668c035cZ", 0),
    (UUIDS[1] + "\n" + UUIDS[2], 0), (UUIDS[2], 1),
])
def test_cached_bad_uuid_fails_closed_without_rank_fallback(
    harness, monkeypatch, stdout, returncode,
):
    monkeypatch.setenv(UUID_QUERY_MODE_ENV, "cached")
    worker = harness.worker()
    monkeypatch.setattr(stock_vllm.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout=stdout, returncode=returncode))
    with pytest.raises(RuntimeError, match="UUID"):
        worker_dual_runtime_snapshot(worker)
    assert not hasattr(worker.model_runner.drafter, "uuid_queries")


def test_cached_missing_subprocess_fails_closed(harness, monkeypatch):
    monkeypatch.setenv(UUID_QUERY_MODE_ENV, "cached")

    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(stock_vllm.subprocess, "run", missing)
    worker = harness.worker()
    with pytest.raises(FileNotFoundError, match="nvidia-smi"):
        worker_dual_runtime_snapshot(worker)
    assert not hasattr(worker.model_runner.drafter, "uuid_queries")


@pytest.mark.parametrize("visible", [None, "", "GPU-1", "1,1"])
def test_cached_missing_binding_cannot_be_guessed_from_rank(harness, monkeypatch, visible):
    monkeypatch.setenv(UUID_QUERY_MODE_ENV, "cached")
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    with pytest.raises(RuntimeError):
        worker_dual_runtime_snapshot(harness.worker())
    assert harness.calls == []


@pytest.mark.parametrize("change", ["active", "visible", "name"])
def test_cached_binding_change_fails_closed(harness, monkeypatch, change):
    monkeypatch.setenv(UUID_QUERY_MODE_ENV, "cached")
    worker = harness.worker()
    worker_dual_runtime_snapshot(worker)
    if change == "active":
        harness.cuda.logical = 1
    elif change == "visible":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    else:
        harness.cuda.name = "different GPU"
    with pytest.raises(RuntimeError, match="device|binding"):
        worker.model_runner.drafter.uuid_queries.for_verification()
    assert len(harness.calls) == 1
    assert worker_dual_uuid_evidence(worker)["uuid_cache_hit_count"] == 0


@pytest.mark.parametrize("bad_mode", ["", "auto", "Cached", " cached "])
def test_unknown_mode_fails_before_startup_query(harness, monkeypatch, bad_mode):
    monkeypatch.setenv(UUID_QUERY_MODE_ENV, bad_mode)
    with pytest.raises(RuntimeError, match="must be live or cached"):
        worker_dual_runtime_snapshot(harness.worker())
    assert harness.calls == []


def test_default_is_live_and_ab_identities_are_equal(harness, monkeypatch):
    assert dual_uuid_query_mode() == "live"
    identities = []
    for mode in ("live", "cached"):
        monkeypatch.setenv(UUID_QUERY_MODE_ENV, mode)
        worker = harness.worker()
        worker_dual_runtime_snapshot(worker)
        identities.append(worker.model_runner.drafter.uuid_queries.for_verification())
    assert identities[0] == identities[1]


def test_live_returns_newly_queried_identity_instead_of_startup_value(harness, monkeypatch):
    worker = harness.worker()
    worker_dual_runtime_snapshot(worker)
    queries = worker.model_runner.drafter.uuid_queries
    assert queries.for_verification()["gpu_uuid"] == UUIDS[2]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    identity = queries.for_verification()
    assert identity["physical_gpu_id"] == 1 and identity["gpu_uuid"] == UUIDS[1]
    assert [int(call[2]) for call in harness.calls] == [2, 2, 1]
    assert queries.evidence()["uuid_verification_subprocess_query_count"] == 2
    assert queries.evidence()["uuid_cache_hit_count"] == 0


def test_cached_startup_rejects_worker_device_active_device_mismatch(harness, monkeypatch):
    monkeypatch.setenv(UUID_QUERY_MODE_ENV, "cached")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2")
    worker = harness.worker(logical=0)
    harness.cuda.logical = 1
    with pytest.raises(RuntimeError, match="binding changed"):
        worker_dual_runtime_snapshot(worker)
    assert [int(call[2]) for call in harness.calls] == [1]
    assert not hasattr(worker.model_runner.drafter, "uuid_queries")


@pytest.mark.parametrize("mode", ["live", "cached"])
def test_two_rank_report_matches_actual_queries_and_verification_batches(
    harness, monkeypatch, mode,
):
    monkeypatch.setenv(UUID_QUERY_MODE_ENV, mode)
    workers, evidence = [], []
    for rank, physical in enumerate((1, 2)):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", str(physical))
        worker = harness.worker(rank=rank)
        workers.append(worker_dual_runtime_snapshot(worker))
        for _ in range(3):
            worker.model_runner.drafter.uuid_queries.for_verification()
        evidence.append(worker_dual_uuid_evidence(worker))
    # Two requests in one verification still cause just one UUID access per rank.
    logs = [{"verify_sequence": sequence, "target_rank_intervals": copy.deepcopy(evidence)}
            for sequence in (0, 0, 1, 1, 2, 2)]
    report = build_dual_uuid_query_report(evidence, workers, logs)
    assert report["valid"], report["errors"]
    assert report["uuid_initial_validation_count"] == 2
    assert report["uuid_verification_access_count"] == 6
    assert report["uuid_verification_subprocess_query_count"] == (6 if mode == "live" else 0)
    assert report["uuid_cache_hit_count"] == (0 if mode == "live" else 6)
    assert {row["gpu_uuid"] for row in report["uuid_query_by_rank"]} == set(UUIDS.values())
    assert len(harness.calls) == (8 if mode == "live" else 2)
    for field, value in (
        ("uuid_initial_validation_count", 0),
        ("uuid_verification_subprocess_query_count", 5),
        ("uuid_cache_hit_count", 5), ("uuid_query_mode", "unknown"),
        ("physical_gpu_id", 9), ("gpu_uuid", UUIDS[2]), ("tp_rank", 1),
    ):
        broken = copy.deepcopy(evidence)
        broken[0][field] = value
        assert not build_dual_uuid_query_report(broken, workers, logs)["valid"], field
    assert not build_dual_uuid_query_report(evidence[:1], workers, logs)["valid"]
    assert not build_dual_uuid_query_report(evidence, workers, [])["valid"]
    logs[0]["target_rank_intervals"][0]["gpu_uuid"] = UUIDS[2]
    assert not build_dual_uuid_query_report(evidence, workers, logs)["valid"]
    logs[0]["target_rank_intervals"] = []
    assert not build_dual_uuid_query_report(evidence, workers, logs)["valid"]


@pytest.mark.parametrize("mode", ["live", "cached"])
def test_real_dual_verify_end_uses_one_access_per_rank_per_batch(harness, monkeypatch, mode):
    monkeypatch.setenv(UUID_QUERY_MODE_ENV, mode)
    worker = harness.worker(rank=1)
    worker_dual_runtime_snapshot(worker)
    observer = DualBatchRemoteProposer.__new__(DualBatchRemoteProposer)
    observer.uuid_queries = worker.model_runner.drafter.uuid_queries
    observer.torch = harness.torch
    observer.tp_rank, observer.tp_world_size = 1, 2
    observer._verify_sequence = 0
    gathered = []
    event = SimpleNamespace(record=lambda: None, synchronize=lambda: None,
                            elapsed_time=lambda other: 0.1)
    harness.cuda.Event = lambda **kwargs: event
    harness.cuda.nvtx = SimpleNamespace(range_pop=lambda: None)
    observer.tp_group = SimpleNamespace(barrier=lambda: None, cpu_group=None)
    observer.dist = SimpleNamespace(
        get_rank=lambda: 1,
        all_gather_object=lambda rows, local, group: gathered.append(local),
    )
    for _ in range(3):
        observer._verify_start = {
            key: {"event": event, "request_id": key, "host_start_ns": 1}
            for key in ("A", "B")
        }
        observer.on_target_verify_end(
            request_ids=["A", "B"], sampled_token_ids=[[4], [5]],
            scheduled_spec_token_ids={"A": [1, 2], "B": [2, 3]},
        )
    assert len(harness.calls) == (4 if mode == "live" else 1)
    assert observer.uuid_queries.evidence()["uuid_verification_access_count"] == 3
    assert all(row["gpu_uuid"] == UUIDS[2] and row["physical_gpu_id"] == 2
               and row["logical_cuda_index"] == 0 for batch in gathered for row in batch)
    assert observer._verify_sequence == 3
    assert observer._verify_start == {}
    observer.on_target_verify_end(request_ids=["C"], sampled_token_ids=[[]],
                                  scheduled_spec_token_ids={})
    assert observer.uuid_queries.evidence()["uuid_verification_access_count"] == 3


@pytest.mark.parametrize("mode", ["live", "cached", "invalid-dual-only-setting"])
def test_target_and_serial_keep_original_paths(harness, monkeypatch, mode):
    monkeypatch.setenv(UUID_QUERY_MODE_ENV, mode)
    worker = harness.worker()
    # Both other runners still use the shared startup snapshot and finalizer.
    assert stock_vllm._worker_runtime_snapshot(worker)["gpu_uuid"] == UUIDS[2]
    assert stock_vllm._worker_performance_finalize(worker)["gpu_uuid"] == UUIDS[2]
    assert len(harness.calls) == 2
    assert not hasattr(worker.model_runner.drafter, "uuid_queries")
    serial = RemoteDraftProposer.__new__(RemoteDraftProposer)
    serial.torch = harness.torch
    serial.tp_group = SimpleNamespace(barrier=lambda: None)
    serial.tp_rank = 0
    serial.internal_to_stable = {"internal": "stable"}
    state = SimpleNamespace(pending_proposal=object(), verify_start_ns=1, verify_end_ns=None)
    serial.requests = {"stable": state}
    serial.hooks_seen = {"verify_end": 0}
    serial.on_target_verify_end(request_ids=["internal"], sampled_token_ids=[[5]],
                                scheduled_spec_token_ids={"internal": [1]})
    assert state.verify_end_ns > 1
    assert serial.hooks_seen["verify_end"] == 1
    assert len(harness.calls) == 2
