"""GPU-only Phase-3B.1 correctness-primitive measurement interfaces."""

from __future__ import annotations

import itertools
import json
import math
import os
import platform
import statistics
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from specrhythm.phase3.config import Phase3Config, resolve_runtime_path
from specrhythm.phase3.engine import CausalLMBackend, EngineUnavailableError, create_backend
from specrhythm.phase3.hardware import (
    capture_hardware_state,
    cuda_visible_devices_mapping,
    physical_gpu_id,
)
from specrhythm.phase3.trace import sha256_file

BACKEND_SEMANTICS = {
    "backend": "hf_correctness",
    "runtime_backend": "transformers",
    "serving_engine": False,
    "kv_cache_reuse": False,
    "packed_tree_verification": False,
    "simulator_latency_surface_compatible": False,
    "result_name": "correctness-backend primitive latency",
}


@dataclass(frozen=True)
class TimingStatistics:
    raw_samples_ms: tuple[float, ...]
    mean_ms: float
    std_ms: float
    cv: float
    min_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    outlier_indices: tuple[int, ...]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = math.ceil(fraction * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def timing_statistics(values: Iterable[float]) -> TimingStatistics:
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("timing statistics require at least one sample")
    mean = statistics.mean(samples)
    std = statistics.pstdev(samples)
    outliers = tuple(
        index
        for index, value in enumerate(samples)
        if std > 0 and abs(value - mean) > 3.0 * std
    )
    return TimingStatistics(
        raw_samples_ms=tuple(samples),
        mean_ms=mean,
        std_ms=std,
        cv=std / mean if mean else 0.0,
        min_ms=min(samples),
        p50_ms=_percentile(samples, 0.50),
        p90_ms=_percentile(samples, 0.90),
        p95_ms=_percentile(samples, 0.95),
        p99_ms=_percentile(samples, 0.99),
        max_ms=max(samples),
        outlier_indices=outliers,
    )


def aggregate_rank_samples(
    rank_records: Iterable[dict[str, Any]], sample_key: str
) -> list[float]:
    records = list(rank_records)
    if not records:
        raise ValueError("cannot aggregate timing without rank records")
    sample_counts = {len(record.get(sample_key, [])) for record in records}
    if len(sample_counts) != 1 or not sample_counts or next(iter(sample_counts)) == 0:
        raise ValueError(f"rank {sample_key} sample counts are missing or inconsistent")
    count = next(iter(sample_counts))
    return [
        max(float(record[sample_key][index]) for record in records)
        for index in range(count)
    ]


def atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _rank_identity(torch: Any, logical_device: int) -> dict[str, Any]:
    distributed = torch.distributed
    initialized = bool(distributed.is_initialized())
    global_rank = int(distributed.get_rank()) if initialized else int(os.getenv("RANK", "0"))
    world_size = (
        int(distributed.get_world_size())
        if initialized
        else int(os.getenv("WORLD_SIZE", "1"))
    )
    local_rank = int(os.getenv("LOCAL_RANK", str(logical_device)))
    properties = torch.cuda.get_device_properties(logical_device)
    return {
        "global_rank": global_rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "logical_cuda_index": logical_device,
        "physical_gpu_id": physical_gpu_id(logical_device, torch.cuda.device_count()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_visible_devices_mapping": cuda_visible_devices_mapping(
            torch.cuda.device_count()
        ),
        "gpu_uuid": str(getattr(properties, "uuid", "")) or None,
        "model_parameter_count": None,
        "parameter_bytes": None,
        "parameter_devices": [],
        "expected_parameter_device": None,
        "model_parameters_on_expected_device": None,
        "forward_input_shape": [],
        "forward_output_shape": [],
        "output_checksum": None,
        "forward_invocations": 0,
    }


class CudaBenchmarkTimer:
    """Measure per-rank CUDA/host samples and aggregate a max-rank critical path."""

    def __init__(self, warmup_iterations: int, measured_iterations: int) -> None:
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError as error:
            raise EngineUnavailableError("GPU benchmark requires PyTorch") from error
        if not torch.cuda.is_available():
            raise EngineUnavailableError(
                "GPU benchmark requires CUDA and never substitutes synthetic latency"
            )
        self.torch = torch
        self.warmup_iterations = warmup_iterations
        self.measured_iterations = measured_iterations

    def _barrier(self) -> None:
        distributed = self.torch.distributed
        if distributed.is_initialized() and distributed.get_world_size() > 1:
            distributed.barrier()

    def _gather(self, value: dict[str, Any]) -> list[dict[str, Any]]:
        distributed = self.torch.distributed
        if not distributed.is_initialized() or distributed.get_world_size() == 1:
            return [value]
        gathered: list[Optional[dict[str, Any]]] = [
            None for _ in range(distributed.get_world_size())
        ]
        distributed.all_gather_object(gathered, value)
        if any(item is None for item in gathered):
            raise RuntimeError("distributed timing gather returned a missing rank")
        return [item for item in gathered if item is not None]

    def measure(
        self,
        operation: Callable[[], None],
        *,
        name: str,
        dimensions: dict[str, Any],
        timing_device: int,
        memory_devices: Iterable[int],
        actual_request_roots: int,
        actual_search_pool_nodes: int,
        actual_verified_candidate_nodes: int,
        actual_target_input_positions: int,
        implementation: str,
        operation_semantics: dict[str, Any],
        rank_metadata_provider: Optional[Callable[[], dict[str, Any]]] = None,
        requires_model_rank_evidence: bool = False,
    ) -> dict[str, Any]:
        torch = self.torch
        memory_device_ids = tuple(sorted(set(memory_devices)))
        with torch.cuda.device(timing_device):
            for _ in range(self.warmup_iterations):
                self._barrier()
                torch.cuda.synchronize(timing_device)
                operation()
                torch.cuda.synchronize(timing_device)
                self._barrier()
            for device in memory_device_ids:
                torch.cuda.reset_peak_memory_stats(device)
            cuda_samples = []
            host_samples = []
            for _ in range(self.measured_iterations):
                self._barrier()
                torch.cuda.synchronize(timing_device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                host_start = time.perf_counter_ns()
                start.record()
                operation()
                end.record()
                end.synchronize()
                host_end = time.perf_counter_ns()
                cuda_samples.append(float(start.elapsed_time(end)))
                host_samples.append((host_end - host_start) / 1_000_000)
                self._barrier()
        rank_record = (
            rank_metadata_provider()
            if rank_metadata_provider is not None
            else _rank_identity(torch, timing_device)
        )
        rank_record.update(
            {
                "cuda_samples_ms": cuda_samples,
                "host_samples_ms": host_samples,
                "device_memory": {
                    str(device): {
                        "allocated_memory_bytes": int(
                            torch.cuda.memory_allocated(device)
                        ),
                        "reserved_memory_bytes": int(torch.cuda.memory_reserved(device)),
                        "max_allocated_memory_bytes": int(
                            torch.cuda.max_memory_allocated(device)
                        ),
                        "max_reserved_memory_bytes": int(
                            torch.cuda.max_memory_reserved(device)
                        ),
                    }
                    for device in memory_device_ids
                },
            }
        )
        if rank_metadata_provider is None:
            primary = rank_record["device_memory"][str(timing_device)]
            rank_record.update(primary)
        rank_records = sorted(
            self._gather(rank_record), key=lambda row: int(row["global_rank"])
        )
        global_cuda = aggregate_rank_samples(rank_records, "cuda_samples_ms")
        global_host = aggregate_rank_samples(rank_records, "host_samples_ms")
        return {
            "operation": name,
            "dimensions": dimensions,
            "warmup_iterations": self.warmup_iterations,
            "measured_iterations": self.measured_iterations,
            "cuda_event": asdict(timing_statistics(global_cuda)),
            "host_wall": asdict(timing_statistics(global_host)),
            "rank_measurements": rank_records,
            "requires_model_rank_evidence": requires_model_rank_evidence,
            "global_latency_definition": (
                "per-iteration maximum latency across all participating ranks"
            ),
            "timing_protocol": {
                "distributed_barrier_before_iteration": True,
                "cuda_synchronize_before_iteration": True,
                "cuda_event_per_rank": True,
                "host_clock_per_rank": True,
                "cuda_event_synchronize_before_host_stop": True,
                "distributed_barrier_after_iteration": True,
                "raw_samples_retained": True,
                "outliers_removed": False,
            },
            "actual_request_roots": actual_request_roots,
            "actual_search_pool_nodes": actual_search_pool_nodes,
            "actual_verified_candidate_nodes": actual_verified_candidate_nodes,
            "actual_target_input_positions": actual_target_input_positions,
            "implementation": implementation,
            "operation_semantics": operation_semantics,
        }


def _context(length: int, vocab_size: int) -> list[int]:
    return [1 % vocab_size] * length


def _model_identity(config: Phase3Config) -> dict[str, Any]:
    result = {}
    for role, model in (("draft", config.draft), ("target", config.target)):
        model_path = Path(resolve_runtime_path(model.model_path))
        config_path = model_path / "config.json"
        if not config_path.is_file():
            raise ValueError(f"{role} model config is missing: {config_path}")
        result[role] = {
            "model_path": str(model_path),
            "configured_revision": model.revision,
            "config_file": config_path.name,
            "config_sha256": sha256_file(config_path),
        }
    return result


def _runtime_versions() -> dict[str, Any]:
    import torch  # type: ignore[import-not-found]
    import transformers  # type: ignore[import-not-found]

    try:
        nccl = torch.cuda.nccl.version()
    except (AttributeError, RuntimeError):
        nccl = None
    if isinstance(nccl, tuple):
        nccl = ".".join(str(item) for item in nccl)
    return {
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "nccl": str(nccl) if nccl is not None else None,
    }


def _model_measurements(
    config: Phase3Config,
    *,
    operation: str,
    backend: CausalLMBackend,
) -> list[dict[str, Any]]:
    benchmark = config.benchmark
    timer = CudaBenchmarkTimer(
        benchmark.warmup_iterations, benchmark.measured_iterations
    )
    metadata_provider = getattr(backend, "benchmark_rank_metadata", None)
    device = getattr(backend, "device", None)
    if metadata_provider is None or device is None or device.index is None:
        raise EngineUnavailableError(
            "GPU model benchmark requires rank metadata and an explicit CUDA device"
        )
    timing_device = int(device.index)
    rows = []
    if operation == "draft":
        combinations = itertools.product(
            benchmark.request_batch_sizes,
            benchmark.search_pool_sizes,
            benchmark.context_lengths,
        )
        for batch, search, context_length in combinations:
            contexts = [_context(context_length, backend.vocab_size) for _ in range(batch)]

            def draft(contexts=contexts, search=search) -> None:
                generated = [[] for _ in contexts]
                for _ in range(search):
                    distributions = backend.next_token_batch(
                        [base + suffix for base, suffix in zip(contexts, generated)], 2
                    )
                    for suffix, distribution in zip(generated, distributions):
                        suffix.append(distribution.top1.token_id)

            rows.append(
                timer.measure(
                    draft,
                    name="T_draft",
                    dimensions={
                        "B_req": batch,
                        "N_search": search,
                        "context_length": context_length,
                    },
                    timing_device=timing_device,
                    memory_devices=(timing_device,),
                    actual_request_roots=batch,
                    actual_search_pool_nodes=batch * search,
                    actual_verified_candidate_nodes=0,
                    actual_target_input_positions=0,
                    implementation="HF correctness collector; batched full-context forwards",
                    operation_semantics={
                        "search_generation_semantics": (
                            "serial greedy full-context replay without KV-cache reuse"
                        ),
                        "number_of_model_forwards": search,
                        "tokens_or_nodes_per_forward": batch,
                        "N_search_definition": (
                            "sequential next-token forwards per request; total generated "
                            "nodes are B_req*N_search"
                        ),
                    },
                    rank_metadata_provider=metadata_provider,
                    requires_model_rank_evidence=True,
                )
            )
    else:
        combinations = itertools.product(
            benchmark.request_batch_sizes,
            benchmark.verify_candidate_sizes,
            benchmark.context_lengths,
        )
        for batch, candidates, context_length in combinations:
            contexts = [_context(context_length, backend.vocab_size) for _ in range(batch)]

            def verify(contexts=contexts, candidates=candidates) -> None:
                generated = [[] for _ in contexts]
                for _ in range(candidates + 1):
                    distributions = backend.next_token_batch(
                        [base + suffix for base, suffix in zip(contexts, generated)], 2
                    )
                    for suffix, distribution in zip(generated, distributions):
                        suffix.append(distribution.top1.token_id)

            rows.append(
                timer.measure(
                    verify,
                    name="T_verify",
                    dimensions={
                        "B_req": batch,
                        "B_cand": candidates,
                        "context_length": context_length,
                        "TP": config.target.tp_size,
                    },
                    timing_device=timing_device,
                    memory_devices=(timing_device,),
                    actual_request_roots=batch,
                    actual_search_pool_nodes=0,
                    actual_verified_candidate_nodes=batch * candidates,
                    actual_target_input_positions=batch * (candidates + 1),
                    implementation=(
                        "HF correctness verifier; serial full-context replay, not packed-tree"
                    ),
                    operation_semantics={
                        "verify_implementation": "serial_full_context_replay",
                        "number_of_target_forwards": candidates + 1,
                        "B_cand_definition": (
                            "non-root candidate steps replayed serially for each request"
                        ),
                        "target_input_positions": batch * (candidates + 1),
                    },
                    rank_metadata_provider=metadata_provider,
                    requires_model_rank_evidence=True,
                )
            )
    return rows


def _selection_measurements(config: Phase3Config) -> list[dict[str, Any]]:
    benchmark = config.benchmark
    timer = CudaBenchmarkTimer(
        benchmark.warmup_iterations, benchmark.measured_iterations
    )
    torch = timer.torch
    device = config.draft.gpu_ids[0]
    rows = []
    for batch, search, verify in itertools.product(
        benchmark.request_batch_sizes,
        benchmark.search_pool_sizes,
        benchmark.verify_candidate_sizes,
    ):
        if verify > search:
            continue
        scores = torch.rand((batch, search), device=f"cuda:{device}")

        def select(scores=scores, verify=verify) -> None:
            torch.topk(scores, k=verify, dim=-1, sorted=True)

        rows.append(
            timer.measure(
                select,
                name="T_select",
                dimensions={"B_req": batch, "N_search": search, "B_verify": verify},
                timing_device=device,
                memory_devices=(device,),
                actual_request_roots=batch,
                actual_search_pool_nodes=batch * search,
                actual_verified_candidate_nodes=batch * verify,
                actual_target_input_positions=0,
                implementation="synthetic device-side torch.topk primitive",
                operation_semantics={
                    "selector_backend": "synthetic_topk",
                    "prefix_closure": False,
                    "tree_materialization": False,
                    "stable_node_bookkeeping": False,
                    "cpu_scheduler": False,
                    "gpu_cpu_synchronization_included": True,
                },
            )
        )
    return rows


def _transfer_pairs(config: Phase3Config) -> list[tuple[str, int, int]]:
    draft = config.draft.gpu_ids[0]
    leader = config.target.gpu_ids[0]
    pairs = [
        ("draft_to_target_leader", draft, leader),
        ("target_leader_to_draft", leader, draft),
    ]
    if len(config.target.gpu_ids) > 1:
        pairs.append(
            ("target_leader_to_target_tp_peer", leader, config.target.gpu_ids[1])
        )
    result = []
    seen = set()
    for value in pairs:
        if value[1] == value[2] or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _transfer_measurements(config: Phase3Config) -> list[dict[str, Any]]:
    benchmark = config.benchmark
    timer = CudaBenchmarkTimer(
        benchmark.warmup_iterations, benchmark.measured_iterations
    )
    torch = timer.torch
    rows = []
    for direction, source, destination in _transfer_pairs(config):
        if source >= torch.cuda.device_count() or destination >= torch.cuda.device_count():
            raise ValueError(
                f"transfer pair {source}->{destination} is not visible; unset or correct "
                "CUDA_VISIBLE_DEVICES"
            )
        peer_available = bool(torch.cuda.can_device_access_peer(source, destination))
        for payload_bytes in benchmark.transfer_payload_bytes:
            payload = torch.empty(
                payload_bytes, dtype=torch.uint8, device=f"cuda:{source}"
            )
            output = torch.empty(
                payload_bytes, dtype=torch.uint8, device=f"cuda:{destination}"
            )

            def transfer(output=output, payload=payload) -> None:
                output.copy_(payload, non_blocking=True)

            with torch.cuda.device(destination):
                row = timer.measure(
                    transfer,
                    name="T_transfer",
                    dimensions={
                        "payload_bytes": payload_bytes,
                        "direction": direction,
                    },
                    timing_device=destination,
                    memory_devices=(source, destination),
                    actual_request_roots=0,
                    actual_search_pool_nodes=0,
                    actual_verified_candidate_nodes=0,
                    actual_target_input_positions=0,
                    implementation="bare direct CUDA tensor copy primitive",
                    operation_semantics={
                        "transport_scope": "bare_device_copy_only",
                        "complete_draft_to_verify_transport": False,
                    },
                )
            mean_ms = float(row["cuda_event"]["mean_ms"])
            row["transfer_metadata"] = {
                "copy_direction": f"cuda:{source}->cuda:{destination}",
                "direction_role": direction,
                "source_logical_cuda_index": source,
                "source_physical_gpu_id": physical_gpu_id(
                    source, torch.cuda.device_count()
                ),
                "source_gpu_uuid": str(
                    getattr(torch.cuda.get_device_properties(source), "uuid", "")
                )
                or None,
                "destination_logical_cuda_index": destination,
                "destination_physical_gpu_id": physical_gpu_id(
                    destination, torch.cuda.device_count()
                ),
                "destination_gpu_uuid": str(
                    getattr(torch.cuda.get_device_properties(destination), "uuid", "")
                )
                or None,
                "cuda_device_can_access_peer": peer_available,
                "p2p_enabled": peer_available,
                "p2p_status_basis": (
                    "torch.cuda.can_device_access_peer plus successful direct CUDA tensor copy"
                ),
                "host_staging": False if peer_available else None,
                "effective_bandwidth_gbps": (
                    payload_bytes / (mean_ms * 1_000_000) if mean_ms > 0 else None
                ),
                "topology_source": "hardware_state_before.nvlink_pcie_topology",
            }
            rows.append(row)
            del output
            del payload
            torch.cuda.empty_cache()
    return rows


def run_latency_benchmark(
    config: Phase3Config, operations: Iterable[str]
) -> dict[str, Any]:
    if config.backend != "transformers":
        raise EngineUnavailableError(
            "latency calibration requires backend=transformers; dry-run timing is forbidden"
        )
    selected = tuple(dict.fromkeys(operations))
    unknown = set(selected) - {"draft", "select", "verify", "transfer"}
    if unknown:
        raise ValueError(f"unknown benchmark operation(s): {sorted(unknown)}")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and selected != ("verify",):
        raise ValueError("torchrun Phase-3 benchmark supports the verify operation only")
    physical_ids = tuple(sorted(set(config.draft.gpu_ids + config.target.gpu_ids)))
    hardware_before = capture_hardware_state(physical_ids)
    rows: list[dict[str, Any]] = []
    for operation in selected:
        if operation == "select":
            rows.extend(_selection_measurements(config))
            continue
        if operation == "transfer":
            rows.extend(_transfer_measurements(config))
            continue
        model_config = config.draft if operation == "draft" else config.target
        backend = create_backend(config.backend, model_config, config.random_seed)
        try:
            rows.extend(_model_measurements(config, operation=operation, backend=backend))
        finally:
            backend.close()
    report = {
        "schema_version": "specrhythm.gpu-latency.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gpu_measurement": True,
        "measurement_scope": "correctness-backend primitive latency",
        "backend_semantics": dict(BACKEND_SEMANTICS),
        "serving_engine_measurement": False,
        "synthetic_fallback_used": False,
        "simulator_latency_surface_compatible": False,
        "python_version": platform.python_version(),
        "runtime_versions": _runtime_versions(),
        "draft_model": asdict(config.draft),
        "target_model": asdict(config.target),
        "model_identity": _model_identity(config),
        "benchmark_config": asdict(config.benchmark),
        "operation_order": list(selected),
        "hardware_state_before": hardware_before,
        "hardware_state_after": capture_hardware_state(physical_ids),
        "measurements": rows,
    }
    from specrhythm.phase3.benchmark_validation import validate_benchmark_report

    report["validation"] = validate_benchmark_report(report)
    return report


def benchmark_markdown(report: dict[str, Any]) -> str:
    validation = report.get("validation", {})
    lines = [
        "# Phase 3B.1 GPU primitive latency summary",
        "",
        "Correctness-backend primitive latency only; not a serving-engine or simulator surface.",
        "",
        f"Validation: **{'PASS' if validation.get('valid') else 'FAIL'}**",
        "",
        "| Operation | Dimensions | CUDA mean/CV/P90/P95/P99 ms | "
        "Host mean/P99 ms | Rank evidence |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in report["measurements"]:
        dimensions = ", ".join(
            f"{key}={value}" for key, value in row["dimensions"].items()
        )
        cuda = row["cuda_event"]
        host = row["host_wall"]
        ranks = ", ".join(
            f"r{rank['global_rank']}:{rank.get('physical_gpu_id')} "
            f"params={rank.get('model_parameter_count')} "
            f"peak={rank.get('max_allocated_memory_bytes')}"
            for rank in row["rank_measurements"]
        )
        lines.append(
            f"| {row['operation']} | {dimensions} | "
            f"{cuda['mean_ms']:.3f}/{cuda['cv']:.4f}/{cuda['p90_ms']:.3f}/"
            f"{cuda['p95_ms']:.3f}/{cuda['p99_ms']:.3f} | "
            f"{host['mean_ms']:.3f}/{host['p99_ms']:.3f} | {ranks} |"
        )
    if validation.get("warnings"):
        lines.extend(["", "## Validation warnings", ""])
        lines.extend(f"- {warning}" for warning in validation["warnings"])
    return "\n".join(lines) + "\n"
