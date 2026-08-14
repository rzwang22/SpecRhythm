"""GPU-only Phase-3B latency interfaces; no synthetic timing fallback is allowed."""

from __future__ import annotations

import itertools
import math
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from specrhythm.phase3.config import Phase3Config
from specrhythm.phase3.engine import CausalLMBackend, EngineUnavailableError, create_backend


@dataclass(frozen=True)
class TimingStatistics:
    mean_ms: float
    std_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float


@dataclass(frozen=True)
class LatencyMeasurement:
    operation: str
    dimensions: dict[str, int]
    warmup_iterations: int
    measured_iterations: int
    cuda_event: TimingStatistics
    host_wall: TimingStatistics
    peak_gpu_memory_bytes: dict[str, int]
    actual_request_roots: int
    actual_search_pool_nodes: int
    actual_verified_candidate_nodes: int
    actual_target_input_positions: int
    implementation: str


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = math.ceil(fraction * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def _statistics(values: list[float]) -> TimingStatistics:
    return TimingStatistics(
        mean_ms=statistics.mean(values),
        std_ms=statistics.pstdev(values),
        p50_ms=_percentile(values, 0.50),
        p90_ms=_percentile(values, 0.90),
        p99_ms=_percentile(values, 0.99),
    )


class CudaBenchmarkTimer:
    """Measure one operation with separate CUDA-event and synchronized host clocks."""

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

    def measure(
        self,
        operation: Callable[[], None],
        *,
        name: str,
        dimensions: dict[str, int],
        devices: Iterable[int],
        actual_request_roots: int,
        actual_search_pool_nodes: int,
        actual_verified_candidate_nodes: int,
        actual_target_input_positions: int,
        implementation: str,
    ) -> LatencyMeasurement:
        torch = self.torch
        device_ids = tuple(sorted(set(devices)))
        for _ in range(self.warmup_iterations):
            operation()
        torch.cuda.synchronize()
        for device in device_ids:
            torch.cuda.reset_peak_memory_stats(device)
        cuda_times = []
        host_times = []
        for _ in range(self.measured_iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            host_start = time.perf_counter_ns()
            start.record()
            operation()
            end.record()
            end.synchronize()
            host_end = time.perf_counter_ns()
            cuda_times.append(float(start.elapsed_time(end)))
            host_times.append((host_end - host_start) / 1_000_000)
        peak = {
            str(device): int(torch.cuda.max_memory_allocated(device))
            for device in device_ids
        }
        return LatencyMeasurement(
            operation=name,
            dimensions=dimensions,
            warmup_iterations=self.warmup_iterations,
            measured_iterations=self.measured_iterations,
            cuda_event=_statistics(cuda_times),
            host_wall=_statistics(host_times),
            peak_gpu_memory_bytes=peak,
            actual_request_roots=actual_request_roots,
            actual_search_pool_nodes=actual_search_pool_nodes,
            actual_verified_candidate_nodes=actual_verified_candidate_nodes,
            actual_target_input_positions=actual_target_input_positions,
            implementation=implementation,
        )


def _context(length: int, vocab_size: int) -> list[int]:
    return [1 % vocab_size] * length


def _model_measurements(
    config: Phase3Config,
    *,
    operation: str,
    backend: CausalLMBackend,
) -> list[LatencyMeasurement]:
    benchmark = config.benchmark
    timer = CudaBenchmarkTimer(
        benchmark.warmup_iterations, benchmark.measured_iterations
    )
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
                    devices=(config.draft.gpu_ids[0],),
                    actual_request_roots=batch,
                    actual_search_pool_nodes=batch * search,
                    actual_verified_candidate_nodes=0,
                    actual_target_input_positions=0,
                    implementation=(
                        "transformers correctness collector; batched full-context forwards"
                    ),
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
                    devices=(
                        range(config.target.tp_size)
                        if config.target.tp_size > 1
                        else (config.target.gpu_ids[0],)
                    ),
                    actual_request_roots=batch,
                    actual_search_pool_nodes=0,
                    actual_verified_candidate_nodes=batch * candidates,
                    actual_target_input_positions=batch * (candidates + 1),
                    implementation=(
                        "transformers correctness verifier; serial full-context forwards, "
                        "not packed-tree serving"
                    ),
                )
            )
    return rows


def _selection_measurements(config: Phase3Config) -> list[LatencyMeasurement]:
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
                devices=(device,),
                actual_request_roots=batch,
                actual_search_pool_nodes=batch * search,
                actual_verified_candidate_nodes=batch * verify,
                actual_target_input_positions=0,
                implementation="device-side top-k selector primitive",
            )
        )
    return rows


def _transfer_measurements(config: Phase3Config) -> list[LatencyMeasurement]:
    benchmark = config.benchmark
    timer = CudaBenchmarkTimer(
        benchmark.warmup_iterations, benchmark.measured_iterations
    )
    torch = timer.torch
    source = config.draft.gpu_ids[0]
    target = config.target.gpu_ids[0]
    if source == target:
        raise ValueError("T_transfer requires distinct configured draft and target GPUs")
    rows = []
    for payload_bytes in benchmark.transfer_payload_bytes:
        payload = torch.empty(payload_bytes, dtype=torch.uint8, device=f"cuda:{source}")
        destination = torch.empty(
            payload_bytes, dtype=torch.uint8, device=f"cuda:{target}"
        )

        def transfer(destination=destination, payload=payload) -> None:
            destination.copy_(payload, non_blocking=True)

        with torch.cuda.device(target):
            rows.append(
                timer.measure(
                    transfer,
                    name="T_transfer",
                    dimensions={"payload_bytes": payload_bytes},
                    devices=(source, target),
                    actual_request_roots=0,
                    actual_search_pool_nodes=0,
                    actual_verified_candidate_nodes=0,
                    actual_target_input_positions=0,
                    implementation="direct CUDA peer tensor copy",
                )
            )
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
    rows: list[LatencyMeasurement] = []
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
    return {
        "schema_version": "specrhythm.gpu-latency.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gpu_measurement": True,
        "serving_engine_measurement": False,
        "synthetic_fallback_used": False,
        "python_version": platform.python_version(),
        "draft_model": asdict(config.draft),
        "target_model": asdict(config.target),
        "measurements": [
            {
                **asdict(row),
                "cuda_event": asdict(row.cuda_event),
                "host_wall": asdict(row.host_wall),
            }
            for row in rows
        ],
    }


def benchmark_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 3 GPU latency summary",
        "",
        "Correctness-backend calibration only; not a serving-engine performance claim.",
        "",
        "| Operation | Dimensions | CUDA mean/P90/P99 ms | Host mean/P90/P99 ms | Peak bytes |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in report["measurements"]:
        dimensions = ", ".join(f"{key}={value}" for key, value in row["dimensions"].items())
        cuda = row["cuda_event"]
        host = row["host_wall"]
        lines.append(
            f"| {row['operation']} | {dimensions} | "
            f"{cuda['mean_ms']:.3f}/{cuda['p90_ms']:.3f}/{cuda['p99_ms']:.3f} | "
            f"{host['mean_ms']:.3f}/{host['p90_ms']:.3f}/{host['p99_ms']:.3f} | "
            f"{max(row['peak_gpu_memory_bytes'].values(), default=0)} |"
        )
    return "\n".join(lines) + "\n"
