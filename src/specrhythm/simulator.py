"""Deterministic discrete-event simulator for Phase-A policy evaluation."""

from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from specrhythm.policies.base import PolicySnapshot, RequestView, SchedulingPolicy
from specrhythm.schema import Workload, WorkloadRequest


@dataclass(frozen=True)
class SimulatorConfig:
    max_active_requests: int = 64
    roof_candidate_budget: int = 192
    max_request_budget: int = 8
    verify_base_ms: float = 7.0
    verify_per_request_ms: float = 0.08
    verify_per_candidate_ms: float = 0.025
    draft_per_candidate_ms: float = 0.018
    fixed_speculative_budget: int = 4
    max_cycles: int = 1_000_000
    seed: int = 0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SimulatorConfig:
        return cls(**value)

    def __post_init__(self) -> None:
        if self.max_active_requests < 1 or self.roof_candidate_budget < 0:
            raise ValueError(
                "active request capacity must be positive and roof budget non-negative"
            )
        if self.max_request_budget < 0 or self.max_cycles < 1:
            raise ValueError("request budget must be non-negative and max_cycles positive")
        latency_values = (
            self.verify_base_ms,
            self.verify_per_request_ms,
            self.verify_per_candidate_ms,
            self.draft_per_candidate_ms,
        )
        if any(value < 0 or not math.isfinite(value) for value in latency_values):
            raise ValueError("latency parameters must be finite and non-negative")


@dataclass
class RuntimeRequest:
    request: WorkloadRequest
    slot: int
    admitted_at_ms: float
    delivered_tokens: int = 0
    elapsed_decode_ms: float = 0.0
    verify_attempts: int = 0
    accepted_draft_tokens: int = 0
    proposed_draft_tokens: int = 0
    recent_acceptance_ratio: float = 0.7


@dataclass(frozen=True)
class RequestResult:
    request_id: str
    task: str
    output_tokens: int
    slo_tpot_ms: float
    decode_latency_ms: float
    tpot_ms: float
    attained: bool


@dataclass(frozen=True)
class SimulationSummary:
    policy: str
    requests: int
    completed_requests: int
    measurement_ms: float
    throughput_tokens_per_s: float
    goodput_tokens_per_s: float
    slo_attainment: float
    p50_tpot_ms: Optional[float]
    p90_tpot_ms: Optional[float]
    p99_tpot_ms: Optional[float]
    proposed_draft_tokens: int
    accepted_draft_tokens: int
    eager_promotions: int
    eager_invalidations: int
    class_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SimulationResult:
    summary: SimulationSummary
    requests: tuple[RequestResult, ...]


def _stable_uniform(seed: int, request_id: str, attempt: int, depth: int) -> float:
    payload = f"{seed}:{request_id}:{attempt}:{depth}".encode()
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / float(2**64)


def _sample_progress(runtime: RuntimeRequest, budget: int, seed: int) -> tuple[int, int, bool]:
    accepted = 0
    probability = runtime.request.acceptance_probability
    for depth in range(1, budget + 1):
        if (
            _stable_uniform(seed, runtime.request.request_id, runtime.verify_attempts, depth)
            < probability
        ):
            accepted += 1
        else:
            break
    # Verification always yields a target token: either a correction or the token after a fully
    # accepted draft. This also makes budget zero equivalent to autoregressive progress.
    progress = accepted + 1
    fully_accepted = budget > 0 and accepted == budget
    return progress, accepted, fully_accepted


def _percentile(values: list[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _build_summary(
    policy_name: str,
    results: list[RequestResult],
    measurement_ms: float,
    proposed: int,
    accepted: int,
    eager_promotions: int,
    eager_invalidations: int,
) -> SimulationSummary:
    duration_s = measurement_ms / 1000.0
    total_tokens = sum(result.output_tokens for result in results)
    good_tokens = sum(result.output_tokens for result in results if result.attained)
    tpots = [result.tpot_ms for result in results]
    tasks = sorted({result.task for result in results})
    class_metrics: dict[str, dict[str, float]] = {}
    for task in tasks:
        subset = [result for result in results if result.task == task]
        class_metrics[task] = {
            "requests": float(len(subset)),
            "attainment": sum(result.attained for result in subset) / len(subset),
            "mean_tpot_ms": statistics.mean(result.tpot_ms for result in subset),
        }
    return SimulationSummary(
        policy=policy_name,
        requests=len(results),
        completed_requests=len(results),
        measurement_ms=measurement_ms,
        throughput_tokens_per_s=total_tokens / duration_s if duration_s else 0.0,
        goodput_tokens_per_s=good_tokens / duration_s if duration_s else 0.0,
        slo_attainment=sum(result.attained for result in results) / len(results)
        if results
        else 0.0,
        p50_tpot_ms=_percentile(tpots, 0.50),
        p90_tpot_ms=_percentile(tpots, 0.90),
        p99_tpot_ms=_percentile(tpots, 0.99),
        proposed_draft_tokens=proposed,
        accepted_draft_tokens=accepted,
        eager_promotions=eager_promotions,
        eager_invalidations=eager_invalidations,
        class_metrics=class_metrics,
    )


def simulate(
    workload: Workload, policy: SchedulingPolicy, config: SimulatorConfig
) -> SimulationResult:
    """Run one policy with dual logical slots and guarded eager promotion."""

    pending = list(workload.requests)
    pending_index = 0
    active: dict[str, RuntimeRequest] = {}
    ready_eager: set[str] = set()
    completed: list[RequestResult] = []
    now_ms = 0.0
    cycle = 0
    proposed_total = 0
    accepted_total = 0
    eager_promotions = 0
    eager_invalidations = 0
    first_admission_ms: Optional[float] = None

    def admit_ready() -> None:
        nonlocal pending_index, first_admission_ms
        while pending_index < len(pending) and len(active) < config.max_active_requests:
            request = pending[pending_index]
            if request.arrival_time_ms > now_ms:
                break
            slot_sizes = [sum(item.slot == slot for item in active.values()) for slot in (0, 1)]
            slot = 0 if slot_sizes[0] <= slot_sizes[1] else 1
            active[request.request_id] = RuntimeRequest(
                request=request,
                slot=slot,
                admitted_at_ms=now_ms,
                recent_acceptance_ratio=request.acceptance_probability,
            )
            if first_admission_ms is None:
                first_admission_ms = now_ms
            pending_index += 1

    while pending_index < len(pending) or active:
        if cycle >= config.max_cycles:
            raise RuntimeError("simulation exceeded max_cycles")
        if not active:
            now_ms = max(now_ms, pending[pending_index].arrival_time_ms)
        admit_ready()
        if not active:
            continue

        verify_slot = cycle % 2
        eligible_ids = {
            request_id for request_id, runtime in active.items() if runtime.slot == verify_slot
        }
        eligible_ids.update(request_id for request_id in ready_eager if request_id in active)
        ready_eager.clear()
        eligible = [active[key] for key in sorted(eligible_ids)]
        if not eligible:
            cycle += 1
            continue

        base_verify_ms = config.verify_base_ms + config.verify_per_request_ms * len(eligible)
        normal_wait_ms = 2.0 * base_verify_ms
        views = tuple(
            RequestView(
                request_id=runtime.request.request_id,
                delivered_tokens=runtime.delivered_tokens,
                elapsed_decode_ms=runtime.elapsed_decode_ms,
                slo_tpot_ms=runtime.request.slo_tpot_ms,
                acceptance_ratio=runtime.recent_acceptance_ratio,
                draft_confidence=runtime.request.acceptance_probability,
                waiting_time_ms=normal_wait_ms,
                max_budget=config.max_request_budget,
            )
            for runtime in eligible
        )
        residual_draft_tokens = (
            math.floor(base_verify_ms / config.draft_per_candidate_ms)
            if config.draft_per_candidate_ms > 0
            else config.roof_candidate_budget
        )
        snapshot = PolicySnapshot(
            requests=views,
            roof_candidate_budget=config.roof_candidate_budget,
            residual_draft_tokens=residual_draft_tokens,
        )
        plan = policy.plan(snapshot)
        if plan.total_candidates > config.roof_candidate_budget:
            raise ValueError(f"policy {policy.name} exceeded the roof candidate budget")
        if set(plan.budgets) - eligible_ids:
            raise ValueError(f"policy {policy.name} allocated a non-eligible request")

        total_candidates = plan.total_candidates
        verify_ms = base_verify_ms + config.verify_per_candidate_ms * total_candidates
        eager_draft_tokens = sum(plan.budgets.get(key, 0) for key in plan.eager_request_ids)
        draft_ms = config.draft_per_candidate_ms * (total_candidates + eager_draft_tokens)
        cycle_ms = verify_ms + max(0.0, draft_ms - verify_ms)
        now_ms += cycle_ms
        for runtime in active.values():
            runtime.elapsed_decode_ms += cycle_ms

        finished_ids: list[str] = []
        eager_set = set(plan.eager_request_ids)
        for runtime in eligible:
            request_id = runtime.request.request_id
            budget = int(plan.budgets.get(request_id, 0))
            if budget < 0 or budget > config.max_request_budget:
                raise ValueError(f"policy {policy.name} returned invalid per-request budget")
            progress, accepted, fully_accepted = _sample_progress(runtime, budget, config.seed)
            remaining = runtime.request.output_tokens - runtime.delivered_tokens
            runtime.delivered_tokens += min(progress, remaining)
            runtime.verify_attempts += 1
            runtime.proposed_draft_tokens += budget
            runtime.accepted_draft_tokens += accepted
            proposed_total += budget
            accepted_total += accepted
            if budget > 0:
                observed = accepted / budget
                runtime.recent_acceptance_ratio = (
                    0.8 * runtime.recent_acceptance_ratio + 0.2 * observed
                )

            if request_id in eager_set:
                if fully_accepted:
                    ready_eager.add(request_id)
                    eager_promotions += 1
                else:
                    eager_invalidations += 1

            if runtime.delivered_tokens >= runtime.request.output_tokens:
                tpot = runtime.elapsed_decode_ms / runtime.request.output_tokens
                completed.append(
                    RequestResult(
                        request_id=request_id,
                        task=runtime.request.task,
                        output_tokens=runtime.request.output_tokens,
                        slo_tpot_ms=runtime.request.slo_tpot_ms,
                        decode_latency_ms=runtime.elapsed_decode_ms,
                        tpot_ms=tpot,
                        attained=tpot <= runtime.request.slo_tpot_ms,
                    )
                )
                finished_ids.append(request_id)

        for request_id in finished_ids:
            active.pop(request_id)
            ready_eager.discard(request_id)
        cycle += 1
        admit_ready()

    measurement_ms = max(0.0, now_ms - (first_admission_ms or 0.0))
    summary = _build_summary(
        policy.name,
        completed,
        measurement_ms,
        proposed_total,
        accepted_total,
        eager_promotions,
        eager_invalidations,
    )
    return SimulationResult(summary=summary, requests=tuple(completed))
