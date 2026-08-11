"""Workload generation, trace composition, and Mooncake normalization."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from specrhythm.schema import Workload, WorkloadRequest


@dataclass(frozen=True)
class TaskProfile:
    name: str
    weight: float
    slo_tpot_ms: float
    input_median: float
    input_sigma: float
    output_median: float
    output_sigma: float
    length_correlation: float
    acceptance_probability: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TaskProfile:
        return cls(**value)


@dataclass(frozen=True)
class ArrivalReplay:
    """A one-to-one, chronologically selected arrival-trace window."""

    source_timestamps_ms: tuple[float, ...]
    arrival_times_ms: tuple[float, ...]
    window_start_ms: float
    window_duration_ms: Optional[float]
    time_scale: float


def load_json(path: Union[str, Path]) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("configuration must be a JSON object")
    return value


def _weighted_choice(rng: random.Random, values: Sequence[Any], weights: Sequence[float]) -> Any:
    if not values or len(values) != len(weights):
        raise ValueError("values and weights must be non-empty and have equal length")
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    threshold = rng.random() * sum(weights)
    cumulative = 0.0
    for value, weight in zip(values, weights):
        cumulative += weight
        if threshold <= cumulative:
            return value
    return values[-1]


def _piecewise_gamma_arrivals(config: dict[str, Any], rng: random.Random) -> list[float]:
    arrivals: list[float] = []
    segment_start_s = 0.0
    for segment in config.get("arrival_segments", []):
        duration_s = float(segment["duration_s"])
        rate = float(segment["rate_per_s"])
        cv = float(segment.get("cv", 1.0))
        if duration_s <= 0 or rate <= 0 or cv <= 0:
            raise ValueError("arrival segment duration, rate, and CV must be positive")

        shape = 1.0 / (cv * cv)
        scale = (1.0 / rate) / shape
        cursor_s = segment_start_s
        segment_end_s = segment_start_s + duration_s
        while True:
            cursor_s += rng.gammavariate(shape, scale)
            if cursor_s >= segment_end_s:
                break
            arrivals.append(cursor_s * 1000.0)
        segment_start_s = segment_end_s
    if not config.get("arrival_segments"):
        raise ValueError("arrival_segments must contain at least one segment")
    return arrivals


def _read_arrival_timestamps(path: Union[str, Path]) -> list[float]:
    source = Path(path)
    timestamps: list[float] = []
    if source.suffix.lower() == ".csv":
        with source.open(encoding="utf-8", newline="") as handle:
            for line_number, row in enumerate(csv.DictReader(handle), start=2):
                raw = row.get("arrival_time_ms", row.get("timestamp", row.get("Timestamp")))
                if raw is None:
                    raise ValueError(
                        f"{source}:{line_number}: CSV requires arrival_time_ms, "
                        "timestamp, or Timestamp"
                    )
                try:
                    timestamp = float(raw)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{source}:{line_number}: timestamp must be numeric"
                    ) from error
                timestamps.append(timestamp)
    else:
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{source}:{line_number}: invalid JSON") from error
                if not isinstance(row, dict):
                    raise ValueError(f"{source}:{line_number}: JSONL row must be an object")
                raw = row.get("arrival_time_ms", row.get("timestamp"))
                if raw is None:
                    raise ValueError(
                        f"{source}:{line_number}: JSONL requires arrival_time_ms or timestamp"
                    )
                try:
                    timestamp = float(raw)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{source}:{line_number}: timestamp must be numeric"
                    ) from error
                timestamps.append(timestamp)
    if any(not math.isfinite(timestamp) or timestamp < 0 for timestamp in timestamps):
        raise ValueError("arrival timestamps must be finite and non-negative")
    return timestamps


def select_arrival_replay(
    path: Union[str, Path],
    *,
    window_start_ms: float = 0.0,
    window_duration_ms: Optional[float] = None,
    time_scale: float = 1.0,
) -> ArrivalReplay:
    """Select a stable, end-exclusive trace window and rebase it to zero.

    The selected source timestamps are sorted chronologically. Equal timestamps keep
    their source-file order because Python's sort is stable. Scaling divides only the
    gaps from the first selected timestamp, so ``time_scale=2`` doubles offered load.
    """

    if not math.isfinite(window_start_ms) or window_start_ms < 0:
        raise ValueError("window_start_ms must be finite and non-negative")
    if not math.isfinite(time_scale) or time_scale <= 0:
        raise ValueError("time_scale must be finite and positive")
    if window_duration_ms is not None and (
        not math.isfinite(window_duration_ms) or window_duration_ms <= 0
    ):
        raise ValueError("window_duration_ms must be finite and positive")

    timestamps = sorted(_read_arrival_timestamps(path))
    window_end_ms = (
        window_start_ms + window_duration_ms if window_duration_ms is not None else None
    )
    selected = tuple(
        timestamp
        for timestamp in timestamps
        if timestamp >= window_start_ms
        and (window_end_ms is None or timestamp < window_end_ms)
    )
    if not selected:
        raise ValueError("the selected arrival window contains no timestamps")
    origin = selected[0]
    scaled = tuple((timestamp - origin) / time_scale for timestamp in selected)
    return ArrivalReplay(
        source_timestamps_ms=selected,
        arrival_times_ms=scaled,
        window_start_ms=float(window_start_ms),
        window_duration_ms=(
            float(window_duration_ms) if window_duration_ms is not None else None
        ),
        time_scale=float(time_scale),
    )


def load_arrival_times(path: Union[str, Path], time_scale: float = 1.0) -> list[float]:
    """Backward-compatible helper for selecting the full arrival trace."""

    replay = select_arrival_replay(path, time_scale=time_scale)
    return list(replay.arrival_times_ms)


def _sample_lengths(
    rng: random.Random,
    profile: TaskProfile,
    max_input: int,
    max_output: int,
    turn_index: int = 0,
) -> tuple[int, int]:
    first = rng.gauss(0.0, 1.0)
    residual = rng.gauss(0.0, 1.0)
    rho = max(-0.99, min(0.99, profile.length_correlation))
    second = rho * first + math.sqrt(1.0 - rho * rho) * residual
    turn_growth = 1.0 + 0.12 * turn_index
    input_tokens = round(
        profile.input_median * math.exp(profile.input_sigma * first) * turn_growth
    )
    output_tokens = round(profile.output_median * math.exp(profile.output_sigma * second))
    return (
        max(1, min(max_input, input_tokens)),
        max(1, min(max_output, output_tokens)),
    )


def _sample_slo(config: dict[str, Any], rng: random.Random, profile: TaskProfile) -> float:
    mode = config.get("slo_mode", "task")
    if mode == "task":
        return profile.slo_tpot_ms
    if mode == "independent":
        classes = config.get("slo_classes", [])
        chosen = _weighted_choice(rng, classes, [float(item["weight"]) for item in classes])
        return float(chosen["tpot_ms"])
    raise ValueError("slo_mode must be 'task' or 'independent'")


def _assign_profiles(
    config: dict[str, Any],
    rng: random.Random,
    profiles: Sequence[TaskProfile],
    request_count: int,
) -> list[TaskProfile]:
    mode = config.get("task_assignment", "random")
    if mode == "random":
        weights = [profile.weight for profile in profiles]
        return [_weighted_choice(rng, profiles, weights) for _ in range(request_count)]
    if mode != "stratified":
        raise ValueError("task_assignment must be 'random' or 'stratified'")

    weight_sum = sum(profile.weight for profile in profiles)
    exact_counts = [request_count * profile.weight / weight_sum for profile in profiles]
    counts = [math.floor(value) for value in exact_counts]
    remainder = request_count - sum(counts)
    ranked = sorted(
        range(len(profiles)),
        key=lambda index: (-(exact_counts[index] - counts[index]), index),
    )
    for index in ranked[:remainder]:
        counts[index] += 1
    assignments = [
        profile
        for profile, count in zip(profiles, counts)
        for _ in range(count)
    ]
    rng.shuffle(assignments)
    return assignments


def generate_replay_workload(config: dict[str, Any], replay: ArrivalReplay) -> Workload:
    """Generate exactly one proxy request for every selected arrival timestamp."""

    conversation = config.get("conversation", {})
    if float(conversation.get("start_probability", 0.0)) != 0.0:
        raise ValueError("strict arrival replay requires conversation.start_probability=0")

    seed = int(config.get("seed", 0))
    rng = random.Random(seed)
    profiles = [TaskProfile.from_dict(value) for value in config.get("task_profiles", [])]
    if not profiles:
        raise ValueError("task_profiles must contain at least one profile")
    if any(profile.weight < 0 for profile in profiles) or sum(p.weight for p in profiles) <= 0:
        raise ValueError("task profile weights must have a positive sum")

    client_count = int(config.get("client_count", 1))
    skew = float(config.get("client_zipf_skew", 0.0))
    if client_count < 1 or skew < 0:
        raise ValueError("client_count must be positive and client_zipf_skew non-negative")
    clients = [f"client-{index:04d}" for index in range(client_count)]
    client_weights = [1.0 / ((index + 1) ** skew) for index in range(client_count)]
    max_input = int(config.get("max_input_tokens", 131072))
    max_output = int(config.get("max_output_tokens", 4096))
    profiles_by_arrival = _assign_profiles(config, rng, profiles, len(replay.arrival_times_ms))

    payload = config.get("payload", {})
    acceptance = config.get("acceptance", {})
    requests: list[WorkloadRequest] = []
    for index, (arrival_ms, source_timestamp_ms, profile) in enumerate(
        zip(replay.arrival_times_ms, replay.source_timestamps_ms, profiles_by_arrival)
    ):
        input_tokens, output_tokens = _sample_lengths(
            rng, profile, max_input, max_output
        )
        requests.append(
            WorkloadRequest(
                request_id=f"request-{index:07d}",
                arrival_time_ms=arrival_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                slo_tpot_ms=_sample_slo(config, rng, profile),
                task=profile.name,
                client_id=_weighted_choice(rng, clients, client_weights),
                conversation_id=None,
                turn_index=None,
                acceptance_probability=profile.acceptance_probability,
                metadata={
                    "source": "mooncake-arrival-replay",
                    "source_timestamp_ms": source_timestamp_ms,
                    "payload_status": payload.get("status", "unspecified"),
                    "acceptance_status": acceptance.get("status", "unspecified"),
                    "seed": seed,
                },
            )
        )

    if len(requests) != len(replay.source_timestamps_ms):
        raise AssertionError("strict replay must preserve the selected arrival count")
    return Workload(
        requests,
        metadata={
            "generator": "specrhythm",
            "seed": seed,
            "arrival_source": "mooncake-strict-replay",
            "workload_family": config.get("workload_family", "trace-replay"),
            "data_status": config.get("data_status", "unspecified"),
            "time_scale": replay.time_scale,
            "window_start_ms": replay.window_start_ms,
            "window_duration_ms": replay.window_duration_ms,
        },
    )


def generate_workload(
    config: dict[str, Any], arrival_times_ms: Optional[Sequence[float]] = None
) -> Workload:
    """Generate a deterministic workload from JSON-compatible configuration."""

    seed = int(config.get("seed", 0))
    rng = random.Random(seed)
    profiles = [TaskProfile.from_dict(value) for value in config.get("task_profiles", [])]
    if not profiles:
        raise ValueError("task_profiles must contain at least one profile")
    if any(profile.weight < 0 for profile in profiles) or sum(p.weight for p in profiles) <= 0:
        raise ValueError("task profile weights must have a positive sum")

    base_arrivals = (
        list(arrival_times_ms)
        if arrival_times_ms is not None
        else _piecewise_gamma_arrivals(config, rng)
    )
    base_arrivals.sort()
    end_ms = (
        max(base_arrivals, default=0.0)
        if arrival_times_ms is not None
        else sum(float(item["duration_s"]) for item in config["arrival_segments"]) * 1000.0
    )
    client_count = int(config.get("client_count", 1))
    skew = float(config.get("client_zipf_skew", 0.0))
    if client_count < 1 or skew < 0:
        raise ValueError("client_count must be positive and client_zipf_skew non-negative")
    clients = [f"client-{index:04d}" for index in range(client_count)]
    client_weights = [1.0 / ((index + 1) ** skew) for index in range(client_count)]
    max_input = int(config.get("max_input_tokens", 131072))
    max_output = int(config.get("max_output_tokens", 4096))
    conversation = config.get("conversation", {})

    provisional: list[dict[str, Any]] = []
    conversation_counter = 0
    for arrival_ms in base_arrivals:
        profile = _weighted_choice(rng, profiles, [item.weight for item in profiles])
        client_id = _weighted_choice(rng, clients, client_weights)
        slo = _sample_slo(config, rng, profile)
        is_conversation = rng.random() < float(conversation.get("start_probability", 0.0))
        conversation_id = None
        turns = 1
        if is_conversation:
            conversation_id = f"conversation-{conversation_counter:06d}"
            conversation_counter += 1
            continuation_probability = float(conversation.get("continuation_probability", 0.0))
            max_turns = int(conversation.get("max_turns", 1))
            while turns < max_turns and rng.random() < continuation_probability:
                turns += 1

        turn_arrival_ms = float(arrival_ms)
        for turn_index in range(turns):
            if turn_arrival_ms > end_ms:
                break
            input_tokens, output_tokens = _sample_lengths(
                rng, profile, max_input, max_output, turn_index
            )
            provisional.append(
                {
                    "arrival_time_ms": turn_arrival_ms,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "slo_tpot_ms": slo,
                    "task": profile.name,
                    "client_id": client_id,
                    "conversation_id": conversation_id,
                    "turn_index": turn_index if conversation_id else None,
                    "acceptance_probability": profile.acceptance_probability,
                    "metadata": {"source": "parametric-or-composed", "seed": seed},
                }
            )
            if turn_index + 1 < turns:
                median_s = float(conversation.get("itt_median_s", 30.0))
                sigma = float(conversation.get("itt_sigma", 1.0))
                turn_arrival_ms += median_s * math.exp(sigma * rng.gauss(0.0, 1.0)) * 1000.0

    provisional.sort(key=lambda value: value["arrival_time_ms"])
    requests = [
        WorkloadRequest(request_id=f"request-{index:07d}", **value)
        for index, value in enumerate(provisional)
    ]
    return Workload(
        requests,
        metadata={
            "generator": "specrhythm",
            "seed": seed,
            "arrival_source": "trace" if arrival_times_ms is not None else "piecewise-gamma",
        },
    )


def import_mooncake(
    path: Union[str, Path],
    time_scale: float = 1.0,
    slo_tpot_ms: float = 50.0,
    acceptance_probability: float = 0.7,
) -> Workload:
    """Normalize the public Mooncake JSONL schema without copying raw content."""

    if time_scale <= 0:
        raise ValueError("time_scale must be positive")
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return Workload([])
    origin = min(float(row["timestamp"]) for row in rows)
    requests = []
    for index, row in enumerate(sorted(rows, key=lambda value: float(value["timestamp"]))):
        metadata: dict[str, Any] = {"source": "mooncake-fast25"}
        if "hash_ids" in row:
            metadata["prefix_block_hash_ids"] = row["hash_ids"]
            metadata["prefix_block_tokens"] = 512
        requests.append(
            WorkloadRequest(
                request_id=f"request-{index:07d}",
                arrival_time_ms=(float(row["timestamp"]) - origin) / time_scale,
                input_tokens=int(row["input_length"]),
                output_tokens=int(row["output_length"]),
                slo_tpot_ms=slo_tpot_ms,
                acceptance_probability=acceptance_probability,
                metadata=metadata,
            )
        )
    return Workload(requests, metadata={"source": str(path), "time_scale": time_scale})


def summarize_workload(workload: Workload) -> dict[str, Any]:
    if not workload.requests:
        return {"requests": 0}
    arrivals = [request.arrival_time_ms for request in workload.requests]
    inputs = [request.input_tokens for request in workload.requests]
    outputs = [request.output_tokens for request in workload.requests]
    duration_s = max(arrivals) / 1000.0

    def percentile(values: list[float], fraction: float) -> float:
        ordered = sorted(values)
        return ordered[round((len(ordered) - 1) * fraction)]

    inter_arrivals = [right - left for left, right in zip(arrivals, arrivals[1:])]
    iat_cv = None
    if len(inter_arrivals) > 1 and statistics.mean(inter_arrivals) > 0:
        iat_cv = statistics.pstdev(inter_arrivals) / statistics.mean(inter_arrivals)
    return {
        "requests": len(workload.requests),
        "duration_s": duration_s,
        "mean_rate_per_s": len(workload.requests) / duration_s if duration_s else None,
        "iat_cv": iat_cv,
        "input_tokens": {"p50": percentile(inputs, 0.5), "p90": percentile(inputs, 0.9)},
        "output_tokens": {"p50": percentile(outputs, 0.5), "p90": percentile(outputs, 0.9)},
        "tasks": {
            task: sum(request.task == task for request in workload.requests)
            for task in sorted({request.task for request in workload.requests})
        },
        "slo_tpot_ms": {
            str(slo): sum(request.slo_tpot_ms == slo for request in workload.requests)
            for slo in sorted({request.slo_tpot_ms for request in workload.requests})
        },
    }
