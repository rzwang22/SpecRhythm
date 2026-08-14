"""Raw-order workload validation and replay correspondence checks."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Union

from specrhythm.workload import apportion_counts, select_arrival_replay


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _slo_label(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


def validate_workload(
    workload_path: Union[str, Path],
    *,
    config: Optional[dict[str, Any]] = None,
    arrival_trace_path: Optional[Union[str, Path]] = None,
    window_start_ms: float = 0.0,
    window_duration_ms: Optional[float] = None,
    time_scale: float = 1.0,
) -> dict[str, Any]:
    """Validate JSONL without normalizing away the original file order."""

    source = Path(workload_path)
    errors: list[dict[str, Any]] = []
    rows: list[tuple[int, dict[str, Any]]] = []

    def add_error(code: str, message: str, line: Optional[int] = None) -> None:
        error: dict[str, Any] = {"code": code, "message": message}
        if line is not None:
            error["line"] = line
        errors.append(error)

    workload_readable = False
    try:
        handle = source.open(encoding="utf-8")
    except OSError as error:
        add_error("workload_unreadable", str(error))
    else:
        workload_readable = True
        with handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    add_error("invalid_json", f"invalid JSON: {error.msg}", line_number)
                    continue
                if not isinstance(row, dict):
                    add_error("invalid_record", "JSONL row must be an object", line_number)
                    continue
                rows.append((line_number, row))

    request_ids: dict[str, int] = {}
    arrivals: list[tuple[int, float]] = []
    tasks: Counter[str] = Counter()
    slos: Counter[str] = Counter()
    r3_family = str((config or {}).get("workload_family", "")).startswith("r3-")
    expected_profiles: dict[str, tuple[float, float]] = {}
    if r3_family:
        try:
            for profile in (config or {})["task_profiles"]:
                name = str(profile["name"])
                if not name.strip():
                    raise ValueError("task profile names must not be empty")
                if name in expected_profiles:
                    raise ValueError(f"duplicate task profile: {name}")
                expected_profiles[name] = (
                    float(profile["weight"]),
                    float(profile["slo_tpot_ms"]),
                )
            if not expected_profiles:
                raise ValueError("task_profiles must not be empty")
            if any(
                not math.isfinite(weight) or weight < 0
                for weight, _ in expected_profiles.values()
            ) or sum(weight for weight, _ in expected_profiles.values()) <= 0:
                raise ValueError("task weights must be finite with a positive sum")
            if any(
                not math.isfinite(slo) or slo <= 0
                for _, slo in expected_profiles.values()
            ):
                raise ValueError("task SLO values must be finite and positive")
        except (KeyError, TypeError, ValueError) as error:
            add_error("invalid_r3_config", f"cannot derive R3 mixture: {error}")

    if workload_readable and not rows:
        add_error("empty_workload", "workload must contain at least one JSON object")

    for line_number, row in rows:
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or not request_id.strip():
            add_error("invalid_request_id", "request_id must be a non-empty string", line_number)
        elif request_id in request_ids:
            add_error(
                "duplicate_request_id",
                f"request_id duplicates line {request_ids[request_id]}",
                line_number,
            )
        else:
            request_ids[request_id] = line_number

        arrival = row.get("arrival_time_ms")
        if not _is_number(arrival) or not math.isfinite(float(arrival)) or float(arrival) < 0:
            add_error(
                "invalid_arrival_time",
                "arrival_time_ms must be finite and non-negative",
                line_number,
            )
        else:
            arrivals.append((line_number, float(arrival)))

        for field in ("input_tokens", "output_tokens"):
            value = row.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                add_error(
                    f"invalid_{field}", f"{field} must be a positive integer", line_number
                )

        slo = row.get("slo_tpot_ms")
        if not _is_number(slo) or not math.isfinite(float(slo)) or float(slo) <= 0:
            add_error("invalid_slo", "slo_tpot_ms must be finite and positive", line_number)
        else:
            slos[_slo_label(float(slo))] += 1

        acceptance = row.get("acceptance_probability")
        if (
            not _is_number(acceptance)
            or not math.isfinite(float(acceptance))
            or not 0 <= float(acceptance) <= 1
        ):
            add_error(
                "invalid_acceptance_probability",
                "acceptance_probability must be finite and in [0, 1]",
                line_number,
            )

        confidence = row.get("draft_confidence", 0.7)
        if (
            not _is_number(confidence)
            or not math.isfinite(float(confidence))
            or not 0 <= float(confidence) <= 1
        ):
            add_error(
                "invalid_draft_confidence",
                "draft_confidence must be finite and in [0, 1]",
                line_number,
            )

        task_value = row.get("task")
        task = str(task_value) if task_value is not None else "unknown"
        tasks[task] += 1

        if r3_family and expected_profiles:
            if task not in expected_profiles:
                add_error(
                    "unexpected_r3_task",
                    f"task {task!r} is not present in the R3 configuration",
                    line_number,
                )
            elif _is_number(slo) and math.isfinite(float(slo)):
                expected_slo = expected_profiles[task][1]
                if not math.isclose(float(slo), expected_slo, rel_tol=0.0, abs_tol=1e-9):
                    add_error(
                        "r3_task_slo_mismatch",
                        f"task {task!r} requires slo_tpot_ms={expected_slo}",
                        line_number,
                    )

        if r3_family and (
            row.get("conversation_id") is not None or row.get("turn_index") is not None
        ):
            add_error(
                "r3_conversation_turn",
                "R3 replay must not contain generated conversation turns",
                line_number,
            )

    if r3_family and expected_profiles and rows:
        names = list(expected_profiles)
        weights = [expected_profiles[name][0] for name in names]
        try:
            counts = apportion_counts(weights, len(rows))
        except ValueError as error:
            add_error("invalid_r3_config", f"cannot apportion R3 mixture: {error}")
        else:
            expected_task_counts = dict(zip(names, counts))
            actual_task_counts = {name: tasks.get(name, 0) for name in names}
            unexpected_tasks = set(tasks) - set(names)
            if actual_task_counts != expected_task_counts or unexpected_tasks:
                add_error(
                    "r3_task_mixture_mismatch",
                    (
                        f"task counts are {dict(sorted(tasks.items()))}, expected "
                        f"{dict(sorted(expected_task_counts.items()))}"
                    ),
                )
            expected_slo_counts: Counter[str] = Counter()
            for name, count in expected_task_counts.items():
                expected_slo_counts[_slo_label(expected_profiles[name][1])] += count
            if slos != expected_slo_counts:
                add_error(
                    "r3_slo_mixture_mismatch",
                    (
                        f"SLO counts are {dict(sorted(slos.items()))}, expected "
                        f"{dict(sorted(expected_slo_counts.items()))}"
                    ),
                )

    for (left_line, left), (right_line, right) in zip(arrivals, arrivals[1:]):
        if right < left:
            add_error(
                "non_monotonic_arrival",
                f"arrival_time_ms is less than line {left_line}",
                right_line,
            )

    if arrival_trace_path is not None:
        try:
            replay = select_arrival_replay(
                arrival_trace_path,
                window_start_ms=window_start_ms,
                window_duration_ms=window_duration_ms,
                time_scale=time_scale,
            )
        except (OSError, ValueError) as error:
            add_error("arrival_trace_invalid", str(error))
        else:
            observed = [arrival for _, arrival in arrivals]
            expected = list(replay.arrival_times_ms)
            if len(rows) != len(expected):
                add_error(
                    "arrival_count_mismatch",
                    (
                        f"workload has {len(rows)} records but replay selects "
                        f"{len(expected)} timestamps"
                    ),
                )
            if len(observed) == len(expected):
                for index, (actual, wanted) in enumerate(zip(observed, expected)):
                    if not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1e-9):
                        add_error(
                            "arrival_timestamp_mismatch",
                            f"arrival index {index} is {actual}, expected {wanted}",
                            arrivals[index][0],
                        )
                        break

    arrival_values = [arrival for _, arrival in arrivals]
    duration_ms = max(arrival_values) - min(arrival_values) if arrival_values else None
    duration_s = duration_ms / 1000.0 if duration_ms is not None else None
    inter_arrivals = [
        right - left for left, right in zip(arrival_values, arrival_values[1:])
    ]
    iat_cv = None
    if len(inter_arrivals) > 1 and statistics.mean(inter_arrivals) > 0:
        iat_cv = statistics.pstdev(inter_arrivals) / statistics.mean(inter_arrivals)
    request_count = len(rows)
    observed_arrival_count = len(arrival_values)
    observed_iat_rate = (
        (observed_arrival_count - 1) / duration_s
        if observed_arrival_count > 1 and duration_s is not None and duration_s > 0
        else None
    )
    window_offered_rate = None
    if (
        window_duration_ms is not None
        and math.isfinite(window_duration_ms)
        and window_duration_ms > 0
        and math.isfinite(time_scale)
        and time_scale > 0
    ):
        scaled_window_s = window_duration_ms / time_scale / 1000.0
        window_offered_rate = request_count / scaled_window_s
    summary = {
        "request_count": request_count,
        "time_range_ms": (
            {"start": min(arrival_values), "end": max(arrival_values)}
            if arrival_values
            else None
        ),
        "duration_s": duration_s,
        "observed_iat_rate_per_s": observed_iat_rate,
        "window_offered_rate_per_s": window_offered_rate,
        "iat_cv": iat_cv,
        "task_counts": dict(sorted(tasks.items())),
        "task_proportions": {
            task: count / request_count for task, count in sorted(tasks.items())
        }
        if request_count
        else {},
        "slo_tpot_ms_counts": dict(sorted(slos.items(), key=lambda item: float(item[0]))),
        "slo_tpot_ms_proportions": {
            slo: count / request_count
            for slo, count in sorted(slos.items(), key=lambda item: float(item[0]))
        }
        if request_count
        else {},
    }
    return {
        "schema_version": "specrhythm.validation.v1",
        "valid": not errors,
        "errors": errors,
        "summary": summary,
    }
