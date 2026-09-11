"""Compact atomic diagnostic checkpoints; never a substitute for qualification."""

from __future__ import annotations

import time
import traceback
from collections import Counter

from specrhythm.serving.common import read_json
from specrhythm.serving.s2_pool import publish


def record_error(directory, error, phase):
    """First failure wins, including when final engine cleanup also fails."""
    primary = directory / "diagnostic-primary-error.json"
    row = {
        "error": str(error),
        "phase": phase,
        "timestamp_ns": time.monotonic_ns(),
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
    }
    try:
        import fcntl

        # Worker and coordinator can fail concurrently (for example at timeout).
        # Serialize only error publication; never overwrite a completed first error.
        with (directory / ".diagnostic-error.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if not primary.exists():
                publish(primary, row)
            else:
                path = directory / "diagnostic-secondary-errors.json"
                rows = read_json(path) if path.exists() else []
                # drive and its caller can observe the same primary exception.
                if read_json(primary)["error"] != row["error"]:
                    publish(path, [*rows, row])
    except Exception as save_error:
        print(f"[diagnostic secondary] cannot retain {phase} error: {save_error}", flush=True)


def checkpoint(
    directory,
    manifest,
    point,
    window,
    clock,
    steps,
    phases,
    *,
    measurement_complete=False,
    drain_complete=False,
):
    from specrhythm.serving.fixed_results import stats

    end = window.end_ns or time.monotonic_ns()
    start = window.start_ns
    selected = [s for s in steps if s["window"] and s["B"]]
    tokens = sum(
        len(c["token_ids"])
        for r in clock.rows.values()
        for c in r["commits"]
        if start is not None and start <= c["timestamp_ns"] <= end
    )
    value = {
        "schema_version": "specrhythm.fixed-measurement-snapshot.v1",
        "mode": point["mode"],
        "point": point,
        "git_commit": manifest.get("git_commit", manifest.get("execution", {}).get("git_commit")),
        "execution_sha256": manifest["sha256"],
        "execution_identity": manifest.get("execution", {}),
        "stop_reason": window.reason,
        "measurement_start_ns": start,
        "measurement_end_ns": end,
        "sample_count": len(selected),
        "completed_steps": len(steps),
        "committed_window_tokens": tokens,
        "window_ms": (end - start) / 1e6 if start is not None else None,
        "step_wall_ms": stats([(s["end_ns"] - s["start_ns"]) / 1e6 for s in selected]),
        "actual_target_batch_histogram": dict(Counter(s["B"] for s in selected)),
        "peak_held_slots": max((p["held_slots"] for p in phases), default=0),
        "peak_cohort_held": {
            c: max((p["cohort_held"][c] for p in phases), default=0) for c in ("A", "B")
        },
        "measurement_complete": measurement_complete,
        "measurement_availability": "UNQUALIFIED_PARTIAL" if selected else "INSUFFICIENT",
        "drain_complete": drain_complete,
        "execution_status": "PENDING",
        "cleanup_status": "PENDING",
        "pending_checks": [
            "final runtime",
            "Target TP/device/token evidence",
            "Draft settlement/shutdown",
            "owned process cleanup",
        ],
        "formal_comparison_eligible": False,
    }
    publish(directory / "measurement-snapshot.json", value)
    return value


def retained_report(directory, report):
    """Works after a killed coordinator too; missing final reports stay secondary."""
    result = dict(report)
    for filename, field in (
        ("measurement-snapshot.json", "measurement_snapshot"),
        ("drain-state.json", "drain"),
        ("diagnostic-secondary-errors.json", "cleanup_diagnostics"),
    ):
        path = directory / filename
        if path.exists():
            result[field] = read_json(path)
    primary = directory / "diagnostic-primary-error.json"
    if primary.exists():
        error = read_json(primary)
        result.update(
            valid=False,
            execution_status="FAILED",
            measurement_status="INVALID",
            errors=[error["error"]],
            primary_error={
                **error,
                "artifact": str(primary),
                "field": error["phase"],
                "actual": error["error"],
                "expected": "success",
            },
        )
    if not result.get("valid"):
        result.update(execution_status="FAILED", measurement_status="INVALID")
    lifecycle = directory / "process-lifecycle.json"
    life = read_json(lifecycle) if lifecycle.exists() else {}
    if life.get("failure_detection") and not result.get("primary_error"):
        detection = life["failure_detection"]
        result["primary_error"] = {
            "error": detection["reason"],
            "actual": detection,
            "expected": "successful owned execution within the drain deadline",
            "field": "owned supervisor",
            "artifact": str(lifecycle),
        }
        result["errors"] = [detection["reason"]]
    if result.get("drain") and life.get("run_valid") is False:
        result["drain"] = {
            **result["drain"],
            "execution_failed": True,
            "supervisor_failure": life.get("failure_detection"),
        }
    result["cleanup_status"] = (
        ("PASS" if life.get("cleanup_valid") else "FAILED") if lifecycle.exists() else "UNKNOWN"
    )
    result["measurement_availability"] = (
        "QUALIFIED"
        if result.get("valid") and result.get("measurement_status") == "PASS"
        else result.get("measurement_snapshot", {}).get("measurement_availability", "UNAVAILABLE")
    )
    if not result.get("valid"):
        result["secondary_diagnostics"] = [
            *result.get("secondary_diagnostics", []),
            *result.get("cleanup_diagnostics", []),
            *[
                {"artifact": str(directory / name), "status": "missing after execution failure"}
                for name in ("runtime.json", "draft-backend-report.json")
                if not (directory / name).exists()
            ],
        ]
    return result


def retained_display(directory):
    value = retained_report(directory, {"valid": False})
    snapshot = value.get("measurement_snapshot", {})
    drain = value.get("drain", {})
    return {
        "artifact": str(directory),
        "primary_error": value.get("primary_error"),
        "measurement_availability": value["measurement_availability"],
        "snapshot": {
            k: snapshot.get(k)
            for k in (
                "sample_count",
                "committed_window_tokens",
                "measurement_complete",
                "drain_complete",
            )
        },
        "drain": {k: drain.get(k) for k in ("status", "phase", "settled_requests", "error")},
        "secondary_diagnostics": value.get("secondary_diagnostics", []),
    }


def point_reports(root):
    reports = []
    for directory in sorted((root / "runs").glob("*")):
        if not directory.is_dir():
            continue
        final = directory / "result.json"
        light = directory / "light-summary.json"
        if final.exists() or light.exists():
            value = read_json(final if final.exists() else light)
        elif (directory / "point.json").exists():
            p = read_json(directory / "point.json")
            exits = directory / "exit-code.json"
            value = {
                "point": p,
                "mode": p["mode"],
                "valid": False,
                "errors": ["final execution report unavailable"],
                "execution_status": "FAILED",
                "measurement_status": "INVALID",
                "effective_exit_code": read_json(exits).get("effective_exit_code")
                if exits.exists()
                else None,
            }
        else:
            continue
        reports.append(retained_report(directory, {**value, "artifact": str(directory)}))
    return reports
