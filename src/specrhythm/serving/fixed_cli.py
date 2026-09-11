"""Foreground fixed64/32 diagnostic. CPU summary/audit/bundle are independent commands."""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import time
import traceback
from pathlib import Path

from specrhythm.serving.common import DataError, read_json, require
from specrhythm.serving.fixed_plan import MODES, point, prepare, settings, stage_points
from specrhythm.serving.s1_console import print_failure
from specrhythm.serving.s1_workload import write_once


def run_point(root, selected, *, probe=False):
    from specrhythm.serving.s2_cli import execute
    from specrhythm.serving.s2_pool import publish

    manifest_path = root / "inputs/execution-manifest.json"
    label = ("capacity" if probe else selected["kind"]) + "-" + selected["mode"]
    if selected["batch"]:
        label += f"-B{selected['batch']}-{selected['half']}"
    stamp = time.strftime("%Y%m%dT%H%M%S") + f"-{time.monotonic_ns()}"
    directory = root / "runs" / (label + "-" + stamp)
    directory.mkdir(parents=True)
    selected = {**selected, "probe": probe}
    write_once(directory / "point.json", selected)
    publish(
        root / "stage.json", {"state": "RUNNING", "directory": str(directory), "point": selected}
    )
    try:
        value = execute(
            root,
            "fixed64-diagnostic",
            selected["runtime_mode"],
            manifest_path,
            directory,
            probe=probe,
            diagnostic=selected,
        )
    except BaseException as error:
        publish(
            root / "stage.json",
            {
                "state": "FAILED",
                "directory": str(directory),
                "point": selected,
                "error": str(error),
            },
        )
        raise
    publish(
        root / "stage.json",
        {
            "state": "COMPLETE",
            "directory": str(directory),
            "point": selected,
            "measurement": value["measurement_status"],
            "full_offline_audit": "PENDING",
        },
    )
    return directory, value


def capacity_passed(root):
    values = [read_json(p) for p in root.glob("runs/*/result.json")]
    require(
        any(r.get("probe") and r["valid"] for r in values),
        "run this diagnostic's resident100/active64 capacity probe first",
    )


def status(root):
    from specrhythm.serving.fixed_artifacts import point_reports

    return {
        "stage": read_json(root / "stage.json") if (root / "stage.json").exists() else None,
        "results": [
            {
                "directory": r["artifact"],
                **{
                    k: r.get(k)
                    for k in (
                        "mode",
                        "execution_status",
                        "measurement_status",
                        "effective_exit_code",
                        "primary_error",
                        "measurement_availability",
                        "cleanup_status",
                    )
                },
                "full_offline_audit": read_json(Path(r["artifact"]) / "offline-audit.json")[
                    "status"
                ]
                if (Path(r["artifact"]) / "offline-audit.json").exists()
                else "PENDING",
            }
            for r in point_reports(root)
        ],
    }


def stop(root, wait_seconds=60):
    from specrhythm.serving.s1_cli import cleanup_attempt

    stage = read_json(root / "stage.json")
    directory = Path(stage["directory"])
    require(directory.resolve().parent == (root / "runs").resolve(), "unsafe active point path")
    if stage["state"] != "RUNNING":
        return {"state": stage["state"], "action": "already stopped"}
    request = directory / "stop-request.json"
    if not request.exists():
        write_once(request, {"requested_ns": time.monotonic_ns(), "reason": "operator_stop"})
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if (directory / "result.json").exists():
            return {"state": "STOPPED", "directory": str(directory)}
        time.sleep(0.1)
    cleanup = cleanup_attempt(directory)
    return {
        "state": "FORCED_OWNED_CLEANUP",
        "directory": str(directory),
        "cleanup": cleanup,
        "interpretation": "execution interrupted; no performance PASS implied",
    }


def bundle(root, output):
    names = (
        "config.json",
        "patch-manifest.json",
        "environment.json",
        "topology.json",
        "diagnostic-config.json",
        "stage.json",
    )
    files = [root / name for name in names if (root / name).is_file()]
    allowed = (
        "point.json",
        "light-summary.json",
        "light-summary.csv",
        "exit-code.json",
        "child-failure.json",
        "draft-child-failure.json",
        "launcher-failure.json",
        "target-cleanup-secondary.json",
        "measurement-snapshot.json",
        "drain-state.json",
        "draft-drain-state.json",
        "diagnostic-primary-error.json",
        "diagnostic-secondary-errors.json",
        "process-lifecycle.json",
    )
    files += [p for name in allowed for p in root.glob("runs/*/" + name)]
    files += list(root.glob("runs/*/fixed-logging-*.json"))
    files += list(root.glob("comparison-*.json"))
    require(not output.exists(), "small bundle output already exists")
    with tarfile.open(output, "x:gz") as tar:
        for path in sorted(files):
            require(not path.is_symlink(), "bundle refuses symlink", artifact=str(path))
            tar.add(path, arcname=str(path.relative_to(root)), recursive=False)
    return {
        "bundle": str(output),
        "files": len(files),
        "bytes": output.stat().st_size,
        "raw_token_request_logs_included": False,
    }


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "command",
        choices=(
            "prepare",
            "capacity",
            "short",
            "stages",
            "status",
            "errors",
            "stop",
            "summary",
            "audit",
            "bundle",
            "child",
            "draft-child",
        ),
    )
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--s1", type=Path)
    p.add_argument("--mode", choices=MODES)
    from specrhythm.serving.fixed_logging import MODES as OBSERVATIONS

    p.add_argument("--observation", choices=OBSERVATIONS, default="original-live")
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--samples", type=int, default=12)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--window-seconds", type=float, default=30)
    p.add_argument("--setup-timeout", type=float, default=900)
    p.add_argument("--drain-timeout", type=float, default=60)
    p.add_argument("--stage-samples", type=int, default=1)
    p.add_argument("--stage-warmups", type=int, default=0)
    p.add_argument("--wait-seconds", type=float, default=60)
    p.add_argument("--directory", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--probe", action="store_true")
    p.add_argument("--output", type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    root = args.root.resolve()
    try:
        if args.command in ("child", "draft-child"):
            try:
                from specrhythm.serving.fixed_observe import install_host_observation

                selected = read_json(Path(os.environ["SR_FIXED_POINT"]))
                from specrhythm.serving.fixed_logging import finish_current

                role = "draft" if args.command == "draft-child" else "coordinator"
                install_host_observation(role)
                if args.command == "draft-child":
                    from specrhythm.phase4.config import load_phase4_config
                    from specrhythm.serving.fixed_draft import FixedDraftBackend, serve

                    serve(
                        load_phase4_config(str(root / "config.json")),
                        args.directory,
                        Path(os.environ["SR_S2_DRAFT_SOCKET"]),
                        selected["runtime_mode"],
                        backend_class=FixedDraftBackend,
                    )
                else:
                    from specrhythm.serving.fixed_runtime import run

                    run(root, args.manifest, args.directory, selected, probe=args.probe)
                finish_current(role)
                return 0
            except BaseException as error:
                from specrhythm.serving.fixed_artifacts import record_error
                from specrhythm.serving.fixed_logging import abort_current

                record_error(args.directory, error, args.command)
                abort_current(error)
                write_once(
                    args.directory
                    / (
                        "draft-child-failure.json"
                        if args.command == "draft-child"
                        else "child-failure.json"
                    ),
                    {
                        "error": str(error),
                        "type": type(error).__name__,
                        "details": getattr(error, "details", {}),
                        "traceback": traceback.format_exc(),
                        "timestamp_ns": time.monotonic_ns(),
                    },
                )
                raise
        if args.command == "prepare":
            require(args.s1 is not None, "prepare requires the existing mixed100 --s1 root")
            options = settings(
                warmup_steps=args.warmup_steps,
                samples=args.samples,
                repeats=args.repeats,
                window_seconds=args.window_seconds,
                setup_timeout=args.setup_timeout,
                drain_timeout=args.drain_timeout,
                observation=args.observation,
            )
            prepare(root, args.s1.resolve(), options)
            value = {"prepared": str(root), "capacity": "PENDING", "GPU_timing": "PENDING"}
        elif args.command == "status":
            value = status(root)
        elif args.command == "errors":
            from specrhythm.serving.fixed_artifacts import retained_display

            selected = args.directory or Path(read_json(root / "stage.json")["directory"])
            print(json.dumps(retained_display(selected), ensure_ascii=False), flush=True)
            print_failure(DataError("retained point errors"), directory=selected, label="fixed")
            return 0
        elif args.command == "stop":
            require(0 <= args.wait_seconds <= 300, "stop wait must be within 0..300 seconds")
            value = stop(root, args.wait_seconds)
        elif args.command in ("summary", "audit", "bundle"):
            from specrhythm.serving.fixed_results import comparisons, offline_audit

            if args.command == "bundle":
                destination = args.output or root / (
                    "small-" + str(time.monotonic_ns()) + ".tar.gz"
                )
                value = bundle(root, destination)
            elif args.command == "audit":
                directories = (
                    [args.directory]
                    if args.directory
                    else [
                        p.parent
                        for p in sorted(root.glob("runs/*/result.json"))
                        if read_json(p)["valid"] and not read_json(p).get("probe")
                    ]
                )
                for directory in directories:
                    require(
                        not (directory / "offline-audit.json").exists(),
                        "offline audit already retained",
                        artifact=str(directory),
                    )
                    audited = offline_audit(root / "inputs/execution-manifest.json", directory)
                    write_once(directory / "offline-audit.json", audited)
                value = {"audited_points": len(directories), "GPU_used": False}
            else:
                value = comparisons(root)
                output = args.output or root / ("comparison-" + str(time.monotonic_ns()) + ".json")
                write_once(output, value)
                value = {"artifact": str(output), **value}
        else:
            from specrhythm.serving.s2_cli import root_lock

            with root_lock(root):
                if args.command == "capacity":
                    directory, value = run_point(root, point("pingpong"), probe=True)
                    value = {"capacity": "PASS", "artifact": str(directory)}
                else:
                    capacity_passed(root)
                    if args.command == "stages":
                        require(
                            args.stage_samples > 0 and args.stage_warmups >= 0,
                            "invalid stage repeat/warmup count",
                        )
                        selected = stage_points(args.stage_samples, args.stage_warmups)
                    else:
                        options = read_json(root / "diagnostic-config.json")["options"]
                        selected = [
                            point(mode, repeat=i)
                            for i in range(options["repeats"])
                            for mode in ([args.mode] if args.mode else MODES)
                        ]
                    done = []
                    for unit in selected:
                        directory, _ = run_point(root, unit)
                        done.append(str(directory))
                        if (directory / "stop-request.json").exists():
                            break  # Operator stop cancels the remaining suite too.
                    value = {"points_complete": done, "full_offline_audit": "PENDING"}
        print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print_failure(
            error,
            directory=getattr(error, "details", {}).get("directory") or args.directory,
            mode=args.mode or args.command,
            label="fixed",
        )
        if args.command in ("child", "draft-child"):
            traceback.print_exc()
        return (
            getattr(error, "details", {}).get(
                "returncode", 130 if isinstance(error, KeyboardInterrupt) else 1
            )
            or 1
        )


if __name__ == "__main__":
    raise SystemExit(main())
