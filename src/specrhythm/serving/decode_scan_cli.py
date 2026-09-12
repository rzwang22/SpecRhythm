"""Foreground resident360 fixed-time decode scan; no implicit GPU in inspection commands."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from specrhythm.serving.common import DataError, require
from specrhythm.serving.decode_scan_plan import (
    BATCHES,
    MODES,
    load,
    options,
    point_manifest,
    prepare,
)
from specrhythm.serving.fixed_artifacts import point_reports
from specrhythm.serving.fixed_cli import run_point, status, stop
from specrhythm.serving.s1_console import print_failure
from specrhythm.serving.s1_workload import write_once
from specrhythm.serving.s2_cli import root_lock

COLUMNS = (
    "mode",
    "batch",
    "sub_batch",
    "repeat",
    "git_commit",
    "workload_sha256",
    "pool_size",
    "execution_status",
    "measurement_status",
    "cleanup_status",
    "capacity_status",
    "stop_reason",
    "formal_comparison_eligible",
    "measured_window_ms",
    "window_overrun_ms",
    "committed_window_tokens",
    "decode_throughput_tok_s",
    "target_steps",
    "complete_rotations",
    "partial_rotations",
    "full_active_time_fraction",
    "natural_completed_requests",
    "refill_requests",
    "diagnostic_cancelled_requests",
    "effective_exit_code",
    "artifact",
)


def key(point):
    return point["mode"], point["batch"], point["repeat"]


def is_probe(row):
    return row.get("probe", row.get("point", {}).get("probe", False))


def successful(root, config):
    good = {}
    for row in point_reports(root):
        if is_probe(row):
            require(
                row.get("valid") and row.get("cleanup_status") == "PASS",
                "scan capacity probe failed; do not auto-continue",
                artifact=row["artifact"],
            )
            continue
        require(
            row.get("formal_comparison_eligible")
            and row.get("valid")
            and row.get("cleanup_status") == "PASS",
            "scan contains failed/stopped/insufficient point; do not auto-continue or rerun",
            artifact=row["artifact"],
            actual=row.get("measurement_status"),
        )
        require(
            row["git_commit"] == config["execution"]["git_commit"]
            and row["workload_sha256"] == config["workload_sha256"],
            "completed scan identity differs",
        )
        require(key(row["point"]) not in good, "duplicate successful scan point")
        good[key(row["point"])] = row
    return good


def run(root, *, batch=None, mode=None, remaining=False, probe=False, single_point=False):
    config = load(root)
    require(not single_point or (mode is not None and batch in BATCHES and not remaining),
            "single-point diagnostic requires explicit mode/batch and no --remaining")
    with root_lock(root):
        good = successful(root, config)
        probed = {
            key(r["point"])
            for r in point_reports(root)
            if is_probe(r) and r.get("valid") and r.get("cleanup_status") == "PASS"
        }
        if not probe and not single_point and (remaining or batch != BATCHES[0]):
            require(
                all(
                    (m, BATCHES[0], r) in good
                    for m in MODES
                    for r in range(config["options"]["repeats"])
                ),
                "first finish all three B16 modes before the remaining scan",
            )
        points = [
            p
            for p in config["points"]
            if (remaining or p["batch"] == batch) and (mode is None or p["mode"] == mode)
        ]
        require(not single_point or len(points) == 1,
                "single-point diagnostic requires exactly one point (repeats=1)")
        require(points, "no scan points selected")
        results = []
        for p in points:
            if key(p) in (probed if probe else good):
                print(
                    f"[scan] keep successful {p['mode']} B={p['batch']} repeat={p['repeat']}",
                    flush=True,
                )
                continue
            print(
                f"[scan] {'capacity/prefill' if probe else 'capacity → prefill → decode'} "
                f"{p['mode']} B={p['batch']} repeat={p['repeat']}",
                flush=True,
            )
            directory, value = run_point(
                root, {**p, **({"test_order": "independent-single-point"}
                              if single_point else {})},
                probe=probe, manifest_path=point_manifest(root, p["batch"])
            )
            require(
                value.get("valid")
                and value.get("cleanup_status") == "PASS"
                and (probe or value.get("formal_comparison_eligible")),
                "scan point not qualified; subsequent GPU points stopped",
                artifact=str(directory),
                actual=value.get("measurement_status"),
                returncode=3,
            )
            results.append(str(directory))
        return {"completed": results, "GPU_performance": "operator execution only"}


def summary(root):
    config = load(root)
    reports = point_reports(root)
    indexed = {key(r["point"]): r for r in reports if is_probe(r) and not r.get("valid")}
    indexed.update({key(r["point"]): r for r in reports if not is_probe(r)})
    rows = []
    for p in config["points"]:
        r = indexed.get(key(p), {})
        row = {k: r.get(k) for k in COLUMNS}
        row.update(
            mode=p["mode"],
            batch=p["batch"],
            repeat=p["repeat"],
            sub_batch=p["batch"] // 2 if p["mode"] == "pingpong" else None,
            pool_size=config["pool_size"],
            workload_sha256=config["workload_sha256"],
            git_commit=config["execution"]["git_commit"],
        )
        if not r:
            row.update(
                execution_status="NOT_RUN",
                measurement_status="NOT_RUN",
                cleanup_status="NOT_RUN",
                formal_comparison_eligible=False,
            )
        if not r.get("valid"):
            row["formal_comparison_eligible"] = False
        if r.get("measurement_snapshot") and not r.get("valid"):
            snapshot = r["measurement_snapshot"]
            row["partial_measurement_snapshot"] = {
                k: snapshot.get(k)
                for k in ("window_ms", "sample_count", "committed_window_tokens", "stop_reason")
            }
        row["actual_target_batch"] = r.get("actual_target_batch")
        row["prepared_pool"] = r.get("prepared_pool")
        row["primary_error"] = r.get("primary_error")
        rows.append(row)
    result = {
        "schema_version": "specrhythm.decode-scan-summary.v1",
        "points": rows,
        "valid_comparison_points": sum(r["formal_comparison_eligible"] for r in rows),
        "excluded_points": [key(r) for r in rows if not r["formal_comparison_eligible"]],
        "metric": "committed_window_tokens / actual measured_window_seconds",
        "cross_run_token_equality": "NOT_REQUIRED",
        "full_offline_audit": "NOT_RUN",
    }
    stamp = str(time.monotonic_ns())
    path = root / ("scan-summary-" + stamp + ".json")
    write_once(path, result)
    with path.with_suffix(".csv").open("x", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[*COLUMNS, "actual_batch_min", "actual_batch_mean", "actual_batch_max"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    **{k: r.get(k) for k in COLUMNS},
                    **{
                        "actual_batch_" + k: (r.get("actual_target_batch") or {}).get(k)
                        for k in ("min", "mean", "max")
                    },
                }
            )
    return {"artifact": str(path), **result}


def bundle(root, output):
    """Project large drain receipts to counts/hash; never copy timelines or request prefixes."""
    import hashlib
    import io
    import tarfile

    from specrhythm.serving.decode_scan_results import compact

    names = (
        "point.json",
        "light-summary.json",
        "measurement-snapshot.json",
        "drain-state.json",
        "draft-drain-state.json",
        "exit-code.json",
        "process-lifecycle.json",
        "diagnostic-primary-error.json",
        "diagnostic-secondary-errors.json",
        "child-failure.json",
        "draft-child-failure.json",
        "launcher-failure.json",
        "actual-capacity.json",
    )
    files = [
        p
        for p in (
            root / n
            for n in (
                "scan-config.json",
                "config.json",
                "patch-manifest.json",
                "environment.json",
                "topology.json",
                "stage.json",
            )
        )
        if p.exists()
    ]
    files += [p for name in names for p in root.glob("runs/*/" + name)]
    files += list(root.glob("runs/*/fixed-logging-*.json"))
    files += list(root.glob("scan-summary-*.json"))
    files += list(root.glob("scan-summary-*.csv"))
    files += list(root.glob("inputs/execution-B*.json"))
    require(not output.exists(), "scan bundle already exists")
    total, hashes = 0, []
    with tarfile.open(output, "x:gz") as tar:
        for path in sorted(files):
            require(not path.is_symlink(), "scan bundle refuses symlink")
            raw = path.read_bytes()
            data = (
                json.dumps(compact(json.loads(raw)), ensure_ascii=False).encode()
                if path.suffix == ".json"
                else raw
            )
            total += len(data)
            require(total <= 10 * 1024**2 - 128 * 1024, "scan small bundle exceeds 10 MiB")
            name = str(path.relative_to(root))
            hashes.append(
                dict(
                    path=name,
                    source_sha256=hashlib.sha256(raw).hexdigest(),
                    exported_sha256=hashlib.sha256(data).hexdigest(),
                )
            )
            item = tarfile.TarInfo(name)
            item.size = len(data)
            tar.addfile(item, io.BytesIO(data))
        data = json.dumps(
            {
                "files": hashes,
                "raw_timelines_included": False,
                "drain_receipts": "count/hash projection; originals unchanged",
            }
        ).encode()
        require(len(data) <= 128 * 1024, "scan bundle manifest capacity exceeded")
        item = tarfile.TarInfo("bundle-sources.json")
        item.size = len(data)
        tar.addfile(item, io.BytesIO(data))
    return {
        "bundle": str(output),
        "files": len(files),
        "bytes": output.stat().st_size,
        "raw_token_request_logs_included": False,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "command",
        choices=("prepare", "capacity", "run", "summary", "bundle", "status", "errors", "stop"),
    )
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--s1", type=Path)
    p.add_argument("--s0", type=Path)
    p.add_argument("--batch", type=int, choices=BATCHES, default=16)
    p.add_argument("--mode", choices=MODES)
    p.add_argument("--remaining", action="store_true")
    p.add_argument("--single-point", action="store_true",
                   help="explicit one-point diagnostic: waive only the B16 order prerequisite")
    p.add_argument("--observation", choices=("buffered-live",), default="buffered-live")
    p.add_argument("--identity-matching", choices=("bound-prefix",), default="bound-prefix")
    p.add_argument("--selection-seed", type=int, default=1666)
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--window-seconds", type=float, default=30)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--setup-timeout", type=float, default=900)
    p.add_argument("--drain-timeout", type=float, default=60)
    p.add_argument("--wait-seconds", type=float, default=65)
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    root = args.root.resolve()
    try:
        if args.command == "prepare":
            require(args.s1 is not None, "scan prepare requires --s1 qualified root")
            value = prepare(
                root,
                args.s1.resolve(),
                options(
                    warmup_steps=args.warmup_steps,
                    repeats=args.repeats,
                    window_seconds=args.window_seconds,
                    setup_timeout=args.setup_timeout,
                    drain_timeout=args.drain_timeout,
                ),
                seed=args.selection_seed,
                s0=args.s0,
            )
        elif args.command in ("run", "capacity"):
            value = run(
                root,
                batch=args.batch,
                mode=args.mode,
                remaining=args.remaining,
                probe=args.command == "capacity",
                single_point=args.single_point,
            )
        elif args.command == "summary":
            value = summary(root)
        elif args.command == "bundle":
            value = bundle(root, args.output or root / f"scan-small-{time.monotonic_ns()}.tar.gz")
        elif args.command == "status":
            value = status(root)
        elif args.command == "errors" and not (root / "stage.json").exists():
            value = {"errors": [], "state": "NOT_STARTED"}
        elif args.command == "errors":
            from specrhythm.serving.fixed_cli import main as fixed_main

            return fixed_main(["errors", "--root", str(root)])
        else:
            require(0 <= args.wait_seconds <= 300, "invalid controlled stop wait")
            value = stop(root, args.wait_seconds)
        print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)
        return 0
    except (DataError, OSError, ValueError, KeyError) as error:
        print_failure(error, label="decode-scan")
        return getattr(error, "details", {}).get("returncode", 1)


if __name__ == "__main__":
    raise SystemExit(main())
