"""One bounded, content-deduplicated operator upload for success or first failure.

Only offline export reads large JSON. Raw protocol/native/host records are retained
once, with their original phase budgets and dropped counts. Source files never change.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

from specrhythm.serving.audit_layer_report import write
from specrhythm.serving.common import read_json, require
from specrhythm.serving.execution_evidence import qualify
from specrhythm.serving.ping_prepost import MODES

FILE_LIMIT = 512 * 1024 * 1024
TOTAL_LIMIT = 1024 * 1024 * 1024
FILE_COUNT = 192
NAMES = {
    "scan-config.json",
    "config.json",
    "patch-manifest.json",
    "environment.json",
    "topology.json",
    "execution-B16.json",
    "execution-manifest.json",
    "point.json",
    "runtime.json",
    "draft-backend-report.json",
    "light-summary.json",
    "process-lifecycle.json",
    "result.json",
    "coverage.json",
    "failure.json",
    "first-failure.json",
    "comparison.json",
    "run-state.json",
    "stage.json",
    "diagnostic-primary-error.json",
    "diagnostic-secondary-errors.json",
    "exit-code.json",
    "launcher-failure.json",
    "child-failure.json",
    "draft-child-failure.json",
    "actual-capacity.json",
    "target-incremental-setup.json",
    "drain-state.json",
    "fixed-logging-coordinator.json",
    "fixed-logging-draft.json",
    "fixed-logging-target-rank-0.json",
    "fixed-logging-target-rank-1.json",
}
RUNTIME_KEYS = {
    "events",
    "capacity",
    "probe",
    "measurement_start_ns",
    "measurement_end_ns",
    "start_ns",
    "end_ns",
    "point",
    "stop_reason",
    "host",
    "target_devices",
    "target_final_memory",
    "target_requests_final",
    "diagnostic_drain",
    "target_steps",
    "requests",
    "prompt_lengths",
    "decode_scan",
    "identity_matching",
    "fixed_logging",
    "draft_shutdown",
    "drain",
    "schema_version",
}
BACKEND_KEYS = {
    "fixed_device",
    "fixed_host",
    "prepost",
    "prepost_physical",
    "draft_audit",
    "provenance",
    "draft_model_forward_count",
    "draft_batch_size_histogram",
    "draft_gpu_event_time_ms",
    "diagnostic_receipts",
    "actual_capacity",
    "schema_version",
    "backend",
    "released_requests",
}


def comparison(directory):
    points = []
    missing = []
    reports = []
    for mode in MODES:
        path = directory / "points" / (mode + "-audit-report.json")
        if not path.exists():
            missing.append(mode)
            points.append(dict(mode=mode, status="MISSING_OR_NOT_STARTED"))
            continue
        require(path.stat().st_size <= 8 * 1024 * 1024, "compact point report too large")
        r = read_json(path)
        reports.append(r)
        points.append(
            dict(
                mode=mode,
                report=str(path.relative_to(directory)),
                qualification=qualify(r),
                **{
                    k: r.get(k)
                    for k in (
                        "throughput_tok_s",
                        "window_ms",
                        "committed_tokens",
                        "steps",
                        "tokens_per_step",
                        "complete_step_wall_ms",
                    )
                },
                overlap=r["pingpong"]["native_overlap"],
            )
        )
    matched = len(reports) == 2 and all(
        reports[0][k] == reports[1][k] for k in ("source_commit", "options", "workload_sha256")
    )
    valid = matched and all(
        qualify(r)["diagnostic_integrity"] == "COMPLETE"
        and all(
            r["original_qualification"][k] == "PASS"
            for k in ("execution_status", "measurement_status", "cleanup_status")
        )
        for r in reports
    )
    joint = directory / "joint/result.json"
    if not joint.exists() or read_json(joint).get("GPU_correctness") != "PASS":
        valid = False
    return dict(
        schema_version="specrhythm.ping-prepost-delivery.v1",
        points=points,
        matched_configuration=matched,
        missing_points=missing,
        joint_correctness="joint/result.json" if joint.exists() else "joint/failure.json",
        valid=valid,
        performance_conclusion="single window mechanism check; "
        "no stable speedup conclusion; use new run tags for later interleaved repeats",
    )


def export(directory, output, *, first_code=0, stage="complete"):
    require(directory.is_dir() and not output.exists(), "new delivery archive required")
    compare_path = directory / "comparison.json"
    if not compare_path.exists():
        try:
            value = comparison(directory)
        except (OSError, ValueError, KeyError) as error:
            value = dict(valid=False, comparison_error=str(error), status="MISSING_OR_INVALID")
        write(value, compare_path)
    candidates = sorted(
        p
        for p in directory.rglob("*")
        if p.is_file()
        and (p.name in NAMES or p.name.endswith(("-audit-report.json", "-evidence-status.json")))
    )
    inventory, objects, emitted, total, failures = [], {}, set(), 0, []
    expected = ["joint/result.json", "comparison.json"]
    for mode in MODES:
        root = directory / "points" / mode
        runs = []
        for path in (root / "runs").glob("*/point.json"):
            try:
                is_probe = read_json(path).get("probe")
            except (OSError, ValueError, AttributeError):
                # Preserve malformed primary evidence in the archive. Failure
                # discovery must not abort export before the first raw file.
                is_probe = False
                failures.append(str(path.relative_to(directory)) + ": invalid point metadata")
            if not is_probe:
                runs.append(path.parent)
        if runs:
            for run in runs:
                expected.extend(
                    str((run / name).relative_to(directory))
                    for name in (
                        "runtime.json",
                        "draft-backend-report.json",
                        "light-summary.json",
                        "process-lifecycle.json",
                    )
                )
        else:
            expected.append("points/" + mode + "/runs/<performance-not-started>")
    for mode in MODES:
        expected.extend(
            [
                "points/" + mode + suffix
                for suffix in ("/scan-config.json", "-audit-report.json", "-evidence-status.json")
            ]
        )
    with tarfile.open(output, "x:gz") as archive:

        def add(name, data):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))

        for index, source in enumerate(candidates):
            name = str(source.relative_to(directory))
            row = dict(path=name)
            inventory.append(row)
            if source.is_symlink() or not source.resolve().is_relative_to(directory.resolve()):
                row["status"] = "OMITTED_NONREGULAR"
                failures.append(name)
                continue
            before = source.stat()
            row["source_bytes"] = before.st_size
            if index >= FILE_COUNT or before.st_size > FILE_LIMIT:
                row["status"] = "OMITTED_LIMIT"
                failures.append(name)
                continue
            try:
                raw = source.read_bytes()
                after = source.stat()
                require(
                    (before.st_size, before.st_mtime_ns, before.st_ino)
                    == (after.st_size, after.st_mtime_ns, after.st_ino),
                    "source changed during export",
                )
                row["source_sha256"] = hashlib.sha256(raw).hexdigest()
                data, projection = raw, None
                if source.name in ("runtime.json", "draft-backend-report.json"):
                    value = json.loads(raw)
                    keys = RUNTIME_KEYS if source.name == "runtime.json" else BACKEND_KEYS
                    if (source.name == "runtime.json"
                            and value.get("point", {}).get("mode") in MODES
                            and "target_final_memory" not in value):
                        row["required_fields_missing"] = ["target_final_memory"]
                        failures.append(name + ": missing target_final_memory (not reconstructed)")

                    projection = dict(
                        omitted_top_level_fields=sorted(set(value) - keys),
                        retained_top_level_fields=sorted(set(value) & keys),
                        event_rows_truncated=False,
                        semantics="keep complete bounded native/host/protocol records; "
                        "large unrelated setup summaries stay on server",
                    )
                    data = json.dumps(
                        {k: v for k, v in value.items() if k in keys}, separators=(",", ":")
                    ).encode()
                digest = hashlib.sha256(data).hexdigest()
                object_path = "objects/" + digest + ".json"
                if digest not in emitted:
                    if total + len(data) > TOTAL_LIMIT:
                        raise ValueError("unique payload total limit exceeded")
                    add(object_path, data)
                    emitted.add(digest)
                    total += len(data)
                row.update(
                    status="INCLUDED",
                    object=object_path,
                    sha256=digest,
                    bytes=len(data),
                    projection=projection,
                )
                objects[name] = object_path
            except (OSError, ValueError) as error:
                row.update(status="EXPORT_ERROR", error=str(error))
                failures.append(name)
        for name in expected:
            if name not in objects:
                inventory.append(dict(path=name, status="MISSING"))
        result = dict(
            schema_version="specrhythm.ping-prepost-archive.v1",
            inventory=inventory,
            logical_paths=objects,
            export_status="INCOMPLETE" if failures else "COMPLETE",
            export_errors=failures,
            export_validation_exit_code=(41 if failures else 42 if first_code == 0 and not
                              read_json(compare_path)["valid"] else 0),
            first_exit_code=first_code,
            failed_stage=stage,
            limits=dict(file_bytes=FILE_LIMIT, unique_payload_bytes=TOTAL_LIMIT, files=FILE_COUNT),
            unique_payload_bytes=total,
            source_results_unchanged=True,
            missing=sum(r["status"] == "MISSING" for r in inventory),
            extraction="logical_paths maps original relative paths to deduplicated JSON objects",
            qualification="archive COMPLETE means eligible files exported; "
            "it does not qualify GPU execution; "
            "missing/failed points remain explicit; see comparison and first-failure",
        )
        add("inventory.json", json.dumps(result, indent=2).encode())
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--first-code", type=int, default=0)
    p.add_argument("--stage", default="complete")
    args = p.parse_args(argv)
    result = export(args.directory, args.output, first_code=args.first_code, stage=args.stage)
    from specrhythm.phase4.manifest import sha256_file

    write(
        {**result, "archive_sha256": sha256_file(args.output)},
        args.directory / "export-status.json",
    )
    print(
        json.dumps({k: result[k] for k in ("export_status", "first_exit_code", "export_errors")})
    )
    if result["export_validation_exit_code"]:
        raise SystemExit(result["export_validation_exit_code"])


if __name__ == "__main__":
    main()
