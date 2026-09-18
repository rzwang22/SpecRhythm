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
from specrhythm.serving.fixed_logging import POST_RUN_LOGS
from specrhythm.serving.ping_prepost import MODES, SCHEDULED_MODES

FILE_LIMIT = 512 * 1024 * 1024
TOTAL_LIMIT = 1024 * 1024 * 1024
FILE_COUNT = 512  # Bounded raw logs/partial files/receipts; byte budgets remain unchanged.
LOGICAL_LIMIT = 512  # Bounded enumeration, including new raw logs and publication states.
NAMES = {
    "dual-batch-plan.json", "smoke-comparison.json",
    "storage-preflight.json", "delivery-status.json", "runner-outcome.json",
    "plugin-report.json", "admission-events.jsonl", "round-events.jsonl",
    "target-diagnostics.jsonl", "transport-events.jsonl",
    "supervisor-decision.json", "draft-report-state.json",
    "runner.log", "target.log", "draft.log", "draft-service.log", "draft-work-events.jsonl",
    "measurement-snapshot.json", "s2-control.json", "draft-service-ready.json",
    "draft-startup.json",
    "ownership.json", "draft-owner.json", "command.json",
    "scan-config.json",
    "config.json",
    "patch-manifest.json",
    "environment.json",
    "topology.json",
    "execution-B16.json", "execution-B64.json", "execution-B128.json",
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
    "drain-state.json", "draft-drain-state.json",
    "startup-cleanup.json",
    "k3-capacity-contract.json", "validation-plan.json",
    "fixed-logging-coordinator.json",
    "fixed-logging-draft.json",
    "fixed-logging-target-rank-0.json",
    "fixed-logging-target-rank-1.json",
}
NAMES.update(POST_RUN_LOGS)
RUNTIME_KEYS = {
    "events",
    "capacity",
    "probe", "validation_profile", "k3_configuration", "full_output_comparison_run",
    "output_equivalence_status", "output_equivalence_scope", "output_equivalence_reason",
    "known_output_difference",
    "measurement_start_ns",
    "measurement_end_ns",
    "start_ns",
    "end_ns",
    "point",
    "stop_reason",
    "host",
    "target_devices",
    "target_final_memory",
    "diagnostic_configuration",
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
    "report_publication",
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


def byte_limits(directory):
    """Explicit scaled K3 export bounds; producer retention/logging is unchanged."""
    from specrhythm.serving.k3 import B16, MODES, configuration_of, matches_geometry

    path = directory / "k3-capacity-contract.json"
    if not path.exists():
        return FILE_LIMIT, TOTAL_LIMIT
    require(path.stat().st_size < 128 * 1024, "invalid static capacity receipt size")
    receipt = read_json(path)
    if configuration_of(receipt) == B16:
        return FILE_LIMIT, TOTAL_LIMIT
    rows = receipt["reservations"]
    require(receipt["static_contract"] == "PASS" and len(rows) == 8
            and {(r["mode"], r["role"]) for r in rows}
            == {(m, role) for m in MODES for role in ("target", "draft")}
            and all(matches_geometry(r["geometry"], r["mode"], configuration_of(receipt))
                    for r in rows),
            "Scaled K3 export capacity requires explicit complete static configuration")
    return 4 * FILE_LIMIT, 4 * TOTAL_LIMIT


def comparison(directory, *, modes=MODES):
    from specrhythm.serving.k3 import configuration_of
    from specrhythm.serving.k3_acceptance import full_batch_receipt
    from specrhythm.serving.k3_validation import EXPLORATION, profile_of, read_plan

    plan = read_plan(directory)
    policy = profile_of(plan)
    exploration = policy == EXPLORATION
    points = []
    missing = []
    reports = []
    for mode in modes:
        path = directory / "points" / (mode + "-audit-report.json")
        if not path.exists():
            missing.append(mode)
            points.append(dict(mode=mode, status="MISSING_OR_NOT_STARTED"))
            continue
        require(path.stat().st_size <= 8 * 1024 * 1024, "compact point report too large")
        from specrhythm.serving.audit_details import restore_file

        r = restore_file(read_json(path), path.parent)
        require(r["mode"] == mode, "comparison report/file mode mismatch", artifact=str(path))
        reports.append(r)
        points.append(
            dict(
                mode=mode,
                report=str(path.relative_to(directory)),
                qualification=qualify(r),
                **{
                    k: r.get(k)
                    for k in (
                        "validation_profile", "full_output_comparison_run",
                        "output_equivalence_status", "measurement_valid", "native_geometry_status",
                        "native_target_geometry",
                        "execution_geometry", "actual_target_batch", "warmup_coverage",
                        "k3_configuration", "request_verification_opportunities",
                        "tokens_per_request_opportunity", "window_ms_per_active_opportunities",
                        "measurement_start_ns", "measurement_end_ns",
                        "capture_target_forward", "control_snapshot_publication",
                        "target_diagnostic_substages", "target_diagnostic_substage_scope",
                        "physical_Draft_calls_per_Target_step", "target_ranks",
                        "effective_target_diagnostic_configuration", "diagnostic_configuration",
                        "selected_initial_request_ids", "planned_initial_request_ids",
                        "selected_request_scope", "execution_path",
                        "throughput_tok_s",
                        "window_ms",
                        "committed_tokens",
                        "steps",
                        "tokens_per_step",
                        "complete_step_wall_ms",
                        "window_average_cadence_ms",
                        "outside_complete_steps_ms",
                    )
                },
                overlap=r["pingpong"]["native_overlap"],
            )
        )
        if mode.endswith("-k3"):
            points[-1]["cycle_accounting"] = {k: v for k, v in
                (r.get("cycle_accounting") or {}).items()
                if k not in ("steps", "request_cycles", "feedback_owner_queue", "cadence_cycles")}
            points[-1]["draft_dispatch"] = {k: v for k, v in
                r["pingpong"].get("draft_dispatch", {}).items() if k != "calls"}
            points[-1]["diagnostic_logging"] = r.get("diagnostic_logging")
            points[-1]["K3_mechanism"] = {
                k: r["pingpong"].get(k) for k in (
                    "outcomes", "generated", "retained", "discarded", "lookahead_rates",
                    "ready_to_admission_ms", "feedback_to_ready_ms")}
            points[-1]["pipeline"] = {k: v for k, v in r["pingpong"].get("pipeline", {}).items()
                                       if k not in ("timeline", "rejection_cycle", "dispatch")}
            points[-1]['dispatch'] = {k: v for k, v in
                r['pingpong'].get('pipeline', {}).get('dispatch', {}).items() if k != 'rows'}
    matched = len(reports) == len(modes) and all(
        reports[0][k] == r[k] for r in reports[1:]
        for k in ("source_commit", "options", "workload_sha256")
    )
    if "serial-eager-k3" in modes:
        from specrhythm.serving.k3 import configuration_of, matches_geometry

        matched = matched and all(matches_geometry(r.get("execution_geometry"), r["mode"],
                                                       configuration_of(r))
                                  for r in reports)
        matched = matched and len({configuration_of(r) for r in reports}) == 1
        matched = matched and bool(reports) and bool(reports[0].get("common_execution")) and all(
            r.get("common_execution") == reports[0]["common_execution"] for r in reports)
    matched = matched and all(profile_of(r) == policy for r in reports)
    if plan:
        matched = matched and all(configuration_of(r) == configuration_of(plan) for r in reports)
    valid = matched and all(
        qualify(r)["diagnostic_integrity"] == "COMPLETE"
        and all(
            r["original_qualification"][k] == "PASS"
            for k in ("execution_status", "measurement_status", "cleanup_status")
        )
        for r in reports
    )
    measurements_valid = valid
    joint = directory / "joint/result.json"
    joint_failure = directory / "joint/failure.json"
    joint_result = read_json(joint) if joint.exists() else {}
    failure_result = read_json(joint_failure) if joint_failure.exists() else {}
    native = {}
    if exploration:
        from specrhythm.serving.k3 import geometry
        from specrhythm.serving.k3_acceptance import full_batch_receipt

        native = {r["mode"]: r.get("native_target_geometry") for r in reports}
        valid = valid and not (directory / "joint").exists() and all(
            r.get("full_output_comparison_run") is False
            and r.get("output_equivalence_status") == "NOT_RUN"
            and r.get("measurement_valid") is True
            and r.get("native_geometry_status") == "PASS"
            and r.get("original_run_details", {}).get("capacity_status") == "PASS"
            and r.get("original_run_details", {}).get("effective_exit_code") == 0
            and full_batch_receipt(native.get(r["mode"]), r["mode"], configuration_of(r))
            and set((native.get(r["mode"]) or {}).get("observed_home_cohorts", []))
                == set(geometry(r["mode"], configuration_of(r))["home_capacities"])
            for r in reports)
    else:
        if not joint.exists() or read_json(joint).get("GPU_correctness") != "PASS":
            valid = False
        if any(m.endswith("-k3") for m in modes) and joint.exists():
            joint_value = read_json(joint)
            valid = valid and joint_value.get("protocol") == "specrhythm.uniform-k3.v1"
            valid = valid and {r["mode"] for r in joint_value.get("runs", [])} == {
                "target", *modes}
        if "serial-eager-k3" in modes and joint.exists():
            # New four-mode entries must retain the actual full-batch fixture, including
            # both Serial B16 runs. Older three-mode historical packages are unchanged.
            from specrhythm.serving.k3_acceptance import full_batch_receipt

            native = {r["mode"]: r.get("native_target_geometry")
                      for r in read_json(joint).get("runs", []) if r["mode"] in modes}
            valid = valid and all(full_batch_receipt(native.get(m), m, configuration_of(reports[0])
                                                              if reports else "k3-b16-v1")
                                  for m in modes)
            if reports and configuration_of(reports[0]) != "k3-b16-v1":
                joint_value = read_json(joint)
                valid = valid and configuration_of(joint_value) == configuration_of(reports[0])
                valid = valid and (joint_value.get("request_count")
                                   == reports[0]["execution_geometry"]["active_limit"])
                valid = valid and (joint_value.get("termination_comparison")
                               == "exact by request ID")
    ratios = []
    if "serial-eager-k3" in modes:
        by_mode = {r["mode"]: r for r in reports}
        for numerator, denominator in (
            ("serial-eager-k3", "serial-k3"), ("pingpong-k3", "serial-k3"),
            ("pingpong-eager-k3", "pingpong-k3"),
            ("pingpong-eager-k3", "serial-eager-k3"),
        ):
            a, b = by_mode.get(numerator, {}), by_mode.get(denominator, {})
            ratios.append(dict(numerator=numerator, denominator=denominator,
                throughput_ratio=a["throughput_tok_s"] / b["throughput_tok_s"]
                if valid and a.get("throughput_tok_s") and b.get("throughput_tok_s") else None,
                interpretation="single-window observed ratio; stability not established"))
    return dict(
        schema_version="specrhythm.ping-prepost-delivery.v1",
        points=points,
        matched_configuration=matched,
        common_execution=reports[0].get("common_execution") if reports else None,
        configuration_comparison="same source/options/workload/models/numerics/resources; "
        "explicit K3 mode geometry may differ; no cross-run GPU UUID comparison",
        missing_points=missing,
        validation_profile=policy,
        full_output_comparison_run=False if exploration else
            True if joint_result.get("GPU_correctness") == "PASS" else
            failure_result.get("full_output_comparison_run"),
        output_equivalence_status="NOT_RUN" if exploration else
            "PASS" if joint.exists() and read_json(joint).get("GPU_correctness") == "PASS"
            else failure_result.get("output_equivalence_status", "NOT_QUALIFIED")
                if failure_result else "NOT_RUN",
        output_equivalence_note="This run does not verify complete output equivalence."
            if exploration else "Separate strict joint diagnostic required.",
        known_output_difference=plan.get("known_output_difference"),
        measurement_valid=measurements_valid,
        native_geometry_status="PASS" if len(native) == len(modes) and all(
            full_batch_receipt(native.get(m), m, configuration_of(reports[0]))
            for m in modes) and reports else "NOT_QUALIFIED",
        native_geometry_source="current warmup/runtime" if exploration else "joint correctness",
        joint_correctness=None if exploration else
            "joint/result.json" if joint.exists() else "joint/failure.json",
        joint_native_target_geometry=native,
        paired_ratios=ratios,
        valid=valid,
        performance_conclusion="single window mechanism check; "
        "no stable speedup conclusion; use new run tags for later interleaved repeats",
    )


def export(directory, output, *, first_code=0, stage="complete", modes=MODES):
    if (directory / "experiment-plan.json").exists():
        from specrhythm.serving.k3_repeat_export import export_repeats

        return export_repeats(directory, output, first_code=first_code, stage=stage, modes=modes)
    require(directory.is_dir() and not output.exists(), "new delivery archive required")
    from specrhythm.serving.k3_validation import EXPLORATION, profile_of, read_plan

    policy_error = None
    try:
        plan = read_plan(directory)
    except (OSError, ValueError, KeyError, TypeError) as error:
        plan = {}
        policy_error = "invalid validation plan: " + str(error)
    exploration = profile_of(plan) == EXPLORATION
    limit_error = None
    try:
        file_limit, total_limit = byte_limits(directory)
    except (OSError, ValueError, KeyError, TypeError) as error:
        file_limit, total_limit = FILE_LIMIT, TOTAL_LIMIT
        limit_error = "invalid export capacity declaration: " + str(error)
    compare_path = directory / "comparison.json"
    if not compare_path.exists():
        try:
            if (directory / "dual-batch-plan.json").exists():
                value = dict(valid=False, status="MISSING_OR_INVALID",
                             comparison_error="double-batch runner comparison missing")
            else:
                value = comparison(directory, modes=modes)
        except (OSError, ValueError, KeyError) as error:
            value = dict(valid=False, comparison_error=str(error), status="MISSING_OR_INVALID")
        write(value, compare_path)
    candidates = sorted(
        p
        for p in directory.rglob("*")
        if p.is_file()
        and (p.name in NAMES or (p.name.startswith("fixed-logging-")
                               and p.suffix == ".json")
             or p.name.endswith(("-audit-report.json", "-evidence-status.json"))
             or ("-audit-report-details." in p.name and p.name.endswith(".jsonl"))
             or (p.name.startswith(".draft-backend-report.json.") and p.name.endswith(".partial")))
    )
    inventory, objects, emitted, total, failures = [], {}, set(), 0, []
    checked_logging = set()
    evidence_errors = [e for e in (limit_error, policy_error) if e]
    expected = ["comparison.json"] if exploration else ["joint/result.json", "comparison.json"]
    if exploration:
        expected.extend(["validation-plan.json", "k3-capacity-contract.json"])
    if (directory / "dual-batch-plan.json").exists():
        from specrhythm.serving.dual_batch_run import MODE, ORDER

        expected.extend(["dual-batch-plan.json", "smoke-comparison.json", "runner-outcome.json"])
        roots = [directory / kind / dispatch
                 for kind in ("capacity", "smoke") for dispatch in ORDER[:2]]
        for index, dispatch in enumerate(ORDER):
            case = directory / "windows" / (str(index) + "-" + dispatch)
            expected.extend(str((case / name).relative_to(directory)) for name in (
                "comparison.json", "points/" + MODE + "-audit-report.json",
                "points/" + MODE + "-evidence-status.json"))
            roots.append(case / "points" / MODE)
        for root in roots:
            runs = sorted((root / "runs").glob("*/point.json"))
            if not runs:
                expected.append(str(root.relative_to(directory)) + "/runs/<not-started>")
            for point in runs:
                expected.extend(str(point.with_name(name).relative_to(directory)) for name in (
                    "point.json", "runtime.json", "draft-backend-report.json",
                    "light-summary.json", "process-lifecycle.json"))
    for mode in modes:
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
    for mode in modes:
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
            try:
                before = source.stat()
            except OSError as error:
                row.update(status="EXPORT_ERROR", error=str(error))
                failures.append(name)
                continue
            row["source_bytes"] = before.st_size
            if index >= LOGICAL_LIMIT or before.st_size > file_limit:
                row["status"] = "OMITTED_LIMIT"
                failures.append(name)
                continue
            try:
                with source.open("rb") as handle:
                    raw = handle.read(file_limit + 1)
                require(len(raw) <= file_limit, "source exceeded file byte budget while reading")
                try:
                    after = source.stat()
                except OSError as error:
                    after = None
                    row["source_stat_error"] = str(error)
                if after is None or (before.st_size, before.st_mtime_ns, before.st_ino) != (
                        after.st_size, after.st_mtime_ns, after.st_ino):
                    row["source_changed_during_read"] = True
                    evidence_errors.append(name + ": changing source; captured bytes retained")
                row["source_sha256"] = hashlib.sha256(raw).hexdigest()
                data, projection = raw, None
                value, json_valid = None, False
                if source.suffix == ".json":
                    try:
                        value = json.loads(raw)
                        json_valid = True
                    except (ValueError, UnicodeError) as error:
                        row["parse_error"] = dict(type=type(error).__name__, error=str(error))
                        row["raw_bytes_retained"] = True
                        evidence_errors.append(name + ": malformed JSON (raw bytes included)")
                if (source.name.startswith("fixed-logging-") and isinstance(value, dict)
                        and value.get("observation") == "deferred-window"
                        and source.parent not in checked_logging):
                    from specrhythm.serving.fixed_logging import qualify as qualify_logging

                    checked_logging.add(source.parent)
                    try:
                        qualify_logging(source.parent, "deferred-window")
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        evidence_errors.append(name + ": diagnostic publication: " + str(error))
                if source.name.endswith("-audit-report.json") and isinstance(value, dict):
                    from specrhythm.serving.audit_details import restore_file

                    try:
                        restore_file(value, source.parent)
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        evidence_errors.append(name + ": audit details: " + str(error))
                if source.name.startswith(".draft-backend-report.json."):
                    row["publication"] = "PARTIAL_UNPUBLISHED"
                    evidence_errors.append(name + ": unfinished report bytes retained")
                if source.name == "draft-report-state.json":
                    if not isinstance(value, dict) or value.get("status") != "COMPLETE":
                        evidence_errors.append(name + ": report publication not COMPLETE")
                    elif not source.with_name("draft-backend-report.json").is_file():
                        evidence_errors.append(name + ": COMPLETE receipt without final report")
                if source.name == "draft-backend-report.json":
                    marker = source.with_name("draft-report-state.json")
                    if marker.exists() or (isinstance(value, dict)
                                           and "report_publication" in value):
                        from specrhythm.phase4.report_publication import qualify_final_report

                        try:
                            row["publication"] = qualify_final_report(source)
                        except (OSError, ValueError, KeyError, TypeError) as error:
                            row["publication_error"] = str(error)
                            evidence_errors.append(name + ": final publication incomplete")
                    else:
                        row["publication"] = "LEGACY_UNDECLARED; existence is not completion"
                        if (directory / "storage-preflight.json").exists():
                            evidence_errors.append(name + ": new entry lacks publication receipt")
                if source.name in ("runtime.json", "draft-backend-report.json") and json_valid:
                    if not isinstance(value, dict):
                        row["parse_error"] = dict(type="SchemaError", error="expected JSON object")
                        row["raw_bytes_retained"] = True
                        evidence_errors.append(name + ": non-object JSON (raw bytes included)")
                    else:
                        keys = RUNTIME_KEYS if source.name == "runtime.json" else BACKEND_KEYS
                        if (source.name == "runtime.json"
                                and value.get("point", {}).get("mode") in SCHEDULED_MODES
                                and "target_final_memory" not in value):
                            row["required_fields_missing"] = ["target_final_memory"]
                            evidence_errors.append(name + ": missing target_final_memory")
                        projection = dict(
                            omitted_top_level_fields=sorted(set(value) - keys),
                            retained_top_level_fields=sorted(set(value) & keys),
                            event_rows_truncated=False,
                            semantics="complete bounded native/host/protocol records; "
                            "unrelated setup summaries stay in retained local run directory")
                        data = json.dumps({k: v for k, v in value.items() if k in keys},
                                          separators=(",", ":")).encode()
                digest = hashlib.sha256(data).hexdigest()
                object_path = "objects/" + digest + ".json"
                if digest not in emitted:
                    require(len(emitted) < FILE_COUNT, "unique payload file budget exceeded")
                    if total + len(data) > total_limit:
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
                if exploration and first_code == 0:
                    evidence_errors.append(name + ": required performance evidence missing")
        try:
            comparison_valid = read_json(compare_path)["valid"] is True
        except (OSError, ValueError, KeyError, TypeError) as error:
            comparison_valid = False
            evidence_errors.append("comparison.json: " + str(error))
        result = dict(
            schema_version="specrhythm.ping-prepost-archive.v1",
            validation_profile=profile_of(plan),
            intentionally_not_run=[dict(path="joint/", reason="explicit performance-exploration",
                output_equivalence_status="NOT_RUN")] if exploration else [],
            inventory=inventory,
            logical_paths=objects,
            archive_integrity="INCOMPLETE" if failures else "COMPLETE",
            evidence_integrity="INCOMPLETE" if evidence_errors or failures else "COLLECTED",
            evidence_errors=evidence_errors,
            export_status="INCOMPLETE" if failures or evidence_errors else "COMPLETE",
            export_errors=failures + evidence_errors,
            export_validation_exit_code=(41 if failures or evidence_errors else
                                         42 if first_code == 0 and not comparison_valid else 0),
            first_exit_code=first_code,
            failed_stage=stage,
            limits=dict(file_bytes=file_limit, unique_payload_bytes=total_limit,
                        files=FILE_COUNT, file_count_scope="unique payloads",
                        unique_files=FILE_COUNT, logical_files=LOGICAL_LIMIT),
            unique_payload_bytes=total,
            source_results_unchanged=True,
            missing=sum(r["status"] == "MISSING" for r in inventory),
            extraction="logical_paths maps relative paths to deduplicated payload objects; "
                       "raw logs/corrupt JSON retain original bytes",
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
    p.add_argument("--k3", action="store_true")
    args = p.parse_args(argv)
    from specrhythm.serving.k3 import MODES as K3_MODES

    result = export(args.directory, args.output, first_code=args.first_code, stage=args.stage,
                    modes=K3_MODES if args.k3 else MODES)
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
