"""Print the first failure layer before exporting evidence; never rewrite run status."""

import argparse
import json
from pathlib import Path

from specrhythm.serving.audit_layer_report import MAX_OUTPUT, write


def read_optional(path, errors):
    if not path.exists():
        return {}
    try:
        if path.is_symlink() or path.stat().st_size > MAX_OUTPUT:
            raise ValueError("not a bounded regular diagnostic artifact")
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("expected object")
        return value
    except (OSError, ValueError) as error:
        errors.append(dict(path=str(path), error=str(error)[:1024]))
        return {}


def summarize(root, exit_code, stage):
    root = Path(root)
    status_path = root.with_name(root.name + "-evidence-status.json")
    errors = []
    status = read_optional(status_path, errors)
    report = read_optional(root.with_name(root.name + "-audit-report.json"), errors)
    missing = status.get("missing_fsync_attribution")
    if missing is None:
        # Older status schemas omit this detail. Read missing rows, never guess names.
        missing = [r for r in report.get("execution_path", {}).get("fsync_by_file", [])
                   if not r.get("log_name") or r["log_name"] == "MISSING_FILENAME"]
    runs = []
    paths = sorted({*list((root / "runs").glob("*/light-summary.json")),
                    *(p.with_name("light-summary.json") for p in
                      (root / "runs").glob("*/supervisor-decision.json"))})
    if len(paths) > 16:
        errors.append(dict(error="light-summary inventory truncated", retained=16,
                           available=len(paths)))
    for path in paths[:16]:
        row = read_optional(path, errors)
        from specrhythm.serving.fixed_artifacts import retained_report

        if (path.parent / "supervisor-decision.json").exists():
            row = retained_report(path.parent, row)
        runs.append(dict(path=str(path), **{k: row.get(k) for k in (
            "capacity_status", "execution_status", "measurement_status", "cleanup_status",
            "formal_comparison_eligible", "effective_exit_code", "errors",
            "qualification_status", "failure_layer", "primary_error",
        )}))
        if (path.parent / "startup-cleanup.json").exists():
            runs[-1]["startup_cleanup"] = read_optional(
                path.parent / "startup-cleanup.json", errors)
        if (path.parent / "exit-code.json").exists():
            runs[-1]["process_exit_codes"] = read_optional(path.parent / "exit-code.json", errors)
    return dict(
        schema_version="specrhythm.execution-first-failure.v1",
        first_exit_code=exit_code,
        failure_layer=status.get("failure_layer") or next(
            (r["failure_layer"] for r in reversed(runs) if r.get("failure_layer")), stage),
        failed_stage=stage,
        original_qualification=status.get("original_qualification"),
        original_run_details=status.get("original_run_details"),
        run_qualifications=runs,
        diagnostic_integrity=status.get("diagnostic_integrity", "MISSING"),
        primary_diagnostic_errors=status.get("errors"),
        missing_fsync_attribution=missing,
        performance_conclusion=status.get("performance_conclusion", "PENDING"),
        evidence_status=str(status_path),
        evidence_packages=[str(root) + suffix for suffix in (
            "-failure-small.tar.gz", "-failure-evidence.tar.gz")],
        package_state="planned; exporter success/existence reported separately",
        summary_read_errors=errors,
        original_results_unchanged=True,
    )


def joint_run_failure(root, mode, error, layer, point=None):
    """Read only the currently invoked mode's actual run, never a prior capacity point."""
    root = Path(root)
    errors = []
    details = getattr(error, "details", {})
    stage = read_optional(root / "stage.json", errors)
    candidate = point or details.get("directory") or stage.get("directory")
    run = Path(candidate) if candidate else None
    if run is not None and run.resolve().parent != (root / "runs").resolve():
        errors.append(dict(error="joint failure run is outside the invoked mode root",
                           path=str(run)))
        run = None
    report, exits = {}, {}
    if run is not None:
        report = read_optional(run / "light-summary.json", errors)
        if not report:
            report = read_optional(run / "result.json", errors)
        exits = read_optional(run / "exit-code.json", errors)
    code = details.get("returncode", 130 if isinstance(error, KeyboardInterrupt) else 1)
    code = code if type(code) is int and 0 < code < 256 else 1
    return dict(
        outer_stage="joint_gpu_correctness", mode=mode, point=mode,
        mode_root=str(root), run_directory=str(run) if run is not None else None,
        failure_layer=report.get("failure_layer") or details.get("failure_layer") or layer,
        primary_error=report.get("primary_error") or dict(error=str(error), **details),
        effective_exit_code=report.get("effective_exit_code", exits.get("effective_exit_code")),
        command_exit_code=code, original_returncode=code,
        qualification_status=report.get("qualification_status"),
        execution_status=report.get("execution_status"),
        measurement_status=report.get("measurement_status"),
        cleanup_status=report.get("cleanup_status"),
        source_report=str(run / "light-summary.json") if run is not None else None,
        missing_run_evidence=not bool(report), summary_read_errors=errors,
    )


def joint_comparison_failure(directory, error, runtimes, receipts):
    """A cross-run comparison belongs to no last-invoked model process."""
    reference = {r["request_id"]: r for r in runtimes["target"]["requests"]}
    sources = {r["mode"]: str(Path(r["point"]) / "runtime.json") for r in receipts}
    mismatches = []
    for mode, runtime in runtimes.items():
        if mode == "target":
            continue
        actual = {r["request_id"]: r for r in runtime["requests"]}
        for rid in sorted(set(reference) | set(actual)):
            a, b = actual.get(rid), reference.get(rid)
            tokens = a is not None and b is not None and (
                a["generated_token_ids"] == b["generated_token_ids"])
            terminal = a is not None and b is not None and a.get("finish_reason") == b.get(
                "finish_reason")
            if not tokens or not terminal:
                mismatches.append(dict(mode=mode, request_id=rid, tokens_equal=tokens,
                    termination_equal=terminal, source=sources[mode], reference=sources["target"]))
    runs = []
    for receipt in receipts:
        errors = []
        path = Path(receipt["point"]) / "light-summary.json"
        report = read_optional(path, errors)
        runs.append(dict(mode=receipt["mode"], run_directory=receipt["point"],
                         source_report=str(path), summary_read_errors=errors,
                         **{k: report.get(k) for k in ("effective_exit_code", "execution_status",
                                                       "cleanup_status")}))
    return dict(outer_stage="joint_gpu_correctness", failure_layer="comparison",
                mode=None, point="comparison", run_directory=str(directory),
                command_exit_code=1, original_returncode=1,
                effective_exit_code=None, process_results=runs,
                mismatched_modes=sorted({r["mode"] for r in mismatches}),
                mismatches=mismatches, evidence_sources=sources,
                primary_error=dict(error=str(error), mismatches=mismatches),
                comparison_scope="all completed modes versus Target-only; "
                "no single failed process")


def summarize_joint(root, exit_code, stage="joint_gpu_correctness"):
    """Project the joint runner's first failure without inventing a failed next mode."""
    root = Path(root)
    errors = []
    failure = read_optional(root / "failure.json", errors)
    return {
        **failure, "first_exit_code": exit_code, "failed_stage": stage, "outer_stage": stage,
        "failure_layer": failure.get("failure_layer", "joint_failure_evidence_missing"),
        "point": failure.get("point", failure.get("mode")),
        "run_directory": failure.get("run_directory"),
        "joint_failure_source": str(root / "failure.json"),
        "summary_read_errors": [*failure.get("summary_read_errors", []), *errors],
        "original_results_unchanged": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = summarize(args.root, args.exit_code, args.stage)
    # Console first: a sidecar write error must not hide the original failure details.
    print(json.dumps(value, sort_keys=True), flush=True)
    write(value, args.output)


if __name__ == "__main__":
    main()
