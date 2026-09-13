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
    paths = sorted((root / "runs").glob("*/light-summary.json"))
    if len(paths) > 16:
        errors.append(dict(error="light-summary inventory truncated", retained=16,
                           available=len(paths)))
    for path in paths[:16]:
        row = read_optional(path, errors)
        runs.append(dict(path=str(path), **{k: row.get(k) for k in (
            "capacity_status", "execution_status", "measurement_status", "cleanup_status",
            "formal_comparison_eligible", "effective_exit_code", "errors",
        )}))
    return dict(
        schema_version="specrhythm.execution-first-failure.v1",
        first_exit_code=exit_code,
        failure_layer=status.get("failure_layer") or stage,
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
