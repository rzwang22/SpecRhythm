"""Phase 4B.4A: concise offline fixed-cohort characterization, without speed gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from specrhythm.phase4.batched_draft_service import write_immutable_report
from specrhythm.phase4.draft_dual_comparison import RAW, positive, read, require, summarize_run
from specrhythm.phase4.draft_metrics import batch_statistics
from specrhythm.phase4.dual_microbatch import FIELDS, positive_size, scheduler_evidence
from specrhythm.phase4.manifest import sha256_file
from specrhythm.phase4.stock_vllm import load_smoke_requests
from specrhythm.phase4.transport import CheckpointJsonl

SIZES = (2, 4, 8, 16, 32, 64, 100)
SCOPE = (
    "Saturated corrected-100 burst, short outputs (max_new_tokens=16); no online/TTFT/SLO claim."
)
TABLE = (
    ("Throughput (tok/s)", "throughput_tokens_per_second"),
    ("Makespan (ms)", "decode_makespan_ms"),
    ("TPOT mean (ms)", "tpot_ms.mean"),
    ("Target forwards", "target_forward_count"),
    ("Draft forwards", "draft_model_forward_count"),
    ("Draft batch p50", "draft_batch_size.p50"),
    ("Draft batch p90", "draft_batch_size.p90"),
    ("Verify batch p50", "verification_batch_size.p50"),
    ("Verify batch p90", "verification_batch_size.p90"),
    ("Draft GPU time (ms)", "draft_gpu_event_time_ms"),
    ("Overlap (ms)", "observed_overlap_ms"),
    ("Accepted length", "mean_accepted_length"),
    ("Dual / Target throughput", "dual_over_target_throughput"),
    ("Dual / Serial throughput", "dual_over_serial_throughput"),
)


def summarize_cell(directory, mode, workload, requested=None):
    result = summarize_run(directory, mode, 100, workload, characterization=mode == "dual")
    result["schema_version"] = "specrhythm.phase4b4a-cell.v1"
    if mode == "dual":
        result.update(dict(zip(FIELDS, (positive_size(requested), None))))
    if not result["valid"]:
        return result
    try:
        requests = load_smoke_requests(workload, 100, require_task_mixture=True)
        require(all(r.maximum_new_tokens == 16 for r in requests), "output limit must remain 16")
        require(
            result["metrics"]["measured_committed_tokens"] == 1487,
            "expected 1487 measured committed tokens",
        )
        if mode == "dual":
            requested = positive_size(requested)
            raw = read(directory / RAW[mode])
            require(
                raw.get("runtime_semantics", {}).get("overlap_requirement") == "characterization",
                "Dual cell must use characterization overlap policy",
            )
            scheduler = CheckpointJsonl(directory / "scheduler-events.jsonl").read()
            fields = scheduler_evidence(scheduler, requested)
            for name in (
                RAW[mode],
                "plugin-report.json",
                "runtime-manifest.json",
                "decode-performance.json",
            ):
                value = read(directory / name)
                require(
                    all(type(value.get(k)) is int and value[k] == fields[k] for k in FIELDS),
                    f"{name}: requested/effective microbatch mismatch",
                )
            verify = CheckpointJsonl(directory / "verification-events.jsonl").read()
            require(
                all(0 < len(r["verify_request_ids"]) <= requested for r in verify),
                "verification exceeds requested microbatch upper bound",
            )
            result.update(fields)
            result["metrics"].update(fields)
            result["diagnostics"]["dual_scheduler_constraints"] = scheduler[0].get(
                "dual_scheduler_constraints"
            )
            cohorts = result["metrics"]["draft_ready_cohorts"]
            result["metrics"]["cohort_distributions"] = {
                op: batch_statistics(
                    {r["request_count"]: r["count"] for r in cohorts if r["operation"] == op}
                )
                for op in ("propose_only", "commit_and_propose", "finish_tail")
            }
        if mode != "target":
            backend = read(directory / "draft-backend-report.json")
            for k in ("min", "p10", "p50", "p90", "max", "mean"):
                require(
                    positive(result["metrics"]["draft_batch_size"][k]),
                    f"Draft batch {k} metric invalid",
                )
            syncs = backend.get("draft_host_sync_count")
            require(type(syncs) is int and syncs >= 0, "Draft host sync count invalid")
            result["metrics"].update(
                {
                    k: backend.get(k)
                    for k in (
                        "draft_batch_statistics_by_purpose",
                        "draft_gpu_event_time_ms_by_purpose",
                        "draft_host_sync_count_by_reason",
                    )
                }
            )
        # One final sidecar binds the common N to every retained runtime file,
        # including unchanged Draft artifacts. No per-verification logging added.
        result["runtime_artifact_sha256"] = {
            p.name: sha256_file(p)
            for p in sorted(directory.iterdir())
            if p.is_file()
            and p.suffix in (".json", ".jsonl")
            and p.name not in ("qualification.json", "dual-microbatch.json")
        }
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        result.update(valid=False, performance_result=False, errors=[str(error)])
    return result


def compare_sweep(root):
    cells = {}
    errors = []
    names = ("target", "serial", *(f"dual-mb{n}" for n in SIZES))
    try:
        preflight = read(root / "preflight.json")
        require(preflight.get("valid") is True and not preflight.get("errors"), "preflight failed")
        for name in names:
            cell = read(root / name / "qualification.json")
            cells[name] = cell
            require(
                cell.get("valid") is True
                and not cell.get("errors")
                and cell.get("performance_result") is True,
                f"{name}: execution failed: {cell.get('errors')}",
            )
            require(
                cell["input_sha256"]["performance"]
                == sha256_file(root / name / "decode-performance.json"),
                f"{name}: measured artifact changed",
            )
            require(
                cell["metrics"]["completed_requests"] == 100
                and cell["metrics"]["measured_committed_tokens"] == 1487,
                f"{name}: request/token accounting differs",
            )
            expected_mode = "dual" if name.startswith("dual-") else name
            require(cell["mode"] == expected_mode, f"{name}: mode differs")
            if expected_mode == "dual":
                n = int(name.removeprefix("dual-mb"))
                require(
                    all(cell.get(k) == n and cell["metrics"].get(k) == n for k in FIELDS),
                    f"{name}: requested/effective upper bound differs",
                )
        identities = [c["experiment_identity"] for c in cells.values()]
        require(
            all(i == identities[0] for i in identities),
            "same-session workload/model/config/server/commit identity differs",
        )
        require(
            identities[0]["execution_git_commit"] == preflight["execution_git_commit"],
            "sweep commit differs from preflight",
        )
        require(
            identities[0]["workload_sha256"]
            == preflight["inputs"]["SR_PHASE4B_WORKLOAD"]["sha256"]
            and identities[0]["config_sha256"] == preflight["config_sha256"],
            "sweep input/config differs from preflight",
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        errors.append(str(error))
    warnings, observations = [], {}
    if not errors:
        m = {k: c["metrics"] for k, c in cells.items()}
        for n in SIZES:
            c = m[f"dual-mb{n}"]
            for mode in ("target", "serial"):
                c[f"dual_over_{mode}_throughput"] = (
                    c["throughput_tokens_per_second"] / m[mode]["throughput_tokens_per_second"]
                )
            c["dual_minus_serial_work"] = {
                k: c[k] - m["serial"][k]
                for k in (
                    "draft_proposal_count",
                    "proposed_tokens",
                    "accepted_draft_tokens",
                    "rejected_draft_tokens",
                    "target_forward_count",
                    "target_query_tokens",
                    "correction_count",
                    "bonus_count",
                    "measured_committed_tokens",
                )
            }
            c["dual_serial_work_exactly_matched"] = not any(c["dual_minus_serial_work"].values())
        first, last = m["dual-mb2"], m["dual-mb100"]
        historical = {"target_forward_count": 268, "draft_model_forward_count": 924}
        if (
            first["draft_batch_size"]["p50"] != 2
            or first["verification_batch_size"]["p50"] != 2
            or any(not v / 2 <= first[k] <= v * 2 for k, v in historical.items())
        ):
            warnings.append(
                "mb2 structure differs substantially from D6: inspect before interpreting "
                "the sweep. Diagnostic heuristic: p50 != 2 or forwards outside 0.5–2x; "
                "historical timings are not qualification gates."
            )
        best = max(SIZES, key=lambda n: m[f"dual-mb{n}"]["throughput_tokens_per_second"])
        observations = {
            "best_observed_microbatch": best,
            "peak_is_intermediate": best not in (2, 100),
            "mb100_minus_mb2": {
                k: last[k] - first[k]
                for k in (
                    "target_forward_count",
                    "draft_model_forward_count",
                    "draft_gpu_event_time_ms",
                    "observed_overlap_ms",
                    "throughput_tokens_per_second",
                )
            },
            "draft_p50_mb2_mb100": [
                first["draft_batch_size"]["p50"],
                last["draft_batch_size"]["p50"],
            ],
            "verify_p50_mb2_mb100": [
                first["verification_batch_size"]["p50"],
                last["verification_batch_size"]["p50"],
            ],
            "mb100_over_serial_throughput": last["dual_over_serial_throughput"],
            "interpretation": "Descriptive single-session curve; no optimal policy established. "
            "Assess fragmentation recovery, any intermediate peak, and remaining Dual overhead "
            "against work differences before choosing the next mechanism.",
        }
    return {
        "schema_version": "specrhythm.phase4b4a-sweep.v1",
        "valid": not errors,
        "errors": errors,
        "cells": cells,
        "baseline_warnings": warnings,
        "observations": observations,
        "scope": SCOPE,
        "historical_d6_is_context_only": True,
        "overlap_is_critical_path_time_saved": False,
        "pure_batching_speedup_claim": False,
        "label": "production vLLM Batched Draft end-to-end improvement",
    }


def render_sweep(report):
    lines = [
        "# Phase 4B.4A — Fixed Dual Microbatch Characterization",
        "",
        report["scope"],
        "",
        f"Valid: {report['valid']}; errors: {report['errors']}",
        "",
    ]
    lines.extend("WARNING: " + w for w in report["baseline_warnings"])
    lines.extend(["", "Metric | " + " | ".join(map(str, SIZES)),
                  "--- | " + " | ".join(["---:"] * len(SIZES))])
    for label, key in TABLE:
        values = []
        for n in SIZES:
            value = report["cells"].get(f"dual-mb{n}", {}).get("metrics", {})
            for part in key.split("."):
                value = value.get(part, "—") if isinstance(value, dict) else "—"
            values.append(f"{value:.6g}" if type(value) is float else str(value))
        lines.append(label + " | " + " | ".join(values))
    lines.extend(["", "Same-session controls (tok/s; makespan ms; TPOT mean ms):", ""])
    for mode in ("target", "serial"):
        m = report["cells"].get(mode, {}).get("metrics", {})
        lines.append(
            f"- {mode}: {m.get('throughput_tokens_per_second')}; "
            f"{m.get('decode_makespan_ms')}; {m.get('tpot_ms', {}).get('mean')}"
        )
    lines.extend(
        [
            "",
            "Observed overlap is not critical-path time saved. No pure-batching claim.",
            "",
            "```json",
            json.dumps(report["observations"], indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cell = sub.add_parser("validate")
    cell.add_argument("--run-root", type=Path, required=True)
    cell.add_argument("--mode", choices=tuple(RAW), required=True)
    cell.add_argument("--workload", type=Path, required=True)
    cell.add_argument("--microbatch-size", type=positive_size)
    compare = sub.add_parser("compare")
    compare.add_argument("--root", type=Path, required=True)
    compare.add_argument("--markdown", type=Path, required=True)
    for p in (cell, compare):
        p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "output must be fresh")
    if args.command == "validate":
        result = summarize_cell(args.run_root, args.mode, args.workload, args.microbatch_size)
        if args.mode == "dual":
            write_immutable_report(
                args.run_root / "dual-microbatch.json",
                {
                    **{k: result.get(k) for k in FIELDS},
                    "valid": result["valid"],
                    "errors": result["errors"],
                    "runtime_artifact_sha256": result.get("runtime_artifact_sha256", {}),
                    "effective_semantics": (
                        "scheduler-loaded upper bound; attained batches reported separately"
                    ),
                },
            )
        print(json.dumps({k: result[k] for k in ("valid", "errors", "metrics")}, indent=2))
    else:
        require(not args.markdown.exists(), "Markdown output must be fresh")
        result = compare_sweep(args.root)
        args.markdown.write_text(render_sweep(result))
        print(render_sweep(result))
    write_immutable_report(args.output, result)
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
