"""120-second externally supervised offline analysis/export; never launches inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path, PurePosixPath

from specrhythm.serving.fixed_attribution import analyze_raw
from specrhythm.serving.fixed_attribution_io import (
    MODES,
    Inputs,
    atomic_json,
    export_file,
    mode_from_path,
    project,
)
from specrhythm.serving.fixed_results import clipped, duration


class Progress:
    def __init__(self, output):
        self.output, self.last = output, 0.0
        self.value = {
            "events_processed": 0,
            "bytes_read": 0,
            "processed_files": [],
            "phase": "starting",
        }

    def __call__(self, **values):
        self.value["events_processed"] += values.pop("events", 0)
        self.value["bytes_read"] += values.pop("bytes_read", 0)
        filename = values.pop("processed_file", None)
        if filename:
            self.value["processed_files"].append(filename)
        self.value.update(values)
        now = time.monotonic()
        if filename or now - self.last >= 1:
            self.last = now
            atomic_json(self.output / "progress.json", self.value)
            print(
                f"[fixed-attribution CPU] {self.value['phase']} "
                f"files={len(self.value['processed_files'])} "
                f"events={self.value['events_processed']} "
                f"current={self.value.get('current_file', '-')}",
                flush=True,
            )


def event_summary(events, start, end):
    """Keep associations, not bulk events; server RPCs have no request correlation ID."""
    operations, spans, proposals = Counter(), [], {}
    failed = 0
    for event in events:
        if event.get("success") is False:
            failed += 1
        a, b = event.get("start_ns"), event.get("end_ns")
        if isinstance(a, int) and isinstance(b, int) and b >= start and a <= end:
            operations[event.get("operation", "unknown")] += 1
            spans.append((a, b))
        result = event.get("result", {})
        proposal = result.get("proposal", {})
        if proposal:
            key = proposal.get("request_id", event.get("request_id")), proposal.get("round_id")
            if key in proposals:
                raise ValueError("duplicate owner result for request/round: " + str(key))
            proposals[key] = {
                "proposal_id": proposal.get("proposal_id"),
                "prefix_version": proposal.get("prefix_version"),
                "prefix_length": proposal.get(
                    "prefix_token_count", proposal.get("parent_prefix_len")
                ),
                "draft_start_ns": proposal.get("draft_start_ns"),
                "draft_end_ns": proposal.get("draft_end_ns"),
                "created_timestamp_ns": proposal.get("created_timestamp_ns"),
                "draft_sync_complete_ns": result.get("draft_sync_complete_ns"),
                "owner_work_start_ns": a,
                "work_record_end_ns": b,
                "source_line": event.get("source_line"),
                "source_record_sha256": event.get("source_record_sha256"),
            }
    return {
        "event_count": len(events),
        "failed_event_count": failed,
        "window_operations": dict(operations),
        "window_interval_union_ms": duration(clipped(spans, start, end)),
        "proposal_identity_count": len(proposals),
        "enqueue_receive_correlation": "UNKNOWN: legacy transport records "
        "omit request/correlation IDs",
        "work_end_semantics": "record timestamp after ready publication; not exact publication",
    }, proposals


def attach_events(raw, event_rows):
    for filename, events in event_rows.items():
        summary, index = event_summary(
            events, raw["measurement_start_ns"], raw["measurement_end_ns"]
        )
        raw.setdefault("event_evidence", {})[filename] = summary
        if filename == "draft-work-events.jsonl":
            matched = 0
            for step in raw["steps"]:
                evidence = []
                for row in step["identities"]:
                    value = index.get((row["request_id"], row["round_id"]))
                    if value is not None:
                        if value["prefix_length"] != row["prefix_length"]:
                            raise ValueError(
                                "owner result prefix differs from actual verification"
                            )
                        evidence.append({"request_id": row["request_id"], **value})
                        matched += 1
                step["owner_proposal_evidence"] = evidence
            summary["associated_verification_requests_in_retained_details"] = matched


def aggregate_clues(attempts):
    # Multiple repeats are left separate; never silently select a favorable attempt.
    by_mode = {}
    raw_by_mode = {}
    for a in attempts:
        s = a.get("retained_summary", {})
        if s.get("valid") is True and all(
            s.get(k) == "PASS"
            for k in ("execution_status", "measurement_status", "cleanup_status")
        ):
            by_mode.setdefault(a["mode"], []).append(s)
            raw_by_mode[a["mode"]] = a.get("raw", {})
    if any(len(v) != 1 for v in by_mode.values()):
        return {"status": "UNKNOWN", "reason": "multiple attempts; inspect each separately"}
    rows = {k: v[0] for k, v in by_mode.items()}
    result = {"status": "AGGREGATE_CLUES_ONLY", "critical_path_attribution": "UNKNOWN"}
    try:
        serial, ping = rows["serial"], rows["pingpong"]
        s = serial["pipeline_stage_gpu_event_ms"]
        p = ping["pipeline_stage_gpu_event_ms"]
        parts = {}
        for mode, values, batch_factor in (("serial", s, 1), ("pingpong", p, 2)):
            raw = raw_by_mode.get(mode, {})
            observed = raw.get("draft_model_gpu_event_ms_by_purpose", {})
            rotations = raw.get("rotations", {}).get("complete_count", 0)
            if not rotations or "commit" not in observed or "proposal" not in observed:
                result["missing"] = "Draft commit/proposal event evidence or complete rotations"
                result["incomplete_proposal_only_proxy_ms"] = {
                    "serial": s["D64"]["mean"] + s["V_SD64"]["mean"],
                    "pingpong": 2 * (p["D32"]["mean"] + p["V_SD32"]["mean"]),
                }
                return result
            parts[mode] = {
                "D_proposal": observed["proposal"]["sum_ms"] / rotations,
                "D_commit_or_prefix_sync": observed["commit"]["sum_ms"] / rotations,
                "V_target": batch_factor * values["V_SD" + str(64 // batch_factor)]["mean"],
                "other_recorded_Draft_GPU": sum(
                    v["sum_ms"] for k, v in observed.items() if k not in ("proposal", "commit")
                )
                / rotations,
            }
        sf, pf = (sum(parts[m].values()) for m in ("serial", "pingpong"))
        result["approximate_recorded_forward_ms_per_rotation"] = parts
        result["forward_semantics"] = (
            "launch-selected observed model forwards only; "
            "not matched initial-state stages, all GPU work, or critical path"
        )
        gap = ping["actual_rotation_ms"]["mean"] - serial["actual_rotation_ms"]["mean"]
        result.update(
            serial_quoted_forward_sum_ms=sf,
            pingpong_nonoverlap_quoted_forward_sum_ms=pf,
            rotation_gap_ms=gap,
            unattributed_arithmetic_gap_ms=gap - (pf - sf),
        )
        if "serial-split" in rows:
            split = rows["serial-split"]
            delta = split["window_ms"] - ping["window_ms"]
            wait = split["host_observation"]["wait_draft_owner"]["union_host_ms"]
            target = (
                split["host_observation"]["target_step"]["union_host_ms"]
                - ping["host_observation"]["target_step"]["union_host_ms"]
            )
            result["split_minus_pingpong"] = {
                "window_ms": delta,
                "split_wait_draft_owner_union_ms": wait,
                "target_step_union_difference_ms": target,
                "arithmetic_remainder_ms": delta - wait - target,
                "interpretation": "source confirms split pre-step owner wait; "
                "not proof of GPU time saved",
            }
    except (KeyError, TypeError):
        result["missing"] = "eligible Serial/PingPong stage means or rotation aggregates"
    return result


def write_outputs(output, report):
    report["aggregate_clues"] = aggregate_clues(report["attempts"])
    if len(json.dumps(report).encode()) > 3 * 1024 * 1024:
        # Keep aggregate counts/quality and completed attempts before dropping detail.
        for attempt in report["attempts"]:
            raw = attempt.get("raw", {})
            if raw:
                raw["steps"] = []
                raw["step_details_truncated"] = True
                raw["owner_during_target_pre_model"] = []
                raw["detail_limit_reason"] = "global output budget; aggregate counts retained"
    atomic_json(output / "attribution.json", report)
    fields = [
        "attempt",
        "mode",
        "execution_status",
        "measurement_status",
        "window_ms",
        "window_throughput_tok_s",
        "committed_window_tokens",
        "reported_rotation_mean_ms",
        "independent_overlap_status",
    ]
    temporary = output / "summary.csv.tmp"
    lines = [
        "# Fixed timing attribution (CPU, retained evidence)",
        "",
        "Analysis does not rerun or qualify GPU execution. Old artifacts remain unchanged.",
        "",
        "| Mode | Window ms | tok/s | Reported rotation ms | Independent overlap |",
        "|---|---:|---:|---:|---|",
    ]
    with temporary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for a in report["attempts"]:
            s = a.get("retained_summary", {})
            rotation = s.get("actual_rotation_ms", {}).get("mean")
            independent = a.get("raw", {}).get("overlap", {}).get("status", "UNKNOWN")
            writer.writerow(
                {
                    "attempt": a["artifact_directory"],
                    "mode": a["mode"],
                    **{k: s.get(k) for k in fields[2:7]},
                    "reported_rotation_mean_ms": rotation,
                    "independent_overlap_status": independent,
                }
            )
            lines.append(
                f"| {a['mode']} | {s.get('window_ms')} | {s.get('window_throughput_tok_s')} "
                f"| {rotation} | {independent} |"
            )
            purpose = a.get("raw", {}).get("draft_model_gpu_event_ms_by_purpose", {})
            lines.append(
                "Draft model-forward sums by purpose (ms): "
                + json.dumps({k: v["sum_ms"] for k, v in purpose.items()}, sort_keys=True)
            )
            lines.extend(
                [
                    "",
                    f"{a['artifact_directory']}: {a['evidence_status']}. "
                    + "; ".join(a.get("gaps", [])),
                ]
            )
    temporary.replace(output / "summary.csv")
    lines.extend(
        [
            "",
            "Host categories are nested inclusive observations; "
            "never sum them as a critical path.",
            "TP CUDA times are separate/max or interval union, never rank sums. "
            "Bounds are CUDA-event overlap, not exact kernel overlap.",
            "Legacy RPC receive/enqueue correlation and timer thread/parent IDs are absent. "
            "This CPU analysis makes no GPU performance improvement claim.",
            "D32/D64 denote proposal-only forwards. Commit/prefix-sync and other recorded "
            "model forwards are separate; 2*max(D_proposal32,V32) "
            "is not an end-to-end prediction.",
            "See attribution.json for counts, interval unions, purpose-separated Draft work, "
            "associations and precise missing evidence.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")


def worker(args):
    output = Path(args.output)
    progress = Progress(output)
    report = {
        "schema_version": "specrhythm.fixed-attribution.v1",
        "status": "RUNNING",
        "input": str(Path(args.input).absolute()),
        "cpu_only": True,
        "full_audit_run": False,
        "formal_comparison_eligible": False,
        "runtime_modified": False,
        "cross_run_equality": "NOT_REQUIRED",
        "attempts": [],
        "errors": [],
        "analyzer_source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                "fixed_attribution.py",
                "fixed_attribution_io.py",
                "fixed_attribution_cli.py",
            )
        },
    }
    write_outputs(output, report)
    sources = None
    exported, budget = [], [0]
    try:
        sources = Inputs(args.input, args.modes, progress)
        groups = {}
        for name in sources.names:
            groups.setdefault(str(PurePosixPath(name).parent), []).append(name)
        if not groups:
            raise ValueError("no allowlisted continuous attempt artifacts found")
        for directory, names in groups.items():
            attempt = {
                "mode": mode_from_path(directory),
                "artifact_directory": directory,
                "evidence_status": "INSUFFICIENT",
                "gaps": [],
            }
            report["attempts"].append(attempt)
            artifacts = {}
            for name in sorted(names, key=lambda n: (not n.endswith("light-summary.json"), n)):
                filename = PurePosixPath(name).name.removesuffix(".gz")
                if filename in artifacts:
                    raise ValueError("ambiguous duplicate raw/compressed artifact: " + name)
                value = project(name, sources.read(name))
                artifacts[filename] = value
                if filename == "light-summary.json":
                    attempt["retained_summary"] = value
                if args.export:
                    exported.append(export_file(output, name, value, budget))
                    atomic_json(
                        output / "export-manifest.json",
                        {
                            "status": "RUNNING",
                            "sources": sources.digests,
                            "exports": exported,
                            "semantics": "allowlisted projections; "
                            "original record hashes label source records",
                        },
                    )
                write_outputs(output, report)
            missing = [
                n for n in ("runtime.json", "draft-backend-report.json") if n not in artifacts
            ]
            attempt["gaps"] = [f"missing {directory}/{n}" for n in missing]
            if not missing and not args.export:
                progress(phase="indexed interval analysis", current_file=directory)
                try:
                    raw = analyze_raw(
                        artifacts["runtime.json"], artifacts["draft-backend-report.json"]
                    )
                    attach_events(
                        raw, {k: v for k, v in artifacts.items() if k.endswith(".jsonl")}
                    )
                    attempt["raw"] = raw
                    attempt["evidence_status"] = raw["status"]
                except (KeyError, TypeError, ValueError) as error:
                    attempt["gaps"].append("raw association/clock evidence: " + str(error))
            for filename in ("draft-work-events.jsonl", "draft-transport.jsonl"):
                if filename not in artifacts:
                    attempt["gaps"].append(
                        "missing optional timing evidence: " + directory + "/" + filename
                    )
            if args.export:
                attempt["evidence_status"] = "EXPORTED" if not missing else "INSUFFICIENT"
            report["source_artifacts"] = sources.digests
            write_outputs(output, report)
        report["status"] = "COMPLETE"
        if args.export:
            atomic_json(
                output / "export-manifest.json",
                {
                    "status": "COMPLETE",
                    "sources": sources.digests,
                    "exports": exported,
                    "compressed_bytes": budget[0],
                    "source_root_unchanged": True,
                    "full_audit_run": False,
                },
            )
        write_outputs(output, report)
        progress(phase="complete")
        for attempt in report["attempts"]:
            print(
                f"[fixed-attribution CPU] {attempt['artifact_directory']}: "
                f"{attempt['evidence_status']}",
                flush=True,
            )
            for gap in attempt["gaps"]:
                print(f"  {gap}", flush=True)
        print(f"[fixed-attribution CPU] report: {output / 'report.md'}", flush=True)
        return 0
    except Exception as error:
        report["status"] = "FAILED"
        report["errors"].append(f"{type(error).__name__}: {error}")
        write_outputs(output, report)
        print(report["errors"][-1], file=sys.stderr, flush=True)
        return 2
    finally:
        if sources:
            sources.close()


def supervise(command, output, timeout):
    """Parent remains able to terminate blocked parse/sort/write; timeout is shared, not reset."""
    started = time.monotonic()
    child = subprocess.Popen(command, start_new_session=True)
    reason, code = None, None
    try:
        code = child.wait(timeout=timeout)
        if code:
            reason = "analysis child exited with code " + str(code)
    except subprocess.TimeoutExpired:
        reason, code = "offline analysis exceeded the single external deadline", 124
    except KeyboardInterrupt:
        reason, code = "operator stopped offline analysis", 130
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        for pattern in ("*.part", "*.tmp"):
            for path in output.rglob(pattern):
                path.unlink()
    if reason:
        path = output / "attribution.json"
        report = json.loads(path.read_text()) if path.exists() else {"attempts": [], "errors": []}
        report["status"] = "TIMEOUT" if code == 124 else "STOPPED" if code == 130 else "FAILED"
        report["errors"].append(reason)
        report["partial_results_preserved"] = True
        write_outputs(output, report)
        print(f"[fixed-attribution CPU] {reason}; partial report: {path}", file=sys.stderr)
    # The final wrapper result is authoritative even if the worker was killed during I/O.
    atomic_json(
        output / "analysis-exit.json",
        {
            "return_code": code,
            "reason": reason,
            "elapsed_seconds": time.monotonic() - started,
            "timeout_seconds": timeout,
            "gpu_run": False,
            "full_audit_run": False,
        },
    )
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="retained root, exported evidence root, or small tar.gz"
    )
    parser.add_argument(
        "--output", required=True, help="new separate output directory (never the source root)"
    )
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument(
        "--export", action="store_true", help="project only allowlisted retained timing evidence"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not math_is_positive(args.timeout):
        parser.error("timeout must be finite and positive")
    if args.worker:
        return worker(args)
    output, source = Path(args.output).resolve(), Path(args.input).resolve()
    if output == source or source in output.parents:
        parser.error("output must be outside the immutable source root")
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(
        output / "attribution.json",
        {
            "status": "STARTING",
            "attempts": [],
            "errors": [],
            "cpu_only": True,
            "formal_comparison_eligible": False,
        },
    )
    command = [
        sys.executable,
        "-m",
        __spec__.name if __spec__ else "specrhythm.serving.fixed_attribution_cli",
        "--worker",
        "--input",
        str(source),
        "--output",
        str(output),
        "--timeout",
        str(args.timeout),
        "--modes",
        *args.modes,
    ]
    if args.export:
        command.append("--export")
    return supervise(command, output, args.timeout)


def math_is_positive(value):
    import math

    return math.isfinite(value) and value > 0


if __name__ == "__main__":
    raise SystemExit(main())
