"""Offline material qualification and five-cell MineDraft-style characterization."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from specrhythm.phase4.batched_draft_service import write_immutable_report
from specrhythm.phase4.draft_dual_comparison import _target_work, read, require, summarize_run
from specrhythm.phase4.draft_metrics import batch_statistics
from specrhythm.phase4.dual_correctness import (
    validate_proposal_lifecycle_events,
    validate_request_state_events,
)
from specrhythm.phase4.dual_microbatch_sweep import summarize_cell
from specrhythm.phase4.dual_overlap_characterization import characterize_overlap
from specrhythm.phase4.dual_rhythm import COHORTS, cohort_for, load_assignment
from specrhythm.phase4.manifest import sha256_file
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.stock_vllm import load_smoke_requests
from specrhythm.phase4.transport import CheckpointJsonl

CELLS = ("target", "serial", "dual-mb2", "dual-mb100", "pingpong")
TABLE = (
    ("Throughput (tok/s)", "throughput_tokens_per_second"),
    ("Makespan (ms)", "decode_makespan_ms"),
    ("TPOT mean (ms)", "tpot_ms.mean"),
    ("Target forwards", "target_forward_count"),
    ("Target batch p50", "target_batch_size.p50"),
    ("Draft forwards", "draft_model_forward_count"),
    ("Draft batch p50", "draft_batch_size.p50"),
    ("Draft GPU time (ms)", "draft_gpu_event_time_ms"),
    ("Accepted length", "mean_accepted_length"),
    ("Overlap (ms)", "observed_overlap_ms"),
)


def jsonl(directory, name):
    return CheckpointJsonl(directory / (name + ".jsonl")).read()


def target_batches(rows, boundary):
    batches = defaultdict(set)
    for row in rows:
        if row["target_forward_start_ns"] >= boundary:
            batches[(row["target_forward_start_ns"], row["target_forward_end_ns"])].add(
                row["request_id"]
            )
    return batches


def union_intervals(intervals):
    merged = []
    for a, b in sorted(set(intervals)):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    return merged


def validate_cohort_execution(assignment, cycles, works, drafts, verifies, commits, boundary):
    """Join new policy evidence to actual work; no numerical-equality gate."""
    require(bool(cycles) and bool(works) and bool(verifies), "missing pingpong execution evidence")
    scheduled = {}
    selected_cycles = []
    previous = None
    for cycle in cycles:
        require(cycle.get("dual_rhythm") == "pingpong", "mixed scheduler rhythm")
        active = cycle["active_request_ids"]
        require(
            len(set(active)) == len(active) and set(active) <= set(assignment),
            "invalid active membership",
        )
        sizes = {c: sum(assignment[r] == c for r in active) for c in COHORTS}
        require(sizes == cycle["active_cohort_sizes"], "active cohort counts disagree")
        ids = cycle["attained_cohort_request_ids"]
        if not ids:
            continue
        c = cohort_for(assignment, ids)
        require(c == cycle["logical_cohort"], "Target scheduled outside selected cohort")
        require(
            set(ids) <= set(cycle["eligible_cohort_request_ids"]),
            "Target scheduled ineligible cohort member",
        )
        require(
            set(active) <= set(cycle["initial_ready_request_ids"]),
            "Target started before initial pipeline fill",
        )
        require(
            previous is None or c != previous or sizes["B" if c == "A" else "A"] == 0,
            "cohort alternation violated while both groups remain active",
        )
        for pid in cycle["ready_proposal_ids"]:
            require(pid not in scheduled, "one proposal scheduled twice")
            scheduled[pid] = c
        previous = c
        selected_cycles.append(cycle)
    require(
        {r["logical_cohort"] for r in selected_cycles} == set(COHORTS),
        "both cohorts must execute and alternate",
    )

    initializations = {
        r["request_id"]: r["result"] for r in drafts if r["operation"] == "initialize"
    }
    require(set(initializations) == set(assignment), "Draft setup membership differs")
    commit_by_id = {r["proposal_id"]: r for r in commits}
    require(len(commit_by_id) == len(commits), "proposal committed more than once")
    prefixes, versions, used, initial_groups = {}, {}, set(), set()
    for work in works:
        c = cohort_for(assignment, [r["request_id"] for r in work["rows"]])
        require(c == work["logical_cohort"], "Draft work changed cohort")
        require(
            boundary <= work["host_start_ns"] < work["host_end_ns"],
            "Draft work outside measured start",
        )
        for row in work["rows"]:
            rid = row["request_id"]
            require(row["logical_cohort"] == c, "Draft row changed stable membership")
            if work["operation"] == "propose_only":
                require(rid not in prefixes, "initial proposal submitted twice")
                init = initializations[rid]
                prefix = tuple(row["committed_token_ids"])
                require(
                    init["prefix_version"] == row["prefix_version"] == 1
                    and init["committed_prefix_hash"] == token_prefix_hash(prefix),
                    "initial proposal differs from authorized bootstrap prefix",
                )
                initial_groups.add(c)
            else:
                require(
                    rid in prefixes and row["prefix_version"] == versions[rid] + 1,
                    "next Draft prefix version is not authorized",
                )
                delta = row["committed_delta"]
                prefix = prefixes[rid] + tuple(delta)
                if work["operation"] == "commit_and_propose":
                    pid = row["proposal_id"]
                    require(
                        pid in commit_by_id and pid not in used,
                        "next Draft lacks a unique Target commit",
                    )
                    commit = commit_by_id[pid]
                    require(
                        commit["request_id"] == rid
                        and commit["prefix_version"] == versions[rid]
                        and commit["prefix_token_sha256"] == token_prefix_hash(prefixes[rid])
                        and commit["round_id"] == row["round_id"]
                        and commit["committed_token_ids"] == delta
                        and commit["terminal"] == row["terminal"]
                        and commit["commit_end_ns"] <= work["host_start_ns"],
                        "Draft prefix not authorized by previous Target commit",
                    )
                    used.add(pid)
                else:
                    require(
                        work["operation"] == "finish_tail"
                        and row["terminal"] is True
                        and len(delta) == 1,
                        "invalid terminal-tail Draft synchronization",
                    )
            require(
                row["prefix_token_sha256"] == token_prefix_hash(prefix),
                "Draft prefix hash mismatch",
            )
            if "committed_token_ids" in row:
                require(
                    tuple(row["committed_token_ids"]) == prefix, "Draft prefix tokens mismatch"
                )
            prefixes[rid], versions[rid] = prefix, row["prefix_version"]
    require(initial_groups == set(COHORTS), "missing initial proposals for one cohort")
    require(
        {
            assignment[d["request_id"]]
            for d in drafts
            if d["result"].get("proposal") and d["result"]["proposal"]["round_id"] == 0
        }
        == set(COHORTS),
        "both cohorts must publish initial proposals",
    )
    require(
        max(w["host_end_ns"] for w in works if w["operation"] == "propose_only")
        <= min(v["verify_host_start_ns"] for v in verifies),
        "initial pipeline fill incomplete",
    )
    require(
        used == set(commit_by_id),
        "Target commits and Draft synchronization do not cover each other",
    )

    verified = {}
    for row in verifies:
        c = cohort_for(assignment, row["verify_request_ids"])
        require(
            row["logical_cohort"] == c and scheduled.get(row["proposal_id"]) == c,
            "Target verifies a proposal outside selected cohort",
        )
        pid = row["proposal_id"]
        require(pid not in verified, "proposal verified more than once")
        verified[pid] = row
    require(
        set(verified) == set(commit_by_id) == set(scheduled),
        "scheduled/verified/committed proposal coverage differs",
    )
    for event in drafts:
        result = event["result"]
        if event["operation"] == "initialize":
            continue
        require(
            result.get("logical_cohort") == assignment[event["request_id"]],
            "published Draft membership differs",
        )
        proposal = result.get("proposal")
        if proposal and proposal["proposal_id"] in verified:
            v = verified[proposal["proposal_id"]]
            require(
                proposal["request_id"] == v["request_id"]
                and proposal["prefix_version"] == v["prefix_version"]
                and proposal["round_id"] == v["round_id"]
                and proposal["proposal_token_ids"] == v["proposal_token_ids"],
                "published/verified proposal identity differs",
            )
    # Host work includes commit materialization as well as proposal GPU work.
    # A valid schedule cannot execute either against Target on the same cohort.
    for work in works:
        for v in verifies:
            start = min(r["host_start_ns"] for r in v["target_rank_intervals"])
            end = max(r["host_end_ns"] for r in v["target_rank_intervals"])
            if work["host_start_ns"] < end and work["host_end_ns"] > start:
                require(
                    work["logical_cohort"] != v["logical_cohort"],
                    "same-cohort or same-request Draft/Target concurrency",
                )
    return selected_cycles


def pingpong_metrics(directory, raw, assignment, boundary, end):
    cycles = [r for r in jsonl(directory, "scheduler-events") if "cycle_id" in r]
    drafts, verifies = (
        jsonl(directory, "draft-work-events"),
        jsonl(directory, "verification-events"),
    )
    backend = read(directory / "draft-backend-report.json")
    works = backend["pingpong_work_records"]
    require(
        backend.get("dual_rhythm") == "pingpong" and backend["assignment"] == assignment,
        "Draft runtime policy/assignment differs",
    )
    selected = validate_cohort_execution(
        assignment, cycles, works, drafts, verifies, jsonl(directory, "proposal-events"), boundary
    )
    states = jsonl(directory, "request-state-events")
    require(not validate_request_state_events(states), "request lifecycle invalid")
    require(
        not validate_proposal_lifecycle_events(jsonl(directory, "proposal-lifecycle-events")),
        "proposal/version lifecycle invalid",
    )
    terminal = {r["request_id"]: r for r in states if r["destination_state"] == "TERMINAL"}
    require(set(terminal) == set(assignment), "requests did not all terminate")
    tail_commits = [
        r
        for r in jsonl(directory, "timing-events")
        if r.get("source") == "dual-proposal-free-target-tail-commit"
    ]
    for w in works:
        if w["operation"] == "finish_tail":
            for row in w["rows"]:
                state = terminal[row["request_id"]]
                require(
                    state["prefix_version"] == row["prefix_version"]
                    and state["committed_prefix_sha256"] == row["prefix_token_sha256"]
                    and state["timestamp_ns"] <= w["host_start_ns"]
                    and any(
                        t["request_id"] == row["request_id"]
                        and t["token_ids"] == row["committed_delta"]
                        and t["timestamp_ns"] <= w["host_start_ns"]
                        for t in tail_commits
                    ),
                    "terminal tail prefix lacks Target commit authorization",
                )
    overlap = characterize_overlap(
        drafts, verifies, jsonl(directory, "overlap-events"), raw.get("worker_ranks", ())
    )
    require(overlap["valid"], str(overlap["errors"]))
    physical = target_batches(jsonl(directory, "target-diagnostics"), boundary)
    # Include proposal-free Target tails in the no-concurrency check.
    for (a, b), ids in physical.items():
        cohort = cohort_for(assignment, ids)
        for w in works:
            if w["host_start_ns"] < b and w["host_end_ns"] > a:
                require(w["logical_cohort"] != cohort, "same-cohort Draft/Target tail concurrency")
    verify_batches = {r["verify_microbatch_id"]: r for r in verifies}
    intersections, concurrent_batches = [], set()
    for v in verifies:
        a, b = v["verify_host_start_ns"], v["verify_host_end_ns"]
        for d in drafts:
            result = d["result"]
            if not result.get("proposal"):
                continue
            i = result["draft_gpu_interval"]
            x, y = max(a, i["host_start_ns"]), min(b, i["host_end_ns"])
            if x < y:
                require(
                    assignment[d["request_id"]] != v["logical_cohort"],
                    "physical overlap is not cross-cohort",
                )
                intersections.append((x, y))
                concurrent_batches.add(v["verify_microbatch_id"])
    merged = union_intervals(intersections)
    draft_by_cohort = {}
    for c in COHORTS:
        histograms = {p: Counter() for p in ("proposal", "commit")}
        for w in works:
            if w["logical_cohort"] == c:
                for p, histogram in w["forward_batch_histograms"].items():
                    histograms[p].update({int(n): count for n, count in histogram.items()})
        draft_by_cohort[c] = {
            "proposal_batch_size": batch_statistics(histograms["proposal"]),
            "proposal_forwards": sum(histograms["proposal"].values()),
            "commit_forwards": sum(histograms["commit"].values()),
        }
    for p in ("proposal", "commit"):
        require(
            sum(d[p + "_forwards"] for d in draft_by_cohort.values())
            == backend["draft_model_forward_count_by_purpose"].get(p, 0),
            "cohort Draft forward accounting differs from actual backend",
        )
    drain_start = min(
        max(terminal[r]["timestamp_ns"] for r in assignment if assignment[r] == c) for c in COHORTS
    )
    waits = [
        r
        for r in cycles
        if r["target_waiting_for_draft"]
        and r["poll_end_ns"] > boundary
        and r["poll_start_ns"] < end
    ]
    return {
        "initial_cohort_sizes": {c: list(assignment.values()).count(c) for c in COHORTS},
        "active_sizes_over_cycles": [
            {
                "cycle_id": r["cycle_id"],
                **r["active_cohort_sizes"],
                "imbalance": r["cohort_imbalance"],
            }
            for r in selected
        ],
        "cohort_size_distribution": {
            c: batch_statistics(Counter(r["active_cohort_sizes"][c] for r in selected))
            for c in COHORTS
        },
        "cycle_count": len(selected),
        "scheduler_poll_count": len(cycles),
        "cohort_switch_count": sum(
            a["logical_cohort"] != b["logical_cohort"] for a, b in zip(selected, selected[1:])
        ),
        "target_by_cohort": {
            c: {
                "verification_batch_size": batch_statistics(
                    Counter(
                        len(v["verify_request_ids"])
                        for v in verify_batches.values()
                        if v["logical_cohort"] == c
                    )
                ),
                "target_forward_count": sum(
                    cohort_for(assignment, ids) == c for ids in physical.values()
                ),
                "singleton_batches": sum(
                    len(v["verify_request_ids"]) == 1
                    for v in verify_batches.values()
                    if v["logical_cohort"] == c
                ),
            }
            for c in COHORTS
        },
        "capacity_clipped_cohorts": [
            {
                k: r[k]
                for k in (
                    "cycle_id",
                    "logical_cohort",
                    "eligible_cohort_request_ids",
                    "attained_cohort_request_ids",
                    "dual_scheduler_constraints",
                    "capacity_constraint_attribution",
                )
            }
            for r in cycles
            if r["capacity_clipped"]
        ],
        "draft_by_cohort": draft_by_cohort,
        "pipeline": {
            **overlap,
            "overlap_interval_count": len(merged),
            "cross_cohort_intersection_count": len(set(intersections)),
            "overlap_fraction_of_makespan": sum(b - a for a, b in merged) / (end - boundary),
            "pipeline_fill_ms": (min(a for a, _ in physical) - boundary) / 1e6,
            "pipeline_drain_ms": max(0, end - drain_start) / 1e6,
            "fill_scope": (
                "measurement start to first Target forward; both initial proposals included"
            ),
            "drain_scope": (
                "first cohort fully terminal to measured end; "
                "terminal sync after end remains cleanup"
            ),
            "target_waiting_for_draft_count": len(waits),
            "target_waiting_for_draft_ms": sum(
                max(0, min(end, r["poll_end_ns"]) - max(boundary, r["poll_start_ns"]))
                for r in waits
            )
            / 1e6,
            "target_wait_scope": (
                "empty policy-wait scheduler calls only; excludes time between polls"
            ),
            "draft_waiting_for_target_count": None,
            "draft_waiting_for_target_ms": None,
            "draft_wait_scope": "unavailable: owner idle intervals are not instrumented",
            "cycles_with_both_stages_active": len(concurrent_batches),
            "cycles_with_only_one_stage_active": len(selected) - len(concurrent_batches),
            "cycle_activity_scope": (
                "Target decode units; both requires actual proposal CUDA overlap; fill is separate"
            ),
        },
        "invariants": {
            "exactly_one_cohort": True,
            "cohort_assignment_stable": True,
            "selected_cohort_verification": True,
            "opposite_cohort_during_overlap": True,
            "no_same_request_concurrency": True,
            "next_prefix_authorized_by_target_commit": True,
            "proposal_version_hash_valid": True,
            "eos_terminal_tail_valid": True,
            "retirement_frees_draft_kv": True,
            "all_requests_complete_once": True,
        },
    }


def qualify(directory, workload, count, *, smoke=False):
    result = summarize_run(
        directory,
        "dual",
        count,
        workload,
        smoke=smoke,
        characterization=True,
        singleton_cohort_smoke=smoke and count == 2,
    )
    if not result["valid"]:
        return result
    try:
        assignment = load_assignment(
            directory / "dual-rhythm.json", workload=workload, count=count
        )
        raw = read(directory / "resident-dual.json")
        result["execution_git_commit"] = read(directory / "runtime-manifest.json")["git_commit"]
        plugin = read(directory / "plugin-report.json")
        require(
            raw.get("dual_rhythm") == plugin.get("dual_rhythm") == "pingpong"
            and raw.get("cohort_assignment") == assignment,
            "runtime did not load pingpong assignment",
        )
        require(plugin.get("sampled_row_tp_consensus") is True, "sampled-row TP consensus failed")
        requests = {
            r.request_id: r
            for r in load_smoke_requests(workload, count, require_task_mixture=count in (5, 100))
        }
        outputs = raw["outputs"]
        require(
            len(outputs) == len({r["request_id"] for r in outputs}) == count
            and {r["request_id"] for r in outputs} == set(assignment),
            "request completion is not exactly once",
        )
        for row in outputs:
            tokens = row["generated_token_ids"]
            require(
                row["generated_tokens"] == len(tokens)
                and 0 < len(tokens) <= requests[row["request_id"]].maximum_new_tokens
                and row["finish_reason"] in ("length", "stop", "max_tokens"),
                "raw completion/token accounting invalid",
            )
        if smoke:
            boundary = plugin["performance_measurement_start_ns"]
            end = max(
                r["timestamp_ns"]
                for r in jsonl(directory, "timing-events")
                if r.get("event") == "measured-token-commit"
            )
            result["metrics"].update(_target_work(directory, boundary, set(assignment)))
        else:
            measurement = read(directory / "decode-performance.json")["measurement"]
            boundary, end = measurement["measurement_start_ns"], measurement["measurement_end_ns"]
        result["pingpong"] = pingpong_metrics(directory, raw, assignment, boundary, end)
        if count == 5:
            require(
                result["pingpong"]["pipeline"]["physical_overlap_valid"],
                "corrected-5 requires physical cross-cohort overlap",
            )
        if count == 100:
            require(
                all(r.maximum_new_tokens == 16 for r in requests.values())
                and result["metrics"]["measured_committed_tokens"] == 1487,
                "corrected-100 output limit/token accounting differs",
            )
        result["dual_rhythm"] = "pingpong"
        result["metrics"]["target_batch_size"] = batch_statistics(
            Counter(
                len(ids)
                for ids in target_batches(
                    jsonl(directory, "target-diagnostics"), boundary
                ).values()
            )
        )
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        result.update(valid=False, performance_result=False, errors=[str(error)])
    return result


def qualify_control(directory, mode, workload, size=None):
    result = summarize_cell(directory, mode, workload, size)
    if result["valid"]:
        boundary = read(directory / "decode-performance.json")["measurement"][
            "measurement_start_ns"
        ]
        result["metrics"]["target_batch_size"] = batch_statistics(
            Counter(
                len(ids)
                for ids in target_batches(
                    jsonl(directory, "target-diagnostics"), boundary
                ).values()
            )
        )
        if mode == "dual":
            require(
                read(directory / "resident-dual.json").get("dual_rhythm", "legacy") == "legacy",
                "legacy control executed another rhythm",
            )
    return result


def compare(root):
    cells, errors, ratios, work, outcome = {}, [], {}, {}, None
    try:
        preflight = read(root / "preflight.json")
        require(preflight.get("valid") is True and not preflight.get("errors"), "preflight failed")
        for name in CELLS:
            cell = cells[name] = read(root / name / "qualification.json")
            require(
                cell.get("valid") is True
                and not cell.get("errors")
                and cell.get("performance_result") is True,
                f"{name}: execution not qualified",
            )
            require(
                cell["input_sha256"]["performance"]
                == sha256_file(root / name / "decode-performance.json"),
                f"{name}: measured input changed",
            )
            require(
                cell["metrics"]["completed_requests"] == 100
                and cell["metrics"]["measured_committed_tokens"] == 1487,
                "request/token count differs",
            )
            require(
                cell["mode"] == (name if name in ("target", "serial") else "dual"),
                "mode identity differs",
            )
            if name.startswith("dual-"):
                from specrhythm.phase4.dual_microbatch import FIELDS

                require(
                    all(cell.get(k) == int(name[7:]) for k in FIELDS),
                    "legacy control size differs",
                )
            if name == "pingpong":
                require(
                    cell.get("dual_rhythm") == "pingpong",
                    "pingpong cell is not the isolated policy",
                )
        identities = [c["experiment_identity"] for c in cells.values()]
        require(
            all(i == identities[0] for i in identities)
            and identities[0]["execution_git_commit"] == preflight["execution_git_commit"],
            "formal cells must share server/commit/workload/model/config identity",
        )
        require(
            identities[0]["workload_sha256"]
            == preflight["inputs"]["SR_PHASE4B_WORKLOAD"]["sha256"]
            and identities[0]["config_sha256"] == preflight["config_sha256"],
            "preflight inputs differ",
        )
        m = {k: v["metrics"] for k, v in cells.items()}
        for name in CELLS[:-1]:
            ratios["pingpong_over_" + name + "_throughput"] = (
                m["pingpong"]["throughput_tokens_per_second"]
                / m[name]["throughput_tokens_per_second"]
            )
        for name in ("serial", "dual-mb2", "dual-mb100"):
            work[name] = {
                k: m["pingpong"][k] - m[name][k]
                for k in (
                    "measured_committed_tokens",
                    "draft_proposal_count",
                    "proposed_tokens",
                    "accepted_draft_tokens",
                    "rejected_draft_tokens",
                    "target_forward_count",
                    "target_query_tokens",
                )
            }
        ratio = ratios["pingpong_over_dual-mb100_throughput"]
        outcome = (
            "P2"
            if abs(ratio - 1) <= 0.05
            else "P1"
            if ratio > 1 and ratios["pingpong_over_serial_throughput"] >= 0.95
            else "P3"
            if ratio < 1 and ratios["pingpong_over_dual-mb2_throughput"] > 1
            else "P4"
        )
    except (OSError, ValueError, TypeError, KeyError, ZeroDivisionError) as error:
        errors.append(str(error))
    return {
        "valid": not errors,
        "errors": errors,
        "cells": cells,
        "ratios": ratios,
        "pingpong_minus_control_work": work,
        "interpretation_case": outcome,
        "interpretation": {
            "P1": (
                "Large persistent cohorts recover batching while retaining useful overlap; "
                "inspect the measured witnesses."
            ),
            "P2": (
                "Coarse batching dominates; this session shows little extra end-to-end "
                "benefit from overlap."
            ),
            "P3": (
                "Some batching is restored, but switching/dependency overhead still "
                "outweighs overlap relative to mb100."
            ),
            "P4": (
                "Inspect residual runtime and synchronization cost before adaptive policy design."
            ),
        }.get(outcome),
        "interpretation_scope": (
            "Descriptive one-session outcome, not causal proof; approximate/approaches "
            "means within 5%, never a validity gate."
        ),
        "label": "MineDraft-style large-cohort ping-pong baseline",
        "performance_label": "production vLLM Batched Draft end-to-end improvement",
        "pure_batching_speedup_claim": False,
        "overlap_is_critical_path_time_saved": False,
        "adaptive_policy_implemented": False,
    }


def render(report):
    lines = [
        f"Valid: {report['valid']}; errors: {report['errors']}",
        "",
        "Metric | Target | Serial | Dual-mb2 | Dual-mb100 | PingPong",
        "--- | " + " | ".join(["---:"] * 5),
    ]
    for label, key in TABLE:
        values = []
        for name in CELLS:
            value = report["cells"].get(name, {}).get("metrics", {})
            for part in key.split("."):
                value = value.get(part, "—") if isinstance(value, dict) else "—"
            values.append(f"{value:.6g}" if type(value) is float else str(value))
        lines.append(label + " | " + " | ".join(values))
    lines.extend(
        [
            "",
            *[f"- {k}: {v:.6g}" for k, v in report["ratios"].items()],
            "",
            f"{report['interpretation_case']}: {report['interpretation']}",
            report["interpretation_scope"],
            "",
            "Work differences: " + json.dumps(report["pingpong_minus_control_work"]),
            "No pure batching claim. Observed overlap is not critical-path time saved.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--run-root", type=Path, required=True)
    validate.add_argument("--workload", type=Path, required=True)
    validate.add_argument("--request-count", type=int, choices=(2, 5, 100), required=True)
    validate.add_argument("--smoke", action="store_true")
    control = sub.add_parser("control")
    control.add_argument("--run-root", type=Path, required=True)
    control.add_argument("--workload", type=Path, required=True)
    control.add_argument("--mode", choices=("target", "serial", "dual"), required=True)
    control.add_argument("--microbatch-size", type=int)
    comparison = sub.add_parser("compare")
    comparison.add_argument("--root", type=Path, required=True)
    comparison.add_argument("--markdown", type=Path, required=True)
    for p in (validate, control, comparison):
        p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "output must be fresh")
    if args.command == "compare":
        require(not args.markdown.exists(), "Markdown output must be fresh")
        result = compare(args.root)
        args.markdown.write_text(render(result))
        print(render(result))
    elif args.command == "control":
        result = qualify_control(args.run_root, args.mode, args.workload, args.microbatch_size)
    else:
        require(not args.smoke or args.request_count == 2, "smoke requires two requests")
        result = qualify(args.run_root, args.workload, args.request_count, smoke=args.smoke)
    write_immutable_report(args.output, result)
    print(json.dumps({k: result[k] for k in ("valid", "errors")}, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
