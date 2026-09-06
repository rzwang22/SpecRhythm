from __future__ import annotations

import json
import shutil
from copy import deepcopy

import pytest
from test_phase4_draft_production_comparison import append_rows
from test_phase4_dual_microbatch_sweep import evidence as _evidence
from test_phase4_dual_microbatch_sweep import sweep as _sweep

from specrhythm.phase4.dual_rhythm import balanced_assignment
from specrhythm.phase4.pingpong_comparison import (
    CELLS,
    compare,
    pingpong_metrics,
    qualify_control,
    render,
    validate_cohort_execution,
)
from specrhythm.phase4.serial import token_prefix_hash

evidence = _evidence
sweep = _sweep


def execution():
    """CPU fixture of two initial proposals and A/B/A/B with opposite-stage overlap."""
    assignment = balanced_assignment("ab")
    cycles, works, drafts, verifies, commits = [], [], [], [], []
    prefix = {"a": [1, 10], "b": [2, 10]}
    for rid in "ab":
        drafts.append(
            {
                "operation": "initialize",
                "request_id": rid,
                "result": {
                    "prefix_version": 1,
                    "committed_prefix_hash": token_prefix_hash(prefix[rid]),
                },
            }
        )
    proposals = {}

    def proposed(rid, round_id):
        p = {
            "proposal_id": f"{rid}{round_id}",
            "request_id": rid,
            "round_id": round_id,
            "prefix_version": round_id + 1,
            "prefix_token_sha256": token_prefix_hash(prefix[rid]),
            "proposal_token_ids": [20, 21],
        }
        proposals[(rid, round_id)] = p
        drafts.append(
            {
                "operation": "propose_only" if round_id == 0 else "commit_and_propose",
                "request_id": rid,
                "result": {"logical_cohort": assignment[rid], "proposal": p},
            }
        )
        return p

    for rid, start, end in (("a", 10, 20), ("b", 21, 30)):
        works.append(
            {
                "logical_cohort": assignment[rid],
                "operation": "propose_only",
                "host_start_ns": start,
                "host_end_ns": end,
                "rows": [
                    {
                        "request_id": rid,
                        "logical_cohort": assignment[rid],
                        "committed_token_ids": list(prefix[rid]),
                        "prefix_version": 1,
                        "prefix_token_sha256": token_prefix_hash(prefix[rid]),
                    }
                ],
            }
        )
        proposed(rid, 0)
    for index, (rid, rnd, vs, ve, ce, ds, de) in enumerate(
        (
            ("a", 0, 40, 50, 55, 60, 80),
            ("b", 0, 65, 75, 85, 90, 110),
            ("a", 1, 95, 105, 115, 120, 125),
            ("b", 1, 130, 140, 145, 150, 155),
        )
    ):
        p = proposals[(rid, rnd)]
        active = ["b"] if index == 3 else list("ab")
        cycles.append(
            {
                "dual_rhythm": "pingpong",
                "logical_cohort": assignment[rid],
                "active_request_ids": active,
                "active_cohort_sizes": {
                    c: sum(assignment[r] == c for r in active) for c in ("A", "B")
                },
                "attained_cohort_request_ids": [rid],
                "eligible_cohort_request_ids": [rid],
                "initial_ready_request_ids": list("ab"),
                "ready_proposal_ids": [p["proposal_id"]],
            }
        )
        verifies.append(
            {
                **p,
                "logical_cohort": assignment[rid],
                "verify_request_ids": [rid],
                "verify_host_start_ns": vs,
                "verify_host_end_ns": ve,
                "target_rank_intervals": [{"host_start_ns": vs, "host_end_ns": ve}],
            }
        )
        commits.append(
            {**p, "committed_token_ids": [99], "terminal": rnd == 1, "commit_end_ns": ce}
        )
        prefix[rid].append(99)
        works.append(
            {
                "logical_cohort": assignment[rid],
                "operation": "commit_and_propose",
                "host_start_ns": ds,
                "host_end_ns": de,
                "rows": [
                    {
                        "request_id": rid,
                        "logical_cohort": assignment[rid],
                        "proposal_id": p["proposal_id"],
                        "round_id": rnd,
                        "prefix_version": rnd + 2,
                        "committed_delta": [99],
                        "terminal": rnd == 1,
                        "committed_token_ids": list(prefix[rid]),
                        "prefix_token_sha256": token_prefix_hash(prefix[rid]),
                    }
                ],
            }
        )
        if rnd == 0:
            proposed(rid, 1)
    return assignment, cycles, works, drafts, verifies, commits, 1


def test_cpu_cross_cohort_execution_validates_all_prefix_dependencies():
    assert len(validate_cohort_execution(*execution())) == 4


@pytest.mark.parametrize(
    "mutation",
    [
        "assignment",
        "mixed",
        "alternation",
        "fill",
        "proposal",
        "no-commit",
        "version",
        "hash",
        "uncommitted-prefix",
        "early-draft",
        "same-request",
        "same-cohort",
        "duplicate-verify",
    ],
)
def test_true_cohort_dependency_failures_block(mutation):
    a, c, w, d, v, p, b = execution()
    if mutation == "assignment":
        w[2]["rows"][0]["logical_cohort"] = "B"
    elif mutation == "mixed":
        c[0]["attained_cohort_request_ids"] = ["a", "b"]
    elif mutation == "alternation":
        c[1], c[2] = c[2], c[1]
    elif mutation == "fill":
        w[1]["host_end_ns"] = 45
    elif mutation == "proposal":
        v[0]["proposal_id"] = "not-scheduled"
    elif mutation == "no-commit":
        p.pop()
    elif mutation == "version":
        w[2]["rows"][0]["prefix_version"] = 1
    elif mutation == "hash":
        w[2]["rows"][0]["prefix_token_sha256"] = "0" * 64
    elif mutation == "uncommitted-prefix":
        w[2]["rows"][0]["committed_delta"] = [888]
    elif mutation == "early-draft":
        w[2]["host_start_ns"] = 54
    elif mutation in ("same-request", "same-cohort"):
        # Extend a prior authorized Draft execution into its own next verification.
        w[2]["host_end_ns"] = 101
    else:
        v.append(deepcopy(v[0]))
    with pytest.raises(ValueError):
        validate_cohort_execution(a, c, w, d, v, p, b)


def test_nonessential_schema_metadata_is_not_a_gate():
    args = execution()
    for work in args[2]:
        work.update(errors=None, historical_provenance=None, jit_warning="diagnostic")
    assert validate_cohort_execution(*args)


def metrics_fixture(directory):
    """Synthetic CPU clocks; no GPU timing claim. Exercise the entire artifact join."""
    from specrhythm.phase4.dual_runner import build_cycle_and_overlap_events

    a, c, w, d, v, p, boundary = execution()
    for i, row in enumerate(c):
        row.update(
            cycle_id=i,
            cohort_imbalance=abs(
                row["active_cohort_sizes"]["A"] - row["active_cohort_sizes"]["B"]
            ),
            capacity_clipped=False,
            poll_start_ns=v[i]["verify_host_start_ns"] - 2,
            poll_end_ns=v[i]["verify_host_start_ns"] - 1,
            target_waiting_for_draft=False,
        )
    for i, row in enumerate(v):
        row["verify_microbatch_id"] = str(i)
        row["target_physical_gpu_ids"] = [1, 2]
        row["target_rank_intervals"] = [
            {
                "request_id": row["request_id"],
                "tp_rank": rank,
                "global_rank": rank,
                "physical_gpu_id": rank + 1,
                "gpu_uuid": f"GPU-{rank}",
                "logical_cuda_index": rank,
                "host_start_ns": row["verify_host_start_ns"],
                "host_end_ns": row["verify_host_end_ns"],
                "cuda_elapsed_ns": row["verify_host_end_ns"] - row["verify_host_start_ns"],
                "cuda_events": True,
                "cuda_synchronized": True,
            }
            for rank in (0, 1)
        ]
    for row in d:
        proposal = row["result"].get("proposal")
        if proposal:
            work = next(
                work
                for work in w
                if work["rows"][0]["request_id"] == row["request_id"]
                and work["rows"][0]["prefix_version"] == proposal["prefix_version"]
            )
            row["result"]["draft_gpu_interval"] = {
                "host_start_ns": work["host_start_ns"],
                "host_end_ns": work["host_end_ns"],
                "cuda_elapsed_ns": work["host_end_ns"] - work["host_start_ns"],
                "physical_gpu_id": 0,
                "cuda_events": True,
                "cuda_synchronized": True,
            }
    for work in w:
        proposal = work["rows"][0].get("terminal") is not True
        work["forward_batch_histograms"] = {
            "proposal": {"1": 1} if proposal else {},
            "commit": {} if work["operation"] == "propose_only" else {"1": 1},
        }
    states, lifecycle = [], []
    for rid in "ab":
        path = [
            "DRAFT_READY",
            "DRAFTING",
            "PROPOSAL_READY",
            "VERIFY_READY",
            "VERIFYING",
            "COMMITTING",
            "DRAFT_SYNC",
            "DRAFT_READY",
            "DRAFTING",
            "PROPOSAL_READY",
            "VERIFY_READY",
            "VERIFYING",
            "COMMITTING",
            "TERMINAL",
        ]
        source = "BOOTSTRAP"
        for index, destination in enumerate(path):
            timestamp = index + 1 if index < len(path) - 1 else (116 if rid == "a" else 146)
            states.append(
                {
                    "request_id": rid,
                    "internal_request_id": "opaque-" + rid,
                    "source_state": source,
                    "destination_state": destination,
                    "timestamp_ns": timestamp,
                    "prefix_version": 1 if index < 5 else 2,
                    "committed_prefix_sha256": "frozen",
                    "reason": "CPU test",
                }
            )
            source = destination
    for proposal in p:
        for index, state in enumerate(("CREATED", "PUBLISHED", "INSTALLED", "CONSUMED")):
            lifecycle.append({**proposal, "lifecycle_state": state, "timestamp_ns": index + 1})
    _, overlaps = build_cycle_and_overlap_events(d, v)
    target = [
        {
            "request_id": row["request_id"],
            "target_forward_start_ns": row["verify_host_start_ns"],
            "target_forward_end_ns": row["verify_host_end_ns"],
        }
        for row in v
    ]
    records = {
        "scheduler-events": c,
        "draft-work-events": d,
        "verification-events": v,
        "proposal-events": p,
        "request-state-events": states,
        "proposal-lifecycle-events": lifecycle,
        "overlap-events": overlaps,
        "target-diagnostics": target,
        "timing-events": [],
    }
    for name, rows in records.items():
        append_rows(directory / (name + ".jsonl"), rows)
    (directory / "draft-backend-report.json").write_text(
        json.dumps(
            {
                "dual_rhythm": "pingpong",
                "assignment": dict(a),
                "pingpong_work_records": w,
                "draft_model_forward_count_by_purpose": {"proposal": 4, "commit": 4},
            }
        )
    )
    return a, boundary, 147


def test_full_metrics_artifact_join_counts_actual_intersections_once(tmp_path):
    assignment, boundary, end = metrics_fixture(tmp_path)
    result = pingpong_metrics(tmp_path, {}, assignment, boundary, end)
    assert result["cycle_count"] == 4 and result["cohort_switch_count"] == 3
    assert result["initial_cohort_sizes"] == {"A": 1, "B": 1}
    assert result["pipeline"]["physical_overlap_valid"]
    assert result["pipeline"]["overlap_interval_count"] == 2
    assert result["pipeline"]["observed_overlap_ms"] == 20 / 1e6
    assert result["pipeline"]["pipeline_fill_ms"] == 39 / 1e6
    assert result["pipeline"]["pipeline_drain_ms"] == 31 / 1e6
    assert result["pipeline"]["cycles_with_both_stages_active"] == 2
    assert result["pipeline"]["cycles_with_only_one_stage_active"] == 2
    assert result["pipeline"]["draft_waiting_for_target_ms"] is None
    assert result["draft_by_cohort"]["A"]["proposal_forwards"] == 2
    assert result["target_by_cohort"]["B"]["target_forward_count"] == 2
    assert len(result["invariants"]) == 10


def test_cohort_forward_accounting_cannot_invent_batching(tmp_path):
    assignment, boundary, end = metrics_fixture(tmp_path)
    path = tmp_path / "draft-backend-report.json"
    value = json.loads(path.read_text())
    value["pingpong_work_records"][0]["forward_batch_histograms"]["proposal"] = {"50": 4}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="forward accounting"):
        pingpong_metrics(tmp_path, {}, assignment, boundary, end)


@pytest.fixture
def five_cells(sweep):
    root, workload = sweep
    for name in CELLS[:-1]:
        mode = name if name in ("target", "serial") else "dual"
        size = int(name[7:]) if mode == "dual" else None
        value = qualify_control(root / name, mode, workload, size)
        assert value["valid"], value["errors"]
        (root / name / "qualification.json").write_text(json.dumps(value))
    # Comparison fixture only: policy execution itself is exercised above and in runtime tests.
    shutil.copytree(root / "dual-mb100", root / "pingpong")
    value = json.loads((root / "pingpong/qualification.json").read_text())
    value.update(dual_rhythm="pingpong")
    (root / "pingpong/qualification.json").write_text(json.dumps(value))
    return root


def test_five_cell_table_and_ratios_no_speed_gate(five_cells):
    result = compare(five_cells)
    assert result["valid"], result["errors"]
    assert len(result["ratios"]) == 4
    assert "Target | Serial | Dual-mb2 | Dual-mb100 | PingPong" in render(result)
    assert "Target batch p50" in render(result)
    assert result["pure_batching_speedup_claim"] is False
    assert result["interpretation_case"] == "P2"


@pytest.mark.parametrize(
    "mutation", ["count", "tokens", "commit", "backend", "mode", "size", "rhythm", "invalid"]
)
def test_material_formal_comparison_failures_block(five_cells, mutation):
    path = five_cells / "dual-mb2/qualification.json"
    if mutation == "rhythm":
        path = five_cells / "pingpong/qualification.json"
    value = json.loads(path.read_text())
    if mutation == "count":
        value["metrics"]["completed_requests"] = 99
    elif mutation == "tokens":
        value["metrics"]["measured_committed_tokens"] -= 1
    elif mutation == "commit":
        value["experiment_identity"]["execution_git_commit"] = "another-commit"
    elif mutation == "backend":
        value["experiment_identity"]["models"] = "different models"
    elif mutation == "mode":
        value["mode"] = "serial"
    elif mutation == "size":
        value["effective_dual_microbatch_size"] = 100
    elif mutation == "rhythm":
        value["dual_rhythm"] = "legacy"
    else:
        value["valid"] = False
    path.write_text(json.dumps(value))
    assert not compare(five_cells)["valid"]


def test_two_singleton_cohort_smoke_exception_does_not_change_d6_default(evidence):
    from specrhythm.phase4.draft_dual_comparison import summarize_run
    from specrhythm.phase4.manifest import sha256_file

    root, workload = evidence
    directory = root / "dual"
    raw_path, backend_path = (
        directory / "resident-dual.json",
        directory / "draft-backend-report.json",
    )
    raw, backend = json.loads(raw_path.read_text()), json.loads(backend_path.read_text())
    raw.update(request_count=2, outputs=raw["outputs"][:2])
    backend["draft_retired_request_count"] = 2
    backend["draft_batch_statistics_by_purpose"]["proposal"]["histogram"] = {"1": 2}
    backend_path.write_text(json.dumps(backend))
    raw["draft_shutdown"]["draft_backend_report_sha256"] = sha256_file(backend_path)
    raw_path.write_text(json.dumps(raw))
    assert not summarize_run(directory, "dual", 2, workload, smoke=True)["valid"]
    assert summarize_run(directory, "dual", 2, workload, smoke=True, singleton_cohort_smoke=True)[
        "valid"
    ]
    assert not summarize_run(directory, "dual", 2, workload, singleton_cohort_smoke=True)["valid"]
