"""Phase-3C.2 common-prefix snapshots and sequential offline selector replay."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from specrhythm.phase3.engine import CausalLMBackend, create_backend
from specrhythm.phase3.phase3c_config import Phase3CConfig, load_frozen_pool_dimensions
from specrhythm.phase3.r3_workload import R3RealRequest
from specrhythm.phase3.real_candidate_trace import (
    ImmutableRequestStore,
    LabeledTraceRecord,
    TargetTrajectoryRecord,
    generate_real_candidate_forest,
    join_forest_and_target,
    target_store,
)
from specrhythm.phase3.selector_diagnosis import (
    ORACLE_SELECTOR,
    POOL_ORDER,
    SELECTOR_ORDER,
    TARGET_BLIND_SELECTORS,
    SelectionResult,
    _evaluate_selection,
    select_target_blind,
    select_within_request_oracle,
)
from specrhythm.phase3.trace import sha256_file

SNAPSHOT_SCHEMA = "specrhythm.phase3c-common-prefix-snapshot.v1"
SEQUENTIAL_SCHEMA = "specrhythm.phase3c-sequential-selector-request.v1"


def _canonical_sha(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def _directory_sha(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*.json") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _snapshot_identity(request_id: str, prefix_position: int) -> str:
    return f"{request_id}@prefix-{prefix_position:02d}"


def _parse_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(value)
    if copied.get("schema_version") != SNAPSHOT_SCHEMA:
        raise ValueError("unsupported common-prefix snapshot schema")
    required = {
        "request_id",
        "task_class",
        "prefix_position",
        "context_length",
        "remaining_target_tokens",
        "forest_sha256",
        "pool_sha256",
        "target_trajectory_sha256",
        "target_continuation",
        "verification_budget",
        "labeled_trace",
    }
    if not required.issubset(copied):
        raise ValueError("common-prefix snapshot is incomplete")
    labeled = LabeledTraceRecord.from_dict(copied["labeled_trace"])
    if labeled.request_id != copied["request_id"]:
        raise ValueError("snapshot/labeled request IDs differ")
    if copied["prefix_position"] < 0 or copied["remaining_target_tokens"] < 1:
        raise ValueError("snapshot prefix/remaining length is invalid")
    if copied["target_continuation"] != list(labeled.target_trajectory.target_token_ids):
        raise ValueError("snapshot continuation differs from its derived target suffix")
    if copied["forest_sha256"] != labeled.forest_sha256:
        raise ValueError("snapshot forest hash differs from labeled trace")
    for ratio in POOL_ORDER:
        expected = _canonical_sha({"node_ids": list(labeled.pool_node_ids[ratio])})
        if copied["pool_sha256"].get(ratio) != expected:
            raise ValueError("snapshot candidate-pool hash differs")
    return copied


def snapshot_store(path: Path) -> ImmutableRequestStore[dict[str, Any]]:
    return ImmutableRequestStore(path, _parse_snapshot, lambda value: value)


def _parse_sequential(value: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(value)
    if copied.get("schema_version") != SEQUENTIAL_SCHEMA:
        raise ValueError("unsupported sequential-selector schema")
    target = list(copied.get("target_token_ids", ()))
    if copied.get("target_token_count") != len(target) or not target:
        raise ValueError("sequential target-token accounting differs")
    results = copied.get("results", ())
    expected = {
        (ratio, selector) for ratio in POOL_ORDER for selector in SELECTOR_ORDER
    }
    actual = {
        (row.get("pool_ratio"), row.get("selector"))
        for row in results
        if isinstance(row, Mapping)
    }
    if actual != expected or len(results) != len(expected):
        raise ValueError("sequential selector matrix is incomplete or duplicated")
    for row in results:
        if row.get("final_committed_target_token_ids") != target:
            raise ValueError("sequential final token sequence differs from target")
        if row.get("final_sequence_matches_target") is not True:
            raise ValueError("sequential target equality flag is false")
        rounds = row.get("rounds", ())
        if not rounds or row.get("proposal_rounds") != len(rounds):
            raise ValueError("sequential proposal-round accounting differs")
        position = 0
        for round_row in rounds:
            if round_row.get("prefix_position") != position:
                raise ValueError("sequential prefix progression is discontinuous")
            accepted = round_row.get("accepted_tokens")
            committed = round_row.get("committed_tokens")
            verified = round_row.get("verified_nodes")
            if not all(isinstance(item, int) for item in (accepted, committed, verified)):
                raise ValueError("sequential token accounting must use integers")
            if (
                not 0 <= accepted <= verified <= 4
                or committed - accepted not in {0, 1}
            ):
                raise ValueError("sequential accepted/committed/verified accounting is invalid")
            position += committed
        if position != len(target):
            raise ValueError("sequential rounds do not commit the complete target")
    return copied


def sequential_store(path: Path) -> ImmutableRequestStore[dict[str, Any]]:
    return ImmutableRequestStore(path, _parse_sequential, lambda value: value)


def _suffix_target(
    target: TargetTrajectoryRecord, prefix_position: int, context_length: int
) -> TargetTrajectoryRecord:
    tokens = target.target_token_ids[prefix_position:]
    probabilities = target.target_log_probabilities[prefix_position:]
    eos = target.target_eos_position
    suffix_eos = eos - prefix_position if eos is not None and eos >= prefix_position else None
    return TargetTrajectoryRecord(
        request_id=target.request_id,
        workload_sha256=target.workload_sha256,
        prompt_length=context_length,
        tokenizer_fingerprint=target.tokenizer_fingerprint,
        target_token_ids=tokens,
        target_log_probabilities=probabilities,
        target_path_length=len(tokens),
        target_eos_position=suffix_eos,
        target_model=target.target_model,
        target_model_revision=target.target_model_revision,
        target_forward_count=len(tokens),
        greedy_decoding=True,
        kv_cache_reuse=False,
    )


def _model_revision(configured: Optional[str], model_id: str) -> str:
    if configured:
        return configured
    path = Path(model_id) / "config.json"
    return f"local-config-sha256:{sha256_file(path)}" if path.is_file() else "unversioned"


def run_common_prefix_snapshot_stage(
    requests: Iterable[R3RealRequest],
    config: Phase3CConfig,
    *,
    workload_path: Path,
    target_dir: Path,
    output_dir: Path,
    resume: bool,
    backend: Optional[CausalLMBackend] = None,
) -> dict[str, Any]:
    """Build each forest once at each frozen target-prefix position."""

    request_list = list(requests)
    targets = target_store(target_dir)
    output = snapshot_store(output_dir)
    if not resume and output.records():
        raise FileExistsError("snapshot output is non-empty; pass --resume to continue")
    dimensions = load_frozen_pool_dimensions(config)
    workload_sha = sha256_file(workload_path)
    owns_backend = backend is None
    backend = backend or create_backend(
        config.runtime.backend, config.runtime.draft, config.runtime.random_seed
    )
    written = 0
    try:
        for request in request_list:
            if not targets.has(request.request_id):
                raise ValueError(f"missing frozen target trajectory for {request.request_id}")
            target = targets.read(request.request_id)
            if target.workload_sha256 != workload_sha:
                raise ValueError(f"target/workload identity differs for {request.request_id}")
            if backend.tokenizer_fingerprint != request.tokenizer_fingerprint:
                raise ValueError(f"draft tokenizer differs for {request.request_id}")
            if tuple(backend.encode(request.prompt_text)) != request.prompt_token_ids:
                raise ValueError(f"draft tokenizer cannot reproduce {request.request_id}")
            full_target_sha = _canonical_sha(target.to_dict())
            for prefix_position in range(target.target_path_length):
                identity = _snapshot_identity(request.request_id, prefix_position)
                if output.has(identity):
                    if not resume:
                        raise FileExistsError(output.path(identity))
                    existing = output.read(identity)
                    if existing["target_trajectory_sha256"] != full_target_sha:
                        raise ValueError("resume target trajectory differs")
                    continue
                context = request.prompt_token_ids + target.target_token_ids[:prefix_position]
                if (
                    len(context) + int(dimensions["candidate_depth"])
                    > config.runtime.context_length
                ):
                    raise ValueError(f"snapshot context exceeds limit for {identity}")
                forest = generate_real_candidate_forest(
                    backend,
                    request,
                    workload_sha256=workload_sha,
                    pool_dimensions=dimensions,
                    model_revision=_model_revision(
                        config.runtime.draft.revision, backend.model_id
                    ),
                    context_token_ids=context,
                    cycle_id=prefix_position,
                )
                suffix = _suffix_target(target, prefix_position, len(context))
                labeled = join_forest_and_target(request, forest, suffix)
                record = {
                    "schema_version": SNAPSHOT_SCHEMA,
                    "request_id": request.request_id,
                    "task_class": request.task_class,
                    "data_split": request.data_split,
                    "prefix_position": prefix_position,
                    "context_length": len(context),
                    "remaining_target_tokens": len(suffix.target_token_ids),
                    "forest_sha256": labeled.forest_sha256,
                    "pool_sha256": {
                        ratio: _canonical_sha({"node_ids": list(labeled.pool_node_ids[ratio])})
                        for ratio in POOL_ORDER
                    },
                    "target_trajectory_sha256": full_target_sha,
                    "target_continuation": list(suffix.target_token_ids),
                    "verification_budget": int(dimensions["verification_budget"]),
                    "common_snapshot_shared_by_all_selectors": True,
                    "target_trajectory_generated_once": True,
                    "labeled_trace": labeled.to_dict(),
                }
                written += int(output.write(identity, record))
    finally:
        if owns_backend:
            backend.close()
    records = output.records()
    return {
        "schema_version": "specrhythm.phase3c-stage-summary.v2",
        "stage": "multi-round-common-prefix-snapshots",
        "request_count": len(request_list),
        "new_snapshots": written,
        "completed_snapshots": len(records),
        "target_trajectory_generated_once": True,
        "selector_specific_forest_generation": False,
        "gpu_performance_result": False,
    }


def _selection_for(
    labeled: LabeledTraceRecord, ratio: str, selector: str, budget: int
) -> SelectionResult:
    pool = set(labeled.pool_node_ids[ratio])
    nodes = tuple(node for node in labeled.nodes if node.runtime_features.stable_node_id in pool)
    if selector == ORACLE_SELECTOR:
        return select_within_request_oracle(nodes, budget)
    return select_target_blind(selector, tuple(node.runtime_features for node in nodes), budget)


def _sequential_one(
    request: R3RealRequest,
    target: TargetTrajectoryRecord,
    snapshots: Mapping[int, Mapping[str, Any]],
    ratio: str,
    selector: str,
) -> dict[str, Any]:
    position = 0
    committed_sequence: list[int] = []
    rounds = []
    while position < target.target_path_length:
        snapshot = snapshots.get(position)
        if snapshot is None:
            raise ValueError(f"missing common snapshot {request.request_id}@prefix-{position}")
        labeled = LabeledTraceRecord.from_dict(snapshot["labeled_trace"])
        budget = int(snapshot["verification_budget"])
        selection = _selection_for(labeled, ratio, selector, budget)
        evaluation = _evaluate_selection(labeled, ratio, selector, selection)
        remaining = target.target_path_length - position
        accepted = int(evaluation["accepted_draft_tokens_per_proposal"])
        committed = min(remaining, accepted + int(accepted < remaining))
        if committed < 1:
            raise AssertionError("sequential replay made no target progress")
        committed_tokens = target.target_token_ids[position : position + committed]
        committed_sequence.extend(committed_tokens)
        rounds.append(
            {
                "round_index": len(rounds),
                "prefix_position": position,
                "forest_sha256": snapshot["forest_sha256"],
                "target_trajectory_sha256": snapshot["target_trajectory_sha256"],
                "selected_node_ids": list(selection.selected_node_ids),
                "accepted_tokens": accepted,
                "committed_tokens": committed,
                "verified_nodes": len(selection.selected_node_ids),
                "correction_or_bonus_tokens": committed - accepted,
            }
        )
        position += committed
    final = tuple(committed_sequence)
    if final != target.target_token_ids:
        raise AssertionError("sequential committed sequence differs from frozen target")
    return {
        "pool_ratio": ratio,
        "selector": selector,
        "proposal_rounds": len(rounds),
        "accepted_tokens": sum(row["accepted_tokens"] for row in rounds),
        "committed_tokens": sum(row["committed_tokens"] for row in rounds),
        "verified_nodes": sum(row["verified_nodes"] for row in rounds),
        "accepted_tokens_per_proposal": statistics.fmean(row["accepted_tokens"] for row in rounds),
        "committed_tokens_per_proposal": statistics.fmean(
            row["committed_tokens"] for row in rounds
        ),
        "verified_nodes_per_proposal": statistics.fmean(row["verified_nodes"] for row in rounds),
        "first_round_acceptance": rounds[0]["accepted_tokens"],
        "later_round_acceptance": (
            statistics.fmean(row["accepted_tokens"] for row in rounds[1:])
            if len(rounds) > 1
            else None
        ),
        "final_committed_target_token_ids": list(final),
        "final_sequence_matches_target": True,
        "rounds": rounds,
    }


def run_sequential_replay_stage(
    requests: Iterable[R3RealRequest],
    *,
    target_dir: Path,
    snapshot_dir: Path,
    output_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    request_list = list(requests)
    targets = target_store(target_dir)
    snapshots_by_request: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for snapshot in snapshot_store(snapshot_dir).records():
        snapshots_by_request[str(snapshot["request_id"])][int(snapshot["prefix_position"])] = (
            snapshot
        )
    output = sequential_store(output_dir)
    if not resume and output.records():
        raise FileExistsError("sequential output is non-empty; pass --resume to continue")
    written = 0
    for request in request_list:
        if output.has(request.request_id):
            if not resume:
                raise FileExistsError(output.path(request.request_id))
            continue
        target = targets.read(request.request_id)
        target_sha = _canonical_sha(target.to_dict())
        if any(
            snapshot["target_trajectory_sha256"] != target_sha
            for snapshot in snapshots_by_request[request.request_id].values()
        ):
            raise ValueError(f"snapshot target identity differs for {request.request_id}")
        results = [
            _sequential_one(
                request,
                target,
                snapshots_by_request[request.request_id],
                ratio,
                selector,
            )
            for ratio in POOL_ORDER
            for selector in SELECTOR_ORDER
        ]
        by_key = {(row["pool_ratio"], row["selector"]): row for row in results}
        for ratio in POOL_ORDER:
            oracle = by_key[(ratio, ORACLE_SELECTOR)]["accepted_tokens"]
            for selector in SELECTOR_ORDER:
                row = by_key[(ratio, selector)]
                row["oracle_regret_per_request"] = oracle - row["accepted_tokens"]
        record = {
            "schema_version": SEQUENTIAL_SCHEMA,
            "request_id": request.request_id,
            "task_class": request.task_class,
            "target_trajectory_sha256": _canonical_sha(target.to_dict()),
            "target_token_count": target.target_path_length,
            "target_token_ids": list(target.target_token_ids),
            "common_snapshots_reused": True,
            "results": results,
        }
        written += int(output.write(request.request_id, record))
    return {
        "schema_version": "specrhythm.phase3c-stage-summary.v2",
        "stage": "multi-round-sequential-selector-replay",
        "new_records": written,
        "completed_records": len(output.records()),
        "gpu_performance_result": False,
        "reports_goodput_or_slo": False,
    }


def _common_snapshot_rows(
    snapshots: Iterable[Mapping[str, Any]],
) -> tuple[list[LabeledTraceRecord], list[dict[str, Any]]]:
    labels = []
    rows = []
    for snapshot in snapshots:
        labeled = LabeledTraceRecord.from_dict(snapshot["labeled_trace"])
        labels.append(labeled)
        prefix = int(snapshot["prefix_position"])
        for ratio in POOL_ORDER:
            ratio_rows = {}
            for selector in SELECTOR_ORDER:
                selection = _selection_for(
                    labeled, ratio, selector, int(snapshot["verification_budget"])
                )
                row = _evaluate_selection(labeled, ratio, selector, selection)
                row["snapshot_id"] = _snapshot_identity(labeled.request_id, prefix)
                row["prefix_depth"] = prefix
                ratio_rows[selector] = row
            oracle = ratio_rows[ORACLE_SELECTOR]["accepted_draft_tokens_per_proposal"]
            for row in ratio_rows.values():
                row["selector_regret"] = oracle - row["accepted_draft_tokens_per_proposal"]
                rows.append(row)
    return labels, rows


def multiround_headroom(
    snapshots: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    _, rows = _common_snapshot_rows(snapshots)
    by_key = {(row["snapshot_id"], row["pool_ratio"], row["selector"]): row for row in rows}
    request_rows = []
    for row in rows:
        if row["selector"] not in TARGET_BLIND_SELECTORS:
            continue
        ratio = str(row["pool_ratio"])
        ratio_index = POOL_ORDER.index(ratio)
        previous = POOL_ORDER[ratio_index - 1] if ratio_index else None
        snapshot_id = str(row["snapshot_id"])
        oracle = by_key[(snapshot_id, ratio, ORACLE_SELECTOR)]
        generator_gain = None
        selector_gain = None
        if previous:
            previous_oracle = by_key[(snapshot_id, previous, ORACLE_SELECTOR)]
            previous_selector = by_key[(snapshot_id, previous, row["selector"])]
            generator_gain = (
                oracle["accepted_draft_tokens_per_proposal"]
                - previous_oracle["accepted_draft_tokens_per_proposal"]
            )
            selector_gain = (
                row["accepted_draft_tokens_per_proposal"]
                - previous_selector["accepted_draft_tokens_per_proposal"]
            )
        identifiable = generator_gain not in (None, 0)
        # Full-pool prefix ignores budget but still requires an unbroken target path.
        first_missing = oracle["first_missing_target_depth"]
        full_prefix = (
            int(first_missing) - 1
            if first_missing is not None
            else int(oracle["eligible_target_trajectory_nodes"])
        )
        budget_constraint = max(0, full_prefix - int(oracle["accepted_draft_tokens_per_proposal"]))
        if full_prefix == 0:
            case = "pool does not contain useful target nodes"
        elif budget_constraint > 0:
            case = "pool contains target nodes but budget cannot reach them"
        elif row["selector_regret"] > 0:
            case = "budget can reach target nodes but selector cannot identify them"
        else:
            case = "selector identifies target nodes and realizes available gain"
        request_rows.append(
            {
                "snapshot_id": snapshot_id,
                "request_id": row["request_id"],
                "task_class": row["task_class"],
                "prefix_depth": row["prefix_depth"],
                "pool_ratio": ratio,
                "selector": row["selector"],
                "generator_coverage_ceiling": generator_gain,
                "selector_regret": row["selector_regret"],
                "budget_constraint": budget_constraint,
                "pool_expansion_utilization": (
                    selector_gain / generator_gain if identifiable else None
                ),
                "pool_expansion_utilization_identifiable": identifiable,
                "case": case,
            }
        )
    aggregates = []
    groups: dict[tuple[str, int, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in request_rows:
        for task in (str(row["task_class"]), "all"):
            groups[
                (task, int(row["prefix_depth"]), str(row["pool_ratio"]), str(row["selector"]))
            ].append(row)
    for (task, depth, ratio, selector), values in sorted(groups.items()):
        identifiable = [
            float(row["pool_expansion_utilization"])
            for row in values
            if row["pool_expansion_utilization_identifiable"]
        ]
        gains = [
            float(row["generator_coverage_ceiling"])
            for row in values
            if row["generator_coverage_ceiling"] is not None
        ]
        aggregates.append(
            {
                "task_class": task,
                "prefix_depth": depth,
                "pool_ratio": ratio,
                "selector": selector,
                "requests": len(values),
                "generator_coverage_ceiling": statistics.fmean(gains) if gains else None,
                "selector_regret": statistics.fmean(
                    float(row["selector_regret"]) for row in values
                ),
                "budget_constraint": statistics.fmean(
                    float(row["budget_constraint"]) for row in values
                ),
                "pool_expansion_utilization": (
                    statistics.fmean(identifiable) if identifiable else None
                ),
                "pool_expansion_utilization_identifiable_requests": len(identifiable),
                "case_counts": {
                    case: sum(row["case"] == case for row in values)
                    for case in sorted({str(row["case"]) for row in values})
                },
            }
        )
    return {"per_snapshot": request_rows, "aggregate": aggregates}


def summarize_multiround(
    *, snapshot_dir: Path, sequential_dir: Path, source_trace_commit: str
) -> dict[str, Any]:
    snapshots = snapshot_store(snapshot_dir).records()
    sequential = sequential_store(sequential_dir).records()
    if not snapshots or not sequential:
        raise ValueError("multi-round summary requires snapshots and sequential replay")
    snapshot_prefixes: dict[str, set[int]] = defaultdict(set)
    for snapshot in snapshots:
        snapshot_prefixes[str(snapshot["request_id"])].add(
            int(snapshot["prefix_position"])
        )
    sequential_ids = {str(record["request_id"]) for record in sequential}
    if set(snapshot_prefixes) != sequential_ids:
        raise ValueError("snapshot and sequential request sets differ")
    for record in sequential:
        expected = set(range(int(record["target_token_count"])))
        if snapshot_prefixes[str(record["request_id"])] != expected:
            raise ValueError("common-prefix snapshot positions are incomplete")
    result_rows = [row for record in sequential for row in record["results"]]
    aggregate = []
    for task in sorted({str(record["task_class"]) for record in sequential}) + ["all"]:
        for ratio in POOL_ORDER:
            for selector in SELECTOR_ORDER:
                values = [
                    row
                    for record in sequential
                    if (task == "all" or record["task_class"] == task)
                    for row in record["results"]
                    if row["pool_ratio"] == ratio and row["selector"] == selector
                ]
                aggregate.append(
                    {
                        "task_class": task,
                        "pool_ratio": ratio,
                        "selector": selector,
                        "requests": len(values),
                        "proposal_rounds_per_request": statistics.fmean(
                            row["proposal_rounds"] for row in values
                        ),
                        "accepted_tokens_per_proposal": statistics.fmean(
                            row["accepted_tokens_per_proposal"] for row in values
                        ),
                        "committed_tokens_per_proposal": statistics.fmean(
                            row["committed_tokens_per_proposal"] for row in values
                        ),
                        "verified_nodes_per_proposal": statistics.fmean(
                            row["verified_nodes_per_proposal"] for row in values
                        ),
                        "total_verified_nodes_per_request": statistics.fmean(
                            row["verified_nodes"] for row in values
                        ),
                        "oracle_regret_per_request": statistics.fmean(
                            row["oracle_regret_per_request"] for row in values
                        ),
                        "first_round_acceptance": statistics.fmean(
                            row["first_round_acceptance"] for row in values
                        ),
                        "later_round_acceptance": (
                            statistics.fmean(
                                row["later_round_acceptance"]
                                for row in values
                                if row["later_round_acceptance"] is not None
                            )
                            if any(row["later_round_acceptance"] is not None for row in values)
                            else None
                        ),
                    }
                )
    return {
        "schema_version": "specrhythm.phase3c-multiround-summary.v1",
        "source_trace_commit": source_trace_commit,
        "source_trace_sha256": {
            "common_prefix_snapshots": _directory_sha(snapshot_dir),
            "sequential_replay": _directory_sha(sequential_dir),
        },
        "request_count": len(sequential),
        "snapshot_count": len(snapshots),
        "target_trajectory_generated_once": True,
        "common_snapshot_shared_by_all_selectors": True,
        "final_target_sequence_match": all(
            row["final_sequence_matches_target"] for row in result_rows
        ),
        "gpu_performance_result": False,
        "reports_goodput_slo_or_speedup": False,
        "aggregate_metrics": aggregate,
        "headroom_decomposition": multiround_headroom(snapshots),
    }


def multiround_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 3C.2 multi-round common-prefix pilot",
        "",
        (
            "Selector signal and token-accounting evidence only; no GPU latency, "
            "goodput, SLO, or speedup claim."
        ),
        "",
        (
            "| Task | Pool | Selector | Rounds/request | Accepted/proposal | "
            "Committed/proposal | Verified/request | Oracle regret/request | First | Later |"
        ),
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["aggregate_metrics"]:
        later = row["later_round_acceptance"]
        later_text = f"{later:.3f}" if later is not None else "n/a"
        lines.append(
            f"| {row['task_class']} | {row['pool_ratio']} | {row['selector']} | "
            f"{row['proposal_rounds_per_request']:.3f} | "
            f"{row['accepted_tokens_per_proposal']:.3f} | "
            f"{row['committed_tokens_per_proposal']:.3f} | "
            f"{row['total_verified_nodes_per_request']:.3f} | "
            f"{row['oracle_regret_per_request']:.3f} | "
            f"{row['first_round_acceptance']:.3f} | {later_text} |"
        )
    return "\n".join(lines) + "\n"
