"""Offline matched-work policy, independent of exact generated-sequence diagnostics."""

from __future__ import annotations

import math
from typing import Any, Mapping


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _integer(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _sha(value: Any, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _tokens(value: Any) -> bool:
    return isinstance(value, list) and all(_integer(token) for token in value)


def _revision_fields_valid(value: Mapping[str, Any]) -> bool:
    return all(
        revision is None or isinstance(revision, str)
        for key, revision in value.items()
        if key == "revision" or key.endswith("_revision")
    )


def _model_identity_valid(value: Any) -> bool:
    models = _mapping(value)
    for role in ("target", "draft"):
        model = _mapping(models.get(role))
        path = model.get("path")
        if (
            not isinstance(path, str)
            or not path.strip()
            or "revision" not in model
            or not _revision_fields_valid(model)
        ):
            return False
    return _revision_fields_valid(models)


def _request_complete(row: Mapping[str, Any]) -> bool:
    """Consume v1 completion evidence and reject concrete contradictions.

    SmokeRequest.maximum_new_tokens / SamplingParams.max_tokens limit TOTAL outputs.
    Historical performance v1 mistakenly copied workload.output_tokens, which is absent
    in R3-real workloads, as maximum_new_tokens=None. Null is unavailable metadata, not
    an output limit of zero. Equal workload SHA and equal measured work remain mandatory.
    Never infer the missing requested limit from the number of observed tokens.
    """

    finish = row.get("finish_reason")
    if (
        not isinstance(finish, str)
        or not finish.strip()
        or finish.strip().lower() in {
            "abort", "aborted", "error", "failed", "cancel", "cancelled", "canceled",
            "incomplete",
        }
        or row.get("completed", True) is not True
        or row.get("finished", True) is not True
        or row.get("token_accounting_valid") is not True
        or not _integer(row.get("measured_committed_output_token_count"), 1)
        or not _tokens(row.get("total_generated_token_ids"))
        or not row["total_generated_token_ids"]
        or "maximum_new_tokens" not in row
    ):
        return False
    maximum = row["maximum_new_tokens"]
    if maximum is None:
        return True  # Explicit legacy v1 omission; overall per-mode validity still gates.
    if not _integer(maximum, 1):
        return False
    generated_count = len(row["total_generated_token_ids"])
    if generated_count > maximum:
        return False
    # A length-limited finish below a known limit is contradictory. Other terminal
    # reasons may legitimately stop early; v1 already checked output.finished.
    return (
        finish.strip().lower() not in {"length", "max_tokens"}
        or generated_count == maximum
    )


def compare_matched_work(values: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Require equivalent provenance and completed work, never cross-mode token identity.

    v1 artifacts bind requested semantics through workload/config SHA and per-request
    output limits. Their git_commit is the execution commit, not the measurement commit.
    No compatibility between different execution commits is assumed.
    """

    checks: dict[str, bool] = {}
    errors: list[str] = []

    def require(check: str, valid: bool, message: str) -> None:
        checks[check] = checks.get(check, True) and bool(valid)
        if not valid:
            errors.append(message)

    def same(check: str, items: list[Any], present: bool = True) -> None:
        require(
            check,
            present and all(item == items[0] for item in items[1:]),
            f"{check}: missing or incompatible matched-work evidence",
        )

    requests = {}
    for mode, value in values.items():
        require(
            "per_mode_artifacts_valid",
            value.get("schema_version") == "specrhythm.phase4b2-decode-performance.v1"
            and value.get("mode") == mode
            and value.get("valid") is True
            and value.get("performance_result") is True
            and value.get("errors") == [],
            f"{mode}: performance artifact is invalid",
        )
        require("cleanup_valid", value.get("cleanup_valid") is True, f"{mode}: cleanup invalid")
        rows = value.get("requests")
        rows_valid = (
            isinstance(rows, list)
            and bool(rows)
            and all(
                isinstance(row, Mapping)
                and isinstance(row.get("request_id"), str)
                and bool(row["request_id"])
                for row in rows
            )
        )
        current = {row["request_id"]: row for row in rows} if rows_valid else {}
        requests[mode] = current
        require(
            "canonical_request_mapping_valid",
            rows_valid and len(current) == len(rows),
            f"{mode}: request rows are empty, malformed, or duplicate",
        )
        metrics = _mapping(value.get("metrics"))
        require(
            "request_count_equal",
            _integer(value.get("request_count"), 1)
            and _integer(metrics.get("completed_requests"), 1)
            and value["request_count"] == metrics["completed_requests"] == len(current),
            f"{mode}: request/completion counts disagree",
        )
        for field in ("decode_makespan_ms", "aggregate_throughput_tokens_per_second"):
            number = metrics.get(field)
            require(
                "metrics_valid",
                type(number) in (int, float) and math.isfinite(number) and number > 0,
                f"{mode}: {field} must be finite and positive",
            )
        count_sum = 0
        for request_id, row in current.items():
            count = row.get("measured_committed_output_token_count")
            measured = row.get("measured_committed_output_token_ids")
            generated = row.get("total_generated_token_ids")
            bootstrap = row.get("bootstrap_token_id")
            count_valid = _integer(count, 1)
            count_sum += count if count_valid else 0
            require(
                "token_accounting_valid",
                count_valid
                and _integer(bootstrap)
                and row.get("setup_committed_output_tokens") == 1
                and row.get("token_accounting_valid") is True
                and _tokens(measured)
                and _tokens(generated)
                and len(measured) == count
                and [bootstrap, *measured] == generated,
                f"{mode}/{request_id}: token accounting invalid",
            )
            require(
                "prompt_provenance_equal",
                _integer(row.get("prompt_token_count"), 1)
                and _sha(row.get("prompt_token_ids_sha256")),
                f"{mode}/{request_id}: prompt provenance missing or malformed",
            )
            require(
                "requests_complete",
                _request_complete(row),
                f"{mode}/{request_id}: failed, incomplete, or invalid output length",
            )
        total = metrics.get("total_measured_committed_output_tokens")
        require(
            "total_measured_token_counts_equal",
            _integer(total, 1) and total == count_sum,
            f"{mode}: total measured count disagrees with request counts",
        )
        measurement = _mapping(value.get("measurement"))
        contract = {
            "clock": "time.monotonic_ns",
            "setup_excluded": True,
            "bootstrap_excluded_from_measured_token_count": True,
            "first_measured_target_forward_consumes_pending_bootstrap": True,
            "first_post_bootstrap_token_counted": True,
            "pre_measurement_tp_barrier": True,
            "pre_measurement_target_cuda_synchronize": True,
            "per_token_cuda_synchronize": False,
            "final_all_target_rank_cuda_synchronize": True,
        }
        require(
            "measurement_boundary_equivalent",
            all(
                measurement.get(key) is expected
                if isinstance(expected, bool)
                else measurement.get(key) == expected
                for key, expected in contract.items()
            ),
            f"{mode}: measurement-boundary contract invalid",
        )
        semantics = _mapping(value.get("mode_semantics"))
        if mode == "target":
            require(
                "mode_execution_valid",
                semantics.get("draft_measured_work") is False
                and semantics.get("proposals") is False,
                "target: measured Draft work must be absent",
            )
        elif mode == "serial":
            require(
                "mode_execution_valid",
                semantics.get("draft_target_overlap") is False
                and semantics.get("initial_proposal_after_measurement_start") is True,
                "serial: mode/boundary evidence invalid",
            )
        else:
            require(
                "dual_overlap_valid",
                semantics.get("natural_draft_target_overlap") is True
                and semantics.get("per_round_global_cuda_synchronize") is False
                and semantics.get("dual_eager") is False,
                "dual-batch: physical overlap/unsynchronized Dual-Batch evidence invalid",
            )

    artifacts = list(values.values())
    same("request_set_equal", [set(rows) for rows in requests.values()])
    same("request_count_equal", [value.get("request_count") for value in artifacts])
    same(
        "total_measured_token_counts_equal",
        [
            _mapping(value.get("metrics")).get("total_measured_committed_output_tokens")
            for value in artifacts
        ],
    )
    for field, check in (
        ("workload_sha256", "workload_equal"),
        ("vllm_commit", "execution_provenance_equal"),
        ("vllm_version", "execution_provenance_equal"),
        ("patch_hashes", "execution_provenance_equal"),
        ("models", "execution_provenance_equal"),
        ("placement", "topology_equal"),
        ("gpu_topology", "topology_equal"),
        ("correctness_mode", "requested_workload_semantics_equal"),
    ):
        items = [value.get(field) for value in artifacts]
        same(check, items, all(bool(item) for item in items))
    commits = []
    for mode, value in values.items():
        commit = value.get("execution_git_commit", value.get("git_commit"))
        commits.append(commit)
        require(
            "execution_provenance_equal",
            _sha(commit, 40) and value.get("git_commit") == commit,
            f"{mode}: execution commit is missing or contradicts legacy git_commit",
        )
        require(
            "execution_provenance_equal",
            _sha(value.get("vllm_commit"), 40)
            and isinstance(value.get("patch_hashes"), list)
            and bool(value["patch_hashes"])
            and all(_sha(item) for item in value["patch_hashes"])
            and _model_identity_valid(value.get("models")),
            f"{mode}: model/vLLM/patch identity missing or malformed",
        )
        placement = _mapping(value.get("placement"))
        require(
            "topology_equal",
            placement.get("target_physical_gpu_ids") == [1, 2]
            and placement.get("target_tensor_parallel_size") == 2
            and placement.get("draft_physical_gpu_ids") == [0]
            and placement.get("draft_tensor_parallel_size") == 1,
            f"{mode}: expected Draft GPU0 / Target GPU1,2 TP2 placement",
        )
    same("execution_provenance_equal", commits)
    for field, check in (
        ("workload", "workload_equal"),
        ("config", "config_equal"),
        ("topology", "topology_equal"),
        ("patch_manifest", "execution_provenance_equal"),
    ):
        items = [_mapping(value.get("artifact_sha256")).get(field) for value in artifacts]
        same(check, items, all(_sha(item) for item in items))
        if field == "workload":
            require(
                check,
                all(item == value.get("workload_sha256") for item, value in zip(items, artifacts)),
                "workload digest contradicts artifact provenance",
            )

    target_rows = requests["target"]
    termination_differences = []
    for mode, current in requests.items():
        if mode == "target":
            continue
        for request_id in sorted(set(target_rows) & set(current)):
            target, row = target_rows[request_id], current[request_id]
            for field, check in (
                ("prompt_token_count", "prompt_provenance_equal"),
                ("prompt_token_ids_sha256", "prompt_provenance_equal"),
                ("bootstrap_token_id", "bootstrap_equal"),
                ("maximum_new_tokens", "requested_workload_semantics_equal"),
                (
                    "measured_committed_output_token_count",
                    "per_request_measured_token_counts_equal",
                ),
            ):
                require(
                    check,
                    target.get(field) == row.get(field),
                    f"target/{mode}/{request_id}: {field} differs",
                )
            for field in ("finish_reason", "termination_reason"):
                if target.get(field) != row.get(field):
                    limits_known = all(
                        _integer(candidate.get("maximum_new_tokens"), 1)
                        for candidate in (target, row)
                    )
                    fixed_length = all(
                        _tokens(candidate.get("total_generated_token_ids"))
                        and len(candidate["total_generated_token_ids"])
                        == candidate["maximum_new_tokens"]
                        for candidate in (target, row)
                    ) if limits_known else None
                    termination_differences.append(
                        {
                            "mode": mode,
                            "request_id": request_id,
                            "field": field,
                            "target": target.get(field),
                            "compared": row.get(field),
                            "fixed_length_completed": fixed_length,
                        }
                    )
                    # Completion and equal measured counts are independently required.
                    # Different successful stop labels do not change matched work.
    return {
        "valid": not errors,
        "errors": errors,
        **checks,
        "request_mapping": "canonical request_id; artifact row order is immaterial",
        "execution_compatibility_policy": "identical execution commit; v1 git_commit fallback",
        "measurement_code_commit_equality_required": False,
        "completion_evidence": {
            "authority": "valid v1 mode artifact, token accounting and terminal evidence",
            "known_output_limit_scope": "total generated tokens, including setup bootstrap",
            "null_output_limit_scope": (
                "legacy v1 metadata unavailable; frozen workload SHA binds it"
            ),
            "null_output_limit_request_counts": {
                mode: sum(row.get("maximum_new_tokens") is None for row in rows.values())
                for mode, rows in requests.items()
            },
        },
        "termination_differences": termination_differences,
    }


def exact_sequence_diagnostic(values: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Compare stored sequences exactly; this evidence never grants performance validity."""

    by_mode = {
        mode: {
            row["request_id"]: row
            for row in value.get("requests", [])
            if isinstance(row, Mapping) and row.get("request_id")
        }
        for mode, value in values.items()
    }
    request_ids = set().union(*(set(rows) for rows in by_mode.values()))
    divergent = set()
    pairs = {}
    for mode, rows in by_mode.items():
        if mode == "target":
            continue
        mismatches = []
        for request_id in sorted(request_ids):
            target, row = by_mode["target"].get(request_id), rows.get(request_id)
            fields = ("measured_committed_output_token_ids", "total_generated_token_ids")
            field = "request_presence"
            if target is not None and row is not None:
                field = next((key for key in fields if target.get(key) != row.get(key)), "")
            if field:
                divergent.add(request_id)
                mismatches.append({"request_id": request_id, "field": field})
        pairs[f"target_equals_{mode}"] = {
            "all_equal": not mismatches,
            "divergent_request_count": len(mismatches),
            "first_mismatches": mismatches[:10],
        }
    return {
        "all_equal": not divergent,
        "no_tolerance": True,
        "divergent_request_count": len(divergent),
        "matching_request_count": len(request_ids - divergent),
        "divergent_request_ids": sorted(divergent),
        "first_mismatches": [
            {"pair": pair, **row}
            for pair, diagnostic in pairs.items()
            for row in diagnostic["first_mismatches"]
        ][:10],
        "pairs": pairs,
    }
