"""Fail-closed Phase-4B.1 stock-reference freeze with immutable diagnostics.

This integration utility deliberately performs exactly one pair of stock Target
runs.  It always freezes both raw runs and their keyed divergence report before
either freezing the reference or returning a nondeterminism failure.  It never
retries a failed pair.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from specrhythm.phase4.config import load_phase4_config
from specrhythm.phase4.manifest import sha256_file
from specrhythm.phase4.reference import (
    _exclusive_freeze,
    build_stock_reference,
    require_stock_vllm_runner,
)
from specrhythm.phase4.stock_vllm import run_stock_smoke

DIAGNOSTIC_SCHEMA = "specrhythm.phase4b1-stock-determinism-diagnostic.v1"


def build_stock_determinism_diagnostic(
    smoke: Mapping[str, Any],
    *,
    git_commit: str,
    config_sha256: str,
    workload_sha256: str,
    installed_runner_sha256: str,
) -> dict[str, Any]:
    """Return exact per-request evidence for the one measured run pair."""

    runs = smoke.get("runs")
    errors: list[str] = []
    if not isinstance(runs, list) or len(runs) != 2:
        runs = [[], []]
        errors.append("stock smoke does not contain exactly two runs")
    first = runs[0] if isinstance(runs[0], list) else []
    second = runs[1] if isinstance(runs[1], list) else []
    if not isinstance(runs[0], list) or not isinstance(runs[1], list):
        errors.append("one or both stock runs are not arrays")

    first_by_id, first_errors = _index_run(first, "run_1")
    second_by_id, second_errors = _index_run(second, "run_2")
    errors.extend(first_errors)
    errors.extend(second_errors)
    first_order = [str(row.get("request_id", "")) for row in first]
    second_order = [str(row.get("request_id", "")) for row in second]
    order_equal = first_order == second_order
    order_divergence = _first_divergence(first_order, second_order)
    if not order_equal:
        errors.append("stock repeated runs have different request order")

    request_ids = list(dict.fromkeys([*first_order, *second_order]))
    comparisons = []
    for request_id in request_ids:
        one = first_by_id.get(request_id)
        two = second_by_id.get(request_id)
        if one is None or two is None:
            comparisons.append(
                {
                    "request_id": request_id,
                    "equal": False,
                    "missing_from_run_1": one is None,
                    "missing_from_run_2": two is None,
                    "first_divergence_position": None,
                    "run_1_token_id": None,
                    "run_2_token_id": None,
                    "semantic_mismatches": ["missing-request"],
                }
            )
            continue
        one_tokens = _token_ids(one)
        two_tokens = _token_ids(two)
        divergence = _first_divergence(one_tokens, two_tokens)
        semantic_mismatches = [
            field
            for field in ("text", "finish_reason", "stop_reason")
            if one.get(field) != two.get(field)
        ]
        comparisons.append(
            {
                "request_id": request_id,
                "equal": divergence is None and not semantic_mismatches,
                "missing_from_run_1": False,
                "missing_from_run_2": False,
                "run_1_generated_token_count": len(one_tokens),
                "run_2_generated_token_count": len(two_tokens),
                "first_divergence_position": (
                    divergence[0] if divergence is not None else None
                ),
                "run_1_token_id": divergence[1] if divergence is not None else None,
                "run_2_token_id": divergence[2] if divergence is not None else None,
                "semantic_mismatches": semantic_mismatches,
                "run_1_finish_reason": one.get("finish_reason"),
                "run_2_finish_reason": two.get("finish_reason"),
                "run_1_stop_reason": one.get("stop_reason"),
                "run_2_stop_reason": two.get("stop_reason"),
            }
        )

    divergent = [row for row in comparisons if row["equal"] is not True]
    computed = not errors and not divergent
    reported = smoke.get("repeated_run_deterministic") is True
    if reported != computed:
        errors.append("stock smoke determinism flag disagrees with raw run comparison")
    if not computed:
        errors.append("stock target-only reference is not deterministic")
    errors = list(dict.fromkeys(errors))
    return {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "valid": not errors,
        "outcome": "deterministic" if not errors else "nondeterministic",
        "errors": errors,
        "retry_count": 0,
        "retry_until_success": False,
        "reference_freeze_eligible": not errors,
        "repeated_run_deterministic_reported": reported,
        "repeated_run_deterministic_computed": computed,
        "request_order_equal": order_equal,
        "first_request_order_divergence_index": (
            order_divergence[0] if order_divergence is not None else None
        ),
        "run_1_request_at_order_divergence": (
            order_divergence[1] if order_divergence is not None else None
        ),
        "run_2_request_at_order_divergence": (
            order_divergence[2] if order_divergence is not None else None
        ),
        "run_1_request_count": len(first),
        "run_2_request_count": len(second),
        "divergent_request_count": len(divergent),
        "first_divergent_request_id": (
            divergent[0]["request_id"] if divergent else None
        ),
        "per_request_comparisons": comparisons,
        "runs": [first, second],
        "correctness_mode": smoke.get("correctness_mode"),
        "batch_invariant_requested": smoke.get("batch_invariant_requested"),
        "batch_invariant_effective": smoke.get("batch_invariant_effective"),
        "batch_invariant_validation": smoke.get("batch_invariant_validation"),
        "worker_ranks": smoke.get("worker_ranks"),
        "provenance": {
            "specrhythm_git_commit": git_commit,
            "config_sha256": config_sha256,
            "workload_sha256": workload_sha256,
            "installed_stock_runner_sha256": installed_runner_sha256,
        },
    }


def freeze_with_diagnostic(args: argparse.Namespace) -> None:
    output_path = Path(args.output).resolve()
    diagnostic_path = Path(args.determinism_diagnostic).resolve()
    for path in (output_path, diagnostic_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite immutable artifact {path}")
    config = load_phase4_config(args.config)
    if (
        config.target_model_runner == "v1"
        and os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "0"
    ):
        raise RuntimeError(
            "Phase 4B.1 stock reference requires VLLM_USE_V2_MODEL_RUNNER=0"
        )
    installed_runner_sha256 = require_stock_vllm_runner()
    git_commit = _git_commit()
    workload_path = Path(args.workload).resolve()
    smoke = run_stock_smoke(
        config,
        role="target",
        workload_path=workload_path,
        environment_path=Path(args.environment).resolve(),
        topology_path=Path(args.topology).resolve(),
        runtime_manifest_path=Path(args.runtime_manifest).resolve(),
        git_commit=git_commit,
        correctness_mode=args.correctness_mode,
        request_count=args.request_count,
    )
    diagnostic = build_stock_determinism_diagnostic(
        smoke,
        git_commit=git_commit,
        config_sha256=sha256_file(config.path),
        workload_sha256=sha256_file(workload_path),
        installed_runner_sha256=installed_runner_sha256,
    )
    _exclusive_freeze(diagnostic_path, diagnostic)
    if diagnostic["valid"] is not True:
        raise RuntimeError(
            "stock target-only reference is not deterministic; immutable diagnostic: "
            + str(diagnostic_path)
        )
    reference = build_stock_reference(
        smoke,
        config,
        workload_path=workload_path,
        git_commit=git_commit,
        installed_runner_sha256=installed_runner_sha256,
    )
    _exclusive_freeze(output_path, reference)


def _index_run(
    rows: Sequence[Any], label: str
) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    errors = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"{label} row {index} is not an object")
            continue
        request_id = str(row.get("request_id", ""))
        if not request_id:
            errors.append(f"{label} row {index} has an empty request ID")
        elif request_id in indexed:
            errors.append(f"{label} contains duplicate request ID {request_id}")
        else:
            indexed[request_id] = row
        tokens = row.get("generated_token_ids")
        if not isinstance(tokens, list) or any(
            not isinstance(item, int) or isinstance(item, bool) for item in tokens
        ):
            errors.append(f"{label} row {index} has invalid generated token IDs")
    return indexed, errors


def _token_ids(row: Mapping[str, Any]) -> list[int]:
    value = row.get("generated_token_ids")
    if not isinstance(value, list):
        return []
    return [int(item) for item in value]


def _first_divergence(
    first: Sequence[Any], second: Sequence[Any]
) -> Optional[tuple[int, Optional[Any], Optional[Any]]]:
    for index, (one, two) in enumerate(zip(first, second)):
        if one != two:
            return index, one, two
    if len(first) != len(second):
        index = min(len(first), len(second))
        return (
            index,
            first[index] if index < len(first) else None,
            second[index] if index < len(second) else None,
        )
    return None


def _git_commit() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"), capture_output=True, text=True, check=False
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else "unknown"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "run exactly one two-run stock Target pair, freeze its determinism "
            "diagnostic, and fail closed before reference creation on divergence"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--request-count", type=int, required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--runtime-manifest", required=True)
    parser.add_argument(
        "--correctness-mode",
        choices=("default", "batch-invariant"),
        default="batch-invariant",
    )
    parser.add_argument("--determinism-diagnostic", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        freeze_with_diagnostic(args)
    except (FileExistsError, FileNotFoundError, ImportError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Phase-4B.1 stock reference freeze failed: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
