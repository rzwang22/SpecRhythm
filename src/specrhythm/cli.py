"""Command-line entry point for workload and simulator experiments."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Sequence

from specrhythm.phase2 import (
    PHASE2_VARIANTS,
    SEARCH_RATIOS,
    common_snapshot_replay,
    run_phase2_end_to_end,
)
from specrhythm.policies import (
    AdaServeFlatProxyPolicy,
    AdaServeStylePolicy,
    ARPolicy,
    DualBatchPolicy,
    DualEagerPolicy,
    LegacyFlatShapingProxyPolicy,
    SerialSDPolicy,
    ShapingDiagnosticPolicy,
    SpecRhythmPolicy,
)
from specrhythm.provenance import build_manifest, pin_source_url
from specrhythm.schema import Workload
from specrhythm.simulator import SimulatorConfig, simulate
from specrhythm.validation import validate_workload
from specrhythm.workload import (
    generate_replay_workload,
    generate_workload,
    import_mooncake,
    load_json,
    select_arrival_replay,
    summarize_workload,
)

POLICY_ORDER = (
    "ar",
    "serial-sd",
    "adaserve-flat-proxy",
    "adaserve",
    "dual-batch",
    "dual-eager",
    "shaping-flat-proxy",
    "shaping",
    "specrhythm-flat-proxy",
    "specrhythm",
)

DIAGNOSTIC_POLICY_ORDER = (
    "shaping-feasible",
    "residual-round-robin",
    "residual-probability",
    "shaping-residual",
    "feasible-residual",
    "shaping-feasible-residual",
)

SIMULATION_POLICY_ORDER = POLICY_ORDER + DIAGNOSTIC_POLICY_ORDER


def _write_json(value: Any, path: Optional[str]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True)
    if path is None:
        print(payload)
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload + "\n", encoding="utf-8")


def _policy(name: str, config: SimulatorConfig) -> Any:
    if name == "ar":
        return ARPolicy()
    if name == "serial-sd":
        return SerialSDPolicy(config.speculative_budget)
    if name == "adaserve":
        return AdaServeStylePolicy(config.n_max_slo)
    if name == "adaserve-flat-proxy":
        return AdaServeFlatProxyPolicy()
    if name == "shaping-flat-proxy":
        return LegacyFlatShapingProxyPolicy(enable_eager=False)
    if name == "specrhythm-flat-proxy":
        return LegacyFlatShapingProxyPolicy(enable_eager=True)
    if name == "dual-batch":
        return DualBatchPolicy(config.speculative_budget)
    if name == "dual-eager":
        return DualEagerPolicy(
            config.speculative_budget,
            max_eager_budget=config.max_eager_budget,
            min_dependency_path_probability=config.min_dependency_path_probability,
        )
    if name == "shaping":
        return SpecRhythmPolicy(
            enable_eager=False,
            n_max_slo=config.n_max_slo,
            residual_score=config.specrhythm_residual_score,
        )
    if name in DIAGNOSTIC_POLICY_ORDER:
        return ShapingDiagnosticPolicy(
            name,
            speculative_budget=config.speculative_budget,
            n_max_slo=config.n_max_slo,
            residual_score=config.specrhythm_residual_score,
        )
    if name == "specrhythm":
        return SpecRhythmPolicy(
            n_max_slo=config.n_max_slo,
            residual_score=config.specrhythm_residual_score,
            max_eager_budget=config.max_eager_budget,
            min_dependency_path_probability=config.min_dependency_path_probability,
        )
    raise ValueError(f"unknown policy: {name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="specrhythm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate or compose a workload")
    generate.add_argument("--config", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--arrival-trace")
    generate.add_argument("--time-scale", type=float, default=1.0)
    generate.add_argument("--window-start-ms", type=float, default=0.0)
    generate.add_argument("--window-duration-ms", type=float)
    generate.add_argument("--manifest")
    generate.add_argument("--source-url")
    generate.add_argument("--source-commit-sha")

    mooncake = subparsers.add_parser("import-mooncake", help="normalize Mooncake JSONL")
    mooncake.add_argument("--input", required=True)
    mooncake.add_argument("--output", required=True)
    mooncake.add_argument("--time-scale", type=float, default=1.0)
    mooncake.add_argument("--slo-tpot-ms", type=float, default=50.0)
    mooncake.add_argument("--acceptance-probability", type=float, default=0.7)
    mooncake.add_argument("--draft-confidence", type=float, default=0.7)

    summary = subparsers.add_parser("summarize", help="summarize a canonical workload")
    summary.add_argument("--workload", required=True)
    summary.add_argument("--output")

    validation = subparsers.add_parser("validate", help="validate a canonical workload")
    validation.add_argument("--workload", required=True)
    validation.add_argument("--config")
    validation.add_argument("--arrival-trace")
    validation.add_argument("--time-scale", type=float, default=1.0)
    validation.add_argument("--window-start-ms", type=float, default=0.0)
    validation.add_argument("--window-duration-ms", type=float)
    validation.add_argument("--output")

    simulation = subparsers.add_parser("simulate", help="run one scheduling policy")
    simulation.add_argument("--workload", required=True)
    simulation.add_argument("--config", required=True)
    simulation.add_argument(
        "--policy",
        choices=SIMULATION_POLICY_ORDER,
        required=True,
        help=(
            "execution mode; adaserve is a tree-aware control-plane baseline under "
            "proxy inputs; adaserve-flat-proxy retains the legacy flat-sequence proxy"
        ),
    )
    simulation.add_argument("--output")
    simulation.add_argument(
        "--cycle-output",
        help="optional full per-cycle JSONL diagnostics (keep outside Git)",
    )
    simulation.add_argument(
        "--eager-output",
        help="optional full per-eager-proposal JSONL diagnostics (keep outside Git)",
    )
    simulation.add_argument(
        "--allocation-output",
        help="optional full per-allocation-opportunity JSONL diagnostics (keep outside Git)",
    )

    compare = subparsers.add_parser("compare", help="compare all Phase-A policies")
    compare.add_argument("--workload", required=True)
    compare.add_argument("--config", required=True)
    compare.add_argument("--output")

    knee = subparsers.add_parser(
        "capacity-knee", help="run a proxy capacity-knee policy sweep"
    )
    knee.add_argument(
        "--workload",
        action="append",
        nargs=2,
        metavar=("TIME_SCALE", "PATH"),
        required=True,
    )
    knee.add_argument("--config", required=True)
    knee.add_argument("--output", required=True)

    eager_grid = subparsers.add_parser(
        "eager-grid", help="run the complete guarded-eager sensitivity grid"
    )
    eager_grid.add_argument("--workload", required=True)
    eager_grid.add_argument("--config", required=True)
    eager_grid.add_argument("--output", required=True)

    phase2_replay = subparsers.add_parser(
        "phase2-replay",
        help="run diagnostic-only common-snapshot oracle headroom replay",
        description=(
            "Replay diagnostic-only structural oracle headroom on common snapshots. "
            "Search cost is metadata-only, all ratios assume fully hidden search, and "
            "the output is not a deployable measured result."
        ),
    )
    phase2_replay.add_argument("--workload", required=True)
    phase2_replay.add_argument("--config", required=True)
    phase2_replay.add_argument("--sample-size", type=int, default=10_000)
    phase2_replay.add_argument("--output", required=True)
    phase2_replay.add_argument(
        "--snapshot-output",
        help="optional compact reproducible snapshot JSONL or JSONL.GZ outside Git",
    )

    phase2_simulate = subparsers.add_parser(
        "phase2-simulate",
        help="run one diagnostic-only fully-hidden-search system upper bound",
        description=(
            "Run a diagnostic-only end-to-end system upper bound. Oracle variants "
            "may leak target outcomes, large-pool search cost is not charged, and the "
            "output is not a deployable measured result."
        ),
    )
    phase2_simulate.add_argument("--workload", required=True)
    phase2_simulate.add_argument("--config", required=True)
    phase2_simulate.add_argument(
        "--variant", choices=tuple(PHASE2_VARIANTS.values()), required=True
    )
    phase2_simulate.add_argument(
        "--search-ratio", choices=SEARCH_RATIOS, type=int, required=True
    )
    phase2_simulate.add_argument("--output", required=True)

    gpu_probe = subparsers.add_parser(
        "gpu-probe", help="write read-only CUDA/NVIDIA environment metadata"
    )
    gpu_probe.add_argument("--output")
    gpu_probe.add_argument(
        "--allow-unavailable",
        action="store_true",
        help="return success after recording explicit no-CUDA errors",
    )

    tp_check = subparsers.add_parser(
        "tp-check", help="validate structural and Transformers TP compatibility"
    )
    tp_check.add_argument("--model-config", required=True)
    tp_check.add_argument("--tp-sizes", nargs="+", type=int, default=(1, 2, 3, 4))
    tp_check.add_argument("--output")

    phase3_run = subparsers.add_parser(
        "phase3-run",
        help="collect draft-only, target-only, or serial real-model traces",
        description=(
            "Run the isolated Phase-3 correctness collector. dry-run emits no GPU timing; "
            "Transformers mode is not a serving-engine performance benchmark."
        ),
    )
    phase3_run.add_argument("--config", required=True)
    phase3_run.add_argument(
        "--mode", choices=("draft-only", "target-only", "serial"), required=True
    )
    phase3_run.add_argument("--input", required=True)
    phase3_run.add_argument("--output-dir", required=True)
    phase3_run.add_argument("--resume", action="store_true")
    phase3_run.add_argument(
        "--max-cycles",
        type=int,
        help="optional interruption-test limit on newly completed cycles",
    )
    phase3_run.add_argument(
        "--environment-metadata",
        help="optional gpu-probe JSON to bind into the run manifest",
    )
    phase3_run.add_argument("--backend", choices=("dry-run", "transformers"))
    phase3_run.add_argument("--draft-model")
    phase3_run.add_argument("--target-model")
    phase3_run.add_argument("--draft-gpus")
    phase3_run.add_argument("--target-gpus")
    phase3_run.add_argument("--draft-tp", type=int)
    phase3_run.add_argument("--target-tp", type=int)
    phase3_run.add_argument("--dtype", choices=("float16", "bfloat16", "float32"))
    phase3_run.add_argument("--context-length", type=int)
    phase3_run.add_argument("--batch-size", type=int)
    phase3_run.add_argument("--search-pool-size", type=int)
    phase3_run.add_argument("--candidate-budget", type=int)
    phase3_run.add_argument("--max-new-tokens", type=int)

    phase3_validate = subparsers.add_parser(
        "phase3-validate", help="validate durable real-trace checkpoints"
    )
    phase3_validate.add_argument("--trace-dir", required=True)
    phase3_validate.add_argument(
        "--target-only-dir",
        help="also require final tokens to equal a target-only reference",
    )
    phase3_validate.add_argument("--output")

    phase3_summary = subparsers.add_parser(
        "phase3-summarize", help="consolidate checkpoints and write a compact summary"
    )
    phase3_summary.add_argument("--trace-dir", required=True)
    phase3_summary.add_argument("--trace-output")
    phase3_summary.add_argument("--output", required=True)

    phase3_benchmark = subparsers.add_parser(
        "phase3-benchmark",
        help=(
            "measure hf_correctness primitives with real CUDA; not a serving engine or "
            "simulator surface"
        ),
        description=(
            "Collect Phase-3B.1 correctness-backend primitive latency with strict per-rank "
            "evidence. Verification is serial full-context replay without KV-cache reuse or "
            "packed-tree execution. There is no dry-run timing fallback."
        ),
    )
    phase3_benchmark.add_argument("--config", required=True)
    phase3_benchmark.add_argument(
        "--operation",
        action="append",
        choices=("draft", "select", "verify", "transfer"),
        required=True,
    )
    phase3_benchmark.add_argument("--output", required=True)
    phase3_benchmark.add_argument("--markdown-output", required=True)
    phase3_benchmark.add_argument("--environment-metadata")

    phase3_benchmark_validate = subparsers.add_parser(
        "phase3-benchmark-validate",
        help="strictly validate a Phase-3B.1 primitive-latency report",
    )
    phase3_benchmark_validate.add_argument("--input", required=True)
    phase3_benchmark_validate.add_argument("--output", required=True)

    phase3_benchmark_compare = subparsers.add_parser(
        "phase3-benchmark-compare",
        help="compare repeated same-commit, same-semantics Phase-3B.1 runs",
    )
    phase3_benchmark_compare.add_argument(
        "--input", action="append", required=True, help="repeat for each run JSON"
    )
    phase3_benchmark_compare.add_argument("--output", required=True)
    phase3_benchmark_compare.add_argument("--markdown-output", required=True)

    phase3_selector_dry_run = subparsers.add_parser(
        "phase3-selector-dry-run",
        help="exercise the real-selector stage contract without recording fake latency",
    )
    phase3_selector_dry_run.add_argument("--request-count", type=int, default=2)
    phase3_selector_dry_run.add_argument("--search-pool-size", type=int, default=16)
    phase3_selector_dry_run.add_argument("--candidate-budget", type=int, default=8)
    phase3_selector_dry_run.add_argument("--output", required=True)

    phase3c_workload = subparsers.add_parser(
        "phase3c-workload-build",
        help="build deterministic R3-real public-text requests with real tokenizer lengths",
    )
    phase3c_workload.add_argument("--config", required=True)
    phase3c_workload.add_argument("--output", required=True)
    phase3c_workload.add_argument("--manifest", required=True)
    phase3c_workload.add_argument("--request-count", type=int)
    phase3c_workload.add_argument("--backend", choices=("dry-run", "transformers"))

    phase3c_draft = subparsers.add_parser(
        "phase3c-draft-forest",
        help="generate one shared nested 1x/2x/4x correctness candidate forest",
    )
    phase3c_draft.add_argument("--config", required=True)
    phase3c_draft.add_argument("--workload", required=True)
    phase3c_draft.add_argument("--output-dir", required=True)
    phase3c_draft.add_argument("--resume", action="store_true")
    phase3c_draft.add_argument("--backend", choices=("dry-run", "transformers"))

    phase3c_target = subparsers.add_parser(
        "phase3c-target-trajectory",
        help="generate one immutable greedy target trajectory per R3-real request",
    )
    phase3c_target.add_argument("--config", required=True)
    phase3c_target.add_argument("--workload", required=True)
    phase3c_target.add_argument("--output-dir", required=True)
    phase3c_target.add_argument("--resume", action="store_true")
    phase3c_target.add_argument("--backend", choices=("dry-run", "transformers"))

    phase3c_join = subparsers.add_parser(
        "phase3c-label-join",
        help="join immutable draft-side features with target-only evaluation labels",
    )
    phase3c_join.add_argument("--workload", required=True)
    phase3c_join.add_argument("--forest-dir", required=True)
    phase3c_join.add_argument("--target-dir", required=True)
    phase3c_join.add_argument("--output-dir", required=True)
    phase3c_join.add_argument("--resume", action="store_true")

    phase3c_replay = subparsers.add_parser(
        "phase3c-selector-replay",
        help="replay fixed-budget target-blind selectors and a within-request oracle",
    )
    phase3c_replay.add_argument("--config", required=True)
    phase3c_replay.add_argument("--labeled-dir", required=True)
    phase3c_replay.add_argument("--output-dir", required=True)
    phase3c_replay.add_argument("--resume", action="store_true")

    phase3c_validate = subparsers.add_parser(
        "phase3c-validate",
        help="validate Phase-3C forest, target, label, and token semantics",
    )
    phase3c_validate.add_argument("--workload", required=True)
    phase3c_validate.add_argument("--forest-dir", required=True)
    phase3c_validate.add_argument("--target-dir", required=True)
    phase3c_validate.add_argument("--labeled-dir", required=True)
    phase3c_validate.add_argument("--selector-dir", required=True)
    phase3c_validate.add_argument("--output", required=True)

    phase3c_summary = subparsers.add_parser(
        "phase3c-summary",
        help="summarize selector learnability without latency, SLO, or speedup claims",
    )
    phase3c_summary.add_argument("--labeled-dir", required=True)
    phase3c_summary.add_argument("--selector-dir", required=True)
    phase3c_summary.add_argument("--source-trace-commit")
    phase3c_summary.add_argument("--workload-manifest")
    phase3c_summary.add_argument("--draft-dir")
    phase3c_summary.add_argument("--target-dir")
    phase3c_summary.add_argument("--output", required=True)
    phase3c_summary.add_argument("--markdown-output", required=True)

    phase3c_resummary = subparsers.add_parser(
        "phase3c-resummary",
        help="migrate existing immutable Phase-3C raw artifacts to the v2 coverage summary",
    )
    phase3c_resummary.add_argument("--labeled-dir", required=True)
    phase3c_resummary.add_argument("--selector-dir", required=True)
    phase3c_resummary.add_argument("--draft-dir", required=True)
    phase3c_resummary.add_argument("--target-dir", required=True)
    phase3c_resummary.add_argument("--source-trace-commit", required=True)
    phase3c_resummary.add_argument("--workload-manifest")
    phase3c_resummary.add_argument("--output", required=True)
    phase3c_resummary.add_argument("--markdown-output", required=True)

    phase3c_snapshots = subparsers.add_parser(
        "phase3c-multiround-snapshots",
        help="build selector-independent forests at every frozen target-prefix position",
    )
    phase3c_snapshots.add_argument("--config", required=True)
    phase3c_snapshots.add_argument("--workload", required=True)
    phase3c_snapshots.add_argument("--target-dir", required=True)
    phase3c_snapshots.add_argument("--output-dir", required=True)
    phase3c_snapshots.add_argument("--resume", action="store_true")
    phase3c_snapshots.add_argument("--backend", choices=("dry-run", "transformers"))

    phase3c_sequential = subparsers.add_parser(
        "phase3c-multiround-replay",
        help="offline sequential selector replay over immutable common-prefix snapshots",
    )
    phase3c_sequential.add_argument("--workload", required=True)
    phase3c_sequential.add_argument("--target-dir", required=True)
    phase3c_sequential.add_argument("--snapshot-dir", required=True)
    phase3c_sequential.add_argument("--output-dir", required=True)
    phase3c_sequential.add_argument("--resume", action="store_true")

    phase3c_multi_summary = subparsers.add_parser(
        "phase3c-multiround-summary",
        help="summarize multi-round selector accounting and headroom without performance claims",
    )
    phase3c_multi_summary.add_argument("--snapshot-dir", required=True)
    phase3c_multi_summary.add_argument("--sequential-dir", required=True)
    phase3c_multi_summary.add_argument("--source-trace-commit", required=True)
    phase3c_multi_summary.add_argument("--output", required=True)
    phase3c_multi_summary.add_argument("--markdown-output", required=True)

    phase3c_multi_validate = subparsers.add_parser(
        "phase3c-multiround-validate",
        help="validate corrected workload, immutable targets, snapshots, and replay",
    )
    phase3c_multi_validate.add_argument("--workload", required=True)
    phase3c_multi_validate.add_argument("--workload-manifest", required=True)
    phase3c_multi_validate.add_argument("--target-dir", required=True)
    phase3c_multi_validate.add_argument("--snapshot-dir", required=True)
    phase3c_multi_validate.add_argument("--sequential-dir", required=True)
    phase3c_multi_validate.add_argument(
        "--expected-request-count", type=int, default=100
    )
    phase3c_multi_validate.add_argument("--output", required=True)

    phase3c_learned = subparsers.add_parser(
        "phase3c-learned-pilot",
        help=(
            "train and replay the diagnostic runtime-feature-only 2x shell ranker; "
            "does not report GPU performance"
        ),
    )
    phase3c_learned.add_argument("--workload", required=True)
    phase3c_learned.add_argument("--target-dir", required=True)
    phase3c_learned.add_argument("--snapshot-dir", required=True)
    phase3c_learned.add_argument("--sequential-dir", required=True)
    phase3c_learned.add_argument("--output-dir", required=True)
    phase3c_learned.add_argument("--source-trace-commit", required=True)
    phase3c_learned.add_argument("--seed", type=int, default=1664)
    phase3c_learned.add_argument("--resume", action="store_true")
    phase3c_learned.add_argument("--output", required=True)
    phase3c_learned.add_argument("--markdown-output", required=True)

    phase4_contract = subparsers.add_parser(
        "phase4-contract-dry-run",
        help="exercise fake Phase-4 adapter contracts without creating a GPU result",
    )
    phase4_contract.add_argument("--output", required=True)

    phase4_dual_contract = subparsers.add_parser(
        "phase4-dual-contract-dry-run",
        help="exercise Phase-4B state/queue contracts without CUDA or vLLM",
    )
    phase4_dual_contract.add_argument("--output", required=True)

    phase4_decode_ready_contract = subparsers.add_parser(
        "phase4-decode-ready-contract-dry-run",
        help="validate ResidentWarmStart contracts without CUDA or vLLM",
    )
    phase4_decode_ready_contract.add_argument("--output", required=True)

    phase4_gate_a_validate = subparsers.add_parser(
        "phase4-gate-a-validate",
        help="validate A-waiting/B-prefill admissibility and owned-process cleanup",
    )
    phase4_gate_a_validate.add_argument("--scheduler-events", required=True)
    phase4_gate_a_validate.add_argument("--lifecycle")
    phase4_gate_a_validate.add_argument("--waiting-request-id", required=True)
    phase4_gate_a_validate.add_argument("--prefill-request-id", required=True)
    phase4_gate_a_validate.add_argument("--output", required=True)

    phase4_probe = subparsers.add_parser(
        "phase4-probe",
        help="validate the frozen vLLM environment and three-GPU topology",
        description=(
            "Probe the independent Python 3.11/vLLM environment. This command exits "
            "nonzero without CUDA and never substitutes synthetic GPU metadata."
        ),
    )
    phase4_probe.add_argument("--config", required=True)
    phase4_probe.add_argument("--vllm-source", required=True)
    phase4_probe.add_argument("--environment-output", required=True)
    phase4_probe.add_argument("--topology-output", required=True)
    phase4_probe.add_argument("--validation-output", required=True)

    phase4_smoke = subparsers.add_parser(
        "phase4-stock-smoke",
        help="run a real stock-vLLM draft TP1 or target TP2 bring-up",
        description=(
            "Run one independent stock vLLM engine twice on five corrected R3-real "
            "requests. This is bring-up, not serving-performance evaluation."
        ),
    )
    phase4_smoke.add_argument("--config", required=True)
    phase4_smoke.add_argument("--role", choices=("draft", "target"), required=True)
    phase4_smoke.add_argument("--workload", required=True)
    phase4_smoke.add_argument("--environment", required=True)
    phase4_smoke.add_argument("--topology", required=True)
    phase4_smoke.add_argument("--runtime-manifest", required=True)
    phase4_smoke.add_argument("--frozen-hf-target-dir")
    phase4_smoke.add_argument(
        "--correctness-mode",
        choices=("default", "batch-invariant"),
        default="default",
    )
    phase4_smoke.add_argument("--target-diagnostics")
    phase4_smoke.add_argument("--request-count", type=int)
    phase4_smoke.add_argument("--output", required=True)

    phase4_validate = subparsers.add_parser(
        "phase4-validate",
        help="validate Phase-4A.0 stock-vLLM bring-up artifacts",
    )
    phase4_validate.add_argument("--config", required=True)
    phase4_validate.add_argument("--environment", required=True)
    phase4_validate.add_argument("--topology", required=True)
    phase4_validate.add_argument("--runtime-manifest", required=True)
    phase4_validate.add_argument("--draft-smoke", required=True)
    phase4_validate.add_argument("--target-smoke", required=True)
    phase4_validate.add_argument("--output", required=True)
    phase4_validate.add_argument("--markdown-output", required=True)

    phase4_reference = subparsers.add_parser(
        "phase4-stock-reference",
        help="generate and immutably freeze the stock-vLLM Target-only reference",
    )
    phase4_reference.add_argument("--config", required=True)
    phase4_reference.add_argument("--workload", required=True)
    phase4_reference.add_argument("--environment", required=True)
    phase4_reference.add_argument("--topology", required=True)
    phase4_reference.add_argument("--runtime-manifest", required=True)
    phase4_reference.add_argument("--legacy-hf-target-dir")
    phase4_reference.add_argument(
        "--correctness-mode",
        choices=("default", "batch-invariant"),
        default="default",
    )
    phase4_reference.add_argument("--target-diagnostics")
    phase4_reference.add_argument("--request-count", type=int)
    phase4_reference.add_argument("--output", required=True)

    phase4_regression = subparsers.add_parser(
        "phase4-target-regression",
        help="prove patched vLLM target-only output equals the frozen stock reference",
    )
    phase4_regression.add_argument("--config", required=True)
    phase4_regression.add_argument("--workload", required=True)
    phase4_regression.add_argument("--environment", required=True)
    phase4_regression.add_argument("--topology", required=True)
    phase4_regression.add_argument("--runtime-manifest", required=True)
    phase4_regression.add_argument("--reference", required=True)
    phase4_regression.add_argument("--patch-manifest", required=True)
    phase4_regression.add_argument("--legacy-hf-target-dir")
    phase4_regression.add_argument(
        "--correctness-mode",
        choices=("default", "batch-invariant"),
        default="default",
    )
    phase4_regression.add_argument("--target-diagnostics")
    phase4_regression.add_argument("--output", required=True)

    phase4_draft_service = subparsers.add_parser(
        "phase4-draft-service",
        help="run the persistent GPU-0 Draft proposer over a local Unix socket",
    )
    phase4_draft_service.add_argument("--config", required=True)
    phase4_draft_service.add_argument("--socket", required=True)
    phase4_draft_service.add_argument("--event-log", required=True)
    phase4_draft_service.add_argument("--ready", required=True)

    phase4_serial = subparsers.add_parser(
        "phase4-serial-run",
        help="run one 1D+2V strict-serial GPU correctness pass",
    )
    phase4_serial.add_argument("--config", required=True)
    phase4_serial.add_argument("--workload", required=True)
    phase4_serial.add_argument("--environment", required=True)
    phase4_serial.add_argument("--topology", required=True)
    phase4_serial.add_argument("--runtime-manifest", required=True)
    phase4_serial.add_argument("--reference", required=True)
    phase4_serial.add_argument("--patch-manifest", required=True)
    phase4_serial.add_argument("--draft-socket", required=True)
    phase4_serial.add_argument("--draft-ready", required=True)
    phase4_serial.add_argument("--round-events", required=True)
    phase4_serial.add_argument("--transport-events", required=True)
    phase4_serial.add_argument("--plugin-report", required=True)
    phase4_serial.add_argument(
        "--correctness-mode",
        choices=("default", "batch-invariant"),
        default="default",
    )
    phase4_serial.add_argument("--target-diagnostics")
    phase4_serial.add_argument("--output", required=True)

    phase4_serial_validate = subparsers.add_parser(
        "phase4-serial-validate",
        help="validate patched Target regression and two strict-serial runs",
    )
    phase4_serial_validate.add_argument("--config", required=True)
    phase4_serial_validate.add_argument("--reference", required=True)
    phase4_serial_validate.add_argument("--patch-manifest", required=True)
    phase4_serial_validate.add_argument("--target-regression", required=True)
    phase4_serial_validate.add_argument("--run", action="append", required=True)
    phase4_serial_validate.add_argument("--round-events", action="append", required=True)
    phase4_serial_validate.add_argument(
        "--transport-events", action="append", required=True
    )
    phase4_serial_validate.add_argument("--output", required=True)
    phase4_serial_validate.add_argument("--markdown-output", required=True)

    phase4_dual_service = subparsers.add_parser(
        "phase4-dual-draft-service",
        help="run the asynchronous persistent GPU-0 Phase-4B Draft service",
    )
    phase4_dual_service.add_argument("--config", required=True)
    phase4_dual_service.add_argument("--socket", required=True)
    phase4_dual_service.add_argument("--event-log", required=True)
    phase4_dual_service.add_argument("--transport-events", required=True)
    phase4_dual_service.add_argument("--ready", required=True)

    phase4_dual_run = subparsers.add_parser(
        "phase4-dual-batch-run",
        help=(
            "run 1D+2V Dual-Batch GPU correctness/overlap-existence collection; "
            "does not report serving performance"
        ),
    )
    phase4_dual_run.add_argument("--config", required=True)
    phase4_dual_run.add_argument("--workload", required=True)
    phase4_dual_run.add_argument("--request-count", type=int, required=True)
    phase4_dual_run.add_argument("--environment", required=True)
    phase4_dual_run.add_argument("--topology", required=True)
    phase4_dual_run.add_argument("--runtime-manifest", required=True)
    phase4_dual_run.add_argument("--reference", required=True)
    phase4_dual_run.add_argument("--patch-manifest", required=True)
    phase4_dual_run.add_argument("--draft-socket", required=True)
    phase4_dual_run.add_argument("--draft-ready", required=True)
    phase4_dual_run.add_argument("--scheduler-events", required=True)
    phase4_dual_run.add_argument("--request-state-events", required=True)
    phase4_dual_run.add_argument("--proposal-events", required=True)
    phase4_dual_run.add_argument("--verification-events", required=True)
    phase4_dual_run.add_argument("--draft-work-events", required=True)
    phase4_dual_run.add_argument("--transport-events", required=True)
    phase4_dual_run.add_argument("--target-diagnostics", required=True)
    phase4_dual_run.add_argument("--plugin-report", required=True)
    phase4_dual_run.add_argument("--output-checkpoint", required=True)
    phase4_dual_run.add_argument("--cycle-events", required=True)
    phase4_dual_run.add_argument("--overlap-events", required=True)
    phase4_dual_run.add_argument("--microbatch-size", type=int, default=1)
    phase4_dual_run.add_argument("--cohort-size", type=int)
    phase4_dual_run.add_argument("--resume", action="store_true")
    phase4_dual_run.add_argument("--output", required=True)

    phase4_dual_validate = subparsers.add_parser(
        "phase4-dual-batch-validate",
        help="read-only validation of two Phase-4B Dual-Batch correctness runs",
    )
    phase4_dual_validate.add_argument(
        "--stock-reference", action="append", required=True
    )
    phase4_dual_validate.add_argument("--target-regression", required=True)
    phase4_dual_validate.add_argument("--run", action="append", required=True)
    phase4_dual_validate.add_argument(
        "--request-state-events", action="append", required=True
    )
    phase4_dual_validate.add_argument(
        "--proposal-events", action="append", required=True
    )
    phase4_dual_validate.add_argument(
        "--cycle-events", action="append", required=True
    )
    phase4_dual_validate.add_argument(
        "--overlap-events", action="append", required=True
    )
    phase4_dual_validate.add_argument(
        "--draft-work-events", action="append", required=True
    )
    phase4_dual_validate.add_argument(
        "--target-diagnostics", action="append", required=True
    )
    phase4_dual_validate.add_argument("--output", required=True)
    phase4_dual_validate.add_argument("--markdown-output", required=True)

    phase4b1_dual_run = subparsers.add_parser(
        "phase4b1-resident-dual-run",
        help=(
            "run real decode-only resident Dual-Batch exact-correctness collection; "
            "never reports performance"
        ),
    )
    for name in (
        "config",
        "workload",
        "environment",
        "topology",
        "patch-manifest",
        "draft-socket",
        "draft-ready",
        "context",
        "decode-ready-manifest",
        "timing-events",
        "setup-control",
        "setup-ready",
        "scheduler-events",
        "request-state-events",
        "proposal-events",
        "proposal-lifecycle-events",
        "verification-events",
        "draft-work-events",
        "transport-events",
        "target-diagnostics",
        "plugin-report",
        "output-checkpoint",
        "cycle-events",
        "overlap-events",
        "runtime-manifest",
        "output",
    ):
        phase4b1_dual_run.add_argument(f"--{name}", required=True)
    phase4b1_dual_run.add_argument(
        "--request-count", type=int, choices=(2, 5, 100), required=True
    )
    phase4b1_dual_run.add_argument("--microbatch-size", type=int, default=2)
    phase4b1_dual_run.add_argument(
        "--test-coordination",
        choices=("none", "one-ready", "two-ready"),
        default="none",
    )
    phase4b1_dual_run.add_argument(
        "--overlap-requirement",
        choices=("required", "separate-gate"),
        default="required",
        help="keep physical overlap mandatory or report it for a separate gate",
    )

    phase4b1_validate = subparsers.add_parser(
        "phase4b1-dual-correctness-validate",
        help="read-only exact Target/Serial/Dual decode-only triangle validation",
    )
    phase4b1_validate.add_argument("--target", required=True)
    phase4b1_validate.add_argument("--serial", required=True)
    phase4b1_validate.add_argument("--dual", action="append", required=True)
    phase4b1_validate.add_argument("--target-manifest", required=True)
    phase4b1_validate.add_argument("--serial-manifest", required=True)
    phase4b1_validate.add_argument("--target-process-lifecycle", required=True)
    phase4b1_validate.add_argument("--serial-process-lifecycle", required=True)
    for name in (
        "dual-manifest",
        "request-state-events",
        "proposal-events",
        "proposal-lifecycle-events",
        "scheduler-events",
        "verification-events",
        "draft-work-events",
        "target-diagnostics",
        "overlap-events",
        "process-lifecycle",
    ):
        phase4b1_validate.add_argument(f"--{name}", action="append", required=True)
    phase4b1_validate.add_argument("--output", required=True)
    phase4b1_validate.add_argument("--markdown-output", required=True)
    phase4b1_validate.add_argument(
        "--overlap-requirement",
        choices=("required", "separate-gate"),
        default="required",
    )
    phase4b1_validate.add_argument(
        "--legacy-source-commit",
        help="explicit immutable source commit for supported read-only revalidation",
    )

    phase4b1_overlap_diagnose = subparsers.add_parser(
        "phase4b1-overlap-diagnose",
        help="read-only nearest Draft/Verify interval diagnosis",
    )
    phase4b1_overlap_diagnose.add_argument(
        "--draft-work-events", action="append", required=True
    )
    phase4b1_overlap_diagnose.add_argument(
        "--verification-events", action="append", required=True
    )
    phase4b1_overlap_diagnose.add_argument(
        "--overlap-events", action="append", required=True
    )
    phase4b1_overlap_diagnose.add_argument("--output", required=True)

    phase4b1_controlled = subparsers.add_parser(
        "phase4b1-dual-controlled-validate",
        help="validate controlled one-wait/two-ready/terminal Gate-1 evidence",
    )
    phase4b1_controlled.add_argument("--asynchronous-scheduler", required=True)
    phase4b1_controlled.add_argument("--coordinated-scheduler", required=True)
    phase4b1_controlled.add_argument("--request-state-events", required=True)
    phase4b1_controlled.add_argument("--output", required=True)

    phase4_resident_target = subparsers.add_parser(
        "phase4-resident-target-run",
        help="run the Phase-4B.0b real-KV decode-only Target correctness gate",
    )
    for name in (
        "config",
        "workload",
        "environment",
        "topology",
        "reference",
        "patch-manifest",
        "draft-socket",
        "draft-ready",
        "context",
        "decode-ready-manifest",
        "timing-events",
        "setup-control",
        "setup-ready",
        "admission-events",
        "target-diagnostics",
        "plugin-report",
        "first-forward",
        "output",
    ):
        phase4_resident_target.add_argument(f"--{name}", required=True)
    phase4_resident_target.add_argument(
        "--request-count", type=int, choices=(2, 5, 100), required=True
    )
    phase4_resident_target.add_argument(
        "--correctness-mode",
        choices=("default", "batch-invariant"),
        default="batch-invariant",
    )

    phase4_resident_serial = subparsers.add_parser(
        "phase4-resident-serial-run",
        help="run the Phase-4B.0b real-KV decode-only Serial correctness gate",
    )
    for name in (
        "config",
        "workload",
        "environment",
        "topology",
        "runtime-manifest",
        "reference",
        "patch-manifest",
        "draft-socket",
        "draft-ready",
        "round-events",
        "transport-events",
        "plugin-report",
        "context",
        "decode-ready-manifest",
        "timing-events",
        "setup-control",
        "setup-ready",
        "admission-events",
        "initial-proposal-events",
        "target-diagnostics",
        "first-forward",
        "output",
    ):
        phase4_resident_serial.add_argument(f"--{name}", required=True)
    phase4_resident_serial.add_argument(
        "--request-count", type=int, choices=(2, 5, 100), required=True
    )
    phase4_resident_serial.add_argument(
        "--correctness-mode",
        choices=("default", "batch-invariant"),
        default="batch-invariant",
    )

    phase4_resident_validate = subparsers.add_parser(
        "phase4-resident-validate",
        help="compare decode-only Target and Serial resident correctness artifacts",
    )
    phase4_resident_validate.add_argument("--target", required=True)
    phase4_resident_validate.add_argument("--serial", required=True)
    phase4_resident_validate.add_argument("--target-manifest", required=True)
    phase4_resident_validate.add_argument("--serial-manifest", required=True)
    phase4_resident_validate.add_argument("--output", required=True)

    phase4_bi_probe = subparsers.add_parser(
        "phase4-batch-invariant-preflight",
        help="fail-closed hardware preflight before creating any vLLM worker",
    )
    phase4_bi_probe.add_argument(
        "--correctness-mode",
        choices=("default", "batch-invariant"),
        default="batch-invariant",
    )
    phase4_bi_probe.add_argument("--output", required=True)

    phase4_bi_validate = subparsers.add_parser(
        "phase4-batch-invariant-validate",
        help="validate two independent C stock and D Serial correctness runs",
    )
    phase4_bi_validate.add_argument("--stock-reference", action="append", required=True)
    phase4_bi_validate.add_argument(
        "--target-regression", action="append", required=True
    )
    phase4_bi_validate.add_argument("--serial-run", action="append", required=True)
    phase4_bi_validate.add_argument("--round-events", action="append", required=True)
    phase4_bi_validate.add_argument(
        "--target-diagnostics", action="append", required=True
    )
    phase4_bi_validate.add_argument(
        "--serial-diagnostics", action="append", required=True
    )
    phase4_bi_validate.add_argument("--output", required=True)
    phase4_bi_validate.add_argument("--markdown-output", required=True)

    phase4_fixed_service = subparsers.add_parser(
        "phase4-fixed-proposal-service",
        help="serve the fixed diagnostic proposal over a local Unix socket",
    )
    phase4_fixed_service.add_argument("--socket", required=True)

    phase4_fixed_run = subparsers.add_parser(
        "phase4-fixed-control-run",
        help="run one single-request K=1/2/4 local or remote fixed-proposal control",
    )
    phase4_fixed_run.add_argument("--config", required=True)
    phase4_fixed_run.add_argument("--workload", required=True)
    phase4_fixed_run.add_argument("--environment", required=True)
    phase4_fixed_run.add_argument("--topology", required=True)
    phase4_fixed_run.add_argument("--patch-manifest", required=True)
    phase4_fixed_run.add_argument(
        "--proposer", choices=("local-static", "remote-fixed"), required=True
    )
    phase4_fixed_run.add_argument(
        "--proposal-budget", type=int, choices=(1, 2, 4), required=True
    )
    phase4_fixed_run.add_argument("--remote-socket")
    phase4_fixed_run.add_argument("--target-diagnostics", required=True)
    phase4_fixed_run.add_argument("--output", required=True)

    phase4_fixed_validate = subparsers.add_parser(
        "phase4-fixed-control-validate",
        help="compare local-static and remote-fixed controls for K=1/2/4",
    )
    phase4_fixed_validate.add_argument("--local-run", action="append", required=True)
    phase4_fixed_validate.add_argument("--remote-run", action="append", required=True)
    phase4_fixed_validate.add_argument(
        "--local-diagnostics", action="append", required=True
    )
    phase4_fixed_validate.add_argument(
        "--remote-diagnostics", action="append", required=True
    )
    phase4_fixed_validate.add_argument("--output", required=True)

    phase4_divergence = subparsers.add_parser(
        "phase4-divergence-diagnose",
        help="prove prefix/logits/position/KV mapping at the first C/D divergence",
    )
    phase4_divergence.add_argument("--stock-diagnostics", required=True)
    phase4_divergence.add_argument("--serial-diagnostics", required=True)
    phase4_divergence.add_argument("--serial-run", required=True)
    phase4_divergence.add_argument("--output", required=True)
    return parser


def _gpu_ids(value: Optional[str]) -> Optional[tuple[int, ...]]:
    if value is None:
        return None
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise SystemExit("GPU IDs must be comma-separated integers") from error
    if not result:
        raise SystemExit("GPU ID list must not be empty")
    return result


def _current_git_commit() -> Optional[str]:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _phase3c_config(path: str, backend: Optional[str] = None) -> Any:
    from dataclasses import replace

    from specrhythm.phase3.phase3c_config import load_phase3c_config

    config = load_phase3c_config(path)
    if backend is not None:
        config = replace(config, runtime=config.runtime.with_overrides(backend=backend))
    return config


def _final_tokens(trace_dir: Path) -> dict[str, tuple[int, ...]]:
    from specrhythm.phase3.trace import TraceStore

    result = {}
    store = TraceStore(trace_dir)
    for request_id in {record.request.request_id for record in store.records()}:
        _, generated, finished = store.resume_state(request_id)
        if not finished:
            raise ValueError(f"request {request_id} is incomplete")
        result[request_id] = generated
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "phase4-contract-dry-run":
        from specrhythm.phase4.fake import run_fake_contract

        _write_json(run_fake_contract(), args.output)
        return 0
    if args.command == "phase4-dual-contract-dry-run":
        from specrhythm.phase4.dual import run_dual_contract_dry_run

        _write_json(run_dual_contract_dry_run(), args.output)
        return 0
    if args.command == "phase4-decode-ready-contract-dry-run":
        from specrhythm.phase4.decode_ready import run_decode_ready_contract_dry_run

        _write_json(run_decode_ready_contract_dry_run(), args.output)
        return 0
    if args.command == "phase4-gate-a-validate":
        from specrhythm.phase4.admissibility import validate_gate_a_construction
        from specrhythm.phase4.transport import CheckpointJsonl

        lifecycle = (
            json.loads(Path(args.lifecycle).read_text(encoding="utf-8"))
            if args.lifecycle
            else None
        )
        report = validate_gate_a_construction(
            CheckpointJsonl(Path(args.scheduler_events).resolve()).read(),
            waiting_request_id=args.waiting_request_id,
            prefill_request_id=args.prefill_request_id,
            lifecycle=lifecycle,
        )
        _write_json(report, args.output)
        return 0 if report["valid"] else 1
    if args.command == "phase4-probe":
        from specrhythm.phase4.config import load_phase4_config
        from specrhythm.phase4.manifest import (
            collect_environment,
            collect_topology,
            validate_environment,
            validate_topology,
        )

        try:
            config = load_phase4_config(args.config)
            environment = collect_environment(Path(args.vllm_source).resolve())
            topology = collect_topology()
            environment_validation = validate_environment(environment, config)
            topology_validation = validate_topology(topology, config)
            validation = {
                "schema_version": "specrhythm.phase4-probe-validation.v1",
                "valid": environment_validation["valid"]
                and topology_validation["valid"],
                "environment": environment_validation,
                "topology": topology_validation,
                "serving_performance_result": False,
            }
        except (FileNotFoundError, ValueError) as error:
            raise SystemExit(f"Phase-4 probe failed: {error}") from error
        _write_json(environment, args.environment_output)
        _write_json(topology, args.topology_output)
        _write_json(validation, args.validation_output)
        return 0 if validation["valid"] else 2
    if args.command == "phase4-stock-smoke":
        from specrhythm.phase4.config import load_phase4_config
        from specrhythm.phase4.stock_vllm import run_stock_smoke

        try:
            report = run_stock_smoke(
                load_phase4_config(args.config),
                role=args.role,
                workload_path=Path(args.workload).resolve(),
                environment_path=Path(args.environment).resolve(),
                topology_path=Path(args.topology).resolve(),
                runtime_manifest_path=Path(args.runtime_manifest).resolve(),
                git_commit=_current_git_commit() or "unknown",
                frozen_target_dir=(
                    Path(args.frozen_hf_target_dir).resolve()
                    if args.frozen_hf_target_dir
                    else None
                ),
                correctness_mode=args.correctness_mode,
                diagnostics_path=(
                    Path(args.target_diagnostics).resolve()
                    if args.target_diagnostics
                    else None
                ),
                request_count=args.request_count,
            )
        except (
            FileExistsError,
            FileNotFoundError,
            ImportError,
            RuntimeError,
            ValueError,
        ) as error:
            raise SystemExit(f"Phase-4 stock smoke failed: {error}") from error
        _write_json(report, args.output)
        return 0
    if args.command == "phase4-validate":
        from specrhythm.phase4.config import load_phase4_config
        from specrhythm.phase4.validation import (
            validate_artifacts,
            validation_markdown,
        )

        try:
            report = validate_artifacts(
                load_phase4_config(args.config),
                environment_path=Path(args.environment).resolve(),
                topology_path=Path(args.topology).resolve(),
                runtime_manifest_path=Path(args.runtime_manifest).resolve(),
                draft_smoke_path=Path(args.draft_smoke).resolve(),
                target_smoke_path=Path(args.target_smoke).resolve(),
            )
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
            report = {
                "schema_version": "specrhythm.phase4-validation.v1",
                "valid": False,
                "errors": [str(error)],
                "warnings": [],
                "serving_performance_result": False,
            }
        _write_json(report, args.output)
        markdown = Path(args.markdown_output)
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(validation_markdown(report), encoding="utf-8")
        return 0 if report["valid"] else 1
    if args.command == "phase4-stock-reference":
        from specrhythm.phase4.config import load_phase4_config
        from specrhythm.phase4.reference import freeze_stock_reference

        try:
            freeze_stock_reference(
                Path(args.output).resolve(),
                load_phase4_config(args.config),
                workload_path=Path(args.workload).resolve(),
                environment_path=Path(args.environment).resolve(),
                topology_path=Path(args.topology).resolve(),
                runtime_manifest_path=Path(args.runtime_manifest).resolve(),
                git_commit=_current_git_commit() or "unknown",
                legacy_hf_target_dir=(
                    Path(args.legacy_hf_target_dir).resolve()
                    if args.legacy_hf_target_dir
                    else None
                ),
                correctness_mode=args.correctness_mode,
                diagnostics_path=(
                    Path(args.target_diagnostics).resolve()
                    if args.target_diagnostics
                    else None
                ),
                request_count=args.request_count,
            )
        except (
            FileExistsError,
            FileNotFoundError,
            ImportError,
            RuntimeError,
            ValueError,
        ) as error:
            raise SystemExit(f"Phase-4 stock reference failed: {error}") from error
        return 0
    if args.command == "phase4-target-regression":
        from specrhythm.phase4.config import load_phase4_config
        from specrhythm.phase4.serial_runner import run_patched_target_regression

        try:
            report = run_patched_target_regression(
                load_phase4_config(args.config),
                workload_path=Path(args.workload).resolve(),
                environment_path=Path(args.environment).resolve(),
                topology_path=Path(args.topology).resolve(),
                runtime_manifest_path=Path(args.runtime_manifest).resolve(),
                reference_path=Path(args.reference).resolve(),
                patch_manifest_path=Path(args.patch_manifest).resolve(),
                git_commit=_current_git_commit() or "unknown",
                legacy_hf_target_dir=(
                    Path(args.legacy_hf_target_dir).resolve()
                    if args.legacy_hf_target_dir
                    else None
                ),
                correctness_mode=args.correctness_mode,
                diagnostics_path=(
                    Path(args.target_diagnostics).resolve()
                    if args.target_diagnostics
                    else None
                ),
            )
        except (FileNotFoundError, ImportError, RuntimeError, ValueError) as error:
            raise SystemExit(f"Phase-4 Target regression failed: {error}") from error
        _write_json(report, args.output)
        return 0 if report["valid"] else 1
    if args.command == "phase4-draft-service":
        from specrhythm.phase4.config import load_phase4_config
        from specrhythm.phase4.draft_service import run_draft_service

        try:
            run_draft_service(
                load_phase4_config(args.config),
                socket_path=Path(args.socket).resolve(),
                event_log_path=Path(args.event_log).resolve(),
                ready_path=Path(args.ready).resolve(),
            )
        except (
            FileExistsError,
            FileNotFoundError,
            ImportError,
            RuntimeError,
            ValueError,
        ) as error:
            raise SystemExit(f"Phase-4 Draft service failed: {error}") from error
        return 0
    if args.command == "phase4-serial-run":
        from specrhythm.phase4.config import load_phase4_config
        from specrhythm.phase4.serial_runner import run_serial_disaggregated
        output_path = Path(args.output).resolve()
        if output_path.exists():
            raise SystemExit(f"refusing to overwrite Serial run artifact {output_path}")
        try:
            report = run_serial_disaggregated(
                load_phase4_config(args.config),
                workload_path=Path(args.workload).resolve(),
                environment_path=Path(args.environment).resolve(),
                topology_path=Path(args.topology).resolve(),
                runtime_manifest_path=Path(args.runtime_manifest).resolve(),
                reference_path=Path(args.reference).resolve(),
                patch_manifest_path=Path(args.patch_manifest).resolve(),
                draft_socket_path=Path(args.draft_socket).resolve(),
                draft_ready_path=Path(args.draft_ready).resolve(),
                round_events_path=Path(args.round_events).resolve(),
                transport_events_path=Path(args.transport_events).resolve(),
                plugin_report_path=Path(args.plugin_report).resolve(),
                git_commit=_current_git_commit() or "unknown",
                correctness_mode=args.correctness_mode,
                diagnostics_path=(
                    Path(args.target_diagnostics).resolve()
                    if args.target_diagnostics
                    else None
                ),
            )
        except (
            FileExistsError,
            FileNotFoundError,
            ImportError,
            RuntimeError,
            ValueError,
        ) as error:
            raise SystemExit(f"Phase-4 Serial run failed: {error}") from error
        _write_json(report, output_path)
        return 0 if report["valid"] else 1
    if args.command == "phase4-serial-validate":
        from specrhythm.phase4.config import load_phase4_config
        from specrhythm.phase4.serial_validation import (
            serial_summary_markdown,
            validate_serial_artifacts,
        )

        try:
            run_paths = [Path(path).resolve() for path in args.run]
            report = validate_serial_artifacts(
                load_phase4_config(args.config),
                reference_path=Path(args.reference).resolve(),
                patch_manifest_path=Path(args.patch_manifest).resolve(),
                target_regression_path=Path(args.target_regression).resolve(),
                run_paths=run_paths,
                round_event_paths=[Path(path).resolve() for path in args.round_events],
                transport_event_paths=[
                    Path(path).resolve() for path in args.transport_events
                ],
            )
            runs = [json.loads(path.read_text(encoding="utf-8")) for path in run_paths]
        except (FileNotFoundError, json.JSONDecodeError, RuntimeError, ValueError) as error:
            report = {
                "schema_version": "specrhythm.phase4a1-validation.v1",
                "valid": False,
                "errors": [str(error)],
                "warnings": [],
                "gpu_correctness_result": True,
                "gpu_performance_result": False,
            }
            runs = []
        _write_json(report, args.output)
        markdown = Path(args.markdown_output).resolve()
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(serial_summary_markdown(report, runs), encoding="utf-8")
        return 0 if report["valid"] else 1
    if args.command == "phase4-dual-draft-service":
        from specrhythm.phase4.config import load_phase4_config
        from specrhythm.phase4.dual_service import run_dual_draft_service

        try:
            run_dual_draft_service(
                load_phase4_config(args.config),
                socket_path=Path(args.socket).resolve(),
                event_log_path=Path(args.event_log).resolve(),
                transport_log_path=Path(args.transport_events).resolve(),
                ready_path=Path(args.ready).resolve(),
            )
        except (
            FileExistsError,
            FileNotFoundError,
            ImportError,
            RuntimeError,
            ValueError,
        ) as error:
            raise SystemExit(f"Phase-4B Draft service failed: {error}") from error
        return 0
    if args.command == "phase4-dual-batch-run":
        from specrhythm.phase4.config import load_phase4_config
        from specrhythm.phase4.dual_runner import run_dual_batch

        try:
            report = run_dual_batch(
                load_phase4_config(args.config),
                workload_path=Path(args.workload).resolve(),
                request_count=args.request_count,
                environment_path=Path(args.environment).resolve(),
                topology_path=Path(args.topology).resolve(),
                runtime_manifest_path=Path(args.runtime_manifest).resolve(),
                reference_path=Path(args.reference).resolve(),
                patch_manifest_path=Path(args.patch_manifest).resolve(),
                draft_socket_path=Path(args.draft_socket).resolve(),
                draft_ready_path=Path(args.draft_ready).resolve(),
                scheduler_events_path=Path(args.scheduler_events).resolve(),
                request_state_events_path=Path(args.request_state_events).resolve(),
                proposal_events_path=Path(args.proposal_events).resolve(),
                verification_events_path=Path(args.verification_events).resolve(),
                draft_work_events_path=Path(args.draft_work_events).resolve(),
                transport_events_path=Path(args.transport_events).resolve(),
                target_diagnostics_path=Path(args.target_diagnostics).resolve(),
                plugin_report_path=Path(args.plugin_report).resolve(),
                output_checkpoint_path=Path(args.output_checkpoint).resolve(),
                cycle_events_path=Path(args.cycle_events).resolve(),
                overlap_events_path=Path(args.overlap_events).resolve(),
                output_path=Path(args.output).resolve(),
                git_commit=_current_git_commit() or "unknown",
                microbatch_size=args.microbatch_size,
                cohort_size=args.cohort_size,
                resume=args.resume,
            )
        except (
            FileExistsError,
            FileNotFoundError,
            ImportError,
            RuntimeError,
            ValueError,
        ) as error:
            raise SystemExit(f"Phase-4B Dual-Batch run failed: {error}") from error
        return 0 if report.get("exact_sequence_match") is True else 1
    if args.command == "phase4-dual-batch-validate":
        from specrhythm.phase4.dual_validation import validate_dual_batch_runs

        try:
            report = validate_dual_batch_runs(
                stock_references=[Path(path).resolve() for path in args.stock_reference],
                target_regression_path=Path(args.target_regression).resolve(),
                run_paths=[Path(path).resolve() for path in args.run],
                state_event_paths=[
                    Path(path).resolve() for path in args.request_state_events
                ],
                proposal_event_paths=[
                    Path(path).resolve() for path in args.proposal_events
                ],
                cycle_event_paths=[Path(path).resolve() for path in args.cycle_events],
                overlap_event_paths=[
                    Path(path).resolve() for path in args.overlap_events
                ],
                draft_work_event_paths=[
                    Path(path).resolve() for path in args.draft_work_events
                ],
                target_diagnostic_paths=[
                    Path(path).resolve() for path in args.target_diagnostics
                ],
                output_path=Path(args.output).resolve(),
                markdown_path=Path(args.markdown_output).resolve(),
            )
        except (FileNotFoundError, json.JSONDecodeError, RuntimeError, ValueError) as error:
            report = {
                "schema_version": "specrhythm.phase4b-dual-batch-validation.v1",
                "valid": False,
                "outcome": "invalid-artifacts",
                "errors": [str(error)],
            }
            _write_json(report, args.output)
            return 1
        return 0 if report["valid"] else 1
    if args.command == "phase4b1-resident-dual-run":
        from specrhythm.phase4.config import load_phase4_config
        from specrhythm.phase4.dual_runner import run_resident_dual_batch

        try:
            report = run_resident_dual_batch(
                load_phase4_config(args.config),
                workload_path=Path(args.workload).resolve(),
                request_count=args.request_count,
                environment_path=Path(args.environment).resolve(),
                topology_path=Path(args.topology).resolve(),
                patch_manifest_path=Path(args.patch_manifest).resolve(),
                draft_socket_path=Path(args.draft_socket).resolve(),
                draft_ready_path=Path(args.draft_ready).resolve(),
                context_path=Path(args.context).resolve(),
                decode_ready_manifest_path=Path(args.decode_ready_manifest).resolve(),
                timing_events_path=Path(args.timing_events).resolve(),
                setup_control_path=Path(args.setup_control).resolve(),
                setup_ready_path=Path(args.setup_ready).resolve(),
                scheduler_events_path=Path(args.scheduler_events).resolve(),
                request_state_events_path=Path(args.request_state_events).resolve(),
                proposal_events_path=Path(args.proposal_events).resolve(),
                proposal_lifecycle_path=Path(
                    args.proposal_lifecycle_events
                ).resolve(),
                verification_events_path=Path(args.verification_events).resolve(),
                draft_work_events_path=Path(args.draft_work_events).resolve(),
                transport_events_path=Path(args.transport_events).resolve(),
                target_diagnostics_path=Path(args.target_diagnostics).resolve(),
                plugin_report_path=Path(args.plugin_report).resolve(),
                output_checkpoint_path=Path(args.output_checkpoint).resolve(),
                cycle_events_path=Path(args.cycle_events).resolve(),
                overlap_events_path=Path(args.overlap_events).resolve(),
                runtime_manifest_path=Path(args.runtime_manifest).resolve(),
                output_path=Path(args.output).resolve(),
                git_commit=_current_git_commit() or "unknown",
                microbatch_size=args.microbatch_size,
                test_coordination=args.test_coordination,
                overlap_requirement=args.overlap_requirement,
            )
        except (
            FileExistsError,
            FileNotFoundError,
            ImportError,
            RuntimeError,
            ValueError,
        ) as error:
            raise SystemExit(f"Phase-4B.1 resident Dual run failed: {error}") from error
        return 0 if report["valid"] else 1
    if args.command == "phase4b1-dual-correctness-validate":
        from specrhythm.phase4.dual_correctness import (
            VALIDATION_SCHEMA,
            validate_phase4b1_dual_correctness,
        )

        try:
            report = validate_phase4b1_dual_correctness(
                target_path=Path(args.target).resolve(),
                serial_path=Path(args.serial).resolve(),
                dual_paths=[Path(path).resolve() for path in args.dual],
                target_manifest_path=Path(args.target_manifest).resolve(),
                serial_manifest_path=Path(args.serial_manifest).resolve(),
                target_process_lifecycle_path=Path(
                    args.target_process_lifecycle
                ).resolve(),
                serial_process_lifecycle_path=Path(
                    args.serial_process_lifecycle
                ).resolve(),
                dual_manifest_paths=[
                    Path(path).resolve() for path in args.dual_manifest
                ],
                state_event_paths=[
                    Path(path).resolve() for path in args.request_state_events
                ],
                proposal_event_paths=[
                    Path(path).resolve() for path in args.proposal_events
                ],
                proposal_lifecycle_paths=[
                    Path(path).resolve() for path in args.proposal_lifecycle_events
                ],
                scheduler_event_paths=[
                    Path(path).resolve() for path in args.scheduler_events
                ],
                verification_event_paths=[
                    Path(path).resolve() for path in args.verification_events
                ],
                draft_work_event_paths=[
                    Path(path).resolve() for path in args.draft_work_events
                ],
                target_diagnostic_paths=[
                    Path(path).resolve() for path in args.target_diagnostics
                ],
                overlap_event_paths=[
                    Path(path).resolve() for path in args.overlap_events
                ],
                process_lifecycle_paths=[
                    Path(path).resolve() for path in args.process_lifecycle
                ],
                output_path=Path(args.output).resolve(),
                markdown_path=Path(args.markdown_output).resolve(),
                overlap_requirement=args.overlap_requirement,
                legacy_source_commit=args.legacy_source_commit,
            )
        except (
            FileNotFoundError,
            json.JSONDecodeError,
            RuntimeError,
            ValueError,
        ) as error:
            report = {
                "schema_version": VALIDATION_SCHEMA,
                "valid": False,
                "outcome": "FAIL",
                "errors": [str(error)],
                "performance_result": False,
            }
            _write_json(report, args.output)
            return 1
        return 0 if report["valid"] else 1
    if args.command == "phase4b1-overlap-diagnose":
        from specrhythm.phase4.dual_correctness import (
            OVERLAP_DIAGNOSTIC_SCHEMA,
            diagnose_overlap_artifacts,
        )

        try:
            report = diagnose_overlap_artifacts(
                draft_work_paths=[
                    Path(path).resolve() for path in args.draft_work_events
                ],
                verification_paths=[
                    Path(path).resolve() for path in args.verification_events
                ],
                overlap_paths=[
                    Path(path).resolve() for path in args.overlap_events
                ],
                output_path=Path(args.output).resolve(),
            )
        except (FileNotFoundError, json.JSONDecodeError, RuntimeError, ValueError) as error:
            report = {
                "schema_version": OVERLAP_DIAGNOSTIC_SCHEMA,
                "valid": False,
                "errors": [str(error)],
                "performance_result": False,
            }
            _write_json(report, args.output)
            return 1
        return 0 if report["valid"] else 1
    if args.command == "phase4b1-dual-controlled-validate":
        from specrhythm.phase4.dual_correctness import (
            CONTROLLED_SCHEMA,
            validate_controlled_gate,
        )

        try:
            report = validate_controlled_gate(
                asynchronous_scheduler_path=Path(
                    args.asynchronous_scheduler
                ).resolve(),
                coordinated_scheduler_path=Path(
                    args.coordinated_scheduler
                ).resolve(),
                state_event_path=Path(args.request_state_events).resolve(),
                output_path=Path(args.output).resolve(),
            )
        except (FileNotFoundError, json.JSONDecodeError, RuntimeError, ValueError) as error:
            report = {
                "schema_version": CONTROLLED_SCHEMA,
                "valid": False,
                "outcome": "FAIL",
                "errors": [str(error)],
                "performance_result": False,
            }
            _write_json(report, args.output)
            return 1
        return 0 if report["valid"] else 1
    if args.command == "phase4-resident-target-run":
        from specrhythm.phase4.config import load_phase4_config
        from specrhythm.phase4.resident_runner import run_resident_target

        try:
            report = run_resident_target(
                load_phase4_config(args.config),
                workload_path=Path(args.workload).resolve(),
                request_count=args.request_count,
                environment_path=Path(args.environment).resolve(),
                topology_path=Path(args.topology).resolve(),
                reference_path=Path(args.reference).resolve(),
                patch_manifest_path=Path(args.patch_manifest).resolve(),
                draft_socket_path=Path(args.draft_socket).resolve(),
                draft_ready_path=Path(args.draft_ready).resolve(),
                context_path=Path(args.context).resolve(),
                decode_ready_manifest_path=Path(args.decode_ready_manifest).resolve(),
                timing_events_path=Path(args.timing_events).resolve(),
                setup_control_path=Path(args.setup_control).resolve(),
                setup_ready_path=Path(args.setup_ready).resolve(),
                admission_events_path=Path(args.admission_events).resolve(),
                target_diagnostics_path=Path(args.target_diagnostics).resolve(),
                plugin_report_path=Path(args.plugin_report).resolve(),
                first_forward_path=Path(args.first_forward).resolve(),
                output_path=Path(args.output).resolve(),
                git_commit=_current_git_commit() or "unknown",
                correctness_mode=args.correctness_mode,
            )
        except (
            FileExistsError,
            FileNotFoundError,
            ImportError,
            RuntimeError,
            ValueError,
        ) as error:
            raise SystemExit(f"Phase-4 resident Target failed: {error}") from error
        return 0 if report["valid"] else 1
    if args.command == "phase4-resident-serial-run":
        from specrhythm.phase4.config import load_phase4_config
        from specrhythm.phase4.manifest import atomic_write_json
        from specrhythm.phase4.resident_runner import build_decode_ready_context
        from specrhythm.phase4.serial_runner import (
            load_patch_manifest,
            run_serial_disaggregated,
        )

        try:
            config = load_phase4_config(args.config)
            patch_manifest_path = Path(args.patch_manifest).resolve()
            patch_manifest = load_patch_manifest(patch_manifest_path, config)
            context_path = Path(args.context).resolve()
            if context_path.exists():
                raise FileExistsError(f"refusing to overwrite resident context {context_path}")
            atomic_write_json(
                context_path,
                build_decode_ready_context(
                    config,
                    patch_manifest=patch_manifest,
                    workload_path=Path(args.workload).resolve(),
                    git_commit=_current_git_commit() or "unknown",
                    correctness_mode=args.correctness_mode,
                ),
            )
            report = run_serial_disaggregated(
                config,
                workload_path=Path(args.workload).resolve(),
                environment_path=Path(args.environment).resolve(),
                topology_path=Path(args.topology).resolve(),
                runtime_manifest_path=Path(args.runtime_manifest).resolve(),
                reference_path=Path(args.reference).resolve(),
                patch_manifest_path=patch_manifest_path,
                draft_socket_path=Path(args.draft_socket).resolve(),
                draft_ready_path=Path(args.draft_ready).resolve(),
                round_events_path=Path(args.round_events).resolve(),
                transport_events_path=Path(args.transport_events).resolve(),
                plugin_report_path=Path(args.plugin_report).resolve(),
                git_commit=_current_git_commit() or "unknown",
                correctness_mode=args.correctness_mode,
                diagnostics_path=Path(args.target_diagnostics).resolve(),
                request_count=args.request_count,
                decode_ready_context_path=context_path,
                decode_ready_manifest_path=Path(
                    args.decode_ready_manifest
                ).resolve(),
                decode_ready_timing_path=Path(args.timing_events).resolve(),
                first_forward_path=Path(args.first_forward).resolve(),
                resident_setup_control_path=Path(args.setup_control).resolve(),
                resident_setup_ready_path=Path(args.setup_ready).resolve(),
                resident_admission_events_path=Path(
                    args.admission_events
                ).resolve(),
                resident_initial_proposal_events_path=Path(
                    args.initial_proposal_events
                ).resolve(),
            )
            _write_json(report, args.output)
        except (
            FileExistsError,
            FileNotFoundError,
            ImportError,
            RuntimeError,
            ValueError,
        ) as error:
            raise SystemExit(f"Phase-4 resident Serial failed: {error}") from error
        return 0 if report["valid"] else 1
    if args.command == "phase4-resident-validate":
        from specrhythm.phase4.resident_runner import validate_resident_pair

        try:
            report = validate_resident_pair(
                target_path=Path(args.target).resolve(),
                serial_path=Path(args.serial).resolve(),
                target_manifest_path=Path(args.target_manifest).resolve(),
                serial_manifest_path=Path(args.serial_manifest).resolve(),
            )
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
            report = {
                "schema_version": "specrhythm.phase4b-resident-pair-validation.v1",
                "valid": False,
                "errors": [str(error)],
                "performance_result": False,
            }
        _write_json(report, args.output)
        return 0 if report["valid"] else 1
    if args.command == "phase4-batch-invariant-preflight":
        from specrhythm.phase4.batch_invariant import (
            probe_batch_invariant_hardware,
        )

        try:
            report = probe_batch_invariant_hardware(args.correctness_mode)
        except (RuntimeError, ValueError) as error:
            report = {
                "schema_version": "specrhythm.phase4-batch-invariant-preflight.v1",
                "valid": False,
                "batch_invariant_effective": False,
                "errors": [str(error)],
            }
        _write_json(report, args.output)
        return 0 if report["valid"] else 2
    if args.command == "phase4-batch-invariant-validate":
        from specrhythm.phase4.correctness_validation import (
            correctness_markdown,
            validate_batch_invariant_experiment,
        )

        try:
            report = validate_batch_invariant_experiment(
                stock_reference_paths=[
                    Path(path).resolve() for path in args.stock_reference
                ],
                target_regression_paths=[
                    Path(path).resolve() for path in args.target_regression
                ],
                serial_run_paths=[Path(path).resolve() for path in args.serial_run],
                round_event_paths=[Path(path).resolve() for path in args.round_events],
                target_diagnostic_paths=[
                    Path(path).resolve() for path in args.target_diagnostics
                ],
                serial_diagnostic_paths=[
                    Path(path).resolve() for path in args.serial_diagnostics
                ],
            )
        except (FileNotFoundError, json.JSONDecodeError, RuntimeError, ValueError) as error:
            report = {
                "schema_version": "specrhythm.phase4a1.1-batch-invariant-validation.v1",
                "valid": False,
                "outcome": "invalid-artifacts",
                "errors": [str(error)],
            }
        _write_json(report, args.output)
        markdown = Path(args.markdown_output).resolve()
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(correctness_markdown(report), encoding="utf-8")
        return 0 if report["valid"] else 1
    if args.command == "phase4-fixed-proposal-service":
        from specrhythm.phase4.fixed_control import run_fixed_proposal_service

        try:
            run_fixed_proposal_service(Path(args.socket).resolve())
        except (FileExistsError, RuntimeError, ValueError) as error:
            raise SystemExit(f"fixed-proposal service failed: {error}") from error
        return 0
    if args.command == "phase4-fixed-control-run":
        from specrhythm.phase4.config import load_phase4_config
        from specrhythm.phase4.serial_runner import run_fixed_proposal_control

        output_path = Path(args.output).resolve()
        if output_path.exists():
            raise SystemExit(f"refusing to overwrite fixed control {output_path}")
        try:
            report = run_fixed_proposal_control(
                load_phase4_config(args.config),
                workload_path=Path(args.workload).resolve(),
                environment_path=Path(args.environment).resolve(),
                topology_path=Path(args.topology).resolve(),
                patch_manifest_path=Path(args.patch_manifest).resolve(),
                diagnostics_path=Path(args.target_diagnostics).resolve(),
                git_commit=_current_git_commit() or "unknown",
                proposer=args.proposer,
                proposal_budget=args.proposal_budget,
                remote_socket_path=(
                    Path(args.remote_socket).resolve() if args.remote_socket else None
                ),
            )
        except (
            FileExistsError,
            FileNotFoundError,
            ImportError,
            RuntimeError,
            ValueError,
        ) as error:
            raise SystemExit(f"fixed-proposal control failed: {error}") from error
        _write_json(report, output_path)
        return 0
    if args.command == "phase4-fixed-control-validate":
        from specrhythm.phase4.correctness_validation import (
            validate_fixed_control_matrix,
        )

        try:
            report = validate_fixed_control_matrix(
                local_run_paths=[Path(path).resolve() for path in args.local_run],
                remote_run_paths=[Path(path).resolve() for path in args.remote_run],
                local_diagnostic_paths=[
                    Path(path).resolve() for path in args.local_diagnostics
                ],
                remote_diagnostic_paths=[
                    Path(path).resolve() for path in args.remote_diagnostics
                ],
            )
        except (FileNotFoundError, json.JSONDecodeError, RuntimeError, ValueError) as error:
            report = {
                "schema_version": "specrhythm.phase4-fixed-control-matrix.v1",
                "valid": False,
                "errors": [str(error)],
            }
        _write_json(report, args.output)
        return 0 if report["valid"] else 1
    if args.command == "phase4-divergence-diagnose":
        from specrhythm.phase4.correctness_validation import (
            diagnose_first_divergence,
        )

        try:
            report = diagnose_first_divergence(
                stock_diagnostics_path=Path(args.stock_diagnostics).resolve(),
                serial_diagnostics_path=Path(args.serial_diagnostics).resolve(),
                serial_run_path=Path(args.serial_run).resolve(),
            )
        except (FileNotFoundError, json.JSONDecodeError, RuntimeError, ValueError) as error:
            report = {
                "schema_version": "specrhythm.phase4-first-divergence.v1",
                "valid": False,
                "errors": [str(error)],
            }
        _write_json(report, args.output)
        return 0 if report["valid"] else 1
    if args.command == "gpu-probe":
        from specrhythm.phase3.probe import probe_gpu_environment

        report = probe_gpu_environment(repo=Path.cwd())
        _write_json(report, args.output)
        return 0 if report["available"] or args.allow_unavailable else 2
    if args.command == "tp-check":
        from specrhythm.phase3.tp import load_model_config, validate_tp_compatibility

        try:
            model_config = load_model_config(args.model_config)
            report = validate_tp_compatibility(model_config, args.tp_sizes)
        except ValueError as error:
            raise SystemExit(f"TP compatibility check failed: {error}") from error
        _write_json(report, args.output)
        return 0 if all(row["supported"] for row in report["results"]) else 1
    if args.command == "phase3-run":
        from specrhythm.phase3.config import load_phase3_config
        from specrhythm.phase3.distributed import TensorParallelTargetPool
        from specrhythm.phase3.runner import (
            build_run_manifest,
            load_prompt_requests,
            run_phase3,
        )

        try:
            config = load_phase3_config(args.config)
            config = config.with_overrides(
                backend=args.backend,
                context_length=args.context_length,
                batch_size=args.batch_size,
                search_pool_size=args.search_pool_size,
                candidate_budget=args.candidate_budget,
                max_new_tokens=args.max_new_tokens,
            )
            draft_gpus = _gpu_ids(args.draft_gpus)
            target_gpus = _gpu_ids(args.target_gpus)
            config = type(config)(
                schema_version=config.schema_version,
                backend=config.backend,
                draft=config.draft.with_overrides(
                    model_path=args.draft_model,
                    gpu_ids=draft_gpus,
                    tp_size=args.draft_tp,
                    dtype=args.dtype,
                ),
                target=config.target.with_overrides(
                    model_path=args.target_model,
                    gpu_ids=target_gpus,
                    tp_size=args.target_tp,
                    dtype=args.dtype,
                ),
                context_length=config.context_length,
                batch_size=config.batch_size,
                search_pool_size=config.search_pool_size,
                candidate_budget=config.candidate_budget,
                candidate_width=config.candidate_width,
                max_new_tokens=config.max_new_tokens,
                random_seed=config.random_seed,
                sampling_configuration=config.sampling_configuration,
                benchmark=config.benchmark,
            )
            input_path = Path(args.input).resolve()
            output_dir = Path(args.output_dir).resolve()
            requests = load_prompt_requests(input_path)
            target_pool = None
            if (
                config.backend == "transformers"
                and args.mode == "serial"
                and config.target.tp_size > 1
            ):
                if int(os.environ.get("WORLD_SIZE", "1")) != 1:
                    raise ValueError(
                        "five-GPU serial mode is a coordinator command; do not wrap it in torchrun"
                    )
                target_pool = TensorParallelTargetPool(
                    config.target, config.random_seed
                )
            try:
                report = run_phase3(
                    requests,
                    config,
                    mode=args.mode,
                    output_dir=output_dir,
                    resume=args.resume,
                    target_backend=target_pool,
                    cycle_limit=args.max_cycles,
                )
            finally:
                if target_pool is not None:
                    target_pool.close()
        except (FileExistsError, ValueError, RuntimeError) as error:
            raise SystemExit(f"Phase-3 run failed: {error}") from error
        if int(os.environ.get("RANK", "0")) == 0:
            command_argv = list(argv) if argv is not None else sys.argv[1:]
            manifest = build_run_manifest(
                config_path=Path(args.config).resolve(),
                input_path=input_path,
                output_dir=output_dir,
                config=config,
                mode=args.mode,
                command=shlex.join(["specrhythm", *command_argv]),
                git_commit=_current_git_commit(),
                environment_metadata_path=(
                    Path(args.environment_metadata).resolve()
                    if args.environment_metadata
                    else None
                ),
                runtime_models=report.get("runtime_models"),
            )
            _write_json(report, str(output_dir / "summary.json"))
            _write_json(manifest, str(output_dir / "manifest.json"))
            _write_json(report, None)
        return 0
    if args.command == "phase3-validate":
        from specrhythm.phase3.trace import TraceStore

        trace_dir = Path(args.trace_dir).resolve()
        report = TraceStore(trace_dir).validate()
        if args.target_only_dir:
            try:
                actual = _final_tokens(trace_dir)
                target = _final_tokens(Path(args.target_only_dir).resolve())
                equivalent = actual == target
                report["target_only_semantic_equivalence"] = equivalent
                if not equivalent:
                    report["valid"] = False
                    report["errors"].append(
                        "final token sequences differ from target-only reference"
                    )
            except ValueError as error:
                report["valid"] = False
                report["errors"].append(str(error))
        _write_json(report, args.output)
        return 0 if report["valid"] else 1
    if args.command == "phase3-summarize":
        from specrhythm.phase3.trace import TraceStore, summarize_records

        store = TraceStore(Path(args.trace_dir).resolve())
        validation = store.validate()
        if not validation["valid"]:
            _write_json(validation, args.output)
            return 1
        report = summarize_records(store.records())
        if args.trace_output:
            report["trace_jsonl_sha256"] = store.write_jsonl(
                Path(args.trace_output).resolve()
            )
        _write_json(report, args.output)
        return 0
    if args.command == "phase3-benchmark":
        from specrhythm.phase3.benchmark import (
            atomic_write_json,
            atomic_write_text,
            benchmark_markdown,
            run_latency_benchmark,
        )
        from specrhythm.phase3.benchmark_validation import validate_benchmark_report
        from specrhythm.phase3.config import load_phase3_config

        try:
            report = run_latency_benchmark(load_phase3_config(args.config), args.operation)
        except (ValueError, RuntimeError) as error:
            raise SystemExit(f"Phase-3 benchmark failed: {error}") from error
        if int(os.environ.get("RANK", "0")) == 0:
            from specrhythm.phase3.trace import sha256_file

            config_path = Path(args.config).resolve()
            report["git_commit"] = _current_git_commit()
            report["config_file"] = config_path.name
            report["config_sha256"] = sha256_file(config_path)
            if args.environment_metadata:
                environment_path = Path(args.environment_metadata).resolve()
                report["environment_metadata_file"] = environment_path.name
                report["environment_metadata_sha256"] = sha256_file(environment_path)
            report["validation"] = validate_benchmark_report(report)
            atomic_write_json(Path(args.output).resolve(), report)
            atomic_write_text(
                Path(args.markdown_output).resolve(), benchmark_markdown(report)
            )
        return 0 if report["validation"]["valid"] else 1
    if args.command == "phase3-benchmark-validate":
        from specrhythm.phase3.benchmark import atomic_write_json
        from specrhythm.phase3.benchmark_validation import validate_benchmark_report

        try:
            report = json.loads(Path(args.input).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"Phase-3 benchmark validation failed: {error}") from error
        validation = validate_benchmark_report(report)
        atomic_write_json(Path(args.output).resolve(), validation)
        return 0 if validation["valid"] else 1
    if args.command == "phase3-benchmark-compare":
        from specrhythm.phase3.benchmark import atomic_write_json, atomic_write_text
        from specrhythm.phase3.benchmark_validation import (
            compare_benchmark_reports,
            comparison_markdown,
        )

        paths = [Path(value).resolve() for value in args.input]
        try:
            reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"Phase-3 benchmark comparison failed: {error}") from error
        comparison = compare_benchmark_reports(
            reports, [f"{path.parent.name}/{path.name}" for path in paths]
        )
        atomic_write_json(Path(args.output).resolve(), comparison)
        atomic_write_text(
            Path(args.markdown_output).resolve(), comparison_markdown(comparison)
        )
        return 0 if comparison["valid"] else 1
    if args.command == "phase3-selector-dry-run":
        from specrhythm.phase3.benchmark import atomic_write_json
        from specrhythm.phase3.selector_benchmark import run_selector_dry_run

        try:
            report = run_selector_dry_run(
                request_count=args.request_count,
                search_pool_size=args.search_pool_size,
                candidate_budget=args.candidate_budget,
            )
        except ValueError as error:
            raise SystemExit(f"Phase-3 selector dry-run failed: {error}") from error
        atomic_write_json(Path(args.output).resolve(), report)
        return 0
    if args.command == "phase3c-workload-build":
        from specrhythm.phase3.r3_workload import build_r3_real_workload

        try:
            command_argv = list(argv) if argv is not None else sys.argv[1:]
            manifest = build_r3_real_workload(
                _phase3c_config(args.config, args.backend),
                output_path=Path(args.output).resolve(),
                manifest_path=Path(args.manifest).resolve(),
                command=shlex.join(["specrhythm", *command_argv]),
                request_count=args.request_count,
                git_commit=_current_git_commit(),
            )
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            raise SystemExit(f"Phase-3C workload build failed: {error}") from error
        _write_json(manifest, None)
        return 0
    if args.command in {"phase3c-draft-forest", "phase3c-target-trajectory"}:
        from specrhythm.phase3.r3_workload import load_r3_workload
        from specrhythm.phase3.real_candidate_trace import (
            run_draft_forest_stage,
            run_target_trajectory_stage,
        )

        try:
            config = _phase3c_config(args.config, args.backend)
            workload_path = Path(args.workload).resolve()
            requests = load_r3_workload(workload_path)
            output_dir = Path(args.output_dir).resolve()
            if args.command == "phase3c-draft-forest":
                report = run_draft_forest_stage(
                    requests,
                    config,
                    workload_path=workload_path,
                    output_dir=output_dir,
                    resume=args.resume,
                )
            else:
                report = run_target_trajectory_stage(
                    requests,
                    config,
                    workload_path=workload_path,
                    output_dir=output_dir,
                    resume=args.resume,
                )
        except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as error:
            raise SystemExit(f"Phase-3C model stage failed: {error}") from error
        _write_json(report, str(output_dir / "stage-summary.json"))
        _write_json(report, None)
        return 0
    if args.command == "phase3c-label-join":
        from specrhythm.phase3.r3_workload import load_r3_workload
        from specrhythm.phase3.real_candidate_trace import run_label_join_stage

        try:
            report = run_label_join_stage(
                load_r3_workload(Path(args.workload).resolve()),
                forest_dir=Path(args.forest_dir).resolve(),
                target_dir=Path(args.target_dir).resolve(),
                output_dir=Path(args.output_dir).resolve(),
                resume=args.resume,
            )
        except (FileExistsError, FileNotFoundError, ValueError) as error:
            raise SystemExit(f"Phase-3C label join failed: {error}") from error
        _write_json(report, str(Path(args.output_dir).resolve() / "stage-summary.json"))
        _write_json(report, None)
        return 0
    if args.command == "phase3c-selector-replay":
        from specrhythm.phase3.selector_diagnosis import run_selector_replay_stage

        try:
            output_dir = Path(args.output_dir).resolve()
            report = run_selector_replay_stage(
                _phase3c_config(args.config),
                labeled_dir=Path(args.labeled_dir).resolve(),
                output_dir=output_dir,
                resume=args.resume,
            )
        except (FileExistsError, FileNotFoundError, ValueError) as error:
            raise SystemExit(f"Phase-3C selector replay failed: {error}") from error
        _write_json(report, str(output_dir / "stage-summary.json"))
        _write_json(report, None)
        return 0
    if args.command == "phase3c-validate":
        from specrhythm.phase3.r3_workload import load_r3_workload
        from specrhythm.phase3.real_candidate_trace import validate_phase3c_artifacts
        from specrhythm.phase3.selector_diagnosis import validate_selector_artifacts

        try:
            report = validate_phase3c_artifacts(
                load_r3_workload(Path(args.workload).resolve()),
                forest_dir=Path(args.forest_dir).resolve(),
                target_dir=Path(args.target_dir).resolve(),
                labeled_dir=Path(args.labeled_dir).resolve(),
            )
            selector_validation = validate_selector_artifacts(
                labeled_dir=Path(args.labeled_dir).resolve(),
                selector_dir=Path(args.selector_dir).resolve(),
            )
            report["selector_validation"] = selector_validation
            if not selector_validation["valid"]:
                report["valid"] = False
                report["errors"].extend(selector_validation["errors"])
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            report = {
                "schema_version": "specrhythm.phase3c-validation.v1",
                "valid": False,
                "errors": [str(error)],
            }
        _write_json(report, args.output)
        return 0 if report["valid"] else 1
    if args.command in {"phase3c-summary", "phase3c-resummary"}:
        from specrhythm.phase3.benchmark import atomic_write_json, atomic_write_text
        from specrhythm.phase3.selector_diagnosis import (
            diagnosis_markdown,
            summarize_selector_diagnosis,
        )

        try:
            report = summarize_selector_diagnosis(
                labeled_dir=Path(args.labeled_dir).resolve(),
                selector_dir=Path(args.selector_dir).resolve(),
                source_trace_commit=args.source_trace_commit,
                workload_manifest_path=(
                    Path(args.workload_manifest).resolve() if args.workload_manifest else None
                ),
                draft_dir=Path(args.draft_dir).resolve() if args.draft_dir else None,
                target_dir=Path(args.target_dir).resolve() if args.target_dir else None,
            )
        except (FileNotFoundError, ValueError) as error:
            raise SystemExit(f"Phase-3C summary failed: {error}") from error
        atomic_write_json(Path(args.output).resolve(), report)
        atomic_write_text(Path(args.markdown_output).resolve(), diagnosis_markdown(report))
        return 0
    if args.command == "phase3c-multiround-snapshots":
        from specrhythm.phase3.multiround import run_common_prefix_snapshot_stage
        from specrhythm.phase3.r3_workload import load_r3_workload

        try:
            output_dir = Path(args.output_dir).resolve()
            report = run_common_prefix_snapshot_stage(
                load_r3_workload(Path(args.workload).resolve()),
                _phase3c_config(args.config, args.backend),
                workload_path=Path(args.workload).resolve(),
                target_dir=Path(args.target_dir).resolve(),
                output_dir=output_dir,
                resume=args.resume,
            )
        except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as error:
            raise SystemExit(f"Phase-3C multi-round snapshot stage failed: {error}") from error
        _write_json(report, str(output_dir / "stage-summary.json"))
        _write_json(report, None)
        return 0
    if args.command == "phase3c-multiround-replay":
        from specrhythm.phase3.multiround import run_sequential_replay_stage
        from specrhythm.phase3.r3_workload import load_r3_workload

        try:
            output_dir = Path(args.output_dir).resolve()
            report = run_sequential_replay_stage(
                load_r3_workload(Path(args.workload).resolve()),
                target_dir=Path(args.target_dir).resolve(),
                snapshot_dir=Path(args.snapshot_dir).resolve(),
                output_dir=output_dir,
                resume=args.resume,
            )
        except (FileExistsError, FileNotFoundError, ValueError) as error:
            raise SystemExit(f"Phase-3C multi-round replay failed: {error}") from error
        _write_json(report, str(output_dir / "stage-summary.json"))
        _write_json(report, None)
        return 0
    if args.command == "phase3c-multiround-summary":
        from specrhythm.phase3.benchmark import atomic_write_json, atomic_write_text
        from specrhythm.phase3.multiround import (
            multiround_markdown,
            summarize_multiround,
        )

        try:
            report = summarize_multiround(
                snapshot_dir=Path(args.snapshot_dir).resolve(),
                sequential_dir=Path(args.sequential_dir).resolve(),
                source_trace_commit=args.source_trace_commit,
            )
        except (FileNotFoundError, ValueError) as error:
            raise SystemExit(f"Phase-3C multi-round summary failed: {error}") from error
        atomic_write_json(Path(args.output).resolve(), report)
        atomic_write_text(Path(args.markdown_output).resolve(), multiround_markdown(report))
        return 0
    if args.command == "phase3c-multiround-validate":
        from specrhythm.phase3.multiround import validate_multiround_artifacts
        from specrhythm.phase3.r3_workload import load_r3_workload

        try:
            workload_path = Path(args.workload).resolve()
            report = validate_multiround_artifacts(
                load_r3_workload(workload_path),
                workload_path=workload_path,
                workload_manifest_path=Path(args.workload_manifest).resolve(),
                target_dir=Path(args.target_dir).resolve(),
                snapshot_dir=Path(args.snapshot_dir).resolve(),
                sequential_dir=Path(args.sequential_dir).resolve(),
                expected_request_count=args.expected_request_count,
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            report = {
                "schema_version": "specrhythm.phase3c-multiround-validation.v1",
                "valid": False,
                "errors": [str(error)],
                "gpu_performance_result": False,
            }
        _write_json(report, args.output)
        return 0 if report["valid"] else 1
    if args.command == "phase3c-learned-pilot":
        from specrhythm.phase3.benchmark import atomic_write_json, atomic_write_text
        from specrhythm.phase3.learned_selector import (
            learned_pilot_markdown,
            run_learned_selector_pilot,
        )
        from specrhythm.phase3.r3_workload import load_r3_workload

        try:
            workload_path = Path(args.workload).resolve()
            output_dir = Path(args.output_dir).resolve()
            report = run_learned_selector_pilot(
                load_r3_workload(workload_path),
                workload_path=workload_path,
                target_dir=Path(args.target_dir).resolve(),
                snapshot_dir=Path(args.snapshot_dir).resolve(),
                sequential_dir=Path(args.sequential_dir).resolve(),
                output_dir=output_dir,
                resume=args.resume,
                seed=args.seed,
                source_trace_commit=args.source_trace_commit,
            )
        except (FileExistsError, FileNotFoundError, KeyError, TypeError, ValueError) as error:
            raise SystemExit(f"Phase-3C learned selector pilot failed: {error}") from error
        stage_summary = {
            "schema_version": "specrhythm.phase3c-stage-summary.v3",
            "stage": "diagnostic-learned-shell-ranker",
            "request_count": report["request_count"],
            "snapshot_count": report["snapshot_count"],
            "new_learned_replay_records": report["new_learned_replay_records"],
            "split_counts": report["split_counts"],
            "decision": report["decision"]["outcome"],
            "runtime_features_only_at_inference": True,
            "gpu_performance_result": False,
        }
        atomic_write_json(output_dir / "stage-summary.json", stage_summary)
        atomic_write_json(Path(args.output).resolve(), report)
        atomic_write_text(
            Path(args.markdown_output).resolve(), learned_pilot_markdown(report)
        )
        return 0
    if args.command == "generate":
        config = load_json(args.config)
        output_path = Path(args.output).resolve()
        config_path = Path(args.config).resolve()
        if output_path == config_path:
            raise SystemExit("--output must not overwrite --config")
        if args.arrival_trace and output_path == Path(args.arrival_trace).resolve():
            raise SystemExit("--output must not overwrite --arrival-trace")
        if args.manifest:
            manifest_path = Path(args.manifest).resolve()
            if output_path == manifest_path:
                raise SystemExit("--output and --manifest must be different files")
            if manifest_path == config_path:
                raise SystemExit("--manifest must not overwrite --config")
            if args.arrival_trace and manifest_path == Path(args.arrival_trace).resolve():
                raise SystemExit("--manifest must not overwrite --arrival-trace")

        source_url = None
        source_commit_sha = None
        if args.manifest:
            source = config.get("source", {})
            source_url = args.source_url or source.get("url")
            source_commit_sha = args.source_commit_sha or source.get("commit_sha")
            if not source_url or not source_commit_sha:
                raise SystemExit(
                    "--manifest requires source URL and commit SHA via CLI or config"
                )
            try:
                source_url = pin_source_url(str(source_url), str(source_commit_sha))
            except ValueError as error:
                raise SystemExit(f"invalid provenance: {error}") from error
        if args.arrival_trace:
            replay = select_arrival_replay(
                args.arrival_trace,
                window_start_ms=args.window_start_ms,
                window_duration_ms=args.window_duration_ms,
                time_scale=args.time_scale,
            )
            workload = generate_replay_workload(config, replay)
        else:
            if args.window_start_ms != 0 or args.window_duration_ms is not None:
                raise SystemExit("arrival windows require --arrival-trace")
            if args.manifest:
                raise SystemExit("provenance manifests currently require --arrival-trace")
            workload = generate_workload(config)
        workload.save_jsonl(args.output)
        if args.manifest:
            command_argv = list(argv) if argv is not None else sys.argv[1:]
            try:
                manifest = build_manifest(
                    config=config,
                    config_path=args.config,
                    source_trace_path=args.arrival_trace,
                    output_workload_path=args.output,
                    source_url=str(source_url),
                    source_commit_sha=str(source_commit_sha),
                    time_scale=args.time_scale,
                    window_start_ms=args.window_start_ms,
                    window_duration_ms=args.window_duration_ms,
                    generation_command=shlex.join(["specrhythm", *command_argv]),
                    request_count=len(workload.requests),
                )
            except ValueError as error:
                raise SystemExit(f"invalid provenance: {error}") from error
            _write_json(manifest, args.manifest)
        _write_json(summarize_workload(workload), None)
        return 0
    if args.command == "import-mooncake":
        workload = import_mooncake(
            args.input,
            time_scale=args.time_scale,
            slo_tpot_ms=args.slo_tpot_ms,
            acceptance_probability=args.acceptance_probability,
            draft_confidence=args.draft_confidence,
        )
        workload.save_jsonl(args.output)
        _write_json(summarize_workload(workload), None)
        return 0
    if args.command == "summarize":
        _write_json(summarize_workload(Workload.load_jsonl(args.workload)), args.output)
        return 0
    if args.command == "validate":
        if args.output:
            report_path = Path(args.output).resolve()
            protected = [Path(args.workload).resolve()]
            protected.extend(
                Path(path).resolve()
                for path in (args.config, args.arrival_trace)
                if path is not None
            )
            if report_path in protected:
                raise SystemExit("validation --output must not overwrite an input file")
        config = load_json(args.config) if args.config else None
        report = validate_workload(
            args.workload,
            config=config,
            arrival_trace_path=args.arrival_trace,
            window_start_ms=args.window_start_ms,
            window_duration_ms=args.window_duration_ms,
            time_scale=args.time_scale,
        )
        _write_json(report, args.output)
        return 0 if report["valid"] else 1

    config = SimulatorConfig.from_dict(load_json(args.config))
    if args.command == "capacity-knee":
        from specrhythm.diagnostics import capacity_knee_report, write_report

        report = capacity_knee_report(
            ((float(scale), path) for scale, path in args.workload), config
        )
        write_report(report, args.output)
        return 0
    workload = Workload.load_jsonl(args.workload)
    if args.command == "eager-grid":
        from specrhythm.diagnostics import eager_grid_report, write_report

        write_report(eager_grid_report(workload, config), args.output)
        return 0
    if args.command == "phase2-replay":
        snapshot_handle = None
        snapshot_sink = None
        if args.snapshot_output:
            snapshot_path = Path(args.snapshot_output)
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            if snapshot_path.suffix == ".gz":
                snapshot_handle = gzip.open(snapshot_path, "wt", encoding="utf-8")
            else:
                snapshot_handle = snapshot_path.open("w", encoding="utf-8")

            def snapshot_sink(value):
                snapshot_handle.write(json.dumps(value, sort_keys=True) + "\n")

        try:
            report = common_snapshot_replay(
                workload,
                config,
                sample_size=args.sample_size,
                snapshot_sink=snapshot_sink,
            )
        finally:
            if snapshot_handle is not None:
                snapshot_handle.close()
        _write_json(report, args.output)
        return 0
    if args.command == "phase2-simulate":
        variant = next(
            key for key, value in PHASE2_VARIANTS.items() if value == args.variant
        )
        result = run_phase2_end_to_end(
            workload,
            config,
            variant=variant,
            search_ratio=args.search_ratio,
        )
        payload = result.summary.to_dict()
        payload["evidence_kind"] = "fully-hidden-search-system-upper-bound"
        payload["deployable_measured_result"] = False
        _write_json(payload, args.output)
        return 0
    if args.command == "simulate":
        cycle_handle = None
        cycle_sink = None
        eager_handle = None
        eager_sink = None
        allocation_handle = None
        allocation_sink = None
        if args.cycle_output:
            cycle_path = Path(args.cycle_output)
            cycle_path.parent.mkdir(parents=True, exist_ok=True)
            cycle_handle = cycle_path.open("w", encoding="utf-8")

            def cycle_sink(value):
                cycle_handle.write(json.dumps(asdict(value), sort_keys=True) + "\n")

        if args.eager_output:
            eager_path = Path(args.eager_output)
            eager_path.parent.mkdir(parents=True, exist_ok=True)
            eager_handle = eager_path.open("w", encoding="utf-8")

            def eager_sink(value):
                eager_handle.write(json.dumps(asdict(value), sort_keys=True) + "\n")

        if args.allocation_output:
            allocation_path = Path(args.allocation_output)
            allocation_path.parent.mkdir(parents=True, exist_ok=True)
            allocation_handle = allocation_path.open("w", encoding="utf-8")

            def allocation_sink(value):
                allocation_handle.write(json.dumps(asdict(value), sort_keys=True) + "\n")

        try:
            result = simulate(
                workload,
                _policy(args.policy, config),
                config,
                cycle_sink=cycle_sink,
                eager_sink=eager_sink,
                allocation_sink=allocation_sink,
            )
        finally:
            if cycle_handle is not None:
                cycle_handle.close()
            if eager_handle is not None:
                eager_handle.close()
            if allocation_handle is not None:
                allocation_handle.close()
        _write_json(result.summary.to_dict(), args.output)
        return 0
    if args.command == "compare":
        names = POLICY_ORDER
        summaries = [
            simulate(workload, _policy(name, config), config).summary.to_dict() for name in names
        ]
        _write_json(
            {
                "schema_version": "specrhythm.comparison.v3",
                "policy_order": list(names),
                "summaries": summaries,
            },
            args.output,
        )
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
