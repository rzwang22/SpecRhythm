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
        help="run real-CUDA latency interfaces; there is no dry-run timing fallback",
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
            benchmark_markdown,
            run_latency_benchmark,
        )
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
            _write_json(report, args.output)
            markdown = Path(args.markdown_output)
            markdown.parent.mkdir(parents=True, exist_ok=True)
            markdown.write_text(benchmark_markdown(report), encoding="utf-8")
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
