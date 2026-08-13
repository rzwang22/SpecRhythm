"""Command-line entry point for workload and Phase-A experiments."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Sequence

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
    "shaping-residual",
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
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
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
