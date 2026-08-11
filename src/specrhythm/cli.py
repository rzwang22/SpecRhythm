"""Command-line entry point for workload and Phase-A experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from specrhythm.policies import ARPolicy, FixedBudgetPolicy, MineDraftPolicy, SpecRhythmPolicy
from specrhythm.schema import Workload
from specrhythm.simulator import SimulatorConfig, simulate
from specrhythm.workload import (
    generate_workload,
    import_mooncake,
    load_arrival_times,
    load_json,
    summarize_workload,
)


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
    if name == "fixed":
        return FixedBudgetPolicy(config.fixed_speculative_budget)
    if name == "minedraft":
        return MineDraftPolicy(config.max_request_budget)
    if name == "shaping":
        return SpecRhythmPolicy(enable_eager=False)
    if name == "specrhythm":
        return SpecRhythmPolicy()
    raise ValueError(f"unknown policy: {name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="specrhythm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate or compose a workload")
    generate.add_argument("--config", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--arrival-trace")
    generate.add_argument("--time-scale", type=float, default=1.0)

    mooncake = subparsers.add_parser("import-mooncake", help="normalize Mooncake JSONL")
    mooncake.add_argument("--input", required=True)
    mooncake.add_argument("--output", required=True)
    mooncake.add_argument("--time-scale", type=float, default=1.0)
    mooncake.add_argument("--slo-tpot-ms", type=float, default=50.0)
    mooncake.add_argument("--acceptance-probability", type=float, default=0.7)

    summary = subparsers.add_parser("summarize", help="summarize a canonical workload")
    summary.add_argument("--workload", required=True)
    summary.add_argument("--output")

    simulation = subparsers.add_parser("simulate", help="run one scheduling policy")
    simulation.add_argument("--workload", required=True)
    simulation.add_argument("--config", required=True)
    simulation.add_argument(
        "--policy",
        choices=["ar", "fixed", "minedraft", "shaping", "specrhythm"],
        required=True,
    )
    simulation.add_argument("--output")

    compare = subparsers.add_parser("compare", help="compare all Phase-A policies")
    compare.add_argument("--workload", required=True)
    compare.add_argument("--config", required=True)
    compare.add_argument("--output")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        config = load_json(args.config)
        arrivals = (
            load_arrival_times(args.arrival_trace, args.time_scale) if args.arrival_trace else None
        )
        workload = generate_workload(config, arrivals)
        workload.save_jsonl(args.output)
        _write_json(summarize_workload(workload), None)
        return 0
    if args.command == "import-mooncake":
        workload = import_mooncake(
            args.input,
            time_scale=args.time_scale,
            slo_tpot_ms=args.slo_tpot_ms,
            acceptance_probability=args.acceptance_probability,
        )
        workload.save_jsonl(args.output)
        _write_json(summarize_workload(workload), None)
        return 0
    if args.command == "summarize":
        _write_json(summarize_workload(Workload.load_jsonl(args.workload)), args.output)
        return 0

    workload = Workload.load_jsonl(args.workload)
    config = SimulatorConfig.from_dict(load_json(args.config))
    if args.command == "simulate":
        result = simulate(workload, _policy(args.policy, config), config)
        _write_json(result.summary.to_dict(), args.output)
        return 0
    if args.command == "compare":
        names = ("ar", "fixed", "minedraft", "shaping", "specrhythm")
        summaries = [
            simulate(workload, _policy(name, config), config).summary.to_dict() for name in names
        ]
        _write_json({"summaries": summaries}, args.output)
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
