"""Ordinary double-batch: bounded semantic smoke, then R/N/N/R fixed B128 windows."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from specrhythm.phase4.manifest import atomic_write_json
from specrhythm.serving.common import read_json, require
from specrhythm.serving.k3 import B16, B128

MODE = "pingpong-k3"
ORDER = ("reference", "dual-batch", "dual-batch", "reference")
CPU_CASES = {
    "P0": ("pingpong-k3", "dual-batch", "reference"),
    "P1": ("pingpong-k3", "shared-command", "block-sets"),
    "S1": ("serial-k3", "shared-command", "block-sets"),
}
CPU_ORDER = ("P0", "P1", "S1", "S1", "P1", "P0")


def smoke(source, root, dispatch, *, mode=MODE, target_cpu=None):
    """16 requests, up to 8 output tokens. No Target-only equivalence claim."""
    from specrhythm.phase4.config import load_phase4_config
    from specrhythm.serving.fixed_cli import run_point
    from specrhythm.serving.k3_gpu_check import coverage
    from specrhythm.serving.prepost_gpu_check import prepare
    from specrhythm.serving.s2_plan import sealed

    config = load_phase4_config(source / "config.json")
    require(not config.sampling.do_sample and config.sampling.temperature == 0.0,
            "paired exact smoke requires existing greedy contract, not merely equal seeds")
    path = prepare(source, root, mode, modes=(mode,), configuration=B16,
                   source_configuration=B128, max_output_tokens=8)
    manifest = read_json(path)
    manifest.pop("sha256")
    manifest["fixed_diagnostic"]["options"]["target_dispatch"] = dispatch
    if target_cpu is not None:
        manifest["fixed_diagnostic"]["options"]["target_cpu"] = target_cpu
    # The bounded fixture has its own declaration, never the source B128 geometry.
    from specrhythm.serving.k3_scale_report import metadata

    manifest["fixed_diagnostic"]["diagnostic_configuration"] = metadata(
        manifest["fixed_diagnostic"]["options"])
    atomic_write_json(path, sealed(manifest))
    selected = dict(mode=mode, runtime_mode=mode, kind="joint-correctness", batch=16,
                    half="A", repeat=0, discard_warmup=False, prepost_correctness=True)
    point, result = run_point(root, selected, manifest_path=path)
    require(result["valid"] and result["cleanup_status"] == "PASS",
            "double-batch semantic smoke execution/cleanup failed")
    runtime = read_json(point / "runtime.json")
    backend = read_json(point / "draft-backend-report.json")
    evidence = coverage(runtime, backend)
    require(evidence["status"] == "COMPLETE", "bounded smoke protocol coverage incomplete")
    # A short smoke need not happen to reject; report coverage, do not invent it.
    require(runtime["stop_reason"] == "all_naturally_completed"
            and len(runtime["requests"]) == 16
            and all(r["state"] == "FINISHED" and r["resources_released"]
                    and 0 < len(r["generated_token_ids"]) <= 8 for r in runtime["requests"]),
            "double-batch semantic smoke incomplete output/release")
    value = dict(mode=mode, target_dispatch=dispatch, run_directory=str(point),
                 coverage=evidence, output_equivalence_status="NOT_RUN",
                 output_equivalence_scope="Target-only not run; paired bounded smoke only",
                 outputs={r["request_id"]: dict(tokens=r["generated_token_ids"],
                     finish_reason=r["finish_reason"]) for r in runtime["requests"]})
    atomic_write_json(root / "result.json", value)
    return value


def run(repo, directory, commit, *, cpu_comparison=False):
    from specrhythm.serving.audit_layer_report import write
    from specrhythm.serving.execution_failure import summarize
    from specrhythm.serving.fixed_artifacts import point_reports
    from specrhythm.serving.k3_acceptance import measurement
    from specrhythm.serving.k3_capacity import preflight as capacity_report
    from specrhythm.serving.k3_validation import plan
    from specrhythm.serving.ping_prepost_delivery import comparison

    order = CPU_ORDER if cpu_comparison else ORDER
    cases = CPU_CASES if cpu_comparison else {
        d: (MODE, d, None) for d in ORDER[:2]}
    # S0 is only a bounded semantic control, never another performance window.
    smoke_cases = {**cases, **({"S0": ("serial-k3", "reference", "reference")}
                              if cpu_comparison else {})}
    mode = MODE
    require(not directory.exists(), "double-batch entry requires a new directory")
    directory.mkdir(parents=True)
    atomic_write_json(directory / "validation-plan.json", plan("performance-exploration", B128))
    atomic_write_json(directory / "dual-batch-plan.json", dict(
        execution_commit=commit, mode=MODE, order=order,
        cases={k: dict(mode=v[0], target_dispatch=v[1], target_cpu=v[2])
               for k, v in cases.items()}, smoke_cases=list(smoke_cases),
        cpu_comparison=cpu_comparison, k3_configuration=B128,
        target_diagnostics="lean", draft_dispatch="unified", observation="deferred-window",
        validation_profile="performance-exploration", output_equivalence_status="NOT_RUN",
        smoke="reference/new 16 requests, max8 output; full Target-only equivalence NOT_RUN",
        performance_seconds=30, warmup_opportunities=256, setup_seconds=900, drain_seconds=60))
    stage, root, first = "static", directory, 0
    summaries, smoke_rows, source_contracts = [], [], []
    persistence_errors = []

    def retain(path, value):
        # Evidence publication errors must not replace an earlier execution code.
        nonlocal first
        try:
            write(value, path)
        except Exception as error:
            persistence_errors.append(dict(path=str(path), error=repr(error)))
            print(f"SECONDARY REPORT ERROR: {path}: {error}", flush=True)
            first = first or 41

    command_index = 0

    def command(args):
        nonlocal command_index
        log = directory / "commands" / str(command_index) / "runner.log"
        command_index += 1
        log.parent.mkdir(parents=True)
        print(f"RUN: stage={stage} mode={mode} root={root} log={log}", flush=True)
        # Keep the CLI's potentially large report in the one evidence package,
        # not in the interactive terminal. No in-memory capture or hot-path I/O.
        with log.open("w") as handle:
            subprocess.run([sys.executable, "-m", "specrhythm.serving.decode_scan_cli",
                            *args], check=True, cwd=repo, stdout=handle,
                           stderr=subprocess.STDOUT)

    def prepare(root, dispatch, cpu=None):
        command(["prepare", "--root", str(root), "--s1", os.environ["SR_FIXED_S1"],
                 "--k3-configuration", B128, "--validation-profile", "performance-exploration",
                 "--draft-audit", "runtime", "--observation", "deferred-window",
                 "--draft-dispatch", "unified", "--target-diagnostics", "lean",
                 "--target-dispatch", dispatch,
                 *(["--target-cpu", cpu] if cpu is not None else []),
                 "--identity-matching", "bound-prefix",
                 "--selection-seed", "1666", "--warmup-steps", "2", "--window-seconds", "30",
                 "--repeats", "1", "--setup-timeout", "900", "--drain-timeout", "60"])
        source = read_json(root / "scan-config.json")
        require(source["workload_sha256"] ==
                "cdaf71adace15d229f5087b98f9fd162a958456226a660184fe03f5d6ebd8ff4",
                "double-batch frozen workload mismatch")
        contract = {k: source[k] for k in ("selection", "workload_sha256")}
        # These two hashes cover timestamped, freshly collected preflight files.
        # Each is validated in its own run; all actual execution settings match.
        contract["execution"] = {k: v for k, v in source["execution"].items()
                                 if k not in ("environment_sha256", "topology_sha256")}
        contract["options"] = {k: v for k, v in source["options"].items()
                               if k not in ("target_dispatch", "target_cpu")}
        require(not source_contracts or contract == source_contracts[0],
                "reference/new execution conditions differ")
        source_contracts.append(contract)

    try:
        require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                        text=True).strip() == commit, "execution SHA mismatch")
        require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=repo),
                "clean execution worktree required")
        atomic_write_json(directory / "k3-capacity-contract.json", capacity_report(B128))
        # Same full-size resident360 capacity, both execution paths, first error stops.
        for name, (mode, dispatch, cpu) in cases.items():
            stage, root = "prepare_capacity", directory / "capacity" / name
            prepare(root, dispatch, cpu)
            stage = "capacity"
            command(["capacity", "--root", str(root), "--single-point", "--batch", "128",
                     "--mode", mode])
        for name, (mode, dispatch, cpu) in smoke_cases.items():
            stage, root = "semantic_smoke", directory / "smoke" / name
            source = directory / "capacity" / ("S1" if name == "S0" else name)
            smoke_rows.append(dict(case=name, **smoke(source, root, dispatch,
                                                      mode=mode, target_cpu=cpu)))
        stage, mode, root = "smoke_comparison", "comparison", directory
        from specrhythm.serving.ordinary_cpu_report import compare_smoke

        smoke_result = compare_smoke(smoke_rows, cpu_comparison=cpu_comparison)
        atomic_write_json(directory / "smoke-comparison.json", smoke_result)
        require(smoke_result["status"] == "PASS", "bounded smoke new output difference; "
                "see smoke-comparison.json first differences")
        for index, name in enumerate(order):
            mode, dispatch, cpu = cases[name]
            case = directory / "windows" / (str(index) + "-" + name)
            case.mkdir(parents=True)
            atomic_write_json(case / "validation-plan.json", plan("performance-exploration", B128))
            atomic_write_json(case / "k3-capacity-contract.json", capacity_report(B128))
            stage, root = "prepare_performance", case / "points" / mode
            prepare(root, dispatch, cpu)
            stage = "performance"
            command(["run", "--root", str(root), "--single-point", "--batch", "128",
                     "--mode", mode])
            stage = "measurement"
            rows = [r for r in point_reports(root) if not r["point"].get("probe")]
            require(len(rows) == 1, "one performance point required")
            measurement(rows[0], read_json(Path(rows[0]["artifact"]) / "runtime.json"),
                        mode, B128, validation_profile="performance-exploration")
            stage = "evidence"
            subprocess.run([sys.executable, "-m", "specrhythm.serving.ping_prepost_evidence",
                "--root", str(root), "--commit", commit, "--k3-configuration", B128,
                "--validation-profile", "performance-exploration",
                "--output", str(case / "points" / (mode + "-audit-report.json")),
                "--status", str(case / "points" / (mode + "-evidence-status.json"))], check=True)
            result = comparison(case, modes=(mode,))
            require(result["valid"], "double-batch window qualification failed")
            from specrhythm.serving.ordinary_cpu_report import (
                recovery_during_prepare,
                window_metrics,
            )

            raw = read_json(Path(rows[0]["artifact"]) / "runtime.json")
            extra = window_metrics(raw)
            recovery = recovery_during_prepare(raw, read_json(
                Path(rows[0]["artifact"]) / "draft-backend-report.json"))
            summaries.append(dict(index=index, case=name, target_dispatch=dispatch,
                target_cpu=cpu, request_measurement=extra, recovery_prepare_class=recovery,
                existing_measurement={k: rows[0].get(k) for k in (
                    "full_request_tpot_ms", "window_commit_interval_ms",
                    "window_commit_interval_semantics", "arrival_to_drain_ms", "drain_ms")},
                **result))
            write(result, case / "comparison.json")
        stage = "complete"
    except Exception as error:
        first = error.returncode if isinstance(error, subprocess.CalledProcessError) else 1
        first = 128 - first if first < 0 else first
        try:
            failure = summarize(root, first, stage)
            failure.update(mode=mode, run_root=str(root), primary_command_error=str(error))
            retain(directory / "first-failure.json", failure)
        except Exception as secondary:
            retain(directory / "first-failure.json", dict(
                mode=mode, stage=stage, run_root=str(root), command_exit_code=first,
                primary_command_error=str(error), secondary_summary_error=repr(secondary)))
        print(f"FIRST FAILURE: stage={stage} mode={mode} root={root} rc={first}: {error}",
              flush=True)
    finally:
        from specrhythm.serving.ordinary_cpu_report import aggregate

        retain(directory / "comparison.json", dict(
                   aggregate=aggregate(summaries),
                   valid=first == 0 and len(summaries) == len(order), windows=summaries,
                   order=order, smoke=smoke_rows, output_equivalence_status="NOT_RUN",
                   inference="all declared windows retained; compare matched I/O/profile; "
                   "no stable speedup or historical output-equivalence claim"))
        retain(directory / "runner-outcome.json", dict(
            first_exit_code=first, stage=stage, mode=mode, run_root=str(root),
            secondary_report_errors=persistence_errors))
    return first


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--cpu-comparison", action="store_true")
    a = p.parse_args()
    raise SystemExit(run(a.repo, a.directory, a.commit, cpu_comparison=a.cpu_comparison))


if __name__ == "__main__":
    main()
