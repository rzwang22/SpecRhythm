"""Foreground S2 server gates. CPU/offline commands never load a GPU framework."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

from specrhythm.phase4.manifest import sha256_file
from specrhythm.phase4.owned_processes import OwnedProcesses, socket_owned_by_pid
from specrhythm.phase4.process_lifecycle import (
    _unix_socket_identity,
    run_owned_target,
    validate_lifecycle_artifact,
)
from specrhythm.serving.common import DataError, read_json, require
from specrhythm.serving.runtime_profile import PROFILE_ENV, load_s2
from specrhythm.serving.s1_cli import cleanup_attempt
from specrhythm.serving.s1_console import RunConsole, print_failure
from specrhythm.serving.s1_preflight import clean_environment, validate_execution_files
from specrhythm.serving.s1_workload import write_once
from specrhythm.serving.s2_plan import MODES, calibrated_policy, check_seal, sealed, slo_policy
from specrhythm.serving.s2_pool import publish
from specrhythm.serving.s2_preflight import (
    calibration_ids,
    capacity_plan,
    execution_manifest,
    freeze,
    preparation,
    prepare,
)
from specrhythm.serving.s2_results import light_summary, qualify, seal_result, verify_result


def environment(mode, manifest_path, directory):
    env = {k: v for k, v in clean_environment(mode).items() if not k.startswith("SR_S2")}
    manifest, _ = load_s2(str(manifest_path))
    env.update(
        {
            PROFILE_ENV: str(manifest_path),
            "SR_S2_MODE": mode,
            "SR_S2_RUN_DIRECTORY": str(directory),
            "SR_S2_CONTROL": str(directory / "s2-control.json"),
            "SR_VLLM_SOURCE": manifest["execution"]["vllm_source"],
            "SR_PHASE4B_DUAL_RHYTHM": "legacy",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        }
    )
    return env


def child_command(kind, root, mode, manifest, directory, probe=False, *, diagnostic=False):
    return [
        sys.executable,
        "-m",
        "specrhythm.serving.fixed_cli" if diagnostic else "specrhythm.serving.s2_cli",
        kind,
        "--root",
        str(root),
        "--mode",
        mode,
        "--manifest",
        str(manifest),
        "--directory",
        str(directory),
        *(["--probe"] if probe else []),
    ]


def failed_report(directory, mode, manifest, rc, error=None):
    primary = None
    for name in ("child-failure.json", "draft-child-failure.json", "launcher-failure.json"):
        path = directory / name
        if path.exists():
            failure = read_json(path)
            primary = {
                "field": name.removesuffix(".json") + ".execution",
                "expected": "successful execution",
                "actual": failure["error"],
                **failure,
                **failure.get("details", {}),
                "artifact": str(path),
            }
            break
    return {
        "schema_version": "specrhythm.s2-result.v1",
        "valid": False,
        "mode": mode,
        "execution_sha256": manifest["sha256"],
        "effective_exit_code": rc,
        "errors": [(primary or {}).get("error") or str(error or f"execution rc={rc}")],
        "primary_error": primary
        or {"field": "execution", "expected": 0, "actual": rc, "artifact": str(directory)},
        "secondary_diagnostics": [
            {"artifact": str(directory / name), "status": "missing after execution failure"}
            for name in ("runtime.json", "draft-backend-report.json")
            if not (directory / name).exists()
        ],
    }


def execute(root, gate, mode, manifest_path, directory, *, probe=False, policy=None,
            diagnostic=None):
    manifest, definitions = load_s2(str(manifest_path))
    validate_execution_files(root, manifest["execution"])
    publish(
        directory / "s2-control.json",
        {
            "schema_version": "specrhythm.s2-control.v1",
            "barrier_ns": None,
            "active_limit": manifest["active_limit"],
            "requests": {
                r.request_id: {
                    "state": "STAGED",
                    "cohort": None,
                    "admission_ns": None,
                    "resources_released": False,
                }
                for r in definitions
            },
        },
    )
    env = environment(mode, manifest_path, directory)
    if diagnostic is not None:
        # Explicit new entry point only; old S2 keeps the original launcher and qualifier.
        env["SR_FIXED_POINT"] = str(directory / "point.json")
        options = manifest["fixed_diagnostic"]["options"]
    socket = Path("/tmp") / ("sr-s2-" + uuid.uuid4().hex[:16] + ".sock")
    env["SR_S2_DRAFT_SOCKET"] = str(socket)
    saved_env = dict(os.environ)
    draft = None
    rc = 1
    report = None
    cleanup_failed = False
    display_mode = mode if diagnostic is None else diagnostic["mode"]
    with RunConsole(directory, display_mode, label=f"S2 {gate}"):
        try:
            os.environ.clear()
            os.environ.update(env)
            token = uuid.uuid4().hex
            with (directory / "draft-service.log").open("x") as log:
                draft = subprocess.Popen(
                    child_command("draft-child", root, mode, manifest_path, directory,
                                  **({"diagnostic": True} if diagnostic is not None else {})),
                    env={
                        **env,
                        "CUDA_VISIBLE_DEVICES": "0",
                        "SR_PHASE4_OWNED_TARGET_TOKEN": token,
                    },
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            owner = OwnedProcesses(draft.pid, target_token=token)
            owner.snapshot()
            write_once(
                directory / "draft-owner.json",
                {
                    "root_pid": draft.pid,
                    "observed": list(owner.observed.values()),
                    "target_token": token,
                    "draft_socket": str(socket),
                },
            )
            deadline = time.monotonic() + (900 if diagnostic is None else options["setup_timeout"])
            while not (socket.is_socket() and (directory / "draft-service-ready.json").is_file()):
                if draft.poll() is not None or time.monotonic() >= deadline:
                    rc = draft.returncode if draft.returncode not in (None, 0) else 124
                    raise DataError("S2 Draft startup failed/timed out", returncode=rc)
                time.sleep(0.1)
            write_once(
                directory / "draft-socket-owner.json",
                {
                    "draft_socket": str(socket),
                    "draft_socket_identity": _unix_socket_identity(socket),
                    "draft_socket_proven": socket_owned_by_pid(socket, draft.pid),
                },
            )
            command = child_command("child", root, mode, manifest_path, directory, probe,
                                    **({"diagnostic": True} if diagnostic is not None else {}))
            write_once(
                directory / "command.json",
                {
                    "argv": command,
                    "execution_sha256": manifest["sha256"],
                    "launch_environment": env,
                    "fresh_GPU_state": True,
                    "probe": probe,
                },
            )
            rc, lifecycle = run_owned_target(
                command,
                target_log=directory / "target.log",
                artifact_path=directory / "process-lifecycle.json",
                draft_pid=draft.pid,
                draft_socket=socket,
                ownership_journal=directory / "ownership.json",
                timeout_seconds=(14400 if diagnostic is None else
                                 2 * options["setup_timeout"] + options["window_seconds"]
                                 + options["drain_timeout"] + 60),
            )
            draft_rc = draft.wait(timeout=15)
            if rc == 0 and draft_rc != 0:
                rc = draft_rc
            write_once(
                directory / "exit-code.json",
                {
                    "effective_exit_code": rc,
                    "coordinator_exit_code": lifecycle["target_exit_status"],
                    "draft_exit_code": draft_rc,
                },
            )
            # Failed execution may have exhausted the launcher's bounded cleanup.
            # Preserve its original rc, but do not seal artifacts while owners remain live.
            if lifecycle.get("owned_cleanup_completed") is not True:
                cleanup_attempt(directory)
            if rc:
                report = failed_report(directory, mode, manifest, rc)
            elif diagnostic is not None:
                from specrhythm.serving.fixed_results import summarize

                report = summarize(manifest_path, directory, diagnostic, probe=probe)
            elif probe:
                from specrhythm.serving.s1_results import draft_backend_checks

                checks = draft_backend_checks(
                    read_json(directory / "draft-backend-report.json"),
                    directory / "draft-backend-report.json",
                )
                errors = validate_lifecycle_artifact(lifecycle)
                require(
                    not errors
                    and lifecycle["run_valid"]
                    and all(r["valid"] for r in checks.values()),
                    "S2 capacity probe cleanup failed",
                    actual={"lifecycle": errors, "backend": checks},
                )
                report = {
                    "schema_version": "specrhythm.s2-result.v1",
                    "valid": True,
                    "probe": True,
                    "mode": mode,
                    "execution_sha256": manifest["sha256"],
                    "errors": [],
                }
            else:
                report = qualify(manifest_path, directory, mode, policy)
            validate_execution_files(root, manifest["execution"])
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                rc = 130
            else:
                rc = getattr(error, "details", {}).get("returncode", rc or 1)
            if not (directory / "launcher-failure.json").exists():
                write_once(
                    directory / "launcher-failure.json",
                    {
                        "error": str(error),
                        "type": type(error).__name__,
                        "details": getattr(error, "details", {}),
                        "returncode": rc,
                    },
                )
            try:
                if (directory / "draft-owner.json").exists():
                    cleanup_attempt(directory)
            except Exception as cleanup_error:
                cleanup_failed = True
                print(f"[S2 {gate} {mode}] secondary cleanup error: {cleanup_error}", flush=True)
            report = failed_report(directory, mode, manifest, rc, error)
        finally:
            os.environ.clear()
            os.environ.update(saved_env)
        # The display thread must drain final raw logs before hashing them.
    if diagnostic is not None:
        from specrhythm.serving.fixed_results import emit_result

        report = emit_result(directory, report, diagnostic)
    elif cleanup_failed:
        write_once(directory / "result.json", report)
    else:
        seal_result(directory, report)
    if not report["valid"]:
        error = DataError(
            report["errors"][0], returncode=rc or 1, directory=str(directory), mode=mode
        )
        print_failure(error, directory=directory, mode=mode, label=f"S2 {gate}")
        raise error
    return report


def attempt(root, gate, mode, manifest_path, *, probe=False, policy=None):
    manifest, _ = load_s2(str(manifest_path))
    base = root / "runs" / gate / manifest_path.parent.name / mode
    base.mkdir(parents=True, exist_ok=True)
    old = sorted(p for p in base.glob("attempt-*") if p.is_dir())
    # Clean every interruption first, even if an older valid result can be reused.
    for directory in old:
        if not (directory / "seal.json").exists():
            if any(
                (directory / name).exists()
                for name in ("draft-owner.json", "ownership.json", "process-lifecycle.json")
            ):
                cleanup_attempt(directory)
            else:
                require(
                    not (directory / "command.json").exists(),
                    "S2 interrupted launch lacks process ownership evidence",
                )
    for directory in reversed(old):
        if (directory / "seal.json").exists():
            value = verify_result(directory)
            require(
                value["execution_sha256"] == manifest["sha256"] and value["mode"] == mode,
                "S2 resume manifest/mode differs",
            )
            if value["valid"]:
                return directory, value
    directory = base / f"attempt-{len(old) + 1:03d}"
    directory.mkdir()
    publish(
        root / "stage.json",
        {"gate": gate, "mode": mode, "directory": str(directory), "state": "RUNNING"},
    )
    try:
        report = execute(root, gate, mode, manifest_path, directory, probe=probe, policy=policy)
    except BaseException as error:
        publish(
            root / "stage.json",
            {
                "gate": gate,
                "mode": mode,
                "directory": str(directory),
                "state": "FAILED",
                "error": str(error),
            },
        )
        raise
    publish(
        root / "stage.json",
        {"gate": gate, "mode": mode, "directory": str(directory), "state": "PASS"},
    )
    return directory, report


@contextlib.contextmanager
def root_lock(root):
    import fcntl

    with (root / "launcher.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise DataError("another S2 foreground launcher owns this root") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def gate(root, selected):
    g0 = read_json(root / "g0.json")
    check_seal(g0)
    policy = read_json(root / "slo-policy.json")
    check_seal(policy)
    require(
        policy["sha256"] == g0["slo_sha256"]
        and read_json(root / "capacity-plan.json")["sha256"] == g0["capacity_sha256"],
        "S2 G0 capacity/SLO changed",
    )
    if selected == "G0":
        return g0
    previous = {"G1": None, "G2": "G1", "G3": "G2"}[selected]
    if previous:
        previous_result = read_json(root / (previous + ".json"))
        check_seal(previous_result)
        require(previous_result["valid"], "previous S2 gate failed")
        for p, sha in previous_result["run_seals"].items():
            directory = root / p
            require(
                verify_result(directory)["valid"] and sha256_file(directory / "seal.json") == sha,
                "previous S2 gate evidence changed",
            )
    seals = {}
    for unit in g0["gates"][selected]:
        path = root / unit["manifest"]
        require(read_json(path)["sha256"] == unit["sha256"], "S2 frozen gate manifest changed")
        for mode in unit["modes"]:
            directory, _ = attempt(root, selected, mode, path, policy=policy)
            seals[str(directory.relative_to(root))] = sha256_file(directory / "seal.json")
    value = sealed(
        {
            "schema_version": "specrhythm.s2-gate.v1",
            "gate": selected,
            "valid": True,
            "g0_sha256": g0["sha256"],
            "run_seals": seals,
            "skipped_capacity_duplicate": not bool(g0["gates"][selected]),
        }
    )
    destination = root / (selected + ".json")
    if destination.exists():
        require(read_json(destination) == value, "S2 completed gate changed")
    else:
        write_once(destination, value)
    return value


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "command",
        choices=(
            "prepare",
            "capacity",
            "calibrate",
            "slo",
            "freeze",
            "gate",
            "status",
            "cleanup",
            "resume",
            "summary",
            "offline",
            "bundle",
            "child",
            "draft-child",
        ),
    )
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--s0", type=Path)
    p.add_argument("--s1", type=Path)
    p.add_argument("--arrival-seed", type=int, default=1667)
    p.add_argument("--order-seed", type=int, default=1668)
    p.add_argument("--selection-seed", type=int, default=1666)
    p.add_argument(
        "--extra-seed",
        type=int,
        nargs=2,
        action="append",
        default=[],
        metavar=("ARRIVAL", "ORDER"),
        help="explicit additional seed pair frozen before main runs",
    )
    p.add_argument("--gate", choices=("G0", "G1", "G2", "G3"), default="G1")
    p.add_argument("--thresholds", help="JSON: per-class ms/timed-token thresholds")
    p.add_argument("--mode", choices=MODES)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--directory", type=Path)
    p.add_argument("--probe", action="store_true")
    p.add_argument("--output", type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    root = args.root.resolve()
    try:
        if args.command in ("child", "draft-child"):
            try:
                if args.command == "draft-child":
                    from specrhythm.phase4.config import load_phase4_config
                    from specrhythm.serving.s2_draft import serve

                    serve(
                        load_phase4_config(str(root / "config.json")),
                        args.directory,
                        Path(os.environ["SR_S2_DRAFT_SOCKET"]),
                        args.mode,
                    )
                else:
                    from specrhythm.serving.s2_runtime import run

                    run(root, args.manifest, args.directory, args.mode, probe=args.probe)
                return 0
            except BaseException as error:
                write_once(
                    args.directory
                    / (
                        "draft-child-failure.json"
                        if args.command == "draft-child"
                        else "child-failure.json"
                    ),
                    {
                        "error": str(error),
                        "type": type(error).__name__,
                        "details": getattr(error, "details", {}),
                        "timestamp_ns": time.monotonic_ns(),
                        "traceback": traceback.format_exc(),
                    },
                )
                raise
        if args.command == "prepare":
            require(args.s0 is not None and args.s1 is not None, "prepare requires --s0 and --s1")
            prepare(
                root,
                args.s0.resolve(),
                args.s1.resolve(),
                arrival_seed=args.arrival_seed,
                order_seed=args.order_seed,
                selection_seed=args.selection_seed,
                extra_seeds=args.extra_seed,
            )
            print(
                f"[S2 G0] prepared {root}; model-loaded capacity and frozen SLO still required",
                flush=True,
            )
            return 0
        if args.command == "status":
            print(
                json.dumps(
                    {
                        "stage": read_json(root / "stage.json")
                        if (root / "stage.json").exists()
                        else None,
                        "gates": {
                            g: (
                                read_json(root / (g + ".json"))
                                if (root / (g + ".json")).exists()
                                else None
                            )
                            for g in ("g0", "G1", "G2", "G3")
                        },
                    },
                    indent=2,
                )
            )
            return 0
        with root_lock(root):
            if args.command in ("gate", "resume"):
                value = gate(root, args.gate)
            elif args.command == "capacity":
                _, selection = preparation(root)
                path = root / "inputs" / "capacity" / "execution-manifest.json"
                if not path.exists():
                    path = execution_manifest(root, path.parent, selection["request_ids"][:10])
                probes = {
                    mode: attempt(root, "capacity", mode, path, probe=True)[0] for mode in MODES
                }
                value = capacity_plan(root, probes)
                value = {
                    "artifact": str(root / "capacity-plan.json"),
                    "sha256": value["sha256"],
                    "sizes": {
                        name: {
                            k: v
                            for k, v in row.items()
                            if k not in ("request_ids", "capacity_attempts")
                        }
                        for name, row in value["sizes"].items()
                    },
                }
            elif args.command == "calibrate":
                require(
                    (root / "capacity-plan.json").exists(),
                    "run real capacity probes before calibration",
                )
                require(
                    not (root / "slo-policy.json").exists(),
                    "SLO already frozen; new policy requires a new root",
                )
                path = root / "inputs" / "calibration" / "execution-manifest.json"
                if not path.exists():
                    path = execution_manifest(
                        root,
                        path.parent,
                        calibration_ids(root),
                        active_limit=1,
                        split="calibration",
                    )
                directory, value = attempt(root, "calibration", "target", path)
                policy = calibrated_policy(
                    value["requests"],
                    {
                        "directory": str(directory),
                        "seal_sha256": sha256_file(directory / "seal.json"),
                    },
                )
                write_once(root / "slo-policy.json", policy)
                value = {"artifact": str(root / "slo-policy.json"), **policy}
            elif args.command == "slo":
                require(args.thresholds is not None, "explicit SLO requires --thresholds")
                value = slo_policy(json.loads(args.thresholds))
                write_once(root / "slo-policy.json", value)
            elif args.command == "freeze":
                value = freeze(root)
            elif args.command == "cleanup":
                value = {
                    str(p.parent): cleanup_attempt(p.parent)
                    for p in root.glob("runs/**/draft-owner.json")
                    if not (p.parent / "seal.json").exists()
                }
            else:
                if args.command == "offline":
                    policy = read_json(root / "slo-policy.json")
                    for p in root.glob("runs/**/seal.json"):
                        old = verify_result(p.parent)
                        if old.get("probe") or not old["valid"]:
                            continue
                        command = read_json(p.parent / "command.json")["argv"]
                        path = Path(command[command.index("--manifest") + 1])
                        recalculated = qualify(
                            path,
                            p.parent,
                            old["mode"],
                            None if "calibration" in p.parts else policy,
                        )
                        require(
                            recalculated == old,
                            "S2 offline requalification differs",
                            artifact=str(p),
                            actual=recalculated.get("errors"),
                        )
                value = light_summary(root)
                destination = args.output or root / (
                    "light-summary-" + time.strftime("%Y%m%dT%H%M%S") + ".json"
                )
                write_once(destination, value)
                print(f"[S2 summary] {destination}", flush=True)
                return 0
        print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        details = getattr(error, "details", {})
        print_failure(
            error,
            directory=details.get("directory") or args.directory,
            mode=args.mode or args.gate,
            label="S2",
        )
        if args.command in ("child", "draft-child"):
            traceback.print_exc()
        return details.get("returncode", 130 if isinstance(error, KeyboardInterrupt) else 1) or 1


if __name__ == "__main__":
    raise SystemExit(main())
