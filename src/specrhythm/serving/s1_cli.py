"""Executable S1 server gates, detached launcher, owned cleanup and immutable resume."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

from specrhythm.phase4.manifest import atomic_write_json, sha256_file
from specrhythm.phase4.owned_processes import OwnedProcesses, process_table
from specrhythm.phase4.process_lifecycle import run_owned_target
from specrhythm.serving.common import DataError, read_json, require
from specrhythm.serving.s1_console import RunConsole, print_failure
from specrhythm.serving.s1_policy import (
    COMPARISON_SCHEMA,
    G0_SCHEMA,
    RESULT_SCHEMA,
    environment_evidence,
    policy_fields,
    validate_policy,
)
from specrhythm.serving.s1_preflight import (
    clean_environment,
    prepare,
    validate_execution_files,
)
from specrhythm.serving.s1_results import (
    compare_results,
    inspect_run,
    seal_run,
    verify_seal,
)
from specrhythm.serving.s1_runtime import resident_capacity, run_consumer
from specrhythm.serving.s1_workload import (
    MODES,
    PROFILE_ENV,
    load_execution,
    write_once,
)

SUBSET = {"G1": "s1-smoke4", "G2": "s1-mixed20", "G3": "s1-mixed100"}
ROTATION = (MODES, ("serial", "pingpong", "target"), ("pingpong", "target", "serial"))


def child_command(*args):
    return [sys.executable, "-m", "specrhythm.serving.s1_cli", *map(str, args)]


def run_plan(gate):
    if gate == "G1":
        return [("target", 0), ("serial", 0), ("pingpong", 0), ("pingpong", 1)]
    if gate == "G2":
        return [(mode, 0) for mode in MODES]
    return [(mode, index) for index, order in enumerate(ROTATION) for mode in order]


def validate_root_policy(root, selected):
    """Reject old roots before launcher/status files can be written or engines started."""
    g0 = read_json(root / "g0.json")
    validate_policy(g0, G0_SCHEMA)
    require(g0.get("valid") is True, "S1-P G0 did not pass")
    for gate in SUBSET if selected == "all" else (selected,):
        manifest, _ = load_execution(root / SUBSET[gate] / "execution-manifest.json")
        require(manifest["execution"] == g0["execution"], "G0/manifest execution identity differs")


def previous_completed(attempts, execution_sha256):
    for directory in sorted(attempts, reverse=True):
        if not (directory / "seal.json").exists():
            continue
        # Corrupt sealed evidence is an error, never silently accepted or skipped.
        report = verify_seal(directory)
        require(
            report.get("execution_sha256") == execution_sha256, "resume execution identity differs"
        )
        if report.get("valid") is True:
            return directory, report
    return None


def recover_owner(saved):
    owner = OwnedProcesses(saved["root_pid"], target_token=saved.get("target_token"))
    owner.observed = {r["pid"]: r for r in saved["observed"]}
    if saved.get("target_token") and sys.platform == "linux":
        marker = ("SR_PHASE4_OWNED_TARGET_TOKEN=" + saved["target_token"]).encode()
        for pid, row in process_table().items():
            try:
                if marker in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
                    owner.observed[pid] = row
            except OSError:
                continue
    require(
        saved["root_pid"] in owner.observed,
        "recovery lacks recorded root PID/start identity; manual process audit required",
    )
    return owner


def cleanup_attempt(directory):
    """Recover only recorded PID-start identities and the unique Target launch token."""
    from specrhythm.phase4.owned_processes import socket_owned_by_pid
    from specrhythm.phase4.process_lifecycle import _unix_socket_identity

    journal_path = directory / "ownership.json"
    startup_path = directory / "draft-owner.json"
    owners, actions = [], []
    journal = read_json(journal_path) if journal_path.is_file() else {}
    if journal:
        owners.append(recover_owner(journal))
    if startup_path.is_file():
        startup = read_json(startup_path)
        owners.append(recover_owner(startup))
        if not journal and (directory / "draft-socket-owner.json").is_file():
            journal = read_json(directory / "draft-socket-owner.json")
        if not journal and startup.get("draft_socket"):
            socket = Path(startup["draft_socket"])
            journal = {
                "draft_socket": str(socket),
                "draft_socket_identity": _unix_socket_identity(socket),
                "draft_socket_proven": any(
                    socket_owned_by_pid(socket, row["pid"]) for row in owners[-1].snapshot()
                ),
            }
    require(
        owners or (directory / "process-lifecycle.json").is_file(),
        "interrupted run lacks process ownership evidence; manual process audit required",
    )
    for selected in (signal.SIGTERM, signal.SIGKILL):
        for owner in owners:
            owner.signal(selected, actions)
        until = time.monotonic() + 5
        while time.monotonic() < until:
            for owner in owners:
                owner.reap(exclude_root=False)
            if all(
                not any(not r["state"].startswith("Z") for r in owner.snapshot())
                for owner in owners
            ):
                break
            time.sleep(0.1)
    remaining = [r for owner in owners for r in owner.snapshot() if not r["state"].startswith("Z")]
    require(not remaining, "owned interrupted processes still alive", processes=remaining)
    socket_path = journal.get("draft_socket")
    if socket_path and Path(socket_path).exists():
        require(
            journal.get("draft_socket_proven") is True
            and _unix_socket_identity(Path(socket_path)) == journal.get("draft_socket_identity"),
            "refusing to remove socket without exact original ownership",
        )
        Path(socket_path).unlink()
    recovery = {
        "valid": True,
        "remaining_owned_pids": [],
        "actions": actions,
        "source_ownership_sha256": sha256_file(journal_path) if journal_path.is_file() else None,
        "live_KV_resumed": False,
    }
    # Sidecar outside the immutable old attempt, retaining old failure evidence.
    write_once(
        directory.parent / f"{directory.name}-cleanup-{uuid.uuid4().hex[:8]}.json", recovery
    )
    return recovery


def run_attempt(root, gate, mode, repeat, manifest_path):
    manifest, _ = load_execution(manifest_path, verify_parent=True)
    validate_execution_files(root, manifest["execution"])
    base = root / gate / f"{repeat}-{mode}"
    base.mkdir(parents=True, exist_ok=True)
    attempts = sorted(base.glob("attempt-*"))
    attempts = [p for p in attempts if p.is_dir()]
    for old in attempts:
        lifecycle = (
            read_json(old / "process-lifecycle.json")
            if (old / "process-lifecycle.json").is_file()
            else {}
        )
        if lifecycle.get("owned_cleanup_completed") is not True:
            cleanup_attempt(old)
    done = previous_completed(attempts, manifest["manifest_sha256"])
    if done:
        return done
    directory = base / f"attempt-{len(attempts) + 1:03d}"
    directory.mkdir()
    atomic_write_json(
        root / "stage.json",
        {
            **policy_fields(),
            "gate": gate,
            "mode": mode,
            "repeat": repeat,
            "directory": str(directory),
            "state": "RUNNING",
        },
    )
    with RunConsole(directory, mode):
        try:
            return execute_attempt(root, mode, manifest_path, manifest, directory)
        except Exception as error:
            # Offline/artifact failures after a failed command must not replace its real status.
            try:
                recorded = read_json(directory / "exit-code.json").get("effective_exit_code")
            except (OSError, ValueError):
                recorded = None
            if type(recorded) is int and recorded != 0:
                raise DataError(str(error), **{
                    **getattr(error, "details", {}),
                    "returncode": recorded, "directory": str(directory), "mode": mode,
                }) from error
            raise


def execute_attempt(root, mode, manifest_path, manifest, directory):
    """Keep direct child log files and the existing owned runner/cleanup path."""
    env = clean_environment(mode, manifest_path)
    env["SR_VLLM_SOURCE"] = manifest["execution"]["vllm_source"]
    socket = Path("/tmp") / f"sr-s1-{uuid.uuid4().hex[:16]}.sock"
    env["SR_S1_DRAFT_SOCKET"] = str(socket)
    draft, draft_log = None, None
    old_environment = dict(os.environ)
    os.environ.clear()
    os.environ.update(env)
    try:
        if mode == "pingpong":
            from specrhythm.phase4.dual_rhythm import write_assignment

            assignment = directory / "dual-rhythm.json"
            env["SR_PHASE4_DUAL_RHYTHM_MANIFEST"] = str(assignment)
            os.environ["SR_PHASE4_DUAL_RHYTHM_MANIFEST"] = str(assignment)
            write_assignment(
                assignment,
                manifest_path.parent / manifest["workload_file"],
                len(manifest["logical"]["request_ids"]),
            )
        if mode != "raw-target":
            draft_token = uuid.uuid4().hex
            draft_log = (directory / "draft-service.log").open("x")
            draft = subprocess.Popen(
                child_command(
                    "draft-child",
                    "--root",
                    root,
                    "--directory",
                    directory,
                    "--mode",
                    mode,
                    "--manifest",
                    manifest_path,
                ),
                env={
                    **env,
                    "CUDA_VISIBLE_DEVICES": "0",
                    "SR_PHASE4_OWNED_TARGET_TOKEN": draft_token,
                },
                stdin=subprocess.DEVNULL,
                stdout=draft_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            owner = OwnedProcesses(draft.pid, target_token=draft_token)
            owner.snapshot()
            write_once(
                directory / "draft-owner.json",
                {
                    "root_pid": draft.pid,
                    "launch_environment": {**environment_evidence(), "CUDA_VISIBLE_DEVICES": "0"},
                    "observed": list(owner.observed.values()),
                    "target_token": draft_token,
                    "draft_socket": str(socket),
                },
            )
            deadline = time.monotonic() + 900
            while not (socket.is_socket() and (directory / "draft-service-ready.json").is_file()):
                if draft.poll() is not None or time.monotonic() >= deadline:
                    draft_status = draft.poll()
                    failure_status = (
                        draft_status if draft_status not in (None, 0)
                        else 124 if draft_status is None else 1
                    )
                    rc, _ = run_owned_target(
                        [sys.executable, "-c", "raise SystemExit(1)"],
                        target_log=directory / "target.log",
                        artifact_path=directory / "process-lifecycle.json",
                        draft_pid=draft.pid,
                        draft_socket=socket,
                        ownership_journal=directory / "ownership.json",
                    )
                    write_once(directory / "exit-code.json", {
                        "effective_exit_code": failure_status,
                        "draft_exit_code": draft_status,
                        "cleanup_exit_code": rc,
                        "coordinator_exit_code": None,
                        "target_generation_started": False,
                    })
                    raise DataError(
                        "Draft startup failed or timed out",
                        returncode=failure_status,
                        draft_exit_code=draft_status,
                        cleanup_exit_code=rc,
                        directory=str(directory),
                        mode=mode,
                    )
                time.sleep(0.2)
            from specrhythm.phase4.owned_processes import socket_owned_by_pid
            from specrhythm.phase4.process_lifecycle import _unix_socket_identity

            write_once(
                directory / "draft-socket-owner.json",
                {
                    "draft_socket": str(socket),
                    "draft_socket_identity": _unix_socket_identity(socket),
                    "draft_socket_proven": socket_owned_by_pid(socket, draft.pid),
                },
            )
        command = child_command(
            "child",
            "--root",
            root,
            "--directory",
            directory,
            "--mode",
            mode,
            "--manifest",
            manifest_path,
        )
        write_once(
            directory / "command.json",
            {
                **policy_fields(),
                "argv": command,
                "mode": mode,
                "execution_sha256": manifest["manifest_sha256"],
                "fresh_engine_lifecycle": True,
                "timeout_seconds": 14400,
                "launch_environment": environment_evidence(),
            },
        )
        rc, lifecycle = run_owned_target(
            command,
            target_log=directory / "target.log",
            artifact_path=directory / "process-lifecycle.json",
            draft_pid=draft.pid if draft else None,
            draft_socket=socket if draft else None,
            timeout_seconds=14400,
            ownership_journal=directory / "ownership.json",
        )
        draft_status = draft.wait(timeout=15) if draft else None
        if rc == 0 and draft_status not in (None, 0):
            rc = draft_status
        write_once(
            directory / "exit-code.json",
            {"effective_exit_code": rc, "coordinator_exit_code": lifecycle["target_exit_status"],
             "draft_exit_code": draft_status},
        )
    except Exception as error:
        write_once(
            directory / "launcher-failure.json",
            {
                "error": str(error),
                "details": getattr(error, "details", {}),
                "draft_exit_code": draft.poll() if draft else None,
            },
        )
        if (directory / "draft-owner.json").exists() or (directory / "ownership.json").exists():
            try:
                cleanup_attempt(directory)
            except Exception as cleanup_error:
                raise DataError(
                    str(error),
                    returncode=getattr(error, "details", {}).get("returncode", 1),
                    original_details=getattr(error, "details", {}),
                    cleanup_error=str(cleanup_error),
                    directory=str(directory),
                    mode=mode,
                ) from error
        raise
    finally:
        if draft_log:
            draft_log.close()
        os.environ.clear()
        os.environ.update(old_environment)
    if rc != 0:
        report = failed_execution_report(root, mode, manifest_path, manifest, directory, rc)
        seal_run(directory, report)
        raise DataError(
            report["errors"][0], returncode=rc, directory=str(directory), mode=mode,
            primary_error=report["primary_error"],
            secondary_diagnostics=report["secondary_diagnostics"],
        )
    # Recheck frozen bytes/metadata after the owned run, outside measurement.
    validate_execution_files(root, manifest["execution"])
    if mode == "raw-target":
        raw = read_json(directory / "raw.json") if (directory / "raw.json").is_file() else {}
        report = {
            "schema_version": RESULT_SCHEMA,
            **policy_fields(),
            "valid": rc == 0 and lifecycle["run_valid"] and raw.get("valid") is True,
            "errors": raw.get("errors", []),
            "mode": mode,
            "execution_sha256": manifest["manifest_sha256"],
            "raw_reference": raw,
        }
    else:
        report = inspect_run(manifest_path, directory, mode)
    seal_run(directory, report)
    require(
        report["valid"],
        "S1 run qualification failed; later gates stopped",
        errors=report.get("errors"),
        error_details=report.get("error_details"),
        draft_backend_checks=report.get("draft_backend_checks"),
        directory=str(directory),
    )
    return directory, report


def failed_execution_report(root, mode, manifest_path, manifest, directory, rc):
    """A failed command stays primary; incomplete downstream reports are diagnostics."""
    secondary, failures = [], []
    for source, name in (("Target", "child-failure.json"), ("Draft", "draft-child-failure.json")):
        path = directory / name
        if path.is_file():
            try:
                failure = read_json(path)
                require(isinstance(failure, dict), "child failure must be an object")
                failures.append({**failure, "source": source, "artifact": str(path)})
            except (OSError, ValueError) as error:
                secondary.append({"artifact": str(path), "error": str(error)})
    failures.sort(key=lambda f: f.get("timestamp_ns", 0))
    primary = failures[0] if failures else {
        "error": f"execution failed with effective exit code {rc}",
        "artifact": str(directory / "exit-code.json"),
    }
    secondary.extend(failures[1:])
    try:
        validate_execution_files(root, manifest["execution"])
    except Exception as error:
        secondary.append({"stage": "post-execution input validation", "error": str(error)})
    report = {}
    if mode != "raw-target":
        try:
            report = inspect_run(manifest_path, directory, mode)
            secondary.extend({"stage": "post-execution qualification", "error": e}
                             for e in report.get("errors", []))
        except Exception as error:
            secondary.append({"stage": "post-execution qualification", "error": str(error)})
    for name in ("raw.json", "plugin-report.json", "draft-backend-report.json"):
        path = directory / name
        if not path.is_file() and (mode != "raw-target" or name == "raw.json"):
            secondary.append({"artifact": str(path), "status": "MISSING_AFTER_EXECUTION_FAILURE"})
    message = f"S1 {primary.get('source', mode)} execution failed: {primary['error']}"
    return {
        **report, "schema_version": RESULT_SCHEMA, **policy_fields(), "mode": mode,
        "execution_sha256": manifest["manifest_sha256"],
        "valid": False, "performance_result": False,
        "errors": [message], "primary_error": primary, "secondary_diagnostics": secondary,
        "error_details": {"returncode": rc, "primary_error": primary},
    }


def require_previous_gate(root, gate):
    previous = {"G2": "G1", "G3": "G2"}.get(gate)
    if previous:
        report = read_json(root / previous / "comparison.json")
        validate_policy(report, COMPARISON_SCHEMA)
        require(report["valid"], "previous S1 gate did not pass", gate=previous)
        for path, checksum in report["run_seals"].items():
            directory = root / path
            require(sha256_file(directory / "seal.json") == checksum, "previous gate seal changed")
            require(verify_seal(directory)["valid"], "previous gate run invalid")
        require(
            offline_comparison(root, previous) == report,
            "previous gate comparison differs from sealed run evidence",
        )


def offline_comparison(root, gate):
    """Independent read-only requalification; this path never dispatches a GPU child."""
    manifest_path = root / SUBSET[gate] / "execution-manifest.json"
    manifest, _ = load_execution(manifest_path, verify_parent=True)
    reports, seals = {mode: [] for mode in MODES}, {}
    for mode, repeat in run_plan(gate):
        done = previous_completed(
            (root / gate / f"{repeat}-{mode}").glob("attempt-*"), manifest["manifest_sha256"]
        )
        require(
            done is not None,
            "offline comparison lacks completed sealed run",
            mode=mode,
            repeat=repeat,
        )
        directory, result = done
        seals[str(directory.relative_to(root))] = sha256_file(directory / "seal.json")
        recomputed = inspect_run(manifest_path, directory, mode)
        require(
            recomputed == result,
            "offline run requalification differs from sealed result",
            directory=str(directory),
            errors=recomputed["errors"],
        )
        reports[mode].append(recomputed)
    result = compare_results(reports)
    result.update(gate=gate, run_seals=seals, run_order=run_plan(gate))
    return json.loads(json.dumps(result))


def gate_capacity(root, gate, manifest_path):
    estimate = read_json(root / "capacity-estimates.json")[SUBSET[gate]]
    valid = all(r["valid"] for r in estimate.values())
    actual = {}
    if gate == "G3":
        _, definitions = load_execution(manifest_path)
        g2 = read_json(root / "G2/comparison.json")
        for path in g2["run_seals"]:
            result = verify_seal(root / path)
            if result.get("mode") not in MODES:
                continue
            runtime = result["effective_runtime"]
            target = runtime["target_worker_ranks"]
            draft = runtime["draft_startup"]
            checks = [
                {
                    "role": "target",
                    "available_blocks": r["s1_effective_capacity"]["num_gpu_blocks"],
                    **resident_capacity(definitions, r["s1_effective_capacity"]["block_size"]),
                }
                for r in target
            ]
            checks.append(
                {
                    "role": "draft",
                    "available_blocks": draft["kv_cache_num_blocks"],
                    **resident_capacity(definitions, draft["block_size"]),
                }
            )
            actual[path] = checks
            valid &= all(c["available_blocks"] >= c["required_blocks"] for c in checks)
    report = {
        **policy_fields(),
        "valid": bool(valid),
        "gate": gate,
        "status": "READY" if valid else "BLOCKED",
        "estimate": estimate,
        "actual_G2_engine_capacity": actual,
        "budgets_or_requests_changed": False,
    }
    destination = root / gate / "capacity-check.json"
    if destination.exists():
        require(read_json(destination) == report, "capacity evidence changed during resume")
    else:
        write_once(destination, report)
    require(
        valid,
        "S1 resident capacity BLOCKED; workload and budgets preserved",
        gate=gate,
        status="BLOCKED",
    )


def gate(root, selected):
    validate_root_policy(root, selected)
    stages = ("G1", "G2", "G3") if selected == "all" else (selected,)
    for current in stages:
        require_previous_gate(root, current)
        manifest_path = root / SUBSET[current] / "execution-manifest.json"
        gate_capacity(root, current, manifest_path)
        reports = {mode: [] for mode in MODES}
        seals = {}
        for mode, repeat in run_plan(current):
            directory, result = run_attempt(root, current, mode, repeat, manifest_path)
            seals[str(directory.relative_to(root))] = sha256_file(directory / "seal.json")
            reports[mode].append(result)
            # Stop on invalid runs or input/config mismatch, never independent output differences.
            if all(reports.values()):
                interim = compare_results(reports)
                if not interim["valid"]:
                    write_once(
                        root / current / f"failed-comparison-{uuid.uuid4().hex[:8]}.json", interim
                    )
                    raise DataError(
                        "S1-P execution/input qualification failed; no later gate",
                        errors=interim["errors"],
                    )
        comparison = compare_results(reports)
        comparison["run_seals"] = seals
        comparison["gate"] = current
        comparison["run_order"] = run_plan(current)
        destination = root / current / "comparison.json"
        if destination.exists():
            require(
                read_json(destination) == json.loads(json.dumps(comparison)),
                "immutable completed gate comparison changed",
            )
        else:
            write_once(destination, comparison)
        require(comparison["valid"], "S1 gate comparison failed")
        atomic_write_json(
            root / "stage.json", {**policy_fields(), "gate": current, "state": "PASS"}
        )
    return 0


def launcher_alive(root):
    lock = root / "launcher.json"
    if not lock.exists():
        return False
    saved = read_json(lock)
    current = process_table().get(saved["pid"])
    return bool(
        current
        and current["start_identity"] == saved["start_identity"]
        and not current["state"].startswith("Z")
    )


def start(root, selected):
    validate_root_policy(root, selected)
    require(not launcher_alive(root), "S1 launcher already running")
    attempt = uuid.uuid4().hex[:12]
    log = root / f"launcher-{attempt}.log"
    with log.open("x") as output:
        process = subprocess.Popen(
            child_command("supervise", "--root", root, "--gate", selected, "--launch-id", attempt),
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(json.dumps({"pid": process.pid, "log": str(log), "launch_id": attempt}))
    return 0


def supervise(root, selected, launch_id):
    import fcntl

    # Kernel lock closes the simultaneous-start race; JSON is only status provenance.
    with (root / "launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = process_table()[os.getpid()]
        atomic_write_json(
            root / "launcher.json",
            {
                **policy_fields(),
                "pid": os.getpid(),
                "start_identity": identity["start_identity"],
                "launch_id": launch_id,
                "gate": selected,
            },
        )
        status = 1
        try:
            status = gate(root, selected)
        except Exception as error:
            status = getattr(error, "details", {}).get("returncode", 1)
            previous_stage = (
                read_json(root / "stage.json") if (root / "stage.json").is_file() else {}
            )
            context = {k: previous_stage[k] for k in ("gate", "mode", "repeat", "directory")
                       if k in previous_stage}
            atomic_write_json(
                root / "stage.json",
                {
                    **policy_fields(),
                    **context,
                    "state": getattr(error, "details", {}).get("status", "FAILED"),
                    "error": str(error),
                    "details": getattr(error, "details", {}),
                },
            )
            directory = getattr(error, "details", {}).get("directory", context.get("directory"))
            if directory and root.resolve() not in Path(directory).resolve().parents:
                directory = None
            print_failure(error, directory=directory, mode=context.get("mode", selected))
        finally:
            value = {
                **policy_fields(), "launch_id": launch_id, "exit_code": status, "gate": selected
            }
            write_once(root / f"exit-code-{launch_id}.json", value)
            atomic_write_json(root / "exit-code.json", value)
        return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "gate",
            "start",
            "resume",
            "status",
            "supervise",
            "child",
            "draft-child",
            "cleanup",
            "verify",
            "bundle",
            "compare",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--s0", type=Path)
    parser.add_argument(
        "--vllm-source", type=Path, default=Path("/root/autodl-tmp/src/vllm-v0.25.1")
    )
    parser.add_argument("--gate", choices=("G1", "G2", "G3", "all"), default="all")
    parser.add_argument("--mode", choices=(*MODES, "raw-target"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--launch-id")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        if args.command == "prepare":
            require(args.s0 is not None, "prepare requires --s0")
            cleaned = clean_environment("target")
            os.environ.clear()
            os.environ.update(cleaned)
            return prepare(root, args.s0.resolve(), args.vllm_source)
        if args.command in ("start", "resume"):
            return start(root, args.gate)
        if args.command in ("supervise", "gate", "child", "draft-child"):
            validate_root_policy(root, args.gate)
        if args.command == "supervise":
            return supervise(root, args.gate, args.launch_id)
        if args.command == "gate":
            return supervise(root, args.gate, uuid.uuid4().hex[:12])
        if args.command == "status":
            print(
                json.dumps(
                    {
                        "alive": launcher_alive(root),
                        **{
                            p: read_json(root / p) if (root / p).is_file() else None
                            for p in ("launcher.json", "stage.json", "exit-code.json")
                        },
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "child":
            result = run_consumer(args.mode, args.manifest, args.directory, root)
            return 0 if result.get("valid") else 1
        if args.command == "draft-child":
            load_execution(args.manifest, verify_parent=True)
            environment_evidence()
            from specrhythm.phase4.batched_draft_service import serve_batched_draft
            from specrhythm.phase4.config import load_phase4_config
            from specrhythm.phase4.dual_service import run_dual_draft_service

            config = load_phase4_config(str(root / "config.json"))
            os.environ[PROFILE_ENV] = str(args.manifest.resolve())
            kwargs = dict(
                socket_path=Path(os.environ["SR_S1_DRAFT_SOCKET"]),
                event_log_path=args.directory
                / (
                    "draft-work-events.jsonl"
                    if args.mode == "pingpong"
                    else "draft-service-events.jsonl"
                ),
                ready_path=args.directory / "draft-service-ready.json",
            )
            if args.mode == "pingpong":
                run_dual_draft_service(
                    config, **kwargs, transport_log_path=args.directory / "transport-events.jsonl"
                )
            else:
                serve_batched_draft(config, **kwargs)
            return 0
        if args.command == "cleanup":
            require(
                not launcher_alive(root),
                "stop/finish the owned S1 launcher before recovery cleanup",
            )
            require(
                args.directory and root in args.directory.resolve().parents,
                "cleanup directory is outside S1 root",
            )
            cleanup_attempt(args.directory)
            return 0
        if args.command == "verify":
            require(args.directory is not None, "verify requires a sealed run --directory")
            result = verify_seal(args.directory)
            require(result["valid"], "sealed run did not pass")
            print("S1 SEALED RUN PASS")
            return 0
        if args.command == "compare":
            require(args.gate != "all", "offline compare requires one explicit --gate")
            result = offline_comparison(root, args.gate)
            write_once(
                root / args.gate / f"offline-comparison-{uuid.uuid4().hex[:8]}.json", result
            )
            print(json.dumps(result, indent=2))
            return 0 if result["valid"] else 1
        if args.command == "bundle":
            return bundle(root)
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
        # Capture the original child exception before teardown/missing reports can obscure it.
        # This runs only after a failure, never in the measured decoding path.
        if args.command in ("child", "draft-child") and args.directory is not None:
            failure_path = args.directory / (
                "child-failure.json" if args.command == "child" else "draft-child-failure.json"
            )
            try:
                write_once(failure_path, {
                    "error": str(error), "error_type": type(error).__name__,
                    "details": getattr(error, "details", {}), "mode": args.mode,
                    "timestamp_ns": time.monotonic_ns(), "traceback": traceback.format_exc(),
                })
            except (OSError, ValueError) as diagnostic_error:
                print(json.dumps({"secondary_diagnostic": str(diagnostic_error),
                                  "artifact": str(failure_path)}), file=sys.stderr)
        print(
            json.dumps(
                {"valid": False, "error": str(error), "details": getattr(error, "details", {})}
            ),
            file=sys.stderr,
        )
        return getattr(error, "details", {}).get("returncode", 1)
    return 1


def bundle(root):
    import tarfile

    require(not launcher_alive(root), "cannot seal while S1 launcher is running")
    for path in root.rglob("ownership.json"):
        saved = read_json(path)
        require(
            not any(not row["state"].startswith("Z") for row in recover_owner(saved).snapshot()),
            "cannot bundle while an owned Target process is alive",
            ownership=str(path),
        )
    for path in root.rglob("draft-owner.json"):
        saved = read_json(path)
        require(
            not any(not row["state"].startswith("Z") for row in recover_owner(saved).snapshot()),
            "cannot bundle while an owned Draft process is alive",
            ownership=str(path),
        )
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "launcher.lock")
    require(not any(p.is_symlink() for p in files), "review bundle cannot follow symlinks")
    inventory_path = root.parent / f"{root.name}-review-inventory.json"
    archive_path = root.parent / f"{root.name}-review.tar.gz"
    write_once(inventory_path, {str(p.relative_to(root)): sha256_file(p) for p in files})
    with archive_path.open("xb") as handle, tarfile.open(fileobj=handle, mode="w:gz") as archive:
        for p in files:
            archive.add(p, arcname=f"{root.name}/{p.relative_to(root)}", recursive=False)
        archive.add(inventory_path, arcname="review-inventory.json", recursive=False)
    print(
        json.dumps(
            {
                "archive": str(archive_path),
                "sha256": sha256_file(archive_path),
                "inventory": str(inventory_path),
                "file_count": len(files),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
