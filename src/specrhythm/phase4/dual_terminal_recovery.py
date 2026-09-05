"""Strict, read-only revalidation of the completed 04e9b61 Dual execution.

An exit status of one is never sufficient. Recovery joins the pinned CLI's final
artifact, complete outputs, all embedded validators, process ownership/cleanup,
final TP synchronization, and the explicit retired-ready evidence.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

from specrhythm.phase4.decode_ready import load_decode_ready_manifest
from specrhythm.phase4.dual_correctness import (
    validate_dual_runner_evidence,
    validate_overlap_witness,
    validate_proposal_lifecycle_events,
    validate_request_state_events,
    validate_round_accounting,
    validate_scheduler_cycles,
)
from specrhythm.phase4.dual_runner import (
    _validate_request_identity_report,
    build_cycle_and_overlap_events,
    summarize_retired_ready_results,
)
from specrhythm.phase4.dual_terminal import _require, build_terminal_reconciliation
from specrhythm.phase4.manifest import sha256_file
from specrhythm.phase4.process_lifecycle import validate_lifecycle_artifact
from specrhythm.phase4.resident_runner import _decode_rows
from specrhythm.phase4.stock_vllm import load_smoke_requests
from specrhythm.phase4.transport import CheckpointJsonl

SOURCE_EXECUTION_COMMIT = "04e9b6141e3846835e6fdee0a42cdb9e8d021e4e"
RECOVERY_SCHEMA = "specrhythm.phase4b2-dual-terminal-revalidation.v1"
JSON_FILES = (
    "resident-dual.json",
    "decode-ready-manifest.json",
    "setup-ready.json",
    "runtime-manifest.json",
    "plugin-report.json",
    "process-lifecycle.json",
)
JSONL_FILES = (
    "output-checkpoint.jsonl",
    "request-state-events.jsonl",
    "scheduler-events.jsonl",
    "proposal-lifecycle-events.jsonl",
    "proposal-events.jsonl",
    "draft-work-events.jsonl",
    "verification-events.jsonl",
    "cycle-events.jsonl",
    "overlap-events.jsonl",
    "timing-events.jsonl",
    "target-diagnostics.jsonl",
    "timestamped-target-log.jsonl",
)


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"artifact is not an object: {path}")
    return value


def _inventory(root: Path) -> dict[str, str]:
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    _require(not any(path.is_symlink() for path in root.rglob("*")), "raw run contains symlinks")
    return {str(path.relative_to(root)): sha256_file(path) for path in paths}


def _read_checkpoint(path: Path) -> list[dict[str, Any]]:
    # Verify framing before removing it for comparisons with embedded JSON rows.
    return [
        {key: value for key, value in row.items() if key != "record_sha256"}
        for row in CheckpointJsonl(path).read()
    ]


def audit_terminal_recovery(
    *,
    run_root: Path,
    workload_path: Path,
    config_path: Path,
    topology_path: Path,
    patch_manifest_path: Path,
    observation_ns: int,
) -> dict[str, Any]:
    """Recompute every recovery gate without writing or relabeling raw artifacts."""

    from specrhythm.phase4.performance import (
        _commit_events,
        _measurement_code_git_commit,
        _validate_final_sync,
        extract_performance_boundary,
    )

    root = run_root.resolve()
    paths = {name: root / name for name in (*JSON_FILES, *JSONL_FILES, "target.log")}
    paths.update(
        workload=workload_path.resolve(),
        config=config_path.resolve(),
        topology=topology_path.resolve(),
        patch_manifest=patch_manifest_path.resolve(),
    )
    _require(
        all(path.is_file() for path in paths.values()), "recovery requires complete raw evidence"
    )
    before = _inventory(root)
    digests = {name: sha256_file(path) for name, path in paths.items()}
    objects = {name: _object(paths[name]) for name in JSON_FILES}
    rows = {
        name: _read_checkpoint(paths[name])
        for name in JSONL_FILES
        if name != "timestamped-target-log.jsonl"
    }
    # The timestamp wrapper writes ordinary JSONL, not CheckpointJsonl records.
    rows["timestamped-target-log.jsonl"] = [
        json.loads(line) for line in paths["timestamped-target-log.jsonl"].read_text().splitlines()
    ]
    _require(
        all(
            isinstance(row, dict)
            and type(row.get("timestamp_ns")) is int
            and isinstance(row.get("line"), str)
            for row in rows["timestamped-target-log.jsonl"]
        ),
        "timestamped Target log is malformed",
    )
    raw = objects["resident-dual.json"]
    manifest = load_decode_ready_manifest(objects["decode-ready-manifest.json"])
    runtime = objects["runtime-manifest.json"]
    _require(
        manifest.specrhythm_git_commit == runtime.get("git_commit") == SOURCE_EXECUTION_COMMIT,
        "terminal recovery is restricted to execution commit " + SOURCE_EXECUTION_COMMIT,
    )
    _require(
        raw.get("schema_version") == "specrhythm.phase4b1-resident-dual-run.v1"
        and raw.get("mode") == "decode-only-dual-batch"
        and raw.get("valid") is False
        and raw.get("request_count") == 100
        and raw.get("phase4b2_performance_candidate") is True,
        "source must be the invalid completed 100-request Phase-4B.2 resident Dual run",
    )
    _require(
        manifest.target_physical_gpu_ids == (1, 2)
        and manifest.draft_physical_gpu_ids == (0,)
        and manifest.target_tensor_parallel_size == 2
        and manifest.draft_tensor_parallel_size == 1,
        "recovery source has the wrong GPU/TP placement",
    )
    _require(manifest.workload_sha256 == digests["workload"], "frozen workload digest differs")
    for name, filename in (
        ("workload", "workload"),
        ("decode_ready_manifest", "decode-ready-manifest.json"),
        ("setup_ready", "setup-ready.json"),
        ("output_checkpoint", "output-checkpoint.jsonl"),
        ("runtime_manifest", "runtime-manifest.json"),
    ):
        _require(
            raw.get("artifact_sha256", {}).get(name) == digests[filename],
            f"raw resident artifact digest differs: {name}",
        )
    for name in ("config", "workload", "topology"):
        _require(
            runtime.get("inputs", {}).get(name + "_sha256") == digests[name],
            f"runtime input digest differs: {name}",
        )
    _require(
        runtime.get("patch_manifest_sha256") == digests["patch_manifest"],
        "runtime patch manifest digest differs",
    )
    _require(
        raw.get("decode_ready_manifest_sha256")
        == runtime.get("decode_ready_manifest_sha256")
        == manifest.manifest_sha256,
        "raw/runtime decode-ready identity differs",
    )
    requests = load_smoke_requests(paths["workload"], 100, require_task_mixture=True)
    _require(
        {
            kind: sum(row.task_class == kind for row in requests)
            for kind in ("code", "chat", "summarization")
        }
        == {"code": 60, "chat": 20, "summarization": 20},
        "recovery requires corrected-100 task mixture",
    )
    outputs = rows["output-checkpoint.jsonl"]
    _require(
        len(outputs) == 100 and raw.get("outputs") == outputs,
        "raw output and complete 100-row checkpoint differ",
    )
    _require(
        raw.get("decode_only_outputs") == _decode_rows(outputs, manifest),
        "raw decode-only output differs from serialized completion",
    )
    plugin = objects["plugin-report.json"]
    _require(
        raw.get("request_identity") == plugin.get("request_identity"),
        "raw/plugin identity binding differs",
    )
    _require(
        raw.get("global_setup_ready") == objects["setup-ready.json"],
        "raw/setup-ready evidence differs",
    )
    _require(
        raw.get("worker_ranks") == runtime.get("worker_ranks"),
        "raw/runtime Target worker identities differ",
    )
    states = rows["request-state-events.jsonl"]
    state_errors = validate_request_state_events(states)
    _require(
        bool(state_errors)
        and raw.get("errors") == state_errors
        and all(
            error.endswith(": final state is DRAFT_SYNC, not TERMINAL") for error in state_errors
        ),
        "raw error set must contain exactly the reproduced DRAFT_SYNC terminal gaps",
    )
    lifecycle = rows["proposal-lifecycle-events.jsonl"]
    proposals = rows["proposal-events.jsonl"]
    scheduler = rows["scheduler-events.jsonl"]
    drafts = rows["draft-work-events.jsonl"]
    cycles, overlaps = build_cycle_and_overlap_events(
        drafts,
        rows["verification-events.jsonl"],
        proposals,
    )
    _require(
        cycles == rows["cycle-events.jsonl"] and overlaps == rows["overlap-events.jsonl"],
        "stored cycle/overlap evidence differs from physical source intervals",
    )
    _require(
        raw.get("overlap_gate")
        == {
            "required_for_run_validity": True,
            "valid": True,
            "errors": [],
        },
        "raw physical overlap gate did not pass",
    )
    _require(
        raw.get("retired_ready_results") == summarize_retired_ready_results(scheduler),
        "raw retired-ready summary differs from scheduler evidence",
    )
    replayed_errors = [
        *validate_proposal_lifecycle_events(lifecycle),
        *validate_scheduler_cycles(
            scheduler, proposal_lifecycle_rows=lifecycle, state_rows=states, draft_rows=drafts
        ),
        *validate_round_accounting(proposals),
        *_validate_request_identity_report(plugin, requests),
        *validate_overlap_witness(overlaps),
        *validate_dual_runner_evidence(raw, manifest, paths["decode-ready-manifest.json"]),
    ]
    _require(
        not replayed_errors, "unrelated Dual validation failure: " + "; ".join(replayed_errors)
    )
    _require(all(row.get("success") is True for row in drafts), "asynchronous Draft work failed")
    _require(
        all(
            row.get("draft_start_ns", -1) >= manifest.measurement_start_ns
            for row in lifecycle
            if row.get("lifecycle_state") == "CREATED"
        ),
        "a proposal predates the decode-ready measurement boundary",
    )
    timing = rows["timing-events.jsonl"]
    boundary, boundary_errors = extract_performance_boundary(timing, consumer="dual-batch")
    _require(
        not boundary_errors and boundary is not None,
        "invalid performance boundary: " + "; ".join(boundary_errors),
    )
    commits, commit_errors = _commit_events("dual-batch", root, timing, boundary)
    _require(not commit_errors and bool(commits), "measured commit evidence is invalid or empty")
    final_sync, sync_errors = _validate_final_sync(
        raw.get("phase4b2_final_sync"),
        2,
        (1, 2),
        max(row["timestamp_ns"] for row in commits),
    )
    _require(not sync_errors, "; ".join(sync_errors))
    _require(
        type(raw.get("run_start_ns")) is int
        and type(raw.get("run_end_ns")) is int
        and raw["run_start_ns"] < boundary < raw["run_end_ns"]
        and all(
            row["final_cuda_synchronize_complete_ns"] <= raw["run_end_ns"] for row in final_sync
        ),
        "generation/final synchronization completion ordering is invalid",
    )
    process = _post_validation_process(objects["process-lifecycle.json"], root, raw)
    messages = [row.get("line", "") for row in rows["timestamped-target-log.jsonl"]]
    messages.append(paths["target.log"].read_text(encoding="utf-8"))
    _require(
        not any(
            marker in message
            for message in messages
            for marker in (
                "Traceback (most recent call last)",
                "EngineDeadError",
                "OutOfMemoryError",
                "EngineCore encountered a fatal error",
                "CUDA error:",
                "Phase-4B.1 resident Dual run failed:",
            )
        ),
        "runtime exception evidence forbids post-validation recovery",
    )
    _require(
        observation_ns > max(raw["run_end_ns"], process["source_process_end_ns"]),
        "reconciliation observation must follow completed execution",
    )
    reconciliation = build_terminal_reconciliation(
        requests=requests,
        outputs=outputs,
        manifest=manifest,
        identity=plugin["request_identity"],
        state_rows=states,
        scheduler_rows=scheduler,
        lifecycle_rows=lifecycle,
        proposal_rows=proposals,
        observation_ns=observation_ns,
    )
    reconciled_ids = reconciliation["reconciled_request_ids"]
    _require(
        set(state_errors)
        == {
            f"{request_id}: final state is DRAFT_SYNC, not TERMINAL"
            for request_id in reconciled_ids
        },
        "reconciliation does not explain exactly the raw validator errors",
    )
    _require(_inventory(root) == before, "raw evidence changed during revalidation")
    _require(
        all(sha256_file(paths[name]) == digest for name, digest in digests.items()),
        "input evidence changed during revalidation",
    )
    return {
        "schema_version": RECOVERY_SCHEMA,
        "valid": True,
        "errors": [],
        "derived_artifact": True,
        "performance_result": False,
        "source_run_root": str(root),
        "source_resident_valid": False,
        "source_resident_errors": state_errors,
        "source_inputs": {
            name: {"path": str(path), "sha256": digests[name]} for name, path in paths.items()
        },
        "source_file_inventory": before,
        "source_artifacts_immutable": True,
        "terminal_state_reconciliation": {**reconciliation, "recovered": True},
        "process_lifecycle_interpretation": process,
        "revalidation_git_commit": _measurement_code_git_commit(),
    }


def _post_validation_process(
    lifecycle: Mapping[str, Any],
    root: Path,
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    errors = validate_lifecycle_artifact(lifecycle)
    _require(not errors, "; ".join(errors))
    reap = lifecycle.get("child_reap_result", {})
    draft = lifecycle.get("draft_shutdown_result", {})
    _require(
        lifecycle.get("target_exit_status") == lifecycle.get("effective_exit_status") == 1
        and lifecycle.get("run_valid") is False
        and lifecycle.get("cleanup_valid") is True
        and lifecycle.get("launch_error") is None
        and lifecycle.get("remaining_owned_pids") == []
        and lifecycle.get("term_kill_actions") == []
        and reap.get("owned_group_empty") is True
        and reap.get("wrapper_exited_with_descendants_alive") is False
        and draft.get("valid") is True
        and draft.get("alive_after_cleanup") is False
        and draft.get("socket_exists_after_cleanup") is False,
        "process status is not an isolated post-validation failure with clean shutdown",
    )
    _require(
        type(lifecycle.get("start_monotonic_ns")) is int
        and type(lifecycle.get("exit_monotonic_ns")) is int
        and lifecycle["start_monotonic_ns"]
        < raw["run_start_ns"]
        < raw["run_end_ns"]
        < lifecycle["exit_monotonic_ns"],
        "completed output is not contained in the owned process lifetime",
    )
    command = lifecycle.get("command", [])
    _require(
        isinstance(command, list)
        and len(command) > 11
        and command[:4]
        == [
            "env",
            "CUDA_VISIBLE_DEVICES=1,2",
            "VLLM_USE_V2_MODEL_RUNNER=0",
            "VLLM_BATCH_INVARIANT=1",
        ]
        and Path(command[4]).name in {"python", "python3"}
        and command[5].endswith("integrations/vllm/phase4b2_timestamp_command.py")
        and command[6:9] == ["--output", str(root / "timestamped-target-log.jsonl"), "--"]
        and command[9:11] == ["specrhythm", "phase4b1-resident-dual-run"],
        "coordinator command is not the pinned resident Dual CLI through its timestamp wrapper",
    )
    options = command[11:]
    _require(
        options.count("--phase4b2-performance") == 1, "missing performance request in command"
    )
    options = [item for item in options if item != "--phase4b2-performance"]
    _require(len(options) % 2 == 0, "malformed resident Dual command options")
    arguments = dict(zip(options[::2], options[1::2]))
    _require(len(arguments) * 2 == len(options), "duplicate resident Dual command options")
    for flag, filename in (
        ("output", "resident-dual.json"),
        ("output-checkpoint", "output-checkpoint.jsonl"),
        ("request-state-events", "request-state-events.jsonl"),
        ("scheduler-events", "scheduler-events.jsonl"),
        ("proposal-lifecycle-events", "proposal-lifecycle-events.jsonl"),
        ("runtime-manifest", "runtime-manifest.json"),
    ):
        _require(
            arguments.get("--" + flag) == str(root / filename),
            f"coordinator command does not bind raw artifact: {flag}",
        )
    for flag, expected in (
        ("request-count", "100"),
        ("microbatch-size", "2"),
        ("test-coordination", "none"),
        ("overlap-requirement", "required"),
    ):
        _require(
            arguments.get("--" + flag) == expected, f"unexpected source execution option: {flag}"
        )
    return {
        "classification": "completed-generation-terminal-state-validation-failure",
        "source_target_exit_status": 1,
        "source_effective_exit_status": 1,
        "source_run_valid": False,
        "cleanup_valid": True,
        "pinned_cli_return_rule": (
            "runner wrote final resident artifact; return 1 when valid=false"
        ),
        "source_process_end_ns": lifecycle["exit_monotonic_ns"],
        "raw_process_lifecycle_modified": False,
    }


def recover_terminal_state(*, output_dir: Path, **inputs: Any) -> dict[str, Any]:
    """Write a new certificate and complete derived trace outside the source root."""

    output_dir = output_dir.resolve()
    source_root = Path(inputs["run_root"]).resolve().parent
    _require(
        output_dir != source_root and source_root not in output_dir.parents,
        "derived artifacts must be outside the immutable three-mode source root",
    )
    _require(not output_dir.exists(), "refusing to reuse derived recovery directory")
    lifecycle = _object(Path(inputs["run_root"]) / "process-lifecycle.json")
    observation_ns = max(time.monotonic_ns(), lifecycle.get("exit_monotonic_ns", 0) + 1)
    report = audit_terminal_recovery(**inputs, observation_ns=observation_ns)
    output_dir.mkdir(parents=True, exist_ok=False)
    state_path = output_dir / "request-state-events.reconciled.jsonl"
    original = _read_checkpoint(Path(inputs["run_root"]) / "request-state-events.jsonl")
    state_path.touch(exist_ok=False)
    for row in [*original, *report["terminal_state_reconciliation"]["events"]]:
        CheckpointJsonl(state_path).append(row)
    report["derived_state_events"] = {"path": str(state_path), "sha256": sha256_file(state_path)}
    with (output_dir / "terminal-state-revalidation.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def validate_recovery_certificate(certificate_path: Path, **inputs: Any) -> dict[str, Any]:
    """A caller must recompute the proof; a certificate's valid flag grants nothing."""

    certificate = _object(certificate_path)
    _require(
        certificate.get("schema_version") == RECOVERY_SCHEMA, "unsupported recovery certificate"
    )
    recomputed = audit_terminal_recovery(
        **inputs,
        observation_ns=certificate["terminal_state_reconciliation"]["observation_timestamp_ns"],
    )
    state = certificate["derived_state_events"]
    state_path = Path(state["path"])
    _require(sha256_file(state_path) == state["sha256"], "derived state trace digest differs")
    expected_rows = [
        *_read_checkpoint(Path(inputs["run_root"]) / "request-state-events.jsonl"),
        *recomputed["terminal_state_reconciliation"]["events"],
    ]
    _require(_read_checkpoint(state_path) == expected_rows, "derived state trace is not the proof")
    _require(
        certificate == {**recomputed, "derived_state_events": state},
        "recovery certificate disagrees with recomputed source evidence",
    )
    return certificate
