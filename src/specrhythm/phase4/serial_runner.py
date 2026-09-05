"""GPU-only Phase-4A.1 Target regression and Serial Disaggregated runner."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Optional

from specrhythm.phase4.batch_invariant import (
    configure_before_worker_creation,
    requested_for_mode,
    validate_batch_invariant_ranks,
)
from specrhythm.phase4.config import Phase4Config
from specrhythm.phase4.manifest import (
    atomic_write_json,
    build_runtime_manifest,
    sha256_file,
    validate_environment,
    validate_topology,
)
from specrhythm.phase4.reference import (
    build_target_regression,
    compare_outputs_to_reference,
    load_reference,
    reference_file_evidence,
    require_exact_resident_reference_reuse,
    require_reference_for_mode,
)
from specrhythm.phase4.resident_initial_proposal import (
    validate_initial_proposal_lifecycle_events,
)
from specrhythm.phase4.resident_setup import (
    build_setup_control,
    load_setup_ready,
    validate_resident_admission_events,
)
from specrhythm.phase4.stock_vllm import (
    _serialize_outputs,
    _update_combined_manifest,
    _visible_physical_ids,
    _worker_performance_finalize,
    _worker_runtime_snapshot,
    load_smoke_requests,
    run_stock_smoke,
    validate_worker_ranks,
)
from specrhythm.phase4.transport import CheckpointJsonl, UnixDraftClient, payload_sha256
from specrhythm.phase4.vllm_diagnostics import validate_kv_monotonicity
from specrhythm.phase4.vllm_installation import locate_installed_vllm_file

PATCHED_VLLM_RUNNER_SHA256 = (
    "2905189397b1517659e6606f5bc36c7ca226330f42255c579207fe38f61f9e19"
)
PATCHED_VLLM_SCHEDULER_SHA256 = (
    "ffaefd61869589f086e6acdf9a0c4f55f80d5dad145ca3f6fff2379f7a4e2455"
)


def _phase4b2_serial_execution_evidence(
    *,
    phase4b2_performance: bool,
    final_sync_rows: list[dict[str, Any]],
    stock_comparison_exact: bool,
) -> dict[str, Any]:
    """Build the one canonical Phase-4B.2 Serial execution-evidence block."""

    return {
        "phase4b2_performance_candidate": phase4b2_performance,
        "phase4b2_final_sync": final_sync_rows,
        "historical_gate3_qualification": {
            "gate3_exact_stock_equivalence": False,
            "exact_stock_trajectory": "96/100",
            "current_run_stock_comparison_exact": stock_comparison_exact,
            "logical_correctness_qualification": True,
            "numerical_qualification": "complete",
            "phase4b2_progression_permitted": True,
            "stock_comparison_excluded_from_phase4b2_validity": (
                phase4b2_performance
            ),
        },
    }


def load_patch_manifest(path: Path, config: Phase4Config) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("vLLM patch manifest root must be an object")
    errors = []
    if value.get("schema_version") != "specrhythm.vllm-base-and-patch-manifest.v1":
        errors.append("unsupported vLLM patch manifest")
    if value.get("vllm_base_commit") != config.expected_vllm_commit:
        errors.append("vLLM patch manifest has the wrong base commit")
    if value.get("verified_source_commit") != config.expected_vllm_commit:
        errors.append("vLLM patch manifest did not verify the pinned source checkout")
    if value.get("operation") != "apply":
        errors.append("vLLM patch manifest is not from the apply operation")
    if value.get("patch_applied") is not True:
        errors.append("vLLM integration patch is not applied")
    if value.get("python_only") is not True or value.get("cpp_cuda_modified") is not False:
        errors.append("Phase-4A.1 permits only the documented Python hook patch")
    if value.get("target_only_behavior_change_when_speculation_disabled") is not False:
        errors.append("patch does not preserve stock target-only semantics")
    if not value.get("patch_sha256"):
        errors.append("vLLM patch checksum is missing")
    if value.get("target_file_sha256_after") != PATCHED_VLLM_RUNNER_SHA256:
        errors.append("vLLM patch manifest has the wrong patched runner checksum")
    if errors:
        raise ValueError("invalid vLLM patch manifest: " + "; ".join(errors))
    return value


def validate_installed_patched_runner(patch_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the installed patch without importing vLLM before mode setup."""

    relative = Path("vllm/v1/worker/gpu_model_runner.py")
    path = locate_installed_vllm_file(relative)
    actual = sha256_file(path)
    expected = str(patch_manifest.get("target_file_sha256_after", ""))
    if actual != expected or actual != PATCHED_VLLM_RUNNER_SHA256:
        raise RuntimeError(
            "installed vLLM runner does not match the recorded Phase-4A.1 patch; "
            f"found SHA256 {actual}"
        )
    return {"file": str(relative), "sha256": actual, "matches_manifest": True}


def validate_installed_patch_stack(patch_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Require both the worker hooks and Phase-4B scheduler hook."""

    runner = validate_installed_patched_runner(patch_manifest)
    if patch_manifest.get("patch_stack_applied") is not True:
        raise RuntimeError("Phase-4B requires the complete ordered vLLM patch stack")
    relative = Path("vllm/v1/core/sched/scheduler.py")
    path = locate_installed_vllm_file(relative)
    actual = sha256_file(path)
    expected = str(patch_manifest.get("scheduler_file_sha256_after", ""))
    if actual != expected or actual != PATCHED_VLLM_SCHEDULER_SHA256:
        raise RuntimeError(
            "installed vLLM scheduler does not match the recorded Phase-4B patch; "
            f"found SHA256 {actual}"
        )
    return {
        "ordered_patch_stack_valid": True,
        "runner": runner,
        "scheduler": {
            "file": str(relative),
            "sha256": actual,
            "matches_manifest": True,
        },
    }


def validate_engine_residency(
    config: Phase4Config,
    *,
    draft_ready: Mapping[str, Any],
    target_worker_ranks: list[Mapping[str, Any]],
    proposer_report: Mapping[str, Any],
) -> list[str]:
    errors = []
    provenance = draft_ready.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    model = provenance.get("model")
    model = model if isinstance(model, Mapping) else {}
    if provenance.get("physical_gpu_id") != config.draft.physical_gpu_ids[0]:
        errors.append("Draft service is not resident on configured physical GPU 0")
    if Path(str(model.get("path", ""))).resolve() != config.draft.resolved_model_path:
        errors.append("Draft service did not load the configured Draft model")
    if Path(str(model.get("path", ""))).resolve() == config.target.resolved_model_path:
        errors.append("Draft GPU loaded the Target model")
    if provenance.get("parameter_count", 0) <= 0:
        errors.append("Draft service has no resident model parameters")
    if provenance.get("full_context_replay_per_round") is not False:
        errors.append("Draft service does not prove cross-round KV reuse")
    errors.extend(validate_worker_ranks(target_worker_ranks, config.target))
    if proposer_report.get("proposer_model_parameter_count") != 0:
        errors.append("Target GPU custom proposer loaded Draft model parameters")
    return errors


def run_patched_target_regression(
    config: Phase4Config,
    *,
    workload_path: Path,
    environment_path: Path,
    topology_path: Path,
    runtime_manifest_path: Path,
    reference_path: Path,
    patch_manifest_path: Path,
    git_commit: str,
    legacy_hf_target_dir: Optional[Path] = None,
    correctness_mode: str = "default",
    diagnostics_path: Optional[Path] = None,
) -> dict[str, Any]:
    _require_v1_runner()
    reference = load_reference(reference_path)
    require_reference_for_mode(reference, correctness_mode)
    patch_manifest = load_patch_manifest(patch_manifest_path, config)
    smoke = run_stock_smoke(
        config,
        role="target",
        workload_path=workload_path,
        environment_path=environment_path,
        topology_path=topology_path,
        runtime_manifest_path=runtime_manifest_path,
        git_commit=git_commit,
        frozen_target_dir=legacy_hf_target_dir,
        correctness_mode=correctness_mode,
        diagnostics_path=diagnostics_path,
    )
    installed_runner = validate_installed_patched_runner(patch_manifest)
    result = build_target_regression(smoke, reference, patch_manifest=patch_manifest)
    result["stock_reference"] = reference_file_evidence(reference_path)
    result["patch_manifest_file_sha256"] = sha256_file(patch_manifest_path)
    result["installed_vllm_runner"] = installed_runner
    return result


def run_serial_disaggregated(
    config: Phase4Config,
    *,
    workload_path: Path,
    environment_path: Path,
    topology_path: Path,
    runtime_manifest_path: Path,
    reference_path: Path,
    patch_manifest_path: Path,
    draft_socket_path: Path,
    draft_ready_path: Path,
    round_events_path: Path,
    transport_events_path: Path,
    plugin_report_path: Path,
    git_commit: str,
    correctness_mode: str = "default",
    diagnostics_path: Optional[Path] = None,
    request_count: Optional[int] = None,
    decode_ready_context_path: Optional[Path] = None,
    decode_ready_manifest_path: Optional[Path] = None,
    decode_ready_timing_path: Optional[Path] = None,
    first_forward_path: Optional[Path] = None,
    resident_setup_control_path: Optional[Path] = None,
    resident_setup_ready_path: Optional[Path] = None,
    resident_admission_events_path: Optional[Path] = None,
    resident_initial_proposal_events_path: Optional[Path] = None,
    phase4b2_performance: bool = False,
) -> dict[str, Any]:
    """Run one real GPU correctness pass; no performance metrics are derived."""

    _require_v1_runner()
    mode_evidence = configure_before_worker_creation(correctness_mode)
    reference = load_reference(reference_path)
    require_reference_for_mode(reference, correctness_mode)
    patch_manifest = load_patch_manifest(patch_manifest_path, config)
    installed_runner = validate_installed_patched_runner(patch_manifest)
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    topology = json.loads(topology_path.read_text(encoding="utf-8"))
    environment_validation = validate_environment(environment, config)
    topology_validation = validate_topology(topology, config)
    if not environment_validation["valid"]:
        raise RuntimeError(
            "invalid Phase-4 environment: " + "; ".join(environment_validation["errors"])
        )
    if not topology_validation["valid"]:
        raise RuntimeError("invalid Phase-4 topology: " + "; ".join(topology_validation["errors"]))
    if _visible_physical_ids() != config.target.physical_gpu_ids:
        raise RuntimeError("Serial Target must see only configured physical GPUs 1,2")
    if not draft_socket_path.is_socket():
        raise RuntimeError("persistent Draft service Unix socket is not ready")
    draft_ready = json.loads(draft_ready_path.read_text(encoding="utf-8"))
    draft_provenance = draft_ready.get("provenance")
    if (
        draft_ready.get("schema_version")
        != "specrhythm.phase4-draft-service-ready.v1"
        or not isinstance(draft_provenance, Mapping)
    ):
        raise RuntimeError("Draft service residency/KV evidence is invalid")
    resident_paths = (
        decode_ready_context_path,
        decode_ready_manifest_path,
        decode_ready_timing_path,
        first_forward_path,
        resident_setup_control_path,
        resident_setup_ready_path,
        resident_admission_events_path,
        resident_initial_proposal_events_path,
    )
    resident_mode = any(path is not None for path in resident_paths)
    if resident_mode and any(path is None for path in resident_paths):
        raise ValueError(
            "resident Serial requires context, manifest, timing and first-forward paths"
        )
    if resident_mode:
        require_exact_resident_reference_reuse(reference, config, workload_path)
        installed_runner = validate_installed_patch_stack(patch_manifest)
    artifact_paths = [round_events_path, transport_events_path, plugin_report_path]
    artifact_paths.extend(path for path in resident_paths[1:] if path is not None)
    if diagnostics_path is not None:
        artifact_paths.append(diagnostics_path)
    for path in artifact_paths:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite Serial checkpoint artifact {path}")
    effective_count = request_count or config.smoke_request_count
    requests = load_smoke_requests(
        workload_path,
        effective_count,
        require_task_mixture=effective_count in {5, 100},
    )
    try:
        import torch
        from vllm import LLM, SamplingParams
    except ImportError as error:
        raise RuntimeError("Phase-4 vLLM GPU environment is not installed") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to generate a fake Serial result")
    os.environ.update(
        {
            "SR_PHASE4_DRAFT_SOCKET": str(draft_socket_path),
            "SR_PHASE4_WORKLOAD": str(workload_path),
            "SR_PHASE4_ROUND_EVENTS": str(round_events_path),
            "SR_PHASE4_TRANSPORT_EVENTS": str(transport_events_path),
            "SR_PHASE4_PLUGIN_REPORT": str(plugin_report_path),
            "SR_PHASE4_REQUEST_COUNT": str(effective_count),
            "SR_PHASE4B2_PERFORMANCE": "1" if phase4b2_performance else "0",
        }
    )
    if resident_mode:
        assert decode_ready_context_path is not None
        assert decode_ready_manifest_path is not None
        assert decode_ready_timing_path is not None
        assert resident_setup_control_path is not None
        assert resident_setup_ready_path is not None
        assert resident_admission_events_path is not None
        assert resident_initial_proposal_events_path is not None
        deferred_initial_proposals_path = resident_setup_ready_path.with_name(
            "initial-proposals-ready.json"
        )
        if deferred_initial_proposals_path.exists():
            raise FileExistsError(
                "refusing to overwrite deferred initial-proposal artifact"
            )
        os.environ.update(
            {
                "SR_PHASE4_DECODE_READY_MODE": "1",
                "SR_PHASE4_DECODE_READY_CONTEXT": str(decode_ready_context_path),
                "SR_PHASE4_DECODE_READY_MANIFEST": str(decode_ready_manifest_path),
                "SR_PHASE4_DECODE_READY_TIMING_EVENTS": str(
                    decode_ready_timing_path
                ),
                "SR_PHASE4_RESIDENT_SETUP": "1",
                "SR_PHASE4_RESIDENT_CONSUMER": "serial",
                "SR_PHASE4_RESIDENT_SETUP_CONTROL": str(
                    resident_setup_control_path
                ),
                "SR_PHASE4_RESIDENT_SETUP_READY": str(resident_setup_ready_path),
                "SR_PHASE4_RESIDENT_ADMISSION_EVENTS": str(
                    resident_admission_events_path
                ),
                "SR_PHASE4_RESIDENT_INITIAL_PROPOSAL_EVENTS": str(
                    resident_initial_proposal_events_path
                ),
                "SR_PHASE4B2_INITIAL_PROPOSALS_READY": str(
                    deferred_initial_proposals_path
                ),
            }
        )
    else:
        for name in (
            "SR_PHASE4_DECODE_READY_MODE",
            "SR_PHASE4_DECODE_READY_CONTEXT",
            "SR_PHASE4_DECODE_READY_MANIFEST",
            "SR_PHASE4_DECODE_READY_TIMING_EVENTS",
            "SR_PHASE4_RESIDENT_SETUP",
            "SR_PHASE4_RESIDENT_CONSUMER",
            "SR_PHASE4_RESIDENT_SETUP_CONTROL",
            "SR_PHASE4_RESIDENT_SETUP_READY",
            "SR_PHASE4_RESIDENT_ADMISSION_EVENTS",
            "SR_PHASE4_RESIDENT_INITIAL_PROPOSAL_EVENTS",
            "SR_PHASE4B2_INITIAL_PROPOSALS_READY",
        ):
            os.environ.pop(name, None)
    if diagnostics_path is not None:
        os.environ["SR_PHASE4_TARGET_DIAGNOSTICS"] = str(diagnostics_path)
    else:
        os.environ.pop("SR_PHASE4_TARGET_DIAGNOSTICS", None)
    started = time.monotonic_ns()
    llm = LLM(
        model=str(config.target.resolved_model_path),
        tokenizer=str(config.target.resolved_tokenizer_path),
        tensor_parallel_size=config.target.tensor_parallel_size,
        dtype=config.target.dtype,
        revision=config.target.revision,
        tokenizer_revision=config.target.tokenizer_revision,
        trust_remote_code=config.target.trust_remote_code,
        seed=config.sampling.seed,
        gpu_memory_utilization=config.target.gpu_memory_utilization,
        max_model_len=config.max_model_len,
        enforce_eager=config.enforce_eager,
        enable_prefix_caching=config.enable_prefix_caching,
        enable_dbo=False,
        scheduler_cls=(
            "specrhythm.phase4.resident_scheduler.ResidentSetupScheduler"
            if resident_mode
            else None
        ),
        speculative_config={
            "model": "specrhythm.phase4.vllm_remote.RemoteDraftProposer",
            "method": "custom_class",
            "num_speculative_tokens": config.proposal_budget,
        },
        disable_log_stats=False,
    )
    startup_end = time.monotonic_ns()
    worker_ranks = llm.collective_rpc(_worker_runtime_snapshot)
    rank_errors = validate_worker_ranks(worker_ranks, config.target)
    batch_validation = validate_batch_invariant_ranks(
        worker_ranks, requested=requested_for_mode(correctness_mode)
    )
    rank_errors.extend(batch_validation["batch_invariant_validation"]["errors"])
    if rank_errors:
        raise RuntimeError("invalid Target worker evidence: " + "; ".join(rank_errors))
    tokenizer = llm.get_tokenizer()
    for request in requests:
        if list(tokenizer.encode(request.prompt_text, add_special_tokens=True)) != list(
            request.prompt_token_ids
        ):
            raise RuntimeError(f"Target tokenizer disagrees for {request.request_id}")
    prompts = [{"prompt_token_ids": list(request.prompt_token_ids)} for request in requests]
    parameters = [
        SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=request.maximum_new_tokens,
            seed=request.sampling_seed,
            n=1,
            logprobs=config.logprobs,
        )
        for request in requests
    ]
    generation_start = time.monotonic_ns()
    if resident_mode:
        assert resident_setup_control_path is not None
        atomic_write_json(
            resident_setup_control_path,
            build_setup_control(
                consumer="serial",
                expected_request_ids=[row.request_id for row in requests],
                setup_start_ns=generation_start,
            ),
        )
    outputs = llm.generate(prompts, parameters, use_tqdm=False)
    final_sync_rows = (
        llm.collective_rpc(_worker_performance_finalize)
        if phase4b2_performance
        else []
    )
    if not phase4b2_performance:
        torch.cuda.synchronize()
    generation_end = time.monotonic_ns()
    serialized = _serialize_outputs(outputs, requests)
    comparison = compare_outputs_to_reference(serialized, reference)
    if not plugin_report_path.is_file():
        raise RuntimeError("Target custom proposer did not emit its lifecycle report")
    plugin_report = json.loads(plugin_report_path.read_text(encoding="utf-8"))
    residency_errors = validate_engine_residency(
        config,
        draft_ready=draft_ready,
        target_worker_ranks=worker_ranks,
        proposer_report=plugin_report,
    )
    if residency_errors:
        raise RuntimeError("invalid engine residency: " + "; ".join(residency_errors))
    round_rows = CheckpointJsonl(round_events_path).read()
    transport_rows = CheckpointJsonl(transport_events_path).read()
    _enrich_divergence(comparison, round_rows, plugin_report)
    draft_shutdown = UnixDraftClient(draft_socket_path).shutdown()
    attention_backends = sorted(
        {str(backend) for row in worker_ranks for backend in row.get("attention_backends", ())}
    )
    manifest = build_runtime_manifest(
        config,
        role="target",
        git_commit=git_commit,
        workload_path=workload_path,
        environment_path=environment_path,
        topology_path=topology_path,
        worker_ranks=worker_ranks,
        attention_backend=",".join(attention_backends) or None,
        correctness_mode=correctness_mode,
        mode_setup=mode_evidence,
        batch_invariant_validation=batch_validation,
    )
    manifest["stage"] = "phase4a1-serial-disaggregated-correctness"
    manifest["result_kind"] = "gpu-correctness-only"
    manifest["framework"]["custom_remote_proposer"] = True
    manifest["framework"]["vllm_patch_sha256"] = patch_manifest["patch_sha256"]
    _update_combined_manifest(runtime_manifest_path, manifest)
    runtime_bundle = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    runtime_bundle["stage"] = "phase4a1-serial-disaggregated-correctness"
    phase4b2_evidence = _phase4b2_serial_execution_evidence(
        phase4b2_performance=phase4b2_performance,
        final_sync_rows=final_sync_rows,
        stock_comparison_exact=comparison["all_sequences_equal"],
    )
    runtime_bundle["phase4a1"] = {
        "mode": "serial-disaggregated",
        "correctness_mode": correctness_mode,
        **batch_validation,
        "gpu_correctness_result": True,
        "gpu_performance_result": False,
        **phase4b2_evidence,
        "draft_service_ready_file": draft_ready_path.name,
        "draft_service_ready_sha256": sha256_file(draft_ready_path),
        "draft_service": draft_ready,
        "stock_reference_file": reference_path.name,
        "stock_reference_sha256": sha256_file(reference_path),
        "patch_manifest_file": patch_manifest_path.name,
        "patch_manifest_sha256": sha256_file(patch_manifest_path),
        "transport": "unix-domain-socket",
        "strict_serial": True,
        "dual_batch_overlap": False,
        "packed_tree_verification": False,
        "reports_goodput": False,
        "reports_slo_attainment": False,
        "reports_speedup": False,
    }
    atomic_write_json(runtime_manifest_path, runtime_bundle)
    accounting = _aggregate_accounting(round_rows, plugin_report)
    kv_errors = validate_kv_monotonicity(round_rows)
    result = {
        "schema_version": "specrhythm.phase4-serial-disaggregated-run.v1",
        "mode": "serial-disaggregated",
        "correctness_mode": correctness_mode,
        **batch_validation,
        "execution_semantics": (
            "Draft completes, local IPC completes, Target batched verification completes, "
            "state synchronization completes, then next-round Draft starts"
        ),
        "gpu_correctness_result": True,
        "gpu_performance_result": False,
        **phase4b2_evidence,
        "reports_goodput": False,
        "reports_slo_attainment": False,
        "reports_speedup": False,
        "serving_engine": True,
        "target_backend": "vllm-v0.25.1-v1-runner",
        "target_verification": "vllm-linear-speculative-batched-verification",
        "packed_tree_verification": False,
        "dual_batch_overlap": False,
        "eager": False,
        "candidate_budget": config.proposal_budget,
        "sampling_configuration": config.sampling.to_dict(),
        "target_runtime_configuration": {
            "physical_gpu_ids": list(config.target.physical_gpu_ids),
            "tensor_parallel_size": config.target.tensor_parallel_size,
            "dtype": config.target.dtype,
            "max_model_len": config.max_model_len,
            "attention_backends": attention_backends,
            "all_reduce_backends": sorted(
                {
                    str(backend)
                    for row in worker_ranks
                    for backend in row.get("all_reduce_backends", ())
                }
            ),
            "VLLM_BATCH_INVARIANT": (
                "1" if requested_for_mode(correctness_mode) else "0"
            ),
        },
        "request_count": len(requests),
        "outputs": serialized,
        "comparison": comparison,
        "exact_sequence_match": comparison["all_sequences_equal"],
        "worker_ranks": worker_ranks,
        "engine_residency": {
            "draft": {
                "physical_gpu_ids": [0],
                "target_model_loaded": False,
                "service_provenance": draft_provenance,
            },
            "target": {
                "physical_gpu_ids": list(config.target.physical_gpu_ids),
                "draft_model_loaded": False,
                "remote_proposer_parameter_count": plugin_report.get(
                    "proposer_model_parameter_count"
                ),
            },
        },
        "accounting": accounting,
        "kv_monotonicity": {
            "valid": not kv_errors,
            "errors": kv_errors,
            "rejected_or_future_tokens_retained": False if not kv_errors else None,
        },
        "strict_serial_timeline": {
            "round_events": len(round_rows),
            "validated_in_runner": _timeline_rows_valid(round_rows),
        },
        "target_microbatch_metadata": {
            "vllm_scheduler_may_microbatch": True,
            "cross_phase_overlap_allowed": False,
        },
        "transport": {
            "type": "unix-domain-socket",
            "serialization": "length-prefixed-canonical-json",
            "event_count": len(transport_rows),
            "host_staging": True,
            "complete_draft_to_verify_transport_benchmark": False,
        },
        "plugin_report": plugin_report,
        "target_diagnostics": (
            {
                "enabled": True,
                "file": diagnostics_path.name,
                "target_only": True,
                "visible_to_draft": False,
            }
            if diagnostics_path is not None
            else {"enabled": False}
        ),
        "draft_shutdown": draft_shutdown,
        "stock_reference": reference_file_evidence(reference_path),
        "patch_manifest": {
            "file": patch_manifest_path.name,
            "file_sha256": sha256_file(patch_manifest_path),
            "payload_sha256": payload_sha256(patch_manifest),
            "patch_sha256": patch_manifest["patch_sha256"],
            "installed_vllm_runner": installed_runner,
        },
        "provenance": {
            "git_commit": git_commit,
            "config_sha256": sha256_file(config.path),
            "workload_sha256": sha256_file(workload_path),
            "vllm_source_commit": config.expected_vllm_commit,
        },
        "observability_wall_time_only": {
            "startup_ms": (startup_end - started) / 1_000_000,
            "generation_ms": (generation_end - generation_start) / 1_000_000,
            "performance_claim_allowed": False,
        },
    }
    resident_errors = []
    if resident_mode:
        assert decode_ready_manifest_path is not None
        assert first_forward_path is not None
        assert resident_setup_ready_path is not None
        assert resident_admission_events_path is not None
        assert resident_initial_proposal_events_path is not None
        from specrhythm.phase4.decode_ready import (
            build_first_target_forward_contract,
            compare_raw_and_decode_outputs,
            load_decode_ready_manifest,
            validate_measurement_boundary,
        )
        from specrhythm.phase4.resident_runner import _decode_rows, _reference_rows

        manifest = load_decode_ready_manifest(
            json.loads(decode_ready_manifest_path.read_text(encoding="utf-8"))
        )
        setup_ready = load_setup_ready(
            resident_setup_ready_path,
            manifest_path=decode_ready_manifest_path,
            consumer="serial",
            expected_request_ids=[row.request_id for row in requests],
        )
        admission_rows = CheckpointJsonl(resident_admission_events_path).read()
        admission_errors = validate_resident_admission_events(
            admission_rows, consumer="serial"
        )
        resident_errors.extend(admission_errors)
        initial_proposal_rows = CheckpointJsonl(
            resident_initial_proposal_events_path
        ).read()
        initial_proposal_errors = validate_initial_proposal_lifecycle_events(
            initial_proposal_rows,
            expected_request_ids=[row.request_id for row in requests],
        )
        resident_errors.extend(initial_proposal_errors)
        diagnostics = CheckpointJsonl(diagnostics_path).read() if diagnostics_path else []
        contracts = []
        first_round_by_request = {
            str(row.get("request_id", "")): row
            for row in round_rows
            if row.get("round_id") == 0
        }
        for ready in manifest.requests:
            matches = [
                row
                for row in diagnostics
                if row.get("request_id") == ready.request_id
                and row.get("committed_prefix_sha256")
                == ready.logical_committed_prefix_sha256
                and row.get("proposal_token_ids")
            ]
            if not matches:
                resident_errors.append(
                    f"{ready.request_id}: first resident Serial verify is missing"
                )
                continue
            observed = matches[0]
            round_row = first_round_by_request.get(ready.request_id)
            if round_row is None:
                resident_errors.append(
                    f"{ready.request_id}: first resident Serial round is missing"
                )
                continue
            try:
                contract = build_first_target_forward_contract(
                    ready,
                    consumer="serial",
                    proposal_token_ids=observed["proposal_token_ids"],
                    target_forward_start_ns=int(observed["target_forward_start_ns"]),
                    target_forward_end_ns=int(observed["target_forward_end_ns"]),
                    output_logits_positions=observed.get("position_ids", ()),
                    accepted_draft_tokens=int(
                        round_row.get("accepted_draft_tokens", 0)
                    ),
                    post_forward_committed_token_ids=round_row.get(
                        "committed_token_ids", ()
                    ),
                    post_forward_target_kv_token_count=(
                        int(round_row.get("logical_target_kv_length", 0)) - 1
                    ),
                    post_forward_prefix_version=ready.prefix_version + 1,
                )
            except (KeyError, TypeError, ValueError) as error:
                resident_errors.append(f"{ready.request_id}: {error}")
                continue
            if observed.get("target_input_token_ids") != contract[
                "verification_input_token_ids"
            ]:
                resident_errors.append(
                    f"{ready.request_id}: Serial verification input differs"
                )
            if observed.get("position_ids") != contract["input_positions"]:
                resident_errors.append(
                    f"{ready.request_id}: Serial verification positions differ"
                )
            if observed.get("physical_kv_num_computed_tokens") != (
                ready.target_materialized_kv_token_count
            ):
                resident_errors.append(
                    f"{ready.request_id}: Serial computed-token/KV count differs"
                )
            contracts.append(contract)
        atomic_write_json(
            first_forward_path,
            {
                "schema_version": "specrhythm.phase4b-first-target-forwards.v1",
                "consumer": "serial",
                "valid": not resident_errors,
                "errors": resident_errors,
                "requests": contracts,
            },
        )
        first_draft_starts = [
            int(row.get("timeline", {}).get("draft_start_ns", 0))
            for row in round_rows
            if row.get("round_id") == 0
            and isinstance(row.get("timeline"), Mapping)
        ]
        boundary_errors = validate_measurement_boundary(
            manifest,
            first_draft_start_ns=min(first_draft_starts) if first_draft_starts else None,
            first_draft_end_ns=min(
                (
                    int(row.get("timeline", {}).get("draft_end_ns", 0))
                    for row in round_rows
                    if row.get("round_id") == 0
                    and isinstance(row.get("timeline"), Mapping)
                ),
                default=None,
            ),
            first_target_decode_start_ns=(
                min(int(row["target_forward_start_ns"]) for row in contracts)
                if contracts
                else None
            ),
            proposal_created_timestamps_ns=first_draft_starts,
        )
        resident_errors.extend(boundary_errors)
        decode_rows = _decode_rows(serialized, manifest)
        raw_decode = compare_raw_and_decode_outputs(
            _reference_rows(reference, requests), decode_rows, manifest
        )
        if not raw_decode["valid"] and not phase4b2_performance:
            resident_errors.extend(raw_decode["errors"])
        result.update(
            {
                "provider_kind": "resident-warm-start",
                "decode_only_outputs": decode_rows,
                "raw_vs_decode": raw_decode,
                "decode_ready_manifest_sha256": manifest.manifest_sha256,
                "first_target_forward_valid": not resident_errors,
                "measurement_boundary_valid": not boundary_errors,
                "end_to_end_pd_deployment": False,
                "kv_connector_handoff": False,
                "resident_errors": resident_errors,
                "global_setup_ready": setup_ready,
                "resident_admission": {
                    "valid": not admission_errors,
                    "errors": admission_errors,
                    "event_count": len(admission_rows),
                },
                "resident_initial_proposal_lifecycle": {
                    "valid": not initial_proposal_errors,
                    "errors": initial_proposal_errors,
                    "event_count": len(initial_proposal_rows),
                    "artifact_file": resident_initial_proposal_events_path.name,
                },
                "phase4b2_initial_proposals_ready": (
                    {
                        "file": deferred_initial_proposals_path.name,
                        "sha256": sha256_file(deferred_initial_proposals_path),
                    }
                    if phase4b2_performance
                    else None
                ),
            }
        )
    result["valid"] = bool(
        (comparison["all_sequences_equal"] or phase4b2_performance)
        and result["strict_serial_timeline"]["validated_in_runner"]
        and accounting["valid"]
        and not kv_errors
        and batch_validation["batch_invariant_validation"]["valid"]
    ) and not resident_errors
    return result


def run_fixed_proposal_control(
    config: Phase4Config,
    *,
    workload_path: Path,
    environment_path: Path,
    topology_path: Path,
    patch_manifest_path: Path,
    diagnostics_path: Path,
    git_commit: str,
    proposer: str,
    proposal_budget: int,
    correctness_mode: str = "batch-invariant",
    remote_socket_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Run one K=1/2/4 single-request diagnostic control, never a benchmark."""

    _require_v1_runner()
    mode_setup = configure_before_worker_creation(correctness_mode)
    if proposer not in {"local-static", "remote-fixed"}:
        raise ValueError("control proposer must be local-static or remote-fixed")
    if proposal_budget not in (1, 2, 4):
        raise ValueError("control proposal budget must be 1, 2, or 4")
    if diagnostics_path.exists():
        raise FileExistsError(f"refusing to overwrite diagnostics {diagnostics_path}")
    if proposer == "remote-fixed" and (
        remote_socket_path is None or not remote_socket_path.is_socket()
    ):
        raise RuntimeError("remote fixed-proposal socket is not ready")
    patch_manifest = load_patch_manifest(patch_manifest_path, config)
    installed_runner = validate_installed_patched_runner(patch_manifest)
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    topology = json.loads(topology_path.read_text(encoding="utf-8"))
    environment_validation = validate_environment(environment, config)
    topology_validation = validate_topology(topology, config)
    if not environment_validation["valid"]:
        raise RuntimeError(
            "invalid Phase-4 environment: "
            + "; ".join(environment_validation["errors"])
        )
    if not topology_validation["valid"]:
        raise RuntimeError(
            "invalid Phase-4 topology: " + "; ".join(topology_validation["errors"])
        )
    if _visible_physical_ids() != config.target.physical_gpu_ids:
        raise RuntimeError("fixed control Target must see configured physical GPUs 1,2")
    requests = load_smoke_requests(
        workload_path, expected_count=1, require_task_mixture=False
    )
    os.environ.update(
        {
            "SR_PHASE4_WORKLOAD": str(workload_path),
            "SR_PHASE4_TARGET_DIAGNOSTICS": str(diagnostics_path),
            "SR_PHASE4_FIXED_K": str(proposal_budget),
        }
    )
    if remote_socket_path is not None:
        os.environ["SR_PHASE4_FIXED_SOCKET"] = str(remote_socket_path)
    try:
        import torch
        from vllm import LLM, SamplingParams
    except ImportError as error:
        raise RuntimeError("Phase-4 vLLM GPU environment is not installed") from error
    proposer_class = (
        "specrhythm.phase4.fixed_control.LocalStaticProposer"
        if proposer == "local-static"
        else "specrhythm.phase4.fixed_control.RemoteFixedProposer"
    )
    llm = LLM(
        model=str(config.target.resolved_model_path),
        tokenizer=str(config.target.resolved_tokenizer_path),
        tensor_parallel_size=config.target.tensor_parallel_size,
        dtype=config.target.dtype,
        revision=config.target.revision,
        tokenizer_revision=config.target.tokenizer_revision,
        trust_remote_code=config.target.trust_remote_code,
        seed=config.sampling.seed,
        gpu_memory_utilization=config.target.gpu_memory_utilization,
        max_model_len=config.max_model_len,
        enforce_eager=config.enforce_eager,
        enable_prefix_caching=config.enable_prefix_caching,
        enable_dbo=False,
        speculative_config={
            "model": proposer_class,
            "method": "custom_class",
            "num_speculative_tokens": proposal_budget,
        },
        disable_log_stats=False,
    )
    worker_ranks = llm.collective_rpc(_worker_runtime_snapshot)
    errors = validate_worker_ranks(worker_ranks, config.target)
    batch_validation = validate_batch_invariant_ranks(
        worker_ranks, requested=requested_for_mode(correctness_mode)
    )
    errors.extend(batch_validation["batch_invariant_validation"]["errors"])
    if errors:
        raise RuntimeError("invalid fixed-control worker evidence: " + "; ".join(errors))
    request = requests[0]
    outputs = llm.generate(
        [{"prompt_token_ids": list(request.prompt_token_ids)}],
        SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=request.maximum_new_tokens,
            seed=request.sampling_seed,
            n=1,
            logprobs=config.logprobs,
        ),
        use_tqdm=False,
    )
    torch.cuda.synchronize()
    serialized = _serialize_outputs(outputs, requests)
    remote_shutdown = None
    if proposer == "remote-fixed" and remote_socket_path is not None:
        remote_shutdown = UnixDraftClient(remote_socket_path).shutdown()
    return {
        "schema_version": "specrhythm.phase4-fixed-proposal-control.v1",
        "control_only": True,
        "gpu_performance_result": False,
        "proposer": proposer,
        "proposal_token_ids": [53143, 2213, 369, 264][:proposal_budget],
        "proposal_budget": proposal_budget,
        "correctness_mode": correctness_mode,
        **batch_validation,
        "mode_setup": mode_setup,
        "request_id": request.request_id,
        "outputs": serialized,
        "worker_ranks": worker_ranks,
        "diagnostics_file": diagnostics_path.name,
        "target_diagnostics_visible_to_draft": False,
        "remote_shutdown": remote_shutdown,
        "patch_provenance": {
            "vllm_source_commit": config.expected_vllm_commit,
            "patch_sha256": patch_manifest["patch_sha256"],
            "installed_vllm_runner": installed_runner,
        },
        "git_commit": git_commit,
    }


def _aggregate_accounting(
    round_rows: list[Mapping[str, Any]], plugin_report: Mapping[str, Any]
) -> dict[str, Any]:
    keys = (
        "proposed_tokens",
        "verified_candidate_tokens",
        "accepted_draft_tokens",
        "rejected_draft_tokens",
        "target_correction_tokens",
        "target_bonus_tokens",
        "committed_tokens",
    )
    totals = {key: sum(int(row.get(key, 0)) for row in round_rows) for key in keys}
    requests = plugin_report.get("requests")
    requests = requests if isinstance(requests, Mapping) else {}
    bootstrap = sum(int(row.get("bootstrap_target_tokens", 0)) for row in requests.values())
    tail = sum(int(row.get("tail_target_tokens", 0)) for row in requests.values())
    final_generated = sum(len(row.get("generated_token_ids", ())) for row in requests.values())
    valid = (
        totals["proposed_tokens"]
        == totals["accepted_draft_tokens"] + totals["rejected_draft_tokens"]
        and totals["committed_tokens"]
        == totals["accepted_draft_tokens"]
        + totals["target_correction_tokens"]
        + totals["target_bonus_tokens"]
        and final_generated == bootstrap + tail + totals["committed_tokens"]
    )
    return {
        **totals,
        "target_bootstrap_tokens": bootstrap,
        "target_tail_tokens": tail,
        "final_generated_tokens": final_generated,
        "valid": valid,
    }


def _enrich_divergence(
    comparison: Mapping[str, Any],
    round_rows: list[Mapping[str, Any]],
    plugin_report: Mapping[str, Any],
) -> None:
    request_meta = plugin_report.get("requests")
    request_meta = request_meta if isinstance(request_meta, Mapping) else {}
    comparisons = comparison.get("requests")
    if not isinstance(comparisons, list):
        return
    by_request: dict[str, list[Mapping[str, Any]]] = {}
    for row in round_rows:
        by_request.setdefault(str(row.get("request_id", "")), []).append(row)
    for item in comparisons:
        if not isinstance(item, dict) or item.get("equal") is True:
            continue
        request_id = str(item.get("request_id", ""))
        position = item.get("first_divergence_position")
        if not isinstance(position, int):
            continue
        meta = request_meta.get(request_id)
        offset = int(meta.get("bootstrap_target_tokens", 0)) if isinstance(meta, Mapping) else 0
        matched_round = None
        for row in by_request.get(request_id, ()):
            committed = int(row.get("committed_tokens", 0))
            if offset <= position < offset + committed:
                matched_round = row
                break
            offset += committed
        if matched_round is not None:
            item["round_id"] = matched_round.get("round_id")
            item["prefix_hash"] = matched_round.get("parent_prefix_hash")
            item["proposal_token_ids"] = matched_round.get("proposal_token_ids")
            item["accepted_prefix_length"] = matched_round.get(
                "accepted_draft_tokens"
            )
            item["logical_target_kv_length"] = matched_round.get(
                "logical_target_kv_length"
            )
            item["target_microbatch_id"] = matched_round.get("target_microbatch_id")


def _timeline_rows_valid(rows: list[Mapping[str, Any]]) -> bool:
    draft_intervals = []
    verify_intervals = []
    for row in rows:
        timeline = row.get("timeline")
        if not isinstance(timeline, Mapping):
            return False
        values = [
            timeline.get(name)
            for name in (
                "draft_start_ns",
                "draft_end_ns",
                "transfer_start_ns",
                "transfer_end_ns",
                "verify_start_ns",
                "verify_end_ns",
                "state_sync_start_ns",
                "state_sync_end_ns",
                "next_round_draft_start_ns",
            )
        ]
        if any(not isinstance(value, int) for value in values) or values != sorted(values):
            return False
        draft_intervals.append((values[0], values[1]))
        verify_intervals.append((values[4], values[5]))
    for draft_start, draft_end in draft_intervals:
        for verify_start, verify_end in verify_intervals:
            if max(draft_start, verify_start) < min(draft_end, verify_end):
                return False
    return True


def _require_v1_runner() -> None:
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "0":
        raise RuntimeError("Phase-4A.1 requires VLLM_USE_V2_MODEL_RUNNER=0")
