"""GPU-only Phase-4B.1 Dual-Batch correctness runner."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from specrhythm.phase4.batch_invariant import (
    configure_before_worker_creation,
    validate_batch_invariant_ranks,
)
from specrhythm.phase4.config import Phase4Config
from specrhythm.phase4.decode_ready import load_decode_ready_manifest
from specrhythm.phase4.dual import validate_cycle_rows
from specrhythm.phase4.dual_commit import phase4_dual_sampling_params
from specrhythm.phase4.dual_correctness import (
    validate_overlap_witness,
    validate_proposal_lifecycle_events,
    validate_request_state_events,
    validate_round_accounting,
    validate_scheduler_cycles,
)
from specrhythm.phase4.dual_service import DualDraftClient
from specrhythm.phase4.dual_terminal import build_terminal_reconciliation
from specrhythm.phase4.manifest import (
    atomic_write_json,
    build_runtime_manifest,
    sha256_file,
    validate_environment,
    validate_topology,
)
from specrhythm.phase4.reference import (
    compare_outputs_to_reference,
    load_reference,
    require_reference_for_mode,
)
from specrhythm.phase4.resident_runner import _decode_rows, build_decode_ready_context
from specrhythm.phase4.resident_setup import (
    build_setup_control,
    load_setup_ready,
)
from specrhythm.phase4.serial_runner import (
    load_patch_manifest,
    validate_installed_patch_stack,
)
from specrhythm.phase4.stock_vllm import (
    _serialize_outputs,
    _visible_physical_ids,
    _worker_performance_finalize,
    _worker_runtime_snapshot,
    load_smoke_requests,
    validate_worker_ranks,
)
from specrhythm.phase4.transport import CheckpointJsonl


def run_dual_batch(
    config: Phase4Config,
    *,
    workload_path: Path,
    request_count: int,
    environment_path: Path,
    topology_path: Path,
    runtime_manifest_path: Path,
    reference_path: Path,
    patch_manifest_path: Path,
    draft_socket_path: Path,
    draft_ready_path: Path,
    scheduler_events_path: Path,
    request_state_events_path: Path,
    proposal_events_path: Path,
    verification_events_path: Path,
    draft_work_events_path: Path,
    transport_events_path: Path,
    target_diagnostics_path: Path,
    plugin_report_path: Path,
    output_checkpoint_path: Path,
    cycle_events_path: Path,
    overlap_events_path: Path,
    output_path: Path,
    git_commit: str,
    microbatch_size: int = 1,
    cohort_size: Optional[int] = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run correctness cohorts; no latency/speedup metric is reported."""

    if request_count < 2 or microbatch_size < 1:
        raise ValueError("Dual-Batch requires at least two requests and positive microbatches")
    if output_path.exists():
        if resume:
            value = json.loads(output_path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or value.get("mode") != "dual-batch":
                raise ValueError("completed resume artifact is not a Dual-Batch run")
            if draft_socket_path.is_socket():
                DualDraftClient(draft_socket_path).call("shutdown", {})
            return value
        raise FileExistsError(f"refusing to overwrite Dual-Batch output {output_path}")
    mode_evidence = configure_before_worker_creation("batch-invariant")
    reference = load_reference(reference_path)
    require_reference_for_mode(reference, "batch-invariant")
    patch_manifest = load_patch_manifest(patch_manifest_path, config)
    installed_runner = validate_installed_patch_stack(patch_manifest)
    environment = _load_object(environment_path)
    topology = _load_object(topology_path)
    environment_validation = validate_environment(environment, config)
    topology_validation = validate_topology(topology, config)
    if not environment_validation["valid"]:
        raise RuntimeError(
            "invalid Phase-4 environment: "
            + "; ".join(environment_validation["errors"])
        )
    if not topology_validation["valid"]:
        raise RuntimeError("invalid Phase-4 topology: " + "; ".join(topology_validation["errors"]))
    if _visible_physical_ids() != config.target.physical_gpu_ids:
        raise RuntimeError("Dual-Batch Target must see only physical GPUs 1,2")
    if not draft_socket_path.is_socket():
        raise RuntimeError("asynchronous Draft service socket is not ready")
    draft_ready = _load_object(draft_ready_path)
    if draft_ready.get("schema_version") != "specrhythm.phase4b-dual-draft-ready.v1":
        raise RuntimeError("Draft service does not expose Phase-4B readiness evidence")
    if draft_ready.get("scheduler_poll_blocks_on_gpu") is not False:
        raise RuntimeError("Draft readiness service is not non-blocking")
    requests = load_smoke_requests(
        workload_path,
        expected_count=request_count,
        require_task_mixture=request_count in {5, 100},
    )
    if request_count == 100:
        counts = {
            task: sum(item.task_class == task for item in requests)
            for task in ("code", "chat", "summarization")
        }
        if counts != {"code": 60, "chat": 20, "summarization": 20}:
            raise ValueError("100-request correctness workload must be 60/20/20")
    completed_rows = CheckpointJsonl(output_checkpoint_path).read() if resume else []
    completed_ids = {str(row.get("request_id", "")) for row in completed_rows}
    if not resume and completed_rows:
        raise FileExistsError("Dual-Batch output checkpoint already exists")
    unknown = completed_ids - {item.request_id for item in requests}
    if unknown:
        raise ValueError(f"resume checkpoint contains unknown requests: {sorted(unknown)}")
    _configure_environment(
        workload_path=workload_path,
        draft_socket_path=draft_socket_path,
        scheduler_events_path=scheduler_events_path,
        request_state_events_path=request_state_events_path,
        proposal_events_path=proposal_events_path,
        verification_events_path=verification_events_path,
        transport_events_path=transport_events_path,
        target_diagnostics_path=target_diagnostics_path,
        plugin_report_path=plugin_report_path,
        microbatch_size=microbatch_size,
        request_count=request_count,
    )
    try:
        import torch
        from vllm import LLM, SamplingParams
    except ImportError as error:
        raise RuntimeError("Phase-4B requires the pinned GPU environment") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to fabricate Dual-Batch artifacts")
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
            "specrhythm.phase4.vllm_dual_scheduler.DualBatchScheduler"
        ),
        speculative_config={
            "model": "specrhythm.phase4.vllm_dual.DualBatchRemoteProposer",
            "method": "custom_class",
            "num_speculative_tokens": config.proposal_budget,
        },
        disable_log_stats=False,
    )
    worker_ranks = llm.collective_rpc(_worker_runtime_snapshot)
    rank_errors = validate_worker_ranks(worker_ranks, config.target)
    batch_validation = validate_batch_invariant_ranks(worker_ranks, requested=True)
    rank_errors.extend(batch_validation["batch_invariant_validation"]["errors"])
    if rank_errors:
        raise RuntimeError("invalid Target worker evidence: " + "; ".join(rank_errors))
    tokenizer = llm.get_tokenizer()
    for request in requests:
        actual = tokenizer.encode(request.prompt_text, add_special_tokens=True)
        if list(actual) != list(request.prompt_token_ids):
            raise RuntimeError(f"Target tokenizer disagrees for {request.request_id}")
    pending = [item for item in requests if item.request_id not in completed_ids]
    effective_cohort = cohort_size or len(requests)
    if effective_cohort < 2 and len(pending) > 1:
        raise ValueError("Dual-Batch cohorts must contain at least two requests")
    checkpoint = CheckpointJsonl(output_checkpoint_path)
    for offset in range(0, len(pending), effective_cohort):
        cohort = pending[offset : offset + effective_cohort]
        if len(cohort) == 1 and pending:
            raise ValueError("a one-request final cohort cannot establish Dual-Batch overlap")
        outputs = _run_stable_cohort(llm, SamplingParams, cohort, config.logprobs)
        serialized = _serialize_outputs(outputs, cohort)
        for row in serialized:
            checkpoint.append(row)
    torch.cuda.synchronize()
    ended = time.monotonic_ns()
    draft_shutdown = DualDraftClient(draft_socket_path).call("shutdown", {})
    plugin_report = _load_object(plugin_report_path)
    identity_errors = _validate_request_identity_report(plugin_report, requests)
    if identity_errors:
        raise RuntimeError(
            "invalid internal/stable request identity evidence: "
            + "; ".join(identity_errors)
        )
    serialized = _ordered_checkpoint_rows(output_checkpoint_path, requests)
    comparison = compare_outputs_to_reference(serialized, reference)
    cycle_rows, overlap_rows = build_cycle_and_overlap_events(
        CheckpointJsonl(draft_work_events_path).read(),
        CheckpointJsonl(verification_events_path).read(),
        CheckpointJsonl(proposal_events_path).read(),
    )
    _write_checkpoint_rows(cycle_events_path, cycle_rows, resume=resume)
    _write_checkpoint_rows(overlap_events_path, overlap_rows, resume=resume)
    cycle_errors = validate_cycle_rows(cycle_rows)
    runtime = build_runtime_manifest(
        config,
        role="target",
        git_commit=git_commit,
        workload_path=workload_path,
        environment_path=environment_path,
        topology_path=topology_path,
        worker_ranks=worker_ranks,
        attention_backend=",".join(
            sorted(
                {
                    str(backend)
                    for row in worker_ranks
                    for backend in row.get("attention_backends", ())
                }
            )
        ),
        correctness_mode="batch-invariant",
        mode_setup=mode_evidence,
        batch_invariant_validation=batch_validation,
    )
    runtime.update(
        {
            "stage": "phase4b1-dual-batch-correctness-readiness",
            "mode": "dual-batch",
            "result_kind": "gpu-correctness-and-overlap-existence-only",
            "serving_engine": True,
            "vllm_dbo_enabled": False,
            "packed_tree_verification": False,
            "reports_speedup": False,
            "reports_goodput": False,
            "reports_slo_attainment": False,
            "ready_only_scheduler": True,
            "scheduler_cls": (
                "specrhythm.phase4.vllm_dual_scheduler.DualBatchScheduler"
            ),
            "patch_manifest_sha256": sha256_file(patch_manifest_path),
            "installed_runner": installed_runner,
            "draft_ready_sha256": sha256_file(draft_ready_path),
        }
    )
    atomic_write_json(runtime_manifest_path, runtime)
    result = {
        "schema_version": "specrhythm.phase4b-dual-batch-run.v1",
        "mode": "dual-batch",
        "stage": "phase4b1-gpu-correctness-readiness",
        "gpu_execution_performed": True,
        "performance_result": False,
        "reports_speedup": False,
        "reports_goodput": False,
        "reports_slo_attainment": False,
        "packed_tree_verification": False,
        "residual_selection": False,
        "eager": False,
        "shaping": False,
        "request_count": len(requests),
        "candidate_budget": config.proposal_budget,
        "microbatch_size": microbatch_size,
        "cohort_size": effective_cohort,
        "checkpoint_resume": True,
        "run_start_ns": started,
        "run_end_ns": ended,
        "outputs": serialized,
        "comparison": comparison,
        "exact_sequence_match": comparison["all_sequences_equal"],
        "worker_ranks": worker_ranks,
        **batch_validation,
        "runtime_semantics": {
            "execution": "dual-batch",
            "draft_target_gpu_overlap_required": True,
            "draft_verify_request_sets_disjoint": True,
            "proposal_handoff": "asynchronous-ready-only-unix-socket",
            "target_blocks_waiting_for_draft": False,
            "one_unverified_proposal_per_request": True,
            "vllm_dbo_enabled": False,
            "target_verification": "vllm-linear-speculative-batched-verification",
            "draft_kv_reuse": True,
            "target_kv_reuse": True,
            "request_identity_adapter": "unique-frozen-prompt-token-prefix",
            "vllm_internal_request_ids_opaque": True,
            "draft_physical_block_identity_observable": False,
            "target_block_identity_source": "target-diagnostics-when-observable",
        },
        "cycle_count": len(cycle_rows),
        "cycle_validation": {"valid": not cycle_errors, "errors": cycle_errors},
        "overlap_intervals_positive": sum(
            int(row.get("overlap_duration_ns", 0)) > 0 for row in overlap_rows
        ),
        "draft_service_shutdown": draft_shutdown,
        "request_identity": plugin_report["request_identity"],
        "artifact_sha256": {
            "workload": sha256_file(workload_path),
            "reference": sha256_file(reference_path),
            "runtime_manifest": sha256_file(runtime_manifest_path),
            "output_checkpoint": sha256_file(output_checkpoint_path),
        },
    }
    atomic_write_json(output_path, result)
    return result


def run_resident_dual_batch(
    config: Phase4Config,
    *,
    workload_path: Path,
    request_count: int,
    environment_path: Path,
    topology_path: Path,
    patch_manifest_path: Path,
    draft_socket_path: Path,
    draft_ready_path: Path,
    context_path: Path,
    decode_ready_manifest_path: Path,
    timing_events_path: Path,
    setup_control_path: Path,
    setup_ready_path: Path,
    scheduler_events_path: Path,
    request_state_events_path: Path,
    proposal_events_path: Path,
    proposal_lifecycle_path: Path,
    verification_events_path: Path,
    draft_work_events_path: Path,
    transport_events_path: Path,
    target_diagnostics_path: Path,
    plugin_report_path: Path,
    output_checkpoint_path: Path,
    cycle_events_path: Path,
    overlap_events_path: Path,
    runtime_manifest_path: Path,
    output_path: Path,
    git_commit: str,
    microbatch_size: int = 2,
    test_coordination: str = "none",
    overlap_requirement: str = "required",
    phase4b2_performance: bool = False,
) -> dict[str, Any]:
    """Run real decode-only resident Dual correctness without performance claims."""

    if request_count not in {2, 5, 100}:
        raise ValueError("Phase-4B.1 allows only 2, 5, or 100 requests")
    if microbatch_size < 1:
        raise ValueError("Dual microbatch size must be positive")
    if test_coordination not in {"none", "one-ready", "two-ready"}:
        raise ValueError("unknown test-only readiness coordination")
    if overlap_requirement not in {"required", "separate-gate"}:
        raise ValueError("unknown overlap requirement")
    artifacts = (
        context_path,
        decode_ready_manifest_path,
        timing_events_path,
        setup_control_path,
        setup_ready_path,
        scheduler_events_path,
        request_state_events_path,
        proposal_events_path,
        proposal_lifecycle_path,
        verification_events_path,
        target_diagnostics_path,
        plugin_report_path,
        output_checkpoint_path,
        cycle_events_path,
        overlap_events_path,
        runtime_manifest_path,
        output_path,
    )
    existing = [str(path) for path in artifacts if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite resident Dual artifacts: " + ", ".join(existing)
        )
    mode_evidence = configure_before_worker_creation("batch-invariant")
    patch_manifest = load_patch_manifest(patch_manifest_path, config)
    installed_runner = validate_installed_patch_stack(patch_manifest)
    environment = _load_object(environment_path)
    topology = _load_object(topology_path)
    environment_validation = validate_environment(environment, config)
    topology_validation = validate_topology(topology, config)
    setup_errors = [
        *environment_validation["errors"],
        *topology_validation["errors"],
    ]
    if setup_errors:
        raise RuntimeError("invalid Phase-4 environment/topology: " + "; ".join(setup_errors))
    if _visible_physical_ids() != config.target.physical_gpu_ids:
        raise RuntimeError("resident Dual Target must see configured physical GPUs 1,2")
    if not draft_socket_path.is_socket() or not draft_ready_path.is_file():
        raise RuntimeError("resident Dual requires the asynchronous GPU-0 Draft service")
    draft_ready = _load_object(draft_ready_path)
    if draft_ready.get("schema_version") != "specrhythm.phase4b-dual-draft-ready.v1":
        raise RuntimeError("resident Dual Draft readiness has the wrong schema")
    requests = load_smoke_requests(
        workload_path,
        request_count,
        require_task_mixture=request_count in {5, 100},
    )
    if request_count == 100:
        counts = {
            task: sum(row.task_class == task for row in requests)
            for task in ("code", "chat", "summarization")
        }
        if counts != {"code": 60, "chat": 20, "summarization": 20}:
            raise ValueError("100-request correctness workload must be 60/20/20")
    context = build_decode_ready_context(
        config,
        patch_manifest=patch_manifest,
        workload_path=workload_path,
        git_commit=git_commit,
        correctness_mode="batch-invariant",
    )
    atomic_write_json(context_path, context)
    _configure_resident_environment(
        workload_path=workload_path,
        draft_socket_path=draft_socket_path,
        scheduler_events_path=scheduler_events_path,
        request_state_events_path=request_state_events_path,
        proposal_events_path=proposal_events_path,
        proposal_lifecycle_path=proposal_lifecycle_path,
        verification_events_path=verification_events_path,
        transport_events_path=transport_events_path,
        target_diagnostics_path=target_diagnostics_path,
        plugin_report_path=plugin_report_path,
        context_path=context_path,
        decode_ready_manifest_path=decode_ready_manifest_path,
        timing_events_path=timing_events_path,
        setup_control_path=setup_control_path,
        setup_ready_path=setup_ready_path,
        microbatch_size=microbatch_size,
        request_count=request_count,
        test_coordination=test_coordination,
    )
    os.environ["SR_PHASE4B2_PERFORMANCE"] = (
        "1" if phase4b2_performance else "0"
    )
    try:
        import torch
        from vllm import LLM, SamplingParams
    except ImportError as error:
        raise RuntimeError("resident Dual requires the pinned GPU environment") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no resident Dual artifact was created")
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
        scheduler_cls="specrhythm.phase4.vllm_dual_scheduler.DualBatchScheduler",
        speculative_config={
            "model": "specrhythm.phase4.vllm_dual.DualBatchRemoteProposer",
            "method": "custom_class",
            "num_speculative_tokens": config.proposal_budget,
        },
        disable_log_stats=False,
    )
    worker_ranks = llm.collective_rpc(_worker_runtime_snapshot)
    rank_errors = validate_worker_ranks(worker_ranks, config.target)
    batch_validation = validate_batch_invariant_ranks(worker_ranks, requested=True)
    rank_errors.extend(batch_validation["batch_invariant_validation"]["errors"])
    if rank_errors:
        raise RuntimeError("invalid resident Dual Target ranks: " + "; ".join(rank_errors))
    tokenizer = llm.get_tokenizer()
    for request in requests:
        actual = tokenizer.encode(request.prompt_text, add_special_tokens=True)
        if list(actual) != list(request.prompt_token_ids):
            raise RuntimeError(f"Target tokenizer disagrees for {request.request_id}")
    prompts = [{"prompt_token_ids": list(row.prompt_token_ids)} for row in requests]
    parameters = [
        phase4_dual_sampling_params(row, SamplingParams, config.logprobs)
        for row in requests
    ]
    atomic_write_json(
        setup_control_path,
        build_setup_control(
            consumer="dual-batch",
            expected_request_ids=[row.request_id for row in requests],
            setup_start_ns=time.monotonic_ns(),
        ),
    )
    started_ns = time.monotonic_ns()
    outputs = llm.generate(prompts, parameters, use_tqdm=False)
    final_sync_rows = (
        llm.collective_rpc(_worker_performance_finalize)
        if phase4b2_performance
        else []
    )
    if not phase4b2_performance:
        torch.cuda.synchronize()
    ended_ns = time.monotonic_ns()
    serialized = _serialize_outputs(outputs, requests)
    checkpoint = CheckpointJsonl(output_checkpoint_path)
    for row in serialized:
        checkpoint.append(row)
    draft_shutdown = DualDraftClient(draft_socket_path).call("shutdown", {})
    manifest = load_decode_ready_manifest(_load_object(decode_ready_manifest_path))
    setup_ready = load_setup_ready(
        setup_ready_path,
        manifest_path=decode_ready_manifest_path,
        consumer="dual-batch",
        expected_request_ids=[row.request_id for row in requests],
    )
    state_rows = CheckpointJsonl(request_state_events_path).read()
    proposal_rows = CheckpointJsonl(proposal_events_path).read()
    lifecycle_rows = CheckpointJsonl(proposal_lifecycle_path).read()
    scheduler_rows = CheckpointJsonl(scheduler_events_path).read()
    draft_rows = CheckpointJsonl(draft_work_events_path).read()
    verification_rows = CheckpointJsonl(verification_events_path).read()
    cycle_rows, overlap_rows = build_cycle_and_overlap_events(
        draft_rows, verification_rows, proposal_rows
    )
    _write_checkpoint_rows(cycle_events_path, cycle_rows, resume=False)
    _write_checkpoint_rows(overlap_events_path, overlap_rows, resume=False)
    decode_rows = _decode_rows(serialized, manifest)
    plugin_report = _load_object(plugin_report_path)
    reconciliation = None
    reconciliation_errors = []
    try:
        reconciliation = build_terminal_reconciliation(
            requests=requests,
            outputs=serialized,
            manifest=manifest,
            identity=plugin_report.get("request_identity", {}),
            state_rows=state_rows,
            scheduler_rows=scheduler_rows,
            lifecycle_rows=lifecycle_rows,
            proposal_rows=proposal_rows,
            observation_ns=time.monotonic_ns(),
        )
    except (KeyError, TypeError, ValueError) as error:
        reconciliation_errors.append(f"terminal reconciliation failed: {error}")
    if reconciliation is not None:
        # Target/Draft work has completed and the service is shut down. Only the
        # coordinator can join final output with retirement and observer evidence.
        for event in reconciliation["events"]:
            CheckpointJsonl(request_state_events_path).append(event)
        state_rows = [*state_rows, *reconciliation["events"]]
    overlap_errors = validate_overlap_witness(overlap_rows)
    errors = [
        *reconciliation_errors,
        *validate_request_state_events(state_rows),
        *validate_proposal_lifecycle_events(lifecycle_rows),
        *validate_scheduler_cycles(
            scheduler_rows,
            proposal_lifecycle_rows=lifecycle_rows,
            state_rows=state_rows,
            draft_rows=draft_rows,
        ),
        *validate_round_accounting(proposal_rows),
        *_validate_request_identity_report(plugin_report, requests),
    ]
    if overlap_requirement == "required":
        errors.extend(overlap_errors)
    measurement_start_ns = manifest.measurement_start_ns
    for row in lifecycle_rows:
        if (
            row.get("lifecycle_state") == "CREATED"
            and int(row.get("draft_start_ns", -1)) < measurement_start_ns
        ):
            errors.append("initial or later Draft proposal predates measurement_start")
    runtime = build_runtime_manifest(
        config,
        role="target",
        git_commit=git_commit,
        workload_path=workload_path,
        environment_path=environment_path,
        topology_path=topology_path,
        worker_ranks=worker_ranks,
        attention_backend=",".join(
            sorted(
                {
                    str(backend)
                    for row in worker_ranks
                    for backend in row.get("attention_backends", ())
                }
            )
        ),
        correctness_mode="batch-invariant",
        mode_setup=mode_evidence,
        batch_invariant_validation=batch_validation,
    )
    runtime.update(
        {
            "stage": "phase4b1-real-decode-only-dual-correctness",
            "performance_result": False,
            "decode_ready_manifest_sha256": manifest.manifest_sha256,
            "installed_runner": installed_runner,
            "patch_manifest_sha256": sha256_file(patch_manifest_path),
        }
    )
    atomic_write_json(runtime_manifest_path, runtime)
    result = {
        "schema_version": "specrhythm.phase4b1-resident-dual-run.v1",
        "mode": "decode-only-dual-batch",
        "stage": "phase4b1-real-decode-only-dual-correctness",
        "valid": not errors,
        "errors": errors,
        "gpu_correctness_result": True,
        "performance_result": False,
        "phase4b2_performance_candidate": phase4b2_performance,
        "phase4b2_final_sync": final_sync_rows,
        "historical_gate3_qualification": {
            "gate3_exact_stock_equivalence": False,
            "logical_correctness_qualification": True,
            "numerical_qualification": "complete",
            "phase4b2_progression_permitted": True,
            "stock_comparison_required_for_phase4b2_validity": False,
        },
        "reports_speedup": False,
        "reports_tpot": False,
        "reports_throughput": False,
        "reports_goodput": False,
        "reports_slo_attainment": False,
        "request_count": request_count,
        "outputs": serialized,
        "decode_only_outputs": decode_rows,
        "decode_ready_manifest_sha256": manifest.manifest_sha256,
        "global_setup_ready": setup_ready,
        "run_start_ns": started_ns,
        "run_end_ns": ended_ns,
        "worker_ranks": worker_ranks,
        **batch_validation,
        "draft_shutdown": draft_shutdown,
        "request_identity": plugin_report.get("request_identity"),
        "runtime_semantics": {
            "execution": "real-asynchronous-dual-batch",
            "target_blind_draft": True,
            "initial_proposal_after_measurement_start": True,
            "proposal_handoff": "asynchronous-ready-only-unix-socket",
            "target_blocks_waiting_for_draft": False,
            "target_verification": "vllm-linear-speculative-batched-verification",
            "packed_tree_verification": False,
            "dual_eager": False,
            "kv_connector": False,
            "test_only_readiness_coordination": test_coordination,
            "overlap_requirement": overlap_requirement,
        },
        "overlap_gate": {
            "required_for_run_validity": overlap_requirement == "required",
            "valid": not overlap_errors,
            "errors": overlap_errors,
        },
        "retired_ready_results": summarize_retired_ready_results(scheduler_rows),
        "terminal_state_reconciliation": reconciliation,
        "evidence_counts": {
            "state_events": len(state_rows),
            "proposal_rounds": len(proposal_rows),
            "proposal_lifecycle_events": len(lifecycle_rows),
            "scheduler_cycles": len(scheduler_rows),
            "verification_events": len(verification_rows),
            "overlap_events": len(overlap_rows),
        },
        "artifact_sha256": {
            "workload": sha256_file(workload_path),
            "decode_ready_manifest": sha256_file(decode_ready_manifest_path),
            "setup_ready": sha256_file(setup_ready_path),
            "output_checkpoint": sha256_file(output_checkpoint_path),
            "runtime_manifest": sha256_file(runtime_manifest_path),
        },
    }
    atomic_write_json(output_path, result)
    return result


def summarize_retired_ready_results(
    scheduler_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Expose late-result evidence without changing performance or validity gates."""

    events = [event for row in scheduler_rows for event in row.get("retired_ready_results", ())]
    return {
        "retired_ready_result_drop_count": len(events),
        "retired_proposal_drop_count": sum(row["result_kind"] == "proposal" for row in events),
        "retired_tail_drop_count": sum(row["result_kind"] == "target-tail" for row in events),
        "events": events,
    }


def build_cycle_and_overlap_events(
    draft_rows: Sequence[Mapping[str, Any]],
    verify_rows: Sequence[Mapping[str, Any]],
    proposal_rows: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    drafts = []
    for row in draft_rows:
        result = row.get("result")
        proposal = result.get("proposal") if isinstance(result, Mapping) else None
        interval = result.get("draft_gpu_interval") if isinstance(result, Mapping) else None
        if isinstance(proposal, Mapping) and isinstance(interval, Mapping):
            drafts.append(
                {
                    "request_id": str(row.get("request_id", "")),
                    "proposal_id": str(proposal.get("proposal_id", "")),
                    "round_id": proposal.get("round_id"),
                    "start": interval.get("host_start_ns"),
                    "end": interval.get("host_end_ns"),
                    "physical_gpu_id": interval.get("physical_gpu_id"),
                    "cuda_elapsed_ns": interval.get("cuda_elapsed_ns"),
                }
            )
    verify_batches: list[Mapping[str, Any]] = []
    seen_batches = set()
    for row in verify_rows:
        batch_id = row.get("verify_microbatch_id")
        if batch_id in seen_batches:
            continue
        seen_batches.add(batch_id)
        verify_batches.append(row)
    cycles = []
    overlaps = []
    for cycle_id, verify in enumerate(verify_batches):
        verify_ids = tuple(str(item) for item in verify.get("verify_request_ids", ()))
        verify_start = verify.get("verify_host_start_ns")
        verify_end = verify.get("verify_host_end_ns")
        candidates = [
            row
            for row in drafts
            if row["request_id"] not in verify_ids
            and isinstance(row["start"], int)
            and isinstance(row["end"], int)
            and isinstance(verify_start, int)
            and isinstance(verify_end, int)
            and row["end"] > verify_start
            and row["start"] < verify_end
        ]
        draft_ids = tuple(dict.fromkeys(row["request_id"] for row in candidates))
        draft_start = min((row["start"] for row in candidates), default=None)
        draft_end = max((row["end"] for row in candidates), default=None)
        overlap_start = (
            max(draft_start, verify_start) if draft_start is not None else None
        )
        overlap_end = min(draft_end, verify_end) if draft_end is not None else None
        duration = (
            max(0, overlap_end - overlap_start)
            if overlap_start is not None and overlap_end is not None
            else 0
        )
        commits = [
            row
            for row in proposal_rows
            if row.get("verify_microbatch_id") == verify.get("verify_microbatch_id")
        ]
        commit_start = min(
            (
                row["commit_start_ns"]
                for row in commits
                if isinstance(row.get("commit_start_ns"), int)
            ),
            default=None,
        )
        commit_end = max(
            (row["commit_end_ns"] for row in commits if isinstance(row.get("commit_end_ns"), int)),
            default=None,
        )
        cycle = {
            "schema_version": "specrhythm.phase4b-dual-cycle.v1",
            "cycle_id": cycle_id,
            "draft_microbatch_id": f"draft-overlap-{cycle_id}" if candidates else None,
            "verify_microbatch_id": verify.get("verify_microbatch_id"),
            "draft_request_ids": list(draft_ids),
            "verify_request_ids": list(verify_ids),
            "draft_start_ns": draft_start,
            "draft_end_ns": draft_end,
            "verify_start_ns": verify_start,
            "verify_end_ns": verify_end,
            "commit_start_ns": commit_start,
            "commit_end_ns": commit_end,
            "overlap_start_ns": overlap_start if duration else None,
            "overlap_end_ns": overlap_end if duration else None,
            "overlap_duration_ns": duration,
        }
        cycles.append(cycle)
        overlaps.append(
            {
                "schema_version": "specrhythm.phase4b-overlap-event.v1",
                "cycle_id": cycle_id,
                "draft_request_ids": list(draft_ids),
                "verify_request_ids": list(verify_ids),
                "request_sets_disjoint": not (set(draft_ids) & set(verify_ids)),
                "draft_physical_gpu_ids": sorted(
                    {row["physical_gpu_id"] for row in candidates}
                ),
                "target_physical_gpu_ids": verify.get("target_physical_gpu_ids", []),
                "draft_cuda_events": all(row["cuda_elapsed_ns"] is not None for row in candidates),
                "target_rank_intervals": verify.get("target_rank_intervals", []),
                "host_interval": [overlap_start, overlap_end] if duration else None,
                "overlap_duration_ns": duration,
                "overlap_ratio_for_diagnostic_only": (
                    duration / max(verify_end - verify_start, 1)
                    if duration and isinstance(verify_start, int) and isinstance(verify_end, int)
                    else 0.0
                ),
                "performance_claim": False,
            }
        )
    return cycles, overlaps


def _run_stable_cohort(
    llm: Any, sampling_params_type: Any, requests: Sequence[Any], logprobs: int
) -> list[Any]:
    for request in requests:
        params = phase4_dual_sampling_params(request, sampling_params_type, logprobs)
        llm.llm_engine.add_request(
            request.request_id,
            {"prompt_token_ids": list(request.prompt_token_ids)},
            params,
            prompt_text=request.prompt_text,
        )
    outputs = []
    while llm.llm_engine.has_unfinished_requests():
        for output in llm.llm_engine.step():
            if output.finished:
                outputs.append(output)
    by_id = {str(output.request_id): output for output in outputs}
    missing = [item.request_id for item in requests if item.request_id not in by_id]
    if missing:
        raise RuntimeError(f"Dual Target did not finish requests: {missing}")
    return [by_id[item.request_id] for item in requests]


def _ordered_checkpoint_rows(path: Path, requests: Sequence[Any]) -> list[dict[str, Any]]:
    rows = CheckpointJsonl(path).read()
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        request_id = str(row.get("request_id", ""))
        if request_id in by_id:
            raise ValueError(f"duplicate output checkpoint request: {request_id}")
        value = dict(row)
        value.pop("record_sha256", None)
        by_id[request_id] = value
    missing = [item.request_id for item in requests if item.request_id not in by_id]
    if missing:
        raise ValueError(f"output checkpoint is incomplete: {missing}")
    return [by_id[item.request_id] for item in requests]


def _write_checkpoint_rows(path: Path, rows: Sequence[Mapping[str, Any]], *, resume: bool) -> None:
    if path.exists():
        if resume:
            existing = CheckpointJsonl(path).read()
            normalized = [
                {
                    key: value
                    for key, value in row.items()
                    if key != "record_sha256"
                }
                for row in existing
            ]
            if normalized == list(rows):
                return
        raise FileExistsError(f"refusing to overwrite checkpoint artifact {path}")
    log = CheckpointJsonl(path)
    for row in rows:
        log.append(row)


def _configure_environment(**paths: Any) -> None:
    mapping = {
        "SR_PHASE4_WORKLOAD": paths["workload_path"],
        "SR_PHASE4_DUAL_DRAFT_SOCKET": paths["draft_socket_path"],
        "SR_PHASE4_DUAL_SCHEDULER_EVENTS": paths["scheduler_events_path"],
        "SR_PHASE4_REQUEST_STATE_EVENTS": paths["request_state_events_path"],
        "SR_PHASE4_PROPOSAL_EVENTS": paths["proposal_events_path"],
        "SR_PHASE4_VERIFICATION_EVENTS": paths["verification_events_path"],
        "SR_PHASE4_TRANSPORT_EVENTS": paths["transport_events_path"],
        "SR_PHASE4_TARGET_DIAGNOSTICS": paths["target_diagnostics_path"],
        "SR_PHASE4_DUAL_PLUGIN_REPORT": paths["plugin_report_path"],
    }
    os.environ["SR_PHASE4_DUAL_BATCH"] = "1"
    os.environ["SR_PHASE4_DUAL_MICROBATCH_SIZE"] = str(paths["microbatch_size"])
    os.environ["SR_PHASE4_REQUEST_COUNT"] = str(paths["request_count"])
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    for key, value in mapping.items():
        os.environ[key] = str(value)


def _configure_resident_environment(**paths: Any) -> None:
    _configure_environment(
        workload_path=paths["workload_path"],
        draft_socket_path=paths["draft_socket_path"],
        scheduler_events_path=paths["scheduler_events_path"],
        request_state_events_path=paths["request_state_events_path"],
        proposal_events_path=paths["proposal_events_path"],
        verification_events_path=paths["verification_events_path"],
        transport_events_path=paths["transport_events_path"],
        target_diagnostics_path=paths["target_diagnostics_path"],
        plugin_report_path=paths["plugin_report_path"],
        microbatch_size=paths["microbatch_size"],
        request_count=paths["request_count"],
    )
    os.environ.update(
        {
            "SR_PHASE4_DUAL_RESIDENT": "1",
            "SR_PHASE4_DECODE_READY_CONTEXT": str(paths["context_path"]),
            "SR_PHASE4_DECODE_READY_MANIFEST": str(
                paths["decode_ready_manifest_path"]
            ),
            "SR_PHASE4_DECODE_READY_TIMING_EVENTS": str(paths["timing_events_path"]),
            "SR_PHASE4_RESIDENT_SETUP_CONTROL": str(paths["setup_control_path"]),
            "SR_PHASE4_RESIDENT_SETUP_READY": str(paths["setup_ready_path"]),
            "SR_PHASE4_PROPOSAL_LIFECYCLE_EVENTS": str(
                paths["proposal_lifecycle_path"]
            ),
            "SR_PHASE4_DUAL_TEST_COORDINATION": str(paths["test_coordination"]),
        }
    )


def _validate_request_identity_report(
    report: Mapping[str, Any], requests: Sequence[Any]
) -> list[str]:
    errors = []
    identity = report.get("request_identity")
    if not isinstance(identity, Mapping):
        return ["plugin report lacks request identity metadata"]
    if identity.get("mapping_source") != "unique frozen prompt_token_ids":
        errors.append("identity mapping does not use the frozen prompt-token prefix")
    if identity.get("suffix_parsing") is not False:
        errors.append("internal request IDs were parsed instead of treated as opaque")
    bindings = identity.get("bindings")
    if not isinstance(bindings, list):
        return [*errors, "identity bindings are missing"]
    if any(not isinstance(row, Mapping) for row in bindings):
        return [*errors, "identity binding is not an object"]
    internal_ids = [str(row.get("internal_request_id", "")) for row in bindings]
    stable_ids = [str(row.get("request_id", "")) for row in bindings]
    expected = {str(request.request_id) for request in requests}
    if not internal_ids or any(not item for item in internal_ids):
        errors.append("identity bindings contain an empty internal request ID")
    if len(internal_ids) != len(set(internal_ids)):
        errors.append("one internal request ID has multiple stable bindings")
    if len(stable_ids) != len(set(stable_ids)):
        errors.append("multiple internal request IDs alias one stable request")
    if set(stable_ids) != expected:
        errors.append("identity bindings do not cover the frozen workload exactly")
    if identity.get("bound_request_count") != len(bindings):
        errors.append("identity binding count is inconsistent")
    return errors


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value
