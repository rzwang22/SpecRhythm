"""GPU-only ResidentWarmStart Target consumer and artifact validation."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from specrhythm.phase4.batch_invariant import (
    configure_before_worker_creation,
    validate_batch_invariant_ranks,
)
from specrhythm.phase4.config import Phase4Config
from specrhythm.phase4.decode_ready import (
    build_first_target_forward_contract,
    compare_decode_consumers,
    compare_raw_and_decode_outputs,
    load_decode_ready_manifest,
    validate_first_target_forward_contract,
    validate_measurement_boundary,
)
from specrhythm.phase4.manifest import (
    atomic_write_json,
    sha256_file,
    validate_environment,
    validate_topology,
)
from specrhythm.phase4.reference import (
    compare_outputs_to_reference,
    load_reference,
    require_exact_resident_reference_reuse,
    require_reference_for_mode,
)
from specrhythm.phase4.resident_setup import (
    build_setup_control,
    load_setup_ready,
    validate_resident_admission_events,
)
from specrhythm.phase4.serial_runner import (
    load_patch_manifest,
    validate_engine_residency,
    validate_installed_patch_stack,
)
from specrhythm.phase4.stock_vllm import (
    _serialize_outputs,
    _visible_physical_ids,
    _worker_runtime_snapshot,
    load_smoke_requests,
    validate_worker_ranks,
)
from specrhythm.phase4.transport import CheckpointJsonl, UnixDraftClient


def build_decode_ready_context(
    config: Phase4Config,
    *,
    patch_manifest: Mapping[str, Any],
    workload_path: Path,
    git_commit: str,
    correctness_mode: str,
) -> dict[str, Any]:
    stack = patch_manifest.get("patch_stack")
    if not isinstance(stack, list) or not stack:
        raise ValueError("decode-ready context requires the ordered vLLM patch stack")
    hashes = []
    for row in stack:
        if not isinstance(row, Mapping) or not row.get("patch_sha256"):
            raise ValueError("vLLM patch stack checksum is missing")
        hashes.append(str(row["patch_sha256"]))
    return {
        "schema_version": "specrhythm.phase4b-decode-ready-context.v1",
        "specrhythm_git_commit": git_commit,
        "vllm_version": config.expected_vllm_version,
        "vllm_commit": config.expected_vllm_commit,
        "vllm_patch_stack_sha256": hashes,
        "target_model_path": str(config.target.resolved_model_path),
        "target_model_revision": config.target.revision,
        "draft_model_path": str(config.draft.resolved_model_path),
        "draft_model_revision": config.draft.revision,
        "tokenizer_revision": config.target.tokenizer_revision,
        "workload_sha256": sha256_file(workload_path),
        "sampling_configuration": config.sampling.to_dict(),
        "batch_invariant_configuration": {
            "correctness_mode": correctness_mode,
            "requested": correctness_mode == "batch-invariant",
            "VLLM_BATCH_INVARIANT": (
                "1" if correctness_mode == "batch-invariant" else "0"
            ),
            "enable_dbo": False,
        },
        "target_physical_gpu_ids": list(config.target.physical_gpu_ids),
        "draft_physical_gpu_ids": list(config.draft.physical_gpu_ids),
        "target_tensor_parallel_size": config.target.tensor_parallel_size,
        "draft_tensor_parallel_size": config.draft.tensor_parallel_size,
    }


def run_resident_target(
    config: Phase4Config,
    *,
    workload_path: Path,
    request_count: int,
    environment_path: Path,
    topology_path: Path,
    reference_path: Path,
    patch_manifest_path: Path,
    draft_socket_path: Path,
    draft_ready_path: Path,
    context_path: Path,
    decode_ready_manifest_path: Path,
    timing_events_path: Path,
    setup_control_path: Path,
    setup_ready_path: Path,
    admission_events_path: Path,
    target_diagnostics_path: Path,
    plugin_report_path: Path,
    first_forward_path: Path,
    output_path: Path,
    git_commit: str,
    correctness_mode: str = "batch-invariant",
) -> dict[str, Any]:
    """Run a real-KV decode-only Target pass; never derive performance metrics."""

    if request_count not in {2, 5}:
        raise ValueError("Phase-4B.0b Gate B allows only 2 or 5 requests")
    for path in (
        context_path,
        decode_ready_manifest_path,
        timing_events_path,
        setup_control_path,
        setup_ready_path,
        admission_events_path,
        target_diagnostics_path,
        plugin_report_path,
        first_forward_path,
        output_path,
    ):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite resident artifact {path}")
    mode_evidence = configure_before_worker_creation(correctness_mode)
    reference = load_reference(reference_path)
    require_reference_for_mode(reference, correctness_mode)
    require_exact_resident_reference_reuse(reference, config, workload_path)
    patch_manifest = load_patch_manifest(patch_manifest_path, config)
    installed_stack = validate_installed_patch_stack(patch_manifest)
    environment = _read_object(environment_path)
    topology = _read_object(topology_path)
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
        raise RuntimeError("resident Target must see only configured Target GPUs")
    if not draft_socket_path.is_socket() or not draft_ready_path.is_file():
        raise RuntimeError("resident Target requires the live persistent Draft service")
    draft_ready = _read_object(draft_ready_path)
    if draft_ready.get("schema_version") != "specrhythm.phase4-draft-service-ready.v1":
        raise RuntimeError("resident Target Draft service evidence has the wrong schema")
    requests = load_smoke_requests(
        workload_path, request_count, require_task_mixture=request_count == 5
    )
    context = build_decode_ready_context(
        config,
        patch_manifest=patch_manifest,
        workload_path=workload_path,
        git_commit=git_commit,
        correctness_mode=correctness_mode,
    )
    atomic_write_json(context_path, context)
    os.environ.update(
        {
            "SR_PHASE4_DRAFT_SOCKET": str(draft_socket_path),
            "SR_PHASE4_WORKLOAD": str(workload_path),
            "SR_PHASE4_REQUEST_COUNT": str(request_count),
            "SR_PHASE4_DECODE_READY_CONTEXT": str(context_path),
            "SR_PHASE4_DECODE_READY_MANIFEST": str(decode_ready_manifest_path),
            "SR_PHASE4_DECODE_READY_TIMING_EVENTS": str(timing_events_path),
            "SR_PHASE4_RESIDENT_SETUP": "1",
            "SR_PHASE4_RESIDENT_CONSUMER": "target-only",
            "SR_PHASE4_RESIDENT_SETUP_CONTROL": str(setup_control_path),
            "SR_PHASE4_RESIDENT_SETUP_READY": str(setup_ready_path),
            "SR_PHASE4_RESIDENT_ADMISSION_EVENTS": str(admission_events_path),
            "SR_PHASE4_TARGET_DIAGNOSTICS": str(target_diagnostics_path),
            "SR_PHASE4_PLUGIN_REPORT": str(plugin_report_path),
        }
    )
    try:
        import torch
        from vllm import LLM, SamplingParams
    except ImportError as error:
        raise RuntimeError("resident Target requires the pinned GPU environment") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no resident GPU result was created")
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
        scheduler_cls="specrhythm.phase4.resident_scheduler.ResidentSetupScheduler",
        speculative_config={
            "model": "specrhythm.phase4.resident_vllm.ResidentTargetProposer",
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
        raise RuntimeError("invalid resident Target worker evidence: " + "; ".join(rank_errors))
    tokenizer = llm.get_tokenizer()
    for request in requests:
        actual = tokenizer.encode(request.prompt_text, add_special_tokens=True)
        if list(actual) != list(request.prompt_token_ids):
            raise RuntimeError(f"Target tokenizer disagrees for {request.request_id}")
    prompts = [{"prompt_token_ids": list(row.prompt_token_ids)} for row in requests]
    parameters = [
        SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=row.maximum_new_tokens,
            seed=row.sampling_seed,
            n=1,
            logprobs=config.logprobs,
        )
        for row in requests
    ]
    atomic_write_json(
        setup_control_path,
        build_setup_control(
            consumer="target-only",
            expected_request_ids=[row.request_id for row in requests],
            setup_start_ns=time.monotonic_ns(),
        ),
    )
    outputs = llm.generate(prompts, parameters, use_tqdm=False)
    torch.cuda.synchronize()
    serialized = _serialize_outputs(outputs, requests)
    draft_shutdown = UnixDraftClient(draft_socket_path).shutdown()
    manifest_value = json.loads(decode_ready_manifest_path.read_text(encoding="utf-8"))
    manifest = load_decode_ready_manifest(manifest_value)
    setup_ready = load_setup_ready(
        setup_ready_path,
        manifest_path=decode_ready_manifest_path,
        consumer="target-only",
        expected_request_ids=[row.request_id for row in requests],
    )
    admission_rows = CheckpointJsonl(admission_events_path).read()
    admission_errors = validate_resident_admission_events(
        admission_rows, consumer="target-only"
    )
    diagnostics = CheckpointJsonl(target_diagnostics_path).read()
    plugin_report = _read_object(plugin_report_path)
    residency_errors = validate_engine_residency(
        config,
        draft_ready=draft_ready,
        target_worker_ranks=worker_ranks,
        proposer_report=plugin_report,
    )
    first_contracts, first_errors = _first_forward_contracts(
        manifest, diagnostics, serialized
    )
    atomic_write_json(
        first_forward_path,
        {
            "schema_version": "specrhythm.phase4b-first-target-forwards.v1",
            "consumer": "target-only",
            "valid": not first_errors,
            "errors": first_errors,
            "requests": first_contracts,
        },
    )
    decode_rows = _decode_rows(serialized, manifest)
    raw_rows = _reference_rows(reference, requests)
    raw_decode = compare_raw_and_decode_outputs(raw_rows, decode_rows, manifest)
    boundary_errors = validate_measurement_boundary(
        manifest,
        first_target_decode_start_ns=(
            min(int(row["target_forward_start_ns"]) for row in first_contracts)
            if first_contracts
            else None
        ),
    )
    if not first_contracts:
        boundary_errors.append("no first Target decode exists after the barrier")
    stock_comparison = compare_outputs_to_reference(serialized, reference)
    errors = first_errors + boundary_errors + residency_errors + admission_errors
    if not raw_decode["valid"]:
        errors.extend(raw_decode["errors"])
    if not stock_comparison["all_sequences_equal"]:
        errors.append("resident Target differs from immutable raw-prompt stock Target")
    result = {
        "schema_version": "specrhythm.phase4b-resident-target-run.v1",
        "mode": "decode-only-target",
        "provider_kind": "resident-warm-start",
        "request_count": request_count,
        "valid": not errors,
        "errors": errors,
        "gpu_correctness_result": True,
        "gpu_performance_result": False,
        "reports_speedup": False,
        "end_to_end_pd_deployment": False,
        "kv_connector_handoff": False,
        "outputs": serialized,
        "decode_only_outputs": decode_rows,
        "raw_vs_decode": raw_decode,
        "stock_comparison": stock_comparison,
        "decode_ready_manifest_sha256": manifest.manifest_sha256,
        "first_target_forward_valid": not first_errors,
        "measurement_boundary_valid": not boundary_errors,
        "global_setup_ready": setup_ready,
        "resident_admission": {
            "valid": not admission_errors,
            "errors": admission_errors,
            "event_count": len(admission_rows),
        },
        "worker_ranks": worker_ranks,
        "batch_invariant": batch_validation,
        "environment_validation": environment_validation,
        "topology_validation": topology_validation,
        "mode_setup": mode_evidence,
        "installed_patch_stack": installed_stack,
        "engine_residency": {
            "valid": not residency_errors,
            "errors": residency_errors,
            "draft_service": draft_ready,
            "target_plugin": plugin_report,
        },
        "draft_shutdown": draft_shutdown,
        "artifact_sha256": {
            "workload": sha256_file(workload_path),
            "reference": sha256_file(reference_path),
            "decode_ready_manifest": sha256_file(decode_ready_manifest_path),
            "first_target_forward": sha256_file(first_forward_path),
            "setup_control": sha256_file(setup_control_path),
            "setup_ready": sha256_file(setup_ready_path),
            "admission_events": sha256_file(admission_events_path),
        },
    }
    atomic_write_json(output_path, result)
    return result


def validate_resident_pair(
    *,
    target_path: Path,
    serial_path: Path,
    target_manifest_path: Path,
    serial_manifest_path: Path,
) -> dict[str, Any]:
    target = _read_object(target_path)
    serial = _read_object(serial_path)
    target_manifest = load_decode_ready_manifest(_read_object(target_manifest_path))
    serial_manifest = load_decode_ready_manifest(_read_object(serial_manifest_path))
    errors = []
    if target.get("valid") is not True:
        errors.append("resident Target run is invalid")
    if serial.get("valid") is not True:
        errors.append("resident Serial run is invalid")
    target_states = {
        row.request_id: (
            row.bootstrap_token_id,
            row.logical_committed_prefix_sha256,
            row.target_materialized_kv_token_count,
            row.draft_materialized_kv_token_count,
        )
        for row in target_manifest.requests
    }
    serial_states = {
        row.request_id: (
            row.bootstrap_token_id,
            row.logical_committed_prefix_sha256,
            row.target_materialized_kv_token_count,
            row.draft_materialized_kv_token_count,
        )
        for row in serial_manifest.requests
    }
    if target_states != serial_states:
        errors.append("Target and Serial did not start from the same logical decode-ready state")
    comparison = compare_decode_consumers(
        target.get("decode_only_outputs", ()),
        serial.get("decode_only_outputs", ()),
    )
    if not comparison["valid"]:
        errors.extend(comparison["errors"])
    return {
        "schema_version": "specrhythm.phase4b-resident-pair-validation.v1",
        "valid": not errors,
        "errors": errors,
        "target_equals_serial": comparison,
        "dual_evaluated": False,
        "performance_result": False,
    }


def _first_forward_contracts(
    manifest: Any,
    diagnostics: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    contracts = []
    errors = []
    output_by_id = {str(row.get("request_id", "")): row for row in outputs}
    for request in manifest.requests:
        matches = [
            row
            for row in diagnostics
            if row.get("request_id") == request.request_id
            and row.get("committed_prefix_sha256")
            == request.logical_committed_prefix_sha256
            and row.get("proposal_token_ids") == []
        ]
        if not matches:
            errors.append(f"{request.request_id}: first timed Target decode is missing")
            continue
        row = matches[0]
        generated = list(output_by_id.get(request.request_id, {}).get("generated_token_ids", ()))
        if len(generated) < 2:
            errors.append(f"{request.request_id}: no timed continuation token was committed")
            continue
        try:
            contract = build_first_target_forward_contract(
                request,
                consumer="target-only",
                target_forward_start_ns=int(row["target_forward_start_ns"]),
                target_forward_end_ns=int(row["target_forward_end_ns"]),
                output_logits_positions=row.get("position_ids", ()),
                accepted_draft_tokens=0,
                post_forward_committed_token_ids=(generated[1],),
                post_forward_target_kv_token_count=(
                    request.target_materialized_kv_token_count + 1
                ),
                post_forward_prefix_version=request.prefix_version + 1,
            )
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{request.request_id}: {error}")
            continue
        if row.get("target_input_token_ids") != contract["verification_input_token_ids"]:
            errors.append(f"{request.request_id}: observed first Target token differs")
        if row.get("position_ids") != contract["input_positions"]:
            errors.append(f"{request.request_id}: observed first Target position differs")
        if row.get("physical_kv_num_computed_tokens") != (
            request.target_materialized_kv_token_count
        ):
            errors.append(
                f"{request.request_id}: observed scheduler computed tokens differ from KV"
            )
        errors.extend(
            f"{request.request_id}: {item}"
            for item in validate_first_target_forward_contract(contract, request)
        )
        contracts.append(contract)
    return contracts, errors


def _decode_rows(
    rows: Sequence[Mapping[str, Any]], manifest: Any
) -> list[dict[str, Any]]:
    states = {row.request_id: row for row in manifest.requests}
    result = []
    for row in rows:
        request_id = str(row["request_id"])
        state = states[request_id]
        generated = list(row.get("generated_token_ids", ()))
        if not generated or generated[0] != state.bootstrap_token_id:
            raise RuntimeError(f"{request_id}: output does not begin with frozen bootstrap")
        result.append(
            {
                "request_id": request_id,
                "generated_token_ids": generated,
                "continuation_token_ids": generated[1:],
                "finish_reason": row.get("finish_reason"),
                "eos_token_id": row.get("stop_reason"),
                "max_token_termination": row.get("finish_reason") == "length",
                "final_logical_length": state.prompt_token_count + len(generated),
            }
        )
    return result


def _reference_rows(reference: Mapping[str, Any], requests: Sequence[Any]) -> list[dict[str, Any]]:
    prompts = {row.request_id: len(row.prompt_token_ids) for row in requests}
    rows = []
    for row in reference.get("outputs", ()):
        request_id = str(row.get("request_id", ""))
        if request_id not in prompts:
            continue
        tokens = list(row.get("generated_token_ids", ()))
        rows.append(
            {
                **row,
                "eos_token_id": row.get("stop_reason"),
                "max_token_termination": row.get("finish_reason") == "length",
                "final_logical_length": prompts[request_id] + len(tokens),
            }
        )
    return rows


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value
