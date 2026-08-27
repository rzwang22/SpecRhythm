"""GPU-only stock vLLM bring-up runner.

Imports of vLLM and PyTorch are intentionally local to runtime functions so
the default dependency-free package remains usable on Python 3.9 and CPU CI.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from specrhythm.phase4.batch_invariant import (
    configure_before_worker_creation,
    requested_for_mode,
    validate_batch_invariant_ranks,
    worker_batch_invariant_evidence,
)
from specrhythm.phase4.config import EngineConfig, Phase4Config
from specrhythm.phase4.manifest import (
    atomic_write_json,
    build_runtime_manifest,
    sha256_file,
    validate_environment,
    validate_topology,
)
from specrhythm.phase4.transport import CheckpointJsonl


@dataclass(frozen=True)
class SmokeRequest:
    request_id: str
    task_class: str
    prompt_text: str
    prompt_token_ids: Tuple[int, ...]
    maximum_new_tokens: int
    sampling_seed: int
    tokenizer_fingerprint: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SmokeRequest":
        request_id = str(value.get("request_id", "")).strip()
        task_class = str(value.get("task_class", "")).strip()
        prompt_text = str(value.get("prompt_text", ""))
        token_ids = tuple(value.get("prompt_token_ids", ()))
        maximum = value.get("maximum_new_tokens")
        seed = value.get("sampling_seed")
        fingerprint = str(value.get("tokenizer_fingerprint", "")).strip()
        if (
            not request_id
            or task_class not in {"code", "chat", "summarization"}
            or not prompt_text
            or not fingerprint
        ):
            raise ValueError("R3-real smoke request identity/prompt/tokenizer is incomplete")
        if task_class == "chat" and not (
            prompt_text.startswith("<|im_start|>user") and "<|im_start|>assistant" in prompt_text
        ):
            raise ValueError("R3-real chat smoke prompt is not Qwen chat-template rendered")
        if not token_ids or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in token_ids
        ):
            raise ValueError("R3-real smoke prompt_token_ids are invalid")
        if value.get("prompt_length") != len(token_ids):
            raise ValueError("R3-real prompt length/token IDs differ")
        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
            raise ValueError("R3-real maximum_new_tokens must be positive")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("R3-real sampling_seed must be an integer")
        return cls(
            request_id,
            task_class,
            prompt_text,
            token_ids,
            maximum,
            seed,
            fingerprint,
        )


def load_smoke_requests(
    path: Path, expected_count: int = 5, *, require_task_mixture: bool = True
) -> list[SmokeRequest]:
    requests = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                requests.append(SmokeRequest.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid R3-real workload line {line_number}: {error}"
                ) from error
            if len(requests) == expected_count:
                break
    if len(requests) != expected_count:
        raise ValueError(f"R3-real smoke requires exactly {expected_count} requests")
    if len({request.request_id for request in requests}) != len(requests):
        raise ValueError("R3-real smoke request IDs must be unique")
    task_counts = {
        task: sum(request.task_class == task for request in requests)
        for task in ("code", "chat", "summarization")
    }
    expected_mixture = {
        5: {"code": 3, "chat": 1, "summarization": 1},
        100: {"code": 60, "chat": 20, "summarization": 20},
    }.get(expected_count)
    if require_task_mixture and expected_mixture is None:
        raise ValueError("task-mixture validation is defined only for 5 or 100 requests")
    if expected_mixture is not None and require_task_mixture and task_counts != expected_mixture:
        raise ValueError(
            f"R3-real {expected_count}-request workload has wrong task mixture: {task_counts}"
        )
    return requests


def _nvidia_uuid(physical_gpu_id: int) -> Optional[str]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(physical_gpu_id),
            "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _visible_physical_ids() -> Tuple[int, ...]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or not raw.strip():
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must explicitly bind one Phase-4 engine to physical GPUs"
        )
    try:
        values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as error:
        raise RuntimeError(
            "Phase-4A currently requires numeric physical IDs in CUDA_VISIBLE_DEVICES"
        ) from error
    if len(values) != len(set(values)):
        raise RuntimeError("CUDA_VISIBLE_DEVICES contains duplicate physical GPU IDs")
    return values


def active_cuda_device_identity(torch: Any, device: Any = None) -> dict[str, Any]:
    """Resolve one worker's identity from its actual active CUDA device.

    vLLM may expose every worker to a different CUDA_VISIBLE_DEVICES view.  A
    process rank is therefore not a CUDA device index.  This helper is shared
    by the authoritative worker snapshot and the verification observer so both
    use the same logical-to-physical mapping contract.
    """

    resolved = torch.cuda.current_device() if device is None else device
    logical_gpu_id = int(
        resolved.index if getattr(resolved, "index", None) is not None else resolved
    )
    visible = _visible_physical_ids()
    if logical_gpu_id < 0 or logical_gpu_id >= len(visible):
        raise RuntimeError("active CUDA device is outside CUDA_VISIBLE_DEVICES")
    physical_gpu_id = visible[logical_gpu_id]
    properties = torch.cuda.get_device_properties(resolved)
    gpu_uuid = _nvidia_uuid(physical_gpu_id)
    if not gpu_uuid:
        raise RuntimeError("active CUDA device UUID is unavailable from nvidia-smi")
    return {
        "logical_cuda_index": logical_gpu_id,
        "physical_gpu_id": physical_gpu_id,
        "gpu_uuid": gpu_uuid,
        "gpu_name": properties.name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _worker_runtime_snapshot(worker: Any) -> dict[str, Any]:
    """Small callable serialized by vLLM collective_rpc to every TP worker."""

    import torch

    torch.cuda.synchronize(worker.device)
    model = worker.get_model()
    parameters = [parameter for parameter in model.parameters() if parameter.numel() > 0]
    devices = sorted({str(parameter.device) for parameter in parameters})
    identity = active_cuda_device_identity(torch, worker.device)
    logical_gpu_id = identity["logical_cuda_index"]
    physical_gpu_id = identity["physical_gpu_id"]
    allocated = int(torch.cuda.memory_allocated(worker.device))
    reserved = int(torch.cuda.memory_reserved(worker.device))
    attention_backends = set()
    for groups in getattr(worker.model_runner, "attn_groups", ()):  # V1 and V2 runners
        for group in groups:
            backend = getattr(group, "backend", None)
            if backend is not None:
                get_name = getattr(backend, "get_name", None)
                attention_backends.add(str(get_name()) if callable(get_name) else backend.__name__)
    expected_device = f"cuda:{logical_gpu_id}"
    result = {
        "global_rank": int(worker.rank),
        "local_rank": int(worker.local_rank),
        "world_size": int(worker.vllm_config.parallel_config.world_size),
        "logical_cuda_index": logical_gpu_id,
        "physical_gpu_id": physical_gpu_id,
        "cuda_visible_devices": identity["cuda_visible_devices"],
        "gpu_uuid": identity["gpu_uuid"],
        "gpu_name": identity["gpu_name"],
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "parameter_bytes": sum(
            parameter.numel() * parameter.element_size() for parameter in parameters
        ),
        "parameter_devices": devices,
        "expected_parameter_device": expected_device,
        "all_parameters_on_expected_device": bool(parameters)
        and all(str(parameter.device) == expected_device for parameter in parameters),
        "allocated_memory_bytes": allocated,
        "reserved_memory_bytes": reserved,
        "max_allocated_memory_bytes": int(torch.cuda.max_memory_allocated(worker.device)),
        "max_reserved_memory_bytes": int(torch.cuda.max_memory_reserved(worker.device)),
        "attention_backends": sorted(attention_backends),
    }
    result.update(worker_batch_invariant_evidence(worker))
    return result


def validate_worker_ranks(rows: Sequence[Mapping[str, Any]], engine: EngineConfig) -> list[str]:
    errors = []
    if len(rows) != engine.tensor_parallel_size:
        errors.append("worker rank count does not equal configured tensor parallel size")
    ranks = {row.get("global_rank") for row in rows}
    if ranks != set(range(engine.tensor_parallel_size)):
        errors.append("worker global ranks are incomplete")
    physical = {row.get("physical_gpu_id") for row in rows}
    if physical != set(engine.physical_gpu_ids):
        errors.append("workers are not bound to the expected physical GPUs")
    uuids = [str(row.get("gpu_uuid", "")) for row in rows]
    if len(rows) > 1 and len(set(uuids)) != len(rows):
        errors.append("worker ranks report aliased GPU UUIDs")
    for row in rows:
        if row.get("world_size") != engine.tensor_parallel_size:
            errors.append("worker reported an unexpected TP world size")
        if row.get("parameter_count", 0) <= 0 or row.get("parameter_bytes", 0) <= 0:
            errors.append("worker rank did not load model parameters")
        if row.get("allocated_memory_bytes", 0) <= 0:
            errors.append("worker rank has zero allocated CUDA memory")
        if not row.get("gpu_uuid"):
            errors.append("worker rank GPU UUID is missing")
        if not row.get("attention_backends"):
            errors.append("worker rank attention backend is missing")
        if not row.get("all_parameters_on_expected_device"):
            errors.append("worker parameters are not entirely on the expected CUDA device")
    return errors


def _metrics(value: Any) -> dict[str, Any]:
    if value is None:
        return {
            "available": False,
            "timebase_note": "vLLM RequestStateStats unavailable",
        }
    fields = (
        "arrival_time",
        "queued_ts",
        "scheduled_ts",
        "first_token_ts",
        "last_token_ts",
        "first_token_latency",
        "num_generation_tokens",
    )
    result = {name: getattr(value, name, None) for name in fields}
    result["available"] = True
    result["timebase_note"] = (
        "arrival_time is frontend wall-clock; queued/scheduled/first/last are "
        "engine-core monotonic"
    )
    scheduled = result.get("scheduled_ts") or 0
    first = result.get("first_token_ts") or 0
    last = result.get("last_token_ts") or 0
    result["prefill_ms"] = (first - scheduled) * 1000 if first >= scheduled > 0 else None
    result["decode_ms"] = (last - first) * 1000 if last >= first > 0 else None
    return result


def _serialize_topk(logprobs: Any) -> list[list[dict[str, Any]]]:
    if logprobs is None:
        return []
    rows = []
    for position in logprobs:
        current = []
        if position:
            for token_id, value in sorted(
                position.items(), key=lambda item: item[1].logprob, reverse=True
            ):
                current.append(
                    {
                        "token_id": int(token_id),
                        "log_probability": float(value.logprob),
                        "rank": value.rank,
                        "decoded_token": value.decoded_token,
                    }
                )
        rows.append(current)
    return rows


def _serialize_outputs(
    outputs: Sequence[Any], requests: Sequence[SmokeRequest]
) -> list[dict[str, Any]]:
    if len(outputs) != len(requests):
        raise RuntimeError("vLLM returned the wrong number of request outputs")
    result = []
    for request, output in zip(requests, outputs):
        if not output.finished or len(output.outputs) != 1:
            raise RuntimeError(f"vLLM request {request.request_id} did not finish exactly once")
        completion = output.outputs[0]
        tokens = list(completion.token_ids)
        result.append(
            {
                "request_id": request.request_id,
                "prompt_length": len(request.prompt_token_ids),
                "generated_token_ids": tokens,
                "generated_tokens": len(tokens),
                "text": completion.text,
                "finish_reason": completion.finish_reason,
                "stop_reason": completion.stop_reason,
                "top_logprobs": _serialize_topk(completion.logprobs),
                "timestamps": _metrics(output.metrics),
                "token_accounting": {
                    "prompt_tokens": len(request.prompt_token_ids),
                    "generated_tokens": len(tokens),
                    "total_tokens": len(request.prompt_token_ids) + len(tokens),
                },
            }
        )
    return result


def _target_references(path: Path) -> dict[str, Mapping[str, Any]]:
    references = {}
    for item in path.rglob("*.json"):
        try:
            value = json.loads(item.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, Mapping) and value.get("request_id"):
            references[str(value["request_id"])] = value
    return references


def compare_frozen_target(
    run: Sequence[Mapping[str, Any]], reference_dir: Optional[Path]
) -> dict[str, Any]:
    if reference_dir is None:
        return {"performed": False, "reason": "no frozen HF target directory supplied"}
    references = _target_references(reference_dir)
    comparisons = []
    all_equal = True
    coverage_complete = True
    for output in run:
        request_id = str(output["request_id"])
        reference = references.get(request_id)
        if reference is None:
            comparisons.append({"request_id": request_id, "equal": False, "error": "missing"})
            all_equal = False
            coverage_complete = False
            continue
        actual = list(output["generated_token_ids"])
        expected = list(reference.get("target_token_ids", ()))
        common = min(len(actual), len(expected))
        mismatch = next(
            (index for index in range(common) if actual[index] != expected[index]), None
        )
        if mismatch is None and len(actual) != len(expected):
            mismatch = common
        equal = mismatch is None
        all_equal = all_equal and equal
        topk = output.get("top_logprobs", [])
        comparisons.append(
            {
                "request_id": request_id,
                "equal": equal,
                "first_divergence_position": mismatch,
                "vllm_token_id": (
                    actual[mismatch] if mismatch is not None and mismatch < len(actual) else None
                ),
                "hf_token_id": (
                    expected[mismatch]
                    if mismatch is not None and mismatch < len(expected)
                    else None
                ),
                "vllm_topk_at_divergence": (
                    topk[mismatch] if mismatch is not None and mismatch < len(topk) else None
                ),
                "hf_target_model_revision": reference.get("target_model_revision"),
            }
        )
    return {
        "performed": True,
        "reference_coverage_complete": coverage_complete,
        "all_tokens_equal": all_equal,
        "reference_request_count": len(references),
        "requests": comparisons,
    }


def _update_combined_manifest(path: Path, role_manifest: Mapping[str, Any]) -> None:
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != "specrhythm.phase4-runtime-bundle.v1":
            raise ValueError("existing runtime manifest has an incompatible schema")
    else:
        value = {
            "schema_version": "specrhythm.phase4-runtime-bundle.v1",
            "stage": "phase4a-stock-vllm-bringup",
            "serving_performance_result": False,
            "roles": {},
        }
    roles = dict(value.get("roles", {}))
    for existing in roles.values():
        if not isinstance(existing, Mapping):
            raise ValueError("existing runtime manifest contains an invalid role")
        if existing.get("git_commit") != role_manifest.get("git_commit"):
            raise ValueError("refusing to combine role manifests from different commits")
        existing_inputs = existing.get("inputs")
        current_inputs = role_manifest.get("inputs")
        if not isinstance(existing_inputs, Mapping) or not isinstance(current_inputs, Mapping):
            raise ValueError("runtime role manifest input provenance is missing")
        for key in (
            "config_sha256",
            "workload_sha256",
            "environment_sha256",
            "topology_sha256",
        ):
            if existing_inputs.get(key) != current_inputs.get(key):
                raise ValueError(f"refusing to combine role manifests with different {key}")
        existing_correctness = existing.get("correctness")
        current_correctness = role_manifest.get("correctness")
        if not isinstance(existing_correctness, Mapping) or not isinstance(
            current_correctness, Mapping
        ):
            raise ValueError("runtime role manifest correctness provenance is missing")
        if existing_correctness.get("mode") != current_correctness.get("mode"):
            raise ValueError(
                "refusing to combine default and batch-invariant runtime manifests"
            )
    roles[str(role_manifest["role"])] = dict(role_manifest)
    value["roles"] = roles
    atomic_write_json(path, value)


def run_stock_smoke(
    config: Phase4Config,
    *,
    role: str,
    workload_path: Path,
    environment_path: Path,
    topology_path: Path,
    runtime_manifest_path: Path,
    git_commit: str,
    frozen_target_dir: Optional[Path] = None,
    correctness_mode: str = "default",
    diagnostics_path: Optional[Path] = None,
    request_count: Optional[int] = None,
    diagnostic_single_run: bool = False,
    numerical_plan_path: Optional[Path] = None,
    numerical_output_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Bring up one independent stock vLLM engine and repeat a frozen workload."""

    from specrhythm.phase4.numerical_diagnostics import (
        configure_numerical_diagnostic,
        validate_numerical_records,
    )

    mode_evidence = configure_before_worker_creation(correctness_mode)
    if role not in {"draft", "target"}:
        raise ValueError("stock smoke role must be draft or target")
    if diagnostics_path is not None:
        if role != "target":
            raise ValueError("Target diagnostics cannot be enabled for the Draft role")
        if diagnostics_path.exists():
            raise FileExistsError(
                f"refusing to overwrite Target diagnostics {diagnostics_path}"
            )
        os.environ["SR_PHASE4_TARGET_DIAGNOSTICS"] = str(diagnostics_path)
        os.environ["SR_PHASE4_WORKLOAD"] = str(workload_path)
    else:
        os.environ.pop("SR_PHASE4_TARGET_DIAGNOSTICS", None)
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
    engine = config.draft if role == "draft" else config.target
    visible = _visible_physical_ids()
    if visible != engine.physical_gpu_ids:
        raise RuntimeError(
            f"{role} requires CUDA_VISIBLE_DEVICES="
            + ",".join(str(item) for item in engine.physical_gpu_ids)
        )
    effective_count = request_count or config.smoke_request_count
    numerical_plan = configure_numerical_diagnostic(
        plan_path=numerical_plan_path,
        output_path=numerical_output_path,
        workload_path=workload_path,
        execution_mode=("stock-style" if diagnostic_single_run else None),
    )
    if diagnostic_single_run and (
        role != "target"
        or effective_count != 100
        or correctness_mode != "batch-invariant"
        or diagnostics_path is None
        or numerical_plan is None
        or numerical_output_path is None
    ):
        raise ValueError(
            "single-run numerical diagnosis requires Target corrected-100, "
            "batch-invariant mode, and both diagnostic outputs"
        )
    if not diagnostic_single_run and numerical_plan is not None:
        raise ValueError("numerical instrumentation is allowed only for a single diagnostic run")
    requests = load_smoke_requests(
        workload_path,
        effective_count,
        require_task_mixture=effective_count in {5, 100},
    )
    try:
        import torch
        from vllm import LLM, SamplingParams
    except ImportError as error:
        raise RuntimeError("the independent Phase-4 vLLM environment is not installed") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to generate a fake stock-engine result")
    started = time.monotonic_ns()
    llm = LLM(
        model=str(engine.resolved_model_path),
        tokenizer=str(engine.resolved_tokenizer_path),
        tensor_parallel_size=engine.tensor_parallel_size,
        dtype=engine.dtype,
        revision=engine.revision,
        tokenizer_revision=engine.tokenizer_revision,
        trust_remote_code=engine.trust_remote_code,
        seed=config.sampling.seed,
        gpu_memory_utilization=engine.gpu_memory_utilization,
        max_model_len=config.max_model_len,
        enforce_eager=config.enforce_eager,
        enable_prefix_caching=config.enable_prefix_caching,
        enable_dbo=False,
        speculative_config=None,
        disable_log_stats=False,
    )
    startup_finished = time.monotonic_ns()
    worker_ranks = llm.collective_rpc(_worker_runtime_snapshot)
    rank_errors = validate_worker_ranks(worker_ranks, engine)
    batch_validation = validate_batch_invariant_ranks(
        worker_ranks, requested=requested_for_mode(correctness_mode)
    )
    rank_errors.extend(batch_validation["batch_invariant_validation"]["errors"])
    if rank_errors:
        raise RuntimeError("invalid vLLM worker evidence: " + "; ".join(rank_errors))
    tokenizer = llm.get_tokenizer()
    for request in requests:
        encoded = tokenizer.encode(request.prompt_text, add_special_tokens=True)
        if list(encoded) != list(request.prompt_token_ids):
            raise RuntimeError(
                f"vLLM tokenizer disagrees with frozen prompt tokens for {request.request_id}"
            )
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
    runs = []
    run_wall_ms = []
    for _ in range(1 if diagnostic_single_run else 2):
        run_started = time.monotonic_ns()
        outputs = llm.generate(prompts, parameters, use_tqdm=False)
        torch.cuda.synchronize()
        run_finished = time.monotonic_ns()
        run_wall_ms.append((run_finished - run_started) / 1_000_000)
        runs.append(_serialize_outputs(outputs, requests))
    deterministic = (
        all(
            first["generated_token_ids"] == second["generated_token_ids"]
            and first["text"] == second["text"]
            and first["finish_reason"] == second["finish_reason"]
            and first["stop_reason"] == second["stop_reason"]
            for first, second in zip(runs[0], runs[1])
        )
        if len(runs) == 2
        else None
    )
    numerical_rows = (
        CheckpointJsonl(numerical_output_path).read()
        if numerical_output_path is not None
        else []
    )
    numerical_errors = (
        validate_numerical_records(
            numerical_rows, numerical_plan, execution_mode="stock-style"
        )
        if numerical_plan is not None
        else []
    )
    if numerical_errors:
        raise RuntimeError(
            "invalid stock-style numerical diagnostics: " + "; ".join(numerical_errors)
        )
    attention_backends = sorted(
        {str(backend) for row in worker_ranks for backend in row.get("attention_backends", ())}
    )
    manifest = build_runtime_manifest(
        config,
        role=role,
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
    _update_combined_manifest(runtime_manifest_path, manifest)
    finished = time.monotonic_ns()
    report = {
        "schema_version": "specrhythm.phase4-stock-smoke.v1",
        "stage": "phase4a-stock-vllm-bringup",
        "role": role,
        "correctness_mode": correctness_mode,
        **batch_validation,
        "backend": "stock-vllm-v1-offline-llm",
        "fake_data": False,
        "gpu_result": True,
        "serving_performance_result": False,
        "performance_scope": "bring-up timestamps only; not a serving performance evaluation",
        "built_in_speculative_decoding": False,
        "vllm_dbo_enabled": False,
        "specrhythm_dual_batch_implemented": False,
        "request_count": len(requests),
        "prompt_token_ids_revalidated": True,
        "workload_tokenizer_fingerprints": sorted(
            {request.tokenizer_fingerprint for request in requests}
        ),
        "worker_ranks": worker_ranks,
        "startup_ms": (startup_finished - started) / 1_000_000,
        "run_wall_ms": run_wall_ms,
        "total_wall_ms": (finished - started) / 1_000_000,
        "sampling": config.sampling.to_dict(),
        "runs": runs,
        "repeated_run_deterministic": deterministic,
        "repeated_run_performed": len(runs) == 2,
        "diagnostic_only": diagnostic_single_run,
        "reference_freeze_eligible": False if diagnostic_single_run else None,
        "stock_reference_replaced": False,
        "numerical_diagnostics": (
            {
                "enabled": True,
                "execution_mode": "stock-style",
                "record_count": len(numerical_rows),
                "valid": not numerical_errors,
                "errors": numerical_errors,
                "plan_sha256": numerical_plan["plan_sha256"],
                "workload_sha256": numerical_plan["workload_sha256"],
                "output_file": numerical_output_path.name,
            }
            if numerical_plan is not None and numerical_output_path is not None
            else {"enabled": False}
        ),
        "frozen_hf_target_comparison": (
            compare_frozen_target(runs[0], frozen_target_dir)
            if role == "target"
            else {"performed": False, "reason": "draft role has no target comparison"}
        ),
        "provenance": {
            "git_commit": git_commit,
            "config_sha256": sha256_file(config.path),
            "workload_sha256": sha256_file(workload_path),
            "runtime_manifest_file": runtime_manifest_path.name,
        },
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
    }
    return report
