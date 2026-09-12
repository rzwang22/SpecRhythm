"""S1 coordinator wiring to the real resident consumers; imports CUDA only in child runs."""

from __future__ import annotations

import os
from pathlib import Path

from specrhythm.phase4.config import load_phase4_config
from specrhythm.serving.common import read_json, require
from specrhythm.serving.s1_policy import EFFECTIVE_SCHEMA, environment_evidence, policy_fields
from specrhythm.serving.s1_workload import (
    PROFILE_ENV,
    active_profile,
    load_execution,
    s1_enabled,
    write_once,
)


def effective_runtime(llm, worker_ranks, config):
    """Once per engine, before generate: save effective configuration and capacity."""
    if not s1_enabled():
        return
    from vllm import SamplingParams

    manifest, definitions = active_profile()
    cfg = llm.llm_engine.vllm_config
    require(
        config.proposal_budget == 4
        and cfg.model_config.max_model_len == 4096
        and cfg.model_config.enforce_eager
        and str(cfg.model_config.dtype) == "torch.bfloat16"
        and cfg.parallel_config.tensor_parallel_size == 2
        and not cfg.parallel_config.enable_dbo
        and not cfg.cache_config.enable_prefix_caching,
        "S1 Target effective numerical/resource configuration changed",
    )
    require(
        all(r.get("batch_invariant_effective") is True for r in worker_ranks),
        "S1 Target batch-invariant mode not effective",
    )
    tokenizer = llm.get_tokenizer()
    generation = cfg.model_config.try_get_generation_config()
    sampling = []
    for row in definitions:
        params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=row.maximum_new_tokens,
            seed=row.sampling_seed,
            n=1,
            logprobs=config.logprobs,
        )
        params.update_from_generation_config(generation, tokenizer.eos_token_id)
        params.update_from_tokenizer(tokenizer)
        eos = sorted(set([params.eos_token_id] + list(params.stop_token_ids or ())))
        require(eos == manifest["execution"]["eos_token_ids"], "S1 effective EOS policy differs")
        require(
            params.ignore_eos is False and params.min_tokens == 0 and not params.stop,
            "S1 effective stopping changed",
        )
        sampling.append(
            {
                "request_id": row.request_id,
                **{
                    k: getattr(params, k)
                    for k in (
                        "temperature",
                        "top_p",
                        "top_k",
                        "max_tokens",
                        "min_tokens",
                        "seed",
                        "n",
                        "ignore_eos",
                        "stop",
                        "stop_token_ids",
                        "eos_token_id",
                        "presence_penalty",
                        "frequency_penalty",
                        "repetition_penalty",
                        "logprobs",
                    )
                },
            }
        )
    capacities = [r["s1_effective_capacity"] for r in worker_ranks]
    required = resident_capacity(definitions, capacities[0]["block_size"])
    for capacity in capacities:
        require(
            capacity["max_model_len"] == 4096 and capacity["max_num_seqs"] >= len(definitions),
            "S1 effective Target context/sequence capacity insufficient",
        )
        require(
            capacity["max_num_batched_tokens"] >= required["worst_step_query_tokens"],
            "S1 effective Target token capacity insufficient",
        )
        require(
            type(capacity["num_gpu_blocks"]) is int
            and capacity["num_gpu_blocks"] >= required["required_blocks"],
            "S1 Target resident KV capacity insufficient before generation",
        )
    draft_startup = Path(os.environ["SR_S1_RUN_DIRECTORY"]) / "draft-startup.json"
    draft = read_json(draft_startup) if draft_startup.is_file() else None
    if os.environ["SR_S1_MODE"] != "raw-target":
        require(draft is not None, "S1 mandatory Draft provider startup evidence missing")
        require(
            draft["tensor_parallel_size"] == 1
            and draft["physical_gpu_id"] == 0
            and draft["dtype"] == "bfloat16"
            and draft["enforce_eager"] is True
            and draft["runner_class"] == "vllm.v1.worker.gpu_model_runner.GPUModelRunner"
            and draft["cache_initialized"] is True,
            "S1 Draft effective backend/resource identity changed",
        )
        draft_needed = resident_capacity(definitions, draft["block_size"])
        require(
            draft["kv_cache_num_blocks"] >= draft_needed["required_blocks"],
            "S1 Draft resident KV capacity insufficient before generation",
        )
        require(
            draft["max_num_seqs"] >= len(definitions)
            and draft["max_num_batched_tokens"] >= len(definitions) * 5,
            "S1 Draft sequence/token capacity insufficient",
        )
    write_once(
        Path(os.environ["SR_S1_RUN_DIRECTORY"]) / "s1-effective-runtime.json",
        {
            "schema_version": EFFECTIVE_SCHEMA,
            **policy_fields(),
            "launch_environment": environment_evidence(),
            "sampling": sampling,
            "target_worker_ranks": worker_ranks,
            "target_required_capacity": required,
            "target_effective_config": {
                "dtype": str(cfg.model_config.dtype),
                "max_model_len": cfg.model_config.max_model_len,
                "tensor_parallel_size": cfg.parallel_config.tensor_parallel_size,
                "enforce_eager": cfg.model_config.enforce_eager,
                "async_scheduling": cfg.scheduler_config.async_scheduling,
                "enable_prefix_caching": cfg.cache_config.enable_prefix_caching,
                "speculative_tokens": getattr(cfg.speculative_config, "num_speculative_tokens", 0),
            },
            "draft_startup": draft,
            "execution_sha256": manifest["manifest_sha256"],
            "measurement_configuration": "existing Phase4B2 counters/diagnostics; no logits probe",
            "reference_continuation_visible_to_draft": False,
            "bootstrap_source": "ordinary Target prefill sampling in this fresh resident engine",
            "matched_bootstrap_injection": False,
        },
    )


def resident_capacity(rows, block_size):
    require(type(block_size) is int and block_size > 0, "invalid physical KV block size")
    lengths = [r.prompt_length + r.maximum_new_tokens + 4 for r in rows]
    return {
        "request_count": len(rows),
        "speculative_reserve_tokens_per_request": 4,
        "maximum_logical_tokens": sum(lengths),
        "block_size": block_size,
        "required_blocks": sum((n + block_size - 1) // block_size for n in lengths),
        "worst_step_query_tokens": len(rows) * 5,
    }


def qualify_raw_target(result, definitions, eos_ids):
    """Optional raw observations retain per-run validity, without a determinism dependency."""
    from specrhythm.phase4.serial import token_prefix_hash

    errors = list(result.get("errors") or [])
    if "valid" in result and result["valid"] is not True:
        errors.append("raw Target reported invalid execution")
    try:
        runs = result["runs"]
        require(bool(runs), "raw Target has no completed run")
        expected = {r.request_id: r for r in definitions}
        for index, run in enumerate(runs):
            actual = {r["request_id"]: r for r in run}
            require(
                set(actual) == set(expected) and len(actual) == len(run),
                "raw Target completed request set differs",
                run=index,
            )
            for rid, row in actual.items():
                definition = expected[rid]
                tokens = row["generated_token_ids"]
                token_prefix_hash(tokens)
                require(
                    0 < len(tokens) <= definition.maximum_new_tokens
                    and not any(t in eos_ids for t in tokens[:-1])
                    and row.get("finished", True) is True,
                    "raw Target invalid token budget/EOS/completion",
                    request_id=rid,
                )
                eos = tokens[-1] in eos_ids
                require(
                    row["finish_reason"] == ("stop" if eos else "length")
                    and (eos or len(tokens) == definition.maximum_new_tokens)
                    and (row["stop_reason"] is None or (eos and row["stop_reason"] == tokens[-1])),
                    "raw Target invalid natural termination",
                    request_id=rid,
                )
                require(
                    row["prompt_length"] == definition.prompt_length
                    and row["generated_tokens"] == len(tokens)
                    and row["token_accounting"] == {
                        "prompt_tokens": definition.prompt_length,
                        "generated_tokens": len(tokens),
                        "total_tokens": definition.prompt_length + len(tokens),
                    },
                    "raw Target invalid per-run token accounting",
                    request_id=rid,
                )
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"raw Target execution evidence: {error}")
    return {**result, **policy_fields(), "valid": not errors, "errors": errors}


def prepare_serial_context(config, manifest, workload, root, context_path):
    """S1 adapter owns this file, like the legacy Serial CLI; runners/workers only read it."""
    from specrhythm.phase4.decode_ready import DecodeReadyProvenance
    from specrhythm.phase4.manifest import sha256_file
    from specrhythm.phase4.resident_runner import build_decode_ready_context
    from specrhythm.phase4.serial_runner import load_patch_manifest

    if context_path.exists():
        raise FileExistsError(f"refusing to overwrite Serial context {context_path}")
    execution = manifest["execution"]
    for name, key in (
        ("config.json", "config_sha256"),
        ("patch-manifest.json", "patch_manifest_sha256"),
    ):
        require(
            sha256_file(root / name) == execution[key],
            "frozen Serial context input changed",
            artifact=str(root / name), expected=execution[key], actual=sha256_file(root / name),
        )
    require(execution["numerical_mode"] == "batch-invariant", "S1 Serial numerical mode changed")
    context = build_decode_ready_context(
        config,
        patch_manifest=load_patch_manifest(root / "patch-manifest.json", config),
        workload_path=workload,
        git_commit=execution["git_commit"],
        correctness_mode=execution["numerical_mode"],
    )
    require(
        context["workload_sha256"] == manifest["logical"]["workload_sha256"],
        "frozen Serial context workload changed",
        artifact=str(workload),
    )
    context["s1_execution_binding"] = {
        "creator": "specrhythm.serving.s1_runtime.prepare_serial_context",
        "execution_sha256": manifest["manifest_sha256"],
        "execution_configuration": execution,
        "config_sha256": execution["config_sha256"],
        "patch_manifest_sha256": execution["patch_manifest_sha256"],
    }
    provenance = DecodeReadyProvenance.from_dict(context)
    write_once(context_path, context)
    require(
        DecodeReadyProvenance.from_dict(read_json(context_path)) == provenance,
        "Serial context readback differs before worker creation",
        artifact=str(context_path),
    )


def run_consumer(mode: str, manifest_path: Path, directory: Path, root: Path):
    """Called only in an owned GPU subprocess by the server launcher."""
    require(mode in ("raw-target", "target", "serial", "pingpong"), "unknown S1 consumer")
    manifest, definitions = load_execution(manifest_path, verify_parent=True)
    environment_evidence()
    os.environ[PROFILE_ENV] = str(manifest_path.resolve())
    os.environ["SR_S1_RUN_DIRECTORY"] = str(directory.resolve())
    os.environ["SR_S1_MODE"] = mode
    workload = manifest_path.parent / manifest["workload_file"]
    config = load_phase4_config(str(root / "config.json"))
    common = dict(
        workload_path=workload,
        request_count=len(definitions),
        environment_path=root / "environment.json",
        topology_path=root / "topology.json",
        git_commit=manifest["execution"]["git_commit"],
    )

    def path(name):
        return directory / name

    if mode == "raw-target":
        from specrhythm.phase4.serial_runner import (
            load_patch_manifest,
            validate_installed_patch_stack,
        )
        from specrhythm.phase4.stock_vllm import run_stock_smoke

        patch = validate_installed_patch_stack(
            load_patch_manifest(root / "patch-manifest.json", config)
        )
        result = run_stock_smoke(
            config,
            role="target",
            **common,
            runtime_manifest_path=path("runtime-manifest.json"),
            correctness_mode="batch-invariant",
            compare_repeated_outputs=False,
        )
        result["s1_raw_target_patch_stack"] = patch
        result["s1_reference_scope"] = "explicit stock Target observation; not an S1-P gate"
        result = qualify_raw_target(result, definitions, manifest["execution"]["eos_token_ids"])
        write_once(path("raw.json"), result)
        return result
    common.update(
        patch_manifest_path=root / "patch-manifest.json",
        draft_socket_path=Path(os.environ["SR_S1_DRAFT_SOCKET"]),
        draft_ready_path=path("draft-service-ready.json"),
        phase4b2_performance=True,
        plugin_report_path=path("plugin-report.json"),
    )
    setup = dict(
        context_path=path("decode-ready-context.json"),
        decode_ready_manifest_path=path("decode-ready-manifest.json"),
        timing_events_path=path("timing-events.jsonl"),
        setup_control_path=path("setup-control.json"),
        setup_ready_path=path("setup-ready.json"),
    )
    if mode == "target":
        from specrhythm.phase4.resident_runner import run_resident_target

        return run_resident_target(
            config,
            **common,
            **setup,
            reference_path=None,
            admission_events_path=path("admission-events.jsonl"),
            target_diagnostics_path=path("target-diagnostics.jsonl"),
            first_forward_path=path("first-target-forward.json"),
            output_path=path("raw.json"),
        )
    if mode == "serial":
        from specrhythm.phase4.serial_runner import run_serial_disaggregated

        prepare_serial_context(config, manifest, workload, root, setup["context_path"])
        result = run_serial_disaggregated(
            config,
            **common,
            reference_path=None,
            correctness_mode="batch-invariant",
            runtime_manifest_path=path("runtime-manifest.json"),
            round_events_path=path("round-events.jsonl"),
            transport_events_path=path("transport-events.jsonl"),
            diagnostics_path=path("target-diagnostics.jsonl"),
            decode_ready_context_path=setup["context_path"],
            decode_ready_manifest_path=setup["decode_ready_manifest_path"],
            decode_ready_timing_path=setup["timing_events_path"],
            first_forward_path=path("first-target-forward.json"),
            resident_setup_control_path=setup["setup_control_path"],
            resident_setup_ready_path=setup["setup_ready_path"],
            resident_admission_events_path=path("admission-events.jsonl"),
            resident_initial_proposal_events_path=path("initial-proposal-events.jsonl"),
        )
        write_once(path("raw.json"), result)
        return result
    from specrhythm.phase4.dual_runner import run_resident_dual_batch

    return run_resident_dual_batch(
        config,
        **common,
        **setup,
        **{
            key + "_path": path(name)
            for key, name in {
                "scheduler_events": "scheduler-events.jsonl",
                "request_state_events": "request-state-events.jsonl",
                "proposal_events": "proposal-events.jsonl",
                "proposal_lifecycle": "proposal-lifecycle-events.jsonl",
                "verification_events": "verification-events.jsonl",
                "draft_work_events": "draft-work-events.jsonl",
                "transport_events": "transport-events.jsonl",
                "target_diagnostics": "target-diagnostics.jsonl",
                "output_checkpoint": "output-checkpoint.jsonl",
                "cycle_events": "cycle-events.jsonl",
                "overlap_events": "overlap-events.jsonl",
                "runtime_manifest": "runtime-manifest.json",
                "output": "raw.json",
            }.items()
        },
        microbatch_size=len(definitions),
        overlap_requirement="characterization",
    )
