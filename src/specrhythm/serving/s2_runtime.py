"""Fresh resident engines and one synchronous Target step per safe coordinator boundary."""

from __future__ import annotations

import os
import time
from pathlib import Path

from specrhythm.phase4.config import load_phase4_config
from specrhythm.phase4.decode_ready import DecodeReadyProvenance, load_decode_ready_manifest
from specrhythm.phase4.dual_uuid import worker_dual_runtime_snapshot, worker_dual_uuid_evidence
from specrhythm.phase4.manifest import sha256_file
from specrhythm.phase4.resident_runner import build_decode_ready_context
from specrhythm.phase4.resident_setup import build_setup_control
from specrhythm.phase4.serial_runner import load_patch_manifest, validate_installed_patch_stack
from specrhythm.phase4.stock_vllm import _worker_runtime_snapshot, validate_worker_ranks
from specrhythm.phase4.transport import UnixDraftClient
from specrhythm.serving.common import read_json, require
from specrhythm.serving.runtime_profile import load_s2
from specrhythm.serving.s1_workload import write_once
from specrhythm.serving.s2_clock import ServingClock
from specrhythm.serving.s2_plan import ACTIVE_LIMIT, POOL_SLOTS, QUERY_LIMIT, capacity_for
from specrhythm.serving.s2_pool import publish

CLASSES = {
    m: (
        "specrhythm.serving.s2_scheduler.S2" + c + "Scheduler",
        "specrhythm.serving.s2_proposer.S2" + c + "Proposer",
    )
    for m, c in (("target", "Target"), ("serial", "Serial"), ("pingpong", "Ping"))
}


def configure(root, manifest_path, directory, mode):
    manifest, definitions = load_s2(str(manifest_path))
    for name, key in (
        ("config.json", "config_sha256"),
        ("patch-manifest.json", "patch_manifest_sha256"),
    ):
        require(
            sha256_file(root / name) == manifest["execution"][key],
            "S2 frozen context input changed",
            artifact=str(root / name),
            expected=manifest["execution"][key],
            actual=sha256_file(root / name),
        )
    config = load_phase4_config(str(root / "config.json"))
    patch = load_patch_manifest(root / "patch-manifest.json", config)
    validate_installed_patch_stack(patch)
    context = build_decode_ready_context(
        config,
        patch_manifest=patch,
        workload_path=manifest_path.parent / manifest["workload_file"],
        git_commit=manifest["execution"]["git_commit"],
        correctness_mode="batch-invariant",
    )
    context["s2_execution_binding"] = {
        "sha256": manifest["sha256"],
        "execution": manifest["execution"],
        "creator": "specrhythm.serving.s2_runtime.configure",
    }
    DecodeReadyProvenance.from_dict(context)
    write_once(directory / "decode-ready-context.json", context)
    paths = {
        "SR_PHASE4_DECODE_READY_CONTEXT": "decode-ready-context.json",
        "SR_PHASE4_DECODE_READY_MANIFEST": "decode-ready-manifest.json",
        "SR_PHASE4_DECODE_READY_TIMING_EVENTS": "timing-events.jsonl",
        "SR_PHASE4_RESIDENT_SETUP_CONTROL": "setup-control.json",
        "SR_PHASE4_RESIDENT_SETUP_READY": "setup-ready.json",
        "SR_PHASE4_RESIDENT_ADMISSION_EVENTS": "admission-events.jsonl",
        "SR_PHASE4_RESIDENT_INITIAL_PROPOSAL_EVENTS": "initial-proposal-events.jsonl",
        "SR_PHASE4B2_INITIAL_PROPOSALS_READY": "initial-proposals-ready.json",
        "SR_PHASE4_TARGET_DIAGNOSTICS": "target-diagnostics.jsonl",
        "SR_PHASE4_PLUGIN_REPORT": "plugin-report.json",
        "SR_PHASE4_DUAL_PLUGIN_REPORT": "plugin-report.json",
        "SR_PHASE4_ROUND_EVENTS": "round-events.jsonl",
        "SR_PHASE4_TRANSPORT_EVENTS": "transport-events.jsonl",
        "SR_PHASE4_DUAL_SCHEDULER_EVENTS": "scheduler-events.jsonl",
        "SR_PHASE4_REQUEST_STATE_EVENTS": "request-state-events.jsonl",
        "SR_PHASE4_PROPOSAL_EVENTS": "proposal-events.jsonl",
        "SR_PHASE4_VERIFICATION_EVENTS": "verification-events.jsonl",
        "SR_PHASE4_PROPOSAL_LIFECYCLE_EVENTS": "proposal-lifecycle-events.jsonl",
    }
    os.environ.update({key: str(directory / name) for key, name in paths.items()})
    os.environ.update(
        SR_PHASE4_WORKLOAD=str(manifest_path.parent / manifest["workload_file"]),
        SR_PHASE4_REQUEST_COUNT=str(len(definitions)),
        SR_PHASE4B2_PERFORMANCE="1",
        SR_PHASE4_DRAFT_SOCKET=os.environ["SR_S2_DRAFT_SOCKET"],
        SR_PHASE4_DUAL_DRAFT_SOCKET=os.environ["SR_S2_DRAFT_SOCKET"],
        SR_PHASE4_RESIDENT_SETUP="1",
        SR_PHASE4_DECODE_READY_MODE="1",
        SR_PHASE4_RESIDENT_CONSUMER="serial" if mode == "serial" else "target-only",
        SR_PHASE4_DUAL_BATCH="1" if mode == "pingpong" else "0",
        SR_PHASE4_DUAL_RESIDENT="1" if mode == "pingpong" else "0",
        SR_PHASE4_DUAL_MICROBATCH_SIZE=str(ACTIVE_LIMIT),
        SR_PHASE4_DUAL_TEST_COORDINATION="none",
    )
    consumer = {"target": "target-only", "serial": "serial", "pingpong": "dual-batch"}[mode]
    write_once(
        directory / "setup-control.json",
        build_setup_control(
            consumer=consumer,
            expected_request_ids=[r.request_id for r in definitions],
            setup_start_ns=time.monotonic_ns(),
        ),
    )
    return config, manifest, definitions


def initialize_pingpong_worker(worker):
    """One startup RPC per TP worker, before any prefill or verification.

    The inherited Dual verification hook requires a query bound to the actual
    worker. Its existing initializer performs the authoritative device snapshot
    and rejects duplicate initialization. Later snapshots only read its evidence.
    """
    return _target_capacity_snapshot(worker, worker_dual_runtime_snapshot(worker))


def target_snapshot(worker):
    """Read current device/capacity evidence without creating or resetting queries."""
    return _target_capacity_snapshot(worker, _worker_runtime_snapshot(worker))


def _target_capacity_snapshot(worker, value):
    import torch

    from specrhythm.serving.s2_draft import cuda_memory

    require(
        len(worker.model_runner.kv_cache_config.kv_cache_groups) == 1,
        "S2 capacity formula requires the frozen dense single-group KV layout",
    )
    capacity = value["s1_effective_capacity"]
    value["s2_capacity"] = {
        **cuda_memory(torch),
        "role": "target",
        "mode": os.environ["SR_S2_MODE"],
        "physical_gpu_id": value["physical_gpu_id"],
        "gpu_uuid": value["gpu_uuid"],
        "block_size": capacity["block_size"],
        "num_gpu_blocks": capacity["num_gpu_blocks"],
        "vocab_size": worker.vllm_config.model_config.get_vocab_size(),
    }
    if os.environ["SR_S2_MODE"] == "pingpong":
        # Capacity probes legitimately have zero verification accesses. These are
        # raw lifetime counters, not the historical nonempty UUID A/B experiment gate.
        value["dual_uuid_query"] = worker_dual_uuid_evidence(worker)
    return value


def target_fence(worker):
    import torch

    torch.cuda.synchronize(worker.device)
    return {"rank": worker.rank, "timestamp_ns": time.monotonic_ns()}


def make_engine(config, mode):
    # Must be configured before importing vLLM. Its in-process client performs exactly
    # one EngineCore step per get_output; there is no autonomous scheduler busy loop.
    require(
        os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0",
        "S2 requires synchronous EngineCore",
    )
    from vllm import LLM

    scheduler, proposer = CLASSES[mode]
    return LLM(
        model=str(config.target.resolved_model_path),
        tokenizer=str(config.target.resolved_tokenizer_path),
        tensor_parallel_size=2,
        dtype=config.target.dtype,
        revision=config.target.revision,
        tokenizer_revision=config.target.tokenizer_revision,
        trust_remote_code=config.target.trust_remote_code,
        seed=config.sampling.seed,
        gpu_memory_utilization=config.target.gpu_memory_utilization,
        max_model_len=4096,
        max_num_seqs=POOL_SLOTS,
        max_num_batched_tokens=QUERY_LIMIT,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_dbo=False,
        async_scheduling=False,
        scheduler_cls=scheduler,
        speculative_config={
            "model": proposer,
            "method": "custom_class",
            "num_speculative_tokens": 4,
        },
        disable_log_stats=False,
    )


def client_for(mode):
    if mode == "pingpong":
        from specrhythm.phase4.dual_service import DualDraftClient

        return DualDraftClient(Path(os.environ["SR_S2_DRAFT_SOCKET"]), timeout_seconds=900)
    return UnixDraftClient(Path(os.environ["SR_S2_DRAFT_SOCKET"]), timeout_seconds=900)


def initial_work(mode, admitted, clock, warm, client, packet):
    definitions = clock.definitions
    if mode == "target":
        return
    if mode == "serial":
        rows = [
            {
                "request_id": rid,
                "round_id": 0,
                "committed_prefix_len": warm[rid].logical_committed_prefix_count,
                "committed_prefix_hash": warm[rid].logical_committed_prefix_sha256,
                "remaining_output_budget": definitions[rid].maximum_new_tokens - 1,
                "eos_token_ids": packet["eos_token_ids"],
            }
            for rid in admitted
            if definitions[rid].maximum_new_tokens > 2
        ]
        if rows:
            reply = client.call(
                "synchronize_and_batch_propose", {"synchronizations": [], "proposals": rows}
            )
            require(
                {r["request_id"] for r in reply["proposals"]} == {r["request_id"] for r in rows},
                "initial S2 Serial proposal request set mismatch",
            )
            for proposal in reply["proposals"]:
                packet["initial_proposals"][proposal["request_id"]] = {
                    "proposal": proposal,
                    "service_send_ns": reply["service_send_ns"],
                    "transport_end_ns": reply["transport_end_ns"],
                }
    else:
        for cohort in ("A", "B"):
            ids = [rid for rid in admitted if clock.rows[rid]["cohort"] == cohort]
            if ids:
                rows = [
                    {
                        "request_id": rid,
                        "prefix_version": warm[rid].prefix_version,
                        "committed_token_ids": list(warm[rid].logical_committed_prefix_token_ids),
                        "prefix_token_sha256": warm[rid].logical_committed_prefix_sha256,
                        "remaining_output_budget": definitions[rid].maximum_new_tokens - 1,
                        "eos_token_ids": packet["eos_token_ids"],
                        "terminal": False,
                        "measurement_start_ns": clock.barrier_ns,
                        "logical_cohort": cohort,
                    }
                    for rid in ids
                ]
                now = time.monotonic_ns()
                for rid in ids:
                    packet["initial_enqueues"][rid] = {"enqueue_start_ns": now}
                client.call("enqueue", {"work_operation": "propose_only", "rows": rows})


def drive(
    llm,
    definitions,
    trace,
    directory,
    mode,
    eos,
    *,
    active_limit=ACTIVE_LIMIT,
    timeout=14400,
    logprobs=5,
):
    from vllm import SamplingParams
    from vllm.v1.engine.core_client import InprocClient

    engine = llm.llm_engine
    require(
        isinstance(engine.engine_core, InprocClient), "S2 engine has an autonomous scheduling loop"
    )
    scheduler = engine.engine_core.engine_core.scheduler
    client = client_for(mode)
    started = time.monotonic_ns()
    setup_outputs = {}
    for row in definitions:
        tokenizer = llm.get_tokenizer()
        require(
            tokenizer.encode(row.prompt_text, add_special_tokens=False)
            == list(row.prompt_token_ids),
            "S2 tokenizer differs from frozen S0 prompt",
            request_id=row.request_id,
        )
        params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=row.maximum_new_tokens,
            seed=row.sampling_seed,
            n=1,
            logprobs=logprobs,
        )
        params.update_from_generation_config(
            engine.vllm_config.model_config.try_get_generation_config(), tokenizer.eos_token_id
        )
        params.update_from_tokenizer(tokenizer)
        require(
            sorted(set([params.eos_token_id] + list(params.stop_token_ids or ()))) == eos,
            "S2 effective EOS differs from frozen policy",
        )
        engine.add_request(
            row.request_id,
            {"prompt_token_ids": list(row.prompt_token_ids)},
            params,
            prompt_text=row.prompt_text,
        )
    deadline = time.monotonic() + timeout
    while not (directory / "setup-ready.json").exists() or len(setup_outputs) < len(definitions):
        require(time.monotonic() < deadline, "S2 setup timeout")
        for output in engine.step():
            require(
                len(output.outputs) == 1 and len(output.outputs[0].token_ids) == 1,
                "S2 setup advanced beyond one bootstrap",
                request_id=output.request_id,
            )
            setup_outputs[output.request_id] = output
    manifest = load_decode_ready_manifest(read_json(directory / "decode-ready-manifest.json"))
    warm = {r.request_id: r for r in manifest.requests}
    require(set(warm) == {r.request_id for r in definitions}, "S2 resident manifest incomplete")
    bootstrap = {}
    for r in definitions:
        w = warm[r.request_id]
        output = setup_outputs[r.request_id]
        token = output.outputs[0].token_ids[0]
        require(
            w.bootstrap_token_id == token
            and w.target_materialized_kv_token_count == r.prompt_length
            and w.draft_materialized_kv_token_count == r.prompt_length + 1
            and w.target_pending_input_token_id == token
            and w.logical_committed_prefix_token_ids == r.prompt_token_ids + (token,),
            "S2 resident bootstrap/KV/next-input contract differs",
            request_id=r.request_id,
        )
        terminal = token in eos or r.maximum_new_tokens == 1
        require(output.finished == terminal, "S2 bootstrap terminal status differs")
        bootstrap[r.request_id] = {
            "token": token,
            "terminal": terminal,
            "finish_reason": output.outputs[0].finish_reason,
        }
    # Finished-in-setup Target/Serial provider requests may still hold Draft prefixes.
    if mode != "pingpong":
        for rid, b in bootstrap.items():
            if b["terminal"]:
                client.call("finish_request", {"request_id": rid})
    pool = scheduler.freeze_pool()
    draft_pool = read_json(directory / "draft-pool.json")
    active_ids = {r for r, b in bootstrap.items() if not b["terminal"]}
    require(
        set(pool["initial"]) == set(draft_pool["rows"]) == active_ids,
        "S2 initial physical resident request set incomplete",
    )
    initial_ranks = llm.collective_rpc(target_snapshot)
    llm.collective_rpc(target_fence)
    setup_end = time.monotonic_ns()
    write_once(
        directory / "resident-pool.json",
        {
            "target": pool,
            "target_rank_initial_memory": initial_ranks,
            "draft": draft_pool,
            "prefill_setup_ns": setup_end - started,
            "selected_requests": len(definitions),
            "resident_requests": len(active_ids),
            "bootstrap_terminal_requests": len(definitions) - len(active_ids),
            "target_bootstrap_materialized": False,
            "draft_bootstrap_materialized": True,
            "restore_method": "fresh process/engine and prefill; no continuation reuse",
        },
    )
    clock = ServingClock(definitions, trace, bootstrap, active_limit=active_limit, mode=mode)
    packet = {"initial_proposals": {}, "initial_enqueues": {}, "eos_token_ids": eos}
    clock.start(time.monotonic_ns())
    last_control = None
    initial_records = []

    def publish_control():
        nonlocal last_control
        current = {**clock.control(), **packet}
        # No control-plane change is needed for an ordinary continuing decode step.
        if current != last_control:
            publish(directory / "s2-control.json", current)
            # Initial metadata dictionaries mutate at admission; keep an independent snapshot.
            import copy

            last_control = copy.deepcopy(current)

    failure = None
    try:
        while not clock.complete:
            require(
                time.monotonic() < deadline, "S2 observation timeout; unfinished requests retained"
            )
            status = client.call("status", {}) if mode == "pingpong" else {}
            require(
                not status.get("failures"),
                "S2 asynchronous Draft execution failed",
                actual=status.get("failures"),
            )
            inflight = set(status.get("inflight_request_ids", ()))
            busy = {clock.rows[r]["cohort"] for r in inflight}
            # Keep finished-but-syncing requests charged against the common active limit.
            release_finished(clock, inflight)
            with clock.lock:
                admitted = clock.admit(time.monotonic_ns(), busy_cohorts=busy)
            publish_control()
            if admitted:
                initial_work(mode, admitted, clock, warm, client, packet)
                initial_records.extend(
                    {
                        "request_id": rid,
                        "admission_ns": clock.rows[rid]["admission_ns"],
                        "serial_initial": packet["initial_proposals"].get(rid),
                        "dual_initial": packet["initial_enqueues"].get(rid),
                    }
                    for rid in admitted
                )
                publish_control()
            if any(r["state"] == "ACTIVE" for r in clock.rows.values()):
                outputs = engine.step()
                now = time.monotonic_ns()
                for output in outputs:
                    rid = output.request_id
                    require(len(output.outputs) == 1, "S2 unexpected output sequence count")
                    tokens = list(output.outputs[0].token_ids)
                    previous = clock.rows[rid]["generated_token_ids"]
                    require(
                        tokens[: len(previous)] == previous,
                        "S2 committed output prefix changed",
                        request_id=rid,
                    )
                    delta = tokens[len(previous) :]
                    if delta:
                        packet["initial_proposals"].pop(rid, None)
                        packet["initial_enqueues"].pop(rid, None)
                        clock.commit(
                            rid,
                            delta,
                            now,
                            finished=output.finished,
                            finish_reason=output.outputs[0].finish_reason,
                        )
                        if output.finished and mode != "pingpong":
                            client.call("finish_request", {"request_id": rid})
                            clock.released([rid], time.monotonic_ns())
                    else:
                        require(not output.finished, "S2 terminal output lacks a commit")
                if not outputs:
                    time.sleep(0.0005)
            else:
                time.sleep(0.001)
        require(not engine.has_unfinished_requests(), "S2 completed ledger but Target still live")
        sync = llm.collective_rpc(target_fence)
        ended = time.monotonic_ns()
        clock._event("drain-complete", ended)
        publish_control()
        return {
            "start_ns": clock.barrier_ns,
            "end_ns": ended,
            "requests": list(clock.rows.values()),
            "events": clock.events,
            "target_steps": scheduler.s2_steps,
            "target_pool_final": scheduler.s2_pool.report(),
            "target_final_sync": sync,
        }
    except Exception as error:
        failure = error
        raise
    finally:
        clock.close(failure=failure)
        write_once(
            directory / "arrival-output-events.json",
            {
                "events": clock.events,
                "requests": list(clock.rows.values()),
                "failure": str(failure) if failure else None,
                "initial_admissions": initial_records,
            },
        )


def release_finished(clock, inflight):
    """Retain the existing active-slot charge until the owner has finished all work."""
    finished = [
        rid for rid, row in clock.rows.items()
        if row["state"] == "FINISHED" and not row["resources_released"] and rid not in inflight
    ]
    clock.released(finished, time.monotonic_ns())


def run(root, manifest_path, directory, mode, *, probe=False):
    config, manifest, definitions = configure(root, manifest_path, directory, mode)
    llm = None
    client = client_for(mode)
    failure = None
    try:
        llm = make_engine(config, mode)
        startup_snapshot = initialize_pingpong_worker if mode == "pingpong" else target_snapshot
        ranks = llm.collective_rpc(startup_snapshot)
        require(
            not validate_worker_ranks(ranks, config.target), "S2 Target TP/device binding invalid"
        )
        require(
            all(r.get("batch_invariant_effective") is True for r in ranks),
            "S2 Target batch-invariant mode not effective",
        )
        cfg = llm.llm_engine.vllm_config
        require(
            not cfg.scheduler_config.async_scheduling
            and cfg.scheduler_config.max_num_seqs == POOL_SLOTS
            and not cfg.cache_config.enable_prefix_caching
            and config.proposal_budget == 4,
            "S2 effective resource/numerical configuration changed",
        )
        draft = read_json(directory / "draft-startup.json")
        capacity = [r["s2_capacity"] for r in ranks] + [draft["s2_capacity"]]
        write_once(
            directory / "actual-capacity.json", {"ranks": capacity, "target_worker_ranks": ranks}
        )
        if probe:
            result = {"probe": True, "ranks": capacity}
        else:
            checks = [
                capacity_for(definitions, rank, active_limit=manifest["active_limit"])
                for rank in capacity
            ]
            require(
                all(r["valid"] for r in checks),
                "S2 fresh engine resident capacity insufficient",
                actual=checks,
            )
            write_once(directory / "capacity-confirmation.json", {"valid": True, "checks": checks})
            result = drive(
                llm,
                definitions,
                manifest["trace"],
                directory,
                mode,
                manifest["execution"]["eos_token_ids"],
                active_limit=manifest["active_limit"],
                logprobs=config.logprobs,
            )
        result["draft_shutdown"] = client.call("shutdown", {})
        result["target_final_memory"] = llm.collective_rpc(target_snapshot)
        write_once(directory / "runtime.json", result)
        return result
    except BaseException as error:
        failure = error
        raise
    finally:
        if llm is not None:
            try:
                llm.llm_engine.engine_core.shutdown()
            except Exception as cleanup_error:
                if failure is None:
                    raise
                write_once(
                    directory / "target-cleanup-secondary.json",
                    {"error": str(cleanup_error), "primary_error": str(failure)},
                )
