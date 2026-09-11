"""Real four-path short windows, fresh initial-state samples and bounded diagnostic drain."""

from __future__ import annotations

import copy
import os
import time

from specrhythm.phase4.stock_vllm import validate_worker_ranks
from specrhythm.serving.common import read_json, require
from specrhythm.serving.fixed_artifacts import checkpoint, record_error
from specrhythm.serving.fixed_drain import settle
from specrhythm.serving.fixed_identity import scheduler_report
from specrhythm.serving.fixed_observe import TIMERS, target_report, target_startup
from specrhythm.serving.fixed_plan import capacity_metadata
from specrhythm.serving.fixed_settle import remaining
from specrhythm.serving.s1_workload import write_once
from specrhythm.serving.s2_clock import ServingClock
from specrhythm.serving.s2_plan import capacity_for
from specrhythm.serving.s2_pool import publish
from specrhythm.serving.s2_runtime import (
    client_for,
    configure,
    initial_work,
    make_engine,
    prepare_resident,
    release_finished,
    target_fence,
)

CLASSES = {
    m: (
        f"specrhythm.serving.fixed_scheduler.Fixed{c}Scheduler",
        f"specrhythm.serving.s2_proposer.S2{c}Proposer",
    )
    for m, c in (("target", "Target"), ("serial", "Serial"), ("pingpong", "Ping"))
}


def population(clock, inflight=()):
    held = [
        r
        for r in clock.rows.values()
        if r.get("admission_ns") is not None and not r["resources_released"]
    ]
    return {
        "runnable": sum(r["state"] == "ACTIVE" and r["request_id"] not in inflight for r in held),
        "draining": sum(r["state"] == "FINISHED" for r in held),
        "active_requests": sum(r["state"] == "ACTIVE" for r in held),
        "draft_inflight_requests": sum(r["request_id"] in inflight for r in held),
        "held_slots": len(held),
        "queued": len(clock.queue),
        "cohort_held": {c: sum(r["cohort"] == c for r in held) for c in ("A", "B")},
    }


def wait_draft(client, timeout, *, sleep=time.sleep, deadline_ns=None):
    """Read actual owner completion. Sleeping only polls; it never simulates work."""
    deadline_ns = deadline_ns or time.monotonic_ns() + int(timeout * 1e9)
    with TIMERS.span("wait_draft_owner"):
        while True:
            client.timeout_seconds = remaining(deadline_ns)
            status = client.call("status", {})
            require(
                not status.get("failures"),
                "diagnostic asynchronous Draft failed",
                actual=status.get("failures"),
            )
            if not status["inflight_request_ids"]:
                return status
            require(
                time.monotonic_ns() < deadline_ns,
                "diagnostic Draft drain timeout",
                actual=status["inflight_request_ids"],
            )
            sleep(0.001)


class Window:
    def __init__(self, options, *, initial_state=False):
        self.options, self.initial_state = options, initial_state
        self.start_ns = self.end_ns = None
        self.warmup_start_ns = time.monotonic_ns()
        self.warmup_end_ns = None
        self.warmup_steps = self.samples = 0
        self.reason = None

    def ready(self, now):
        if self.start_ns is None and (
            self.initial_state or self.warmup_steps >= self.options["warmup_steps"]
        ):
            self.warmup_end_ns = self.start_ns = now
        return self.start_ns is not None

    def time_expired(self, now):
        if (
            self.start_ns is not None
            and now - self.start_ns >= self.options["window_seconds"] * 1e9
        ):
            self.reason = "time_budget"
            return True
        return False

    def step_completed(self, actual_batch, now):
        if actual_batch:
            if self.start_ns is None:
                self.warmup_steps += 1
            else:
                self.samples += 1
        if self.start_ns is not None:
            limit = 1 if self.initial_state else self.options["samples"]
            if self.samples >= limit:
                self.reason = "sample_budget"
            elif now - self.start_ns >= self.options["window_seconds"] * 1e9:
                self.reason = "time_budget"
        return self.reason is not None


def commit_outputs(clock, outputs, packet, client, runtime_mode):
    for output in outputs:
        rid = output.request_id
        require(len(output.outputs) == 1, "diagnostic unexpected output sequence count")
        tokens = list(output.outputs[0].token_ids)
        previous = clock.rows[rid]["generated_token_ids"]
        require(
            tokens[: len(previous)] == previous,
            "diagnostic committed prefix changed",
            request_id=rid,
        )
        delta = tokens[len(previous) :]
        if delta:
            packet["initial_proposals"].pop(rid, None)
            packet["initial_enqueues"].pop(rid, None)
            with TIMERS.span("commit", request_id=rid):
                clock.commit(
                    rid,
                    delta,
                    time.monotonic_ns(),
                    finished=output.finished,
                    finish_reason=output.outputs[0].finish_reason,
                )
            if output.finished and runtime_mode != "pingpong":
                client.call("finish_request", {"request_id": rid})
                clock.released([rid], time.monotonic_ns())
        else:
            require(not output.finished, "diagnostic terminal output lacks commit")


def drive(llm, manifest, definitions, directory, point, options, *, logprobs=5, probe=False):
    mode, runtime_mode = point["mode"], point["runtime_mode"]
    scan = point.get("scan", False)
    setup_timeout = options["setup_timeout"]
    if scan and os.environ.get("SR_FIXED_SCAN_SETUP_DEADLINE_NS"):
        setup_timeout = remaining(int(os.environ["SR_FIXED_SCAN_SETUP_DEADLINE_NS"]))
    prepared = prepare_resident(
        llm,
        definitions,
        directory,
        runtime_mode,
        manifest["execution"]["eos_token_ids"],
        setup_timeout,
        logprobs,
    )
    prefill_complete_ns = time.monotonic_ns() if scan else None
    engine, scheduler, client, warm, bootstrap, _ = prepared
    diag = manifest["fixed_diagnostic"]
    initial = point["kind"] == "initial-state"
    if scan:
        require(not getattr(scheduler, "defer_block_free", False),
                "scan pre-forward stop requires synchronous allocator without deferred frees")
    active = manifest["active_limit"] if scan else (
        point["batch"] if initial and runtime_mode != "pingpong" else 64)
    grouped = runtime_mode == "pingpong"
    cohort_capacity = active // 2 if scan else 32
    maximum_batch = cohort_capacity if grouped else active
    trace = copy.deepcopy(manifest["trace"])
    if initial and point["half"] == "B":
        # Explicit isolated B-half point, retaining the source's frozen order inside each half.
        trace.pop("sha256")
        rows = trace["rows"]
        trace["rows"] = rows[32:64] + rows[:32] + rows[64:]
        from specrhythm.serving.s2_plan import sealed

        trace = sealed(trace)
    assignment = {rid: c for c, ids in diag["cohorts"].items() for rid in ids} if grouped else {}
    clock = ServingClock(
        definitions,
        trace,
        bootstrap,
        active_limit=active,
        mode=runtime_mode,
        per_cohort_capacity=cohort_capacity if grouped else None,
        fixed_assignment=assignment,
    )
    packet = {
        "initial_proposals": {},
        "initial_enqueues": {},
        "eos_token_ids": manifest["execution"]["eos_token_ids"],
    }
    clock.start(time.monotonic_ns(), threaded=False)
    if scan:
        clock.observe(clock.barrier_ns)
    window_options = dict(options)
    if grouped and not initial and not scan:
        # A requested sample unit is one 64-request rotation (two Target steps).
        # Actual partial/cohort-incomplete rotations remain separately labelled.
        window_options["samples"] *= 2
        window_options["warmup_steps"] *= 2
    if scan:
        from specrhythm.serving.decode_scan_window import ScanShapeStop, ScanWindow

        window = ScanWindow(window_options, active, grouped)
    else:
        window = Window(window_options, initial_state=initial)
    last = None
    polls, steps, phases = [], [], []
    first_admission = True
    failure = None
    complete = False
    observation_deadline = time.monotonic() + options["setup_timeout"] + options["window_seconds"]

    def publish_control(inflight=()):
        nonlocal last
        current = {
            **clock.control(),
            **packet,
            "max_requests_per_target_forward": maximum_batch,
            **({"decode_scan_full_batch": maximum_batch} if scan else {}),
            "diagnostic_phase": (
                "drain" if window.end_ns else "measurement" if window.start_ns else "warmup"
            ),
            "population": population(clock, inflight),
        }
        if current != last:
            publish(directory / "s2-control.json", current)
            last = copy.deepcopy(current)

    try:
        while not clock.complete and not probe:
            now = time.monotonic_ns()
            if scan:
                require(len(polls) < 10000, "scan polling evidence capacity exceeded")
            if scan and window.start_ns is None:
                remaining(int(os.environ.get("SR_FIXED_SCAN_SETUP_DEADLINE_NS",
                                             int(observation_deadline * 1e9))))
            else:
                require(
                    time.monotonic() < observation_deadline, "diagnostic warmup/window timeout")
            if (directory / "stop-request.json").exists():
                window.reason = "operator_stop"
                break
            if not scan:
                window.ready(now)
            if window.time_expired(now):
                break
            clock.observe(now)
            status = client.call("status", {}) if grouped else {}
            require(
                not status.get("failures"),
                "diagnostic asynchronous Draft failed",
                actual=status.get("failures"),
            )
            inflight = set(status.get("inflight_request_ids", ()))
            polls.append({"timestamp_ns": now, "inflight_request_ids": sorted(inflight)})
            release_finished(clock, inflight)
            if scan and population(clock)["active_requests"] + len(clock.queue) < active:
                window.reason = "pool_exhausted_before_window"
                break
            with TIMERS.span("refill"):
                admitted = clock.admit(
                    time.monotonic_ns(), busy_cohorts={clock.rows[r]["cohort"] for r in inflight}
                )
            publish_control(inflight)
            if admitted:
                if initial and mode == "pingpong" and first_admission:
                    # Prepare A's real proposal, then submit B's real Draft immediately
                    # before A's Target step. Actual device overlap can still be zero.
                    a = [r for r in admitted if clock.rows[r]["cohort"] == "A"]
                    b = [r for r in admitted if clock.rows[r]["cohort"] == "B"]
                    initial_work(runtime_mode, a, clock, warm, client, packet)
                    wait_draft(client, options["drain_timeout"])
                    publish_control()
                    initial_work(runtime_mode, b, clock, warm, client, packet)
                else:
                    with TIMERS.span("initial_draft"):
                        initial_work(runtime_mode, admitted, clock, warm, client, packet)
                first_admission = False
                publish_control()
            # This is the only serial-split execution difference. No Target step
            # can begin with any owner work pending, including terminal materialization.
            if mode == "serial-split":
                status = wait_draft(client, options["drain_timeout"])
            elif grouped:
                status = client.call("status", {})
            inflight = set(status.get("inflight_request_ids", ()))
            require(
                not status.get("failures"),
                "diagnostic asynchronous Draft failed",
                actual=status.get("failures"),
            )
            publish_control(inflight)
            if window.time_expired(time.monotonic_ns()):
                break
            pop = population(clock, inflight)
            if scan and window.start_ns is None:
                if window.ready(time.monotonic_ns(), population=pop):
                    window.window_state.update(
                        request_ids=[
                            rid for rid, r in clock.rows.items() if r["state"] == "ACTIVE"],
                        cohorts={c: [rid for rid, r in clock.rows.items()
                                     if r["state"] == "ACTIVE" and r["cohort"] == c]
                                 for c in ("A", "B")},
                        inflight_request_ids=sorted(inflight),
                        next_cohort=getattr(scheduler, "selected_cohort", None),
                    )
                    # Keep the normal asynchronous pipeline; no new fence or proposal.
                    deadline = window.start_ns + int(
                        (options["window_seconds"] + options["drain_timeout"]) * 1e9)
                    publish(directory / "drain-state.json", {
                        "phase": "scan_window_and_atomic_step", "status": "RUNNING",
                        "start_ns": window.start_ns, "deadline_ns": deadline,
                    })
                    publish_control(inflight)
                    checkpoint(directory, manifest, point, window, clock, steps, phases)
                elif window.warmup_rotations >= options["warmup_steps"]:
                    # Restore the full starting population without extra warmup forwards.
                    time.sleep(0.0005)
                    continue
            phase = (
                "full-load" if pop["active_requests"] == active
                else ("fill" if not steps else "tail")
            )
            phases.append({"timestamp_ns": time.monotonic_ns(), "phase": phase, **pop})
            before = len(scheduler.s2_steps)
            start = time.monotonic_ns()
            if scan and window.time_expired(start):
                break
            try:
                with TIMERS.span("target_step"):
                    outputs = engine.step()
            except Exception as error:
                if not scan or not isinstance(error, ScanShapeStop):
                    raise
                window.rejected_step = error.evidence
                window.reason = "partial_batch_prevented"
                break
            scheduled = scheduler.s2_steps[before:]
            require(len(scheduled) == 1, "diagnostic requires one scheduler step per engine step")
            row = scheduled[0]
            committed = False
            try:
                with TIMERS.span("output_commit"):
                    commit_outputs(clock, outputs, packet, client, runtime_mode)
                committed = True
            finally:
                end = time.monotonic_ns()
                # Preserve a completed engine step even if output/drain RPC fails.
                steps.append(
                    {
                        **row,
                        "start_ns": start,
                        "end_ns": end,
                        "window": window.start_ns is not None,
                        "population": pop,
                        "supply_phase": phase,
                        "output_commit_complete": committed,
                    }
                )
            stopping = window.step_completed(
                steps[-1] if scan else len(row["request_ids"]), end)
            checkpoint(directory, manifest, point, window, clock, steps, phases)
            if stopping:
                break
            if not outputs:
                with TIMERS.span("wait_ready", reason="no admissible proposal/Target work"):
                    time.sleep(0.0005)
        window.end_ns = time.monotonic_ns()
        window.reason = window.reason or ("capacity_probe" if probe else
                                         "pool_exhausted_before_window" if scan else
                                         "all_naturally_completed")
        # Stop submissions at this safe Target boundary. Work enqueued by the last
        # issued step is real work and remains charged through the drain below.
        clock._event("diagnostic-stop-submission", window.end_ns, reason=window.reason)
        complete = (window.reason == "time_budget" if scan else window.reason != "operator_stop"
                    ) and window.samples > 0
        checkpoint(
            directory, manifest, point, window, clock, steps, phases, measurement_complete=complete
        )
        with TIMERS.span("drain"):
            devices, draft_shutdown, final_ranks, drain = settle(
                llm,
                engine,
                scheduler,
                client,
                clock,
                warm,
                steps,
                directory,
                runtime_mode,
                options["drain_timeout"],
                publish_control,
                wait_draft,
            )
        checkpoint(
            directory,
            manifest,
            point,
            window,
            clock,
            steps,
            phases,
            measurement_complete=complete,
            drain_complete=True,
        )
        end = time.monotonic_ns()
        clock._event("diagnostic-drain-complete", end)
        publish_control()
        return {
            "schema_version": "specrhythm.fixed-runtime.v2",
            "point": point,
            "capacity": {
                **(diag["capacity"][mode] if scan else capacity_metadata(mode)),
                "resident_request_count": sum(not b["terminal"] for b in bootstrap.values()),
                "active_request_limit": active,
                "max_requests_per_target_forward": maximum_batch,
            },
            **({"decode_scan": {**window.evidence(), "prefill_complete_ns": prefill_complete_ns},
                "probe": probe} if scan else {}),
            "measurement_start_ns": window.start_ns,
            "measurement_end_ns": window.end_ns,
            "warmup_start_ns": window.warmup_start_ns,
            "warmup_end_ns": window.warmup_end_ns,
            "warmup_steps": window.warmup_steps,
            "sample_count": window.samples,
            "stop_reason": window.reason,
            "start_ns": clock.barrier_ns,
            "end_ns": end,
            "requests": list(clock.rows.values()),
            "events": clock.events,
            "prompt_lengths": {r.request_id: r.prompt_length for r in definitions},
            "target_steps": steps,
            "population": phases,
            "draft_status": polls,
            "target_devices": devices,
            "host": TIMERS.report(),
            "identity_matching": scheduler_report(scheduler),
            "target_pool_final": scheduler.s2_pool.report(),
            "target_requests_final": len(scheduler.requests),
            "draft_shutdown": draft_shutdown,
            "diagnostic_drain": drain,
            "target_final_memory": final_ranks,
            "full_request_slo_attainment": None,
            "full_request_slo_reason": "diagnostic window, not a complete serving run",
        }
    except BaseException as error:
        failure = error
        record_error(directory, error, "measurement_or_drain")
        window.end_ns = window.end_ns or time.monotonic_ns()
        window.reason = window.reason or "execution_failure"
        try:
            checkpoint(
                directory,
                manifest,
                point,
                window,
                clock,
                steps,
                phases,
                measurement_complete=complete,
            )
        except Exception as secondary:
            record_error(directory, secondary, "measurement_snapshot")
        raise
    finally:
        try:
            clock.close(failure=failure)
            write_once(
                directory / "arrival-output-events.json",
                {
                    "requests": list(clock.rows.values()),
                    "events": clock.events,
                    "failure": str(failure) if failure else None,
                },
            )
        except Exception as error:
            record_error(directory, error, "arrival_report")
            if failure is None:
                raise


def run(root, manifest_path, directory, point, *, probe=False):
    mode = point["runtime_mode"]
    config, manifest, definitions = configure(root, manifest_path, directory, mode)
    require("fixed_diagnostic" in manifest, "missing explicit fixed diagnostic manifest")
    scan = point.get("scan", False)
    capacity_spec = (manifest["fixed_diagnostic"]["capacity"][point["mode"]]
                     if scan else capacity_metadata(point["mode"]))
    active = capacity_spec["active_request_limit"]
    os.environ["SR_PHASE4_DUAL_MICROBATCH_SIZE"] = str(
        capacity_spec["max_requests_per_target_forward"] if scan else 32)
    options = manifest["fixed_diagnostic"]["options"]
    llm = None
    failure = None
    started = time.monotonic_ns()
    try:
        llm = make_engine(config, mode, classes=CLASSES,
                          sequence_limit=capacity_spec["target_sequence_limit"], query_limit=4096)
        ranks = llm.collective_rpc(target_startup)
        require(not validate_worker_ranks(ranks, config.target), "fixed Target TP/device invalid")
        require(
            all(r.get("batch_invariant_effective") is True for r in ranks),
            "fixed Target batch-invariant mode not effective",
        )
        cfg = llm.llm_engine.vllm_config
        require(
            not cfg.scheduler_config.async_scheduling
            and cfg.scheduler_config.max_num_seqs == capacity_spec["target_sequence_limit"]
            and cfg.scheduler_config.max_num_batched_tokens == 4096
            and cfg.model_config.max_model_len == 4096
            and not cfg.cache_config.enable_prefix_caching,
            "fixed effective engine limits/config differ from requested values",
        )
        draft = read_json(directory / "draft-startup.json")
        capacity = [r["s2_capacity"] for r in ranks] + [draft["s2_capacity"]]
        checks = [capacity_for(definitions, r, active_limit=active) for r in capacity]
        for check in checks:
            check["block_deficit"] = max(0, check["required_blocks"] - check["num_gpu_blocks"])
            check["workspace_deficit_bytes"] = max(
                0,
                check["additional_workspace_reserve_bytes"]
                + check["resident_draft_logits_bytes"]
                - check["free_memory_bytes_after_engine_init"],
            )

        actual = {
            "ranks": capacity,
            "target_worker_ranks": ranks,
            "checks": checks,
            "metadata": capacity_spec,
            **({"execution_sha256": manifest["sha256"], "point": point,
                "workload_sha256": manifest["workload_sha256"]} if scan else {}),
            "target_effective_by_rank": [r["s1_effective_capacity"] for r in ranks],
            "draft_effective": draft.get("fixed_engine_limits"),
        }
        write_once(directory / "actual-capacity.json", actual)
        require(
            all(r["valid"] for r in checks),
            "scan resident360 capacity insufficient" if scan else
            "fixed resident100/active64 capacity insufficient",
            artifact=str(directory / "actual-capacity.json"),
            actual=checks,
            expected=capacity_spec,
        )
        if scan:
            require(capacity_spec["resident_request_requirement"] == len(definitions) == 360
                    and point["batch"] == active, "scan pool/active capacity binding differs")
            effective = draft.get("fixed_engine_limits", {})
            maximum = capacity_spec["max_requests_per_target_forward"]
            require(effective.get("max_num_seqs", 0) >= maximum
                    and effective.get("max_num_batched_tokens", 0) >= max(5 * maximum,
                        max(r.prompt_length + 1 for r in definitions))
                    and 5 * maximum <= cfg.scheduler_config.max_num_batched_tokens,
                    "scan actual sequence/query-position capacity insufficient", actual=effective)
            # Every scan point (including capacity-only probes) prepares all360 KV.
            result = drive(llm, manifest, definitions, directory, point, options,
                           logprobs=config.logprobs, probe=probe)
        elif probe:
            probe_started = time.monotonic_ns()
            probe_deadline = probe_started + int(options["drain_timeout"] * 1e9)
            publish(
                directory / "measurement-snapshot.json",
                {
                    "schema_version": "specrhythm.fixed-measurement-snapshot.v1",
                    "mode": point["mode"],
                    "point": point,
                    "git_commit": manifest["execution"]["git_commit"],
                    "execution_sha256": manifest["sha256"],
                    "stop_reason": "capacity_probe",
                    "measurement_start_ns": None,
                    "measurement_end_ns": None,
                    "sample_count": 0,
                    "committed_window_tokens": 0,
                    "measurement_complete": False,
                    "measurement_availability": "NOT_APPLICABLE",
                    "drain_complete": False,
                    "formal_comparison_eligible": False,
                    "pending_checks": ["Draft shutdown", "owned process cleanup"],
                },
            )
            probe_drain = {
                "schema_version": "specrhythm.fixed-drain.v1",
                "start_ns": probe_started,
                "deadline_ns": probe_deadline,
                "status": "RUNNING",
                "phase": "capacity_shutdown",
                "settled_requests": 0,
                "receipts": [],
            }
            publish(directory / "drain-state.json", probe_drain)
            llm.collective_rpc(target_fence, timeout=remaining(probe_deadline))
            probe_client = client_for(mode)
            probe_client.timeout_seconds = remaining(probe_deadline)
            result = {
                "probe": True,
                "capacity": capacity_metadata(point["mode"], 0),
                "draft_shutdown": probe_client.call("shutdown", {}),
                "target_devices": llm.collective_rpc(
                    target_report, timeout=remaining(probe_deadline)
                ),
            }
            require(
                result["draft_shutdown"].get("shutdown") is True, "Draft probe shutdown incomplete"
            )
            from specrhythm.serving.fixed_logging import finalize_drain

            probe_drain["logging_finalization"] = finalize_drain(llm, directory, probe_deadline)
            remaining(probe_deadline)
            probe_drain.update(
                status="COMPLETE", phase="await_coordinator_exit", end_ns=time.monotonic_ns()
            )
            publish(directory / "drain-state.json", probe_drain)
        else:
            result = drive(
                llm, manifest, definitions, directory, point, options, logprobs=config.logprobs
            )
        if not probe or scan:
            result["startup_and_state_preparation_ms"] = (result["start_ns"] - started) / 1e6
        else:
            result["identity_matching"] = scheduler_report(
                llm.llm_engine.engine_core.engine_core.scheduler
            )
        result["capacity"].update(
            target_effective_by_rank=actual["target_effective_by_rank"],
            draft_effective=actual["draft_effective"],
            physical_KV_capacity_by_worker=capacity,
        )
        result["engine_and_execution_host_ms"] = (time.monotonic_ns() - started) / 1e6
        write_once(directory / "runtime.json", result)
        return result
    except BaseException as error:
        failure = error
        record_error(directory, error, "runtime")
        raise
    finally:
        if llm is not None:
            try:
                llm.llm_engine.engine_core.shutdown()
            except Exception as error:
                record_error(directory, error, "target_engine_cleanup")
                if failure is None:
                    raise
