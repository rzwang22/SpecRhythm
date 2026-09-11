"""One bounded stop transaction after the last issued diagnostic Target step."""

from __future__ import annotations

import time
from collections import Counter

from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.serving.common import require
from specrhythm.serving.fixed_artifacts import record_error
from specrhythm.serving.fixed_observe import target_report
from specrhythm.serving.fixed_settle import remaining
from specrhythm.serving.s2_pool import publish
from specrhythm.serving.s2_runtime import target_fence, target_snapshot


def settle(
    llm,
    engine,
    scheduler,
    client,
    clock,
    warm,
    steps,
    directory,
    runtime_mode,
    timeout,
    publish_control,
    wait_draft,
):
    started = time.monotonic_ns()
    deadline = started + int(timeout * 1e9)
    state = {
        "schema_version": "specrhythm.fixed-drain.v1",
        "status": "RUNNING",
        "start_ns": started,
        "deadline_ns": deadline,
        "settled_requests": 0,
        "phase": "wait_owner",
        "receipts": [],
        "cleanup_status": "PENDING",
    }

    def update(phase):
        state["phase"] = phase
        publish(directory / "drain-state.json", state)
        client.timeout_seconds = remaining(deadline)
        return client.timeout_seconds

    # The external owned-process supervisor watches this same absolute deadline,
    # including blocked worker calls and final engine shutdown after this returns.
    try:
        update("wait_owner")
        if runtime_mode == "pingpong":
            wait_draft(client, remaining(deadline), deadline_ns=deadline)
        llm.collective_rpc(target_fence, timeout=update("target_fence"))
        devices = llm.collective_rpc(target_report, timeout=update("target_evidence"))
        update("target_abort")
        cancelled = [rid for rid, r in clock.rows.items() if r["state"] != "FINISHED"]
        engine.abort_request(cancelled)
        require(not engine.has_unfinished_requests(), "Target requests remain after abort")
        require(
            not scheduler.requests and not getattr(scheduler, "deferred_frees", ()),
            "Target allocator still owns requests/deferred blocks after abort",
        )
        state["target_release_ns"] = time.monotonic_ns()
        verified = Counter(
            r["request_id"] for s in steps for r in s["rows"] if r["candidate_positions"] > 0
        )
        for rid, row in clock.rows.items():
            update("draft_settle")
            final = (*clock.definitions[rid].prompt_token_ids, *row["generated_token_ids"])
            payload = {
                "request_id": rid,
                "deadline_ns": deadline,
                "committed_prefix": list(final),
                "committed_prefix_hash": token_prefix_hash(final),
                "natural_terminal": row["state"] == "FINISHED",
                "admitted": row.get("admission_ns") is not None,
                "runtime_mode": runtime_mode,
                "next_round_id": verified[rid],
                "prefix_version": warm[rid].prefix_version + len(row["commits"]),
                "bootstrap_prefix_hash": warm[rid].logical_committed_prefix_sha256,
            }
            receipt = (
                client.call("execute", {"work_operation": "diagnostic_settle", "row": payload})
                if runtime_mode == "pingpong"
                else client.call("diagnostic_settle", payload)
            )
            require(
                receipt.get("released") is True and receipt.get("request_id") == rid,
                "Draft diagnostic release receipt missing",
                request_id=rid,
            )
            remaining(deadline)
            state["receipts"].append(receipt)
            state["settled_requests"] += 1
            # Physical Target and Draft completion both precede logical release.
            if rid in cancelled:
                row["state"] = "DIAGNOSTIC_CANCELLED"
                clock._event("diagnostic-cancelled", time.monotonic_ns(), request_id=rid)
                row["resources_released"] = True
                clock._event("cancelled-resources-released", time.monotonic_ns(), request_id=rid)
            elif not row["resources_released"]:
                clock.released([rid], time.monotonic_ns())
        publish_control()
        update("draft_shutdown")
        draft_shutdown = client.call("shutdown", {})
        require(draft_shutdown.get("shutdown") is True, "Draft shutdown incomplete")
        final = llm.collective_rpc(target_snapshot, timeout=update("target_final_evidence"))
        remaining(deadline)
        state.update(status="COMPLETE", phase="await_coordinator_exit", end_ns=time.monotonic_ns())
        state["drain_ms"] = (state["end_ns"] - started) / 1e6
        publish(directory / "drain-state.json", state)
        return devices, draft_shutdown, final, state
    except BaseException as error:
        record_error(directory, error, "drain:" + state["phase"])
        state.update(status="FAILED", error=str(error), end_ns=time.monotonic_ns())
        try:
            publish(directory / "drain-state.json", state)
        except Exception as secondary:
            record_error(directory, secondary, "drain_report")
        raise
