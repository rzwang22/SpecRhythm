"""Owner-thread runnable inventory; physical selection remains one next step/request."""

from specrhythm.serving.common import require

MODES = ("legacy", "unified")


def settled_promotions(machine):
    """No new token needed. The backend still validates every dependency and fences."""
    result = []
    for row in machine._pending_rows():
        rid = row["request_id"]
        plan, _ = machine.parent_plans[rid, row["round_id"]]
        work = machine.works.get(rid)
        job = machine.core._state(rid).continuations[work.work_id] if work else None
        if (
            not plan.terminal
            and not plan.correction_count
            and job
            and job.completed
            and not job.aborted
            and job.generated
            and (len(job.generated) == work.limit or job.generated[-1] in work.eos_token_ids)
        ):
            result.append(row)
    return result


def snapshot(machine, bindings):
    """Called immediately before materialize, with authoritative owner-only state."""
    from specrhythm.continuation.trace import TRACE

    phase = TRACE.phase
    require(machine.dispatch_counts.get(phase, 0) < 4096,
            "Draft dispatch inventory budget exhausted; evidence incomplete")
    machine.dispatch_counts[phase] = machine.dispatch_counts.get(phase, 0) + 1
    selected = {b["request_id"]: b for b in bindings}
    require(len(selected) == len(bindings), "physical dispatch duplicate request")
    pending = {r["request_id"]: r for r in machine._pending_rows()}
    inventory, decorated, terminated = [], [], []
    promotions = {r["request_id"] for r in settled_promotions(machine)}
    for rid in machine.homes:
        state = machine.core._state(rid)
        work = machine.works.get(rid)
        job = state.continuations[work.work_id] if work else None
        # All GPU calls are fenced before returning to this single owner. No async
        # in-flight job is silently classified as executable by another thread.
        kind, reason, runnable = None, None, False
        if state.terminal:
            require(rid not in selected, "terminal request selected for Draft")
            terminated.append(rid)
            continue
        elif rid in pending and (job is None or job.completed):
            kind, runnable = "parent_materialization", True
        elif rid in machine.normal:
            kind, runnable = "normal_extension", True
        elif job and not job.completed and not job.aborted:
            kind, runnable = "lookahead", True
        elif rid in machine.ready:
            reason = "already_READY"
        elif rid in machine.claims:
            reason = (
                "waiting_target_feedback" if rid not in pending else "waiting_own_previous_step"
            )
        else:
            reason = "no_current_draft_work"
        parent_key = (rid, state.prefix_version - (1 if kind == "normal_extension" else 0))
        parent = machine.parent_plans.get(parent_key)
        rejected = bool(parent and parent[0].correction_count)
        source = (
            "eager_rejection"
            if rejected and machine.enabled and rid in machine.eligible
            else "rejection"
            if rejected
            else "ordinary"
        )
        if kind == "lookahead":
            source = "conditional_lookahead"
        row = dict(
            request_id=rid,
            home=machine.homes[rid],
            prefix_version=state.prefix_version,
            proposal_id=state.proposal_id,
            task_kind=kind,
            source=source,
            executable=runnable,
            selected=rid in selected,
            reason=reason if not runnable else None,
        )
        if runnable and rid not in selected:
            # Only full promotions are non-GPU settlements. Everything else must
            # be selected when it fits. Never infer a dependency from an empty batch.
            promotion = rid in promotions
            row["reason"] = "promotion_needs_no_forward" if promotion else "capacity"
            if machine.draft_dispatch == "unified" and not promotion:
                require(
                    len(selected) >= machine.backend.physical_batch_ceiling,
                    "unified dispatch omitted executable work despite free capacity",
                )
        inventory.append(row)
        if rid in selected:
            require(runnable, "dispatch selected non-executable request")
            decorated.append(
                {
                    **selected[rid],
                    "home": row["home"],
                    "source": source,
                    "task_kind": kind,
                    "prefix_version": state.prefix_version,
                    "proposal_id": state.proposal_id,
                }
            )
    require(len(inventory) <= machine.active_limit, "dispatch active inventory exceeds geometry")
    machine.dispatch_peak_live = max(machine.dispatch_peak_live, len(inventory))
    require(len(decorated) == len(bindings), "dispatch selected unknown request")
    by_id = {r["request_id"]: r for r in decorated}
    return [by_id[b["request_id"]] for b in bindings], dict(
        policy=machine.draft_dispatch,
        inventory=inventory,
        terminated=dict(reason="terminated", request_ids=terminated),
        no_other_executable_work=not any(
            r["executable"] and not r["selected"] and r["reason"] != "promotion_needs_no_forward"
            for r in inventory
        ),
        in_flight="none: single owner token-step fence before next command",
    )
