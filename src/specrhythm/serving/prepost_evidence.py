"""Offline pre/post mechanism checks; preserve failed evidence as a separate layer."""

from __future__ import annotations

from collections import Counter

from specrhythm.continuation.prepost import PARAMETERS, PROTOCOL, PURPOSES
from specrhythm.serving.eager_latency import trace_rows
from specrhythm.serving.fixed_results import stats


def analyze(runtime, backend):
    start, end = runtime["measurement_start_ns"], runtime["measurement_end_ns"]
    protocol, physical = backend.get("prepost", {}), backend.get("prepost_physical", {})
    errors, cycles = [], []
    if protocol.get("protocol") != PROTOCOL or physical.get("parameters") != PARAMETERS:
        errors.append("protocol/physical parameter binding missing")
    retentions = {
        "owner": protocol.get("retention"),
        "cycles": protocol.get("cycle_retention"),
        "draft_forwards": physical.get("retention"),
    }
    retentions.update(
        {
            "target-rank-" + str(d["device"]["identity"]["global_rank"]): {
                k: v for k, v in d.get("prepost_samples", {}).items() if k != "rows"
            }
            for d in runtime["target_devices"]
        }
    )
    for role, retention in retentions.items():
        if not retention or retention.get("layout") != "phased":
            errors.append(role + " phased evidence missing")
        elif (
            trace_rows({"causal_timeline": retention}, start, end)["measurement_status"]
            != "COMPLETE"
        ):
            errors.append(role + " measurement evidence missing or truncated")
    if protocol.get("pending_work") or not protocol.get("owner_stopped"):
        errors.append("owner work not retired")
    native = backend["fixed_device"]["forwards"]
    host = physical.get("forwards", [])
    settlements = protocol.get("cycles", [])
    eager = runtime["point"]["mode"] == "serial-eager-prepost3"
    target_samples = [
        r
        for d in runtime["target_devices"]
        if d["device"]["identity"]["global_rank"] == 0
        for r in d.get("prepost_samples", {}).get("rows", [])
    ]
    for index, step in enumerate(runtime["target_steps"]):
        if not step.get("window"):
            continue
        a, b = step["start_ns"], step["end_ns"]
        gpu = [f for f in native if a <= f["host_start_ns"] <= b]
        forward = [f for f in host if a <= f["start_ns"] <= b]
        parents = [r for c in settlements if a <= c["start_ns"] <= b for r in c["requests"]]
        samples = [r for r in target_samples if a <= r["timestamp_ns"] <= b]
        purposes = Counter(f.get("purpose") for f in gpu)
        lengths = [r["candidate_positions"] for r in step["rows"]]
        actual_lengths = {
            r["internal_request_id"]: r["actual_candidate_length"]
            for s in samples
            for r in s["requests"]
        }
        expected_lengths = {
            r["internal_request_id"]: r["candidate_positions"] for r in step["rows"]
        }
        if len(samples) != 1 or actual_lengths != expected_lengths:
            errors.append(f"step {index}: Target parsed/scheduled mixed length evidence differs")
        if any(r["query_positions"] != r["candidate_positions"] + 1 for r in step["rows"]):
            errors.append(f"step {index}: Target effective positions/root mismatch")
        if purposes["proposal"] or purposes["commit"] or purposes["eager"]:
            errors.append(f"step {index}: legacy Draft forward in new steady loop")
        if purposes["prepost_post"] > 1 or (eager and purposes["prepost_extension"]):
            errors.append(f"step {index}: extra common/recovery extension forwards")
        if purposes["prepost_lookahead"] > (3 if eager else 0):
            errors.append(f"step {index}: lookahead bound exceeded")
        if purposes["prepost_extension"] > (0 if eager else 3):
            errors.append(f"step {index}: Serial extension bound exceeded")
        if len(forward) != sum(purposes[p] for p in PURPOSES):
            errors.append(f"step {index}: physical host/native forward accounting differs")
        if any(any(n != 1 for n in f["materialized_positions"]) for f in forward):
            errors.append(f"step {index}: hidden multi-position materialization")
        if any(r["common_forward"] for r in parents) and purposes["prepost_post"] != 1:
            errors.append(f"step {index}: common forward not uniquely recorded")
        if len(parents) != step["B"]:
            errors.append(f"step {index}: missing parent settlement rows")
        for r in parents:
            if r["bonus"] or r["committed_tokens"] != r["accepted"] + r["correction"]:
                errors.append(f"step {index}: no-bonus committed accounting differs")
            if r["lookahead_generated"] > 3 or r["lookahead_retained"] > r["lookahead_generated"]:
                errors.append(f"step {index}: lookahead token conservation differs")
            if eager and not r["terminal"]:
                if r["accepted"] < r["parent_length"] and r["next_candidate_length"] != 1:
                    errors.append(f"step {index}: rejected request did not recover short")
                if (
                    r["lookahead_retained"] == 3
                    and not r["next_candidate_EOS"]
                    and r["remaining_output_budget"] >= 4
                    and r["next_candidate_length"] != 4
                ):
                    errors.append(
                        f"step {index}: successful three-token reuse did not extend to four"
                    )
        if sum(r["committed_tokens"] for r in parents) != step["committed_tokens"]:
            errors.append(f"step {index}: authoritative step/parent committed count differs")
        cycles.append(
            dict(
                step_index=index,
                boundary_ns=[a, b],
                long_requests=lengths.count(4),
                short_requests=lengths.count(1),
                effective_candidate_lengths=lengths,
                terminal_clipped_lengths=[n for n in lengths if n not in (1, 4)],
                target_effective_positions=sum(r["query_positions"] for r in step["rows"]),
                committed_tokens=step["committed_tokens"],
                accepted_candidates=sum(r["accepted"] for r in parents),
                actual_verified_candidates=sum(lengths),
                acceptance_rate=(
                    sum(r["accepted"] for r in parents) / sum(lengths) if sum(lengths) else None
                ),
                lookahead_generated=sum(r["lookahead_generated"] for r in parents),
                lookahead_reuse_rate=(
                    sum(r["lookahead_retained"] for r in parents)
                    / sum(r["lookahead_generated"] for r in parents)
                    if any(r["lookahead_generated"] for r in parents)
                    else None
                ),
                lookahead_discard_rate=(
                    sum(r["lookahead_generated"] - r["lookahead_retained"] for r in parents)
                    / sum(r["lookahead_generated"] for r in parents)
                    if any(r["lookahead_generated"] for r in parents)
                    else None
                ),
                forward_counts=dict(purposes),
                physical_B_by_purpose={
                    p: [f["B"] for f in gpu if f.get("purpose") == p] for p in PURPOSES
                },
                GPU_ms_by_purpose={
                    p: sum(f["gpu_event_ms"] for f in gpu if f.get("purpose") == p)
                    for p in PURPOSES
                },
                physical_positions=sum(sum(f["materialized_positions"]) for f in forward),
                physically_generated_candidates=sum(f["generated_tokens"] for f in forward),
                reused_tokens=sum(r["lookahead_retained"] for r in parents),
                discarded_tokens=sum(
                    r["lookahead_generated"] - r["lookahead_retained"] for r in parents
                ),
                recovery_requests=sum(
                    not r["terminal"] and not r["lookahead_retained"] for r in parents
                ),
                success_ready_to_step_end_ms=stats(
                    [
                        (b - r["candidate_ready_ns"]) / 1e6
                        for r in parents
                        if r["lookahead_retained"]
                    ]
                ),
                lookahead_complete_to_common_ms=stats(
                    [
                        (r["common_start_ns"] - r["lookahead_completed_ns"]) / 1e6
                        for r in parents
                        if r["lookahead_retained"] and r["common_start_ns"] is not None
                    ]
                ),
                feedback_to_ready_ms=stats(
                    [(r["candidate_ready_ns"] - r["feedback_ns"]) / 1e6 for r in parents]
                ),
                # Only compact parent identities/landmarks, no duplicated raw native/trace events.
                parents=parents,
            )
        )
    if not cycles:
        errors.append("no measurement cycles")
    mixed = sum(c["long_requests"] > 0 and c["short_requests"] > 0 for c in cycles)
    if eager and not mixed:
        errors.append("no measured mixed 1/4 Target batch; mechanism coverage incomplete")
    return dict(
        protocol=PROTOCOL,
        parameters=PARAMETERS,
        status="FAILED" if errors else "COMPLETE",
        errors=errors,
        retentions=retentions,
        cycles=cycles,
        mixed_target_steps=mixed,
        window_bounds_ns=[start, end],
        semantics="per-step physical launch sets; no sum of host/GPU/process times; "
        "terminal EOS/output budget may clip real candidates; zero repair is observed only "
        "with complete host/native accounting; extra work outside full steps remains in "
        "the parent report/window throughput",
        GPU_correctness="separate joint check",
        performance_conclusion="PENDING paired GPU evidence; no acceleration threshold",
    )
