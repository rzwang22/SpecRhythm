"""Cross-step joins for one authoritative PingPong owner, no additive overlap claims."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from specrhythm.continuation.prepost import PARAMETERS
from specrhythm.serving.audit_layer_report import (
    MAX_FILE,
    MAX_POINT,
    categories,
    device,
    exclusive_lanes,
    write,
)
from specrhythm.serving.common import require
from specrhythm.serving.eager_evidence_export import Reader, bounds, duration, intersect
from specrhythm.serving.eager_latency import exclusive_main, trace_rows
from specrhythm.serving.execution_evidence import delta, execution_path, qualify
from specrhythm.serving.fixed_results import stats
from specrhythm.serving.k3 import PARAMETERS as K3_PARAMETERS
from specrhythm.serving.k3 import PROTOCOL as K3_PROTOCOL
from specrhythm.serving.k3 import accounting_complete
from specrhythm.serving.ping_prepost import MODES, PROTOCOL, PURPOSES, SCHEDULED_MODES


def queue_landmarks(host, key):
    rows = host.get("causal_timeline", {}).get("rows", [])
    found = {}
    for r in rows:
        if r.get("operation") != "eager_enqueue" or not any(
            (ref.get("request_id"), ref.get("round_id")) == key for ref in r.get("requests", [])
        ):
            continue
        category = r.get("category")
        if category in ("owner_queue_submit", "owner_dequeue"):
            found.setdefault(category, []).append(r["start_ns"])
    return {k: v[0] if len(v) == 1 else None for k, v in found.items()}


def mechanism(runtime, backend):
    start, end = runtime["measurement_start_ns"], runtime["measurement_end_ns"]
    uniform = runtime["point"]["mode"].endswith("-k3")
    expected_protocol = K3_PROTOCOL if uniform else PROTOCOL
    parameters = K3_PARAMETERS if uniform else PARAMETERS
    protocol, physical = backend.get("prepost", {}), backend.get("prepost_physical", {})
    ping, errors = protocol.get("pingpong", {}), []
    if ping.get("protocol") != expected_protocol or physical.get("parameters") not in (
        parameters,
        {**parameters, "target_request_ceiling": 8},
    ):
        errors.append("protocol parameters/binding missing")
    if uniform and physical.get("parameters") == K3_PARAMETERS:
        from specrhythm.serving.k3 import configuration_of, matches_geometry

        if not matches_geometry(ping.get("geometry"), runtime["point"]["mode"],
                                configuration_of(runtime["point"])):
            errors.append("K3 actual owner execution geometry missing/different")
    if uniform:
        from specrhythm.serving.k3 import configuration_of, geometry

        limit = geometry(runtime["point"]["mode"], configuration_of(runtime["point"]))[
            "draft_physical_batch_ceiling"]
        for f in physical.get("forwards", []):
            ids = [b["request_id"] for b in f.get("bindings", [])]
            if type(f.get("B")) is not int or not (0 < f["B"] <= limit) \
                    or len(ids) != len(set(ids)) or len(ids) != f["B"]:
                errors.append("K3 physical Draft request ceiling/uniqueness mismatch")
    if uniform and not accounting_complete(protocol.get("candidate_accounting")):
        errors.append("K3 final lifetime candidate accounting missing/inconsistent")
    retentions = {
        "owner": protocol.get("retention"),
        "admissions": ping.get("retention"),
        "physical": physical.get("retention"),
    }
    for d in runtime["target_devices"]:
        retentions["target-samples-" + str(d["device"]["identity"]["global_rank"])] = {
            k: v for k, v in d.get("prepost_samples", {}).items() if k != "rows"
        }
    for role, retention in retentions.items():
        if (
            not retention
            or trace_rows({"causal_timeline": retention}, start, end)["measurement_status"]
            != "COMPLETE"
        ):
            errors.append(role + " measurement records missing/truncated")
    if (
        protocol.get("pending_work")
        or not protocol.get("owner_stopped")
        or any(ping.get(k) for k in ("claims", "ready", "pending_normal"))
    ):
        errors.append("owner did not retire all work/claims/readiness")
    events = ping.get("events", [])
    ready = {(e["request_id"], e["prefix_version"]): e for e in events if e["event"] == "ready"}
    settlements = {
        (e["request_id"], e["prefix_version"]): e for e in events if e["event"] == "settlement"
    }
    claims = {
        (c["request_id"], c["prefix_version"]): c
        for e in events
        if e["event"] == "admission"
        for c in e["claims"]
    }
    all_claims = [c for e in events if e["event"] == "admission" for c in e["claims"]]
    if len(claims) != len(all_claims):
        errors.append("same prefix/proposal claimed more than once")
    consumes = [(e["request_id"], e["prefix_version"]) for e in events if e["event"] == "consume"]
    if len(consumes) != len(set(consumes)) or any(k not in claims for k in consumes):
        errors.append("duplicate/unclaimed consumption")
    native = backend["fixed_device"].get("forwards", [])
    physical_rows = physical.get("forwards", [])
    by_key, mapped, roles = defaultdict(list), [], defaultdict(list)
    mapping_complete = True
    native_for_host = {}
    for f in physical_rows:
        matches = [
            g
            for g in native
            if g.get("purpose") == f["purpose"]
            and g["B"] == f["B"]
            and f["start_ns"] <= g["host_start_ns"] <= f["end_ns"]
        ]
        relevant = start <= f["start_ns"] <= end
        if relevant and len(matches) != 1:
            mapping_complete = False
            errors.append("physical/native forward mapping incomplete")
        if not f.get("worker_fenced") or any(n != 1 for n in f["materialized_positions"]):
            errors.append("unfenced or hidden multi-position token forward")
        if len(f.get("bindings", [])) != f["B"]:
            errors.append("physical row binding/B mismatch")
        for b in f.get("bindings", []):
            by_key[b["request_id"], b["round_id"]].append((f, b))
        if len(matches) == 1:
            mapped.append(matches[0])
            native_for_host[f["start_ns"]] = matches[0]
            for kind in {b.get("work_kind") for b in f["bindings"]}:
                roles[kind].append(matches[0])
    window_native = [g for g in native if start <= g["host_start_ns"] <= end]
    if any(g.get("purpose") not in PURPOSES for g in window_native):
        errors.append("unexpected legacy forward in PingPong measurement loop")
    if any(g not in mapped for g in window_native if g.get("purpose") in PURPOSES):
        mapping_complete = False
        errors.append("native forward lacks role mapping")
    eager = runtime["point"]["mode"] in (MODES[1], "serial-eager-k3", "pingpong-eager-k3")
    cycles = []
    targets = [g for d in runtime["target_devices"] for g in d["device"].get("forwards", [])]
    for step in runtime["target_steps"]:
        if not step.get("window") or not step["B"]:
            continue
        admission = step.get("ping_admission", {})
        step_claims = admission.get("claims", [])
        lengths = {r["request_id"]: r["candidate_positions"] for r in step["rows"]}
        if (
            len(step_claims) != step["B"]
            or len(lengths) != step["B"]
            or step["B"] > ping.get("target_batch_ceiling", 8)
            or {c["request_id"] for c in step_claims} != set(lengths)
        ):
            errors.append("Target claim/actual batch mismatch")
        parsed = [
            s
            for d in runtime["target_devices"]
            if d["device"]["identity"]["global_rank"] == 0
            for s in d.get("prepost_samples", {}).get("rows", [])
            if step["start_ns"] <= s["timestamp_ns"] <= step["end_ns"]
        ]
        expected = {r["internal_request_id"]: r["candidate_positions"] for r in step["rows"]}
        if (
            len(parsed) != 1
            or {
                r["internal_request_id"]: r["actual_candidate_length"]
                for s in parsed
                for r in s["requests"]
            }
            != expected
        ):
            errors.append("Target actual ragged sampling lengths missing/different")
        if any(r["query_positions"] != r["candidate_positions"] + 1 for r in step["rows"]):
            errors.append("Target root/candidate position conservation failed")
        tg = [g for g in targets if step["start_ns"] <= g["host_start_ns"] <= step["end_ns"]]
        target_end = max((g.get("end_upper_ns", 0) for g in tg), default=None)
        target_end_lower = max((g.get("end_lower_ns", 0) for g in tg), default=None)
        items = []
        for c in step_claims:
            key = c["request_id"], c["prefix_version"]
            p = settlements.get(key)
            f = by_key[key]
            look = [(g, b) for g, b in f if b.get("work_kind") == "lookahead"]
            common = [(g, b) for g, b in f if b.get("work_kind") in ("post", "repair")]
            extensions = [b for _, b in f if b.get("work_kind") == "normal_extension"]
            if len(look) > 3 or len(common) > 1 or (eager and extensions and not uniform):
                errors.append("per-parent lookahead/common/recovery forward bound violated")
            if p is None:
                errors.append("measured parent settlement missing (including drain)")
                continue
            if len(p["committed_tokens"]) != p["accepted"] + p["correction"]:
                errors.append("committed accepted/correction conservation failed")
            if not 0 <= p["retained"] <= p["generated"] <= 3:
                errors.append("lookahead generated/reused/discarded conservation failed")
            if eager and not uniform and not p["terminal"] and p["next_candidate_length"]:
                if p["accepted"] < p["parent_length"] and p["next_candidate_length"] != 1:
                    errors.append("rejection recovery did not publish P1")
                if p["retained"] == 3 and p["next_candidate_length"] != 4:
                    # Lookahead ending in EOS is published without tail extension.
                    following = ready.get((key[0], key[1] + 1), {})
                    if not following.get("candidate_EOS"):
                        errors.append("complete non-EOS lookahead failed to publish P4")
            nxt = ready.get((key[0], key[1] + 1))
            if uniform:
                current = ready.get(key, {})
                budget = current.get("remaining_output_budget")
                reason = current.get("short_reason")
                if type(budget) is not int or not (lengths[key[0]] == min(3, budget)
                    or (0 < lengths[key[0]] < min(3, budget) and reason == "candidate_EOS")):
                    errors.append("incomplete/non-tail K3 entered Target")
                if not eager and look:
                    errors.append("disabled eager generated conditional work")
                if p["retained"] == 3 and p["common_start_ns"] is not None:
                    errors.append("complete K3 reuse performed an unnecessary tail forward")
            definite_steps = (
                sum(g["end_ns"] <= target_end for g, _ in look) if target_end else None
            )
            native_look = [native_for_host.get(g["start_ns"], {}) for g, _ in look]
            first_gpu = min(native_look, key=lambda g: g.get("host_start_ns", 0), default={})
            queue = queue_landmarks(backend.get("fixed_host", {}), key)
            complete_counts = None
            if (
                target_end
                and target_end_lower
                and all(
                    type(g.get(k)) is int
                    for g in native_look
                    for k in ("end_lower_ns", "end_upper_ns")
                )
            ):
                complete_counts = dict(
                    lower=sum(g["end_upper_ns"] <= target_end_lower for g in native_look),
                    upper=sum(g["end_lower_ns"] <= target_end for g in native_look),
                )
            items.append(
                dict(
                    request_id=key[0],
                    prefix_version=key[1],
                    home_cohort=c["home_cohort"],
                    opportunity_cohort=c["opportunity_cohort"],
                    proposal_id=c["proposal_id"],
                    continuation_id=p["continuation_id"],
                    decision_version=c["decision_version"],
                    source="promoted" if c["promoted"] else "normal",
                    P=lengths.get(key[0]),
                    short_reason=ready.get(key, {}).get("short_reason"),
                    admission_to_Target_GPU_ms={side: delta(c["claimed_ns"],
                        min((g.get("start_" + bound + "_ns") for g in tg
                             if g.get("start_" + bound + "_ns") is not None), default=None))
                        for side, bound in (("lower", "lower"), ("upper", "upper"))},
                    parent_outcome="terminal"
                    if p["terminal"]
                    else "rejection"
                    if p["correction"]
                    else "promotion"
                    if p["retained"]
                    else "full_accept_without_reuse",
                    generated=p["generated"],
                    retained=p["retained"],
                    discarded=p["generated"] - p["retained"],
                    committed_tokens=len(p["committed_tokens"]),
                    ready_to_admission_ms=delta(c["ready_ns"], c["claimed_ns"]),
                    feedback_to_ready_ms=delta(
                        p["feedback_ns"], nxt.get("ready_ns") if nxt else None
                    ),
                    exposed_lookahead_wait_ms=max(
                        0, delta(p["feedback_ns"], p["lookahead_completed_ns"])
                    )
                    if p["lookahead_completed_ns"]
                    else None,
                    common_start_ns=p["common_start_ns"],
                    common_end_ns=p["common_end_ns"],
                    feedback_ns=p["feedback_ns"],
                    ready_ns=nxt.get("ready_ns") if nxt else None,
                    lookahead_fenced_before_Target_end_upper=definite_steps,
                    owner_queue_ms=delta(
                        queue.get("owner_queue_submit"), queue.get("owner_dequeue")
                    ),
                    enqueue_to_first_GPU_ms={
                        side: delta(
                            queue.get("owner_queue_submit"),
                            first_gpu.get("start_" + bound + "_ns"),
                        )
                        for side, bound in (("lower", "lower"), ("upper", "upper"))
                    },
                    dequeue_to_first_GPU_ms={
                        side: delta(
                            queue.get("owner_dequeue"), first_gpu.get("start_" + bound + "_ns")
                        )
                        for side, bound in (("lower", "lower"), ("upper", "upper"))
                    },
                    native_lookahead_steps_at_Target_end=complete_counts,
                    lookahead_forward_count=len(look),
                    common_forward_count=len(common),
                    next_P=(nxt or {}).get("candidate_length") if uniform
                    else p["next_candidate_length"],
                    settlement_private_candidate_count=p["next_candidate_length"],
                )
            )
        cycles.append(
            dict(
                target_batch_id=admission.get("target_batch_id"),
                B=step["B"],
                P_histogram=dict(Counter(str(v) for v in lengths.values())),
                actual_input_positions=sum(r["query_positions"] for r in step["rows"]),
                step_wall_ms=(step["end_ns"] - step["start_ns"]) / 1e6,
                requests=items,
            )
        )
    native_complete = all(
        all(
            type(g.get(k)) is int
            for k in ("start_lower_ns", "start_upper_ns", "end_lower_ns", "end_upper_ns")
        )
        for g in [*window_native, *targets]
    ) and bool(window_native and targets)
    overlap = {}
    for label, group in {
        "all": mapped,
        "ordinary_extension": roles["normal_extension"],
        "eager_lookahead": roles["lookahead"],
        "common": roles["post"],
    }.items():
        overlap[label] = (
            {
                side: duration(
                    intersect(bounds(targets, start, end, inner), bounds(group, start, end, inner))
                )
                for side, inner in (("lower_ms", True), ("upper_ms", False))
            }
            if native_complete and mapping_complete
            else None
        )
    admissions = [
        r for r in events if r["event"] == "admission" and start <= r["timestamp_ns"] <= end
    ]
    flat = [r for c in cycles for r in c["requests"]]
    generated = sum(r["generated"] for r in flat)
    from specrhythm.serving.k3_physical_dispatch_evidence import summarize as dispatch_summary

    dispatch = dispatch_summary(physical_rows, native_for_host, start, end)
    errors.extend(dispatch.get("errors", []))
    return dict(
        draft_dispatch=dispatch,
        protocol=expected_protocol,
        candidate_accounting=protocol.get("candidate_accounting") if uniform else None,
        status="INCOMPLETE" if errors else "COMPLETE",
        errors=sorted(set(errors)),
        retentions=retentions,
        cycles=cycles,
        per_role_window_forwards={
            k: dict(
                count=len([g for g in rows if start <= g["host_start_ns"] <= end]),
                B_histogram=dict(
                    Counter(str(g["B"]) for g in rows if start <= g["host_start_ns"] <= end)
                ),
                GPU_event_sum_ms=sum(
                    g["gpu_event_ms"] for g in rows if start <= g["host_start_ns"] <= end
                ),
            )
            for k, rows in roles.items()
        },
        native_overlap=overlap,
        native_overlap_status=(
            "MISSING"
            if overlap["all"] is None
            else "ZERO"
            if overlap["all"]["upper_ms"] == 0
            else "RECORDED_OVERLAP"
            if overlap["all"]["lower_ms"] > 0
            else "UNCERTAIN"
        ),
        outcomes=dict(Counter(r["parent_outcome"] for r in flat)),
        generated=generated,
        retained=sum(r["retained"] for r in flat),
        discarded=sum(r["discarded"] for r in flat),
        lookahead_rates=dict(
            denominator_generated_candidates=generated,
            reuse_fraction=sum(r["retained"] for r in flat) / generated if generated else None,
            discard_fraction=sum(r["discarded"] for r in flat) / generated if generated else None,
            parent_request_opportunities=len(flat),
            scope="measured Target parent request cycles, including their later settlement; "
            "zero generated means fractions not applicable, distinct from GPU launch window"),
        ready_to_admission_ms=stats([r["ready_to_admission_ms"] for r in flat]),
        feedback_to_ready_ms=stats(
            [r["feedback_to_ready_ms"] for r in flat if r["feedback_to_ready_ms"] is not None]
        ),
        cross_home_admissions=sum(
            c["home_cohort"] != c["opportunity_cohort"] for a in admissions for c in a["claims"]
        ),
        deferred_request_opportunities=dict(
            Counter(r["reason"] for a in admissions for r in a["deferred"])
        ),
        unfilled_opportunities=sum(a["actual_B"] < a["capacity"] for a in admissions),
        semantics=[
            "parent metrics follow measured Target parents through later steps/drain",
            "native forward metrics follow actual host launch in the window",
            "mixed GPU intervals belong to multiple roles; total uses union once",
            "host fence completion is an upper bound on GPU completion, not a device event",
            "missing endpoints are null; no throughput subtraction or speedup threshold",
        ],
    )


def analyze(runtime, backend, light, *, draft_dispatch=None):
    start, end = runtime["measurement_start_ns"], runtime["measurement_end_ns"]
    targets = runtime["target_devices"]
    require(len(targets) == 2, "Target TP ranks missing")
    require(
        len(
            {d["device"]["identity"]["gpu_uuid"] for d in targets}
            | {backend["fixed_device"]["identity"]["gpu_uuid"]}
        )
        == 3,
        "within-run Draft/Target identity isolation failed",
    )
    hosts = {"coordinator": runtime["host"], "draft": backend["fixed_host"]}
    hosts.update(
        {"target-rank-" + str(d["device"]["identity"]["global_rank"]): d["host"] for d in targets}
    )
    traces = {name: trace_rows(h, start, end) for name, h in hosts.items()}
    path = execution_path(runtime, backend, start, end, feedback_operation="pp_feedback")
    # Async settlement crosses Target steps; discard Serial-specific within-step joins.
    path = dict(
        fsync_by_file=path["fsync_by_file"],
        cycles=[
            dict(
                index=c["index"],
                boundary_ns=c["boundary_ns"],
                postprocessing=c["postprocessing"],
                latencies_ms={
                    k: v
                    for k, v in c["latencies_ms"].items()
                    if k
                    in (
                        "target_GPU_end_upper_to_sampled_hook",
                        "sampled_hook_to_payload_ready",
                        "payload_ready_to_transport",
                        "transport_to_service_receive",
                        "service_receive_to_owner_dequeue",
                    )
                },
            )
            for c in path["cycles"]
        ],
    )
    steps = [s for s in runtime["target_steps"] if s.get("window") and s["B"]]
    ping = mechanism(runtime, backend)
    dispatch_policy = draft_dispatch
    if dispatch_policy is not None:
        require(backend.get("prepost", {}).get("pingpong", {}).get("draft_dispatch")
                == dispatch_policy, "actual owner dispatch differs from manifest")
        require(ping["draft_dispatch"]["status"] == "COMPLETE",
                "configured Draft dispatch inventory missing/incomplete")
    if light["mode"].endswith("-k3"):
        from specrhythm.serving.k3_evidence import pipeline

        ping["pipeline"] = pipeline(runtime, backend)
        if ping["pipeline"]["evidence_integrity"] != "COMPLETE":
            ping["status"] = "INCOMPLETE"
            ping["errors"].extend(ping["pipeline"]["errors"])
    from specrhythm.serving.k3_validation import matching, not_run

    matching(runtime, runtime["point"], light)
    policy_fields = not_run(runtime)
    if policy_fields:
        from specrhythm.serving.k3 import configuration_of
        from specrhythm.serving.k3_acceptance import measurement

        proof = measurement(light, runtime, light["mode"], configuration_of(runtime))
        policy_fields.update(native_geometry_status="PASS", native_target_geometry=proof,
                             measurement_valid=True)
    policy_fields.pop("k3_configuration", None)  # Geometry is recorded independently below.
    from specrhythm.serving.k3_scale_report import capture_summary, diagnostic_substages

    if runtime["point"].get("k3_configuration") == "k3-b128-v1":
        require(all(d.get("diagnostic_configuration", {}).get("target_diagnostics_enabled")
                    is True for d in targets),
                "B128 effective Target logits diagnostics missing/disabled")
    calls = ping.get("draft_dispatch", {}).get("unique_physical_calls")
    from specrhythm.serving.k3_cycle_evidence import compact_report, cycle_report, factor_wait_scope

    if light["mode"].endswith("-k3"):
        ping["pipeline"] = factor_wait_scope(ping["pipeline"])

    return dict(
        cycle_accounting=compact_report(cycle_report(runtime, backend))
        if light["mode"].endswith("-k3") else None,
        capture_target_forward=capture_summary(
            {k: v for k, v in hosts.items() if k.startswith("target-rank-")}, start, end),
        target_diagnostic_substages=diagnostic_substages(
            runtime, {k: v for k, v in hosts.items() if k.startswith("target-rank-")}, start, end),
        target_diagnostic_substage_scope="children of capture, not additive with parent; "
        "metadata/mapping/encoding residual is inclusive, not all sampling or pure CPU",
        control_snapshot_publication=capture_summary(
            {"coordinator": hosts["coordinator"]}, start, end, "control_json_write"),
        physical_Draft_calls_per_Target_step=calls / len(steps) if calls is not None else None,
        effective_target_diagnostic_configuration={
            str(d["device"]["identity"]["global_rank"]): d.get("diagnostic_configuration")
            for d in targets},
        **policy_fields,
        mode=light["mode"],
        **({"k3_configuration": runtime["point"]["k3_configuration"]}
           if "k3_configuration" in runtime["point"] else {}),
        request_verification_opportunities=sum(s["B"] for s in steps),
        tokens_per_request_opportunity=light["committed_window_tokens"]
        / sum(s["B"] for s in steps),
        window_ms_per_active_opportunities=(end - start) / 1e6 * runtime["point"]["batch"]
        / sum(s["B"] for s in steps),
        measurement_start_ns=start, measurement_end_ns=end,
        execution_geometry=backend.get("prepost", {}).get("pingpong", {}).get("geometry"),
        actual_target_batch=dict(Counter(s["B"] for s in steps)),
        warmup_coverage={k: light.get("scan_warmup_boundary", {}).get(k) for k in (
            "schema_version", "request_opportunities", "opportunities_by_request",
            "unique_requests", "completed_steps")},
        draft_audit=backend["draft_audit"],
        original_qualification={
            k: light[k]
            for k in (
                "execution_status",
                "measurement_status",
                "cleanup_status",
                "formal_comparison_eligible",
            )
        },
        original_run_details={
            k: light.get(k) for k in ("capacity_status", "effective_exit_code", "errors")
        },
        steps=len(steps),
        window_ms=light["measured_window_ms"],
        committed_tokens=light["committed_window_tokens"],
        throughput_tok_s=light["decode_throughput_tok_s"],
        tokens_per_step=light["committed_window_tokens"] / len(steps),
        window_average_cadence_ms=(end - start) / 1e6 / len(steps),
        complete_step_wall_ms=stats([(s["end_ns"] - s["start_ns"]) / 1e6 for s in steps]),
        outside_complete_steps_ms=(end - start) / 1e6
        - duration([(s["start_ns"], s["end_ns"]) for s in steps]),
        target_ranks={
            str(d["device"]["identity"]["global_rank"]): device(d["device"], start, end)
            for d in targets
        },
        draft_device=device(backend["fixed_device"], start, end),
        pingpong=ping,
        diagnostic_logging=light.get("diagnostic_logging"),
        target_TP_union={
            side: duration(
                bounds([g for d in targets for g in d["device"]["forwards"]], start, end, inner)
            )
            for side, inner in (("lower_ms", True), ("upper_ms", False))
        },
        execution_path=path,
        trace_coverage={
            name: {k: v for k, v in t.items() if k != "rows"} for name, t in traces.items()
        },
        coordinator_exclusive=exclusive_main(
            hosts["coordinator"].get("intervals", []), traces["coordinator"]["rows"], start, end
        ),
        exclusive_process_threads=exclusive_lanes(
            [r for h in hosts.values() for r in h.get("intervals", [])]
            + [r for t in traces.values() for r in t["rows"]],
            start,
            end,
        ),
        inclusive_host_lanes={
            name: categories(h.get("intervals", []), start, end) for name, h in hosts.items()
        },
        recording_self_cost="NOT_ISOLATED; inclusive observed functions are not recorder overhead",
        raw_events_embedded=False,
        original_results_unchanged=True,
    )


def report(root, output, status, commit):
    reader = Reader(root, MAX_FILE, MAX_POINT, 100000)
    config = reader.read(root / "scan-config.json", required=True)
    require(config["execution"]["git_commit"] == commit, "source SHA differs")
    points = [
        p.parent
        for p in (root / "runs").glob("*/point.json")
        if not reader.read(p, required=True).get("probe")
    ]
    require(len(points) == 1, "exactly one new PingPong performance point required")
    point = points[0]
    light = reader.read(point / "light-summary.json", required=True)
    require(
        light["mode"] in SCHEDULED_MODES
        and light["git_commit"] == commit
        and light["workload_sha256"] == config["workload_sha256"],
        "point binding differs",
    )
    from specrhythm.serving.k3_validation import matching

    matching(config, light)
    runtime = reader.read(point / "runtime.json", required=True)
    from specrhythm.phase4.target_profile import qualify as qualify_target_profile

    qualify_target_profile(runtime, config["options"])
    value = analyze(
        runtime,
        reader.read(point / "draft-backend-report.json", required=True),
        light,
        draft_dispatch=config["options"].get("draft_dispatch"),
    )
    value.update(
        source_commit=commit,
        options=config["options"],
        workload_sha256=config["workload_sha256"],
        diagnostic_configuration=runtime.get("diagnostic_configuration"),
        planned_initial_request_ids=(config["selection"]["request_ids"][:light["point"]["batch"]]
                                     if "selection" in config else None),
        selected_initial_request_ids=runtime.get("decode_scan", {}).get(
            "window_initial_population", {}).get("request_ids"),
        selected_request_scope="actual active IDs at measurement start; planned IDs are "
                               "frozen selection prefix before setup EOS/refill",
        inventory=list(reader.inventory.values()),
    )
    if light["mode"].endswith("-k3"):
        value["common_execution"] = {k: config["execution"][k] for k in (
            "models", "config_sha256", "patch_manifest_sha256", "numerical_mode",
            "engine_core", "async_scheduling", "eos_token_ids", "launch_environment", "capacity")}
    write(value, output)
    qualification = qualify(value)
    from specrhythm.serving.target_dispatch import qualification_errors

    dispatch_errors = qualification_errors(value, runtime, config["options"])
    if dispatch_errors:
        qualification["errors"].extend(dispatch_errors)
        qualification["diagnostic_integrity"] = "FAILED"
        qualification["failure_layer"] = "diagnostic_evidence"
    write(qualification, status)
    return qualification


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--status", type=Path, required=True)
    p.add_argument("--commit", required=True)
    from specrhythm.serving.common import read_json
    from specrhythm.serving.k3 import CONFIGURATIONS, configuration_of
    from specrhythm.serving.k3_validation import PROFILES, profile_of

    p.add_argument("--k3-configuration", choices=CONFIGURATIONS)
    p.add_argument("--validation-profile", choices=PROFILES)
    args = p.parse_args(argv)
    config = read_json(args.root / "scan-config.json")
    require(args.k3_configuration is None or configuration_of(config) == args.k3_configuration,
            "diagnostic requested geometry differs")
    require(args.validation_profile is None or profile_of(config) == args.validation_profile,
            "diagnostic requested validation_profile differs")
    result = report(args.root, args.output, args.status, args.commit)
    print(json.dumps(result))
    require(
        result["diagnostic_integrity"] == "COMPLETE",
        "PingPong diagnostic evidence failed",
        **result,
    )


if __name__ == "__main__":
    main()
