"""Resident wrapper/stock phase accounting from bounded production host spans."""

from specrhythm.serving.eager_evidence_export import duration
from specrhythm.serving.fixed_results import stats
from specrhythm.serving.k3 import RESIDENT_POLICY as POLICY

PHASES = ("binding", "readiness", "decisions", "stock", "initial_finish", "admission_records")


def resident_step(step, spans):
    a, b = step.get("schedule_start_ns"), step.get("schedule_end_ns")
    errors, found = [], {}
    if type(a) is not int or type(b) is not int:
        return dict(
            status="INCOMPLETE",
            errors=["resident scheduler endpoints missing"],
            phases_ms=None,
            work=None,
            wrapper_unaccounted_ms=None,
        )
    for name in ("stock_schedule", *PHASES):
        rows = [
            r
            for r in spans
            if r.get("category") == "target_resident_" + name
            and a <= r["start_ns"] <= r["end_ns"] <= b
        ]
        if len(rows) != 1:
            errors.append("resident phase missing/ambiguous: " + name)
        else:
            found[name] = rows[0]
    work = {}
    for name in PHASES:
        r = found.get(name, {})
        if r.get("resident_schedule_policy") not in (POLICY, "k3-normalize-once-v1"):
            errors.append("resident policy missing/different: " + name)
        for key in ("resident_cycle", "request_table_size", "pid", "thread_id"):
            if type(r.get(key)) is not int:
                errors.append("resident phase field missing/type: " + name + "." + key)
        work[name] = r.get("work")
    policies = {r.get("resident_schedule_policy") for n, r in found.items() if n in PHASES}
    if len(policies) != 1:
        errors.append("resident policy differs within call")
    required = dict(
        binding=("live_requests", "binding_token_visits", "normalized_rows"),
        decisions=("decision_rows",),
        admission_records=("decision_rows", "admission_records"),
    )
    if policies == {POLICY}:
        required["admission_records"] += ("checkpoint_payload_encodings", "shared_field_snapshots")
    for name, keys in required.items():
        for key in keys:
            value = (work.get(name) or {}).get(key)
            if type(value) is not int or value < 0:
                errors.append("resident work field missing/type: " + name + "." + key)
    if not errors:
        children = [found[n] for n in PHASES]
        parent = found["stock_schedule"]
        if (
            len({(r["pid"], r["thread_id"], r["resident_cycle"]) for r in children}) != 1
            or any(
                (r["pid"], r["thread_id"]) != (parent["pid"], parent["thread_id"])
                or not parent["start_ns"] <= r["start_ns"] <= r["end_ns"] <= parent["end_ns"]
                for r in children
            )
            or any(x["end_ns"] > y["start_ns"] for x, y in zip(children, children[1:]))
        ):
            errors.append("resident phases violate ordered, same-lane nested boundaries")
        live = work["binding"]["live_requests"]
        if not (
            live
            == work["binding"]["normalized_rows"]
            == work["decisions"]["decision_rows"]
            == work["admission_records"]["admission_records"]
            == work["admission_records"]["decision_rows"]
            <= found["binding"]["request_table_size"]
            and work["binding"]["binding_token_visits"] >= live
        ):
            errors.append("resident work conservation differs")
    valid = not errors
    return dict(
        status="COMPLETE" if valid else "INCOMPLETE",
        errors=errors,
        policy=next(iter(policies)) if len(policies) == 1 else None,
        phases_ms={
            n: (found[n]["end_ns"] - found[n]["start_ns"]) / 1e6 if n in found else None
            for n in PHASES
        },
        work=work,
        wrapper_unaccounted_ms=(
            (found["stock_schedule"]["end_ns"] - found["stock_schedule"]["start_ns"]) / 1e6
            - duration([(found[n]["start_ns"], found[n]["end_ns"]) for n in PHASES])
        )
        if valid
        else None,
    )


def summarize(rows):
    evidence = [r["resident_schedule"] for r in rows]
    errors = [
        f"step {r['step_index']}: {e}" for r in rows for e in r["resident_schedule"]["errors"]
    ]
    if not rows:
        errors.append("no measured resident scheduler steps")
    return dict(
        status="INCOMPLETE" if errors else "COMPLETE",
        errors=errors,
        policy=(evidence[0].get("policy") if evidence and
                len({e.get("policy") for e in evidence}) == 1 else None),
        measured_steps=len(rows),
        phases_ms={
            n: stats(
                [
                    e["phases_ms"][n]
                    for e in evidence
                    if e["phases_ms"] and e["phases_ms"][n] is not None
                ]
            )
            for n in PHASES
        },
        work_totals=None
        if errors
        else {
            k: sum(e["work"]["binding"][k] for e in evidence)
            for k in ("live_requests", "binding_token_visits", "normalized_rows")
        },
        admission_records=None
        if errors
        else sum(e["work"]["admission_records"]["admission_records"] for e in evidence),
        checkpoint_payload_encodings=(sum(e["work"]["admission_records"][
            "checkpoint_payload_encodings"] for e in evidence)
            if not errors and all(e.get("policy") == POLICY for e in evidence) else None),
        shared_field_snapshots=(sum(e["work"]["admission_records"]["shared_field_snapshots"]
            for e in evidence)
            if not errors and all(e.get("policy") == POLICY for e in evidence) else None),
        wrapper_unaccounted_ms=stats(
            [
                e["wrapper_unaccounted_ms"]
                for e in evidence
                if e["wrapper_unaccounted_ms"] is not None
            ]
        ),
        semantics="six disjoint children nested in target_resident_stock_schedule; stock is "
        "the actual pinned Scheduler.schedule call including its predicate callbacks; binding "
        "includes normalization/matching, decisions include initial proposal checks, records "
        "include construction/append; nested hash/audit/JSON spans are NOT additive",
        scale_scope="successful current-row int conversion visits; includes generated suffix; "
        "not all token accesses elsewhere in scheduler; rows counted, no per-token events",
        limitations="recorder self cost and stock internals not independently isolated; "
        "missing evidence stays missing; no throughput extrapolation",
    )
