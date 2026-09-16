"""Offline four-mode entry checks, using actual native TP request sets, not labels."""

import math

from specrhythm.serving.common import require
from specrhythm.serving.fixed_results import device_batches, stats
from specrhythm.serving.k3 import B16, B64, configuration_of, geometry, matches_geometry


def full_batch_receipt(proof, mode, configuration=B16):
    """Compact comparison check; raw correctness records remain the source of truth."""
    if not isinstance(proof, dict) or proof.get("mode") != mode:
        return False
    if not matches_geometry(proof.get("execution_geometry"), mode, configuration):
        return False
    if type(proof.get("full_batch_steps")) is not int or proof["full_batch_steps"] < 1:
        return False
    first = proof.get("first_full_batch")
    if not isinstance(first, dict) or not isinstance(first.get("ranks"), dict):
        return False
    n = geometry(mode, configuration)["target_request_ceiling"]
    ids = first.get("request_ids")
    if (not isinstance(ids, list) or not all(isinstance(r, str) for r in ids)
            or len(ids) != n or len(set(ids)) != n):
        return False
    ranks = first["ranks"]
    return set(ranks) == {"0", "1"} and all(
        isinstance(f, dict) and type(f.get("B")) is int and f["B"] == n
        and isinstance(f.get("internal_request_ids"), list)
        and all(isinstance(r, str) for r in f["internal_request_ids"])
        and len(f["internal_request_ids"]) == len(set(f["internal_request_ids"])) == n
        and type(f.get("host_start_ns")) is int
        and type(f.get("gpu_event_ms")) in (int, float)
        and math.isfinite(f["gpu_event_ms"]) and f["gpu_event_ms"] > 0
        for f in ranks.values()) and set(ranks["0"]["internal_request_ids"]) == set(
            ranks["1"]["internal_request_ids"])


def native_geometry(runtime, mode, *, full_fixture=False, configuration=None):
    declared = configuration_of(runtime["point"])
    require(configuration is None or configuration == declared,
            "K3 native configuration mismatch")
    configuration = declared
    g = geometry(mode, configuration)
    point, capacity = runtime["point"], runtime["capacity"]
    require(point["mode"] == point["runtime_mode"] == mode, "K3 native report mode mismatch")
    require(type(point["batch"]) is int and point["batch"] == g["active_limit"],
            "K3 native active batch mismatch")
    require(configuration_of(capacity) == configuration
            and matches_geometry(capacity["execution_geometry"], mode, configuration),
            "K3 native geometry mismatch")
    ceiling = g["target_request_ceiling"]
    if configuration == B64:
        require(type(capacity["active_request_limit"]) is int
                and capacity["active_request_limit"] == g["active_limit"]
                and type(capacity["per_cohort_capacity"]) is int
                and capacity["per_cohort_capacity"] == max(g["home_capacities"].values())
                and type(capacity["cohort_count"]) is int
                and capacity["cohort_count"] == len(g["home_capacities"]),
                "B64 runtime capacity/home geometry mismatch")
    require(type(capacity["max_requests_per_target_forward"]) is int
            and capacity["max_requests_per_target_forward"] == ceiling,
            "K3 native capacity/geometry ceiling mismatch")
    steps = runtime["target_steps"]
    by_step = device_batches(runtime["target_devices"], steps)
    full, partial = [], []
    seen_homes = set()
    for i, step in enumerate(steps):
        b, ids, rows = step["B"], step["request_ids"], step["rows"]
        require(type(b) is int and 0 <= b <= ceiling, "K3 actual Target batch exceeds geometry")
        require(len(ids) == len(set(ids)) == len(rows) == b
                and set(ids) == {r["request_id"] for r in rows}
                and len({r["internal_request_id"] for r in rows}) == b,
                "K3 scheduled Target request cardinality mismatch")
        if configuration == B64:
            pop = step["population"]
            require(type(pop["active_requests"]) is int and 0 <= pop["active_requests"] <= 64
                    and type(pop["held_slots"]) is int and 0 <= pop["held_slots"] <= 64
                    and all(type(pop["cohort_held"][h]) is int
                            and 0 <= pop["cohort_held"][h] <= g["home_capacities"].get(h, 0)
                            for h in ("A", "B")), "B64 active/home occupancy exceeds geometry")
            homes = step["home_cohorts"]
            claims = step["ping_admission"]["claims"]
            require(set(homes) == set(ids) and len(claims) == b
                    and {c["request_id"]: c["home_cohort"] for c in claims} == homes
                    and set(homes.values()) <= set(g["home_capacities"]),
                    "B64 native request/home/claim association mismatch")
            seen_homes.update(homes.values())
        if not b:
            continue
        for f in by_step[i].values():
            require(type(f["B"]) is int and f["B"] == b
                    and len(f["internal_request_ids"]) == len(set(f["internal_request_ids"])) == b,
                    "K3 native forward distinct-request cardinality mismatch")
        if b == ceiling:
            full.append(i)
        else:
            # Preserve references to raw owner unfilled_reason/deferred and lifecycle
            # evidence. Partial request batches are distinct from EOS/K3 token tails.
            partial.append(dict(step_index=i, B=b,
                                target_batch_id=step.get("ping_admission", {}).get(
                                    "target_batch_id"),
                                population=step.get("population")))
    require(not full_fixture or bool(full),
            "K3 fixed full-batch fixture lacks one native Target forward at configured ceiling",
            mode=mode, expected_distinct_requests=ceiling)
    require(not full_fixture or configuration != B64
            or seen_homes == set(g["home_capacities"]), "B64 fixture missing home coverage")
    first = full[0] if full else None
    return dict(mode=mode, execution_geometry=g, full_batch_steps=len(full),
                observed_home_cohorts=sorted(seen_homes),
                evidence_scope="all recorded warmup/runtime verification steps",
                first_full_batch=None if first is None else dict(
                    step_index=first, request_ids=steps[first]["request_ids"],
                    ranks={str(rank): dict(host_start_ns=f["host_start_ns"],
                                          gpu_event_ms=f["gpu_event_ms"], B=f["B"],
                                          internal_request_ids=f["internal_request_ids"])
                           for rank, f in by_step[first].items()}),
                partial_batches=partial,
                partial_reason_source="runtime target_steps population and ping_admission; "
                "owner events admission.unfilled_reason/deferred by target_batch_id; "
                "request lifecycle retained without reconstruction")


def measurement(report, runtime, mode, configuration=B16, validation_profile=None):
    """The foreground runner's gate. No hardware, timing or historical report mutation."""
    from specrhythm.serving.k3_validation import EXPLORATION, matching

    policy = matching(report, runtime, runtime["point"])
    require(validation_profile is None or policy == validation_profile,
            "K3 measurement validation_profile mismatch")
    g = geometry(mode, configuration)
    require(configuration_of(report["point"]) == configuration,
            "K3 measurement requested configuration mismatch")
    require(report["mode"] == report["point"]["mode"] == mode,
            "K3 measurement report mode mismatch")
    require(report["point"] == runtime["point"], "K3 measurement/runtime point mismatch")
    require(type(report["batch"]) is int and report["batch"] == g["active_limit"]
            and type(report["point"]["batch"]) is int
            and report["point"]["batch"] == g["active_limit"],
            "K3 measurement active batch mismatch")
    require(report["probe"] is False and report["point"]["probe"] is False
            and runtime["probe"] is False, "K3 measurement must be an explicit non-probe")
    require(matches_geometry(report["execution_geometry"], mode, configuration),
            "K3 measurement geometry mismatch")
    require(type(report["sub_batch"]) is int
            and report["sub_batch"] == g["target_request_ceiling"],
            "K3 report Target ceiling mismatch")
    require(all(report[k] == "PASS" for k in
                ("capacity_status", "execution_status", "measurement_status", "cleanup_status"))
            and report["formal_comparison_eligible"] is True
            and type(report["effective_exit_code"]) is int and report["effective_exit_code"] == 0,
            "K3 original run qualification failed")
    require(report["stop_reason"] == "time_budget" and report["measured_window_ms"] >= 30000,
            "K3 measurement did not complete the fixed window")
    proof = native_geometry(runtime, mode, configuration=configuration,
                            full_fixture=policy == EXPLORATION)
    values = [s["B"] for s in runtime["target_steps"] if s["window"] and s["B"]]
    require(type(report["target_steps"]) is int and report["target_steps"] == len(values),
            "K3 reported measured step count mismatch")
    actual = report["actual_target_batch"]
    require(values and type(actual["min"]) is int and type(actual["max"]) is int
            and type(actual["count"]) is int
            and actual == dict(min=min(values), **stats(values))
            and 1 <= actual["min"] <= actual["max"] <= g["target_request_ceiling"],
            "K3 reported actual batch differs from measured native steps")
    return proof
