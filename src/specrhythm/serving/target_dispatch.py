"""One independent optimization of synchronous control serialization, not scheduling policy."""

import os

ENV = "SR_K3_TARGET_DISPATCH"
POLICIES = ("reference", "encode-once")


def policy():
    value = os.environ.get(ENV, "reference")
    if value not in POLICIES:
        raise ValueError("invalid Target dispatch encoding policy: " + value)
    return value


def qualification_errors(value, runtime, options):
    if "target_dispatch" not in options:
        return []  # Preserve historical reports and defaults.
    errors = []
    expected = options["target_dispatch"]
    if runtime.get("diagnostic_configuration", {}).get("target_dispatch") != expected:
        errors.append("declared Target dispatch policy differs from runtime")
    cycle = value.get("cycle_accounting") or {}
    errors.extend(cycle.get("association_errors", []))
    if not cycle.get("matched_step_summary", {}).get("complete") \
            or cycle.get("matched_step_summary", {}).get("incomplete"):
        errors.append("matched dispatch step landmarks missing/incomplete")
    if cycle.get("request_cycle_summary", {}).get("incomplete"):
        errors.append("in-window request cycle landmarks missing/incomplete")
    spans = runtime.get("host", {}).get("causal_timeline", {}).get("rows", [])
    for step in cycle.get("steps", []):
        a = step["landmarks_ns"]["claim"]
        b = step["landmarks_ns"]["GPU_start_lower"]
        rows = [r for r in spans if r["category"] == "control_snapshot_publish"
                and r.get("file_name") == "s2-control.json" and a <= r["start_ns"] <= b]
        if not rows or any(r.get("encoding_policy") != expected for r in rows):
            errors.append(f"step {step['key']}: control encoding evidence missing/different")
    return errors
