"""Unique physical calls. Row roles are membership, never additive GPU time."""

from collections import Counter


def summarize(physical, native_for_host, start, end):
    rows = [r for r in physical if start <= r["start_ns"] <= end]
    declared = any(r.get("dispatch") is not None for r in rows)
    if not declared:
        return dict(
            status="NOT_COLLECTED", recovery_exclusive_calls=None, missed_compatible_calls=None
        )
    errors, ids, composition, batches = [], set(), Counter(), Counter()
    exclusive, mixed, missed = 0, 0, 0
    calls = []
    for row in rows:
        identity, dispatch = row.get("physical_forward_id"), row.get("dispatch")
        if not identity or identity in ids or dispatch is None:
            errors.append("physical identity/dispatch inventory missing or duplicate")
            continue
        ids.add(identity)
        bindings = row["bindings"]
        recovery = sum(b.get("source") in ("rejection", "eager_rejection") for b in bindings)
        ordinary = sum(b.get("source") == "ordinary" for b in bindings)
        bucket = (
            "recovery_only" if recovery == row["B"] else "recovery_mixed" if recovery else "other"
        )
        exclusive += int(bucket == "recovery_only")
        mixed += int(bucket == "recovery_mixed")
        composition[bucket] += 1
        batches[str(row["B"])] += 1
        available = [
            r
            for r in dispatch["inventory"]
            if r["executable"]
            and not r["selected"]
            and r["reason"] != "promotion_needs_no_forward"
        ]
        missed += int(
            bucket == "recovery_only" and any(r["source"] == "ordinary" for r in available)
        )
        native = native_for_host.get(row["start_ns"])
        if native is None:
            errors.append("dispatch native interval missing")
        calls.append(
            dict(
                physical_forward_id=identity,
                B=row["B"],
                composition=bucket,
                recovery_rows=recovery,
                ordinary_rows=ordinary,
                materialized_positions=row["materialized_positions"],
                sources=dict(Counter(b["source"] for b in bindings)),
                unselected_reasons=dict(
                    Counter(r["reason"] for r in dispatch["inventory"] if not r["selected"])
                ),
                native={
                    k: v
                    for k, v in (native or {}).items()
                    if k
                    in (
                        "forward_id", "native_forward_id", "physical_forward_id",
                        "gpu_elapsed_ms",
                        "gpu_event_elapsed_ms", "gpu_event_ms",
                        "start_lower_ns",
                        "start_upper_ns",
                        "end_lower_ns",
                        "end_upper_ns",
                    )
                },
            )
        )
    return dict(
        status="INCOMPLETE" if errors else "COMPLETE",
        errors=sorted(set(errors)),
        scope="host dispatch start inside measurement; native mapping by "
              "purpose/B/host containment",
        unique_physical_calls=len(ids),
        B_histogram=dict(batches),
        composition=dict(composition),
        recovery_exclusive_calls=exclusive,
        recovery_mixed_calls=mixed,
        missed_compatible_calls=missed,
        calls=calls,
        additive_role_time=False,
        historical_inventory="NOT_COLLECTED",
    )
