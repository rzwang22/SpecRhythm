"""Offline scale metrics; capture and scheduling are deliberately unchanged."""

from collections import defaultdict
from statistics import mean, median

from specrhythm.serving.eager_evidence_export import duration, touching


def capture_summary(hosts, start, end, category="target_forward_diagnostics"):
    """Clip spans to the measured window, and union separately on each thread."""
    result = {}
    for producer, host in hosts.items():
        rows = [
            r
            for r in host.get("intervals", [])
            if r["category"] == category and touching(r, start, end)
        ]
        lanes = defaultdict(list)
        missing = 0
        for r in rows:
            if r.get("pid") is None or r.get("thread_id") is None:
                missing += 1
            else:
                lanes[r["pid"], r["thread_id"]].append(
                    (max(start, r["start_ns"]), min(end, r["end_ns"]))
                )
        values = sorted((min(end, r["end_ns"]) - max(start, r["start_ns"])) / 1e6 for r in rows)
        result[producer] = dict(
            observed_category=category,
            status="OBSERVED" if rows and not missing else "MISSING",
            count=len(rows),
            mean_ms=mean(values) if values else None,
            p50_ms=median(values) if values else None,
            # Nearest rank; unlike interpolation this remains an observed duration.
            p95_ms=values[max(0, (95 * len(values) + 99) // 100 - 1)] if values else None,
            inclusive_sum_ms=sum(values) if values else None,
            missing_thread_rows=missing,
            threads=[
                dict(
                    pid=pid,
                    thread_id=tid,
                    union_ms=duration(spans),
                    window_share=duration(spans) / ((end - start) / 1e6),
                )
                for (pid, tid), spans in sorted(lanes.items())
            ],
            scope="window-clipped inclusive capture spans; union within each thread; "
            "contains child work and waits; not additive across ranks/threads/GPU",
        )
    return result


def metadata(options):
    """Declared fixed-runtime configuration, alongside raw worker effective metadata."""
    return dict(
        capture_function="specrhythm.phase4.vllm_diagnostics.capture_target_forward",
        target_logits_diagnostics="ENABLED; existing raw top-10/argmax/log-softmax path",
        numerical_diagnostic_plan="UNCHANGED; effective worker environment recorded separately",
        observation=options["observation"],
        draft_audit=options["draft_audit"],
        identity_matching=options["identity_matching"],
        draft_dispatch=options.get("draft_dispatch"),
        capture_algorithm_changed=False,
        deferred_bounds=dict(
            records_per_process=100000,
            encoded_bytes_per_process=256 * 1024**2,
            plugin_report_bytes=64 * 1024**2,
        ),
        dispatch_records_per_phase=4096,
        export_bounds=dict(
            single_source_bytes=2 * 1024**3,
            per_repeat_unique_bytes=4 * 1024**3,
            combined_unique_bytes=8 * 1024**3,
            logical_files=1200,
        ),
        overflow_policy="FAIL evidence; no capacity flush, silent dropping or deadline extension",
    )
