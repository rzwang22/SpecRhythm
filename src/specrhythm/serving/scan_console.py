"""Terminal summaries only; complete result bytes remain in the run artifacts."""

from pathlib import Path

FIELDS = (
    "mode", "batch", "probe", "valid", "execution_status", "measurement_status",
    "cleanup_status", "capacity_status", "effective_exit_code", "stop_reason",
    "validation_profile", "output_equivalence_status", "native_geometry_status",
    "target_steps", "committed_window_tokens", "measured_window_ms", "decode_throughput_tok_s",
)


def summary(value, report):
    errors = value.get("errors")
    return {
        "console_profile": "summary; full evidence retained in report",
        **{k: value[k] for k in FIELDS if k in value},
        "error_count": len(errors) if isinstance(errors, list) else None,
        "first_errors": [str(e)[:500] for e in errors[:3]] if isinstance(errors, list) else None,
        "full_report": str(Path(report)),
    }
