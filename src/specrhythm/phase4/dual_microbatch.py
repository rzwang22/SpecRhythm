"""One fixed-cohort control; no batching policy or GPU imports."""

from __future__ import annotations

import os
import re

ENV = "SR_PHASE4B_DUAL_MICROBATCH_SIZE"
FIELDS = ("requested_dual_microbatch_size", "effective_dual_microbatch_size")


def positive_size(value):
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        value = int(value)
    if type(value) is not int or value < 1:
        raise ValueError(f"{ENV} must be a positive integer")
    return value


def selected_size(environ=None):
    return positive_size((os.environ if environ is None else environ).get(ENV, "2"))


def evidence(requested, effective):
    return dict(zip(FIELDS, (positive_size(requested), positive_size(effective))))


def scheduler_evidence(rows, requested):
    """Read back the upper bound used by the scheduler, not an observed quantile."""
    requested = positive_size(requested)
    if not rows:
        raise ValueError("Dual scheduler microbatch evidence is missing")
    for row in rows:
        if any(type(row.get(key)) is not int or row[key] != requested for key in FIELDS):
            raise ValueError("requested/effective Dual scheduler microbatch differs")
        if len(row.get("verify_request_ids", ())) > requested:
            raise ValueError("Dual verification cohort exceeds its microbatch upper bound")
    return evidence(requested, rows[0][FIELDS[1]])


if __name__ == "__main__":
    print(selected_size())
