"""Strict, single-assignment absolute monotonic deadline for diagnostic drain RPCs."""

import json
import time

MISSING = object()


class DeadlineContractError(ValueError):
    """Missing, malformed or changed deadline; never an elapsed-time failure."""


class DeadlineExpired(TimeoutError):
    """A well-formed absolute deadline has actually expired."""


def deadline_context(value=MISSING, *, mode, directory, phase, now_ns=None):
    now = time.monotonic_ns() if now_ns is None else now_ns
    valid = type(value) is int and 0 < value <= (1 << 63) - 1
    context = dict(
        mode=mode, run_directory=str(directory), phase=phase,
        deadline_present=value is not MISSING,
        received_deadline_ns=None if value is MISSING else value,
        received_deadline_type="missing" if value is MISSING else type(value).__name__,
        now_ns=now,
    )
    if valid:
        context["remaining_ns"] = value - now
    return context


def validate_deadline(value=MISSING, *, mode, directory, phase, original=None, now_ns=None):
    context = deadline_context(value, mode=mode, directory=directory, phase=phase, now_ns=now_ns)
    reason = (
        "missing_deadline" if value is MISSING else
        "invalid_deadline" if "remaining_ns" not in context else
        "conflicting_deadline" if original is not None and value != original else
        "expired_deadline" if context["remaining_ns"] <= 0 else None
    )
    if reason:
        context.update(reason=reason, original_deadline_ns=original)
        cls = DeadlineExpired if reason == "expired_deadline" else DeadlineContractError
        error = cls(reason + ": " + json.dumps(context, sort_keys=True))
        error.deadline_context = context
        raise error
    return value


class DrainDeadline:
    """No cached PASS: every use revalidates the received value and remaining time."""

    def __init__(self, mode, directory):
        self.mode, self.directory, self.value = mode, directory, None

    def bind(self, payload, phase):
        value = validate_deadline(payload.get("deadline_ns", MISSING), mode=self.mode,
                                  directory=self.directory, phase=phase, original=self.value)
        self.value = value
        return value
