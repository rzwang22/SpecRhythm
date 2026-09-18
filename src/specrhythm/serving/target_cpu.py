"""Equivalent full block audit implementation; no state or proof is cached."""

import os

POLICIES = ("reference", "block-sets")


def policy():
    value = os.environ.get("SR_K3_TARGET_CPU", "reference")
    if value not in POLICIES:
        raise ValueError("invalid Target CPU policy: " + value)
    return value
