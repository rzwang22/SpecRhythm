"""Versioned S1-P acceptance contract; independent of historical Phase4 validators."""

from __future__ import annotations

import os

from specrhythm.serving.common import require

EXECUTION_SCHEMA = "specrhythm.s1-execution.v2"
RESULT_SCHEMA = "specrhythm.s1-result.v2"
COMPARISON_SCHEMA = "specrhythm.s1-comparison.v2"
EFFECTIVE_SCHEMA = "specrhythm.s1-effective-runtime.v2"
SEAL_SCHEMA = "specrhythm.s1-seal.v2"
G0_SCHEMA = "specrhythm.s1-g0.v2"
REQUIRED_ENVIRONMENT = {"OMP_NUM_THREADS": "1", "VLLM_ALLOW_INSECURE_SERIALIZATION": "1"}
DISABLED_ENVIRONMENT = ("USE_TORCH", "USE_TF", "USE_FLAX")


def policy_fields():
    return {
        "policy_schema_version": "specrhythm.s1-policy.v1",
        "acceptance_policy": "s1-performance-v1",
        "cross_run_token_equality_required": False,
        "cross_mode_token_equality_required": False,
        "cross_run_output_length_equality_required": False,
        "cross_run_output_comparison": {"performed": False, "status": "NOT_REQUIRED"},
        "execution_validity_required": True,
        "measurement_validity_required": True,
        "cleanup_validity_required": True,
    }


def validate_policy(value, schema=None):
    require(
        all(value.get(k) == v for k, v in policy_fields().items()),
        "S1-P acceptance policy mismatch; prepare a fresh S1-P root",
    )
    if schema is not None:
        require(value.get("schema_version") == schema, "unsupported S1-P artifact schema")


def environment_evidence():
    actual = {k: os.environ.get(k) for k in (*REQUIRED_ENVIRONMENT, *DISABLED_ENVIRONMENT)}
    require(
        actual == {**REQUIRED_ENVIRONMENT, **dict.fromkeys(DISABLED_ENVIRONMENT)},
        "S1-P requires explicit thread/local callable RPC environment and inference enabled",
        actual=actual,
    )
    return actual
