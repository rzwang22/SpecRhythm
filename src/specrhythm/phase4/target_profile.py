"""Explicit Target forensics selection; never a sampling or execution switch."""

import os
from pathlib import Path
from types import MappingProxyType

ENV = "SR_FIXED_TARGET_DIAGNOSTICS"
PROFILES = ("full", "lean")
NOT_COLLECTED = "NOT_COLLECTED_BY_PROFILE"
NUMERICAL_FIELDS = ("top_raw_logits", "top_target_logprobs", "selected_target_token_id")


def profile():
    value = os.environ.get(ENV, "full")
    if value not in PROFILES:
        raise ValueError("unknown Target diagnostic profile: " + value)
    if value == "lean" and os.environ.get("SR_PHASE4_NUMERICAL_DIAGNOSTIC_PLAN"):
        raise ValueError("lean Target diagnostics conflicts with full numerical diagnostic plan")
    return value


def coverage(value):
    if value not in PROFILES:
        raise ValueError("unknown Target diagnostic profile")
    return dict(
        target_diagnostic_profile=value,
        numerical_forensics="ENABLED_ON_TP_RANK0" if value == "full" else NOT_COLLECTED,
        numerical_scope="rank0 input/numerical rows; both ranks retain native forward evidence",
        required_evidence="live prefix/input/positions/speculative mapping; native TP forwards; "
        "actual sampler and committed-token accounting",
        sampling="unchanged vLLM sampler; optional diagnostic argmax is never feedback",
    )


def initialize(runner):
    """Once per worker/runner, before decode. Bindings keep the proposer's lifecycle.

    Only immutable definitions are retained. The existing identity owner compares
    the full CURRENT prompt and validates aliases on every bind; its historical
    mappings survive retirement. No KV, generated tokens or prefix version is cached.
    A new workload/identity owner requires a new initialization (never implicit reuse).
    """
    if profile() != "lean":
        return
    from specrhythm.serving.runtime_profile import load_runtime_requests

    path = Path(os.environ["SR_PHASE4_WORKLOAD"]).resolve()
    count = sum(bool(x.strip()) for x in path.read_text(encoding="utf-8").splitlines())
    definitions = load_runtime_requests(path, expected_count=count, require_task_mixture=False)
    identity = runner.drafter.identity
    prompts = {r.request_id: tuple(r.prompt_token_ids) for r in definitions}
    if dict(identity.stable_prompts) != prompts:
        raise ValueError("Target diagnostic workload disagrees with proposer identity")
    runner.target_diagnostic_workload = (
        str(path), identity, MappingProxyType({r.request_id: r for r in definitions})
    )


def definitions(runner):
    value = getattr(runner, "target_diagnostic_workload", None)
    if value is None:
        raise RuntimeError("lean Target diagnostics not initialized before forward")
    path, identity, rows = value
    # No I/O here. The run manifest seals the immutable workload; live input is
    # checked separately below. Replacing the owner/path invalidates this index.
    if os.environ.get("SR_PHASE4_WORKLOAD") != path or runner.drafter.identity is not identity:
        raise RuntimeError("Target diagnostic workload/identity owner changed")
    return rows, identity


def qualify(runtime, options):
    """Only explicit new experiments require the new declaration; old artifacts stay readable."""
    if "target_diagnostics" not in options:
        return
    from specrhythm.serving.common import require

    expected = coverage(options["target_diagnostics"])
    declared = runtime.get("diagnostic_configuration", {})
    require(all(declared.get(k) == v for k, v in expected.items()),
            "Target diagnostic profile declaration mismatch")
    for worker in runtime["target_devices"]:
        actual = worker.get("diagnostic_configuration", {})
        require(actual.get("target_diagnostics_enabled") is True
                and all(actual.get(k) == v for k, v in expected.items()),
                "Target diagnostic profile/effective coverage mismatch",
                actual=actual)
        for row in worker["target_rows"]:
            require(row.get("target_diagnostic_profile", "full") == options["target_diagnostics"],
                    "Target input evidence profile mismatch", actual=row.get("request_id"))
            if options["target_diagnostics"] == "lean":
                require(row.get("numerical_forensics") == NOT_COLLECTED,
                        "lean Target input evidence coverage missing")
