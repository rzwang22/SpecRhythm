"""Read-only contract against the pinned vLLM source; never import its GPU modules."""

import os
from pathlib import Path

import pytest


def test_held_ready_cohort_is_not_erased_by_other_input_batch():
    path = os.environ.get("SR_PHASE4B3_SOURCE_AUDIT")
    if not path:
        pytest.skip("pinned source checkout is exercised by Linux source-contract CI")
    root = Path(path)
    runner = (root / "vllm/v1/worker/gpu_model_runner.py").read_text()
    scheduler = (root / "vllm/v1/core/sched/scheduler.py").read_text()
    assert "unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)" in runner
    removal = runner.index("for req_id in unscheduled_req_ids:")
    assert "self.input_batch.remove_request(req_id)" in runner[removal : removal + 500]
    scheduled = scheduler.index(
        "scheduled_spec_decode_tokens[request.request_id] = spec_token_ids"
    )
    assert "request.spec_token_ids = []" in scheduler[scheduled : scheduled + 1100]


def test_exact_existing_patch_stack_no_pingpong_patch():
    from specrhythm.phase4.config import load_phase4_config

    config = load_phase4_config("configs/phase4b_dual_batch_1d2v.yaml")
    assert config.expected_vllm_commit == "752a3a504485790a2e8491cacbb35c137339ad34"
    assert config.proposal_budget == 4 and config.enforce_eager
    # The installed-stack manager remains the only source of patch requirements.
    source = Path("integrations/vllm/manage_patch.py").read_text()
    assert "pingpong" not in source.lower()
