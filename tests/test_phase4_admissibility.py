from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from specrhythm.phase4.admissibility import (
    AdmissibilitySnapshot,
    ExecutionPhase,
    ProposalEvidence,
    ScheduledOperation,
    SchedulerRequestState,
    decide_admissibility,
    decision_event,
    select_admissible,
    validate_admissibility_events,
    validate_gate_a_construction,
)
from specrhythm.phase4.process_lifecycle import (
    run_owned_target,
    validate_lifecycle_artifact,
)


def snapshot(
    state: SchedulerRequestState,
    *,
    request_id: str = "A",
    internal_id: str = "opaque-A",
    phase: ExecutionPhase = ExecutionPhase.TIMED_DECODE,
    proposal: bool = False,
) -> AdmissibilitySnapshot:
    evidence = (
        ProposalEvidence(
            request_id=request_id,
            internal_request_id=internal_id,
            prefix_version=1,
            prefix_token_count=3,
            prefix_token_sha256="prefix",
            round_id=0,
            proposal_token_ids=(10, 11),
            ready_timestamp_ns=90,
        )
        if proposal
        else None
    )
    return AdmissibilitySnapshot(
        internal_request_id=internal_id,
        stable_request_id=request_id,
        state=state,
        execution_phase=phase,
        prefix_version=1,
        round_id=0,
        prefix_token_count=3,
        prefix_token_sha256="prefix",
        num_computed_tokens=3,
        num_output_tokens=1,
        spec_token_ids=(10, 11) if proposal else (),
        proposal=evidence,
        now_ns=100,
    )


@pytest.mark.parametrize(
    "state",
    [SchedulerRequestState.WAITING_DRAFT, SchedulerRequestState.DRAFTING],
)
def test_waiting_and_drafting_requests_are_not_admissible(state):
    decision = decide_admissibility(snapshot(state))
    assert decision.admissible is False
    assert decision.operation is ScheduledOperation.NONE


def test_matching_proposal_tail_prefill_and_terminal_contract():
    ready = decide_admissibility(
        snapshot(SchedulerRequestState.PROPOSAL_READY, proposal=True)
    )
    assert ready.admissible is True
    assert ready.operation is ScheduledOperation.VERIFY
    verify = decide_admissibility(
        snapshot(SchedulerRequestState.VERIFY_READY, proposal=True)
    )
    assert verify.admissible is True
    assert decide_admissibility(
        snapshot(SchedulerRequestState.TARGET_TAIL_READY)
    ).operation is ScheduledOperation.TARGET_TAIL
    prefill = decide_admissibility(
        snapshot(
            SchedulerRequestState.WAITING_DRAFT,
            phase=ExecutionPhase.SETUP_PREFILL,
        )
    )
    assert prefill.operation is ScheduledOperation.PREFILL
    assert decide_admissibility(
        snapshot(SchedulerRequestState.TERMINAL)
    ).admissible is False


@pytest.mark.parametrize(
    "change",
    [
        {"request_id": "B"},
        {"internal_request_id": "other"},
        {"prefix_version": 2},
        {"prefix_token_count": 4},
        {"prefix_token_sha256": "stale"},
        {"round_id": 1},
        {"consumed": True},
        {"expires_timestamp_ns": 99},
        {"proposal_token_ids": (99,)},
    ],
)
def test_stale_or_mismatched_proposal_is_fail_closed(change):
    base = snapshot(SchedulerRequestState.PROPOSAL_READY, proposal=True)
    values = dict(base.proposal.__dict__)
    values.update(change)
    changed = AdmissibilitySnapshot(
        **{**base.__dict__, "proposal": ProposalEvidence(**values)}
    )
    assert decide_admissibility(changed).admissible is False


def test_waiting_a_does_not_block_or_charge_prefill_b():
    waiting = snapshot(SchedulerRequestState.WAITING_DRAFT)
    prefill = snapshot(
        SchedulerRequestState.WAITING_DRAFT,
        request_id="B",
        internal_id="opaque-B",
        phase=ExecutionPhase.SETUP_PREFILL,
    )
    selected, remaining = select_admissible([waiting, prefill], token_budget=1)
    assert selected == ["B"]
    assert remaining == 0


def test_ready_proposal_is_immediately_admissible_and_events_are_auditable():
    ready = snapshot(SchedulerRequestState.PROPOSAL_READY, proposal=True)
    decision = decide_admissibility(ready)
    row = decision_event(
        ready,
        decision,
        cycle_id=4,
        scheduler_step=9,
        scheduled=True,
        target_input_positions=(3, 4, 5),
    )
    assert row["proposal_ready_timestamp_ns"] == 90
    assert row["scheduled_operation"] == "verify"
    assert validate_admissibility_events([row]) == []


def test_gate_a_artifact_proves_waiting_a_does_not_block_prefill_b():
    waiting = snapshot(SchedulerRequestState.WAITING_DRAFT)
    prefill = snapshot(
        SchedulerRequestState.WAITING_DRAFT,
        request_id="B",
        internal_id="opaque-B",
        phase=ExecutionPhase.SETUP_PREFILL,
    )
    cycle = {
        "cycle_id": 3,
        "scheduled_request_ids": ["B"],
        "request_admissibility": [
            decision_event(
                waiting,
                decide_admissibility(waiting),
                cycle_id=3,
                scheduler_step=4,
                scheduled=False,
            ),
            decision_event(
                prefill,
                decide_admissibility(prefill),
                cycle_id=3,
                scheduler_step=4,
                scheduled=True,
                target_input_positions=(0, 1),
            ),
        ],
    }
    report = validate_gate_a_construction(
        [cycle],
        waiting_request_id="A",
        prefill_request_id="B",
        lifecycle={"cleanup_valid": True, "remaining_owned_pids": []},
    )
    assert report["valid"] is True
    assert report["matching_cycle_ids"] == [3]


def test_scheduler_adapter_has_no_step_based_draft_readiness_workaround():
    root = Path(__file__).parents[1]
    source = (root / "src/specrhythm/phase4/vllm_dual_scheduler.py").read_text(
        encoding="utf-8"
    )
    assert "current_step +" not in source
    assert "next_decode_eligible_step =" not in source
    patch = (
        root
        / "integrations/vllm/patches/0002-scheduler-request-admissibility-hook.patch"
    ).read_text(encoding="utf-8")
    assert "req_index += 1" in patch
    assert "consumes no budget or KV allocation" in patch
    guard = (root / "src/specrhythm/phase4/vllm_dual.py").read_text(encoding="utf-8")
    assert "unproposed Target decode advanced a live request" in guard


def test_owned_process_group_propagates_normal_and_nonzero_status(tmp_path):
    normal_status, normal = run_owned_target(
        [sys.executable, "-c", "raise SystemExit(0)"],
        target_log=tmp_path / "normal.log",
        artifact_path=tmp_path / "normal.json",
        poll_seconds=0.01,
    )
    assert normal_status == 0
    assert validate_lifecycle_artifact(normal) == []
    failed_status, failed = run_owned_target(
        [sys.executable, "-c", "raise SystemExit(7)"],
        target_log=tmp_path / "failed.log",
        artifact_path=tmp_path / "failed.json",
        poll_seconds=0.01,
    )
    assert failed_status == 7
    assert failed["target_exit_status"] == 7
    assert failed["cleanup_valid"] is True


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group contract")
def test_wrapper_exit_with_live_grandchild_fails_and_cleans_owned_group(tmp_path):
    script = "import os,time; pid=os.fork(); os._exit(0) if pid else time.sleep(60)"
    status, report = run_owned_target(
        [sys.executable, "-c", script],
        target_log=tmp_path / "tree.log",
        artifact_path=tmp_path / "tree.json",
        graceful_seconds=0.2,
        kill_seconds=0.2,
        poll_seconds=0.01,
    )
    assert status == 125
    assert report["child_reap_result"]["wrapper_exited_with_descendants_alive"] is True
    assert report["remaining_owned_pids"] == []
    assert any(row["signal"] == "SIGTERM" for row in report["term_kill_actions"])


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group contract")
def test_timeout_kills_only_owned_group_and_preserves_unrelated_process(tmp_path):
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        script = (
            "import os,signal,time; pid=os.fork(); "
            "os._exit(9) if pid else (signal.signal(signal.SIGTERM, signal.SIG_IGN), "
            "time.sleep(60))"
        )
        status, report = run_owned_target(
            [sys.executable, "-c", script],
            target_log=tmp_path / "kill.log",
            artifact_path=tmp_path / "kill.json",
            graceful_seconds=0.1,
            kill_seconds=0.2,
            poll_seconds=0.01,
        )
        assert status == 9
        assert any(row["signal"] == "SIGKILL" for row in report["term_kill_actions"])
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=2)


def test_incomplete_cleanup_guard_blocks_next_run(tmp_path):
    guard = tmp_path / "owned.active"
    guard.write_text("prior run incomplete\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="cleanup is incomplete"):
        run_owned_target(
            [sys.executable, "-c", "raise SystemExit(0)"],
            target_log=tmp_path / "guard.log",
            artifact_path=tmp_path / "guard.json",
            guard_path=guard,
        )
    assert guard.exists()
