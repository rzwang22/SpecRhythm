from __future__ import annotations

import json
from pathlib import Path

import pytest

from specrhythm import cli
from specrhythm.phase4.admissibility import (
    AdmissibilitySnapshot,
    ExecutionPhase,
    ProposalEvidence,
    ScheduledOperation,
    SchedulerRequestState,
    decide_admissibility,
    decision_event,
    select_admissible,
)
from specrhythm.phase4.decode_ready import (
    DecodeReadyProvenance,
    ResidentSetupObservation,
    ResidentWarmStartProvider,
)
from specrhythm.phase4.dual_correctness import (
    compare_consumer_triangle,
    diagnose_overlap_artifacts,
    diagnose_overlap_timing,
    validate_controlled_gate,
    validate_draft_sync,
    validate_overlap_witness,
    validate_phase4b1_dual_correctness,
    validate_proposal_lifecycle_events,
    validate_request_state_events,
    validate_round_accounting,
    validate_scheduler_cycles,
    validate_verification_contracts,
)
from specrhythm.phase4.dual_service import DualDraftMachine
from specrhythm.phase4.manifest import sha256_file
from specrhythm.phase4.process_lifecycle import LIFECYCLE_SCHEMA
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.transport import CheckpointJsonl


class Backend:
    backend_name = "test"

    def __init__(self) -> None:
        self.prefixes = {}
        self.propose_calls = 0

    @property
    def provenance(self):
        return {"physical_gpu_id": 0}

    def initialize(self, request_id, tokens):
        self.prefixes[request_id] = tuple(tokens)

    def propose(self, request_id, budget, eos):
        del eos
        self.propose_calls += 1
        return (10, 11)[:budget], min(2, budget)

    def rollback(self, request_id, accepted):
        del request_id, accepted

    def append_target_token(self, request_id, token):
        self.prefixes[request_id] += (token,)

    def finish(self, request_id):
        del request_id

    def shutdown(self):
        pass


def test_decode_ready_initialization_generates_no_proposal_until_boundary():
    backend = Backend()
    machine = DualDraftMachine(backend)
    prefix = (1, 2, 3)
    row = {
        "request_id": "A",
        "committed_token_ids": list(prefix),
        "prefix_version": 1,
        "prefix_token_sha256": token_prefix_hash(prefix),
        "remaining_output_budget": 4,
        "eos_token_ids": [],
        "terminal": False,
    }
    initialized = machine.initialize(row)
    assert initialized["initial_proposal_generated"] is False
    assert initialized["logical_draft_kv_length"] == len(prefix)
    assert backend.propose_calls == 0
    proposal = machine.propose_only({**row, "measurement_start_ns": 1})
    assert proposal["proposal"]["round_id"] == 0
    assert backend.propose_calls == 1


def test_draft_sync_precedes_next_real_draft_and_state_evidence():
    round_row = _round("A", "p0", 0, proposal=(10,), prefix_version=1)
    draft = {
        "request_id": "A",
        "operation": "commit_and_propose",
        "success": True,
        "result": {
            "request_id": "A",
            "round_id": 0,
            "committed_token_ids": [10, 20],
            "rollback_length": 0,
            "logical_draft_kv_length": 5,
            "draft_sync_complete_ns": 80,
            "proposal": {"proposal_id": "p1"},
            "draft_gpu_interval": {"host_start_ns": 90},
            "terminal": False,
        },
    }
    states = [
        {
            "request_id": "A",
            "proposal_id": "p1",
            "destination_state": "DRAFT_READY",
            "timestamp_ns": 80,
        },
        {
            "request_id": "A",
            "proposal_id": "p1",
            "destination_state": "DRAFTING",
            "timestamp_ns": 90,
        },
    ]
    assert validate_draft_sync([draft], [round_row], states) == []
    draft["result"]["draft_gpu_interval"]["host_start_ns"] = 79
    assert any(
        "before correction/bonus sync" in item
        for item in validate_draft_sync([draft], [round_row], states)
    )


def test_state_machine_and_proposal_lifecycle_fail_closed():
    states = _state_rows("A", "pA")
    assert validate_request_state_events(states) == []
    broken = [dict(row) for row in states]
    broken[3]["source_state"] = "DRAFT_READY"
    assert any("non-contiguous" in item for item in validate_request_state_events(broken))
    lifecycle = _lifecycle_rows("A", "pA", 0, (10,))
    assert validate_proposal_lifecycle_events(lifecycle) == []
    duplicate = [*lifecycle, dict(lifecycle[-1])]
    assert any(
        "invalid proposal lifecycle" in item
        for item in validate_proposal_lifecycle_events(duplicate)
    )


def test_one_ready_one_waiting_two_ready_and_terminal_are_explicit():
    waiting = _snapshot("A", SchedulerRequestState.DRAFTING, proposal=False)
    ready_b = _snapshot("B", SchedulerRequestState.PROPOSAL_READY, proposal=True)
    selected, remaining = select_admissible([waiting, ready_b], token_budget=2)
    assert selected == ["B"]
    assert remaining == 1
    ready_a = _snapshot("A", SchedulerRequestState.PROPOSAL_READY, proposal=True)
    selected, remaining = select_admissible([ready_a, ready_b], token_budget=2)
    assert selected == ["A", "B"] and remaining == 0
    terminal = _snapshot("A", SchedulerRequestState.TERMINAL, proposal=False)
    assert decide_admissibility(terminal).admissible is False


def test_first_and_subsequent_verification_positions_and_accounting():
    round_row = _round("A", "pA", 0, proposal=(10, 11), prefix_version=1)
    diagnostics = [
        {
            "request_id": "A",
            "round_id": 0,
            "proposal_id": "pA",
            "target_input_token_ids": [3, 10, 11],
            "position_ids": [2, 3, 4],
            "physical_kv_num_computed_tokens": 2,
            "logical_committed_prefix_count": 3,
            "target_pending_input_token_id": 3,
            "committed_prefix_token_ids": [1, 2, 3],
            "target_kv_contains_rejected_or_future_tokens": False,
        }
    ]
    verifies = [{"proposal_id": "pA", "proposal_token_ids": [10, 11]}]
    assert validate_verification_contracts([round_row], verifies, diagnostics) == []
    assert validate_round_accounting([round_row]) == []
    diagnostics[0]["position_ids"] = [3, 4, 5]
    assert any(
        "positions" in item
        for item in validate_verification_contracts([round_row], verifies, diagnostics)
    )


@pytest.mark.parametrize("reason", ["eos", "max_tokens"])
def test_terminal_truncation_is_explicit(reason):
    row = _round("A", "pA", 0, proposal=(10,), prefix_version=1)
    row["terminal_truncation_reason"] = reason
    assert validate_round_accounting([row]) == []
    row["terminal_truncation_reason"] = "silent"
    assert any("truncation" in item for item in validate_round_accounting([row]))


def test_triangle_compares_tokens_termination_and_first_divergence():
    good = _run_outputs()
    assert compare_consumer_triangle([good, good, good])["valid"] is True
    changed = json.loads(json.dumps(good))
    changed["decode_only_outputs"][0]["generated_token_ids"][1] = 999
    report = compare_consumer_triangle([good, good, changed])
    assert report["valid"] is False
    assert report["comparisons"][1]["divergences"][0][
        "first_divergence_position"
    ] == 1


def test_cross_request_overlap_witness_requires_real_placement():
    witness = _overlap()
    assert validate_overlap_witness([witness]) == []
    witness["request_sets_disjoint"] = False
    assert validate_overlap_witness([witness])


def test_overlap_diagnostic_reports_exact_nearest_zero_duration_pair(tmp_path):
    draft = [
        {
            "request_id": "B",
            "result": {
                "proposal": {"proposal_id": "pB", "round_id": 0},
                "draft_gpu_interval": {
                    "host_start_ns": 10,
                    "host_end_ns": 20,
                    "physical_gpu_id": 0,
                },
            },
        }
    ]
    verification = [
        {
            "verify_microbatch_id": "vA",
            "verify_request_ids": ["A"],
            "verify_host_start_ns": 25,
            "verify_host_end_ns": 40,
            "target_physical_gpu_ids": [1, 2],
        }
    ]
    report = diagnose_overlap_timing(
        draft, verification, [{"overlap_duration_ns": 0}]
    )
    assert report["valid"] is True
    assert report["positive_overlap_observed"] is False
    assert report["nearest_pair"] == {
        "draft_request_id": "B",
        "proposal_id": "pB",
        "draft_round_id": 0,
        "verify_microbatch_id": "vA",
        "verify_request_ids": ["A"],
        "draft_host_interval_ns": [10, 20],
        "verify_host_interval_ns": [25, 40],
        "signed_intersection_ns": -5,
        "overlap_duration_ns": 0,
        "separation_ns": 5,
        "ordering": "draft-before-verify",
        "draft_physical_gpu_id": 0,
        "target_physical_gpu_ids": [1, 2],
    }

    paths = []
    for name, rows in (
        ("draft", draft),
        ("verify", verification),
        ("overlap", [{"overlap_duration_ns": 0}]),
    ):
        path = tmp_path / f"{name}.jsonl"
        for row in rows:
            CheckpointJsonl(path).append(row)
        paths.append(path)
    before = {str(path): sha256_file(path) for path in paths}
    artifact = diagnose_overlap_artifacts(
        draft_work_paths=[paths[0]],
        verification_paths=[paths[1]],
        overlap_paths=[paths[2]],
        output_path=tmp_path / "diagnosis.json",
    )
    assert artifact["valid"] is True
    assert artifact["input_artifacts_immutable"] is True
    assert before == {str(path): sha256_file(path) for path in paths}
    cli_output = tmp_path / "diagnosis-cli.json"
    assert (
        cli.main(
            [
                "phase4b1-overlap-diagnose",
                "--draft-work-events",
                str(paths[0]),
                "--verification-events",
                str(paths[1]),
                "--overlap-events",
                str(paths[2]),
                "--output",
                str(cli_output),
            ]
        )
        == 0
    )
    assert json.loads(cli_output.read_text())["runs"][0]["nearest_pair"][
        "separation_ns"
    ] == 5


def test_controlled_gate_proves_waiting_batching_and_terminal_progress(tmp_path):
    async_path = tmp_path / "async.jsonl"
    coordinated_path = tmp_path / "coordinated.jsonl"
    state_path = tmp_path / "states.jsonl"
    waiting = _snapshot("A", SchedulerRequestState.DRAFTING, proposal=False)
    ready = _snapshot("B", SchedulerRequestState.VERIFY_READY, proposal=True)
    async_rows = [
        {
            "cycle_id": 0,
            "poll_start_ns": 15,
            "request_admissibility": [
                decision_event(
                    waiting,
                    decide_admissibility(waiting),
                    cycle_id=0,
                    scheduler_step=0,
                    scheduled=False,
                ),
                decision_event(
                    ready,
                    decide_admissibility(ready),
                    cycle_id=0,
                    scheduler_step=0,
                    scheduled=True,
                ),
            ],
        },
        {
            "cycle_id": 1,
            "poll_start_ns": 30,
            "request_admissibility": [
                decision_event(
                    ready,
                    decide_admissibility(ready),
                    cycle_id=1,
                    scheduler_step=1,
                    scheduled=True,
                )
            ],
        },
    ]
    for row in async_rows:
        CheckpointJsonl(async_path).append(row)
    CheckpointJsonl(coordinated_path).append(_scheduler_cycle())
    CheckpointJsonl(state_path).append(
        {
            "request_id": "A",
            "destination_state": "TERMINAL",
            "timestamp_ns": 20,
        }
    )
    report = validate_controlled_gate(
        asynchronous_scheduler_path=async_path,
        coordinated_scheduler_path=coordinated_path,
        state_event_path=state_path,
        output_path=tmp_path / "controlled.json",
    )
    assert report["valid"] is True


def test_scheduler_validator_allows_real_waiting_setup_prefill_event():
    snapshot = _snapshot(
        "A",
        SchedulerRequestState.WAITING_DRAFT,
        proposal=False,
        phase=ExecutionPhase.SETUP_PREFILL,
    )
    decision = decide_admissibility(snapshot)
    assert decision.operation is ScheduledOperation.PREFILL
    row = decision_event(
        snapshot,
        decision,
        cycle_id=0,
        scheduler_step=0,
        scheduled=True,
        target_input_positions=(0, 1, 2),
    )
    assert validate_scheduler_cycles([_scheduler_cycle_for(row)]) == []


def test_scheduler_validator_rejects_waiting_timed_decode_scheduling():
    snapshot = _snapshot(
        "A", SchedulerRequestState.WAITING_DRAFT, proposal=False
    )
    row = decision_event(
        snapshot,
        decide_admissibility(snapshot),
        cycle_id=0,
        scheduler_step=0,
        scheduled=True,
        target_input_positions=(2,),
    )
    errors = validate_scheduler_cycles([_scheduler_cycle_for(row)])
    assert any("waiting request consumed Target budget" in item for item in errors)


def test_scheduler_validator_rejects_drafting_timed_decode_positions():
    snapshot = _snapshot("A", SchedulerRequestState.DRAFTING, proposal=False)
    row = decision_event(
        snapshot,
        decide_admissibility(snapshot),
        cycle_id=0,
        scheduler_step=0,
        scheduled=False,
        target_input_positions=(2,),
    )
    errors = validate_scheduler_cycles([_scheduler_cycle_for(row)])
    assert any("waiting request owns Target positions" in item for item in errors)


def test_scheduler_validator_accepts_exact_legal_target_tail():
    snapshot = _snapshot(
        "A", SchedulerRequestState.TARGET_TAIL_READY, proposal=False
    )
    decision = decide_admissibility(snapshot)
    assert decision.operation is ScheduledOperation.TARGET_TAIL
    row = decision_event(
        snapshot,
        decision,
        cycle_id=0,
        scheduler_step=0,
        scheduled=True,
        target_input_positions=(2,),
    )
    assert validate_scheduler_cycles([_scheduler_cycle_for(row)]) == []


def test_scheduler_validator_rejects_arbitrary_timed_decode_advancement():
    snapshot = _snapshot(
        "A", SchedulerRequestState.TARGET_TAIL_READY, proposal=False
    )
    row = decision_event(
        snapshot,
        decide_admissibility(snapshot),
        cycle_id=0,
        scheduler_step=0,
        scheduled=True,
        target_input_positions=(2,),
    )
    row["scheduled_operation"] = "arbitrary-target-work"
    errors = validate_scheduler_cycles([_scheduler_cycle_for(row)])
    assert any("unexplained live Target advancement" in item for item in errors)


@pytest.mark.parametrize(
    ("overlap_requirement", "overlap_duration_ns", "expected_valid"),
    (
        ("required", 5, True),
        ("required", 0, False),
        ("separate-gate", 0, True),
    ),
)
def test_full_triangle_validator_is_read_only_and_raw_order_is_diagnostic(
    tmp_path, overlap_requirement, overlap_duration_ns, expected_valid
):
    manifest = _manifest().to_dict()
    manifest_paths = [tmp_path / f"m-{index}.json" for index in range(4)]
    for path in manifest_paths:
        _json(path, manifest)
    target_path = tmp_path / "target.json"
    serial_path = tmp_path / "serial.json"
    _json(target_path, _run_outputs())
    _json(serial_path, _run_outputs())
    dual_paths = [tmp_path / "dual-1.json", tmp_path / "dual-2.json"]
    for path in dual_paths:
        _json(path, _run_outputs())
    target_process = tmp_path / "target-process.json"
    serial_process = tmp_path / "serial-process.json"
    _json(target_process, _process_lifecycle())
    _json(serial_process, _process_lifecycle())
    artifacts = {name: [] for name in (
        "states", "rounds", "lifecycles", "schedulers", "verifies", "drafts",
        "diagnostics", "overlaps", "processes"
    )}
    for run_index in range(2):
        order = ("A", "B") if run_index == 0 else ("B", "A")
        values = {
            "states": [row for request in order for row in _state_rows(request, f"p{request}")],
            "rounds": [_round(request, f"p{request}", 0) for request in order],
            "lifecycles": [
                row
                for request in order
                for row in _lifecycle_rows(request, f"p{request}", 0, (10,))
            ],
            "schedulers": [_scheduler_cycle()],
            "verifies": [
                {"proposal_id": f"p{request}", "proposal_token_ids": [10]}
                for request in order
            ],
            "drafts": [_draft_row(request) for request in order],
            "diagnostics": [_diagnostic(request) for request in order],
            "overlaps": [
                {**_overlap(), "overlap_duration_ns": overlap_duration_ns}
            ],
        }
        for name, rows in values.items():
            path = tmp_path / f"{name}-{run_index}.jsonl"
            for row in rows:
                CheckpointJsonl(path).append(row)
            artifacts[name].append(path)
        process = tmp_path / f"process-{run_index}.json"
        _json(process, _process_lifecycle())
        artifacts["processes"].append(process)
    inputs = [
        target_path,
        serial_path,
        target_process,
        serial_process,
        *dual_paths,
        *manifest_paths,
        *sum(artifacts.values(), []),
    ]
    before = {str(path): sha256_file(path) for path in inputs}
    report = validate_phase4b1_dual_correctness(
        target_path=target_path,
        serial_path=serial_path,
        dual_paths=dual_paths,
        target_manifest_path=manifest_paths[0],
        serial_manifest_path=manifest_paths[1],
        target_process_lifecycle_path=target_process,
        serial_process_lifecycle_path=serial_process,
        dual_manifest_paths=manifest_paths[2:],
        state_event_paths=artifacts["states"],
        proposal_event_paths=artifacts["rounds"],
        proposal_lifecycle_paths=artifacts["lifecycles"],
        scheduler_event_paths=artifacts["schedulers"],
        verification_event_paths=artifacts["verifies"],
        draft_work_event_paths=artifacts["drafts"],
        target_diagnostic_paths=artifacts["diagnostics"],
        overlap_event_paths=artifacts["overlaps"],
        process_lifecycle_paths=artifacts["processes"],
        output_path=tmp_path / "validation.json",
        markdown_path=tmp_path / "validation.md",
        overlap_requirement=overlap_requirement,
    )
    assert report["valid"] is expected_valid
    assert report["outcome"] == ("A" if expected_valid else "FAIL")
    assert report["overlap_requirement"] == overlap_requirement
    assert report["overlap_gate"]["valid"] is bool(overlap_duration_ns)
    assert report["dual_runs"][0]["overlap"]["valid"] is bool(
        overlap_duration_ns
    )
    assert report["repeat_comparisons"][0]["raw_event_order_equal"] is False
    assert before == {str(path): sha256_file(path) for path in inputs}


def _snapshot(
    request_id,
    state,
    *,
    proposal,
    phase=ExecutionPhase.TIMED_DECODE,
):
    evidence = ProposalEvidence(
        request_id=request_id,
        internal_request_id=f"opaque-{request_id}",
        prefix_version=1,
        prefix_token_count=3,
        prefix_token_sha256="h",
        round_id=0,
        proposal_token_ids=(10,),
        ready_timestamp_ns=2,
    ) if proposal else None
    return AdmissibilitySnapshot(
        internal_request_id=f"opaque-{request_id}",
        stable_request_id=request_id,
        state=state,
        execution_phase=phase,
        prefix_version=1,
        round_id=0,
        prefix_token_count=3,
        prefix_token_sha256="h",
        num_computed_tokens=2,
        num_output_tokens=1,
        spec_token_ids=(10,) if proposal else (),
        proposal=evidence,
        now_ns=3,
    )


def _scheduler_cycle_for(row):
    return {
        "cycle_id": row["cycle_id"],
        "scheduled_request_ids": (
            [row["request_id"]] if row["scheduled"] else []
        ),
        "request_admissibility": [row],
    }


def _state_rows(request_id, proposal_id):
    transitions = (
        ("BOOTSTRAP", "DRAFT_READY"),
        ("DRAFT_READY", "DRAFTING"),
        ("DRAFTING", "PROPOSAL_READY"),
        ("PROPOSAL_READY", "VERIFY_READY"),
        ("VERIFY_READY", "VERIFYING"),
        ("VERIFYING", "COMMITTING"),
        ("COMMITTING", "TERMINAL"),
    )
    return [
        {
            "request_id": request_id,
            "internal_request_id": f"opaque-{request_id}",
            "source_state": source,
            "destination_state": destination,
            "prefix_version": 1 if index < 5 else 2,
            "round_id": 0,
            "committed_prefix_length": 3 if index < 5 else 5,
            "committed_prefix_sha256": token_prefix_hash(
                (1, 2, 3) if index < 5 else (1, 2, 3, 10, 20)
            ),
            "proposal_id": proposal_id if index >= 2 else None,
            "reason": "test",
            "timestamp_ns": 10 + index,
        }
        for index, (source, destination) in enumerate(transitions)
    ]


def _lifecycle_rows(request_id, proposal_id, round_id, proposal):
    return [
        {
            "proposal_id": proposal_id,
            "request_id": request_id,
            "internal_request_id": f"opaque-{request_id}",
            "round_id": round_id,
            "prefix_version": 1,
            "prefix_token_count": 3,
            "prefix_token_sha256": "h",
            "proposal_token_ids": list(proposal),
            "proposal_length": len(proposal),
            "draft_start_ns": 50,
            "draft_end_ns": 51,
            "lifecycle_state": state,
            "timestamp_ns": 51 + index,
            "reason": "test",
        }
        for index, state in enumerate(("CREATED", "PUBLISHED", "INSTALLED", "CONSUMED"))
    ]


def _round(request_id, proposal_id, round_id, proposal=(10,), prefix_version=1):
    return {
        "request_id": request_id,
        "round_id": round_id,
        "proposal_id": proposal_id,
        "prefix_version": prefix_version,
        "proposal_token_ids": list(proposal),
        "accepted_draft_token_ids": list(proposal),
        "rejected_draft_token_ids": [],
        "accepted_draft_tokens": len(proposal),
        "rejected_draft_tokens": 0,
        "target_correction_token_ids": [],
        "target_bonus_token_ids": [20],
        "committed_token_ids": [*proposal, 20],
        "terminal": True,
        "terminal_truncation_reason": "max_tokens",
        "commit_end_ns": 60,
    }


def _scheduler_cycle():
    rows = []
    for request_id in ("A", "B"):
        snap = _snapshot(request_id, SchedulerRequestState.VERIFY_READY, proposal=True)
        rows.append(
            decision_event(
                snap,
                decide_admissibility(snap),
                cycle_id=0,
                scheduler_step=0,
                scheduled=True,
                target_input_positions=(2, 3),
            )
        )
    return {
        "cycle_id": 0,
        "scheduled_request_ids": ["A", "B"],
        "verify_request_ids": ["A", "B"],
        "request_admissibility": rows,
    }


def _draft_row(request_id):
    return {
        "request_id": request_id,
        "operation": "commit_and_propose",
        "start_ns": 61,
        "success": True,
        "result": {
            "request_id": request_id,
            "round_id": 0,
            "committed_token_ids": [10, 20],
            "rollback_length": 0,
            "logical_draft_kv_length": 5,
            "draft_sync_complete_ns": 62,
            "terminal": True,
        },
    }


def _diagnostic(request_id):
    return {
        "request_id": request_id,
        "round_id": 0,
        "proposal_id": f"p{request_id}",
        "target_input_token_ids": [3, 10],
        "position_ids": [2, 3],
        "physical_kv_num_computed_tokens": 2,
        "logical_committed_prefix_count": 3,
        "target_pending_input_token_id": 3,
        "committed_prefix_token_ids": [1, 2, 3],
        "target_kv_contains_rejected_or_future_tokens": False,
    }


def _overlap():
    return {
        "overlap_duration_ns": 5,
        "request_sets_disjoint": True,
        "draft_request_ids": ["B"],
        "verify_request_ids": ["A"],
        "draft_physical_gpu_ids": [0],
        "target_physical_gpu_ids": [1, 2],
        "draft_cuda_events": True,
        "target_rank_intervals": [{"physical_gpu_id": 1}, {"physical_gpu_id": 2}],
    }


def _run_outputs():
    rows = [
        {
            "request_id": request_id,
            "generated_token_ids": [3, 10, 20],
            "continuation_token_ids": [10, 20],
            "finish_reason": "length",
            "eos_token_id": None,
            "max_token_termination": True,
            "final_logical_length": 5,
        }
        for request_id in ("A", "B")
    ]
    return {
        "valid": True,
        "decode_only_outputs": rows,
        "runtime_semantics": {"target_blind_draft": True},
    }


def _manifest():
    provenance = DecodeReadyProvenance(
        specrhythm_git_commit="1" * 40,
        vllm_version="0.25.1",
        vllm_commit="752a3a504485790a2e8491cacbb35c137339ad34",
        vllm_patch_stack_sha256=("a" * 64,),
        target_model_path="target",
        target_model_revision=None,
        draft_model_path="draft",
        draft_model_revision=None,
        tokenizer_revision=None,
        workload_sha256="b" * 64,
        sampling_configuration={"temperature": 0.0, "seed": 1664},
        batch_invariant_configuration={"requested": True, "enable_dbo": False},
        target_physical_gpu_ids=(1, 2),
        draft_physical_gpu_ids=(0,),
        target_tensor_parallel_size=2,
        draft_tensor_parallel_size=1,
    )
    observations = [
        ResidentSetupObservation(
            request_id=request_id,
            internal_target_request_id=f"opaque-{request_id}",
            prompt_token_ids=(1, 2),
            bootstrap_token_id=3,
            target_materialized_kv_token_count=2,
            target_num_computed_tokens=2,
            draft_materialized_kv_token_count=3,
            bootstrap_ready_ns=11 + index,
            draft_initialization_complete_ns=13 + index,
        )
        for index, request_id in enumerate(("A", "B"))
    ]
    return ResidentWarmStartProvider().prepare(
        observations,
        provenance,
        setup_start_ns=10,
        setup_complete_ns=20,
        global_barrier_ns=30,
        measurement_start_ns=40,
    )


def _process_lifecycle():
    return {
        "schema_version": LIFECYCLE_SCHEMA,
        "coordinator_pid": 123,
        "pgid": 123,
        "session_id": 123,
        "child_reap_result": {"coordinator_reaped": True, "owned_group_empty": True},
        "remaining_owned_pids": [],
        "cleanup_valid": True,
        "draft_shutdown_result": {
            "required": True,
            "socket_path": "/tmp/test.sock",
            "socket_exists_after_cleanup": False,
            "alive_after_cleanup": False,
            "valid": True,
        },
    }


def _json(path: Path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
