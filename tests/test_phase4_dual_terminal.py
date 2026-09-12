"""Synthetic CPU reproduction of the reported completed-run terminal gap.

This fixture models the reported single DRAFT_SYNC gap; it is not a copy of A800
data and must never be presented as new GPU evidence.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict

import pytest

from specrhythm import cli
from specrhythm.phase4.admissibility import (
    AdmissibilitySnapshot,
    ExecutionPhase,
    ProposalEvidence,
    SchedulerRequestState,
    decide_admissibility,
    decision_event,
)
from specrhythm.phase4.decode_ready import (
    DecodeReadyProvenance,
    ResidentSetupObservation,
    ResidentWarmStartProvider,
)
from specrhythm.phase4.dual import DualProposal, proposal_identity
from specrhythm.phase4.dual_correctness import validate_request_state_events
from specrhythm.phase4.dual_runner import (
    build_cycle_and_overlap_events,
    summarize_retired_ready_results,
)
from specrhythm.phase4.dual_terminal import CLOSURE_REASON, build_terminal_reconciliation
from specrhythm.phase4.dual_terminal_recovery import (
    SOURCE_EXECUTION_COMMIT,
    audit_terminal_recovery,
    recover_terminal_state,
    validate_recovery_certificate,
)
from specrhythm.phase4.manifest import sha256_file
from specrhythm.phase4.performance import build_decode_performance_result
from specrhythm.phase4.performance_boundary import PERFORMANCE_EVENT, PERFORMANCE_EVENT_SCHEMA
from specrhythm.phase4.process_lifecycle import LIFECYCLE_SCHEMA
from specrhythm.phase4.resident_runner import _decode_rows
from specrhythm.phase4.resident_setup import build_setup_ready
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.stock_vllm import SmokeRequest
from specrhythm.phase4.transport import CheckpointJsonl

AFFECTED = "r3-22887f929fd54d97814c2bd3"


def _json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _checkpoint(path, rows):
    path.write_text("")
    for row in rows:
        CheckpointJsonl(path).append(
            {key: value for key, value in row.items() if key != "record_sha256"}
        )


def _proposal(request, prefix, version, round_id, base):
    tokens = (11,) if round_id == 0 else (13, 14)
    return DualProposal(
        request_id=request.request_id,
        round_id=round_id,
        proposal_id=proposal_identity(request.request_id, round_id, version, tokens),
        prefix_version=version,
        prefix_token_count=len(prefix),
        prefix_token_sha256=token_prefix_hash(prefix),
        draft_kv_length_before=len(prefix),
        draft_kv_length_after=len(prefix) + len(tokens),
        proposal_token_ids=tokens,
        created_timestamp_ns=base + 100,
        draft_start_ns=base + 10,
        draft_end_ns=base + 90,
    )


def _lifecycle(proposal, internal, states, base):
    return [
        {
            **proposal.to_dict(),
            "internal_request_id": internal,
            "lifecycle_state": state,
            "timestamp_ns": base + 100 + index,
            "reason": "request-retired-before-ready" if state == "DROPPED_STALE" else "fixture",
        }
        for index, state in enumerate(states)
    ]


def make_evidence(count=2):
    requests = [
        SmokeRequest(
            request_id=AFFECTED if index == 0 else f"request-{index}",
            task_class="code" if index < 60 else "chat" if index < 80 else "summarization",
            prompt_text=(
                "fixture prompt"
                if not 60 <= index < 80
                else "<|im_start|>user\nfixture<|im_end|>\n<|im_start|>assistant\n"
            ),
            prompt_token_ids=(100 + index, 200 + index),
            maximum_new_tokens=8 if index == 0 else 3,
            sampling_seed=1664,
            tokenizer_fingerprint="synthetic-cpu-fixture",
        )
        for index in range(count)
    ]
    observations = [
        ResidentSetupObservation(
            request_id=row.request_id,
            internal_target_request_id=f"opaque-{index}",
            prompt_token_ids=row.prompt_token_ids,
            bootstrap_token_id=10,
            target_materialized_kv_token_count=2,
            target_num_computed_tokens=2,
            draft_materialized_kv_token_count=3,
            bootstrap_ready_ns=20 + index,
            draft_initialization_complete_ns=200 + index,
        )
        for index, row in enumerate(requests)
    ]
    manifest = ResidentWarmStartProvider().prepare(
        observations,
        DecodeReadyProvenance(
            specrhythm_git_commit=SOURCE_EXECUTION_COMMIT,
            vllm_version="0.25.1",
            vllm_commit="752a3a504485790a2e8491cacbb35c137339ad34",
            vllm_patch_stack_sha256=("a" * 64, "b" * 64, "c" * 64),
            target_model_path="/models/target",
            target_model_revision=None,
            draft_model_path="/models/draft",
            draft_model_revision=None,
            tokenizer_revision=None,
            workload_sha256="f" * 64,
            sampling_configuration={"temperature": 0.0},
            batch_invariant_configuration={"requested": True},
            target_physical_gpu_ids=(1, 2),
            draft_physical_gpu_ids=(0,),
            target_tensor_parallel_size=2,
            draft_tensor_parallel_size=1,
        ),
        setup_start_ns=1,
        setup_complete_ns=500,
        global_barrier_ns=510,
        measurement_start_ns=520,
    )
    identity = {
        "mapping_source": "unique frozen prompt_token_ids",
        "suffix_parsing": False,
        "bound_request_count": count,
        "bindings": [
            {"request_id": row.request_id, "internal_request_id": f"opaque-{index}"}
            for index, row in enumerate(requests)
        ],
    }
    outputs, states, lifecycles, proposals, cycles = [], [], [], [], []
    for index, request in enumerate(requests):
        internal = f"opaque-{index}"
        base = 10000 + index * 1000
        prefix = (*request.prompt_token_ids, 10)
        final = (*prefix, 11, 12)
        proposal = _proposal(request, prefix, 1, 0, base)
        outputs.append(
            {
                "request_id": request.request_id,
                "prompt_length": 2,
                "generated_token_ids": [10, 11, 12],
                "generated_tokens": 3,
                "text": "synthetic",
                "finish_reason": "stop" if index == 0 else "length",
                "stop_reason": None,
                "top_logprobs": [],
                "timestamps": {},
                "token_accounting": {"prompt_tokens": 2, "generated_tokens": 3, "total_tokens": 5},
            }
        )
        chain = (
            "BOOTSTRAP",
            "DRAFT_READY",
            "DRAFTING",
            "PROPOSAL_READY",
            "VERIFY_READY",
            "VERIFYING",
            "COMMITTING",
            "DRAFT_SYNC" if index == 0 else "TERMINAL",
        )
        for offset, (source, destination) in enumerate(zip(chain, chain[1:])):
            committed = final if offset >= 5 else prefix
            states.append(
                {
                    "schema_version": "specrhythm.phase4b-request-state-event.v1",
                    "request_id": request.request_id,
                    "internal_request_id": internal,
                    "source_state": source,
                    "destination_state": destination,
                    "prefix_version": 2 if offset >= 5 else 1,
                    "round_id": 1 if offset >= 5 else 0,
                    "committed_prefix_length": len(committed),
                    "committed_prefix_sha256": token_prefix_hash(committed),
                    "proposal_id": proposal.proposal_id,
                    "reason": "synthetic-runtime-transition",
                    "timestamp_ns": base + offset * 50,
                }
            )
        lifecycles.extend(
            _lifecycle(
                proposal,
                internal,
                ("CREATED", "PUBLISHED", "INSTALLED", "CONSUMED"),
                base,
            )
        )
        proposals.append(
            {
                **proposal.to_dict(),
                "accepted_draft_token_ids": [11],
                "rejected_draft_token_ids": [],
                "accepted_draft_tokens": 1,
                "rejected_draft_tokens": 0,
                "target_correction_token_ids": [],
                "target_bonus_token_ids": [12],
                "committed_token_ids": [11, 12],
                "terminal": index != 0,
                "terminal_truncation_reason": "max_tokens" if index != 0 else None,
                "verify_microbatch_id": f"verify-{index}",
                "commit_start_ns": base + 230,
                "commit_end_ns": base + 240,
            }
        )
        snapshot = AdmissibilitySnapshot(
            internal_request_id=internal,
            stable_request_id=request.request_id,
            state=SchedulerRequestState.VERIFY_READY,
            execution_phase=ExecutionPhase.TIMED_DECODE,
            prefix_version=1,
            round_id=0,
            prefix_token_count=3,
            prefix_token_sha256=token_prefix_hash(prefix),
            num_computed_tokens=2,
            num_output_tokens=1,
            spec_token_ids=(11,),
            now_ns=base + 150,
            proposal=ProposalEvidence(
                request_id=request.request_id,
                internal_request_id=internal,
                prefix_version=1,
                round_id=0,
                prefix_token_count=3,
                prefix_token_sha256=token_prefix_hash(prefix),
                proposal_token_ids=(11,),
                ready_timestamp_ns=base + 100,
            ),
        )
        cycles.append(
            {
                "schema_version": "specrhythm.phase4b-scheduler-cycle.v1",
                "cycle_id": index * 2,
                "poll_start_ns": base + 140,
                "poll_end_ns": base + 160,
                "scheduled_request_ids": [request.request_id],
                "verify_request_ids": [request.request_id],
                "request_admissibility": [
                    decision_event(
                        snapshot,
                        decide_admissibility(snapshot),
                        cycle_id=index * 2,
                        scheduler_step=index,
                        scheduled=True,
                        target_input_positions=(2, 3),
                    )
                ],
                "retired_ready_results": [],
            }
        )
    late = _proposal(requests[0], (*requests[0].prompt_token_ids, 10, 11, 12), 2, 1, 10400)
    lifecycles.extend(
        _lifecycle(late, "opaque-0", ("CREATED", "PUBLISHED", "DROPPED_STALE"), 10400)
    )
    cycles.insert(
        1,
        {
            "schema_version": "specrhythm.phase4b-scheduler-cycle.v1",
            "cycle_id": 1,
            "poll_start_ns": 10490,
            "poll_end_ns": 10510,
            "scheduled_request_ids": [],
            "verify_request_ids": [],
            "request_admissibility": [],
            "retired_ready_results": [
                {
                    "schema_version": "specrhythm.phase4b2-retired-ready-result.v1",
                    "request_id": AFFECTED,
                    "internal_request_id": "opaque-0",
                    "result_kind": "proposal",
                    "proposal_id": late.proposal_id,
                    "target_tail_ready_ns": None,
                    "timestamp_ns": 10503,
                    "reason": "request-retired-before-ready",
                    "discarded": True,
                    "installed": False,
                    "verified": False,
                }
            ],
        },
    )
    return {
        "requests": requests,
        "outputs": outputs,
        "manifest": manifest,
        "identity": identity,
        "state_rows": states,
        "scheduler_rows": cycles,
        "lifecycle_rows": lifecycles,
        "proposal_rows": proposals,
        "observation_ns": 10_000_000,
    }


@pytest.fixture
def evidence():
    return make_evidence()


def test_reported_gap_closes_and_all_runtime_evidence_is_unchanged(evidence):
    before = deepcopy(evidence)
    assert validate_request_state_events(evidence["state_rows"]) == [
        f"{AFFECTED}: final state is DRAFT_SYNC, not TERMINAL",
    ]
    result = build_terminal_reconciliation(**evidence)
    assert result["reconciled_request_ids"] == [AFFECTED]
    event = result["events"][0]
    assert event["source_state"] == "DRAFT_SYNC" and event["destination_state"] == "TERMINAL"
    assert event["reason"] == CLOSURE_REASON
    assert not event["proposal_installed"] and not event["proposal_verified"]
    assert not event["proposal_committed"]
    assert evidence == before
    assert validate_request_state_events([*evidence["state_rows"], event]) == []


def test_normal_terminal_unchanged_and_reconciliation_is_idempotent(evidence):
    result = build_terminal_reconciliation(**evidence)
    evidence["state_rows"].extend(result["events"])
    before = deepcopy(evidence)
    again = build_terminal_reconciliation(**evidence)
    assert again["events"] == [] and again["reconciled_request_ids"] == []
    assert evidence == before


def test_post_terminal_physical_commit_is_never_trimmed_by_reconciliation(evidence):
    # Same false historical commit shape as the real 84-vs-83 prefix: a
    # successful final output cannot authorize rewriting already committed work.
    request = evidence["requests"][0]
    final = request.prompt_token_ids + tuple(evidence["outputs"][0]["generated_token_ids"])
    for row in evidence["state_rows"]:
        if row["request_id"] == AFFECTED and row["destination_state"] in {
            "COMMITTING", "DRAFT_SYNC",
        }:
            row["committed_prefix_length"] = len(final) + 1
            row["committed_prefix_sha256"] = token_prefix_hash(final + (151643,))
    before = deepcopy(evidence)
    with pytest.raises(ValueError, match="state prefix/hash contradicts final output"):
        build_terminal_reconciliation(**evidence)
    assert evidence == before


def test_tail_ready_legal_predecessor_with_same_completed_prefix(evidence):
    last = [row for row in evidence["state_rows"] if row["request_id"] == AFFECTED][-1]
    evidence["state_rows"].append(
        {
            **last,
            "source_state": "DRAFT_SYNC",
            "destination_state": "TARGET_TAIL_READY",
            "timestamp_ns": last["timestamp_ns"] + 1,
        }
    )
    event = evidence["scheduler_rows"][1]["retired_ready_results"][0]
    event.update(result_kind="target-tail", proposal_id=None, target_tail_ready_ns=10500)
    result = build_terminal_reconciliation(**evidence)
    assert result["events"][0]["source_state"] == "TARGET_TAIL_READY"


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-output",
        "duplicate-output",
        "failed-output",
        "nonterminal-output",
        "unfinished",
        "short-length",
        "too-long",
        "wrong-stop",
        "string-stop",
        "output-accounting",
        "identity",
        "alias",
        "prefix-hash",
        "prefix-length",
        "bootstrap",
        "failed-state",
        "illegal-transition",
        "bad-version",
        "bad-time",
        "closure-time",
        "no-retirement",
        "retired-identity",
        "retired-installed",
        "retired-verified",
        "still-live",
        "late-installed",
        "late-committed",
        "late-prefix",
        "terminal-then-nonterminal",
        "duplicate-retirement",
        "retired-before-drop",
        "missing-poll-start",
        "retired-round-bool",
    ],
)
def test_invalid_completion_or_conflicting_evidence_is_never_repaired(evidence, mutation):
    output = evidence["outputs"][0]
    state = [row for row in evidence["state_rows"] if row["request_id"] == AFFECTED][-1]
    event = evidence["scheduler_rows"][1]["retired_ready_results"][0]
    if mutation == "missing-output":
        evidence["outputs"].pop(0)
    elif mutation == "duplicate-output":
        evidence["outputs"].append(deepcopy(output))
    elif mutation == "failed-output":
        output["finish_reason"] = "abort"
    elif mutation == "nonterminal-output":
        output["finish_reason"] = None
    elif mutation == "unfinished":
        output["finished"] = False
    elif mutation == "short-length":
        output["finish_reason"] = "length"
    elif mutation == "too-long":
        output["generated_token_ids"] *= 3
    elif mutation == "wrong-stop":
        output["stop_reason"] = 99
    elif mutation == "string-stop":
        output["stop_reason"] = "unrequested stop string"
    elif mutation == "output-accounting":
        output["generated_tokens"] = 1
    elif mutation == "identity":
        state["internal_request_id"] = "opaque-1"
    elif mutation == "alias":
        evidence["identity"]["bindings"][1]["internal_request_id"] = "opaque-0"
    elif mutation == "prefix-hash":
        state["committed_prefix_sha256"] = "a" * 64
    elif mutation == "prefix-length":
        state["committed_prefix_length"] = 4
        state["committed_prefix_sha256"] = token_prefix_hash((100, 200, 10, 11))
    elif mutation == "bootstrap":
        output["generated_token_ids"][0] = 999
    elif mutation == "failed-state":
        state["destination_state"] = "FAILED"
    elif mutation == "illegal-transition":
        evidence["state_rows"][2]["source_state"] = "BOOTSTRAP"
    elif mutation == "bad-version":
        evidence["state_rows"][2]["prefix_version"] = -1
    elif mutation == "bad-time":
        evidence["state_rows"][2]["timestamp_ns"] = 0
    elif mutation == "closure-time":
        evidence["observation_ns"] = event["timestamp_ns"]
    elif mutation == "no-retirement":
        evidence["scheduler_rows"][1]["retired_ready_results"] = []
    elif mutation == "retired-identity":
        event["internal_request_id"] = "opaque-1"
    elif mutation == "retired-installed":
        event["installed"] = True
    elif mutation == "retired-verified":
        event["verified"] = True
    elif mutation == "still-live":
        evidence["scheduler_rows"][1]["request_admissibility"] = [{"request_id": AFFECTED}]
    elif mutation == "late-installed":
        evidence["lifecycle_rows"][-2]["lifecycle_state"] = "INSTALLED"
    elif mutation == "late-committed":
        evidence["proposal_rows"].append({"proposal_id": event["proposal_id"]})
    elif mutation == "late-prefix":
        evidence["lifecycle_rows"][-1]["prefix_token_sha256"] = "b" * 64
    elif mutation == "duplicate-retirement":
        evidence["scheduler_rows"][1]["retired_ready_results"].append(deepcopy(event))
    elif mutation == "retired-before-drop":
        event["timestamp_ns"] = 10501
    elif mutation == "missing-poll-start":
        evidence["scheduler_rows"][1].pop("poll_start_ns")
    elif mutation == "retired-round-bool":
        evidence["lifecycle_rows"][-1]["round_id"] = True
    else:
        evidence["state_rows"][1]["destination_state"] = "TERMINAL"
    before = deepcopy(evidence)
    with pytest.raises((ValueError, TypeError, KeyError)):
        build_terminal_reconciliation(**evidence)
    assert evidence == before


def test_conflicting_output_after_reconciliation_still_fails(evidence):
    evidence["state_rows"].extend(build_terminal_reconciliation(**evidence)["events"])
    evidence["outputs"][0]["generated_token_ids"][-1] = 999
    with pytest.raises(ValueError, match="prefix/hash"):
        build_terminal_reconciliation(**evidence)


def make_disk_run(tmp_path):
    from dataclasses import replace

    data = make_evidence(100)
    source = tmp_path / "source"
    root = source / "dual"
    root.mkdir(parents=True)
    paths = {"run_root": root}
    for key, name in (
        ("workload_path", "corrected-100.jsonl"),
        ("config_path", "config.json"),
        ("topology_path", "topology.json"),
        ("patch_manifest_path", "patch.json"),
    ):
        paths[key] = source / name
    _jsonl(
        paths["workload_path"],
        [{**asdict(row), "prompt_length": len(row.prompt_token_ids)} for row in data["requests"]],
    )
    for key in ("config_path", "topology_path", "patch_manifest_path"):
        _json(paths[key], {"fixture": key})
    manifest = replace(
        data["manifest"], workload_sha256=sha256_file(paths["workload_path"])
    ).with_hash()
    data["manifest"] = manifest
    _json(root / "decode-ready-manifest.json", manifest.to_dict())
    setup = build_setup_ready(
        manifest,
        consumer="dual-batch",
        manifest_path=root / "decode-ready-manifest.json",
        ready_published_ns=600,
    )
    _json(root / "setup-ready.json", setup)
    _json(root / "plugin-report.json", {"request_identity": data["identity"]})
    workers = [
        {
            "global_rank": rank,
            "world_size": 2,
            "physical_gpu_id": physical,
            "gpu_uuid": f"GPU-{physical}",
            "parameter_count": 1,
            "parameter_bytes": 2,
            "allocated_memory_bytes": 2,
            "all_parameters_on_expected_device": True,
        }
        for rank, physical in enumerate((1, 2))
    ]
    final_ns = 200000
    runtime = {
        "git_commit": SOURCE_EXECUTION_COMMIT,
        "decode_ready_manifest_sha256": manifest.manifest_sha256,
        "patch_manifest_sha256": sha256_file(paths["patch_manifest_path"]),
        "inputs": {
            name + "_sha256": sha256_file(paths[name + "_path"])
            for name in ("config", "workload", "topology")
        },
        "worker_ranks": workers,
    }
    _json(root / "runtime-manifest.json", runtime)
    rank_intervals = [
        {
            "global_rank": rank,
            "tp_rank": rank,
            "logical_cuda_index": rank,
            "physical_gpu_id": physical,
            "gpu_uuid": f"GPU-{physical}",
            "cuda_events": True,
            "cuda_synchronized": True,
        }
        for rank, physical in enumerate((1, 2))
    ]
    drafts = [
        {
            "request_id": data["requests"][1].request_id,
            "success": True,
            "result": {
                "proposal": {"proposal_id": "overlap-proposal", "round_id": 0},
                "draft_gpu_interval": {
                    "host_start_ns": 10150,
                    "host_end_ns": 10190,
                    "physical_gpu_id": 0,
                    "cuda_elapsed_ns": 40,
                },
            },
        }
    ]
    verifies = [
        {
            "verify_microbatch_id": "verify-0",
            "verify_request_ids": [AFFECTED],
            "verify_host_start_ns": 10160,
            "verify_host_end_ns": 10200,
            "target_physical_gpu_ids": [1, 2],
            "target_rank_intervals": rank_intervals,
        }
    ]
    cycles, overlaps = build_cycle_and_overlap_events(drafts, verifies, data["proposal_rows"])
    for name, rows in {
        "output-checkpoint": data["outputs"],
        "request-state-events": data["state_rows"],
        "scheduler-events": data["scheduler_rows"],
        "proposal-lifecycle-events": data["lifecycle_rows"],
        "proposal-events": data["proposal_rows"],
        "draft-work-events": drafts,
        "verification-events": verifies,
        "cycle-events": cycles,
        "overlap-events": overlaps,
        "target-diagnostics": [],
        "timing-events": [
            {
                "schema_version": PERFORMANCE_EVENT_SCHEMA,
                "event": PERFORMANCE_EVENT,
                "consumer": "dual-batch",
                "timestamp_ns": 1000,
                "setup_ready_published_ns": 600,
                "pre_measurement_tp_barrier": True,
                "pre_measurement_target_cuda_synchronize": True,
                "setup_excluded": True,
                "bootstrap_excluded_from_measured_tokens": True,
            }
        ],
        "timestamped-target-log": [{"timestamp_ns": final_ns + 10, "line": "Draft shut down"}],
    }.items():
        writer = _jsonl if name == "timestamped-target-log" else _checkpoint
        writer(root / f"{name}.jsonl", rows)
    (root / "target.log").write_text(
        "synthetic completed run; validation found one terminal gap\n"
    )
    raw = {
        "schema_version": "specrhythm.phase4b1-resident-dual-run.v1",
        "mode": "decode-only-dual-batch",
        "request_count": 100,
        "valid": False,
        "errors": validate_request_state_events(data["state_rows"]),
        "phase4b2_performance_candidate": True,
        "phase4b2_final_sync": [
            {
                "local_rank": rank,
                "physical_gpu_id": physical,
                "final_cuda_synchronize_complete_ns": final_ns - 10 + rank,
            }
            for rank, physical in enumerate((1, 2))
        ],
        "outputs": data["outputs"],
        "decode_only_outputs": _decode_rows(data["outputs"], manifest),
        "decode_ready_manifest_sha256": manifest.manifest_sha256,
        "global_setup_ready": setup,
        "worker_ranks": workers,
        "request_identity": data["identity"],
        "overlap_gate": {"required_for_run_validity": True, "valid": True, "errors": []},
        "draft_shutdown": {
            "shutdown": True,
            "request_count": 100,
            "failures": {},
            "inflight_request_ids": [],
            "work_queue_depth": 0,
        },
        "retired_ready_results": summarize_retired_ready_results(data["scheduler_rows"]),
        "run_start_ns": 550,
        "run_end_ns": final_ns,
        "artifact_sha256": {
            "workload": sha256_file(paths["workload_path"]),
            **{
                key: sha256_file(root / name)
                for key, name in (
                    ("decode_ready_manifest", "decode-ready-manifest.json"),
                    ("setup_ready", "setup-ready.json"),
                    ("output_checkpoint", "output-checkpoint.jsonl"),
                    ("runtime_manifest", "runtime-manifest.json"),
                )
            },
        },
    }
    _json(root / "resident-dual.json", raw)
    command = [
        "env",
        "CUDA_VISIBLE_DEVICES=1,2",
        "VLLM_USE_V2_MODEL_RUNNER=0",
        "VLLM_BATCH_INVARIANT=1",
        "python",
        "integrations/vllm/phase4b2_timestamp_command.py",
        "--output",
        str(root / "timestamped-target-log.jsonl"),
        "--",
        "specrhythm",
        "phase4b1-resident-dual-run",
    ]
    for flag, name in (
        ("output", "resident-dual.json"),
        ("output-checkpoint", "output-checkpoint.jsonl"),
        ("request-state-events", "request-state-events.jsonl"),
        ("scheduler-events", "scheduler-events.jsonl"),
        ("proposal-lifecycle-events", "proposal-lifecycle-events.jsonl"),
        ("runtime-manifest", "runtime-manifest.json"),
    ):
        command.extend(("--" + flag, str(root / name)))
    command.extend(
        (
            "--request-count",
            "100",
            "--microbatch-size",
            "2",
            "--test-coordination",
            "none",
            "--overlap-requirement",
            "required",
            "--phase4b2-performance",
        )
    )
    process = {
        "schema_version": LIFECYCLE_SCHEMA,
        "coordinator_pid": 10,
        "pgid": 10,
        "session_id": 10,
        "command": command,
        "target_exit_status": 1,
        "effective_exit_status": 1,
        "run_valid": False,
        "cleanup_valid": True,
        "launch_error": None,
        "remaining_owned_pids": [],
        "term_kill_actions": [],
        "start_monotonic_ns": 5,
        "exit_monotonic_ns": final_ns + 100,
        "child_reap_result": {
            "coordinator_reaped": True,
            "owned_group_empty": True,
            "wrapper_exited_with_descendants_alive": False,
        },
        "draft_shutdown_result": {
            "valid": True,
            "alive_after_cleanup": False,
            "socket_exists_after_cleanup": False,
        },
    }
    _json(root / "process-lifecycle.json", process)
    return paths


@pytest.fixture
def disk_run(tmp_path):
    return make_disk_run(tmp_path)


def test_strict_offline_recovery_and_measurement_preserve_raw_execution(disk_run, tmp_path):
    source = disk_run["run_root"]
    before = {str(path): path.read_bytes() for path in source.parent.rglob("*") if path.is_file()}
    report = recover_terminal_state(**disk_run, output_dir=tmp_path / "recovery")
    assert report["valid"] is True
    assert report["terminal_state_reconciliation"]["recovered"] is True
    assert report["terminal_state_reconciliation"]["reconciled_request_ids"] == [AFFECTED]
    assert report["source_resident_errors"] == [
        f"{AFFECTED}: final state is DRAFT_SYNC, not TERMINAL"
    ]
    assert (
        validate_request_state_events(
            CheckpointJsonl(tmp_path / "recovery/request-state-events.reconciled.jsonl").read()
        )
        == []
    )
    measured = build_decode_performance_result(
        mode="dual-batch",
        **disk_run,
        terminal_revalidation_path=tmp_path / "recovery/terminal-state-revalidation.json",
        output_path=tmp_path / "recovery/decode-performance.json",
    )
    assert measured["valid"] is True, measured["errors"]
    assert measured["execution_git_commit"] == SOURCE_EXECUTION_COMMIT
    assert measured["terminal_state_reconciliation"]["source_target_exit_status"] == 1
    assert measured["terminal_state_reconciliation"]["source_resident_valid"] is False
    assert measured["metrics"]["completed_requests"] == 100
    original_measurement = build_decode_performance_result(
        mode="dual-batch",
        **disk_run,
        output_path=tmp_path / "original-invalid-performance.json",
    )
    assert original_measurement["valid"] is False
    assert measured["metrics"] == original_measurement["metrics"]
    assert measured["requests"] == original_measurement["requests"]
    assert measured["measurement"] == original_measurement["measurement"]
    assert {
        str(path): path.read_bytes() for path in source.parent.rglob("*") if path.is_file()
    } == before
    with pytest.raises(ValueError, match="reuse"):
        recover_terminal_state(**disk_run, output_dir=tmp_path / "recovery")


@pytest.mark.parametrize(
    "mutation",
    [
        "other-commit",
        "other-error",
        "missing-output",
        "overlap",
        "one-rank",
        "cleanup",
        "crash-status",
        "launch-error",
        "survivors",
        "command",
        "traceback",
        "state-error",
        "retired-summary",
        "raw-output",
        "draft-failure",
        "input-digest",
    ],
)
def test_offline_recovery_rejects_unrelated_failures(disk_run, tmp_path, mutation):
    root = disk_run["run_root"]
    raw = json.loads((root / "resident-dual.json").read_text())
    process = json.loads((root / "process-lifecycle.json").read_text())
    if mutation == "other-commit":
        runtime = json.loads((root / "runtime-manifest.json").read_text())
        runtime["git_commit"] = "1" * 40
        _json(root / "runtime-manifest.json", runtime)
    elif mutation == "other-error":
        raw["errors"].append("another validator failed")
    elif mutation == "missing-output":
        checkpoint = CheckpointJsonl(root / "output-checkpoint.jsonl").read()[1:]
        _checkpoint(root / "output-checkpoint.jsonl", checkpoint)
    elif mutation == "overlap":
        raw["overlap_gate"]["valid"] = False
    elif mutation == "one-rank":
        raw["phase4b2_final_sync"].pop()
    elif mutation == "cleanup":
        process["cleanup_valid"] = False
    elif mutation == "crash-status":
        process["target_exit_status"] = 137
    elif mutation == "launch-error":
        process["launch_error"] = "failed launch"
    elif mutation == "survivors":
        process["remaining_owned_pids"] = [123]
    elif mutation == "command":
        process["command"][10] = "unrelated-coordinator"
    elif mutation == "traceback":
        (root / "target.log").write_text("Traceback (most recent call last)\nEngineDeadError\n")
    elif mutation == "state-error":
        states = CheckpointJsonl(root / "request-state-events.jsonl").read()
        states[2]["source_state"] = "BOOTSTRAP"
        _checkpoint(root / "request-state-events.jsonl", states)
    elif mutation == "retired-summary":
        raw["retired_ready_results"]["retired_proposal_drop_count"] = 0
    elif mutation == "raw-output":
        raw["outputs"][0]["finish_reason"] = "abort"
    elif mutation == "draft-failure":
        raw["draft_shutdown"]["failures"] = {AFFECTED: "failure"}
    else:
        _json(disk_run["config_path"], {"changed": True})
    _json(root / "resident-dual.json", raw)
    _json(root / "process-lifecycle.json", process)
    with pytest.raises((ValueError, KeyError, TypeError)):
        recover_terminal_state(**disk_run, output_dir=tmp_path / "recovery")
    assert not (tmp_path / "recovery").exists()


def test_certificate_is_recomputed_and_cannot_override_tampered_raw(disk_run, tmp_path):
    recover_terminal_state(**disk_run, output_dir=tmp_path / "recovery")
    certificate = tmp_path / "recovery/terminal-state-revalidation.json"
    proof = json.loads(certificate.read_text())
    proof["terminal_state_reconciliation"]["events"][0]["committed_prefix_sha256"] = "f" * 64
    _json(certificate, proof)
    with pytest.raises(ValueError, match="certificate disagrees"):
        validate_recovery_certificate(certificate, **disk_run)


def test_recovery_cannot_write_into_raw_root_or_override_other_modes(disk_run, tmp_path):
    with pytest.raises(ValueError, match="outside"):
        recover_terminal_state(**disk_run, output_dir=disk_run["run_root"] / "recovery")
    with pytest.raises(ValueError, match="restricted to Dual"):
        build_decode_performance_result(
            mode="serial",
            **disk_run,
            output_path=tmp_path / "out.json",
            terminal_revalidation_path=tmp_path / "fake.json",
        )


def test_raw_invalid_stays_invalid_without_explicit_revalidation(disk_run, tmp_path):
    report = build_decode_performance_result(
        mode="dual-batch", **disk_run, output_path=tmp_path / "invalid.json"
    )
    assert report["valid"] is False
    assert "underlying resident run is invalid" in report["errors"]
    assert "owned Target process run is invalid" in report["errors"]


def test_recovery_cli_is_cpu_only_and_requires_explicit_source(disk_run, tmp_path):
    command = ["phase4b2-reconcile-dual-terminal", "--output-dir", str(tmp_path / "recovery")]
    for key, value in disk_run.items():
        flag = "--" + key.removesuffix("_path").replace("_", "-")
        command.extend((flag, str(value)))
    assert cli.main(command) == 0
    proof = audit_terminal_recovery(**disk_run, observation_ns=10_000_000)
    assert proof["terminal_state_reconciliation"]["reconciled_request_ids"] == [AFFECTED]
