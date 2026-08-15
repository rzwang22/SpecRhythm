from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import pytest
from test_phase4 import CONFIG, WORKLOAD

from specrhythm import cli
from specrhythm.phase4.config import load_phase4_config
from specrhythm.phase4.draft_service import DraftStateMachine
from specrhythm.phase4.reference import (
    _exclusive_freeze,
    build_stock_reference,
    build_target_regression,
    validate_stock_reference,
)
from specrhythm.phase4.serial import (
    PROTOCOL_VERSION,
    AcceptanceDecision,
    Proposal,
    SerialTimeline,
    greedy_acceptance,
    token_prefix_hash,
)
from specrhythm.phase4.serial_runner import (
    PATCHED_VLLM_RUNNER_SHA256,
    _timeline_rows_valid,
    load_patch_manifest,
    validate_engine_residency,
)
from specrhythm.phase4.transport import CheckpointJsonl
from specrhythm.phase4.vllm_remote import _assert_target_information_isolated


@pytest.fixture
def phase4_config(tmp_path, monkeypatch):
    draft = tmp_path / "Qwen3-0.6B"
    target = tmp_path / "Qwen3-32B"
    for path, model_type in ((draft, "qwen3-draft"), (target, "qwen3-target")):
        path.mkdir()
        (path / "config.json").write_text(
            json.dumps({"model_type": model_type}), encoding="utf-8"
        )
        (path / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "Qwen3Tokenizer"}), encoding="utf-8"
        )
    monkeypatch.setenv("SR_DRAFT_MODEL", str(draft))
    monkeypatch.setenv("SR_TARGET_MODEL", str(target))
    return load_phase4_config(str(CONFIG))


class RecordingDraftBackend:
    backend_name = "recording-test-draft"

    def __init__(self, proposals: Sequence[Sequence[int]]) -> None:
        self.proposals = [tuple(row) for row in proposals]
        self.prefixes: dict[str, Tuple[int, ...]] = {}
        self.pending: dict[str, Tuple[int, ...]] = {}
        self.initializations = 0
        self.propose_calls = 0
        self.rollback_calls = []
        self.appended = []
        self.finished = set()
        self.shutdown_called = False

    @property
    def provenance(self) -> Mapping[str, Any]:
        return {
            "backend": self.backend_name,
            "full_context_prefill_per_request": 1,
            "full_context_replay_per_round": False,
            "persistent_cross_round_kv": True,
        }

    def initialize(self, request_id: str, committed_token_ids: Sequence[int]) -> None:
        self.initializations += 1
        self.prefixes[request_id] = tuple(committed_token_ids)

    def propose(
        self, request_id: str, budget: int, eos_token_ids: Sequence[int]
    ) -> tuple[Tuple[int, ...], int]:
        del eos_token_ids
        self.propose_calls += 1
        proposal = self.proposals.pop(0)[:budget]
        self.pending[request_id] = proposal
        return proposal, max(len(proposal) - 1, 0)

    def rollback(self, request_id: str, accepted_draft_tokens: int) -> None:
        proposal = self.pending[request_id]
        self.prefixes[request_id] += proposal[:accepted_draft_tokens]
        self.rollback_calls.append((request_id, accepted_draft_tokens))

    def append_target_token(self, request_id: str, token_id: int) -> None:
        self.prefixes[request_id] += (token_id,)
        self.appended.append((request_id, token_id))
        self.pending.pop(request_id, None)

    def finish(self, request_id: str) -> None:
        self.finished.add(request_id)
        self.pending.pop(request_id, None)

    def shutdown(self) -> None:
        self.shutdown_called = True


def initialize(machine: DraftStateMachine, prefix=(1, 2)):
    return machine.initialize("request", prefix, token_prefix_hash(prefix))


def propose(machine: DraftStateMachine, round_id=0, prefix=(1, 2), remaining=8):
    result = machine.batch_propose(
        [
            {
                "request_id": "request",
                "round_id": round_id,
                "committed_prefix_len": len(prefix),
                "committed_prefix_hash": token_prefix_hash(prefix),
                "remaining_output_budget": remaining,
                "eos_token_ids": [99],
            }
        ]
    )
    return result["proposals"]


@pytest.mark.parametrize(
    ("proposal_tokens", "committed", "terminal", "accepted", "rejected", "correction", "bonus"),
    [
        ((10, 11, 12), (20,), False, (), (10, 11, 12), (20,), ()),
        ((10, 11, 12), (10, 20), False, (10,), (11, 12), (20,), ()),
        ((10, 11), (10, 11, 20), False, (10, 11), (), (), (20,)),
        ((10, 99), (10, 99), True, (10, 99), (), (), ()),
        ((10, 11), (10, 99), True, (10,), (11,), (99,), ()),
    ],
)
def test_greedy_acceptance_cases(
    proposal_tokens,
    committed,
    terminal,
    accepted,
    rejected,
    correction,
    bonus,
):
    result = greedy_acceptance(proposal_tokens, committed, terminal=terminal)
    assert result.accepted_draft_token_ids == accepted
    assert result.rejected_draft_token_ids == rejected
    assert result.target_correction_token_ids == correction
    assert result.target_bonus_token_ids == bonus
    assert result.accounting["proposed_tokens"] == len(proposal_tokens)
    assert result.accounting["committed_tokens"] == len(committed)


def test_zero_proposal_target_tail_and_max_boundary():
    result = greedy_acceptance((), (42,), terminal=True)
    assert result.target_correction_token_ids == (42,)
    backend = RecordingDraftBackend([(10,)])
    machine = DraftStateMachine(backend)
    initialize(machine)
    assert propose(machine, remaining=1) == []
    assert backend.propose_calls == 0


def test_correction_and_bonus_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        AcceptanceDecision((), (), (1,), (2,), (1, 2), False)


def test_state_machine_rolls_back_and_appends_target_correction():
    backend = RecordingDraftBackend([(10, 11, 12)])
    machine = DraftStateMachine(backend)
    initialize(machine)
    rows = propose(machine)
    proposal_value = Proposal.from_dict(rows[0])
    assert proposal_value.protocol_version == PROTOCOL_VERSION
    assert proposal_value.transport_payload_bytes > 0
    final = (1, 2, 10, 20)
    classified = machine.synchronize_committed_prefix(
        "request", 0, (10, 20), token_prefix_hash(final), terminal=False
    )
    assert classified["decision"]["accepted_draft_tokens"] == 1
    machine.rollback_rejected_suffix("request", 0)
    with pytest.raises(ValueError, match="already rolled back"):
        machine.rollback_rejected_suffix("request", 0)
    state = machine.append_target_correction_or_bonus("request", 0)
    assert backend.rollback_calls == [("request", 1)]
    assert backend.appended == [("request", 20)]
    assert backend.prefixes["request"] == final
    assert state["committed_prefix_hash"] == token_prefix_hash(final)


def test_full_acceptance_appends_bonus_and_preserves_prefix_hash():
    backend = RecordingDraftBackend([(10, 11)])
    machine = DraftStateMachine(backend)
    initialize(machine)
    propose(machine)
    final = (1, 2, 10, 11, 20)
    machine.synchronize_committed_prefix(
        "request", 0, (10, 11, 20), token_prefix_hash(final), terminal=False
    )
    machine.rollback_rejected_suffix("request", 0)
    state = machine.append_target_correction_or_bonus("request", 0)
    assert state["committed_prefix_len"] == len(final)
    assert backend.prefixes["request"] == final


def test_draft_eos_finishes_and_blocks_later_proposal():
    backend = RecordingDraftBackend([(10, 99), (30,)])
    machine = DraftStateMachine(backend)
    initialize(machine)
    propose(machine)
    final = (1, 2, 10, 99)
    machine.synchronize_committed_prefix(
        "request", 0, (10, 99), token_prefix_hash(final), terminal=True
    )
    machine.rollback_rejected_suffix("request", 0)
    machine.append_target_correction_or_bonus("request", 0)
    assert "request" in backend.finished
    with pytest.raises(ValueError, match="finished"):
        propose(machine, 1, final)


def test_cancelled_request_cannot_reenter_draft_batch():
    backend = RecordingDraftBackend([(10, 11)])
    machine = DraftStateMachine(backend)
    initialize(machine)
    assert machine.cancel("request")["cancelled"] is True
    with pytest.raises(ValueError, match="finished"):
        propose(machine)


def test_stale_duplicate_out_of_order_and_hash_mismatch_are_rejected():
    backend = RecordingDraftBackend([(10, 11)])
    machine = DraftStateMachine(backend)
    initialize(machine)
    with pytest.raises(ValueError, match="out-of-order"):
        propose(machine, round_id=1)
    propose(machine)
    with pytest.raises(ValueError, match="unverified"):
        propose(machine)
    with pytest.raises(ValueError, match="hash mismatch"):
        machine.synchronize_committed_prefix("request", 0, (20,), "bad", terminal=False)


def test_strict_serial_timeline_rejects_overlap():
    SerialTimeline(1, 2, 2, 3, 3, 4, 4, 5, 5)
    with pytest.raises(ValueError, match="overlap"):
        SerialTimeline(1, 3, 2, 4, 4, 5, 5, 6, 6)

    first = {
        "timeline": dict(
            zip(
                (
                    "draft_start_ns",
                    "draft_end_ns",
                    "transfer_start_ns",
                    "transfer_end_ns",
                    "verify_start_ns",
                    "verify_end_ns",
                    "state_sync_start_ns",
                    "state_sync_end_ns",
                    "next_round_draft_start_ns",
                ),
                (1, 2, 2, 3, 5, 6, 6, 7, 7),
            )
        )
    }
    second = {
        "timeline": dict(
            zip(
                first["timeline"],
                (5, 7, 7, 8, 8, 9, 9, 10, 10),
            )
        )
    }
    assert not _timeline_rows_valid([first, second])


def test_target_information_isolation():
    _assert_target_information_isolated(
        {"committed_delta": [1, 2], "committed_prefix_hash": "abc"}
    )
    with pytest.raises(ValueError, match="leaked"):
        _assert_target_information_isolated({"target_logits": [0.1, 0.2]})


def test_no_full_context_replay_and_one_initialization():
    backend = RecordingDraftBackend([(10, 11), (12, 13)])
    machine = DraftStateMachine(backend)
    initialize(machine)
    propose(machine)
    final = (1, 2, 20)
    machine.synchronize_committed_prefix(
        "request", 0, (20,), token_prefix_hash(final), terminal=False
    )
    machine.rollback_rejected_suffix("request", 0)
    machine.append_target_correction_or_bonus("request", 0)
    propose(machine, round_id=1, prefix=final)
    assert backend.initializations == 1
    assert backend.provenance["full_context_replay_per_round"] is False


def test_checkpoint_detects_partial_and_checksum_corruption(tmp_path):
    path = tmp_path / "events.jsonl"
    log = CheckpointJsonl(path)
    log.append({"request_id": "one", "round_id": 0})
    assert log.read()[0]["request_id"] == "one"
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="partial"):
        log.read()


def smoke_report():
    outputs = [
        {
            "request_id": f"r{index}",
            "generated_token_ids": [index + 1],
            "finish_reason": "length",
            "stop_reason": None,
            "top_logprobs": [],
        }
        for index in range(5)
    ]
    return {
        "role": "target",
        "repeated_run_deterministic": True,
        "sampling": {"temperature": 0.0},
        "runs": [outputs, json.loads(json.dumps(outputs))],
        "frozen_hf_target_comparison": {
            "performed": True,
            "all_tokens_equal": False,
        },
    }


def test_stock_reference_is_immutable_and_hf_is_advisory(phase4_config, tmp_path):
    rows = [json.loads(line) for line in WORKLOAD.read_text().splitlines()]
    smoke = smoke_report()
    for output, row in zip(smoke["runs"][0], rows):
        output["request_id"] = row["request_id"]
    for output, row in zip(smoke["runs"][1], rows):
        output["request_id"] = row["request_id"]
    reference = build_stock_reference(
        smoke, phase4_config, workload_path=WORKLOAD, git_commit="f" * 40
    )
    assert not validate_stock_reference(reference)
    assert reference["legacy_hf_trajectory"]["can_fail_serving_correctness"] is False
    assert len(reference["workload"]["requests"]) == 5
    assert reference["workload"]["requests"][0]["prompt_token_ids"]
    path = tmp_path / "stock-target-reference.json"
    _exclusive_freeze(path, reference)
    with pytest.raises(FileExistsError):
        _exclusive_freeze(path, reference)


def test_stock_reference_rejects_nondeterministic_repeated_run(
    phase4_config,
):
    rows = [json.loads(line) for line in WORKLOAD.read_text().splitlines()]
    smoke = smoke_report()
    for run in smoke["runs"]:
        for output, row in zip(run, rows):
            output["request_id"] = row["request_id"]
    smoke["runs"][1][2]["generated_token_ids"] = [999]
    with pytest.raises(ValueError, match="different token IDs"):
        build_stock_reference(
            smoke, phase4_config, workload_path=WORKLOAD, git_commit="f" * 40
        )


def test_patched_target_only_must_equal_stock(phase4_config):
    rows = [json.loads(line) for line in WORKLOAD.read_text().splitlines()]
    smoke = smoke_report()
    for run in smoke["runs"]:
        for output, row in zip(run, rows):
            output["request_id"] = row["request_id"]
    reference = build_stock_reference(
        smoke, phase4_config, workload_path=WORKLOAD, git_commit="f" * 40
    )
    patch = {"patch_applied": True}
    regression = build_target_regression(smoke, reference, patch_manifest=patch)
    assert regression["valid"]
    nondeterministic = json.loads(json.dumps(smoke))
    nondeterministic["repeated_run_deterministic"] = False
    assert not build_target_regression(
        nondeterministic, reference, patch_manifest=patch
    )["valid"]
    changed = json.loads(json.dumps(smoke))
    changed["runs"][0][0]["generated_token_ids"] = [999]
    assert not build_target_regression(changed, reference, patch_manifest=patch)["valid"]


def test_patch_manifest_is_pinned_to_apply_operation(phase4_config, tmp_path):
    path = tmp_path / "patch.json"
    value = {
        "schema_version": "specrhythm.vllm-base-and-patch-manifest.v1",
        "operation": "apply",
        "vllm_base_commit": phase4_config.expected_vllm_commit,
        "verified_source_commit": phase4_config.expected_vllm_commit,
        "patch_applied": True,
        "python_only": True,
        "cpp_cuda_modified": False,
        "target_only_behavior_change_when_speculation_disabled": False,
        "patch_sha256": "a" * 64,
        "target_file_sha256_after": PATCHED_VLLM_RUNNER_SHA256,
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    assert load_patch_manifest(path, phase4_config)["operation"] == "apply"
    value["operation"] = "check"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="apply operation"):
        load_patch_manifest(path, phase4_config)


def test_wrong_model_or_gpu_residency_fails(phase4_config):
    ready = {
        "provenance": {
            "physical_gpu_id": 1,
            "model": {"path": str(phase4_config.target.resolved_model_path)},
            "parameter_count": 10,
            "full_context_replay_per_round": False,
        }
    }
    workers = [
        {
            "global_rank": rank,
            "world_size": 2,
            "physical_gpu_id": physical,
            "parameter_count": 10,
            "parameter_bytes": 20,
            "allocated_memory_bytes": 30,
            "gpu_uuid": f"GPU-{physical}",
            "attention_backends": ["FLASH_ATTN"],
            "all_parameters_on_expected_device": True,
        }
        for rank, physical in enumerate((1, 2))
    ]
    errors = validate_engine_residency(
        phase4_config,
        draft_ready=ready,
        target_worker_ranks=workers,
        proposer_report={"proposer_model_parameter_count": 1},
    )
    assert any("Target model" in error for error in errors)
    assert any("Target GPU" in error for error in errors)


def test_cpu_cannot_emit_draft_gpu_result(phase4_config, tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(SystemExit, match="Draft service failed"):
        cli.main(
            [
                "phase4-draft-service",
                "--config",
                str(phase4_config.path),
                "--socket",
                str(tmp_path / "draft.sock"),
                "--event-log",
                str(tmp_path / "events.jsonl"),
                "--ready",
                str(tmp_path / "ready.json"),
            ]
        )
    assert not (tmp_path / "ready.json").exists()


def test_phase4_config_freezes_k4_and_v1(phase4_config):
    assert phase4_config.proposal_budget == 4
    assert phase4_config.target_model_runner == "v1"
    with pytest.raises(ValueError, match="K=4"):
        replace(phase4_config, proposal_budget=3)


def test_phase4_serial_sources_parse_as_python39():
    import ast

    root = Path(__file__).parents[1]
    for name in (
        "serial.py",
        "transport.py",
        "draft_service.py",
        "reference.py",
        "vllm_remote.py",
        "serial_runner.py",
        "serial_validation.py",
    ):
        path = root / "src" / "specrhythm" / "phase4" / name
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 9))
