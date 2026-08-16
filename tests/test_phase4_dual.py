from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import pytest

from specrhythm import cli
from specrhythm.phase4.dual import (
    DualCycle,
    DualProposal,
    DualRequest,
    ProposalReadyQueue,
    RequestState,
    form_dynamic_microbatches,
    proposal_identity,
    token_prefix_hash,
)
from specrhythm.phase4.dual_runner import (
    _ordered_checkpoint_rows,
    _validate_request_identity_report,
    _write_checkpoint_rows,
    build_cycle_and_overlap_events,
)
from specrhythm.phase4.dual_service import (
    AsyncDualDraftController,
    DualDraftMachine,
)
from specrhythm.phase4.dual_validation import _validate_identity_domains
from specrhythm.phase4.request_identity import (
    FrozenPromptIdentityMap,
    resolve_stable_ready_request,
)
from specrhythm.phase4.transport import CheckpointJsonl


def make_request(request_id="A", prefix=(1, 2), *, maximum=16):
    request = DualRequest(request_id, prefix, maximum)
    request.finish_bootstrap((3,), terminal=False)
    return request


def make_proposal(request: DualRequest, tokens=(10, 11), **overrides):
    values = {
        "request_id": request.request_id,
        "round_id": request.next_round_id,
        "prefix_version": request.prefix_version,
        "prefix_token_count": request.committed_token_count,
        "prefix_token_sha256": request.prefix_sha256,
        "draft_kv_length_before": request.committed_token_count,
        "draft_kv_length_after": request.committed_token_count + len(tokens),
        "proposal_token_ids": tuple(tokens),
        "created_timestamp_ns": 2,
        "draft_start_ns": 1,
        "draft_end_ns": 2,
    }
    values.update(overrides)
    values.setdefault(
        "proposal_id",
        proposal_identity(
            values["request_id"],
            values["round_id"],
            values["prefix_version"],
            values["proposal_token_ids"],
        ),
    )
    return DualProposal(**values)


def test_legal_state_machine_correction_rollback_and_sync():
    request = make_request()
    request.start_draft()
    request.publish_proposal(make_proposal(request, (10, 11, 12)))
    request.mark_verify_ready()
    request.start_verify()
    decision = request.commit((10, 20), terminal=False)
    assert decision.accepted_draft_token_ids == (10,)
    assert decision.rejected_draft_token_ids == (11, 12)
    assert request.state is RequestState.DRAFT_SYNC
    request.complete_draft_sync(request.committed_token_count)
    assert request.state is RequestState.DRAFT_READY
    assert request.prefix_version == 2


def test_illegal_transition_and_same_request_concurrency_rejected():
    request = make_request()
    with pytest.raises(ValueError, match="illegal"):
        request.transition(RequestState.VERIFYING)
    request.start_draft()
    with pytest.raises(ValueError, match="no proposal"):
        request.start_verify()
    with pytest.raises(ValueError, match="illegal"):
        request.start_draft()


def test_one_inflight_proposal_and_stale_parent_rejected():
    request = make_request()
    request.start_draft()
    request.publish_proposal(make_proposal(request))
    with pytest.raises(ValueError, match="only be published"):
        request.publish_proposal(make_proposal(request, (12,)))
    request.mark_verify_ready()
    request.committed_token_ids += (99,)
    with pytest.raises(ValueError, match="stale proposal"):
        request.start_verify()


def test_ready_queue_nonblocking_and_stale_discard():
    first = make_request("A")
    second = make_request("B", (4, 5))
    first.start_draft()
    second.start_draft()
    good = make_proposal(first)
    stale = make_proposal(second, prefix_token_sha256="f" * 64)
    queue = ProposalReadyQueue()
    queue.publish(good)
    queue.publish(stale)
    started = time.monotonic()
    ready, discarded = queue.take({"A": first, "B": second}, limit=4)
    assert time.monotonic() - started < 0.1
    assert ready == [good]
    assert discarded[0]["proposal_id"] == stale.proposal_id


def test_dynamic_microbatch_swap_and_empty_side():
    first = make_request("A")
    second = make_request("B", (4, 5))
    first.start_draft()
    first.publish_proposal(make_proposal(first))
    assert form_dynamic_microbatches([first, second], microbatch_size=1) == (("B",), ("A",))
    second.start_draft()
    second.publish_proposal(make_proposal(second, (20,)))
    first.mark_verify_ready()
    first.start_verify()
    first.commit((10, 11, 30), terminal=False)
    first.complete_draft_sync(first.committed_token_count)
    assert form_dynamic_microbatches([first, second], microbatch_size=1) == (("A",), ("B",))
    first.start_draft()
    assert form_dynamic_microbatches([first], microbatch_size=1) == ((), ())


def test_overlap_interval_has_no_artificial_duration():
    no_overlap = DualCycle(0, "D0", "V0", ("B",), ("A",), 1, 2, 2, 3, 3, 4)
    assert no_overlap.overlap_duration_ns == 0
    overlap = DualCycle(1, "D1", "V1", ("A",), ("B",), 5, 15, 10, 20, 20, 21)
    assert overlap.overlap_interval == (10, 15)
    with pytest.raises(ValueError, match="not disjoint"):
        DualCycle(2, "D", "V", ("A",), ("A",), 1, 2, 1, 2, 2, 3)


class RecordingBackend:
    backend_name = "test-persistent"

    def __init__(self, proposals: Sequence[Sequence[int]]) -> None:
        self.proposals = [tuple(item) for item in proposals]
        self.prefixes: dict[str, Tuple[int, ...]] = {}
        self.pending: dict[str, Tuple[int, ...]] = {}
        self.rollback_calls = []
        self.appended = []

    @property
    def provenance(self) -> Mapping[str, Any]:
        return {"physical_gpu_id": 0, "persistent_cross_round_kv": True}

    def initialize(self, request_id: str, committed_token_ids: Sequence[int]) -> None:
        self.prefixes[request_id] = tuple(committed_token_ids)

    def propose(self, request_id: str, budget: int, eos_token_ids: Sequence[int]):
        del eos_token_ids
        proposal = self.proposals.pop(0)[:budget]
        self.pending[request_id] = proposal
        return proposal, len(proposal)

    def rollback(self, request_id: str, accepted_draft_tokens: int) -> None:
        proposal = self.pending[request_id]
        self.prefixes[request_id] += proposal[:accepted_draft_tokens]
        self.rollback_calls.append((request_id, accepted_draft_tokens))

    def append_target_token(self, request_id: str, token_id: int) -> None:
        self.prefixes[request_id] += (token_id,)
        self.pending.pop(request_id, None)
        self.appended.append((request_id, token_id))

    def finish(self, request_id: str) -> None:
        self.pending.pop(request_id, None)

    def shutdown(self) -> None:
        pass


def test_async_draft_controller_correction_bonus_and_one_inflight(tmp_path):
    backend = RecordingBackend([(10, 11), (30, 31)])
    machine = DualDraftMachine(backend)
    controller = AsyncDualDraftController(machine, CheckpointJsonl(tmp_path / "work.jsonl"))
    prefix = (1, 2, 3)
    row = {
        "request_id": "A",
        "committed_token_ids": list(prefix),
        "prefix_version": 1,
        "prefix_token_sha256": token_prefix_hash(prefix),
        "remaining_output_budget": 8,
        "eos_token_ids": [99],
        "terminal": False,
    }
    assert controller.enqueue("bootstrap_and_propose", [row])["blocking_on_draft_gpu"] is False
    with pytest.raises(ValueError, match="in flight"):
        controller.enqueue("bootstrap_and_propose", [row])
    ready = _wait_ready(controller)
    proposal = ready["proposal"]
    final = prefix + (10, 20)
    commit = {
        "request_id": "A",
        "proposal_id": proposal["proposal_id"],
        "round_id": 0,
        "committed_delta": [10, 20],
        "prefix_version": 2,
        "prefix_token_sha256": token_prefix_hash(final),
        "remaining_output_budget": 6,
        "eos_token_ids": [99],
        "terminal": False,
    }
    controller.enqueue("commit_and_propose", [commit])
    second = _wait_ready(controller)
    assert second["proposal"]["round_id"] == 1
    assert backend.rollback_calls == [("A", 1)]
    assert backend.appended == [("A", 20)]
    assert controller.shutdown()["shutdown"] is True


def test_build_overlap_uses_disjoint_real_intervals():
    draft = [
        {
            "request_id": "B",
            "result": {
                "proposal": {"proposal_id": "p", "round_id": 0},
                "draft_gpu_interval": {
                    "host_start_ns": 10,
                    "host_end_ns": 30,
                    "physical_gpu_id": 0,
                    "cuda_elapsed_ns": 15,
                },
            },
        }
    ]
    verify = [
        {
            "verify_microbatch_id": "v0",
            "verify_request_ids": ["A"],
            "verify_host_start_ns": 20,
            "verify_host_end_ns": 40,
            "target_physical_gpu_ids": [1, 2],
            "target_rank_intervals": [{"cuda_events": True}, {"cuda_events": True}],
        }
    ]
    cycles, overlaps = build_cycle_and_overlap_events(draft, verify)
    assert cycles[0]["overlap_duration_ns"] == 10
    assert overlaps[0]["request_sets_disjoint"] is True
    assert overlaps[0]["draft_physical_gpu_ids"] == [0]


def test_dual_contract_cli_is_cuda_free_and_target_only_default_unchanged(tmp_path):
    output = tmp_path / "dual-dry-run.json"
    assert cli.main(["phase4-dual-contract-dry-run", "--output", str(output)]) == 0
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["gpu_execution_performed"] is False
    assert value["dynamic_swap"] is True
    assert value["target_only_default_behavior_changed"] is False
    assert value["vllm_imported"] is False


def test_request_checkpoint_resume_is_keyed_and_non_overwriting(tmp_path):
    path = tmp_path / "outputs.jsonl"
    first = {"request_id": "A", "generated_token_ids": [1], "finish_reason": "length"}
    second = {"request_id": "B", "generated_token_ids": [2], "finish_reason": "stop"}
    log = CheckpointJsonl(path)
    log.append(second)
    log.append(first)
    definitions = [
        type("Request", (), {"request_id": "A"})(),
        type("Request", (), {"request_id": "B"})(),
    ]
    assert [row["request_id"] for row in _ordered_checkpoint_rows(path, definitions)] == [
        "A",
        "B",
    ]
    derived = tmp_path / "cycles.jsonl"
    _write_checkpoint_rows(derived, [{"cycle_id": 0}], resume=False)
    before = derived.read_bytes()
    _write_checkpoint_rows(derived, [{"cycle_id": 0}], resume=True)
    assert derived.read_bytes() == before
    with pytest.raises(FileExistsError):
        _write_checkpoint_rows(derived, [{"cycle_id": 1}], resume=True)


def test_finished_request_cannot_reenter_draft_or_verify():
    request = DualRequest("A", (1, 2), 1)
    request.finish_bootstrap((3,), terminal=True)
    assert request.state is RequestState.FINISHED
    with pytest.raises(ValueError, match="illegal"):
        request.start_draft()
    with pytest.raises(ValueError, match="no proposal"):
        request.start_verify()


def test_opaque_internal_id_maps_to_stable_prompt_without_suffix_parsing():
    identity = FrozenPromptIdentityMap({"r3-abc": (1, 2), "r3-def": (3, 4)})
    internal_id = "r3-abc-9997a85a"
    states = {identity.bind(internal_id, (1, 2, 10)): {"prefix_version": 1}}
    assert set(states) == {"r3-abc"}
    assert identity.stable_id(internal_id) == "r3-abc"
    assert identity.internal_id("r3-abc") == internal_id


def test_internal_id_binding_is_stable_and_prompt_changes_fail_closed():
    identity = FrozenPromptIdentityMap({"A": (1, 2), "B": (3, 4)})
    assert identity.bind("opaque", (1, 2, 9)) == "A"
    assert identity.bind("opaque", (1, 2, 10)) == "A"
    with pytest.raises(RuntimeError, match="changed stable prompt identity"):
        identity.bind("opaque", (3, 4, 11))


def test_prompt_identity_rejects_zero_and_multiple_matches():
    identity = FrozenPromptIdentityMap({"short": (1,), "long": (1, 2)})
    with pytest.raises(RuntimeError, match="no frozen workload prompt"):
        identity.match((9, 9))
    with pytest.raises(RuntimeError, match="ambiguously"):
        identity.match((1, 2, 3))
    with pytest.raises(ValueError, match="unique frozen prompt"):
        FrozenPromptIdentityMap({"A": (1, 2), "B": (1, 2)})


def test_two_internal_ids_cannot_alias_one_stable_request():
    identity = FrozenPromptIdentityMap({"A": (1, 2)})
    identity.bind("internal-A-1", (1, 2, 3))
    with pytest.raises(RuntimeError, match="alias one stable request"):
        identity.bind("internal-A-2", (1, 2, 4))


def test_scheduler_handoff_resolves_only_the_mapped_internal_request():
    identity = FrozenPromptIdentityMap({"A": (1, 2), "B": (3, 4)})
    internal_a = "A-engine-suffix"
    internal_b = "B-engine-suffix"
    identity.bind(internal_a, (1, 2, 10))
    identity.bind(internal_b, (3, 4, 20))
    request_a = type("VllmRequest", (), {"request_id": internal_a})()
    request_b = type("VllmRequest", (), {"request_id": internal_b})()
    internal_id, request = resolve_stable_ready_request(
        "A", identity, {internal_a: request_a, internal_b: request_b}
    )
    assert internal_id == internal_a
    assert request is request_a
    with pytest.raises(RuntimeError, match="no mapped vLLM request"):
        resolve_stable_ready_request("A", identity, {internal_b: request_b})


def test_identity_translation_keeps_stale_prefix_guard_fail_closed():
    identity = FrozenPromptIdentityMap({"A": (1, 2)})
    assert identity.bind("A-engine-suffix", (1, 2, 3)) == "A"
    request = make_request("A", (1, 2))
    request.start_draft()
    request.publish_proposal(make_proposal(request))
    request.mark_verify_ready()
    request.committed_token_ids += (99,)
    with pytest.raises(ValueError, match="stale proposal"):
        request.start_verify()


def test_identity_report_requires_one_to_one_frozen_workload_coverage():
    requests = [
        type("Request", (), {"request_id": "A"})(),
        type("Request", (), {"request_id": "B"})(),
    ]
    report = {
        "request_identity": {
            "mapping_source": "unique frozen prompt_token_ids",
            "suffix_parsing": False,
            "bound_request_count": 2,
            "bindings": [
                {"internal_request_id": "A-x", "request_id": "A"},
                {"internal_request_id": "B-y", "request_id": "B"},
            ],
        }
    }
    assert _validate_request_identity_report(report, requests) == []
    report["request_identity"]["bindings"][1]["request_id"] = "A"
    assert any(
        "alias" in error
        for error in _validate_request_identity_report(report, requests)
    )


def test_dual_validator_rejects_internal_ids_in_stable_event_fields():
    identity = {
        "mapping_source": "unique frozen prompt_token_ids",
        "suffix_parsing": False,
        "bindings": [{"internal_request_id": "A-suffix", "request_id": "A"}],
    }
    run = {"outputs": [{"request_id": "A"}], "request_identity": identity}
    assert _validate_identity_domains(run, [], [], [], [], []) == []
    errors = _validate_identity_domains(
        run,
        [{"request_id": "A-suffix"}],
        [],
        [],
        [],
        [],
    )
    assert any("non-stable request IDs" in error for error in errors)


def test_target_failure_terminates_draft_without_unbounded_wait(tmp_path):
    helper = (
        Path(__file__).parents[1]
        / "integrations"
        / "vllm"
        / "phase4b_run_helpers.sh"
    )
    environment = dict(os.environ)
    environment.update(
        {
            "PHASE4B_HELPER": str(helper),
            "TARGET_LOG": str(tmp_path / "target.log"),
            "DRAFT_LOG": str(tmp_path / "draft.log"),
            "PHASE4B_CLEANUP_POLLS": "5",
            "PHASE4B_CLEANUP_SLEEP_SECONDS": "0.01",
            "PHASE4B_PYTHON": sys.executable,
        }
    )
    completed = subprocess.run(
        [
            "bash",
            "-c",
            """
source "$PHASE4B_HELPER"
: >"$DRAFT_LOG"
sleep 60 &
draft_pid="$!"
if phase4b_run_target_with_cleanup "$draft_pid" "$TARGET_LOG" "$DRAFT_LOG" -- \
    bash -c 'echo target-failed; exit 7'; then
  target_status=0
else
  target_status="$?"
fi
test "$target_status" -eq 7
! kill -0 "$draft_pid" 2>/dev/null
""",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "target-failed" in completed.stderr


def _wait_ready(controller: AsyncDualDraftController) -> Mapping[str, Any]:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        result = controller.poll_ready(1)
        if result["ready"]:
            return result["ready"][0]
        time.sleep(0.001)
    raise AssertionError("asynchronous proposal did not become ready")
