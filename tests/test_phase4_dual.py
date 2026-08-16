from __future__ import annotations

import json
import time
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
    _write_checkpoint_rows,
    build_cycle_and_overlap_events,
)
from specrhythm.phase4.dual_service import (
    AsyncDualDraftController,
    DualDraftMachine,
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


def _wait_ready(controller: AsyncDualDraftController) -> Mapping[str, Any]:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        result = controller.poll_ready(1)
        if result["ready"]:
            return result["ready"][0]
        time.sleep(0.001)
    raise AssertionError("asynchronous proposal did not become ready")
