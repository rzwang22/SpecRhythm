from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from specrhythm.phase4.resident_initial_proposal import (
    InitialProposalState,
    ResidentInitialProposalLifecycle,
    validate_initial_proposal_lifecycle_events,
)
from specrhythm.phase4.resident_setup import resident_admission_decision
from specrhythm.phase4.serial import PROTOCOL_VERSION, Proposal, token_prefix_hash


class FakeRequest:
    def __init__(self, request_id: str, prefix: Sequence[int]) -> None:
        self.request_id = request_id
        self.all_token_ids = list(prefix)
        self.num_output_tokens = 1
        self.num_computed_tokens = len(prefix) - 1
        self._spec_token_ids: list[int] = []
        self.spec_assignments = 0
        self.finished = False

    @property
    def spec_token_ids(self) -> list[int]:
        return self._spec_token_ids

    @spec_token_ids.setter
    def spec_token_ids(self, value: Sequence[int]) -> None:
        self.spec_assignments += 1
        self._spec_token_ids = list(value)

    def is_finished(self) -> bool:
        return self.finished


def _proposal(request_id: str, prefix: Sequence[int], tokens=(90, 91)) -> Proposal:
    return Proposal(
        protocol_version=PROTOCOL_VERSION,
        request_id=request_id,
        round_id=0,
        parent_prefix_len=len(prefix),
        parent_prefix_hash=token_prefix_hash(prefix),
        proposal_token_ids=tuple(tokens),
        proposal_eos=False,
        draft_start_ns=100,
        draft_end_ns=101,
        transport_payload_bytes=10,
        model_provenance={"model": "draft"},
        runtime_provenance={"device": 0},
    )


def _lifecycle(prefixes=None):
    prefixes = prefixes or {"A": (1, 2, 3), "B": (4, 5, 6)}
    requests = {
        f"internal-{request_id}": FakeRequest(f"internal-{request_id}", prefix)
        for request_id, prefix in prefixes.items()
    }
    events = []
    lifecycle = ResidentInitialProposalLifecycle(
        expected_request_ids=tuple(prefixes),
        stable_to_internal_request_id={
            request_id: f"internal-{request_id}" for request_id in prefixes
        },
        proposals=tuple(
            _proposal(request_id, prefix) for request_id, prefix in prefixes.items()
        ),
        emit=events.append,
    )
    lifecycle.publish(requests, cycle_id=2)
    return lifecycle, requests, events


def test_first_ready_schedule_installs_both_matching_initial_proposals():
    lifecycle, requests, events = _lifecycle()
    lifecycle.prepare_for_schedule(requests, cycle_id=2)
    assert lifecycle.state_for("A") is InitialProposalState.INSTALLED
    assert lifecycle.state_for("B") is InitialProposalState.INSTALLED
    assert requests["internal-A"].spec_token_ids == [90, 91]
    assert requests["internal-B"].spec_token_ids == [90, 91]
    assert [row["event"] for row in events] == [
        "published",
        "published",
        "installed",
        "installed",
    ]


def test_scheduled_spec_decode_evidence_consumes_both_once():
    lifecycle, requests, events = _lifecycle()
    lifecycle.prepare_for_schedule(requests, cycle_id=2)
    for request in requests.values():
        request.spec_token_ids = []  # pinned vLLM clears after scheduling
    lifecycle.finish_schedule(
        requests,
        scheduled_tokens={"internal-A": 3, "internal-B": 3},
        scheduled_spec_decode_tokens={
            "internal-A": [90, 91],
            "internal-B": [90, 91],
        },
        cycle_id=2,
    )
    assert lifecycle.state_for("A") is InitialProposalState.CONSUMED
    assert lifecycle.state_for("B") is InitialProposalState.CONSUMED
    consumed = next(
        row
        for row in events
        if row["request_id"] == "A"
        and row["event"] == "consumed-by-speculative-verification"
    )
    assert {
        "request_id",
        "internal_request_id",
        "lifecycle_state",
        "proposal_id",
        "round_id",
        "proposal_parent_prefix_len",
        "proposal_parent_prefix_hash",
        "live_request_prefix_len",
        "live_request_prefix_hash",
        "num_output_tokens",
        "num_computed_tokens",
        "request_spec_token_ids",
        "initial_proposal_token_ids",
        "scheduled_spec_decode_tokens_member",
        "scheduler_cycle_id",
    } <= set(consumed)
    assert validate_initial_proposal_lifecycle_events(
        events, expected_request_ids=("A", "B")
    ) == []


def test_consumed_round_zero_is_not_reinstalled_or_checked_against_later_prefix():
    lifecycle, requests, _ = _lifecycle({"A": (1, 2, 3)})
    request = requests["internal-A"]
    lifecycle.prepare_for_schedule(requests, cycle_id=2)
    request.spec_token_ids = []
    lifecycle.finish_schedule(
        requests,
        scheduled_tokens={"internal-A": 3},
        scheduled_spec_decode_tokens={"internal-A": [90, 91]},
        cycle_id=2,
    )
    request.all_token_ids.extend((90, 42))
    request.num_output_tokens += 2
    request.spec_token_ids = [200]  # normal RemoteDraftProposer round progression
    assignments = request.spec_assignments
    lifecycle.prepare_for_schedule(requests, cycle_id=3)
    lifecycle.finish_schedule(
        requests,
        scheduled_tokens={"internal-A": 2},
        scheduled_spec_decode_tokens={"internal-A": [200]},
        cycle_id=3,
    )
    assert request.spec_assignments == assignments
    assert request.spec_token_ids == [200]
    assert lifecycle.state_for("A") is InitialProposalState.CONSUMED


def test_installed_but_unscheduled_retains_without_reinstallation():
    lifecycle, requests, events = _lifecycle({"A": (1, 2, 3)})
    request = requests["internal-A"]
    lifecycle.prepare_for_schedule(requests, cycle_id=2)
    assignments = request.spec_assignments
    lifecycle.finish_schedule(
        requests,
        scheduled_tokens={},
        scheduled_spec_decode_tokens={},
        cycle_id=2,
    )
    lifecycle.prepare_for_schedule(requests, cycle_id=3)
    assert request.spec_assignments == assignments
    assert lifecycle.state_for("A") is InitialProposalState.INSTALLED
    assert any(row["event"] == "installed-not-scheduled" for row in events)


def test_installed_but_unscheduled_changed_prefix_fails_closed():
    lifecycle, requests, events = _lifecycle({"A": (1, 2, 3)})
    lifecycle.prepare_for_schedule(requests, cycle_id=2)
    requests["internal-A"].all_token_ids.append(99)
    with pytest.raises(RuntimeError, match="parent is stale before consumption"):
        lifecycle.finish_schedule(
            requests,
            scheduled_tokens={},
            scheduled_spec_decode_tokens={},
            cycle_id=2,
        )
    failure = events[-1]
    assert failure["event"] == "validation-failed"
    assert failure["live_request_prefix_len"] == 4
    assert failure["proposal_parent_prefix_len"] == 3


@pytest.mark.parametrize("replacement", ((), (90,), (90, 92)))
def test_installed_but_unscheduled_missing_or_changed_spec_fails_closed(replacement):
    lifecycle, requests, _ = _lifecycle({"A": (1, 2, 3)})
    lifecycle.prepare_for_schedule(requests, cycle_id=2)
    requests["internal-A"].spec_token_ids = replacement
    with pytest.raises(RuntimeError, match="disappeared or changed"):
        lifecycle.finish_schedule(
            requests,
            scheduled_tokens={},
            scheduled_spec_decode_tokens={},
            cycle_id=2,
        )


def test_true_stale_parent_before_first_installation_still_fails_closed():
    lifecycle, requests, events = _lifecycle({"A": (1, 2, 3)})
    requests["internal-A"].all_token_ids.append(99)
    with pytest.raises(RuntimeError, match="parent is stale before consumption") as error:
        lifecycle.prepare_for_schedule(requests, cycle_id=2)
    assert "proposal_parent_prefix_hash" in str(error.value)
    assert events[-1]["event"] == "validation-failed"


def test_installed_request_cannot_be_scheduled_as_plain_target_decode():
    lifecycle, requests, _ = _lifecycle({"A": (1, 2, 3)})
    lifecycle.prepare_for_schedule(requests, cycle_id=2)
    requests["internal-A"].spec_token_ids = []
    with pytest.raises(RuntimeError, match="scheduled without its initial proposal"):
        lifecycle.finish_schedule(
            requests,
            scheduled_tokens={"internal-A": 1},
            scheduled_spec_decode_tokens={},
            cycle_id=2,
        )


def test_target_only_resident_admission_semantics_are_unchanged():
    assert resident_admission_decision(
        num_output_tokens=1,
        global_decode_ready=True,
        consumer="target-only",
        has_initial_proposal=False,
    ) == (True, "global-decode-ready")


def test_scheduler_consumes_only_from_stock_scheduled_spec_evidence():
    source = (
        Path(__file__).parents[1]
        / "src/specrhythm/phase4/resident_scheduler.py"
    ).read_text(encoding="utf-8")
    stock_call = source.index("output = super().schedule")
    finish_call = source.index(".finish_schedule(")
    assert stock_call < finish_call
    assert (
        "scheduled_spec_decode_tokens=output.scheduled_spec_decode_tokens" in source
    )
    assert "def _install_initial_proposals" not in source
    assert "if self._resident_consumer != \"serial\":" in source
