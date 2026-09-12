"""Failure atomicity at future eligibility-provider boundaries."""

import pytest

from specrhythm.continuation.core import ParentVerification, RollingContinuation
from specrhythm.continuation.cpu import generate
from specrhythm.continuation.policy import EagerDecision


class FailingProvider:
    fault = None

    def evaluate(self, request_state, step_context):
        assert step_context.step_index == 0
        if self.fault == "exception":
            raise RuntimeError("provider unavailable")
        if self.fault == "type":
            return None
        if self.fault == "regression":
            return EagerDecision(True, True, 1)
        if self.fault == "unversioned":
            return EagerDecision(True, False, 2)
        return EagerDecision(True, True, 2)


@pytest.mark.parametrize("fault", ["exception", "type", "regression", "unversioned"])
def test_failed_provider_leaves_parent_receipt_retryable_without_partial_commit(fault):
    provider = FailingProvider()
    machine = RollingContinuation("owner", provider)
    machine.register("r", (10,))
    normal = machine.schedule_normal_recovery("r")
    parent = machine.record_normal_completion(normal, generate(normal, lambda p: p[-1] + 1))
    machine.start_verification("r", parent.proposal_id)
    future = machine.begin_continuation("r")
    machine.record_continuation_completion(future, generate(future, lambda p: p[-1] + 1))
    receipt = ParentVerification(
        "owner", "r", parent.proposal_id, 0, (10,), (11, 12, 13, 14, 15)
    )
    before = machine.state("r")
    provider.fault = fault
    with pytest.raises((ValueError, RuntimeError)):
        machine.resolve_parent_verification(receipt)
    assert machine.state("r") == before
    provider.fault = None
    assert machine.resolve_parent_verification(receipt)
    machine.promote_continuation("r", future.work_id)
    assert machine.state("r").accounting.committed_tokens == 5
    assert not machine.resolve_parent_verification(receipt)


@pytest.mark.parametrize("fault", ["exception", "type"])
def test_failed_registration_does_not_leave_ghost_request(fault):
    provider = FailingProvider()
    provider.fault = fault
    machine = RollingContinuation("owner", provider)
    with pytest.raises((ValueError, RuntimeError)):
        machine.register("r", (10,))
    with pytest.raises(ValueError, match="unknown request"):
        machine.state("r")
    provider.fault = None
    assert machine.register("r", (10,)).eager_decision.eligible
