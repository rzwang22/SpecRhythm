"""Measure the actual provider snapshot path across bounded rolling histories."""

import itertools
from dataclasses import FrozenInstanceError

import pytest

from specrhythm.continuation import core as core_module
from specrhythm.continuation.core import ParentVerification, RollingContinuation
from specrhythm.continuation.cpu import generate
from specrhythm.continuation.policy import (
    EagerDecision,
    EagerRequestView,
    EagerStepContext,
    StaticEagerEligibility,
)
from specrhythm.continuation.trace import TRACE


def rolling_history(rounds):
    core = RollingContinuation('cpu-history', StaticEagerEligibility({'r'}))
    core.register('r', tuple(range(256)), max_output_tokens=10000)
    work = core.schedule_normal_recovery('r')
    proposal = core.record_normal_completion(work, generate(work, lambda p: p[-1]+1))
    for _ in range(rounds):
        core.start_verification('r', proposal.proposal_id)
        future = core.begin_continuation('r')
        completion = generate(future, lambda p: p[-1]+1)
        core.record_continuation_completion(future, completion)
        core.resolve_parent_verification(ParentVerification(
            core.owner_id, 'r', proposal.proposal_id, proposal.prefix_version,
            proposal.parent_prefix, proposal.tokens+(completion.generated_tokens[0],)))
        proposal = core.promote_continuation('r', future.work_id)
    return core


def test_rolling_provider_never_deepcopies_history(monkeypatch):
    monkeypatch.setattr(TRACE, 'enabled', True)
    monkeypatch.setattr(TRACE, 'rows', [])
    monkeypatch.setattr(TRACE, 'attempts', {})
    monkeypatch.setattr(TRACE, 'sequence', itertools.count())
    original = core_module.copy.deepcopy
    state_copies = []

    def guard(value, memo=None):
        if isinstance(value, core_module.RequestState):
            # The public registration return remains a defensive snapshot.
            assert not value.proposals and not value.continuations
            state_copies.append(value.request_id)
        return original(value, memo)

    monkeypatch.setattr(core_module.copy, 'deepcopy', guard)
    core = rolling_history(32)
    rows = [r for r in TRACE.rows if r['category'] == 'eligibility_snapshot']
    assert len(rows) == 1+32*4  # registration, start, begin, parent, promotion
    assert all(r['snapshot_kind'] == 'immutable_request_identity' for r in rows)
    assert all(r['copied_history_objects'] == 0 for r in rows)
    assert rows[-1]['history_proposals'] > rows[1]['history_proposals']
    assert rows[-1]['history_continuations'] > rows[1]['history_continuations']
    assert state_copies == ['r']
    assert core._state('r').accounting.committed_tokens == 160
    core.shutdown()


def test_provider_view_is_frozen_fresh_and_cannot_mutate_owner_state():
    class InspectingProvider:
        def __init__(self):
            self.views = []
            self.contexts = []

        def evaluate(self, view, context):
            assert isinstance(view, EagerRequestView)
            assert vars(view) == {'request_id': 'r'}
            with pytest.raises(FrozenInstanceError):
                view.request_id = 'changed'
            for field in ('proposals', 'continuations', 'committed_prefix', 'accounting'):
                assert not hasattr(view, field)
            self.views.append(view)
            self.contexts.append(context)
            # Even deliberately bypassing frozen assignment only changes this view.
            object.__setattr__(view, 'request_id', 'provider-local')
            return EagerDecision(True, True, 0)

    provider = InspectingProvider()
    core = RollingContinuation('isolated', provider)
    before = core.register('r', (1, 2))
    context = EagerStepContext(37)
    core.evaluate_eager_eligibility('r', context)
    assert provider.views[0] is not provider.views[1]
    assert provider.contexts[-1] is context
    assert core.state('r') == before
    core.shutdown()


@pytest.mark.parametrize('decision, message', [
    (EagerDecision(True, True, 1), 'regressed'),
    (EagerDecision(True, False, 2), 'requires a new version'),
])
def test_view_keeps_provider_version_checks_and_live_switch(decision, message):
    class Provider:
        current = EagerDecision(True, True, 2)

        def evaluate(self, view, context):
            assert view.request_id == 'r'
            return self.current

    provider = Provider()
    core = RollingContinuation('versions', provider)
    core.register('r', (1, 2))
    before = core.state('r')
    provider.current = decision
    with pytest.raises(ValueError, match=message):
        core.evaluate_eager_eligibility('r')
    assert core.state('r') == before
    provider.current = EagerDecision(True, False, 3)
    assert not core.evaluate_eager_eligibility('r').can_admit
    provider.current = EagerDecision(True, True, 4)
    assert core.evaluate_eager_eligibility('r').can_admit
    assert core.state('r').eager_decision.decision_version == 4
    core.shutdown()
