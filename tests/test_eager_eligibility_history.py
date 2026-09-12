"""Measure the actual provider snapshot path across bounded rolling histories."""

import itertools

from specrhythm.continuation.core import ParentVerification, RollingContinuation
from specrhythm.continuation.cpu import generate
from specrhythm.continuation.policy import StaticEagerEligibility
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


def test_provider_history_copy_growth_is_explicitly_observed(monkeypatch):
    monkeypatch.setattr(TRACE, 'enabled', True)
    monkeypatch.setattr(TRACE, 'rows', [])
    monkeypatch.setattr(TRACE, 'attempts', {})
    monkeypatch.setattr(TRACE, 'sequence', itertools.count())
    core = rolling_history(32)
    rows = [r for r in TRACE.rows if r['category'] == 'eligibility_snapshot']
    assert len(rows) == 1+32*4  # registration, start, begin, parent, promotion
    assert all(r['snapshot_kind'] == 'deepcopy' for r in rows)
    assert all(r['copied_history_objects'] == r['history_proposals']+r['history_continuations']
               for r in rows)
    assert rows[-1]['copied_history_objects'] > rows[1]['copied_history_objects']
    assert core._state('r').accounting.committed_tokens == 160
    core.shutdown()
