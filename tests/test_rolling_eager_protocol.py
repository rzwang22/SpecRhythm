"""Execute the rolling protocol with deterministic CPU Draft and Target work.

The Target oracle commits one token at a time using the existing greedy
acceptance contract. No expected lifecycle status is supplied by the backend.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from specrhythm.continuation.core import ParentVerification, RollingContinuation
from specrhythm.continuation.cpu import generate
from specrhythm.continuation.policy import EagerDecision, StaticEagerEligibility
from specrhythm.phase4.serial import greedy_acceptance


def next_token(prefix):
    return prefix[-1] + 1 if prefix else 1


def make_owner(*, request_id="req", max_output_tokens=100, eos_token_ids=(), provider=None):
    owner = RollingContinuation(
        "owner", provider if provider is not None else StaticEagerEligibility({request_id})
    )
    owner.register(
        request_id,
        (0,),
        home_cohort="A",
        max_output_tokens=max_output_tokens,
        eos_token_ids=eos_token_ids,
    )
    return owner


def normal_proposal(owner, request_id="req", *, oracle=next_token, eos_token_ids=()):
    work = owner.schedule_normal_recovery(request_id)
    result = generate(work, oracle, eos_token_ids=eos_token_ids)
    proposal = owner.record_normal_completion(work, result)
    return proposal, work, result


def target_receipt(owner, proposal, *, oracle=next_token, terminal_reason=None):
    """Build actual greedy committed output, including correction or bonus."""

    state = owner.state(proposal.request_id)
    prefix = proposal.parent_prefix
    delta = []
    for candidate in proposal.tokens:
        actual = oracle(prefix + tuple(delta))
        delta.append(actual)
        if actual in state.eos_token_ids:
            terminal_reason = "eos"
            break
        if len(delta) == state.remaining_output_tokens:
            terminal_reason = "length"
            break
        if actual != candidate:
            break
    else:
        if terminal_reason is None:
            delta.append(oracle(prefix + tuple(delta)))
    if delta and delta[-1] in state.eos_token_ids:
        terminal_reason = "eos"
    elif len(delta) == state.remaining_output_tokens:
        terminal_reason = "length"
    decision = greedy_acceptance(proposal.tokens, delta, terminal=terminal_reason is not None)
    return ParentVerification(
        owner_id=state.owner_id,
        request_id=state.request_id,
        proposal_id=proposal.proposal_id,
        prefix_version=proposal.prefix_version,
        parent_prefix=proposal.parent_prefix,
        committed_delta=decision.committed_token_ids,
        terminal_reason=terminal_reason,
    )


def continuation_round(owner, proposal, *, draft_first=True, target_oracle=next_token):
    owner.start_verification("req", proposal.proposal_id)
    work = owner.begin_continuation("req")
    assert work is not None
    completion = generate(work, next_token)
    receipt = target_receipt(owner, proposal, oracle=target_oracle)
    cid = owner.state("req").active_continuation_id
    assert cid is not None
    if draft_first:
        owner.record_continuation_completion(work, completion)
        assert owner.continuation("req", cid).status == "waiting_parent"
        assert owner.state("req").committed_prefix == proposal.parent_prefix
        owner.resolve_parent_verification(receipt)
    else:
        owner.resolve_parent_verification(receipt)
        if receipt.committed_delta == proposal.tokens + (completion.generated_tokens[0],):
            assert owner.continuation("req", cid).status == "waiting_draft"
        owner.record_continuation_completion(work, completion)
    return work, completion, receipt, cid


@pytest.mark.parametrize("first_draft_first", [True, False])
def test_continuous_success_rejection_recovery_and_continuous_success(first_draft_first):
    owner = make_owner()
    proposal, initial, _ = normal_proposal(owner)
    assert initial.candidate_length == 4
    assert proposal.tokens == (1, 2, 3, 4)
    proposal_ids = [proposal.proposal_id]
    continuation_ids = []
    trace = []

    # P0 -> E1 -> E2 -> rejection of E3 -> R3 -> E4 -> E5 -> E6.
    for stage in range(6):
        rejecting = stage == 2
        reject_at = len(proposal.parent_prefix) + 2

        def oracle(prefix, rejecting=rejecting, reject_at=reject_at):
            if rejecting and len(prefix) == reject_at:
                return 1000
            return next_token(prefix)

        work, completion, receipt, cid = continuation_round(
            owner,
            proposal,
            draft_first=first_draft_first if stage % 2 == 0 else not first_draft_first,
            target_oracle=oracle,
        )
        assert work.candidate_length == 4
        assert len(completion.generated_tokens) == 5
        continuation_ids.append(cid)
        state = owner.state("req")
        assert state.committed_prefix == proposal.parent_prefix + receipt.committed_delta
        assert state.committed_prefix_version == stage + 1
        assert owner.evaluate_eager_eligibility("req").eligible
        if rejecting:
            invalid = owner.continuation("req", cid)
            assert invalid.status == "invalidated"
            assert invalid.reason == "parent_rejected"
            assert state.recovery_required
            assert state.current_proposal_id is None
            corrected_prefix = state.committed_prefix
            proposal, recovery, _ = normal_proposal(owner)
            assert recovery.dependency_prefix == corrected_prefix
            assert proposal.tokens == (1001, 1002, 1003, 1004)
            assert not owner.state("req").recovery_required
            trace.append("reject/recover")
        else:
            assert owner.continuation("req", cid).status == "promotable"
            proposal = owner.promote_continuation("req", cid)
            assert proposal.tokens == completion.generated_tokens[1:]
            assert proposal.parent_prefix == state.committed_prefix
            assert proposal.source_continuation_id == cid
            assert proposal.prefix_version == state.committed_prefix_version
            trace.append("promote")
        proposal_ids.append(proposal.proposal_id)

    assert trace == ["promote", "promote", "reject/recover", "promote", "promote", "promote"]
    assert len(set(proposal_ids)) == 7
    assert len(set(continuation_ids)) == 6
    accounting = owner.state("req").accounting
    assert len(owner.state("req").committed_prefix) - 1 == accounting.committed_tokens == 28
    assert accounting.parent_accepted_tokens == 22
    assert accounting.correction_tokens == 1
    assert accounting.bonus_tokens == 5
    assert accounting.normal_generated_tokens == 8
    assert accounting.early_generated_tokens == 30
    assert accounting.bridge_generated_tokens == 6
    assert accounting.reused_candidate_tokens == 20
    assert accounting.reused_bridge_tokens == 5
    assert accounting.discarded_early_tokens == 5
    assert accounting.recovery_generated_tokens == 4
    assert accounting.recovery_jobs == 1
    assert accounting.committed_tokens == (
        accounting.parent_accepted_tokens + accounting.correction_tokens + accounting.bonus_tokens
    )


@pytest.mark.parametrize("draft_first", [True, False])
def test_full_parent_acceptance_with_wrong_bonus_requires_recovery(draft_first):
    owner = make_owner()
    proposal, _, _ = normal_proposal(owner)

    def bonus_mismatch(prefix):
        return 99 if len(prefix) == len(proposal.parent_prefix) + 4 else next_token(prefix)

    _, _, _, cid = continuation_round(
        owner, proposal, draft_first=draft_first, target_oracle=bonus_mismatch
    )
    invalid = owner.continuation("req", cid)
    assert invalid.status == "invalidated"
    assert invalid.reason == "bridge_mismatch"
    assert owner.state("req").committed_prefix == (0, 1, 2, 3, 4, 99)
    assert owner.state("req").recovery_required
    recovered, work, _ = normal_proposal(owner)
    assert work.dependency_prefix == (0, 1, 2, 3, 4, 99)
    assert recovered.tokens == (100, 101, 102, 103)
    assert owner.evaluate_eager_eligibility("req").eligible
    owner.finish_or_cancel("req")
    released = owner.continuation("req", cid)
    assert released.status == "released"
    assert released.reason == "bridge_mismatch"
    assert released.release_reason == "cancelled"


def test_eligibility_admission_and_validity_are_independent():
    owner = make_owner()
    proposal, _, _ = normal_proposal(owner)
    owner.start_verification("req", proposal.proposal_id)
    assert owner.evaluate_eager_eligibility("req").eligible
    assert owner.begin_continuation("req", admitted=False) is None
    assert owner.evaluate_eager_eligibility("req").eligible
    work = owner.begin_continuation("req", admitted=True)
    assert work is not None
    cid = owner.state("req").active_continuation_id
    with pytest.raises((ValueError, RuntimeError)):
        owner.promote_continuation("req", cid)
    with pytest.raises((ValueError, RuntimeError)):
        owner.begin_continuation("req")
    owner.record_continuation_completion(work, generate(work, next_token))
    assert owner.state("req").current_proposal_id == proposal.proposal_id
    with pytest.raises((ValueError, RuntimeError)):
        owner.promote_continuation("req", cid)


def test_duplicate_completions_feedback_and_consumption_do_not_double_count():
    owner = make_owner()
    proposal, normal, normal_completion = normal_proposal(owner)
    before = owner.state("req")
    duplicate = owner.record_normal_completion(normal, normal_completion)
    assert duplicate.proposal_id == proposal.proposal_id
    assert owner.state("req") == before
    work, completion, receipt, cid = continuation_round(owner, proposal)
    before = owner.state("req")
    owner.record_continuation_completion(work, completion)
    assert not owner.resolve_parent_verification(receipt)
    assert owner.state("req") == before
    promoted = owner.promote_continuation("req", cid)
    before = owner.state("req")
    with pytest.raises((ValueError, RuntimeError)):
        owner.promote_continuation("req", cid)
    assert owner.state("req") == before
    owner.start_verification("req", promoted.proposal_id)
    assert owner.continuation("req", cid).status == "consumed"
    owner.record_continuation_completion(work, completion)
    assert owner.state("req").current_proposal_id == promoted.proposal_id
    assert not owner.resolve_parent_verification(receipt)


def test_conflicting_duplicate_worker_or_target_messages_are_rejected_atomically():
    owner = make_owner()
    proposal, _, _ = normal_proposal(owner)
    work, completion, receipt, _ = continuation_round(owner, proposal)
    before = owner.state("req")
    with pytest.raises(ValueError, match="conflicting duplicate"):
        owner.record_continuation_completion(
            work,
            replace(completion, generated_tokens=completion.generated_tokens[:-1] + (900,)),
        )
    assert owner.state("req") == before
    with pytest.raises(ValueError, match="conflicting duplicate"):
        owner.resolve_parent_verification(
            replace(receipt, committed_delta=receipt.committed_delta[:-1] + (900,))
        )
    assert owner.state("req") == before


@pytest.mark.parametrize("changed", ["owner", "request", "proposal", "version", "prefix"])
def test_parent_receipt_requires_full_identity_and_dependency(changed):
    owner = make_owner()
    proposal, _, _ = normal_proposal(owner)
    owner.start_verification("req", proposal.proposal_id)
    work = owner.begin_continuation("req")
    owner.record_continuation_completion(work, generate(work, next_token))
    receipt = target_receipt(owner, proposal)
    mutations = {
        "owner": {"owner_id": "other-owner"},
        "request": {"request_id": "other-request"},
        "proposal": {"proposal_id": "other-proposal"},
        "version": {"prefix_version": receipt.prefix_version + 1},
        # Same length, different content: length-only validation would be unsafe.
        "prefix": {"parent_prefix": (999,)},
    }
    before = owner.state("req")
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        owner.resolve_parent_verification(replace(receipt, **mutations[changed]))
    assert owner.state("req") == before
    assert owner.resolve_parent_verification(receipt)


def test_worker_completion_cannot_replace_work_identity_or_dependency():
    owner = make_owner()
    proposal, _, _ = normal_proposal(owner)
    owner.start_verification("req", proposal.proposal_id)
    work = owner.begin_continuation("req")
    completion = generate(work, next_token)
    before = owner.state("req")
    for forged in (
        replace(work, owner_id="foreign"),
        replace(work, request_id="foreign"),
        replace(work, parent_proposal_id="foreign"),
        replace(work, prefix_version=work.prefix_version + 1),
        replace(work, dependency_prefix=(99,) + work.dependency_prefix[1:]),
    ):
        with pytest.raises((ValueError, RuntimeError, KeyError)):
            owner.record_continuation_completion(forged, completion)
        assert owner.state("req") == before
    with pytest.raises((ValueError, RuntimeError)):
        owner.record_continuation_completion(work, replace(completion, work_id="foreign"))
    assert owner.state("req") == before


def test_late_rejected_generation_cannot_invalidate_recovered_proposal():
    owner = make_owner()
    proposal, _, _ = normal_proposal(owner)
    owner.start_verification("req", proposal.proposal_id)
    work = owner.begin_continuation("req")
    completion = generate(work, next_token)
    cid = owner.state("req").active_continuation_id
    receipt = target_receipt(owner, proposal, oracle=lambda prefix: 99)
    owner.resolve_parent_verification(receipt)
    recovered, _, _ = normal_proposal(owner)
    owner.record_continuation_completion(work, completion)
    assert owner.continuation("req", cid).status == "invalidated"
    assert owner.state("req").current_proposal_id == recovered.proposal_id
    assert recovered.parent_prefix == (0, 99)
    assert not owner.state("req").recovery_required
    assert not owner.resolve_parent_verification(receipt)


def test_request_snapshots_cannot_mutate_authoritative_state():
    owner = make_owner()
    snapshot = owner.state("req")
    before = owner.state("req")
    snapshot.recovery_required = not before.recovery_required
    snapshot.accounting.committed_tokens = 900
    assert owner.state("req") == before


def test_proposal_and_continuation_snapshots_cannot_change_live_dependency():
    owner = make_owner()
    proposal, _, _ = normal_proposal(owner)
    owner.start_verification("req", proposal.proposal_id)
    work = owner.begin_continuation("req")
    owner.record_continuation_completion(work, generate(work, next_token))
    before = owner.state("req")
    proposal_copy = owner.proposal("req", proposal.proposal_id)
    proposal_copy.tokens = (999,)
    continuation_copy = owner.continuation("req", before.active_continuation_id)
    continuation_copy.dependency_prefix = (999,)
    assert owner.state("req") == before
    assert owner.resolve_parent_verification(target_receipt(owner, proposal))
    owner.promote_continuation("req", before.active_continuation_id)


def test_distinct_request_and_owner_work_isolation():
    owner = make_owner()
    owner.register("other", (50,), home_cohort="B", max_output_tokens=100)
    foreign = RollingContinuation("foreign", StaticEagerEligibility({"req"}))
    foreign.register("req", (100,), max_output_tokens=100)
    proposal, _, _ = normal_proposal(owner)
    other_before = owner.state("other")
    foreign_before = foreign.state("req")
    work, completion, _, cid = continuation_round(owner, proposal)
    assert owner.state("other") == other_before
    assert foreign.state("req") == foreign_before
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        foreign.record_continuation_completion(work, completion)
    owner.promote_continuation("req", cid)
    assert foreign.state("req") == foreign_before
    assert not owner.evaluate_eager_eligibility("other").eligible


def test_mutations_must_execute_on_the_owning_thread():
    owner = make_owner()
    before = owner.state("req")
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(owner.schedule_normal_recovery, "req")
        with pytest.raises((ValueError, RuntimeError)):
            future.result()
    assert owner.state("req") == before


class SwitchProvider:
    def __init__(self):
        self.enabled = True
        self.version = 0

    def evaluate(self, request_state, step_context):
        return EagerDecision(True, self.enabled, self.version)

    def switch(self, enabled):
        self.enabled = enabled
        self.version += 1


def test_disable_during_target_and_draft_then_reenable_from_true_prefix():
    provider = SwitchProvider()
    owner = make_owner(provider=provider)
    proposal, _, _ = normal_proposal(owner)
    owner.start_verification("req", proposal.proposal_id)
    work = owner.begin_continuation("req")
    completion = generate(work, next_token)
    cid = owner.state("req").active_continuation_id
    provider.switch(False)
    decision = owner.evaluate_eager_eligibility("req")
    assert decision.eligible and not decision.can_admit
    assert owner.begin_continuation("req") is None
    owner.record_continuation_completion(work, completion)
    assert owner.continuation("req", cid).status in {"cancelled", "released"}
    assert owner.resolve_parent_verification(target_receipt(owner, proposal))
    assert owner.state("req").committed_prefix == (0, 1, 2, 3, 4, 5)
    recovered, _, _ = normal_proposal(owner)
    provider.switch(True)
    assert owner.evaluate_eager_eligibility("req").can_admit
    owner.start_verification("req", recovered.proposal_id)
    assert owner.begin_continuation("req") is not None
    assert owner.state("req").committed_prefix == (0, 1, 2, 3, 4, 5)


def test_disable_after_promotion_keeps_legal_proposal_for_normal_verification():
    provider = SwitchProvider()
    owner = make_owner(provider=provider)
    proposal, _, _ = normal_proposal(owner)
    _, _, _, cid = continuation_round(owner, proposal)
    promoted = owner.promote_continuation("req", cid)
    before_prefix = owner.state("req").committed_prefix
    provider.switch(False)
    owner.evaluate_eager_eligibility("req")
    with pytest.raises((ValueError, RuntimeError)):
        owner.start_verification("req", promoted.proposal_id, eager=True)
    assert owner.state("req").current_proposal_id == promoted.proposal_id
    assert owner.state("req").committed_prefix == before_prefix
    owner.start_verification("req", promoted.proposal_id, eager=False)
    assert owner.begin_continuation("req") is None
    owner.resolve_parent_verification(target_receipt(owner, promoted))
    assert owner.state("req").committed_prefix == tuple(range(11))


def test_kv_keeps_sampled_last_token_unmaterialized_then_fills_parent_and_bridge():
    owner = make_owner()
    proposal, normal, initial = normal_proposal(owner)
    assert normal.base_kv_prefix == (0,)
    assert initial.generated_tokens == (1, 2, 3, 4)
    assert initial.materialized_kv_frontier == 4
    assert owner.state("req").materialized_kv_prefix == (0, 1, 2, 3)
    work, completion, _, cid = continuation_round(owner, proposal)
    assert work.base_kv_prefix == (0, 1, 2, 3)
    assert work.dependency_prefix == (0, 1, 2, 3, 4)
    assert completion.generated_tokens == (5, 6, 7, 8, 9)
    assert completion.materialized_kv_frontier == 9
    pending = owner.continuation("req", cid)
    assert pending.predicted_bridge == 5
    assert pending.dependency_prefix == (0, 1, 2, 3, 4, 5)
    assert pending.continuation_tokens == (6, 7, 8, 9)
    owner.promote_continuation("req", cid)
    assert owner.state("req").materialized_kv_prefix == tuple(range(9))
    assert owner.state("req").committed_prefix == tuple(range(6))
    assert owner.state("req").accounting.draft_materialized_tokens == 8


def test_rejection_kv_is_cropped_to_true_prefix_and_correction_is_materialized_on_recovery():
    owner = make_owner()
    proposal, _, _ = normal_proposal(owner)
    _, _, _, cid = continuation_round(
        owner,
        proposal,
        target_oracle=lambda prefix: 50 if len(prefix) == 3 else next_token(prefix),
    )
    assert owner.state("req").committed_prefix == (0, 1, 2, 50)
    assert owner.state("req").materialized_kv_prefix == (0, 1, 2)
    assert owner.continuation("req", cid).materialized_kv_frontier == 0
    recovered, recovery, completion = normal_proposal(owner)
    assert recovery.base_kv_prefix == (0, 1, 2)
    assert recovery.dependency_prefix == (0, 1, 2, 50)
    assert recovered.tokens == (51, 52, 53, 54)
    assert completion.materialized_kv_frontier == 7
    assert owner.state("req").materialized_kv_prefix == (0, 1, 2, 50, 51, 52, 53)


@pytest.mark.parametrize("damage", ["short", "long", "frontier", "wrong_id"])
def test_incomplete_or_invalid_cpu_completion_never_becomes_ready(damage):
    owner = make_owner()
    proposal, _, _ = normal_proposal(owner)
    owner.start_verification("req", proposal.proposal_id)
    work = owner.begin_continuation("req")
    completion = generate(work, next_token)
    mutations = {
        "short": {"generated_tokens": completion.generated_tokens[:-1]},
        "long": {"generated_tokens": completion.generated_tokens + (10,)},
        "frontier": {"materialized_kv_frontier": completion.materialized_kv_frontier + 1},
        "wrong_id": {"work_id": "old-work"},
    }
    before = owner.state("req")
    with pytest.raises(ValueError):
        owner.record_continuation_completion(work, replace(completion, **mutations[damage]))
    assert owner.state("req") == before
    owner.record_continuation_completion(work, completion)
    assert owner.state("req").current_proposal_id == proposal.proposal_id


@pytest.mark.parametrize("eos_token", [2, 5, 7])
@pytest.mark.parametrize("draft_first", [True, False])
def test_eos_in_parent_bridge_or_promoted_candidates_releases_all_state(eos_token, draft_first):
    owner = make_owner(eos_token_ids=(eos_token,))
    proposal, _, _ = normal_proposal(owner)
    for _ in range(3):
        owner.start_verification("req", proposal.proposal_id)
        work = owner.begin_continuation("req")
        completion = generate(work, next_token) if work is not None else None
        if draft_first and work is not None:
            owner.record_continuation_completion(work, completion)
        receipt = target_receipt(owner, proposal)
        owner.resolve_parent_verification(receipt)
        if not draft_first and work is not None:
            owner.record_continuation_completion(work, completion)
        state = owner.state("req")
        if state.finished:
            break
        assert state.active_continuation_id is not None
        proposal = owner.promote_continuation("req", state.active_continuation_id)
    assert state.finished and state.terminal_reason == "eos"
    assert state.committed_prefix == tuple(range(eos_token + 1))
    assert state.accounting.committed_tokens == eos_token
    assert state.materialized_kv_frontier == 0
    assert state.current_proposal_id is None
    assert state.active_continuation_id is None
    assert not state.recovery_required
    assert all(p.status == "released" and not p.tokens for p in state.proposals.values())
    assert all(
        c.status == "released" and not c.continuation_tokens and not c.dependency_prefix
        for c in state.continuations.values()
    )
    assert owner.begin_continuation("req") is None
    with pytest.raises(ValueError):
        owner.schedule_normal_recovery("req")


@pytest.mark.parametrize("max_output_tokens", [1, 2, 4, 5, 6, 7, 9, 10, 11])
def test_output_length_boundaries_include_short_candidates_and_zero_candidate_tail(
    max_output_tokens,
):
    owner = make_owner(max_output_tokens=max_output_tokens)
    candidate_lengths = []
    proposal, _, _ = normal_proposal(owner)
    for step in range(max_output_tokens + 1):
        candidate_lengths.append(len(proposal.tokens))
        owner.start_verification("req", proposal.proposal_id)
        work = owner.begin_continuation("req")
        completion = generate(work, next_token) if work is not None else None
        if work is not None and step % 2 == 0:
            owner.record_continuation_completion(work, completion)
        owner.resolve_parent_verification(target_receipt(owner, proposal))
        if work is not None and step % 2 != 0:
            owner.record_continuation_completion(work, completion)
        state = owner.state("req")
        if state.finished:
            break
        if state.active_continuation_id is not None:
            proposal = owner.promote_continuation("req", state.active_continuation_id)
        else:
            proposal, _, _ = normal_proposal(owner)
    assert state.finished and state.terminal_reason == "length"
    assert state.committed_prefix == tuple(range(max_output_tokens + 1))
    assert state.accounting.committed_tokens == max_output_tokens
    assert state.remaining_output_tokens == 0
    assert all(length <= 4 for length in candidate_lengths)
    if max_output_tokens in (1, 6, 11):
        assert 0 in candidate_lengths
    assert state.materialized_kv_frontier == 0


@pytest.mark.parametrize(
    "phase", ["normal", "draft", "waiting_parent", "waiting_draft", "promoted"]
)
def test_cancellation_and_late_work_never_reactivate_request(phase):
    owner = make_owner()
    if phase == "normal":
        work = owner.schedule_normal_recovery("req")
        completion = generate(work, next_token)
        receipt = None
    else:
        proposal, _, _ = normal_proposal(owner)
        owner.start_verification("req", proposal.proposal_id)
        work = owner.begin_continuation("req")
        completion = generate(work, next_token)
        receipt = target_receipt(owner, proposal)
        if phase in ("waiting_parent", "promoted"):
            owner.record_continuation_completion(work, completion)
        if phase in ("waiting_draft", "promoted"):
            owner.resolve_parent_verification(receipt)
        if phase == "promoted":
            owner.promote_continuation("req", owner.state("req").active_continuation_id)
    prefix_before = owner.state("req").committed_prefix
    owner.finish_or_cancel("req")
    owner.finish_or_cancel("req")
    if phase == "normal":
        assert owner.record_normal_completion(work, completion) is None
    else:
        assert owner.record_continuation_completion(work, completion) == "released"
        assert not owner.resolve_parent_verification(receipt)
    state = owner.state("req")
    assert state.finished and state.terminal_reason == "cancelled"
    assert state.committed_prefix == prefix_before
    assert state.current_proposal_id is None
    assert state.active_continuation_id is None
    assert state.normal_work_id is None
    assert state.materialized_kv_frontier == 0
    assert owner.begin_continuation("req") is None


def test_owner_shutdown_releases_every_request_and_rejects_new_requests():
    owner = make_owner()
    owner.register("other", (50,), home_cohort="B")
    proposal, _, _ = normal_proposal(owner)
    owner.start_verification("req", proposal.proposal_id)
    continuation_work = owner.begin_continuation("req")
    normal_work = owner.schedule_normal_recovery("other")
    owner.shutdown()
    owner.shutdown()
    for request_id in ("req", "other"):
        state = owner.state(request_id)
        assert state.finished and state.terminal_reason == "owner_shutdown"
        assert state.materialized_kv_frontier == 0
        assert state.accounting.committed_tokens == 0
    owner.record_continuation_completion(
        continuation_work, generate(continuation_work, next_token)
    )
    assert owner.record_normal_completion(normal_work, generate(normal_work, next_token)) is None
    with pytest.raises(ValueError):
        owner.register("new", (0,))


@pytest.mark.parametrize("disable_mode", ["ineligible", "disabled"])
def test_eager_off_executes_only_normal_draft_with_unchanged_target_output(disable_mode):
    provider = (
        StaticEagerEligibility(set())
        if disable_mode == "ineligible"
        else StaticEagerEligibility({"req"}, enabled=False)
    )
    owner = make_owner(provider=provider, max_output_tokens=16)
    for _ in range(5):
        proposal, _, _ = normal_proposal(owner)
        owner.start_verification("req", proposal.proposal_id)
        assert owner.begin_continuation("req") is None
        owner.resolve_parent_verification(target_receipt(owner, proposal))
        state = owner.state("req")
        if state.finished:
            break
    assert state.finished
    assert state.committed_prefix == tuple(range(17))
    assert state.accounting.committed_tokens == 16
    assert state.accounting.early_generated_tokens == 0
    assert state.accounting.bridge_generated_tokens == 0
    assert state.accounting.reused_candidate_tokens == 0
    assert state.accounting.recovery_jobs == 0
    assert state.accounting.recovery_generated_tokens == 0
    assert not state.continuations


def test_target_eos_correction_releases_inflight_continuation():
    owner = make_owner(eos_token_ids=(99,))
    proposal, _, _ = normal_proposal(owner)
    owner.start_verification("req", proposal.proposal_id)
    work = owner.begin_continuation("req")
    completion = generate(work, next_token)
    receipt = target_receipt(
        owner, proposal, oracle=lambda prefix: 99 if len(prefix) == 3 else next_token(prefix)
    )
    owner.resolve_parent_verification(receipt)
    owner.record_continuation_completion(work, completion)
    state = owner.state("req")
    assert state.finished and state.terminal_reason == "eos"
    assert state.committed_prefix == (0, 1, 2, 99)
    assert state.accounting.parent_accepted_tokens == 2
    assert state.accounting.correction_tokens == 1
    assert state.accounting.bonus_tokens == 0
    assert state.accounting.discarded_early_tokens == 5


@pytest.mark.parametrize("reject_parent", [True, False])
def test_admission_deferred_full_acceptance_is_normal_but_rejection_is_recovery(reject_parent):
    owner = make_owner()
    proposal, _, _ = normal_proposal(owner)
    owner.start_verification("req", proposal.proposal_id)
    assert owner.begin_continuation("req", admitted=False) is None
    receipt = target_receipt(
        owner, proposal, oracle=(lambda prefix: 99) if reject_parent else next_token
    )
    owner.resolve_parent_verification(receipt)
    assert owner.state("req").recovery_required is reject_parent
    _, work, _ = normal_proposal(owner)
    assert work.kind == ("recovery" if reject_parent else "normal")
    assert owner.state("req").accounting.recovery_jobs == int(reject_parent)
    assert owner.state("req").accounting.recovery_generated_tokens == 4 * int(reject_parent)
    assert owner.evaluate_eager_eligibility("req").eligible


def test_imported_prefix_version_advances_and_fences_old_parent_receipts():
    owner = RollingContinuation("owner", StaticEagerEligibility({"req"}))
    owner.register("req", (0,), committed_prefix_version=17)
    proposal, _, _ = normal_proposal(owner)
    assert proposal.prefix_version == 17
    _, _, receipt, cid = continuation_round(owner, proposal)
    assert owner.state("req").committed_prefix_version == 18
    promoted = owner.promote_continuation("req", cid)
    assert promoted.prefix_version == 18
    owner.start_verification("req", promoted.proposal_id)
    before = owner.state("req")
    with pytest.raises(ValueError):
        owner.resolve_parent_verification(replace(receipt, prefix_version=0))
    assert owner.state("req") == before
