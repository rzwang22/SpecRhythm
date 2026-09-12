"""Owner-local rolling continuation protocol, independent of GPU execution.

Only Target receipts advance committed output. Draft workers receive immutable
work and return fenced completion evidence; all mutations run on the owner thread.
"""

from __future__ import annotations

import copy
import hashlib
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

from specrhythm.continuation.policy import (
    EagerDecision,
    EagerEligibilityProvider,
    EagerStepContext,
)
from specrhythm.phase4.draft_batch import token_tuple
from specrhythm.phase4.dual_commit import dual_greedy_acceptance
from specrhythm.phase4.serial import token_prefix_hash

NORMAL_CANDIDATE_LENGTH = 4
EAGER_CANDIDATE_LENGTH = 4


class ContinuationStatus(str, Enum):
    GENERATING = "generating"
    WAITING_PARENT = "waiting_parent"
    WAITING_DRAFT = "waiting_draft"
    PROMOTABLE = "promotable"
    PROMOTED = "promoted"
    CONSUMED = "consumed"
    INVALIDATED = "invalidated"
    CANCELLED = "cancelled"
    RELEASED = "released"


@dataclass(frozen=True)
class DraftWork:
    owner_id: str
    request_id: str
    work_id: str
    parent_proposal_id: Optional[str]
    prefix_version: int
    dependency_prefix: Tuple[int, ...]
    candidate_length: int
    kind: str
    base_kv_prefix: Tuple[int, ...]
    decision_version: int
    eos_token_ids: Tuple[int, ...]


@dataclass(frozen=True)
class DraftCompletion:
    work_id: str
    generated_tokens: Tuple[int, ...]
    materialized_kv_frontier: int


@dataclass(frozen=True)
class ParentVerification:
    owner_id: str
    request_id: str
    proposal_id: str
    prefix_version: int
    parent_prefix: Tuple[int, ...]
    committed_delta: Tuple[int, ...]
    terminal_reason: Optional[str] = None


@dataclass
class Proposal:
    proposal_id: str
    tokens: Tuple[int, ...]
    prefix_version: int
    parent_prefix: Tuple[int, ...]
    parent_proposal_id: Optional[str] = None
    source_continuation_id: Optional[str] = None
    status: str = "ready"
    request_id: str = ""
    owner_id: str = ""


@dataclass
class Continuation:
    continuation_id: str
    parent_proposal_id: str
    prefix_version: int
    generation: int
    dependency_prefix: Tuple[int, ...]
    continuation_tokens: Tuple[int, ...] = ()
    predicted_bridge: Optional[int] = None
    materialized_kv_frontier: int = 0
    status: ContinuationStatus = ContinuationStatus.GENERATING
    reason: Optional[str] = None
    release_reason: Optional[str] = None
    parent_confirmed: bool = False
    completed: bool = False
    generated_count: int = 0
    discarded_counted: bool = False


@dataclass
class Accounting:
    committed_tokens: int = 0
    parent_accepted_tokens: int = 0
    correction_tokens: int = 0
    bonus_tokens: int = 0
    normal_generated_tokens: int = 0
    early_generated_tokens: int = 0
    bridge_generated_tokens: int = 0
    reused_candidate_tokens: int = 0
    reused_bridge_tokens: int = 0
    discarded_early_tokens: int = 0
    recovery_generated_tokens: int = 0
    recovery_jobs: int = 0
    draft_materialized_tokens: int = 0


@dataclass
class RequestState:
    request_id: str
    owner_id: str
    home_cohort: str
    committed_prefix: Tuple[int, ...]
    max_output_tokens: int
    eos_token_ids: Tuple[int, ...]
    committed_prefix_version: int = 0
    current_proposal_id: Optional[str] = None
    active_continuation_id: Optional[str] = None
    normal_work_id: Optional[str] = None
    recovery_required: bool = False
    finished: bool = False
    terminal_reason: Optional[str] = None
    eager_decision: Optional[EagerDecision] = None
    materialized_kv_prefix: Tuple[int, ...] = ()
    accounting: Accounting = field(default_factory=Accounting)
    proposals: dict[str, Proposal] = field(default_factory=dict)
    continuations: dict[str, Continuation] = field(default_factory=dict)

    @property
    def committed_prefix_length(self) -> int:
        return len(self.committed_prefix)

    @property
    def committed_prefix_hash(self) -> str:
        return token_prefix_hash(self.committed_prefix)

    @property
    def materialized_kv_frontier(self) -> int:
        return len(self.materialized_kv_prefix)

    @property
    def remaining_output_tokens(self) -> int:
        return self.max_output_tokens - self.accounting.committed_tokens


def _fingerprint(value: object) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def _common_prefix(left: tuple, right: tuple) -> tuple:
    size = 0
    while size < min(len(left), len(right)) and left[size] == right[size]:
        size += 1
    return left[:size]


class RollingContinuation:
    """One instance per existing owner; workers must marshal results back here.

    IDs are never reused, even for replacement requests or a new owner instance.
    Snapshots are defensive copies. Receipts/work tombstones retain only digests
    after release so late notifications cannot rehydrate speculative state.
    """

    def __init__(self, owner_id: str, provider: EagerEligibilityProvider) -> None:
        if not isinstance(owner_id, str) or not owner_id:
            raise ValueError("owner identity must be nonempty")
        self.owner_id = owner_id
        self.provider = provider
        self._thread = threading.get_ident()
        self._session = uuid.uuid4().hex
        self._sequence = 0
        self._requests: dict[str, RequestState] = {}
        self._works: dict[str, DraftWork] = {}
        self._work_digests: dict[str, str] = {}
        self._completion_digests: dict[str, str] = {}
        self._normal_results: dict[str, str] = {}
        self._receipt_digests: dict[str, str] = {}
        self._closed = False

    def _check(self) -> None:
        if threading.get_ident() != self._thread:
            raise RuntimeError("continuation mutation/read must run on its owner thread")

    def _state(self, request_id: str) -> RequestState:
        self._check()
        if request_id not in self._requests:
            raise ValueError("unknown request identity for this owner")
        return self._requests[request_id]

    def _id(self, kind: str) -> str:
        self._sequence += 1
        return f"{self.owner_id}:{self._session}:{kind}:{self._sequence}"

    def register(
        self, request_id, prefix, *, home_cohort="A", max_output_tokens=100,
        eos_token_ids=(), committed_prefix_version=0,
    ) -> RequestState:
        self._check()
        tokens = token_tuple(prefix)
        if (self._closed or not isinstance(request_id, str) or not request_id
                or request_id in self._requests or not tokens):
            raise ValueError("closed owner, empty prefix, or duplicate request identity")
        if type(max_output_tokens) is not int or max_output_tokens < 1:
            raise ValueError("output length must be positive")
        if home_cohort not in ("A", "B"):
            raise ValueError("home cohort must be A or B")
        if type(committed_prefix_version) is not int or committed_prefix_version < 0:
            raise ValueError("committed prefix version must be a nonnegative integer")
        self._requests[request_id] = RequestState(
            request_id, self.owner_id, home_cohort, tokens, max_output_tokens,
            token_tuple(eos_token_ids), materialized_kv_prefix=tokens,
            committed_prefix_version=committed_prefix_version,
        )
        try:
            self.evaluate_eager_eligibility(request_id)
        except Exception:
            del self._requests[request_id]
            raise
        return self.state(request_id)

    def state(self, request_id) -> RequestState:
        return copy.deepcopy(self._state(request_id))

    def proposal(self, request_id, proposal_id) -> Proposal:
        return copy.deepcopy(self._state(request_id).proposals[proposal_id])

    def continuation(self, request_id, continuation_id) -> Continuation:
        return copy.deepcopy(self._state(request_id).continuations[continuation_id])

    def evaluate_eager_eligibility(self, request_id, step_context=None):
        state = self._state(request_id)
        decision = self.provider.evaluate(
            copy.deepcopy(state), EagerStepContext() if step_context is None else step_context
        )
        if not isinstance(decision, EagerDecision):
            raise ValueError("provider must return a validated EagerDecision")
        previous = state.eager_decision
        if previous is not None:
            if decision.decision_version < previous.decision_version:
                raise ValueError("eager decision version regressed")
            if decision.decision_version == previous.decision_version and decision != previous:
                raise ValueError("changed eager decision requires a new version")
        state.eager_decision = decision
        if not (decision.eligible and decision.enabled) and state.active_continuation_id:
            self.discard_continuation(
                request_id, state.active_continuation_id, reason="eager_disabled"
            )
        return decision

    def _work(self, state, kind, dependency, budget, parent=None) -> DraftWork:
        work = DraftWork(
            self.owner_id, state.request_id, self._id(kind), parent,
            state.committed_prefix_version, dependency, budget, kind,
            _common_prefix(state.materialized_kv_prefix, dependency),
            state.eager_decision.decision_version, state.eos_token_ids,
        )
        self._works[work.work_id] = work
        self._work_digests[work.work_id] = _fingerprint(work)
        return work

    def schedule_normal_recovery(self, request_id) -> DraftWork:
        state = self._state(request_id)
        if state.finished or state.current_proposal_id or state.active_continuation_id:
            raise ValueError("normal draft requires a live request without pending proposal")
        if state.normal_work_id:
            return self._works[state.normal_work_id]
        kind = "recovery" if state.recovery_required else "normal"
        work = self._work(
            state, kind, state.committed_prefix,
            min(NORMAL_CANDIDATE_LENGTH, max(state.remaining_output_tokens - 1, 0)),
        )
        state.normal_work_id = work.work_id
        if kind == "recovery":
            state.accounting.recovery_jobs += 1
        return work

    def _validate_completion(self, work, completion):
        state = self._state(work.request_id)
        if work.owner_id != self.owner_id:
            raise ValueError("Draft completion belongs to another owner")
        if self._work_digests.get(work.work_id) != _fingerprint(work):
            raise ValueError("unknown or stale Draft work identity/dependency/version")
        if completion.work_id != work.work_id:
            raise ValueError("completion work identity mismatch")
        tokens = token_tuple(completion.generated_tokens)
        budget = work.candidate_length + int(work.kind == "continuation")
        if len(tokens) > budget or (len(tokens) < budget and (
            not tokens or tokens[-1] not in work.eos_token_ids
        )):
            raise ValueError("incomplete Draft data or candidate length mismatch")
        if any(t in work.eos_token_ids for t in tokens[:-1]):
            raise ValueError("Draft generation continued after EOS")
        expected = len(work.dependency_prefix) + max(len(tokens) - 1, 0)
        if (type(completion.materialized_kv_frontier) is not int
                or completion.materialized_kv_frontier != expected):
            raise ValueError("Draft completion materialized KV frontier mismatch")
        old = self._completion_digests.get(work.work_id)
        digest = _fingerprint(completion)
        if old is not None and old != digest:
            raise ValueError("conflicting duplicate Draft completion")
        return state, tokens, old is not None, digest

    def _count_completion(self, state, work, tokens, digest):
        self._completion_digests[work.work_id] = digest
        state.accounting.draft_materialized_tokens += (
            len(work.dependency_prefix) - len(work.base_kv_prefix) + max(len(tokens) - 1, 0)
        )
        self._works.pop(work.work_id, None)

    def record_normal_completion(self, work, completion) -> Optional[Proposal]:
        state, tokens, duplicate, digest = self._validate_completion(work, completion)
        if work.kind not in ("normal", "recovery"):
            raise ValueError("normal completion requires a normal/recovery work item")
        if duplicate:
            pid = self._normal_results.get(work.work_id)
            return self.proposal(work.request_id, pid) if pid and not state.finished else None
        if not state.finished and (
            state.normal_work_id != work.work_id
            or state.committed_prefix_version != work.prefix_version
            or state.committed_prefix != work.dependency_prefix
        ):
            raise ValueError("stale normal completion prefix/version")
        self._count_completion(state, work, tokens, digest)
        state.accounting.normal_generated_tokens += len(tokens)
        if work.kind == "recovery":
            state.accounting.recovery_generated_tokens += len(tokens)
        if state.finished:
            return None
        proposal = Proposal(self._id("proposal"), tokens, work.prefix_version,
                            work.dependency_prefix, request_id=state.request_id,
                            owner_id=self.owner_id)
        state.proposals[proposal.proposal_id] = proposal
        state.current_proposal_id = proposal.proposal_id
        state.normal_work_id = None
        state.recovery_required = False
        state.materialized_kv_prefix = work.dependency_prefix + tokens[:-1]
        self._normal_results[work.work_id] = proposal.proposal_id
        return copy.deepcopy(proposal)

    def start_verification(self, request_id, proposal_id, *, eager=False) -> Proposal:
        state = self._state(request_id)
        decision = self.evaluate_eager_eligibility(request_id)
        proposal = state.proposals.get(proposal_id)
        if state.finished or state.recovery_required or proposal is None or (
            state.current_proposal_id != proposal_id or proposal.status != "ready"
            or proposal.prefix_version != state.committed_prefix_version
            or proposal.parent_prefix != state.committed_prefix
        ):
            raise ValueError("proposal is not a current legal READY proposal")
        if eager and not (decision.eligible and decision.enabled
                          and proposal.source_continuation_id is not None):
            raise ValueError("eager verification admission is disabled or ineligible")
        proposal.status = "verifying"
        if proposal.source_continuation_id:
            state.continuations[proposal.source_continuation_id].status = (
                ContinuationStatus.CONSUMED
            )
        return copy.deepcopy(proposal)

    def begin_continuation(self, request_id, *, admitted=True) -> Optional[DraftWork]:
        state = self._state(request_id)
        decision = self.evaluate_eager_eligibility(request_id)
        if state.finished or not (decision.eligible and decision.enabled) or not admitted:
            return None
        parent = state.proposals.get(state.current_proposal_id)
        if parent is None or parent.status != "verifying":
            raise ValueError("continuation requires an in-flight Target parent")
        if state.active_continuation_id or any(
            c.parent_proposal_id == parent.proposal_id for c in state.continuations.values()
        ):
            raise ValueError("only one continuation may be started for a parent")
        remaining = state.remaining_output_tokens - len(parent.tokens) - 1
        budget = min(EAGER_CANDIDATE_LENGTH, max(remaining - 1, 0))
        if budget == 0 or any(t in state.eos_token_ids for t in parent.tokens):
            return None
        work = self._work(state, "continuation", parent.parent_prefix + parent.tokens,
                          budget, parent.proposal_id)
        state.continuations[work.work_id] = Continuation(
            work.work_id, parent.proposal_id, work.prefix_version, self._sequence,
            work.dependency_prefix,
        )
        state.active_continuation_id = work.work_id
        return work

    def record_continuation_completion(self, work, completion) -> ContinuationStatus:
        state, tokens, duplicate, digest = self._validate_completion(work, completion)
        if work.kind != "continuation":
            raise ValueError("continuation completion requires continuation work")
        continuation = state.continuations[work.work_id]
        if duplicate:
            return continuation.status
        self._count_completion(state, work, tokens, digest)
        continuation.generated_count = len(tokens)
        continuation.completed = True
        state.accounting.early_generated_tokens += len(tokens)
        state.accounting.bridge_generated_tokens += int(bool(tokens))
        if state.finished or continuation.status in (
            ContinuationStatus.INVALIDATED, ContinuationStatus.CANCELLED,
            ContinuationStatus.RELEASED,
        ):
            self._count_discard(state, continuation)
            return continuation.status
        continuation.predicted_bridge = tokens[0]
        continuation.continuation_tokens = tokens[1:]
        continuation.dependency_prefix = work.dependency_prefix + tokens[:1]
        continuation.materialized_kv_frontier = completion.materialized_kv_frontier
        if continuation.parent_confirmed:
            self._resolve_continuation(state, continuation)
        else:
            continuation.status = ContinuationStatus.WAITING_PARENT
        return continuation.status

    def record_continuation_abort(self, work, generated_tokens, materialized_kv_frontier):
        """Retire partial GPU work after its owner has fenced and invalidated it.

        This receipt never creates completed reusable candidate data. It records
        actual work stopped at a token boundary, including a zero-work admission.
        """
        state = self._state(work.request_id)
        if work.owner_id != self.owner_id or (
            self._work_digests.get(work.work_id) != _fingerprint(work)
        ):
            raise ValueError("unknown or foreign aborted continuation")
        continuation = state.continuations[work.work_id]
        if continuation.status not in (
            ContinuationStatus.INVALIDATED, ContinuationStatus.CANCELLED,
            ContinuationStatus.RELEASED,
        ):
            raise ValueError("partial work must be invalidated before retirement")
        tokens = token_tuple(generated_tokens)
        maximum = len(work.dependency_prefix) + max(len(tokens) - 1, 0)
        if (len(tokens) > work.candidate_length + 1
                or type(materialized_kv_frontier) is not int
                or not len(work.base_kv_prefix) <= materialized_kv_frontier <= maximum
                or (tokens and materialized_kv_frontier != maximum)):
            raise ValueError("invalid partial GPU generation/frontier evidence")
        abort_digest = _fingerprint(("aborted", tokens, materialized_kv_frontier))
        if continuation.completed:
            # Repeated retirement must preserve exact physical evidence, whether
            # the first receipt was a completed branch or a partial abort.
            completion_digest = _fingerprint(
                DraftCompletion(work.work_id, tokens, materialized_kv_frontier)
            )
            if self._completion_digests.get(work.work_id) not in (
                abort_digest, completion_digest,
            ):
                raise ValueError("conflicting already completed GPU retirement")
            return continuation.status
        self._completion_digests[work.work_id] = abort_digest
        self._works.pop(work.work_id, None)
        continuation.completed = True
        continuation.generated_count = len(tokens)
        state.accounting.early_generated_tokens += len(tokens)
        state.accounting.bridge_generated_tokens += int(bool(tokens))
        state.accounting.draft_materialized_tokens += (
            materialized_kv_frontier - len(work.base_kv_prefix)
        )
        self._count_discard(state, continuation)
        return continuation.status

    def resolve_parent_verification(self, receipt: ParentVerification) -> bool:
        state = self._state(receipt.request_id)
        if receipt.owner_id != self.owner_id:
            raise ValueError("Target receipt belongs to another owner")
        digest = _fingerprint(receipt)
        old = self._receipt_digests.get(receipt.proposal_id)
        if old is not None:
            if old != digest:
                raise ValueError("conflicting duplicate Target receipt")
            return False
        if state.finished:
            return False
        proposal = state.proposals.get(receipt.proposal_id)
        if proposal is None or state.current_proposal_id != receipt.proposal_id or (
            proposal.status != "verifying" or receipt.prefix_version != proposal.prefix_version
            or receipt.prefix_version != state.committed_prefix_version
            or receipt.parent_prefix != proposal.parent_prefix
            or receipt.parent_prefix != state.committed_prefix
        ):
            raise ValueError("stale Target parent proposal/prefix/version")
        delta = token_tuple(receipt.committed_delta)
        terminal = receipt.terminal_reason is not None
        if receipt.terminal_reason not in (None, "eos", "length", "cancelled"):
            raise ValueError("unknown Target terminal reason")
        if not delta or len(delta) > state.remaining_output_tokens:
            raise ValueError("Target delta exceeds remaining output or is empty")
        eos = any(t in state.eos_token_ids for t in delta)
        if any(t in state.eos_token_ids for t in delta[:-1]):
            raise ValueError("Target committed tokens beyond EOS")
        if eos and receipt.terminal_reason != "eos":
            raise ValueError("Target EOS requires EOS termination")
        if receipt.terminal_reason == "eos" and not eos:
            raise ValueError("EOS termination lacks an EOS token")
        if receipt.terminal_reason == "length" and len(delta) != state.remaining_output_tokens:
            raise ValueError("length termination before output budget")
        if not terminal and len(delta) == state.remaining_output_tokens:
            raise ValueError("output budget exhausted without termination")
        decision = dual_greedy_acceptance(proposal.tokens, delta, terminal=terminal)
        # A failing/regressed provider must not leave half-applied Target output
        # that a duplicate receipt would subsequently consider resolved.
        if not terminal:
            self.evaluate_eager_eligibility(receipt.request_id)
        self._receipt_digests[receipt.proposal_id] = digest
        state.committed_prefix += delta
        state.committed_prefix_version += 1
        state.accounting.committed_tokens += len(delta)
        state.accounting.parent_accepted_tokens += len(decision.accepted_draft_token_ids)
        state.accounting.correction_tokens += len(decision.target_correction_token_ids)
        state.accounting.bonus_tokens += len(decision.target_bonus_token_ids)
        proposal.status = "consumed"
        state.current_proposal_id = None
        state.materialized_kv_prefix = _common_prefix(
            state.materialized_kv_prefix, state.committed_prefix
        )
        if terminal:
            self.finish_or_cancel(receipt.request_id, reason=receipt.terminal_reason)
            return True
        cid = state.active_continuation_id
        if cid is None:
            state.recovery_required = bool(decision.rejected_draft_token_ids) or any(
                c.parent_proposal_id == proposal.proposal_id
                and c.status in (ContinuationStatus.CANCELLED, ContinuationStatus.INVALIDATED)
                for c in state.continuations.values()
            )
        else:
            continuation = state.continuations[cid]
            continuation.parent_confirmed = True
            if decision.rejected_draft_token_ids:
                self.discard_continuation(receipt.request_id, cid, reason="parent_rejected")
            else:
                self._resolve_continuation(state, continuation)
        return True

    def _resolve_continuation(self, state, continuation):
        if not continuation.completed:
            continuation.status = ContinuationStatus.WAITING_DRAFT
            return
        parent = state.proposals[continuation.parent_proposal_id]
        actual_base = parent.parent_prefix + parent.tokens
        if (state.committed_prefix_version != continuation.prefix_version + 1
                or state.committed_prefix[:-1] != actual_base):
            self.discard_continuation(state.request_id, continuation.continuation_id,
                                      reason="dependency_mismatch")
        elif continuation.predicted_bridge != state.committed_prefix[-1]:
            self.discard_continuation(state.request_id, continuation.continuation_id,
                                      reason="bridge_mismatch")
        elif continuation.dependency_prefix != state.committed_prefix:
            self.discard_continuation(state.request_id, continuation.continuation_id,
                                      reason="dependency_mismatch")
        elif not continuation.continuation_tokens:
            self.discard_continuation(state.request_id, continuation.continuation_id,
                                      reason="empty_continuation")
        else:
            continuation.status = ContinuationStatus.PROMOTABLE

    def promote_continuation(self, request_id, continuation_id) -> Proposal:
        state = self._state(request_id)
        decision = self.evaluate_eager_eligibility(request_id)
        continuation = state.continuations[continuation_id]
        if state.finished or not (decision.eligible and decision.enabled) or (
            state.active_continuation_id != continuation_id
            or continuation.status != ContinuationStatus.PROMOTABLE
            or state.current_proposal_id is not None
            or continuation.dependency_prefix != state.committed_prefix
            or continuation.prefix_version + 1 != state.committed_prefix_version
        ):
            raise ValueError("continuation is not valid for promotion or already consumed")
        proposal = Proposal(
            self._id("proposal"), continuation.continuation_tokens,
            state.committed_prefix_version, state.committed_prefix,
            continuation.parent_proposal_id, continuation_id,
            request_id=state.request_id, owner_id=self.owner_id,
        )
        state.proposals[proposal.proposal_id] = proposal
        state.current_proposal_id = proposal.proposal_id
        state.active_continuation_id = None
        state.recovery_required = False
        state.materialized_kv_prefix = (
            continuation.dependency_prefix + continuation.continuation_tokens[:-1]
        )
        continuation.status = ContinuationStatus.PROMOTED
        state.accounting.reused_candidate_tokens += len(proposal.tokens)
        state.accounting.reused_bridge_tokens += 1
        return copy.deepcopy(proposal)

    @staticmethod
    def _count_discard(state, continuation):
        if continuation.completed and not continuation.discarded_counted:
            state.accounting.discarded_early_tokens += continuation.generated_count
            continuation.discarded_counted = True

    def discard_continuation(self, request_id, continuation_id, reason="discarded"):
        state = self._state(request_id)
        continuation = state.continuations[continuation_id]
        if continuation.status in (
            ContinuationStatus.PROMOTED, ContinuationStatus.CONSUMED,
        ):
            raise ValueError("promoted proposal has its own lifecycle")
        if continuation.status in (
            ContinuationStatus.INVALIDATED, ContinuationStatus.CANCELLED,
            ContinuationStatus.RELEASED,
        ):
            return
        continuation.status = (ContinuationStatus.CANCELLED if reason == "eager_disabled"
                               else ContinuationStatus.INVALIDATED)
        continuation.reason = reason
        self._count_discard(state, continuation)
        continuation.dependency_prefix = ()
        continuation.continuation_tokens = ()
        continuation.predicted_bridge = None
        continuation.materialized_kv_frontier = 0
        self._works.pop(continuation_id, None)
        if state.active_continuation_id == continuation_id:
            state.active_continuation_id = None
        if not state.current_proposal_id and not state.finished:
            state.recovery_required = True

    def finish_or_cancel(self, request_id, reason="cancelled"):
        state = self._state(request_id)
        if state.finished:
            return
        if reason not in ("eos", "length", "cancelled", "owner_shutdown"):
            raise ValueError("unknown finish reason")
        state.finished = True
        state.terminal_reason = reason
        state.current_proposal_id = None
        state.active_continuation_id = None
        state.normal_work_id = None
        state.recovery_required = False
        state.materialized_kv_prefix = ()
        for proposal in state.proposals.values():
            proposal.status = "released"
            proposal.tokens = ()
            proposal.parent_prefix = ()
        for continuation in state.continuations.values():
            if continuation.status not in (
                ContinuationStatus.PROMOTED, ContinuationStatus.CONSUMED,
            ):
                self._count_discard(state, continuation)
            continuation.status = ContinuationStatus.RELEASED
            continuation.release_reason = reason
            continuation.dependency_prefix = ()
            continuation.continuation_tokens = ()
            continuation.predicted_bridge = None
            continuation.materialized_kv_frontier = 0
        for wid, work in list(self._works.items()):
            if work.request_id == request_id:
                del self._works[wid]

    def shutdown(self):
        self._check()
        for request_id in self._requests:
            self.finish_or_cancel(request_id, reason="owner_shutdown")
        self._closed = True
