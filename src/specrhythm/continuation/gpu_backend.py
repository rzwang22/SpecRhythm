"""Opt-in physical paged-KV execution for the shared continuation protocol.

The existing request handle remains the sole KV owner. Pending continuation
tokens extend that allocation; committed prefix/proposal fields remain frozen
until a validated parent settlement. Blocks above a rolled-back logical frontier
are retained as private capacity and overwritten before reuse, then freed by the
worker's fenced request release. Baseline ``commit_frontier`` is unchanged.
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Tuple

from specrhythm.continuation.core import DraftCompletion, DraftWork
from specrhythm.continuation.trace import TRACE
from specrhythm.phase4.draft_batch import (
    DraftCommitPlan,
    DraftMaterialization,
    commit_frontier,
    token_tuple,
    unique_ids,
)
from specrhythm.phase4.draft_metrics import batch_statistics
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.vllm_draft_backend import VllmBatchedDraftBackend


def _digest(value):
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


@dataclass
class _GPUContinuation:
    work: DraftWork
    generated: list[int] = field(default_factory=list)
    complete: bool = False
    aborted: bool = False
    started_ns: Optional[int] = None
    completed_ns: Optional[int] = None


class GPUContinuationBackendMixin:
    """Add bounded, batchable GPU steps to an existing resident Draft backend.

Compose before VllmBatchedDraftBackend or FixedDraftBackend. Enrollment launches
no GPU work. Each step fences its own real forward before returning, allowing the
same owner to process invalidation/termination ahead of the next token step.
"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gpu_continuations: dict[str, _GPUContinuation] = {}
        self._gpu_retired: dict[str, dict] = {}
        self._gpu_rebases: dict[tuple, tuple] = {}
        self._gpu_owner_id = None
        self._gpu_writing = False
        self.gpu_continuation_records = []
        self.metrics.batches["eager"] = Counter()
        self._provenance["rolling_eager_gpu"] = {
            "protocol": "shared-continuation-v1",
            "one_existing_request_kv_owner": True,
            "incremental_batched_steps": True,
            "completion_evidence": "worker materialize/greedy/fence",
            "rollback_blocks": "retain private high-water blocks; overwrite invalid suffix",
            "full_prefix_replay": False,
        }

    def _gpu_check(self):
        self._check()
        if self._gpu_writing:
            raise RuntimeError("GPU KV mutation attempted before the active write fence")

    @TRACE.observe("physical_gpu_audit")
    def _gpu_audit(self, request_ids=(), *, admitting=False, settling=False):
        audit = getattr(self, "_audit", None)
        if audit is not None:
            packet = audit()
            if admitting and any(
                packet["requests"][rid]["state"] != "ACTIVE" for rid in request_ids
            ):
                raise ValueError("eager GPU work requires admitted ACTIVE requests")
            if settling and any(
                packet["requests"][rid]["state"] not in ("ACTIVE", "FINISHED")
                for rid in request_ids
            ):
                raise ValueError("S2 Draft commit before admission")

    def _gpu_job(self, work):
        self._gpu_check()
        job = self._gpu_continuations.get(work.work_id)
        if job is None or job.work != work:
            raise ValueError("unknown, retired or conflicting GPU continuation work")
        return job

    def begin_gpu_continuation(self, work: DraftWork):
        return self.begin_gpu_continuations((work,))[work.work_id]

    @TRACE.observe("physical_begin_gpu_continuations")
    def begin_gpu_continuations(self, works):
        """Validate the whole owner-local batch before publishing any enrollment.

        Enrollment changes bookkeeping only, not KV/frontiers or the allocator.
        One fresh pool/control audit covers this stable batch. Nothing is cached
        across owner commands: stepping audits again, before and after KV writes.
        """
        works = tuple(works)
        self._gpu_check()
        unique_ids([w.request_id for w in works])
        unique_ids([w.work_id for w in works])
        owners = {w.owner_id for w in works}
        if self._gpu_owner_id is not None:
            owners.add(self._gpu_owner_id)
        if len(owners) > 1:
            raise ValueError("GPU continuation belongs to another protocol owner")
        pending = []
        for work in works:
            if work.work_id in self._gpu_continuations:
                self._gpu_job(work)
                continue
            self._validate_gpu_admission(work)
            pending.append(work)
        if pending:
            self._gpu_audit([w.request_id for w in pending], admitting=True)
            for work in pending:
                self._gpu_continuations[work.work_id] = _GPUContinuation(work)
            self._gpu_owner_id = pending[0].owner_id
            self.metrics.counters["eager_enrolled"] += len(pending)
        return {w.work_id: self.gpu_continuation_snapshot(w) for w in works}

    @TRACE.observe("physical_validate_gpu_admission")
    def _validate_gpu_admission(self, work):
        if work.work_id in self._gpu_retired:
            raise ValueError("retired GPU continuation cannot restart")
        state = self.states[work.request_id]
        if (
            work.kind != "continuation" or not work.parent_proposal_id
            or not 1 <= work.candidate_length <= 4
            or state.proposal is None or state.next_round != work.prefix_version
            or state.prefix + state.proposal != work.dependency_prefix
            or state.materialized != len(state.prefix) + max(len(state.proposal) - 1, 0)
            or state.next_logits is None
        ):
            raise ValueError("GPU continuation parent/proposal/version/frontier mismatch")
        if work.base_kv_prefix != work.dependency_prefix[:len(work.base_kv_prefix)] or (
            len(work.base_kv_prefix) > state.materialized
        ):
            raise ValueError("GPU continuation base KV does not match physical parent")
        if len(work.dependency_prefix) + work.candidate_length + 1 > self.max_model_len:
            raise ValueError("GPU continuation exceeds model context capacity")
        if any(j.work.request_id == work.request_id for j in self._gpu_continuations.values()):
            raise ValueError("request already owns a future GPU continuation")

    def step_gpu_continuation(self, work):
        return self.step_gpu_continuations((work,))[work.work_id]

    def step_gpu_continuations(self, works):
        self._gpu_check()
        unique_ids([w.request_id for w in works])
        jobs = [self._gpu_job(work) for work in works]
        if any(job.aborted for job in jobs):
            raise ValueError("aborted GPU continuation cannot execute another step")
        active = [job for job in jobs if not job.complete]
        if not active:
            return {job.work.work_id: self._gpu_completion(job) for job in jobs}
        self._gpu_audit([job.work.request_id for job in active])
        started = time.monotonic_ns()
        rows = []
        for job in active:
            state = self.states[job.work.request_id]
            context = job.work.dependency_prefix + tuple(job.generated)
            # The first row fills the parent's sampled final token; later rows
            # fill the bridge and candidate inputs, one position per request.
            if state.materialized != len(context) - 1:
                raise ValueError("GPU continuation progress/logits frontier mismatch")
            rows.append(DraftMaterialization(state.internal_id, context, state.materialized))
        try:
            self._gpu_writing = True
            with TRACE.span("eager_token_forward", bindings=[{
                "request_id": j.work.request_id, "round_id": j.work.prefix_version,
                "proposal_id": j.work.parent_proposal_id, "continuation_id": j.work.work_id,
                "token_step": len(j.generated),
                "token_role": "bridge" if not j.generated else "candidate",
            } for j in active], B=len(active)):
                logits = self.worker.materialize(rows, "eager")
            sampled = self.worker.greedy([logits[row.request_id] for row in rows])
            if len(sampled) != len(active):
                raise RuntimeError("GPU continuation sampler row count mismatch")
            token_tuple(sampled)
            # This is the safe owner boundary. No rebase/release can execute
            # until all submitted writes and their sample have completed.
            self.worker.fence("eager_step_complete")
            self._gpu_writing = False
            completed = time.monotonic_ns()
            self.metrics.active_steps[len(active)] += 1
            for job, row, token in zip(active, rows, sampled):
                state = self.states[job.work.request_id]
                state.materialized = len(row.context)
                state.next_logits = logits[row.request_id]
                if not job.generated:
                    self.metrics.counters["eager_started"] += 1
                job.generated.append(token)
                job.started_ns = job.started_ns or started
                job.complete = (
                    token in job.work.eos_token_ids
                    or len(job.generated) == job.work.candidate_length + 1
                )
                if job.complete:
                    job.completed_ns = completed
                    self.metrics.counters["eager_completed"] += 1
                self.metrics.counters["eager_generated_tokens"] += 1
                self.metrics.counters["eager_materialized_tokens"] += len(row.suffix)
                self.metrics.counters["eager_bridge_tokens"] += int(len(job.generated) == 1)
            record = {
                "operation": "eager_gpu_step", "host_start_ns": started,
                "host_end_ns": completed, "request_ids": [j.work.request_id for j in active],
                "work_ids": [j.work.work_id for j in active], "B": len(active),
                "materialized_positions": [r.valid_length for r in rows],
                "worker_fenced": True,
                "gpu_overlap": "UNKNOWN; correlate actual device timeline events",
            }
            self.gpu_continuation_records.append(record)
            if hasattr(self, "history"):
                self.history.append(dict(record))
            self._gpu_audit()
            return {job.work.work_id: self._gpu_completion(job) for job in jobs}
        except Exception:
            # A failed forward may still have launched writes. Attempt a fence
            # before permitting shutdown; never claim normal recoverability.
            try:
                self.worker.fence("eager_failed_step")
            finally:
                self._gpu_writing = False
                self._fail()
            raise

    def _gpu_completion(self, job):
        if not job.complete:
            return None
        return DraftCompletion(
            job.work.work_id, tuple(job.generated), self.states[job.work.request_id].materialized
        )

    def gpu_continuation_snapshot(self, work):
        self._gpu_check()
        retired = self._gpu_retired.get(work.work_id)
        if retired is not None:
            if retired["work_digest"] != _digest(work):
                raise ValueError("conflicting retired GPU continuation work")
            return dict(retired)
        job = self._gpu_job(work)
        return {
            "work_digest": _digest(work), "work_id": work.work_id,
            "generated_tokens": tuple(job.generated),
            "materialized_kv_frontier": self.states[work.request_id].materialized,
            "complete": job.complete, "aborted": job.aborted, "retired": False,
            "started_ns": job.started_ns, "completed_ns": job.completed_ns,
        }

    def abort_gpu_continuation(self, work):
        self._gpu_check()
        if work.work_id in self._gpu_retired:
            return self.gpu_continuation_snapshot(work)
        job = self._gpu_job(work)
        if not job.aborted:
            self.worker.fence("eager_abort")
            job.aborted = True
            self.metrics.counters["eager_aborted"] += 1
        return self.gpu_continuation_snapshot(work)

    def _gpu_retire(self, work, reason, snapshot=None):
        snapshot = dict(snapshot or self.gpu_continuation_snapshot(work))
        snapshot.update(retired=True, retire_reason=reason)
        self._gpu_retired[work.work_id] = snapshot
        del self._gpu_continuations[work.work_id]
        return snapshot

    def rebase_gpu_parent(
        self, plan: DraftCommitPlan, *, work=None,
        promoted_tokens: Optional[Tuple[int, ...]] = None,
    ):
        return self.rebase_gpu_parents(((plan, work, promoted_tokens),))[plan.request_id]

    def _prepare_gpu_parent(self, plan, work, promoted_tokens):
        """Pure per-request validation/row planning; no KV or ledger publication."""
        state = self.states[plan.request_id]
        if len(plan.final_prefix) > self.max_model_len:
            raise ValueError("GPU settlement exceeds model context")
        if work is None:
            if promoted_tokens is not None:
                raise ValueError("GPU promotion requires its physical continuation work")
            if any(j.work.request_id == plan.request_id for j in self._gpu_continuations.values()):
                raise ValueError("GPU parent settlement omitted its active continuation")
            terminal_tail = (state.proposal is None and not plan.proposal and plan.terminal
                             and len(plan.target_tail) == 1 and plan.accepted == 0)
            if (state.next_round != plan.round_id or state.prefix != plan.parent_prefix
                    or (state.proposal != plan.proposal and not terminal_tail)):
                raise ValueError("stale Draft commit proposal/prefix/round")
            valid = commit_frontier(plan, state.materialized)
            refresh = (valid == len(plan.final_prefix) and not plan.terminal
                       and valid != state.materialized)
            frontier = valid - int(refresh)
            row = (DraftMaterialization(state.internal_id, plan.final_prefix, frontier)
                   if frontier < len(plan.final_prefix) else None)
            return row, valid, refresh
        job = self._gpu_job(work)
        if (
            work.request_id != plan.request_id or work.prefix_version != plan.round_id
            or state.next_round != plan.round_id or state.prefix != plan.parent_prefix
            or state.proposal != plan.proposal
            or work.dependency_prefix != plan.parent_prefix + plan.proposal
        ):
            raise ValueError("GPU settlement parent/request/proposal/version mismatch")
        expected = len(work.dependency_prefix) + len(job.generated) - 1
        if not job.generated:
            expected = len(plan.parent_prefix) + max(len(plan.proposal) - 1, 0)
            commit_frontier(plan, state.materialized)
        if state.materialized != expected:
            raise ValueError("GPU extended frontier lacks matching worker progress evidence")
        if promoted_tokens is not None:
            promoted_tokens = token_tuple(promoted_tokens)
            if (
                job.aborted or not job.complete or plan.terminal
                or plan.accepted != len(plan.proposal) or plan.bonus_count != 1
                or tuple(job.generated[:1]) != plan.target_tail
                or not promoted_tokens or promoted_tokens != tuple(job.generated[1:])
                or plan.final_prefix != work.dependency_prefix + tuple(job.generated[:1])
            ):
                raise ValueError("GPU continuation is not valid for physical promotion")
            return None, state.materialized, False
        valid = min(state.materialized, len(plan.parent_prefix) + plan.accepted)
        row = None
        if valid < len(plan.final_prefix) or not plan.terminal:
            # Preserve the existing shortened-frontier logit refresh rule.
            frontier = valid - int(valid == len(plan.final_prefix))
            row = DraftMaterialization(state.internal_id, plan.final_prefix, frontier)
        return row, valid, False

    @TRACE.observe("physical_rebase_gpu_parents")
    def rebase_gpu_parents(self, settlements):
        """Validate together, materialize compatible repair rows once, then publish.

        Same owner/model/allocator, ragged suffixes, and per-row frontier evidence
        are shared with the ordinary backend. The worker enforces its existing
        sequence/token limits; no silent B1 fallback or partial publication.
        """
        self._gpu_check()
        settlements = tuple(settlements)
        unique_ids([p.request_id for p, _, _ in settlements])
        result, prepared = {}, []
        for plan, work, tokens in settlements:
            fingerprint = _digest((plan, work, tokens))
            key = (plan.request_id, plan.round_id)
            old = self._gpu_rebases.get(key)
            if old is not None:
                if old[0] != fingerprint:
                    raise ValueError("conflicting duplicate GPU parent settlement")
                result[plan.request_id] = dict(old[1])
                continue
            row, valid, refresh = self._prepare_gpu_parent(plan, work, tokens)
            prepared.append((plan, work, tokens, row, valid, refresh, fingerprint))
        if not prepared:
            return result
        if all(work is None for _, work, *_ in prepared):
            # Eager disabled/tail-only batches keep the ordinary batched path,
            # including its S2 admission audits and baseline commit_frontier.
            ordinary = super().commit_many([p for p, *_ in prepared])
            for plan, _, _, row, _, _, fingerprint in prepared:
                value = {**ordinary[plan.request_id],
                         "materialized_query_tokens": len(row.suffix) if row else 0}
                self._gpu_rebases[(plan.request_id, plan.round_id)] = (fingerprint, dict(value))
                result[plan.request_id] = value
            return result
        self._gpu_audit([p.request_id for p, *_ in prepared], settling=True)
        try:
            self.worker.fence("eager_parent_settlement")
            snapshots = {}
            for plan, work, tokens, *_ in prepared:
                if work is not None:
                    snapshots[plan.request_id] = self.gpu_continuation_snapshot(work)
                    job = self._gpu_job(work)
                    if tokens is None:
                        # The shared fence above covers these bookkeeping aborts.
                        if not job.aborted:
                            job.aborted = True
                            self.metrics.counters["eager_aborted"] += 1
                        snapshots[plan.request_id]["aborted"] = True
            rows = [row for _, _, _, row, *_ in prepared if row is not None]
            logits = {}
            if rows:
                self._gpu_writing = True
                for plan, _, _, row, *_ in prepared:
                    if row is not None:
                        self.states[plan.request_id].next_logits = None
                with TRACE.span("parent_kv_materialize", B=len(rows),
                                request_ids=[r.request_id for r in rows]):
                    logits = self.worker.materialize(rows, "commit")
                self.worker.fence("eager_rebase_complete")
                self._gpu_writing = False
            for plan, work, tokens, row, valid, refresh, _ in prepared:
                state = self.states[plan.request_id]
                query_tokens = len(row.suffix) if row else 0
                if tokens is None:
                    state.materialized = len(plan.final_prefix)
                    state.next_logits = logits.get(state.internal_id, state.next_logits)
                    if plan.terminal:
                        state.next_logits = None
                    state.proposal = None
                else:
                    state.proposal = token_tuple(tokens)
                    self.metrics.counters["eager_promoted"] += 1
                    self.metrics.counters["eager_promoted_candidates"] += len(tokens)
                if work is not None:
                    if tokens is None:
                        self.metrics.counters["eager_discarded_tokens"] += len(
                            self._gpu_job(work).generated)
                        self.metrics.counters["eager_rebase_materialized_tokens"] += query_tokens
                    retired = self._gpu_retire(
                        work, "promoted" if tokens is not None else "discarded",
                        snapshots[plan.request_id])
                    continuation = {"continuation": retired,
                                    "promoted_candidates": len(tokens or ())}
                else:
                    self.metrics.counters["rollback_requests"] += 1
                    self.metrics.counters["commit_materialized_tokens"] += query_tokens
                    self.metrics.counters["logit_refresh_requests"] += int(refresh)
                    continuation = {"valid_prefix_before_materialization": valid}
                state.prefix = plan.final_prefix
                state.next_round += 1
                for name, count in (("commits", 1), ("correction_tokens", plan.correction_count),
                                    ("bonus_tokens", plan.bonus_count),
                                    ("invalidated_tokens", len(plan.proposal)-plan.accepted)):
                    self.metrics.counters[name] += count
                result[plan.request_id] = {
                    "materialized_kv_length": state.materialized,
                    "committed_prefix_hash": token_prefix_hash(state.prefix),
                    "materialized_query_tokens": query_tokens, **continuation,
                    **self.worker.request_evidence(state.internal_id),
                }
            if any(work is None for _, work, *_ in prepared):
                self.metrics.counters["commit_batches"] += 1
                self.metrics.counters["correction_bonus_batches"] += int(any(
                    work is None and row is not None and plan.target_tail
                    for plan, work, _, row, *_ in prepared))
            # Validate the new pool before releasing terminal allocations. Release
            # is another physical-state change, so audit it separately when needed.
            self._gpu_audit()
            terminal_ids = [p.request_id for p, *_ in prepared if p.terminal]
            if terminal_ids:
                self.finish_many(terminal_ids)
                self._gpu_audit()
            for plan, _, _, _, _, _, fingerprint in prepared:
                self._gpu_rebases[(plan.request_id, plan.round_id)] = (
                    fingerprint, dict(result[plan.request_id]))
            return result
        except Exception:
            try:
                self.worker.fence("eager_failed_settlement")
            finally:
                self._gpu_writing = False
                self._fail()
            raise

    def finish_many(self, request_ids):
        self._gpu_check()
        unique_ids(request_ids)
        if set(request_ids) - set(self.states) - self.retired:
            raise ValueError("cannot finish an unknown Draft request")
        if not request_ids:
            return
        for job in list(self._gpu_continuations.values()):
            if job.work.request_id in request_ids:
                self.abort_gpu_continuation(job.work)
                self._gpu_retire(job.work, "released")
        self.worker.fence("eager_release_boundary")
        return super().finish_many(request_ids)

    def commit_many(self, plans):
        self._gpu_check()
        active = {job.work.request_id for job in self._gpu_continuations.values()}
        if any(plan.request_id in active for plan in plans):
            raise ValueError("active continuation requires explicit GPU parent settlement")
        return super().commit_many(plans)

    def report(self):
        result = super().report()
        purposes = ("proposal", "commit", "eager")
        measured = sum((self.metrics.batches[p] for p in purposes), Counter())
        stats = batch_statistics(measured)
        result.update({
            "draft_model_forward_count": sum(self.metrics.forwards[p] for p in purposes),
            "draft_batch_count": stats["count"],
            "draft_batch_size_histogram": stats["histogram"],
            "draft_gpu_event_time_ms": sum(self.metrics.gpu_ms[p] for p in purposes),
            "true_batching_observed": bool(stats["max"] and stats["max"] > 1),
            "eager_gpu_live_work": len(self._gpu_continuations),
            "eager_gpu_retired_work": len(self._gpu_retired),
            "eager_gpu_write_inflight": self._gpu_writing,
            "eager_gpu_steps": list(self.gpu_continuation_records),
            "counter_scope": "proposal/commit/eager; setup and warmup separately reported",
        })
        for key in ("min", "p10", "p50", "p90", "max", "mean"):
            result[f"draft_batch_size_{key}"] = stats[key]
        return result

    def shutdown(self):
        if self.closed:
            return
        # Worker/backend failures still require physical teardown. Never clear the
        # in-flight ledger before the worker's mandatory shutdown fence/release.
        if not self.failed:
            self.finish_many(tuple(self.states))
        super().shutdown()
        if self.closed:
            self._gpu_continuations.clear()


class RollingVllmDraftBackend(GPUContinuationBackendMixin, VllmBatchedDraftBackend):
    """Standalone real-worker backend for GPU correctness runs."""
