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

    def _gpu_audit(self, request_ids=(), *, admitting=False):
        audit = getattr(self, "_audit", None)
        if audit is not None:
            packet = audit()
            if admitting and any(
                packet["requests"][rid]["state"] != "ACTIVE" for rid in request_ids
            ):
                raise ValueError("eager GPU work requires admitted ACTIVE requests")

    def _gpu_job(self, work):
        self._gpu_check()
        job = self._gpu_continuations.get(work.work_id)
        if job is None or job.work != work:
            raise ValueError("unknown, retired or conflicting GPU continuation work")
        return job

    def begin_gpu_continuation(self, work: DraftWork):
        self._gpu_check()
        if work.work_id in self._gpu_continuations:
            self._gpu_job(work)
            return self.gpu_continuation_snapshot(work)
        if work.work_id in self._gpu_retired:
            raise ValueError("retired GPU continuation cannot restart")
        if self._gpu_owner_id is not None and work.owner_id != self._gpu_owner_id:
            raise ValueError("GPU continuation belongs to another protocol owner")
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
        self._gpu_audit((work.request_id,), admitting=True)
        self._gpu_owner_id = work.owner_id
        self._gpu_continuations[work.work_id] = _GPUContinuation(work)
        self.metrics.counters["eager_enrolled"] += 1
        return self.gpu_continuation_snapshot(work)

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
        """Apply a core-authorized parent settlement to the physical allocation.

        Promotion is the sole extended-frontier path: validate full acceptance,
        exact bridge, full branch identity, completion and candidate content.
        Otherwise crop logically to accepted parent KV and overwrite only the
        missing correction/bonus suffix. This never relaxes commit_frontier.
        """
        self._gpu_check()
        key = (plan.request_id, plan.round_id)
        fingerprint = _digest((plan, work, promoted_tokens))
        old = self._gpu_rebases.get(key)
        if old is not None:
            if old[0] != fingerprint:
                raise ValueError("conflicting duplicate GPU parent settlement")
            return dict(old[1])
        if work is None:
            if promoted_tokens is not None:
                raise ValueError("GPU promotion requires its physical continuation work")
            if any(j.work.request_id == plan.request_id for j in self._gpu_continuations.values()):
                raise ValueError("GPU parent settlement omitted its active continuation")
            result = super().commit_many((plan,))[plan.request_id]
            self._gpu_rebases[key] = (fingerprint, dict(result))
            return result
        job = self._gpu_job(work)
        state = self.states[plan.request_id]
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
        if len(plan.final_prefix) > self.max_model_len:
            raise ValueError("GPU settlement exceeds model context")
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
        self._gpu_audit()
        try:
            self.worker.fence("eager_parent_settlement")
            snapshot = self.gpu_continuation_snapshot(work)
            if promoted_tokens is None:
                self.abort_gpu_continuation(work)
                snapshot["aborted"] = True
                valid = min(state.materialized, len(plan.parent_prefix) + plan.accepted)
                if valid < len(plan.final_prefix) or not plan.terminal:
                    # Recompute last-position logits if the accepted prefix is
                    # shortened without a tail. Cached deeper logits are invalid.
                    if valid == len(plan.final_prefix):
                        valid -= 1
                    row = DraftMaterialization(state.internal_id, plan.final_prefix, valid)
                    state.next_logits = None
                    self._gpu_writing = True
                    logits = self.worker.materialize((row,), "commit")
                    self.worker.fence("eager_rebase_complete")
                    self._gpu_writing = False
                    state.next_logits = logits[state.internal_id]
                    self.metrics.counters["eager_rebase_materialized_tokens"] += len(row.suffix)
                state.materialized = len(plan.final_prefix)
                if plan.terminal:
                    state.next_logits = None
                state.proposal = None
                self.metrics.counters["eager_discarded_tokens"] += len(job.generated)
            else:
                state.proposal = promoted_tokens
                self.metrics.counters["eager_promoted"] += 1
                self.metrics.counters["eager_promoted_candidates"] += len(promoted_tokens)
            state.prefix = plan.final_prefix
            state.next_round += 1
            self.metrics.counters["commits"] += 1
            self.metrics.counters["correction_tokens"] += plan.correction_count
            self.metrics.counters["bonus_tokens"] += plan.bonus_count
            self.metrics.counters["invalidated_tokens"] += len(plan.proposal) - plan.accepted
            retired = self._gpu_retire(
                work, "promoted" if promoted_tokens is not None else "discarded", snapshot
            )
            result = {
                "materialized_kv_length": state.materialized,
                "committed_prefix_hash": token_prefix_hash(state.prefix),
                "continuation": {**snapshot, "retired": retired["retired"]},
                "promoted_candidates": len(promoted_tokens or ()),
                **self.worker.request_evidence(state.internal_id),
            }
            if plan.terminal:
                # No further logits are consumed. Fenced release frees the whole
                # private allocation, including invalid speculative high water.
                self.finish_many((plan.request_id,))
            self._gpu_audit()
            self._gpu_rebases[key] = (fingerprint, dict(result))
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
