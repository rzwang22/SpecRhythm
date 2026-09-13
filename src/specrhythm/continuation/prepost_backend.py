"""Physical three-lookahead / one-common-step Draft implementation, no bridge token."""

from __future__ import annotations

import time
from collections import Counter

from specrhythm.continuation.gpu_backend import GPUContinuationBackendMixin
from specrhythm.continuation.prepost import PARAMETERS, PROTOCOL, PURPOSES, Lookahead
from specrhythm.continuation.prepost_records import Records
from specrhythm.continuation.trace import TRACE
from specrhythm.phase4.draft_batch import DraftMaterialization, token_tuple, unique_ids
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.serving.common import require


class PrePostBackendMixin(GPUContinuationBackendMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepost_jobs, self.prepost_retired = {}, {}
        self.prepost_forwards = Records()
        self.metrics.batches.update({p: Counter() for p in PURPOSES})
        self._provenance["prepost"] = dict(
            protocol=PROTOCOL,
            **PARAMETERS,
            candidate_root_is_committed=False,
            target_bonus_is_committed=False,
        )

    @TRACE.observe("prepost_admission")
    def enroll_prepost(self, works):
        self._gpu_check()
        unique_ids([w.request_id for w in works])
        if not works:
            return {}
        unique_ids([w.work_id for w in works])
        for work in works:
            self._validate_prepost(work)
            require(
                self.states[work.request_id].materialized == len(work.dependency_prefix) - 1,
                "prepost admission frontier mismatch",
            )
            require(
                1 <= work.limit <= 3 and work.work_id not in self.prepost_retired,
                "invalid or retired prepost work",
            )
            require(work.work_id not in self.prepost_jobs, "duplicate prepost enrollment")
            require(
                not any(j.work.request_id == work.request_id for j in self.prepost_jobs.values()),
                "overlapping request work",
            )
        self._gpu_audit([w.request_id for w in works], admitting=True)
        for work in works:
            self.prepost_jobs[work.work_id] = Lookahead(work)
        return {w.work_id: self.prepost_jobs[w.work_id] for w in works}

    def _validate_prepost(self, work):
        state = self.states[work.request_id]
        require(
            state.prefix == work.parent_prefix
            and state.next_round == work.prefix_version
            and state.proposal == work.proposal_tokens,
            "prepost physical dependency/proposal/version mismatch",
        )
        require(
            len(work.dependency_prefix) + work.limit + 1 <= self.max_model_len,
            "prepost lookahead exceeds model capacity",
        )

    def _forward_prepost(self, rows, purpose, bindings, sample_ids):
        """One actual ragged GPU forward and one batch fence; no per-request repair."""
        self._gpu_check()
        if not rows:
            return {}, {}, None
        started = time.monotonic_ns()
        before = self.metrics.forwards[purpose]
        try:
            self._gpu_writing = True
            with TRACE.span(
                purpose,
                bindings=bindings,
                B=len(rows),
                materialized_positions=[len(r.suffix) for r in rows],
            ):
                logits = self.worker.materialize(rows, purpose)
                sampled = (
                    self.worker.greedy([logits[rid] for rid in sample_ids]) if sample_ids else ()
                )
                require(len(sampled) == len(sample_ids), "prepost sampled batch mismatch")
                token_tuple(sampled)
                self.worker.fence(purpose + "_complete")
            self._gpu_writing = False
            require(
                self.metrics.forwards[purpose] == before + 1,
                "prepost operation must be one physical model forward",
            )
            row = dict(
                purpose=purpose,
                start_ns=started,
                end_ns=time.monotonic_ns(),
                B=len(rows),
                request_ids=[r.request_id for r in rows],
                bindings=bindings,
                materialized_positions=[len(r.suffix) for r in rows],
                generated_tokens=len(sampled),
                worker_fenced=True,
            )
            self.prepost_forwards.append(row)
            return logits, dict(zip(sample_ids, sampled)), row
        except BaseException:
            try:
                self.worker.fence("prepost_failed_write")
            finally:
                self._gpu_writing = False
                self._fail()
            raise

    def step_prepost(self, works):
        self._gpu_check()
        unique_ids([w.request_id for w in works])
        active = []
        for work in works:
            self._validate_prepost(work)
            job = self.prepost_jobs.get(work.work_id)
            require(
                job is not None and job.work == work and not job.aborted,
                "late/cancelled prepost step",
            )
            if not job.completed:
                active.append(job)
        self._gpu_audit([j.work.request_id for j in active])
        rows, bindings = [], []
        for job in active:
            state = self.states[job.work.request_id]
            context = job.work.dependency_prefix + tuple(job.generated)
            require(state.materialized == len(context) - 1, "prepost lookahead KV frontier")
            rows.append(DraftMaterialization(state.internal_id, context, state.materialized))
            bindings.append(
                dict(
                    request_id=job.work.request_id,
                    round_id=job.work.prefix_version,
                    continuation_id=job.work.work_id,
                    token_step=len(job.generated) + 1,
                )
            )
        logits, sampled, record = self._forward_prepost(
            rows, "prepost_lookahead", bindings, [r.request_id for r in rows]
        )
        for job, row in zip(active, rows):
            state = self.states[job.work.request_id]
            state.materialized, state.next_logits = len(row.context), logits[row.request_id]
            job.started_ns = job.started_ns or record["start_ns"]
            job.generated.append(sampled[row.request_id])
            job.completed = (
                len(job.generated) == job.work.limit or job.generated[-1] in job.work.eos_token_ids
            )
            if job.completed:
                job.completed_ns = record["end_ns"]
                job.status = "ready" if job.status == "waiting_draft" else "draft_complete"
        self._gpu_audit([j.work.request_id for j in active])
        return {w.work_id: self.prepost_jobs[w.work_id] for w in works}

    def abort_prepost(self, work):
        self._gpu_check()
        job = self.prepost_jobs.get(work.work_id)
        require(job is not None and job.work == work, "unknown prepost abort")
        self.worker.fence("prepost_invalidate")
        job.aborted, job.completed, job.status = True, True, "invalidated"
        job.completed_ns = job.completed_ns or time.monotonic_ns()
        return tuple(job.generated)

    def post_prepost(self, entries, *, prepare_next=True):
        """entries=(commit plan, work-or-None, retained tokens). Mix success/recovery.

        A rejected correction is this row's input and its logits give the sole
        recovery candidate. A success inputs a3 and produces a4. Terminal KV
        repair, if necessary, participates without generating a candidate.
        """
        self._gpu_check()
        unique_ids([p.request_id for p, _, _ in entries])
        if not entries:
            return {}
        rows, bindings, seed_ids, choices = [], [], [], {}
        self._gpu_audit([p.request_id for p, _, _ in entries], settling=True)
        for plan, work, retained in entries:
            state = self.states[plan.request_id]
            require(
                state.prefix == plan.parent_prefix
                and state.next_round == plan.round_id
                and state.proposal == plan.proposal
                and plan.bonus_count == 0,
                "stale prepost settlement",
            )
            retained = token_tuple(retained)
            if work is not None:
                self._validate_prepost(work)
                require(work.request_id == plan.request_id, "cross-request prepost settlement")
                job = self.prepost_jobs.get(work.work_id)
                require(job is not None and job.work == work, "missing prepost dependency")
                if retained:
                    require(
                        job.completed
                        and not job.aborted
                        and tuple(job.generated) == retained
                        and plan.accepted == len(plan.proposal)
                        and not plan.target_tail,
                        "invalid lookahead reuse",
                    )
                else:
                    require(job.aborted or not job.generated, "unretired invalid lookahead")
            else:
                require(not retained, "lookahead lacks work identity")
                require(
                    state.materialized == len(plan.parent_prefix) + len(plan.proposal) - 1,
                    "ordinary prepost proposal frontier mismatch",
                )
            stop_lookahead = bool(retained and retained[-1] in work.eos_token_ids)
            context = plan.final_prefix + retained
            if retained:
                require(state.materialized == len(context) - 1, "retained lookahead frontier")
                valid = state.materialized
            else:
                valid = min(
                    state.materialized,
                    len(plan.parent_prefix) + plan.accepted,
                    len(context) - int(prepare_next and not plan.terminal),
                )
            # Runtime checks require the committed prefix materialized even on release.
            needs_row = not stop_lookahead and (
                (not plan.terminal and prepare_next) or valid < len(plan.final_prefix)
            )
            if needs_row:
                require(
                    valid == len(context) - 1, "prepost common forward needs one input position"
                )
                row = DraftMaterialization(state.internal_id, context, valid)
                rows.append(row)
                bindings.append(
                    dict(
                        request_id=plan.request_id,
                        round_id=plan.round_id,
                        continuation_id=work.work_id if work else None,
                        role="terminal_repair"
                        if plan.terminal
                        else "long_tail"
                        if retained
                        else "recovery_or_serial_seed",
                    )
                )
                if not plan.terminal and prepare_next:
                    seed_ids.append(state.internal_id)
            choices[plan.request_id] = (context, retained, needs_row, valid)
        self.worker.fence("prepost_parent_settlement")
        logits, sampled, record = self._forward_prepost(
            rows, "prepost_post" if prepare_next else "prepost_repair", bindings, seed_ids
        )
        result = {}
        for plan, work, _ in entries:
            state = self.states[plan.request_id]
            context, retained, wrote, valid = choices[plan.request_id]
            tokens = (
                ()
                if plan.terminal or not prepare_next
                else retained
                + ((sampled[state.internal_id],) if state.internal_id in sampled else ())
            )
            state.prefix, state.next_round = plan.final_prefix, plan.round_id + 1
            state.materialized = (
                len(context)
                if wrote
                else (
                    len(plan.final_prefix)
                    if plan.terminal or not prepare_next
                    else len(context) - 1
                )
            )
            state.next_logits = logits.get(state.internal_id, state.next_logits)
            state.proposal = tokens or None
            if work is not None:
                self.prepost_retired[work.work_id] = self.prepost_jobs.pop(work.work_id).work
            for name, count in (
                ("commits", 1),
                ("correction_tokens", plan.correction_count),
                ("invalidated_tokens", len(plan.proposal) - plan.accepted),
            ):
                self.metrics.counters[name] += count
            result[plan.request_id] = dict(
                tokens=tokens,
                materialized_kv_length=state.materialized,
                committed_prefix_hash=token_prefix_hash(state.prefix),
                materialized_query_tokens=int(wrote),
                retained_tokens=len(retained),
                common_forward=record,
                **self.worker.request_evidence(state.internal_id),
            )
        self._gpu_audit([p.request_id for p, _, _ in entries])
        self.finish_many([p.request_id for p, _, _ in entries if p.terminal])
        return result

    def extend_prepost(self, plans):
        """Serial control only: three extensions after the common seed, EOS/budget clipped."""
        self._gpu_check()
        unique_ids([p.request_id for p in plans])
        active = list(plans)
        while active:
            continuing = []
            for plan in active:
                state = self.states[plan.request_id]
                require(
                    state.prefix == plan.prefix
                    and state.next_round == plan.round_id
                    and state.proposal,
                    "stale prepost serial extension",
                )
                if (
                    len(state.proposal) < plan.budget
                    and state.proposal[-1] not in plan.eos_token_ids
                ):
                    continuing.append(plan)
            active = continuing
            if not active:
                break
            self._gpu_audit([p.request_id for p in active], admitting=True)
            rows = []
            for plan in active:
                state = self.states[plan.request_id]
                context = state.prefix + state.proposal
                require(state.materialized == len(context) - 1, "serial extension frontier")
                rows.append(DraftMaterialization(state.internal_id, context, state.materialized))
            logits, sampled, _ = self._forward_prepost(
                rows,
                "prepost_extension",
                [dict(request_id=p.request_id, round_id=p.round_id) for p in active],
                [r.request_id for r in rows],
            )
            for plan, row in zip(active, rows):
                state = self.states[plan.request_id]
                state.materialized, state.next_logits = len(row.context), logits[row.request_id]
                state.proposal += (sampled[row.request_id],)
            self._gpu_audit([p.request_id for p in active])
        return {p.request_id: self.states[p.request_id].proposal for p in plans}

    def finish_many(self, ids):
        self._gpu_check()
        require(
            not any(j.work.request_id in ids for j in self.prepost_jobs.values()),
            "prepost release requires explicit work retirement",
        )
        return super().finish_many(ids)

    def commit_many(self, plans):
        require(
            not any(
                p.request_id == j.work.request_id
                for p in plans
                for j in self.prepost_jobs.values()
            ),
            "prepost requires explicit parent settlement",
        )
        return super().commit_many(plans)

    def shutdown(self):
        if not self.closed and self.prepost_jobs:
            self.worker.fence("prepost_shutdown_work")
            for wid, job in self.prepost_jobs.items():
                self.prepost_retired[wid] = job.work
            self.prepost_jobs.clear()
        return super().shutdown()

    def report(self):
        from specrhythm.phase4.draft_metrics import batch_statistics

        value = super().report()
        purposes = ("proposal", "commit", *PURPOSES)
        measured = sum((self.metrics.batches[p] for p in purposes), Counter())
        stats = batch_statistics(measured)
        value.update(
            draft_model_forward_count=sum(self.metrics.forwards[p] for p in purposes),
            draft_batch_count=stats["count"],
            draft_batch_size_histogram=stats["histogram"],
            draft_gpu_event_time_ms=sum(self.metrics.gpu_ms[p] for p in purposes),
            true_batching_observed=bool(stats["max"] and stats["max"] > 1),
            counter_scope="lifecycle proposal/commit/prepost (including decode warmup/drain); "
            "setup prefill/model warmup have separate purposes; window by timestamps",
        )
        for key in ("min", "p10", "p50", "p90", "max", "mean"):
            value["draft_batch_size_" + key] = stats[key]
        value["prepost_physical"] = dict(
            protocol=PROTOCOL,
            parameters=PARAMETERS,
            forwards=self.prepost_forwards.rows(),
            retention={k: v for k, v in self.prepost_forwards.report().items() if k != "rows"},
            pending_work=list(self.prepost_jobs),
        )
        return value
