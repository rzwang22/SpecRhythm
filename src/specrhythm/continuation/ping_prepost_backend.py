"""One Draft GPU: merge independent normal, lookahead and common token rows."""

from collections import Counter

from specrhythm.continuation.prepost_backend import PrePostBackendMixin
from specrhythm.phase4.draft_batch import DraftMaterialization, unique_ids
from specrhythm.serving.common import require
from specrhythm.serving.ping_prepost import PURPOSES


class PingPrePostBackendMixin(PrePostBackendMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics.batches["prepost_mixed"] = Counter()
        self.background = ((), ())

    def _background_rows(self, works, plans, excluded=()):
        rows, bindings, updates = [], [], []
        stable = [w.request_id for w in works] + [p.request_id for p in plans]
        unique_ids(stable)
        require(not set(stable) & set(excluded), "common/background request alias")
        self._gpu_audit(stable)
        for work in works:
            self._validate_prepost(work)
            job = self.prepost_jobs.get(work.work_id)
            require(
                job is not None and job.work == work and not job.aborted and not job.completed,
                "invalid mixed lookahead",
            )
            state = self.states[work.request_id]
            context = work.dependency_prefix + tuple(job.generated)
            require(state.materialized == len(context) - 1, "mixed lookahead frontier")
            rows.append(DraftMaterialization(state.internal_id, context, state.materialized))
            bindings.append(
                dict(
                    request_id=work.request_id,
                    round_id=work.prefix_version,
                    continuation_id=work.work_id,
                    work_kind="lookahead",
                    token_step=len(job.generated) + 1,
                )
            )
            updates.append((state, job, work))
        for plan in plans:
            state = self.states[plan.request_id]
            require(
                state.prefix == plan.prefix
                and state.next_round == plan.round_id
                and state.proposal
                and len(state.proposal) < plan.budget
                and state.proposal[-1] not in plan.eos_token_ids,
                "invalid mixed normal extension",
            )
            context = state.prefix + state.proposal
            require(state.materialized == len(context) - 1, "mixed normal frontier")
            rows.append(DraftMaterialization(state.internal_id, context, state.materialized))
            bindings.append(
                dict(
                    request_id=plan.request_id,
                    round_id=plan.round_id,
                    work_kind="normal_extension",
                    token_step=len(state.proposal),
                )
            )
            updates.append((state, None, plan))
        return rows, bindings, updates

    def _apply_background(self, rows, updates, logits, sampled, record):
        for row, (state, job, intent) in zip(rows, updates):
            state.materialized, state.next_logits = len(row.context), logits[row.request_id]
            token = sampled[row.request_id]
            if job is None:
                state.proposal += (token,)
            else:
                job.started_ns = job.started_ns or record["start_ns"]
                job.generated.append(token)
                job.completed = len(job.generated) == intent.limit or token in intent.eos_token_ids
                if job.completed:
                    job.completed_ns = record["end_ns"]
                    job.status = "ready" if job.status == "waiting_draft" else "draft_complete"
        self._gpu_audit([i.request_id for _, _, i in updates])

    def token_step(self, works, plans):
        rows, bindings, updates = self._background_rows(works, plans)
        if not rows:
            return None
        purpose = (
            "prepost_mixed"
            if works and plans
            else "prepost_lookahead"
            if works
            else "prepost_extension"
        )
        logits, sampled, record = super()._forward_prepost(
            rows, purpose, bindings, [r.request_id for r in rows]
        )
        self._apply_background(rows, updates, logits, sampled, record)
        return record

    def _forward_prepost(self, rows, purpose, bindings, sample_ids):
        # Called by inherited post_prepost after parent validation and the rollback fence.
        works, plans = self.background
        self.background = ((), ())
        if purpose != "prepost_post" or not (works or plans):
            return super()._forward_prepost(
                rows,
                purpose,
                [
                    {**r, "work_kind": "post" if purpose == "prepost_post" else "repair"}
                    for r in bindings
                ],
                sample_ids,
            )
        extra, roles, updates = self._background_rows(
            works, plans, excluded=[r["request_id"] for r in bindings]
        )
        result = super()._forward_prepost(
            [*rows, *extra],
            "prepost_mixed",
            [*({**r, "work_kind": "post"} for r in bindings), *roles],
            [*sample_ids, *(r.request_id for r in extra)],
        )
        logits, sampled, record = result
        self._apply_background(extra, updates, logits, sampled, record)
        return result

    def report(self):
        from specrhythm.phase4.draft_metrics import batch_statistics

        value = super().report()
        histogram = sum(
            (self.metrics.batches[p] for p in (*PURPOSES, "proposal", "commit")), Counter()
        )
        stats = batch_statistics(histogram)
        value.update(
            draft_model_forward_count=sum(
                self.metrics.forwards[p] for p in (*PURPOSES, "proposal", "commit")
            ),
            draft_batch_count=stats["count"],
            draft_batch_size_histogram=stats["histogram"],
            draft_gpu_event_time_ms=sum(
                self.metrics.gpu_ms[p] for p in (*PURPOSES, "proposal", "commit")
            ),
            true_batching_observed=bool(stats["max"] and stats["max"] > 1),
        )
        for k in ("min", "p10", "p50", "p90", "max", "mean"):
            value["draft_batch_size_" + k] = stats[k]
        value["prepost_physical"]["mixed_forward_semantics"] = (
            "one physical batch; work_kind binds each row; "
            "mixed interval belongs to the union once"
        )
        return value
