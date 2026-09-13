"""Owner-confined, no-bonus rolling pre/post authority (old protocols unchanged)."""

from __future__ import annotations

import time
from collections import Counter

from specrhythm.continuation.prepost import (
    PARAMETERS,
    PROTOCOL,
    PrePostLedger,
    PrePostState,
    PrePostWork,
    acceptance,
)
from specrhythm.continuation.prepost_records import Records
from specrhythm.continuation.trace import TRACE
from specrhythm.phase4.draft_batch import DraftProposalPlan, unique_ids
from specrhythm.phase4.serial import PROTOCOL_VERSION, Proposal, token_prefix_hash
from specrhythm.serving.common import require
from specrhythm.serving.fixed_settle import DiagnosticSerialMachine, remaining


class PrePostMachine(DiagnosticSerialMachine):
    def __init__(self, backend, *, request_ids, eager=True, report_path=None):
        super().__init__(backend, candidate_budget=4, report_path=report_path)
        self.core = PrePostLedger()
        self.eligible = frozenset(request_ids)
        self.enabled, self.decision_version = eager, 0
        self.serial_control = not eager
        self.works, self.work_times = {}, {}
        self.verify_bindings, self.sync_bindings = {}, {}
        self.parent_plans, self.parent_results = {}, {}
        self.feedback_times = {}
        self.cycles = Records()
        self.counters, self.events = Counter(), Records()
        self.owner_stopped = False

    def _event(self, phase, *, request_ids=(), start_ns=None, end_ns=None, **counts):
        now = time.monotonic_ns()
        event = dict(
            phase=phase,
            request_ids=list(request_ids),
            start_ns=now if start_ns is None else start_ns,
            end_ns=now if end_ns is None else end_ns,
            counter_delta=counts,
        )
        self.events.append(event)
        self.counters.update(counts)
        TRACE.event("prepost_" + phase, **event)

    def eager_report(self):
        return dict(
            protocol=PROTOCOL,
            parameters=PARAMETERS,
            counters=dict(self.counters),
            cycles=self.cycles.rows(),
            cycle_retention={k: v for k, v in self.cycles.report().items() if k != "rows"},
            events=self.events.rows(),
            retention={k: v for k, v in self.events.report().items() if k != "rows"},
            pending_work=list(self.works),
            owner_stopped=self.owner_stopped,
            gpu_qualification="PENDING",
        )

    def _publish_proposal(self, rid, tokens, started, ended, *, source=None):
        state, physical = self.core._state(rid), self.backend.states[rid]
        require(
            tuple(tokens) == physical.proposal and 1 <= len(tokens) <= 4,
            "prepost ready proposal differs from physical candidates",
        )
        state.proposal_id = f"{rid}:prepost:{state.prefix_version}"
        state.proposal_tokens = tuple(tokens)
        proposal = Proposal(
            PROTOCOL_VERSION,
            rid,
            state.prefix_version,
            len(state.committed_prefix),
            state.prefix_hash,
            tuple(tokens),
            tokens[-1] in state.eos_token_ids,
            started,
            ended,
            0,
            self.backend.provenance,
            dict(
                backend=self.backend.backend_name,
                prepost_protocol=PROTOCOL,
                rolling_proposal_id=state.proposal_id,
                rolling_prefix_version=state.prefix_version,
                source_continuation_id=source,
                materialized_kv_length=physical.materialized,
                full_context_replay=False,
                persistent_cross_round_kv=True,
                actual_candidate_length=len(tokens),
            ),
        )
        if source is not None or state.prefix_version:
            if hasattr(self.backend, "fixed_proposals"):
                self.backend.fixed_proposals.append(
                    dict(
                        start_ns=started,
                        end_ns=ended,
                        B=1,
                        request_ids=[rid],
                        context_lengths=[len(state.committed_prefix)],
                        budgets=[len(tokens)],
                        candidate_lengths=[len(tokens)],
                        round_ids=[state.prefix_version],
                        first_token_from_cached_logits=False,
                        batch_participation_only=True,
                        prepost_protocol=PROTOCOL,
                    )
                )
            self.backend.metrics.counters["proposals"] += 1
            self.backend.metrics.counters["proposed_tokens"] += len(tokens)
        legacy = self._state(rid)
        legacy.pending_proposal = proposal
        legacy.proposal_count += 1
        return proposal

    @TRACE.observe("machine_batch_propose")
    def batch_propose(self, rows):
        unique_ids([r["request_id"] for r in rows])
        started = time.monotonic_ns()
        plans = []
        for row in rows:
            rid = row["request_id"]
            legacy = self._state(rid)
            require(
                not legacy.finished
                and row["round_id"] == legacy.next_round_id
                and row["committed_prefix_len"] == len(legacy.committed_token_ids)
                and row["committed_prefix_hash"] == token_prefix_hash(legacy.committed_token_ids)
                and type(row["remaining_output_budget"]) is int
                and row["remaining_output_budget"] > 0,
                "stale prepost proposal entrance",
            )
            if rid not in self.core.states:
                self.core.states[rid] = PrePostState(
                    legacy.committed_token_ids,
                    legacy.next_round_id,
                    row["remaining_output_budget"],
                    tuple(row.get("eos_token_ids", ())),
                )
            state = self.core._state(rid)
            state.check(legacy.committed_token_ids, row["round_id"])
            require(state.remaining == row["remaining_output_budget"], "prepost budget mismatch")
            if legacy.pending_proposal is None:
                require(not state.proposal_id, "lost prepost ready proposal")
                plans.append(
                    DraftProposalPlan(
                        rid, state.prefix_version, state.committed_prefix, 1, state.eos_token_ids
                    )
                )
        # Setup/refill already materialized the legal prefix. Sampling its cached
        # logits gives a real first candidate with zero new GPU forwards.
        tokens = self.backend.propose_many(plans)
        ended = time.monotonic_ns()
        for plan in plans:
            self._publish_proposal(plan.request_id, tokens[plan.request_id], started, ended)
        if plans:
            self._event(
                "initial_seed",
                request_ids=[p.request_id for p in plans],
                initial_candidates=len(plans),
            )
        return dict(
            proposals=[self._state(r["request_id"]).pending_proposal.to_dict() for r in rows],
            draft_batch_start_ns=started,
            draft_batch_end_ns=time.monotonic_ns(),
            draft_microbatch_count=int(bool(plans)),
            actual_batch_model_forward_count=0,
            single_persistent_model=True,
        )

    @TRACE.observe("machine_verify_start")
    def verify_start(self, rows):
        unique_ids([r["request_id"] for r in rows])
        works, new = [], []
        for row in rows:
            rid, version = row["request_id"], row["round_id"]
            key = rid, version
            if key in self.verify_bindings:
                require(self.verify_bindings[key] == row, "conflicting duplicate prepost verify")
                continue
            state = self.core._state(rid)
            state.check(self._state(rid).committed_token_ids, version)
            require(
                state.proposal_id == row["proposal_id"]
                and state.prefix_hash == row["parent_prefix_hash"]
                and len(state.committed_prefix) == row["parent_prefix_len"]
                and state.proposal_tokens == tuple(row["proposal_tokens"]),
                "prepost verify dependency mismatch",
            )
            new.append(row)
            limit = min(3, max(state.remaining - len(state.proposal_tokens) - 1, 0))
            if self.enabled and rid in self.eligible and limit:
                works.append(
                    PrePostWork(
                        f"{state.proposal_id}:lookahead:{self.decision_version}",
                        rid,
                        version,
                        state.proposal_id,
                        state.committed_prefix,
                        state.proposal_tokens,
                        limit,
                        state.eos_token_ids,
                    )
                )
        jobs = self.backend.enroll_prepost(works)
        for row in new:
            self.verify_bindings[row["request_id"], row["round_id"]] = dict(row)
        for work in works:
            self.works[work.request_id] = work
            self.core._state(work.request_id).continuations[work.work_id] = jobs[work.work_id]
            self.work_times[work.work_id] = [None, None]
        self._event("admission", request_ids=[w.request_id for w in works], admissions=len(works))
        return {"accepted": True}

    @TRACE.observe("machine_step")
    def step(self):
        works = [
            w
            for rid, w in self.works.items()
            if not self.core._state(rid).continuations[w.work_id].completed
        ]
        if not works:
            return False
        jobs = self.backend.step_prepost(works)
        for work in works:
            job = jobs[work.work_id]
            if self.work_times[work.work_id][0] is None:
                self.work_times[work.work_id][0] = job.started_ns
                self._event("start", request_ids=[work.request_id], started=1)
            self._event("token", request_ids=[work.request_id], early_generated_tokens=1)
            if job.completed:
                self.work_times[work.work_id][1] = job.completed_ns
                self._event("complete", request_ids=[work.request_id], completed=1)
        return True

    def _abort(self, rid, reason):
        work = self.works.get(rid)
        if work is None:
            return
        job = self.core._state(rid).continuations[work.work_id]
        if job.aborted:
            return
        tokens = self.backend.abort_prepost(work)
        self.work_times[work.work_id][1] = job.completed_ns
        self._event("abort_" + reason, request_ids=[rid], discarded_early_tokens=len(tokens))

    @TRACE.observe("machine_prepare_synchronizations")
    def prepare_synchronizations(self, rows, *, draining=False):
        unique_ids([r["request_id"] for r in rows])
        pending = []
        for row in rows:
            rid, key = row["request_id"], (row["request_id"], row["round_id"])
            if key in self.sync_bindings:
                require(self.sync_bindings[key] == row, "conflicting duplicate Target feedback")
                continue
            legacy = self._pending(rid, row["round_id"])
            state = self.core._state(rid)
            state.check(legacy.committed_token_ids, row["round_id"])
            require(key in self.verify_bindings or draining, "feedback precedes verification")
            decision = acceptance(
                state.proposal_tokens, row["committed_delta"], terminal=row["terminal"]
            )
            require(
                len(decision.committed_token_ids) <= state.remaining,
                "prepost output budget exceeded",
            )
            plan = self._commit_plan(rid, row["round_id"], decision, row["committed_prefix_hash"])
            pending.append((row, key, plan, decision))
        for row, key, plan, decision in pending:
            rid = plan.request_id
            self.sync_bindings[key], self.parent_plans[key] = dict(row), (plan, decision)
            self.feedback_times[key] = time.monotonic_ns()
            rejected = bool(decision.rejected_draft_token_ids)
            if rejected or plan.terminal or draining:
                self._abort(rid, "rejection" if rejected else "terminal_or_drain")
            elif rid in self.works:
                job = self.core._state(rid).continuations[self.works[rid].work_id]
                if not job.completed:
                    job.status = "waiting_draft"
            self._event(
                "parent_result",
                request_ids=[rid],
                parent_rejections=int(rejected),
                parent_full_accepts=int(not rejected),
            )

    @TRACE.observe("machine_finish_synchronizations")
    def finish_synchronizations(self, rows, *, draining=False):
        pending = []
        for row in rows:
            rid, key = row["request_id"], (row["request_id"], row["round_id"])
            if key in self.parent_results:
                continue
            plan, decision = self.parent_plans[key]
            work = self.works.get(rid)
            job = self.core._state(rid).continuations[work.work_id] if work else None
            if job and not job.completed:
                return None
            retained = tuple(job.generated) if job and not job.aborted else ()
            pending.append((plan, decision, work, retained))
        started = time.monotonic_ns()
        physical = self.backend.post_prepost(
            [(p, w, t) for p, _, w, t in pending], prepare_next=not draining
        )
        extensions = []
        for plan, decision, _work, retained in pending:
            rid, state = plan.request_id, self.core._state(plan.request_id)
            result = self._publish_commit(plan, decision, physical[rid])
            state.committed_prefix, state.prefix_version = plan.final_prefix, plan.round_id + 1
            state.remaining -= len(decision.committed_token_ids)
            state.terminal, state.proposal_id, state.proposal_tokens = plan.terminal, "", ()
            self.parent_results[rid, plan.round_id] = result
            self.works.pop(rid, None)
            if (
                not plan.terminal
                and not draining
                and (self.serial_control or rid not in self.eligible)
            ):
                extensions.append(
                    DraftProposalPlan(
                        rid,
                        state.prefix_version,
                        state.committed_prefix,
                        min(4, state.remaining),
                        state.eos_token_ids,
                    )
                )
            self._event(
                "settled",
                request_ids=[rid],
                committed_tokens=len(decision.committed_token_ids),
                accepted_tokens=plan.accepted,
                correction_tokens=plan.correction_count,
                promotions=int(bool(retained)),
                reused_early_tokens=len(retained),
                recovery_requests=int(not retained and not plan.terminal),
            )
        self.backend.extend_prepost(extensions)
        ended = time.monotonic_ns()
        cycle_rows = []
        for plan, _, work, retained in pending:
            state = self.core._state(plan.request_id)
            next_tokens = (
                self.backend.states[plan.request_id].proposal
                if not plan.terminal and not draining
                else ()
            )
            cycle_rows.append(
                dict(
                    request_id=plan.request_id,
                    round_id=plan.round_id,
                    parent_length=len(plan.proposal),
                    accepted=plan.accepted,
                    correction=plan.correction_count,
                    bonus=plan.bonus_count,
                    committed_tokens=len(plan.final_prefix) - len(plan.parent_prefix),
                    lookahead_generated=(
                        len(state.continuations[work.work_id].generated) if work else 0
                    ),
                    lookahead_completed_ns=(
                        state.continuations[work.work_id].completed_ns if work else None
                    ),
                    lookahead_retained=len(retained),
                    next_candidate_length=len(next_tokens),
                    next_candidate_EOS=bool(
                        next_tokens and next_tokens[-1] in state.eos_token_ids
                    ),
                    remaining_output_budget=state.remaining,
                    terminal=plan.terminal,
                    feedback_ns=self.feedback_times[plan.request_id, plan.round_id],
                    common_forward=physical[plan.request_id]["common_forward"] is not None,
                    common_start_ns=(physical[plan.request_id]["common_forward"] or {}).get(
                        "start_ns"
                    ),
                    common_end_ns=(physical[plan.request_id]["common_forward"] or {}).get(
                        "end_ns"
                    ),
                    candidate_ready_ns=ended,
                )
            )
            if not plan.terminal and not draining:
                self._publish_proposal(
                    plan.request_id,
                    self.backend.states[plan.request_id].proposal,
                    started,
                    ended,
                    source=work.work_id if retained else None,
                )
        if cycle_rows:
            self.cycles.append(dict(start_ns=started, end_ns=ended, requests=cycle_rows))
        return [self.parent_results[r["request_id"], r["round_id"]] for r in rows]

    def _retire_work(self, rid):
        self._abort(rid, "release")
        work = self.works.pop(rid, None)
        if work is not None:
            self.backend.prepost_retired[work.work_id] = self.backend.prepost_jobs.pop(
                work.work_id
            ).work

    def finish_authoritative(self, row):
        rid = row["request_id"]
        if "committed_prefix" not in row:
            return dict(request_id=rid, awaiting_authoritative_terminal=True)
        legacy = self._state(rid)
        final = tuple(row["committed_prefix"])
        require(
            token_prefix_hash(final) == row["committed_prefix_hash"]
            and final[: len(legacy.committed_token_ids)] == legacy.committed_token_ids,
            "terminal prefix mismatch",
        )
        if rid in self.backend.retired:
            require(legacy.committed_token_ids == final, "late terminal differs")
            return dict(request_id=rid, finished=True)
        delta = final[len(legacy.committed_token_ids) :]
        if delta:
            require(legacy.pending_proposal is not None, "unproposed prepost terminal delta")
            sync = dict(
                request_id=rid,
                round_id=legacy.next_round_id,
                committed_delta=list(delta),
                committed_prefix_hash=token_prefix_hash(final),
                terminal=True,
            )
            self.prepare_synchronizations([sync], draining=True)
            self.finish_synchronizations([sync])
        else:
            self._retire_work(rid)
            self.backend.finish_many([rid])
            legacy.finished, legacy.pending_proposal, legacy.pending_decision = True, None, None
            if rid in self.core.states:
                self.core._state(rid).terminal = True
        return dict(request_id=rid, finished=True)

    def diagnostic_settle(self, row):
        remaining(row["deadline_ns"])
        rid = row["request_id"]
        legacy = self.requests.get(rid)
        if legacy is not None:
            final = tuple(row["committed_prefix"])
            require(
                final[: len(legacy.committed_token_ids)] == legacy.committed_token_ids,
                "drain prefix regressed",
            )
            delta = final[len(legacy.committed_token_ids) :]
            if delta:
                # Controlled stop: commit authority and release, generating no next candidates.
                sync = dict(
                    request_id=rid,
                    round_id=legacy.next_round_id,
                    committed_delta=list(delta),
                    committed_prefix_hash=token_prefix_hash(final),
                    terminal=row["natural_terminal"],
                )
                self.prepare_synchronizations([sync], draining=True)
                self.finish_synchronizations([sync], draining=True)
            self._retire_work(rid)
        result = super().diagnostic_settle(row)
        if rid in self.core.states:
            self.core._state(rid).terminal = True
        return result

    def set_eager(self, enabled, version):
        require(
            type(enabled) is bool and type(version) is int and version >= self.decision_version,
            "stale prepost switch",
        )
        require(
            version != self.decision_version or enabled == self.enabled,
            "conflicting prepost switch version",
        )
        self.enabled, self.decision_version = enabled, version
        if not enabled:
            for rid in self.works:
                self._abort(rid, "disabled")
        return dict(enabled=enabled, decision_version=version)

    def shutdown(self, deadline_ns=None):
        if deadline_ns is not None:
            remaining(deadline_ns)
        require(
            not self.works
            and not self.backend.prepost_jobs
            and not any(s.pending_proposal for s in self.requests.values()),
            "prepost shutdown requires fenced settlement",
        )
        self.backend.shutdown()
        return dict(
            shutdown=True, pending_work=[], physical_live_requests=len(self.backend.states)
        )
