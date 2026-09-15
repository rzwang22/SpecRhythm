"""Single authoritative GPU owner; independent parent settlement and atomic ready claims."""

import time
from dataclasses import asdict

from specrhythm.continuation.policy import EagerDecision
from specrhythm.continuation.prepost_records import Records
from specrhythm.continuation.scheduling import SchedulingView, select_target_admissions
from specrhythm.phase4.draft_batch import DraftProposalPlan, unique_ids
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.serving.common import require
from specrhythm.serving.ping_prepost import OWNER_ID, PROTOCOL
from specrhythm.serving.prepost_machine import PrePostMachine


class PingPrePostMachine(PrePostMachine):
    target_batch_ceiling = 8
    home_capacities = {"A": 8, "B": 8}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.homes, self.ready, self.claims, self.normal = {}, {}, {}, {}
        self.ping_events = Records()
        self.opportunity = -1
        self.draining = False
        self.eligibility_universe = self.eligible

    def _ping(self, event, **values):
        self.ping_events.append(dict(event=event, timestamp_ns=time.monotonic_ns(), **values))

    def _publish_proposal(self, rid, tokens, started, ended, *, source=None):
        require(rid not in self.ready and rid not in self.claims, "duplicate ready publication")
        proposal = super()._publish_proposal(rid, tokens, started, ended, source=source)
        self.ready[rid] = dict(
            proposal=proposal.to_dict(), ready_ns=ended, promoted=source is not None
        )
        self._ping(
            "ready",
            request_id=rid,
            prefix_version=proposal.round_id,
            proposal_id=proposal.runtime_provenance["rolling_proposal_id"],
            continuation_id=source,
            candidate_length=len(tokens),
            candidate_EOS=proposal.proposal_eos,
            ready_ns=ended,
            home_cohort=self.homes.get(rid),
            decision_version=self.decision_version,
        )
        return proposal

    def register(self, rows):
        unique_ids([r["request_id"] for r in rows])
        require(not self.draining, "registration after stop")
        for r in rows:
            rid, home = r["request_id"], r["home_cohort"]
            require(home in self.home_capacities and rid not in self.homes, "duplicate/moved home")
        for r in rows:
            state = self._state(r["request_id"])
            require(
                not state.finished
                and state.pending_proposal is None
                and r["round_id"] == state.next_round_id
                and r["committed_prefix_len"] == len(state.committed_token_ids)
                and r["committed_prefix_hash"] == token_prefix_hash(state.committed_token_ids)
                and type(r["remaining_output_budget"]) is int
                and r["remaining_output_budget"] > 0,
                "stale PingPong initial registration",
            )
        for r in rows:
            self.homes[r["request_id"]] = r["home_cohort"]
        result = self.batch_propose(rows)
        return dict(
            registered=[r["request_id"] for r in rows],
            actual_batch_model_forward_count=result["actual_batch_model_forward_count"],
        )

    def views(self, active):
        result = []
        for rid in self.homes:  # Immutable admission order, not a quota-limited global FIFO.
            if rid not in active:
                continue
            state = self.core._state(rid)
            ready = self.ready.get(rid)
            result.append(
                SchedulingView(
                    rid,
                    OWNER_ID,
                    self.homes[rid],
                    state.proposal_id or None,
                    state.prefix_version,
                    EagerDecision(rid in self.eligible, self.enabled, self.decision_version),
                    is_ready=ready is not None,
                    is_legal=ready is not None,
                    promoted=bool(ready and ready["promoted"]),
                    verifying=rid in self.claims,
                    active=not state.terminal,
                    normal_draft_pending=rid in self.normal,
                    continuation_pending=rid in self.works,
                )
            )
        return result

    def admit(self, payload):
        require(
            not self.draining and payload["opportunity"] > self.opportunity,
            "stale/duplicate admission opportunity",
        )
        capacity, normal = payload["capacity"], payload["normal_cohort"]
        require(
            type(capacity) is int
            and 0 < capacity <= self.target_batch_ceiling
            and normal in self.home_capacities,
            "Target claim exceeds configured home/ceiling",
        )
        active = payload["active_request_ids"]
        unique_ids(active)
        require(len(active) <= 16 and set(active) <= set(self.homes), "invalid active inventory")
        views = self.views(active)
        selected = select_target_admissions(views, normal_cohort=normal, capacity=capacity)
        fallback = None
        if (
            not selected and len(self.home_capacities) == 2
        ):  # A busy home must not hide legal ordinary work in the other home.
            fallback = "B" if normal == "A" else "A"
            selected = select_target_admissions(views, normal_cohort=fallback, capacity=capacity)
        self.opportunity = payload["opportunity"]
        batch_id = f"ping-prepost-target-{self.opportunity}-{normal}"
        rows = []
        for intent in selected:
            rid = intent.request_id
            require(rid not in self.claims, "proposal already claimed")
            ready = self.ready.pop(rid)
            row = dict(
                **asdict(intent),
                **ready,
                target_batch_id=batch_id,
                admission_opportunity=self.opportunity,
                opportunity_cohort=normal,
                claim_id=batch_id + ":" + intent.proposal_id,
                claimed_ns=time.monotonic_ns(),
                consumed=False,
            )
            self.claims[rid] = row
            rows.append(dict(row))
        selected_ids = {r["request_id"] for r in rows}
        self._ping(
            "admission",
            target_batch_id=batch_id,
            admission_opportunity=self.opportunity,
            opportunity_cohort=normal,
            fallback_normal_cohort=fallback,
            capacity=capacity,
            actual_B=len(rows),
            claims=rows,
            ready_inventory=[v.request_id for v in views if v.is_ready],
            deferred=[
                dict(
                    request_id=v.request_id,
                    home_cohort=v.home_cohort,
                    reason=(
                        "dependency"
                        if not v.is_ready
                        else "capacity"
                        if (v.promoted and v.decision.can_admit)
                        or v.home_cohort == (fallback or normal)
                        else "home_policy"
                    ),
                )
                for v in views
                if v.request_id not in selected_ids
            ],
            unfilled=capacity - len(rows),
            unfilled_reason="no_more_promoted_or_selected_home_ready_work"
            if len(rows) < capacity
            else None,
        )
        return dict(claims=rows, target_batch_id=batch_id, opportunity_cohort=normal)

    def verify_start(self, rows):
        for row in rows:
            claim = self.claims.get(row["request_id"])
            require(
                claim is not None
                and not claim["consumed"]
                and claim["proposal_id"] == row["proposal_id"]
                and claim["prefix_version"] == row["round_id"],
                "unclaimed/stale/consumed Target proposal",
            )
        super().verify_start(rows)
        for row in rows:
            claim = self.claims[row["request_id"]]
            claim["consumed"] = True
            self._ping(
                "consume",
                request_id=row["request_id"],
                claim_id=claim["claim_id"],
                target_batch_id=claim["target_batch_id"],
                prefix_version=row["round_id"],
                proposal_id=row["proposal_id"],
                decision_version=self.decision_version,
            )

    def feedback(self, rows):
        for row in rows:
            key = row["request_id"], row["round_id"]
            if key not in self.sync_bindings:
                claim = self.claims.get(row["request_id"])
                require(claim is not None and claim["consumed"], "feedback without consumed claim")
        self.prepare_synchronizations(rows)
        # Acknowledge validated feedback only. No physical frontier/ready claim is fabricated.
        now = time.monotonic_ns()
        self._ping("feedback", rows=rows, owner_feedback_ns=now)
        return dict(
            synchronizations=[
                dict(
                    request_id=r["request_id"],
                    decision={"committed_token_ids": r["committed_delta"]},
                    physical_settlement="PENDING",
                )
                for r in rows
            ],
            proposals=[],
            state_sync_start_ns=now,
            state_sync_end_ns=time.monotonic_ns(),
            acknowledgement="feedback_validated; not KV completion",
        )

    def _pending_rows(self):
        return [
            self.sync_bindings[key] for key in self.parent_plans if key not in self.parent_results
        ]

    def has_work(self):
        return bool(
            self.normal
            or self._pending_rows()
            or any(
                not self.core._state(r).continuations[w.work_id].completed
                for r, w in self.works.items()
            )
        )

    def _background(self, excluded=()):
        works = [
            w
            for rid, w in self.works.items()
            if rid not in excluded and not self.core._state(rid).continuations[w.work_id].completed
        ]
        plans = [p for rid, p in self.normal.items() if rid not in excluded]
        return works, plans

    def _account_background(self, works, plans, before):
        for w in works:
            job = self.core._state(w.request_id).continuations[w.work_id]
            n = len(job.generated) - before[w.work_id]
            if n:
                if self.work_times[w.work_id][0] is None:
                    self.work_times[w.work_id][0] = job.started_ns
                    self._event("start", request_ids=[w.request_id], started=1)
                self._event("token", request_ids=[w.request_id], early_generated_tokens=n)
                if job.completed:
                    self.work_times[w.work_id][1] = job.completed_ns
                    self._event("complete", request_ids=[w.request_id], completed=1)
        for p in plans:
            state = self.backend.states[p.request_id]
            if len(state.proposal) == p.budget or state.proposal[-1] in p.eos_token_ids:
                self.normal.pop(p.request_id)
                self._publish_proposal(
                    p.request_id,
                    state.proposal,
                    self.feedback_times.get((p.request_id, p.round_id - 1), time.monotonic_ns()),
                    time.monotonic_ns(),
                )

    def step(self):
        ready = []
        for row in self._pending_rows():
            rid = row["request_id"]
            work = self.works.get(rid)
            if work is None or self.core._state(rid).continuations[work.work_id].completed:
                ready.append(row)
        works, plans = self._background([r["request_id"] for r in ready])
        before = {
            w.work_id: len(self.core._state(w.request_id).continuations[w.work_id].generated)
            for w in works
        }
        if ready:
            self.backend.background = (works, plans)
            try:
                self.finish_synchronizations(ready)
            finally:
                self.backend.background = ((), ())
        elif works or plans:
            self.backend.token_step(works, plans)
        else:
            return False
        self._account_background(works, plans, before)
        return True

    def finish_synchronizations(self, rows, *, draining=False):
        entries = []
        for row in rows:
            rid, key = row["request_id"], (row["request_id"], row["round_id"])
            if key in self.parent_results:
                continue
            plan, decision = self.parent_plans[key]
            work = self.works.get(rid)
            job = self.core._state(rid).continuations[work.work_id] if work else None
            if job and not job.completed:
                return None
            entries.append(
                (plan, decision, work, tuple(job.generated) if job and not job.aborted else ())
            )
        started = time.monotonic_ns()
        physical = self.backend.post_prepost(
            [(p, w, t) for p, _, w, t in entries], prepare_next=not (draining or self.draining)
        )
        for plan, decision, work, retained in entries:
            rid, key = plan.request_id, (plan.request_id, plan.round_id)
            state = self.core._state(rid)
            self.parent_results[key] = self._publish_commit(plan, decision, physical[rid])
            state.committed_prefix, state.prefix_version = plan.final_prefix, plan.round_id + 1
            state.remaining -= len(decision.committed_token_ids)
            state.terminal, state.proposal_id, state.proposal_tokens = plan.terminal, "", ()
            self.works.pop(rid, None)
            claim = self.claims.pop(rid, None)
            common = (
                physical[rid]["common_forward"]
                if physical[rid]["materialized_query_tokens"]
                else None
            )
            self._ping(
                "settlement",
                request_id=rid,
                prefix_version=plan.round_id,
                target_batch_id=claim["target_batch_id"] if claim else None,
                parent_length=len(plan.proposal),
                accepted=plan.accepted,
                correction=plan.correction_count,
                committed_tokens=list(decision.committed_token_ids),
                feedback_ns=self.feedback_times[key],
                settlement_start_ns=started,
                settlement_end_ns=time.monotonic_ns(),
                retained=len(retained),
                generated=len(state.continuations[work.work_id].generated) if work else 0,
                continuation_id=work.work_id if work else None,
                terminal=plan.terminal,
                common_start_ns=(common or {}).get("start_ns"),
                common_end_ns=(common or {}).get("end_ns"),
                lookahead_completed_ns=(
                    state.continuations[work.work_id].completed_ns if work else None
                ),
                remaining_output_budget=state.remaining,
                next_candidate_length=len(self.backend.states[rid].proposal or ())
                if rid in self.backend.states
                else 0,
            )
            self._event(
                "settled",
                request_ids=[rid],
                promotions=int(bool(retained)),
                reused_early_tokens=len(retained),
                committed_tokens=len(decision.committed_token_ids),
                accepted_tokens=plan.accepted,
                correction_tokens=plan.correction_count,
                recovery_requests=int(not retained and not plan.terminal),
            )
            if not plan.terminal and not (draining or self.draining):
                tokens = self.backend.states[rid].proposal
                normal = (self.uniform_candidate_length or
                          not (self.enabled and rid in self.eligible))
                budget = min(self.uniform_candidate_length or 4, state.remaining)
                if normal and len(tokens) < budget and tokens[-1] not in state.eos_token_ids:
                    self.normal[rid] = DraftProposalPlan(
                        rid,
                        state.prefix_version,
                        state.committed_prefix,
                        budget,
                        state.eos_token_ids,
                    )
                else:
                    self._publish_proposal(
                        rid,
                        tokens,
                        started,
                        time.monotonic_ns(),
                        source=work.work_id if retained else None,
                    )
        return [self.parent_results[r["request_id"], r["round_id"]] for r in rows]

    def finish_authoritative(self, row):
        rid = row["request_id"]
        if any(r["request_id"] == rid for r in self._pending_rows()):
            # Terminal feedback is settleable; batch with other ready parents.
            self.step()
        result = super().finish_authoritative(row)
        if result.get("finished"):
            self.ready.pop(rid, None)
            self.normal.pop(rid, None)
            self.claims.pop(rid, None)
        return result

    def begin_drain(self):
        self.draining = True
        for rid in list(self.works):
            self._abort(rid, "drain")
        for rid in list(self.normal):
            self._describe_cancelled_normal(rid)
        return dict(stopping=True)

    def _describe_cancelled_normal(self, rid):
        if rid in self.normal:
            # Last token step already fenced. Private drain descriptor only: never
            # enters ready/claims, and cannot be admitted once stop is received.
            now = time.monotonic_ns()
            PrePostMachine._publish_proposal(
                self, rid, self.backend.states[rid].proposal, now, now
            )
            self.normal.pop(rid)
            self._ping(
                "cancel_normal",
                request_id=rid,
                discarded_candidates=len(self.backend.states[rid].proposal),
            )

    def diagnostic_settle(self, row):
        rid = row["request_id"]
        if rid in self.normal:
            require(
                tuple(row["committed_prefix"]) == self.requests[rid].committed_token_ids
                and row["committed_prefix_hash"] == token_prefix_hash(row["committed_prefix"]),
                "cancel normal prefix differs",
            )
            self._describe_cancelled_normal(rid)
        result = super().diagnostic_settle(row)
        self.ready.pop(rid, None)
        self.normal.pop(rid, None)
        self.claims.pop(rid, None)
        return result

    def update_eligibility(self, enabled, version, eligible):
        require(
            version > self.decision_version
            and set(eligible) <= self.eligibility_universe | set(self.homes),
            "invalid/stale static eligibility update",
        )
        removed = self.eligible - set(eligible)
        self.set_eager(enabled, version)
        self.eligible = frozenset(eligible)
        for rid in removed & self.works.keys():
            self._abort(rid, "eligibility")
        return dict(enabled=self.enabled, decision_version=version, eligible=sorted(self.eligible))

    def eager_report(self):
        result = super().eager_report()
        result["pingpong"] = dict(
            protocol=self.protocol if self.uniform_candidate_length else PROTOCOL,
            events=self.ping_events.rows(),
            retention={k: v for k, v in self.ping_events.report().items() if k != "rows"},
            pending_normal=list(self.normal),
            claims=list(self.claims),
            ready=list(self.ready),
            home_cohorts=dict(self.homes),
            active_limit=16,
            target_batch_ceiling=self.target_batch_ceiling,
        )
        return result
