"""Serial wire compatibility around the single shared continuation authority.

All methods execute on the existing Draft model's owner. Legacy DraftRequest
fields are a projection for existing transport/settlement consumers, never an
independent eager decision or dependency state machine.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict, replace

from specrhythm.continuation.core import (
    ContinuationStatus,
    DraftCompletion,
    ParentVerification,
    RollingContinuation,
)
from specrhythm.continuation.policy import StaticEagerEligibility
from specrhythm.continuation.trace import TRACE
from specrhythm.phase4.draft_batch import DraftCommitPlan, unique_ids
from specrhythm.phase4.dual_commit import dual_greedy_acceptance
from specrhythm.phase4.serial import PROTOCOL_VERSION, Proposal, token_prefix_hash
from specrhythm.serving.common import require
from specrhythm.serving.fixed_settle import DiagnosticSerialMachine, remaining


class EagerSerialMachine(DiagnosticSerialMachine):
    def __init__(self, backend, *, request_ids, candidate_budget=4, report_path=None):
        require(candidate_budget == 4, "serial-eager candidate length is frozen at four")
        super().__init__(backend, candidate_budget=4, report_path=report_path)
        self.core = RollingContinuation("serial-draft-owner", StaticEagerEligibility(request_ids))
        self.registered = set()
        self.works = {}
        self.parent_plans = {}
        self.parent_results = {}
        self.verify_bindings = {}
        self.sync_bindings = {}
        self.work_times = {}
        self.counters = Counter()
        self.events = []
        self.accounted = Counter()
        self.owner_stopped = False

    def _event(self, phase, *, request_ids=(), start_ns=None, end_ns=None,
               causal_bindings=None, **counts):
        now = time.monotonic_ns() if end_ns is None else end_ns
        delta = Counter(counts)
        for rid in request_ids:
            values = asdict(self.core._state(rid).accounting)
            for name, value in values.items():
                key = (rid, name)
                change = value - self.accounted[key]
                if change:
                    delta[name] += change
                self.accounted[key] = value
        if TRACE.enabled:
            bindings = [] if causal_bindings is None else causal_bindings
            for rid in request_ids if causal_bindings is None else ():
                state = self.core._state(rid)
                work = self.works.get(rid)
                bindings.append(dict(request_id=rid, round_id=(work.prefix_version if work else
                                                                state.committed_prefix_version),
                                     proposal_id=(work.parent_proposal_id if work else
                                                  state.current_proposal_id),
                                     continuation_id=work.work_id if work else None,
                                     continuation_status=(state.continuations[work.work_id].status
                                                          if work else None)))
            TRACE.event("protocol_"+phase, bindings=bindings,
                        counter_delta=dict(delta), start_ns=now if start_ns is None else start_ns,
                        end_ns=now)
        self.counters.update(delta)
        self.events.append(
            {
                "phase": phase,
                "request_ids": list(request_ids),
                "start_ns": now if start_ns is None else start_ns,
                "end_ns": now,
                "counter_delta": dict(delta),
            }
        )

    def eager_report(self):
        from specrhythm.serving.eager_results import COUNTERS

        return {
            "schema_version": "specrhythm.rolling-eager-gpu.v1",
            "counters": {name: self.counters[name] for name in COUNTERS},
            "events": list(self.events),
            "pending_work": list(self.works),
            "owner_stopped": self.owner_stopped,
            "GPU_overlap": "UNKNOWN",
            "protocol": "specrhythm.continuation",
        }

    def _attach(self, row):
        rid = row["request_id"]
        if rid not in self.registered:
            state = self._state(rid)
            self.core.register(
                rid,
                state.committed_token_ids,
                max_output_tokens=row["remaining_output_budget"],
                eos_token_ids=row.get("eos_token_ids", ()),
                committed_prefix_version=state.next_round_id,
            )
            self.registered.add(rid)

    @TRACE.observe("machine_batch_propose")
    def batch_propose(self, rows):
        unique_ids([r["request_id"] for r in rows])
        normal, reused, works = [], [], {}
        for row in rows:
            self._attach(row)
            rid = row["request_id"]
            state = self.core._state(rid)
            require(
                row["round_id"] == self._state(rid).next_round_id
                and row["committed_prefix_len"] == state.committed_prefix_length
                and row["committed_prefix_hash"] == state.committed_prefix_hash
                and row["remaining_output_budget"] == state.remaining_output_tokens,
                "serial-eager normal/proposed prefix or budget mismatch",
            )
            if state.current_proposal_id:
                proposal = state.proposals[state.current_proposal_id]
                require(proposal.status == "ready", "unconfirmed eager cannot enter READY")
                if proposal.tokens:
                    reused.append(self._state(rid).pending_proposal.to_dict())
            else:
                works[rid] = self.core.schedule_normal_recovery(rid)
                normal.append(row)
        kinds = {rid: work.kind for rid, work in works.items()}
        with TRACE.span("normal_recovery_proposal", request_kinds=kinds,
                        requests=[{"request_id": r["request_id"], "round_id": r["round_id"]}
                                  for r in normal]):
            result = super().batch_propose(normal)
        for value in result["proposals"]:
            rid = value["request_id"]
            work = works.pop(rid)
            physical = self.backend.states[rid]
            proposal = self.core.record_normal_completion(
                work,
                DraftCompletion(
                    work.work_id, tuple(value["proposal_token_ids"]), physical.materialized
                ),
            )
            legacy = self._state(rid).pending_proposal
            legacy = replace(
                legacy,
                runtime_provenance={
                    **legacy.runtime_provenance,
                    "rolling_proposal_id": proposal.proposal_id,
                    "rolling_prefix_version": proposal.prefix_version,
                    "source_continuation_id": proposal.source_continuation_id,
                },
            )
            self._state(rid).pending_proposal = legacy
            value.update(legacy.to_dict())
            self._event("normal_draft", request_ids=(rid,), start_ns=value["draft_start_ns"])
        for rid, work in works.items():
            require(work.candidate_length == 0, "missing physical normal proposal completion")
            self.core.record_normal_completion(
                work, DraftCompletion(work.work_id, (), self.backend.states[rid].materialized)
            )
            self._event("target_tail_ready", request_ids=(rid,))
        result["proposals"].extend(reused)
        return result

    @TRACE.observe("machine_verify_start")
    def verify_start(self, rows):
        unique_ids([r["request_id"] for r in rows])
        for row in rows:
            rid = row["request_id"]
            old = self.verify_bindings.get((rid, row["round_id"]))
            if old is not None:
                require(old == row, "conflicting duplicate eager verify admission")
                continue
            legacy = self._state(rid).pending_proposal
            state = self.core._state(rid)
            require(
                legacy is not None
                and legacy.round_id == row["round_id"]
                and state.current_proposal_id == row["proposal_id"]
                and state.committed_prefix_hash == row["parent_prefix_hash"]
                and state.committed_prefix_length == row["parent_prefix_len"]
                and tuple(row["proposal_tokens"]) == legacy.proposal_token_ids,
                "eager verify-start identity/proposal/full dependency mismatch",
            )
        works = []
        for row in rows:
            rid, key = row["request_id"], (row["request_id"], row["round_id"])
            if key in self.verify_bindings:
                continue
            self.verify_bindings[key] = dict(row)
            proposal = self.core.start_verification(rid, row["proposal_id"])
            self._event(
                "verify_start",
                request_ids=(rid,),
                verified_promoted_candidates=(
                    len(proposal.tokens) if proposal.source_continuation_id else 0
                ),
            )
            work = self.core.begin_continuation(rid)
            if work is not None:
                works.append(work)
        # No owner command or GPU write interleaves with this validation batch.
        self.backend.begin_gpu_continuations(works)
        for work in works:
            self.works[work.request_id] = work
            self.work_times[work.work_id] = [None, None]
            self._event("eager_admission", request_ids=(work.request_id,), admissions=1)
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
        started = time.monotonic_ns()
        completions = self.backend.step_gpu_continuations(works)
        for work in works:
            if self.work_times[work.work_id][0] is None:
                self.work_times[work.work_id][0] = started
                self._event(
                    "eager_start", request_ids=(work.request_id,), start_ns=started, started=1
                )
            result = completions[work.work_id]
            if result is not None:
                self.work_times[work.work_id][1] = time.monotonic_ns()
                self.core.record_continuation_completion(work, result)
                self._event(
                    "eager_complete", request_ids=(work.request_id,), start_ns=started, completed=1
                )
        return True

    @TRACE.observe("machine_abort")
    def _abort(self, rid, reason):
        work = self.works.get(rid)
        if work is None:
            return
        state = self.core._state(rid)
        c = state.continuations[work.work_id]
        if c.status not in (
            ContinuationStatus.INVALIDATED,
            ContinuationStatus.CANCELLED,
            ContinuationStatus.RELEASED,
        ):
            self.core.discard_continuation(rid, work.work_id, reason=reason)
        evidence = self.backend.abort_gpu_continuation(work)
        # Waiting feedback may be released by a provider switch instead of a
        # full completion. Its wait ends at this real fenced abort boundary.
        if self.work_times[work.work_id][1] is None:
            self.work_times[work.work_id][1] = time.monotonic_ns()
        self.core.record_continuation_abort(
            work, evidence["generated_tokens"], evidence["materialized_kv_frontier"]
        )
        self._event("eager_abort", request_ids=(rid,))

    @TRACE.observe("machine_prepare_synchronizations")
    def prepare_synchronizations(self, rows, *, draining=False):
        for row in rows:
            rid = row["request_id"]
            key = (rid, row["round_id"])
            if key in self.sync_bindings:
                require(self.sync_bindings[key] == row, "conflicting duplicate Target feedback")
                continue
            legacy = self._pending(rid, row["round_id"]).pending_proposal
            state = self.core._state(rid)
            proposal = state.proposals[state.current_proposal_id]
            final = state.committed_prefix + tuple(row["committed_delta"])
            require(
                token_prefix_hash(final) == row["committed_prefix_hash"],
                "Target feedback full prefix hash mismatch",
            )
            decision = dual_greedy_acceptance(
                legacy.proposal_token_ids, row["committed_delta"], terminal=row["terminal"]
            )
            reason = None
            if row["terminal"]:
                reason = "eos" if final[-1] in state.eos_token_ids else "length"
            plan = DraftCommitPlan(
                rid,
                legacy.round_id,
                state.committed_prefix,
                legacy.proposal_token_ids,
                len(decision.accepted_draft_token_ids),
                decision.target_correction_token_ids + decision.target_bonus_token_ids,
                row["committed_prefix_hash"],
                row["terminal"],
                len(decision.target_correction_token_ids),
                len(decision.target_bonus_token_ids),
            )
            self.core.resolve_parent_verification(
                ParentVerification(
                    self.core.owner_id,
                    rid,
                    proposal.proposal_id,
                    proposal.prefix_version,
                    proposal.parent_prefix,
                    decision.committed_token_ids,
                    reason,
                )
            )
            self.sync_bindings[key] = dict(row)
            self.parent_plans[key] = (plan, decision)
            self._event(
                "parent_result",
                request_ids=(rid,),
                causal_bindings=[dict(request_id=rid, round_id=proposal.prefix_version,
                    proposal_id=proposal.proposal_id,
                    continuation_id=self.works[rid].work_id if rid in self.works else None)],
                parent_full_accepts=int(not decision.rejected_draft_token_ids),
                parent_rejections=int(bool(decision.rejected_draft_token_ids)),
                accepted_promoted_candidates=(
                    len(decision.accepted_draft_token_ids)
                    if proposal.source_continuation_id
                    else 0
                ),
            )
            if rid in self.works:
                c = self.core._state(rid).continuations[self.works[rid].work_id]
                if (
                    draining
                    or row["terminal"]
                    or c.status
                    in (
                        ContinuationStatus.INVALIDATED,
                        ContinuationStatus.CANCELLED,
                    )
                ):
                    self._abort(rid, "drain" if draining else "parent_invalid")

    @TRACE.observe("machine_finish_synchronizations")
    def finish_synchronizations(self, rows):
        for row in rows:
            rid = row["request_id"]
            if rid in self.works:
                c = self.core._state(rid).continuations[self.works[rid].work_id]
                if c.status == ContinuationStatus.WAITING_DRAFT:
                    return None
        pending = []
        for row in rows:
            rid, key = row["request_id"], (row["request_id"], row["round_id"])
            if key in self.parent_results:
                continue
            plan, decision = self.parent_plans[key]
            work = self.works.get(rid)
            c = self.core._state(rid).continuations[work.work_id] if work else None
            tokens = (
                c.continuation_tokens if c and c.status == ContinuationStatus.PROMOTABLE else None
            )
            if work and tokens is None:
                self._abort(rid, "dependency_invalid")
            pending.append((plan, decision, work, c, tokens))
        physical_results = self.backend.rebase_gpu_parents([
            (plan, work, tokens) for plan, _, work, _, tokens in pending
        ])
        for plan, decision, work, c, tokens in pending:
            rid, key = plan.request_id, (plan.request_id, plan.round_id)
            physical = physical_results[rid]
            state = self.core._state(rid)
            state.accounting.draft_materialized_tokens += physical["materialized_query_tokens"]
            if tokens is None and not plan.terminal:
                require(
                    physical["materialized_kv_length"] == len(plan.final_prefix),
                    "recovery prefix was not physically materialized",
                )
                state.materialized_kv_prefix = plan.final_prefix
            # Logical committed length excludes the already materialized future suffix.
            result = self._publish_commit(plan, decision, physical)
            result["state"]["logical_draft_kv_length"] = len(plan.final_prefix)
            if tokens is not None:
                promoted = self.core.promote_continuation(rid, work.work_id)
                started, ended = self.work_times[work.work_id]
                legacy = Proposal(
                    PROTOCOL_VERSION,
                    rid,
                    plan.round_id + 1,
                    len(plan.final_prefix),
                    token_prefix_hash(plan.final_prefix),
                    tokens,
                    bool(tokens and tokens[-1] in self.core._state(rid).eos_token_ids),
                    started,
                    ended,
                    0,
                    self.backend.provenance,
                    {
                        "backend": self.backend.backend_name,
                        "rolling_proposal_id": promoted.proposal_id,
                        "rolling_prefix_version": promoted.prefix_version,
                        "source_continuation_id": work.work_id,
                        "full_context_replay": False,
                        "persistent_cross_round_kv": True,
                        "materialized_kv_length": physical["materialized_kv_length"],
                    },
                )
                self._state(rid).pending_proposal = legacy
                self._state(rid).proposal_count += 1
            self.works.pop(rid, None)
            self.parent_results[key] = result
            self._event(
                "parent_settled",
                request_ids=(rid,),
                causal_bindings=[dict(request_id=rid, round_id=plan.round_id,
                    proposal_id=(work.parent_proposal_id if work else
                                 self.verify_bindings.get(key, {}).get("proposal_id")),
                    continuation_id=work.work_id if work else None,
                    promoted_proposal_id=state.current_proposal_id if tokens else None)],
                promotions=int(tokens is not None),
                bridge_matches=int(tokens is not None),
                bridge_mismatches=int(c is not None and c.reason == "bridge_mismatch"),
            )
        return [self.parent_results[(row["request_id"], row["round_id"])] for row in rows]

    def finish_authoritative(self, row):
        rid = row["request_id"]
        if "committed_prefix" not in row:
            # Proposer-free terminal tails have no authoritative delta in this RPC.
            # The coordinator sends its actual canonical prefix immediately afterward.
            return {"request_id": rid, "awaiting_authoritative_terminal": True}
        final = tuple(row["committed_prefix"])
        require(
            final and token_prefix_hash(final) == row["committed_prefix_hash"],
            "terminal prefix evidence mismatch",
        )
        legacy = self._state(rid)
        if rid in self.backend.retired:
            require(legacy.committed_token_ids == final, "late terminal prefix differs")
            return {"request_id": rid, "finished": True}
        delta = final[len(legacy.committed_token_ids) :]
        require(
            final[: len(legacy.committed_token_ids)] == legacy.committed_token_ids,
            "terminal prefix regressed",
        )
        if legacy.pending_proposal is not None and delta:
            sync = {
                "request_id": rid,
                "round_id": legacy.next_round_id,
                "committed_delta": list(delta),
                "committed_prefix_hash": token_prefix_hash(final),
                "terminal": True,
            }
            self.prepare_synchronizations((sync,), draining=True)
            self.finish_synchronizations((sync,))
        elif delta:
            require(len(delta) == 1, "terminal tail must contain exactly one Target token")
            if rid not in self.registered:
                # An admitted max_new_tokens=2 request has only this one token
                # after bootstrap and never passed the normal-proposal entrance.
                self.core.register(
                    rid,
                    legacy.committed_token_ids,
                    max_output_tokens=1,
                    eos_token_ids=tuple(row.get("eos_token_ids", ())),
                    committed_prefix_version=legacy.next_round_id,
                )
                self.registered.add(rid)
            plan = DraftCommitPlan(
                rid,
                legacy.next_round_id,
                legacy.committed_token_ids,
                (),
                0,
                delta,
                token_prefix_hash(final),
                True,
                1,
                0,
            )
            self.backend.rebase_gpu_parent(plan)
            if rid in self.registered:
                state = self.core._state(rid)
                pid = state.current_proposal_id
                if pid is None:
                    work = self.core.schedule_normal_recovery(rid)
                    require(work.candidate_length == 0, "unexpected unproposed terminal tail")
                    pid = self.core.record_normal_completion(
                        work, DraftCompletion(work.work_id, (), len(state.committed_prefix))
                    ).proposal_id
                parent = self.core.start_verification(rid, pid)
                reason = "eos" if delta[-1] in state.eos_token_ids else "length"
                self.core.resolve_parent_verification(
                    ParentVerification(
                        self.core.owner_id,
                        rid,
                        pid,
                        parent.prefix_version,
                        parent.parent_prefix,
                        delta,
                        reason,
                    )
                )
                self._event("terminal_tail", request_ids=(rid,))
            legacy.committed_token_ids, legacy.finished = final, True
        else:
            self._abort(rid, "terminal")
            self.backend.finish_many((rid,))
            self.works.pop(rid, None)
            if rid in self.registered:
                self.core.finish_or_cancel(rid)
                self._event("terminal_release", request_ids=(rid,))
            legacy.finished = True
            legacy.pending_proposal = legacy.pending_decision = None
        return {"request_id": rid, "finished": True}

    def diagnostic_settle(self, row):
        remaining(row["deadline_ns"])
        rid = row["request_id"]
        legacy = self.requests.get(rid)
        if legacy is not None and rid in self.registered:
            final = tuple(row["committed_prefix"])
            delta = final[len(legacy.committed_token_ids) :]
            if delta and legacy.pending_proposal is not None:
                sync = {
                    "request_id": rid,
                    "round_id": legacy.next_round_id,
                    "committed_delta": list(delta),
                    "committed_prefix_hash": row["committed_prefix_hash"],
                    "terminal": row["natural_terminal"],
                }
                self.prepare_synchronizations((sync,), draining=True)
                self.finish_synchronizations((sync,))
            elif delta:
                require(row["natural_terminal"], "unresolved nonterminal proposal-free delta")
                self.finish_authoritative({**row, "terminal": True})
            self._abort(rid, "drain")
        result = super().diagnostic_settle(row)
        self.works.pop(rid, None)
        if rid in self.registered:
            self.core.finish_or_cancel(rid)
            self._event("diagnostic_release", request_ids=(rid,))
        return result

    def set_eager(self, enabled, version):
        old = self.core.provider
        self.core.provider = StaticEagerEligibility(old.request_ids, enabled, version)
        for rid in self.registered:
            self.core.evaluate_eager_eligibility(rid)
            if not enabled and rid in self.works:
                self._abort(rid, "eager_disabled")
        return {"enabled": enabled, "decision_version": version}

    def shutdown(self, deadline_ns=None):
        if deadline_ns is not None:
            remaining(deadline_ns)
        require(
            not self.works and not any(s.pending_proposal for s in self.requests.values()),
            "eager shutdown has unresolved work; settle under the shared deadline first",
        )
        self.core.shutdown()
        self.backend.shutdown()
        if deadline_ns is not None:
            remaining(deadline_ns)
        return {
            "shutdown": True,
            "pending_work": [],
            "physical_live_requests": len(self.backend.states),
        }
