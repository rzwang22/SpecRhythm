"""Uniform K3 over the real independent-settlement PingPong owner."""

import time
from dataclasses import replace

from specrhythm.continuation.prepost import PrePostState
from specrhythm.phase4.draft_batch import DraftProposalPlan, unique_ids
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.serving.common import require
from specrhythm.serving.k3 import PARAMETERS, PROTOCOL
from specrhythm.serving.ping_prepost_machine import PingPrePostMachine


class K3Machine(PingPrePostMachine):
    protocol, parameters = PROTOCOL, PARAMETERS
    uniform_candidate_length = 3

    def batch_propose(self, rows):
        """Only cached seeds here; all extensions run in preemptible owner steps."""
        unique_ids([r["request_id"] for r in rows])
        plans = []
        for row in rows:
            rid = row["request_id"]
            legacy = self._state(rid)
            require(
                not legacy.finished
                and legacy.pending_proposal is None
                and rid not in self.core.states
                and row["round_id"] == legacy.next_round_id
                and row["committed_prefix_len"] == len(legacy.committed_token_ids)
                and row["committed_prefix_hash"] == token_prefix_hash(legacy.committed_token_ids)
                and type(row["remaining_output_budget"]) is int
                and row["remaining_output_budget"] > 0,
                "stale K3 registration",
            )
            plans.append(
                DraftProposalPlan(
                    rid,
                    legacy.next_round_id,
                    legacy.committed_token_ids,
                    1,
                    tuple(row.get("eos_token_ids", ())),
                )
            )
        seeds = self.backend.propose_many(plans)
        self.counters["initial_seed_candidates"] += sum(map(len, seeds.values()))
        now = time.monotonic_ns()
        for plan, row in zip(plans, rows):
            rid = plan.request_id
            state = self.core.states[rid] = PrePostState(
                plan.prefix, plan.round_id, row["remaining_output_budget"], plan.eos_token_ids
            )
            tokens = seeds[rid]
            budget = min(3, state.remaining)
            if len(tokens) == budget or tokens[-1] in state.eos_token_ids:
                self._publish_proposal(rid, tokens, now, now)
            else:
                self.normal[rid] = replace(plan, budget=budget)
            self._ping(
                "initial_seed",
                request_id=rid,
                candidate_count=len(tokens),
                new_model_forwards=0,
                prefix_version=state.prefix_version,
            )
        return dict(actual_batch_model_forward_count=0)

    def _publish_proposal(self, rid, tokens, started, ended, *, source=None):
        state = self.core._state(rid)
        require(
            len(tokens) == min(3, state.remaining)
            or (0 < len(tokens) < min(3, state.remaining) and tokens[-1] in state.eos_token_ids),
            "incomplete K3 cannot be READY",
        )
        proposal = super()._publish_proposal(rid, tokens, started, ended, source=source)
        if state.prefix_version == 0 and source is None:
            # propose_many already counted the cached seed. The two owner steps
            # finish that same initial proposal; they do not create new proposals.
            self.backend.metrics.counters["proposed_tokens"] += len(tokens) - 1
        return proposal

    def _account_background(self, works, plans, before):
        super()._account_background(works, plans, before)
        # Each normal plan just completed one actual fenced token step, including
        # when combined with settlement or eager work in the same physical batch.
        self.counters["normal_extension_candidates"] += len(plans)

    def diagnostic_settle(self, row):
        rid = row["request_id"]
        repeated = rid in self.diagnostic_receipts
        consumed = self.claims.get(rid, {}).get("consumed", False)
        result = super().diagnostic_settle(row)
        if not repeated and not consumed:
            self.counters["discarded_unsubmitted_candidates"] += result[
                "discarded_proposal_tokens"
            ]
        return result

    def eager_report(self):
        result = super().eager_report()
        c = self.counters
        generated = sum(c[k] for k in (
            "initial_seed_candidates", "normal_extension_candidates",
            "post_seed_candidates", "early_generated_tokens",
        ))
        accounted = sum(c[k] for k in (
            "target_submitted_candidates", "discarded_early_tokens",
            "discarded_unsubmitted_candidates",
        ))
        result["candidate_accounting"] = dict(
            scope="owner lifetime; candidates, not committed tokens or forward calls",
            generated=generated,
            submitted=c["target_submitted_candidates"],
            discarded_early=c["discarded_early_tokens"],
            discarded_unsubmitted=c["discarded_unsubmitted_candidates"],
            live_unsubmitted=generated - accounted,
            # Submitted means the accepted verify-start hook. Native Target
            # association separately proves that its forward actually ran.
            submitted_boundary="validated verify_start, once per consumed claim",
        )
        return result

    def _ping(self, event, **values):
        if event == "consume":
            self.counters["target_submitted_candidates"] += len(
                self.core._state(values["request_id"]).proposal_tokens
            )
        if event == "settlement":
            self.counters["post_seed_candidates"] += max(
                0, values["next_candidate_length"] - values["retained"]
            )
        if event == "ready":
            state = self.core._state(values["request_id"])
            values.update(
                remaining_output_budget=state.remaining,
                short_reason=(
                    "candidate_EOS"
                    if values["candidate_EOS"]
                    else "output_budget"
                    if state.remaining < 3
                    else None
                ),
            )
        if event == "admission":
            values["target_available"] = True
            values["waiting_inventory"] = [
                dict(
                    request_id=r["request_id"],
                    prefix_version=self.core._state(r["request_id"]).prefix_version,
                    reason="normal_or_recovery_incomplete"
                    if r["request_id"] in self.normal
                    else "feedback_or_settlement_pending"
                    if r["request_id"] in self.claims
                    else "lookahead_pending"
                    if r["request_id"] in self.works
                    else r["reason"],
                )
                for r in values["deferred"]
            ]
        super()._ping(event, **values)

    def views(self, active):
        # Both PingPong modes have exactly the same admission policy. Reuse does
        # not give an extra scheduling priority; static eligibility controls work only.
        return [replace(v, promoted=False) for v in super().views(active)]
