"""Serial-only, batched greedy Draft with private persistent paged KV."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

from specrhythm.phase4.config import Phase4Config
from specrhythm.phase4.draft_batch import (
    DraftCommitPlan,
    DraftMaterialization,
    DraftProposalPlan,
    commit_frontier,
    token_tuple,
    unique_ids,
)
from specrhythm.phase4.draft_metrics import DraftMetrics
from specrhythm.phase4.serial import token_prefix_hash

BACKEND_ENV = "SR_PHASE4_DRAFT_BACKEND"


def selected_draft_backend(environ: Optional[Mapping[str, str]] = None) -> str:
    value = (os.environ if environ is None else environ).get(BACKEND_ENV, "hf-persistent")
    if value not in {"hf-persistent", "vllm-batched"}:
        raise ValueError(f"{BACKEND_ENV} must be hf-persistent or vllm-batched")
    return value


@dataclass
class DraftRequestHandle:
    internal_id: str
    prefix: Tuple[int, ...]
    materialized: int
    next_logits: Any
    next_round: int = 0
    proposal: Optional[Tuple[int, ...]] = None


class VllmBatchedDraftBackend:
    backend_name = "vllm-batched-paged-kv-draft"

    def __init__(self, config: Phase4Config, *, worker: Any = None) -> None:
        self.metrics = DraftMetrics()
        self.owner_thread = threading.get_ident()
        self.states: dict[str, DraftRequestHandle] = {}
        self.retired: set[str] = set()
        self.closed = False
        self.failed = False
        self.max_model_len = config.max_model_len
        if worker is None:
            from specrhythm.phase4.vllm_draft_worker import VllmDraftWorker

            worker = VllmDraftWorker(config, self.metrics)
        else:
            worker.metrics = self.metrics
        self.worker = worker
        self._provenance = {
            **worker.provenance,
            "backend": self.backend_name,
            "serving_performance_backend": True,
            "full_context_prefill_per_request": 1,
            "full_context_replay_per_round": False,
            "persistent_cross_round_kv": True,
            "proposal_first_token_source": "committed-prefix cached logits",
        }

    @property
    def provenance(self) -> Mapping[str, Any]:
        return self._provenance

    def _check(self) -> None:
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError("Draft worker must stay on its construction/CUDA owner thread")
        if self.closed or self.failed:
            raise RuntimeError("Draft backend is closed or failed; restart the service")

    def _fail(self) -> None:
        self.failed = self.metrics.failed = True

    def initialize(self, request_id: str, committed_token_ids: Sequence[int]) -> None:
        self.initialize_many(((request_id, committed_token_ids),))

    def initialize_many(self, rows: Sequence[tuple[str, Sequence[int]]]) -> None:
        self._check()
        unique_ids([row[0] for row in rows])
        plans = []
        for request_id, values in rows:
            prefix = token_tuple(values)
            if request_id in self.states or request_id in self.retired:
                raise ValueError("Draft request ID cannot be initialized twice")
            if not prefix or len(prefix) > self.max_model_len:
                raise ValueError("Draft initialization prefix exceeds the model context")
            # Opaque identity, not a CUDA rank or an InputBatch row.
            plans.append((request_id, DraftMaterialization(f"sr-draft:{request_id}", prefix, 0)))
        if not plans:
            return
        try:
            logits = self.worker.materialize([p for _, p in plans], "setup")
            self.worker.fence("setup")
            for request_id, row in plans:
                self.states[request_id] = DraftRequestHandle(
                    row.request_id, row.context, len(row.context), logits[row.request_id]
                )
        except Exception:
            self._fail()
            raise

    def propose_many(self, plans: Sequence[DraftProposalPlan]) -> dict[str, Tuple[int, ...]]:
        self._check()
        unique_ids([p.request_id for p in plans])
        for plan in plans:
            state = self.states[plan.request_id]
            if (
                state.proposal is not None
                or state.next_round != plan.round_id
                or state.prefix != plan.prefix
                or state.materialized != len(plan.prefix)
                or state.next_logits is None
            ):
                raise ValueError("stale Draft proposal/prefix/round/materialized state")
            if len(plan.prefix) + plan.budget > self.max_model_len:
                raise ValueError("Draft proposal exceeds model context")
        tokens: dict[str, list[int]] = {p.request_id: [] for p in plans}
        active = [p for p in plans if p.budget]
        previous_end = None
        try:
            while active:
                started = time.monotonic_ns()
                if previous_end is not None:
                    self.metrics.host_gap_ns += started - previous_end
                    self.metrics.host_gap_count += 1
                self.metrics.active_steps[len(active)] += 1
                sampled = self.worker.greedy(
                    [self.states[p.request_id].next_logits for p in active]
                )
                if len(sampled) != len(active):
                    raise RuntimeError("batched Draft sampler row count mismatch")
                token_tuple(sampled)
                continuing = []
                rows = []
                for plan, token in zip(active, sampled):
                    produced = tokens[plan.request_id]
                    produced.append(token)
                    if token not in plan.eos_token_ids and len(produced) < plan.budget:
                        state = self.states[plan.request_id]
                        rows.append(
                            DraftMaterialization(
                                state.internal_id,
                                state.prefix + tuple(produced),
                                state.materialized,
                            )
                        )
                        continuing.append(plan)
                if rows:
                    logits = self.worker.materialize(rows, "proposal")
                    for plan, row in zip(continuing, rows):
                        state = self.states[plan.request_id]
                        state.materialized = len(row.context)
                        state.next_logits = logits[row.request_id]
                active = continuing
                previous_end = time.monotonic_ns()
            if any(tokens.values()):
                self.worker.fence("proposal_complete")
            for plan in plans:
                proposal = tuple(tokens[plan.request_id])
                if proposal:
                    self.states[plan.request_id].proposal = proposal
                    self.metrics.counters["proposals"] += 1
                    self.metrics.counters["proposed_tokens"] += len(proposal)
            return {request_id: tuple(values) for request_id, values in tokens.items()}
        except Exception:
            self._fail()
            raise

    def commit_many(self, plans: Sequence[DraftCommitPlan]) -> dict[str, dict[str, Any]]:
        self._check()
        unique_ids([p.request_id for p in plans])
        rows = []
        frontiers = {}
        refreshes = 0
        for plan in plans:
            state = self.states[plan.request_id]
            if (
                state.next_round != plan.round_id
                or state.prefix != plan.parent_prefix
                or state.proposal != plan.proposal
            ):
                raise ValueError("stale Draft commit proposal/prefix/round")
            if len(plan.final_prefix) > self.max_model_len:
                raise ValueError("Draft commit exceeds model context")
            frontier = commit_frontier(plan, state.materialized)
            frontiers[plan.request_id] = frontier
            if frontier == len(plan.final_prefix):
                if not plan.terminal and frontier != state.materialized:
                    frontier -= 1  # Refresh logits at a shortened accepted frontier.
                    refreshes += 1
                else:
                    continue
            rows.append(DraftMaterialization(state.internal_id, plan.final_prefix, frontier))
        result = {}
        try:
            logits = self.worker.materialize(rows, "commit") if rows else {}
            if rows:
                self.worker.fence("commit_complete")
            for plan in plans:
                state = self.states[plan.request_id]
                state.prefix = plan.final_prefix
                state.materialized = len(plan.final_prefix)
                state.next_logits = logits.get(state.internal_id, state.next_logits)
                state.next_round += 1
                state.proposal = None
                self.metrics.counters["commits"] += 1
                self.metrics.counters["rollback_requests"] += 1
                self.metrics.counters["invalidated_tokens"] += len(plan.proposal) - plan.accepted
                self.metrics.counters["correction_tokens"] += plan.correction_count
                self.metrics.counters["bonus_tokens"] += plan.bonus_count
                result[plan.request_id] = {
                    "materialized_kv_length": state.materialized,
                    "valid_prefix_before_materialization": frontiers[plan.request_id],
                    "committed_prefix_hash": token_prefix_hash(state.prefix),
                    **self.worker.request_evidence(state.internal_id),
                }
            self.metrics.counters["commit_materialized_tokens"] += sum(len(r.suffix) for r in rows)
            self.metrics.counters["logit_refresh_requests"] += refreshes
            if rows and any(p.target_tail for p in plans):
                self.metrics.counters["correction_bonus_batches"] += 1
            if plans:
                self.metrics.counters["commit_batches"] += 1
            self.finish_many([p.request_id for p in plans if p.terminal])
            return result
        except Exception:
            self._fail()
            raise

    def finish_many(self, request_ids: Sequence[str]) -> None:
        self._check()
        unique_ids(request_ids)
        ids = [rid for rid in request_ids if rid in self.states]
        if set(request_ids) - set(ids) - self.retired:
            raise ValueError("cannot finish an unknown Draft request")
        if ids:
            try:
                self.worker.release([self.states[rid].internal_id for rid in ids])
                for rid in ids:
                    del self.states[rid]
                    self.retired.add(rid)
            except Exception:
                self._fail()
                raise

    def finish(self, request_id: str) -> None:
        self.finish_many((request_id,))

    def report(self) -> dict[str, Any]:
        return {
            **self.metrics.snapshot(self.backend_name),
            "provenance": self.provenance,
            "draft_live_requests_final": len(self.states),
            "draft_retired_request_count": len(self.retired),
            "backend_shutdown_complete": self.closed,
            "worker_resources": self.worker.resource_evidence(),
        }

    def shutdown(self) -> None:
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError("Draft shutdown must run on the owner thread")
        if self.closed:
            return
        try:
            self.worker.shutdown()
        finally:
            self.states.clear()
            self.closed = True
