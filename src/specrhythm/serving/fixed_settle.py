"""Diagnostic-only settlement on the existing Draft CUDA owner, without proposing."""

from __future__ import annotations

import queue
import time
from pathlib import Path

from specrhythm.phase4.batched_draft_service import BatchedDraftStateMachine
from specrhythm.phase4.draft_batch import DraftCommitPlan
from specrhythm.phase4.draft_service import DraftUnixServer
from specrhythm.phase4.dual_batched_draft import BatchedDualDraftController
from specrhythm.phase4.dual_commit import dual_greedy_acceptance
from specrhythm.phase4.dual_service import _Work
from specrhythm.phase4.serial import greedy_acceptance, token_prefix_hash
from specrhythm.serving.common import digest, require
from specrhythm.serving.s2_draft import S2DualMachine
from specrhythm.serving.s2_pool import ResidentPoolAudit, publish


def remaining(deadline_ns):
    value = (deadline_ns - time.monotonic_ns()) / 1e9
    require(value > 0, "diagnostic total drain deadline expired")
    return value


class Settlement:
    """Publish logical cancellation only after real owner release has succeeded."""

    dual = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagnostic_receipts = {}
        self.release_receipts = {}
        finish = self.backend.finish_many

        def observed_finish(ids):
            before = {rid: self.backend.states[rid] for rid in ids if rid in self.backend.states}
            result = finish(ids)
            now = time.monotonic_ns()
            for rid, physical in before.items():
                require(
                    rid not in self.backend.states and rid in self.backend.retired,
                    "physical finish returned without retiring KV",
                    request_id=rid,
                )
                self.release_receipts[rid] = {
                    "resources_released_ns": now,
                    "internal_request_id": physical.internal_id,
                    "released_prefix_hash": token_prefix_hash(physical.prefix),
                }
            return result

        self.backend.finish_many = observed_finish

    def diagnostic_settle(self, row):
        deadline = row["deadline_ns"]
        remaining(deadline)
        rid = row["request_id"]
        binding = digest({k: v for k, v in row.items() if k != "deadline_ns"})
        receipts = getattr(self, "diagnostic_receipts", {})
        self.diagnostic_receipts = receipts
        if rid in receipts:
            require(receipts[rid]["binding"] == binding, "changed repeated diagnostic settlement")
            return receipts[rid]
        started = time.monotonic_ns()
        try:
            result = self._settle(row)
            result.update(binding=binding, start_ns=started, end_ns=time.monotonic_ns())
            receipts[rid] = result
            self.backend.diagnostic_receipts = receipts
            self._save_drain("SETTLING")
            remaining(deadline)
            return result
        except Exception as error:
            if self.report_path is not None:
                from specrhythm.serving.fixed_artifacts import record_error

                record_error(Path(self.report_path).parent, error, "draft_settlement")
            self._save_drain("FAILED", error=error, request_id=rid)
            raise

    def _save_drain(self, status, *, error=None, request_id=None):
        if self.report_path is None:
            return
        value = {
            "schema_version": "specrhythm.fixed-draft-drain.v1",
            "status": status,
            "settled_requests": len(self.diagnostic_receipts),
            "live_physical_requests": len(self.backend.states),
            "unresolved_logical_proposals": sum(
                getattr(s, "proposal" if self.dual else "pending_proposal") is not None
                for s in self.requests.values()
            ),
            "receipts": list(self.diagnostic_receipts.values()),
            "primary_error": str(error) if error else None,
            "request_id": request_id,
        }
        try:
            publish(Path(self.report_path).with_name("draft-drain-state.json"), value)
        except Exception as secondary:
            if error is None:
                raise
            from specrhythm.serving.fixed_artifacts import record_error

            record_error(Path(self.report_path).parent, secondary, "draft_drain_report")

    def _settle(self, row):
        rid = row["request_id"]
        final = tuple(row["committed_prefix"])
        require(
            final and token_prefix_hash(final) == row["committed_prefix_hash"],
            "invalid authoritative diagnostic prefix",
            request_id=rid,
        )
        terminal = row["natural_terminal"]
        require(type(terminal) is bool, "diagnostic terminal evidence must be explicit")
        state = self.requests.get(rid)
        physical = self.backend.states.get(rid)
        proposal = getattr(state, "proposal" if self.dual else "pending_proposal", None)
        proposal_tokens = () if proposal is None else proposal.proposal_token_ids
        parent = (
            tuple(state.committed_token_ids)
            if state is not None
            else (physical.prefix if physical is not None else ())
        )
        next_round = state.next_round_id if state is not None else 0
        result = {
            "request_id": rid,
            "logical_initialized": state is not None,
            "admitted": row["admitted"],
            "natural_terminal": terminal,
            "disposition": "NATURAL_TERMINAL" if terminal else "DIAGNOSTIC_CANCELLED",
            "authoritative_prefix_hash": row["committed_prefix_hash"],
            "authoritative_prefix_count": len(final),
            "authoritative_next_round_id": row["next_round_id"],
            "authoritative_prefix_version": row["prefix_version"],
            "draft_prefix_before_hash": token_prefix_hash(parent),
            "draft_prefix_before_count": len(parent),
            "committed_sync_tokens": 0,
            "discarded_proposal_tokens": 0,
            "discarded_proposal_id": None,
            "new_proposals_generated": 0,
            "physical_gpu_id": self.backend.provenance["physical_gpu_id"],
            "gpu_uuid": self.backend.provenance["gpu_uuid"],
        }
        target_only = row["runtime_mode"] == "target"
        if state is None:
            require(
                not row["admitted"]
                and row["next_round_id"] == 0
                and physical is not None
                and parent == final
                and physical.proposal is None,
                "uninitialized logical request is not an untouched resident",
                request_id=rid,
            )
        if physical is None:
            require(
                state is not None
                and state.finished
                and proposal is None
                and rid in self.backend.retired
                and rid in self.release_receipts
                and terminal,
                "missing physical Draft state lacks natural release evidence",
                request_id=rid,
            )
            # Ordinary Target-only never advances Draft; ordinary Serial may release
            # its proposal-free terminal tail before the coordinator sees the output.
            require(
                target_only
                or (
                    next_round == row["next_round_id"]
                    and (
                        parent == final
                        or (final[: len(parent)] == parent and len(final) == len(parent) + 1)
                    )
                ),
                "pre-released Draft final prefix differs",
                request_id=rid,
            )
            if self.dual:
                require(
                    parent == final and state.prefix_version == row["prefix_version"],
                    "pre-released Dual final prefix/version differs",
                    request_id=rid,
                )
            result.update(
                released=True,
                release_kind="already_naturally_released",
                **self.release_receipts[rid],
            )
            return result
        self.backend._check()
        require(
            physical.prefix == parent
            and physical.proposal == (None if proposal is None else proposal_tokens)
            and physical.next_round == next_round,
            "Draft logical/physical prefix or proposal mismatch",
            request_id=rid,
        )
        require(
            getattr(state, "pending_decision", None) is None,
            "partial Serial synchronization requires failure investigation",
            request_id=rid,
        )
        # Check private ownership before mutation; prove unrelated KV remains intact.
        before = self.backend.physical_rows()
        ResidentPoolAudit("diagnostic-release").check(before, {})
        unrelated = {k: v for k, v in before.items() if k != rid}
        internal = physical.internal_id
        if target_only:
            require(
                proposal is None
                and physical.next_round == 0
                and token_prefix_hash(parent) == row["bootstrap_prefix_hash"],
                "Target-only Draft provider was mutated",
                request_id=rid,
            )
            result["sync_reason"] = "Target-only provider KV is unused after real bootstrap"
        else:
            require(
                final[: len(parent)] == parent,
                "diagnostic committed prefix regressed",
                request_id=rid,
            )
            delta = final[len(parent) :]
            expected_round = next_round + int(bool(delta) and proposal is not None)
            require(
                row["next_round_id"] == expected_round,
                "diagnostic final round differs from actual Target verification",
                request_id=rid,
            )
            if self.dual and state is not None:
                require(
                    row["prefix_version"] == state.prefix_version + int(bool(delta)),
                    "diagnostic final prefix version mismatch",
                    request_id=rid,
                )
            if delta:
                if proposal is None:
                    require(
                        terminal and len(delta) == 1,
                        "proposal-free Draft sync must be a real terminal tail",
                        request_id=rid,
                    )
                    accepted, target_tokens, corrections, bonuses = 0, delta, 1, 0
                else:
                    decision = (dual_greedy_acceptance if self.dual else greedy_acceptance)(
                        proposal_tokens, delta, terminal=terminal
                    )
                    accepted = len(decision.accepted_draft_token_ids)
                    target_tokens = (
                        decision.target_correction_token_ids + decision.target_bonus_token_ids
                    )
                    corrections, bonuses = (
                        len(decision.target_correction_token_ids),
                        len(decision.target_bonus_token_ids),
                    )
                plan = DraftCommitPlan(
                    rid,
                    physical.next_round,
                    parent,
                    proposal_tokens,
                    accepted,
                    target_tokens,
                    row["committed_prefix_hash"],
                    terminal,
                    corrections,
                    bonuses,
                )
                # Existing production KV commit validates/truncates/materializes/fences.
                # No _propose_many / synchronize_and_batch_propose continuation is invoked.
                self.backend.commit_many((plan,))
                state.committed_token_ids = final
                state.next_round_id = expected_round
                if self.dual:
                    state.prefix_version = row["prefix_version"]
                    state.proposal = None
                else:
                    state.pending_proposal = state.pending_decision = None
                    state.rollback_applied = False
                state.finished = terminal
                proposal = None
                result["committed_sync_tokens"] = len(delta)
            elif proposal is not None:
                result["discarded_proposal_tokens"] = len(proposal_tokens)
                result["discarded_proposal_id"] = getattr(proposal, "proposal_id", None)
                result["discarded_round_id"] = proposal.round_id
        remaining(row["deadline_ns"])
        if rid in self.backend.states:
            self.backend.worker.fence("diagnostic_release")
            self.backend.finish_many((rid,))
        after = self.backend.physical_rows()
        require(
            rid not in self.backend.states and rid in self.backend.retired and after == unrelated,
            "diagnostic release failed or changed unrelated KV",
            request_id=rid,
        )
        if state is not None:
            # Normal EOS and diagnostic cancellation remain distinguishable. In
            # particular, cancellation does not set the natural `finished` flag.
            if self.dual:
                state.proposal = None
            else:
                state.pending_proposal = state.pending_decision = None
                state.rollback_applied = False
        result.update(
            released=True,
            release_kind="owner_private_KV_release",
            internal_request_id=internal,
            unrelated_KV_sha256=digest(unrelated),
            resources_released_ns=self.release_receipts[rid]["resources_released_ns"],
        )
        return result


class DiagnosticSerialMachine(Settlement, BatchedDraftStateMachine):
    pass  # Normal shutdown still executes DraftStateMachine's unresolved-proposal guard.


class DiagnosticSerialServer(DraftUnixServer):
    def _dispatch(self, operation, payload):
        if operation == "diagnostic_settle":
            return self.machine.diagnostic_settle(payload)
        return super()._dispatch(operation, payload)


class DiagnosticDualMachine(Settlement, S2DualMachine):
    dual = True

    def execute_batch(self, operation, rows):
        if operation == "diagnostic_settle":
            return [self.diagnostic_settle(row) for row in rows]
        return super().execute_batch(operation, rows)

    def shutdown(self):
        require(
            not any(s.proposal is not None for s in self.requests.values()),
            "diagnostic Dual shutdown has unresolved proposals",
        )
        return super().shutdown()


class DiagnosticDualController(BatchedDualDraftController):
    def execute(self, operation, row, *, timeout_seconds=120.0):
        if operation != "diagnostic_settle":
            return super().execute(operation, row, timeout_seconds=timeout_seconds)
        rid = row["request_id"]
        response = queue.Queue(maxsize=1)
        with self._lock:
            require(
                not self._inflight and not self._failures,
                "diagnostic settlement requires successful idle owner",
            )
            self._inflight.add(rid)
        self._work.put(_Work(operation, {"rows": (dict(row),)}, response))
        try:
            success, result = response.get(timeout=remaining(row["deadline_ns"]))
        except queue.Empty as error:
            raise TimeoutError("diagnostic owner settlement deadline expired") from error
        if not success:
            raise RuntimeError(str(result))
        with self._lock:
            # Remove transport references only after the actual owner receipt.
            self._ready.pop(rid, None)
            self._claimed.pop(rid, None)
            self._ready_order = [r for r in self._ready_order if r != rid]
        return result
