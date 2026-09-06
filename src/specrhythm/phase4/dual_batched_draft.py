"""D6 adapter: existing Dual wire cohorts on one production Draft owner thread."""

from __future__ import annotations

import queue
import threading
import time
from collections import Counter
from pathlib import Path

from specrhythm.phase4.batched_draft_service import write_immutable_report
from specrhythm.phase4.draft_batch import DraftCommitPlan, DraftProposalPlan, unique_ids
from specrhythm.phase4.dual import DualProposal, proposal_identity
from specrhythm.phase4.dual_commit import dual_greedy_acceptance
from specrhythm.phase4.dual_service import (
    AsyncDualDraftController,
    DualDraftMachine,
    DualDraftUnixServer,
    _Work,
)
from specrhythm.phase4.manifest import sha256_file
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.transport import CheckpointJsonl
from specrhythm.phase4.vllm_draft_backend import VllmBatchedDraftBackend


class BatchedDualDraftMachine(DualDraftMachine):
    """Keep Dual identities/acceptance; batch only the already submitted rows."""

    def __init__(self, backend, *, candidate_budget=4, report_path=None):
        super().__init__(backend, candidate_budget=candidate_budget)
        self.report_path = report_path
        self.cohorts = Counter()
        self.cohort_id = 0

    def execute_batch(self, operation, rows):
        unique_ids([str(r.get("request_id", "")) for r in rows])
        self.cohorts[(operation, len(rows))] += 1
        if operation == "bootstrap_and_propose":
            initialized = {r["request_id"]: self.initialize(r) for r in rows}
            active = [r for r in rows if not initialized[r["request_id"]]["terminal"]]
            proposed = self._propose_many(active)
            return [proposed.get(r["request_id"], initialized[r["request_id"]]) for r in rows]
        if operation == "propose_only":
            for row in rows:
                state = self._state(row["request_id"])
                if (
                    row.get("prefix_version") != state.prefix_version
                    or tuple(row.get("committed_token_ids", ())) != state.committed_token_ids
                    or row.get("prefix_token_sha256")
                    != token_prefix_hash(state.committed_token_ids)
                    or type(row.get("measurement_start_ns")) is not int
                    or row["measurement_start_ns"] <= 0
                ):
                    raise ValueError("proposal-only prefix/version/measurement boundary mismatch")
            results = self._propose_many(rows)
            for row in rows:
                interval = results[row["request_id"]].get("draft_gpu_interval")
                if interval and interval["host_start_ns"] < row["measurement_start_ns"]:
                    raise RuntimeError("initial Draft proposal preceded measurement boundary")
            return [results[r["request_id"]] for r in rows]
        if operation in ("commit_and_propose", "finish_tail"):
            return self._commit_many(rows, target_tail=operation == "finish_tail")
        raise ValueError("unsupported batched Dual operation")

    def _propose_many(self, rows):
        plans, results = [], {}
        for row in rows:
            rid = row["request_id"]
            state = self._state(rid)
            if state.finished or state.proposal is not None:
                raise ValueError("request cannot create another live proposal")
            remaining = row.get("remaining_output_budget")
            if type(remaining) is not int or remaining < 1:
                raise ValueError("remaining output budget must be positive")
            budget = min(self.candidate_budget, remaining - 1)
            if budget:
                plans.append(
                    DraftProposalPlan(
                        rid,
                        state.next_round_id,
                        state.committed_token_ids,
                        budget,
                        tuple(row.get("eos_token_ids", ())),
                    )
                )
            else:
                results[rid] = {
                    "request_id": rid,
                    "target_tail": True,
                    "proposal": None,
                    "target_tail_ready_ns": time.monotonic_ns(),
                }
        if not plans:
            return results
        # Preserve Dual's synchronized CUDA proposal interval, once per real cohort.
        # The backend's existing per-forward metrics remain the GPU work counter.
        torch = getattr(self.backend.worker, "torch", None)
        if torch is not None:
            torch.cuda.synchronize()
            self.backend.metrics.syncs["dual_interval_start"] += 1
        start = None if torch is None else torch.cuda.Event(enable_timing=True)
        end = None if torch is None else torch.cuda.Event(enable_timing=True)
        started = time.monotonic_ns()
        if start is not None:
            start.record()
        before = self.backend.metrics.forwards["proposal"]
        tokens = self.backend.propose_many(plans)
        if end is not None:
            end.record()
            torch.cuda.synchronize()
            self.backend.metrics.syncs["dual_interval_end"] += 1
        ended = time.monotonic_ns()
        forwards = self.backend.metrics.forwards["proposal"] - before
        elapsed = None if start is None else int(start.elapsed_time(end) * 1_000_000)
        cohort = self.cohort_id
        self.cohort_id += 1
        for plan in plans:
            state = self._state(plan.request_id)
            proposal_tokens = tokens[plan.request_id]
            proposal = DualProposal(
                request_id=plan.request_id,
                round_id=state.next_round_id,
                proposal_id=proposal_identity(
                    plan.request_id, state.next_round_id, state.prefix_version, proposal_tokens
                ),
                prefix_version=state.prefix_version,
                prefix_token_count=len(state.committed_token_ids),
                prefix_token_sha256=token_prefix_hash(state.committed_token_ids),
                draft_kv_length_before=len(state.committed_token_ids),
                draft_kv_length_after=len(state.committed_token_ids) + len(proposal_tokens),
                proposal_token_ids=proposal_tokens,
                created_timestamp_ns=ended,
                draft_start_ns=started,
                draft_end_ns=ended,
            )
            state.proposal = proposal
            results[plan.request_id] = {
                "request_id": plan.request_id,
                "target_tail": False,
                "proposal": proposal.to_dict(),
                "draft_cohort_id": cohort,
                "draft_cohort_request_ids": [p.request_id for p in plans],
                "logical_draft_kv_length": len(state.committed_token_ids),
                "draft_gpu_interval": {
                    "physical_gpu_id": self.backend.provenance.get("physical_gpu_id"),
                    "host_start_ns": started,
                    "host_end_ns": ended,
                    "cuda_elapsed_ns": elapsed,
                    "cuda_events": start is not None,
                    "cuda_synchronized": True,
                    "number_of_model_forwards": forwards,
                    "shared_cohort_interval": True,
                },
            }
        return results

    def _commit_many(self, rows, *, target_tail):
        plans, decisions = [], {}
        for row in rows:
            rid = row["request_id"]
            state = self._state(rid)
            proposal = state.proposal
            terminal = bool(row.get("terminal", False))
            delta = tuple(row.get("committed_delta", ()))
            if state.finished or row.get("prefix_version") != state.prefix_version + 1:
                raise ValueError("Dual commit finished request or non-monotonic prefix version")
            if target_tail:
                if proposal is not None or not terminal or len(delta) != 1:
                    raise ValueError("Target tail must be one terminal token without a proposal")
                accepted, proposal_tokens, target_tokens = 0, (), delta
                correction, bonus = 1, 0
            else:
                if (
                    proposal is None
                    or row.get("proposal_id") != proposal.proposal_id
                    or row.get("round_id") != proposal.round_id
                ):
                    raise ValueError("Dual Draft commit proposal/round identity mismatch")
                decision = dual_greedy_acceptance(
                    proposal.proposal_token_ids, delta, terminal=terminal
                )
                decisions[rid] = decision
                accepted = len(decision.accepted_draft_token_ids)
                proposal_tokens = proposal.proposal_token_ids
                target_tokens = (
                    decision.target_correction_token_ids + decision.target_bonus_token_ids
                )
                correction = len(decision.target_correction_token_ids)
                bonus = len(decision.target_bonus_token_ids)
            plans.append(
                DraftCommitPlan(
                    rid,
                    state.next_round_id,
                    state.committed_token_ids,
                    proposal_tokens,
                    accepted,
                    target_tokens,
                    row.get("prefix_token_sha256", ""),
                    terminal,
                    correction,
                    bonus,
                )
            )
        # Full cohort validation above precedes any backend mutation.
        physical = self.backend.commit_many(plans)
        sync_complete = time.monotonic_ns()
        results = {}
        for plan, row in zip(plans, rows):
            state = self._state(plan.request_id)
            old_proposal = state.proposal
            state.committed_token_ids = plan.final_prefix
            state.prefix_version = row["prefix_version"]
            if not target_tail:
                state.next_round_id += 1
            state.proposal = None
            state.finished = plan.terminal
            result = {
                "request_id": plan.request_id,
                "proposal": None,
                "committed_token_ids": list(plan.final_prefix[len(plan.parent_prefix) :]),
                "logical_draft_kv_length": len(plan.final_prefix),
                "prefix_version": state.prefix_version,
                "prefix_token_sha256": plan.final_prefix_hash,
                "draft_sync_complete_ns": sync_complete,
                "terminal": plan.terminal,
                "target_tail": target_tail,
                **physical[plan.request_id],
            }
            if old_proposal is not None:
                decision = decisions[plan.request_id]
                result.update(
                    round_id=old_proposal.round_id,
                    proposal_id=old_proposal.proposal_id,
                    accepted_draft_token_ids=list(decision.accepted_draft_token_ids),
                    rejected_draft_token_ids=list(decision.rejected_draft_token_ids),
                    target_correction_token_ids=list(decision.target_correction_token_ids),
                    target_bonus_token_ids=list(decision.target_bonus_token_ids),
                    accepted_draft_tokens=plan.accepted,
                    rejected_draft_tokens=len(plan.proposal) - plan.accepted,
                    rollback_length=len(plan.proposal) - plan.accepted,
                    correction_length=plan.correction_count,
                    bonus_length=plan.bonus_count,
                )
            results[plan.request_id] = result
        active = [r for r in rows if not self._state(r["request_id"]).finished]
        for rid, proposed in self._propose_many(active).items():
            results[rid].update(
                {key: value for key, value in proposed.items() if key != "logical_draft_kv_length"}
            )
        return [results[r["request_id"]] for r in rows]

    def shutdown(self):
        result = super().shutdown()
        if self.report_path is not None:
            report = {
                **self.backend.report(),
                "dual_cohorts": [
                    {"operation": op, "request_count": size, "count": count}
                    for (op, size), count in sorted(self.cohorts.items())
                ],
                "cohort_policy": "one existing enqueue message; no accumulation",
            }
            write_immutable_report(self.report_path, report)
            result["draft_backend_report_sha256"] = sha256_file(self.report_path)
        return result


class BatchedDualDraftController(AsyncDualDraftController):
    """Same nonblocking protocol, with creation/commit/shutdown on one owner."""

    def __init__(self, factory, event_log):
        self._factory = factory
        self._startup = threading.Event()
        self._startup_error = None
        self._shutdown_result = None
        super().__init__(None, event_log)
        self._startup.wait()
        if self._startup_error is not None:
            raise RuntimeError(f"production Dual Draft startup failed: {self._startup_error}")

    def enqueue(self, operation, rows):
        if operation not in {
            "bootstrap_and_propose",
            "propose_only",
            "commit_and_propose",
            "finish_tail",
        }:
            raise ValueError("unsupported asynchronous Draft operation")
        request_ids = [str(r.get("request_id", "")) for r in rows]
        if not request_ids:
            raise ValueError("asynchronous Draft rows require request IDs")
        unique_ids(request_ids)
        with self._lock:
            if set(request_ids) & self._inflight:
                raise ValueError("request already has Draft work in flight")
            self._inflight.update(request_ids)
        self._work.put(_Work(operation, {"rows": tuple(dict(r) for r in rows)}))
        return {"enqueued": request_ids, "blocking_on_draft_gpu": False}

    def shutdown(self, *, failed=False):
        if self._shutdown_result is not None:
            return self._shutdown_result
        response = queue.Queue(maxsize=1)
        self._work.put(_Work("shutdown", {"failed": failed}, response))
        while True:
            try:
                success, result = response.get(timeout=0.1)
                break
            except queue.Empty:
                if not self._thread.is_alive():
                    raise RuntimeError(
                        "production Dual Draft owner stopped before shutdown"
                    ) from None
        self._thread.join()
        if not success:
            raise RuntimeError(str(result))
        self._shutdown_result = {**result, **self.status()}
        return self._shutdown_result

    def _run(self):
        try:
            self.machine = self._factory()
        except Exception as error:
            self._startup_error = error
            return
        finally:
            self._startup.set()
        while True:
            work = self._work.get()
            rows = [work.row] if work.operation == "initialize" else work.row.get("rows", ())
            started = time.monotonic_ns()
            try:
                if work.operation == "shutdown":
                    if work.row.get("failed"):
                        self.machine.backend._fail()
                    result = self.machine.shutdown()
                    work.response.put((True, result))
                    return
                results = (
                    [self.machine.initialize(work.row)]
                    if work.operation == "initialize"
                    else self.machine.execute_batch(work.operation, rows)
                )
                # Publish the original cohort in input order after its physical work completes.
                with self._lock:
                    for row, result in zip(rows, results):
                        rid = row["request_id"]
                        self._inflight.remove(rid)
                        self._claimed.pop(rid, None)
                        if result.get("proposal") is not None or (
                            result.get("target_tail") and not result.get("terminal")
                        ):
                            self._ready[rid] = result
                            self._ready_order.append(rid)
                for row, result in zip(rows, results):
                    self.event_log.append(
                        {
                            "schema_version": "specrhythm.phase4b-draft-work.v1",
                            "operation": work.operation,
                            "request_id": row["request_id"],
                            "start_ns": started,
                            "end_ns": time.monotonic_ns(),
                            "success": True,
                            "result": result,
                        }
                    )
                if work.response is not None:
                    work.response.put((True, results[0]))
            except Exception as error:
                self.machine.backend._fail()
                message = f"{type(error).__name__}: {error}"
                with self._lock:
                    for row in rows:
                        self._inflight.discard(row["request_id"])
                        self._failures[row["request_id"]] = message
                if work.response is not None:
                    work.response.put((False, message))
                for row in rows:
                    try:
                        self.event_log.append(
                            {
                                "schema_version": "specrhythm.phase4b-draft-work.v1",
                                "operation": work.operation,
                                "request_id": row["request_id"],
                                "start_ns": started,
                                "end_ns": time.monotonic_ns(),
                                "success": False,
                                "error": message,
                            }
                        )
                    except OSError:
                        # Keep the original failure in memory and the owner alive
                        # for cleanup even if its error checkpoint cannot be written.
                        break
                if work.operation == "shutdown":
                    return
            finally:
                self._work.task_done()


def serve_vllm_dual(config, *, socket_path, event_log_path, transport_log_path, ready_path):
    report_path = ready_path.with_name("draft-backend-report.json")
    startup_path = ready_path.with_name("draft-startup.json")
    if any(Path(p).exists() for p in (report_path, startup_path, ready_path, socket_path)):
        raise FileExistsError("production Dual artifacts must be fresh")

    def factory():
        backend = VllmBatchedDraftBackend(config)
        try:
            write_immutable_report(startup_path, backend.provenance)
            return BatchedDualDraftMachine(
                backend, candidate_budget=config.proposal_budget, report_path=report_path
            )
        except Exception:
            backend._fail()
            backend.shutdown()
            write_immutable_report(report_path, backend.report())
            raise

    controller = BatchedDualDraftController(factory, CheckpointJsonl(event_log_path))
    server = DualDraftUnixServer(
        socket_path,
        controller,
        ready_path=ready_path,
        transport_log=CheckpointJsonl(transport_log_path),
    )
    try:
        server.serve()
    finally:
        controller.shutdown(failed=server.running)
