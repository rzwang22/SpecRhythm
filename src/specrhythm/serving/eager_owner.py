"""One model owner with FIFO messages and preemptible, fenced token steps.

The socket thread only copies/enqueues immutable JSON messages. It never mutates
protocol state or GPU KV. Feedback already queued is handled before another token
step; a step already launched must return its physical fence first.
"""

from __future__ import annotations

import copy
import queue
import threading
import time
from types import SimpleNamespace

from specrhythm.continuation.trace import TRACE, references
from specrhythm.serving.fixed_settle import remaining


class EagerOwner:
    def __init__(self, factory, *, timeout_seconds=900):
        self.commands = queue.Queue()
        self.ready = queue.Queue(maxsize=1)
        self.timeout_seconds = timeout_seconds
        self.failure = None
        self.closed = False
        self.machine = None
        self._thread = threading.Thread(
            target=self._run, args=(factory,), name="serial-eager-owner"
        )
        self._thread.start()
        ok, value = self.ready.get(timeout=timeout_seconds)
        if not ok:
            self._thread.join(timeout=1)
            raise RuntimeError(str(value))
        self.backend = SimpleNamespace(**value)

    def _run(self, factory):
        waiting = []
        try:
            self.machine = machine = factory()
            self.ready.put(
                (
                    True,
                    {
                        "backend_name": machine.backend.backend_name,
                        "provenance": dict(machine.backend.provenance),
                    },
                )
            )
            while not self.closed:
                # Drain queued control messages before any next speculative token.
                try:
                    command = self.commands.get(timeout=0 if self._has_work() or waiting else 0.1)
                except queue.Empty:
                    command = None
                while command is not None:
                    operation, payload, response = command
                    causal = payload.pop("_causal", {})
                    TRACE.event("owner_dequeue", operation=operation, **causal)
                    if TRACE.enabled:
                        payload["_causal_context"] = causal
                    try:
                        with TRACE.span("owner_dispatch", operation=operation, **causal):
                            if operation == "eager_enqueue":
                                machine.verify_start(payload["requests"])
                                value = {"enqueued": True}
                            elif operation == "synchronize_and_batch_propose":
                                started = payload.pop("_owner_enqueue_ns")
                                machine.prepare_synchronizations(
                                    payload.get("synchronizations", ()))
                                pending = [
                                    w.work_id
                                    for rid, w in machine.works.items()
                                    if machine.core._state(rid).continuations[w.work_id].status
                                    == "waiting_draft"
                                ]
                                if TRACE.enabled:
                                    payload["_causal_wait_start"] = time.monotonic_ns()
                                    blocked_ids = {rid for rid, w in machine.works.items()
                                                   if w.work_id in pending}
                                    payload["_causal_ready_ids"] = [r["request_id"]
                                        for r in payload.get("synchronizations", ())
                                        if r["request_id"] not in blocked_ids]
                                waiting.append((payload, response, started, pending))
                                response = None
                                value = None
                            else:
                                value = self._dispatch(operation, payload)
                            if response is not None:
                                response.put((True, value))
                    except Exception as error:
                        if response is None:
                            raise
                        response.put((False, error))
                        # A failed immutable message does not mutate unrelated work.
                        # Device failures, however, make the owner unrecoverable.
                        if machine.backend.failed:
                            raise
                    if self.closed:
                        break
                    try:
                        command = self.commands.get_nowait()
                    except queue.Empty:
                        command = None
                for item in list(waiting):
                    payload, response, started, pending = item
                    try:
                        gate_end = time.monotonic_ns()
                        sync = machine.finish_synchronizations(payload.get("synchronizations", ()))
                        if sync is None:
                            continue
                        ended = time.monotonic_ns()
                        if TRACE.enabled and pending:
                            TRACE.event("owner_batch_gate",
                                        start_ns=payload["_causal_wait_start"],
                                        end_ns=gate_end, dependency_work_ids=pending,
                                        ready_request_ids=payload.get("_causal_ready_ids", []),
                                        batch_gate=True, **payload.get("_causal_context", {}))
                        proposals = machine.batch_propose(payload.get("proposals", ()))
                        if pending:
                            wait_end = max(machine.work_times[wid][1] for wid in pending)
                            wait_end = max(started, wait_end)
                            machine._event(
                                "unhidden_wait",
                                start_ns=started,
                                end_ns=wait_end,
                                unhidden_wait_ns=wait_end - started,
                            )
                        TRACE.event("owner_candidates_return",
                                    **payload.get("_causal_context", {}))
                        response.put(
                            (
                                True,
                                {
                                    "synchronizations": sync,
                                    "state_sync_start_ns": started,
                                    "state_sync_end_ns": ended,
                                    **proposals,
                                },
                            )
                        )
                    except Exception as error:
                        # Retire this waiter before escalating a device failure.
                        # The outer failure handler notifies remaining waiters; a
                        # second put here would block on an abandoned size-one
                        # response queue and prevent physical backend shutdown.
                        waiting.remove(item)
                        response.put((False, error))
                        if machine.backend.failed:
                            raise
                        continue
                    waiting.remove(item)
                if not self.closed and self._has_work():
                    machine.step()
            machine.owner_stopped = True
        except BaseException as error:
            self.failure = error
            if self.machine is None:
                self.ready.put((False, error))
            for _, response, _, _ in waiting:
                response.put((False, error))
            while not self.commands.empty():
                _, _, response = self.commands.get_nowait()
                if response is not None:
                    response.put((False, error))
            if self.machine is not None:
                try:
                    self.machine.backend._fail()
                    self.machine.backend.shutdown()
                finally:
                    self.machine.owner_stopped = True

    def _has_work(self):
        return self.machine is not None and any(
            not self.machine.core._state(rid).continuations[work.work_id].completed
            for rid, work in self.machine.works.items()
        )

    def _dispatch(self, operation, payload):
        machine = self.machine
        if operation == "initialize":
            return machine.initialize(
                payload["request_id"],
                payload["committed_token_ids"],
                payload["committed_prefix_hash"],
            )
        if operation == "batch_propose":
            return machine.batch_propose(payload["requests"])
        if operation == "finish_request":
            return machine.finish_authoritative(payload)
        if operation == "cancel_request":
            raise ValueError("eager cancellation requires authoritative diagnostic_settle")
        if operation == "diagnostic_settle":
            return machine.diagnostic_settle(payload)
        if operation == "eager_switch":
            return machine.set_eager(payload["enabled"], payload["decision_version"])
        if operation == "status":
            return {
                "inflight_request_ids": [
                    rid
                    for rid, work in machine.works.items()
                    if not machine.core._state(rid).continuations[work.work_id].completed
                ],
                "failures": [],
                "pending_work": list(machine.works),
            }
        if operation == "shutdown":
            result = machine.shutdown(payload.get("deadline_ns"))
            self.closed = True
            return result
        raise ValueError(f"unsupported eager operation: {operation}")

    def call(self, operation, payload):
        if self.failure is not None:
            raise RuntimeError(f"eager owner failed: {self.failure}")
        if self.closed:
            raise RuntimeError("eager owner is closed")
        rows = payload.get("requests", payload.get("synchronizations", ()))
        refs = references(rows) if TRACE.enabled else []
        with TRACE.span("owner_payload_copy", operation=operation, requests=refs):
            frozen = copy.deepcopy(payload)
        if TRACE.enabled:
            frozen["_causal"] = {"queue_submit_ns": time.monotonic_ns(), "requests": refs,
                                 "batch_id": frozen.pop("_causal_batch_id", None)}
        TRACE.event("owner_queue_submit", operation=operation, requests=refs,
                    queue_submit_ns=frozen.get("_causal", {}).get("queue_submit_ns"))
        if operation == "eager_enqueue":
            # Confirm task registration in the mailbox only. In particular, do
            # not wait for the owner thread, model forward, sampling or a fence.
            with TRACE.span("owner_queue_put", operation=operation, requests=refs):
                self.commands.put((operation, frozen, None))
            TRACE.event("owner_enqueue_ack", operation=operation, requests=refs)
            return {
                "enqueued": True,
                "enqueue_ns": time.monotonic_ns(),
                "request_ids": [r["request_id"] for r in frozen["requests"]],
            }
        response = queue.Queue(maxsize=1)
        if operation == "synchronize_and_batch_propose":
            frozen["_owner_enqueue_ns"] = time.monotonic_ns()
        with TRACE.span("owner_queue_put", operation=operation, requests=refs):
            self.commands.put((operation, frozen, response))
        timeout = (
            remaining(payload["deadline_ns"]) if "deadline_ns" in payload else self.timeout_seconds
        )
        try:
            with TRACE.span("owner_response_wait", operation=operation, requests=refs):
                ok, value = response.get(timeout=timeout)
        except queue.Empty as error:
            raise TimeoutError(
                "eager owner response deadline expired; physical work retained"
            ) from error
        if not ok:
            raise RuntimeError(str(value))
        if operation == "shutdown":
            self._thread.join(
                timeout=remaining(payload["deadline_ns"])
                if "deadline_ns" in payload
                else self.timeout_seconds
            )
            if self._thread.is_alive():
                raise TimeoutError("eager owner remains alive after shutdown deadline")
        return value

    def close(self, deadline_ns):
        """Bounded failure join; never clear an in-flight GPU ledger to claim exit."""
        if not self.closed and self.failure is None:
            self.call("shutdown", {"deadline_ns": deadline_ns})
        self._thread.join(timeout=remaining(deadline_ns))
        if self._thread.is_alive():
            raise TimeoutError("eager owner has not physically exited")
