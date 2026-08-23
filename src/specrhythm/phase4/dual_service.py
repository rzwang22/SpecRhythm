"""Asynchronous GPU-0 Draft service used by the Phase-4B adapter.

Socket operations only enqueue work or inspect completed proposals.  Model
forwards execute on one background worker, so a Target scheduler poll never
waits for Draft GPU work and per-request Draft KV mutations remain serialized.
"""

from __future__ import annotations

import os
import queue
import socket
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from specrhythm.phase4.config import Phase4Config
from specrhythm.phase4.draft_service import DraftBackend, HFPersistentDraftBackend
from specrhythm.phase4.dual import (
    DUAL_PROTOCOL_VERSION,
    DualProposal,
    proposal_identity,
)
from specrhythm.phase4.manifest import atomic_write_json
from specrhythm.phase4.serial import greedy_acceptance, token_prefix_hash
from specrhythm.phase4.transport import (
    CheckpointJsonl,
    receive_message,
    send_message,
)


@dataclass
class _DraftState:
    committed_token_ids: Tuple[int, ...]
    prefix_version: int
    next_round_id: int
    proposal: Optional[DualProposal] = None
    finished: bool = False


class DualDraftMachine:
    """Persistent Draft KV authority for linear Dual-Batch proposals."""

    def __init__(self, backend: DraftBackend, *, candidate_budget: int = 4) -> None:
        if not 1 <= candidate_budget <= 4:
            raise ValueError("candidate budget must be in [1, 4]")
        self.backend = backend
        self.candidate_budget = candidate_budget
        self.requests: dict[str, _DraftState] = {}

    def initialize(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Materialize the complete decode-ready prefix without proposing."""

        request_id = str(row.get("request_id", ""))
        tokens = tuple(row.get("committed_token_ids", ()))
        prefix_version = row.get("prefix_version", -1)
        if request_id in self.requests:
            raise ValueError("Dual Draft request was initialized more than once")
        if not request_id or not tokens or prefix_version < 1:
            raise ValueError("Dual Draft bootstrap evidence is incomplete")
        if token_prefix_hash(tokens) != row.get("prefix_token_sha256"):
            raise ValueError("Dual Draft bootstrap prefix hash mismatch")
        self.backend.initialize(request_id, tokens)
        self.requests[request_id] = _DraftState(tokens, prefix_version, 0)
        if bool(row.get("terminal", False)):
            self.requests[request_id].finished = True
            self.backend.finish(request_id)
        return {
            "request_id": request_id,
            "terminal": bool(row.get("terminal", False)),
            "proposal": None,
            "initial_proposal_generated": False,
            "logical_draft_kv_length": len(tokens),
            "committed_prefix_hash": token_prefix_hash(tokens),
            "prefix_version": prefix_version,
            "next_round_id": 0,
        }

    def bootstrap_and_propose(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Legacy B.0 operation retained for immutable artifact reproducibility."""

        initialized = self.initialize(row)
        if initialized["terminal"]:
            return initialized
        return self._propose(str(row["request_id"]), row)

    def propose_only(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Create a proposal only after the decode-ready measurement boundary."""

        request_id = str(row.get("request_id", ""))
        state = self._state(request_id)
        if row.get("prefix_version") != state.prefix_version:
            raise ValueError("proposal-only prefix version mismatch")
        if tuple(row.get("committed_token_ids", ())) != state.committed_token_ids:
            raise ValueError("proposal-only committed prefix mismatch")
        if token_prefix_hash(state.committed_token_ids) != row.get(
            "prefix_token_sha256"
        ):
            raise ValueError("proposal-only committed prefix hash mismatch")
        measurement_start_ns = row.get("measurement_start_ns")
        if not isinstance(measurement_start_ns, int) or measurement_start_ns <= 0:
            raise ValueError("proposal-only requires a positive measurement boundary")
        result = self._propose(request_id, row)
        interval = result.get("draft_gpu_interval")
        if (
            isinstance(interval, Mapping)
            and interval.get("host_start_ns", 0) < measurement_start_ns
        ):
            raise RuntimeError("initial Draft proposal started before measurement_start")
        return result

    def commit_and_propose(self, row: Mapping[str, Any]) -> dict[str, Any]:
        request_id = str(row.get("request_id", ""))
        state = self._state(request_id)
        proposal = state.proposal
        if state.finished or proposal is None:
            raise ValueError("Dual Draft commit has no live proposal")
        if row.get("proposal_id") != proposal.proposal_id:
            raise ValueError("Dual Draft commit proposal identity mismatch")
        if row.get("round_id") != proposal.round_id:
            raise ValueError("Dual Draft commit round mismatch")
        decision = greedy_acceptance(
            proposal.proposal_token_ids,
            row.get("committed_delta", ()),
            terminal=bool(row.get("terminal", False)),
        )
        final_prefix = state.committed_token_ids + decision.committed_token_ids
        new_version = row.get("prefix_version", -1)
        if new_version != state.prefix_version + 1:
            raise ValueError("committed prefix version is not monotonic")
        if token_prefix_hash(final_prefix) != row.get("prefix_token_sha256"):
            raise ValueError("Target/Draft committed prefix hash mismatch")
        self.backend.rollback(request_id, len(decision.accepted_draft_token_ids))
        target_tokens = (
            decision.target_correction_token_ids + decision.target_bonus_token_ids
        )
        if target_tokens:
            self.backend.append_target_token(request_id, target_tokens[0])
        state.committed_token_ids = final_prefix
        state.prefix_version = new_version
        state.next_round_id += 1
        state.proposal = None
        # The next proposal must begin only after correction/bonus has been
        # incorporated into the persistent Draft state.
        draft_sync_complete_ns = time.monotonic_ns()
        terminal = bool(row.get("terminal", False))
        if terminal:
            state.finished = True
            self.backend.finish(request_id)
            next_result: Mapping[str, Any] = {"proposal": None}
        else:
            next_result = self._propose(request_id, row)
        return {
            "request_id": request_id,
            "round_id": proposal.round_id,
            "proposal_id": proposal.proposal_id,
            "accepted_draft_token_ids": list(decision.accepted_draft_token_ids),
            "rejected_draft_token_ids": list(decision.rejected_draft_token_ids),
            "target_correction_token_ids": list(decision.target_correction_token_ids),
            "target_bonus_token_ids": list(decision.target_bonus_token_ids),
            "committed_token_ids": list(decision.committed_token_ids),
            "accepted_draft_tokens": len(decision.accepted_draft_token_ids),
            "rejected_draft_tokens": len(decision.rejected_draft_token_ids),
            "rollback_length": len(decision.rejected_draft_token_ids),
            "correction_length": len(decision.target_correction_token_ids),
            "bonus_length": len(decision.target_bonus_token_ids),
            "logical_draft_kv_length": len(final_prefix),
            "draft_physical_request_block_identity": None,
            "draft_physical_request_block_observable": False,
            "prefix_version": state.prefix_version,
            "prefix_token_sha256": token_prefix_hash(final_prefix),
            "draft_sync_complete_ns": draft_sync_complete_ns,
            "terminal": terminal,
            "proposal": next_result.get("proposal"),
            "draft_gpu_interval": next_result.get("draft_gpu_interval"),
            "target_tail": next_result.get("target_tail", False),
            "target_tail_ready_ns": next_result.get("target_tail_ready_ns"),
        }

    def finish_tail(self, row: Mapping[str, Any]) -> dict[str, Any]:
        request_id = str(row.get("request_id", ""))
        state = self._state(request_id)
        if state.finished or state.proposal is not None:
            raise ValueError("Target tail can only finish a proposal-free live request")
        delta = tuple(row.get("committed_delta", ()))
        if len(delta) != 1 or bool(row.get("terminal", False)) is not True:
            raise ValueError("Target tail must commit one terminal token")
        self.backend.append_target_token(request_id, delta[0])
        state.committed_token_ids += delta
        new_version = row.get("prefix_version", -1)
        if new_version != state.prefix_version + 1:
            raise ValueError("Target-tail prefix version is not monotonic")
        if token_prefix_hash(state.committed_token_ids) != row.get("prefix_token_sha256"):
            raise ValueError("Target-tail committed prefix hash mismatch")
        state.prefix_version = new_version
        state.finished = True
        self.backend.finish(request_id)
        return {
            "request_id": request_id,
            "target_tail": True,
            "committed_token_ids": list(delta),
            "logical_draft_kv_length": len(state.committed_token_ids),
            "prefix_version": state.prefix_version,
            "terminal": True,
            "proposal": None,
        }

    def _propose(self, request_id: str, row: Mapping[str, Any]) -> dict[str, Any]:
        state = self._state(request_id)
        if state.finished or state.proposal is not None:
            raise ValueError("request cannot create another live proposal")
        remaining = row.get("remaining_output_budget", -1)
        if not isinstance(remaining, int) or isinstance(remaining, bool) or remaining < 1:
            raise ValueError("remaining output budget must be positive")
        budget = min(self.candidate_budget, max(remaining - 1, 0))
        if budget == 0:
            return {
                "request_id": request_id,
                "target_tail": True,
                "target_tail_ready_ns": time.monotonic_ns(),
                "proposal": None,
            }
        self._cuda_synchronize()
        start_event = self._cuda_event()
        end_event = self._cuda_event()
        started = time.monotonic_ns()
        if start_event is not None:
            start_event.record()
        torch = getattr(self.backend, "torch", None)
        nvtx = (
            torch.cuda.nvtx.range(f"specrhythm:draft:{request_id}")
            if torch is not None
            else nullcontext()
        )
        with nvtx:
            tokens, model_forwards = self.backend.propose(
                request_id, budget, tuple(row.get("eos_token_ids", ()))
            )
        if end_event is not None:
            end_event.record()
        self._cuda_synchronize()
        ended = time.monotonic_ns()
        proposal_tokens = tuple(tokens)
        proposal = DualProposal(
            request_id=request_id,
            round_id=state.next_round_id,
            proposal_id=proposal_identity(
                request_id, state.next_round_id, state.prefix_version, proposal_tokens
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
        cuda_elapsed_ns = None
        if start_event is not None and end_event is not None:
            cuda_elapsed_ns = int(start_event.elapsed_time(end_event) * 1_000_000)
        return {
            "request_id": request_id,
            "target_tail": False,
            "proposal": proposal.to_dict(),
            "draft_gpu_interval": {
                "physical_gpu_id": self.backend.provenance.get("physical_gpu_id"),
                "host_start_ns": started,
                "host_end_ns": ended,
                "cuda_elapsed_ns": cuda_elapsed_ns,
                "cuda_events": start_event is not None,
                "cuda_synchronized": True,
                "number_of_model_forwards": model_forwards,
            },
            "logical_draft_kv_length": len(state.committed_token_ids),
            "draft_physical_request_block_identity": None,
            "draft_physical_request_block_observable": False,
        }

    def shutdown(self) -> dict[str, Any]:
        self.backend.shutdown()
        return {"request_count": len(self.requests), "shutdown": True}

    def _state(self, request_id: str) -> _DraftState:
        try:
            return self.requests[request_id]
        except KeyError as error:
            raise ValueError(f"unknown Dual Draft request: {request_id}") from error

    def _cuda_synchronize(self) -> None:
        torch = getattr(self.backend, "torch", None)
        if torch is not None:
            torch.cuda.synchronize()

    def _cuda_event(self) -> Any:
        torch = getattr(self.backend, "torch", None)
        return None if torch is None else torch.cuda.Event(enable_timing=True)


@dataclass(frozen=True)
class _Work:
    operation: str
    row: Mapping[str, Any]
    response: Optional[queue.Queue[tuple[bool, Any]]] = None


class AsyncDualDraftController:
    """One model-worker queue with immediate, lock-bounded control calls."""

    def __init__(self, machine: DualDraftMachine, event_log: CheckpointJsonl) -> None:
        self.machine = machine
        self.event_log = event_log
        self._work: queue.Queue[Optional[_Work]] = queue.Queue()
        self._lock = threading.Lock()
        self._ready: dict[str, dict[str, Any]] = {}
        self._ready_order: list[str] = []
        self._claimed: dict[str, dict[str, Any]] = {}
        self._inflight: set[str] = set()
        self._failures: dict[str, str] = {}
        self._thread = threading.Thread(target=self._run, name="dual-draft-gpu", daemon=True)
        self._thread.start()

    def enqueue(self, operation: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if operation not in {
            "bootstrap_and_propose",
            "propose_only",
            "commit_and_propose",
            "finish_tail",
        }:
            raise ValueError("unsupported asynchronous Draft operation")
        request_ids = [str(row.get("request_id", "")) for row in rows]
        if not request_ids or any(not item for item in request_ids):
            raise ValueError("asynchronous Draft rows require request IDs")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("asynchronous Draft request IDs must be unique")
        with self._lock:
            conflicts = [item for item in request_ids if item in self._inflight]
            if conflicts:
                raise ValueError(f"request already has Draft work in flight: {conflicts}")
            self._inflight.update(request_ids)
        for row in rows:
            self._work.put(_Work(operation, dict(row)))
        return {"enqueued": request_ids, "blocking_on_draft_gpu": False}

    def execute(
        self, operation: str, row: Mapping[str, Any], *, timeout_seconds: float = 120.0
    ) -> dict[str, Any]:
        """Run setup-only work on the Draft worker and return its evidence.

        This blocking operation is permitted only for pre-measurement resident
        initialization. Timed proposal work continues to use ``enqueue``.
        """

        if operation != "initialize":
            raise ValueError("synchronous Dual Draft execution is setup-only")
        request_id = str(row.get("request_id", ""))
        if not request_id:
            raise ValueError("synchronous Draft setup requires a request ID")
        response: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)
        with self._lock:
            if request_id in self._inflight:
                raise ValueError(f"request already has Draft work in flight: {request_id}")
            self._inflight.add(request_id)
        self._work.put(_Work(operation, dict(row), response))
        try:
            success, value = response.get(timeout=timeout_seconds)
        except queue.Empty as error:
            raise RuntimeError("timed out waiting for resident Draft initialization") from error
        if not success:
            raise RuntimeError(str(value))
        if not isinstance(value, dict):
            raise RuntimeError("resident Draft initialization returned invalid evidence")
        return value

    def poll_ready(self, limit: int) -> dict[str, Any]:
        if limit < 1:
            raise ValueError("poll limit must be positive")
        with self._lock:
            selected_ids = self._ready_order[:limit]
            del self._ready_order[: len(selected_ids)]
            selected = []
            for request_id in selected_ids:
                value = self._ready.pop(request_id)
                self._claimed[request_id] = value
                selected.append(value)
            return {
                "ready": selected,
                "pending_request_ids": sorted(self._inflight),
                "failures": dict(self._failures),
                "blocking_on_draft_gpu": False,
            }

    def claimed(self, request_ids: Sequence[str]) -> dict[str, Any]:
        with self._lock:
            values = []
            for request_id in request_ids:
                value = self._claimed.get(request_id)
                if value is None:
                    raise ValueError(f"request has no claimed proposal: {request_id}")
                values.append(value)
            return {"claimed": values}

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ready_request_ids": list(self._ready_order),
                "claimed_request_ids": sorted(self._claimed),
                "inflight_request_ids": sorted(self._inflight),
                "failures": dict(self._failures),
                "work_queue_depth": self._work.qsize(),
            }

    def shutdown(self) -> dict[str, Any]:
        self._work.join()
        self._work.put(None)
        self._thread.join(timeout=30)
        if self._thread.is_alive():
            raise RuntimeError("Dual Draft worker did not terminate")
        return {**self.machine.shutdown(), **self.status()}

    def _run(self) -> None:
        while True:
            work = self._work.get()
            try:
                if work is None:
                    return
                started = time.monotonic_ns()
                try:
                    method = getattr(self.machine, work.operation)
                    result = method(work.row)
                    request_id = str(work.row["request_id"])
                    with self._lock:
                        self._inflight.remove(request_id)
                        self._claimed.pop(request_id, None)
                        if result.get("proposal") is not None or (
                            result.get("target_tail") and not result.get("terminal")
                        ):
                            self._ready[request_id] = result
                            self._ready_order.append(request_id)
                    if work.response is not None:
                        work.response.put((True, result))
                    self.event_log.append(
                        {
                            "schema_version": "specrhythm.phase4b-draft-work.v1",
                            "operation": work.operation,
                            "request_id": request_id,
                            "start_ns": started,
                            "end_ns": time.monotonic_ns(),
                            "success": True,
                            "result": result,
                        }
                    )
                except Exception as error:
                    request_id = str(work.row.get("request_id", ""))
                    message = f"{type(error).__name__}: {error}"
                    with self._lock:
                        self._inflight.discard(request_id)
                        self._failures[request_id] = message
                    if work.response is not None:
                        work.response.put((False, message))
                    self.event_log.append(
                        {
                            "schema_version": "specrhythm.phase4b-draft-work.v1",
                            "operation": work.operation,
                            "request_id": request_id,
                            "start_ns": started,
                            "end_ns": time.monotonic_ns(),
                            "success": False,
                            "error": message,
                        }
                    )
            finally:
                self._work.task_done()


class DualDraftClient:
    def __init__(self, socket_path: Path, *, timeout_seconds: float = 30.0) -> None:
        self.socket_path = socket_path
        self.timeout_seconds = timeout_seconds

    def call(self, operation: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        message = {
            "protocol_version": DUAL_PROTOCOL_VERSION,
            "operation": operation,
            "payload": dict(payload),
        }
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout_seconds)
            connection.connect(str(self.socket_path))
            send_message(connection, message)
            response, _ = receive_message(connection)
        if response.get("protocol_version") != DUAL_PROTOCOL_VERSION:
            raise RuntimeError("Dual Draft service returned an incompatible protocol")
        if response.get("ok") is not True:
            raise RuntimeError(str(response.get("error", "Dual Draft service failed")))
        result = response.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("Dual Draft service result is not an object")
        return result


class DualDraftUnixServer:
    def __init__(
        self,
        socket_path: Path,
        controller: AsyncDualDraftController,
        *,
        ready_path: Path,
        transport_log: CheckpointJsonl,
    ) -> None:
        self.socket_path = socket_path
        self.controller = controller
        self.ready_path = ready_path
        self.transport_log = transport_log
        self.running = True

    def serve(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            raise FileExistsError(
                f"refusing to replace an existing Dual Draft socket {self.socket_path}"
            )
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o600)
            server.listen(32)
            atomic_write_json(
                self.ready_path,
                {
                    "schema_version": "specrhythm.phase4b-dual-draft-ready.v1",
                    "protocol_version": DUAL_PROTOCOL_VERSION,
                    "socket_file": self.socket_path.name,
                    "asynchronous_gpu_worker": True,
                    "scheduler_poll_blocks_on_gpu": False,
                    "provenance": dict(self.controller.machine.backend.provenance),
                },
            )
            while self.running:
                connection, _ = server.accept()
                with connection:
                    self._handle(connection)
        self.socket_path.unlink(missing_ok=True)

    def _handle(self, connection: socket.socket) -> None:
        started = time.monotonic_ns()
        operation = ""
        try:
            request, request_bytes = receive_message(connection)
            if request.get("protocol_version") != DUAL_PROTOCOL_VERSION:
                raise ValueError("incompatible Dual Draft protocol")
            operation = str(request.get("operation", ""))
            payload = request.get("payload")
            if not isinstance(payload, Mapping):
                raise ValueError("Dual Draft payload must be an object")
            if operation == "enqueue":
                result = self.controller.enqueue(
                    str(payload.get("work_operation", "")), payload.get("rows", ())
                )
            elif operation == "execute":
                row = payload.get("row")
                if not isinstance(row, Mapping):
                    raise ValueError("synchronous Draft execution row must be an object")
                result = self.controller.execute(
                    str(payload.get("work_operation", "")), row
                )
            elif operation == "poll_ready":
                result = self.controller.poll_ready(payload.get("limit", 1))
            elif operation == "claimed":
                result = self.controller.claimed(payload.get("request_ids", ()))
            elif operation == "status":
                result = self.controller.status()
            elif operation == "shutdown":
                result = self.controller.shutdown()
                self.running = False
            else:
                raise ValueError(f"unsupported Dual Draft operation: {operation}")
            response = {
                "protocol_version": DUAL_PROTOCOL_VERSION,
                "ok": True,
                "result": result,
            }
        except Exception as error:
            request_bytes = 0
            response = {
                "protocol_version": DUAL_PROTOCOL_VERSION,
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
                "result": {},
            }
        response_bytes = send_message(connection, response)
        self.transport_log.append(
            {
                "schema_version": "specrhythm.phase4b-transport-event.v1",
                "transport": "unix-domain-socket",
                "serialization": "length-prefixed-canonical-json",
                "operation": operation,
                "start_ns": started,
                "end_ns": time.monotonic_ns(),
                "request_payload_bytes": request_bytes,
                "response_payload_bytes": response_bytes,
                "loopback_local_only": True,
                "host_staging": True,
                "gpu_kernel_time": False,
                "blocks_on_draft_gpu": operation == "execute",
                "measurement_region": operation != "execute",
                "success": response["ok"],
            }
        )


def run_dual_draft_service(
    config: Phase4Config,
    *,
    socket_path: Path,
    event_log_path: Path,
    transport_log_path: Path,
    ready_path: Path,
) -> None:
    backend = HFPersistentDraftBackend(config)
    machine = DualDraftMachine(backend, candidate_budget=config.proposal_budget)
    controller = AsyncDualDraftController(machine, CheckpointJsonl(event_log_path))
    DualDraftUnixServer(
        socket_path,
        controller,
        ready_path=ready_path,
        transport_log=CheckpointJsonl(transport_log_path),
    ).serve()
