"""Persistent GPU-0 Draft service for Phase-4A.1 correctness runs."""

from __future__ import annotations

import os
import socket
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple

from specrhythm.phase4.config import Phase4Config
from specrhythm.phase4.manifest import atomic_write_json, model_revision_manifest
from specrhythm.phase4.serial import (
    PROTOCOL_VERSION,
    AcceptanceDecision,
    Proposal,
    greedy_acceptance,
    token_prefix_hash,
)
from specrhythm.phase4.transport import (
    CheckpointJsonl,
    canonical_json_bytes,
    receive_message,
    send_message,
)


class DraftBackend(Protocol):
    backend_name: str

    @property
    def provenance(self) -> Mapping[str, Any]: ...

    def initialize(self, request_id: str, committed_token_ids: Sequence[int]) -> None: ...

    def propose(
        self, request_id: str, budget: int, eos_token_ids: Sequence[int]
    ) -> tuple[Tuple[int, ...], int]: ...

    def rollback(self, request_id: str, accepted_draft_tokens: int) -> None: ...

    def append_target_token(self, request_id: str, token_id: int) -> None: ...

    def finish(self, request_id: str) -> None: ...

    def shutdown(self) -> None: ...


@dataclass
class DraftRequest:
    request_id: str
    committed_token_ids: Tuple[int, ...]
    next_round_id: int = 0
    pending_proposal: Optional[Proposal] = None
    pending_decision: Optional[AcceptanceDecision] = None
    rollback_applied: bool = False
    finished: bool = False
    initialization_count: int = 1
    proposal_count: int = 0


class DraftStateMachine:
    """Validate every Draft lifecycle transition before touching model KV."""

    def __init__(self, backend: DraftBackend, *, candidate_budget: int = 4) -> None:
        if not 1 <= candidate_budget <= 4:
            raise ValueError("candidate budget must be in [1, 4]")
        self.backend = backend
        self.candidate_budget = candidate_budget
        self.requests: dict[str, DraftRequest] = {}

    def initialize(
        self, request_id: str, committed_token_ids: Sequence[int], prefix_hash: str
    ) -> dict[str, Any]:
        tokens = tuple(committed_token_ids)
        if not request_id or not tokens:
            raise ValueError("Draft initialization requires request identity and prefix")
        if request_id in self.requests:
            raise ValueError("Draft request was initialized more than once")
        if token_prefix_hash(tokens) != prefix_hash:
            raise ValueError("Draft initialization prefix hash mismatch")
        self.backend.initialize(request_id, tokens)
        self.requests[request_id] = DraftRequest(request_id, tokens)
        return {
            "request_id": request_id,
            "committed_prefix_len": len(tokens),
            "committed_prefix_hash": prefix_hash,
            "initialization_count": 1,
        }

    def batch_propose(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if len({str(row.get("request_id", "")) for row in rows}) != len(rows):
            raise ValueError("batch_propose request IDs must be unique")
        proposals = []
        batch_started = time.monotonic_ns()
        for row in rows:
            request_id = str(row.get("request_id", ""))
            state = self._state(request_id)
            if state.finished:
                raise ValueError("finished request cannot produce another proposal")
            if state.pending_proposal is not None:
                raise ValueError("request already has an unverified proposal")
            round_id = row.get("round_id")
            if round_id != state.next_round_id:
                raise ValueError("stale, duplicate, or out-of-order Draft round")
            if row.get("committed_prefix_len") != len(state.committed_token_ids):
                raise ValueError("Draft proposal parent prefix length mismatch")
            prefix_hash = token_prefix_hash(state.committed_token_ids)
            if row.get("committed_prefix_hash") != prefix_hash:
                raise ValueError("Draft proposal parent prefix hash mismatch")
            remaining = row.get("remaining_output_budget")
            if not isinstance(remaining, int) or isinstance(remaining, bool) or remaining < 1:
                raise ValueError("remaining output budget must be positive")
            budget = min(self.candidate_budget, max(remaining - 1, 0))
            if budget == 0:
                continue
            eos = tuple(row.get("eos_token_ids", ()))
            started = time.monotonic_ns()
            tokens, model_forwards = self.backend.propose(request_id, budget, eos)
            ended = time.monotonic_ns()
            proposal = Proposal(
                protocol_version=PROTOCOL_VERSION,
                request_id=request_id,
                round_id=state.next_round_id,
                parent_prefix_len=len(state.committed_token_ids),
                parent_prefix_hash=prefix_hash,
                proposal_token_ids=tokens,
                proposal_eos=bool(tokens and tokens[-1] in eos),
                draft_start_ns=started,
                draft_end_ns=ended,
                transport_payload_bytes=0,
                model_provenance=self.backend.provenance,
                runtime_provenance={
                    "backend": self.backend.backend_name,
                    "number_of_model_forwards": model_forwards,
                    "full_context_replay": False,
                    "persistent_cross_round_kv": True,
                },
            )
            for _ in range(3):
                encoded_size = len(canonical_json_bytes(proposal.to_dict()))
                if proposal.transport_payload_bytes == encoded_size:
                    break
                proposal = replace(proposal, transport_payload_bytes=encoded_size)
            state.pending_proposal = proposal
            state.proposal_count += 1
            proposals.append(proposal.to_dict())
        return {
            "proposals": proposals,
            "draft_batch_start_ns": batch_started,
            "draft_batch_end_ns": time.monotonic_ns(),
            "draft_microbatch_count": len(rows),
            "single_persistent_model": True,
        }

    def synchronize_committed_prefix(
        self,
        request_id: str,
        round_id: int,
        committed_delta: Sequence[int],
        committed_prefix_hash: str,
        *,
        terminal: bool,
    ) -> dict[str, Any]:
        state = self._pending(request_id, round_id)
        proposal = state.pending_proposal
        assert proposal is not None
        decision = greedy_acceptance(
            proposal.proposal_token_ids, committed_delta, terminal=terminal
        )
        final_prefix = state.committed_token_ids + decision.committed_token_ids
        if token_prefix_hash(final_prefix) != committed_prefix_hash:
            raise ValueError("Target/Draft committed prefix hash mismatch")
        state.pending_decision = decision
        return {
            "request_id": request_id,
            "round_id": round_id,
            "decision": {
                "accepted_draft_token_ids": list(decision.accepted_draft_token_ids),
                "rejected_draft_token_ids": list(decision.rejected_draft_token_ids),
                "target_correction_token_ids": list(decision.target_correction_token_ids),
                "target_bonus_token_ids": list(decision.target_bonus_token_ids),
                "committed_token_ids": list(decision.committed_token_ids),
                **decision.accounting,
                "terminal": terminal,
            },
        }

    def rollback_rejected_suffix(self, request_id: str, round_id: int) -> dict[str, Any]:
        state = self._decision(request_id, round_id)
        decision = state.pending_decision
        assert decision is not None
        if state.rollback_applied:
            raise ValueError("rejected Draft suffix was already rolled back")
        self.backend.rollback(request_id, len(decision.accepted_draft_token_ids))
        state.rollback_applied = True
        return {
            "request_id": request_id,
            "round_id": round_id,
            "accepted_draft_tokens": len(decision.accepted_draft_token_ids),
            "invalidated_draft_tokens": len(decision.rejected_draft_token_ids),
        }

    def append_target_correction_or_bonus(self, request_id: str, round_id: int) -> dict[str, Any]:
        state = self._decision(request_id, round_id)
        decision = state.pending_decision
        assert decision is not None
        if not state.rollback_applied:
            raise ValueError("Draft KV rollback must precede correction/bonus append")
        target_tokens = decision.target_correction_token_ids + decision.target_bonus_token_ids
        if target_tokens:
            self.backend.append_target_token(request_id, target_tokens[0])
        state.committed_token_ids += decision.committed_token_ids
        state.next_round_id += 1
        state.finished = decision.terminal
        state.pending_proposal = None
        state.pending_decision = None
        state.rollback_applied = False
        if state.finished:
            self.backend.finish(request_id)
        return {
            "request_id": request_id,
            "round_id": round_id,
            "committed_prefix_len": len(state.committed_token_ids),
            "committed_prefix_hash": token_prefix_hash(state.committed_token_ids),
            "logical_draft_kv_length": len(state.committed_token_ids),
            "finished": state.finished,
        }

    def synchronize_and_batch_propose(
        self,
        synchronizations: Sequence[Mapping[str, Any]],
        proposals: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        sync_results = []
        sync_start = time.monotonic_ns()
        for row in synchronizations:
            request_id = str(row.get("request_id", ""))
            round_id = row.get("round_id", -1)
            classified = self.synchronize_committed_prefix(
                request_id,
                round_id,
                row.get("committed_delta", ()),
                str(row.get("committed_prefix_hash", "")),
                terminal=bool(row.get("terminal", False)),
            )
            rollback = self.rollback_rejected_suffix(request_id, round_id)
            appended = self.append_target_correction_or_bonus(request_id, round_id)
            sync_results.append({**classified, "rollback": rollback, "state": appended})
        sync_end = time.monotonic_ns()
        proposal_result = (
            self.batch_propose(proposals)
            if proposals
            else {
                "proposals": [],
                "draft_batch_start_ns": sync_end,
                "draft_batch_end_ns": sync_end,
                "draft_microbatch_count": 0,
                "single_persistent_model": True,
            }
        )
        return {
            "synchronizations": sync_results,
            "state_sync_start_ns": sync_start,
            "state_sync_end_ns": sync_end,
            **proposal_result,
        }

    def cancel(self, request_id: str) -> dict[str, Any]:
        state = self._state(request_id)
        if state.finished:
            raise ValueError("finished request cannot be cancelled")
        state.finished = True
        self.backend.finish(request_id)
        return {"request_id": request_id, "cancelled": True}

    def finish(self, request_id: str) -> dict[str, Any]:
        state = self._state(request_id)
        if state.pending_proposal is not None:
            raise ValueError("cannot finish a request with an unresolved proposal")
        if not state.finished:
            state.finished = True
            self.backend.finish(request_id)
        return {"request_id": request_id, "finished": True}

    def shutdown(self) -> dict[str, Any]:
        unresolved = [
            request_id
            for request_id, state in self.requests.items()
            if state.pending_proposal is not None
        ]
        if unresolved:
            raise ValueError(f"cannot shutdown with unresolved proposals: {unresolved}")
        self.backend.shutdown()
        return {"shutdown": True, "request_count": len(self.requests)}

    def _state(self, request_id: str) -> DraftRequest:
        try:
            return self.requests[request_id]
        except KeyError as error:
            raise ValueError(f"unknown Draft request: {request_id}") from error

    def _pending(self, request_id: str, round_id: int) -> DraftRequest:
        state = self._state(request_id)
        if state.finished or state.pending_proposal is None:
            raise ValueError("request has no live proposal to synchronize")
        if state.pending_proposal.round_id != round_id:
            raise ValueError("stale, duplicate, or out-of-order Target synchronization")
        if state.pending_decision is not None:
            raise ValueError("Target synchronization was already classified")
        return state

    def _decision(self, request_id: str, round_id: int) -> DraftRequest:
        state = self._state(request_id)
        if state.finished or state.pending_proposal is None:
            raise ValueError("request has no live proposal to mutate")
        if state.pending_proposal.round_id != round_id:
            raise ValueError("stale, duplicate, or out-of-order Target synchronization")
        if state.pending_decision is None:
            raise ValueError("committed prefix must be synchronized before KV mutation")
        return state


@dataclass
class _HFRequest:
    cache: Any
    next_logits: Any
    committed_length: int
    base_next_logits: Any = None
    proposal: Tuple[int, ...] = ()
    logits_after_prefix: list[Any] = field(default_factory=list)


class HFPersistentDraftBackend:
    """One resident HF model with per-request mutable KV for correctness only.

    This is deliberately not presented as a serving-performance backend. It
    performs one initial full-prefix prefill per request, then only single-token
    forwards and cache cropping across rounds.
    """

    backend_name = "hf-persistent-kv-correctness-draft"

    def __init__(self, config: Phase4Config) -> None:
        visible = _visible_physical_ids()
        if visible != config.draft.physical_gpu_ids:
            raise RuntimeError("Draft service must see only configured physical GPU 0")
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except ImportError as error:
            raise RuntimeError("Draft service requires the Phase-4 GPU environment") from error
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Draft service requires exactly one visible CUDA GPU")
        torch.manual_seed(config.sampling.seed)
        torch.cuda.manual_seed_all(config.sampling.seed)
        dtype = torch.bfloat16 if config.draft.dtype == "bfloat16" else torch.float16
        self.torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(
            str(config.draft.resolved_model_path),
            revision=config.draft.revision,
            dtype=dtype,
            trust_remote_code=config.draft.trust_remote_code,
        ).to("cuda:0")
        self.model.eval()
        parameters = [parameter for parameter in self.model.parameters() if parameter.numel()]
        if not parameters or any(str(parameter.device) != "cuda:0" for parameter in parameters):
            raise RuntimeError("Draft model parameters are not exclusively resident on GPU 0")
        target_path = config.target.resolved_model_path
        if config.draft.resolved_model_path == target_path:
            raise RuntimeError("Draft service was configured with the Target model")
        self._provenance = {
            "backend": self.backend_name,
            "model": model_revision_manifest(
                config.draft.resolved_model_path, config.draft.revision
            ),
            "parameter_count": sum(parameter.numel() for parameter in parameters),
            "parameter_bytes": sum(
                parameter.numel() * parameter.element_size() for parameter in parameters
            ),
            "logical_cuda_index": 0,
            "physical_gpu_id": visible[0],
            "full_context_prefill_per_request": 1,
            "full_context_replay_per_round": False,
            "persistent_cross_round_kv": True,
            "serving_performance_result": False,
        }
        self.states: dict[str, _HFRequest] = {}

    @property
    def provenance(self) -> Mapping[str, Any]:
        return self._provenance

    def initialize(self, request_id: str, committed_token_ids: Sequence[int]) -> None:
        if request_id in self.states:
            raise ValueError("HF Draft request already exists")
        inputs = self.torch.tensor(
            [list(committed_token_ids)], dtype=self.torch.long, device="cuda:0"
        )
        with self.torch.inference_mode():
            output = self.model(input_ids=inputs, use_cache=True)
        cache = output.past_key_values
        if cache is None or not callable(getattr(cache, "crop", None)):
            raise RuntimeError(
                "Transformers cache lacks in-place crop(); source-level Draft KV rollback blocked"
            )
        self.states[request_id] = _HFRequest(
            cache=cache,
            next_logits=output.logits[:, -1, :],
            committed_length=len(committed_token_ids),
        )

    def propose(
        self, request_id: str, budget: int, eos_token_ids: Sequence[int]
    ) -> tuple[Tuple[int, ...], int]:
        state = self.states[request_id]
        if state.proposal:
            raise ValueError("HF Draft request still has an unresolved proposal")
        state.base_next_logits = state.next_logits
        state.logits_after_prefix = [state.next_logits]
        tokens = []
        forwards = 0
        for index in range(budget):
            token_id = int(self.torch.argmax(state.next_logits, dim=-1).item())
            tokens.append(token_id)
            if token_id in eos_token_ids or index + 1 == budget:
                break
            state.next_logits = self._append_token(state, token_id)
            state.logits_after_prefix.append(state.next_logits)
            forwards += 1
        state.proposal = tuple(tokens)
        return state.proposal, forwards

    def rollback(self, request_id: str, accepted_draft_tokens: int) -> None:
        state = self.states[request_id]
        if not 0 <= accepted_draft_tokens <= len(state.proposal):
            raise ValueError("accepted Draft length is outside the proposal")
        target_length = state.committed_length + accepted_draft_tokens
        state.cache.crop(target_length)
        if accepted_draft_tokens < len(state.logits_after_prefix):
            state.next_logits = state.logits_after_prefix[accepted_draft_tokens]
        elif accepted_draft_tokens == len(state.proposal) and state.proposal:
            state.next_logits = self._append_token(state, state.proposal[-1])
        else:
            state.next_logits = state.base_next_logits
        state.committed_length = target_length

    def append_target_token(self, request_id: str, token_id: int) -> None:
        state = self.states[request_id]
        state.next_logits = self._append_token(state, token_id)
        state.committed_length += 1
        state.proposal = ()
        state.logits_after_prefix = []

    def finish(self, request_id: str) -> None:
        self.states.pop(request_id, None)

    def shutdown(self) -> None:
        self.states.clear()
        del self.model
        self.torch.cuda.empty_cache()

    def _append_token(self, state: _HFRequest, token_id: int) -> Any:
        token = self.torch.tensor([[token_id]], dtype=self.torch.long, device="cuda:0")
        with self.torch.inference_mode():
            output = self.model(
                input_ids=token,
                past_key_values=state.cache,
                use_cache=True,
            )
        state.cache = output.past_key_values
        return output.logits[:, -1, :]


def _visible_physical_ids() -> Tuple[int, ...]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or not raw.strip():
        raise RuntimeError("CUDA_VISIBLE_DEVICES must explicitly bind the Draft service")
    try:
        result = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as error:
        raise RuntimeError("Draft service requires numeric CUDA_VISIBLE_DEVICES") from error
    return result


class DraftUnixServer:
    def __init__(
        self,
        socket_path: Path,
        machine: DraftStateMachine,
        *,
        event_log: CheckpointJsonl,
    ) -> None:
        self.socket_path = socket_path
        self.machine = machine
        self.event_log = event_log
        self.running = True

    def serve(self, ready_path: Path) -> None:
        if self.socket_path.exists():
            raise FileExistsError(f"refusing to replace existing socket {self.socket_path}")
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o600)
            server.listen(8)
            atomic_write_json(
                ready_path,
                {
                    "schema_version": "specrhythm.phase4-draft-service-ready.v1",
                    "protocol_version": PROTOCOL_VERSION,
                    "socket_file": self.socket_path.name,
                    "pid": os.getpid(),
                    "backend": self.machine.backend.backend_name,
                    "provenance": dict(self.machine.backend.provenance),
                    "gpu_correctness_result": True,
                    "gpu_performance_result": False,
                },
            )
            while self.running:
                connection, _ = server.accept()
                with connection:
                    self._handle(connection)
        self.socket_path.unlink(missing_ok=True)

    def _handle(self, connection: socket.socket) -> None:
        received_ns = time.monotonic_ns()
        try:
            request, request_bytes = receive_message(connection)
            if request.get("protocol_version") != PROTOCOL_VERSION:
                raise ValueError("incompatible Draft service protocol")
            operation = str(request.get("operation", ""))
            payload = request.get("payload")
            if not isinstance(payload, Mapping):
                raise ValueError("Draft service payload must be an object")
            result = self._dispatch(operation, payload)
            result["service_receive_ns"] = received_ns
            result["service_send_ns"] = time.monotonic_ns()
            result["request_payload_bytes"] = request_bytes
            response = {
                "protocol_version": PROTOCOL_VERSION,
                "ok": True,
                "result": result,
            }
            response_bytes = send_message(connection, response)
            self.event_log.append(
                {
                    "schema_version": "specrhythm.phase4-draft-service-event.v1",
                    "operation": operation,
                    "received_ns": received_ns,
                    "completed_ns": time.monotonic_ns(),
                    "request_payload_bytes": request_bytes,
                    "response_payload_bytes": response_bytes,
                    "success": True,
                }
            )
        except Exception as error:
            send_message(
                connection,
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                    "result": {},
                },
            )

    def _dispatch(self, operation: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if operation == "initialize":
            return self.machine.initialize(
                str(payload.get("request_id", "")),
                payload.get("committed_token_ids", ()),
                str(payload.get("committed_prefix_hash", "")),
            )
        if operation == "batch_propose":
            return self.machine.batch_propose(payload.get("requests", ()))
        if operation == "synchronize_committed_prefix":
            return self.machine.synchronize_committed_prefix(
                str(payload.get("request_id", "")),
                payload.get("round_id", -1),
                payload.get("committed_delta", ()),
                str(payload.get("committed_prefix_hash", "")),
                terminal=bool(payload.get("terminal", False)),
            )
        if operation == "rollback_rejected_suffix":
            return self.machine.rollback_rejected_suffix(
                str(payload.get("request_id", "")), payload.get("round_id", -1)
            )
        if operation == "append_target_correction_or_bonus":
            return self.machine.append_target_correction_or_bonus(
                str(payload.get("request_id", "")), payload.get("round_id", -1)
            )
        if operation == "synchronize_and_batch_propose":
            return self.machine.synchronize_and_batch_propose(
                payload.get("synchronizations", ()), payload.get("proposals", ())
            )
        if operation == "cancel_request":
            return self.machine.cancel(str(payload.get("request_id", "")))
        if operation == "finish_request":
            return self.machine.finish(str(payload.get("request_id", "")))
        if operation == "shutdown":
            result = self.machine.shutdown()
            self.running = False
            return result
        raise ValueError(f"unsupported Draft service operation: {operation}")


def run_draft_service(
    config: Phase4Config,
    *,
    socket_path: Path,
    event_log_path: Path,
    ready_path: Path,
) -> None:
    backend = HFPersistentDraftBackend(config)
    server = DraftUnixServer(
        socket_path,
        DraftStateMachine(backend, candidate_budget=config.proposal_budget),
        event_log=CheckpointJsonl(event_log_path),
    )
    server.serve(ready_path)
