"""vLLM v0.25.1 custom proposer backed by the local Draft service.

This module is imported inside Target TP workers. Only TP rank zero uses the
Unix socket; the result is broadcast to the other Target rank. No model or
Target-side logits are stored in this proposer.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from specrhythm.phase4.manifest import atomic_write_json
from specrhythm.phase4.request_identity import FrozenPromptIdentityMap
from specrhythm.phase4.serial import Proposal, RoundRecord, SerialTimeline, greedy_acceptance
from specrhythm.phase4.stock_vllm import load_smoke_requests
from specrhythm.phase4.transport import CheckpointJsonl, UnixDraftClient
from specrhythm.phase4.vllm_diagnostics import TARGET_ONLY_FIELDS


@dataclass
class _TargetRequest:
    request_id: str
    prompt_token_ids: Tuple[int, ...]
    maximum_new_tokens: int
    committed_token_ids: Tuple[int, ...]
    generated_token_ids: Tuple[int, ...]
    next_round_id: int = 0
    pending_proposal: Optional[Proposal] = None
    transfer_start_ns: int = 0
    transfer_end_ns: int = 0
    verify_start_ns: int = 0
    verify_end_ns: int = 0
    target_batch_id: Optional[str] = None
    target_batch_request_ids: Tuple[str, ...] = ()
    bootstrap_target_tokens: int = 0
    tail_target_tokens: int = 0
    finished: bool = False


class RemoteDraftProposer:
    """Experimental custom-class proposer for the pinned vLLM V1 runner."""

    def __init__(self, vllm_config: Any) -> None:
        try:
            import torch
            from vllm.distributed.parallel_state import get_tp_group
        except ImportError as error:
            raise RuntimeError("RemoteDraftProposer must run inside pinned vLLM") from error
        self.torch = torch
        self.tp_group = get_tp_group()
        self.tp_rank = int(self.tp_group.rank_in_group)
        self.tp_world_size = int(self.tp_group.world_size)
        if self.tp_world_size != 2:
            raise RuntimeError("Phase-4A.1 Target proposer requires TP=2")
        socket_path = _required_path("SR_PHASE4_DRAFT_SOCKET")
        workload_path = _required_path("SR_PHASE4_WORKLOAD")
        self.round_log = CheckpointJsonl(_required_path("SR_PHASE4_ROUND_EVENTS"))
        self.transport_log = CheckpointJsonl(_required_path("SR_PHASE4_TRANSPORT_EVENTS"))
        self.report_path = _required_path("SR_PHASE4_PLUGIN_REPORT")
        self.client = UnixDraftClient(socket_path, transport_log=self.transport_log)
        requests = load_smoke_requests(workload_path)
        self.definitions = {request.request_id: request for request in requests}
        self.identity = FrozenPromptIdentityMap.from_definitions(requests)
        eos = getattr(vllm_config.model_config.hf_config, "eos_token_id", None)
        if isinstance(eos, int):
            self.eos_token_ids = (eos,)
        elif isinstance(eos, (list, tuple)):
            self.eos_token_ids = tuple(int(item) for item in eos)
        else:
            self.eos_token_ids = ()
        self.requests: dict[str, _TargetRequest] = {}
        # Kept as a public diagnostic alias for existing Phase-4A reports.
        self.internal_to_stable = self.identity.internal_to_stable
        self.round_records: list[dict[str, Any]] = []
        self.hooks_seen = {"verify_start": 0, "verify_end": 0}
        self.verify_phase_sequence = 0
        self._write_report()

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: Any,
        token_ids_cpu: Any,
        *,
        request_ids: Optional[Sequence[str]] = None,
        slot_mappings: Any = None,
    ) -> list[list[int]]:
        del sampled_token_ids, slot_mappings
        if request_ids is None:
            raise RuntimeError(
                "Phase-4A.1 hook patch is missing: custom proposer did not receive request_ids"
            )
        if self.tp_rank == 0:
            result = self._rank_zero_propose(request_ids, num_tokens_no_spec, token_ids_cpu)
        else:
            result = None
        result = self.tp_group.broadcast_object(result, src=0)
        if not isinstance(result, Mapping):
            raise RuntimeError("Target TP rank did not receive remote Draft proposals")
        proposals = result.get("draft_token_ids")
        if not isinstance(proposals, list) or len(proposals) != len(request_ids):
            raise RuntimeError("remote Draft proposal batch shape is invalid")
        return [list(row) for row in proposals]

    def on_target_verify_start(
        self,
        *,
        request_ids: Sequence[str],
        scheduled_spec_token_ids: Mapping[str, Sequence[int]],
    ) -> None:
        self.torch.cuda.synchronize()
        self.tp_group.barrier()
        now = time.monotonic_ns()
        if self.tp_rank != 0:
            return
        target_batch_id = f"target-verification-phase-{self.verify_phase_sequence}"
        self.verify_phase_sequence += 1
        stable_batch = tuple(
            self.internal_to_stable[str(internal_id)]
            for internal_id in request_ids
            if internal_id in scheduled_spec_token_ids
            and str(internal_id) in self.internal_to_stable
        )
        for internal_id in request_ids:
            if internal_id not in scheduled_spec_token_ids:
                continue
            stable = self.internal_to_stable.get(str(internal_id))
            if stable is None:
                raise RuntimeError("verify-start hook observed an unmapped vLLM request")
            state = self.requests[stable]
            if state.pending_proposal is None or state.verify_start_ns:
                raise RuntimeError("verify-start hook has no unique live proposal")
            state.verify_start_ns = now
            state.target_batch_id = target_batch_id
            state.target_batch_request_ids = stable_batch
            self.hooks_seen["verify_start"] += 1

    def on_target_verify_end(
        self,
        *,
        request_ids: Sequence[str],
        sampled_token_ids: Sequence[Sequence[int]],
        scheduled_spec_token_ids: Mapping[str, Sequence[int]],
    ) -> None:
        del sampled_token_ids
        self.torch.cuda.synchronize()
        self.tp_group.barrier()
        now = time.monotonic_ns()
        if self.tp_rank != 0:
            return
        for internal_id in request_ids:
            if internal_id not in scheduled_spec_token_ids:
                continue
            stable = self.internal_to_stable.get(str(internal_id))
            if stable is None:
                continue
            state = self.requests[stable]
            if state.pending_proposal is None:
                continue
            if not state.verify_start_ns or state.verify_end_ns:
                raise RuntimeError("verify-end hook is missing or duplicated")
            state.verify_end_ns = now
            self.hooks_seen["verify_end"] += 1

    @property
    def supports_mm_inputs(self) -> bool:
        return False

    def _rank_zero_propose(
        self, request_ids: Sequence[str], num_tokens_no_spec: Any, token_ids_cpu: Any
    ) -> dict[str, Any]:
        current_rows = []
        for index, internal_id in enumerate(request_ids):
            count = int(num_tokens_no_spec[index])
            tokens = tuple(int(item) for item in token_ids_cpu[index, :count].tolist())
            stable_id = self.identity.bind(str(internal_id), tokens)
            definition = self.definitions[stable_id]
            generated = self._logical_generated(
                tokens[len(definition.prompt_token_ids) :], definition.maximum_new_tokens
            )
            current_rows.append((stable_id, definition, tokens, generated))

        synchronizations = []
        initializations = []
        proposal_rows = []
        terminal_ids = set()
        finish_without_pending = []
        for stable_id, definition, _physical_tokens, generated in current_rows:
            logical_prefix = definition.prompt_token_ids + generated
            terminal = self._terminal(generated, definition.maximum_new_tokens)
            state = self.requests.get(stable_id)
            if state is None:
                if not generated:
                    raise RuntimeError("stock vLLM bootstrap token is missing")
                state = _TargetRequest(
                    request_id=stable_id,
                    prompt_token_ids=definition.prompt_token_ids,
                    maximum_new_tokens=definition.maximum_new_tokens,
                    committed_token_ids=logical_prefix,
                    generated_token_ids=generated,
                    bootstrap_target_tokens=len(generated),
                    finished=terminal,
                )
                self.requests[stable_id] = state
                initializations.append(
                    {
                        "request_id": stable_id,
                        "committed_token_ids": list(logical_prefix),
                        "committed_prefix_hash": _prefix_hash(logical_prefix),
                    }
                )
            else:
                if state.pending_proposal is None:
                    if generated != state.generated_token_ids:
                        state.tail_target_tokens += len(generated) - len(state.generated_token_ids)
                        state.generated_token_ids = generated
                        state.committed_token_ids = logical_prefix
                    state.finished = terminal
                else:
                    if not _starts_with(logical_prefix, state.committed_token_ids):
                        raise RuntimeError("Target committed prefix regressed or diverged")
                    delta = logical_prefix[len(state.committed_token_ids) :]
                    synchronizations.append(
                        {
                            "request_id": stable_id,
                            "round_id": state.pending_proposal.round_id,
                            "committed_delta": list(delta),
                            "committed_prefix_hash": _prefix_hash(logical_prefix),
                            "terminal": terminal,
                        }
                    )
            if terminal:
                terminal_ids.add(stable_id)
            else:
                remaining = definition.maximum_new_tokens - len(generated)
                proposal_rows.append(
                    {
                        "request_id": stable_id,
                        "round_id": state.next_round_id
                        + (1 if state.pending_proposal is not None else 0),
                        "committed_prefix_len": len(logical_prefix),
                        "committed_prefix_hash": _prefix_hash(logical_prefix),
                        "remaining_output_budget": remaining,
                        "eos_token_ids": list(self.eos_token_ids),
                    }
                )
            if terminal and state.pending_proposal is None:
                finish_without_pending.append(stable_id)

        for row in initializations:
            self.client.call("initialize", row)
        outgoing = {
            "synchronizations": synchronizations,
            "proposals": proposal_rows,
        }
        _assert_target_information_isolated(outgoing)
        response = self.client.call("synchronize_and_batch_propose", outgoing)
        for stable_id in finish_without_pending:
            self.client.call("finish_request", {"request_id": stable_id})
        proposal_by_request = {
            str(row["request_id"]): Proposal.from_dict(row)
            for row in response.get("proposals", ())
        }
        sync_by_request = {
            str(row["request_id"]): row for row in response.get("synchronizations", ())
        }
        for stable_id, definition, _physical_tokens, generated in current_rows:
            state = self.requests[stable_id]
            logical_prefix = definition.prompt_token_ids + generated
            sync = sync_by_request.get(stable_id)
            if sync is not None:
                self._finalize_round(
                    state,
                    logical_prefix,
                    generated,
                    sync,
                    response,
                    proposal_by_request.get(stable_id),
                    terminal=stable_id in terminal_ids,
                )
            proposal = proposal_by_request.get(stable_id)
            if proposal is not None:
                if proposal.parent_prefix_hash != _prefix_hash(logical_prefix):
                    raise RuntimeError("remote proposal has a stale parent prefix")
                state.pending_proposal = proposal
                state.transfer_start_ns = int(response["service_send_ns"])
                state.transfer_end_ns = int(response["transport_end_ns"])
                state.verify_start_ns = 0
                state.verify_end_ns = 0
                state.target_batch_id = None
                state.target_batch_request_ids = ()
            state.finished = stable_id in terminal_ids
        self._write_report()
        return {
            "draft_token_ids": [
                list(proposal_by_request[stable_id].proposal_token_ids)
                if stable_id in proposal_by_request
                else []
                for stable_id, _definition, _tokens, _generated in current_rows
            ],
            "proposal_request_ids": [row[0] for row in current_rows],
        }

    def _finalize_round(
        self,
        state: _TargetRequest,
        logical_prefix: Tuple[int, ...],
        generated: Tuple[int, ...],
        synchronization: Mapping[str, Any],
        response: Mapping[str, Any],
        next_proposal: Optional[Proposal],
        *,
        terminal: bool,
    ) -> None:
        proposal = state.pending_proposal
        if proposal is None:
            raise RuntimeError("cannot finalize a missing proposal")
        if not state.verify_start_ns or not state.verify_end_ns:
            raise RuntimeError("patched verify timing hooks did not run")
        delta = logical_prefix[len(state.committed_token_ids) :]
        decision = greedy_acceptance(proposal.proposal_token_ids, delta, terminal=terminal)
        remote_decision = synchronization.get("decision")
        if not isinstance(remote_decision, Mapping) or list(
            decision.committed_token_ids
        ) != remote_decision.get("committed_token_ids"):
            raise RuntimeError("Target and Draft classified different committed tokens")
        sync_start = int(response["state_sync_start_ns"])
        sync_end = int(response["state_sync_end_ns"])
        next_start = next_proposal.draft_start_ns if next_proposal else sync_end
        timeline = SerialTimeline(
            draft_start_ns=proposal.draft_start_ns,
            draft_end_ns=proposal.draft_end_ns,
            transfer_start_ns=state.transfer_start_ns,
            transfer_end_ns=state.transfer_end_ns,
            verify_start_ns=state.verify_start_ns,
            verify_end_ns=state.verify_end_ns,
            state_sync_start_ns=sync_start,
            state_sync_end_ns=sync_end,
            next_round_draft_start_ns=next_start,
        )
        remaining = state.maximum_new_tokens - len(generated)
        record = RoundRecord(
            request_id=state.request_id,
            round_id=proposal.round_id,
            parent_prefix_len=proposal.parent_prefix_len,
            parent_prefix_hash=proposal.parent_prefix_hash,
            proposal_token_ids=proposal.proposal_token_ids,
            decision=decision,
            remaining_output_budget=remaining,
            logical_target_kv_length=len(logical_prefix),
            logical_draft_kv_length=int(
                synchronization.get("state", {}).get("logical_draft_kv_length", -1)
            ),
            timeline=timeline,
            target_microbatch_id=state.target_batch_id,
        )
        row = {
            "schema_version": "specrhythm.phase4-round-event.v1",
            **record.to_dict(),
            "committed_prefix_hash": _prefix_hash(logical_prefix),
            "target_authority": True,
            "candidate_linear_sequence": True,
            "packed_tree_verification": False,
            "target_batch_request_ids": list(state.target_batch_request_ids),
            "vllm_dbo_microbatching": False,
        }
        self.round_log.append(row)
        self.round_records.append(row)
        state.committed_token_ids = logical_prefix
        state.generated_token_ids = generated
        state.next_round_id += 1
        state.pending_proposal = None

    def _logical_generated(
        self, generated: Sequence[int], maximum_new_tokens: int
    ) -> Tuple[int, ...]:
        result = list(generated[:maximum_new_tokens])
        for index, token_id in enumerate(result):
            if token_id in self.eos_token_ids:
                result = result[: index + 1]
                break
        return tuple(result)

    def _terminal(self, generated: Sequence[int], maximum_new_tokens: int) -> bool:
        return len(generated) >= maximum_new_tokens or bool(
            generated and generated[-1] in self.eos_token_ids
        )

    def _write_report(self) -> None:
        if self.tp_rank != 0:
            return
        atomic_write_json(
            self.report_path,
            {
                "schema_version": "specrhythm.phase4-remote-proposer-report.v1",
                "proposer_model_parameter_count": 0,
                "target_logits_observed": False,
                "target_future_tokens_observed": False,
                "oracle_labels_observed": False,
                "transport": "unix-domain-socket",
                "target_tp_world_size": self.tp_world_size,
                "target_rank0_only_transport": True,
                "hook_counts": dict(self.hooks_seen),
                "request_count": len(self.requests),
                "round_count": len(self.round_records),
                "requests": {
                    request_id: {
                        "generated_token_ids": list(state.generated_token_ids),
                        "bootstrap_target_tokens": state.bootstrap_target_tokens,
                        "tail_target_tokens": state.tail_target_tokens,
                        "next_round_id": state.next_round_id,
                        "finished": state.finished,
                    }
                    for request_id, state in self.requests.items()
                },
                "gpu_correctness_result": True,
                "gpu_performance_result": False,
            },
        )


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required by the remote Draft proposer")
    return Path(value).resolve()


def _starts_with(values: Sequence[int], prefix: Sequence[int]) -> bool:
    return len(values) >= len(prefix) and tuple(values[: len(prefix)]) == tuple(prefix)


def _prefix_hash(values: Sequence[int]) -> str:
    from specrhythm.phase4.serial import token_prefix_hash

    return token_prefix_hash(values)


def _assert_target_information_isolated(value: Mapping[str, Any]) -> None:
    forbidden = set(TARGET_ONLY_FIELDS) | {
        "target_logits",
        "target_logprobs",
        "target_topk",
        "future_target_tokens",
        "oracle_label",
    }

    def walk(item: Any) -> None:
        if isinstance(item, Mapping):
            overlap = forbidden & set(item)
            if overlap:
                raise ValueError(f"Target-only information leaked to Draft: {sorted(overlap)}")
            for child in item.values():
                walk(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                walk(child)

    walk(value)
