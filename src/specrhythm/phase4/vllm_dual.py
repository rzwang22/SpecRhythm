"""Pinned-vLLM custom proposer observer for Phase-4B Dual-Batch.

Proposal generation is never performed in this Target callback.  Rank zero
only enqueues committed-prefix work to the asynchronous GPU-0 Draft service;
the custom scheduler consumes already-ready proposals independently.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from specrhythm.phase4.dual import DualProposal
from specrhythm.phase4.dual_service import DualDraftClient
from specrhythm.phase4.manifest import atomic_write_json
from specrhythm.phase4.request_identity import FrozenPromptIdentityMap
from specrhythm.phase4.serial import greedy_acceptance, token_prefix_hash
from specrhythm.phase4.stock_vllm import load_smoke_requests
from specrhythm.phase4.transport import CheckpointJsonl


@dataclass
class _Request:
    prompt_token_ids: Tuple[int, ...]
    maximum_new_tokens: int
    committed_token_ids: Tuple[int, ...]
    generated_token_ids: Tuple[int, ...]
    prefix_version: int = 1
    next_round_id: int = 0
    pending_proposal: Optional[DualProposal] = None
    terminal: bool = False
    lifecycle: str = "BOOTSTRAP"


class DualBatchRemoteProposer:
    """Target-side observer and non-blocking Draft work submitter."""

    def __init__(self, vllm_config: Any) -> None:
        try:
            import torch
            import torch.distributed as dist
            from vllm.distributed.parallel_state import get_tp_group
        except ImportError as error:
            raise RuntimeError("DualBatchRemoteProposer requires pinned vLLM") from error
        if os.environ.get("SR_PHASE4_DUAL_BATCH") != "1":
            raise RuntimeError("Dual proposer is fail-closed unless explicitly enabled")
        self.torch = torch
        self.dist = dist
        self.tp_group = get_tp_group()
        self.tp_rank = int(self.tp_group.rank_in_group)
        self.tp_world_size = int(self.tp_group.world_size)
        if self.tp_world_size != 2:
            raise RuntimeError("Phase-4B.1 Target requires TP=2")
        self.client = DualDraftClient(_required_path("SR_PHASE4_DUAL_DRAFT_SOCKET"))
        self.proposal_log = CheckpointJsonl(_required_path("SR_PHASE4_PROPOSAL_EVENTS"))
        self.verification_log = CheckpointJsonl(
            _required_path("SR_PHASE4_VERIFICATION_EVENTS")
        )
        self.state_log = CheckpointJsonl(_required_path("SR_PHASE4_REQUEST_STATE_EVENTS"))
        self.report_path = _required_path("SR_PHASE4_DUAL_PLUGIN_REPORT")
        request_count = int(os.environ.get("SR_PHASE4_REQUEST_COUNT", "5"))
        definitions = load_smoke_requests(
            _required_path("SR_PHASE4_WORKLOAD"),
            request_count,
            require_task_mixture=request_count in {5, 100},
        )
        self.definitions = {item.request_id: item for item in definitions}
        self.identity = FrozenPromptIdentityMap.from_definitions(definitions)
        eos = getattr(vllm_config.model_config.hf_config, "eos_token_id", None)
        if isinstance(eos, int):
            self.eos_token_ids = (eos,)
        elif isinstance(eos, (list, tuple)):
            self.eos_token_ids = tuple(int(item) for item in eos)
        else:
            self.eos_token_ids = ()
        self.requests: dict[str, _Request] = {}
        self.internal_to_stable = self.identity.internal_to_stable
        self.stable_to_internal = self.identity.stable_to_internal
        self._verify_start: dict[str, dict[str, Any]] = {}
        self._verify_batch_by_request: dict[str, str] = {}
        self._verified_ids: set[str] = set()
        existing_verifications = self.verification_log.read()
        self._verify_sequence = 1 + max(
            (
                int(row.get("verify_sequence", -1))
                for row in existing_verifications
            ),
            default=-1,
        )
        self._write_report()

    @property
    def supports_mm_inputs(self) -> bool:
        return False

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
            raise RuntimeError("Phase-4 worker hook did not provide request IDs")
        if self.tp_rank == 0:
            result = self._rank_zero_update(request_ids, num_tokens_no_spec, token_ids_cpu)
        else:
            result = None
        result = self.tp_group.broadcast_object(result, src=0)
        if not isinstance(result, Mapping) or result.get("ok") is not True:
            raise RuntimeError("Target TP ranks did not agree on Dual-Batch state update")
        # Ready proposals are injected by DualBatchScheduler before execution.
        # Returning empty rows prevents the stock post-step path from creating
        # a second, unverified proposal for any request.
        return [[] for _ in request_ids]

    def on_target_verify_start(
        self,
        *,
        request_ids: Sequence[str],
        scheduled_spec_token_ids: Mapping[str, Sequence[int]],
    ) -> None:
        internal_verify_ids = [
            str(item) for item in request_ids if item in scheduled_spec_token_ids
        ]
        if not internal_verify_ids:
            return
        if self.tp_rank == 0:
            stable_verify_ids = [
                self.identity.stable_id(internal_id)
                for internal_id in internal_verify_ids
            ]
            if len(set(stable_verify_ids)) != len(stable_verify_ids):
                raise RuntimeError("Target verify aliases multiple internal IDs to one request")
            claimed = self.client.call("claimed", {"request_ids": stable_verify_ids})
            service_rows = claimed.get("claimed", ())
            if not isinstance(service_rows, list) or len(service_rows) != len(
                stable_verify_ids
            ):
                raise RuntimeError("claimed stable proposal metadata is incomplete")
            service_by_stable = {
                str(row.get("request_id")): row
                for row in service_rows
                if isinstance(row, Mapping)
            }
            if set(service_by_stable) != set(stable_verify_ids):
                raise RuntimeError("claimed proposal stable request set is inconsistent")
            claimed_rows = []
            for internal_id, stable_id in zip(internal_verify_ids, stable_verify_ids):
                row = service_by_stable.get(stable_id)
                if row is None:
                    raise RuntimeError("claimed stable proposal metadata is incomplete")
                claimed_rows.append(
                    {
                        **row,
                        "request_id": stable_id,
                        "internal_request_id": internal_id,
                    }
                )
        else:
            claimed_rows = None
        claimed_rows = self.tp_group.broadcast_object(claimed_rows, src=0)
        if not isinstance(claimed_rows, list) or len(claimed_rows) != len(
            internal_verify_ids
        ):
            raise RuntimeError("claimed proposal metadata is incomplete")
        by_internal = {
            str(row.get("internal_request_id")): row for row in claimed_rows
        }
        if set(by_internal) != set(internal_verify_ids):
            raise RuntimeError("claimed proposal internal request set is inconsistent")
        verify_batch_id = f"verify-{self._verify_sequence}"
        for internal_id in internal_verify_ids:
            result = by_internal.get(internal_id)
            proposal_value = result.get("proposal") if isinstance(result, Mapping) else None
            if not isinstance(proposal_value, Mapping):
                raise RuntimeError("Target verify lacks claimed proposal metadata")
            proposal = DualProposal.from_dict(proposal_value)
            stable_id = str(result.get("request_id"))
            if proposal.request_id != stable_id:
                raise RuntimeError("claimed proposal belongs to a different stable request")
            expected = tuple(int(item) for item in scheduled_spec_token_ids[internal_id])
            if proposal.proposal_token_ids != expected:
                raise RuntimeError("scheduled tokens differ from claimed proposal")
            if self.tp_rank == 0:
                if self.identity.stable_id(internal_id) != stable_id:
                    raise RuntimeError("scheduler/proposer request identity mapping disagrees")
                state = self.requests.get(stable_id)
                if state is None or state.pending_proposal is not None:
                    raise RuntimeError("request cannot own two unverified proposals")
                prefix = state.committed_token_ids
                if (
                    proposal.prefix_version != state.prefix_version
                    or proposal.prefix_token_count != len(prefix)
                    or proposal.prefix_token_sha256 != token_prefix_hash(prefix)
                ):
                    raise RuntimeError("stale proposal reached Target verify boundary")
                state.pending_proposal = proposal
                self._verify_batch_by_request[stable_id] = verify_batch_id
                self._transition(stable_id, "PROPOSAL_READY")
                self._transition(stable_id, "VERIFY_READY")
                self._transition(stable_id, "VERIFYING")
        self.tp_group.barrier()
        self.torch.cuda.synchronize()
        self.torch.cuda.nvtx.range_push(
            "specrhythm:verify:" + ",".join(
                str(by_internal[item]["request_id"]) for item in internal_verify_ids
            )
        )
        start_event = self.torch.cuda.Event(enable_timing=True)
        start_event.record()
        self._verify_start = {
            internal_id: {
                "event": start_event,
                "host_start_ns": time.monotonic_ns(),
                "proposal": by_internal[internal_id]["proposal"],
                "request_id": str(by_internal[internal_id]["request_id"]),
            }
            for internal_id in internal_verify_ids
        }

    def on_target_verify_end(
        self,
        *,
        request_ids: Sequence[str],
        sampled_token_ids: Sequence[Sequence[int]],
        scheduled_spec_token_ids: Mapping[str, Sequence[int]],
    ) -> None:
        del sampled_token_ids
        internal_verify_ids = [
            str(item) for item in request_ids if item in scheduled_spec_token_ids
        ]
        if not internal_verify_ids:
            return
        if set(internal_verify_ids) != set(self._verify_start):
            raise RuntimeError("verify-end internal request set differs from verify-start")
        end_event = self.torch.cuda.Event(enable_timing=True)
        end_event.record()
        end_event.synchronize()
        self.torch.cuda.nvtx.range_pop()
        host_end = time.monotonic_ns()
        self.tp_group.barrier()
        physical = _physical_gpu_id()
        uuid = self.torch.cuda.get_device_properties(0).uuid
        local_rows = []
        for internal_id in internal_verify_ids:
            start = self._verify_start.get(internal_id)
            if start is None:
                raise RuntimeError("verify-end has no matching CUDA start event")
            stable_id = str(start["request_id"])
            local_rows.append(
                {
                    "request_id": stable_id,
                    "internal_request_id": internal_id,
                    "global_rank": int(self.dist.get_rank()),
                    "local_rank": int(os.environ.get("LOCAL_RANK", self.tp_rank)),
                    "tp_rank": self.tp_rank,
                    "logical_cuda_index": 0,
                    "physical_gpu_id": physical,
                    "gpu_uuid": str(uuid),
                    "host_start_ns": start["host_start_ns"],
                    "host_end_ns": host_end,
                    "cuda_elapsed_ns": int(start["event"].elapsed_time(end_event) * 1_000_000),
                    "cuda_events": True,
                    "cuda_synchronized": True,
                }
            )
        gathered: list[Any] = [None] * self.tp_world_size
        self.dist.all_gather_object(gathered, local_rows, group=self.tp_group.cpu_group)
        if self.tp_rank == 0:
            rank_rows = [row for group in gathered for row in group]
            stable_verify_ids = [
                str(self._verify_start[internal_id]["request_id"])
                for internal_id in internal_verify_ids
            ]
            batch_ids = {
                self._verify_batch_by_request[stable_id]
                for stable_id in stable_verify_ids
            }
            if len(batch_ids) != 1:
                raise RuntimeError("one Target verification batch has mixed proposal metadata")
            verify_batch_id = next(iter(batch_ids))
            for internal_id, stable_id in zip(internal_verify_ids, stable_verify_ids):
                proposal = DualProposal.from_dict(
                    self._verify_start[internal_id]["proposal"]
                )
                evidence = [row for row in rank_rows if row["request_id"] == stable_id]
                if len(evidence) != self.tp_world_size:
                    raise RuntimeError("Target verify is missing TP-rank identity evidence")
                self.verification_log.append(
                    {
                        "schema_version": "specrhythm.phase4b-verification-event.v1",
                        "request_id": stable_id,
                        "internal_request_id": internal_id,
                        "round_id": proposal.round_id,
                        "proposal_id": proposal.proposal_id,
                        "prefix_version": proposal.prefix_version,
                        "proposal_token_ids": list(proposal.proposal_token_ids),
                        "verify_microbatch_id": verify_batch_id,
                        "verify_sequence": self._verify_sequence,
                        "verify_request_ids": stable_verify_ids,
                        "internal_verify_request_ids": internal_verify_ids,
                        "target_rank_intervals": evidence,
                        # Conservative TP interval: every rank is inside its
                        # synchronized verify region throughout this window.
                        "verify_host_start_ns": max(
                            row["host_start_ns"] for row in evidence
                        ),
                        "verify_host_end_ns": min(
                            row["host_end_ns"] for row in evidence
                        ),
                        "target_tp": self.tp_world_size,
                        "target_physical_gpu_ids": sorted(
                            {row["physical_gpu_id"] for row in evidence}
                        ),
                    }
                )
                self._verified_ids.add(stable_id)
        self._verify_sequence += 1
        self._verify_start = {}

    def _rank_zero_update(
        self, request_ids: Sequence[str], num_tokens_no_spec: Any, token_ids_cpu: Any
    ) -> dict[str, Any]:
        bootstrap_rows = []
        commit_rows = []
        tail_rows = []
        for index, request_id_value in enumerate(request_ids):
            internal_request_id = str(request_id_value)
            count = int(num_tokens_no_spec[index])
            physical_tokens = tuple(
                int(item) for item in token_ids_cpu[index, :count].tolist()
            )
            request_id = self.identity.bind(internal_request_id, physical_tokens)
            definition = self.definitions[request_id]
            generated = _logical_generated(
                physical_tokens[len(definition.prompt_token_ids) :],
                definition.maximum_new_tokens,
            )
            prefix = definition.prompt_token_ids + generated
            terminal = _terminal(generated, definition.maximum_new_tokens, self.eos_token_ids)
            state = self.requests.get(request_id)
            if state is None:
                if not generated:
                    continue
                state = _Request(
                    prompt_token_ids=definition.prompt_token_ids,
                    maximum_new_tokens=definition.maximum_new_tokens,
                    committed_token_ids=prefix,
                    generated_token_ids=generated,
                    terminal=terminal,
                )
                self.requests[request_id] = state
                bootstrap_rows.append(
                    self._work_row(request_id, state, terminal=terminal)
                )
                if terminal:
                    self._transition(request_id, "FINISHED")
                else:
                    self._transition(request_id, "DRAFT_READY")
                    self._transition(request_id, "DRAFTING")
                continue
            if request_id in self._verified_ids:
                commit_start_ns = time.monotonic_ns()
                proposal = state.pending_proposal
                if proposal is None:
                    raise RuntimeError("verified request lost its proposal identity")
                if not _starts_with(prefix, state.committed_token_ids):
                    raise RuntimeError("Target committed prefix regressed or diverged")
                delta = prefix[len(state.committed_token_ids) :]
                decision = greedy_acceptance(
                    proposal.proposal_token_ids, delta, terminal=terminal
                )
                state.prefix_version += 1
                state.next_round_id += 1
                state.committed_token_ids = prefix
                state.generated_token_ids = generated
                state.terminal = terminal
                commit_rows.append(
                    {
                        **self._work_row(request_id, state, terminal=terminal),
                        "proposal_id": proposal.proposal_id,
                        "round_id": proposal.round_id,
                        "committed_delta": list(delta),
                    }
                )
                self.proposal_log.append(
                    {
                        "schema_version": "specrhythm.phase4b-proposal-event.v1",
                        **proposal.to_dict(),
                        "accepted_draft_tokens": len(decision.accepted_draft_token_ids),
                        "rejected_draft_tokens": len(decision.rejected_draft_token_ids),
                        "target_correction_token_ids": list(
                            decision.target_correction_token_ids
                        ),
                        "target_bonus_token_ids": list(decision.target_bonus_token_ids),
                        "committed_token_ids": list(decision.committed_token_ids),
                        "terminal": terminal,
                        "stale": False,
                        "verify_microbatch_id": self._verify_batch_by_request.pop(
                            request_id
                        ),
                        "commit_start_ns": commit_start_ns,
                        "commit_end_ns": time.monotonic_ns(),
                    }
                )
                state.pending_proposal = None
                self._transition(request_id, "COMMITTING")
                if terminal:
                    self._transition(request_id, "FINISHED")
                else:
                    self._transition(request_id, "DRAFT_SYNC")
                    self._transition(request_id, "DRAFT_READY")
                    self._transition(request_id, "DRAFTING")
            elif generated != state.generated_token_ids:
                # A proposal-free one-token tail is the only legal Target-only
                # decode after bootstrap.
                delta = prefix[len(state.committed_token_ids) :]
                if state.pending_proposal is not None or len(delta) != 1 or not terminal:
                    raise RuntimeError("unproposed Target decode advanced a live request")
                state.prefix_version += 1
                state.committed_token_ids = prefix
                state.generated_token_ids = generated
                state.terminal = True
                tail_rows.append(
                    {
                        "request_id": request_id,
                        "committed_delta": list(delta),
                        "prefix_version": state.prefix_version,
                        "prefix_token_sha256": token_prefix_hash(prefix),
                        "terminal": True,
                    }
                )
                # The asynchronous Draft worker returned a proposal-free tail.
                self._transition(request_id, "DRAFT_READY")
                self._transition(request_id, "VERIFY_READY")
                self._transition(request_id, "VERIFYING")
                self._transition(request_id, "COMMITTING")
                self._transition(request_id, "FINISHED")
        self._verified_ids.clear()
        if bootstrap_rows:
            self.client.call(
                "enqueue",
                {"work_operation": "bootstrap_and_propose", "rows": bootstrap_rows},
            )
        if commit_rows:
            self.client.call(
                "enqueue", {"work_operation": "commit_and_propose", "rows": commit_rows}
            )
        if tail_rows:
            self.client.call(
                "enqueue", {"work_operation": "finish_tail", "rows": tail_rows}
            )
        self._write_report()
        return {"ok": True, "blocking_on_draft_gpu": False}

    def _work_row(self, request_id: str, state: _Request, *, terminal: bool) -> dict[str, Any]:
        remaining = state.maximum_new_tokens - len(state.generated_token_ids)
        return {
            "request_id": request_id,
            "committed_token_ids": list(state.committed_token_ids),
            "prefix_version": state.prefix_version,
            "prefix_token_sha256": token_prefix_hash(state.committed_token_ids),
            "remaining_output_budget": remaining,
            "eos_token_ids": list(self.eos_token_ids),
            "terminal": terminal,
        }

    def _transition(self, request_id: str, destination: str) -> None:
        state = self.requests[request_id]
        source = state.lifecycle
        state.lifecycle = destination
        self.state_log.append(
            {
                "schema_version": "specrhythm.phase4b-request-state-event.v1",
                "request_id": request_id,
                "source_state": source,
                "destination_state": destination,
                "prefix_version": state.prefix_version,
                "round_id": state.next_round_id,
                "committed_prefix_length": len(state.committed_token_ids),
                "committed_prefix_sha256": token_prefix_hash(state.committed_token_ids),
                "timestamp_ns": time.monotonic_ns(),
            }
        )

    def _write_report(self) -> None:
        if self.tp_rank != 0:
            return
        atomic_write_json(
            self.report_path,
            {
                "schema_version": "specrhythm.phase4b-dual-proposer-report.v1",
                "mode": "dual-batch",
                "proposer_model_parameter_count": 0,
                "proposal_generation_in_target_callback": False,
                "ready_proposal_injection": (
                    "specrhythm.phase4.vllm_dual_scheduler.DualBatchScheduler"
                ),
                "target_callback_blocks_on_draft_gpu": False,
                "one_unverified_proposal_per_request": True,
                "dependency_speculation": False,
                "request_identity": {
                    "stable_key": "frozen workload request_id",
                    "internal_key": "opaque vLLM request_id",
                    "mapping_source": "unique frozen prompt_token_ids",
                    "suffix_parsing": False,
                    "bound_request_count": len(self.internal_to_stable),
                    "bindings": [
                        {
                            "internal_request_id": internal_id,
                            "request_id": stable_id,
                        }
                        for internal_id, stable_id in sorted(
                            self.internal_to_stable.items()
                        )
                    ],
                },
                "request_count": len(self.requests),
                "target_tp": self.tp_world_size,
            },
        )


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required Dual-Batch environment variable is missing: {name}")
    return Path(value)


def _physical_gpu_id() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        values = [item.strip() for item in visible.split(",") if item.strip()]
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank < len(values) and values[local_rank].isdigit():
            return int(values[local_rank])
    return int(os.environ.get("LOCAL_RANK", "0"))


def _logical_generated(values: Sequence[int], maximum: int) -> Tuple[int, ...]:
    return tuple(int(item) for item in values if int(item) >= 0)[:maximum]


def _terminal(values: Sequence[int], maximum: int, eos: Sequence[int]) -> bool:
    return len(values) >= maximum or bool(values and values[-1] in eos)


def _starts_with(values: Sequence[int], prefix: Sequence[int]) -> bool:
    return tuple(values[: len(prefix)]) == tuple(prefix)
