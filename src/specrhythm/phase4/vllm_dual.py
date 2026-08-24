"""Pinned-vLLM custom proposer observer for Phase-4B Dual-Batch.

Proposal generation is never performed in this Target callback.  Rank zero
only enqueues committed-prefix work to the asynchronous GPU-0 Draft service;
the custom scheduler consumes already-ready proposals independently.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from specrhythm.phase4.decode_ready import (
    DecodeReadyProvenance,
    ResidentSetupObservation,
    ResidentWarmStartProvider,
    validate_decode_ready_manifest,
)
from specrhythm.phase4.dual import DualProposal
from specrhythm.phase4.dual_service import DualDraftClient
from specrhythm.phase4.manifest import atomic_write_json
from specrhythm.phase4.request_identity import FrozenPromptIdentityMap
from specrhythm.phase4.resident_setup import (
    IncrementalResidentSetup,
    build_setup_ready,
    load_setup_control,
    observation_static_fields,
    observation_to_dict,
)
from specrhythm.phase4.serial import greedy_acceptance, token_prefix_hash
from specrhythm.phase4.stock_vllm import (
    active_cuda_device_identity,
    load_smoke_requests,
)
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
        self.resident_decode_ready = os.environ.get("SR_PHASE4_DUAL_RESIDENT") == "1"
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
        self.setup_complete = False
        self.setup_tracker: Optional[IncrementalResidentSetup] = None
        self.measurement_start_ns: Optional[int] = None
        if self.resident_decode_ready:
            self.manifest_path = _required_path("SR_PHASE4_DECODE_READY_MANIFEST")
            self.setup_control_path = _required_path("SR_PHASE4_RESIDENT_SETUP_CONTROL")
            self.setup_ready_path = _required_path("SR_PHASE4_RESIDENT_SETUP_READY")
            self.timing_log = CheckpointJsonl(
                _required_path("SR_PHASE4_DECODE_READY_TIMING_EVENTS")
            )
            context = json.loads(
                _required_path("SR_PHASE4_DECODE_READY_CONTEXT").read_text(
                    encoding="utf-8"
                )
            )
            if not isinstance(context, Mapping):
                raise RuntimeError("decode-ready context must be an object")
            self.provenance = DecodeReadyProvenance.from_dict(context)
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
        if self.resident_decode_ready and not self.setup_complete:
            setup = (
                self._rank_zero_observe_setup(
                    request_ids, num_tokens_no_spec, token_ids_cpu
                )
                if self.tp_rank == 0
                else None
            )
            setup = self.tp_group.broadcast_object(setup, src=0)
            if not isinstance(setup, Mapping) or setup.get("valid") is not True:
                raise RuntimeError("incremental resident Dual setup observation failed")
            if setup.get("complete") is True:
                self._complete_global_setup(setup)
            return [[] for _ in request_ids]
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

    def _rank_zero_observe_setup(
        self, request_ids: Sequence[str], num_tokens_no_spec: Any, token_ids_cpu: Any
    ) -> dict[str, Any]:
        tracker = self._setup_tracker()
        for index, internal_id_value in enumerate(request_ids):
            internal_id = str(internal_id_value)
            count = int(num_tokens_no_spec[index])
            tokens = tuple(int(item) for item in token_ids_cpu[index, :count].tolist())
            stable_id = self.identity.bind(internal_id, tokens)
            definition = self.definitions[stable_id]
            generated = tokens[len(definition.prompt_token_ids) :]
            if len(generated) != 1:
                raise RuntimeError("resident Dual setup must sample one bootstrap token")
            logical = definition.prompt_token_ids + (generated[0],)
            existing = tracker.get(stable_id)
            if existing is not None:
                candidate = ResidentSetupObservation(
                    request_id=stable_id,
                    internal_target_request_id=internal_id,
                    prompt_token_ids=definition.prompt_token_ids,
                    bootstrap_token_id=generated[0],
                    target_materialized_kv_token_count=len(definition.prompt_token_ids),
                    target_num_computed_tokens=len(definition.prompt_token_ids),
                    draft_materialized_kv_token_count=len(logical),
                    bootstrap_ready_ns=existing.bootstrap_ready_ns,
                    draft_initialization_complete_ns=(
                        existing.draft_initialization_complete_ns
                    ),
                )
                if observation_static_fields(candidate) != observation_static_fields(
                    existing
                ):
                    raise RuntimeError(
                        f"resident Dual bootstrap changed for {stable_id}"
                    )
                tracker.record(existing)
                continue
            bootstrap_ready_ns = time.monotonic_ns()
            terminal = _terminal(
                generated, definition.maximum_new_tokens, self.eos_token_ids
            )
            initialized = self.client.call(
                "execute",
                {
                    "work_operation": "initialize",
                    "row": {
                        "request_id": stable_id,
                        "committed_token_ids": list(logical),
                        "prefix_version": 1,
                        "prefix_token_sha256": token_prefix_hash(logical),
                        "terminal": terminal,
                    },
                },
            )
            if (
                initialized.get("logical_draft_kv_length") != len(logical)
                or initialized.get("committed_prefix_hash")
                != token_prefix_hash(logical)
                or initialized.get("initial_proposal_generated") is not False
            ):
                raise RuntimeError("Draft did not materialize decode-ready state exactly")
            state = _Request(
                prompt_token_ids=definition.prompt_token_ids,
                maximum_new_tokens=definition.maximum_new_tokens,
                committed_token_ids=logical,
                generated_token_ids=generated,
                terminal=terminal,
            )
            self.requests[stable_id] = state
            self._transition(
                stable_id,
                "TERMINAL" if terminal else "DRAFT_READY",
                reason="decode-ready-bootstrap-complete",
            )
            observation = ResidentSetupObservation(
                request_id=stable_id,
                internal_target_request_id=internal_id,
                prompt_token_ids=definition.prompt_token_ids,
                bootstrap_token_id=generated[0],
                target_materialized_kv_token_count=len(definition.prompt_token_ids),
                target_num_computed_tokens=len(definition.prompt_token_ids),
                draft_materialized_kv_token_count=len(logical),
                bootstrap_ready_ns=bootstrap_ready_ns,
                draft_initialization_complete_ns=time.monotonic_ns(),
            )
            tracker.record(observation)
            self.timing_log.append(
                {
                    "schema_version": "specrhythm.phase4b-decode-ready-timing.v1",
                    "event": "bootstrap-draft-ready",
                    "timestamp_ns": observation.draft_initialization_complete_ns,
                    "request_id": stable_id,
                    "internal_target_request_id": internal_id,
                    "initial_proposal_generated": False,
                }
            )
        if not tracker.complete:
            self._write_report()
            return {
                "valid": True,
                "complete": False,
                "observed_request_ids": list(tracker.observed_request_ids),
            }
        setup_complete_ns = time.monotonic_ns()
        provisional = ResidentWarmStartProvider().prepare(
            tracker.observations,
            self.provenance,
            setup_start_ns=tracker.setup_start_ns,
            setup_complete_ns=setup_complete_ns,
            global_barrier_ns=setup_complete_ns,
            measurement_start_ns=setup_complete_ns,
        )
        errors = validate_decode_ready_manifest(provisional)
        if errors:
            raise RuntimeError("resident Dual setup validation failed: " + "; ".join(errors))
        return {
            "valid": True,
            "complete": True,
            "setup_start_ns": tracker.setup_start_ns,
            "setup_complete_ns": setup_complete_ns,
            "observations": [observation_to_dict(row) for row in tracker.observations],
        }

    def _complete_global_setup(self, setup: Mapping[str, Any]) -> None:
        self.tp_group.barrier()
        self.torch.cuda.synchronize()
        barrier_ns = time.monotonic_ns()
        measurement_start_ns = time.monotonic_ns() if self.tp_rank == 0 else None
        measurement_start_ns = self.tp_group.broadcast_object(
            measurement_start_ns, src=0
        )
        if not isinstance(measurement_start_ns, int):
            raise RuntimeError("resident Dual measurement boundary was not broadcast")
        if self.tp_rank == 0:
            observations = setup.get("observations")
            if not isinstance(observations, list):
                raise RuntimeError("resident Dual setup observations are missing")
            manifest = ResidentWarmStartProvider().prepare(
                [ResidentSetupObservation.from_dict(row) for row in observations],
                self.provenance,
                setup_start_ns=int(setup["setup_start_ns"]),
                setup_complete_ns=int(setup["setup_complete_ns"]),
                global_barrier_ns=barrier_ns,
                measurement_start_ns=measurement_start_ns,
            )
            atomic_write_json(self.manifest_path, manifest.to_dict())
            ready = build_setup_ready(
                manifest,
                consumer="dual-batch",
                manifest_path=self.manifest_path,
                ready_published_ns=time.monotonic_ns(),
            )
            atomic_write_json(self.setup_ready_path, ready)
            self.timing_log.append(
                {
                    "schema_version": "specrhythm.phase4b-decode-ready-timing.v1",
                    "event": "measurement-start",
                    "timestamp_ns": measurement_start_ns,
                    "global_barrier_ns": barrier_ns,
                    "manifest_sha256": manifest.manifest_sha256,
                    "initial_proposal_generated": False,
                }
            )
            initial_rows = []
            for request_id, state in self.requests.items():
                if state.terminal:
                    continue
                self._transition(
                    request_id,
                    "DRAFTING",
                    reason="post-measurement-initial-proposal",
                )
                initial_rows.append(
                    {
                        **self._work_row(request_id, state, terminal=False),
                        "measurement_start_ns": measurement_start_ns,
                    }
                )
            if initial_rows:
                self.client.call(
                    "enqueue",
                    {"work_operation": "propose_only", "rows": initial_rows},
                )
            published: Optional[Mapping[str, Any]] = ready
        else:
            published = None
        published = self.tp_group.broadcast_object(published, src=0)
        if (
            not isinstance(published, Mapping)
            or published.get("global_decode_ready") is not True
        ):
            raise RuntimeError("resident Dual setup-ready publication was not broadcast")
        self.setup_complete = True
        self.measurement_start_ns = measurement_start_ns
        self._write_report()

    def _setup_tracker(self) -> IncrementalResidentSetup:
        if self.setup_tracker is None:
            expected = tuple(self.definitions)
            control = load_setup_control(
                self.setup_control_path,
                consumer="dual-batch",
                expected_request_ids=expected,
            )
            self.setup_tracker = IncrementalResidentSetup(
                expected, int(control["setup_start_ns"])
            )
        return self.setup_tracker

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
                if state.lifecycle == "DRAFT_SYNC":
                    sync_complete_ns = result.get("draft_sync_complete_ns")
                    if (
                        not isinstance(sync_complete_ns, int)
                        or sync_complete_ns > proposal.draft_start_ns
                    ):
                        raise RuntimeError(
                            "next Draft proposal lacks an ordered sync boundary"
                        )
                    self._transition(
                        stable_id,
                        "DRAFT_READY",
                        proposal_id=proposal.proposal_id,
                        reason="draft-sync-complete",
                        timestamp_ns=sync_complete_ns,
                    )
                    self._transition(
                        stable_id,
                        "DRAFTING",
                        proposal_id=proposal.proposal_id,
                        reason="asynchronous-draft-work-observed",
                        timestamp_ns=proposal.draft_start_ns,
                    )
                self._transition(
                    stable_id,
                    "PROPOSAL_READY",
                    proposal_id=proposal.proposal_id,
                    reason="proposal-published",
                )
                self._transition(
                    stable_id,
                    "VERIFY_READY",
                    proposal_id=proposal.proposal_id,
                    reason="proposal-installed",
                )
                self._transition(
                    stable_id,
                    "VERIFYING",
                    proposal_id=proposal.proposal_id,
                    reason="target-verification-start",
                )
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
        identity = active_cuda_device_identity(self.torch)
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
                    "logical_cuda_index": identity["logical_cuda_index"],
                    "physical_gpu_id": identity["physical_gpu_id"],
                    "gpu_uuid": identity["gpu_uuid"],
                    "cuda_visible_devices": identity["cuda_visible_devices"],
                    "device_identity_source": (
                        "active-cuda-device-plus-visible-physical-binding"
                    ),
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
                identity_errors = validate_target_rank_identity(
                    evidence, self.tp_world_size
                )
                if identity_errors:
                    raise RuntimeError(
                        "invalid Target verify TP identity evidence: "
                        + "; ".join(identity_errors)
                    )
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
                        "target_device_identity_semantics": (
                            "per-rank-active-cuda-device-cross-checked-by-runner-worker-snapshot"
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
                    self._transition(request_id, "TERMINAL", reason="bootstrap-terminal")
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
                        "accepted_draft_token_ids": list(
                            decision.accepted_draft_token_ids
                        ),
                        "rejected_draft_token_ids": list(
                            decision.rejected_draft_token_ids
                        ),
                        "accepted_draft_tokens": len(decision.accepted_draft_token_ids),
                        "rejected_draft_tokens": len(decision.rejected_draft_token_ids),
                        "target_correction_token_ids": list(
                            decision.target_correction_token_ids
                        ),
                        "target_bonus_token_ids": list(decision.target_bonus_token_ids),
                        "committed_token_ids": list(decision.committed_token_ids),
                        "terminal": terminal,
                        "terminal_truncation_reason": (
                            "max_tokens"
                            if terminal
                            and len(generated) >= state.maximum_new_tokens
                            else "eos"
                            if terminal
                            and generated
                            and generated[-1] in self.eos_token_ids
                            else None
                        ),
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
                    self._transition(
                        request_id,
                        "TERMINAL",
                        proposal_id=proposal.proposal_id,
                        reason="verified-terminal-commit",
                    )
                else:
                    self._transition(
                        request_id,
                        "DRAFT_SYNC",
                        proposal_id=proposal.proposal_id,
                        reason="commit-awaiting-draft-sync",
                    )
            elif generated != state.generated_token_ids:
                # A proposal-free one-token tail is the only legal Target-only
                # decode after bootstrap.
                delta = prefix[len(state.committed_token_ids) :]
                if state.pending_proposal is not None or len(delta) != 1 or not terminal:
                    raise RuntimeError("unproposed Target decode advanced a live request")
                claimed = self.client.call(
                    "claimed", {"request_ids": [request_id]}
                ).get("claimed")
                if not isinstance(claimed, list) or len(claimed) != 1:
                    raise RuntimeError("Target tail lacks claimed Draft readiness evidence")
                tail_result = claimed[0]
                if (
                    not isinstance(tail_result, Mapping)
                    or tail_result.get("target_tail") is not True
                    or tail_result.get("proposal") is not None
                ):
                    raise RuntimeError("unproposed Target decode lacks a legal Draft tail")
                tail_ready_ns = tail_result.get("target_tail_ready_ns")
                if not isinstance(tail_ready_ns, int):
                    raise RuntimeError("Target tail readiness timestamp is missing")
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
                self._transition(
                    request_id,
                    "TARGET_TAIL_READY",
                    reason="proposal-free-terminal-tail-ready",
                    timestamp_ns=tail_ready_ns,
                )
                self._transition(
                    request_id,
                    "VERIFYING",
                    reason="target-tail-verification-start",
                )
                self._transition(
                    request_id, "COMMITTING", reason="target-tail-commit"
                )
                self._transition(
                    request_id, "TERMINAL", reason="target-tail-terminal"
                )
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

    def _transition(
        self,
        request_id: str,
        destination: str,
        *,
        proposal_id: Optional[str] = None,
        reason: str = "runtime-transition",
        timestamp_ns: Optional[int] = None,
    ) -> None:
        state = self.requests[request_id]
        source = state.lifecycle
        legal = {
            "BOOTSTRAP": {"DRAFT_READY", "TERMINAL"},
            "DRAFT_READY": {"DRAFTING", "TARGET_TAIL_READY", "TERMINAL"},
            "DRAFTING": {"PROPOSAL_READY", "TARGET_TAIL_READY", "FAILED"},
            "PROPOSAL_READY": {"VERIFY_READY", "DROPPED_STALE", "FAILED"},
            "VERIFY_READY": {"VERIFYING", "DROPPED_STALE", "FAILED"},
            "VERIFYING": {"COMMITTING", "FAILED"},
            "COMMITTING": {"DRAFT_SYNC", "TERMINAL", "FAILED"},
            "DRAFT_SYNC": {"DRAFT_READY", "TARGET_TAIL_READY", "TERMINAL", "FAILED"},
            "TARGET_TAIL_READY": {"VERIFYING", "TERMINAL", "FAILED"},
            "TERMINAL": set(),
            "FAILED": set(),
        }
        if destination not in legal.get(source, set()):
            raise RuntimeError(f"illegal Dual request transition {source}->{destination}")
        state.lifecycle = destination
        self.state_log.append(
            {
                "schema_version": "specrhythm.phase4b-request-state-event.v1",
                "request_id": request_id,
                "internal_request_id": self.stable_to_internal.get(request_id),
                "source_state": source,
                "destination_state": destination,
                "prefix_version": state.prefix_version,
                "round_id": state.next_round_id,
                "committed_prefix_length": len(state.committed_token_ids),
                "committed_prefix_sha256": token_prefix_hash(state.committed_token_ids),
                "proposal_id": proposal_id,
                "reason": reason,
                "timestamp_ns": timestamp_ns or time.monotonic_ns(),
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
                "resident_decode_ready": self.resident_decode_ready,
                "setup_complete": self.setup_complete,
                "measurement_start_ns": self.measurement_start_ns,
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


def validate_target_rank_identity(
    rows: Sequence[Mapping[str, Any]],
    tensor_parallel_size: int,
    authoritative_worker_rows: Sequence[Mapping[str, Any]] = (),
) -> list[str]:
    """Validate per-forward TP identities and optional worker-snapshot agreement."""

    errors = []
    if len(rows) != tensor_parallel_size:
        errors.append("Target verify rank count differs from TP size")
    if {row.get("tp_rank") for row in rows} != set(range(tensor_parallel_size)):
        errors.append("Target verify TP ranks are incomplete")
    if len({row.get("global_rank") for row in rows}) != len(rows):
        errors.append("Target verify global ranks are not unique")
    physical = [row.get("physical_gpu_id") for row in rows]
    if any(not isinstance(item, int) for item in physical):
        errors.append("Target verify physical GPU identity is missing")
    elif len(set(physical)) != len(rows):
        errors.append("Target verify TP ranks alias one physical GPU")
    uuids = [str(row.get("gpu_uuid", "")) for row in rows]
    if any(not item for item in uuids):
        errors.append("Target verify GPU UUID is missing")
    elif len(set(uuids)) != len(rows):
        errors.append("Target verify TP ranks report the same GPU UUID")
    if any(
        row.get("cuda_events") is not True
        or row.get("cuda_synchronized") is not True
        for row in rows
    ):
        errors.append("Target verify rank timing lacks synchronized CUDA events")
    if authoritative_worker_rows:
        workers = {row.get("global_rank"): row for row in authoritative_worker_rows}
        for row in rows:
            worker = workers.get(row.get("global_rank"))
            if worker is None:
                errors.append("Target verify rank lacks authoritative worker evidence")
                continue
            if (
                row.get("physical_gpu_id") != worker.get("physical_gpu_id")
                or row.get("gpu_uuid") != worker.get("gpu_uuid")
                or row.get("logical_cuda_index")
                != worker.get("logical_cuda_index")
            ):
                errors.append(
                    "Target verify logical/physical identity disagrees with worker evidence"
                )
    return list(dict.fromkeys(errors))


def _logical_generated(values: Sequence[int], maximum: int) -> Tuple[int, ...]:
    return tuple(int(item) for item in values if int(item) >= 0)[:maximum]


def _terminal(values: Sequence[int], maximum: int, eos: Sequence[int]) -> bool:
    return len(values) >= maximum or bool(values and values[-1] in eos)


def _starts_with(values: Sequence[int], prefix: Sequence[int]) -> bool:
    return tuple(values[: len(prefix)]) == tuple(prefix)
