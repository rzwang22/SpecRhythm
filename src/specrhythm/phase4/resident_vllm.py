"""Pinned-vLLM resident warm-start Target-only consumer plugin."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from specrhythm.phase4.decode_ready import (
    DecodeReadyProvenance,
    ResidentSetupObservation,
    ResidentWarmStartProvider,
    validate_decode_ready_manifest,
)
from specrhythm.phase4.manifest import atomic_write_json
from specrhythm.phase4.request_identity import FrozenPromptIdentityMap
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.stock_vllm import load_smoke_requests
from specrhythm.phase4.transport import CheckpointJsonl, UnixDraftClient


class ResidentTargetProposer:
    """Return no proposals while preserving Target and Draft decode-ready KV."""

    def __init__(self, vllm_config: Any) -> None:
        try:
            import torch
            from vllm.distributed.parallel_state import get_tp_group
        except ImportError as error:
            raise RuntimeError("ResidentTargetProposer requires pinned vLLM") from error
        self.torch = torch
        self.tp_group = get_tp_group()
        self.tp_rank = int(self.tp_group.rank_in_group)
        self.tp_world_size = int(self.tp_group.world_size)
        if self.tp_world_size != 2:
            raise RuntimeError("resident Target consumer requires TP=2")
        count = int(os.environ.get("SR_PHASE4_REQUEST_COUNT", "5"))
        definitions = load_smoke_requests(
            _required_path("SR_PHASE4_WORKLOAD"),
            count,
            require_task_mixture=count == 5,
        )
        self.definitions = {row.request_id: row for row in definitions}
        self.identity = FrozenPromptIdentityMap.from_definitions(definitions)
        self.client = UnixDraftClient(_required_path("SR_PHASE4_DRAFT_SOCKET"))
        self.manifest_path = _required_path("SR_PHASE4_DECODE_READY_MANIFEST")
        self.timing_log = CheckpointJsonl(
            _required_path("SR_PHASE4_DECODE_READY_TIMING_EVENTS")
        )
        self.report_path = _required_path("SR_PHASE4_PLUGIN_REPORT")
        context = json.loads(
            _required_path("SR_PHASE4_DECODE_READY_CONTEXT").read_text(encoding="utf-8")
        )
        if not isinstance(context, Mapping):
            raise RuntimeError("decode-ready context must be an object")
        self.provenance = DecodeReadyProvenance.from_dict(context)
        self.expected_request_count = count
        self.setup_complete = False
        self.measurement_start_ns: Optional[int] = None
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
            raise RuntimeError("resident Target requires the request-identity worker hook")
        if not self.setup_complete:
            setup = (
                self._rank_zero_setup(request_ids, num_tokens_no_spec, token_ids_cpu)
                if self.tp_rank == 0
                else None
            )
            setup = self.tp_group.broadcast_object(setup, src=0)
            if not isinstance(setup, Mapping) or setup.get("valid") is not True:
                raise RuntimeError("resident setup did not validate before the TP barrier")
            self.tp_group.barrier()
            self.torch.cuda.synchronize()
            barrier_ns = time.monotonic_ns()
            measurement_start_ns = (
                time.monotonic_ns() if self.tp_rank == 0 else None
            )
            measurement_start_ns = self.tp_group.broadcast_object(
                measurement_start_ns, src=0
            )
            if not isinstance(measurement_start_ns, int):
                raise RuntimeError("resident measurement boundary was not broadcast")
            if self.tp_rank == 0:
                self._write_manifest(
                    setup,
                    barrier_ns=barrier_ns,
                    measurement_start_ns=measurement_start_ns,
                )
            self.tp_group.barrier()
            self.setup_complete = True
            self.measurement_start_ns = measurement_start_ns
            self._write_report()
        return [[] for _ in request_ids]

    def _rank_zero_setup(
        self, request_ids: Sequence[str], num_tokens_no_spec: Any, token_ids_cpu: Any
    ) -> dict[str, Any]:
        if len(request_ids) != self.expected_request_count:
            raise RuntimeError(
                "resident setup requires every frozen request in one initial prefill batch"
            )
        setup_start_ns = int(os.environ.get("SR_PHASE4_SETUP_START_NS", "0"))
        if setup_start_ns <= 0:
            raise RuntimeError("resident setup start timestamp is missing")
        observations = []
        for index, internal_id in enumerate(request_ids):
            count = int(num_tokens_no_spec[index])
            tokens = tuple(int(item) for item in token_ids_cpu[index, :count].tolist())
            stable_id = self.identity.bind(str(internal_id), tokens)
            definition = self.definitions[stable_id]
            generated = tokens[len(definition.prompt_token_ids) :]
            if len(generated) != 1:
                raise RuntimeError("resident setup must sample exactly one bootstrap token")
            logical = definition.prompt_token_ids + (generated[0],)
            initialized = self.client.call(
                "initialize",
                {
                    "request_id": stable_id,
                    "committed_token_ids": list(logical),
                    "committed_prefix_hash": token_prefix_hash(logical),
                },
            )
            if initialized.get("committed_prefix_len") != len(logical):
                raise RuntimeError("Draft did not materialize the complete committed prefix")
            observations.append(
                {
                    "request_id": stable_id,
                    "internal_target_request_id": str(internal_id),
                    "prompt_token_ids": list(definition.prompt_token_ids),
                    "bootstrap_token_id": generated[0],
                    "target_materialized_kv_token_count": len(
                        definition.prompt_token_ids
                    ),
                    "target_num_computed_tokens": len(definition.prompt_token_ids),
                    "draft_materialized_kv_token_count": len(logical),
                }
            )
        setup_complete_ns = time.monotonic_ns()
        # Validate all token/KV invariants before allowing the TP barrier. The
        # final immutable manifest receives the actual barrier/start timestamps.
        provisional = ResidentWarmStartProvider().prepare(
            [ResidentSetupObservation(**row) for row in observations],
            self.provenance,
            setup_start_ns=setup_start_ns,
            setup_complete_ns=setup_complete_ns,
            global_barrier_ns=setup_complete_ns,
            measurement_start_ns=setup_complete_ns,
        )
        errors = validate_decode_ready_manifest(provisional)
        self.timing_log.append(
            {
                "schema_version": "specrhythm.phase4b-decode-ready-timing.v1",
                "event": "pre-barrier-validation",
                "timestamp_ns": setup_complete_ns,
                "valid": not errors,
                "errors": errors,
                "initial_proposal_generated": False,
            }
        )
        if errors:
            raise RuntimeError("resident setup validation failed: " + "; ".join(errors))
        return {
            "valid": True,
            "setup_start_ns": setup_start_ns,
            "setup_complete_ns": setup_complete_ns,
            "observations": observations,
        }

    def _write_manifest(
        self,
        setup: Mapping[str, Any],
        *,
        barrier_ns: int,
        measurement_start_ns: int,
    ) -> None:
        observations = setup.get("observations")
        if not isinstance(observations, list):
            raise RuntimeError("resident setup observations are missing")
        manifest = ResidentWarmStartProvider().prepare(
            [ResidentSetupObservation(**row) for row in observations],
            self.provenance,
            setup_start_ns=int(setup["setup_start_ns"]),
            setup_complete_ns=int(setup["setup_complete_ns"]),
            global_barrier_ns=barrier_ns,
            measurement_start_ns=measurement_start_ns,
        )
        atomic_write_json(self.manifest_path, manifest.to_dict())
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

    def _write_report(self) -> None:
        if self.tp_rank != 0:
            return
        atomic_write_json(
            self.report_path,
            {
                "schema_version": "specrhythm.phase4b-resident-target-proposer.v1",
                "provider_kind": "resident-warm-start",
                "consumer": "target-only",
                "proposal_generation": False,
                "proposer_model_parameter_count": 0,
                "target_logits_observed": False,
                "setup_complete": self.setup_complete,
                "measurement_start_ns": self.measurement_start_ns,
                "target_tp_world_size": self.tp_world_size,
            },
        )


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required for resident decode-ready execution")
    return Path(value).resolve()
