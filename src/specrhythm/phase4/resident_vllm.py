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
from specrhythm.phase4.resident_setup import (
    IncrementalResidentSetup,
    build_setup_ready,
    load_setup_control,
    observation_static_fields,
    observation_to_dict,
)
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
        self.setup_control_path = _required_path("SR_PHASE4_RESIDENT_SETUP_CONTROL")
        self.setup_ready_path = _required_path("SR_PHASE4_RESIDENT_SETUP_READY")
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
        self.setup_tracker: Optional[IncrementalResidentSetup] = None
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
                self._rank_zero_observe_setup(
                    request_ids, num_tokens_no_spec, token_ids_cpu
                )
                if self.tp_rank == 0
                else None
            )
            setup = self.tp_group.broadcast_object(setup, src=0)
            if not isinstance(setup, Mapping) or setup.get("valid") is not True:
                raise RuntimeError("incremental resident setup observation failed")
            if setup.get("complete") is True:
                self._complete_global_setup(setup)
        return [[] for _ in request_ids]

    def _rank_zero_observe_setup(
        self, request_ids: Sequence[str], num_tokens_no_spec: Any, token_ids_cpu: Any
    ) -> dict[str, Any]:
        if not request_ids:
            raise RuntimeError("resident setup callback cannot be empty")
        tracker = self._setup_tracker()
        for index, internal_id in enumerate(request_ids):
            count = int(num_tokens_no_spec[index])
            tokens = tuple(int(item) for item in token_ids_cpu[index, :count].tolist())
            stable_id = self.identity.bind(str(internal_id), tokens)
            definition = self.definitions[stable_id]
            generated = tokens[len(definition.prompt_token_ids) :]
            if len(generated) != 1:
                raise RuntimeError("resident setup must sample exactly one bootstrap token")
            logical = definition.prompt_token_ids + (generated[0],)
            existing = tracker.get(stable_id)
            if existing is not None:
                candidate = ResidentSetupObservation(
                    request_id=stable_id,
                    internal_target_request_id=str(internal_id),
                    prompt_token_ids=definition.prompt_token_ids,
                    bootstrap_token_id=generated[0],
                    target_materialized_kv_token_count=len(
                        definition.prompt_token_ids
                    ),
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
                        f"resident bootstrap identity/prefix changed for {stable_id}"
                    )
                tracker.record(existing)
                self._log_observation(existing, duplicate=True)
                continue
            bootstrap_ready_ns = time.monotonic_ns()
            initialized = self.client.call(
                "initialize",
                {
                    "request_id": stable_id,
                    "committed_token_ids": list(logical),
                    "committed_prefix_hash": token_prefix_hash(logical),
                },
            )
            if (
                initialized.get("committed_prefix_len") != len(logical)
                or initialized.get("committed_prefix_hash")
                != token_prefix_hash(logical)
            ):
                raise RuntimeError("Draft did not materialize the complete committed prefix")
            observation = ResidentSetupObservation(
                request_id=stable_id,
                internal_target_request_id=str(internal_id),
                prompt_token_ids=definition.prompt_token_ids,
                bootstrap_token_id=generated[0],
                target_materialized_kv_token_count=len(definition.prompt_token_ids),
                target_num_computed_tokens=len(definition.prompt_token_ids),
                draft_materialized_kv_token_count=len(logical),
                bootstrap_ready_ns=bootstrap_ready_ns,
                draft_initialization_complete_ns=time.monotonic_ns(),
            )
            tracker.record(observation)
            self._log_observation(observation, duplicate=False)
        if not tracker.complete:
            self._write_report()
            return {
                "valid": True,
                "complete": False,
                "expected_request_count": self.expected_request_count,
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
        self.timing_log.append(
            {
                "schema_version": "specrhythm.phase4b-decode-ready-timing.v1",
                "event": "pre-barrier-validation",
                "timestamp_ns": setup_complete_ns,
                "valid": not errors,
                "errors": errors,
                "initial_proposal_generated": False,
                "observed_request_ids": list(tracker.observed_request_ids),
            }
        )
        if errors:
            raise RuntimeError("resident setup validation failed: " + "; ".join(errors))
        return {
            "valid": True,
            "complete": True,
            "setup_start_ns": tracker.setup_start_ns,
            "setup_complete_ns": setup_complete_ns,
            "observations": [
                observation_to_dict(row) for row in tracker.observations
            ],
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
            raise RuntimeError("resident measurement boundary was not broadcast")
        if self.tp_rank == 0:
            manifest = self._write_manifest(
                setup,
                barrier_ns=barrier_ns,
                measurement_start_ns=measurement_start_ns,
            )
            ready_published_ns = time.monotonic_ns()
            ready = build_setup_ready(
                manifest,
                consumer="target-only",
                manifest_path=self.manifest_path,
                ready_published_ns=ready_published_ns,
            )
            atomic_write_json(self.setup_ready_path, ready)
            self.timing_log.append(
                {
                    "schema_version": "specrhythm.phase4b-decode-ready-timing.v1",
                    "event": "global-setup-ready-published",
                    "timestamp_ns": ready_published_ns,
                    "consumer": "target-only",
                    "request_count": self.expected_request_count,
                    "initial_proposal_generated": False,
                }
            )
            published: Optional[Mapping[str, Any]] = ready
        else:
            published = None
        published = self.tp_group.broadcast_object(published, src=0)
        if (
            not isinstance(published, Mapping)
            or published.get("global_decode_ready") is not True
        ):
            raise RuntimeError("resident global setup-ready publication was not broadcast")
        self.setup_complete = True
        self.measurement_start_ns = measurement_start_ns
        self._write_report()

    def _setup_tracker(self) -> IncrementalResidentSetup:
        if self.setup_tracker is None:
            expected = tuple(self.definitions)
            control = load_setup_control(
                self.setup_control_path,
                consumer="target-only",
                expected_request_ids=expected,
            )
            self.setup_tracker = IncrementalResidentSetup(
                expected, int(control["setup_start_ns"])
            )
        return self.setup_tracker

    def _log_observation(
        self, observation: ResidentSetupObservation, *, duplicate: bool
    ) -> None:
        self.timing_log.append(
            {
                "schema_version": "specrhythm.phase4b-decode-ready-timing.v1",
                "event": "bootstrap-draft-ready",
                "timestamp_ns": observation.draft_initialization_complete_ns,
                "bootstrap_ready_ns": observation.bootstrap_ready_ns,
                "draft_initialization_complete_ns": (
                    observation.draft_initialization_complete_ns
                ),
                "request_id": observation.request_id,
                "internal_target_request_id": (
                    observation.internal_target_request_id
                ),
                "duplicate_identical": duplicate,
                "initial_proposal_generated": False,
            }
        )

    def _write_manifest(
        self,
        setup: Mapping[str, Any],
        *,
        barrier_ns: int,
        measurement_start_ns: int,
    ) -> Any:
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
        return manifest

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
                "setup_observed_request_count": (
                    len(self.setup_tracker.observations)
                    if self.setup_tracker is not None
                    else 0
                ),
                "measurement_start_ns": self.measurement_start_ns,
                "target_tp_world_size": self.tp_world_size,
            },
        )


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required for resident decode-ready execution")
    return Path(value).resolve()
