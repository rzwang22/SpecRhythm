"""Incremental resident setup and cross-process readiness contracts.

This module is dependency-free.  TP worker rank zero records bootstrap/Draft
observations and atomically publishes readiness.  The EngineCore scheduler
consumes that artifact; the two processes never rely on shared Python state.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from specrhythm.phase4.decode_ready import (
    DecodeReadyManifest,
    ResidentSetupObservation,
    load_decode_ready_manifest,
)
from specrhythm.phase4.manifest import sha256_file
from specrhythm.phase4.serial import Proposal

SETUP_CONTROL_SCHEMA = "specrhythm.phase4b-resident-setup-control.v1"
SETUP_READY_SCHEMA = "specrhythm.phase4b-resident-setup-ready.v1"
ADMISSION_EVENT_SCHEMA = "specrhythm.phase4b-resident-admission-event.v1"
RESIDENT_CONSUMERS = {"target-only", "serial", "dual-batch"}


class IncrementalResidentSetup:
    """Collect one immutable setup observation per frozen stable request."""

    def __init__(self, expected_request_ids: Sequence[str], setup_start_ns: int) -> None:
        expected = tuple(str(item) for item in expected_request_ids)
        if not expected or len(set(expected)) != len(expected):
            raise ValueError("resident setup requires unique expected request IDs")
        if setup_start_ns <= 0:
            raise ValueError("resident setup start timestamp must be positive")
        self.expected_request_ids = expected
        self.setup_start_ns = setup_start_ns
        self._observations: dict[str, ResidentSetupObservation] = {}
        self._completion_transition_count = 0

    @property
    def complete(self) -> bool:
        return set(self._observations) == set(self.expected_request_ids)

    @property
    def observed_request_ids(self) -> tuple[str, ...]:
        return tuple(
            request_id
            for request_id in self.expected_request_ids
            if request_id in self._observations
        )

    @property
    def observations(self) -> tuple[ResidentSetupObservation, ...]:
        return tuple(self._observations[item] for item in self.observed_request_ids)

    @property
    def completion_transition_count(self) -> int:
        return self._completion_transition_count

    def get(self, request_id: str) -> Optional[ResidentSetupObservation]:
        return self._observations.get(str(request_id))

    def record(self, observation: ResidentSetupObservation) -> bool:
        """Record an observation; return False for an identical duplicate."""

        request_id = observation.request_id
        if request_id not in self.expected_request_ids:
            raise RuntimeError(f"unexpected resident setup request: {request_id}")
        _validate_observation_timestamps(observation, self.setup_start_ns)
        previous = self._observations.get(request_id)
        if previous is not None:
            if previous != observation:
                raise RuntimeError(
                    f"resident bootstrap observation changed for {request_id}"
                )
            return False
        was_complete = self.complete
        self._observations[request_id] = observation
        if not was_complete and self.complete:
            self._completion_transition_count += 1
        return True


def build_setup_control(
    *, consumer: str, expected_request_ids: Sequence[str], setup_start_ns: int
) -> dict[str, Any]:
    if consumer not in RESIDENT_CONSUMERS:
        raise ValueError("resident setup consumer must be target-only, serial, or dual-batch")
    request_ids = [str(item) for item in expected_request_ids]
    if not request_ids or len(set(request_ids)) != len(request_ids):
        raise ValueError("resident setup control requires unique request IDs")
    if setup_start_ns <= 0:
        raise ValueError("resident setup control timestamp must be positive")
    return {
        "schema_version": SETUP_CONTROL_SCHEMA,
        "consumer": consumer,
        "expected_request_ids": request_ids,
        "setup_start_ns": setup_start_ns,
    }


def load_setup_control(
    path: Path, *, consumer: str, expected_request_ids: Sequence[str]
) -> dict[str, Any]:
    value = _read_object(path)
    errors = []
    if value.get("schema_version") != SETUP_CONTROL_SCHEMA:
        errors.append("unsupported resident setup control schema")
    if value.get("consumer") != consumer:
        errors.append("resident setup control consumer differs")
    if value.get("expected_request_ids") != [str(item) for item in expected_request_ids]:
        errors.append("resident setup control request set/order differs")
    if not isinstance(value.get("setup_start_ns"), int) or value["setup_start_ns"] <= 0:
        errors.append("resident setup control timestamp is invalid")
    if errors:
        raise RuntimeError("invalid resident setup control: " + "; ".join(errors))
    return value


def build_setup_ready(
    manifest: DecodeReadyManifest,
    *,
    consumer: str,
    manifest_path: Path,
    initial_proposals: Sequence[Proposal] = (),
    ready_published_ns: int,
) -> dict[str, Any]:
    if consumer not in RESIDENT_CONSUMERS:
        raise ValueError(
            "resident readiness consumer must be target-only, serial, or dual-batch"
        )
    if ready_published_ns < manifest.measurement_start_ns:
        raise ValueError("resident readiness preceded measurement_start")
    proposals = tuple(initial_proposals)
    proposal_ids = {proposal.request_id for proposal in proposals}
    expected_ids = {request.request_id for request in manifest.requests}
    if consumer in {"target-only", "dual-batch"} and proposals:
        raise ValueError(f"{consumer} readiness cannot contain proposals")
    if consumer == "serial" and proposal_ids != expected_ids:
        raise ValueError("Serial readiness requires one initial proposal per request")
    if len(proposal_ids) != len(proposals):
        raise ValueError("resident readiness proposal IDs must be unique")
    for proposal in proposals:
        if proposal.round_id != 0:
            raise ValueError("initial resident proposal must be round zero")
        if proposal.draft_start_ns < manifest.measurement_start_ns:
            raise ValueError("initial Serial proposal predates measurement_start")
        request = next(
            row for row in manifest.requests if row.request_id == proposal.request_id
        )
        if proposal.parent_prefix_len != request.logical_committed_prefix_count:
            raise ValueError("initial proposal parent length differs from manifest")
        if proposal.parent_prefix_hash != request.logical_committed_prefix_sha256:
            raise ValueError("initial proposal parent hash differs from manifest")
    payload = {
        "schema_version": SETUP_READY_SCHEMA,
        "consumer": consumer,
        "global_decode_ready": True,
        "expected_request_ids": [row.request_id for row in manifest.requests],
        "observed_request_ids": [row.request_id for row in manifest.requests],
        "stable_to_internal_request_id": {
            row.request_id: row.internal_target_request_id for row in manifest.requests
        },
        "manifest_file": manifest_path.name,
        "manifest_sha256": manifest.manifest_sha256,
        "manifest_file_sha256": sha256_file(manifest_path),
        "setup_complete_ns": manifest.setup_complete_ns,
        "global_barrier_ns": manifest.global_barrier_ns,
        "measurement_start_ns": manifest.measurement_start_ns,
        "ready_published_ns": ready_published_ns,
        "initial_proposals": [proposal.to_dict() for proposal in proposals],
        "initial_proposal_generated_before_measurement": False,
    }
    return {**payload, "artifact_sha256": _payload_sha256(payload)}


def load_setup_ready(
    path: Path,
    *,
    manifest_path: Path,
    consumer: str,
    expected_request_ids: Sequence[str],
) -> dict[str, Any]:
    value = _read_object(path)
    errors = validate_setup_ready(
        value,
        manifest_path=manifest_path,
        consumer=consumer,
        expected_request_ids=expected_request_ids,
    )
    if errors:
        raise RuntimeError("invalid resident setup readiness: " + "; ".join(errors))
    return value


def validate_setup_ready(
    value: Mapping[str, Any],
    *,
    manifest_path: Path,
    consumer: str,
    expected_request_ids: Sequence[str],
) -> list[str]:
    errors = []
    expected = [str(item) for item in expected_request_ids]
    if value.get("schema_version") != SETUP_READY_SCHEMA:
        errors.append("unsupported setup-ready schema")
    if value.get("consumer") != consumer:
        errors.append("setup-ready consumer differs")
    if value.get("global_decode_ready") is not True:
        errors.append("global decode readiness is not true")
    if value.get("expected_request_ids") != expected:
        errors.append("setup-ready expected request IDs differ")
    if value.get("observed_request_ids") != expected:
        errors.append("setup-ready observations are incomplete or reordered")
    if not manifest_path.is_file():
        errors.append("decode-ready manifest is missing")
        return errors
    try:
        manifest = load_decode_ready_manifest(_read_object(manifest_path))
    except (OSError, RuntimeError, ValueError) as error:
        errors.append(f"decode-ready manifest is invalid: {error}")
        return errors
    if value.get("manifest_file") != manifest_path.name:
        errors.append("setup-ready manifest filename differs")
    if value.get("manifest_sha256") != manifest.manifest_sha256:
        errors.append("setup-ready manifest payload hash differs")
    if value.get("manifest_file_sha256") != sha256_file(manifest_path):
        errors.append("setup-ready manifest file hash differs")
    for name in (
        "setup_complete_ns",
        "global_barrier_ns",
        "measurement_start_ns",
    ):
        if value.get(name) != getattr(manifest, name):
            errors.append(f"setup-ready {name} differs from manifest")
    published = value.get("ready_published_ns")
    if not isinstance(published, int) or published < manifest.measurement_start_ns:
        errors.append("setup-ready publication timestamp is invalid")
    if value.get("initial_proposal_generated_before_measurement") is not False:
        errors.append("setup-ready claims an early initial proposal")
    mappings = value.get("stable_to_internal_request_id")
    expected_mappings = {
        row.request_id: row.internal_target_request_id for row in manifest.requests
    }
    if mappings != expected_mappings:
        errors.append("setup-ready stable/internal request mapping differs")
    proposals_value = value.get("initial_proposals")
    if not isinstance(proposals_value, list):
        errors.append("setup-ready initial proposals are not a list")
        proposals_value = []
    proposals = []
    for row in proposals_value:
        if not isinstance(row, Mapping):
            errors.append("setup-ready initial proposal is not an object")
            continue
        try:
            proposals.append(Proposal.from_dict(row))
        except (TypeError, ValueError) as error:
            errors.append(f"setup-ready initial proposal is invalid: {error}")
    proposal_ids = [proposal.request_id for proposal in proposals]
    if consumer in {"target-only", "dual-batch"} and proposals:
        errors.append(f"{consumer} setup-ready contains proposals")
    if consumer == "serial" and proposal_ids != expected:
        errors.append("Serial setup-ready proposal request set/order differs")
    for proposal in proposals:
        request = next(
            (row for row in manifest.requests if row.request_id == proposal.request_id),
            None,
        )
        if request is None:
            errors.append("setup-ready proposal belongs to an unknown request")
            continue
        if proposal.round_id != 0:
            errors.append("setup-ready initial proposal is not round zero")
        if proposal.draft_start_ns < manifest.measurement_start_ns:
            errors.append("setup-ready initial proposal predates measurement_start")
        if (
            proposal.parent_prefix_len != request.logical_committed_prefix_count
            or proposal.parent_prefix_hash != request.logical_committed_prefix_sha256
        ):
            errors.append("setup-ready initial proposal parent prefix differs")
    payload = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _payload_sha256(payload):
        errors.append("setup-ready artifact hash is invalid")
    return errors


def resident_admission_decision(
    *,
    num_output_tokens: int,
    global_decode_ready: bool,
    consumer: str,
    has_initial_proposal: bool,
) -> tuple[bool, str]:
    """Return the explicit setup/timed-decode scheduler decision."""

    if num_output_tokens < 0:
        raise ValueError("resident request output-token count cannot be negative")
    if consumer not in RESIDENT_CONSUMERS:
        raise ValueError("resident setup consumer must be target-only, serial, or dual-batch")
    if num_output_tokens == 0:
        return True, "setup-prefill-bootstrap"
    if not global_decode_ready:
        return False, "bootstrap-ready-awaiting-global-boundary"
    if consumer == "serial" and num_output_tokens == 1 and not has_initial_proposal:
        return False, "serial-initial-proposal-not-installed"
    if consumer == "dual-batch" and num_output_tokens == 1 and not has_initial_proposal:
        return False, "dual-initial-proposal-not-installed"
    return True, "global-decode-ready"


def validate_resident_admission_events(
    rows: Sequence[Mapping[str, Any]], *, consumer: str
) -> list[str]:
    errors = []
    frozen_ids = set()
    released_ids = set()
    for index, row in enumerate(rows):
        label = f"resident admission row {index}"
        if row.get("schema_version") != ADMISSION_EVENT_SCHEMA:
            errors.append(f"{label} has an unsupported schema")
            continue
        if row.get("consumer") != consumer:
            errors.append(f"{label} has the wrong consumer")
        output_count = row.get("num_output_tokens")
        ready = row.get("global_decode_ready") is True
        admissible = row.get("admissible") is True
        scheduled = row.get("scheduled") is True
        request_id = str(row.get("request_id", ""))
        if not isinstance(output_count, int) or output_count < 0 or not request_id:
            errors.append(f"{label} has invalid request state")
            continue
        if output_count >= 1 and not ready:
            frozen_ids.add(request_id)
            if admissible or scheduled:
                errors.append(f"{label} advanced after bootstrap before global readiness")
        if output_count >= 1 and ready:
            released_ids.add(request_id)
            measurement_start = row.get("measurement_start_ns")
            timestamp = row.get("timestamp_ns")
            if (
                not isinstance(measurement_start, int)
                or not isinstance(timestamp, int)
                or timestamp < measurement_start
            ):
                errors.append(f"{label} was released before measurement_start")
        if (
            consumer == "serial"
            and output_count == 1
            and ready
            and row.get("initial_proposal_installed") is not True
        ):
            errors.append(f"{label} released Serial without an initial proposal")
        if (
            consumer == "dual-batch"
            and output_count == 1
            and ready
            and row.get("initial_proposal_installed") is not True
            and row.get("reason") != "dual-initial-proposal-not-installed"
        ):
            errors.append(f"{label} released Dual without an initial proposal")
        if row.get("explicit_request_predicate") is not True:
            errors.append(f"{label} did not use the explicit request predicate")
        if row.get("current_step_arithmetic") is not False:
            errors.append(f"{label} used current_step arithmetic")
    if rows and not released_ids:
        errors.append("resident admission evidence contains no post-boundary release")
    return errors


def observation_to_dict(observation: ResidentSetupObservation) -> dict[str, Any]:
    return observation.to_dict()


def observation_static_fields(observation: ResidentSetupObservation) -> tuple[Any, ...]:
    return (
        observation.request_id,
        observation.internal_target_request_id,
        observation.prompt_token_ids,
        observation.bootstrap_token_id,
        observation.target_materialized_kv_token_count,
        observation.target_num_computed_tokens,
        observation.draft_materialized_kv_token_count,
    )


def _validate_observation_timestamps(
    observation: ResidentSetupObservation, setup_start_ns: int
) -> None:
    if not (
        setup_start_ns
        <= observation.bootstrap_ready_ns
        <= observation.draft_initialization_complete_ns
    ):
        raise RuntimeError(
            f"resident setup timestamps are invalid for {observation.request_id}"
        )


def _payload_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value
