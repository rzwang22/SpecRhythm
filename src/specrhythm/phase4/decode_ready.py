"""Decode-ready provider, manifest, first-forward, and correctness contracts.

This module is dependency-free. GPU adapters supply observations from resident
Target and Draft engines; consumers are forbidden from depending on how those
resident states were produced.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple

from specrhythm.phase4.serial import token_prefix_hash

DECODE_READY_SCHEMA = "specrhythm.phase4b-decode-ready-manifest.v1"
FIRST_FORWARD_SCHEMA = "specrhythm.phase4b-first-target-forward.v1"


@dataclass(frozen=True)
class DecodeReadyRequest:
    request_id: str
    internal_target_request_id: str
    prompt_token_count: int
    prompt_token_ids_sha256: str
    bootstrap_token_id: int
    committed_output_token_count: int
    logical_committed_prefix_count: int
    logical_committed_prefix_sha256: str
    logical_committed_prefix_token_ids: Tuple[int, ...]
    target_materialized_kv_token_count: int
    target_pending_input_token_id: int
    target_pending_input_position: int
    target_num_computed_tokens: int
    target_num_computed_tokens_relation: str
    draft_materialized_kv_token_count: int
    prefix_version: int
    next_round_id: int
    target_decode_ready: bool
    draft_decode_ready: bool
    initial_proposal_generated: bool
    bootstrap_ready_ns: int
    draft_initialization_complete_ns: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["logical_committed_prefix_token_ids"] = list(
            self.logical_committed_prefix_token_ids
        )
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DecodeReadyRequest":
        fields = dict(value)
        fields["logical_committed_prefix_token_ids"] = tuple(
            int(item) for item in value.get("logical_committed_prefix_token_ids", ())
        )
        fields.setdefault("bootstrap_ready_ns", 0)
        fields.setdefault("draft_initialization_complete_ns", 0)
        return cls(**fields)


@dataclass(frozen=True)
class DecodeReadyManifest:
    schema_version: str
    provider_kind: str
    specrhythm_git_commit: str
    vllm_version: str
    vllm_commit: str
    vllm_patch_stack_sha256: Tuple[str, ...]
    target_model_path: str
    target_model_revision: Optional[str]
    draft_model_path: str
    draft_model_revision: Optional[str]
    tokenizer_revision: Optional[str]
    workload_sha256: str
    sampling_configuration_json: str
    batch_invariant_configuration_json: str
    target_physical_gpu_ids: Tuple[int, ...]
    draft_physical_gpu_ids: Tuple[int, ...]
    target_tensor_parallel_size: int
    draft_tensor_parallel_size: int
    setup_start_ns: int
    setup_complete_ns: int
    global_barrier_ns: int
    measurement_start_ns: int
    requests: Tuple[DecodeReadyRequest, ...]
    kv_connector_handoff: bool = False
    performance_result: bool = False
    manifest_sha256: str = ""

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_kind": self.provider_kind,
            "specrhythm_git_commit": self.specrhythm_git_commit,
            "vllm_version": self.vllm_version,
            "vllm_commit": self.vllm_commit,
            "vllm_patch_stack_sha256": list(self.vllm_patch_stack_sha256),
            "models": {
                "target": {
                    "path": self.target_model_path,
                    "revision": self.target_model_revision,
                },
                "draft": {
                    "path": self.draft_model_path,
                    "revision": self.draft_model_revision,
                },
                "tokenizer_revision": self.tokenizer_revision,
            },
            "workload_sha256": self.workload_sha256,
            "sampling_configuration": json.loads(self.sampling_configuration_json),
            "batch_invariant_configuration": json.loads(
                self.batch_invariant_configuration_json
            ),
            "placement": {
                "target_physical_gpu_ids": list(self.target_physical_gpu_ids),
                "draft_physical_gpu_ids": list(self.draft_physical_gpu_ids),
                "target_tensor_parallel_size": self.target_tensor_parallel_size,
                "draft_tensor_parallel_size": self.draft_tensor_parallel_size,
            },
            "timestamps": {
                "setup_start_ns": self.setup_start_ns,
                "setup_complete_ns": self.setup_complete_ns,
                "global_barrier_ns": self.global_barrier_ns,
                "measurement_start_ns": self.measurement_start_ns,
            },
            "requests": [request.to_dict() for request in self.requests],
            "kv_connector_handoff": self.kv_connector_handoff,
            "performance_result": self.performance_result,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "manifest_sha256": self.manifest_sha256}

    def with_hash(self) -> "DecodeReadyManifest":
        return replace(self, manifest_sha256=_payload_sha256(self.payload()))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DecodeReadyManifest":
        models = _mapping(value.get("models"), "models")
        target = _mapping(models.get("target"), "models.target")
        draft = _mapping(models.get("draft"), "models.draft")
        placement = _mapping(value.get("placement"), "placement")
        timestamps = _mapping(value.get("timestamps"), "timestamps")
        requests = value.get("requests")
        if not isinstance(requests, list):
            raise ValueError("decode-ready requests must be a list")
        return cls(
            schema_version=str(value.get("schema_version", "")),
            provider_kind=str(value.get("provider_kind", "")),
            specrhythm_git_commit=str(value.get("specrhythm_git_commit", "")),
            vllm_version=str(value.get("vllm_version", "")),
            vllm_commit=str(value.get("vllm_commit", "")),
            vllm_patch_stack_sha256=tuple(value.get("vllm_patch_stack_sha256", ())),
            target_model_path=str(target.get("path", "")),
            target_model_revision=_optional_string(target.get("revision")),
            draft_model_path=str(draft.get("path", "")),
            draft_model_revision=_optional_string(draft.get("revision")),
            tokenizer_revision=_optional_string(models.get("tokenizer_revision")),
            workload_sha256=str(value.get("workload_sha256", "")),
            sampling_configuration_json=_canonical_json(
                _mapping(value.get("sampling_configuration"), "sampling_configuration")
            ),
            batch_invariant_configuration_json=_canonical_json(
                _mapping(
                    value.get("batch_invariant_configuration"),
                    "batch_invariant_configuration",
                )
            ),
            target_physical_gpu_ids=tuple(placement.get("target_physical_gpu_ids", ())),
            draft_physical_gpu_ids=tuple(placement.get("draft_physical_gpu_ids", ())),
            target_tensor_parallel_size=int(placement.get("target_tensor_parallel_size", 0)),
            draft_tensor_parallel_size=int(placement.get("draft_tensor_parallel_size", 0)),
            setup_start_ns=int(timestamps.get("setup_start_ns", -1)),
            setup_complete_ns=int(timestamps.get("setup_complete_ns", -1)),
            global_barrier_ns=int(timestamps.get("global_barrier_ns", -1)),
            measurement_start_ns=int(timestamps.get("measurement_start_ns", -1)),
            requests=tuple(DecodeReadyRequest.from_dict(row) for row in requests),
            kv_connector_handoff=value.get("kv_connector_handoff") is True,
            performance_result=value.get("performance_result") is True,
            manifest_sha256=str(value.get("manifest_sha256", "")),
        )


@dataclass(frozen=True)
class ResidentSetupObservation:
    request_id: str
    internal_target_request_id: str
    prompt_token_ids: Tuple[int, ...]
    bootstrap_token_id: int
    target_materialized_kv_token_count: int
    target_num_computed_tokens: int
    draft_materialized_kv_token_count: int
    bootstrap_ready_ns: int = 0
    draft_initialization_complete_ns: int = 0


@dataclass(frozen=True)
class DecodeReadyProvenance:
    specrhythm_git_commit: str
    vllm_version: str
    vllm_commit: str
    vllm_patch_stack_sha256: Tuple[str, ...]
    target_model_path: str
    target_model_revision: Optional[str]
    draft_model_path: str
    draft_model_revision: Optional[str]
    tokenizer_revision: Optional[str]
    workload_sha256: str
    sampling_configuration: Mapping[str, Any]
    batch_invariant_configuration: Mapping[str, Any]
    target_physical_gpu_ids: Tuple[int, ...]
    draft_physical_gpu_ids: Tuple[int, ...]
    target_tensor_parallel_size: int
    draft_tensor_parallel_size: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DecodeReadyProvenance":
        return cls(
            specrhythm_git_commit=str(value.get("specrhythm_git_commit", "")),
            vllm_version=str(value.get("vllm_version", "")),
            vllm_commit=str(value.get("vllm_commit", "")),
            vllm_patch_stack_sha256=tuple(value.get("vllm_patch_stack_sha256", ())),
            target_model_path=str(value.get("target_model_path", "")),
            target_model_revision=_optional_string(value.get("target_model_revision")),
            draft_model_path=str(value.get("draft_model_path", "")),
            draft_model_revision=_optional_string(value.get("draft_model_revision")),
            tokenizer_revision=_optional_string(value.get("tokenizer_revision")),
            workload_sha256=str(value.get("workload_sha256", "")),
            sampling_configuration=_mapping(
                value.get("sampling_configuration"), "sampling_configuration"
            ),
            batch_invariant_configuration=_mapping(
                value.get("batch_invariant_configuration"),
                "batch_invariant_configuration",
            ),
            target_physical_gpu_ids=tuple(value.get("target_physical_gpu_ids", ())),
            draft_physical_gpu_ids=tuple(value.get("draft_physical_gpu_ids", ())),
            target_tensor_parallel_size=int(value.get("target_tensor_parallel_size", 0)),
            draft_tensor_parallel_size=int(value.get("draft_tensor_parallel_size", 0)),
        )


class DecodeReadyProvider(Protocol):
    provider_kind: str

    def prepare(
        self,
        observations: Sequence[ResidentSetupObservation],
        provenance: DecodeReadyProvenance,
        *,
        setup_start_ns: int,
        setup_complete_ns: int,
        global_barrier_ns: int,
        measurement_start_ns: int,
    ) -> DecodeReadyManifest:
        ...


class ResidentWarmStartProvider:
    """Build a handoff-neutral manifest from already resident real KV state."""

    provider_kind = "resident-warm-start"

    def prepare(
        self,
        observations: Sequence[ResidentSetupObservation],
        provenance: DecodeReadyProvenance,
        *,
        setup_start_ns: int,
        setup_complete_ns: int,
        global_barrier_ns: int,
        measurement_start_ns: int,
    ) -> DecodeReadyManifest:
        if not observations:
            raise ValueError("resident setup produced no requests")
        if len({row.request_id for row in observations}) != len(observations):
            raise ValueError("resident setup request IDs must be unique")
        requests = []
        for row in observations:
            logical = row.prompt_token_ids + (row.bootstrap_token_id,)
            requests.append(
                DecodeReadyRequest(
                    request_id=row.request_id,
                    internal_target_request_id=row.internal_target_request_id,
                    prompt_token_count=len(row.prompt_token_ids),
                    prompt_token_ids_sha256=token_prefix_hash(row.prompt_token_ids),
                    bootstrap_token_id=row.bootstrap_token_id,
                    committed_output_token_count=1,
                    logical_committed_prefix_count=len(logical),
                    logical_committed_prefix_sha256=token_prefix_hash(logical),
                    logical_committed_prefix_token_ids=logical,
                    target_materialized_kv_token_count=(
                        row.target_materialized_kv_token_count
                    ),
                    target_pending_input_token_id=row.bootstrap_token_id,
                    target_pending_input_position=row.target_materialized_kv_token_count,
                    target_num_computed_tokens=row.target_num_computed_tokens,
                    target_num_computed_tokens_relation=(
                        "scheduler num_computed_tokens equals materialized KV tokens; "
                        "allocated block capacity is not counted"
                    ),
                    draft_materialized_kv_token_count=(
                        row.draft_materialized_kv_token_count
                    ),
                    prefix_version=1,
                    next_round_id=0,
                    target_decode_ready=True,
                    draft_decode_ready=True,
                    initial_proposal_generated=False,
                    bootstrap_ready_ns=row.bootstrap_ready_ns,
                    draft_initialization_complete_ns=(
                        row.draft_initialization_complete_ns
                    ),
                )
            )
        manifest = DecodeReadyManifest(
            schema_version=DECODE_READY_SCHEMA,
            provider_kind=self.provider_kind,
            specrhythm_git_commit=provenance.specrhythm_git_commit,
            vllm_version=provenance.vllm_version,
            vllm_commit=provenance.vllm_commit,
            vllm_patch_stack_sha256=provenance.vllm_patch_stack_sha256,
            target_model_path=provenance.target_model_path,
            target_model_revision=provenance.target_model_revision,
            draft_model_path=provenance.draft_model_path,
            draft_model_revision=provenance.draft_model_revision,
            tokenizer_revision=provenance.tokenizer_revision,
            workload_sha256=provenance.workload_sha256,
            sampling_configuration_json=_canonical_json(
                provenance.sampling_configuration
            ),
            batch_invariant_configuration_json=_canonical_json(
                provenance.batch_invariant_configuration
            ),
            target_physical_gpu_ids=provenance.target_physical_gpu_ids,
            draft_physical_gpu_ids=provenance.draft_physical_gpu_ids,
            target_tensor_parallel_size=provenance.target_tensor_parallel_size,
            draft_tensor_parallel_size=provenance.draft_tensor_parallel_size,
            setup_start_ns=setup_start_ns,
            setup_complete_ns=setup_complete_ns,
            global_barrier_ns=global_barrier_ns,
            measurement_start_ns=measurement_start_ns,
            requests=tuple(requests),
        ).with_hash()
        errors = validate_decode_ready_manifest(manifest)
        if errors:
            raise ValueError("invalid resident decode-ready state: " + "; ".join(errors))
        return manifest


def validate_decode_ready_manifest(manifest: DecodeReadyManifest) -> list[str]:
    errors = []
    if manifest.schema_version != DECODE_READY_SCHEMA:
        errors.append("unsupported decode-ready schema")
    if manifest.provider_kind != "resident-warm-start":
        errors.append("unsupported decode-ready provider")
    if not manifest.specrhythm_git_commit:
        errors.append("SpecRhythm commit is missing")
    if not manifest.vllm_version or not manifest.vllm_commit:
        errors.append("pinned vLLM provenance is missing")
    if not manifest.vllm_patch_stack_sha256 or any(
        len(value) != 64 for value in manifest.vllm_patch_stack_sha256
    ):
        errors.append("ordered vLLM patch hashes are missing or invalid")
    if not manifest.target_model_path or not manifest.draft_model_path:
        errors.append("resident model provenance is missing")
    if len(manifest.workload_sha256) != 64:
        errors.append("workload checksum is invalid")
    if (
        manifest.target_tensor_parallel_size != len(manifest.target_physical_gpu_ids)
        or manifest.draft_tensor_parallel_size != len(manifest.draft_physical_gpu_ids)
    ):
        errors.append("GPU placement and tensor-parallel configuration differ")
    if manifest.kv_connector_handoff:
        errors.append("ResidentWarmStartProvider cannot claim KVConnector handoff")
    if manifest.performance_result:
        errors.append("decode-ready setup cannot be a performance result")
    if not (
        0 <= manifest.setup_start_ns
        <= manifest.setup_complete_ns
        <= manifest.global_barrier_ns
        <= manifest.measurement_start_ns
    ):
        errors.append("setup/barrier/measurement timestamps are not ordered")
    if not manifest.requests:
        errors.append("decode-ready manifest has no requests")
    if len({request.request_id for request in manifest.requests}) != len(
        manifest.requests
    ):
        errors.append("decode-ready request IDs are not unique")
    for request in manifest.requests:
        errors.extend(_validate_request(request))
        if not (
            manifest.setup_start_ns
            <= request.bootstrap_ready_ns
            <= request.draft_initialization_complete_ns
            <= manifest.setup_complete_ns
        ):
            errors.append(
                f"{request.request_id}: per-request setup timestamps are not ordered"
            )
    expected_hash = _payload_sha256(manifest.payload())
    if manifest.manifest_sha256 != expected_hash:
        errors.append("decode-ready manifest hash is invalid")
    return errors


def load_decode_ready_manifest(value: Mapping[str, Any]) -> DecodeReadyManifest:
    manifest = DecodeReadyManifest.from_dict(value)
    errors = validate_decode_ready_manifest(manifest)
    if errors:
        raise ValueError("invalid DecodeReadyManifest: " + "; ".join(errors))
    return manifest


def build_first_target_forward_contract(
    request: DecodeReadyRequest,
    *,
    consumer: str,
    proposal_token_ids: Sequence[int] = (),
    target_forward_start_ns: int,
    target_forward_end_ns: int,
    output_logits_positions: Optional[Sequence[int]] = None,
    accepted_draft_tokens: Optional[int] = None,
    post_forward_committed_token_ids: Sequence[int] = (),
    post_forward_target_kv_token_count: Optional[int] = None,
    post_forward_prefix_version: Optional[int] = None,
) -> dict[str, Any]:
    if consumer not in {"target-only", "serial", "dual-batch"}:
        raise ValueError("unknown decode-ready consumer")
    proposal = tuple(int(item) for item in proposal_token_ids)
    if consumer == "target-only" and proposal:
        raise ValueError("Target-only first decode cannot contain a proposal")
    inputs = (request.bootstrap_token_id,) + proposal
    start = request.target_pending_input_position
    positions = tuple(range(start, start + len(inputs)))
    logits_positions = (
        tuple(int(item) for item in output_logits_positions)
        if output_logits_positions is not None
        else positions
    )
    value = {
        "schema_version": FIRST_FORWARD_SCHEMA,
        "request_id": request.request_id,
        "consumer": consumer,
        "prefix_version_before": request.prefix_version,
        "pre_forward_target_materialized_kv_token_count": (
            request.target_materialized_kv_token_count
        ),
        "pending_input_token_id": request.target_pending_input_token_id,
        "proposal_token_ids": list(proposal),
        "verification_input_token_ids": list(inputs),
        "input_positions": list(positions),
        "output_logits_positions": list(logits_positions),
        "accepted_draft_tokens": accepted_draft_tokens,
        "rejected_draft_tokens": (
            None if accepted_draft_tokens is None else len(proposal) - accepted_draft_tokens
        ),
        "post_forward_committed_token_ids": list(post_forward_committed_token_ids),
        "post_forward_target_kv_token_count": post_forward_target_kv_token_count,
        "prefix_version_after": post_forward_prefix_version,
        "target_forward_start_ns": target_forward_start_ns,
        "target_forward_end_ns": target_forward_end_ns,
    }
    errors = validate_first_target_forward_contract(value, request)
    if errors:
        raise ValueError("invalid first Target forward: " + "; ".join(errors))
    return value


def validate_first_target_forward_contract(
    value: Mapping[str, Any], request: DecodeReadyRequest
) -> list[str]:
    errors = []
    if value.get("schema_version") != FIRST_FORWARD_SCHEMA:
        errors.append("unsupported first-forward schema")
    proposal = tuple(value.get("proposal_token_ids", ()))
    expected_tokens = (request.bootstrap_token_id,) + proposal
    if tuple(value.get("verification_input_token_ids", ())) != expected_tokens:
        errors.append("first Target input omitted or duplicated bootstrap/proposal tokens")
    expected_positions = tuple(
        range(
            request.target_pending_input_position,
            request.target_pending_input_position + len(expected_tokens),
        )
    )
    if tuple(value.get("input_positions", ())) != expected_positions:
        errors.append("first Target input positions are off by one")
    if value.get("pre_forward_target_materialized_kv_token_count") != (
        request.target_materialized_kv_token_count
    ):
        errors.append("pre-forward Target KV count differs from manifest")
    if value.get("pending_input_token_id") != request.bootstrap_token_id:
        errors.append("first Target pending token is not the bootstrap token")
    start = value.get("target_forward_start_ns")
    end = value.get("target_forward_end_ns")
    if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end:
        errors.append("first Target forward timestamps are invalid")
    if value.get("consumer") == "target-only" and proposal:
        errors.append("Target-only first forward contains proposal tokens")
    accepted = value.get("accepted_draft_tokens")
    rejected = value.get("rejected_draft_tokens")
    if accepted is not None:
        if (
            not isinstance(accepted, int)
            or not isinstance(rejected, int)
            or accepted < 0
            or rejected < 0
            or accepted + rejected != len(proposal)
        ):
            errors.append("first Target accepted/rejected mapping is invalid")
    post_kv = value.get("post_forward_target_kv_token_count")
    post_tokens = value.get("post_forward_committed_token_ids")
    if post_kv is not None:
        if not isinstance(post_tokens, list) or not post_tokens:
            errors.append("post-forward committed tokens are missing")
        elif post_kv != request.target_materialized_kv_token_count + len(post_tokens):
            errors.append("post-forward Target KV accounting is invalid")
        if value.get("prefix_version_after") != request.prefix_version + 1:
            errors.append("post-forward prefix version did not advance exactly once")
    return errors


def validate_measurement_boundary(
    manifest: DecodeReadyManifest,
    *,
    first_draft_start_ns: Optional[int] = None,
    first_draft_end_ns: Optional[int] = None,
    first_target_decode_start_ns: Optional[int] = None,
    proposal_created_timestamps_ns: Sequence[int] = (),
) -> list[str]:
    errors = []
    start = manifest.measurement_start_ns
    if any(timestamp < start for timestamp in proposal_created_timestamps_ns):
        errors.append("an initial proposal existed before measurement_start_ns")
    if first_draft_start_ns is not None:
        if first_draft_end_ns is None or not start <= first_draft_start_ns < first_draft_end_ns:
            errors.append("first Draft interval crosses the measurement boundary")
    if (
        first_target_decode_start_ns is not None
        and first_target_decode_start_ns < start
    ):
        errors.append("first Target decode started before measurement_start_ns")
    return errors


def compare_raw_and_decode_outputs(
    raw_rows: Sequence[Mapping[str, Any]],
    decode_rows: Sequence[Mapping[str, Any]],
    manifest: DecodeReadyManifest,
) -> dict[str, Any]:
    raw = _unique_outputs(raw_rows)
    decode = _unique_outputs(decode_rows)
    checks = []
    errors = []
    for request in manifest.requests:
        raw_row = raw.get(request.request_id)
        decode_row = decode.get(request.request_id)
        if raw_row is None or decode_row is None:
            errors.append(f"request {request.request_id} is missing from output comparison")
            continue
        raw_tokens = tuple(raw_row.get("generated_token_ids", ()))
        continuation = tuple(decode_row.get("continuation_token_ids", ()))
        reconstructed = (request.bootstrap_token_id,) + continuation
        row_checks = {
            "generated_token_ids": raw_tokens == reconstructed,
            "bootstrap_token": bool(raw_tokens)
            and raw_tokens[0] == request.bootstrap_token_id,
            "finish_reason": raw_row.get("finish_reason")
            == decode_row.get("finish_reason"),
            "eos_token_id": raw_row.get("eos_token_id")
            == decode_row.get("eos_token_id"),
            "max_token_termination": raw_row.get("max_token_termination")
            == decode_row.get("max_token_termination"),
            "final_logical_length": raw_row.get("final_logical_length")
            == decode_row.get("final_logical_length"),
        }
        if not all(row_checks.values()):
            errors.append(f"request {request.request_id} raw/decode output differs")
        checks.append({"request_id": request.request_id, "checks": row_checks})
    return {
        "schema_version": "specrhythm.phase4b-raw-vs-decode-comparison.v1",
        "valid": not errors,
        "errors": errors,
        "requests": checks,
    }


def compare_decode_consumers(
    target_rows: Sequence[Mapping[str, Any]],
    serial_rows: Sequence[Mapping[str, Any]],
    dual_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    target = _unique_outputs(target_rows)
    serial = _unique_outputs(serial_rows)
    dual = _unique_outputs(dual_rows or ()) if dual_rows is not None else None
    errors = []
    keys = set(target)
    if set(serial) != keys:
        errors.append("Target and Serial request sets differ")
    if dual is not None and set(dual) != keys:
        errors.append("Target and Dual request sets differ")
    fields = ("generated_token_ids", "finish_reason", "eos_token_id", "final_logical_length")
    checks = []
    for request_id in sorted(keys & set(serial)):
        target_serial = all(
            target[request_id].get(key) == serial[request_id].get(key)
            for key in fields
        )
        target_dual = None
        if dual is not None and request_id in dual:
            target_dual = all(
                target[request_id].get(key) == dual[request_id].get(key)
                for key in fields
            )
        if not target_serial or target_dual is False:
            errors.append(f"decode consumers differ for {request_id}")
        checks.append(
            {
                "request_id": request_id,
                "target_equals_serial": target_serial,
                "target_equals_dual": target_dual,
            }
        )
    return {
        "schema_version": "specrhythm.phase4b-decode-consumer-comparison.v1",
        "valid": not errors,
        "errors": errors,
        "dual_evaluated": dual is not None,
        "requests": checks,
    }


def run_decode_ready_contract_dry_run() -> dict[str, Any]:
    provenance = DecodeReadyProvenance(
        specrhythm_git_commit="cpu-dry-run",
        vllm_version="0.25.1",
        vllm_commit="752a3a504485790a2e8491cacbb35c137339ad34",
        vllm_patch_stack_sha256=("a" * 64, "b" * 64, "c" * 64),
        target_model_path="/models/target",
        target_model_revision=None,
        draft_model_path="/models/draft",
        draft_model_revision=None,
        tokenizer_revision=None,
        workload_sha256="d" * 64,
        sampling_configuration={"temperature": 0.0, "seed": 1664},
        batch_invariant_configuration={"requested": True, "enable_dbo": False},
        target_physical_gpu_ids=(1, 2),
        draft_physical_gpu_ids=(0,),
        target_tensor_parallel_size=2,
        draft_tensor_parallel_size=1,
    )
    observation = ResidentSetupObservation(
        request_id="dry-run",
        internal_target_request_id="opaque-0",
        prompt_token_ids=(1, 2, 3),
        bootstrap_token_id=4,
        target_materialized_kv_token_count=3,
        target_num_computed_tokens=3,
        draft_materialized_kv_token_count=4,
        bootstrap_ready_ns=12,
        draft_initialization_complete_ns=18,
    )
    manifest = ResidentWarmStartProvider().prepare(
        [observation],
        provenance,
        setup_start_ns=10,
        setup_complete_ns=20,
        global_barrier_ns=30,
        measurement_start_ns=40,
    )
    target = build_first_target_forward_contract(
        manifest.requests[0],
        consumer="target-only",
        target_forward_start_ns=41,
        target_forward_end_ns=42,
    )
    serial = build_first_target_forward_contract(
        manifest.requests[0],
        consumer="serial",
        proposal_token_ids=(5, 6),
        target_forward_start_ns=43,
        target_forward_end_ns=44,
    )
    return {
        "schema_version": "specrhythm.phase4b-decode-ready-dry-run.v1",
        "gpu_execution_performed": False,
        "performance_result": False,
        "manifest": manifest.to_dict(),
        "manifest_validation": {
            "valid": not validate_decode_ready_manifest(manifest),
            "errors": validate_decode_ready_manifest(manifest),
        },
        "first_target_forward_contracts": [target, serial],
        "measurement_boundary": {
            "valid": not validate_measurement_boundary(
                manifest,
                first_draft_start_ns=43,
                first_draft_end_ns=44,
                first_target_decode_start_ns=41,
            )
        },
        "kv_connector_implemented": False,
        "dual_gpu_outcome_claimed": False,
    }


def _validate_request(request: DecodeReadyRequest) -> list[str]:
    prefix = request.logical_committed_prefix_token_ids
    errors = []
    if not request.request_id or not request.internal_target_request_id:
        errors.append("decode-ready request identity is empty")
    if request.prompt_token_count + 1 != request.logical_committed_prefix_count:
        errors.append(f"{request.request_id}: logical prefix is not prompt+bootstrap")
    if len(prefix) != request.logical_committed_prefix_count:
        errors.append(f"{request.request_id}: logical prefix token count differs")
    if token_prefix_hash(prefix) != request.logical_committed_prefix_sha256:
        errors.append(f"{request.request_id}: logical prefix hash differs")
    if token_prefix_hash(prefix[:-1]) != request.prompt_token_ids_sha256:
        errors.append(f"{request.request_id}: prompt hash differs")
    if not prefix or prefix[-1] != request.bootstrap_token_id:
        errors.append(f"{request.request_id}: pending token is not logical prefix[-1]")
    if request.committed_output_token_count != 1:
        errors.append(f"{request.request_id}: setup must commit exactly one output token")
    if request.target_materialized_kv_token_count + 1 != (
        request.logical_committed_prefix_count
    ):
        errors.append(f"{request.request_id}: Target KV + 1 invariant failed")
    if request.target_pending_input_position != request.target_materialized_kv_token_count:
        errors.append(f"{request.request_id}: pending Target position is invalid")
    if request.target_pending_input_token_id != request.bootstrap_token_id:
        errors.append(f"{request.request_id}: pending Target token differs")
    if request.target_num_computed_tokens != request.target_materialized_kv_token_count:
        errors.append(f"{request.request_id}: scheduler computed-token count differs from KV")
    if request.draft_materialized_kv_token_count != request.logical_committed_prefix_count:
        errors.append(f"{request.request_id}: Draft KV does not cover the logical prefix")
    if request.prefix_version != 1 or request.next_round_id != 0:
        errors.append(f"{request.request_id}: initial prefix/round is invalid")
    if not request.target_decode_ready or not request.draft_decode_ready:
        errors.append(f"{request.request_id}: resident engines are not decode-ready")
    if request.initial_proposal_generated:
        errors.append(f"{request.request_id}: proposal was generated during setup")
    return errors


def _unique_outputs(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {}
    for row in rows:
        request_id = str(row.get("request_id", ""))
        if not request_id or request_id in result:
            raise ValueError("output rows require unique non-empty request IDs")
        result[request_id] = row
    return result


def _payload_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _optional_string(value: Any) -> Optional[str]:
    return str(value) if value is not None else None
