"""Explicit bookkeeping-output to physical InputBatch row projection."""

from __future__ import annotations

import hashlib
import json
import operator
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

ROW_CONTEXT_SCHEMA = "specrhythm.vllm-sampled-row-context.v1"


class PhysicalTokenRows:
    """Small projected CPU rows with the indexing used by the existing observer."""

    def __init__(self, rows: Sequence[Sequence[int]]) -> None:
        self.rows = tuple(tuple(row) for row in rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, key: Any) -> Any:
        index, section = key
        return _TokenSlice(self.rows[index][section])


class _TokenSlice(tuple):
    def tolist(self) -> list[int]:
        return list(self)


@dataclass(frozen=True)
class SampledRows:
    request_ids: tuple[str, ...]
    sampled_tokens: tuple[tuple[int, ...], ...]
    physical_indices: tuple[int, ...]
    physical_tokens: tuple[tuple[int, ...], ...]
    materialized_counts: Optional[tuple[int, ...]]

    @property
    def logical_signature(self) -> str:
        # Physical slots may differ between ranks; the logical projection may not.
        payload = (self.request_ids, self.sampled_tokens, self.physical_tokens,
                   self.materialized_counts)
        return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def _ids(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) or not item for item in value
    ) or len(set(value)) != len(value):
        raise ValueError(f"{name} must contain unique nonempty request IDs")
    return tuple(value)


def _indices(value: Any, ids: Sequence[str], name: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(ids):
        raise ValueError(f"{name} request mapping is missing or inconsistent")
    if any(type(value[key]) is not int or value[key] != index for index, key in enumerate(ids)):
        raise ValueError(f"{name} request mapping disagrees with authoritative row order")
    return dict(value)


def align_sampled_rows(
    context: Mapping[str, Any], sampled_token_ids: Sequence[Sequence[int]],
    num_tokens_no_spec: Any, token_ids_cpu: Any, *,
    physical_request_ids: Sequence[str],
    target_materialized_token_counts: Optional[Sequence[int]],
) -> SampledRows:
    if not isinstance(context, Mapping) or context.get("schema_version") != ROW_CONTEXT_SCHEMA:
        raise ValueError("Dual requires the pinned sampled-row-context vLLM patch")
    sampled_ids = _ids(context.get("sampled_request_ids"), "sampled_request_ids")
    physical_ids = _ids(context.get("physical_request_ids"), "physical_request_ids")
    _indices(context.get("req_id_to_sampled_index"), sampled_ids, "bookkeeping-output")
    physical_map = _indices(context.get("req_id_to_physical_index"), physical_ids, "physical")
    if tuple(physical_request_ids) != physical_ids:
        raise ValueError("physical request IDs changed after the row-context snapshot")
    scheduled = _ids(context.get("scheduled_request_ids"), "scheduled_request_ids")
    verified = _ids(context.get("scheduled_spec_request_ids"), "scheduled_spec_request_ids")
    if set(sampled_ids) != set(scheduled) or not set(verified) <= set(sampled_ids):
        raise ValueError("sampled request mapping differs from scheduled request evidence")
    if len(sampled_ids) != len(sampled_token_ids):
        raise ValueError(
            f"sampled/ID row count mismatch: sampled={len(sampled_token_ids)}, "
            f"output_ids={len(sampled_ids)}"
        )
    if len(num_tokens_no_spec) < len(physical_ids) or len(token_ids_cpu) < len(physical_ids):
        raise ValueError("physical storage capacity does not cover mapped rows")
    if target_materialized_token_counts is not None and (
        len(target_materialized_token_counts) != len(physical_ids)
    ):
        raise ValueError("materialized counts do not cover physical request rows")
    indices, tokens, materialized, sampled = [], [], [], []
    for sampled_index, request_id in enumerate(sampled_ids):
        if request_id not in physical_map:
            raise ValueError("sampled request is missing its physical row mapping")
        physical_index = physical_map[request_id]
        count = operator.index(num_tokens_no_spec[physical_index])
        if count < 0:
            raise ValueError("physical token count is negative")
        row = tuple(int(token) for token in token_ids_cpu[physical_index, :count].tolist())
        if len(row) != count:
            raise ValueError("physical token count exceeds row storage")
        delta = tuple(sampled_token_ids[sampled_index])
        if any(type(token) is not int or token < 0 for token in delta):
            raise ValueError("valid sampled tokens must be non-negative integers")
        indices.append(physical_index)
        tokens.append(row)
        sampled.append(delta)
        if target_materialized_token_counts is not None:
            materialized.append(operator.index(target_materialized_token_counts[physical_index]))
    return SampledRows(
        sampled_ids, tuple(sampled), tuple(indices), tuple(tokens),
        tuple(materialized) if target_materialized_token_counts is not None else None,
    )
