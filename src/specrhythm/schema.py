"""Canonical workload records and JSONL serialization."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Union


@dataclass(frozen=True)
class WorkloadRequest:
    """One request entering the decode scheduler.

    arrival_time_ms is relative time. slo_tpot_ms applies to end-to-end decode time from arrival,
    including queueing before active-set admission. Acceptance probability and draft confidence
    are separate Phase-A calibration inputs, not properties inferred from token lengths.
    """

    request_id: str
    arrival_time_ms: float
    input_tokens: int
    output_tokens: int
    slo_tpot_ms: float
    task: str = "unknown"
    model: str = "default"
    client_id: str = "client-0"
    conversation_id: Optional[str] = None
    turn_index: Optional[int] = None
    acceptance_probability: float = 0.7
    metadata: dict[str, Any] = field(default_factory=dict)
    draft_confidence: float = 0.7

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be a non-empty string")
        if (
            not isinstance(self.arrival_time_ms, (int, float))
            or isinstance(self.arrival_time_ms, bool)
            or not math.isfinite(self.arrival_time_ms)
            or self.arrival_time_ms < 0
        ):
            raise ValueError("arrival_time_ms must be finite and non-negative")
        if (
            not isinstance(self.input_tokens, int)
            or isinstance(self.input_tokens, bool)
            or self.input_tokens < 1
            or not isinstance(self.output_tokens, int)
            or isinstance(self.output_tokens, bool)
            or self.output_tokens < 1
        ):
            raise ValueError("input_tokens and output_tokens must be positive integers")
        if (
            not isinstance(self.slo_tpot_ms, (int, float))
            or isinstance(self.slo_tpot_ms, bool)
            or not math.isfinite(self.slo_tpot_ms)
            or self.slo_tpot_ms <= 0
        ):
            raise ValueError("slo_tpot_ms must be finite and positive")
        if (
            not isinstance(self.acceptance_probability, (int, float))
            or isinstance(self.acceptance_probability, bool)
            or not math.isfinite(self.acceptance_probability)
            or not 0 <= self.acceptance_probability <= 1
        ):
            raise ValueError("acceptance_probability must be finite and in [0, 1]")
        if (
            not isinstance(self.draft_confidence, (int, float))
            or isinstance(self.draft_confidence, bool)
            or not math.isfinite(self.draft_confidence)
            or not 0 <= self.draft_confidence <= 1
        ):
            raise ValueError("draft_confidence must be finite and in [0, 1]")
        if self.turn_index is not None and (
            not isinstance(self.turn_index, int)
            or isinstance(self.turn_index, bool)
            or self.turn_index < 0
        ):
            raise ValueError("turn_index must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> WorkloadRequest:
        return cls(**value)


@dataclass
class Workload:
    requests: list[WorkloadRequest]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = [request.request_id for request in self.requests]
        if len(ids) != len(set(ids)):
            raise ValueError("request_id values must be unique")
        self.requests.sort(key=lambda request: (request.arrival_time_ms, request.request_id))

    def save_jsonl(self, path: Union[str, Path]) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for request in self.requests:
                handle.write(json.dumps(request.to_dict(), sort_keys=True) + "\n")

    @classmethod
    def load_jsonl(cls, path: Union[str, Path]) -> Workload:
        source = Path(path)
        with source.open(encoding="utf-8") as handle:
            requests = [
                WorkloadRequest.from_dict(json.loads(line)) for line in handle if line.strip()
            ]
        return cls(requests=requests, metadata={"source": str(source)})

    @classmethod
    def from_requests(cls, requests: Iterable[WorkloadRequest]) -> Workload:
        return cls(list(requests))
