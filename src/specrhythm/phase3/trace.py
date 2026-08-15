"""Versioned real-model trace schema with durable, non-overwriting checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


def _finite(name: str, value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _probability(name: str, value: Any) -> float:
    result = _finite(name, value)
    if not 0 <= result <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _nonnegative_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class RequestCycle:
    request_id: str
    cycle_id: int
    prompt_length: int
    context_length: int
    generated_tokens: tuple[int, ...]
    slo_class: str
    draft_model: str
    target_model: str
    random_seed: int
    sampling_configuration: dict[str, Any]
    mode: str

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        _nonnegative_int("cycle_id", self.cycle_id)
        _nonnegative_int("prompt_length", self.prompt_length)
        _nonnegative_int("context_length", self.context_length)
        if self.context_length != self.prompt_length + len(self.generated_tokens):
            raise ValueError("context_length must equal prompt plus generated token count")
        if any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in self.generated_tokens
        ):
            raise ValueError("generated_tokens must contain non-negative integer token IDs")
        if not self.slo_class:
            raise ValueError("SLO_class must not be empty")
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise ValueError("random_seed must be an integer")
        if self.mode not in {"draft-only", "target-only", "serial"}:
            raise ValueError("invalid Phase-3 trace mode")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["SLO_class"] = value.pop("slo_class")
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RequestCycle":
        return cls(
            request_id=str(value["request_id"]),
            cycle_id=value["cycle_id"],
            prompt_length=value["prompt_length"],
            context_length=value["context_length"],
            generated_tokens=tuple(value.get("generated_tokens", ())),
            slo_class=str(value["SLO_class"]),
            draft_model=str(value["draft_model"]),
            target_model=str(value["target_model"]),
            random_seed=value["random_seed"],
            sampling_configuration=dict(value["sampling_configuration"]),
            mode=str(value["mode"]),
        )


@dataclass(frozen=True)
class CandidateNodeRecord:
    stable_node_id: str
    parent_id: Optional[str]
    depth: int
    token_id: int
    local_probability: float
    path_probability: float
    draft_logit: float
    entropy: float
    top1_top2_margin: float
    sibling_rank: int
    prefix_closed: bool
    selected_for_verification: bool

    def __post_init__(self) -> None:
        if not self.stable_node_id:
            raise ValueError("stable_node_id must not be empty")
        if self.parent_id == "":
            raise ValueError("parent_id must be null or non-empty")
        if _nonnegative_int("candidate depth", self.depth) < 1:
            raise ValueError("candidate depth must be positive")
        _nonnegative_int("candidate token_id", self.token_id)
        _probability("local_probability", self.local_probability)
        _probability("path_probability", self.path_probability)
        _finite("draft_logit", self.draft_logit)
        if _finite("entropy", self.entropy) < 0:
            raise ValueError("entropy must be non-negative")
        if _finite("top1_top2_margin", self.top1_top2_margin) < 0:
            raise ValueError("top1_top2_margin must be non-negative")
        _nonnegative_int("sibling_rank", self.sibling_rank)
        if not self.prefix_closed:
            raise ValueError("materialized candidate nodes must be prefix closed")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateNodeRecord":
        return cls(**dict(value))


@dataclass(frozen=True)
class TargetOutcomeRecord:
    stable_node_id: str
    target_token_id: int
    target_log_probability: float
    on_target_path: bool
    accepted: bool
    committed: bool
    accepted_prefix_length: int

    def __post_init__(self) -> None:
        if not self.stable_node_id:
            raise ValueError("target outcome stable_node_id must not be empty")
        _nonnegative_int("target_token_id", self.target_token_id)
        _finite("target_log_probability", self.target_log_probability)
        _nonnegative_int("accepted_prefix_length", self.accepted_prefix_length)
        if self.committed and not self.accepted:
            raise ValueError("a committed candidate must be accepted")
        if self.accepted and not self.on_target_path:
            raise ValueError("an accepted candidate must be on the target path")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TargetOutcomeRecord":
        return cls(**dict(value))


@dataclass(frozen=True)
class CycleAccounting:
    request_roots: int
    search_pool_nodes: int
    verified_candidate_nodes: int
    accepted_candidate_nodes: int
    committed_candidate_tokens: int
    committed_target_tokens: int

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _nonnegative_int(name, value)
        if self.request_roots != 1:
            raise ValueError("each per-request cycle must account for exactly one root")
        if self.accepted_candidate_nodes > self.verified_candidate_nodes:
            raise ValueError("accepted candidates cannot exceed verified candidates")
        if self.committed_candidate_tokens != self.accepted_candidate_nodes:
            raise ValueError("accepted candidate nodes must commit exactly once")
        if self.committed_target_tokens > 1:
            raise ValueError("a cycle can commit at most one target root/bonus token")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CycleAccounting":
        return cls(**dict(value))


@dataclass(frozen=True)
class SelectorCandidate:
    """Draft-side features available to a deployable, non-oracle selector."""

    stable_node_id: str
    parent_id: Optional[str]
    depth: int
    token_id: int
    local_probability: float
    path_probability: float
    draft_logit: float
    entropy: float
    top1_top2_margin: float
    sibling_rank: int
    prefix_closed: bool


@dataclass(frozen=True)
class SelectorCycleView:
    request: RequestCycle
    candidates: tuple[SelectorCandidate, ...]


@dataclass(frozen=True)
class RealTraceRecord:
    request: RequestCycle
    candidate_nodes: tuple[CandidateNodeRecord, ...]
    target_outcomes: tuple[TargetOutcomeRecord, ...]
    accounting: CycleAccounting
    root_target_token_id: Optional[int]
    committed_token_ids: tuple[int, ...]
    request_finished: bool
    schema_version: str = "specrhythm.real-trace.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "specrhythm.real-trace.v1":
            raise ValueError("unsupported real-trace schema version")
        if self.root_target_token_id is not None:
            _nonnegative_int("root_target_token_id", self.root_target_token_id)
        if any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in self.committed_token_ids
        ):
            raise ValueError("committed_token_ids must contain non-negative integer IDs")
        by_id = {node.stable_node_id: node for node in self.candidate_nodes}
        if len(by_id) != len(self.candidate_nodes):
            raise ValueError("candidate stable_node_id values must be unique per cycle")
        for node in self.candidate_nodes:
            if node.parent_id is None:
                if node.depth != 1:
                    raise ValueError("only depth-one candidates may have no parent")
            else:
                parent = by_id.get(node.parent_id)
                if parent is None or parent.depth + 1 != node.depth:
                    raise ValueError("candidate forest has a missing or invalid parent")
                if node.path_probability > parent.path_probability + 1e-12:
                    raise ValueError("candidate path probability increases below its parent")
            if node.selected_for_verification and node.parent_id is not None:
                if not by_id[node.parent_id].selected_for_verification:
                    raise ValueError("verified candidate selection is not prefix closed")
        outcomes = {outcome.stable_node_id: outcome for outcome in self.target_outcomes}
        if len(outcomes) != len(self.target_outcomes):
            raise ValueError("target outcomes must be unique per candidate")
        if not set(outcomes).issubset(by_id):
            raise ValueError("target outcome references an unknown candidate")
        for outcome in self.target_outcomes:
            candidate = by_id[outcome.stable_node_id]
            if outcome.accepted and not candidate.selected_for_verification:
                raise ValueError("unverified candidate cannot be accepted")
        selected = sum(node.selected_for_verification for node in self.candidate_nodes)
        accepted = sum(outcome.accepted for outcome in self.target_outcomes)
        if self.accounting.search_pool_nodes != len(self.candidate_nodes):
            raise ValueError("search-pool accounting does not match candidate records")
        if self.accounting.verified_candidate_nodes != selected:
            raise ValueError("verified accounting does not match selected candidates")
        if self.accounting.accepted_candidate_nodes != accepted:
            raise ValueError("accepted accounting does not match target outcomes")
        if self.target_outcomes and any(
            outcome.accepted_prefix_length != accepted for outcome in self.target_outcomes
        ):
            raise ValueError("target outcomes disagree on accepted prefix length")
        expected_committed = (
            self.accounting.committed_candidate_tokens
            + self.accounting.committed_target_tokens
        )
        if len(self.committed_token_ids) != expected_committed:
            raise ValueError("committed token accounting does not match token IDs")
        if self.accounting.committed_target_tokens != int(
            self.root_target_token_id is not None
        ):
            raise ValueError("root target-token accounting is inconsistent")

    def selector_view(self) -> SelectorCycleView:
        return SelectorCycleView(
            self.request,
            tuple(
                SelectorCandidate(
                    stable_node_id=node.stable_node_id,
                    parent_id=node.parent_id,
                    depth=node.depth,
                    token_id=node.token_id,
                    local_probability=node.local_probability,
                    path_probability=node.path_probability,
                    draft_logit=node.draft_logit,
                    entropy=node.entropy,
                    top1_top2_margin=node.top1_top2_margin,
                    sibling_rank=node.sibling_rank,
                    prefix_closed=node.prefix_closed,
                )
                for node in self.candidate_nodes
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_cycle": self.request.to_dict(),
            "candidate_nodes": [asdict(node) for node in self.candidate_nodes],
            "target_outcomes": [asdict(outcome) for outcome in self.target_outcomes],
            "accounting": asdict(self.accounting),
            "root_target_token_id": self.root_target_token_id,
            "committed_token_ids": list(self.committed_token_ids),
            "request_finished": self.request_finished,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RealTraceRecord":
        return cls(
            request=RequestCycle.from_dict(value["request_cycle"]),
            candidate_nodes=tuple(
                CandidateNodeRecord.from_dict(item) for item in value["candidate_nodes"]
            ),
            target_outcomes=tuple(
                TargetOutcomeRecord.from_dict(item) for item in value["target_outcomes"]
            ),
            accounting=CycleAccounting.from_dict(value["accounting"]),
            root_target_token_id=value.get("root_target_token_id"),
            committed_token_ids=tuple(value.get("committed_token_ids", ())),
            request_finished=bool(value["request_finished"]),
            schema_version=str(value["schema_version"]),
        )


def stable_node_id(
    request_id: str, cycle_id: int, parent_id: Optional[str], token_id: int, sibling_rank: int
) -> str:
    payload = f"{request_id}\0{cycle_id}\0{parent_id}\0{token_id}\0{sibling_rank}".encode()
    return "node-" + hashlib.sha256(payload).hexdigest()[:24]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TraceStore:
    """One immutable file per completed cycle, safe for process interruption/resume."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.cycles = root / "cycles"
        self.cycles.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _request_key(request_id: str) -> str:
        return hashlib.sha256(request_id.encode()).hexdigest()[:20]

    def record_path(self, request_id: str, cycle_id: int) -> Path:
        return self.cycles / f"{self._request_key(request_id)}-{cycle_id:08d}.json"

    def write(self, record: RealTraceRecord) -> bool:
        path = self.record_path(record.request.request_id, record.request.cycle_id)
        payload = json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
        if path.exists():
            existing = RealTraceRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if existing != record:
                raise FileExistsError(
                    f"completed trace cycle differs and will not be overwritten: {path}"
                )
            return False
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                existing = RealTraceRecord.from_dict(
                    json.loads(path.read_text(encoding="utf-8"))
                )
                if existing != record:
                    raise FileExistsError(
                        f"completed trace cycle differs and will not be overwritten: {path}"
                    ) from None
                return False
            return True
        finally:
            temporary.unlink(missing_ok=True)

    def records(self) -> list[RealTraceRecord]:
        records = []
        for path in sorted(self.cycles.glob("*.json")):
            records.append(
                RealTraceRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
            )
        return sorted(records, key=lambda item: (item.request.request_id, item.request.cycle_id))

    def request_records(self, request_id: str) -> list[RealTraceRecord]:
        return [record for record in self.records() if record.request.request_id == request_id]

    def resume_state(self, request_id: str) -> tuple[int, tuple[int, ...], bool]:
        records = self.request_records(request_id)
        generated = tuple(
            token for record in records for token in record.committed_token_ids
        )
        if not records:
            return 0, generated, False
        expected = list(range(len(records)))
        actual = [record.request.cycle_id for record in records]
        if actual != expected:
            raise ValueError(f"trace checkpoint has a cycle gap for request {request_id}")
        return len(records), generated, records[-1].request_finished

    def validate(self) -> dict[str, Any]:
        errors = []
        records: list[RealTraceRecord] = []
        for path in sorted(self.cycles.glob("*.json")):
            try:
                record = RealTraceRecord.from_dict(
                    json.loads(path.read_text(encoding="utf-8"))
                )
                if path != self.record_path(
                    record.request.request_id, record.request.cycle_id
                ):
                    raise ValueError("checkpoint filename does not match request/cycle key")
                records.append(record)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                errors.append(f"{path.name}: {error}")
        requests = sorted({record.request.request_id for record in records})
        for request_id in requests:
            try:
                self.resume_state(request_id)
            except ValueError as error:
                errors.append(str(error))
        return {
            "schema_version": "specrhythm.real-trace-validation.v1",
            "valid": not errors,
            "record_count": len(records),
            "request_count": len(requests),
            "errors": errors,
        }

    def write_jsonl(self, output: Path) -> str:
        payload = "".join(
            json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
            for record in self.records()
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, output)
        return sha256_file(output)


def summarize_records(records: Iterable[RealTraceRecord]) -> dict[str, Any]:
    values = list(records)
    accounting = {
        field: sum(getattr(record.accounting, field) for record in values)
        for field in CycleAccounting.__dataclass_fields__
    }
    return {
        "schema_version": "specrhythm.real-trace-summary.v1",
        "record_count": len(values),
        "request_count": len({record.request.request_id for record in values}),
        "finished_request_count": len(
            {record.request.request_id for record in values if record.request_finished}
        ),
        "accounting": accounting,
        "gpu_measurement": False,
    }
