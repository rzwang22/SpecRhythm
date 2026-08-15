"""Phase-3C candidate-forest, target-trajectory, and label-join stages."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, TypeVar

from specrhythm.phase3.distributed import TensorParallelTargetPool
from specrhythm.phase3.engine import CausalLMBackend, create_backend
from specrhythm.phase3.phase3c_config import (
    Phase3CConfig,
    load_frozen_pool_dimensions,
)
from specrhythm.phase3.r3_workload import R3RealRequest
from specrhythm.phase3.trace import sha256_file, stable_node_id

RUNTIME_FEATURES = (
    "stable_node_id",
    "parent_id",
    "depth",
    "token_id",
    "local_probability",
    "log_local_probability",
    "path_probability",
    "log_path_probability",
    "draft_logit",
    "entropy",
    "top1_top2_margin",
    "sibling_rank",
    "branch_rank",
    "parent_probability",
    "cumulative_entropy",
    "remaining_depth",
)
TARGET_ONLY_LABELS = (
    "on_target_path",
    "target_prefix_match",
    "accepted_if_selected",
    "committed_if_selected",
)


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class RuntimeCandidateNode:
    stable_node_id: str
    parent_id: Optional[str]
    depth: int
    token_id: int
    local_probability: float
    log_local_probability: float
    path_probability: float
    log_path_probability: float
    draft_logit: float
    entropy: float
    top1_top2_margin: float
    sibling_rank: int
    branch_rank: int
    parent_probability: float
    cumulative_entropy: float
    remaining_depth: int

    def __post_init__(self) -> None:
        if not self.stable_node_id or self.depth < 1 or self.token_id < 0:
            raise ValueError("invalid runtime candidate identity")
        if not 0 <= self.local_probability <= 1:
            raise ValueError("local_probability must be in [0, 1]")
        if not 0 <= self.path_probability <= 1:
            raise ValueError("path_probability must be in [0, 1]")
        if not 0 <= self.parent_probability <= 1:
            raise ValueError("parent_probability must be in [0, 1]")
        for name in (
            "log_local_probability",
            "log_path_probability",
            "draft_logit",
            "entropy",
            "top1_top2_margin",
            "cumulative_entropy",
        ):
            _finite(name, getattr(self, name))
        if self.entropy < 0 or self.top1_top2_margin < 0 or self.cumulative_entropy < 0:
            raise ValueError("entropy and margin features must be non-negative")
        if min(self.sibling_rank, self.branch_rank, self.remaining_depth) < 0:
            raise ValueError("rank and remaining-depth features must be non-negative")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeCandidateNode":
        unknown = set(value) - set(RUNTIME_FEATURES)
        if unknown:
            raise TargetFeatureLeakageError(
                f"runtime selector view contains forbidden fields: {sorted(unknown)}"
            )
        return cls(**dict(value))


class TargetFeatureLeakageError(ValueError):
    """Raised when a target-blind interface receives target-only fields."""


@dataclass(frozen=True)
class CandidateForestRecord:
    request_id: str
    workload_sha256: str
    prompt_length: int
    draft_model: str
    draft_model_revision: str
    tokenizer_fingerprint: str
    phase2_config_file: str
    phase2_config_sha256: str
    pool_definition: str
    verification_budget: int
    nodes: tuple[RuntimeCandidateNode, ...]
    pool_node_ids: dict[str, tuple[str, ...]]
    search_pool_nodes: dict[str, int]
    selected_verify_nodes: dict[str, tuple[str, ...]]
    target_path_nodes: tuple[str, ...]
    actual_draft_forward_count: int
    kv_cache_reuse: bool
    search_generation_semantics: str
    schema_version: str = "specrhythm.phase3c-candidate-forest.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "specrhythm.phase3c-candidate-forest.v1":
            raise ValueError("unsupported candidate-forest schema")
        by_id = {node.stable_node_id: node for node in self.nodes}
        if len(by_id) != len(self.nodes):
            raise ValueError("candidate stable node IDs must be unique")
        positions = {node.stable_node_id: index for index, node in enumerate(self.nodes)}
        for node in self.nodes:
            if node.parent_id is None:
                if node.depth != 1:
                    raise ValueError("only depth-one candidate nodes may be roots")
            else:
                parent = by_id.get(node.parent_id)
                if parent is None or parent.depth + 1 != node.depth:
                    raise ValueError("candidate forest is not prefix closed")
                if positions[parent.stable_node_id] >= positions[node.stable_node_id]:
                    raise ValueError("candidate parent must precede its child")
        previous: set[str] = set()
        for ratio in ("1x", "2x", "4x"):
            ids = self.pool_node_ids.get(ratio, ())
            if len(ids) != self.search_pool_nodes.get(ratio):
                raise ValueError("candidate pool count does not match node IDs")
            current = set(ids)
            if not previous.issubset(current):
                raise ValueError("candidate pools must be nested")
            if not current.issubset(by_id):
                raise ValueError("candidate pool contains an unknown node")
            for node_id in ids:
                parent_id = by_id[node_id].parent_id
                if parent_id is not None and parent_id not in current:
                    raise ValueError("candidate pool is not prefix closed")
            previous = current
        if tuple(self.pool_node_ids["4x"]) != tuple(node.stable_node_id for node in self.nodes):
            raise ValueError("4x pool must be the complete shared candidate forest")
        if self.selected_verify_nodes or self.target_path_nodes:
            raise ValueError("draft-only forest must not contain selection or target labels")
        if len(self.phase2_config_sha256) != 64 or self.verification_budget < 1:
            raise ValueError("candidate forest has invalid frozen-config provenance")
        if self.actual_draft_forward_count < 1 or self.kv_cache_reuse:
            raise ValueError("Phase-3C correctness forest must record forwards and no KV reuse")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "workload_sha256": self.workload_sha256,
            "prompt_length": self.prompt_length,
            "draft_model": self.draft_model,
            "draft_model_revision": self.draft_model_revision,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "phase2_config_file": self.phase2_config_file,
            "phase2_config_sha256": self.phase2_config_sha256,
            "pool_definition": self.pool_definition,
            "verification_budget": self.verification_budget,
            "runtime_available_features": list(RUNTIME_FEATURES),
            "target_only_labels": list(TARGET_ONLY_LABELS),
            "candidate_nodes": [asdict(node) for node in self.nodes],
            "pool_node_ids": {key: list(value) for key, value in self.pool_node_ids.items()},
            "search_pool_nodes": self.search_pool_nodes,
            "selected_verify_nodes": {},
            "target_path_nodes": [],
            "actual_draft_forward_count": self.actual_draft_forward_count,
            "kv_cache_reuse": self.kv_cache_reuse,
            "search_generation_semantics": self.search_generation_semantics,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateForestRecord":
        if tuple(value.get("runtime_available_features", ())) != RUNTIME_FEATURES:
            raise ValueError("candidate forest runtime-feature declaration changed")
        if tuple(value.get("target_only_labels", ())) != TARGET_ONLY_LABELS:
            raise ValueError("candidate forest target-label declaration changed")
        return cls(
            request_id=str(value["request_id"]),
            workload_sha256=str(value["workload_sha256"]),
            prompt_length=value["prompt_length"],
            draft_model=str(value["draft_model"]),
            draft_model_revision=str(value["draft_model_revision"]),
            tokenizer_fingerprint=str(value["tokenizer_fingerprint"]),
            phase2_config_file=str(value["phase2_config_file"]),
            phase2_config_sha256=str(value["phase2_config_sha256"]),
            pool_definition=str(value["pool_definition"]),
            verification_budget=value["verification_budget"],
            nodes=tuple(
                RuntimeCandidateNode.from_dict(item) for item in value["candidate_nodes"]
            ),
            pool_node_ids={
                str(key): tuple(ids) for key, ids in value["pool_node_ids"].items()
            },
            search_pool_nodes={
                str(key): int(count) for key, count in value["search_pool_nodes"].items()
            },
            selected_verify_nodes={
                str(key): tuple(ids)
                for key, ids in value.get("selected_verify_nodes", {}).items()
            },
            target_path_nodes=tuple(value.get("target_path_nodes", ())),
            actual_draft_forward_count=value["actual_draft_forward_count"],
            kv_cache_reuse=bool(value["kv_cache_reuse"]),
            search_generation_semantics=str(value["search_generation_semantics"]),
            schema_version=str(value["schema_version"]),
        )


@dataclass(frozen=True)
class TargetTrajectoryRecord:
    request_id: str
    workload_sha256: str
    prompt_length: int
    tokenizer_fingerprint: str
    target_token_ids: tuple[int, ...]
    target_log_probabilities: tuple[float, ...]
    target_path_length: int
    target_eos_position: Optional[int]
    target_model: str
    target_model_revision: str
    target_forward_count: int
    greedy_decoding: bool
    kv_cache_reuse: bool
    schema_version: str = "specrhythm.phase3c-target-trajectory.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "specrhythm.phase3c-target-trajectory.v1":
            raise ValueError("unsupported target-trajectory schema")
        if not self.request_id or not self.target_token_ids:
            raise ValueError("target trajectory must contain an identity and tokens")
        if len(self.target_token_ids) != len(self.target_log_probabilities):
            raise ValueError("target token/log-probability lengths differ")
        if self.target_path_length != len(self.target_token_ids):
            raise ValueError("target_path_length does not match target tokens")
        if self.target_forward_count != len(self.target_token_ids):
            raise ValueError("serial target forward count must equal generated token count")
        if self.target_eos_position is not None and not (
            0 <= self.target_eos_position < len(self.target_token_ids)
        ):
            raise ValueError("target_eos_position is outside the target trajectory")
        if not self.greedy_decoding or self.kv_cache_reuse:
            raise ValueError("Phase-3C target must be greedy without KV-cache reuse")
        for value in self.target_log_probabilities:
            _finite("target_log_probability", value)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["target_token_ids"] = list(self.target_token_ids)
        value["target_log_probabilities"] = list(self.target_log_probabilities)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TargetTrajectoryRecord":
        copied = dict(value)
        copied["target_token_ids"] = tuple(copied["target_token_ids"])
        copied["target_log_probabilities"] = tuple(
            copied["target_log_probabilities"]
        )
        return cls(**copied)


@dataclass(frozen=True)
class LabeledCandidateNode:
    runtime_features: RuntimeCandidateNode
    target_only_labels: dict[str, Any]

    def __post_init__(self) -> None:
        if set(self.target_only_labels) != set(TARGET_ONLY_LABELS):
            raise ValueError("labeled node has an incomplete target-only label set")
        if not isinstance(self.target_only_labels["on_target_path"], bool):
            raise ValueError("on_target_path must be boolean")
        if not isinstance(self.target_only_labels["target_prefix_match"], int):
            raise ValueError("target_prefix_match must be an integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_features": asdict(self.runtime_features),
            "target_only_labels": self.target_only_labels,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LabeledCandidateNode":
        return cls(
            RuntimeCandidateNode.from_dict(value["runtime_features"]),
            dict(value["target_only_labels"]),
        )


@dataclass(frozen=True)
class LabeledTraceRecord:
    request_id: str
    task_class: str
    data_split: str
    workload_sha256: str
    forest_sha256: str
    target_trajectory_sha256: str
    nodes: tuple[LabeledCandidateNode, ...]
    pool_node_ids: dict[str, tuple[str, ...]]
    search_pool_nodes: dict[str, int]
    selected_verify_nodes: dict[str, tuple[str, ...]]
    target_path_nodes: tuple[str, ...]
    target_path_node_ids_by_pool: dict[str, tuple[str, ...]]
    missing_target_depths_by_pool: dict[str, tuple[int, ...]]
    target_trajectory: TargetTrajectoryRecord
    schema_version: str = "specrhythm.phase3c-labeled-trace.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "specrhythm.phase3c-labeled-trace.v1":
            raise ValueError("unsupported labeled-trace schema")
        if self.request_id != self.target_trajectory.request_id:
            raise ValueError("target trajectory is attached to the wrong request")
        by_id = {node.runtime_features.stable_node_id: node for node in self.nodes}
        if len(by_id) != len(self.nodes):
            raise ValueError("labeled candidate IDs must be unique")
        previous: set[str] = set()
        for ratio in ("1x", "2x", "4x"):
            pool = set(self.pool_node_ids[ratio])
            if not previous.issubset(pool):
                raise ValueError("labeled candidate pools must remain nested")
            if not set(self.target_path_node_ids_by_pool[ratio]).issubset(pool):
                raise ValueError("target path node is outside its candidate pool")
            previous = pool
        if self.selected_verify_nodes:
            raise ValueError("label join must not preselect verification nodes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "task_class": self.task_class,
            "data_split": self.data_split,
            "workload_sha256": self.workload_sha256,
            "forest_sha256": self.forest_sha256,
            "target_trajectory_sha256": self.target_trajectory_sha256,
            "runtime_available_features": list(RUNTIME_FEATURES),
            "target_only_labels": list(TARGET_ONLY_LABELS),
            "candidate_nodes": [node.to_dict() for node in self.nodes],
            "pool_node_ids": {key: list(value) for key, value in self.pool_node_ids.items()},
            "search_pool_nodes": self.search_pool_nodes,
            "selected_verify_nodes": {},
            "target_path_nodes": list(self.target_path_nodes),
            "target_path_node_ids_by_pool": {
                key: list(value) for key, value in self.target_path_node_ids_by_pool.items()
            },
            "missing_target_depths_by_pool": {
                key: list(value) for key, value in self.missing_target_depths_by_pool.items()
            },
            "target_trajectory": self.target_trajectory.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LabeledTraceRecord":
        if tuple(value.get("runtime_available_features", ())) != RUNTIME_FEATURES:
            raise ValueError("labeled trace runtime-feature declaration changed")
        if tuple(value.get("target_only_labels", ())) != TARGET_ONLY_LABELS:
            raise ValueError("labeled trace target-label declaration changed")
        return cls(
            request_id=str(value["request_id"]),
            task_class=str(value["task_class"]),
            data_split=str(value["data_split"]),
            workload_sha256=str(value["workload_sha256"]),
            forest_sha256=str(value["forest_sha256"]),
            target_trajectory_sha256=str(value["target_trajectory_sha256"]),
            nodes=tuple(
                LabeledCandidateNode.from_dict(item) for item in value["candidate_nodes"]
            ),
            pool_node_ids={key: tuple(ids) for key, ids in value["pool_node_ids"].items()},
            search_pool_nodes={
                key: int(count) for key, count in value["search_pool_nodes"].items()
            },
            selected_verify_nodes={
                key: tuple(ids)
                for key, ids in value.get("selected_verify_nodes", {}).items()
            },
            target_path_nodes=tuple(value["target_path_nodes"]),
            target_path_node_ids_by_pool={
                key: tuple(ids)
                for key, ids in value["target_path_node_ids_by_pool"].items()
            },
            missing_target_depths_by_pool={
                key: tuple(depths)
                for key, depths in value["missing_target_depths_by_pool"].items()
            },
            target_trajectory=TargetTrajectoryRecord.from_dict(value["target_trajectory"]),
            schema_version=str(value["schema_version"]),
        )


RecordT = TypeVar("RecordT")


class ImmutableRequestStore:
    """One atomic immutable JSON checkpoint per request and stage."""

    def __init__(
        self,
        root: Path,
        parser: Callable[[Mapping[str, Any]], RecordT],
        serializer: Callable[[RecordT], Mapping[str, Any]],
    ) -> None:
        self.root = root
        self.records_dir = root / "requests"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.parser = parser
        self.serializer = serializer

    @staticmethod
    def request_key(request_id: str) -> str:
        return hashlib.sha256(request_id.encode()).hexdigest()[:24]

    def path(self, request_id: str) -> Path:
        return self.records_dir / f"{self.request_key(request_id)}.json"

    def has(self, request_id: str) -> bool:
        return self.path(request_id).is_file()

    def read(self, request_id: str) -> RecordT:
        value = json.loads(self.path(request_id).read_text(encoding="utf-8"))
        return self.parser(value)

    def records(self) -> list[RecordT]:
        return [
            self.parser(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(self.records_dir.glob("*.json"))
        ]

    def write(self, request_id: str, record: RecordT) -> bool:
        path = self.path(request_id)
        payload = json.dumps(
            self.serializer(record), sort_keys=True, separators=(",", ":")
        ) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != payload:
                raise FileExistsError(f"immutable checkpoint differs: {path}")
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
            except FileExistsError as error:
                if path.read_text(encoding="utf-8") != payload:
                    raise FileExistsError(
                        f"immutable checkpoint differs: {path}"
                    ) from error
                return False
            return True
        finally:
            temporary.unlink(missing_ok=True)


def forest_store(path: Path) -> ImmutableRequestStore[CandidateForestRecord]:
    return ImmutableRequestStore(
        path, CandidateForestRecord.from_dict, CandidateForestRecord.to_dict
    )


def target_store(path: Path) -> ImmutableRequestStore[TargetTrajectoryRecord]:
    return ImmutableRequestStore(
        path, TargetTrajectoryRecord.from_dict, TargetTrajectoryRecord.to_dict
    )


def labeled_store(path: Path) -> ImmutableRequestStore[LabeledTraceRecord]:
    return ImmutableRequestStore(path, LabeledTraceRecord.from_dict, LabeledTraceRecord.to_dict)


@dataclass(frozen=True)
class _Frontier:
    parent_id: Optional[str]
    path_tokens: tuple[int, ...]
    depth: int
    probability: float
    cumulative_entropy: float
    branch_rank: int


def generate_real_candidate_forest(
    backend: CausalLMBackend,
    request: R3RealRequest,
    *,
    workload_sha256: str,
    pool_dimensions: Mapping[str, Any],
    model_revision: str,
    context_token_ids: Optional[tuple[int, ...]] = None,
    cycle_id: int = 0,
) -> CandidateForestRecord:
    width = int(pool_dimensions["candidate_width"])
    maximum_depth = int(pool_dimensions["candidate_depth"])
    maximum_nodes = int(pool_dimensions["pool_node_counts"]["4x"])
    frontier: list[tuple[float, int, str, _Frontier]] = []
    root = _Frontier(None, (), 0, 1.0, 0.0, 0)
    heapq.heappush(frontier, (-1.0, 0, "", root))
    nodes: list[RuntimeCandidateNode] = []
    forward_count = 0
    context = context_token_ids if context_token_ids is not None else request.prompt_token_ids
    while frontier and len(nodes) < maximum_nodes:
        _, _, _, state = heapq.heappop(frontier)
        distribution = backend.next_token(list(context + state.path_tokens), width)
        forward_count += 1
        for sibling_rank, token in enumerate(distribution.ranked_tokens[:width]):
            if len(nodes) >= maximum_nodes:
                break
            depth = state.depth + 1
            node_id = stable_node_id(
                request.request_id, cycle_id, state.parent_id, token.token_id, sibling_rank
            )
            probability = max(token.probability, 1e-300)
            path_probability = state.probability * token.probability
            branch_rank = sibling_rank if state.parent_id is None else state.branch_rank
            node = RuntimeCandidateNode(
                stable_node_id=node_id,
                parent_id=state.parent_id,
                depth=depth,
                token_id=token.token_id,
                local_probability=token.probability,
                log_local_probability=math.log(probability),
                path_probability=path_probability,
                log_path_probability=math.log(max(path_probability, 1e-300)),
                draft_logit=token.logit,
                entropy=distribution.entropy,
                top1_top2_margin=distribution.top1_top2_margin,
                sibling_rank=sibling_rank,
                branch_rank=branch_rank,
                parent_probability=state.probability,
                cumulative_entropy=state.cumulative_entropy + distribution.entropy,
                remaining_depth=maximum_depth - depth,
            )
            nodes.append(node)
            if depth < maximum_depth and (
                backend.eos_token_id is None or token.token_id != backend.eos_token_id
            ):
                child = _Frontier(
                    node_id,
                    state.path_tokens + (token.token_id,),
                    depth,
                    path_probability,
                    node.cumulative_entropy,
                    branch_rank,
                )
                heapq.heappush(
                    frontier,
                    (-path_probability, depth, node_id, child),
                )
    if len(nodes) != maximum_nodes:
        raise ValueError(
            f"draft forest realized {len(nodes)} nodes; frozen 4x pool requires {maximum_nodes}"
        )
    pool_ids = {
        ratio: tuple(node.stable_node_id for node in nodes[:count])
        for ratio, count in pool_dimensions["pool_node_counts"].items()
    }
    return CandidateForestRecord(
        request_id=request.request_id,
        workload_sha256=workload_sha256,
        prompt_length=len(context),
        draft_model=backend.model_id,
        draft_model_revision=model_revision,
        tokenizer_fingerprint=backend.tokenizer_fingerprint,
        phase2_config_file=Path(pool_dimensions["phase2_config_path"]).name,
        phase2_config_sha256=sha256_file(Path(pool_dimensions["phase2_config_path"])),
        pool_definition="base_nodes=candidate_tree_width*candidate_tree_depth",
        verification_budget=int(pool_dimensions["verification_budget"]),
        nodes=tuple(nodes),
        pool_node_ids=pool_ids,
        search_pool_nodes={key: len(value) for key, value in pool_ids.items()},
        selected_verify_nodes={},
        target_path_nodes=(),
        actual_draft_forward_count=forward_count,
        kv_cache_reuse=False,
        search_generation_semantics=(
            "best-first prefix expansion; one full-context model forward per expanded parent; "
            "all 1x/2x/4x pools are prefixes of one shared 4x forest"
        ),
    )


def generate_target_trajectory(
    backend: CausalLMBackend,
    request: R3RealRequest,
    *,
    workload_sha256: str,
    model_revision: str,
) -> TargetTrajectoryRecord:
    tokens = []
    log_probabilities = []
    eos_position = None
    for index in range(request.maximum_new_tokens):
        distribution = backend.next_token(
            list(request.prompt_token_ids) + tokens, 2
        )
        token = distribution.top1
        tokens.append(token.token_id)
        log_probabilities.append(math.log(max(token.probability, 1e-300)))
        if backend.eos_token_id is not None and token.token_id == backend.eos_token_id:
            eos_position = index
            break
    return TargetTrajectoryRecord(
        request_id=request.request_id,
        workload_sha256=workload_sha256,
        prompt_length=request.prompt_length,
        tokenizer_fingerprint=backend.tokenizer_fingerprint,
        target_token_ids=tuple(tokens),
        target_log_probabilities=tuple(log_probabilities),
        target_path_length=len(tokens),
        target_eos_position=eos_position,
        target_model=backend.model_id,
        target_model_revision=model_revision,
        target_forward_count=len(tokens),
        greedy_decoding=True,
        kv_cache_reuse=False,
    )


def _model_revision(configured: Optional[str], model_id: str) -> str:
    if configured:
        return configured
    config_path = Path(model_id) / "config.json"
    if config_path.is_file():
        return f"local-config-sha256:{sha256_file(config_path)}"
    return "unversioned-local-or-dry-run-model"


def _stage_preflight(store: ImmutableRequestStore[Any], resume: bool) -> None:
    if not resume and any(store.records_dir.glob("*.json")):
        raise FileExistsError("stage output is non-empty; pass --resume to continue")


def run_draft_forest_stage(
    requests: Iterable[R3RealRequest],
    config: Phase3CConfig,
    *,
    workload_path: Path,
    output_dir: Path,
    resume: bool,
    backend: Optional[CausalLMBackend] = None,
) -> dict[str, Any]:
    store = forest_store(output_dir)
    _stage_preflight(store, resume)
    workload_sha = sha256_file(workload_path)
    dimensions = load_frozen_pool_dimensions(config)
    owns_backend = backend is None
    backend = backend or create_backend(
        config.runtime.backend, config.runtime.draft, config.runtime.random_seed
    )
    written = 0
    try:
        for request in requests:
            if store.has(request.request_id):
                if not resume:
                    raise FileExistsError(store.path(request.request_id))
                existing = store.read(request.request_id)
                if (
                    existing.workload_sha256 != workload_sha
                    or existing.tokenizer_fingerprint != backend.tokenizer_fingerprint
                    or existing.draft_model != backend.model_id
                ):
                    raise ValueError(
                        f"resume identity differs for draft request {request.request_id}"
                    )
                continue
            if backend.tokenizer_fingerprint != request.tokenizer_fingerprint:
                raise ValueError(
                    f"draft tokenizer differs from workload for {request.request_id}"
                )
            if tuple(backend.encode(request.prompt_text)) != request.prompt_token_ids:
                raise ValueError(
                    f"draft tokenizer does not reproduce prompt IDs for {request.request_id}"
                )
            if request.prompt_length + request.maximum_new_tokens > config.runtime.context_length:
                raise ValueError(f"request {request.request_id} exceeds context_length")
            record = generate_real_candidate_forest(
                backend,
                request,
                workload_sha256=workload_sha,
                pool_dimensions=dimensions,
                model_revision=_model_revision(
                    config.runtime.draft.revision, backend.model_id
                ),
            )
            written += int(store.write(request.request_id, record))
    finally:
        if owns_backend:
            backend.close()
    records = store.records()
    return {
        "schema_version": "specrhythm.phase3c-stage-summary.v1",
        "stage": "draft-forest",
        "backend": config.runtime.backend,
        "gpu_measurement": False,
        "serving_engine": False,
        "packed_tree_verification": False,
        "new_records": written,
        "completed_records": len(records),
        "pool_node_counts": dimensions["pool_node_counts"],
        "verification_budget": dimensions["verification_budget"],
        "kv_cache_reuse": False,
    }


def _create_target_backend(config: Phase3CConfig) -> CausalLMBackend:
    if config.runtime.backend == "transformers" and config.runtime.target.tp_size > 1:
        return TensorParallelTargetPool(config.runtime.target, config.runtime.random_seed)
    return create_backend(
        config.runtime.backend, config.runtime.target, config.runtime.random_seed
    )


def run_target_trajectory_stage(
    requests: Iterable[R3RealRequest],
    config: Phase3CConfig,
    *,
    workload_path: Path,
    output_dir: Path,
    resume: bool,
    backend: Optional[CausalLMBackend] = None,
) -> dict[str, Any]:
    store = target_store(output_dir)
    _stage_preflight(store, resume)
    workload_sha = sha256_file(workload_path)
    owns_backend = backend is None
    backend = backend or _create_target_backend(config)
    written = 0
    try:
        for request in requests:
            if store.has(request.request_id):
                if not resume:
                    raise FileExistsError(store.path(request.request_id))
                existing = store.read(request.request_id)
                if (
                    existing.workload_sha256 != workload_sha
                    or existing.tokenizer_fingerprint != backend.tokenizer_fingerprint
                    or existing.target_model != backend.model_id
                ):
                    raise ValueError(
                        f"resume identity differs for target request {request.request_id}"
                    )
                continue
            if backend.tokenizer_fingerprint != request.tokenizer_fingerprint:
                raise ValueError(
                    f"target tokenizer differs from workload for {request.request_id}"
                )
            if request.prompt_length + request.maximum_new_tokens > config.runtime.context_length:
                raise ValueError(f"request {request.request_id} exceeds context_length")
            record = generate_target_trajectory(
                backend,
                request,
                workload_sha256=workload_sha,
                model_revision=_model_revision(
                    config.runtime.target.revision, backend.model_id
                ),
            )
            written += int(store.write(request.request_id, record))
    finally:
        if owns_backend:
            backend.close()
    records = store.records()
    return {
        "schema_version": "specrhythm.phase3c-stage-summary.v1",
        "stage": "target-trajectory",
        "backend": config.runtime.backend,
        "gpu_measurement": False,
        "serving_engine": False,
        "packed_tree_verification": False,
        "new_records": written,
        "completed_records": len(records),
        "target_outputs_immutable": True,
        "target_forward_count": sum(record.target_forward_count for record in records),
        "kv_cache_reuse": False,
    }


def _node_paths(nodes: Iterable[RuntimeCandidateNode]) -> dict[str, tuple[int, ...]]:
    paths: dict[str, tuple[int, ...]] = {}
    for node in nodes:
        parent = () if node.parent_id is None else paths[node.parent_id]
        paths[node.stable_node_id] = parent + (node.token_id,)
    return paths


def _record_sha(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def join_forest_and_target(
    request: R3RealRequest,
    forest: CandidateForestRecord,
    target: TargetTrajectoryRecord,
) -> LabeledTraceRecord:
    if forest.request_id != request.request_id or target.request_id != request.request_id:
        raise ValueError("label join request IDs do not match")
    if forest.workload_sha256 != target.workload_sha256:
        raise ValueError("draft and target stages used different workloads")
    if forest.tokenizer_fingerprint != target.tokenizer_fingerprint:
        raise ValueError("draft and target tokenizers are not shared")
    paths = _node_paths(forest.nodes)
    target_tokens = target.target_token_ids
    labeled = []
    target_nodes = []
    for node in forest.nodes:
        path = paths[node.stable_node_id]
        common = 0
        for actual, expected in zip(path, target_tokens):
            if actual != expected:
                break
            common += 1
        on_path = len(path) <= len(target_tokens) and common == len(path)
        if on_path:
            target_nodes.append(node.stable_node_id)
        labeled.append(
            LabeledCandidateNode(
                runtime_features=node,
                target_only_labels={
                    "on_target_path": on_path,
                    "target_prefix_match": common,
                    "accepted_if_selected": on_path,
                    "committed_if_selected": on_path,
                },
            )
        )
    by_id = {node.runtime_features.stable_node_id: node for node in labeled}
    by_pool = {}
    missing = {}
    for ratio, ids in forest.pool_node_ids.items():
        pool = set(ids)
        hits = tuple(node_id for node_id in target_nodes if node_id in pool)
        by_pool[ratio] = hits
        covered_depths = {by_id[node_id].runtime_features.depth for node_id in hits}
        maximum_depth = min(
            max((node.runtime_features.depth for node in labeled), default=0),
            len(target_tokens),
        )
        missing[ratio] = tuple(
            depth for depth in range(1, maximum_depth + 1) if depth not in covered_depths
        )
    return LabeledTraceRecord(
        request_id=request.request_id,
        task_class=request.task_class,
        data_split=request.data_split,
        workload_sha256=forest.workload_sha256,
        forest_sha256=_record_sha(forest.to_dict()),
        target_trajectory_sha256=_record_sha(target.to_dict()),
        nodes=tuple(labeled),
        pool_node_ids=forest.pool_node_ids,
        search_pool_nodes=forest.search_pool_nodes,
        selected_verify_nodes={},
        target_path_nodes=tuple(target_nodes),
        target_path_node_ids_by_pool=by_pool,
        missing_target_depths_by_pool=missing,
        target_trajectory=target,
    )


def run_label_join_stage(
    requests: Iterable[R3RealRequest],
    *,
    forest_dir: Path,
    target_dir: Path,
    output_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    forests = forest_store(forest_dir)
    targets = target_store(target_dir)
    output = labeled_store(output_dir)
    _stage_preflight(output, resume)
    written = 0
    for request in requests:
        if output.has(request.request_id):
            if not resume:
                raise FileExistsError(output.path(request.request_id))
            continue
        if not forests.has(request.request_id) or not targets.has(request.request_id):
            raise ValueError(f"missing draft or target checkpoint for {request.request_id}")
        record = join_forest_and_target(
            request, forests.read(request.request_id), targets.read(request.request_id)
        )
        written += int(output.write(request.request_id, record))
    records = output.records()
    missing_full = sum(bool(record.missing_target_depths_by_pool["1x"]) for record in records)
    missing_k4 = sum(
        any(depth <= 4 for depth in record.missing_target_depths_by_pool["1x"])
        for record in records
    )
    return {
        "schema_version": "specrhythm.phase3c-stage-summary.v1",
        "stage": "label-join",
        "new_records": written,
        "completed_records": len(records),
        "runtime_target_feature_isolation": True,
        "target_trajectory_regenerated": False,
        "requests_with_missing_1x_target_coverage": missing_full,
        "requests_with_missing_full_1x_eligible_target_path": missing_full,
        "requests_with_missing_1x_verification_horizon_k4": missing_k4,
        "missing_coverage_interpretation": (
            "full eligible-path missing is not a first-round failure rate; use the K=4 count"
        ),
    }


def validate_phase3c_artifacts(
    requests: Iterable[R3RealRequest],
    *,
    forest_dir: Path,
    target_dir: Path,
    labeled_dir: Path,
) -> dict[str, Any]:
    request_list = list(requests)
    forests = forest_store(forest_dir)
    targets = target_store(target_dir)
    labels = labeled_store(labeled_dir)
    errors = []
    expected_ids = {request.request_id for request in request_list}
    stage_ids = {
        "forest": {record.request_id for record in forests.records()},
        "target": {record.request_id for record in targets.records()},
        "labeled": {record.request_id for record in labels.records()},
    }
    for stage, actual_ids in stage_ids.items():
        if actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids)
            extra = sorted(actual_ids - expected_ids)
            errors.append(
                f"{stage} request set differs: missing={missing[:5]}, extra={extra[:5]}"
            )
    for request in request_list:
        try:
            forest = forests.read(request.request_id)
            target = targets.read(request.request_id)
            labeled = labels.read(request.request_id)
            expected = join_forest_and_target(request, forest, target)
            if labeled != expected:
                errors.append(f"{request.request_id}: label join is not deterministic")
            selected = set(labeled.target_path_node_ids_by_pool["4x"])
            accepted = tuple(
                node.runtime_features.token_id
                for node in labeled.nodes
                if node.runtime_features.stable_node_id in selected
            )
            # Accepting a covered prefix and then replaying the target remainder must
            # reproduce the one immutable target trajectory exactly.
            reconstructed = accepted + target.target_token_ids[len(accepted) :]
            if reconstructed != target.target_token_ids:
                errors.append(f"{request.request_id}: speculative/target tokens differ")
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            errors.append(f"{request.request_id}: {error}")
    return {
        "schema_version": "specrhythm.phase3c-validation.v1",
        "valid": not errors,
        "errors": errors,
        "request_count": len(request_list),
        "forest_records": len(forests.records()),
        "target_records": len(targets.records()),
        "labeled_records": len(labels.records()),
        "checks": {
            "nested_pools": True,
            "stable_node_identity": True,
            "prefix_closure": True,
            "target_trajectory_ratio_independent": True,
            "target_blind_feature_isolation": True,
            "speculative_target_token_semantics": not errors,
        },
    }
