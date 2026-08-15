"""Configuration for the isolated Phase-3C real-trace pilot."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from specrhythm.phase3.config import Phase3Config


@dataclass(frozen=True)
class PublicDatasetConfig:
    task_class: str
    dataset: str
    adapter: str
    path: str
    split: str
    source_license_or_url: str

    @classmethod
    def from_dict(cls, task_class: str, value: Mapping[str, Any]) -> "PublicDatasetConfig":
        fields = {
            name: str(value.get(name, "")).strip()
            for name in ("dataset", "adapter", "path", "split", "source_license_or_url")
        }
        missing = [name for name, item in fields.items() if not item]
        if missing:
            raise ValueError(f"workload.sources.{task_class} missing: {', '.join(missing)}")
        allowed = {
            "code": {"humaneval", "mbpp"},
            "chat": {"sharegpt", "openassistant"},
            "summarization": {"cnn_dailymail", "xsum", "govreport"},
        }
        if task_class not in allowed or fields["adapter"] not in allowed[task_class]:
            raise ValueError(
                f"unsupported {task_class} dataset adapter: {fields['adapter']}"
            )
        return cls(task_class=task_class, **fields)

    @property
    def resolved_path(self) -> Path:
        expanded = os.path.expandvars(os.path.expanduser(self.path))
        if "$" in expanded:
            raise ValueError(f"unresolved dataset path: {self.path}")
        return Path(expanded).resolve()


@dataclass(frozen=True)
class R3WorkloadConfig:
    arrival_trace: str
    tokenizer_model: str
    request_count: int
    maximum_new_tokens: int
    time_scale: float
    sources: tuple[PublicDatasetConfig, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "R3WorkloadConfig":
        raw_sources = value.get("sources")
        if not isinstance(raw_sources, Mapping):
            raise ValueError("workload.sources must be a mapping")
        sources = tuple(
            PublicDatasetConfig.from_dict(task, raw_sources.get(task, {}))
            for task in ("code", "chat", "summarization")
        )
        arrival_trace = str(value.get("arrival_trace", "")).strip()
        tokenizer_model = str(value.get("tokenizer_model", "")).strip()
        if not arrival_trace or not tokenizer_model:
            raise ValueError("workload arrival_trace and tokenizer_model are required")
        request_count = value.get("request_count", 100)
        maximum_new_tokens = value.get("maximum_new_tokens", 16)
        time_scale = value.get("time_scale", 1.0)
        if (
            not isinstance(request_count, int)
            or isinstance(request_count, bool)
            or request_count < 1
        ):
            raise ValueError("workload.request_count must be a positive integer")
        if (
            not isinstance(maximum_new_tokens, int)
            or isinstance(maximum_new_tokens, bool)
            or maximum_new_tokens < 1
        ):
            raise ValueError("workload.maximum_new_tokens must be a positive integer")
        if not isinstance(time_scale, (int, float)) or time_scale <= 0:
            raise ValueError("workload.time_scale must be positive")
        return cls(
            arrival_trace,
            tokenizer_model,
            request_count,
            maximum_new_tokens,
            float(time_scale),
            sources,
        )

    def resolve(self, value: str) -> Path:
        expanded = os.path.expandvars(os.path.expanduser(value))
        if "$" in expanded:
            raise ValueError(f"unresolved workload path: {value}")
        return Path(expanded).resolve()


@dataclass(frozen=True)
class FrozenPoolConfig:
    phase2_config_path: str
    ratios: tuple[int, ...]
    verification_budget: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenPoolConfig":
        path = str(value.get("phase2_config_path", "")).strip()
        if not path:
            raise ValueError("candidate_pool.phase2_config_path is required")
        ratios = tuple(value.get("ratios", (1, 2, 4)))
        if ratios != (1, 2, 4):
            raise ValueError("Phase-3C candidate pool ratios must be exactly 1, 2, 4")
        budget = value.get("verification_budget")
        if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
            raise ValueError("candidate_pool.verification_budget must be positive")
        return cls(path, ratios, budget)

    def resolved_phase2_path(self, config_path: Path) -> Path:
        expanded = os.path.expandvars(os.path.expanduser(self.phase2_config_path))
        if "$" in expanded:
            raise ValueError(f"unresolved Phase-2 config path: {self.phase2_config_path}")
        path = Path(expanded)
        return (config_path.parent / path).resolve() if not path.is_absolute() else path.resolve()


@dataclass(frozen=True)
class Phase3CConfig:
    path: Path
    runtime: Phase3Config
    workload: R3WorkloadConfig
    candidate_pool: FrozenPoolConfig


def _read_mapping(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as error:
            raise ValueError("non-JSON YAML requires the optional PyYAML dependency") from error
        value = yaml.safe_load(text)
    if not isinstance(value, Mapping):
        raise ValueError("Phase-3C config root must be a mapping")
    return value


def load_phase3c_config(path: str) -> Phase3CConfig:
    config_path = Path(path).resolve()
    value = _read_mapping(config_path)
    if value.get("schema_version") != "specrhythm.phase3c-config.v1":
        raise ValueError("unsupported Phase-3C config schema_version")
    workload = R3WorkloadConfig.from_dict(value.get("workload", {}))

    def relative_to_config(raw: str) -> str:
        expanded = os.path.expandvars(os.path.expanduser(raw))
        if "$" in expanded or Path(expanded).is_absolute():
            return raw
        return str((config_path.parent / expanded).resolve())

    workload = replace(
        workload,
        arrival_trace=relative_to_config(workload.arrival_trace),
        sources=tuple(
            replace(source, path=relative_to_config(source.path))
            for source in workload.sources
        ),
    )
    runtime = Phase3Config.from_dict(value.get("runtime", {}))
    if workload.tokenizer_model != runtime.draft.model_path:
        raise ValueError("R3 workload tokenizer_model must be the configured draft model")
    return Phase3CConfig(
        path=config_path,
        runtime=runtime,
        workload=workload,
        candidate_pool=FrozenPoolConfig.from_dict(value.get("candidate_pool", {})),
    )


def load_frozen_pool_dimensions(config: Phase3CConfig) -> dict[str, Any]:
    """Derive node counts from the checked Phase-2 width/depth definition."""

    path = config.candidate_pool.resolved_phase2_path(config.path)
    value = _read_mapping(path)
    width = value.get("candidate_tree_width")
    depth = value.get("candidate_tree_depth")
    speculative_budget = value.get("speculative_budget")
    if any(
        not isinstance(item, int) or isinstance(item, bool) or item < 1
        for item in (width, depth, speculative_budget)
    ):
        raise ValueError("Phase-2 config has invalid tree or speculative-budget dimensions")
    if config.candidate_pool.verification_budget != speculative_budget:
        raise ValueError(
            "Phase-3C verification budget must match frozen Phase-2 speculative_budget"
        )
    base_nodes = width * depth
    expected_maximum = base_nodes * max(config.candidate_pool.ratios)
    if config.runtime.candidate_width != width:
        raise ValueError("Phase-3C candidate width must match frozen Phase-2 width")
    if config.runtime.search_pool_size != expected_maximum:
        raise ValueError("Phase-3C search_pool_size must match the frozen 4x pool")
    if config.runtime.candidate_budget != speculative_budget:
        raise ValueError("Phase-3C candidate_budget must match frozen speculative_budget")
    return {
        "phase2_config_path": path,
        "candidate_width": width,
        "candidate_depth": depth,
        "base_pool_nodes": base_nodes,
        "pool_node_counts": {
            f"{ratio}x": base_nodes * ratio for ratio in config.candidate_pool.ratios
        },
        "verification_budget": speculative_budget,
    }
