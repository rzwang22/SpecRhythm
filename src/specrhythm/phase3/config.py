"""Configuration loading for Phase-3 runners and microbenchmarks."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


def _positive_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _gpu_ids(name: str, value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must contain at least one GPU ID")
    result = tuple(value)
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in result):
        raise ValueError(f"{name} must contain non-negative integer GPU IDs")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate GPU IDs")
    return result


@dataclass(frozen=True)
class ModelRuntimeConfig:
    model_path: str
    gpu_ids: tuple[int, ...]
    tp_size: int
    dtype: str = "bfloat16"
    revision: Optional[str] = None
    trust_remote_code: bool = False

    @classmethod
    def from_dict(cls, name: str, value: Mapping[str, Any]) -> "ModelRuntimeConfig":
        model_path = str(value.get("model_path", "")).strip()
        if not model_path:
            raise ValueError(f"{name}.model_path must not be empty")
        gpu_ids = _gpu_ids(f"{name}.gpu_ids", value.get("gpu_ids"))
        tp_size = _positive_int(f"{name}.tp_size", value.get("tp_size"))
        if tp_size != len(gpu_ids):
            raise ValueError(f"{name}.tp_size must equal the number of GPU IDs")
        dtype = str(value.get("dtype", "bfloat16"))
        if dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError(f"unsupported {name}.dtype: {dtype}")
        revision = value.get("revision")
        if revision is not None:
            revision = str(revision)
        trust = value.get("trust_remote_code", False)
        if not isinstance(trust, bool):
            raise ValueError(f"{name}.trust_remote_code must be boolean")
        return cls(model_path, gpu_ids, tp_size, dtype, revision, trust)

    def with_overrides(
        self,
        *,
        model_path: Optional[str] = None,
        gpu_ids: Optional[Sequence[int]] = None,
        tp_size: Optional[int] = None,
        dtype: Optional[str] = None,
    ) -> "ModelRuntimeConfig":
        values = replace(
            self,
            model_path=model_path or self.model_path,
            gpu_ids=tuple(gpu_ids) if gpu_ids is not None else self.gpu_ids,
            tp_size=tp_size if tp_size is not None else self.tp_size,
            dtype=dtype or self.dtype,
        )
        return self.from_dict(
            "model",
            {
                "model_path": values.model_path,
                "gpu_ids": values.gpu_ids,
                "tp_size": values.tp_size,
                "dtype": values.dtype,
                "revision": values.revision,
                "trust_remote_code": values.trust_remote_code,
            },
        )


@dataclass(frozen=True)
class BenchmarkConfig:
    warmup_iterations: int = 3
    measured_iterations: int = 10
    request_batch_sizes: tuple[int, ...] = (1, 4)
    search_pool_sizes: tuple[int, ...] = (8, 32)
    verify_candidate_sizes: tuple[int, ...] = (4, 16)
    context_lengths: tuple[int, ...] = (128, 512)
    transfer_payload_bytes: tuple[int, ...] = (4096, 65536)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BenchmarkConfig":
        def positive_tuple(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
            raw = value.get(name, default)
            if not isinstance(raw, (list, tuple)) or not raw:
                raise ValueError(f"benchmark.{name} must be a non-empty list")
            return tuple(_positive_int(f"benchmark.{name}", item) for item in raw)

        return cls(
            warmup_iterations=_positive_int(
                "benchmark.warmup_iterations", value.get("warmup_iterations", 3)
            ),
            measured_iterations=_positive_int(
                "benchmark.measured_iterations", value.get("measured_iterations", 10)
            ),
            request_batch_sizes=positive_tuple("request_batch_sizes", (1, 4)),
            search_pool_sizes=positive_tuple("search_pool_sizes", (8, 32)),
            verify_candidate_sizes=positive_tuple("verify_candidate_sizes", (4, 16)),
            context_lengths=positive_tuple("context_lengths", (128, 512)),
            transfer_payload_bytes=positive_tuple(
                "transfer_payload_bytes", (4096, 65536)
            ),
        )


@dataclass(frozen=True)
class Phase3Config:
    schema_version: str
    backend: str
    draft: ModelRuntimeConfig
    target: ModelRuntimeConfig
    context_length: int
    batch_size: int
    search_pool_size: int
    candidate_budget: int
    candidate_width: int
    max_new_tokens: int
    random_seed: int
    sampling_configuration: dict[str, Any] = field(default_factory=dict)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Phase3Config":
        schema = str(value.get("schema_version", ""))
        if schema != "specrhythm.phase3-config.v1":
            raise ValueError("Phase-3 config schema_version must be specrhythm.phase3-config.v1")
        backend = str(value.get("backend", "transformers"))
        if backend not in {"dry-run", "transformers"}:
            raise ValueError("Phase-3 backend must be dry-run or transformers")
        sampling = dict(value.get("sampling_configuration", {"do_sample": False}))
        if sampling.get("do_sample", False):
            raise ValueError("Phase-3 v0.1 supports deterministic greedy sampling only")
        config = cls(
            schema_version=schema,
            backend=backend,
            draft=ModelRuntimeConfig.from_dict("draft", value.get("draft", {})),
            target=ModelRuntimeConfig.from_dict("target", value.get("target", {})),
            context_length=_positive_int("context_length", value.get("context_length")),
            batch_size=_positive_int("batch_size", value.get("batch_size")),
            search_pool_size=_positive_int(
                "search_pool_size", value.get("search_pool_size")
            ),
            candidate_budget=_positive_int(
                "candidate_budget", value.get("candidate_budget")
            ),
            candidate_width=_positive_int(
                "candidate_width", value.get("candidate_width")
            ),
            max_new_tokens=_positive_int(
                "max_new_tokens", value.get("max_new_tokens")
            ),
            random_seed=int(value.get("random_seed", 1664)),
            sampling_configuration=sampling,
            benchmark=BenchmarkConfig.from_dict(value.get("benchmark", {})),
        )
        if config.candidate_budget > config.search_pool_size:
            raise ValueError("candidate_budget cannot exceed search_pool_size")
        return config

    def with_overrides(self, **values: Any) -> "Phase3Config":
        allowed = {
            "backend",
            "context_length",
            "batch_size",
            "search_pool_size",
            "candidate_budget",
            "max_new_tokens",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown Phase-3 override(s): {sorted(unknown)}")
        retained = {key: value for key, value in values.items() if value is not None}
        updated = replace(self, **retained)
        if updated.backend not in {"dry-run", "transformers"}:
            raise ValueError("Phase-3 backend must be dry-run or transformers")
        for name in (
            "context_length",
            "batch_size",
            "search_pool_size",
            "candidate_budget",
            "max_new_tokens",
        ):
            _positive_int(name, getattr(updated, name))
        if updated.candidate_budget > updated.search_pool_size:
            raise ValueError("candidate_budget cannot exceed search_pool_size")
        return updated


def _read_config(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as error:
            raise ValueError(
                "non-JSON YAML requires the optional Phase-3 dependency PyYAML"
            ) from error
        value = yaml.safe_load(text)
    if not isinstance(value, Mapping):
        raise ValueError("Phase-3 config root must be a mapping")
    return value


def load_phase3_config(path: str) -> Phase3Config:
    """Load JSON-compatible YAML without making PyYAML a base dependency."""

    return Phase3Config.from_dict(_read_config(Path(path)))


def resolve_runtime_path(value: str, *, dry_run: bool = False) -> str:
    """Expand environment variables and reject unresolved real-runtime paths."""

    expanded = os.path.expandvars(os.path.expanduser(value))
    if not dry_run and "$" in expanded:
        raise ValueError(f"unresolved environment variable in path: {value}")
    return expanded
