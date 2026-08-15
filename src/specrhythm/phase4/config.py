"""Configuration and semantic guardrails for the frozen Phase-4A workflow."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from specrhythm.phase4.contracts import GreedySamplingContract

VLLM_VERSION = "0.25.1"
VLLM_COMMIT = "752a3a504485790a2e8491cacbb35c137339ad34"
PYTHON_SERIES = "3.11"
PYTORCH_VERSION = "2.11.0"


def _nonempty(name: str, value: Any) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _gpu_ids(name: str, value: Any) -> Tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must contain physical GPU IDs")
    result = tuple(value)
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in result):
        raise ValueError(f"{name} must contain non-negative integer physical GPU IDs")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


@dataclass(frozen=True)
class EngineConfig:
    role: str
    model_path: str
    tokenizer_path: str
    physical_gpu_ids: Tuple[int, ...]
    tensor_parallel_size: int
    dtype: str
    revision: Optional[str]
    tokenizer_revision: Optional[str]
    trust_remote_code: bool
    gpu_memory_utilization: float

    @classmethod
    def from_dict(cls, role: str, value: Mapping[str, Any]) -> "EngineConfig":
        if role not in {"draft", "target"}:
            raise ValueError("engine role must be draft or target")
        model = _nonempty(f"{role}.model_path", value.get("model_path"))
        tokenizer = _nonempty(f"{role}.tokenizer_path", value.get("tokenizer_path", model))
        gpus = _gpu_ids(f"{role}.physical_gpu_ids", value.get("physical_gpu_ids"))
        tp = value.get("tensor_parallel_size")
        if not isinstance(tp, int) or isinstance(tp, bool) or tp != len(gpus):
            raise ValueError(f"{role}.tensor_parallel_size must equal its physical GPU count")
        dtype = str(value.get("dtype", "bfloat16"))
        if dtype not in {"bfloat16", "float16"}:
            raise ValueError(f"unsupported Phase-4 dtype: {dtype}")
        trust = value.get("trust_remote_code", False)
        if not isinstance(trust, bool):
            raise ValueError(f"{role}.trust_remote_code must be boolean")
        utilization = value.get("gpu_memory_utilization", 0.85)
        if not isinstance(utilization, (int, float)) or not 0 < float(utilization) < 1:
            raise ValueError(f"{role}.gpu_memory_utilization must be in (0, 1)")
        revision = value.get("revision")
        tokenizer_revision = value.get("tokenizer_revision")
        return cls(
            role=role,
            model_path=model,
            tokenizer_path=tokenizer,
            physical_gpu_ids=gpus,
            tensor_parallel_size=tp,
            dtype=dtype,
            revision=None if revision is None else str(revision),
            tokenizer_revision=(None if tokenizer_revision is None else str(tokenizer_revision)),
            trust_remote_code=trust,
            gpu_memory_utilization=float(utilization),
        )

    @property
    def resolved_model_path(self) -> Path:
        value = os.path.expandvars(os.path.expanduser(self.model_path))
        if "$" in value:
            raise ValueError(f"unresolved {self.role} model path: {self.model_path}")
        return Path(value).resolve()

    @property
    def resolved_tokenizer_path(self) -> Path:
        value = os.path.expandvars(os.path.expanduser(self.tokenizer_path))
        if "$" in value:
            raise ValueError(f"unresolved {self.role} tokenizer path: {self.tokenizer_path}")
        return Path(value).resolve()


@dataclass(frozen=True)
class ModeContract:
    name: str
    draft_enabled: bool
    target_enabled: bool
    cross_engine_overlap: bool
    built_in_vllm_speculative: bool
    vllm_dbo: bool

    @classmethod
    def from_dict(cls, name: str, value: Mapping[str, Any]) -> "ModeContract":
        mode = cls(
            name,
            bool(value.get("draft_enabled")),
            bool(value.get("target_enabled")),
            bool(value.get("cross_engine_overlap")),
            bool(value.get("built_in_vllm_speculative")),
            bool(value.get("vllm_dbo")),
        )
        expected = {
            "target-only": (False, True, False),
            "serial-disaggregated": (True, True, False),
            "dual-batch": (True, True, True),
        }
        if name not in expected:
            raise ValueError(f"unsupported Phase-4 mode: {name}")
        if (mode.draft_enabled, mode.target_enabled, mode.cross_engine_overlap) != expected[name]:
            raise ValueError(f"{name} has an invalid engine/overlap contract")
        if mode.built_in_vllm_speculative:
            raise ValueError(
                "vLLM colocated speculative decoding cannot represent a disaggregated mode"
            )
        if mode.vllm_dbo:
            raise ValueError("vLLM DBO cannot be labeled as SpecRhythm draft/verify Dual-Batch")
        return mode


@dataclass(frozen=True)
class Phase4Config:
    path: Path
    draft: EngineConfig
    target: EngineConfig
    modes: Tuple[ModeContract, ...]
    sampling: GreedySamplingContract
    max_model_len: int
    smoke_request_count: int
    logprobs: int
    enforce_eager: bool
    enable_prefix_caching: bool
    proposal_budget: int
    target_model_runner: str
    enable_thinking: bool
    expected_vllm_version: str
    expected_vllm_commit: str
    expected_python_series: str
    expected_pytorch_version: str

    def __post_init__(self) -> None:
        if set(self.draft.physical_gpu_ids) & set(self.target.physical_gpu_ids):
            raise ValueError("draft and target physical GPUs must be disjoint")
        if tuple(mode.name for mode in self.modes) != (
            "target-only",
            "serial-disaggregated",
            "dual-batch",
        ):
            raise ValueError("Phase-4 future mode order is frozen")
        if self.expected_vllm_version != VLLM_VERSION:
            raise ValueError(f"Phase-4 freezes vLLM {VLLM_VERSION}")
        if self.expected_vllm_commit != VLLM_COMMIT:
            raise ValueError(f"Phase-4 freezes vLLM commit {VLLM_COMMIT}")
        if self.expected_python_series != PYTHON_SERIES:
            raise ValueError("Phase-4 integration environment must use Python 3.11")
        if self.expected_pytorch_version != PYTORCH_VERSION:
            raise ValueError("Phase-4 integration environment must use PyTorch 2.11.0")
        if self.max_model_len < 1 or self.smoke_request_count != 5 or self.logprobs < 5:
            raise ValueError("invalid Phase-4 context/request-count/logprob contract")
        if self.proposal_budget != 4:
            raise ValueError("Phase-4A.1 freezes the linear proposal budget at K=4")
        if self.target_model_runner != "v1":
            raise ValueError("Phase-4A.1 custom proposer requires frozen vLLM model runner v1")
        if self.enable_thinking:
            raise ValueError("Phase-4 freezes Qwen chat rendering with enable_thinking=false")
        if not self.enforce_eager:
            raise ValueError("stock bring-up must use enforce_eager for deterministic inspection")


def _read_mapping(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as error:
            raise ValueError("non-JSON YAML requires optional PyYAML") from error
        value = yaml.safe_load(text)
    if not isinstance(value, Mapping):
        raise ValueError("Phase-4 config root must be a mapping")
    return value


def load_phase4_config(path: str) -> Phase4Config:
    config_path = Path(path).resolve()
    value = _read_mapping(config_path)
    if value.get("schema_version") != "specrhythm.phase4a-config.v1":
        raise ValueError("unsupported Phase-4 config schema_version")
    pins = value.get("framework_freeze", {})
    modes = value.get("future_modes")
    if not isinstance(modes, Mapping):
        raise ValueError("future_modes must be a mapping")
    sampling = value.get("sampling", {})
    return Phase4Config(
        path=config_path,
        draft=EngineConfig.from_dict("draft", value.get("draft", {})),
        target=EngineConfig.from_dict("target", value.get("target", {})),
        modes=tuple(
            ModeContract.from_dict(name, modes.get(name, {}))
            for name in ("target-only", "serial-disaggregated", "dual-batch")
        ),
        sampling=GreedySamplingContract(
            do_sample=sampling.get("do_sample", False),
            temperature=float(sampling.get("temperature", 0.0)),
            top_p=float(sampling.get("top_p", 1.0)),
            n=sampling.get("n", 1),
            best_of=sampling.get("best_of", 1),
            seed=sampling.get("seed", 1664),
        ),
        max_model_len=value.get("max_model_len", 4096),
        smoke_request_count=value.get("smoke_request_count", 5),
        logprobs=value.get("logprobs", 5),
        enforce_eager=bool(value.get("enforce_eager", True)),
        enable_prefix_caching=bool(value.get("enable_prefix_caching", False)),
        proposal_budget=value.get("proposal_budget", 4),
        target_model_runner=str(value.get("target_model_runner", "v1")),
        enable_thinking=bool(value.get("enable_thinking", False)),
        expected_vllm_version=str(pins.get("vllm_version", "")),
        expected_vllm_commit=str(pins.get("vllm_commit", "")),
        expected_python_series=str(pins.get("python_series", "")),
        expected_pytorch_version=str(pins.get("pytorch_version", "")),
    )
