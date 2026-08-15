"""Deterministic R3-real public-text workload construction."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import random
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Protocol, TextIO

from specrhythm.phase3.engine import DryRunBackend
from specrhythm.phase3.phase3c_config import Phase3CConfig, PublicDatasetConfig
from specrhythm.phase3.trace import sha256_file
from specrhythm.workload import apportion_counts, select_arrival_replay

TASK_CLASSES = ("code", "chat", "summarization")
TASK_WEIGHTS = (0.6, 0.2, 0.2)
TASK_SLO_MS = {"code": 40.0, "chat": 50.0, "summarization": 150.0}


class PromptTokenizer(Protocol):
    model_id: str
    tokenizer_fingerprint: str

    def encode(self, prompt: str) -> list[int]: ...


class TransformersPromptTokenizer:
    """CPU tokenizer-only adapter; it does not load a language model."""

    def __init__(self, model_path: str, *, revision: Optional[str] = None) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "real R3 workload tokenization requires the optional transformers dependency"
            ) from error
        expanded = os.path.expandvars(os.path.expanduser(model_path))
        if "$" in expanded:
            raise ValueError(f"unresolved tokenizer model path: {model_path}")
        self.model_id = expanded
        self.tokenizer = AutoTokenizer.from_pretrained(expanded, revision=revision)
        payload = {
            "vocab": self.tokenizer.get_vocab(),
            "special_tokens_map": self.tokenizer.special_tokens_map,
        }
        self.tokenizer_fingerprint = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        ).hexdigest()

    def encode(self, prompt: str) -> list[int]:
        return list(self.tokenizer.encode(prompt, add_special_tokens=True))


@dataclass(frozen=True)
class R3RealRequest:
    request_id: str
    task_class: str
    source_dataset: str
    source_record_id: str
    source_split: str
    source_license_or_url: str
    raw_text_sha256: str
    prompt_text: str
    prompt_token_ids: tuple[int, ...]
    prompt_length: int
    tokenizer_fingerprint: str
    maximum_new_tokens: int
    slo_class: str
    arrival_timestamp: float
    sampling_seed: int
    data_split: str

    def __post_init__(self) -> None:
        if not self.request_id or self.task_class not in TASK_CLASSES:
            raise ValueError("invalid R3-real request identity or task_class")
        for name in (
            "source_dataset",
            "source_record_id",
            "source_split",
            "source_license_or_url",
            "raw_text_sha256",
            "prompt_text",
            "tokenizer_fingerprint",
            "slo_class",
            "data_split",
        ):
            if not getattr(self, name):
                raise ValueError(f"R3-real request {name} must not be empty")
        if len(self.raw_text_sha256) != 64:
            raise ValueError("raw_text_sha256 must be a SHA256 hex digest")
        if not self.prompt_token_ids or any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in self.prompt_token_ids
        ):
            raise ValueError("prompt_token_ids must contain non-negative integers")
        if self.prompt_length != len(self.prompt_token_ids):
            raise ValueError("prompt_length must equal the actual token-ID count")
        if (
            not isinstance(self.maximum_new_tokens, int)
            or isinstance(self.maximum_new_tokens, bool)
            or self.maximum_new_tokens < 1
        ):
            raise ValueError("maximum_new_tokens must be a positive integer")
        if (
            not isinstance(self.arrival_timestamp, (int, float))
            or isinstance(self.arrival_timestamp, bool)
            or not math.isfinite(self.arrival_timestamp)
            or self.arrival_timestamp < 0
        ):
            raise ValueError("arrival_timestamp must be finite and non-negative")
        if not isinstance(self.sampling_seed, int) or isinstance(
            self.sampling_seed, bool
        ):
            raise ValueError("sampling_seed must be an integer")
        if self.data_split not in {"diagnostic/train", "validation", "test"}:
            raise ValueError("invalid request-level data split")
        if self.slo_class != f"{int(TASK_SLO_MS[self.task_class])}ms":
            raise ValueError("SLO_class does not match the task metadata mapping")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["prompt_token_ids"] = list(self.prompt_token_ids)
        value["SLO_class"] = value.pop("slo_class")
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "R3RealRequest":
        copied = dict(value)
        copied["slo_class"] = str(copied.pop("SLO_class"))
        copied["prompt_token_ids"] = tuple(copied["prompt_token_ids"])
        return cls(**copied)


@dataclass(frozen=True)
class _SourcePrompt:
    record_id: str
    raw_text: str
    prompt_text: str


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def _rows(path: Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"required public dataset file is missing: {path}")
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from error
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number}: record must be a JSON object")
            yield line_number, value


def _record_id(row: Mapping[str, Any], line_number: int) -> str:
    for key in ("task_id", "conversation_id", "id", "record_id", "question_id"):
        if row.get(key) not in (None, ""):
            return str(row[key])
    return f"line-{line_number:08d}"


def _first_text(row: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _chat_prompt(row: Mapping[str, Any]) -> str:
    conversations = row.get("conversations", row.get("messages"))
    if isinstance(conversations, list):
        for message in conversations:
            if not isinstance(message, Mapping):
                continue
            role = str(message.get("from", message.get("role", ""))).lower()
            text = _first_text(message, ("value", "content", "text"))
            if role in {"human", "user", "prompter"} and text:
                return text
    return _first_text(row, ("prompt", "instruction", "text"))


def adapt_public_dataset(source: PublicDatasetConfig) -> list[_SourcePrompt]:
    """Read supported exported JSONL formats without downloading or substituting data."""

    prompts = []
    seen = set()
    for line_number, row in _rows(source.resolved_path):
        record_id = _record_id(row, line_number)
        if record_id in seen:
            raise ValueError(f"duplicate source record ID in {source.dataset}: {record_id}")
        seen.add(record_id)
        if source.adapter == "humaneval":
            raw = _first_text(row, ("prompt",))
            prompt = raw
        elif source.adapter == "mbpp":
            raw = _first_text(row, ("text", "prompt"))
            prompt = raw
        elif source.adapter in {"sharegpt", "openassistant"}:
            raw = _chat_prompt(row)
            prompt = raw
        elif source.adapter == "cnn_dailymail":
            raw = _first_text(row, ("article",))
            prompt = f"Summarize the following document:\n\n{raw}" if raw else ""
        elif source.adapter == "xsum":
            raw = _first_text(row, ("document",))
            prompt = f"Summarize the following document:\n\n{raw}" if raw else ""
        elif source.adapter == "govreport":
            raw = _first_text(row, ("report", "document"))
            prompt = f"Summarize the following document:\n\n{raw}" if raw else ""
        else:  # guarded by config parsing
            raise ValueError(f"unsupported adapter: {source.adapter}")
        if raw and prompt:
            prompts.append(_SourcePrompt(record_id, raw, prompt))
    if not prompts:
        raise ValueError(f"dataset {source.dataset} contains no valid {source.adapter} prompts")
    return prompts


def stable_request_split(request_id: str) -> str:
    """Stable request-level 70/15/15 split; nodes never choose their own split."""

    bucket = int(hashlib.sha256(request_id.encode()).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "diagnostic/train"
    if bucket < 85:
        return "validation"
    return "test"


def _sample_source(
    source: PublicDatasetConfig, prompts: list[_SourcePrompt], count: int, seed: int
) -> list[_SourcePrompt]:
    ranked = sorted(
        prompts,
        key=lambda item: (
            hashlib.sha256(
                f"{seed}\0{source.dataset}\0{item.record_id}\0{item.raw_text}".encode()
            ).hexdigest(),
            item.record_id,
        ),
    )
    if len(ranked) < count:
        raise ValueError(
            f"dataset {source.dataset} has {len(ranked)} valid records; {count} required"
        )
    return ranked


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def workload_payload(requests: Iterable[R3RealRequest]) -> str:
    return "".join(
        json.dumps(request.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
        for request in requests
    )


def load_r3_workload(path: Path) -> list[R3RealRequest]:
    requests = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                request = R3RealRequest.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: invalid R3-real workload row") from error
            if request.request_id in seen:
                raise ValueError(f"duplicate R3-real request_id: {request.request_id}")
            seen.add(request.request_id)
            requests.append(request)
    if not requests:
        raise ValueError("R3-real workload contains no requests")
    arrivals = [request.arrival_timestamp for request in requests]
    if arrivals != sorted(arrivals):
        raise ValueError("R3-real arrival timestamps must be monotonic")
    return requests


def create_workload_tokenizer(config: Phase3CConfig) -> PromptTokenizer:
    if config.runtime.backend == "dry-run":
        return DryRunBackend(config.workload.tokenizer_model, config.runtime.random_seed)
    return TransformersPromptTokenizer(
        config.workload.tokenizer_model, revision=config.runtime.draft.revision
    )


def build_r3_real_workload(
    config: Phase3CConfig,
    *,
    output_path: Path,
    manifest_path: Path,
    command: str,
    request_count: Optional[int] = None,
    tokenizer: Optional[PromptTokenizer] = None,
    git_commit: Optional[str] = None,
) -> dict[str, Any]:
    count = request_count if request_count is not None else config.workload.request_count
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("request_count must be a positive integer")
    counts = apportion_counts(TASK_WEIGHTS, count)
    source_by_task = {source.task_class: source for source in config.workload.sources}
    sampled = {
        task: _sample_source(
            source_by_task[task],
            adapt_public_dataset(source_by_task[task]),
            task_count,
            config.runtime.random_seed,
        )
        for task, task_count in zip(TASK_CLASSES, counts)
    }
    arrival_path = config.workload.resolve(config.workload.arrival_trace)
    replay = select_arrival_replay(
        arrival_path, time_scale=config.workload.time_scale
    )
    if len(replay.arrival_times_ms) < count:
        raise ValueError(
            f"Mooncake trace has {len(replay.arrival_times_ms)} arrivals; {count} required"
        )
    assignments = [
        task for task, task_count in zip(TASK_CLASSES, counts) for _ in range(task_count)
    ]
    random.Random(config.runtime.random_seed).shuffle(assignments)
    cursors = {task: 0 for task in TASK_CLASSES}
    context_rejections = {task: 0 for task in TASK_CLASSES}
    owns_tokenizer = tokenizer is None
    tokenizer = tokenizer or create_workload_tokenizer(config)
    requests = []
    try:
        for index, (task, arrival) in enumerate(
            zip(assignments, replay.arrival_times_ms[:count])
        ):
            source = source_by_task[task]
            while True:
                if cursors[task] >= len(sampled[task]):
                    raise ValueError(
                        f"dataset {source.dataset} has insufficient prompts within "
                        f"context_length={config.runtime.context_length}"
                    )
                item = sampled[task][cursors[task]]
                cursors[task] += 1
                token_ids = tuple(tokenizer.encode(item.prompt_text))
                if (
                    len(token_ids) + config.workload.maximum_new_tokens
                    <= config.runtime.context_length
                ):
                    break
                context_rejections[task] += 1
            raw_hash = hashlib.sha256(item.raw_text.encode()).hexdigest()
            request_id = "r3-" + hashlib.sha256(
                f"{config.runtime.random_seed}\0{task}\0{source.dataset}\0"
                f"{item.record_id}\0{raw_hash}".encode()
            ).hexdigest()[:24]
            requests.append(
                R3RealRequest(
                    request_id=request_id,
                    task_class=task,
                    source_dataset=source.dataset,
                    source_record_id=item.record_id,
                    source_split=source.split,
                    source_license_or_url=source.source_license_or_url,
                    raw_text_sha256=raw_hash,
                    prompt_text=item.prompt_text,
                    prompt_token_ids=token_ids,
                    prompt_length=len(token_ids),
                    tokenizer_fingerprint=tokenizer.tokenizer_fingerprint,
                    maximum_new_tokens=config.workload.maximum_new_tokens,
                    slo_class=f"{int(TASK_SLO_MS[task])}ms",
                    arrival_timestamp=float(arrival),
                    sampling_seed=config.runtime.random_seed + index,
                    data_split=stable_request_split(request_id),
                )
            )
    finally:
        if owns_tokenizer and hasattr(tokenizer, "close"):
            tokenizer.close()  # type: ignore[attr-defined]
    payload = workload_payload(requests)
    _atomic_write(output_path, payload)
    workload_sha = hashlib.sha256(payload.encode()).hexdigest()
    tokenizer_config = Path(tokenizer.model_id) / "tokenizer_config.json"
    sources = []
    for source, used in zip(config.workload.sources, counts):
        sources.append(
            {
                "task_class": source.task_class,
                "dataset": source.dataset,
                "adapter": source.adapter,
                "source_split": source.split,
                "source_license_or_url": source.source_license_or_url,
                "source_file": source.resolved_path.name,
                "source_file_sha256": sha256_file(source.resolved_path),
                "records_used": used,
                "records_rejected_for_context": context_rejections[source.task_class],
            }
        )
    manifest = {
        "schema_version": "specrhythm.r3-real-workload-manifest.v1",
        "workload_family": "R3-real-pilot",
        "evidence_scope": "schema-and-selector-signal-pilot-only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "seed": config.runtime.random_seed,
        "request_count": len(requests),
        "task_counts": dict(zip(TASK_CLASSES, counts)),
        "task_slo_tpot_ms_metadata_only": TASK_SLO_MS,
        "slo_used_for_scheduling": False,
        "arrival_source_file": arrival_path.name,
        "arrival_source_sha256": sha256_file(arrival_path),
        "arrival_selection": "first-N-valid-chronological-rebased-to-zero",
        "arrival_timestamp_unit": "ms",
        "time_scale": config.workload.time_scale,
        "tokenizer_model": Path(tokenizer.model_id).name,
        "tokenizer_fingerprint": tokenizer.tokenizer_fingerprint,
        "tokenizer_config_sha256": (
            sha256_file(tokenizer_config) if tokenizer_config.is_file() else None
        ),
        "config_file": config.path.name,
        "config_sha256": sha256_file(config.path),
        "prompt_lengths_are_proxy": False,
        "sources": sources,
        "output_file": output_path.name,
        "output_workload_sha256": workload_sha,
        "command": command,
    }
    _atomic_write(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    return manifest
