"""Independent serving-workload schema, not the legacy 2/5/100 SmokeRequest."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from specrhythm.serving.common import (
    CONFIG_SCHEMA,
    REQUEST_SCHEMA,
    TASKS,
    finite,
    integer,
    is_hash,
    read_json,
    require,
    text_hash,
)


@dataclass(frozen=True)
class ServingWorkloadRequest:
    schema_version: str
    workload_id: str
    split: str
    request_id: str
    task_class: str
    language: str
    source_ref: dict
    messages: list[dict]
    prompt_text: str
    prompt_token_ids: list[int]
    prompt_length: int
    prompt_sha256: str
    tokenizer_fingerprint: str
    arrival_time_ms: float
    arrival_source_ref: dict
    maximum_new_tokens: int
    sampling_seed: int
    slo_class: str

    def __post_init__(self):
        require(self.schema_version == REQUEST_SCHEMA, "unsupported serving request schema")
        for field in ("workload_id", "split", "request_id"):
            require(
                isinstance(getattr(self, field), str) and getattr(self, field).strip(),
                "missing serving request identity",
                field=field,
            )
        require(
            self.task_class in TASKS and self.slo_class == self.task_class,
            "invalid serving task/SLO class",
        )
        require(self.language == "English", "S0 requires English requests")
        require(
            isinstance(self.messages, list)
            and len(self.messages) == 1
            and isinstance(self.messages[0], dict)
            and set(self.messages[0]) == {"role", "content"}
            and self.messages[0]["role"] == "user"
            and isinstance(self.messages[0]["content"], str)
            and self.messages[0]["content"].strip(),
            "S0 requires one user message",
        )
        require(isinstance(self.prompt_text, str) and self.prompt_text, "empty templated prompt")
        require(
            isinstance(self.prompt_token_ids, list)
            and self.prompt_token_ids
            and all(integer(t) for t in self.prompt_token_ids),
            "invalid prompt token IDs",
        )
        require(
            integer(self.prompt_length, 1) and self.prompt_length == len(self.prompt_token_ids),
            "prompt_length differs from token IDs",
        )
        require(self.prompt_sha256 == text_hash(self.prompt_text), "prompt SHA256 mismatch")
        require(is_hash(self.tokenizer_fingerprint), "invalid tokenizer fingerprint")
        require(finite(self.arrival_time_ms), "arrival_time_ms must be finite and nonnegative")
        require(
            integer(self.maximum_new_tokens, 1) and integer(self.sampling_seed),
            "invalid generation budget or sampling seed",
        )
        require(
            isinstance(self.source_ref, dict)
            and all(
                isinstance(self.source_ref.get(k), str) and self.source_ref[k]
                for k in ("dataset_id", "revision", "config", "split", "id", "file")
            )
            and is_hash(self.source_ref["revision"], 40)
            and is_hash(self.source_ref.get("file_sha256"))
            and integer(self.source_ref.get("original_row_index"))
            and "group_id" in self.source_ref
            and (
                self.source_ref["group_id"] is None or isinstance(self.source_ref["group_id"], str)
            ),
            "invalid dataset source reference",
        )
        require(
            isinstance(self.arrival_source_ref, dict)
            and all(
                isinstance(self.arrival_source_ref.get(k), str) and self.arrival_source_ref[k]
                for k in ("repository_id", "file")
            )
            and is_hash(self.arrival_source_ref.get("revision"), 40)
            and is_hash(self.arrival_source_ref.get("file_sha256"))
            and integer(self.arrival_source_ref.get("original_line_index"), 1)
            and finite(self.arrival_source_ref.get("source_timestamp_ms")),
            "invalid arrival source reference",
        )

    @classmethod
    def from_dict(cls, value: Any) -> ServingWorkloadRequest:
        require(isinstance(value, dict), "serving JSONL row must be an object")
        return cls(**value)

    def to_dict(self) -> dict:
        return asdict(self)


def load_requests(path: Path) -> list[ServingWorkloadRequest]:
    """Read every row; quotas/counts belong to configuration, not this loader."""
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line, raw in enumerate(handle, 1):
            require(bool(raw.strip()), "blank serving JSONL record", line=line)
            try:
                rows.append(ServingWorkloadRequest.from_dict(json.loads(raw)))
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"serving JSONL line {line}: {error}") from error
    require(bool(rows), "serving workload is empty")
    require(len({r.request_id for r in rows}) == len(rows), "duplicate serving request ID")
    return rows


def load_config(path: Path) -> dict:
    config = read_json(path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "unsupported S0 configuration")
    require(
        config.get("language") == "English"
        and config.get("request_style") == "single-turn"
        and config.get("enable_thinking") is False
        and config.get("add_generation_prompt") is True
        and config.get("natural_eos") is True,
        "invalid S0 prompt/generation semantics",
    )
    require(integer(config.get("seed")) and integer(config.get("arrival_seed")), "invalid seed")
    require(
        integer(config.get("max_model_len"), 1)
        and integer(config.get("speculative_reserve_tokens")),
        "invalid context budget",
    )
    require(config.get("base_time_scale") == 1.0, "canonical S0 arrivals must be unscaled")
    require(
        config.get("slo_policy_ref") is None and config.get("calibration_status") == "pending",
        "S0 cannot claim a calibrated SLO",
    )
    require(set(config["splits"]) == {"main", "calibration"}, "S0 requires two independent splits")
    require(
        set(config["class_limits"]) == set(config["instructions"]) == set(TASKS),
        "S0 requires four task classes",
    )
    for task in TASKS:
        require(isinstance(config["instructions"][task], str), "instruction must be text")
        limits = config["class_limits"][task]
        require(
            integer(limits["prompt_tokens"], 1) and integer(limits["maximum_new_tokens"], 1),
            "class budgets must be positive integers",
        )
    for split in config["splits"].values():
        require(
            isinstance(split["workload_id"], str) and split["workload_id"], "empty workload ID"
        )
        require(
            set(split["quotas"]) == set(TASKS)
            and all(integer(n, 1) for n in split["quotas"].values()),
            "invalid exact quotas",
        )
    require(
        config["arrival"]
        == {
            "timestamp_field": "timestamp",
            "unit": "ms",
            "sort": "timestamp,original_line_index",
            "selection": "first-main-then-calibration",
            "start_index": 0,
        },
        "S0 requires the original first main/calibration trace positions",
    )
    return config


def require_calibrated(config: dict) -> None:
    """Future SLO consumers must call this on a separate experiment configuration."""
    require(
        config.get("calibration_status") == "complete" and config.get("slo_policy_ref"),
        "SLO policy is pending; it cannot be used as a calibrated experiment",
    )
