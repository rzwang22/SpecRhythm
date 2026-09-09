"""Explicit resident S1 input profile; legacy SmokeRequest remains unchanged."""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from specrhythm.phase4.manifest import sha256_file
from specrhythm.serving.common import digest, integer, read_json, require
from specrhythm.serving.schema import ServingWorkloadRequest, load_requests

PROFILE_ENV = "SR_S1_EXECUTION_MANIFEST"
SCHEMA = "specrhythm.s1-execution.v1"
S0_COMMIT = "0dd5750384eb63bd4d2ef3b32163d819862e0fbd"
FROZEN = {
    "main1000.jsonl": "19979cc335b48a2a5c2d64e2a18c29b0b523d9a970733f4c6744eb5cea4779ef",
    "calibration200.jsonl": "7a1278243b7d2aa5a71d4082eafb22f29dfa5e281b047c1443295684eb7652b9",
    "workload-manifest.json": "a2cb9bda2ad723cf6ac1050bca7e0a7818574d6617b9352bc30db70b0cce9703",
}
SEMANTIC = "05b5f2efad0a4ac1771c9a18d847c08d95d360432e1c9cfa55a82ba2f63cba02"
QUOTAS = {
    "s1-smoke4": dict(chat=1, code=1, summarization=1, reasoning=1),
    "s1-mixed20": dict(chat=6, code=6, summarization=4, reasoning=4),
    "s1-mixed100": dict(chat=30, code=30, summarization=20, reasoning=20),
}
MODES = ("target", "serial", "pingpong")


def write_once(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def verify_s0(directory: Path) -> dict:
    """Read sealed evidence, not the absent original sources or model tokenizer."""
    inventory = {}
    for line in (directory / "checksums.sha256").read_text().splitlines():
        checksum, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        path = directory / name
        require(
            not Path(name).is_absolute() and ".." not in Path(name).parts,
            "unsafe S0 checksum path",
        )
        require(path.is_file() and not path.is_symlink(), "missing S0 inventory file", file=name)
        require(
            name not in inventory and sha256_file(path) == checksum,
            "S0 inventory checksum mismatch",
            file=name,
        )
        inventory[name] = checksum
    require(len(inventory) == 35, "S0 sealed inventory must contain 35 entries")
    for name, checksum in FROZEN.items():
        require(inventory.get(name) == checksum, "S0 frozen identity mismatch", file=name)
    manifest = read_json(directory / "workload-manifest.json")
    require(manifest["semantic_workload_sha256"] == SEMANTIC, "S0 semantic identity mismatch")
    server = read_json(directory / "server-tokenizer-validation.json")
    require(
        server.get("valid") is True and server.get("validated_prompt_count") == 1200,
        "S0 server tokenizer validation missing",
    )
    require(
        server.get("workload_manifest_sha256") == FROZEN["workload-manifest.json"],
        "S0 tokenizer report parent binding mismatch",
    )
    # The report's exact shape is frozen by the acceptance bundle.
    require(
        server.get("workload_files") == {k: v for k, v in FROZEN.items() if k.endswith(".jsonl")},
        "S0 tokenizer report workload binding mismatch",
    )
    acceptance = read_json(directory / "server-acceptance-summary.json")
    require(
        acceptance.get("machine_data_acceptance") == "PASS"
        and acceptance.get("server_local_tokenizer_verification") == "PASS"
        and acceptance.get("rebuild") == "PASS",
        "S0 server acceptance missing",
    )
    return {
        "directory": str(directory.resolve()),
        "inventory": inventory,
        "checksums_sha256": sha256_file(directory / "checksums.sha256"),
        "semantic_workload_sha256": SEMANTIC,
        "data_git_commit": S0_COMMIT,
        "evidence_scope": "sealed server reports and file bindings; no source/tokenizer rerun",
    }


def select_subset(rows, quotas):
    selected, seen = [], Counter()
    for row in rows:
        if seen[row.task_class] < quotas.get(row.task_class, 0):
            selected.append(row)
            seen[row.task_class] += 1
    require(dict(seen) == quotas, "insufficient class quota in frozen main workload")
    return selected


def create_manifest(directory, subset, rows, parent, execution):
    require(subset in QUOTAS, "unknown S1 gate subset")
    chosen = select_subset(rows, QUOTAS[subset])
    directory.mkdir(parents=True, exist_ok=False)
    workload = directory / "requests.jsonl"
    # Preserve every field, including arrival/source provenance. Never tokenize here.
    with workload.open("x", encoding="utf-8") as handle:
        for row in chosen:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    ids = [r.request_id for r in chosen]
    logical = {
        "parent_semantic_sha256": SEMANTIC,
        "subset": subset,
        "request_ids": ids,
        "workload_sha256": sha256_file(workload),
        "arrival_replay_enabled": False,
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
            "ignore_eos": False,
            "min_tokens": 0,
            "stop": [],
            "stop_token_ids": [],
            "enable_thinking": False,
        },
        "requests": [
            {
                "request_id": r.request_id,
                "maximum_new_tokens": r.maximum_new_tokens,
                "sampling_seed": r.sampling_seed,
            }
            for r in chosen
        ],
    }
    value = {
        "schema_version": SCHEMA,
        "logical": logical,
        "logical_sha256": digest(logical),
        "parent": parent,
        "execution": execution,
        "workload_file": workload.name,
        "request_ids_sha256": digest(ids),
        "assignment": {rid: "AB"[i % 2] for i, rid in enumerate(ids)},
    }
    value["manifest_sha256"] = digest(value)
    write_once(directory / "execution-manifest.json", value)
    return value


def load_execution(path: Path, *, verify_parent=False):
    value = read_json(path)
    require(value.get("schema_version") == SCHEMA, "unsupported S1 execution schema")
    require(
        value.get("manifest_sha256")
        == digest({k: v for k, v in value.items() if k != "manifest_sha256"}),
        "S1 execution manifest hash mismatch",
    )
    logical = value["logical"]
    require(value["logical_sha256"] == digest(logical), "S1 logical identity mismatch")
    require(logical.get("arrival_replay_enabled") is False, "S1 cannot replay arrivals")
    workload = path.parent / value["workload_file"]
    require(workload.resolve().parent == path.resolve().parent, "unsafe S1 workload path")
    require(sha256_file(workload) == logical["workload_sha256"], "S1 workload hash mismatch")
    rows = load_requests(workload)
    ids = [r.request_id for r in rows]
    require(
        ids == logical["request_ids"] and digest(ids) == value["request_ids_sha256"],
        "S1 stable request order mismatch",
    )
    require(
        value["assignment"] == {rid: "AB"[i % 2] for i, rid in enumerate(ids)},
        "S1 immutable cohort assignment mismatch",
    )
    require(
        logical["requests"]
        == [
            dict(
                request_id=r.request_id,
                maximum_new_tokens=r.maximum_new_tokens,
                sampling_seed=r.sampling_seed,
            )
            for r in rows
        ],
        "S1 per-request sampling identity mismatch",
    )
    capacity = value["execution"]["capacity"]
    require(len(rows) <= capacity["max_num_seqs"], "S1 sequence capacity insufficient")
    for r in rows:
        require(
            r.prompt_length + r.maximum_new_tokens + 4 <= capacity["max_model_len"],
            "S1 context including speculative reserve insufficient",
            request_id=r.request_id,
        )
    if verify_parent:
        require(
            verify_s0(Path(value["parent"]["directory"])) == value["parent"],
            "S0 sealed parent changed",
        )
        original = load_requests(Path(value["parent"]["directory"]) / "main1000.jsonl")
        require(
            [r.to_dict() for r in rows]
            == [r.to_dict() for r in select_subset(original, QUOTAS[logical["subset"]])],
            "S1 subset differs from frozen selection",
        )
    return value, rows


@dataclass(frozen=True)
class ResidentServingRequest:
    """Lossless transport adapter: only the container type of token IDs changes."""

    source: ServingWorkloadRequest

    @property
    def prompt_token_ids(self):
        return tuple(self.source.prompt_token_ids)

    def __getattr__(self, name):
        return getattr(self.source, name)


def s1_enabled():
    return bool(os.environ.get(PROFILE_ENV))


@lru_cache(maxsize=8)
def _profile(path):
    return load_execution(Path(path))


def active_profile():
    require(s1_enabled(), "S1 execution manifest is required")
    return _profile(os.environ[PROFILE_ENV])


def load_runtime_requests(path, expected_count=5, require_task_mixture=True):
    if not s1_enabled():
        from specrhythm.phase4.stock_vllm import load_smoke_requests

        return load_smoke_requests(path, expected_count, require_task_mixture=require_task_mixture)
    manifest, rows = active_profile()
    require(
        integer(expected_count, 1) and len(rows) == expected_count,
        "S1 runtime request count mismatch",
    )
    require(
        sha256_file(Path(path)) == manifest["logical"]["workload_sha256"],
        "S1 runtime workload differs from execution manifest",
    )
    return tuple(ResidentServingRequest(r) for r in rows)


def target_options():
    if not s1_enabled():
        return {}
    capacity = active_profile()[0]["execution"]["capacity"]
    return {k: capacity[k] for k in ("max_num_seqs", "max_num_batched_tokens")}


def setup_terminal_ids(manifest):
    if not s1_enabled():
        return set()
    execution, rows = active_profile()
    definitions = {r.request_id: r for r in rows}
    eos = execution["execution"]["eos_token_ids"]
    return {
        r.request_id
        for r in manifest.requests
        if definitions[r.request_id].maximum_new_tokens == 1 or r.bootstrap_token_id in eos
    }


def initial_proposal_excluded_ids(manifest):
    if not s1_enabled():
        return set()
    _, rows = active_profile()
    return setup_terminal_ids(manifest) | {r.request_id for r in rows if r.maximum_new_tokens <= 2}


def initial_target_tail(request_id, output_count):
    if not s1_enabled() or output_count != 1:
        return False
    return any(
        r.request_id == request_id and r.maximum_new_tokens == 2 for r in active_profile()[1]
    )


def validate_prompt_ids(tokenizer, request):
    encoded = tokenizer.encode(request.prompt_text, add_special_tokens=not s1_enabled())
    require(
        list(encoded) == list(request.prompt_token_ids),
        "vLLM tokenizer disagrees with frozen prompt IDs",
        request_id=request.request_id,
    )


def require_reference_or_s1(reference_path, performance):
    require(
        reference_path is not None or (s1_enabled() and performance),
        "legacy resident execution requires its immutable stock reference",
    )


def pending_reference_comparison():
    return {
        "performed": False,
        "all_sequences_equal": None,
        "valid": None,
        "errors": [],
        "reason": "S1 exact comparison required after independent runs",
    }
