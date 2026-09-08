"""S0 canonical data construction and immutable artifacts, without a request runner."""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import platform
import socket
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from specrhythm.serving.arrival import arrival_summary, read_arrivals, windows
from specrhythm.serving.common import (
    ALGORITHM,
    MANIFEST_SCHEMA,
    REQUEST_SCHEMA,
    TASKS,
    canonical,
    digest,
    distribution,
    file_inventory,
    read_json,
    require,
    text_hash,
    write_json,
)
from specrhythm.serving.schema import ServingWorkloadRequest, load_config
from specrhythm.serving.selection import choose
from specrhythm.serving.sources import verify_sources
from specrhythm.serving.tokenization import verify_locked_tokenizer


def plan(config: dict, lock: dict, sources: Path, tokenizer) -> tuple[dict, dict, dict]:
    require(
        tokenizer.is_real or lock["data_kind"] == "synthetic-fixture",
        "a fixture tokenizer cannot construct a real workload",
    )
    arrival_windows = windows(read_arrivals(lock, sources), config)
    selected, selection = choose(lock, sources, config, tokenizer)
    output = {}
    for split in ("main", "calibration"):
        # A separate seed orders exact class-slot quotas, not the source timestamps.
        slots = [(task, i) for task in TASKS for i in range(len(selected[split][task]))]
        slots.sort(
            key=lambda slot: digest(
                ["arrival-class-slot", config["arrival_seed"], split, slot[0], slot[1]]
            )
        )
        consumed = Counter()
        requests = []
        for (task, _), arrival in zip(slots, arrival_windows[split]):
            candidate = selected[split][task][consumed[task]]
            consumed[task] += 1
            ref = candidate["source_ref"]
            request_id = "sr-" + digest(
                [
                    config["workload_family"],
                    task,
                    ref["dataset_id"],
                    ref["revision"],
                    ref["id"],
                    text_hash(candidate["prompt_text"]),
                    tokenizer.fingerprint,
                ]
            )
            requests.append(
                ServingWorkloadRequest(
                    schema_version=REQUEST_SCHEMA,
                    workload_id=config["splits"][split]["workload_id"],
                    split=split,
                    request_id=request_id,
                    task_class=task,
                    language=config["language"],
                    **candidate,
                    prompt_length=len(candidate["prompt_token_ids"]),
                    prompt_sha256=text_hash(candidate["prompt_text"]),
                    tokenizer_fingerprint=tokenizer.fingerprint,
                    arrival_time_ms=arrival["source_timestamp_ms"]
                    - arrival["selected_first_timestamp_ms"],
                    arrival_source_ref=arrival,
                    maximum_new_tokens=config["class_limits"][task]["maximum_new_tokens"],
                    sampling_seed=int(digest(["sampling", config["seed"], request_id])[:8], 16),
                    slo_class=task,
                )
            )
        output[split] = requests
    return output, selection, arrival_windows


def output_names(config: dict) -> dict:
    return {
        s: f"{s}{sum(config['splits'][s]['quotas'].values())}.jsonl"
        for s in ("main", "calibration")
    }


def manifest(config, lock, tokenizer, arrivals, hashes) -> dict:
    core = {
        "schema_version": MANIFEST_SCHEMA,
        "data_kind": lock["data_kind"],
        "description": "mixed-task real-text workload with Mooncake arrival replay"
        if lock["data_kind"] == "real-public"
        else "synthetic fixture contract; not real data",
        "workload_family": config["workload_family"],
        "config_sha256": digest(config),
        "source_lock_sha256": digest(lock),
        "sources": {
            k: {"revision": v["revision"], "files": {f["path"]: f["sha256"] for f in v["files"]}}
            for k, v in lock["sources"].items()
        },
        "algorithm": ALGORITHM,
        "seed": config["seed"],
        "arrival_seed": config["arrival_seed"],
        "splits": config["splits"],
        "tokenizer": tokenizer.metadata,
        "prompt_contract": {
            k: config[k]
            for k in (
                "instruction_version",
                "instructions",
                "language",
                "request_style",
                "enable_thinking",
                "add_generation_prompt",
                "class_limits",
                "max_model_len",
                "speculative_reserve_tokens",
                "natural_eos",
            )
        },
        "arrival": {
            "unit": "ms",
            "base_time_scale": 1.0,
            "runtime_consumption": "not implemented",
            "windows": {
                s: {
                    "sorted_start_index": rows[0]["sorted_index"],
                    "sorted_end_index_exclusive": rows[-1]["sorted_index"] + 1,
                    "original_line_indices": [r["original_line_index"] for r in rows],
                    "selected_first_timestamp_ms": rows[0]["source_timestamp_ms"],
                }
                for s, rows in arrivals.items()
            },
        },
        "slo_policy_ref": None,
        "calibration_status": "pending",
        "files": hashes,
    }
    return {**core, "semantic_workload_sha256": digest(core)}


def summary(requests: dict, selection: dict) -> dict:
    return {
        "by_split": {
            split: {
                "request_count": len(rows),
                "arrival": arrival_summary(rows),
                "by_class": {
                    task: {
                        "count": sum(r.task_class == task for r in rows),
                        "prompt_length_tokens": distribution(
                            [r.prompt_length for r in rows if r.task_class == task]
                        ),
                        "maximum_new_tokens": sorted(
                            {r.maximum_new_tokens for r in rows if r.task_class == task}
                        ),
                        "generation_length_scope": (
                            "upper bound including first token; not measured output"
                        ),
                    }
                    for task in TASKS
                },
            }
            for split, rows in requests.items()
        },
        "filtering": {
            task: {
                **stats,
                "input_filter_fraction": sum(
                    v
                    for k, v in stats["filters"].items()
                    if k not in ("prompt_length_limit", "context_budget")
                )
                / stats["source_rows"]
                if stats["source_rows"]
                else None,
                "examined_length_filter_fraction": sum(
                    stats["filters"].get(k, 0) for k in ("prompt_length_limit", "context_budget")
                )
                / stats["length_checked"]
                if stats["length_checked"]
                else None,
            }
            for task, stats in selection["by_class"].items()
        },
        "actual_output_distribution": None,
        "output_cap_truncation_rate": None,
        "inference_performed": False,
    }


def sample_review(requests: dict) -> str:
    lines = [
        "# S0 prompt review",
        "",
        "First five main-set prompts per class in arrival order.",
        "No model outputs or reference answers are included.",
        "",
    ]
    for task in TASKS:
        for row in [r for r in requests["main"] if r.task_class == task][:5]:
            fence = "`" * max(
                4,
                max((len(s) for s in __import__("re").findall(r"`+", row.prompt_text)), default=0)
                + 1,
            )
            lines.extend(
                [
                    f"## {task}: {row.request_id}",
                    "",
                    "Source: " + json.dumps(row.source_ref, ensure_ascii=False),
                    "",
                    f"Prompt tokens: {row.prompt_length}; maximum new tokens: "
                    f"{row.maximum_new_tokens}",
                    "",
                    fence + "text",
                    row.prompt_text,
                    fence,
                    "",
                ]
            )
    return "\n".join(lines)


def build_info(sources: Path, tokenizer_path: Path) -> dict:
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", "--no-optional-locks", "-c", "core.fsmonitor=false", *args],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=30,
            ).strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            return None

    versions = {}
    for name in ("transformers", "tokenizers", "huggingface-hub", "pyarrow", "jinja2"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    status = git("status", "--porcelain")
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "versions": versions,
        "execution_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "sources_directory": str(sources.resolve()),
        "tokenizer_directory": str(tokenizer_path),
        "server_local_tokenizer_verification": "PENDING",
        "model_inference": False,
    }


def build(
    config_path: Path,
    lock_path: Path,
    sources: Path,
    tokenizer,
    output: Path,
    *,
    tokenizer_path: Path,
    allow_synthetic=False,
) -> dict:
    from specrhythm.serving.validation import validate

    require(not output.exists(), "build output directory must be fresh", directory=str(output))
    output.mkdir(parents=True)
    started = time.monotonic()
    info = build_info(sources, tokenizer_path)
    try:
        with (
            (output / "stdout.log").open("x") as stdout,
            (output / "stderr.log").open("x") as stderr,
        ):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                config, lock = load_config(config_path), read_json(lock_path)
                verify_sources(lock, sources, allow_synthetic=allow_synthetic)
                verify_locked_tokenizer(tokenizer, lock, sources)
                requests, selection, arrivals = plan(config, lock, sources, tokenizer)
                write_json(output / "source-lock.json", lock)
                write_json(output / "workload-config.json", config)
                names = output_names(config)
                for split, rows in requests.items():
                    with (output / names[split]).open("xb") as handle:
                        for row in rows:
                            handle.write(canonical(row.to_dict()))
                write_json(output / "selection-report.json", selection)
                write_json(output / "dataset-summary.json", summary(requests, selection))
                (output / "sample-review.md").write_text(sample_review(requests), encoding="utf-8")
                write_json(
                    output / "workload-manifest.json",
                    manifest(
                        config,
                        lock,
                        tokenizer,
                        arrivals,
                        file_inventory(output, list(names.values())),
                    ),
                )
                result = validate(
                    output,
                    sources,
                    tokenizer,
                    allow_synthetic=allow_synthetic,
                    expected_lock=lock_path,
                    expected_config=config_path,
                )
                write_json(output / "validation.json", result)
                print(
                    json.dumps(
                        {
                            "valid": result["valid"],
                            "errors": result["errors"],
                            "request_counts": {s: len(r) for s, r in requests.items()},
                        }
                    )
                )
                require(
                    result["valid"],
                    "constructed workload failed independent validation",
                    errors=result["errors"],
                )
        return result
    except (ValueError, OSError, TypeError, KeyError, RuntimeError) as error:
        failure = {
            "valid": False,
            "errors": [str(error)],
            "details": getattr(error, "details", {}),
        }
        write_json(output / "failure-report.json", failure)
        if not (output / "validation.json").exists():
            write_json(output / "validation.json", failure)
        raise
    finally:
        info["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "build-info.json", info)
