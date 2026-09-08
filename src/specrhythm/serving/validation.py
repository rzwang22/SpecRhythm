"""Read-only source-to-token/arrival validation; builder validity flags are not trusted."""

from __future__ import annotations

import tarfile
from collections import Counter
from pathlib import Path

from specrhythm.serving.arrival import read_arrivals, windows
from specrhythm.serving.common import (
    TASKS,
    canonical,
    digest,
    file_inventory,
    normalized_prompt,
    read_json,
    require,
    sha256_file,
    text_hash,
    write_json,
)
from specrhythm.serving.schema import load_config, load_requests
from specrhythm.serving.selection import COLUMNS, extract
from specrhythm.serving.sources import records, source_path, verify_sources
from specrhythm.serving.tokenization import verify_locked_tokenizer


def validate(
    directory: Path,
    sources: Path,
    tokenizer,
    *,
    allow_synthetic=False,
    expected_lock: Path = None,
    expected_config: Path = None,
) -> dict:
    from specrhythm.serving.builder import manifest, output_names, plan, sample_review, summary

    result = {
        "schema_version": "specrhythm.serving-validation.v1",
        "valid": False,
        "errors": [],
        "code_fixture_validation": "PENDING",
        "real_data_validation": "PENDING",
        "server_local_tokenizer_verification": "PENDING",
        "inputs_unchanged": False,
        "model_inference_performed": False,
    }
    before, source_before = {}, {}
    lock = None
    try:
        config = load_config(directory / "workload-config.json")
        lock = read_json(directory / "source-lock.json")
        if expected_lock:
            require(
                lock == read_json(expected_lock), "build source lock differs from trusted lock"
            )
        if expected_config:
            require(
                config == load_config(expected_config), "build config differs from expected config"
            )
        source_before = verify_sources(lock, sources, allow_synthetic=allow_synthetic)
        verify_locked_tokenizer(tokenizer, lock, sources)
        require(
            tokenizer.is_real or lock["data_kind"] == "synthetic-fixture",
            "real-data validation requires a real tokenizer",
        )
        names = output_names(config)
        artifact_names = [
            *names.values(),
            "source-lock.json",
            "workload-config.json",
            "workload-manifest.json",
            "selection-report.json",
            "dataset-summary.json",
            "sample-review.md",
        ]
        before = file_inventory(directory, artifact_names)
        arrivals = windows(read_arrivals(lock, sources), config)
        requests = {s: load_requests(directory / filename) for s, filename in names.items()}
        seen_ids, seen_sources, seen_groups, seen_prompts, seen_lines = (
            set(),
            set(),
            set(),
            set(),
            set(),
        )
        for split, rows in requests.items():
            require(
                len(rows) == sum(config["splits"][split]["quotas"].values()),
                "full-file request count differs",
                split=split,
                actual=len(rows),
            )
            require(
                dict(Counter(r.task_class for r in rows)) == config["splits"][split]["quotas"],
                "class quotas differ",
                split=split,
            )
            previous = -1
            for row, arrival in zip(rows, arrivals[split]):
                ref = row.source_ref
                require(
                    row.split == split
                    and row.workload_id == config["splits"][split]["workload_id"],
                    "request workload/split differs",
                )
                require(row.request_id not in seen_ids, "request identity overlaps between splits")
                seen_ids.add(row.request_id)
                sid = (ref["dataset_id"], ref["id"])
                require(sid not in seen_sources, "source sample duplicated across joint workload")
                seen_sources.add(sid)
                if ref["group_id"]:
                    group = (ref["dataset_id"], ref["group_id"])
                    require(
                        group not in seen_groups, "source group overlaps between requests/splits"
                    )
                    seen_groups.add(group)
                normalized = text_hash(normalized_prompt(row.prompt_text))
                require(normalized not in seen_prompts, "normalized prompt duplicated")
                seen_prompts.add(normalized)
                require(
                    row.tokenizer_fingerprint == tokenizer.fingerprint,
                    "tokenizer fingerprint differs",
                )
                require(
                    tokenizer.render(row.messages) == row.prompt_text, "prompt/template mismatch"
                )
                require(
                    tokenizer.encode(row.prompt_text) == row.prompt_token_ids,
                    "prompt token IDs differ",
                )
                limits = config["class_limits"][row.task_class]
                require(
                    row.maximum_new_tokens == limits["maximum_new_tokens"]
                    and row.prompt_length <= limits["prompt_tokens"]
                    and row.prompt_length
                    + row.maximum_new_tokens
                    + config["speculative_reserve_tokens"]
                    <= config["max_model_len"],
                    "prompt/generation context budget violated",
                )
                require(
                    row.arrival_source_ref == arrival,
                    "original Mooncake row/window mapping differs",
                )
                require(
                    row.arrival_time_ms
                    == arrival["source_timestamp_ms"] - arrival["selected_first_timestamp_ms"]
                    and row.arrival_time_ms >= previous,
                    "arrival timestamp/order differs from original trace",
                )
                previous = row.arrival_time_ms
                require(
                    arrival["original_line_index"] not in seen_lines, "arrival windows overlap"
                )
                seen_lines.add(arrival["original_line_index"])
                source = lock["sources"][row.task_class]
                require(
                    all(
                        ref[k] == source[k] for k in ("dataset_id", "revision", "config", "split")
                    ),
                    "request source provenance differs from lock",
                )
                expected_id = "sr-" + digest(
                    [
                        config["workload_family"],
                        row.task_class,
                        ref["dataset_id"],
                        ref["revision"],
                        ref["id"],
                        text_hash(row.prompt_text),
                        tokenizer.fingerprint,
                    ]
                )
                require(
                    row.request_id == expected_id
                    and row.sampling_seed
                    == int(digest(["sampling", config["seed"], expected_id])[:8], 16),
                    "request identity/seed differs",
                )
        # Independently re-read selected source records and re-extract whitelisted input fields.
        by_location = {}
        for rows in requests.values():
            for row in rows:
                ref = row.source_ref
                key = (row.task_class, ref["file"], ref["original_row_index"])
                require(key not in by_location, "source row selected twice")
                by_location[key] = row
        found = set()
        for task in TASKS:
            for file in lock["sources"][task]["files"]:
                if file["role"] != "data":
                    continue
                for index, source_row in records(
                    source_path(sources, task, file["path"]), COLUMNS[task]
                ):
                    key = (task, file["path"], index)
                    if key not in by_location:
                        continue
                    row = by_location[key]
                    prompt, reason = extract(task, source_row, config)
                    require(
                        reason is None
                        and prompt["id"] == row.source_ref["id"]
                        and prompt["group_id"] == row.source_ref["group_id"]
                        and row.source_ref["file_sha256"] == file["sha256"]
                        and row.messages == [{"role": "user", "content": prompt["content"]}],
                        "prompt does not match the original allowed source fields",
                    )
                    found.add(key)
        require(found == set(by_location), "selected source row is missing")
        # Recompute deterministic selection from the entire locked candidate pool. This also
        # verifies all filter/dedup accounting, not just rows the builder chose to expose.
        expected, selection, expected_arrivals = plan(config, lock, sources, tokenizer)
        for split, rows in expected.items():
            wanted = b"".join(canonical(row.to_dict()) for row in rows)
            require(
                (directory / names[split]).read_bytes() == wanted,
                "workload differs from deterministic source selection",
                split=split,
            )
        actual_manifest = read_json(directory / "workload-manifest.json")
        require(
            actual_manifest
            == manifest(
                config,
                lock,
                tokenizer,
                expected_arrivals,
                {name: before[name] for name in names.values()},
            ),
            "core manifest/source/output hash mismatch",
        )
        require(
            read_json(directory / "selection-report.json") == selection,
            "selection/filter report differs from original sources",
        )
        require(
            read_json(directory / "dataset-summary.json") == summary(expected, selection),
            "dataset summary differs from verified requests",
        )
        require(
            (directory / "sample-review.md").read_text(encoding="utf-8")
            == sample_review(expected),
            "sample review differs from original selected prompts",
        )
        result.update(
            valid=True,
            validated_artifact_sha256=before,
            semantic_workload_sha256=actual_manifest["semantic_workload_sha256"],
            request_counts={s: len(rows) for s, rows in requests.items()},
        )
        result[
            "real_data_validation"
            if lock["data_kind"] == "real-public"
            else "code_fixture_validation"
        ] = "PASS"
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        result.update(valid=False, errors=[str(error)], details=getattr(error, "details", {}))
    finally:
        try:
            if before:
                require(
                    file_inventory(directory, list(before)) == before,
                    "validator input artifacts changed during validation",
                )
            if source_before:
                require(
                    verify_sources(lock, sources, allow_synthetic=allow_synthetic)
                    == source_before,
                    "source files changed during validation",
                )
            result["inputs_unchanged"] = bool(before and source_before)
        except (OSError, ValueError, TypeError, KeyError) as error:
            result.update(valid=False, inputs_unchanged=False)
            result["errors"].append(str(error))
    return result


def rebuild_check(left: Path, right: Path) -> dict:
    result = {"valid": False, "errors": []}
    try:
        for root in (left, right):
            require(
                read_json(root / "validation.json").get("valid") is True,
                "rebuild comparison requires completed data validation",
            )
        a, b = (
            read_json(left / "workload-manifest.json"),
            read_json(right / "workload-manifest.json"),
        )
        require(a == b, "core manifests differ across rebuild directories")
        for name, expected in a["files"].items():
            require(
                sha256_file(left / name) == sha256_file(right / name) == expected,
                "JSONL differs across rebuilds",
                file=name,
            )
        result.update(
            valid=True,
            semantic_workload_sha256=a["semantic_workload_sha256"],
            files=list(a["files"]),
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["errors"] = [str(error)]
    return result


def seal(directory: Path, *, bundle: Path = None) -> dict:
    require(not (directory / "checksums.sha256").exists(), "checksum inventory must be fresh")
    if bundle:
        require(not bundle.exists(), "review bundle must be fresh")
        require(
            directory.resolve() not in bundle.resolve().parents,
            "bundle must be outside build directory",
        )
    names = sorted(p.name for p in directory.iterdir() if p.is_file())
    require(
        "validation.json" in names and read_json(directory / "validation.json").get("valid"),
        "cannot seal an invalid build",
    )
    validated = read_json(directory / "validation.json")["validated_artifact_sha256"]
    require(
        file_inventory(directory, list(validated)) == validated,
        "artifact changed after data validation; refusing to seal",
    )
    hashes = file_inventory(directory, names)
    (directory / "checksums.sha256").write_text(
        "".join(f"{value}  {name}\n" for name, value in hashes.items()), encoding="utf-8"
    )
    if bundle:
        with tarfile.open(bundle, "x:gz") as archive:
            for name in [*names, "checksums.sha256"]:
                archive.add(directory / name, arcname=directory.name + "/" + name, recursive=False)
    return {"valid": True, "artifacts": hashes, "self_excluded": "checksums.sha256"}


def write_validation(output: Path, report: dict, directory: Path) -> None:
    require(not output.exists(), "validation report output must be fresh")
    # Reports may be additional immutable files, never replace validated input artifacts.
    protected = {
        directory / name for name in read_json(directory / "workload-manifest.json")["files"]
    }
    require(output.resolve() not in {p.resolve() for p in protected}, "cannot overwrite workload")
    write_json(output, report)
