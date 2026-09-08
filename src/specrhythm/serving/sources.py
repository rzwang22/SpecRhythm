"""Official fixed-revision files, portable source locks and read-only imports."""

from __future__ import annotations

import fnmatch
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from specrhythm.serving.common import (
    LOCK_SCHEMA,
    TASKS,
    is_hash,
    read_json,
    require,
    sha256_file,
    source_path,
    write_json,
)

DATASETS = {
    "chat": ("OpenAssistant/oasst1", "default", "data/train-*.parquet", 1),
    "code": ("codeparrot/apps", "all", "train.jsonl", 1),
    "summarization": ("abisee/cnn_dailymail", "3.0.0", "3.0.0/train-*.parquet", 3),
    "reasoning": ("openai/gsm8k", "main", "main/train-*.parquet", 1),
}
MOONCAKE = "kvcache-ai/Mooncake"
TRACE_FILE = "FAST25-release/traces/conversation_trace.jsonl"
TOKENIZER = "Qwen/Qwen3-0.6B"
TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "README.md",
    "LICENSE",
}


def _api(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "SpecRhythm-S0-source-resolver"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _hub_file(
    repo: str, revision: str, filename: str, cache: Path, *, model=False, offline=False
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError("install SpecRhythm[workload] for Hugging Face acquisition") from error
    return Path(
        hf_hub_download(
            repo_id=repo,
            repo_type="model" if model else "dataset",
            revision=revision,
            filename=filename,
            cache_dir=str(cache / "hub"),
            local_files_only=offline,
            token=False,
        )
    )


def _http_file(url: str, cache: Path, *, offline=False) -> Path:
    from specrhythm.serving.common import text_hash

    destination = cache / "specrhythm-s0" / text_hash(url)
    if destination.is_file():
        return destination
    require(not offline, "locked source is absent from offline cache", url=url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Download to a temporary file; an interrupted transfer is never a cache hit.
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                shutil.copyfileobj(response, handle)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, destination)
    return destination


def _link(source: Path, destination: Path, expected: str) -> None:
    require(sha256_file(source) == expected, "source checksum mismatch", file=str(source))
    if destination.exists() or destination.is_symlink():
        require(
            destination.is_file() and sha256_file(destination) == expected,
            "existing source import changed; refusing to overwrite",
            file=str(destination),
        )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve())


def resolve(output: Path, sources: Path, cache: Path, revisions=None) -> dict:
    """Resolve once, download exact original files, hash bytes, then freeze the lock."""
    require(not output.exists(), "source lock output must be fresh")
    revisions = revisions or {}
    lock = {"schema_version": LOCK_SCHEMA, "data_kind": "real-public", "sources": {}}
    for task, (repo, config, pattern, expected) in DATASETS.items():
        info = _api(
            f"https://huggingface.co/api/datasets/{repo}/revision/" + revisions.get(task, "main")
        )
        revision = info["sha"]
        require(is_hash(revision, 40), "dataset did not resolve to an immutable commit", repo=repo)
        names = [r["rfilename"] for r in info["siblings"]]
        data = sorted(n for n in names if fnmatch.fnmatchcase(n, pattern))
        require(
            len(data) == expected,
            "official source file layout changed",
            dataset=repo,
            pattern=pattern,
            actual=data,
        )
        metadata = sorted(n for n in names if n in {"README.md", "LICENSE", "apps.py"})
        files = []
        for filename in [*data, *metadata]:
            path = _hub_file(repo, revision, filename, cache)
            entry = {
                "path": filename,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "role": "data" if filename in data else "attribution",
                "url": f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}",
            }
            files.append(entry)
            _link(path, source_path(sources, task, filename), entry["sha256"])
        lock["sources"][task] = {
            "dataset_id": repo,
            "revision": revision,
            "storage_revision": revision,
            "config": config,
            "split": "train",
            "files": files,
            "url": f"https://huggingface.co/datasets/{repo}/tree/{revision}",
            "license_declared": info.get("cardData", {}).get("license", "not-declared"),
            "attribution": {
                "dataset_card_url": f"https://huggingface.co/datasets/{repo}/blob/"
                f"{revision}/README.md",
                "citation": "see retained dataset card",
            },
            "file_mapping": "official repository files at the same source commit; no conversion",
        }
    revision = _api(
        f"https://api.github.com/repos/{MOONCAKE}/commits/" + revisions.get("arrival", "main")
    )["sha"]
    require(is_hash(revision, 40), "trace did not resolve to an immutable commit")
    files = []
    for filename in (TRACE_FILE, "FAST25-release/README.md", "LICENSE-APACHE"):
        url = f"https://raw.githubusercontent.com/{MOONCAKE}/{revision}/{filename}"
        path = _http_file(url, cache)
        entry = {
            "path": filename,
            "url": url,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "role": "data" if filename == TRACE_FILE else "attribution",
        }
        files.append(entry)
        _link(path, source_path(sources, "arrival", filename), entry["sha256"])
    lock["sources"]["arrival"] = {
        "repository_id": MOONCAKE,
        "revision": revision,
        "files": files,
        "url": f"https://github.com/{MOONCAKE}/tree/{revision}/FAST25-release/traces",
        "license_declared": "Apache-2.0; see retained repository LICENSE-APACHE",
        "attribution": "Mooncake FAST25 release; timestamp-only replay, not original payloads",
    }
    info = _api(
        f"https://huggingface.co/api/models/{TOKENIZER}/revision/"
        + revisions.get("tokenizer", "main")
    )
    revision = info["sha"]
    files = []
    for filename in sorted({r["rfilename"] for r in info["siblings"]} & TOKENIZER_FILES):
        path = _hub_file(TOKENIZER, revision, filename, cache, model=True)
        entry = {
            "path": filename,
            "url": f"https://huggingface.co/{TOKENIZER}/resolve/{revision}/{filename}",
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "role": "tokenizer",
        }
        files.append(entry)
        _link(path, source_path(sources, "tokenizer", filename), entry["sha256"])
    lock["sources"]["tokenizer"] = {
        "repository_id": TOKENIZER,
        "revision": revision,
        "files": files,
        "license_declared": info.get("cardData", {}).get("license", "not-declared"),
    }
    validate_lock(lock)
    write_json(output, lock)
    return lock


def validate_lock(lock: dict, *, allow_synthetic=False) -> None:
    require(lock.get("schema_version") == LOCK_SCHEMA, "unsupported source lock")
    require(
        lock.get("data_kind") == "real-public"
        or (allow_synthetic and lock.get("data_kind") == "synthetic-fixture"),
        "synthetic source locks require explicit fixture mode",
    )
    require(set(lock["sources"]) == {*TASKS, "arrival", "tokenizer"}, "incomplete source lock")
    for group, source in lock["sources"].items():
        require(
            is_hash(source.get("revision"), 40),
            "source revision must be an exact commit",
            source=group,
        )
        require(
            source.get("files") and source.get("license_declared"),
            "missing source files or license declaration",
            source=group,
        )
        names = [r["path"] for r in source["files"]]
        require(len(names) == len(set(names)), "duplicate source file", source=group)
        for row in source["files"]:
            source_path(Path("."), group, row["path"])
            require(is_hash(row.get("sha256")), "source file SHA256 missing", source=group)
        if group in DATASETS:
            repo, config, pattern, expected = DATASETS[group]
            require(
                source.get("dataset_id") == repo
                and source.get("config") == config
                and source.get("split") == "train",
                "dataset/config/split substitution",
            )
            if lock["data_kind"] == "real-public":
                data = [r["path"] for r in source["files"] if r["role"] == "data"]
                require(
                    len(data) == expected and all(fnmatch.fnmatchcase(n, pattern) for n in data),
                    "real source lock does not cover the audited full train files",
                )
        if group == "tokenizer" and lock["data_kind"] == "real-public":
            require(
                source.get("repository_id") == TOKENIZER
                and {"tokenizer.json", "tokenizer_config.json"} <= set(names)
                and set(names) <= TOKENIZER_FILES,
                "invalid tokenizer-only source files",
            )
        if group == "arrival" and lock["data_kind"] == "real-public":
            require(
                source.get("repository_id") == MOONCAKE
                and [r["path"] for r in source["files"] if r["role"] == "data"] == [TRACE_FILE],
                "S0 requires the official Mooncake conversation trace",
            )


def verify_sources(lock: dict, root: Path, *, allow_synthetic=False) -> dict:
    validate_lock(lock, allow_synthetic=allow_synthetic)
    hashes = {}
    for group, source in lock["sources"].items():
        for row in source["files"]:
            path = source_path(root, group, row["path"])
            require(path.is_file(), "required source file missing", file=str(path))
            actual = sha256_file(path)
            require(actual == row["sha256"], "source checksum mismatch", file=str(path))
            hashes[f"{group}/{row['path']}"] = actual
    return hashes


def fetch(lock_path: Path, sources: Path, cache: Path, *, import_map=None, offline=False) -> dict:
    lock = read_json(lock_path)
    validate_lock(lock)
    imported = read_json(import_map) if import_map else {}
    for group, source in lock["sources"].items():
        for row in source["files"]:
            key = f"{group}/{row['path']}"
            destination = source_path(sources, group, row["path"])
            if destination.is_file():
                require(
                    sha256_file(destination) == row["sha256"],
                    "existing source checksum mismatch",
                    file=str(destination),
                )
                continue
            if key in imported:
                path = Path(imported[key])
            elif group == "arrival":
                path = _http_file(row["url"], cache, offline=offline)
            else:
                path = _hub_file(
                    source.get("dataset_id", source.get("repository_id")),
                    source["revision"],
                    row["path"],
                    cache,
                    model=group == "tokenizer",
                    offline=offline,
                )
            _link(path, destination, row["sha256"])
    return verify_sources(lock, sources)


def records(path: Path, columns: list[str]):
    """Column projection for official Parquet; no dataset loading scripts execute."""
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise RuntimeError("install SpecRhythm[workload] to read Parquet") from error
        reader = parquet.ParquetFile(path)
        require(
            set(columns) <= set(reader.schema.names),
            "required dataset fields missing",
            file=str(path),
            required=columns,
            actual=reader.schema.names,
        )
        index = 0
        for batch in reader.iter_batches(batch_size=512, columns=columns):
            for row in batch.to_pylist():
                yield index, row
                index += 1
    else:
        require(path.suffix == ".jsonl", "supported source formats are JSONL and Parquet")
        with path.open(encoding="utf-8") as handle:
            for index, raw in enumerate(handle):
                require(raw.strip(), "blank source JSONL row", file=str(path), row=index)
                row = json.loads(raw)
                require(
                    isinstance(row, dict) and set(columns) <= set(row),
                    "required dataset fields missing",
                    file=str(path),
                    row=index,
                    required=columns,
                )
                yield index, {k: row[k] for k in columns}
