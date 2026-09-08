"""CPU-only Phase S0 source, workload and acceptance commands."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from specrhythm.serving.common import read_json, require, sha256_file, write_json

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO / "configs/workloads/mixed-real-v1.json"
DEFAULT_LOCK = REPO / "configs/workloads/mixed-real-v1-source-lock.json"
DEFAULT_DRAFT = Path("/root/autodl-tmp/models/Qwen3-0.6B")
DEFAULT_TARGET = Path("/root/autodl-tmp/models/Qwen3-32B")
DEFAULT_CACHE = Path("/root/.cache/huggingface")
DEFAULT_RESULTS = Path("/root/autodl-tmp/SpecRhythm-data/results/phase-s0")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    resolve = commands.add_parser(
        "source-resolve", help="resolve once and freeze original file hashes"
    )
    resolve.add_argument("--output", type=Path, required=True)
    resolve.add_argument("--revisions", type=Path)
    fetch = commands.add_parser("source-fetch", help="fetch/import only the locked source files")
    fetch.add_argument("--import-map", type=Path)
    fetch.add_argument("--offline", action="store_true")
    for command in (resolve, fetch):
        command.add_argument("--sources-dir", type=Path, required=True)
        command.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    fetch.add_argument("--source-lock", type=Path, default=DEFAULT_LOCK)
    build = commands.add_parser("build", help="construct exact quotas; no inference")
    build.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser(
        "validate", help="read-only full source/token/arrival validation"
    )
    validate.add_argument("--run-root", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    for command in (build, validate):
        command.add_argument("--sources-dir", type=Path, required=True)
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        command.add_argument("--source-lock", type=Path, default=DEFAULT_LOCK)
        command.add_argument("--tokenizer-path", type=Path, default=DEFAULT_DRAFT)
    check = commands.add_parser(
        "tokenizer-check", help="compare real Draft/Target tokenizers only"
    )
    check.add_argument("--draft-tokenizer", type=Path, default=DEFAULT_DRAFT)
    check.add_argument("--target-tokenizer", type=Path, default=DEFAULT_TARGET)
    check.add_argument("--run-root", type=Path)
    check.add_argument("--output", type=Path, required=True)
    inspect = commands.add_parser("inspect", help="inspect data; this does not certify validation")
    inspect.add_argument("--run-root", type=Path, required=True)
    rebuild = commands.add_parser("rebuild-check", help="compare independently validated builds")
    rebuild.add_argument("--left", type=Path, required=True)
    rebuild.add_argument("--right", type=Path, required=True)
    rebuild.add_argument("--output", type=Path, required=True)
    seal = commands.add_parser("seal", help="freeze checksums and a small review bundle")
    seal.add_argument("--run-root", type=Path, required=True)
    seal.add_argument("--bundle", type=Path)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "source-resolve":
            from specrhythm.serving.sources import resolve

            lock = resolve(
                args.output,
                args.sources_dir,
                args.cache_dir,
                read_json(args.revisions) if args.revisions else None,
            )
            report = {
                "valid": True,
                "data_kind": lock["data_kind"],
                "source_lock_sha256": sha256_file(args.output),
            }
        elif args.command == "source-fetch":
            from specrhythm.serving.sources import fetch

            hashes = fetch(
                args.source_lock,
                args.sources_dir,
                args.cache_dir,
                import_map=args.import_map,
                offline=args.offline,
            )
            report = {"valid": True, "source_files_verified": len(hashes)}
        elif args.command in ("build", "validate"):
            from specrhythm.serving.builder import build
            from specrhythm.serving.tokenization import QwenTokenizer
            from specrhythm.serving.validation import validate, write_validation

            require(not args.output.exists(), "output must be fresh")
            tokenizer = QwenTokenizer(args.tokenizer_path)
            if args.command == "build":
                report = build(
                    args.config,
                    args.source_lock,
                    args.sources_dir,
                    tokenizer,
                    args.output,
                    tokenizer_path=args.tokenizer_path,
                )
            else:
                report = validate(
                    args.run_root,
                    args.sources_dir,
                    tokenizer,
                    expected_lock=args.source_lock,
                    expected_config=args.config,
                )
                write_validation(args.output, report, args.run_root)
        elif args.command == "tokenizer-check":
            from specrhythm.serving.schema import load_requests
            from specrhythm.serving.tokenization import QwenTokenizer, check_tokenizers

            require(not args.output.exists(), "tokenizer report output must be fresh")
            left, right = QwenTokenizer(args.draft_tokenizer), QwenTokenizer(args.target_tokenizer)
            requests = []
            binding = {}
            if args.run_root:
                manifest = read_json(args.run_root / "workload-manifest.json")
                for name, expected in manifest["files"].items():
                    require(
                        sha256_file(args.run_root / name) == expected, "workload checksum differs"
                    )
                    requests.extend(load_requests(args.run_root / name))
                binding = {
                    "workload_manifest_sha256": sha256_file(
                        args.run_root / "workload-manifest.json"
                    ),
                    "workload_files": manifest["files"],
                }
            report = check_tokenizers(left, right, requests)
            if not args.run_root:
                report.update(status="READY", server_prompt_encoding_validation="PENDING")
            report.update(valid=True, **binding)
            write_json(args.output, report)
        elif args.command == "inspect":
            from specrhythm.serving.schema import load_requests

            manifest = read_json(args.run_root / "workload-manifest.json")
            report = {
                "inspection_only": True,
                "semantic_workload_sha256": manifest["semantic_workload_sha256"],
                "files": {},
            }
            for name in manifest["files"]:
                rows = load_requests(args.run_root / name)
                report["files"][name] = {
                    "requests": len(rows),
                    "quotas": dict(Counter(r.task_class for r in rows)),
                    "sha256": sha256_file(args.run_root / name),
                }
        elif args.command == "rebuild-check":
            from specrhythm.serving.validation import rebuild_check

            require(not args.output.exists(), "rebuild report output must be fresh")
            report = rebuild_check(args.left, args.right)
            write_json(args.output, report)
        else:
            from specrhythm.serving.validation import seal

            report = seal(args.run_root, bundle=args.bundle)
    except (ValueError, OSError, TypeError, KeyError, RuntimeError) as error:
        message = (
            f"missing required data/config field: {error.args[0]}"
            if isinstance(error, KeyError)
            else str(error)
        )
        report = {"valid": False, "errors": [message], "details": getattr(error, "details", {})}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("valid", True) else 1


if __name__ == "__main__":
    sys.exit(main())
