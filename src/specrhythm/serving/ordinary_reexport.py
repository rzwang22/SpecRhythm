"""Repackage retained ordinary CPU evidence; never start inference or edit sources."""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from specrhythm.phase4.manifest import atomic_write_json, sha256_file
from specrhythm.serving.common import read_json, require
from specrhythm.serving.delivery_budget import count_budget, export_summary
from specrhythm.serving.k3_local_run import deliver, prepare_local, validate_archive
from specrhythm.serving.ping_prepost_delivery import export


def small_record(path):
    require(path.is_file() and path.stat().st_size < 128 * 1024,
            "missing/oversized reexport source receipt", path=str(path))
    value = read_json(path)
    require(isinstance(value, dict), "reexport receipt must be an object", path=str(path))
    return dict(path=str(path), sha256=sha256_file(path), value=value)


def run(source, *, expected_execution, exporter_commit, local_base, persistent, tag,
        minimum_free_bytes=8 * 1024**3):
    source = source.resolve(strict=True)
    require(source.is_dir(), "raw retained directory required, not the incomplete tar archive")
    for value in (expected_execution, exporter_commit):
        require(re.fullmatch(r"[0-9a-f]{40}", value) is not None, "full commit SHA required")
    budget = count_budget(source)
    require(budget["profile"] == "ordinary-cpu-13-runs-v1", "ordinary CPU plan required")
    plan = small_record(source / "dual-batch-plan.json")
    require(plan["value"]["execution_commit"] == expected_execution,
            "source execution SHA differs; never relabel old results")
    outcome = small_record(source / "runner-outcome.json")
    first, stage = outcome["value"]["first_exit_code"], outcome["value"]["stage"]
    require(type(first) is int and 0 <= first <= 255 and isinstance(stage, str),
            "invalid original runner exit/stage")
    planned_root = (Path(local_base) / tag).resolve()
    require(planned_root != source and planned_root not in source.parents
            and source not in planned_root.parents,
            "reexport output must be separate from retained source")
    require(Path(persistent).resolve() != source
            and source not in Path(persistent).resolve().parents,
            "persistent copy must not write inside retained source")
    root, persistent, storage = prepare_local(local_base, persistent, tag,
                                              minimum_free_bytes=minimum_free_bytes)
    require(root != source and root not in source.parents and source not in root.parents,
            "reexport output must be separate from retained source")
    previous = source.parent / "delivery-status.json"
    provenance = dict(schema_version="specrhythm.ordinary-reexport.v1",
        source_directory=str(source), source_execution_commit=expected_execution,
        exporter_commit=exporter_commit, source_plan=plan, source_runner_outcome=outcome,
        previous_delivery=small_record(previous) if previous.exists() else None,
        previous_delivery_status="PRESENT" if previous.exists() else "NOT_AVAILABLE",
        storage=storage, source_read_only=True,
        scope="archive repair only; no inference, no new measurement or GPU qualification")
    state = dict(reexport_provenance=provenance, original_execution_exit_code=first,
                 original_stage=stage, export_exit_code=0, delivery_exit_code=0,
                 receipt_errors=[], local_root=str(root))
    archive = root / ("pingpong-k3-delivery-" + tag + ".tar.gz")
    upload = None
    try:
        result = export(source, archive, first_code=first, stage=stage, modes=(),
                        source_read_only=True, provenance=provenance)
        state["export_exit_code"] = result["export_validation_exit_code"]
        state["export_summary"] = export_summary(result)
        print(json.dumps(state["export_summary"]), flush=True)
        verified = validate_archive(archive)
        state["local_archive"] = dict(path=str(archive), **verified)
        upload = archive
    except Exception as error:
        state.update(export_exit_code=41, archive_error=repr(error))
    if upload is not None:
        try:
            state["persistent_copy"] = deliver(archive, persistent, verified)
            upload = Path(state["persistent_copy"]["path"])
        except Exception as error:
            state.update(delivery_exit_code=43, delivery_error=repr(error),
                         failed_copy=getattr(error, "delivery_temporary", None))
            # Keep the first locally verified package. A fallback package also
            # contains the actual copy failure, without editing any old source.
            fallback = root / ("pingpong-k3-delivery-" + tag + "-local-fallback.tar.gz")
            try:
                result = export(source, fallback, first_code=first, stage=stage, modes=(),
                    source_read_only=True, provenance={**provenance, "delivery_failure": {
                        k: state[k] for k in (
                            "delivery_exit_code", "delivery_error", "failed_copy")}})
                state["fallback_archive"] = dict(path=str(fallback), **validate_archive(fallback))
                state["export_exit_code"] = (state["export_exit_code"]
                                            or result["export_validation_exit_code"])
                upload = fallback
            except Exception as secondary:
                state["fallback_error"] = repr(secondary)
                state["export_exit_code"] = state["export_exit_code"] or 41
    code = first or state["export_exit_code"] or state["delivery_exit_code"]
    state.update(final_exit_code=code, upload_path=str(upload) if upload else None)
    try:
        atomic_write_json(root / "reexport-status.json", state)
    except Exception as error:
        state["receipt_errors"].append(repr(error))
        code = code or 44
        state["final_exit_code"] = code
    print(json.dumps({k: state[k] for k in (
        "original_execution_exit_code", "original_stage", "export_exit_code",
        "delivery_exit_code", "final_exit_code", "upload_path", "local_root",
        "archive_error", "delivery_error", "fallback_error", "receipt_errors") if k in state}),
        flush=True)
    if upload is not None:
        print("UPLOAD ONLY: " + str(upload), flush=True)
    else:
        print("NO ARCHIVE: retained reexport directory " + str(root), flush=True)
    return code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--expected-execution", required=True)
    p.add_argument("--exporter-commit", required=True)
    p.add_argument("--tag", default="ordinary-cpu-reexport-" +
                   time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + str(os.getpid()))
    args = p.parse_args()
    try:
        repo = Path(__file__).resolve().parents[3]
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        require(actual == args.exporter_commit, "exporter SHA mismatch")
        require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=repo),
                "clean exporter worktree required")
        code = run(args.source, expected_execution=args.expected_execution,
            exporter_commit=args.exporter_commit, tag=args.tag,
            local_base=Path(os.environ.get("SR_K3_LOCAL_BASE", "/tmp/specrhythm-runs")),
            persistent=Path(os.environ.get("SR_PING_RESULTS",
                "/root/autodl-tmp/SpecRhythm-data/results/rolling-eager")))
    except Exception as error:
        print("Reexport preflight failed; retained source unchanged: " + repr(error),
              file=sys.stderr)
        code = 44
    raise SystemExit(code)


if __name__ == "__main__":
    main()
