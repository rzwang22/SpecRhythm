"""Operator-only joint Target TP2 + Draft check through the production service/runner.

A separate, recorded 16-request/32-output fixture is never a performance result.
All requests finish naturally, then both protocols compare full output to Target-only.
Owned process launch, physical audits, shutdown and deadline handling are reused.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import traceback
from pathlib import Path

from specrhythm.continuation.prepost import MODES
from specrhythm.phase4.manifest import sha256_file
from specrhythm.serving.common import read_json, require
from specrhythm.serving.fixed_plan import capacity_metadata
from specrhythm.serving.s1_workload import write_once
from specrhythm.serving.s2_plan import sealed


def prepare(source, root, mode, *, modes=MODES, configuration="k3-b16-v1",
            source_configuration=None, max_output_tokens=32):
    from specrhythm.serving.k3 import configuration_fields, configuration_of, geometry

    fields = configuration_fields(configuration)
    count = geometry("serial-k3", configuration)["active_limit"]
    require(
        mode in ("target", *modes) and not root.exists(), "new joint correctness root required"
    )
    source_configuration = source_configuration or configuration
    source_count = geometry("serial-k3", source_configuration)["active_limit"]
    require(type(max_output_tokens) is int and 1 <= max_output_tokens <= 32,
            "invalid bounded correctness output budget")
    base = read_json(source / f"inputs/execution-B{source_count}.json")
    raw = [
        json.loads(line) for line in (source / "inputs/requests.jsonl").read_text().splitlines()
    ]
    # Same frozen prompts/seeds/models, explicit bounded output fixture only here.
    rows = [{**r, "maximum_new_tokens": min(r["maximum_new_tokens"], max_output_tokens)}
            for r in raw[:count]]
    require(len(rows) == count and configuration_of(base) == source_configuration,
            "joint correctness source count/configuration mismatch")
    root.mkdir(parents=True)
    for name in ("config.json", "patch-manifest.json", "environment.json", "topology.json"):
        shutil.copyfile(source / name, root / name)
    inputs = root / "inputs"
    inputs.mkdir()
    with (inputs / "requests.jsonl").open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    ids = [r["request_id"] for r in rows]
    manifest = copy.deepcopy(base)
    manifest.pop("sha256")
    if source_configuration != configuration:
        manifest.pop("k3_configuration", None)
        manifest.pop("validation_profile", None)
    if configuration != "k3-b16-v1":
        from specrhythm.serving.k3_validation import STRICT

        manifest["validation_profile"] = STRICT
    manifest.update(
        request_ids=ids,
        actual_N=count,
        requested_N=count,
        active_limit=count,
        **fields,
        workload_sha256=sha256_file(inputs / "requests.jsonl"),
    )
    trace = dict(base["trace"])
    trace.pop("sha256")
    trace["rows"] = [r for r in trace["rows"] if r["request_id"] in ids]
    manifest["trace"] = sealed(trace)
    manifest["execution"]["capacity"]["resident_pool"] = count
    diag = manifest["fixed_diagnostic"]
    diag["scenario"] = f"joint correctness; complete {count}-request outputs; not performance"
    diag["options"].update(
        samples=4096,
        warmup_steps=0,
        window_seconds=900,
        draft_audit="full" if mode == "target" else "runtime",
    )
    diag["capacity"] = {
        mode: capacity_metadata(
            mode, active_limit=count, resident_requirement=count, target_sequence_limit=512,
            k3_configuration=configuration
        )
    }
    diag.update(initial_request_ids=ids, replacement_request_ids=[], cohorts={"A": [], "B": []})
    manifest["prepost_correctness_fixture"] = dict(
        source_workload_sha256=base["workload_sha256"],
        request_count=count,
        max_output_tokens=max_output_tokens,
        complete_outputs_required=True,
        performance_result=False,
    )
    path = inputs / "execution-manifest.json"
    write_once(path, sealed(manifest))
    return path


def compare_outputs(runtimes, *, modes=MODES, require_mixed=True, request_count=16,
                    compare_termination=False):
    require(set(runtimes) == {"target", *modes}, "joint check lacks a mode/reference")
    values = {}
    for mode, runtime in runtimes.items():
        require(
            runtime["stop_reason"] == "all_naturally_completed"
            and len(runtime["requests"]) == request_count
            and all(
                r["state"] == "FINISHED" and r["resources_released"] for r in runtime["requests"]
            ),
            "joint output/cleanup incomplete",
        )
        values[mode] = {r["request_id"]: r["generated_token_ids"] for r in runtime["requests"]}
        if compare_termination:
            require(len(values[mode]) == request_count and all(
                len(r["generated_token_ids"]) <= 32 and isinstance(r.get("finish_reason"), str)
                and r["finish_reason"] for r in runtime["requests"]),
                "B64 correctness output budget/termination evidence invalid")
    if compare_termination:
        terminal = {m: {r["request_id"]: r["finish_reason"] for r in v["requests"]}
                    for m, v in runtimes.items()}
        require(all(terminal[m] == terminal["target"] for m in modes),
                "B64 correctness termination reasons differ")
    if not require_mixed:
        require(all(len(v) == request_count and set(v) == set(values["target"])
                    for v in values.values()),
                "K3 complete-output request identity set differs")
    comparisons = [
        dict(
            mode=mode,
            request_id=rid,
            exact=values[mode].get(rid) == tokens,
            target_tokens=tokens,
            actual_tokens=values[mode].get(rid),
        )
        for mode in modes
        for rid, tokens in values["target"].items()
    ]
    require(
        all(r["exact"] for r in comparisons),
        "joint Target-only full output mismatch",
        comparisons=comparisons,
    )
    mixed = [
        s
        for s in runtimes[modes[1]]["target_steps"]
        if {r["candidate_positions"] for r in s.get("rows", [])} >= {1, 4}
    ]
    if require_mixed:
        require(mixed, "joint correctness coverage missing actual mixed 1/4 Target forward")
    return dict(
        valid=True,
        request_count=request_count,
        termination_comparison="exact by request ID" if compare_termination else "legacy",
        lengths_compared=True,
        GPU_correctness="PASS",
        comparisons=comparisons,
        mixed_target_steps=len(mixed) if require_mixed else None,
        mixed_length_requirement="P1/P4" if require_mixed else "uniform K3 with budget/EOS tails",
        performance="NOT_TESTED",
        overlap="NOT_QUALIFIED",
    )


def run(source):
    from specrhythm.serving.fixed_cli import run_point

    directory = source / "prepost-joint-gpu-check"
    require(not directory.exists(), "joint correctness evidence already exists")
    directory.mkdir()
    runtimes, receipts = {}, []
    try:
        for mode in ("target", *MODES):
            root = directory / mode
            path = prepare(source, root, mode)
            selected = dict(
                mode=mode,
                runtime_mode=mode,
                kind="joint-correctness",
                batch=16,
                half="A",
                repeat=0,
                discard_warmup=False,
                prepost_correctness=True,
            )
            point, result = run_point(root, selected, manifest_path=path)
            require(
                result["valid"]
                and result["execution_status"] == "PASS"
                and result["cleanup_status"] == "PASS",
                "joint execution/cleanup failed",
            )
            runtimes[mode] = read_json(point / "runtime.json")
            # Native TP rows must prove one actual forward, including mixed batches.
            from specrhythm.serving.fixed_results import device_batches

            device_batches(runtimes[mode]["target_devices"], runtimes[mode]["target_steps"])
            require(
                all(
                    not r["structural_errors"]
                    for d in runtimes[mode]["target_devices"]
                    for r in d["target_rows"]
                ),
                "joint Target positions/attention structural validation failed",
            )
            receipts.append(
                dict(mode=mode, directory=str(point), execution="PASS", cleanup="PASS")
            )
        result = compare_outputs(runtimes)
        write_once(directory / "result.json", {**result, "runs": receipts})
        return result
    except BaseException as error:
        try:
            write_once(
                directory / "failure.json",
                dict(
                    error=str(error),
                    original_returncode=getattr(error, "details", {}).get("returncode", 1),
                    completed_runs=receipts,
                    GPU_correctness="FAILED",
                    performance="NOT_TESTED",
                ),
            )
        except Exception as recording_error:
            print("secondary joint failure-record error: " + str(recording_error), file=sys.stderr)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run(args.root)))
    except Exception as error:
        traceback.print_exc()
        code = getattr(error, "details", {}).get("returncode", 1)
        raise SystemExit(code if type(code) is int and 0 < code < 256 else 1) from error


if __name__ == "__main__":
    main()
