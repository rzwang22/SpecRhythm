"""Operator-only joint complete-output check, including actual cross-admission coverage."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from specrhythm.serving.common import read_json, require
from specrhythm.serving.fixed_results import device_batches
from specrhythm.serving.ping_prepost import MODES, PROTOCOL
from specrhythm.serving.prepost_gpu_check import compare_outputs, prepare
from specrhythm.serving.s1_workload import write_once


def coverage(runtime, backend):
    ping = backend.get("prepost", {}).get("pingpong", {})
    require(ping.get("protocol") == PROTOCOL, "joint check lacks actual PingPong owner")
    events = ping.get("events", [])
    admissions = [e for e in events if e.get("event") == "admission" and e["actual_B"]]
    trajectories = {}
    for event in admissions:
        for claim in event["claims"]:
            trajectories.setdefault(claim["request_id"], []).append(claim)
    crossing = [
        rid
        for rid, rows in trajectories.items()
        if any(
            [r["opportunity_cohort"] for r in rows[i : i + 3]] == ["A", "B", "A"]
            for i in range(len(rows) - 2)
        )
    ]
    settle = [r for r in events if r.get("event") == "settlement"]
    recoveries = [
        r
        for r in settle
        if r["accepted"] < r["parent_length"]
        and r["next_candidate_length"] == 1
        and not r["terminal"]
    ]
    restored = [
        r
        for r in recoveries
        if any(
            s["request_id"] == r["request_id"]
            and s["prefix_version"] > r["prefix_version"]
            and s["retained"] == 3
            and s["next_candidate_length"] == 4
            for s in settle
        )
    ]
    missing = []
    if not crossing:
        missing.append("consecutive A/B/A admission")
    if not recoveries or not restored:
        missing.append("rejection P1 then restored P4")
    require(
        not ping.get("claims") and not ping.get("ready") and not ping.get("pending_normal"),
        "joint owner did not release all claims/readiness",
    )
    require(
        all(0 < s["B"] <= 8 for s in runtime["target_steps"] if s["B"]),
        "joint Target exceeds B16/ceiling8",
    )
    return dict(
        status="COMPLETE" if not missing else "INCOMPLETE",
        missing=missing,
        cross_ABA_request_ids=crossing,
        rejection_P1=len(recoveries),
        restored_P4=len(restored),
        actual_Target_ceiling=8,
    )


def run(source, directory):
    from specrhythm.serving.fixed_cli import run_point

    require(not directory.exists(), "joint correctness needs a fresh root")
    directory.mkdir(parents=True)
    runtimes, backends, receipts = {}, {}, []
    output_status, layer = "PENDING", "joint_execution"
    try:
        for mode in ("target", *MODES):
            root = directory / mode
            path = prepare(source, root, mode, modes=MODES)
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
            runtime = runtimes[mode] = read_json(point / "runtime.json")
            backends[mode] = read_json(point / "draft-backend-report.json")
            device_batches(runtime["target_devices"], runtime["target_steps"])
            require(
                all(
                    not r["structural_errors"]
                    for d in runtime["target_devices"]
                    for r in d["target_rows"]
                ),
                "joint actual Target layout invalid",
            )
            receipts.append(dict(mode=mode, point=str(point), execution="PASS", cleanup="PASS"))
        layer = "joint_output_and_mixed_verification"
        result = compare_outputs(runtimes, modes=MODES)
        output_status = "PASS"
        layer = "joint_coverage"
        cov = coverage(runtimes[MODES[1]], backends[MODES[1]])
        write_once(directory / "coverage.json", cov)
        require(cov["status"] == "COMPLETE", "joint real protocol coverage incomplete", **cov)
        value = dict(
            **result,
            coverage=cov,
            runs=receipts,
            protocol=PROTOCOL,
            output_correctness=output_status,
            single_shared_correctness=True,
        )
        write_once(directory / "result.json", value)
        return value
    except BaseException as error:
        try:
            write_once(
                directory / "failure.json",
                dict(
                    error=str(error),
                    failure_layer=layer,
                    output_correctness=output_status,
                    original_returncode=getattr(error, "details", {}).get("returncode", 1),
                    completed_runs=receipts,
                    GPU_correctness="NOT_QUALIFIED",
                    performance="NOT_TESTED",
                ),
            )
        except Exception as secondary:
            print("secondary joint failure recording: " + str(secondary), file=sys.stderr)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run(args.source, args.output)))
    except Exception as error:
        traceback.print_exc()
        code = getattr(error, "details", {}).get("returncode", 1)
        raise SystemExit(code if type(code) is int and 0 < code < 256 else 1) from error


if __name__ == "__main__":
    main()
