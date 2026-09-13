"""Operator-only K3 full-output comparison through the shared production launcher."""

import argparse
import json
import traceback
from pathlib import Path

from specrhythm.serving.common import require
from specrhythm.serving.k3 import MODES, PROTOCOL, accounting_complete
from specrhythm.serving.ping_prepost_gpu_check import run as joint_run


def coverage(runtime, backend):
    ping = backend.get("prepost", {}).get("pingpong", {})
    require(ping.get("protocol") == PROTOCOL, "K3 owner protocol missing")
    require(
        accounting_complete(backend.get("prepost", {}).get("candidate_accounting")),
        "K3 final lifetime candidate accounting missing/inconsistent",
    )
    events = ping.get("events", [])
    ready = {(r["request_id"], r["prefix_version"]): r for r in events if r["event"] == "ready"}
    recoveries = [
        r for r in events if r["event"] == "settlement" and r["correction"] and not r["terminal"]
    ]
    restored = [
        r
        for r in recoveries
        if ready.get((r["request_id"], r["prefix_version"] + 1), {}).get("candidate_length") == 3
    ]
    require(
        not any(ping.get(k) for k in ("claims", "ready", "pending_normal")),
        "K3 correctness retains owner resources",
    )
    missing = []
    if not restored:
        missing.append("actual rejection followed by complete K3 READY")
    homes = {c["home_cohort"] for e in events if e["event"] == "admission" for c in e["claims"]}
    if homes != {"A", "B"}:
        missing.append("both homes verified")
    for step in runtime["target_steps"]:
        if not step["B"]:
            continue
        require(0 < step["B"] <= 8, "K3 Target request ceiling violated")
        for c in step["ping_admission"]["claims"]:
            r = ready.get((c["request_id"], c["prefix_version"]))
            require(r is not None, "K3 actual claim lacks READY evidence")
            n = len(c["proposal"]["proposal_token_ids"])
            require(
                n == min(3, r["remaining_output_budget"])
                or (
                    0 < n < min(3, r["remaining_output_budget"])
                    and r["short_reason"] == "candidate_EOS"
                ),
                "invalid actual K3 claim length",
            )
    return dict(
        status="INCOMPLETE" if missing else "COMPLETE",
        missing=missing,
        rejection_K3_recoveries=len(restored),
        protocol=PROTOCOL,
        pipeline_behavior="separate native evidence; output PASS is not overlap",
    )


def run(source, directory):
    return joint_run(
        source,
        directory,
        modes=MODES,
        protocol=PROTOCOL,
        coverage_check=coverage,
        require_mixed=False,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        print(json.dumps(run(args.source, args.output)))
    except Exception as error:
        traceback.print_exc()
        code = getattr(error, "details", {}).get("returncode", 1)
        raise SystemExit(code if type(code) is int and 0 < code < 256 else 1) from error


if __name__ == "__main__":
    main()
