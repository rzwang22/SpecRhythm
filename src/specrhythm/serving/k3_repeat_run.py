"""Two declared scaled K3 repetitions, one package; no retries and no extra parameter grid."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from specrhythm.phase4.manifest import atomic_write_json
from specrhythm.serving.common import read_json, require
from specrhythm.serving.k3 import B64, B128, MODES, geometry
from specrhythm.serving.k3_validation import plan

CONFIGS = {"io-only": "legacy", "unified": "unified"}
REPEATS = ("0-forward", "1-reverse")


def declaration(configuration, k3_configuration=B64):
    require(configuration in CONFIGS, "unknown K3 execution experiment")
    require(k3_configuration in (B64, B128), "repetitions require explicit B64/B128")
    return dict(
        schema_version="specrhythm.k3-repeat.v1",
        configuration=configuration,
        observation="deferred-window",
        draft_dispatch=CONFIGS[configuration],
        validation_profile="performance-exploration",
        k3_configuration=k3_configuration,
        output_equivalence_status="NOT_RUN",
        **({"capacity_policy": "four first-repeat probes; every fresh performance engine "
             "also rechecks actual loaded capacity"} if k3_configuration == B128 else {}),
        repeats=[
            dict(name=n, modes=list(MODES if i == 0 else reversed(MODES)))
            for i, n in enumerate(REPEATS)
        ],
    )


def read_declaration(directory):
    value = read_json(directory / "experiment-plan.json")
    require(
        value == declaration(value["configuration"], value["k3_configuration"]),
        "invalid repeated experiment declaration",
    )
    return value


def compare(directory):
    from specrhythm.serving.ping_prepost_delivery import comparison

    config = read_declaration(directory)
    repeats = []
    for name in REPEATS:
        root = directory / "repeats" / name
        if not root.exists():
            repeats.append(dict(repeat=name, valid=False, status="NOT_STARTED"))
            continue
        try:
            row = comparison(root, modes=MODES)
        except (ValueError, OSError, KeyError, TypeError) as error:
            row = dict(valid=False, error=str(error))
        if row.get("valid"):
            paths = [root / "points" / (m + "-audit-report.json") for m in MODES]
            row["experiment_matched"] = all(
                read_json(p)["options"].get("observation") == config["observation"]
                and read_json(p)["options"].get("draft_dispatch") == config["draft_dispatch"]
                and read_json(p).get("k3_configuration") == config["k3_configuration"]
                for p in paths
            )
            row["valid"] = row["experiment_matched"]
        repeats.append(dict(repeat=name, **row))
    values = {
        m: [
            p["throughput_tok_s"]
            for r in repeats
            for p in r.get("points", [])
            if p["mode"] == m and p.get("throughput_tok_s") is not None
        ]
        for m in MODES
    }
    return dict(
        **config,
        repetitions=repeats,
        valid=all(r["valid"] for r in repeats),
        throughput_ranges={
            m: dict(
                values=v,
                minimum=min(v) if v else None,
                maximum=max(v) if v else None,
                count=len(v),
            )
            for m, v in values.items()
        },
        inference="two single windows per mode; no stable speedup claim; "
        "dispatch attribution requires the other same-I/O configuration",
    )


def run(directory, repo, commit, configuration, k3_configuration=B64):
    require(not directory.exists(), "new repeated experiment directory required")
    config = declaration(configuration, k3_configuration)
    directory.mkdir(parents=True)
    atomic_write_json(directory / "experiment-plan.json", config)
    atomic_write_json(
        directory / "validation-plan.json", plan("performance-exploration", k3_configuration)
    )
    first, outcome = 0, dict(first_exit_code=0, stage="complete", mode=None)
    for i, name in enumerate(REPEATS):
        root = directory / "repeats" / name
        root.parent.mkdir(exist_ok=True)
        env = {
            **os.environ,
            "SR_K3_REPEAT_CHILD": "1",
            "SR_K3_SKIP_CAPACITY": "1" if k3_configuration == B128 and i else "0",
            "SR_K3_MANAGED_LOCAL": "1",
            "SR_K3_LOCAL_DELIVERY": str(root),
            "SR_K3_OBSERVATION": config["observation"],
            "SR_K3_DRAFT_DISPATCH": config["draft_dispatch"],
            "SR_K3_REVERSE": str(i),
            "SR_K3_VALIDATION_PROFILE": "performance-exploration",
        }
        code = subprocess.call(
            [
                "bash",
                str(
                    repo
                    / (
                        "scripts/run_k3_b"
                        + str(geometry(MODES[0], k3_configuration)["active_limit"])
                        + ".sh"
                    )
                ),
                commit,
            ],
            env=env,
        )
        code = 128 - code if code < 0 else code
        try:
            outcome = read_json(root / "runner-outcome.json")
            require(outcome["first_exit_code"] == code, "repeat outcome/exit mismatch")
        except (OSError, ValueError, KeyError, TypeError) as error:
            outcome = dict(
                first_exit_code=code or 44,
                stage="repeat_runner",
                mode=None,
                run_root=str(root),
                outcome_error=str(error),
            )
        first = outcome["first_exit_code"]
        outcome["repeat"] = name
        if first:
            failure = root / "first-failure.json"
            if failure.exists():
                try:
                    shutil.copyfile(failure, directory / "first-failure.json")
                except OSError as error:
                    outcome["failure_summary_copy_error"] = repr(error)
                    print(json.dumps(outcome), file=sys.stderr)
            break
    atomic_write_json(directory / "runner-outcome.json", outcome)
    try:
        comparison = compare(directory)
        atomic_write_json(directory / "comparison.json", comparison)
        if not first and not comparison["valid"]:
            first = 42
            outcome.update(first_exit_code=first, stage="repeated_comparison", mode=None)
            atomic_write_json(
                directory / "first-failure.json",
                dict(
                    failure_layer="diagnostic_evidence",
                    stage="repeated_comparison",
                    mode=None,
                    command_exit_code=first,
                    primary_error="repeated comparison incomplete",
                    evidence=str(directory / "comparison.json"),
                ),
            )
    except (OSError, ValueError, KeyError, TypeError) as error:
        # A secondary reporting failure must never replace an existing child failure.
        detail = dict(stage="repeated_comparison", error=repr(error), original_exit_code=first)
        print(json.dumps(detail), file=sys.stderr)
        first = first or 44
        outcome.update(first_exit_code=first, comparison_error=detail)
        try:
            atomic_write_json(directory / "repeat-wrapper-errors.json", detail)
        except OSError as secondary:
            print("secondary receipt error: " + repr(secondary), file=sys.stderr)
    atomic_write_json(directory / "runner-outcome.json", outcome)
    return first


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--configuration", choices=CONFIGS, required=True)
    p.add_argument("--k3-configuration", choices=(B64, B128), default=B64)
    args = p.parse_args()
    try:
        code = run(
            args.directory, args.repo, args.commit, args.configuration, args.k3_configuration
        )
    except Exception as error:
        print(json.dumps(dict(stage="repeat_wrapper", error=str(error))), file=sys.stderr)
        code = 44
    raise SystemExit(code)


if __name__ == "__main__":
    main()
