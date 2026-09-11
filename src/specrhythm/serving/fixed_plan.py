"""Independent fixed-concurrency diagnostic inputs; no S2 capacity search or SLO gates."""

from __future__ import annotations

import json
import math
import shutil
from collections import Counter
from pathlib import Path

from specrhythm.phase4.config import load_phase4_config
from specrhythm.phase4.manifest import (
    collect_environment,
    collect_topology,
    sha256_file,
    validate_environment,
    validate_topology,
)
from specrhythm.serving.common import digest, require
from specrhythm.serving.s1_preflight import git_identity, validate_execution_files
from specrhythm.serving.s1_workload import load_execution, write_once
from specrhythm.serving.s2_plan import sealed

MODES = ("target", "serial", "serial-split", "pingpong")
SCENARIO = "prefill-complete, all requests ready; fixed-concurrency finite supply"
POLICY = {
    f"cross_run_{key}_equality": "NOT_REQUIRED" for key in ("token", "length", "EOS", "round")
}


def settings(
    *,
    warmup_steps=2,
    samples=12,
    repeats=1,
    window_seconds=30,
    setup_timeout=900,
    drain_timeout=60,
    observation="original-live",
):
    for name, value, minimum in (
        ("warmup_steps", warmup_steps, 0),
        ("samples", samples, 1),
        ("repeats", repeats, 1),
    ):
        require(type(value) is int and value >= minimum, "invalid diagnostic count", field=name)
    for name, value in (
        ("window_seconds", window_seconds),
        ("setup_timeout", setup_timeout),
        ("drain_timeout", drain_timeout),
    ):
        require(
            type(value) in (int, float) and math.isfinite(value) and value > 0,
            "invalid diagnostic timeout",
            field=name,
        )
    require(observation == "original-live", "only original live UUID/full pool audits supported")
    return dict(
        warmup_steps=warmup_steps,
        samples=samples,
        repeats=repeats,
        window_seconds=window_seconds,
        setup_timeout=setup_timeout,
        drain_timeout=drain_timeout,
        observation=observation,
    )


def capacity_metadata(mode, resident_count=None):
    require(mode in MODES, "unknown fixed diagnostic mode", actual=mode)
    grouped = mode in ("serial-split", "pingpong")
    return {
        "resident_request_requirement": 100,
        "resident_request_count": resident_count,
        "resident_count_semantics": "actual after prefill; null before state preparation",
        "active_request_limit": 64,
        "cohort_count": 2 if grouped else 0,
        "per_cohort_capacity": 32 if grouped else None,
        "max_requests_per_target_forward": 32 if grouped else 64,
        "target_sequence_limit": 128,
        "target_query_token_limit": 4096,
        "draft_sequence_limit": 128,
        "draft_query_token_limit": 4096,
        "max_model_len": 4096,
        "proposal_budget": 4,
        "actual_KV_limits": "model-loaded per-rank actual-capacity.json; never guessed",
    }


def point(mode, *, kind="continuous", batch=None, half="A", repeat=0, warmup=False):
    require(mode in MODES and kind in ("continuous", "initial-state"), "invalid diagnostic point")
    require(half in ("A", "B"), "invalid shape half")
    if kind == "initial-state":
        require(batch in (32, 64), "initial-state B must be 32 or 64")
    return dict(
        mode=mode,
        kind=kind,
        batch=batch,
        half=half,
        repeat=repeat,
        discard_warmup=warmup,
        runtime_mode="pingpong" if mode == "serial-split" else mode,
    )


def stage_points(repeats=1, warmups=0):
    # Every sample restores by a fresh process + real prefill. No continuation KV replay.
    for i in range(warmups + repeats):
        for mode in ("target", "serial"):
            for batch, half in ((32, "A"), (32, "B"), (64, "A")):
                yield point(
                    mode,
                    kind="initial-state",
                    batch=batch,
                    half=half,
                    repeat=i,
                    warmup=i < warmups,
                )
        yield point("pingpong", kind="initial-state", batch=32, repeat=i, warmup=i < warmups)


def build_manifest(execution, ids, workload_sha, options):
    require(len(ids) == len(set(ids)) == 100, "fixed diagnostic requires original mixed100")
    trace = sealed(
        {
            "schema_version": "specrhythm.fixed-ready-trace.v1",
            "kind": SCENARIO,
            "rows": [{"request_id": rid, "arrival_offset_seconds": 0.0} for rid in ids],
            "original_S0_arrivals_modified": False,
            "arrival_replay_enabled": False,
        }
    )
    return sealed(
        {
            "schema_version": "specrhythm.s2-execution.v1",  # Reuse serving input/KV contracts.
            "execution": execution,
            "request_ids": ids,
            "workload_file": "requests.jsonl",
            "workload_sha256": workload_sha,
            "trace": trace,
            "active_limit": 64,
            "actual_N": 100,
            "requested_N": 100,
            "fixed_diagnostic": {
                "schema_version": "specrhythm.fixed-diagnostic.v1",
                "scenario": SCENARIO,
                "options": options,
                "capacity": {m: capacity_metadata(m) for m in MODES},
                "initial_request_ids": ids[:64],
                "cohorts": {"A": ids[:32], "B": ids[32:64]},
                "replacement_request_ids": ids[64:],
                "request_order_sha256": digest(ids),
                "restore": "fresh process/engine + real prefill per point; no KV replay",
                "stage_definition": "initial-state K4; shape mismatch excluded explicitly",
                "fixed_shape_vs_continuous": "separate sample populations, never pooled",
                **POLICY,
            },
        }
    )


def prepare(root, s1, options):
    old, rows = load_execution(s1 / "s1-mixed100/execution-manifest.json", verify_parent=True)
    ids = old["logical"]["request_ids"]
    by_id = {r.request_id: r for r in rows}
    require(
        Counter(r.task_class for r in rows)
        == {"chat": 30, "code": 30, "reasoning": 20, "summarization": 20},
        "source is not the qualified mixed100 library",
    )
    require(not root.exists(), "new diagnostic requires a new result root", artifact=str(root))
    commit = git_identity()
    root.mkdir(parents=True)
    for name in ("config.json", "patch-manifest.json"):
        shutil.copyfile(s1 / name, root / name)
    config = load_phase4_config(str(root / "config.json"))
    require(
        config.proposal_budget == 4 and config.target.tensor_parallel_size == 2,
        "fixed diagnostic requires K4 and Target TP2",
    )
    execution = dict(old["execution"])
    execution.update(
        git_commit=commit,
        arrival_replay_enabled=False,
        capacity=capacity_metadata("target"),
        engine_core="synchronous InprocClient",
        async_scheduling=False,
    )
    environment = collect_environment(Path(execution["vllm_source"]))
    topology = collect_topology()
    require(
        validate_environment(environment, config)["valid"]
        and validate_topology(topology, config)["valid"],
        "fixed diagnostic environment/device binding invalid",
    )
    for name, value in (("environment", environment), ("topology", topology)):
        write_once(root / (name + ".json"), value)
        execution[name + "_sha256"] = sha256_file(root / (name + ".json"))
    validate_execution_files(root, execution)
    inputs = root / "inputs"
    inputs.mkdir()
    workload = inputs / "requests.jsonl"
    # Preserve the exact S0 RequestDefinition rows, not the runtime adapter dataclass.
    from specrhythm.serving.schema import load_requests

    source = {r.request_id: r for r in load_requests(s1 / "s1-mixed100" / old["workload_file"])}
    require(set(source) == set(by_id), "mixed100 source row identity mismatch")
    with workload.open("x") as handle:
        for rid in ids:
            handle.write(
                json.dumps(source[rid].to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            )
    manifest = build_manifest(execution, ids, sha256_file(workload), options)
    write_once(inputs / "execution-manifest.json", manifest)
    write_once(
        root / "diagnostic-config.json",
        {
            **manifest["fixed_diagnostic"],
            "source_s1": str(s1),
            "source_manifest_sha256": old["manifest_sha256"],
            "execution": execution,
            "execution_manifest_sha256": manifest["sha256"],
        },
    )
    return manifest
