"""Independent frozen resident360/B16..128 time-window scan; no model execution here."""

from __future__ import annotations

import json
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
from specrhythm.serving.common import digest, read_json, require
from specrhythm.serving.fixed_plan import POLICY, capacity_metadata, settings
from specrhythm.serving.s1_preflight import git_identity, validate_execution_files
from specrhythm.serving.s1_workload import load_execution, verify_s0, write_once
from specrhythm.serving.s2_plan import MODES, RATIO, check_seal, sealed, selection_order
from specrhythm.serving.schema import load_requests

BATCHES = (16, 32, 64, 128)
EXPLICIT_MODES = (*MODES, "serial-eager")
POOL_SIZE = 360
SCHEMA = "specrhythm.decode-scan.v1"
BOUNDARY = "prefilled resident pool; post-warmup full-batch decode; actual stop before drain"


def options(**kwargs):
    value = settings(observation="buffered-live", identity_matching="bound-prefix", **kwargs)
    value["samples"] = None  # Results, never a measurement stop budget.
    return value


def select(main, small_ids, seed=1666):
    ranked = selection_order(main, small_ids, seed)
    by_id = {r.request_id: r for r in main}
    chosen = [by_id[rid] for rid in ranked["request_ids"][:POOL_SIZE]]
    require(
        len({r.request_id for r in chosen}) == POOL_SIZE, "scan pool must be unique resident360"
    )
    require(
        Counter(r.task_class for r in chosen) == {k: 36 * v for k, v in RATIO.items()},
        "scan pool class ratio differs",
    )
    require(
        all(r.prompt_length + r.maximum_new_tokens + 4 <= 4096 for r in chosen),
        "scan pool exceeds unchanged context including K4",
    )
    return chosen, sealed(
        {
            "selection_seed": seed,
            "algorithm": ranked["algorithm"],
            "request_ids": [r.request_id for r in chosen],
            "request_order_sha256": digest([r.request_id for r in chosen]),
            "source_selection_sha256": ranked["sha256"],
            "class_counts": dict(Counter(r.task_class for r in chosen)),
        }
    )


def selected_point(mode, batch, repeat=0):
    require(mode in EXPLICIT_MODES and batch in BATCHES, "unknown decode scan mode/B")
    return dict(
        mode=mode,
        runtime_mode=mode,
        kind="decode-scan",
        batch=batch,
        half="A",
        repeat=repeat,
        discard_warmup=False,
        scan=True,
    )


def manifest(execution, ids, workload_sha, opts, batch):
    require(
        batch in BATCHES and len(ids) == len(set(ids)) == POOL_SIZE,
        "invalid scan batch or resident pool",
    )
    trace = sealed(
        {
            "schema_version": "specrhythm.fixed-ready-trace.v1",
            "kind": BOUNDARY,
            "rows": [{"request_id": rid, "arrival_offset_seconds": 0.0} for rid in ids],
            "arrival_replay_enabled": False,
            "original_S0_arrivals_modified": False,
        }
    )
    return sealed(
        {
            "schema_version": "specrhythm.s2-execution.v1",
            "execution": execution,
            "request_ids": ids,
            "workload_file": "requests.jsonl",
            "workload_sha256": workload_sha,
            "trace": trace,
            "active_limit": batch,
            "actual_N": POOL_SIZE,
            "requested_N": POOL_SIZE,
            "fixed_diagnostic": {
                "schema_version": SCHEMA,
                "scenario": BOUNDARY,
                "options": opts,
                "capacity": {
                    m: capacity_metadata(
                        m,
                        active_limit=batch,
                        resident_requirement=POOL_SIZE,
                        target_sequence_limit=512,
                    )
                    for m in EXPLICIT_MODES
                },
                "initial_request_ids": ids[:batch],
                "cohorts": {"A": ids[: batch // 2], "B": ids[batch // 2 : batch]},
                "replacement_request_ids": ids[batch:],
                "request_order_sha256": digest(ids),
                "refill_rule": ("FIFO prepared requests; vacant nonbusy cohort, balance A then B"),
                "bootstrap_terminal_rule": "skip natural setup terminals; record admissions",
                "pingpong_readiness_policy": "scan-global-fifo-full-cohort-v1",
                "restore": "fresh process/engine + all360 real prefill per point; no KV replay",
                **POLICY,
            },
        }
    )


def prepare(root, s1, opts, *, seed=1666, s0=None):
    require(not root.exists(), "decode scan requires a new result root", artifact=str(root))
    old, _ = load_execution(s1 / "s1-mixed100/execution-manifest.json", verify_parent=True)
    require(read_json(s1 / "G3/comparison.json")["valid"] is True, "scan requires qualified S1 G3")
    s0 = s0 or Path(old["parent"]["directory"])
    parent = verify_s0(s0)
    chosen, selection = select(
        load_requests(s0 / "main1000.jsonl"), old["logical"]["request_ids"], seed
    )
    commit = git_identity()
    root.mkdir(parents=True)
    for name in ("config.json", "patch-manifest.json"):
        shutil.copyfile(s1 / name, root / name)
    config = load_phase4_config(str(root / "config.json"))
    require(
        config.proposal_budget == 4 and config.target.tensor_parallel_size == 2,
        "scan requires unchanged K4/Target TP2",
    )
    execution = dict(old["execution"])
    execution.update(
        git_commit=commit,
        arrival_replay_enabled=False,
        capacity={
            "max_model_len": 4096,
            "max_num_seqs": 512,
            "max_num_batched_tokens": 4096,
            "resident_pool": POOL_SIZE,
        },
        engine_core="synchronous InprocClient",
        async_scheduling=False,
    )
    environment = collect_environment(Path(execution["vllm_source"]))
    topology = collect_topology()
    require(
        validate_environment(environment, config)["valid"]
        and validate_topology(topology, config)["valid"],
        "scan environment/device binding invalid",
    )
    for name, value in (("environment", environment), ("topology", topology)):
        write_once(root / (name + ".json"), value)
        execution[name + "_sha256"] = sha256_file(root / (name + ".json"))
    validate_execution_files(root, execution)
    inputs = root / "inputs"
    inputs.mkdir()
    workload = inputs / "requests.jsonl"
    with workload.open("x") as handle:
        for row in chosen:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    work_sha = sha256_file(workload)
    manifests = {}
    for batch in BATCHES:
        value = manifest(execution, selection["request_ids"], work_sha, opts, batch)
        name = f"execution-B{batch}.json"
        write_once(inputs / name, value)
        manifests[str(batch)] = {"path": "inputs/" + name, "sha256": value["sha256"]}
    result = sealed(
        {
            "schema_version": SCHEMA,
            "source_s1": str(s1),
            "parent": parent,
            "execution": execution,
            "selection": selection,
            "pool_size": POOL_SIZE,
            "workload_sha256": work_sha,
            "options": opts,
            "manifests": manifests,
            "points": [
                selected_point(m, b, r)
                for r in range(opts["repeats"])
                for b in BATCHES
                for m in MODES
            ],
            "optional_points": [
                selected_point("serial-eager", b, r)
                for r in range(opts["repeats"])
                for b in BATCHES
            ],
            "boundary": BOUNDARY,
            "capacity": "PENDING per fresh point before decode",
            **POLICY,
        }
    )
    write_once(root / "scan-config.json", result)
    return result


def load(root):
    value = read_json(root / "scan-config.json")
    check_seal(value)
    require(value["schema_version"] == SCHEMA, "not a decode scan root")
    return value


def point_manifest(root, batch):
    entry = load(root)["manifests"][str(batch)]
    path = root / entry["path"]
    require(read_json(path)["sha256"] == entry["sha256"], "scan point manifest binding changed")
    return path
