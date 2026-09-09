"""Independent S2 frozen inputs; model capacity is confirmed by owned server probes."""

from __future__ import annotations

import json
import shutil
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
from specrhythm.serving.s1_preflight import git_identity, validate_execution_files
from specrhythm.serving.s1_workload import load_execution, verify_s0, write_once
from specrhythm.serving.s2_plan import (
    ACTIVE_LIMIT,
    MODES,
    POOL_SLOTS,
    QUERY_LIMIT,
    RATES,
    check_seal,
    freeze_sizes,
    poisson_trace,
    sealed,
    selection_order,
)
from specrhythm.serving.schema import load_requests


def prepare(
    root, s0, s1, *, arrival_seed=1667, order_seed=1668, selection_seed=1666, extra_seeds=()
):
    pairs = [(arrival_seed, order_seed), *map(tuple, extra_seeds)]
    require(len(pairs) == len(set(pairs)), "S2 seed pairs must be unique")
    parent = verify_s0(s0)
    old, _ = load_execution(s1 / "s1-mixed100" / "execution-manifest.json", verify_parent=True)
    require(
        read_json(s1 / "G3" / "comparison.json")["valid"] is True, "S2 requires validated S1-P G3"
    )
    require(
        old["execution"]["git_commit"] == "5a00049e2eabf09f535fdd5f187f77406f6dcfe2",
        "S2 baseline differs from requested validated S1 commit",
    )
    root.mkdir(parents=True, exist_ok=False)
    for name in ("config.json", "patch-manifest.json"):
        shutil.copyfile(s1 / name, root / name)
    execution = dict(old["execution"])
    execution.update(
        git_commit=git_identity(),
        arrival_replay_enabled=True,
        capacity={
            "max_model_len": 4096,
            "max_num_seqs": POOL_SLOTS,
            "max_num_batched_tokens": QUERY_LIMIT,
            "active_limit": ACTIVE_LIMIT,
        },
        engine_core="synchronous InprocClient",
        async_scheduling=False,
    )
    config = load_phase4_config(str(root / "config.json"))
    environment = collect_environment(Path(execution["vllm_source"]))
    topology = collect_topology()
    require(
        validate_environment(environment, config)["valid"]
        and validate_topology(topology, config)["valid"],
        "S2 pinned environment/device topology invalid",
    )
    for name, value in (("environment", environment), ("topology", topology)):
        write_once(root / (name + ".json"), value)
    for name, key in (
        ("config", "config_sha256"),
        ("patch-manifest", "patch_manifest_sha256"),
        ("environment", "environment_sha256"),
        ("topology", "topology_sha256"),
    ):
        execution[key] = sha256_file(root / (name + ".json"))
    validate_execution_files(root, execution)
    main = load_requests(s0 / "main1000.jsonl")
    selected = selection_order(main, old["logical"]["request_ids"], selection_seed)
    write_once(root / "selection.json", selected)
    write_once(
        root / "preparation.json",
        sealed(
            {
                "schema_version": "specrhythm.s2-preparation.v1",
                "parent": parent,
                "s1_baseline_directory": str(s1),
                "s1_execution_sha256": old["manifest_sha256"],
                "execution": execution,
                "seed_pairs": [{"arrival_seed": a, "order_seed": o} for a, o in pairs],
                "arrival_seed": arrival_seed,
                "order_seed": order_seed,
                "rates": list(RATES),
                "selection_sha256": selected["sha256"],
                "static_capacity_estimate": {
                    "requested_sizes": [100, 500],
                    "active_limit": ACTIVE_LIMIT,
                    "formula": (
                        "all prefix ceil(block) + worst 128 growth through full output cap "
                        "+ K4 + one partial copy block/active + max(32,5% blocks)"
                    ),
                    "logits_reserve": "Draft N*vocab_size*4 bytes",
                    "additional_workspace_bytes": 512 * 1024**2,
                    "snapshot_copies": 0,
                    "actual_model_loaded_rank_capacity_confirmed": False,
                    "permission_to_time": False,
                },
            }
        ),
    )
    return root


def preparation(root):
    value = read_json(root / "preparation.json")
    check_seal(value)
    selected = read_json(root / "selection.json")
    check_seal(selected)
    require(selected["sha256"] == value["selection_sha256"], "S2 frozen selection changed")
    return value, selected


def source_rows(root, split="main"):
    value, _ = preparation(root)
    require(
        verify_s0(Path(value["parent"]["directory"])) == value["parent"],
        "S2 sealed S0 source changed",
    )
    return load_requests(
        Path(value["parent"]["directory"])
        / ("main1000.jsonl" if split == "main" else "calibration200.jsonl")
    )


def execution_manifest(
    root,
    directory,
    ids,
    *,
    rate=0.25,
    requested=None,
    active_limit=ACTIVE_LIMIT,
    split="main",
    arrival_seed=None,
    order_seed=None,
):
    prep, _ = preparation(root)
    by_id = {r.request_id: r for r in source_rows(root, split)}
    rows = [by_id[r] for r in ids]
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "requests.jsonl"
    with path.open("x") as handle:
        for r in rows:
            handle.write(json.dumps(r.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    trace = poisson_trace(
        ids,
        rate,
        prep["arrival_seed"] if arrival_seed is None else arrival_seed,
        prep["order_seed"] if order_seed is None else order_seed,
    )
    write_once(directory / "trace.json", trace)
    value = sealed(
        {
            "schema_version": "specrhythm.s2-execution.v1",
            "execution": prep["execution"],
            "workload_file": path.name,
            "workload_sha256": sha256_file(path),
            "request_ids": ids,
            "request_set_sha256": digest(sorted(ids)),
            "source_split": split,
            "original_S0_arrivals_retained": True,
            "trace": trace,
            "active_limit": active_limit,
            "requested_N": requested or len(ids),
            "actual_N": len(ids),
            "restore": "fresh process + fresh real prefill + allocator-resident GPU KV",
        }
    )
    write_once(directory / "execution-manifest.json", value)
    return directory / "execution-manifest.json"


def capacity_plan(root, probe_directories):
    from specrhythm.serving.s2_results import verify_result

    ranks = []
    for mode in MODES:
        directory = probe_directories[mode]
        require(
            verify_result(directory)["valid"] is True,
            "S2 capacity probe did not cleanly complete",
            mode=mode,
        )
        ranks.extend(read_json(directory / "actual-capacity.json")["ranks"])
    _, selected = preparation(root)
    plan = freeze_sizes(source_rows(root), selected, ranks)
    destination = root / "capacity-plan.json"
    if destination.exists():
        require(read_json(destination) == plan, "S2 capacity changed after freeze; use a new root")
    else:
        write_once(destination, plan)
    return plan


def freeze(root):
    prep, selection = preparation(root)
    plan, policy = read_json(root / "capacity-plan.json"), read_json(root / "slo-policy.json")
    check_seal(plan)
    check_seal(policy)
    validate_execution_files(root, prep["execution"])
    specs = {"G1": [("smoke", 10, selection["request_ids"][:10], 0.25)]}
    for gate, size in (("G2", "small"), ("G3", "large")):
        chosen = plan["sizes"][size]
        specs[gate] = [
            (size, chosen["requested_N"], chosen["request_ids"], rate) for rate in RATES
        ]
        if gate == "G3" and not chosen["distinct_scale"]:
            specs[gate] = []
    gates = {}
    for gate, units in specs.items():
        gates[gate] = []
        for scale, requested, ids, rate in units:
            for seeds in prep["seed_pairs"]:
                a, o = seeds["arrival_seed"], seeds["order_seed"]
                directory = root / "inputs" / f"{gate}-{scale}-qps{rate:g}-a{a}-o{o}"
                path = directory / "execution-manifest.json"
                if path.exists():
                    from specrhythm.serving.runtime_profile import load_s2

                    old, _ = load_s2(str(path))
                    require(
                        old["request_ids"] == ids
                        and old["requested_N"] == requested
                        and old["execution"] == prep["execution"]
                        and old["trace"] == poisson_trace(ids, rate, a, o),
                        "S2 partial G0 inputs changed",
                    )
                else:
                    path = execution_manifest(
                        root,
                        directory,
                        ids,
                        rate=rate,
                        requested=requested,
                        arrival_seed=a,
                        order_seed=o,
                    )
                gates[gate].append(
                    {
                        "manifest": str(path.relative_to(root)),
                        "sha256": read_json(path)["sha256"],
                        "modes": list(MODES),
                        "scale": scale,
                        "rate": rate,
                        "arrival_seed": a,
                        "order_seed": o,
                    }
                )
    value = sealed(
        {
            "schema_version": "specrhythm.s2-g0.v1",
            "valid": True,
            "gates": gates,
            "execution": prep["execution"],
            "capacity_sha256": plan["sha256"],
            "slo_sha256": policy["sha256"],
            "large_capacity_limited": not plan["sizes"]["large"]["distinct_scale"],
            "actual_residency_rechecked_before_every_observation": True,
        }
    )
    if (root / "g0.json").exists():
        require(read_json(root / "g0.json") == value, "S2 frozen G0 changed")
    else:
        write_once(root / "g0.json", value)
    return value


def calibration_ids(root, seed=1670):
    rows = source_rows(root, "calibration")
    ids = []
    for task in ("chat", "code", "reasoning", "summarization"):
        ids.extend(
            r.request_id
            for r in sorted(
                [r for r in rows if r.task_class == task],
                key=lambda r: (digest([seed, r.request_id]), r.request_id),
            )[:5]
        )
    require(
        len(ids) == 20 and len(set(ids)) == 20,
        "S2 calibration requires five unique requests per class",
    )
    return ids
