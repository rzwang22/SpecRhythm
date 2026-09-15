"""K3 logical candidates versus the unchanged conservative K4 capacity interface."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from specrhythm.serving.common import require
from specrhythm.serving.k3 import (
    B16,
    CONFIGURATIONS,
    MODES,
    PARAMETERS,
    configuration_of,
    geometry,
    matches_geometry,
)
from specrhythm.serving.s1_workload import write_once
from specrhythm.serving.s2_plan import capacity_for

SCHEMA = "specrhythm.k3-capacity.v2"
LEGACY_SCHEMA = "specrhythm.k3-capacity.v1"
LEGACY_MINIMUM_RESERVE = 4


def effective_capacity(actual):
    """B64 requires loaded engine limits, in addition to the raw KV/workspace ranks."""
    from specrhythm.serving.k3 import B64

    meta = actual["metadata"]
    if configuration_of(meta) != B64:
        return  # Historical B16 validation and evidence schema are unchanged.
    g = meta["execution_geometry"]
    targets, draft = actual.get("target_effective_by_rank"), actual.get("draft_effective")
    require(isinstance(targets, list) and len(targets) == 2 and isinstance(draft, dict),
            "B64 loaded sequence/query capacity evidence missing")
    workers = actual.get("target_worker_ranks")
    require(isinstance(workers, list) and len(workers) == 2
            and targets == [r.get("s1_effective_capacity") for r in workers],
            "B64 loaded capacity differs from raw worker snapshots")
    prompt_query = max(r["prompt_length"] + 1 for r in actual["capacity_request_budgets"])
    for role, row, batch, query in [
        *(('target', r, g["target_request_ceiling"], 4 * g["target_request_ceiling"])
          for r in targets),
        ('draft', draft, g["draft_physical_batch_ceiling"],
         max(prompt_query, 4 * g["draft_physical_batch_ceiling"])),
    ]:
        require(isinstance(row, dict)
                and type(row.get("max_num_seqs")) is int
                and type(row.get("max_num_batched_tokens")) is int
                and row["max_num_seqs"] >= batch
                and row["max_num_batched_tokens"] >= query,
                "B64 loaded sequence/query capacity insufficient", role=role,
                required_requests=batch, required_query_positions=query, actual=row)


def reservation(mode, role):
    require(mode in MODES and role in ("target", "draft"), "unknown K3 capacity mode/role")
    k, eager = PARAMETERS.get("candidate_length"), PARAMETERS.get("eager_candidate_limit")
    require(type(k) is int and k == 3 and type(eager) is int and eager == 3,
            "K3 capacity requires explicit integer candidate_length/eager_candidate_limit=3")
    look = eager if mode in ("serial-eager-k3", "pingpong-eager-k3") and role == "draft" else 0
    required = k + look
    return dict(candidate_length=k, eager_continuation_candidates=look,
                required_speculative_positions=required,
                reserved_speculative_positions=max(LEGACY_MINIMUM_RESERVE, required),
                legacy_minimum_reserve=LEGACY_MINIMUM_RESERVE)


def budgets(definitions):
    rows = []
    for d in definitions:
        require(isinstance(d.request_id, str) and d.request_id
                and type(d.prompt_length) is int and d.prompt_length > 0
                and type(d.maximum_new_tokens) is int and d.maximum_new_tokens > 0,
                "invalid K3 capacity request budget")
        rows.append(dict(request_id=d.request_id, prompt_length=d.prompt_length,
                         maximum_new_tokens=d.maximum_new_tokens))
    require(rows and len({r["request_id"] for r in rows}) == len(rows),
            "missing/duplicate K3 capacity request budgets")
    return rows


def check(definitions, rank, *, mode, active_limit, metadata, legacy=False):
    require(isinstance(rank, dict) and isinstance(metadata, dict),
            "K3 raw rank capacity/metadata missing", mode=mode)
    require(type(active_limit) is int and active_limit > 0, "invalid K3 active capacity")
    for key in ("block_size", "num_gpu_blocks", "vocab_size"):
        require(type(rank.get(key)) is int and rank[key] > 0,
                "invalid K3 raw rank capacity", field=key, mode=mode)
    require(type(rank.get("free_memory_bytes")) is int and rank["free_memory_bytes"] >= 0,
            "invalid K3 raw rank capacity", field="free_memory_bytes", mode=mode)
    role = rank.get("role")
    policy = reservation(mode, role)
    require(rank.get("mode") == mode and type(rank.get("physical_gpu_id")) is int
            and rank["physical_gpu_id"] in ((0,) if role == "draft" else (1, 2))
            and isinstance(rank.get("gpu_uuid"), str) and rank["gpu_uuid"],
            "K3 capacity rank binding differs", mode=mode, role=role)
    declarations = metadata.get("speculative_reservations")
    require(isinstance(declarations, dict) and isinstance(declarations.get(role), dict),
            "K3 capacity reservation metadata missing", mode=mode, role=role)
    declared = declarations[role]
    require(all(type(declared.get(k)) is int and declared[k] == v for k, v in policy.items())
            and type(metadata.get("candidate_length")) is int
            and metadata["candidate_length"] == 3
            and type(metadata.get("proposal_budget")) is int and metadata["proposal_budget"] == 3
            and metadata.get("draft_speculative_capacity_tokens")
            == reservation(mode, "draft")["reserved_speculative_positions"],
            "K3 capacity metadata/required/reserved positions differ", mode=mode, role=role)
    configuration = configuration_of(metadata)
    plan = geometry(mode, configuration)
    if configuration != B16:
        require(type(metadata["per_cohort_capacity"]) is int
                and metadata["per_cohort_capacity"] == max(plan["home_capacities"].values())
                and type(metadata["cohort_count"]) is int
                and metadata["cohort_count"] == len(plan["home_capacities"]),
                "B64 capacity home declaration differs")
    declared_geometry = metadata.get("execution_geometry")
    if not legacy or declared_geometry is not None:
        require(matches_geometry(declared_geometry, mode, configuration)
                and active_limit == plan["active_limit"]
                and type(metadata["max_requests_per_target_forward"]) is int
                and metadata["max_requests_per_target_forward"] == plan["target_request_ceiling"],
                "K3 physical batch declaration differs")
        batch = plan["target_request_ceiling"] if role == "target" else plan[
            "draft_physical_batch_ceiling"]
        require(
            type(metadata.get(role + "_sequence_limit")) is int
            and type(metadata.get(role + "_query_token_limit")) is int
            and metadata[role + "_sequence_limit"] >= batch
            and metadata[role + "_query_token_limit"] >= batch * (4 if role == "target" else 1),
            "K3 physical batch/query capacity insufficient",
        )
    budgets(definitions)
    value = capacity_for(definitions, rank, active_limit=active_limit,
                         speculative_tokens=policy["reserved_speculative_positions"])
    # Old fields retain their reserve-based meaning; K3 records them explicitly even
    # at reserve=4, while old modes retain their original sparse/default schema.
    value.update(policy, speculative_capacity_tokens=policy["reserved_speculative_positions"],
                 extra_speculative_tokens=policy["reserved_speculative_positions"] - 4)
    return value


def qualify(actual, mode, definitions=None):
    """Offline replay from original budgets/rank data, never hardware queries."""
    require(actual.get("capacity_schema") in (SCHEMA, LEGACY_SCHEMA),
            "K3 capacity schema missing", mode=mode)
    rows = actual.get("capacity_request_budgets")
    require(isinstance(rows, list) and all(isinstance(r, dict) for r in rows),
            "K3 raw capacity request budgets missing", mode=mode)
    try:
        reconstructed = [SimpleNamespace(**r) for r in rows]
        require(budgets(reconstructed) == rows, "K3 capacity budget fields differ")
    except (AttributeError, TypeError) as error:
        raise ValueError("K3 raw capacity request budget fields missing/invalid") from error
    if definitions is not None:
        require(budgets(definitions) == rows, "K3 capacity budgets differ from frozen workload")
    ranks, checks, meta = actual.get("ranks"), actual.get("checks"), actual.get("metadata")
    require(isinstance(ranks, list) and len(ranks) == 3
            and isinstance(checks, list) and len(checks) == 3 and isinstance(meta, dict),
            "K3 capacity requires three raw ranks/checks and metadata", mode=mode)
    seen = set()
    for rank, recorded in zip(ranks, checks):
        require(isinstance(recorded, dict), "K3 capacity check record missing", mode=mode)
        expected = check(reconstructed, rank, mode=mode,
                         active_limit=meta.get("active_request_limit"), metadata=meta,
                         legacy=actual["capacity_schema"] == LEGACY_SCHEMA)
        for key, value in expected.items():
            require(type(recorded.get(key)) is type(value) and recorded[key] == value,
                    "K3 capacity arithmetic/record differs", mode=mode, role=rank["role"],
                    field=key, expected=value, actual=recorded.get(key))
        require(expected["valid"], "K3 physical capacity insufficient", role=rank["role"])
        seen.add((rank["role"], rank["physical_gpu_id"]))
    require(seen == {("draft", 0), ("target", 1), ("target", 2)},
            "K3 capacity rank missing/duplicated")
    effective_capacity(actual)
    return dict(schema_version=SCHEMA, status="PASS", candidate_length=3,
                speculative_reservations=meta["speculative_reservations"],
                scope="raw rank/budget arithmetic; output and process cleanup are separate")


def preflight(configuration=B16):
    """Static interface exercise only, before loading models; no physical PASS."""
    from specrhythm.serving.fixed_plan import capacity_metadata
    from specrhythm.serving.k3 import configuration_fields

    rows = []
    for mode in MODES:
        active = geometry(mode, configuration)["active_limit"]
        meta = capacity_metadata(mode, active_limit=active, resident_requirement=360,
                                 target_sequence_limit=512, k3_configuration=configuration)
        for role in ("target", "draft"):
            policy = reservation(mode, role)
            # These explicit arithmetic fixtures are not device observations.
            value = check([SimpleNamespace(request_id=str(i), prompt_length=16,
                                           maximum_new_tokens=32) for i in range(360)],
                dict(role=role, mode=mode, block_size=16, num_gpu_blocks=10000,
                     physical_gpu_id=0 if role == "draft" else 1,
                     gpu_uuid="STATIC_ARITHMETIC_FIXTURE_NOT_DEVICE_EVIDENCE", vocab_size=16,
                     free_memory_bytes=1024**3), mode=mode, active_limit=active, metadata=meta)
            require(value["extra_speculative_tokens"] >= 0
                    if "extra_speculative_tokens" in value else True,
                    "negative legacy capacity increment")
            rows.append(dict(mode=mode, role=role,
                             geometry=geometry(mode, configuration), **policy))
    return dict(schema_version=SCHEMA, static_contract="PASS", GPU_capacity="PENDING",
                **configuration_fields(configuration),
                scope="synthetic arithmetic only; no model, device, allocation or UUID query",
                reservations=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--k3-configuration", choices=CONFIGURATIONS, default=B16)
    args = parser.parse_args()
    result = preflight(args.k3_configuration)
    write_once(args.output, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
