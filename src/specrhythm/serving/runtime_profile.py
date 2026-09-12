"""Dispatch resident input contracts; absence of S2 delegates unchanged to S1/legacy."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from specrhythm.phase4.manifest import sha256_file
from specrhythm.serving import s1_workload as legacy
from specrhythm.serving.common import read_json, require
from specrhythm.serving.s2_plan import check_seal
from specrhythm.serving.schema import load_requests

PROFILE_ENV = "SR_S2_EXECUTION_MANIFEST"


def s2_enabled():
    return bool(os.environ.get(PROFILE_ENV))


def s1_enabled():
    """Historical internal name: per-request serving budgets/EOS apply to S1 and S2."""
    return s2_enabled() or legacy.s1_enabled()


def load_s2(path):
    path = Path(path)
    value = read_json(path)
    check_seal(value)
    require(value["schema_version"] == "specrhythm.s2-execution.v1", "wrong S2 execution schema")
    workload = path.parent / value["workload_file"]
    require(workload.resolve().parent == path.resolve().parent, "unsafe S2 workload path")
    require(sha256_file(workload) == value["workload_sha256"], "S2 workload hash mismatch")
    rows = load_requests(workload)
    require([r.request_id for r in rows] == value["request_ids"], "S2 request identity mismatch")
    for row in rows:
        require(
            row.prompt_length + row.maximum_new_tokens + 4 <= 4096,
            "S2 context including speculation exceeds frozen capacity",
            request_id=row.request_id,
        )
    return value, tuple(legacy.ResidentServingRequest(r) for r in rows)


@lru_cache(maxsize=16)
def _cached_profile(path):
    return load_s2(path)


def active_profile():
    if s2_enabled():
        return _cached_profile(os.environ[PROFILE_ENV])
    return legacy.active_profile()


def load_runtime_requests(path, expected_count, **kwargs):
    if not s2_enabled():
        return legacy.load_runtime_requests(path, expected_count, **kwargs)
    value, rows = active_profile()
    require(
        len(rows) == expected_count and sha256_file(path) == value["workload_sha256"],
        "S2 runtime workload/count mismatch",
    )
    return rows


def target_options():
    if not s2_enabled():
        return legacy.target_options()
    from specrhythm.serving.s2_plan import POOL_SLOTS, QUERY_LIMIT

    return dict(max_num_seqs=POOL_SLOTS, max_num_batched_tokens=QUERY_LIMIT)


def setup_terminal_ids(manifest):
    if not s2_enabled():
        return legacy.setup_terminal_ids(manifest)
    value, rows = active_profile()
    budgets = {r.request_id: r.maximum_new_tokens for r in rows}
    return {
        r.request_id
        for r in manifest.requests
        if budgets[r.request_id] == 1
        or r.bootstrap_token_id in value["execution"]["eos_token_ids"]
    }


def initial_proposal_excluded_ids(manifest):
    if not s2_enabled():
        return legacy.initial_proposal_excluded_ids(manifest)
    # S2 defers every initial proposal until arrival/admission.
    return {r.request_id for r in manifest.requests}


def initial_target_tail(request_id, output_count):
    if not s2_enabled():
        return legacy.initial_target_tail(request_id, output_count)
    return output_count == 1 and any(
        r.request_id == request_id and r.maximum_new_tokens == 2 for r in active_profile()[1]
    )


# These are used only by existing entry points; their behavior remains unchanged.
require_reference_or_s1 = legacy.require_reference_or_s1
pending_reference_comparison = legacy.pending_reference_comparison
