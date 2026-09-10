"""S2 stage dependencies; terminal drain is an evidenced operation, not a name exemption."""

from specrhythm.serving.common import require
from specrhythm.serving.s1_results import interval_duration


def _tail_evidence(work, backend, runtime, states, drafts, workers, check):
    evidence = work.get("terminal_drain")
    check(isinstance(evidence, dict), "S2 terminal drain evidence missing", "terminal receipt")
    ids = set(work["request_ids"])
    rows = evidence.get("rows", [])
    results = evidence.get("results", [])
    release = evidence.get("release", {})
    check(
        evidence.get("schema_version") == "specrhythm.s2-terminal-drain.v1"
        and {r["request_id"] for r in rows} == ids
        and len(rows) == len(ids) == len(results)
        and {r["request_id"] for r in results} == ids
        and set(evidence.get("retired_request_ids", [])) == ids
        and set(release.get("request_ids", [])) == ids,
        "S2 terminal drain request/release scope differs",
        "same unique request set throughout dispatch/materialization/release",
    )
    check(
        evidence.get("proposal_forward_count") == 0
        and all(r.get("terminal") is True for r in rows)
        and all(
            r.get("terminal") is True
            and r.get("target_tail") is True
            and "proposal" in r
            and r["proposal"] is None
            for r in results
        ),
        "S2 terminal drain is nonterminal or generates a proposal",
        "terminal input/result, no proposal and zero proposal forwards",
    )
    provenance = backend["provenance"]
    check(
        len(workers) == 2
        and evidence.get("physical_gpu_id") == provenance.get("physical_gpu_id")
        and evidence.get("gpu_uuid") == provenance.get("gpu_uuid")
        and bool(evidence.get("gpu_uuid"))
        and all(
            w.get("gpu_uuid")
            and w["gpu_uuid"] != evidence["gpu_uuid"]
            and w.get("physical_gpu_id") != evidence["physical_gpu_id"]
            for w in workers
        ),
        "S2 terminal drain lacks independent Draft/Target device binding",
        "actual Draft device/UUID distinct from both Target workers",
    )
    unrelated = set(evidence.get("unrelated_request_ids", []))
    check(
        not ids & unrelated
        and release.get("private_kv_checked_requests") == len(ids) + len(unrelated)
        and set(release.get("materialized", {})) == ids
        and evidence.get("unrelated_before_sha256")
        == evidence.get("unrelated_after_sha256")
        == release.get("unrelated_before_release_sha256")
        and bool(evidence.get("unrelated_before_sha256")),
        "S2 terminal drain KV ownership/isolation evidence differs",
        "private blocks checked before release; unrelated prefixes/blocks unchanged",
    )
    start, end = work["host_start_ns"], work["host_end_ns"]
    released = release.get("resources_released_ns")
    check(
        type(released) is int and start <= released <= end <= runtime["end_ns"],
        "S2 terminal drain release/work outside observation",
        "work start <= actual KV release <= work end <= drain end",
    )
    raw = {r["request_id"]: r for r in runtime["requests"]}
    completions = []
    for row, result in zip(rows, results):
        rid = row["request_id"]
        before = evidence.get("before", {}).get(rid, {})
        materialized = release["materialized"][rid]
        terminal = [
            s
            for s in states
            if s.get("request_id") == rid and s.get("destination_state") == "TERMINAL"
        ]
        check(
            row["request_id"] == result["request_id"]
            and len(terminal) == 1
            and terminal[0]["timestamp_ns"] <= start
            and before.get("finished") is False
            and before.get("proposal_present") is False
            and row["prefix_version"] == before["prefix_version"] + 1
            and row["prefix_version"] == result["prefix_version"] == terminal[0]["prefix_version"]
            and len(row.get("committed_delta", [])) == 1
            and row["committed_delta"] == result.get("committed_token_ids")
            and row["prefix_token_sha256"]
            == result.get("prefix_token_sha256")
            == terminal[0]["committed_prefix_sha256"]
            == materialized.get("prefix_sha256")
            == result.get("committed_prefix_hash")
            and before["prefix_length"] + 1
            == result.get("materialized_kv_length")
            == materialized.get("materialized_tokens")
            == terminal[0]["committed_prefix_length"]
            and result.get("draft_physical_request_block_observable") is True
            and result.get("draft_physical_request_block_identity")
            == materialized.get("block_ids")
            and released <= result.get("draft_sync_complete_ns", 0) <= end,
            "S2 terminal drain lacks authorized final prefix/terminal materialization",
            "native Target TERMINAL before dispatch; exact next-version one-token final sync",
            "request-state-events.jsonl",
        )
        prior = [
            s
            for s in states
            if s.get("request_id") == rid
            and s.get("prefix_version") == before["prefix_version"]
            and s.get("committed_prefix_sha256") == before["prefix_sha256"]
            and s.get("committed_prefix_length") == before["prefix_length"]
            and s.get("timestamp_ns", end + 1) <= start
        ]
        check(
            bool(prior), "S2 terminal drain parent prefix is not evidenced", "native parent state"
        )
        matches = [
            d
            for d in drafts
            if d.get("operation") == "finish_tail"
            and d.get("request_id") == rid
            and d.get("success") is True
            and d.get("start_ns", end + 1) <= start <= end <= d.get("end_ns", 0)
            and all(d.get("result", {}).get(k) == v for k, v in result.items())
        ]
        check(
            len(matches) == 1,
            "S2 terminal drain lacks matching successful owner dispatch",
            "one successful physical finish_tail work result",
            "draft-work-events.jsonl",
        )
        check(
            not any(
                rid in w["request_ids"] and w["host_start_ns"] >= end
                for w in backend["s2_work_records"]
                if w is not work
            ),
            "S2 Draft work after terminal drain",
            "no subsequent proposal/commit for retired request",
        )
        slot = [
            e["timestamp_ns"]
            for e in runtime["events"]
            if e.get("event") == "resources-released" and e.get("request_id") == rid
        ]
        check(
            len(slot) == 1
            and raw[rid]["state"] == "FINISHED"
            and raw[rid]["resources_released"] is True
            and raw[rid]["completion_ns"] <= slot[0]
            and end <= slot[0] <= runtime["end_ns"],
            "S2 active slot released before terminal drain completed",
            "actual completion and whole Draft work precede slot release; release precedes drain",
            "runtime.json",
        )
        completions.append(
            {
                "request_id": rid,
                "token_completion_ns": raw[rid]["completion_ns"],
                "draft_resources_released_ns": released,
                "active_slot_released_ns": slot[0],
            }
        )
    return completions


def stage_overlap(backend, forwards, runtime, states, drafts, workers, directory):
    """Keep legacy opposite-cohort host union; report drain separately from proposal CUDA pairs."""
    raw = {r["request_id"]: r for r in runtime["requests"]}
    cross, active, drain, same_drain, drain_work = [], [], [], [], []
    drain_pairs, completions = [], []
    paths = {
        name: str(directory / name)
        for name in (
            "draft-backend-report.json",
            "draft-work-events.jsonl",
            "request-state-events.jsonl",
            "target-diagnostics.jsonl",
            "runtime.json",
            "actual-capacity.json",
        )
    }
    for work in backend.get("s2_work_records", []):
        ids = set(work["request_ids"])
        pairs = []
        for f in forwards:
            a, b = max(work["host_start_ns"], f["start_ns"]), min(work["host_end_ns"], f["end_ns"])
            if b > a:
                pairs.append((f, a, b))
        context = {
            "operation": work["operation"],
            "terminal_by_request": work.get("terminal_by_request"),
            "terminal_evidence": work.get("terminal_drain"),
            "draft_request_ids": work["request_ids"],
            "draft_cohort": work["logical_cohort"],
            "draft_host_interval_ns": [work["host_start_ns"], work["host_end_ns"]],
            "target_pairs": [
                {
                    "target_request_ids": f["request_ids"],
                    "target_kind": f["kind"],
                    "target_cohorts": {r: raw[r]["cohort"] for r in f["request_ids"]},
                    "target_host_interval_ns": [f["start_ns"], f["end_ns"]],
                    "same_request_ids": sorted(ids & set(f["request_ids"])),
                    "host_overlap_ms": (b - a) / 1e6,
                }
                for f, a, b in pairs
            ],
        }

        def check(ok, message, expected, artifact="draft-backend-report.json", context=context):
            require(
                ok,
                message,
                field="s2_stage_dependency",
                expected=expected,
                actual=context,
                artifact=paths[artifact],
                artifacts=paths,
            )

        check(
            bool(ids)
            and len(ids) == len(work["request_ids"])
            and ids <= raw.keys()
            and all(raw[r]["cohort"] == work["logical_cohort"] for r in ids)
            and runtime["start_ns"]
            <= work["host_start_ns"]
            < work["host_end_ns"]
            <= runtime["end_ns"],
            "S2 Draft work identity/cohort/observation interval invalid",
            "admitted measured work",
        )
        for f, _a, _b in pairs:
            check(
                not ids & set(f["request_ids"]),
                "S2 concurrent Draft/Target ownership collision",
                "disjoint request sets for every operation",
            )
        terminal = work["operation"] == "finish_tail" and "terminal_drain" in work
        if terminal:
            try:
                completions.extend(
                    _tail_evidence(work, backend, runtime, states, drafts, workers, check)
                )
            except (KeyError, TypeError, IndexError):
                check(
                    False,
                    "S2 terminal drain evidence incomplete/malformed",
                    "complete native terminal, KV materialization and release evidence",
                )
            drain_work.append((work["host_start_ns"], work["host_end_ns"]))
        for index, (f, a, b) in enumerate(pairs):
            same = any(raw[r]["cohort"] == work["logical_cohort"] for r in f["request_ids"])
            check(
                not same or (
                    terminal and f["kind"] == "verify"
                    and len({raw[r]["cohort"] for r in f["request_ids"]}) == 1
                ),
                "S2 same-cohort stages overlap",
                "active Draft requires opposite cohort; same-cohort verification "
                "needs validated terminal drain",
            )
            if not same:
                cross.append((a, b))
                if not terminal:
                    active.append((a, b))
            if terminal:
                check(
                    set(f["request_ids"]) <= set(work["terminal_drain"]["unrelated_request_ids"]),
                    "S2 terminal drain lacks the concurrent requests in its independent KV scope",
                    "Target request prefixes/blocks covered by the unchanged unrelated Draft pool",
                )
                drain.append((a, b))
                if same:
                    same_drain.append((a, b))
                drain_pairs.append({**context, "target_pairs": [context["target_pairs"][index]]})
    return {
        "stage_dependency_contract": "specrhythm.s2-terminal-drain.v1",
        "stage_host_overlap_ms": interval_duration(cross) / 1e6,
        "active_stage_host_overlap_ms": interval_duration(active) / 1e6,
        "terminal_drain": {
            "work_count": len(drain_work),
            "work_host_ms": interval_duration(drain_work) / 1e6,
            "host_overlap_ms": interval_duration(drain) / 1e6,
            "same_cohort_host_overlap_ms": interval_duration(same_drain) / 1e6,
            "pair_count": len(drain_pairs),
            "pairs": drain_pairs,
            "completions": completions,
            "definition": (
                "synchronized host envelopes including final KV sync/release; "
                "not exact GPU kernel overlap"
            ),
        },
        "overlap_definition": (
            "legacy union of opposite-cohort Draft/Target host envelopes "
            "(including cross-cohort drain); "
            "active-only and terminal drain reported separately; not additive or kernel self time"
        ),
    }
