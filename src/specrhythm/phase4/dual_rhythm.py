"""CPU-only, immutable burst assignment for the isolated ping-pong baseline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import MappingProxyType

from specrhythm.phase4.batched_draft_service import write_immutable_report
from specrhythm.phase4.manifest import sha256_file
from specrhythm.phase4.stock_vllm import load_smoke_requests

SELECTOR = "SR_PHASE4B_DUAL_RHYTHM"
MANIFEST = "SR_PHASE4_DUAL_RHYTHM_MANIFEST"
COHORTS = ("A", "B")
LEGACY_CLASSES = (
    "specrhythm.phase4.vllm_dual_scheduler.DualBatchScheduler",
    "specrhythm.phase4.vllm_dual.DualBatchRemoteProposer",
)
PINGPONG_CLASSES = (
    "specrhythm.phase4.vllm_pingpong_scheduler.PingPongScheduler",
    "specrhythm.phase4.vllm_pingpong.PingPongRemoteProposer",
)


def selected_rhythm(environ=None):
    value = (os.environ if environ is None else environ).get(SELECTOR, "legacy")
    if value not in ("legacy", "pingpong"):
        raise ValueError(f"{SELECTOR} must be legacy or pingpong")
    return value


def balanced_assignment(request_ids):
    ids = tuple(request_ids)
    if (
        len(ids) < 2
        or len(set(ids)) != len(ids)
        or any(not isinstance(r, str) or not r.strip() for r in ids)
    ):
        raise ValueError("pingpong requires at least two unique frozen request IDs")
    return MappingProxyType({rid: COHORTS[i % 2] for i, rid in enumerate(ids)})


def write_assignment(path, workload, count):
    rows = load_smoke_requests(workload, count, require_task_mixture=count in (5, 100))
    assignment = balanced_assignment(r.request_id for r in rows)
    write_immutable_report(
        path,
        {
            "schema_version": "specrhythm.phase4b4b-rhythm.v1",
            "dual_rhythm": "pingpong",
            "workload_sha256": sha256_file(workload),
            "request_count": count,
            "request_ids": list(assignment),
            "assignment": dict(assignment),
            "assignment_policy": "frozen workload order alternating A/B; no transfers or arrivals",
            "initial_sizes": {c: list(assignment.values()).count(c) for c in COHORTS},
            "readiness_capacity": count,
            "scheduler_class": PINGPONG_CLASSES[0],
            "proposer_class": PINGPONG_CLASSES[1],
        },
    )


def load_assignment(path=None, *, workload=None, count=None):
    path = path or os.environ.get(MANIFEST)
    if not path:
        raise ValueError("pingpong requires the immutable startup assignment manifest")
    value = json.loads(Path(path).read_text())
    assignment = value["assignment"]
    # Preserve frozen order explicitly even when the report writer sorts object keys.
    expected = balanced_assignment(value["request_ids"])
    if (
        value.get("dual_rhythm") != "pingpong"
        or assignment != expected
        or value.get("request_count") != len(expected)
        or value.get("readiness_capacity") != len(expected)
    ):
        raise ValueError("invalid or unbalanced pingpong assignment")
    if count is not None and count != len(expected):
        raise ValueError("pingpong assignment request count mismatch")
    if workload is not None:
        rows = load_smoke_requests(
            workload, len(expected), require_task_mixture=len(expected) in (5, 100)
        )
        if sha256_file(workload) != value.get("workload_sha256") or list(expected) != [
            r.request_id for r in rows
        ]:
            raise ValueError("pingpong assignment differs from frozen workload")
    return expected


def cohort_for(assignment, request_ids):
    ids = tuple(request_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("empty or duplicated cohort work")
    try:
        cohorts = {assignment[rid] for rid in ids}
    except KeyError as error:
        raise ValueError("unknown request in pingpong work") from error
    if len(cohorts) != 1:
        raise ValueError("one execution unit cannot mix logical cohorts")
    return cohorts.pop()


class AnnotatedLog:
    """Decorate an existing append, with no extra checkpoint or fsync."""

    def __init__(self, log, annotate):
        self.log, self.annotate = log, annotate

    def read(self):
        return self.log.read()

    def append(self, row):
        self.log.append({**row, **self.annotate(row)})


class CohortDraftClient:
    """Split only the initial enqueue; commit submissions must already be one cohort."""

    def __init__(self, client, assignment):
        self.client, self.assignment = client, assignment

    def call(self, operation, payload):
        if operation != "enqueue":
            return self.client.call(operation, payload)
        rows = payload["rows"]
        ids = [r["request_id"] for r in rows]
        if len(set(ids)) != len(ids) or any(r not in self.assignment for r in ids):
            raise ValueError("invalid pingpong enqueue identity")
        if payload["work_operation"] == "propose_only":
            groups = [[r for r in rows if self.assignment[r["request_id"]] == c] for c in COHORTS]
        else:
            cohort_for(self.assignment, ids)
            groups = [rows]
        responses = []
        for group in groups:
            if group:
                responses.append(
                    self.client.call(
                        operation,
                        {
                            **payload,
                            "rows": [
                                {**r, "logical_cohort": self.assignment[r["request_id"]]}
                                for r in group
                            ],
                        },
                    )
                )
        return {"cohort_enqueues": responses}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workload", type=Path)
    parser.add_argument("--request-count", type=int)
    args = parser.parse_args()
    mode = selected_rhythm()
    if mode == "pingpong" and os.environ.get("SR_PHASE4_DRAFT_BACKEND") != "vllm-batched":
        parser.error("pingpong requires production vllm-batched Draft")
    if args.output:
        if mode != "pingpong" or args.workload is None or args.request_count is None:
            parser.error("assignment requires pingpong, workload and request count")
        write_assignment(args.output, args.workload, args.request_count)
    else:
        print(mode)


if __name__ == "__main__":
    main()
