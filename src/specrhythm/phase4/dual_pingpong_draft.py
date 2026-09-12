"""Cohort checks and in-memory evidence around the qualified production Draft machine."""

from __future__ import annotations

import time

from specrhythm.phase4.batched_draft_service import write_immutable_report
from specrhythm.phase4.dual_batched_draft import BatchedDualDraftMachine
from specrhythm.phase4.dual_rhythm import COHORTS, cohort_for, load_assignment
from specrhythm.phase4.manifest import sha256_file


class PingPongDraftMachine(BatchedDualDraftMachine):
    def __init__(self, backend, *, candidate_budget=4, report_path=None, assignment=None):
        super().__init__(backend, candidate_budget=candidate_budget, report_path=None)
        self.pingpong_report_path = report_path
        self.assignment = load_assignment() if assignment is None else assignment
        self.work_records = []

    def initialize(self, row):
        cohort_for(self.assignment, [row["request_id"]])
        return super().initialize(row)

    def execute_batch(self, operation, rows):
        cohort = cohort_for(self.assignment, [r["request_id"] for r in rows])
        if any(r.get("logical_cohort") != cohort for r in rows):
            raise ValueError("Draft enqueue changed a stable logical cohort")
        if operation not in ("propose_only", "commit_and_propose", "finish_tail"):
            raise ValueError("pingpong only supports the qualified resident decode path")
        metrics = self.backend.metrics
        before = {p: dict(metrics.batches[p]) for p in ("proposal", "commit")}
        started = time.monotonic_ns()
        results = super().execute_batch(operation, rows)
        ended = time.monotonic_ns()
        self.work_records.append(
            {
                "logical_cohort": cohort,
                "operation": operation,
                "host_start_ns": started,
                "host_end_ns": ended,
                "rows": [dict(r) for r in rows],
                "forward_batch_histograms": {
                    p: {
                        str(n): count - before[p].get(n, 0)
                        for n, count in metrics.batches[p].items()
                        if count > before[p].get(n, 0)
                    }
                    for p in before
                },
            }
        )
        return [{**r, "logical_cohort": cohort, "dual_rhythm": "pingpong"} for r in results]

    def shutdown(self):
        result = super().shutdown()
        if self.pingpong_report_path is not None:
            write_immutable_report(
                self.pingpong_report_path,
                {
                    **self.backend.report(),
                    "dual_rhythm": "pingpong",
                    "assignment": dict(self.assignment),
                    "initial_cohort_sizes": {
                        c: list(self.assignment.values()).count(c) for c in COHORTS
                    },
                    "dual_cohorts": [
                        {"operation": op, "request_count": size, "count": count}
                        for (op, size), count in sorted(self.cohorts.items())
                    ],
                    "cohort_policy": "persistent A/B; asynchronous owner; no accumulation",
                    "pingpong_work_records": self.work_records,
                },
            )
            result["draft_backend_report_sha256"] = sha256_file(self.pingpong_report_path)
        return result
