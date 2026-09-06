"""Ping-pong submission adapter; Target verification/acceptance remain inherited."""

from __future__ import annotations

import os
from pathlib import Path

from specrhythm.phase4.dual_rhythm import (
    PINGPONG_CLASSES,
    AnnotatedLog,
    CohortDraftClient,
    cohort_for,
    load_assignment,
)
from specrhythm.phase4.vllm_dual import DualBatchRemoteProposer


class PingPongRemoteProposer(DualBatchRemoteProposer):
    def __init__(self, vllm_config):
        self.rhythm_report_fields = {
            "dual_rhythm": "pingpong",
            "ready_proposal_injection": PINGPONG_CLASSES[0],
        }
        super().__init__(vllm_config)
        self.assignment = load_assignment(
            workload=Path(os.environ["SR_PHASE4_WORKLOAD"]), count=len(self.definitions)
        )
        self.client = CohortDraftClient(self.client, self.assignment)
        self.proposal_log = AnnotatedLog(self.proposal_log, self._cohort_evidence)
        self.verification_log = AnnotatedLog(self.verification_log, self._cohort_evidence)

    def _cohort_evidence(self, row):
        ids = row.get("verify_request_ids", [row["request_id"]])
        return {"dual_rhythm": "pingpong", "logical_cohort": cohort_for(self.assignment, ids)}
