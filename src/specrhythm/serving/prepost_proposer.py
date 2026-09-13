"""Explicit no-bonus variant; transport hooks and barriers retain their boundaries."""

from specrhythm.continuation.prepost import PROTOCOL, acceptance
from specrhythm.serving.eager_proposer import EagerSerialProposer


class PrePostProposer(EagerSerialProposer):
    acceptance_rule = staticmethod(acceptance)
    protocol_metadata = {"prepost_protocol": PROTOCOL, "target_bonus_is_committed": False}

    def _gpu_qualification_report(self):
        return {
            "gpu_correctness_result": False,
            "gpu_performance_result": False,
            "prepost_qualification": "PENDING",
            "prepost_protocol": PROTOCOL,
        }
