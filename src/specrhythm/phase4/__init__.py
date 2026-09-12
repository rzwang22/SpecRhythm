"""Isolated Phase-4 serving-engine integration contracts.

This package deliberately has no import-time dependency on vLLM, PyTorch, or
the simulator.  GPU-only imports live in :mod:`specrhythm.phase4.stock_vllm`.
"""

from specrhythm.phase4.contracts import (
    CandidateBatch,
    CandidateNode,
    DraftEngineAdapter,
    EngineEvent,
    GreedySamplingContract,
    RequestState,
    TargetEngineAdapter,
    VerificationBatch,
    VerificationResult,
)

__all__ = [
    "CandidateBatch",
    "CandidateNode",
    "DraftEngineAdapter",
    "EngineEvent",
    "GreedySamplingContract",
    "RequestState",
    "TargetEngineAdapter",
    "VerificationBatch",
    "VerificationResult",
]
