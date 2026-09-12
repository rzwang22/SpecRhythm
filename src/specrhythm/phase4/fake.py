"""Explicitly fake adapters used only to test Phase-4 contracts on CPU."""

from __future__ import annotations

from specrhythm.phase4.contracts import (
    CandidateBatch,
    CandidateNode,
    MonotonicEventClock,
    RequestState,
    VerificationBatch,
    VerificationResult,
)


class FakeDraftEngineAdapter:
    backend_name = "fake-contract-test-only"
    fake_backend = True

    def __init__(self) -> None:
        self.clock = MonotonicEventClock()

    def propose(self, request: RequestState, candidate_budget: int) -> CandidateBatch:
        if request.finished:
            raise ValueError("cannot draft for a finished fake request")
        if candidate_budget < 1:
            raise ValueError("candidate_budget must be positive")
        nodes = []
        parent = None
        for index in range(candidate_budget):
            node_id = f"{request.request_id}:{request.prefix_epoch}:{index}"
            nodes.append(
                CandidateNode(
                    stable_node_id=node_id,
                    parent_id=parent,
                    depth=index + 1,
                    token_id=(request.prefix_epoch + index + 1) % 32000,
                    local_probability=0.75,
                    path_probability=0.75 ** (index + 1),
                )
            )
            parent = node_id
        event = self.clock.next(request.request_id, "draft", "fake-candidates")
        return CandidateBatch(
            request_id=request.request_id,
            prefix_epoch=request.prefix_epoch,
            committed_prefix_token_ids=request.committed_token_ids,
            nodes=tuple(nodes),
            created_monotonic_ns=event.monotonic_ns,
            adapter_backend=self.backend_name,
            fake_data=True,
        )


class FakeTargetEngineAdapter:
    backend_name = "fake-contract-test-only"
    fake_backend = True

    def __init__(self) -> None:
        self.clock = MonotonicEventClock()

    def verify(self, request: RequestState, batch: VerificationBatch) -> VerificationResult:
        if request.finished:
            raise ValueError("cannot verify a finished fake request")
        if batch.request_id != request.request_id or batch.prefix_epoch != request.prefix_epoch:
            raise ValueError("fake verification batch does not match request state")
        by_id = {node.stable_node_id: node for node in batch.candidate_nodes}
        accepted = batch.selected_node_ids[:1]
        tokens = tuple(by_id[node_id].token_id for node_id in accepted)
        event = self.clock.next(request.request_id, "target", "fake-verification")
        return VerificationResult(
            request_id=request.request_id,
            prefix_epoch=request.prefix_epoch,
            verified_node_ids=batch.selected_node_ids,
            accepted_node_ids=accepted,
            committed_token_ids=tokens,
            target_bonus_token_ids=(),
            finished=False,
            completed_monotonic_ns=event.monotonic_ns,
            target_backend=self.backend_name,
            fake_data=True,
        )


def run_fake_contract() -> dict[str, object]:
    """Exercise the protocol without presenting the output as a GPU result."""

    request = RequestState("fake-request", (1, 2, 3))
    draft = FakeDraftEngineAdapter()
    target = FakeTargetEngineAdapter()
    candidates = draft.propose(request, 3)
    verification = VerificationBatch.from_candidates(
        candidates,
        [node.stable_node_id for node in candidates.nodes[:2]],
        candidates.created_monotonic_ns + 1,
    )
    result = target.verify(request, verification)
    updated = request.apply(result, 0)
    return {
        "schema_version": "specrhythm.phase4-contract-dry-run.v1",
        "backend": "fake-contract-test-only",
        "fake_data": True,
        "gpu_result": False,
        "serving_performance_result": False,
        "drafted_nodes": candidates.drafted_nodes,
        "accounting": result.accounting,
        "updated_prefix_epoch": updated.prefix_epoch,
    }
