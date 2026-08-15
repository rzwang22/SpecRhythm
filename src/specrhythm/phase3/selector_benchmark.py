"""Testable selector-stage contract without fake GPU timing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

SELECTOR_STAGES = (
    "tree_construction",
    "prefix_closed_pruning",
    "stable_node_bookkeeping",
    "cpu_scheduler",
    "gpu_cpu_synchronization",
)


@dataclass(frozen=True)
class SelectorNode:
    stable_node_id: str
    request_id: str
    parent_id: Optional[str]
    depth: int
    score: float


class SelectorBenchmarkAdapter(Protocol):
    """Future real selector adapters must expose all five separately timed stages."""

    name: str

    def tree_construction(
        self, request_count: int, search_pool_size: int
    ) -> list[SelectorNode]: ...

    def prefix_closed_pruning(
        self, nodes: list[SelectorNode], candidate_budget: int
    ) -> list[SelectorNode]: ...

    def stable_node_bookkeeping(self, nodes: list[SelectorNode]) -> dict[str, int]: ...

    def cpu_scheduler(self, nodes: list[SelectorNode]) -> list[str]: ...

    def gpu_cpu_synchronization(self) -> None: ...


class DryRunSelectorAdapter:
    """Dependency-free lifecycle adapter; it deliberately records no latency."""

    name = "dry_run_prefix_closed_selector"

    def tree_construction(
        self, request_count: int, search_pool_size: int
    ) -> list[SelectorNode]:
        nodes = []
        for request_index in range(request_count):
            parent = None
            for depth in range(1, search_pool_size + 1):
                node_id = f"r{request_index}:n{depth}"
                nodes.append(
                    SelectorNode(
                        stable_node_id=node_id,
                        request_id=f"r{request_index}",
                        parent_id=parent,
                        depth=depth,
                        score=1.0 / depth,
                    )
                )
                parent = node_id
        return nodes

    def prefix_closed_pruning(
        self, nodes: list[SelectorNode], candidate_budget: int
    ) -> list[SelectorNode]:
        by_request: dict[str, list[SelectorNode]] = {}
        for node in nodes:
            by_request.setdefault(node.request_id, []).append(node)
        selected = []
        depth = 0
        while len(selected) < candidate_budget:
            made_progress = False
            for request_id in sorted(by_request):
                request_nodes = by_request[request_id]
                if depth < len(request_nodes) and len(selected) < candidate_budget:
                    selected.append(request_nodes[depth])
                    made_progress = True
            if not made_progress:
                break
            depth += 1
        return selected

    def stable_node_bookkeeping(self, nodes: list[SelectorNode]) -> dict[str, int]:
        return {node.stable_node_id: index for index, node in enumerate(nodes)}

    def cpu_scheduler(self, nodes: list[SelectorNode]) -> list[str]:
        ordered = sorted(nodes, key=lambda node: (-node.score, node.stable_node_id))
        return [node.stable_node_id for node in ordered]

    def gpu_cpu_synchronization(self) -> None:
        return None


def _is_prefix_closed(nodes: list[SelectorNode]) -> bool:
    selected = {node.stable_node_id for node in nodes}
    return all(node.parent_id is None or node.parent_id in selected for node in nodes)


def run_selector_dry_run(
    *, request_count: int, search_pool_size: int, candidate_budget: int
) -> dict[str, Any]:
    if request_count < 1 or search_pool_size < 1 or candidate_budget < 1:
        raise ValueError("selector dry-run dimensions must be positive")
    if candidate_budget > request_count * search_pool_size:
        raise ValueError("candidate budget exceeds the dry-run search forest")
    adapter: SelectorBenchmarkAdapter = DryRunSelectorAdapter()
    forest = adapter.tree_construction(request_count, search_pool_size)
    selected = adapter.prefix_closed_pruning(forest, candidate_budget)
    bookkeeping = adapter.stable_node_bookkeeping(selected)
    schedule = adapter.cpu_scheduler(selected)
    adapter.gpu_cpu_synchronization()
    prefix_closed = _is_prefix_closed(selected)
    if not prefix_closed:
        raise AssertionError("selector dry-run produced a non-prefix-closed selection")
    return {
        "schema_version": "specrhythm.selector-benchmark-interface.v1",
        "adapter": adapter.name,
        "gpu_measurement": False,
        "latency_samples_recorded": False,
        "synthetic_latency_used": False,
        "stages": [
            {"name": stage, "exercised": True, "timed": False}
            for stage in SELECTOR_STAGES
        ],
        "request_count": request_count,
        "search_pool_size_per_request": search_pool_size,
        "candidate_budget": candidate_budget,
        "forest_node_count": len(forest),
        "selected_node_count": len(selected),
        "prefix_closed": prefix_closed,
        "stable_node_count": len(bookkeeping),
        "scheduled_node_count": len(schedule),
        "selected_stable_node_ids": [node.stable_node_id for node in selected],
    }
