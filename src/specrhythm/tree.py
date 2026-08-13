"""Deterministic candidate-tree primitives for control-plane simulation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Iterable, Optional


@dataclass(frozen=True)
class CandidateTreeNode:
    node_id: str
    parent_id: Optional[str]
    depth: int
    token_confidence: float
    path_probability: float

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("tree node_id must not be empty")
        if not isinstance(self.depth, int) or isinstance(self.depth, bool) or self.depth < 0:
            raise ValueError("tree depth must be a non-negative integer")
        for value in (self.token_confidence, self.path_probability):
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("tree probabilities must be finite numbers")
            if not 0 <= value <= 1:
                raise ValueError("tree probabilities must be in [0, 1]")


@dataclass(frozen=True)
class CandidateTree:
    request_id: str
    root: CandidateTreeNode
    nodes: tuple[CandidateTreeNode, ...]
    _by_id: dict[str, CandidateTreeNode] = field(
        init=False, repr=False, compare=False, hash=False
    )
    _children_by_parent: dict[str, tuple[CandidateTreeNode, ...]] = field(
        init=False, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("candidate tree request_id must not be empty")
        if self.root.parent_id is not None or self.root.depth != 0:
            raise ValueError("candidate tree root must have no parent and depth zero")
        by_id = {node.node_id: node for node in self.nodes}
        if len(by_id) != len(self.nodes) or by_id.get(self.root.node_id) != self.root:
            raise ValueError("candidate tree nodes must uniquely contain the root")
        for node in self.nodes:
            if node is self.root:
                continue
            parent = by_id.get(node.parent_id or "")
            if parent is None or node.depth != parent.depth + 1:
                raise ValueError("candidate tree node has an invalid parent or depth")
            if node.path_probability > parent.path_probability + 1e-12:
                raise ValueError("child path probability cannot exceed its parent")
        children: dict[str, list[CandidateTreeNode]] = {}
        for node in self.candidate_nodes:
            children.setdefault(node.parent_id or "", []).append(node)
        object.__setattr__(self, "_by_id", by_id)
        object.__setattr__(
            self,
            "_children_by_parent",
            {key: tuple(value) for key, value in children.items()},
        )

    @property
    def by_id(self) -> dict[str, CandidateTreeNode]:
        return self._by_id

    @property
    def children_by_parent(self) -> dict[str, tuple[CandidateTreeNode, ...]]:
        return self._children_by_parent

    @property
    def candidate_nodes(self) -> tuple[CandidateTreeNode, ...]:
        return tuple(node for node in self.nodes if node.node_id != self.root.node_id)

    @property
    def width(self) -> int:
        counts: dict[int, int] = {}
        for node in self.candidate_nodes:
            counts[node.depth] = counts.get(node.depth, 0) + 1
        return max(counts.values(), default=0)

    @property
    def depth(self) -> int:
        return max((node.depth for node in self.nodes), default=0)

    def ancestors(self, node_id: str, *, include_root: bool = False) -> tuple[str, ...]:
        by_id = self.by_id
        if node_id not in by_id:
            raise ValueError(f"unknown candidate-tree node: {node_id}")
        path = []
        node = by_id[node_id]
        while node.parent_id is not None:
            path.append(node.node_id)
            node = by_id[node.parent_id]
        if include_root:
            path.append(self.root.node_id)
        return tuple(reversed(path))


@dataclass(frozen=True)
class SelectedProposalTree:
    selected_node_ids: tuple[str, ...]
    dependency_edges: tuple[tuple[str, str], ...]
    candidate_budget: int
    root_in_candidate_budget: bool = False

    def __post_init__(self) -> None:
        if len(set(self.selected_node_ids)) != len(self.selected_node_ids):
            raise ValueError("selected tree node IDs must be unique")
        if self.candidate_budget < 0:
            raise ValueError("candidate tree budget must be non-negative")
        if self.candidate_budget != len(self.selected_node_ids):
            raise ValueError("candidate budget must equal the selected tree-node count")

    @property
    def expected_progress(self) -> float:
        """Compatibility placeholder; use expected_tree_progress with a candidate tree."""

        return float(self.candidate_budget)


@dataclass(frozen=True)
class TreeVerificationOutcome:
    accepted_branch_node_ids: tuple[str, ...]
    committed_progress: int


def validate_prefix_closure(
    tree: CandidateTree, selected: SelectedProposalTree
) -> None:
    """Raise when a selected node omits a non-root ancestor or edge."""

    by_id = tree.by_id
    selected_ids = set(selected.selected_node_ids)
    expected_edges = set()
    for node_id in selected.selected_node_ids:
        node = by_id.get(node_id)
        if node is None or node_id == tree.root.node_id:
            raise ValueError("selected proposal contains an unknown node or explicit root")
        if node.parent_id != tree.root.node_id and node.parent_id not in selected_ids:
            raise ValueError("selected proposal tree is not prefix closed")
        expected_edges.add((node.parent_id or "", node_id))
    if set(selected.dependency_edges) != expected_edges:
        raise ValueError("selected proposal dependency edges do not match its nodes")


def make_selected_tree(
    tree: CandidateTree, selected_node_ids: Iterable[str]
) -> SelectedProposalTree:
    node_ids = tuple(selected_node_ids)
    by_id = tree.by_id
    edges = tuple((by_id[node_id].parent_id or "", node_id) for node_id in node_ids)
    selected = SelectedProposalTree(node_ids, edges, len(node_ids), False)
    validate_prefix_closure(tree, selected)
    return selected


def expected_tree_progress(tree: CandidateTree, selected: SelectedProposalTree) -> float:
    validate_prefix_closure(tree, selected)
    by_id = tree.by_id
    return sum(by_id[node_id].path_probability for node_id in selected.selected_node_ids)


def maximum_expected_candidate_progress(tree: CandidateTree, budget: int) -> float:
    """Return the maximum prefix-closed expected progress under a node budget.

    Candidate path probabilities are non-increasing along every edge. Sorting by
    probability, then shallower depth, therefore cannot select a descendant before
    its ancestor and yields the maximum-weight prefix-closed set.
    """

    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0:
        raise ValueError("candidate progress budget must be a non-negative integer")
    ranked = sorted(
        tree.candidate_nodes,
        key=lambda node: (-node.path_probability, node.depth, node.node_id),
    )
    selected = ranked[:budget]
    chosen = {node.node_id for node in selected}
    for node in selected:
        if node.parent_id != tree.root.node_id and node.parent_id not in chosen:
            raise AssertionError("maximum-progress selection lost prefix closure")
    return sum(node.path_probability for node in selected)


def predicted_dependency_path(
    tree: CandidateTree, selected: SelectedProposalTree
) -> tuple[str, ...]:
    """Return the deepest, then highest-probability selected root-to-node path."""

    validate_prefix_closure(tree, selected)
    if not selected.selected_node_ids:
        return ()
    by_id = tree.by_id
    leaf = max(
        (by_id[node_id] for node_id in selected.selected_node_ids),
        key=lambda node: (node.depth, node.path_probability, node.node_id),
    )
    return tree.ancestors(leaf.node_id)


def select_sequence_path(tree: CandidateTree, budget: int) -> SelectedProposalTree:
    """Select one deterministic highest-probability path from a candidate tree."""

    selected = []
    parent_id = tree.root.node_id
    for _ in range(budget):
        children = [
            node for node in tree.candidate_nodes if node.parent_id == parent_id
        ]
        if not children:
            break
        node = max(children, key=lambda item: (item.path_probability, item.node_id))
        selected.append(node.node_id)
        parent_id = node.node_id
    return make_selected_tree(tree, selected)


def truncate_selected_tree(
    tree: CandidateTree, selected: SelectedProposalTree, budget: int
) -> SelectedProposalTree:
    """Truncate a prefix-closed selection to a smaller node budget."""

    if budget < 0:
        raise ValueError("truncated tree budget must be non-negative")
    return make_selected_tree(tree, selected.selected_node_ids[:budget])


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & ((1 << 64) - 1)
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & ((1 << 64) - 1)
    return value ^ (value >> 31)


class CandidateTreeOracle:
    """Generate policy-independent candidate trees and target branches deterministically."""

    def __init__(
        self,
        seed: int,
        *,
        injected_trees: Optional[dict[tuple[str, int], CandidateTree]] = None,
        injected_branches: Optional[dict[tuple[str, int], tuple[str, ...]]] = None,
    ) -> None:
        self.seed = seed
        self._trees = dict(injected_trees or {})
        self._branches = dict(injected_branches or {})
        self._request_seeds: dict[str, int] = {}

    def _request_seed(self, request_id: str) -> int:
        if request_id not in self._request_seeds:
            payload = f"{self.seed}:{request_id}".encode()
            self._request_seeds[request_id] = int.from_bytes(
                hashlib.sha256(payload).digest()[:8], "big"
            )
        return self._request_seeds[request_id]

    def _event(self, request_id: str, prefix: int, depth: int, salt: int) -> float:
        value = self._request_seed(request_id)
        value ^= prefix * 0xD6E8FEB86659FD93
        value ^= depth * 0xA5A3564E27F8862B
        value ^= salt * 0x9E3779B97F4A7C15
        return _splitmix64(value) / float(2**64)

    def tree(
        self,
        request_id: str,
        committed_prefix_len: int,
        *,
        width: int,
        depth: int,
        draft_confidence: float,
    ) -> CandidateTree:
        key = (request_id, committed_prefix_len)
        if key in self._trees:
            return self._trees[key]
        if width < 1 or depth < 0:
            raise ValueError("candidate tree width must be positive and depth non-negative")
        confidence = max(0.0, min(1.0, draft_confidence))
        root = CandidateTreeNode("root", None, 0, 1.0, 1.0)
        nodes = [root]
        frontier = [root]
        for level in range(1, depth + 1):
            candidates = []
            for parent in frontier:
                branches = range(width) if level == 1 else range(1)
                raw = [
                    0.5
                    + self._event(
                        request_id,
                        committed_prefix_len,
                        level,
                        len(nodes) + branch,
                    )
                    for branch in branches
                ]
                scale = confidence / sum(raw) if level == 1 else confidence / max(raw)
                for branch, value in enumerate(raw):
                    conditional = value * scale
                    node_id = f"d{level}-p{parent.node_id}-b{branch}"
                    candidates.append(
                        CandidateTreeNode(
                            node_id=node_id,
                            parent_id=parent.node_id,
                            depth=level,
                            token_confidence=conditional,
                            path_probability=parent.path_probability * conditional,
                        )
                    )
            frontier = sorted(candidates, key=lambda node: node.node_id)
            nodes.extend(frontier)
        result = CandidateTree(request_id, root, tuple(nodes))
        # Generated trees are intentionally not cached: proposal objects retain the tree
        # only through verification, while full R3 runs contain millions of distinct
        # request-prefix keys. Deterministic regeneration preserves cross-policy fairness
        # without turning the oracle into an unbounded result store.
        return result

    def verify(
        self,
        tree: CandidateTree,
        selected: SelectedProposalTree,
        *,
        committed_prefix_len: int,
        acceptance_probability: float,
    ) -> TreeVerificationOutcome:
        validate_prefix_closure(tree, selected)
        key = (tree.request_id, committed_prefix_len)
        selected_ids = set(selected.selected_node_ids)
        by_parent = tree.children_by_parent
        accepted = []
        parent_id = tree.root.node_id
        injected = self._branches.get(key)
        for depth in range(1, tree.depth + 1):
            if injected is not None:
                node_id = injected[depth - 1] if depth <= len(injected) else None
            else:
                children = by_parent.get(parent_id, ())
                if not children:
                    break
                event = self._event(tree.request_id, committed_prefix_len, depth, 1)
                if event >= acceptance_probability:
                    break
                choice = int(
                    self._event(tree.request_id, committed_prefix_len, depth, 2)
                    * len(children)
                )
                node_id = children[min(choice, len(children) - 1)].node_id
            if node_id is None or node_id not in selected_ids:
                break
            node = tree.by_id.get(node_id)
            if node is None or node.parent_id != parent_id:
                break
            accepted.append(node_id)
            parent_id = node_id
        return TreeVerificationOutcome(tuple(accepted), len(accepted) + 1)
