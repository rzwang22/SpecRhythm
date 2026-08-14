"""Deterministic proposal-lifecycle simulator for Phase-A semantic validation."""

from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Optional

from specrhythm.policies.base import PolicySnapshot, RequestView, SchedulingPolicy, StepPlan
from specrhythm.schema import Workload, WorkloadRequest
from specrhythm.tree import (
    CandidateTree,
    CandidateTreeOracle,
    SelectedProposalTree,
    expected_tree_progress,
    predicted_dependency_path,
    select_sequence_path,
    truncate_selected_tree,
)


@dataclass(frozen=True)
class SimulatorConfig:
    max_active_requests: int = 64
    roof_candidate_budget: int = 192
    max_request_budget: int = 8
    verify_base_ms: float = 7.0
    verify_per_request_ms: float = 0.08
    verify_per_candidate_ms: float = 0.025
    draft_per_candidate_ms: float = 0.018
    speculative_budget: int = 4
    candidate_tree_width: int = 2
    candidate_tree_depth: int = 8
    n_max_slo: int = 8
    max_eager_budget: int = 2
    min_dependency_path_probability: float = 0.10
    specrhythm_residual_score: str = "urgency-path-probability"
    max_cycles: int = 1_000_000
    seed: int = 0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SimulatorConfig:
        return cls(**value)

    def __post_init__(self) -> None:
        integer_values = (
            self.max_active_requests,
            self.roof_candidate_budget,
            self.max_request_budget,
            self.speculative_budget,
            self.candidate_tree_width,
            self.candidate_tree_depth,
            self.n_max_slo,
            self.max_eager_budget,
            self.max_cycles,
            self.seed,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) for value in integer_values):
            raise ValueError("simulator count, budget, cycle, and seed values must be integers")
        if self.max_active_requests < 1 or self.roof_candidate_budget < 0:
            raise ValueError(
                "active request capacity must be positive and roof budget non-negative"
            )
        if (
            self.max_request_budget < 0
            or self.speculative_budget < 0
            or self.candidate_tree_width < 1
            or self.candidate_tree_depth < 0
            or self.n_max_slo < 0
            or self.max_eager_budget < 0
            or self.max_cycles < 1
        ):
            raise ValueError("tree/request budgets must be valid and max_cycles positive")
        latency_values = (
            self.verify_base_ms,
            self.verify_per_request_ms,
            self.verify_per_candidate_ms,
            self.draft_per_candidate_ms,
            self.min_dependency_path_probability,
        )
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value < 0
            or not math.isfinite(value)
            for value in latency_values
        ):
            raise ValueError("latency parameters must be finite and non-negative")
        if self.specrhythm_residual_score not in {
            "path-probability",
            "urgency-path-probability",
        }:
            raise ValueError("invalid SpecRhythm residual score")


@dataclass(frozen=True)
class Proposal:
    request_id: str
    parent_prefix_len: int
    prefix_epoch: int
    budget: int
    drafted_tokens: int
    source: str
    drafted_at_cycle: int
    candidate_tree: Optional[CandidateTree] = None
    selected_tree: Optional[SelectedProposalTree] = None
    dependency_path: tuple[str, ...] = ()
    slo_tpot_ms: float = 0.0
    dependency_path_probability: float = 0.0

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("proposal request_id must not be empty")
        integer_values = (
            self.parent_prefix_len,
            self.prefix_epoch,
            self.budget,
            self.drafted_tokens,
            self.drafted_at_cycle,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in integer_values
        ):
            raise ValueError("proposal prefix, epoch, budget, tokens, and cycle must be integers")
        if self.drafted_tokens > self.budget:
            raise ValueError("proposal drafted_tokens must not exceed budget")
        if self.source not in {"normal", "eager"}:
            raise ValueError("proposal source must be 'normal' or 'eager'")
        if self.selected_tree is not None:
            if self.candidate_tree is None:
                raise ValueError("a selected proposal tree requires its candidate tree")
            if self.selected_tree.candidate_budget != self.drafted_tokens:
                raise ValueError(
                    "selected tree budget must equal drafted tokens: "
                    f"{self.request_id} selected={self.selected_tree.candidate_budget} "
                    f"drafted={self.drafted_tokens} budget={self.budget}"
                )
        if not 0 <= self.dependency_path_probability <= 1:
            raise ValueError("dependency path probability must be in [0, 1]")

    @property
    def key(self) -> tuple[Any, ...]:
        return (
            self.request_id,
            self.parent_prefix_len,
            self.prefix_epoch,
            self.budget,
            self.drafted_tokens,
            self.source,
            self.drafted_at_cycle,
            tuple(self.selected_tree.selected_node_ids) if self.selected_tree else (),
            self.dependency_path,
            self.slo_tpot_ms,
            self.dependency_path_probability,
        )


@dataclass
class RuntimeRequest:
    request: WorkloadRequest
    slot: int
    admitted_at_ms: float
    normal_proposal: Optional[Proposal] = None
    eager_proposal: Optional[Proposal] = None
    committed_prefix_len: int = 0
    prefix_epoch: int = 0
    finished: bool = False
    elapsed_decode_ms: float = 0.0
    service_latency_ms: float = 0.0
    recent_acceptance_ratio: float = 0.7
    draft_confidence: float = 0.7

    def __post_init__(self) -> None:
        if self.slot not in {0, 1}:
            raise ValueError("runtime slot must be 0 or 1")


@dataclass(frozen=True)
class RuntimeRequestState:
    request_id: str
    normal_proposal: Optional[Proposal]
    eager_proposal: Optional[Proposal]
    committed_prefix_len: int
    prefix_epoch: int
    finished: bool


@dataclass(frozen=True)
class RequestResult:
    request_id: str
    task: str
    output_tokens: int
    slo_tpot_ms: float
    queueing_latency_ms: float
    service_latency_ms: float
    decode_latency_ms: float
    tpot_ms: float
    attained: bool


@dataclass(frozen=True)
class CycleDiagnostic:
    cycle: int
    active_requests: int
    pending_requests: int
    candidate_roof: int
    normal_budget: int
    eager_budget: int
    budget_by_slo_class: dict[str, int]
    eager_budget_by_slo_class: dict[str, int]
    draft_latency_ms: float
    verify_latency_ms: float
    cycle_latency_ms: float
    predicted_cycle_latency_ms: float
    prediction_error_ms: float
    verify_requests: int = 0
    verified_candidate_nodes: int = 0
    committed_candidate_tokens: int = 0
    root_progress: int = 0
    total_progress: int = 0
    base_request_ids: tuple[str, ...] = ()
    base_candidate_nodes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    base_work_preserved: bool = True


@dataclass(frozen=True)
class AllocationOpportunityDiagnostic:
    request_id: str
    slo_tpot_ms: float
    drafted_at_cycle: int
    progress_gap: float
    required_total_progress: float
    required_candidate_progress: float
    maximum_attainable_candidate_progress: float
    maximum_attainable_total_progress: float
    one_cycle_feasible: bool
    stage1_budget: int
    stage2_budget: int
    base_budget: int
    selected_expected_progress: float
    realized_committed_progress: int
    realized_candidate_progress: int
    root_progress: int
    remaining_output_tokens: int


@dataclass(frozen=True)
class EagerDiagnostic:
    request_id: str
    slo_tpot_ms: float
    drafted_at_cycle: int
    dependency_path: tuple[str, ...]
    dependency_path_probability: float
    eager_budget: int
    normal_budget_displaced: int
    outcome: str
    promoted_progress: int
    promoted_progress_per_drafted_token: float


@dataclass(frozen=True)
class RequestAllocationDiagnostic:
    request_id: str
    slo_tpot_ms: float
    allocated_candidate_nodes: int
    expected_progress: float
    realized_candidate_progress: int
    allocation_opportunities: int
    attained: bool


@dataclass(frozen=True)
class SimulationSummary:
    schema_version: str
    model_status: str
    input_tokens_modeled: bool
    context_dependent_latency_modeled: bool
    proxy_parameter_status: dict[str, str]
    simulator_parameters: dict[str, Any]
    policy: str
    display_name: str
    execution_mode: str
    allocator: str
    base_allocator: str
    residual_selector: str
    eager_semantics: str
    requests: int
    completed_requests: int
    first_arrival_ms: float
    last_arrival_ms: float
    drain_completion_ms: float
    arrival_span_ms: float
    processing_and_drain_ms: float
    makespan_ms: float
    measurement_ms: float
    raw_generated_tokens: int
    slo_good_tokens: int
    raw_throughput_tokens_per_s: float
    throughput_tokens_per_s: float
    goodput_tokens_per_s: float
    slo_attainment: float
    p50_tpot_ms: Optional[float]
    p90_tpot_ms: Optional[float]
    p99_tpot_ms: Optional[float]
    mean_queueing_latency_ms: float
    mean_service_latency_ms: float
    mean_decode_latency_ms: float
    p50_queueing_latency_ms: Optional[float]
    p50_service_latency_ms: Optional[float]
    p50_decode_latency_ms: Optional[float]
    normal_drafted_proposals: int
    eager_drafted_proposals: int
    verified_proposals: int
    fully_accepted_proposals: int
    promoted_proposals: int
    invalidated_proposals: int
    discarded_at_eos_proposals: int
    normal_drafted_tokens: int
    eager_drafted_tokens: int
    verified_tokens: int
    accepted_tokens: int
    promoted_tokens: int
    invalidated_tokens: int
    discarded_at_eos_tokens: int
    draft_compute_ms: float
    verify_compute_ms: float
    eager_promotions: int
    eager_invalidations: int
    eager_discarded_at_eos_proposals: int
    eager_invalidated_tokens: int
    eager_discarded_at_eos_tokens: int
    eager_promotion_proposal_ratio: float
    eager_invalidation_proposal_ratio: float
    eager_eos_discard_proposal_ratio: float
    eager_promotion_token_ratio: float
    eager_invalidation_token_ratio: float
    eager_eos_discard_token_ratio: float
    draft_compute_waste_ratio: float
    eager_compute_waste_ratio: float
    root_in_candidate_budget: bool
    root_progress_definition: str
    candidate_roof_definition: str
    target_input_positions: str
    verify_latency_inputs: dict[str, Any]
    tree_drafted_nodes: int
    tree_verified_nodes: int
    tree_accepted_nodes: int
    tree_invalidated_nodes: int
    tree_discarded_at_eos_nodes: int
    baseline_root_progress: int
    candidate_expected_progress: float
    candidate_realized_progress: int
    mean_candidate_tree_width: float
    mean_candidate_tree_depth: float
    mean_candidate_tree_nodes: float
    selected_path_probability_distribution: dict[str, float]
    allocator_stage_metrics: dict[str, int]
    cycles: int
    requests_completed_per_cycle: float
    mean_verify_batch: float
    candidate_roof_utilization: float
    one_cycle_infeasible_opportunity_ratio: float
    stage1_nodes_to_one_cycle_infeasible: int
    stage1_infeasible_node_ratio: float
    verified_candidate_nodes_per_cycle: float
    candidate_committed_tokens_per_cycle: float
    root_progress_per_cycle: float
    total_progress_per_cycle: float
    accepted_candidate_tokens_per_verified_node: float
    selected_expected_progress_per_opportunity: float
    realized_committed_progress_per_opportunity: float
    base_preservation_violations: int
    cycle_diagnostics: tuple[CycleDiagnostic, ...] = ()
    cycle_diagnostics_truncated: int = 0
    eager_diagnostics: tuple[EagerDiagnostic, ...] = ()
    eager_diagnostics_truncated: int = 0
    request_allocation_diagnostics: tuple[RequestAllocationDiagnostic, ...] = ()
    allocation_opportunity_diagnostics: tuple[AllocationOpportunityDiagnostic, ...] = ()
    allocation_opportunity_diagnostics_truncated: int = 0
    class_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    slo_class_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def proposed_draft_tokens(self) -> int:
        return self.normal_drafted_tokens + self.eager_drafted_tokens

    @property
    def accepted_draft_tokens(self) -> int:
        return self.accepted_tokens

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["proposed_draft_tokens"] = self.proposed_draft_tokens
        value["accepted_draft_tokens"] = self.accepted_draft_tokens
        return value


@dataclass(frozen=True)
class SimulationResult:
    summary: SimulationSummary
    requests: tuple[RequestResult, ...]
    final_states: tuple[RuntimeRequestState, ...]


class AcceptanceOracle:
    """Prefix-indexed max-K acceptance traces shared by every policy and budget."""

    def __init__(
        self,
        seed: int,
        max_k: int,
        traces: Optional[dict[tuple[str, int], tuple[bool, ...]]] = None,
    ) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("oracle seed must be an integer")
        if not isinstance(max_k, int) or isinstance(max_k, bool) or max_k < 0:
            raise ValueError("oracle max_k must be a non-negative integer")
        self.seed = seed
        self.max_k = max_k
        self._cache: dict[tuple[str, int], tuple[bool, ...]] = {}
        for key, trace in (traces or {}).items():
            if len(trace) != max_k or any(not isinstance(value, bool) for value in trace):
                raise ValueError("injected acceptance traces must contain max_k booleans")
            self._cache[key] = trace

    def trace(
        self, request_id: str, committed_target_prefix_len: int, probability: float
    ) -> tuple[bool, ...]:
        if not 0 <= probability <= 1 or not math.isfinite(probability):
            raise ValueError("acceptance probability must be finite and in [0, 1]")
        key = (request_id, committed_target_prefix_len)
        if key not in self._cache:
            values = []
            for depth in range(1, self.max_k + 1):
                payload = (
                    f"{self.seed}:{request_id}:{committed_target_prefix_len}:{depth}"
                ).encode()
                integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
                values.append(integer / float(2**64) < probability)
            self._cache[key] = tuple(values)
        return self._cache[key]

    def accepted_prefix(
        self,
        request_id: str,
        committed_target_prefix_len: int,
        budget: int,
        probability: float,
    ) -> int:
        if (
            not isinstance(budget, int)
            or isinstance(budget, bool)
            or not 0 <= budget <= self.max_k
        ):
            raise ValueError("oracle budget must be an integer in [0, max_k]")
        accepted = 0
        for matches in self.trace(request_id, committed_target_prefix_len, probability)[:budget]:
            if not matches:
                break
            accepted += 1
        return accepted


@dataclass(frozen=True)
class _VerificationOutcome:
    parent: Proposal
    valid_parent: bool
    accepted_tokens: int
    fully_accepted: bool
    accepted_branch_node_ids: tuple[str, ...] = ()
    committed_progress: int = 0
    root_progress: int = 0


@dataclass
class _LifecycleLedger:
    normal_drafted_proposals: int = 0
    eager_drafted_proposals: int = 0
    verified_proposals: int = 0
    fully_accepted_proposals: int = 0
    promoted_proposals: int = 0
    invalidated_proposals: int = 0
    discarded_at_eos_proposals: int = 0
    normal_drafted_tokens: int = 0
    eager_drafted_tokens: int = 0
    verified_tokens: int = 0
    accepted_tokens: int = 0
    promoted_tokens: int = 0
    invalidated_tokens: int = 0
    discarded_at_eos_tokens: int = 0
    draft_compute_ms: float = 0.0
    verify_compute_ms: float = 0.0
    eager_promotions: int = 0
    eager_invalidations: int = 0
    eager_invalidated_tokens: int = 0
    eager_discarded_at_eos_tokens: int = 0
    eager_discarded_at_eos_proposals: int = 0
    eager_accepted_tokens: int = 0
    tree_drafted_nodes: int = 0
    tree_verified_nodes: int = 0
    tree_accepted_nodes: int = 0
    tree_invalidated_nodes: int = 0
    tree_discarded_at_eos_nodes: int = 0
    selected_path_probability_count: int = 0
    selected_path_probability_sum: float = 0.0
    selected_path_probability_histogram: list[int] = field(
        default_factory=lambda: [0] * 101
    )
    candidate_tree_samples: int = 0
    candidate_tree_width_sum: int = 0
    candidate_tree_depth_sum: int = 0
    candidate_tree_node_sum: int = 0
    class_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    request_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    eager_events: dict[tuple[Any, ...], dict[str, Any]] = field(default_factory=dict)
    completed_eager_events: list[dict[str, Any]] = field(default_factory=list)
    baseline_root_progress: int = 0
    candidate_expected_progress: float = 0.0
    slo_stage_nodes: int = 0
    residual_stage_nodes: int = 0
    eager_detail_limit: int = 10_000
    eager_events_truncated: int = 0
    allocation_events: dict[tuple[Any, ...], dict[str, Any]] = field(default_factory=dict)
    pending_allocation_events: dict[tuple[str, int], dict[str, Any]] = field(
        default_factory=dict
    )
    completed_allocation_events: list[AllocationOpportunityDiagnostic] = field(
        default_factory=list
    )
    allocation_detail_limit: int = 10_000
    allocation_events_truncated: int = 0
    allocation_opportunities: int = 0
    one_cycle_infeasible_opportunities: int = 0
    stage1_nodes_to_one_cycle_infeasible: int = 0
    selected_expected_progress_total: float = 0.0
    realized_committed_progress_total: int = 0
    base_selected_nodes: int = 0
    candidate_roof_used: int = 0
    candidate_roof_capacity: int = 0
    base_preservation_violations: int = 0
    verify_request_positions: int = 0
    eager_sink: Optional[Callable[[EagerDiagnostic], None]] = field(
        default=None, repr=False
    )
    allocation_sink: Optional[Callable[[AllocationOpportunityDiagnostic], None]] = field(
        default=None, repr=False
    )
    drafted: set[tuple[Any, ...]] = field(default_factory=set)
    promoted: set[tuple[Any, ...]] = field(default_factory=set)

    def record_drafted(self, proposal: Proposal, draft_per_candidate_ms: float) -> None:
        if proposal.key in self.drafted:
            raise AssertionError("proposal was drafted more than once")
        self.drafted.add(proposal.key)
        if proposal.source == "normal":
            self.normal_drafted_proposals += 1
            self.normal_drafted_tokens += proposal.drafted_tokens
        else:
            self.eager_drafted_proposals += 1
            self.eager_drafted_tokens += proposal.drafted_tokens
            self.eager_events[proposal.key] = {
                "request_id": proposal.request_id,
                "slo_tpot_ms": proposal.slo_tpot_ms,
                "drafted_at_cycle": proposal.drafted_at_cycle,
                "dependency_path": proposal.dependency_path,
                "dependency_path_probability": proposal.dependency_path_probability,
                "eager_budget": proposal.drafted_tokens,
                "normal_budget_displaced": proposal.drafted_tokens,
                "outcome": "pending",
                "promoted_progress": 0,
            }
        self.draft_compute_ms += proposal.drafted_tokens * draft_per_candidate_ms
        label = f"{proposal.slo_tpot_ms:g}"
        stats = self.class_stats.setdefault(
            label,
            {
                "proposal_opportunities": 0,
                "scheduled_proposals": 0,
                "budget_histogram": {},
                "budget_sum": 0,
                "requested_progress_gap_sum": 0.0,
                "expected_progress_sum": 0.0,
                "drafted_tokens": 0,
                "verified_tokens": 0,
                "accepted_tokens": 0,
                "invalidated_tokens": 0,
            },
        )
        stats["drafted_tokens"] += proposal.drafted_tokens
        if proposal.selected_tree is not None and proposal.candidate_tree is not None:
            self.tree_drafted_nodes += proposal.selected_tree.candidate_budget
            by_id = proposal.candidate_tree.by_id
            for node_id in proposal.selected_tree.selected_node_ids:
                probability = by_id[node_id].path_probability
                self.selected_path_probability_count += 1
                self.selected_path_probability_sum += probability
                self.selected_path_probability_histogram[
                    min(100, int(probability * 100))
                ] += 1
            self.candidate_tree_samples += 1
            self.candidate_tree_width_sum += proposal.candidate_tree.width
            self.candidate_tree_depth_sum += proposal.candidate_tree.depth
            self.candidate_tree_node_sum += len(proposal.candidate_tree.candidate_nodes)

    def record_verified(self, proposal: Proposal, accepted_tokens: int) -> None:
        self._record_terminal(proposal)
        if not 0 <= accepted_tokens <= proposal.drafted_tokens:
            raise AssertionError("accepted tokens must be a subset of verified tokens")
        self.verified_tokens += proposal.drafted_tokens
        self.accepted_tokens += accepted_tokens
        stats = self.class_stats[f"{proposal.slo_tpot_ms:g}"]
        stats["verified_tokens"] += proposal.drafted_tokens
        stats["accepted_tokens"] += accepted_tokens
        self.request_stats[proposal.request_id]["realized_candidate_progress"] += (
            accepted_tokens
        )
        self.verified_proposals += 1
        if accepted_tokens == proposal.drafted_tokens and proposal.drafted_tokens > 0:
            self.fully_accepted_proposals += 1
        if proposal.source == "eager":
            self.eager_accepted_tokens += accepted_tokens
        if proposal.selected_tree is not None:
            self.tree_verified_nodes += proposal.selected_tree.candidate_budget
            self.tree_accepted_nodes += accepted_tokens

    def record_invalidated(self, proposal: Proposal, *, at_eos: bool = False) -> None:
        self._record_terminal(proposal)
        self.resolve_allocation(
            proposal,
            committed_progress=0,
            candidate_progress=0,
            root_progress=0,
        )
        if at_eos:
            self.discarded_at_eos_proposals += 1
            self.discarded_at_eos_tokens += proposal.drafted_tokens
            if proposal.source == "eager":
                self.eager_discarded_at_eos_tokens += proposal.drafted_tokens
                self.eager_discarded_at_eos_proposals += 1
            if proposal.selected_tree is not None:
                self.tree_discarded_at_eos_nodes += proposal.selected_tree.candidate_budget
        else:
            self.invalidated_proposals += 1
            self.invalidated_tokens += proposal.drafted_tokens
            self.class_stats[f"{proposal.slo_tpot_ms:g}"]["invalidated_tokens"] += (
                proposal.drafted_tokens
            )
            if proposal.source == "eager":
                self.eager_invalidated_tokens += proposal.drafted_tokens
            if proposal.selected_tree is not None:
                self.tree_invalidated_nodes += proposal.selected_tree.candidate_budget
        if proposal.source == "eager":
            self.eager_events[proposal.key]["outcome"] = (
                "eos-discard" if at_eos else "invalidation"
            )
            event = self.eager_events.pop(proposal.key)
            self._emit_eager_event(event)
            if len(self.completed_eager_events) < self.eager_detail_limit:
                self.completed_eager_events.append(event)
            else:
                self.eager_events_truncated += 1

    def record_allocation(
        self,
        request_id: str,
        slo_tpot_ms: float,
        drafted_at_cycle: int,
        budget: int,
        requested_gap: float,
        expected_progress: float,
        slo_stage_budget: int,
        residual_stage_budget: int,
        base_budget: int,
        required_total_progress: float,
        required_candidate_progress: float,
        maximum_attainable_candidate_progress: float,
        maximum_attainable_total_progress: float,
        one_cycle_feasible: bool,
        remaining_output_tokens: int,
    ) -> None:
        label = f"{slo_tpot_ms:g}"
        stats = self.class_stats.setdefault(
            label,
            {
                "proposal_opportunities": 0,
                "scheduled_proposals": 0,
                "budget_histogram": {},
                "budget_sum": 0,
                "requested_progress_gap_sum": 0.0,
                "expected_progress_sum": 0.0,
                "drafted_tokens": 0,
                "verified_tokens": 0,
                "accepted_tokens": 0,
                "invalidated_tokens": 0,
            },
        )
        stats["proposal_opportunities"] += 1
        stats["scheduled_proposals"] += int(budget > 0)
        histogram = stats["budget_histogram"]
        histogram[str(budget)] = histogram.get(str(budget), 0) + 1
        stats["budget_sum"] += budget
        stats["requested_progress_gap_sum"] += requested_gap
        stats["expected_progress_sum"] += expected_progress
        self.candidate_expected_progress += expected_progress
        self.slo_stage_nodes += slo_stage_budget
        self.residual_stage_nodes += residual_stage_budget
        self.allocation_opportunities += 1
        self.one_cycle_infeasible_opportunities += int(not one_cycle_feasible)
        if not one_cycle_feasible:
            self.stage1_nodes_to_one_cycle_infeasible += slo_stage_budget
        self.selected_expected_progress_total += expected_progress
        self.base_selected_nodes += base_budget
        event = {
            "request_id": request_id,
            "slo_tpot_ms": slo_tpot_ms,
            "drafted_at_cycle": drafted_at_cycle,
            "progress_gap": requested_gap,
            "required_total_progress": required_total_progress,
            "required_candidate_progress": required_candidate_progress,
            "maximum_attainable_candidate_progress": (
                maximum_attainable_candidate_progress
            ),
            "maximum_attainable_total_progress": maximum_attainable_total_progress,
            "one_cycle_feasible": one_cycle_feasible,
            "stage1_budget": slo_stage_budget,
            "stage2_budget": residual_stage_budget,
            "base_budget": base_budget,
            "selected_expected_progress": expected_progress,
            "realized_committed_progress": 0,
            "realized_candidate_progress": 0,
            "root_progress": 0,
            "remaining_output_tokens": remaining_output_tokens,
        }
        key = (request_id, drafted_at_cycle)
        self.allocation_events[key] = event
        self.pending_allocation_events[(request_id, drafted_at_cycle)] = event
        request_stats = self.request_stats.setdefault(
            request_id,
            {
                "slo_tpot_ms": slo_tpot_ms,
                "allocated_candidate_nodes": 0,
                "expected_progress": 0.0,
                "realized_candidate_progress": 0,
                "allocation_opportunities": 0,
            },
        )
        request_stats["allocated_candidate_nodes"] += budget
        request_stats["expected_progress"] += expected_progress
        request_stats["allocation_opportunities"] += 1

    def resolve_allocation(
        self,
        proposal: Proposal,
        *,
        committed_progress: int,
        candidate_progress: int,
        root_progress: int,
    ) -> None:
        if proposal.source != "normal":
            return
        key = (proposal.request_id, proposal.drafted_at_cycle)
        event = self.pending_allocation_events.pop(key, None)
        if event is None:
            return
        event["realized_committed_progress"] = committed_progress
        event["realized_candidate_progress"] = candidate_progress
        event["root_progress"] = root_progress
        self.realized_committed_progress_total += committed_progress
        diagnostic = AllocationOpportunityDiagnostic(**event)
        if self.allocation_sink is not None:
            self.allocation_sink(diagnostic)
        if len(self.completed_allocation_events) < self.allocation_detail_limit:
            self.completed_allocation_events.append(diagnostic)
        else:
            self.allocation_events_truncated += 1
        self.allocation_events.pop(key, None)

    def record_promotion(self, proposal: Proposal) -> None:
        if proposal.source != "eager" or proposal.key not in self.drafted:
            raise AssertionError("only a drafted eager proposal can be promoted")
        if proposal.key in self.promoted:
            raise AssertionError("proposal was promoted twice or after terminal disposition")
        self.promoted.add(proposal.key)
        self.promoted_proposals += 1
        self.promoted_tokens += proposal.drafted_tokens
        self.eager_promotions += 1
        self.eager_events[proposal.key]["outcome"] = "promotion"
        self.eager_events[proposal.key]["promoted_progress"] = proposal.drafted_tokens
        event = self.eager_events.pop(proposal.key)
        self._emit_eager_event(event)
        if len(self.completed_eager_events) < self.eager_detail_limit:
            self.completed_eager_events.append(event)
        else:
            self.eager_events_truncated += 1

    def _emit_eager_event(self, event: dict[str, Any]) -> None:
        if self.eager_sink is None:
            return
        self.eager_sink(
            EagerDiagnostic(
                **event,
                promoted_progress_per_drafted_token=(
                    event["promoted_progress"] / event["eager_budget"]
                    if event["eager_budget"]
                    else 0.0
                ),
            )
        )

    def _record_terminal(self, proposal: Proposal) -> None:
        if proposal.key not in self.drafted:
            raise AssertionError("proposal reached a terminal state before being drafted")
        self.drafted.remove(proposal.key)
        self.promoted.discard(proposal.key)

    def assert_conservation(self) -> None:
        if self.drafted:
            raise AssertionError("every drafted proposal must have exactly one terminal state")
        drafted_tokens = self.normal_drafted_tokens + self.eager_drafted_tokens
        terminal_tokens = (
            self.verified_tokens + self.invalidated_tokens + self.discarded_at_eos_tokens
        )
        if drafted_tokens != terminal_tokens:
            raise AssertionError("drafted proposal tokens are not conserved")
        if self.accepted_tokens > self.verified_tokens:
            raise AssertionError("accepted tokens exceed verified tokens")
        if self.promoted_tokens > self.eager_drafted_tokens:
            raise AssertionError("promoted tokens exceed eager drafted tokens")
        drafted_proposals = self.normal_drafted_proposals + self.eager_drafted_proposals
        terminal_proposals = (
            self.verified_proposals
            + self.invalidated_proposals
            + self.discarded_at_eos_proposals
        )
        if drafted_proposals != terminal_proposals:
            raise AssertionError("drafted proposals are not conserved")
        if self.promoted_proposals > self.eager_drafted_proposals:
            raise AssertionError("promoted proposals exceed eager drafted proposals")
        if (
            self.promoted_proposals
            + self.eager_invalidations
            + self.eager_discarded_at_eos_proposals
            != self.eager_drafted_proposals
        ):
            raise AssertionError(
                "every eager proposal must be promoted, invalidated, or discarded"
            )
        if (
            self.promoted_tokens
            + self.eager_invalidated_tokens
            + self.eager_discarded_at_eos_tokens
            != self.eager_drafted_tokens
        ):
            raise AssertionError("eager tokens are not conserved at admission resolution")
        if self.tree_drafted_nodes != (
            self.tree_verified_nodes
            + self.tree_invalidated_nodes
            + self.tree_discarded_at_eos_nodes
        ):
            raise AssertionError("tree nodes must have exactly one terminal disposition")
        if self.tree_accepted_nodes > self.tree_verified_nodes:
            raise AssertionError("accepted tree nodes exceed verified tree nodes")
        if self.pending_allocation_events or self.allocation_events:
            raise AssertionError("every normal allocation opportunity must be resolved")


def eager_is_promotable(
    runtime: RuntimeRequest,
    parent: Proposal,
    eager: Proposal,
    *,
    parent_fully_accepted: bool,
    accepted_branch_node_ids: tuple[str, ...] = (),
) -> bool:
    """Check every guarded-commit dependency for one eager proposal."""

    dependency_satisfied = (
        tuple(accepted_branch_node_ids[: len(eager.dependency_path)])
        == eager.dependency_path
        if eager.dependency_path
        else parent_fully_accepted
    )
    dependency_length = (
        len(eager.dependency_path) if eager.dependency_path else parent.drafted_tokens
    )
    return (
        dependency_satisfied
        and not runtime.finished
        and eager.source == "eager"
        and eager.request_id == parent.request_id == runtime.request.request_id
        and eager.parent_prefix_len
        == parent.parent_prefix_len + dependency_length + 1
        and eager.parent_prefix_len == runtime.committed_prefix_len
        and eager.prefix_epoch == runtime.prefix_epoch
        and eager.prefix_epoch == parent.prefix_epoch + 1
    )


def _proposal_matches_runtime(runtime: RuntimeRequest, proposal: Proposal) -> bool:
    return (
        proposal.request_id == runtime.request.request_id
        and proposal.parent_prefix_len == runtime.committed_prefix_len
        and proposal.prefix_epoch == runtime.prefix_epoch
        and not runtime.finished
    )


def _percentile(values: list[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _build_summary(
    policy: SchedulingPolicy,
    results: list[RequestResult],
    measurement_ms: float,
    ledger: _LifecycleLedger,
    config: SimulatorConfig,
    *,
    first_arrival_ms: float,
    last_arrival_ms: float,
    drain_completion_ms: float,
    class_diagnostics: dict[str, dict[str, float]],
    cycles: int,
    cycle_diagnostics: list[CycleDiagnostic],
    cycle_diagnostics_truncated: int,
) -> SimulationSummary:
    duration_s = measurement_ms / 1000.0
    total_tokens = sum(result.output_tokens for result in results)
    good_tokens = sum(result.output_tokens for result in results if result.attained)
    tpots = [result.tpot_ms for result in results]
    queueing = [result.queueing_latency_ms for result in results]
    service = [result.service_latency_ms for result in results]
    decode = [result.decode_latency_ms for result in results]
    tasks = sorted({result.task for result in results})
    class_metrics: dict[str, dict[str, Any]] = {}
    for task in tasks:
        subset = [result for result in results if result.task == task]
        class_metrics[task] = {
            "requests": float(len(subset)),
            "attainment": sum(result.attained for result in subset) / len(subset),
            "mean_tpot_ms": statistics.mean(result.tpot_ms for result in subset),
            "mean_queueing_latency_ms": statistics.mean(
                result.queueing_latency_ms for result in subset
            ),
            "mean_service_latency_ms": statistics.mean(
                result.service_latency_ms for result in subset
            ),
            "mean_decode_latency_ms": statistics.mean(
                result.decode_latency_ms for result in subset
            ),
        }
    slo_class_metrics: dict[str, dict[str, Any]] = {}
    for slo in sorted({result.slo_tpot_ms for result in results}):
        subset = [result for result in results if result.slo_tpot_ms == slo]
        slo_class_metrics[f"{slo:g}"] = {
            "requests": float(len(subset)),
            "attainment": sum(result.attained for result in subset) / len(subset),
            "slo_good_tokens": sum(
                result.output_tokens for result in subset if result.attained
            ),
            "mean_tpot_ms": statistics.mean(result.tpot_ms for result in subset),
            "mean_queueing_latency_ms": statistics.mean(
                result.queueing_latency_ms for result in subset
            ),
            "mean_service_latency_ms": statistics.mean(
                result.service_latency_ms for result in subset
            ),
            "mean_decode_latency_ms": statistics.mean(
                result.decode_latency_ms for result in subset
            ),
        }
        diagnostics = ledger.class_stats.get(f"{slo:g}")
        if diagnostics:
            opportunities = diagnostics["proposal_opportunities"]
            histogram = diagnostics["budget_histogram"]
            slo_class_metrics[f"{slo:g}"].update(
                {
                    "mean_allocated_candidate_budget": diagnostics["budget_sum"]
                    / opportunities,
                    "budget_histogram": histogram,
                    "zero_budget_request_ratio": histogram.get("0", 0) / opportunities,
                    "scheduled_request_ratio": diagnostics["scheduled_proposals"]
                    / opportunities,
                    "mean_requested_progress_gap": diagnostics[
                        "requested_progress_gap_sum"
                    ]
                    / opportunities,
                    "mean_expected_progress": diagnostics["expected_progress_sum"]
                    / opportunities,
                    "mean_realized_verified_progress": (
                        diagnostics["accepted_tokens"] / opportunities
                    ),
                    "drafted_tokens": diagnostics["drafted_tokens"],
                    "verified_tokens": diagnostics["verified_tokens"],
                    "accepted_tokens": diagnostics["accepted_tokens"],
                    "invalidated_tokens": diagnostics["invalidated_tokens"],
                }
            )
    eager_proposals = ledger.eager_drafted_proposals
    eager_tokens = ledger.eager_drafted_tokens
    drafted_tokens = ledger.normal_drafted_tokens + eager_tokens
    def histogram_percentile(fraction: float) -> float:
        target = max(1, math.ceil(ledger.selected_path_probability_count * fraction))
        seen = 0
        for bucket, count in enumerate(ledger.selected_path_probability_histogram):
            seen += count
            if seen >= target:
                return bucket / 100.0
        return 0.0
    return SimulationSummary(
        schema_version="specrhythm.simulation-summary.v4",
        model_status="simulator-proxy-not-gpu-measured",
        input_tokens_modeled=False,
        context_dependent_latency_modeled=False,
        proxy_parameter_status={
            "draft_latency": "context-independent simulator proxy",
            "verify_latency": "context-independent simulator proxy",
            "acceptance": "workload proxy; not GPU-measured",
            "draft_confidence": "workload proxy; not GPU-calibrated",
            "candidate_roof": "configured simulator proxy",
        },
        simulator_parameters=asdict(config),
        policy=policy.name,
        display_name=policy.display_name,
        execution_mode=policy.execution_mode,
        allocator=policy.allocator,
        base_allocator=getattr(policy, "base_allocator", "not-applicable"),
        residual_selector=getattr(policy, "residual_selector", "not-applicable"),
        eager_semantics=policy.eager_semantics,
        requests=len(results),
        completed_requests=len(results),
        first_arrival_ms=first_arrival_ms,
        last_arrival_ms=last_arrival_ms,
        drain_completion_ms=drain_completion_ms,
        arrival_span_ms=max(0.0, last_arrival_ms - first_arrival_ms),
        processing_and_drain_ms=max(0.0, drain_completion_ms - last_arrival_ms),
        makespan_ms=measurement_ms,
        measurement_ms=measurement_ms,
        raw_generated_tokens=total_tokens,
        slo_good_tokens=good_tokens,
        raw_throughput_tokens_per_s=total_tokens / duration_s if duration_s else 0.0,
        throughput_tokens_per_s=total_tokens / duration_s if duration_s else 0.0,
        goodput_tokens_per_s=good_tokens / duration_s if duration_s else 0.0,
        slo_attainment=(
            sum(result.attained for result in results) / len(results) if results else 0.0
        ),
        p50_tpot_ms=_percentile(tpots, 0.50),
        p90_tpot_ms=_percentile(tpots, 0.90),
        p99_tpot_ms=_percentile(tpots, 0.99),
        mean_queueing_latency_ms=statistics.mean(queueing) if queueing else 0.0,
        mean_service_latency_ms=statistics.mean(service) if service else 0.0,
        mean_decode_latency_ms=statistics.mean(decode) if decode else 0.0,
        p50_queueing_latency_ms=_percentile(queueing, 0.50),
        p50_service_latency_ms=_percentile(service, 0.50),
        p50_decode_latency_ms=_percentile(decode, 0.50),
        normal_drafted_proposals=ledger.normal_drafted_proposals,
        eager_drafted_proposals=ledger.eager_drafted_proposals,
        verified_proposals=ledger.verified_proposals,
        fully_accepted_proposals=ledger.fully_accepted_proposals,
        promoted_proposals=ledger.promoted_proposals,
        invalidated_proposals=ledger.invalidated_proposals,
        discarded_at_eos_proposals=ledger.discarded_at_eos_proposals,
        normal_drafted_tokens=ledger.normal_drafted_tokens,
        eager_drafted_tokens=ledger.eager_drafted_tokens,
        verified_tokens=ledger.verified_tokens,
        accepted_tokens=ledger.accepted_tokens,
        promoted_tokens=ledger.promoted_tokens,
        invalidated_tokens=ledger.invalidated_tokens,
        discarded_at_eos_tokens=ledger.discarded_at_eos_tokens,
        draft_compute_ms=ledger.draft_compute_ms,
        verify_compute_ms=ledger.verify_compute_ms,
        eager_promotions=ledger.eager_promotions,
        eager_invalidations=ledger.eager_invalidations,
        eager_discarded_at_eos_proposals=ledger.eager_discarded_at_eos_proposals,
        eager_invalidated_tokens=ledger.eager_invalidated_tokens,
        eager_discarded_at_eos_tokens=ledger.eager_discarded_at_eos_tokens,
        eager_promotion_proposal_ratio=(
            ledger.promoted_proposals / eager_proposals if eager_proposals else 0.0
        ),
        eager_invalidation_proposal_ratio=(
            ledger.eager_invalidations / eager_proposals if eager_proposals else 0.0
        ),
        eager_eos_discard_proposal_ratio=(
            ledger.eager_discarded_at_eos_proposals / eager_proposals
            if eager_proposals
            else 0.0
        ),
        eager_promotion_token_ratio=(
            ledger.promoted_tokens / eager_tokens if eager_tokens else 0.0
        ),
        eager_invalidation_token_ratio=(
            ledger.eager_invalidated_tokens / eager_tokens if eager_tokens else 0.0
        ),
        eager_eos_discard_token_ratio=(
            ledger.eager_discarded_at_eos_tokens / eager_tokens if eager_tokens else 0.0
        ),
        draft_compute_waste_ratio=(
            (drafted_tokens - ledger.accepted_tokens) / drafted_tokens
            if drafted_tokens
            else 0.0
        ),
        eager_compute_waste_ratio=(
            (eager_tokens - ledger.eager_accepted_tokens) / eager_tokens
            if eager_tokens
            else 0.0
        ),
        root_in_candidate_budget=False,
        root_progress_definition=(
            "one target token committed by each valid non-EOS proposal verification; "
            "counted once and separately from accepted candidate nodes"
        ),
        candidate_roof_definition=(
            "maximum non-root candidate nodes drafted in one cycle; normal and eager "
            "candidate nodes share the configured roof"
        ),
        target_input_positions=(
            "one root target-input position per verified request, outside the candidate roof"
        ),
        verify_latency_inputs={
            "surface": "T_verify(B_req, B_cand, C)",
            "formula": (
                "verify_base_ms + verify_per_request_ms * target_input_positions + "
                "verify_per_candidate_ms * selected_candidate_nodes"
            ),
            "request_count_modeled": True,
            "candidate_node_count_modeled": True,
            "context_length_modeled": False,
            "profiling_requirement": (
                "GPU calibration must sweep request/root positions and candidate "
                "positions jointly across context C"
            ),
        },
        tree_drafted_nodes=ledger.tree_drafted_nodes,
        tree_verified_nodes=ledger.tree_verified_nodes,
        tree_accepted_nodes=ledger.tree_accepted_nodes,
        tree_invalidated_nodes=ledger.tree_invalidated_nodes,
        tree_discarded_at_eos_nodes=ledger.tree_discarded_at_eos_nodes,
        baseline_root_progress=ledger.baseline_root_progress,
        candidate_expected_progress=ledger.candidate_expected_progress,
        candidate_realized_progress=ledger.tree_accepted_nodes,
        mean_candidate_tree_width=(
            ledger.candidate_tree_width_sum / ledger.candidate_tree_samples
            if ledger.candidate_tree_samples
            else 0.0
        ),
        mean_candidate_tree_depth=(
            ledger.candidate_tree_depth_sum / ledger.candidate_tree_samples
            if ledger.candidate_tree_samples
            else 0.0
        ),
        mean_candidate_tree_nodes=(
            ledger.candidate_tree_node_sum / ledger.candidate_tree_samples
            if ledger.candidate_tree_samples
            else 0.0
        ),
        selected_path_probability_distribution={
            "count": float(ledger.selected_path_probability_count),
            "mean": (
                ledger.selected_path_probability_sum
                / ledger.selected_path_probability_count
                if ledger.selected_path_probability_count
                else 0.0
            ),
            "p50": histogram_percentile(0.50),
            "p90": histogram_percentile(0.90),
        },
        allocator_stage_metrics={
            "slo_stage_selected_nodes": ledger.slo_stage_nodes,
            "residual_stage_selected_nodes": ledger.residual_stage_nodes,
            "base_selected_nodes": ledger.base_selected_nodes,
        },
        cycles=cycles,
        requests_completed_per_cycle=len(results) / cycles if cycles else 0.0,
        mean_verify_batch=(
            ledger.verify_request_positions / cycles if cycles else 0.0
        ),
        candidate_roof_utilization=(
            ledger.candidate_roof_used / ledger.candidate_roof_capacity
            if ledger.candidate_roof_capacity
            else 0.0
        ),
        one_cycle_infeasible_opportunity_ratio=(
            ledger.one_cycle_infeasible_opportunities / ledger.allocation_opportunities
            if ledger.allocation_opportunities
            else 0.0
        ),
        stage1_nodes_to_one_cycle_infeasible=(
            ledger.stage1_nodes_to_one_cycle_infeasible
        ),
        stage1_infeasible_node_ratio=(
            ledger.stage1_nodes_to_one_cycle_infeasible / ledger.slo_stage_nodes
            if ledger.slo_stage_nodes
            else 0.0
        ),
        verified_candidate_nodes_per_cycle=(
            ledger.verified_tokens / cycles if cycles else 0.0
        ),
        candidate_committed_tokens_per_cycle=(
            ledger.accepted_tokens / cycles if cycles else 0.0
        ),
        root_progress_per_cycle=(
            ledger.baseline_root_progress / cycles if cycles else 0.0
        ),
        total_progress_per_cycle=(
            (ledger.baseline_root_progress + ledger.accepted_tokens) / cycles
            if cycles
            else 0.0
        ),
        accepted_candidate_tokens_per_verified_node=(
            ledger.accepted_tokens / ledger.verified_tokens
            if ledger.verified_tokens
            else 0.0
        ),
        selected_expected_progress_per_opportunity=(
            ledger.selected_expected_progress_total / ledger.allocation_opportunities
            if ledger.allocation_opportunities
            else 0.0
        ),
        realized_committed_progress_per_opportunity=(
            ledger.realized_committed_progress_total / ledger.allocation_opportunities
            if ledger.allocation_opportunities
            else 0.0
        ),
        base_preservation_violations=ledger.base_preservation_violations,
        cycle_diagnostics=tuple(cycle_diagnostics),
        cycle_diagnostics_truncated=cycle_diagnostics_truncated,
        eager_diagnostics=tuple(
            EagerDiagnostic(
                **event,
                promoted_progress_per_drafted_token=(
                    event["promoted_progress"] / event["eager_budget"]
                    if event["eager_budget"]
                    else 0.0
                ),
            )
            for event in ledger.completed_eager_events
        ),
        eager_diagnostics_truncated=ledger.eager_events_truncated,
        request_allocation_diagnostics=tuple(
            RequestAllocationDiagnostic(
                request_id=result.request_id,
                attained=result.attained,
                **ledger.request_stats.get(
                    result.request_id,
                    {
                        "slo_tpot_ms": result.slo_tpot_ms,
                        "allocated_candidate_nodes": 0,
                        "expected_progress": 0.0,
                        "realized_candidate_progress": 0,
                        "allocation_opportunities": 0,
                    },
                ),
            )
            for result in sorted(results, key=lambda item: item.request_id)
        ),
        allocation_opportunity_diagnostics=tuple(
            ledger.completed_allocation_events
        ),
        allocation_opportunity_diagnostics_truncated=(
            ledger.allocation_events_truncated
        ),
        class_metrics=class_metrics,
        slo_class_metrics=slo_class_metrics,
    )


def _request_view(
    runtime: RuntimeRequest,
    config: SimulatorConfig,
    waiting_time_ms: float,
    *,
    parent: Optional[Proposal] = None,
    candidate_tree: Optional[CandidateTree] = None,
    estimated_next_iteration_latency_ms: float = 0.0,
) -> Optional[RequestView]:
    if runtime.finished:
        return None
    parent_prefix = runtime.committed_prefix_len
    proposal_budget = 0
    parent_full_acceptance_probability = 0.0
    if parent is not None:
        if not _proposal_matches_runtime(runtime, parent) or parent.drafted_tokens <= 0:
            return None
        dependency_path = (
            predicted_dependency_path(parent.candidate_tree, parent.selected_tree)
            if parent.candidate_tree is not None and parent.selected_tree is not None
            else ()
        )
        dependency_length = len(dependency_path) or parent.drafted_tokens
        full_progress = min(
            dependency_length + 1,
            runtime.request.output_tokens - parent.parent_prefix_len,
        )
        parent_prefix = parent.parent_prefix_len + full_progress
        proposal_budget = parent.drafted_tokens
        recent = max(0.0, min(1.0, runtime.recent_acceptance_ratio))
        confidence = max(0.0, min(1.0, runtime.draft_confidence))
        parent_full_acceptance_probability = (
            parent.candidate_tree.by_id[dependency_path[-1]].path_probability
            if dependency_path and parent.candidate_tree is not None
            else recent**parent.drafted_tokens * confidence
        )
    remaining = runtime.request.output_tokens - parent_prefix
    if remaining <= 0:
        return None
    return RequestView(
        request_id=runtime.request.request_id,
        committed_prefix_len=runtime.committed_prefix_len,
        elapsed_decode_ms=runtime.elapsed_decode_ms,
        slo_tpot_ms=runtime.request.slo_tpot_ms,
        recent_acceptance_ratio=runtime.recent_acceptance_ratio,
        draft_confidence=runtime.draft_confidence,
        waiting_time_ms=waiting_time_ms,
        max_budget=min(config.max_request_budget, max(0, remaining - 1)),
        proposal_budget=min(proposal_budget, max(0, remaining - 1)),
        parent_full_acceptance_probability=parent_full_acceptance_probability,
        candidate_tree=candidate_tree or (parent.candidate_tree if parent is not None else None),
        parent_selected_tree=parent.selected_tree if parent is not None else None,
        estimated_next_iteration_latency_ms=estimated_next_iteration_latency_ms,
    )


def _plan(
    policy: SchedulingPolicy,
    normal_runtimes: list[RuntimeRequest],
    eager_parents: list[tuple[RuntimeRequest, Proposal]],
    verify_ms: float,
    config: SimulatorConfig,
    tree_oracle: CandidateTreeOracle,
    previous_cycle_ms: float,
) -> StepPlan:
    estimated_draft_ms = config.draft_per_candidate_ms * min(
        config.roof_candidate_budget,
        sum(
            min(config.max_request_budget, runtime.request.output_tokens)
            for runtime in normal_runtimes
        ),
    )
    estimated_cycle_ms = (
        previous_cycle_ms
        if previous_cycle_ms > 0
        else cycle_latency_ms(policy.execution_mode, estimated_draft_ms, verify_ms)
    )
    # A newly drafted proposal in the alternating dual-batch pipeline is verified in
    # the following slot cycle, so progress is observed after two exposed cycles.
    # Serial execution has no overlapped waiting cycle: its first-order estimate is
    # the complete D + V cycle itself.
    waiting_cycles = 2 if policy.execution_mode == "dual" else 1
    estimated_progress_latency_ms = waiting_cycles * estimated_cycle_ms
    waiting_time_ms = estimated_progress_latency_ms
    normal_views = tuple(
        view
        for runtime in normal_runtimes
        for tree in [
            tree_oracle.tree(
                runtime.request.request_id,
                runtime.committed_prefix_len,
                width=config.candidate_tree_width,
                depth=min(
                    config.candidate_tree_depth,
                    config.max_request_budget,
                    runtime.request.output_tokens - runtime.committed_prefix_len,
                ),
                draft_confidence=runtime.draft_confidence,
            )
        ]
        for view in [
            _request_view(
                runtime,
                config,
                waiting_time_ms,
                candidate_tree=tree,
                estimated_next_iteration_latency_ms=estimated_progress_latency_ms,
            )
        ]
        if view is not None
    )
    eager_views = tuple(
        view
        for runtime, parent in eager_parents
        for view in [
            _request_view(
                runtime,
                config,
                waiting_time_ms,
                parent=parent,
                estimated_next_iteration_latency_ms=estimated_progress_latency_ms,
            )
        ]
        if view is not None
    )
    residual_draft_tokens = (
        math.floor(verify_ms / config.draft_per_candidate_ms)
        if config.draft_per_candidate_ms > 0
        else config.roof_candidate_budget
    )
    snapshot = PolicySnapshot(
        normal_requests=normal_views,
        eager_requests=eager_views,
        roof_candidate_budget=config.roof_candidate_budget,
        residual_draft_tokens=residual_draft_tokens,
    )
    plan = policy.plan(snapshot)
    # Sequence baselines still verify one path through the same deterministic tree.
    if not plan.normal_trees:
        trees = {}
        by_request = {view.request_id: view for view in normal_views}
        for request_id, budget in plan.normal_budgets.items():
            tree = by_request[request_id].candidate_tree
            if tree is not None:
                trees[request_id] = select_sequence_path(tree, budget)
        plan = StepPlan(
            normal_budgets=plan.normal_budgets,
            eager_budgets=plan.eager_budgets,
            normal_trees=trees,
            candidate_trees={
                request_id: by_request[request_id].candidate_tree
                for request_id in trees
                if by_request[request_id].candidate_tree is not None
            },
            eager_dependency_paths=plan.eager_dependency_paths,
            expected_progress={
                request_id: expected_tree_progress(
                    by_request[request_id].candidate_tree, selected
                )
                for request_id, selected in trees.items()
                if by_request[request_id].candidate_tree is not None
            },
            requested_progress_gap={
                request_id: by_request[request_id].continuous_progress_gap
                for request_id in plan.normal_budgets
            },
            slo_stage_budgets={request_id: 0 for request_id in plan.normal_budgets},
            residual_stage_budgets=dict(plan.normal_budgets),
            normal_budget_displaced_by_eager=plan.normal_budget_displaced_by_eager,
            required_total_progress={
                request_id: by_request[request_id].required_total_progress
                for request_id in plan.normal_budgets
            },
            required_candidate_progress={
                request_id: by_request[request_id].required_candidate_progress
                for request_id in plan.normal_budgets
            },
            maximum_attainable_candidate_progress={
                request_id: by_request[
                    request_id
                ].maximum_attainable_candidate_progress
                for request_id in plan.normal_budgets
            },
            maximum_attainable_total_progress={
                request_id: by_request[request_id].maximum_attainable_total_progress
                for request_id in plan.normal_budgets
            },
            one_cycle_feasible={
                request_id: by_request[request_id].one_cycle_feasible
                for request_id in plan.normal_budgets
            },
        )
    normal_ids = {view.request_id for view in normal_views}
    eager_ids = {view.request_id for view in eager_views}
    if set(plan.normal_budgets) - normal_ids:
        raise ValueError(f"policy {policy.name} allocated a non-draftable normal request")
    if set(plan.eager_budgets) - eager_ids:
        raise ValueError(f"policy {policy.name} allocated a non-draftable eager request")
    if plan.eager_budgets and not getattr(policy, "eager_enabled", False):
        raise ValueError(f"policy {policy.name} allocated eager work while eager is disabled")
    if plan.total_candidates + plan.total_eager_candidates > config.roof_candidate_budget:
        raise ValueError(f"policy {policy.name} exceeded the roof candidate budget")
    by_id = {view.request_id: view for view in normal_views}
    eager_by_id = {view.request_id: view for view in eager_views}
    for request_id, budget in plan.normal_budgets.items():
        if (
            not isinstance(budget, int)
            or isinstance(budget, bool)
            or budget < 0
            or budget > by_id[request_id].max_budget
        ):
            raise ValueError(f"policy {policy.name} returned an invalid normal budget")
        base_tree = plan.base_normal_trees.get(request_id)
        selected_tree = plan.normal_trees.get(request_id)
        if base_tree is not None:
            if selected_tree is None or not set(base_tree.selected_node_ids).issubset(
                selected_tree.selected_node_ids
            ):
                raise ValueError(f"policy {policy.name} evicted base candidate nodes")
            if budget < plan.base_normal_budgets.get(request_id, 0):
                raise ValueError(f"policy {policy.name} reduced a base request budget")
    for request_id, budget in plan.eager_budgets.items():
        if (
            not isinstance(budget, int)
            or isinstance(budget, bool)
            or budget <= 0
            or budget > eager_by_id[request_id].max_budget
        ):
            raise ValueError(f"policy {policy.name} returned an invalid eager budget")
    return plan


def _draft_normal(
    runtime: RuntimeRequest,
    budget: int,
    cycle: int,
    config: SimulatorConfig,
    ledger: _LifecycleLedger,
    *,
    candidate_tree: Optional[CandidateTree] = None,
    selected_tree: Optional[SelectedProposalTree] = None,
) -> Proposal:
    if runtime.finished or runtime.normal_proposal is not None:
        raise AssertionError("cannot draft a normal proposal for a finished or occupied request")
    remaining = runtime.request.output_tokens - runtime.committed_prefix_len
    drafted_tokens = min(budget, remaining)
    if (
        candidate_tree is not None
        and selected_tree is not None
        and selected_tree.candidate_budget != drafted_tokens
    ):
        selected_tree = truncate_selected_tree(
            candidate_tree, selected_tree, drafted_tokens
        )
    proposal = Proposal(
        request_id=runtime.request.request_id,
        parent_prefix_len=runtime.committed_prefix_len,
        prefix_epoch=runtime.prefix_epoch,
        budget=budget,
        drafted_tokens=drafted_tokens,
        source="normal",
        drafted_at_cycle=cycle,
        candidate_tree=candidate_tree,
        selected_tree=selected_tree,
        slo_tpot_ms=runtime.request.slo_tpot_ms,
    )
    runtime.normal_proposal = proposal
    ledger.record_drafted(proposal, config.draft_per_candidate_ms)
    return proposal


def _draft_eager(
    runtime: RuntimeRequest,
    parent: Proposal,
    budget: int,
    cycle: int,
    config: SimulatorConfig,
    ledger: _LifecycleLedger,
    dependency_path: tuple[str, ...] = (),
    candidate_tree: Optional[CandidateTree] = None,
    selected_tree: Optional[SelectedProposalTree] = None,
    normal_budget_displaced: int = 0,
    dependency_path_probability: float = 0.0,
) -> Optional[Proposal]:
    if runtime.finished or runtime.eager_proposal is not None:
        raise AssertionError("cannot draft eager work for a finished or occupied request")
    if not _proposal_matches_runtime(runtime, parent) or parent.drafted_tokens <= 0:
        return None
    dependency_length = len(dependency_path) or parent.drafted_tokens
    full_progress = min(
        dependency_length + 1,
        runtime.request.output_tokens - parent.parent_prefix_len,
    )
    anticipated_prefix = parent.parent_prefix_len + full_progress
    remaining = runtime.request.output_tokens - anticipated_prefix
    if remaining <= 0:
        return None
    drafted_tokens = min(budget, remaining)
    if (
        candidate_tree is not None
        and selected_tree is not None
        and selected_tree.candidate_budget != drafted_tokens
    ):
        selected_tree = truncate_selected_tree(
            candidate_tree, selected_tree, drafted_tokens
        )
    proposal = Proposal(
        request_id=runtime.request.request_id,
        parent_prefix_len=anticipated_prefix,
        prefix_epoch=parent.prefix_epoch + 1,
        budget=budget,
        drafted_tokens=drafted_tokens,
        source="eager",
        drafted_at_cycle=cycle,
        dependency_path=dependency_path,
        slo_tpot_ms=runtime.request.slo_tpot_ms,
        candidate_tree=candidate_tree,
        selected_tree=selected_tree,
        dependency_path_probability=dependency_path_probability,
    )
    if proposal.drafted_tokens <= 0:
        return None
    runtime.eager_proposal = proposal
    ledger.record_drafted(proposal, config.draft_per_candidate_ms)
    ledger.eager_events[proposal.key]["normal_budget_displaced"] = normal_budget_displaced
    return proposal


def _proposal_draft_ms(proposals: list[Proposal], config: SimulatorConfig) -> float:
    return sum(proposal.drafted_tokens for proposal in proposals) * (
        config.draft_per_candidate_ms
    )


def _verify_proposal(
    runtime: RuntimeRequest,
    oracle: AcceptanceOracle,
    tree_oracle: CandidateTreeOracle,
    ledger: _LifecycleLedger,
) -> _VerificationOutcome:
    proposal = runtime.normal_proposal
    if proposal is None:
        raise AssertionError("verification requires a stored proposal")
    runtime.normal_proposal = None
    if not _proposal_matches_runtime(runtime, proposal):
        ledger.record_invalidated(proposal)
        return _VerificationOutcome(proposal, False, 0, False)

    accepted_branch: tuple[str, ...] = ()
    verified_progress: Optional[int] = None
    if (
        proposal.candidate_tree is not None
        and proposal.selected_tree is not None
    ):
        tree_outcome = tree_oracle.verify(
            proposal.candidate_tree,
            proposal.selected_tree,
            committed_prefix_len=proposal.parent_prefix_len,
            acceptance_probability=runtime.request.acceptance_probability,
        )
        accepted_branch = tree_outcome.accepted_branch_node_ids
        accepted = len(accepted_branch)
        verified_progress = tree_outcome.committed_progress
    else:
        accepted = oracle.accepted_prefix(
            proposal.request_id,
            proposal.parent_prefix_len,
            proposal.drafted_tokens,
            runtime.request.acceptance_probability,
        )
    remaining = runtime.request.output_tokens - runtime.committed_prefix_len
    accepted = min(accepted, remaining)
    progress = min(
        verified_progress if verified_progress is not None else accepted + 1,
        remaining,
    )
    ledger.baseline_root_progress += int(progress > 0)
    fully_accepted = proposal.drafted_tokens > 0 and accepted == proposal.drafted_tokens
    ledger.record_verified(proposal, accepted)
    root_progress = int(progress > 0)
    ledger.resolve_allocation(
        proposal,
        committed_progress=progress,
        candidate_progress=accepted,
        root_progress=root_progress,
    )
    runtime.committed_prefix_len += progress
    runtime.prefix_epoch += 1
    runtime.finished = runtime.committed_prefix_len >= runtime.request.output_tokens
    if proposal.drafted_tokens > 0:
        observed = accepted / proposal.drafted_tokens
        runtime.recent_acceptance_ratio = (
            0.8 * runtime.recent_acceptance_ratio + 0.2 * observed
        )
    return _VerificationOutcome(
        proposal,
        True,
        accepted,
        fully_accepted,
        accepted_branch,
        progress,
        root_progress,
    )


def _verify_ar(runtime: RuntimeRequest) -> None:
    if runtime.finished:
        raise AssertionError("finished AR request cannot be verified")
    runtime.committed_prefix_len += 1
    runtime.prefix_epoch += 1
    runtime.finished = runtime.committed_prefix_len >= runtime.request.output_tokens


def _resolve_eager(
    runtime: RuntimeRequest,
    outcome: _VerificationOutcome,
    ledger: _LifecycleLedger,
) -> None:
    eager = runtime.eager_proposal
    if eager is None:
        return
    runtime.eager_proposal = None
    if eager_is_promotable(
        runtime,
        outcome.parent,
        eager,
        parent_fully_accepted=outcome.valid_parent and outcome.fully_accepted,
        accepted_branch_node_ids=outcome.accepted_branch_node_ids,
    ):
        if runtime.normal_proposal is not None:
            raise AssertionError("promotion would overwrite a stored normal proposal")
        runtime.normal_proposal = eager
        runtime.slot = 1 - runtime.slot
        ledger.record_promotion(eager)
        return
    if runtime.finished:
        ledger.record_invalidated(eager, at_eos=True)
    else:
        ledger.eager_invalidations += 1
        ledger.record_invalidated(eager)


def _discard_finished_proposals(runtime: RuntimeRequest, ledger: _LifecycleLedger) -> None:
    if not runtime.finished:
        return
    for field_name in ("normal_proposal", "eager_proposal"):
        proposal = getattr(runtime, field_name)
        if proposal is not None:
            ledger.record_invalidated(proposal, at_eos=True)
            setattr(runtime, field_name, None)


def _complete_request(runtime: RuntimeRequest) -> RequestResult:
    if not runtime.finished:
        raise AssertionError("cannot complete an unfinished request")
    tpot = runtime.elapsed_decode_ms / runtime.request.output_tokens
    return RequestResult(
        request_id=runtime.request.request_id,
        task=runtime.request.task,
        output_tokens=runtime.request.output_tokens,
        slo_tpot_ms=runtime.request.slo_tpot_ms,
        queueing_latency_ms=runtime.admitted_at_ms - runtime.request.arrival_time_ms,
        service_latency_ms=runtime.service_latency_ms,
        decode_latency_ms=runtime.elapsed_decode_ms,
        tpot_ms=tpot,
        attained=tpot <= runtime.request.slo_tpot_ms,
    )


def _verify_ms(runtimes: list[RuntimeRequest], config: SimulatorConfig) -> float:
    if not runtimes:
        return 0.0
    candidates = sum(
        runtime.normal_proposal.drafted_tokens
        for runtime in runtimes
        if runtime.normal_proposal is not None
    )
    return (
        config.verify_base_ms
        + config.verify_per_request_ms * len(runtimes)
        + config.verify_per_candidate_ms * candidates
    )


def _snapshot(runtime: RuntimeRequest) -> RuntimeRequestState:
    return RuntimeRequestState(
        request_id=runtime.request.request_id,
        normal_proposal=runtime.normal_proposal,
        eager_proposal=runtime.eager_proposal,
        committed_prefix_len=runtime.committed_prefix_len,
        prefix_epoch=runtime.prefix_epoch,
        finished=runtime.finished,
    )


def cycle_latency_ms(execution_mode: str, draft_ms: float, verify_ms: float) -> float:
    """Return exposed cycle time for the named execution semantics."""

    if draft_ms < 0 or verify_ms < 0:
        raise ValueError("draft and verify latency must be non-negative")
    if execution_mode == "serial":
        return draft_ms + verify_ms
    if execution_mode == "dual":
        return max(draft_ms, verify_ms)
    if execution_mode == "ar":
        return verify_ms
    raise ValueError("execution_mode must be 'ar', 'serial', or 'dual'")


def simulate(
    workload: Workload,
    policy: SchedulingPolicy,
    config: SimulatorConfig,
    *,
    acceptance_oracle: Optional[AcceptanceOracle] = None,
    candidate_tree_oracle: Optional[CandidateTreeOracle] = None,
    cycle_sink: Optional[Callable[[CycleDiagnostic], None]] = None,
    eager_sink: Optional[Callable[[EagerDiagnostic], None]] = None,
    allocation_sink: Optional[Callable[[AllocationOpportunityDiagnostic], None]] = None,
) -> SimulationResult:
    """Run an explicit AR, serial-SD, or dual-batch proposal state machine."""

    execution_mode = getattr(policy, "execution_mode", None)
    if execution_mode not in {"ar", "serial", "dual"}:
        raise ValueError("policy execution_mode must be 'ar', 'serial', or 'dual'")
    oracle = acceptance_oracle or AcceptanceOracle(config.seed, config.max_request_budget)
    tree_oracle = candidate_tree_oracle or CandidateTreeOracle(config.seed)
    if oracle.max_k < config.max_request_budget:
        raise ValueError("acceptance oracle max_k is smaller than max_request_budget")

    pending = list(workload.requests)
    pending_index = 0
    active: dict[str, RuntimeRequest] = {}
    all_runtimes: dict[str, RuntimeRequest] = {}
    completed: list[RequestResult] = []
    ledger = _LifecycleLedger(
        eager_sink=eager_sink, allocation_sink=allocation_sink
    )
    class_diagnostics: dict[str, dict[str, Any]] = {}
    cycle_diagnostics: list[CycleDiagnostic] = []
    cycle_diagnostics_truncated = 0
    now_ms = 0.0
    cycle = 0
    previous_cycle_ms = 0.0
    simulation_start_ms = pending[0].arrival_time_ms if pending else 0.0

    def record_cycle(diagnostic: CycleDiagnostic) -> None:
        nonlocal cycle_diagnostics_truncated
        if cycle_sink is not None:
            cycle_sink(diagnostic)
        if len(cycle_diagnostics) < 10_000:
            cycle_diagnostics.append(diagnostic)
        else:
            cycle_diagnostics_truncated += 1

    def should_materialize_cycle() -> bool:
        return cycle_sink is not None or len(cycle_diagnostics) < 10_000

    def admit_ready() -> None:
        nonlocal pending_index
        while pending_index < len(pending) and len(active) < config.max_active_requests:
            request = pending[pending_index]
            if request.arrival_time_ms > now_ms:
                break
            slot_sizes = [sum(item.slot == slot for item in active.values()) for slot in (0, 1)]
            slot = 0 if slot_sizes[0] <= slot_sizes[1] else 1
            runtime = RuntimeRequest(
                request=request,
                slot=slot,
                admitted_at_ms=now_ms,
                elapsed_decode_ms=now_ms - request.arrival_time_ms,
                recent_acceptance_ratio=request.acceptance_probability,
                draft_confidence=request.draft_confidence,
            )
            active[request.request_id] = runtime
            all_runtimes[request.request_id] = runtime
            pending_index += 1

    def finish_ready() -> None:
        finished_ids = [key for key, runtime in active.items() if runtime.finished]
        for request_id in finished_ids:
            runtime = active.pop(request_id)
            _discard_finished_proposals(runtime, ledger)
            completed.append(_complete_request(runtime))

    while pending_index < len(pending) or active:
        if cycle >= config.max_cycles:
            raise RuntimeError("simulation exceeded max_cycles")
        if not active:
            now_ms = max(now_ms, pending[pending_index].arrival_time_ms)
        admit_ready()
        if not active:
            continue

        if execution_mode == "ar":
            verify_runtimes = [
                runtime
                for runtime in sorted(active.values(), key=lambda item: item.request.request_id)
                if not runtime.finished
            ]
            verify_ms = _verify_ms(verify_runtimes, config)
            ledger.verify_compute_ms += verify_ms
            ledger.verify_request_positions += len(verify_runtimes)
            for runtime in active.values():
                runtime.elapsed_decode_ms += verify_ms
                runtime.service_latency_ms += verify_ms
            for runtime in verify_runtimes:
                _verify_ar(runtime)
            if should_materialize_cycle():
                record_cycle(CycleDiagnostic(
                    cycle=cycle,
                    active_requests=len(active),
                    pending_requests=len(pending) - pending_index,
                    candidate_roof=0,
                    normal_budget=0,
                    eager_budget=0,
                    budget_by_slo_class={},
                    eager_budget_by_slo_class={},
                    draft_latency_ms=0.0,
                    verify_latency_ms=verify_ms,
                    cycle_latency_ms=verify_ms,
                    predicted_cycle_latency_ms=previous_cycle_ms or verify_ms,
                    prediction_error_ms=verify_ms - (previous_cycle_ms or verify_ms),
                    verify_requests=len(verify_runtimes),
                    root_progress=len(verify_runtimes),
                    total_progress=len(verify_runtimes),
                ))
            else:
                cycle_diagnostics_truncated += 1
            now_ms += verify_ms
            finish_ready()
            previous_cycle_ms = verify_ms
        else:
            verify_slot = cycle % 2
            verify_runtimes = [
                runtime
                for runtime in sorted(active.values(), key=lambda item: item.request.request_id)
                if runtime.slot == verify_slot
                and runtime.normal_proposal is not None
                and not runtime.finished
            ]
            normal_runtimes = [
                runtime
                for runtime in sorted(active.values(), key=lambda item: item.request.request_id)
                if runtime.slot != verify_slot
                and runtime.normal_proposal is None
                and not runtime.finished
            ]
            eager_parents = (
                [
                    (runtime, runtime.normal_proposal)
                    for runtime in verify_runtimes
                    if runtime.normal_proposal is not None
                ]
                if policy.eager_enabled
                else []
            )

            verify_ms = _verify_ms(verify_runtimes, config)
            verified_candidates = sum(
                runtime.normal_proposal.drafted_tokens
                for runtime in verify_runtimes
                if runtime.normal_proposal is not None
            )
            if verified_candidates > config.roof_candidate_budget:
                raise AssertionError("stored verify batch exceeded the roof candidate budget")
            ledger.verify_compute_ms += verify_ms
            ledger.verify_request_positions += len(verify_runtimes)
            root_before_cycle = ledger.baseline_root_progress
            accepted_before_cycle = ledger.accepted_tokens
            plan = _plan(
                policy,
                normal_runtimes,
                eager_parents,
                verify_ms,
                config,
                tree_oracle,
                previous_cycle_ms,
            )
            drafted: list[Proposal] = []
            normal_by_id = {
                runtime.request.request_id: runtime for runtime in normal_runtimes
            }
            for runtime in normal_runtimes:
                budget = plan.normal_budgets.get(runtime.request.request_id, 0)
                ledger.record_allocation(
                    runtime.request.request_id,
                    runtime.request.slo_tpot_ms,
                    cycle,
                    budget,
                    plan.requested_progress_gap.get(runtime.request.request_id, 0.0),
                    plan.expected_progress.get(runtime.request.request_id, 0.0),
                    plan.slo_stage_budgets.get(runtime.request.request_id, 0),
                    plan.residual_stage_budgets.get(runtime.request.request_id, budget),
                    plan.base_normal_budgets.get(runtime.request.request_id, 0),
                    plan.required_total_progress.get(runtime.request.request_id, 0.0),
                    plan.required_candidate_progress.get(runtime.request.request_id, 0.0),
                    plan.maximum_attainable_candidate_progress.get(
                        runtime.request.request_id, 0.0
                    ),
                    plan.maximum_attainable_total_progress.get(
                        runtime.request.request_id, 1.0
                    ),
                    plan.one_cycle_feasible.get(runtime.request.request_id, True),
                    runtime.request.output_tokens - runtime.committed_prefix_len,
                )
                selected = plan.normal_trees.get(runtime.request.request_id)
                tree = None
                if selected is not None:
                    tree = plan.candidate_trees.get(runtime.request.request_id)
                drafted.append(
                    _draft_normal(
                        runtime,
                        budget,
                        cycle,
                        config,
                        ledger,
                        candidate_tree=tree,
                        selected_tree=selected,
                    )
                )
            parent_by_id = {
                runtime.request.request_id: parent for runtime, parent in eager_parents
            }
            for runtime, parent in eager_parents:
                budget = plan.eager_budgets.get(runtime.request.request_id)
                if budget is not None:
                    dependency_path = plan.eager_dependency_paths.get(
                        runtime.request.request_id,
                        predicted_dependency_path(parent.candidate_tree, parent.selected_tree)
                        if parent.candidate_tree is not None
                        and parent.selected_tree is not None
                        else (),
                    )
                    anticipated_prefix = parent.parent_prefix_len + (
                        len(dependency_path) or parent.drafted_tokens
                    ) + 1
                    eager_tree = (
                        tree_oracle.tree(
                            runtime.request.request_id,
                            anticipated_prefix,
                            width=1,
                            depth=min(
                                budget,
                                runtime.request.output_tokens - anticipated_prefix,
                            ),
                            draft_confidence=runtime.draft_confidence,
                        )
                        if anticipated_prefix < runtime.request.output_tokens
                        else None
                    )
                    eager_selected = (
                        select_sequence_path(eager_tree, budget)
                        if eager_tree is not None
                        else None
                    )
                    proposal = _draft_eager(
                        runtime,
                        parent,
                        min(budget, config.max_eager_budget),
                        cycle,
                        config,
                        ledger,
                        dependency_path=dependency_path,
                        normal_budget_displaced=plan.normal_budget_displaced_by_eager.get(
                            runtime.request.request_id, 0
                        ),
                        dependency_path_probability=(
                            parent.candidate_tree.by_id[
                                dependency_path[-1]
                            ].path_probability
                            if dependency_path and parent.candidate_tree is not None
                            else 0.0
                        ),
                        candidate_tree=eager_tree,
                        selected_tree=eager_selected,
                    )
                    if proposal is not None:
                        drafted.append(proposal)
            draft_ms = _proposal_draft_ms(drafted, config)

            cycle_ms = cycle_latency_ms(execution_mode, draft_ms, verify_ms)
            budget_by_slo: dict[str, int] = {}
            for request_id, budget in plan.normal_budgets.items():
                slo = f"{normal_by_id[request_id].request.slo_tpot_ms:g}"
                budget_by_slo[slo] = budget_by_slo.get(slo, 0) + budget
            eager_budget_by_slo: dict[str, int] = {}
            for runtime, _ in eager_parents:
                budget = plan.eager_budgets.get(runtime.request.request_id, 0)
                slo = f"{runtime.request.slo_tpot_ms:g}"
                eager_budget_by_slo[slo] = eager_budget_by_slo.get(slo, 0) + budget
            predicted = previous_cycle_ms or cycle_latency_ms(
                execution_mode,
                config.draft_per_candidate_ms * config.roof_candidate_budget,
                verify_ms,
            )
            diagnostic_base_trees = (
                plan.normal_trees
                if policy.name == "dual-batch"
                else plan.base_normal_trees
            )
            base_candidate_nodes = {
                request_id: diagnostic_base_trees[request_id].selected_node_ids
                for request_id in diagnostic_base_trees
            }
            base_work_preserved = all(
                set(nodes).issubset(
                    plan.normal_trees[request_id].selected_node_ids
                )
                for request_id, nodes in base_candidate_nodes.items()
            )
            if not base_work_preserved:
                ledger.base_preservation_violations += 1
            planned_candidates = plan.total_candidates + plan.total_eager_candidates
            ledger.candidate_roof_used += planned_candidates
            if planned_candidates > 0:
                ledger.candidate_roof_capacity += config.roof_candidate_budget
            diagnostic = CycleDiagnostic(
                    cycle=cycle,
                    active_requests=len(active),
                    pending_requests=len(pending) - pending_index,
                    candidate_roof=config.roof_candidate_budget,
                    normal_budget=plan.total_candidates,
                    eager_budget=plan.total_eager_candidates,
                    budget_by_slo_class=budget_by_slo,
                    eager_budget_by_slo_class=eager_budget_by_slo,
                    draft_latency_ms=draft_ms,
                    verify_latency_ms=verify_ms,
                    cycle_latency_ms=cycle_ms,
                    predicted_cycle_latency_ms=predicted,
                    prediction_error_ms=cycle_ms - predicted,
                    verify_requests=len(verify_runtimes),
                    verified_candidate_nodes=verified_candidates,
                    base_request_ids=tuple(sorted(base_candidate_nodes)),
                    base_candidate_nodes=base_candidate_nodes,
                    base_work_preserved=base_work_preserved,
                )
            for runtime in active.values():
                runtime.elapsed_decode_ms += cycle_ms
                runtime.service_latency_ms += cycle_ms
            for runtime in verify_runtimes:
                outcome = _verify_proposal(runtime, oracle, tree_oracle, ledger)
                if runtime.request.request_id in parent_by_id:
                    _resolve_eager(runtime, outcome, ledger)
            root_progress_cycle = ledger.baseline_root_progress - root_before_cycle
            candidate_progress_cycle = ledger.accepted_tokens - accepted_before_cycle
            diagnostic = replace(
                diagnostic,
                committed_candidate_tokens=candidate_progress_cycle,
                root_progress=root_progress_cycle,
                total_progress=root_progress_cycle + candidate_progress_cycle,
            )
            if should_materialize_cycle():
                record_cycle(diagnostic)
            else:
                cycle_diagnostics_truncated += 1
            now_ms += cycle_ms
            finish_ready()
            previous_cycle_ms = cycle_ms

        cycle += 1
        admit_ready()

    ledger.assert_conservation()
    measurement_ms = max(0.0, now_ms - simulation_start_ms)
    completed.sort(key=lambda item: item.request_id)
    final_states = tuple(_snapshot(all_runtimes[key]) for key in sorted(all_runtimes))
    if any(
        not state.finished
        or state.normal_proposal is not None
        or state.eager_proposal is not None
        for state in final_states
    ):
        raise AssertionError("completed simulation retained unfinished or dangling proposal state")
    first_arrival = pending[0].arrival_time_ms if pending else 0.0
    last_arrival = pending[-1].arrival_time_ms if pending else first_arrival
    summary = _build_summary(
        policy,
        completed,
        measurement_ms,
        ledger,
        config,
        first_arrival_ms=first_arrival,
        last_arrival_ms=last_arrival,
        drain_completion_ms=now_ms,
        class_diagnostics=class_diagnostics,
        cycles=cycle,
        cycle_diagnostics=cycle_diagnostics,
        cycle_diagnostics_truncated=cycle_diagnostics_truncated,
    )
    return SimulationResult(
        summary=summary,
        requests=tuple(completed),
        final_states=final_states,
    )
