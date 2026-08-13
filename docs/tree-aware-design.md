# Tree-aware control-plane specification

This document freezes the simulator algorithm before capacity-knee experiments. Results may
diagnose this definition, but do not retroactively change it. The implementation is a deterministic
control-plane model with proxy latency/confidence inputs, not the AdaServe or SpecRhythm GPU
system.

## Accounting convention

The committed target token is the tree root and provides one unit of baseline progress. It is not
a drafted candidate and is not charged to `roof_candidate_budget`; every summary therefore emits
`root_in_candidate_budget=false`. A selected non-root node consumes exactly one candidate unit.
Expected candidate progress is the sum of selected nodes' path probabilities. Realized verified
progress is the length of the target-accepted branch; target correction/root progress is reported
separately from accepted candidate nodes.

The AdaServe artifact commit `7f07ba2bf98c505c86dd4c1bacb6384454fa37a5`
uses a total batch-token budget that includes one root per request. This repository converts that
to a non-root candidate roof explicitly instead of silently mixing conventions.

The latency proxy uses both kinds of work, without charging either twice:

```text
V = verify_base_ms
    + verify_per_request_ms * target_input_positions
    + verify_per_candidate_ms * selected_candidate_nodes
```

`target_input_positions` is the number of requests verified in the cycle: each contributes one
root/target input position. `selected_candidate_nodes` is the number of stored non-root candidate
nodes in those proposals. Consequently, adding a request can add base verification latency even
when its candidate budget is zero. Candidate-roof utilization uses drafted non-root nodes over
`roof_candidate_budget` for cycles with a non-empty allocation opportunity; pure verification or
idle-slot cycles are excluded. The reported Dual-Batch and shaping utilization values therefore
have the same denominator and exclude roots.

## Frozen one-cycle feasibility diagnostic

For an allocation opportunity, the projected-progress quantity remains:

```text
A_i = max(0, (elapsed_latency_i + estimated_next_cycle_latency) / tpot_slo_i
             - generated_tokens_i)
required_total_progress_i = A_i
required_candidate_progress_i = max(0, A_i - 1)
maximum_attainable_total_progress_i
  = 1 + maximum_expected_candidate_progress_i
one_cycle_feasible_i
  = required_total_progress_i <= maximum_attainable_total_progress_i
```

`A_i` is total required progress and has not already subtracted a future root. Exactly one
guaranteed root is therefore removed when deriving the candidate gap and restored when computing
maximum total progress. It is not subtracted or added anywhere else. The maximum expected
candidate progress is the highest-weight prefix-closed selection within the request's existing
candidate tree and per-request cap. This label only says whether the current projected cycle can
recover the SLO; `one_cycle_infeasible` does not mean globally or permanently unsalvageable.

## Deterministic tree and verifier

`CandidateTreeOracle(seed)` indexes a candidate tree and target branch by
`(request_id, committed_prefix_len)`. Every policy sees the same tree and target outcome. Nodes
carry a conditional token confidence and cumulative path probability. The verifier walks only one
target branch and returns its node IDs plus committed progress. Selection is valid only when every
non-root ancestor is selected. With width 1, the model is exactly sequence speculative decoding.

## AdaServe tree-aware allocation

For request `i`:

```text
A_i = max(0, (L_i + estimated_next_iteration_latency) / tau_i - N_i)
candidate_gap_i = max(0, A_i - 1)          # root progress is separate
A_cap_i = min(candidate_gap_i, sum(path_probability of attainable nodes))
```

Requests are processed in descending continuous `A_i` with request ID as the deterministic tie
break. Stage 1 repeatedly chooses the highest-path-probability eligible node until expected
candidate progress covers `A_cap_i`, budget is exhausted, or the independent cap `n_max_slo` is
reached. Stage 2 globally ranks all remaining prefix-eligible nodes by path probability. It does
not multiply by urgency. The overall per-request cap is `max_request_budget`.

This mirrors the paper's Figure 5 selection rule and artifact structure while retaining proxy
trees and latency surfaces. It is therefore named `adaserve-tree-aware`, not a complete AdaServe
reproduction.

## SpecRhythm tree-aware allocation

The exploratory tree is generated before selection. Node marginal expected progress is its
cumulative path probability. Stage 1 considers requests whose projected candidate gap is positive
and not yet covered. It selects the eligible node maximizing:

```text
(1 + max(0, candidate_gap_i - selected_expected_progress_i))
    * node_path_probability
```

Stage 1 stops when every gap is covered, the roof is exhausted, or no node remains under
`min(n_max_slo, max_request_budget)`. Stage 2 uses the frozen paper-style score:

```text
residual urgency * node expected progress
```

The diagnostic ablation `path-probability` removes residual urgency and is AdaServe-like. The
default `urgency-path-probability` is not changed based on which experiment is favorable. All
ties use score, request ID, then node ID. Prefix closure and `max_request_budget` remain mandatory.

## Phase-1 guarded shaping diagnostics

The following policies are diagnostic ablations only. They do not alter or replace the default
`shaping`, `dual-eager`, or `specrhythm` algorithms.

- `shaping-feasible` changes only stage-1 eligibility: a positive-gap request must also be
  `one_cycle_feasible`. Infeasible requests remain eligible for the unchanged stage 2.
- `shaping-residual` first invokes the same round-robin allocator as `dual-batch` on the same
  snapshot. Its request set, per-request budgets, selected path nodes, and root opportunities are
  frozen. Shaping may only add prefix-closed nodes using unused roof; a zero residual roof is
  exactly the Dual-Batch plan.
- `shaping-feasible-residual` combines frozen base work with feasible-only residual stage 1;
  infeasible requests still participate in residual stage 2.

Every residual plan carries its counterfactual base request IDs and candidate node IDs. Runtime
validation rejects any plan that removes a base node, reduces a base budget, breaks prefix closure,
or exceeds the shared candidate roof. The three variants add no urgency threshold or tuned knob.

## Rolling eager dependency

An eager continuation depends on the deepest, then highest-probability selected root-to-node
path, not the entire selected tree. Its admission probability is that dependency path's cumulative
probability. It may promote exactly when the verified target branch contains the dependency path,
the prefix epoch matches, and EOS has not been reached. Rejection on a non-dependency branch is
irrelevant. Normal and eager budgets are separately emitted even though both consume the shared
candidate roof and draft window. `normal_budget_displaced` is not assumed equal to eager budget:
it is the decrease in same-cycle normal candidates versus a no-eager counterfactual, attributed in
deterministic eager-admission order.

## Cycle prediction

The first allocation uses a one-step cycle estimate computed from the configured roof:

```text
serial = estimated draft + estimated verify
dual   = max(estimated draft, estimated verify)
```

Later allocations use the previous realized cycle latency as the documented first-order estimate.
For projected progress, serial uses one complete `D + V` cycle. An alternating dual-batch proposal
is drafted in the current slot cycle and verified in the next slot cycle, so it uses
`2 * max(D, V)`. Every cycle reports predicted latency, realized latency, and signed prediction
error; the multiplier affects projected-progress latency, not the per-cycle diagnostic.
