# Phase 1.5: base-preserving residual selection

This short diagnostic isolates residual selection after freezing the exact Dual-Batch base plan.
It is a pure-Python R3-proxy experiment, not a GPU result and not a new default policy.

## Controlled methods

| Policy | Frozen base | Residual selector |
| --- | --- | --- |
| `dual-batch` | original Dual-Batch | none |
| `residual-round-robin` | exact Dual-Batch request set, budgets, and path nodes | one node per request per pass; local deterministic probability tie break |
| `residual-probability` | same | global path probability only |
| `shaping-residual` | same | current SLO-aware two-stage selector |
| `feasible-residual` | same | feasibility-gated SLO-aware two-stage selector |

All residual allocators use the same `CandidateTreeOracle(seed=1664)` forest and target outcome,
candidate roof 192, per-request cap 8, active-set semantics, and latency proxy. Constructive tests
compare the four plans on the same snapshot and require identical base request IDs, base candidate
node IDs, and candidate-tree objects. Each final tree must be a prefix-closed superset of its base
tree and must not exceed the roof. All full runs report zero base-preservation violations.

## Results

Queue is mean queueing latency in seconds. `Roof` is the common non-root candidate utilization
metric. `Progress/cycle` is root plus committed candidate progress. `A/V` is accepted candidate
tokens divided by verified candidate nodes.

| Load | Policy | Goodput tok/s | Attainment | Raw tok/s | Queue s | P90 TPOT ms | Roof | Progress/cycle | A/V |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.75× | Dual-Batch | 2557.9 | .990 | 2565.3 | .36 | 28.4 | .543 | 63.61 | .358 |
| 2.75× | Residual-Round-Robin | 2566.9 | .997 | 2569.0 | .22 | 25.5 | .911 | 69.67 | .255 |
| 2.75× | Residual-Probability | 2567.0 | .998 | 2568.4 | .18 | 24.8 | .909 | 69.61 | .256 |
| 2.75× | Shaping-Residual | 2567.5 | .998 | 2568.9 | .19 | 25.0 | .910 | 69.62 | .255 |
| 2.75× | Feasible-Residual | 2566.8 | .997 | 2568.4 | .19 | 24.9 | .909 | 69.61 | .255 |
| 3.0× | Dual-Batch | 1968.3 | .673 | 2746.2 | 5.91 | 119.9 | .598 | 69.93 | .358 |
| 3.0× | Residual-Round-Robin | 2255.5 | .753 | 2770.4 | 3.53 | 80.0 | .953 | 76.98 | .266 |
| 3.0× | Residual-Probability | 2452.3 | .816 | 2781.9 | 2.43 | 60.9 | .953 | 77.22 | .268 |
| 3.0× | Shaping-Residual | 2146.6 | .728 | 2761.8 | 4.30 | 93.6 | .953 | 76.70 | .265 |
| 3.0× | Feasible-Residual | 2144.7 | .727 | 2762.8 | 4.31 | 93.6 | .953 | 76.71 | .265 |
| 3.25× | Dual-Batch | 1291.6 | .421 | 2837.0 | 22.70 | 373.5 | .626 | 73.24 | .358 |
| 3.25× | Residual-Round-Robin | 1478.5 | .470 | 2875.1 | 17.49 | 297.3 | .974 | 80.87 | .272 |
| 3.25× | Residual-Probability | 1587.4 | .502 | 2892.6 | 15.05 | 259.8 | .974 | 81.32 | .275 |
| 3.25× | Shaping-Residual | 1414.4 | .457 | 2865.1 | 19.17 | 321.8 | .974 | 80.57 | .271 |
| 3.25× | Feasible-Residual | 1415.0 | .457 | 2865.1 | 19.14 | 321.6 | .974 | 80.57 | .271 |

Residual utilization is aligned within 0.15 percentage points at 2.75×, within 0.06 points at
3.0×, and within 0.05 points at 3.25×. The key comparison is therefore not explained by a
material roof-fill difference:

| Load | Probability − Shaping goodput | Attainment | Queue | Raw throughput |
| --- | ---: | ---: | ---: | ---: |
| 2.75× | -0.5 tok/s | -0.01 points | -0.01 s | -0.4 tok/s |
| 3.0× | +305.7 tok/s | +8.81 points | -1.87 s | +20.1 tok/s |
| 3.25× | +173.0 tok/s | +4.49 points | -4.12 s | +27.5 tok/s |

## Decision

This is the third hypothesized outcome: **Residual-Probability is materially better than
Shaping-Residual at and above the proxy knee**. Filling otherwise-idle roof is beneficial, and
probability-based candidate selection adds further value, but the current SLO-weighted stage
reduces candidate efficiency and amplifies queueing. Feasibility gating does not repair it.

The current SLO-stage formula is therefore rejected as a forward mechanism. It remains available
only as a named provenance/diagnostic policy; it must not be promoted into a new default or called
`Base-Preserving Residual Shaping`. The next mechanism should investigate candidate selection or
Overdraft-and-Prune rather than tune the rejected SLO-stage weights on these proxy results.

## Roof versus hardware capacity

`roof_candidate_budget` is only a scalar allocation cap on non-root candidate nodes. It is not a
measured hardware-capacity frontier. A request also contributes one root/target input position,
and the current proxy charges both axes:

```text
T_verify = T(B_req, B_cand, C)
proxy V = verify_base
          + verify_per_request * B_req
          + verify_per_candidate * B_cand
```

Thus filling residual candidate roof is not free in this simulator: the added candidates enter
the next verification cost, while all preserved requests retain their root-position cost. The
proxy is internally consistent under its additive, context-independent approximation. GPU
calibration must replace it with a jointly measured surface over request/root positions and
candidate positions across context `C`; profiling only a candidate-node roof is insufficient.

Detailed JSON summaries remain outside Git under
`SpecRhythm-data/results/simulator-semantics-v0.2/phase1.5-residual-selection/`.
