# SpecRhythm project status

Last updated: 2026-08-14

Maintenance rule: every code-changing PR updates this file with its scope, status, evidence,
known limitations, and next gate before that PR is considered complete.

## Goal

Build a reproducible path for evaluating SpecRhythm's SLO-aware speculative-decoding policy,
first as a dependency-free semantic simulator and workload harness, then with measured GPU
calibration, and only afterward as a vLLM or SGLang integration. Simulator outputs are used to
reject inconsistent policies and design controlled experiments; they are not GPU performance
claims.

## Roadmap

1. **Workload and provenance:** deterministic Mooncake replay, R3 proxy construction,
   validation, manifests, and raw-data hygiene.
2. **Simulator semantics:** persistent proposal lifecycle, deterministic candidate trees,
   tree-aware AdaServe/SpecRhythm allocators, queueing-aware SLO accounting, path-dependent eager
   continuation, diagnostics, and constructive counterexamples.
3. **GPU calibration:** measure context/batch/candidate-dependent `D(B,K,C)` and `V(B,K,C)`,
   acceptance traces, confidence calibration, and candidate roof on a pinned model/engine/GPU.
4. **Engine prototype:** implement the validated control plane as a narrow plugin/prototype in
   the selected serving engine.
5. **Evaluation:** paired workload sweeps, ablations, class-level SLO attainment, goodput, waste,
   confidence intervals, and failure analysis.

## Pull request progress

| PR | Status | Scope | Evidence / boundary |
| --- | --- | --- | --- |
| [#1 workload-v0.1](https://github.com/rzwang22/SpecRhythm/pull/1) | merged | strict Mooncake replay, R3 proxy config, validator, manifest, fixture tests and docs | workload plumbing only; proxy payload and illustrative acceptance |
| [#2 simulator-semantics-v0.2](https://github.com/rzwang22/SpecRhythm/pull/2) | frozen draft; Phase 2 complete, not merged | proposal lifecycle, deterministic tree oracle, tree-aware allocators, base-preserving residual controls, Phase-2 nested search pools and common-snapshot oracle replay, path-aware eager and accounting | pure-Python proxy and oracle upper bounds only; no deployable oracle, measured search cost, GPU integration, or performance claim |
| [#3 gpu-integration-v0.1](https://github.com/rzwang22/SpecRhythm/pull/3) | draft; Phase 3.0 CPU implementation/readiness review | isolated real-trace schema, immutable checkpoints, GPU/NVIDIA probe, TP validator, Transformers correctness backend, 1D4V serial runner, CUDA benchmark interfaces and server runbook | developed and dry-run tested on a non-NVIDIA Mac; no CUDA result, serving-engine benchmark, Dual-Batch overlap, or speedup claim |

## Phase 3.0: GPU readiness and real-trace runner

Phase 3.0 is stacked on the frozen PR #2 head and does not modify its simulator algorithms or
reported results. The default package remains dependency free. An optional GPU extra pins PyTorch
`>=2.7.1,<2.8` and Transformers `>=4.56.1,<4.57`; Transformers is used as a correctness collector
because stable per-candidate draft logits, entropy, and margin are required. It is not the final
serving engine.

The real trace schema separates selector-visible draft features from target-only labels. Completed
request/cycle records are immutable and independently validatable, making interruption/resume
safe. The Phase 3A runner implements deterministic draft-only, target-only, and serial
draft-then-verify collection. A five-GPU coordinator keeps draft TP=1 on GPU 0 and a persistent
target TP=4 worker group on GPUs 1–4. It does not implement Dual-Batch overlap.

The latency API exposes `T_draft(B_req,N_search,C)`, `T_select(B_req,N_search,B_verify)`,
`T_verify(B_req,B_cand,C,TP)`, and `T_transfer(payload_bytes)` with separate warmup/measured
iterations, CUDA-event/host-wall distributions, memory peaks, and explicit root/search/verify
counts. It has no CPU timing fallback. The current verifier is a full-context correctness path,
not an optimized packed-tree serving kernel, so any future server output remains calibration data
rather than a performance claim until engine integration and validation are complete.

## Phase 1.5: residual selection

Four residual policies freeze the exact same-state Dual-Batch request set, budgets, path nodes,
candidate forest, roof, and deterministic target outcome, then differ only in residual selection:
request round-robin, path probability, current SLO-aware two-stage, or feasibility-gated two-stage.
All runs report zero base-preservation violations, and residual roof utilization is aligned within
0.15 percentage points at 2.75×, 0.06 points at 3.0×, and 0.05 points at 3.25×.

The decisive result is Probability versus Shaping. At 3.0×, probability raises goodput
2146.6→2452.3 tokens/s, raises attainment 0.728→0.816, and lowers mean queueing 4.30→2.43 seconds.
At 3.25×, it raises goodput 1414.4→1587.4, raises attainment 0.457→0.502, and lowers queueing
19.17→15.05 seconds. The two are effectively tied below the knee at 2.75×.

This rejects the current SLO-stage formula as a forward mechanism: filling idle roof helps, while
global path-probability selection is more efficient than the SLO-weighted residual stage under
pressure. The rejected policies remain only for provenance and diagnosis. The next mechanism gate
is candidate selection or Overdraft-and-Prune, not further tuning of these SLO weights.

The scalar candidate roof is explicitly not a GPU capacity claim. The proxy charges both request
root positions and candidate positions, while GPU calibration must measure the joint surface
`T_verify(B_req, B_cand, C)`. Full results and definitions are in
[phase1.5-residual-selection.md](phase1.5-residual-selection.md).

## Phase 2: oracle headroom

Phase 2 leaves every existing policy unchanged and exposes two separate diagnostic commands.
`phase2-replay` performs the primary same-snapshot causal comparison, while `phase2-simulate`
reports secondary end-to-end, fully-hidden-search upper bounds. Neither command is part of the
normal policy order.

The historical target oracle samples its next target child from the tree passed to verification,
so changing pool width would also change ground truth. The isolated Phase-2 oracle instead freezes
the historical target trajectory on the immutable 1× tree, then adds deterministic, prefix-closed
branches. This makes `A_1×` strictly comparable with Residual-Probability and keeps target truth
constant across ratios and selectors. It also means the canonical target is always present in the
1× pool: this proxy can measure selector, cross-request residual-allocation, and full-tree gaps,
but it cannot identify real missing-target or better-drafter pool-coverage headroom.

The primary replay uses 10,000 deterministic, stratified snapshots per load and keeps
`B_verify`, request roots, the target outcome, and the proxy verification surface fixed while
expanding metadata-only `B_search` to 1×/2×/4×/8×. Variants A/B/C/D isolate the current
target-blind selector, within-request target oracle, global residual oracle, and full-tree oracle
ceiling respectively. All are marked diagnostic-only; B/C/D explicitly leak target outcomes and
all ratios assume search is fully hidden. Detailed definitions, sampling coverage, results, and
decision boundaries are in [phase2-oracle-headroom.md](phase2-oracle-headroom.md).

The completion audit reused all valid interrupted-run artifacts and executed only nine missing
3.25× cells. Final coverage is 3/3 common replays with exactly 10,000 corrected-queue snapshots
each, 9/9 references, and 48/48 end-to-end oracle cells. A_1× reproduces Phase-1.5
Residual-Probability at all three loads with exact integer equality and floating-point absolute
tolerance `1e-12`.

On common snapshots at 8×, B−A adds 16.36/17.07/17.55 committed candidates per cycle across
2.75×/3.0×/3.25×; C−B adds 1.69/3.18/4.02; D−C adds 2.49/2.83/3.01. All
360,000 ordered dominance checks pass. A loses 9.26/9.34/9.33 candidates per cycle versus A_1×
because added branches are distractors around a target already fully covered by 1×.

End to end, A's 1×→8× goodput falls 2567.0→1761.1, 2452.3→1114.0, and
1587.4→742.1 tokens/s. B largely removes the selector loss; at 3.25× it reaches
2952.6/.9412 goodput/attainment at 4×. C reaches 3030.2/.9963 and D 3033.7/.9988, with C/D
core outcomes unchanged by ratio because their oracle already finds the 1× target. These are
fully-hidden-search system ceilings, not deployable or GPU-measured performance.

## Phase 1: shaping causal diagnosis

Three opt-in diagnostics were added without changing the default algorithms:
`shaping-feasible`, `shaping-residual`, and `shaping-feasible-residual`. One-cycle feasibility uses
the frozen total progress gap, counts one future root exactly once, and does not label a request
globally unsalvageable. Residual variants freeze the same-state Dual-Batch request set, per-request
budget, selected candidate path, and root opportunities before shaping otherwise-unused roof.

The scoped full-R3 proxy results are:

| Load | Policy | Goodput tok/s | Attainment | Queue s | P90 TPOT ms | Total progress/cycle | Infeasible opportunities | Stage-1 → infeasible |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.75× | Dual-Batch | 2557.9 | 0.990 | 0.36 | 28.4 | 63.61 | 0.055 | 0 |
| 2.75× | shaping | 2566.5 | 0.996 | 0.25 | 26.3 | 69.74 | 0.035 | 336,357 (85.7% of stage 1) |
| 2.75× | feasible | 2565.7 | 0.996 | 0.25 | 26.3 | 69.74 | 0.035 | 0 |
| 2.75× | residual | 2567.5 | 0.998 | 0.19 | 25.0 | 69.62 | 0.027 | 128,113 (92.0%) |
| 2.75× | feasible + residual | 2566.8 | 0.997 | 0.19 | 24.9 | 69.61 | 0.027 | 0 |
| 3.0× | Dual-Batch | 1968.3 | 0.673 | 5.91 | 119.9 | 69.93 | 0.409 | 0 |
| 3.0× | shaping | 1746.8 | 0.632 | 11.50 | 228.3 | 74.60 | 0.425 | 3,597,597 (99.0%) |
| 3.0× | feasible | 1750.3 | 0.633 | 11.28 | 224.3 | 74.64 | 0.424 | 0 |
| 3.0× | residual | 2146.6 | 0.728 | 4.30 | 93.6 | 76.70 | 0.353 | 1,177,804 (99.4%) |
| 3.0× | feasible + residual | 2144.7 | 0.727 | 4.31 | 93.6 | 76.71 | 0.354 | 0 |
| 3.25× | Dual-Batch | 1291.6 | 0.421 | 22.70 | 373.5 | 73.24 | 0.640 | 0 |
| 3.25× | shaping | 1117.4 | 0.394 | 37.43 | 612.5 | 76.81 | 0.655 | 5,421,587 (99.6%) |
| 3.25× | feasible | 1116.2 | 0.394 | 37.29 | 609.8 | 76.87 | 0.654 | 0 |
| 3.25× | residual | 1414.4 | 0.457 | 19.17 | 321.8 | 80.57 | 0.609 | 1,761,091 (99.8%) |
| 3.25× | feasible + residual | 1415.0 | 0.457 | 19.14 | 321.6 | 80.57 | 0.608 | 0 |

The feasible-only guard removes essentially all stage-1 allocation to one-cycle-infeasible
requests but leaves goodput almost unchanged. Preserving Dual-Batch breadth/base opportunities is
the materially positive intervention at 3.0× and 3.25×. Combining the guards adds no material gain
over residual preservation alone. This supports breadth/root opportunity cost as the dominant
modeled cause at the proxy knee; it does not validate a new default or a GPU performance claim.

All residual runs report zero base-preservation violations. Their base trees come from the exact
Dual-Batch allocator and sequence-path materializer on each variant's same cycle state; residual
nodes are prefix closed and never evict base work. Because earlier scheduling choices can make
independent runs reach different later states, the invariant is a same-state counterfactual, not
a claim that cycle IDs across divergent runs always contain identical active request sets.

Detailed summaries remain outside Git under
`SpecRhythm-data/results/simulator-semantics-v0.2/phase1-shaping-diagnosis/`.
The full table, class-good-token breakdown, progress accounting, and utilization-denominator audit
are in [phase1-shaping-diagnosis.md](phase1-shaping-diagnosis.md).

## Superseded flat-proxy evidence

- The superseded flat revision had 80 passing tests. The current tree-aware/Phase-2 revision has
  115 passing tests and is Ruff clean, including prefix closure, Figure 5 selection, width-1
  degeneration, path-dependent eager, goodput denominator, proposal/token/tree-node conservation,
  and both Phase-2 CLI commands. The recovery baseline had 114 tests; one integration test was
  added to exercise `phase2-replay` and `phase2-simulate` through the CLI.
- Full Mooncake R3 proxy replay validates at 12,031 requests for 1×, 2×, 4×, and 8×
  time scales, corresponding to 3.401, 6.802, 13.605, and 27.210 requests/s.
- 1× and 2× remain mostly below the configured proxy capacity; 4× and 8× show large queueing
  delay and SLO violations, confirming that pending time is included.
- These results used a **legacy flat-sequence shaping proxy**. They diagnose that proxy only and
  cannot compare AdaServe's or SpecRhythm's tree-aware algorithms.
- Eager compute-waste ratios are high (roughly 0.76–0.88 across the reported proxy sweeps).
  Confidence, admission threshold, latency surfaces, and roof calibration remain open gates.

Generated workloads, validation reports, and comparison JSON remain in the external data tree
and are not committed.

## Tree-aware capacity-knee evidence

The full R3 proxy sweep covers 2.0×, 2.25×, 2.5×, 2.75×, 3.0×, 3.25×, 3.5×, 3.75×, and 4.0×.
The proxy knee is between 2.75× and 3.0×: shaping attainment falls from 0.996 to 0.632 and mean
queueing rises from 0.25 s to 11.50 s. At 3.0×, tree-aware shaping reaches 1746.8 good tokens/s
versus 1968.3 for Dual-Batch; SpecRhythm reaches 1895.8 versus 2645.2 for Dual-Eager. These are
negative proxy results for the frozen tree-aware control plane, not evidence about GPU execution.

At 3.0×, Dual-Batch to shaping adds 2,050,577/522,522 candidate nodes to the 40/50 ms classes,
but realized accepted-node gains are only 99,604/29,814. It also adds 148,942 nodes to the 150 ms
class while realized progress falls by 24,880. There are 3,814 tight requests that receive more
candidates but still miss SLO, and 385 previously attained 150 ms requests lose attainment.
Goodput falls through both numerator (-216,670 SLO-good tokens) and denominator (+29.68 s
makespan). The allocator transfers budget, but proxy expected progress does not translate into
sufficient realized progress.

At the same 3.0× point, flat→tree goodput changes are 498.3→485.3 for AdaServe proxy→tree,
1747.6→1746.8 for shaping, and 1791.9→1895.8 for SpecRhythm with eager. Tree semantics therefore
change both allocation and eager outcomes; the legacy flat result is retained for provenance but
cannot substitute for the tree-aware control plane.

The residual-score ablation at 3.0× gives identical shaping output for path probability and
urgency × path probability under this workload/configuration (1746.8 good tokens/s). The frozen
default remains urgency × path probability; equality here is a diagnostic result, not a reason to
change it.

The complete eager grid at 3.0× spans budgets 1/2/4 and dependency thresholds 0.1/0.2/0.3/0.5;
no cell is filtered. At threshold 0.1, Dual-Eager goodput is 2593.0/2645.2/2651.1 for budgets
1/2/4, while SpecRhythm is 2208.5/1895.8/1827.5. Raising the threshold to 0.5 removes all
Dual-Eager admissions and nearly all SpecRhythm admissions. In the detailed final default run,
SpecRhythm's admitted dependency paths have slightly higher mean probability than Dual-Eager
(0.273 versus 0.258), but consume more eager tokens (910,391 versus 636,371). Counterfactual
same-cycle allocation attributes 846,688 displaced normal nodes to SpecRhythm eager versus only
96 to Dual-Eager. This rejects the "lower-probability admission" explanation and instead
identifies admission volume and normal-budget displacement as the leading mechanism to inspect.
The default is not changed after observing results. These are control-plane sensitivity findings,
not GPU performance claims.

## Current simulator contract

The comparison order includes AR, Serial SD, retained flat-proxy diagnostics, tree-aware
AdaServe, Dual-Batch, Dual-Batch + Rolling Eager, tree-aware shaping, and SpecRhythm. Serial SD
and Dual-Batch share allocation and differ only in exposed cycle latency. Tree-aware AdaServe implements the paper's two-stage
node selection with proxy trees/latencies; it is not the complete AdaServe system. Exact formulas
and root accounting are frozen in [tree-aware-design.md](tree-aware-design.md).

Serial SD and Dual-Batch produce identical allocations for the same fixed logical batch. Their
different cycle duration may alter later trace admissions and queueing, which is an intended
system-level consequence of overlap.

Every request reports queueing, service, and end-to-end decode latency. Every proposal is tracked
to exactly one terminal state, with proposal/token promotion, invalidation, EOS discard, and
compute-waste ratios.

## Open work and known limitations

- `input_tokens` is stored but not used by the current latency model.
- Context-dependent draft/verify latency is not implemented.
- `D(B,K,C)`, `V(B,K,C)`, acceptance, confidence, and candidate roof are proxy inputs until GPU
  calibration.
- R3 proxy lengths are sampled and are not HumanEval, Alpaca, or CNN/DailyMail payloads.
- No vLLM, SGLang, MineDraft, packed-tree verification kernel, or Dual-Batch GPU runtime is
  integrated. The optional Transformers path is a Phase-3 correctness/calibration backend only.
- No current result may be cited as evidence of real GPU speedup or full AdaServe/SpecRhythm
  reproduction.
