# SpecRhythm project status

Last updated: 2026-08-12

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
| [#2 simulator-semantics-v0.2](https://github.com/rzwang22/SpecRhythm/pull/2) | draft, tree-aware control-plane revision implemented; review pending | proposal lifecycle, deterministic tree oracle, independent AdaServe/SpecRhythm allocators, class/cycle diagnostics, path-aware eager and accounting | 2×-4× capacity-knee and eager-grid evidence complete; pure Python proxy only; no GPU integration or performance claim |

## Superseded flat-proxy evidence

- The superseded flat revision had 80 passing tests. The tree-aware revision has 95 passing tests,
  Ruff clean, including prefix closure, Figure 5 selection, width-1 degeneration, path-dependent
  eager, goodput denominator, and proposal/token/tree-node conservation.
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
- No PyTorch, vLLM, SGLang, MineDraft, or remote-GPU runtime is integrated.
- No current result may be cited as evidence of real GPU speedup or full AdaServe/SpecRhythm
  reproduction.
