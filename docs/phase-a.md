# Phase A: strategy proof protocol

Phase A asks whether the control policy is internally sound and produces a measurable benefit
under calibrated conditions. It does **not** claim that a pure-Python simulator predicts vLLM or
SGLang performance.

## Falsifiable hypotheses

1. **Overlap hypothesis:** a dual-batch execution structure reduces exposed draft time once the
   verification window is large enough to hide useful drafting work.
2. **Eager hypothesis:** guarded eager continuation reduces projected progress gaps for tight-SLO
   requests, and its useful promotions exceed invalidated continuations by enough to improve
   goodput.
3. **Shaping hypothesis:** under the same roofline candidate budget, SLO- and
   acceptance-aware allocation produces more SLO-valid tokens than uniform allocation.

The third hypothesis is the cleanest Phase-A target. If it fails after calibration, engine
integration should stop until the policy or workload assumptions are corrected.

## Policy invariants

- The sum of verified candidates never exceeds the profiled roofline budget.
- A request never receives more than its per-request cap.
- Candidate depth d + 1 is allocated only after depth d, preserving prefix dependencies.
- Eager continuations become ready only after the entire dependency prefix is accepted.
- Active but unserved requests accumulate decode time and therefore become more urgent.
- AR and all speculative policies use the same arrivals, lengths, SLOs, and deterministic
  per-request acceptance stream.

## Required comparisons

| Label | Execution | Allocation | Eager |
| --- | --- | --- | --- |
| AR | dual-slot control | zero candidates | no |
| Fixed SD | dual-slot control | fixed per request | no |
| MineDraft-like | dual batch | round-robin to the same roof budget, SLO-unaware | no |
| + shaping | dual batch | two-pass SLO-aware | no |
| SpecRhythm | dual batch | two-pass SLO-aware | guarded |

The CLI exposes every row, including the shaping-only ablation.

## Proof gates

### A1 — logic and accounting

Unit tests must cover budget limits, prefix allocation, urgency ordering, deterministic workload
generation, guarded eager promotion, and goodput accounting. These tests establish correctness of
the model, not performance.

### A2 — constructed counterexamples

Run small scenarios where the expected decision is obvious:

- equal acceptance, unequal SLO: the tighter request must receive at least as much budget;
- equal SLO, unequal acceptance: residual budget should favor the higher-value prefix;
- zero roof budget: every policy reduces to one verified target token per selected request;
- rejection-heavy request: eager invalidations must rise and its allocation must contract;
- homogeneous SLO: shaping should not manufacture a large advantage.

### A3 — calibrated workload sweep

Replace illustrative values in configs/simulator.json with measurements from the exact target
model, draft model, GPU count, tensor parallelism, context buckets, and serving engine. Sweep:

- active batch size: 8, 16, 32, 64;
- offered load: 0.5, 0.7, 0.9, 1.0, and 1.1 times measured AR capacity;
- SLO mixture: 6:2:2 plus balanced and all-homogeneous controls;
- trace shape: steady, rate-shift, periodic, high-frequency, and 1 ms microburst;
- acceptance profile: task-specific measured distributions, not a single mean.

Report paired runs with at least five seeds or trace windows. Include mean, bootstrap 95% confidence
interval, and per-SLO-class attainment. A useful Phase-A success criterion is a repeatable goodput
gain over MineDraft-like allocation without reducing the tight-class attainment, accompanied by a
bounded eager invalidation ratio. The threshold should be chosen before observing final results.

## GPU calibration contract

The remote profiler should emit compact records with at least:

~~~text
engine_version, model_pair, gpu_type, gpu_count, tp_size,
batch_size, context_bucket, candidate_tokens,
draft_ms, verify_ms, accepted_prefix_length,
draft_confidences, request_id, task, seed
~~~

From these records, derive the roofline budget and empirical conditional distributions used by the
simulator. Never infer speculative acceptance from input/output lengths alone.
