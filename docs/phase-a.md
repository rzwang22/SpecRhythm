# Phase A: strategy proof protocol

Phase A asks whether the control policy is internally sound under constructed and, later,
calibrated conditions. Simulator-semantics v0.2 establishes proposal causality and accounting. It
does **not** claim that a pure-Python simulator predicts vLLM, SGLang, or GPU performance.

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
- Every drafted proposal reaches exactly one terminal state: verified, invalidated, or discarded
  because the request reached EOS.
- Eager continuations are promoted only if the exact stored parent proposal is fully accepted, its
  prefix epoch matches, and the request has not reached EOS.
- Active but unserved requests accumulate decode time and therefore become more urgent.
- AR and all speculative policies use the same arrivals, lengths, and SLOs. Speculative policies
  query a shared max-K acceptance trace indexed by `(request_id, committed_target_prefix_len)` and
  truncate that trace to their chosen budget.

## Required comparisons

| Label | Execution | Allocation | Eager |
| --- | --- | --- | --- |
| AR | target decode only | zero candidates | no |
| Serial SD | `draft_ms + verify_ms` | SLO-unaware round-robin | no |
| AdaServe flat proxy | `draft_ms + verify_ms` | legacy flat-sequence shaping proxy | no |
| AdaServe tree-aware | `draft_ms + verify_ms` | two-stage path-probability tree selection | no |
| Dual-Batch | `max(draft_ms, verify_ms)` | round-robin, SLO-unaware | no |
| Dual-Batch + Rolling Eager | dual batch | round-robin, SLO-unaware | guarded |
| + shaping | dual batch | two-stage tree-aware | no |
| SpecRhythm | dual batch | two-stage tree-aware | path-dependent guarded |

The CLI also exposes `adaserve-flat-proxy`, `shaping-flat-proxy`, and
`specrhythm-flat-proxy` only to explain superseded results. Tree-aware AdaServe is a controlled
paper-algorithm model with proxy trees and latencies, not a complete AdaServe system reproduction.
See [tree-aware-design.md](tree-aware-design.md).

The cumulative comparisons are `serial-sd → adaserve` for shaping under serial execution,
`serial-sd → dual-batch` for overlap, `dual-batch → dual-eager` for rolling eager,
`dual-batch → shaping` for individual budget shaping, and `shaping → specrhythm` for guarded
eager on top of shaping.

For a fixed logical batch, Serial SD and Dual-Batch call the same SLO-unaware round-robin
allocator and share every budget, roof, oracle, and active-set parameter. Only their exposed
cycle formula differs. Across a timed arrival trace, that formula can change later admission
times and hence the active set; this queueing effect is part of the comparison, not a second
allocator difference.

## Proposal state machine

Each admitted request owns `normal_proposal`, `eager_proposal`, `committed_prefix_len`,
`prefix_epoch`, and `finished` state. A proposal records its request, parent prefix, epoch, budget,
drafted token count, normal/eager source, and draft cycle.

In a dual cycle, the simulator first identifies the stored normal proposals to verify. During that
same logical interval it drafts normal proposals for the other slot and may draft eager
continuations whose parent is being verified. Verification then advances only the committed target
prefix. A fully accepted parent may promote its matching eager proposal into the next verify slot;
partial acceptance, an epoch/prefix mismatch, or EOS invalidates or discards it. Promotion does not
count as verification and cannot duplicate draft compute.

The summary separately reports normal/eager drafted candidates, verified candidates, accepted
candidates, promoted eager candidates, invalidated candidates, EOS-discarded candidates, and
logical draft/verify compute time at both proposal and token granularity. It also reports
queueing latency from arrival to admission, resident service latency, and end-to-end decode
latency. The conservation invariant is:

~~~text
normal_drafted + eager_drafted
  = verified + invalidated + discarded_at_eos
accepted <= verified
promoted <= eager_drafted
~~~

Promoted tokens later enter exactly one terminal state, normally verification. These values are
semantic accounting fields, not measured GPU counters.

Eager admission uses the estimated probability that the entire stored parent proposal will be
accepted, plus a separate draft-confidence signal and progress gap. Long, rejection-heavy
parents are suppressed or receive shorter continuations; eager budget is not copied from the
parent budget. Simulator-semantics v0.2 uses an explicit minimum estimated parent
full-acceptance probability of `0.10`; this is a proxy control threshold to calibrate, not a
paper- or GPU-measured constant.

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

Only after the v0.2 state machine passes review should a future phase replace illustrative values
in configs/simulator.json with measurements from the exact target
model, draft model, GPU count, tensor parallelism, context buckets, and serving engine. Sweep:

- active batch size: 8, 16, 32, 64;
- offered load: 0.5, 0.7, 0.9, 1.0, and 1.1 times measured AR capacity;
- SLO mixture: 6:2:2 plus balanced and all-homogeneous controls;
- trace shape: steady, rate-shift, periodic, high-frequency, and 1 ms microburst;
- acceptance profile: task-specific measured distributions, not a single mean.

Report paired runs with at least five seeds or trace windows. Include mean, bootstrap 95% confidence
interval, and per-SLO-class attainment. A useful Phase-A success criterion is a repeatable goodput
gain over the SLO-unaware dual-batch allocation without reducing tight-class attainment, with a
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

The current simulator does not feed `input_tokens` into latency. Context-dependent latency is
therefore unimplemented, and all draft/verify surfaces, acceptance/confidence inputs, and roof
budgets remain proxy parameters until calibrated on the target GPU and engine.
