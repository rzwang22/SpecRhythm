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
2. **Simulator semantics:** persistent proposal lifecycle, comparable baselines, queueing-aware
   SLO accounting, guarded eager continuation, and constructive counterexamples.
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
| [#2 simulator-semantics-v0.2](https://github.com/rzwang22/SpecRhythm/pull/2) | draft, implementation complete; review pending | proposal lifecycle, prefix oracle, fair baselines, queueing-aware latency, AdaServe-style and cumulative ablations, eager/accounting semantics | pure Python control-plane model; no GPU engine integration or performance claim |

## Latest validation evidence

- Python 3.9: 80 tests pass; Ruff and lifecycle stress checks pass.
- Full Mooncake R3 proxy replay validates at 12,031 requests for 1×, 2×, 4×, and 8×
  time scales, corresponding to 3.401, 6.802, 13.605, and 27.210 requests/s.
- 1× and 2× remain mostly below the configured proxy capacity; 4× and 8× show large queueing
  delay and SLO violations, confirming that pending time is included.
- The proxy sweep is not uniformly favorable: at 4×, SpecRhythm is slightly below shaping-only
  goodput/attainment, and AdaServe-style is below Serial SD at 4× and 8×.
- Eager compute-waste ratios are high (roughly 0.76–0.88 across the reported proxy sweeps).
  Confidence, admission threshold, latency surfaces, and roof calibration remain open gates.

Generated workloads, validation reports, and comparison JSON remain in the external data tree
and are not committed.

## Current simulator contract

The comparison order is AR, Serial SD, AdaServe-style, Dual-Batch, Dual-Batch + Rolling Eager,
Dual-Batch + Individual Budget Shaping, and SpecRhythm. Serial SD and Dual-Batch share allocation
and differ only in exposed cycle latency. AdaServe-style denotes serial execution plus the local
SLO-aware allocator; it is not the complete AdaServe scheduler.

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
