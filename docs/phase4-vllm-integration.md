# Phase 4 vLLM serving integration

Phase 4 is a separate stacked workflow based on the frozen Phase 3 head. It does not change the
simulator or reinterpret Phase 3 correctness-backend measurements as serving latency.

## Phase 4A.0 scope

The framework freeze is vLLM `v0.25.1` at commit
`752a3a504485790a2e8491cacbb35c137339ad34`, in a standalone Python 3.11 environment with PyTorch
2.11.0. The default SpecRhythm package remains dependency free and Python 3.9 compatible.

The target-fair three-GPU layout is:

| Role | Model | Physical GPUs | TP |
| --- | --- | --- | ---: |
| Draft | Qwen3-0.6B | 0 | 1 |
| Target | Qwen3-32B | 1, 2 | 2 |

`target-only`, `serial-disaggregated`, and `dual-batch` are frozen future mode contracts. This
phase brings up the Draft and Target engines independently; it implements none of those
cross-engine execution modes except ordinary target-only generation. vLLM colocated speculative
decoding is not a substitute for either disaggregated mode. vLLM DBO is a within-model-executor
microbatch overlap and is not SpecRhythm Dual-Batch; both built-in speculative decoding and vLLM
DBO are disabled here.

The config's `enforce_eager=true` is vLLM's flag for eager PyTorch execution (CUDA Graphs off)
during transparent bring-up. It is unrelated to the SpecRhythm rolling-Eager mechanism, which is
not implemented in this phase.

## Integration boundary

`specrhythm.phase4` defines dependency-free `DraftEngineAdapter`, `TargetEngineAdapter`,
`CandidateBatch`, `VerificationBatch`, `VerificationResult`, `RequestState`, monotonic events and
greedy-sampling contracts. They can express this future transition without importing simulator
policies or latency proxies:

```text
request state
  -> draft candidate nodes + draft-only metadata
  -> prefix-closed verification batch
  -> target accepted/committed tokens
  -> monotonic request state update
```

The fake adapters exist only for Mac/CI protocol tests. Every fake artifact contains
`fake_data=true`, `gpu_result=false`, and `serving_performance_result=false`.

The GPU command uses stock vLLM `LLM.generate` on a deterministic 3/1/1 task-stratified five-row
subset of already-rendered, already-tokenized requests from the corrected R3-real workload. It
selects the earliest required rows in trace order; it does not create handwritten prompts or
reapply a chat template. Both engines consume the same prompt token IDs, greedy settings,
tokenizer path, maximum new-token values and request seeds. Each engine runs the batch twice and
records token IDs, text, top log probabilities, request timestamps, startup/wall-clock bring-up
timestamps, and per-rank model/device evidence.

For Target, the validator optionally compares every token with the immutable Phase 3 HF target
trajectory. A mismatch is retained as a warning with the first divergent position, both token IDs,
the vLLM top-k row, and the HF model revision. It never rewrites the old target trace.

## Artifact boundary

One server run directory contains:

```text
environment.json
topology.json
probe-validation.json
runtime-manifest.json
draft-smoke.json
target-tp2-smoke.json
validation.json
summary.md
phase4a.log
```

Model files, result JSON, logs and traces stay outside Git under `SpecRhythm-data`. Startup,
prefill, decode and wall-clock timestamps are bring-up observability only. This phase makes no
goodput, SLO, latency-speedup, packed-tree, or end-to-end serving-performance claim.

## Exit gate

Proceed only after the 3×A800 artifacts show the exact vLLM/source/Python/PyTorch freeze, disjoint
physical GPU placement, complete TP ranks with nonzero local parameters/memory, deterministic
five-request greedy outputs, and an explained token-level relationship to the frozen HF target.
The next phase must separately design cross-engine transport and verification; it must not infer
them from these ordinary-generation timestamps.
