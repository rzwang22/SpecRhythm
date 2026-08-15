# SpecRhythm

SpecRhythm is a pure-Python research harness for testing SLO-aware speculative-decoding
policies before integrating them into vLLM or SGLang. The default install remains dependency
free. Phase 3 adds an isolated, optional PyTorch/Transformers correctness collector and GPU
calibration interface; it does not change the control-plane simulator or claim serving-engine
performance. Phase 4A adds a separate Python 3.11/vLLM v0.25.1 integration workflow for stock
engine bring-up; vLLM is not installed into the default simulator environment.

The repository currently provides:

- a canonical JSONL workload schema;
- piecewise-Gamma synthetic arrivals and strict, windowed Mooncake timestamp replay;
- task-conditioned, correlated input/output token lengths;
- JSON validation reports and checksum-based provenance manifests;
- seven explicitly named cumulative-ablation modes from AR through SpecRhythm;
- a deterministic proposal-lifecycle simulator with guarded eager promotion;
- queueing/service/end-to-end latency, goodput, SLO-attainment, and TPOT metrics;
- tests for budget, prefix, determinism, and accounting invariants.
- a versioned real-model trace schema, resumable checkpoints, GPU environment probe, conservative
  TP validator, and CUDA-only latency interfaces.
- a deterministic R3-real public-text pilot builder, nested real candidate-forest checkpoints,
  immutable target labels, and fixed-budget offline selector diagnosis.
- dependency-free Draft/Target adapter contracts plus strict stock-vLLM environment, topology,
  rank-placement, determinism, token-comparison, and bring-up validation tools.

## Quick start

Python 3.9 or newer is sufficient.

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[dev]'

specrhythm generate \
  --config configs/synthetic.json \
  --output data/processed/synthetic.jsonl

specrhythm compare \
  --workload data/processed/synthetic.jsonl \
  --config configs/simulator.json \
  --output results/phase-a.json

pytest
ruff check .
~~~

To build the R3 proxy from a chronological Mooncake window while sampling illustrative payload
lengths from the 6:2:2 code/chat/summarization mixture:

~~~bash
specrhythm generate \
  --config configs/workloads/r3-mooncake-622-proxy.json \
  --arrival-trace data/raw/conversation_trace.jsonl \
  --window-start-ms 0 \
  --window-duration-ms 600000 \
  --time-scale 2.0 \
  --source-commit-sha MOONCAKE_COMMIT_SHA \
  --output data/processed/r3-proxy.jsonl \
  --manifest data/manifests/r3-proxy.manifest.json

specrhythm validate \
  --workload data/processed/r3-proxy.jsonl \
  --config configs/workloads/r3-mooncake-622-proxy.json \
  --arrival-trace data/raw/conversation_trace.jsonl \
  --window-start-ms 0 \
  --window-duration-ms 600000 \
  --time-scale 2.0 \
  --output data/manifests/r3-proxy.validation.json
~~~

The replay selects the end-exclusive window `[start, start + duration)`, sorts its source
timestamps chronologically, rebases the first selected timestamp to zero, and emits exactly one
request per selected timestamp. Do not use `configs/synthetic.json` for R3 replay because that
configuration deliberately models additional conversation turns.

To preserve Mooncake's observed timestamp and token-length fields instead:

~~~bash
specrhythm import-mooncake \
  --input data/raw/conversation_trace.jsonl \
  --output data/processed/mooncake-normalized.jsonl
~~~

`time-scale=2.0` compresses inter-arrival gaps by two and therefore approximately doubles the
arrival rate. The R3 proxy is a plumbing, SLO-shaping, and policy-logic test only: it does not
contain HumanEval, Alpaca, or CNN/DailyMail payloads, and its acceptance probabilities are not GPU
measurements. Raw traces, generated workloads, manifests, and results should remain outside Git.

## Repository map

~~~text
configs/                 Reproducible workload and simulator inputs
docs/phase-a.md          Hypotheses, ablations, and proof gates
docs/workload-design.md  Dataset construction and validation protocol
docs/development.md      Mac/GitHub/remote-GPU workflow
docs/phase3-gpu-runbook.md  Exact Phase-3 server commands and safety boundaries
docs/phase4-vllm-server-runbook.md  Exact independent vLLM bring-up commands
src/specrhythm/          Workload, policy, simulation, and CLI code
tests/                   Unit and integration tests
~~~

The simulator is useful for rejecting bad policies and checking invariants. It is not evidence
of a speedup or a GPU performance predictor. The current latency surfaces and acceptance inputs
remain illustrative; engine integration and performance claims require a later measured model.
See [docs/phase-a.md](docs/phase-a.md) for the evidence standard.

## Simulator mode contract

| Mode | Execution | Allocation | Eager |
| --- | --- | --- | --- |
| `ar` | target-only full active batch | none | no |
| `serial-sd` | serial `D + V` | SLO-unaware round-robin | no |
| `adaserve-flat-proxy` | serial `D + V` | legacy flat-sequence shaping proxy | no |
| `adaserve` | serial `D + V` | AdaServe tree-aware control-plane allocator | no |
| `dual-batch` | dual `max(D,V)` | SLO-unaware round-robin | no |
| `dual-eager` | dual `max(D,V)` | SLO-unaware round-robin | guarded rolling |
| `shaping-flat-proxy` | dual `max(D,V)` | legacy flat-sequence shaping proxy | no |
| `shaping` | dual `max(D,V)` | SpecRhythm tree-aware shaping | no |
| `shaping-feasible` | dual `max(D,V)` | shaping with one-cycle-feasible stage 1 | no |
| `residual-round-robin` | dual `max(D,V)` | frozen Dual-Batch base + request-round-robin fill | no |
| `residual-probability` | dual `max(D,V)` | frozen Dual-Batch base + path-probability fill | no |
| `shaping-residual` | dual `max(D,V)` | frozen Dual-Batch base + residual shaping | no |
| `feasible-residual` | dual `max(D,V)` | frozen base + feasible residual shaping | no |
| `shaping-feasible-residual` | dual `max(D,V)` | frozen base + feasible residual stage 1 | no |
| `specrhythm-flat-proxy` | dual `max(D,V)` | legacy flat-sequence shaping proxy | guarded rolling |
| `specrhythm` | dual `max(D,V)` | SpecRhythm tree-aware shaping | path-dependent rolling |

`serial-sd` and `dual-batch` intentionally share request ordering, per-request budgets,
candidate roof, maximum budget, acceptance trace, and active-set limit. Their only execution
difference for a fixed logical batch is `D + V` versus `max(D,V)`. On an arrival trace, that
wall-clock difference can legitimately change when later requests enter the active set; both
modes still apply the same selection and allocation rules. `adaserve-flat-proxy` preserves the
old linear allocator for diagnostics only. `adaserve` uses an independent tree-aware allocator,
but candidate trees, latency, confidence, and roof remain proxies; it is not a complete AdaServe
reproduction. The exact frozen formulas are in
[docs/tree-aware-design.md](docs/tree-aware-design.md).

The three `shaping-*` additions above are Phase-1 causal diagnostics, not proposed defaults.
`one_cycle_infeasible` means only that the frozen projected cycle cannot recover the SLO; it does
not classify a request as permanently unsalvageable. Residual variants preserve the complete
same-state Dual-Batch base plan before using otherwise-unused candidate roof.

Phase 1.5 aligns residual roof utilization and finds that `residual-probability` materially
outperforms `shaping-residual` at 3.0× and 3.25×. The current SLO-stage formula is therefore kept
only as a diagnostic/provenance path, not promoted as a forward mechanism. See
[docs/phase1.5-residual-selection.md](docs/phase1.5-residual-selection.md).

Phase 2 is exposed through separate diagnostic commands, not through the normal policy order or
cumulative ablation. It replays common Residual-Probability snapshots with nested search pools and
target-leaking oracle ceilings:

```bash
specrhythm phase2-replay --workload /external/r3-3.0x.jsonl \
  --config configs/simulator.json --sample-size 10000 \
  --output /external/phase2/r3-3.0x.json \
  --snapshot-output /external/phase2/r3-3.0x-snapshots.jsonl.gz

specrhythm phase2-simulate --workload /external/r3-3.0x.jsonl \
  --config configs/simulator.json --variant oracle-global-residual \
  --search-ratio 4 --output /external/phase2/r3-3.0x-c-4x.json
```

All Phase-2 variants are diagnostic only. Ratios above 1 assume fully hidden search and do not
include a measured large-pool draft latency. See
[docs/phase2-oracle-headroom.md](docs/phase2-oracle-headroom.md).

## Phase 3 GPU-readiness boundary

Phase 3A uses an optional Transformers `>=4.56.1,<4.57` correctness backend with PyTorch
`>=2.7.1,<2.8`. This backend was selected because the trace requires draft logits, entropy,
top-1/top-2 margin, and stable per-node features that serving-engine public APIs do not expose as
a stable contract. vLLM/SGLang integration remains a later engine-prototype step.

The real trace keeps draft-side selector features separate from target-side labels, writes one
immutable file per completed request/cycle, and resumes without replacing completed records. The
runner supports `draft-only`, `target-only`, and serial draft-then-verify modes. The reference
five-GPU configuration uses draft GPU 0 at TP=1 and a persistent target worker group on GPUs 1–4
at TP=4. A three-GPU fallback uses draft GPU 0 at TP=1 and target GPUs 1–2 at TP=2; it is captured
in `configs/phase3_trace_1d2v.yaml` and `configs/phase3_latency_1d2v.yaml`. There is no Dual-Batch
overlap in this revision.

The Phase 3B.1 CUDA microbenchmark records raw per-rank CUDA-event and synchronized host-wall
samples, model/device/memory/forward evidence, max-rank critical-path latency, observed hardware
state, bidirectional bare-copy metadata, strict validation, and same-commit repeated-run
comparisons. The current Transformers verifier recomputes full contexts without KV-cache reuse and
is not a packed-tree serving kernel. Every report forbids simulator-surface use and is a
correctness-backend primitive measurement, not a vLLM/SGLang throughput claim. CPU dry-runs test
schemas and the selector-stage contract but never emit synthetic GPU latency. See
[docs/phase3-gpu-runbook.md](docs/phase3-gpu-runbook.md) before using a GPU host.

Phase 3C.1/3C.2 use `configs/phase3c_r3_real_1d2v.yaml` for a three-GPU R3-real
selector pilot. Public prompts, Mooncake arrivals, Qwen token IDs, candidate forests and target
trajectories stay outside Git.
The pipeline is split into workload build, draft forest, target trajectory, label join, selector
replay, validation and summary commands so every expensive stage can resume independently.
Its output diagnoses fixed-denominator target-path recall, pool density, shell utilization,
multi-round acceptance and target-blind selector regret against a within-request oracle; it never
reports goodput, SLO attainment or GPU speedup. See
[docs/phase3-real-trace.md](docs/phase3-real-trace.md) for its evidence boundary.

The completed audit contains 3/3 common replays (10,000 corrected-queue snapshots each), 9/9
references, and all 48 end-to-end cells. A_1× exactly reproduces Residual-Probability at all
three loads. The dominant oracle gap is within-request candidate selection; however, the canonical
target is already fully covered by the frozen 1× pool, so this experiment cannot identify real
missing-target coverage or claim that 8× search is free.

`input_tokens` is preserved in workloads but is not yet an input to the latency surface.
Context-dependent latency is not implemented. Until GPU calibration, `D(B,K,C)`, `V(B,K,C)`,
acceptance, confidence, and the candidate roof are simulator/proxy parameters only.
The current eager full-parent admission threshold (`0.10`) is also an explicit proxy parameter,
not a measured system constant.
The candidate roof constrains only non-root nodes; it is not a measured hardware frontier. GPU
profiling must measure `T_verify(B_req, B_cand, C)` jointly rather than sweeping candidate nodes
alone.

Full per-cycle diagnostics are intentionally streamed outside Git:

```bash
specrhythm simulate --workload /path/to/r3.jsonl --config configs/simulator.json \
  --policy specrhythm --output /external/results/summary.json \
  --cycle-output /external/results/cycles.jsonl \
  --eager-output /external/results/eager-proposals.jsonl \
  --allocation-output /external/results/allocation-opportunities.jsonl
```

The summary retains at most 10,000 cycle/eager/allocation detail rows and reports truncation
counts; full streams can remain outside Git. All class histograms, feasibility/stage totals,
progress rates, accounting totals, goodput, and throughput remain exact online aggregates.

Project goals, roadmap, semantic boundaries, and per-PR progress are maintained in
[docs/project-status.md](docs/project-status.md). The scoped Phase-1 causal results are in
[docs/phase1-shaping-diagnosis.md](docs/phase1-shaping-diagnosis.md).
