# SpecRhythm

SpecRhythm is a pure-Python research harness for testing SLO-aware speculative-decoding
policies before integrating them into vLLM or SGLang. The initial release contains no GPU,
PyTorch, vLLM, or SGLang dependency. It is deliberately a control-plane model: measured GPU
profiles will replace the illustrative latency and acceptance parameters in Phase B.

The repository currently provides:

- a canonical JSONL workload schema;
- piecewise-Gamma synthetic arrivals and strict, windowed Mooncake timestamp replay;
- task-conditioned, correlated input/output token lengths;
- JSON validation reports and checksum-based provenance manifests;
- seven explicitly named cumulative-ablation modes from AR through SpecRhythm;
- a deterministic proposal-lifecycle simulator with guarded eager promotion;
- queueing/service/end-to-end latency, goodput, SLO-attainment, and TPOT metrics;
- tests for budget, prefix, determinism, and accounting invariants.

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

`input_tokens` is preserved in workloads but is not yet an input to the latency surface.
Context-dependent latency is not implemented. Until GPU calibration, `D(B,K,C)`, `V(B,K,C)`,
acceptance, confidence, and the candidate roof are simulator/proxy parameters only.
The current eager full-parent admission threshold (`0.10`) is also an explicit proxy parameter,
not a measured system constant.

Full per-cycle diagnostics are intentionally streamed outside Git:

```bash
specrhythm simulate --workload /path/to/r3.jsonl --config configs/simulator.json \
  --policy specrhythm --output /external/results/summary.json \
  --cycle-output /external/results/cycles.jsonl \
  --eager-output /external/results/eager-proposals.jsonl
```

The summary retains at most 10,000 cycle/eager detail rows and reports truncation counts; all
class histograms, means, accounting totals, goodput, and throughput remain exact online aggregates.

Project goals, roadmap, semantic boundaries, and per-PR progress are maintained in
[docs/project-status.md](docs/project-status.md).
