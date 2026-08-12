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
- AR, fixed-budget, MineDraft-like uniform, shaping-only, and full SpecRhythm policies;
- a deterministic dual-batch discrete-event simulator;
- goodput, SLO-attainment, throughput, and TPOT metrics;
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
of a speedup until its latency and acceptance inputs are calibrated on the intended model pair,
GPU topology, and engine. See [docs/phase-a.md](docs/phase-a.md) for the evidence standard.
