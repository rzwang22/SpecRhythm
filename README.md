# SpecRhythm

SpecRhythm is a pure-Python research harness for testing SLO-aware speculative-decoding
policies before integrating them into vLLM or SGLang. The initial release contains no GPU,
PyTorch, vLLM, or SGLang dependency. It is deliberately a control-plane model: measured GPU
profiles will replace the illustrative latency and acceptance parameters in Phase B.

The repository currently provides:

- a canonical JSONL workload schema;
- piecewise-Gamma synthetic arrivals and Mooncake timestamp composition;
- task-conditioned, correlated input/output token lengths;
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

To use a Mooncake trace only for its arrival process while sampling task payloads from the
configured mixture:

~~~bash
specrhythm generate \
  --config configs/synthetic.json \
  --arrival-trace data/raw/conversation_trace.jsonl \
  --time-scale 2.0 \
  --output data/processed/mooncake-composed.jsonl
~~~

To preserve Mooncake's observed timestamp and token-length fields instead:

~~~bash
specrhythm import-mooncake \
  --input data/raw/conversation_trace.jsonl \
  --output data/processed/mooncake-normalized.jsonl
~~~

time-scale=2.0 compresses inter-arrival gaps by two and therefore approximately doubles the
arrival rate. Raw traces, generated workloads, and results are ignored by Git by default.

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
