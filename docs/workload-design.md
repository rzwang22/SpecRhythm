# Workload construction protocol

The workload should be a versioned data product with four separable layers. This prevents arrival
statistics, prompt semantics, SLO assignment, and speculative acceptance from being accidentally
coupled.

## 1. Four-layer model

1. **Arrival layer:** relative timestamps, client/session identity, and model route.
2. **Payload layer:** task or dataset ID, tokenizer-specific input length, requested/observed output
   length, and optional reusable-prefix metadata.
3. **SLO layer:** TPOT target and its provenance (task contract or AR-baseline multiplier).
4. **Speculation layer:** per-step draft confidence, accepted prefix length, model pair, context,
   batch size, and measured draft/verify latency.

The public serving traces do not contain layer 4. It must be collected on the remote GPU for the
exact model pair. A synthetic acceptance probability is suitable only for unit tests and sensitivity
analysis.

## 2. Canonical record

The repository's JSONL schema includes:

~~~json
{
  "request_id": "request-0000001",
  "arrival_time_ms": 1234.5,
  "input_tokens": 512,
  "output_tokens": 128,
  "slo_tpot_ms": 40,
  "task": "code",
  "model": "default",
  "client_id": "client-0001",
  "conversation_id": null,
  "turn_index": null,
  "acceptance_probability": 0.7,
  "metadata": {"source": "mooncake-fast25"}
}
~~~

Use relative time and retain a provenance manifest containing source URL, upstream commit SHA,
download date, tokenizer, transformation command, seed, and output checksum.

## 3. What each reference contributes

### SpecRhythm

The paper replays and rate-rescales production timestamps, then samples three task classes: HumanEval
code completion at 40 ms/token, Alpaca chat at 50 ms/token, and CNN/DailyMail summarization at
150 ms/token. Its default mixture is 6:2:2 and it evaluates decode time after prefill has completed.
This is the primary reproduction workload, but it must be supplemented with acceptance and latency
profiles.

### AdaSpec

[AdaSpec](https://arxiv.org/pdf/2503.05096) selects 10-minute moderate-rate windows representing
large rate shifts, periodic fluctuation, and rapid oscillation. It samples 80 examples from each of
six SpecBench tasks and defines its TPOT SLO from the P90 AR TPOT, scaled by 0.8–1.2. Use this as a
robustness family: fixed task SLOs and baseline-derived SLOs answer different questions and should
be reported separately.

### ServeGen

[ServeGen](https://www.usenix.org/system/files/nsdi26-xiang-servegen.pdf) explicitly separates a
trace (arrival timestamps) from a dataset (request attributes), then composes workloads per client.
Its production study shows shifting rate and CV, client heterogeneity, rate–length correlation, and
conversation inter-turn effects. Therefore, do not globally shuffle lengths onto timestamps for the
final dataset. Preserve or synthesize clients first, preserve conversation ITTs, and aggregate the
client streams last.

### FineServe

[FineServe](https://arxiv.org/pdf/2607.19349) uses per-model replay or parametric synthesis. Its
parametric arrival model fits Gamma inter-arrivals in 300-second windows and adds Negative-Binomial
counts in 1 ms slots for microburst-heavy groups. Payloads use log-normal input distributions and
architecture/task-aware input–output dependence. This motivates a two-timescale arrival generator
and conditional, rather than independent, length sampling.

### Mooncake FAST'25 traces

The [Mooncake release](https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release/traces)
contains conversation, tool/agent, and public-dataset synthetic JSONL traces. Records provide a
relative millisecond timestamp, input/output token counts, and remapped 512-token prefix-block hash
IDs. The conversation and tool/agent traces cover one hour; the synthetic trace uses Poisson arrivals
while preserving multi-turn order. Download upstream data into data/raw/; do not vendor it here.
The upstream release reports:

| Trace | Requests | Mean input | Mean output | Arrival type |
| --- | ---: | ---: | ---: | --- |
| conversation | 12,031 | 12,035 | 343 | production timestamps |
| tool/agent | 23,608 | 8,596 | 182 | production timestamps |
| synthetic | 3,993 | 15,325 | 149 | Poisson |

The hash IDs describe reusable KV-prefix blocks; they are not client or conversation identifiers.

## 4. Dataset families

Build and freeze the following families rather than one monolithic trace:

| Family | Arrival source | Payload source | Purpose |
| --- | --- | --- | --- |
| R0 | controlled Poisson | task-conditioned public data | sanity baseline |
| R1 | piecewise Gamma | same payloads | long-timescale burst/rate shift |
| R2 | Gamma + 1 ms NB | same payloads | microburst and batch pressure |
| R3 proxy | Mooncake replay | sampled proxy lengths | workload/shaping/logic validation |
| Final R3 | Mooncake replay | HumanEval/Alpaca/CNN-DM | main SpecRhythm reproduction |
| R4 | Mooncake replay | Mooncake lengths/hashes | long-context/prefix sensitivity |
| R5 | client/session synthesis | task-conditional payloads | ServeGen-style causal realism |

For the committed R3 proxy configuration, use Mooncake timestamps only and sample illustrative
token lengths according to the 6:2:2 mixture. It has the task SLOs 40/50/150 ms, but it is not the
final R3 because no HumanEval, Alpaca, or CNN/DailyMail records have been materialized or tokenized.
Its acceptance probabilities are explicitly illustrative and must be replaced by measurements for
the exact draft/target model pair on the remote GPU. For final R3, replace the proxy payload layer
with those public datasets while retaining the selected arrival window. For R4, preserve
Mooncake's joint timestamp, input/output length, and prefix-hash fields. Mixing these
interpretations would invalidate conclusions.

## 5. Strict R3 proxy replay

`specrhythm generate --arrival-trace ...` uses strict replay. It reads the timestamp field, orders
timestamps chronologically, selects the end-exclusive interval
`[window-start-ms, window-start-ms + window-duration-ms)`, rebases the first selected timestamp to
zero, and divides inter-arrival gaps by `time-scale`. It never synthesizes continuation turns and
the output record count must equal the selected timestamp count. The dedicated
`configs/workloads/r3-mooncake-622-proxy.json` sets conversation start probability to zero and uses
deterministic stratified assignment so a 30-request workload has exactly 18/6/6 tasks.

The validator reads the output file in its original order. It checks syntax, identifiers, arrival
ordering and replay correspondence, positive token/SLO values, acceptance bounds, task and SLO
mixtures, request count, rate, time range, and IAT CV. A validation failure returns a non-zero exit
code. The manifest binds source trace, configuration, and output with SHA256 digests and records the
trace commit, replay window, scale, command, software versions, and generation time.

## 6. Scenario matrix

Cross each dataset family with:

- offered load relative to measured AR capacity: 0.5, 0.7, 0.9, 1.0, 1.1;
- SLO assignment: task-coupled 40/50/150, independently permuted control, homogeneous controls,
  and AR-P90 multipliers 0.8/1.0/1.2;
- batch cap: 8, 16, 32, 64;
- task mix: 6:2:2, balanced, code-heavy, and summarization-heavy;
- model-pair acceptance profile and context-length bucket.

The independently permuted SLO control is important: it tests whether the strategy benefits from
SLO urgency itself rather than merely exploiting task-correlated acceptance.

## 7. Splits and validation

Split chronological source traces into calibration, validation, and held-out test windows. Do not
randomly split individual rows, because adjacent requests share rate regimes, clients, and often
conversation state.

Validate generated workloads at multiple timescales:

- inter-arrival CDF and CV in 5-minute windows;
- request-count mean/variance and zero/one/many probabilities in 1 ms slots;
- per-second rate autocorrelation and periodicity;
- client rate share, client CV, and rate–payload correlation;
- session length and inter-turn-time distributions;
- input/output marginal quantiles and their joint conditional curve;
- per-task, per-model, and per-SLO mixture;
- concurrency and active-batch distributions under an AR reference server;
- confidence calibration and accepted-prefix distributions by task/context/depth.

The generator passes only if it matches the intended characteristics at both macro and micro scales.
A similar mean request rate is not sufficient.

## 8. Server build example

Run code from the GitHub checkout and keep all generated data in the external data tree:

~~~bash
cd /home/rzwang/SpecRhythm
git fetch origin
git switch codex/workload-v0.1
git pull --ff-only origin codex/workload-v0.1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
mkdir -p /home/rzwang/data/SpecRhythm-data/processed/workload-v0.1/r3-proxy
mkdir -p /home/rzwang/data/SpecRhythm-data/manifests/workload-v0.1
MOONCAKE_TRACE=/home/rzwang/data/SpecRhythm-data/raw/Mooncake/FAST25-release/traces/conversation_trace.jsonl
MOONCAKE_COMMIT_SHA="$(git -C /home/rzwang/data/SpecRhythm-data/raw/Mooncake rev-parse HEAD)"
MOONCAKE_SOURCE_URL="https://github.com/kvcache-ai/Mooncake/blob/${MOONCAKE_COMMIT_SHA}/FAST25-release/traces/conversation_trace.jsonl"
R3_OUTPUT=/home/rzwang/data/SpecRhythm-data/processed/workload-v0.1/r3-proxy/r3-mooncake-622-window-0-600000-scale-1.jsonl
R3_MANIFEST=/home/rzwang/data/SpecRhythm-data/manifests/workload-v0.1/r3-mooncake-622-window-0-600000-scale-1.manifest.json
R3_VALIDATION=/home/rzwang/data/SpecRhythm-data/manifests/workload-v0.1/r3-mooncake-622-window-0-600000-scale-1.validation.json
specrhythm generate --config configs/workloads/r3-mooncake-622-proxy.json --arrival-trace "$MOONCAKE_TRACE" --window-start-ms 0 --window-duration-ms 600000 --time-scale 1 --source-url "$MOONCAKE_SOURCE_URL" --source-commit-sha "$MOONCAKE_COMMIT_SHA" --output "$R3_OUTPUT" --manifest "$R3_MANIFEST"
specrhythm validate --workload "$R3_OUTPUT" --config configs/workloads/r3-mooncake-622-proxy.json --arrival-trace "$MOONCAKE_TRACE" --window-start-ms 0 --window-duration-ms 600000 --time-scale 1 --output "$R3_VALIDATION"
specrhythm summarize --workload "$R3_OUTPUT"
~~~

The `Mooncake` directory must be the same Git checkout from which the trace was copied; otherwise,
provide the actual source commit explicitly. The manifest always hashes the local trace bytes, so
the trace identity remains independently auditable.

## 9. Recommended build order

1. Normalize Mooncake records and create immutable source manifests.
2. Produce R3 and R4 with several fixed trace windows and rate scales.
3. Tokenize public task datasets using the exact target tokenizer; store IDs and lengths, not copied
   prompt text, in this repository.
4. Measure AR capacity and P90 TPOT on the remote server.
5. Collect step-level speculative profiles for the chosen target/draft pair.
6. Fit 300-second Gamma parameters and, where dispersion requires it, 1 ms NB parameters.
7. Validate and freeze version workload-v0.1; only then run the final strategy sweep.
