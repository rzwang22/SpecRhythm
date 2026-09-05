# Phase 4B.1 correctness validator schema

`phase4b1-dual-correctness-validate` emits
`specrhythm.phase4b1-dual-correctness-validation.v2`. It is read-only: every Target, Serial, Dual,
manifest, JSONL and process-lifecycle input is SHA256-hashed before and after validation.

The top-level decision is intentionally binary:

```json
{"valid": true, "outcome": "A", "errors": []}
```

or:

```json
{"valid": false, "outcome": "FAIL", "errors": ["..."]}
```

`triangle` contains completed request sets and pairwise Target-versus-consumer comparisons. Each
divergence names the stable request, first token position, both token IDs and both termination
tuples. Logical manifest identity covers workload, sampling/batch-invariant configuration,
prompt hash, bootstrap, committed prefix token IDs/hash/count, Target KV/pending state, Draft KV,
prefix version and next round.

Each `dual_runs` entry reports independent `valid/errors` objects for request state, proposal
lifecycle, scheduler, token accounting, verification input, Draft sync, Target-blind isolation,
runner-only invariants, overlap and cleanup. Target-tail validation distinguishes retained,
already-`CONSUMED` proposal metadata from a live proposal and requires Draft readiness plus an
ordered one-token terminal state sequence. `repeat_comparisons` uses `(request_id, round_id)` and compares proposal,
accepted/rejected, correction/bonus, committed, terminal and prefix-version fields. Raw
cross-request event order is retained as `raw_event_order_equal` diagnostic metadata and never
overrides keyed semantic equality.

`embedded_verdict_authority` records the historical `run.valid/errors`, the current recomputed
components, structurally proven superseded errors and any remaining fatal embedded error. Legacy
revalidation is disabled unless the exact supported source commit is supplied. It never filters
arbitrary errors by substring. `overlap_evidence` separately reports temporal overlap, raw-event
device attribution, authoritative worker topology and hardware-placement qualification. A
controlled gate needs semantic validity; an overlap-required gate needs at least one qualified
real witness, not one witness in every coordinated control run.

`input_artifacts_immutable`, `input_sha256_before` and `input_sha256_after` prove validator
non-mutation. `performance_result` is always false and the claim boundary explicitly forbids
TPOT, throughput, goodput, SLO, speedup or overlap-benefit interpretation.
