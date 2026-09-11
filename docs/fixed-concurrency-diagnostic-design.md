# Fixed-concurrency batch control and timing decomposition

Implementation on the serving PR; real GPU timing/performance is **PENDING**. This is an
independent diagnostic, not S2 qualification or Poisson serving. Existing S1/S2 defaults,
chain drafting, sampled-row mapping, acceptance, EOS, K4, private KV ownership, terminal
materialization and five vLLM patches remain unchanged. Cross-run/mode token, length,
EOS and round equality are **NOT_REQUIRED**.

## Work and capacity

The source is the existing frozen S1 mixed100 library (30/30/20/20). Exact source rows,
prompts, IDs, output budgets and order are retained. All 100 requests undergo real
resident prefill/bootstrap before the ready barrier. The first 64 source IDs are fixed;
A is the first 32, B the next 32. Remaining IDs refill in frozen FIFO order. This
initial-state definition is shared across modes; a naturally terminal bootstrap is
reported and cannot be manufactured into an active sample.

| Mode | Required resident pool | Active slots | Cohorts | Per-cohort cap | Target forward cap |
|---|---:|---:|---|---:|---:|
| target | 100 | 64 | none | null | 64 |
| serial | 100 | 64 | none | null | 64 |
| serial-split | 100 | 64 | A/B | 32 | 32 |
| pingpong | 100 | 64 | A/B | 32 | 32 |

Target engine sequence capacity is 128, query capacity 4096, context capacity 4096.
Draft engine sequence/query limits and per-rank physical KV block capacities are also
recorded from the real engines. These are distinct from resident count, slots and
per-forward request count. A model-loaded three-rank probe checks precisely the 100
resident / 64 active case using the existing conservative private-block formula,
including output growth, K4, partial blocks, workspace and logits reserve. A deficit
fails explicitly and is recorded in `actual-capacity.json`; there is no 500-request
search, SLO calibration or automatic capacity reduction. Each execution rechecks its
fresh engine and then validates actual resident pools.

`resident_request_requirement=100` is distinct from `resident_request_count`: the latter
is null before preparation, zero for the model-loaded probe, and the actual nonterminal
resident count after real prefill. Runtime/light metadata also records actual engine/KV
limits and any isolated B32 point's active/forward cap. Bootstrap-terminal requests are
counted explicitly in `resident-pool.json`, never misreported as resident decode work.

`ServingClock` gains optional `per_cohort_capacity` and a fixed initial assignment.
Defaults are unchanged (S2 still uses total128 without a new per-cohort cap). Admission
requires observed readiness/arrival, free held slots and a nonbusy cohort with room.
Finished-but-draining requests continue to occupy slots. Only the coordinator between
completed Target steps changes admission; inflight requests never move. Partial batches
proceed without waiting for future arrivals or enough requests to fill a batch.

## Execution paths and controlled stop

The common S2 real prefill/bootstrap/pool validation is extracted as `prepare_resident`;
the S2 caller preserves its existing whole-run deadline and behavior. The diagnostic
selects its own scheduler classes/limits. Target/Serial reuse their real proposers;
serial-split/PingPong reuse the same S2 Ping scheduler, proposer, cohort client and
single-owner `S2DualMachine`. The sole serial-split ordering difference is a successful
owner-status drain before **every** Target step. It includes terminal work, and waits
for actual completion. Poll sleeping is backoff only, never simulated stage duration.
The light result rejects physical/host Draft-Target overlap for serial-split.

`short` runs four modes once by default. `--samples` and `--warmup-steps` configured at
prepare are rotation units: one Target step for target/serial, two for grouped modes.
Only nonempty steps count. Actual batch histograms and complete full-size A/B rotations
are reported; partial or incomplete rotations do not become full-size samples. A time
budget can end any point sooner. The foreground script never invokes S2's gate grid.
Fill/full-load/tail wall durations sweep actual admission, completion and release events,
including control/owner waits between steps. Full-active and full-held fractions are
separate; a finished request can occupy a held slot without providing active decode work.

The window begins after the configured warmup, at a coordinator boundary. It ends at
the safe boundary after the last issued step (time limits can overshoot by one real
step). A time check after an owner wait prevents starting another Target step after
the budget expires. Stop prevents further admission/Target submission, drains work
already issued by the final step, fences Target, aborts unfinished Target requests,
shuts down Draft on its owner, and records release only after those actions. Pending
proposals can be discarded by diagnostic cancellation; they are never counted as
accepted or completed output. Total drain has an explicit budget applied to Draft
socket waits and final Target RPCs. The existing owned process launcher remains the
outer timeout/failure cleanup authority. A separate `stop` command requests a safe
stop, then performs bounded owned cleanup if needed.

Natural `FINISHED`, `DIAGNOSTIC_CANCELLED`, and failed execution remain distinct.
Window throughput counts only actual coordinator commits timestamped inside the
window. All issued work, including next proposals and final sync outside that window,
remains in the raw report and arrival-to-drain cost. There is no full-request SLO
attainment result. `full_request_tpot_ms` uses only sufficiently observed naturally
completed requests; `window_commit_interval_ms` measures commit-batch intervals, not
invented per-token timestamps. Missing statistics are null with count/reason.

## Stage samples and interpretation

`stages` is a separate initial-state experiment: AR32 A, AR32 B, AR64, Serial SD32 A,
SD32 B, SD64, and A verification32 with B Draft32 submitted concurrently. Serial
initial proposals supply isolated D32/D64; the corresponding real model forward
supplies V_SD. AR is separately named V_AR. Every point/repeat is a fresh process,
fresh engine and real prefill, never restoration of stale continuation KV. Optional
stage warmups are explicitly discarded fresh-process trials; the default does not
double all experiments. Engine profiling/prefill warmup is distinct from same-shape
trial warmup. Setup/restoration cost, configured decode warmup, window and drain are
reported separately; fresh restoration is part of setup, not a fabricated zero-time reset.

K=4 means up to four actual candidates. Draft's first candidate comes from validated
cached committed-prefix logits, so full K4 proposals normally need three new Draft
model forwards, not four. Natural EOS or budget truncation can change shape. Actual
candidate lengths, contexts, request sets, dtype/attention backend and config are
retained. Missing B32/B64/K4 samples remain insufficient. Split-cost derivation requires
both 32 halves, their exact 64 union, equal per-request initial contexts, dtype/backend
and execution identity. No token equality gate is introduced.

Continuous pipeline samples are a separate population. `actual_rotation_ms` measures
real rotation completion intervals including intervening control/waits. The concurrent
initial point reports the actually observed B Draft and A Target stage costs and joint
host span, plus independently measured event overlap. Async submission alone proves
no physical overlap. A zero/uncertain overlap remains visible, not forced to PASS.

The comparison emits D32, D64, V_SD32, V_SD64, V_AR32/64, actual rotation time and own
per-mode g. It derives `2*V_SD32 - V_SD64` from matched halves and D32/V_SD32. Main
comparisons are Serial→Serial-split (split cost), split→PingPong (overlap net effect),
Serial→PingPong (final net effect), and Target→speculative (token progress). Actual
work counters accompany wall/throughput ratios; the label is **production vLLM Batched
Draft end-to-end improvement**. There is no pure-batching claim.

Idealized `D64+V_SD64+H_serial` and `2*(max(D32,V_SD32)+H_pingpong)` assume balanced full
batches and comparable context/K. GPU-only predictions and observed wall residuals
are shown separately. Nested host timers cannot be added to manufacture H. Residual
time is unexplained, not automatically attributed to acceptance or GPU count.

## Observation and validation boundaries

Only the diagnostic children/workers install the in-memory observer. `original-live`
is the sole supported observation setting: live verification UUID queries, real
startup/device binding, full pool/prefix/block guards and existing logging/fsync all
remain. No cached-UUID or reduced-audit optimization is silently enabled. Legacy
launchers strip stale `SR_FIXED_*` settings. The same observer applies across modes.

Timers cover scheduler, IPC, control JSON, serialization, live UUID validation and
actual `nvidia-smi` subprocesses, resident/private-block scans, prefix records, required
CUDA/TP synchronization, checkpoint writes and fsync. They are inclusive categories;
nested or overlapping times are not additive. Intervals use one host monotonic clock.

New model-forward CUDA events are read after the existing final Target fence / Draft
end-of-window fence. The only additional synchronization is outside the per-round
path (startup anchor and window end). Every Target TP rank is retained; the stage scalar
is **max rank duration**, not sum. Cross-device event intervals are projected onto a
bracketed startup host anchor. The overlap report retains anchor uncertainty as
lower/upper event-envelope overlap, unions TP intervals and intersects with Draft's
union. It is not exact kernel overlap or saved critical-path time. Forward gaps use
same-device event differences; control wait reasons are separately timed.

The default post-run path reads compact runtime/worker/backend evidence, not giant
Target JSONL logs. Producer-side compact capture retains actual input positions and
existing diagnostic validation results. Light checks cover execution/owned cleanup,
backend identity and shutdown, request/token conservation, budgets/EOS, release order,
base/root exactly once, candidate/query conservation, committed-round binding, actual
TP model event coverage and serial-split nonoverlap. Aggregation uses sort/bisect or
interval unions, never Draft×Target Cartesian comparison. Results print immediately,
with raw stdout/stderr still mirrored to the terminal and retained files.

`execution_status`, `measurement_status`, and `full_offline_audit_status` are independent.
`valid` in this diagnostic means its execution/material checks passed; it does not
claim full S2 qualification. Missing shapes are `INSUFFICIENT`; the complete offline
artifact audit remains `PENDING` until explicitly run. `audit` checks checksummed raw
logs and the inherited stage dependency/terminal-drain contract separately on CPU.
All original artifacts remain immutable. Audit produces a separate artifact rather
than rewriting the light result. The small bundle includes config/light/error artifacts
only, excluding raw token/request diagnostics.
