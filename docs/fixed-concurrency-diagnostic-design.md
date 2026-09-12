# Fixed-concurrency batch control and timing decomposition

Current opt-in persistence/timing addendum: [design](fixed-buffered-runtime-design.md) and
[two-mode buffered-live runbook](fixed-buffered-runtime-runbook.md). Existing instructions
below describe the original-live baseline and remain available; no old artifacts are changed.

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

The window begins after configured warmup and ends after the last issued step. An
issued step can overshoot its time budget; its actual time and committed tokens remain
charged. No new admission, Target step or Draft proposal starts during stop. The
fixed-only `DiagnosticSerialMachine` / `DiagnosticDualMachine` add explicit settlement;
normal S1/S2 machines and algorithms retain their existing behavior.

At the failure revision `89a962127f2e9a10a2564736e2124dae0abe51d8`, `drive` aborted Target
and immediately called Draft `shutdown`. Serial's completed step had already created
64 next-round pending proposals. The inherited `DraftStateMachine.shutdown` correctly
refused them; `cancel_request` would not have cleared those proposals either. Dual's
owner could be idle while its ready/claimed proposals still needed settlement. Final
`runtime.json` was written only on successful return, losing the measurement at this
failure boundary. The retained small bundle confirms this call path and lacks the
full runtime/backend report; no old data is repaired or reclassified as PASS.

Stop now waits for issued owner work, fences/reads Target evidence and physically
aborts unfinished Target requests. On the existing Draft owner it settles each actual
resident ID against the coordinator's final committed prefix/hash, verified round count
and prefix version. If the final Target commit is not yet in Draft, the existing greedy
acceptance and `DraftCommitPlan` KV materialization run without a propose continuation.
A next proposal whose parent already equals that prefix is explicitly discarded with
ID/round/count evidence. An uninitialized logical request is eligible only as an
untouched, unadmitted physical resident; the protocol does not create logical state.
Target-only releases its unused bootstrap Draft KV without replaying AR tokens into it.
Natural releases/tails retain their real completion/release evidence, including Serial's
ordinary proposal-free terminal tail contract. Inconsistent or partially applied state
fails closed, retaining the measurement for investigation.

The release path checks real logical/physical prefix, round and proposal agreement,
whole-pool private-block ownership, and unchanged unrelated KV. A completion fence and
actual allocator release precede logical cancellation/release. Idempotency receipts bind
the full stop authority; repeats never release twice, while changed authority fails.
Normal Serial shutdown still rejects unresolved proposals; diagnostic Dual also rejects
unresolved proposals. All physical requests must be gone before the final shutdown.
No cancellation sets natural EOS, output-length completion, or acceptance counters.

One absolute `drain-state.json.deadline_ns` starts before drain. All socket/owner/Target
waits consume its remaining budget. The existing owned process supervisor watches that
same deadline through final coordinator/engine shutdown and can TERM/KILL blocked
worker/RPC work. Bounded process termination grace is separately recorded after the
deadline; expiry is failure (effective rc=124), never successful drain. Operator stop
prevents subsequent suite points, with the existing owned-only cleanup fallback.

`measurement-snapshot.json` is atomically checkpointed without fsync after completed
steps and before risky drain, and best-effort on exception. It contains only compact
identity/count/timing/population summaries, no resident prefix/KV dump. Window commits
stay frozen during drain. Measurement, drain and whole execution durations remain
separate; all actual wait/sync/release work is retained. The first error is saved before
secondary report/engine-cleanup handling. Missing final reports cannot erase the
snapshot. `summary` and small `bundle` include partial evidence without invoking full
CPU `audit`; failed or incomplete/operator-stopped points are excluded from numeric
comparisons. A snapshot never asserts final PASS. See the versioned
[artifact schema](fixed-concurrency-diagnostic-schema.md).

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
# Retained timing attribution (2026-09-11)

The four continuous points at `b42401585a6b32e36fd0b874eb106a8f75ee9ea4` were
reported execution/measurement/cleanup PASS by the operator. The supplied small bundle
contains aggregates but no runtime/backend raw timelines. The evidence and source audit
are recorded in [the b424 attribution](fixed64-b424-timing-attribution.md).

This revision adds only `fixed_attribution`, its read-only projection layer and a
separate supervised CPU command. Runtime, instrumentation, log persistence, live UUID,
models/K4/workload, scheduler and batch limits remain unchanged. No logging or scheduling
optimization is justified as a measured critical-path improvement from this small bundle.

The tool joins scheduled stable/internal IDs to both TP forward rows, committed
request/prefix rounds to actual coordinator commits, and Draft request/round/context
plans to actual proposal forwards. Actual A32/B32 cohorts are paired by round and
disjoint request membership; incomplete or interleaved boundaries remain unknown.
It checks recorded sample counts and independent Draft purpose counters before using
CUDA bounds to classify zero overlap. TP durations are never added. Model-event overlap
is bounded by the original clock uncertainty, never labeled exact kernel overlap.

Host category counts include nested calls. Per-source unions and cross-source unions
are separate from inclusive sums. Missing thread/parent-call IDs prevent exclusive
self-time attribution. RPC duration includes synchronous service work where the path
blocks; it is not pure network cost. The unwrapped TP collectives and GPU postprocessing
also prevent treating these timers as exhaustive primitive-level coverage. The tool
retains this limitation and does not convert observed overhead into predicted speedup.

The source retains no request-correlated enqueue/owner-receive or exact ready-publication
event. Existing proposal start/end, work-record endpoints, scheduler/model-launch bounds
and commit timestamps retain their own semantics. An absent event is not an invented
timestamp or zero duration. Default reports/export have a 10 MiB combined output budget;
sorting/indexing/interval scans replace Cartesian joins, and a parent process enforces
one 120-second CPU deadline with partial output on failure. This never invokes full audit
or inference. See [the CPU-first runbook](fixed-attribution-runbook.md).
