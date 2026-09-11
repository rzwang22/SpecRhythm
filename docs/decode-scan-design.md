# Resident360 fixed-time post-prefill decode scan

This independent experiment extends the serving branch containing the `e78e4c7`
bound-prefix alias repair. GPU performance/qualification is **PENDING**. It does
not reclassify older artifacts. PR #4 remains Draft/Open; PR #2/#3 are unchanged.

## Frozen experiment

`specrhythm-decode-scan` / `scripts/run_decode_scan.sh` select only Target, Serial
and PingPong at total active B=16,32,64,128. Target/Serial forwards have B requests;
PingPong has two cohorts of B/2 and one cohort per Target forward. K remains four.
No arrival-rate scan, speculative policy, scheduler optimization or overlap gate
is introduced. The old fixed64 entry retains resident100, active64, its sample12
budget, original-live/linear defaults and existing four-mode/stage semantics.
S1/S2 defaults are unchanged.

Preparation verifies the qualified S1 mixed100 and its frozen S0 parent. It reuses
S2's deterministic stratified order over the **unique original main1000**; the
first360 original definitions have 108 chat,108 code,72 reasoning,72 summarization
requests. The original S1 hundred are included without duplication. Seed1666,
request order, original prompts/output budgets/sampling/EOS and one workload-file
hash are shared by every point. Initial IDs, initial A/B assignment and FIFO
remaining order are sealed in each B manifest. Natural bootstrap-terminal requests
are skipped by admission, with actual admissions recorded separately. No request
is recycled and no output budget is extended.

## Real preparation, capacity and ownership

Each capacity probe and each decode point starts fresh owned Draft/Target engines.
Capacity is checked on all three actual GPU workers for this mode, B, execution
manifest and resident360 workload. The existing private-KV formula reserves all
360 prompts, worst active continuation including K, partial blocks, 5% block
margin, Draft resident logits and workspace; EOS is not assumed to save capacity.
There is no fallback to resident100, B64 or a smaller pool.

All three modes use Target max_num_seqs=512 to retain the 360 actual requests,
max_model_len=4096 and max_num_batched_tokens=4096. The existing Draft engine has
max_num_seqs=128 and query limit4096. A B128 Serial step needs at most 640 query
positions, PingPong B128 needs at most320; Target-only needs128. Runtime checks
the actual effective sequence/query limits and each prompt+bootstrap, in addition
to block/workspace capacity. All points use the same model, tokenizer, dtype,
TP2/device assignment, synchronous InprocClient and original eager configuration.

`fixed_runtime.run → configure → make_engine → target_startup → capacity_for →
prepare_resident` is the shared real path. Preparation adds all360 requests with
original parameters, waits for each real bootstrap and decode-ready manifest,
checks Target prompt KV / Draft prompt+bootstrap KV / next-input hashes, freezes
private resident blocks, verifies both owner sets and performs the existing fence.
`prefill_complete_ns` is captured after this returns, before the decode clock.
Capacity-only probes also execute this preparation and settle all360 states with
zero verification accesses; no measurement is hidden in a probe.

Timed refill uses the already prepared remaining pool. Existing S2 checks prohibit
initialization after the barrier, eviction, timed re-prefill and private block
aliasing. Normal commit/prefix materialization, proposals, KV updates, refill
management, waits and live UUID queries remain real timed work. Slots are held
through terminal drain until physical release evidence permits reuse.

## Window and full-batch contract

Default warmup is two complete total-B rotations. Target/Serial use two full B
steps; PingPong uses four full B/2 steps paired by opposite cohort and disjoint
actual request IDs. State and tokens are retained. After warmup, any necessary
refill restores B active requests before the single measurement boundary. Waiting
for terminal release at this boundary does not generate extra warmup steps.
The normal pipeline state (active IDs/cohorts, next cohort, in-flight IDs) is
recorded without a new GPU fence or fabricated proposal.

The window is 30 seconds with **no sample budget** (`samples=null`). Only commits
at/after the measurement boundary are counted. No new Target step is issued after
the deadline. A previously issued synchronous atomic step may finish past it;
its full commits and actual elapsed wall time are retained, with overrun reported.
No extra B-half step completes a final partial PingPong rotation after expiry.

Every actual nonempty forward must have B (or B/2) rows on both TP workers. Stock
selection is unchanged. The scan-only `decode_scan_full_batch` guard intercepts a
partial selection **before EngineCore dispatches the model**, records rejected IDs
and shape, then performs the existing bounded drain. Stock scheduler mutations are
not rolled back or presented as commits. The scan rejects deferred block freeing;
the pinned synchronous allocator abort path releases this unexecuted selection.
If too few active+queued requests remain to restore B, it stops before more work.
`INSUFFICIENT / pool_exhausted_before_window` and `partial_batch_prevented` retain
partial measurements, exclude comparison and stop subsequent points. An unexpected
partial physical forward is a material failure. Empty scheduler polls and refill
gaps remain inside the continuous window, never filtered out of the denominator.

Metric: `decode_throughput_tok_s = committed_window_tokens / (measured_window_ms/1000)`.
Actual root/base and accepted candidates are counted once per logical request;
prompt, warmup, rejected candidates and unverified proposals are excluded. TP ranks
are identity/physical-forward checks, not additive token or time contributors.
This is host-inclusive post-prefill decode throughput, not pure forward throughput,
whole-request latency or completion-throughput. There is no TPOT/SLO/goodput gate.
Cross-run token/length/EOS/round equality remains NOT_REQUIRED.

## Failure, deadlines and evidence

The owned supervisor enforces one setup900 deadline covering Draft load, Target
load, all360 prefill and warmup. At window start it watches window30 plus a drain60
atomic-step grace; each final drain then has its existing single60 deadline for
wait/fence/sync/cancel/release/shutdown/final log flush. RPC waits consume remaining
budget and the existing supervisor can terminate blocked workers. The fallback
whole-process timeout also remains. Operator stop never starts a following point.

Snapshots are written atomically at window opening, after completed steps and
before risky drain. Settlement reconciles the last actual prefix/version/round,
discards unused proposals explicitly, preserves natural terminals and labels other
requests DIAGNOSTIC_CANCELLED. The original pending-proposal shutdown rejection,
first-error retention, real exits and owned cleanup are unchanged. Failed flush,
RPC or release cannot become successful logical cleanup.

buffered-live and bound-prefix are explicit frozen scan settings. Buffers retain
the original bounded256-record/1MiB flush/checksum contract. Protocol control
remains synchronous; required live UUID accesses remain live. Existing timers and
traces are reused, with no new per-token dumps. Scan evidence has a fail-closed
10,000 scheduler-poll/nonempty-step safety ceiling, not a success/sample stop rule.
Summary/inspection do not invoke the full CPU audit or initialize a GPU framework.

## Schema and qualification

`scan-config.json` / per-B execution manifests: `specrhythm.decode-scan.v1`, seed,
selection/order/workload hashes, point list, execution identity, capacity settings,
options and initial/refill definitions. `actual-capacity.json` additionally binds
mode/B/point and execution/workload hashes to real worker checks.

Runtime keeps the existing fixed-runtime.v2 request, step, TP, commit and drain
contracts plus `decode_scan`: prefill boundary, warmup rotations, complete/partial
measurement rotations, window initial population, rejected selection, sample_limit.
Each step retains real rows/context/query/root/candidate positions and timing.

`result.json` / `light-summary.json` / CSV include commit, workload, pool/B/sub-batch,
execution/measurement/cleanup/capacity status, reason, actual batch min/mean/max,
full_active_time_fraction from admission/completion interval sweep, actual wall and
overrun, committed tokens/tok_s, Target steps, rotations, natural/cancel/refill
counts and actual admission order. Full-active fraction measures occupancy time;
forward shape is independently enforced. Empty optional statistics are null/count0.

Only valid execution + measurement PASS + cleanup PASS is comparison-eligible.
Physical preparation, token/position/TP/request/lifecycle checks, logging receipts,
identity and live UUID evidence block materially invalid results. Clean stops and
pool shortage remain excluded; failed capacity prevents decode. The small12-point
summary lists NOT_RUN and failed/insufficient/stopped points separately from valid
ones. Successful points are skipped on explicit continuation; failure prevents
automatic continuation/retry or configuration changes.

Small bundles export no raw timelines, model data or KV tensors. Large settlement
receipt arrays are represented by count/released/new-proposal totals and source
hash; original receipts remain untouched. File source/export hashes disclose these
projections. Logging integrity receipts remain intact. Output has a10MiB bound.

## CPU evidence and pending gate

Tests run the real fixed/S2 scheduler selection for all12 mode/B combinations,
actual prepare_resident for360 IDs in each mode, the real Draft state machine and
coordinator through >12 steps/refill/time overrun/pool exhaustion/stop/failure,
and real asynchronous Dual ownership through a final partial rotation and drain.
Additional result tests join physical TP forwards/positions/commit records; CLI
tests exercise success skipping, failure blocking and small inspection/export.
Hardware computations/transports are substituted, not shutdown success or identity
logic. Existing bounded-process, live UUID, logging and S1/S2 regressions remain.
Pinned source contracts verify schedule-before-dispatch and non-deferred abort.
GPU capacity, sustained full batches and throughput still require the server run.
