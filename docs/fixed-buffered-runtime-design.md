# Fixed64/32 bounded diagnostic persistence and timing correction

Scope: Draft/Open PR #4, fixed diagnostic entry only, based on e7452fc2a9425ced1be73afc9af88008572af765.
No AutoDL connection or GPU execution. GPU performance PENDING. Existing S1/S2 and
all old artifacts keep their contracts. No scheduling, model, K=4, 64/32 capacity,
cohort, acceptance, prefix/version, KV, live UUID, barrier or stop-protocol change.

## Evidence and decision

The supplied attribution JSON SHA256 is
`20aa9225b4727ea109e5215c72823619fed656749faf2a2e4e2cea2d5a1ffb0c`.
The archive contains only attribution/summary/report/progress/exit, not raw runtime,
backend or offending projected CUDA rows. Analysis of GPU commit b42401585a6b32e36fd0b874eb106a8f75ee9ea4
completed in 7.8573 s under analyzer e7452fc; no GPU/full audit.

PingPong independently has ZERO event overlap, no coverage or join errors, 12 complete
rotations and 24 Target steps. All 24 cross-cohort owner operations overlap Target
preprocessing and finish before its model launch, mean advance 32.1208 ms. B32 Target
step means: 787.1608 ms total, 181.8099 ms before model hook, 109.8925 ms CUDA forward.
Post-hook host time includes GPU completion waiting and other operations, not pure CPU.
Zero is supported for recorded model CUDA events; it is not exact kernel overlap or
proof that every GPU operation is covered.

| Approximate recorded forward ms / full64 rotation | Serial | PingPong |
|---|---:|---:|
| D_proposal | 60.6260 | 131.3952 |
| D_commit_or_prefix_sync | 19.9362 | 60.1338 |
| V_target (max rank per step) | 158.8003 | 219.7850 |
| Recorded forward sum | 239.3625 | 411.3140 |
| Actual rotation mean | 1144.1404 | 1596.2965 |
| Actual rotation P50 | 1134.1648 | 1584.4505 |
| Actual rotation P90 | 1167.8149 | 1686.0722 |
| Throughput tok/s | 167.2180 | 119.0190 |
| Committed progress / request / verification | 2.9895833 | 2.9687500 |

Draft sums are selected by real host launches inside the window and divided by 12
complete rotations. They are approximate allocations, not matched initial-state stages
or causal critical paths. Other *recorded* Draft purposes in this window have count=0;
unhooked GPU work remains UNKNOWN. The 452.1561 ms wall gap minus 171.9515 ms recorded
forward gap leaves 280.2046 ms arithmetically unattributed. Earlier proposal-only
320.402 ms residual omitted commit work. Neither residual means CPU computation time.

PingPong rank0 checkpoint/fsync unions are 3351.374/2673.842 ms, TP barrier 20.521 ms;
rank1 barrier 1687.010 ms. Each rank executes 24 nvidia-smi queries, unions
1846.100/1859.459 ms. Rank1 writes zero checkpoint logs because the real proposer and
Target diagnostic publication paths run on rank0; live UUID queries still run on both.
Coordinator checkpoint/fsync unions are 1169.676/964.641 ms; Draft owner/server
851.237/511.386 ms. Timers are inclusive and overlap across categories/processes.
Do not sum them or claim the same milliseconds will be saved by batching persistence.
The rank0 publication before TP synchronization gives a source-supported imbalance
candidate. Necessary rank barriers and live validation remain intact.

Serial-split minus PingPong wall is 5701.2366 ms; split pre-step owner wait is
5606.6305 ms, Target step-union difference 48.3395 ms, remainder 46.2666 ms. The split
path explicitly waits before stepping Target. This accounts for host envelopes, not
5701 ms of GPU overlap. Fine-grained correlated RPC receive/enqueue and timer parent/
thread IDs remain missing from the historical producer; exact critical-path attribution
is UNKNOWN.

## Protocol audit and independent configuration

`prepare --observation original-live|buffered-live` is frozen in diagnostic options and
manifest. Default remains `original-live`. Only the fixed launcher propagates
`SR_FIXED_OBSERVATION`; S1/S2 clean_environment strips SR_FIXED variables. Both modes
still execute the original live UUID validation at the same verification hooks.

`fixed_observe.install_host_observation` installs the CPU adapter
`fixed_logging.install_checkpoint_logging` in coordinator and Draft before model
construction, and in Target startup RPC before verification. Forked Target workers
can inherit the installed adapter earlier; each append/read resolves the current PID,
so the child creates its own manager and never flushes the parent's copied queue.
Counters/digests cover writes after installation; any earlier native startup writes
retain original persistence and are excluded from those counters. `CheckpointJsonl.append` originally hashes canonical
payload JSON, serializes the framed record, opens, writes, flushes, fsyncs and closes
on every event. These two serializations encode different content and are retained.
`original-live` delegates that exact append. Buffered mode produces the same bytes,
checksum, timestamps and per-file order, while batching write/flush/fsync. In-memory
Target diagnostic contract capture/validation still happens immediately after append.

Whitelist (only files directly inside this attempt): target-diagnostics, round-events,
transport-events, proposal-events, verification-events, request-state-events,
scheduler-events, proposal-lifecycle-events, draft-work-events, draft-transport JSONL.
These are post-run diagnostic evidence, not owner protocol state. Scheduler and
verification logs have startup recovery readers; read-your-writes synchronously flushes
before the unchanged checksum reader. New attempts use fresh directories. No process
uses another process's whitelisted log as its live enqueue/ready/claim channel.

Real protocol state resides in Dual controller memory/locks and framed Unix sockets.
`s2_pool.publish` atomically replaces control JSON; ready/setup JSON, admission and
initial-proposal/timing records, ownership/release/error reports remain immediate.
The whitelist intentionally excludes unknown paths and potential recovery journals.
Protocol ready publication does not depend on diagnostic flush. Draft owner writes its
work log and server writes its transport log under the same process-local RLock.
Each whitelisted file has one process writer; the buffer preserves each file's order.

Each process buffer is bounded by 256 records AND 1 MiB of encoded bytes. Full buffers
synchronously flush with backpressure; no background thread/queue. One oversized record
bypasses the queue after older records flush, so scratch memory may hold one encoded
record plus the bounded queue. Per-file handles are bounded by the whitelist. No drops.
Record counts/digests advance only after write/flush/fsync/close success; partially
written failure batches are conservatively unconfirmed and cannot be retried as success.

## Drain and failure path

After real pending-proposal settlement and resource release, `fixed_drain.settle` calls
normal Draft shutdown and final Target snapshot. Buffered finalization then waits for
Draft's completion receipt, sends a CPU-only final-flush collective RPC to both Target
workers, and flushes coordinator logs. No GPU fence/barrier was added. Draft's receipt
is emitted **after** server exit and owner join: the shutdown reply can precede the
last transport/work-log append, so backend.shutdown alone is not sufficient evidence.
The zero-verification capacity path also executes finalization, allowing empty buffers.

All final flushes share the original absolute drain deadline. RPC timeouts consume its
remainder. A blocking file write/fsync cannot be interrupted merely by Python deadline
checks: the existing owned-process supervisor watches the same deadline and performs
bounded TERM/KILL cleanup on a blocked coordinator/worker. Final flush is included in
drain/end timestamps, never token accounting. Total process time and cleanup status
remain separately governed by the real lifecycle report.

Four COMPLETE receipts (coordinator, Draft, ranks0/1), produced=written, empty queue,
within-bound peaks, same deadline and final flush inside drain are required for buffered
qualification. Missing/RUNNING/FAILED receipts reject qualification. Abrupt process death
leaves RUNNING counts non-final; no missing record is fabricated. Logger/flush errors,
original execution errors and cleanup errors are retained; first error wins. No normal
shutdown guard, stop cancellation disposition or physical-before-logical release check
is removed. Light summary/bundle include receipts and failure snapshots without full audit.

## Target preprocessing review

`FixedBatch.schedule -> PoolScheduler.schedule -> native schedule` reads current control,
audits physical resident ownership before and after schedule, builds actual prefix/block
records, and publishes scheduler diagnostics. The two audits surround allocator/state
changes; they are not redundant. `S2PingProposer._admitted_initial`, dynamic cohort
annotation/client assignment, claimed proposal acquisition and inherited verify hooks
use changing admitted/prefix/version/proposal state. Rank0 performs per-request
validation/recording before TP barrier; rank1 must synchronize the same verification.
Live query/all-gather after completion validates real device binding. All retained.

Repeated `assignment()` control reads are candidates for a separately proven operation
snapshot, but no such cache or metadata switch is introduced here. This leaves one
runtime experiment variable: diagnostic persistence. Frozen cohort guesses, skipping
checks or moving required work to a later round would not be justified by these traces.

## Time contracts and independent clock repair

Summary and attribution now distinguish proposal, commit/prefix-sync, Target and other
recorded Draft forward costs. Sum selection by host launch is explicitly different
from physical interval clipping at window edges. TP is separately reported or max per
joined step, and physical unions deduplicate ranks and nesting. Outside the union of
all recorded GPU intervals is uncovered observation time, not CPU-only time. Stage
D32/D64 remain explicitly labeled proposal-only compatibility aliases. The old
`2*max(D32,V32)` end-to-end prediction is unavailable until commit/dependency costs are
known; it must not be used to estimate measured PingPong rotation. Full-request SLO/TPOT
cannot be inferred from controlled cancellation.

Clock-only producer repair: `round(integer_anchor + float_offset)` rounds the anchor to
binary64 before adding and rounding the offset. At anchor=17357310142137541, width=1 ns,
offset=1.0 ns, it generates width=4 ns (3 ns error). New `integer-anchor-v2` computes
`integer_anchor + round(float_offset)` separately for start and end. Width is exact;
each offset-rounding error is at most 0.5 ns, duration rounding at most 1 ns, in addition
to the original CUDA elapsed-measurement uncertainty. Both integer anchor endpoints and
projection version are retained. No timing fence is added. New-version analysis requires
exact width; the existing legacy width tolerance of 2 ns is unchanged.

This proves a producer rounding defect, **not** the cause of the real Serial-split row
(which was not supplied). Legacy Serial-split remains UNKNOWN. CPU analysis now reports
up to 16 actual failing projected rows per attempt, with source/identity/index, bounds,
uncertainty, host timestamps and width residuals. Reanalysis writes a new sibling output
and does not mutate the old JSON. 120 s external deadline, partial output and 10 MiB
output budget remain. No full audit or inference entry is called.

## Regression boundary

Tests exercise the actual startup checkpoint adapter, original checksum reader, real
Serial coordinator/server/state-machine shutdown and Dual owner ready/inflight/settlement
paths with GPU computation replaced. Separate tests cover both TP finalizer callbacks,
empty buffers, real supervisor killing a blocked fsync, sticky flush/report errors,
receipt qualification, byte equality/order/bounds/counts, projection proof/corruption,
purpose separation/window clipping/TP union and large bounded bad-row scans. These are
CPU contracts, not GPU performance validation. Server next step: buffered Serial once,
then buffered PingPong once only after Serial execution/measurement/cleanup PASS.

Local final validation: 1736 passed, 3 GPU-marked skips; Ruff, compileall, Bash/runbook
syntax, Python 3.9 grammar and git diff --check passed. Linux CI runs the final pushed
commit (3.9/3.12 full suites, 3.11 contracts and pinned vLLM source contracts); its
final status is reported in the delivery. No CPU result is a GPU performance claim.
