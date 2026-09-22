# K3 B64 diagnostic persistence and Draft dispatch

Historical reference: execution `899b54a6b58c0ea07046582c5a17934f630ac040`.
The four single-window throughputs were 132.04 / 138.07 / 131.51 / 119.76 tok/s.
The last point's admission and plugin-report fsync were slower. These inclusive,
cross-process intervals cannot be subtracted from throughput or added as wall time.
Complete output equivalence remains NOT_RUN; historical Serial output differences
are unresolved.

## I/O-only

`observation=deferred-window` is explicit. The original and buffered-live paths
remain available, including the unchanged B16 default.

|Producer / files|Consumer|New behavior|
|---|---|---|
|CheckpointJsonl: round-events, target-diagnostics, transport-events, proposal/verification/request-state/scheduler/proposal-lifecycle/draft-work/draft-transport streams|Offline audit; in-process CheckpointJsonl readers|Immediate immutable framing/checksum and protocol capture; bounded RAM; readers validate a memory snapshot without flushing|
|Resident admission-events with the audited serial-consumer schema|Offline resident audit|Same live decision/checks and complete rows; delayed persistence|
|Other admission/control records|Startup/readiness/control protocol|Existing live write path|
|RemoteDraftProposer plugin-report|Post-run phase4 performance and serving qualifiers|Keep existing proposer state; coalesce build requests; build and atomically publish once during finalization|
|Control/deadline/drain/ownership/error/lifecycle snapshots|Coordinator, supervisor, Draft owner|Unchanged live publication|

Bounds are per process: 100,000 encoded records, 256 MiB pending encoded bytes,
one deferred plugin report at most 64 MiB serialized. Overflow fails; it never
flushes synchronously to create space. Peak encoded bytes/records, producer and
written counts, per-phase enqueue/encoding cost, final write cost, per-file fsync,
physical byte digest, final report digest and shared absolute deadline are recorded.
Python allocator overhead is not claimed as measured RSS. Existing trace phase is
used, with no extra control reads or GPU synchronization. A receipt's own final
publication and pre-observer startup I/O are outside its recorded fsync coverage.

The 60-second absolute drain deadline is never regenerated. Incomplete/failing
flushes remain FAILED, retain acquired bytes where possible under an existing
deadline, and cannot qualify based on exit code alone. Abrupt process death can
lose unwritten RAM; its absent/incomplete final receipt remains an evidence failure.
CPU tests cover real proposer → checkpoint adapter → worker finalizer → physical
JSON/JSONL reread. They do not establish A100/DPC behavior or GPU performance.

## Unified dispatch and what was already batched

`PingPrePostMachine.step` already collects pending parent settlements plus
`_background` normal extensions/lookahead. `PingPrePostBackendMixin._forward_prepost`
merges those ragged rows in **one** `worker.materialize`; `token_step` also mixes
ordinary and eager work. A rejection requires real correction/seed and subsequent
autoregressive extensions; it is not synonymous with a recovery-only forward.
No historical dispatch-time runnable inventory exists, so the count of historical
missed ordinary batching opportunities is **unknown**, not zero.

Both new configurations record the same owner-thread inventory immediately before
physical dispatch. Row bindings include source, home, prefix version, proposal,
round, continuation and task kind; a per-backend physical ID is joined to the
native device interval by purpose/B/host containment. Deferred reasons distinguish
Target feedback, own next-step dependency, READY, termination, capacity and no
current work. The owner is single threaded and each returned token step is fenced;
there is no unfenced in-flight request presented as runnable. The raw physical
report contains the complete bounded inventories; the compact report retains
counts and joins, not a second copy of every inventory.

The `unified` policy checks that every executable non-promotion task fits the same
physical call whenever capacity permits. It reuses the existing materializer,
including correction/seed plus another request's ordinary extension. A request
contributes at most one next step. With no peer work, recovery proceeds immediately.
There is no new batching wait, sleep, quota, group barrier or change to Target
admission. The physical ceiling remains 64. Recovery-only calls can remain legal
when peers are READY or awaiting feedback.

One confirmed ordering change: full K3 reuse needs no new forward. Previously its
READY publication followed the whole common materialize call for unrelated
requests. In unified mode, completed legal promotions are validated, fenced and
published first, then the step returns to the **real owner command loop**. Ordinary
and rejection work stay pending for the next common token step. Serial's all-Draft
idle gate still rejects admission until that work completes. A newly arrived
feedback/stop/claim can be handled at this boundary. Successful reuse never adds a
fourth candidate. EOS/budget truncation, invalidation and rollback are unchanged.

This does not establish fewer physical calls on the historical workload. Tests
show legacy and unified already mix simultaneously runnable recovery and ordinary
rows. The new inventory distinguishes true missing opportunities from real
unavailability. Compare `io-only` and `unified` at the **same final execution SHA**,
with identical diagnostic collection, to assess the READY boundary and dispatch
checks; never attribute a historical logging-mode difference to scheduling.

## Evidence scope and validation

Run `SR_K3_EXPERIMENT=io-only` or `unified` through the dedicated pinned entry.
Both select deferred-window, performance-exploration, frozen resident360, K3,
B64/A32+B32, 128 actual warmup opportunities, setup900/drain60 and a 30-second
window. The order is Serial, Serial-eager, PingPong, PingPong-eager, then reversed.
Each repetition has separate capacity/performance roots. There is no Target-only
or independent full-output prerequisite. Output equivalence stays NOT_RUN.

All eight measured points and first failure are flattened into one deduplicated
archive, retaining the original bounded per-repeat exports. Aggregate export limits
are explicit: two repetitions, 8 GiB unique payloads, 1,024 unique / 1,200 logical
files. No truncation counts as complete. Local run storage, verified DPC copy,
local fallback and sole outer UPLOAD ONLY remain unchanged. Both repetitions'
raw throughputs and ranges are reported. Self-parent eager overlap and other-home
normal/recovery overlap retain existing native interval union semantics. Role
membership must not multiply physical forward counts or add role GPU intervals.

Known test development failures: the first focused old logger run had two
subprocess/drain-budget failures; the same unchanged tests subsequently passed.
This does not establish their original cause or resolve prior CI flakiness. A new
module initially collided with an existing Target dispatch analysis module and
caused collection ImportError; it was renamed to `k3_physical_dispatch_evidence`,
and the existing module was restored byte-for-byte. No timeout/assertion was loosened.

Dispatch inventory is additionally limited to 4,096 physical token dispatches per
phase and at most the configured live request limit per snapshot (64 here).
Terminated request identities have a separate bounded list; inactive resident KV
is not reread for this diagnostic. Budget exhaustion fails before another dispatch;
limits, per-phase counts and peak live rows are in the owner report. Native records
also carry per-producer/run forward indices for setup/refill calls. Token dispatch
summaries do not silently include those calls as normal/recovery work.

Separating completed promotion settlement can add a necessary fence/audit boundary
before the remaining common materialization. The GPU experiment must measure that
cost too; no positive net effect is assumed. Diagnostic encode/enqueue time remains
inside the window, and write-record/flush-batch metrics are not OS syscall counts.

First local full run: six failures. Two old shell tests returned 127 because this
invocation used an absolute Python executable without its environment on PATH;
the child helper executes `python`. The remaining four were subprocess timeouts
(three geometry-entry cases at 20 s and an old teardown case at 5 s). The eight
corresponding parametrized baseline cases at `5511f89` passed in an independent
read-only source copy with the environment on PATH. This establishes the baseline
check result, not the cause of the intermittent timeouts. Final full validation
uses the same Python environment on PATH; no test limit/assertion was changed.

Final local full-suite invocation with the correct environment: **2 failed,
2,931 passed, 3 skipped** (890.87 s). The remaining failures are the existing
`test_target_failure_terminates_draft_without_unbounded_wait` subprocess limit
(5 s), and `test_actual_fork_cannot_share_parent_buffer_or_forge_child_complete`
(the child's valid 2 s diagnostic deadline expired). The original buffered-live
`DiagnosticLogs` class and teardown helper/test are unchanged by this work. The
same fork case also passed once in the read-only `5511f89` baseline. These checks
do not establish the cause of the full-suite failures; full local validation is
**not PASS**, and no timeout, assertion or budget was relaxed. New targeted cases
and fixed-entry checks do not erase these failures.

Final affected tests on Python 3.9 and 3.12: **126 passed each**. The pinned entry's
actual Bash launch/first-error/single-upload cases: **5 passed**. The two configs
use the same source; only the explicit dispatch policy differs. CPU tests replace
GPU worker execution and do not prove A100 overlap, performance or filesystem
behavior. Output equivalence is deliberately NOT_RUN. Remote CI is reported from
the actual pushed SHA separately; before upload there is no new CI result.
