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
