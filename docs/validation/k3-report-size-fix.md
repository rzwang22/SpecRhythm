# Lean-reference B128 diagnostic publication and console repair

Source archive: `pingpong-k3-delivery-a100-k3-b128-lean-reference-20260918T045239Z-465.tar.gz`.
Size 118623801 bytes, SHA256 `ecc5beb6d3a7deee198eb360afe09c6aeb46120ec40f8e9e569f16a91f050b17`.
All 252 logical paths / 230 unique objects verified for size/SHA256. Original input
is read-only; derived replay lives separately. Execution source was
`c1fca49f3f846ef94f0759b585fddaf091d78b62`, profile lean-reference.

## Original failure and scope

All four capacity probes passed. Forward-repeat Serial K3, Serial-eager K3 and
PingPong K3 completed their 30 s performance windows and normal cleanup. The third
point failed while publishing the derived audit report; forward PingPong-eager
and the entire reverse repeat did not start. Original error/exit 1 and diagnostic
failure remain unchanged. Export and verified DPC delivery succeeded. No GPU
crash, cleanup timeout, trace loss or output-equivalence failure is inferred.
Output equivalence is intentionally NOT_RUN.

|Completed point|Steps|Tokens|Window ms|Measured tok/s|
|---|---:|---:|---:|---:|
|serial-k3|26|7533|30299.183341|248.620562|
|serial-eager-k3|28|8112|30457.163301|266.341285|
|pingpong-k3|66|9547|30236.772606|315.741370|

These are original qualified measurements, not a complete paired experiment or
stable performance conclusion. The optimization configuration was not tested.

The precise production chain is `ping_prepost_evidence.report -> analyze ->
audit_layer_report.write`. The third point's analysis body alone encodes to
8416255 bytes (before source/config/inventory additions), exceeding 8388608.
Of that body, `pingpong.cycles` occupies 4524094 bytes and `pingpong.pipeline`
3340008 bytes. New cycle accounting is 307859 bytes. The prior budget check used
52/55-step historical ordinary windows; this run has 66. A small historical
projection was insufficient coverage for data that grows with request/step count.
This is an implementation/validation omission, not a GPU failure.

Separately, `decode_scan_results.emit_result` printed the entire retained result,
including admission histories and logging receipts. The terminal text is long
report JSON, not thousands of new errors or per-token hot-path disk writes.

## Repair and invariants

For K3 only, completed result publication prints a fixed small field selection
with statuses, original exit code, step/token/window/throughput and full-report
path. Complete JSON/CSV artifacts retain their data. Print occurs after publication;
a failed write cannot print the success summary. Scaled prepare prints its config
path instead of the resident360 config. Legacy non-K3 console behavior is retained.
GPU work, K3, sampling, geometry, warmup/window, control encoding switches, READY,
logging buffers and absolute drain deadlines are unchanged.

K3 audit publication moves four derived row tables into ordered JSONL shards:
`pingpong.cycles`, `pipeline.dispatch.rows`, `pipeline.timeline`, and
`draft_dispatch.calls`. The summary retains all aggregates, status, native geometry,
step ledgers and declared representative cycles. The index specifies table paths,
row counts, binding, per-file bytes and SHA256. Missing data is not converted to an
empty table: absent fields remain absent. Every original row is recoverable.

The summary limit stays 8 MiB. Each shard is also at most 8 MiB; at most 8 shards / 64 MiB
of joined detail per point. This is a declared bounded detail budget, not an
increase of the summary limit or unbounded history. The original raw/total archive
budgets remain. A single oversized row, exhausted shard budget, malformed/missing
shard or write/readback error fails explicitly. Partial files remain exportable.
No files are written on the measurement path by this change.

All original qualification and dispatch checks execute before projection.
The publisher closes shards, writes the index last, re-reads and verifies it and
the shards before marking report publication complete. Comparison restores and
checks the detail contract. The single exporter includes shards, rejects missing
or damaged referenced evidence, and the flattened archive retains logical paths
and digests. Users still upload only one archive per invocation.

If report publication fails, evidence-status records FAILED plus original run
qualification, exception type/details and report path. First-failure projection
includes this record. A secondary status-write failure is reported separately and
cannot replace the original exception/code. No forced COMPLETE or larger timeout.

## Real evidence replay and CPU regression

Replayed the real archived config/point/runtime/backend/light-summary through the
actual report producer and qualifier, writing to a separate derived directory.
New summary: 513123 bytes. All 459 joined rows are in one 7957178-byte shard, whose
SHA256 is `4c1a3ad40533d02855f01dc0cbf9c7394bbbb24268ee6f6f1addec4d8f9d51f8`.
Lossless restoration and requalification give COMPLETE with no errors. This is
new offline validation, not a rewrite of the original diagnostic failure or a
new GPU result.

CPU tests generate owner/backend joins, stress tables beyond the original 8 MiB
limit, publish, export and restore through the real archive. They reject missing,
corrupt, misordered, count/binding-conflicting shards and retain primary publication
failure through status/export. Actual drive -> summarize -> emit_result runs cover
four K3 modes for capacity/performance, preserving full files and bounded output.
Initial fixture setup used a prepost-only machine with a K3 argument; corrected
to the real K3 test machine. This was a test wiring error, not a production failure.

## Fixed delivery and validation boundaries

Execution commit: `24a5042d8bce503699bb040f293541f853e1e6b1`.
Fixed launcher: `a7d34cdbe5e65606a14eab3cec9b3704a8f53d74`.
Both lean configurations use the identical report repair. Actual fixed-launcher
regressions passed 7/7 on Python 3.12 and 7/7 on Python 3.9, including first-error
exit and exactly one upload path. The 31-test Python 3.9 affected suite passed.
Ruff passed; compileall on 3.9/3.12, 410-file Python 3.9 syntax parsing and all
20 Bash entry syntax checks passed. Original no-GPU production replay succeeded.

The first affected Python 3.12 suite had 65 passes and two existing 20-second
subprocess harness timeouts in `test_k3_actual_joint_first_failure_and_export`
(`False-pingpong-eager-k3`, `True-target`). Those failures are retained; no timeout
was increased and a later successful run cannot establish their cause.

Prior HEAD `bbd0481` CI run 35261230780 finished with Python 3.12 failure:
`test_real_owner_dispatch_initial_work_wait_and_window_drain[pingpong-True]`
raised `diagnostic owner settlement deadline expired` (3066 passed, 1 failed,
36 skipped). Python 3.9 was cancelled; both contract jobs passed. This precedes
the report fix and is separate from the server's 8 MiB failure. Its scheduling
root cause is unresolved; this delivery does not claim to fix it. No CI retry
has been used to replace that failure. Current-entry CI is tracked separately.

Final local full Python 3.12 run: 3105 passed, 17 failed, 3 skipped (1223.44s).
Nine new console-test failures were a fixture isolation defect: prior production
launch fixtures leave `SR_PHASE4_NUMERICAL_DIAGNOSTIC_PLAN` in the test process,
which correctly conflicts with lean mode. The console fixture now isolates its
own workload/profile inputs, following the existing target-lean fixture; real
profile conflict rejection is unchanged. With an explicitly contaminated input
environment, the affected/lean/entry suites passed 55 tests (2 source-audit skips)
on Python 3.12; report/console suites passed all 24 tests on Python 3.9.
The other eight full-suite failures were six old 20s subprocess timeouts, one
supervisor deadline assertion (124 instead of 0), and one fork-buffer 2s drain
expiry. Their causes remain unresolved. No assertion or timeout was relaxed.
No second full-suite success is claimed. Full failure IDs and focused outcomes
are retained in [local checks](k3-report-local-checks.json).

The test-input isolation follow-up changes no production source; pinned execution
24a5042 and launcher a7d34cd still contain the complete server repair. Server
revalidation is PENDING. Original run qualifications and diagnostic failure stay
unchanged, and the incomplete eight-window experiment is not presented as complete.

Completed fixed-entry CI snapshot (run35312914255, a7d34cd): Python3.12 had
3082 passed/9 failed/36 skipped. Its nine failures are exactly the inherited
full-plan conflict in console fixtures, addressed by b09827f test isolation.
Both contract jobs passed; Python3.9 was cancelled. Subsequent CI is PENDING,
not presumed passed from the focused result. The previous CI failure is retained.
