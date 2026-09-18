# Lean-reference B128 diagnostic publication and console repair

Source archive: `pingpong-k3-delivery-a100-k3-b128-lean-reference-20260918T045239Z-465.tar.gz`.
Size118623801 bytes, SHA256`ecc5beb6d3a7deee198eb360afe09c6aeb46120ec40f8e9e569f16a91f050b17`.
All252 logical paths /230 unique objects verified for size/SHA256. Original input
is read-only; derived replay lives separately. Execution source was
`c1fca49f3f846ef94f0759b585fddaf091d78b62`, profile lean-reference.

## Original failure and scope

All four capacity probes passed. Forward-repeat Serial K3, Serial-eager K3 and
PingPong K3 completed their30s performance windows and normal cleanup. The third
point failed while publishing the derived audit report; forward PingPong-eager
and the entire reverse repeat did not start. Original error/exit1 and diagnostic
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
8416255 bytes (before source/config/inventory additions), exceeding8388608.
Of that body, `pingpong.cycles` occupies4524094 bytes and `pingpong.pipeline`
3340008 bytes. New cycle accounting is307859 bytes. The prior budget check used
52/55-step historical ordinary windows; this run has66. A small historical
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

The summary limit stays8MiB. Each shard is also at most8MiB; at most8 shards/64MiB
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
New summary:513123 bytes. All459 joined rows are in one7957178-byte shard, whose
SHA256 is`4c1a3ad40533d02855f01dc0cbf9c7394bbbb24268ee6f6f1addec4d8f9d51f8`.
Lossless restoration and requalification give COMPLETE with no errors. This is
new offline validation, not a rewrite of the original diagnostic failure or a
new GPU result.

CPU tests generate owner/backend joins, stress tables beyond the original8MiB
limit, publish, export and restore through the real archive. They reject missing,
corrupt, misordered, count/binding-conflicting shards and retain primary publication
failure through status/export. Actual drive -> summarize -> emit_result runs cover
four K3 modes for capacity/performance, preserving full files and bounded output.
Initial fixture setup used a prepost-only machine with a K3 argument; corrected
to the real K3 test machine. This was a test wiring error, not a production failure.
