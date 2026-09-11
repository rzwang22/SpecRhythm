# Fixed diagnostic stop artifacts

This contract applies to new fixed64/32 attempts. Historical artifacts are immutable;
no old result is silently upgraded. Normal S1/S2 result contracts are unchanged.
All `*_ns` execution timestamps use the same host monotonic clock. UUID/device identity
comes from the real worker, never a rank guess. Cross-run token/length/EOS/round equality
remains `NOT_REQUIRED`.

| Artifact | Schema / evidence |
|---|---|
| `runtime.json` | `specrhythm.fixed-runtime.v2`; existing window, actual steps/commits, population, device/timing evidence plus `diagnostic_drain` |
| `measurement-snapshot.json` | `specrhythm.fixed-measurement-snapshot.v1`; atomic compact checkpoint before drain and after completed steps, best-effort on error |
| `drain-state.json` | `specrhythm.fixed-drain.v1`; one absolute deadline, phase, settled count, compact receipts, start/end and completion/failure |
| `draft-drain-state.json` | `specrhythm.fixed-draft-drain.v1`; owner-side completed receipts, live physical count, unresolved proposal count and first settlement error |
| `diagnostic-primary-error.json` | first failure with phase, actual exception, timestamp and traceback; saved before later cleanup/report attempts |
| `diagnostic-secondary-errors.json` | later distinct cleanup/report failures; cannot replace the first error |
| `exit-code.json`, `process-lifecycle.json` | actual coordinator/Draft/effective exits, timeout detection, owned termination actions and remaining PIDs |
| `result.json`, `light-summary.json/.csv` | qualified execution/measurement status; partial snapshot/drain retained separately when final reports are absent |

The measurement snapshot records mode/point, git commit, immutable execution SHA and
configuration identity, stop reason, window start/end, completed steps/sample count,
actual committed window tokens, step wall statistics, actual batch histogram and peak
held-slot/cohort occupancy. `measurement_complete` describes reaching the intended
window boundary; it does not mean qualification passed. `drain_complete` is separate.
`measurement_availability=UNQUALIFIED_PARTIAL` permits inspection only; zero measured
steps are `INSUFFICIENT`, a capacity probe is `NOT_APPLICABLE`. A snapshot always has
`formal_comparison_eligible=false` and pending checks. It never substitutes for final
Target/device/accounting/lifecycle evidence. The finalized light result can separately
report `QUALIFIED` after its existing material checks succeed.

Stop receipts contain request ID, initialized/admitted/terminal status, disposition,
authoritative prefix hash/count/version/round, previous Draft prefix hash/count,
committed synchronization count, unused proposal ID/round/count, and
`new_proposals_generated=0`. They bind actual worker UUID/device, internal request ID,
private/unrelated KV audit digest and physical release time. Full prefixes are sent
once to the stop protocol, not repeatedly written into compact diagnostics. Serial
proposals use the existing request/round identity (no fabricated proposal UUID).

`NATURAL_TERMINAL` requires real Target terminal evidence. `DIAGNOSTIC_CANCELLED`
never supplies EOS, normal completion time, accepted tokens, or committed tokens.
Actual allocator success is required before logical release. Already-natural release
receipts retain the physical release timestamp captured when it happened. A repeated
identical settlement returns the original receipt; a changed authority fails closed.
A receipt can survive a later deadline/RPC failure without upgrading execution to PASS.

The coordinator waits for issued asynchronous work, fences Target, physically aborts
remaining Target requests, settles/releases Draft state on its actual owner and calls
normal shutdown. V2 qualification requires every request's receipt, zero physical live
Draft requests **before** shutdown, matching final prefix/disposition, and physical
release no later than logical release. Ordinary Serial's unresolved-proposal shutdown
guard remains intact. The zero-verification capacity probe has no resident requests
and requires no proposal or settlement receipts.

`deadline_ns` is set once before drain and remains active through coordinator teardown.
Every RPC/owner wait consumes its remainder. The owned supervisor also reads it while
the coordinator is blocked; expiry records failure and effective rc=124, followed by
the existing bounded TERM/KILL grace. `COMPLETE` in drain-state means the stop protocol
finished; only the final process-lifecycle artifact establishes clean process exit.
A killed worker can leave drain-state at `RUNNING`; the summary reports the supervisor
failure alongside that last observed state, without rewriting the artifact.

Measurement tokens include only real commits inside the original window. Drain sync
replays already committed authority into Draft KV and cannot add window tokens. Window
wall time includes an issued step's actual overshoot; drain and total execution costs
remain separately reported. Nonzero rc, incomplete cleanup/accounting/evidence, or an
operator-interrupted budget prevents formal numeric comparison. Empty/incomplete shape
samples remain insufficient. `partial_measurements` displays retained failures
separately with `formal_comparison_eligible=false`.

Default summary and small bundle read compact artifacts only. They work without final
runtime/backend reports and include snapshots, drain summaries, first/secondary errors,
exit and ownership evidence. Full request/token logs remain in the original root;
`audit` is a separate optional CPU command, never an implicit short/summary/bundle gate.
# Offline attribution artifacts (separate from runtime qualification)

`specrhythm.fixed-attribution.v1` is emitted only into a new sibling analysis directory.
`status=COMPLETE` means analysis completed, never GPU qualification. Each attempt has a
retained summary, explicit missing-file/association gaps, and optional raw analysis.
Raw `OBSERVED` means supported associations/bounds, not a complete causal critical path.
Old results are not upgraded. `formal_comparison_eligible=false` is unconditional here.

- `attribution.json`: per-attempt source paths/hashes, analyzer source hashes, aggregate
  clues, actual step/rotation associations, bounded CUDA event overlap, host inclusive
  costs/unions and purpose-separated Draft forwards. Details can be capped; counts and
  aggregate statistics use all eligible events. No raw input traces are copied here.
- `summary.csv`, `report.md`: compact retained metrics with **independent** overlap status.
  A historical reported ZERO stays separate from UNKNOWN when raw coverage is absent.
- `progress.json`: current/processed files, phase, input bytes and processed record count.
- `analysis-exit.json`: actual child exit, elapsed time, shared deadline, no-GPU/no-audit
  evidence. Codes 0/2/124/130 mean completed analysis/input failure/timeout/operator stop.
- `export-manifest.json`, `evidence/runs/.../*.gz`: optional allowlisted projections with
  original uncompressed file hashes and output hashes. `source_record_sha256` is the
  original JSONL record checksum, not a checksum claiming the redacted row is original.
  `source_line` is a source locator, not a fabricated runtime event ID. Original clock
  bounds, uncertainty, identity, cohort/round/prefix metadata and small commit tokens
  survive. Full prefixes, generated sequences, KV maps and general logs do not.

CUDA status is UNKNOWN for missing/inconsistent coverage or clock identity, UNCERTAIN
for positive upper bound only, ZERO for zero upper bound **with complete supported
coverage**, and POSITIVE for positive lower bound. `exact_kernel_overlap=false` always.
Exclusive host self-time and critical-path time saved remain null where not observable.
Rows outside the formal window are not silently charged to measured token accounting.
The runtime measurement/drain schemas below are unchanged.

## Opt-in buffered-live receipts and timing addendum

See [runtime design](fixed-buffered-runtime-design.md) and
[two-mode runbook](fixed-buffered-runtime-runbook.md). `options.observation` is frozen;
`original-live` remains the default. `buffered-live` adds one
`fixed-logging-<pid>.json` (`specrhythm.fixed-logging.v1`) per process:

- role/PID, observation, max_buffer_records/bytes, produced_records, written_records,
  pending_records/bytes, flush_count, fsync_count, peak_buffer_records/bytes,
  oversized_direct_records, append_blocking_ms, flush_blocking_ms, flush_reasons;
- per-stream produced/written counts and confirmed written-byte SHA256; original-mode
  bytes are delegated to native append and its incremental digest is not computed;
- final_flush start/end/deadline/records, status, counts_final, integrity_complete,
  primary logger error and bounded secondary errors. RUNNING counts are not final.

`append_blocking_ms` includes nested capacity flush; do not add it to flush_blocking_ms.
These counters cover CheckpointJsonl writes; host log_fsync timers can include other
mandatory file writes. Final receipts are separate from pre-shutdown backend/worker
snapshots, and are retained in small bundles. Four complete, conserving receipts and
matching drain times are mandatory only for buffered mode. No old result is upgraded.

`recorded_gpu_costs` (`specrhythm.fixed-recorded-gpu-costs.v1`) separates
D_proposal / D_commit_or_prefix_sync / V_target / other_recorded_Draft_GPU, window launch
counts/sums, approximate per64-rotation allocation, per-rank Target sums, clipped physical
union bounds and outside-observation time. `all_gpu_work_covered=false` is explicit.
`pipeline_stage_semantics` labels D32/D64 aliases as proposal-only. New light results
also expose target_host_envelopes_by_rank and host_costs_by_process without raw intervals.
CUDA device reports add projection_version=integer-anchor-v2, anchor_before_ns and
anchor_after_ns; existing elapsed times are retained, projection width becomes exact.
Raw offline analysis adds clock_failure_details with inspected/failure counts and at
most 16 original failing rows per attempt; legacy UNKNOWN remains UNKNOWN.
