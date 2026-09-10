# S2 schemas and retained evidence

All new documents use `specrhythm.s2-*.v1`; S1 v2 is unchanged. Canonical sealed JSON hashes
exclude the top-level `sha256` field. Per-attempt `seal.json` maps every retained regular file
name to its byte SHA256. A failed cleanup leaves an unsealed attempt for owned recovery.

| Artifact | Contract |
|---|---|
| `preparation.json` | Sealed S0 parent, verified S1 mixed100 source, current execution/config/model/patch/environment identities, declared seed pairs, static capacity formula. Static preparation is not permission to time. |
| `selection.json` | Selection seed/rule; nested 500-ID order; preferred S1 set hash; selected set hash. |
| capacity probe `actual-capacity.json` | Three actual rank records per mode: role, mode, physical GPU ID, queried UUID, block size/count, vocabulary size, real free/allocated/reserved/peak CUDA memory, effective Target worker evidence. |
| `capacity-plan.json` | Common requested/actual small and large sizes, class counts, nested IDs, each attempted decrement and nine capacity checks, shrink reason, distinct-scale flag, active/pool/query limits. |
| `slo-policy.json` | Per-class positive ms/token thresholds; explicit or independent calibration source; metric; engineering-not-paper label; optional class baselines and calibration seal reference. |
| `inputs/*/requests.jsonl` | Unmodified S0 request rows. Original Mooncake time and provenance remain here. |
| `inputs/*/trace.json` | `arrival_rate_qps`, `arrival_seed`, `order_seed`, set/order/unit-sample hashes; rows with request ID, unit exponential interval and high-precision planned arrival offset in seconds. |
| `inputs/*/execution-manifest.json` | `specrhythm.s2-execution.v1`; workload byte hash, exact IDs and set hash, separate trace, original split, requested/actual N, common active limit, frozen execution, fresh restoration definition. |
| `g0.json`, `G1.json`–`G3.json` | Frozen gate inputs and SLO/capacity hashes, then pass/skip status and immutable attempt-seal hashes. Empty G3 is an explicit capacity duplicate skip. |
| `decode-ready-context.json` | Existing validated provenance metadata plus S2 execution binding. Created once by the S2 coordinator before `LLM(...)`; never used as KV storage. |
| `decode-ready-manifest.json` | Existing real resident provider observations, exact prompt/bootstrap/positions/Target and Draft materialized lengths. Its historical internal setup boundary is not the S2 observation clock. |
| `draft-pool.json` | Actual owner-thread startup snapshot of resident Draft prefixes and private block IDs; timestamp and initial GPU memory. Written only outside timing. |
| `resident-pool.json` | Actual initial Target/Draft pool evidence, selected/resident/setup-terminal counts, per-Target-rank initial memory, prefill/setup time and fresh restoration method. |
| `s2-control.json` | Single-writer atomic transient control: barrier, states, cohort, admission, resource release and only still-needed initial proposal/enqueue metadata. No fsync, no token interpolation. Final value retained under the attempt seal. |
| `arrival-output-events.json` | Actual arrived/admitted/commit/finished/released/failed/barrier/drain events and final per-request state; original initial admission metadata. Full token arrays live here, not in the light summary. |
| `runtime.json` | Barrier/end ns, final requests/commits, Target step IDs/B/Q/cohort, final pool audit, final two-rank fence, real Draft shutdown response and Target memory. |
| `draft-backend-report.json` | Existing actual backend name, real proposal/commit metrics, cleanup/failure counters; S2 pool audit, live count before teardown, memory and dynamic cohort work records. |
| native `*.jsonl`, `plugin-report.json` | Existing Target structural diagnostics, round/commit/proposal/TP/lifecycle evidence. Offline S2 checks join these independently against the output ledger. |
| `population.json` | Monotonic event-time active-held and queued counts. FINISHED-but-syncing requests remain in active-held. |
| `result.json` | S2 validity, explicit errors/primary_error, mode/trace/SLO/execution identity, per-request aggregate metrics, overall/class summaries, offered/observed arrival spans, throughput/goodput, actual forward/B/Q/acceptance/overlap/resource metrics. |
| `command.json`, `exit-code.json`, `process-lifecycle.json`, `*-owner.json` | Real process commands, raw exit codes, PID/start identities, exact socket ownership and cleanup. Recovery sidecars are outside old attempts. |
| `light-summary-*.json` | Uploadable aggregate results, validity and bounded diagnostics plus raw directories, artifact names and hashes. Excludes full token arrays, full population, block tables, repeated raw logs and KV tensors. |

Timing fields are integer `monotonic_ns` on the same host. Rate/trace offsets and milliseconds
are finite numeric values. Optional distributions use `count=0` with mean/percentiles=None.
The request metrics are `queue_ms`, `arrival_handling_lag_ms`, `first_timed_token_wait_ms`,
`request_tpot_ms`, and `queue_inclusive_decode_avg_ms_per_token`. Bootstrap/timed/total output
counts are separate. SLO-good is defined only for requests with at least one timed token.

For PingPong, worker snapshot rows additionally contain `dual_uuid_query`, from the existing
`worker_dual_uuid_evidence` reader. This appears in startup `actual-capacity.json` Target worker
rows, prefill `resident-pool.json` Target initial-memory rows, and final `runtime.json` Target
memory rows. Fields retain `uuid_query_mode`, `uuid_initial_validation_count`,
`uuid_verification_subprocess_query_count`, `uuid_cache_hit_count`,
`uuid_verification_access_count`, and the real rank/device/UUID binding. Snapshot reads never
reset counters; one startup validation per rank and zero verification accesses are normal for
capacity probes. The nonempty-verification UUID A/B gate is not a capacity qualification rule.
Verification UUID intervals remain in the existing `verification-events.jsonl`; final counters
are read after the unchanged observation end and Draft shutdown. Target/Serial snapshots do
not acquire a Dual query object or these Dual-only fields.

Valid execution never depends on cross-run output equality or positive overlap/SLO attainment.
A missing required field, source/identity mismatch, incorrect accounting, native commit
mismatch or cleanup failure remains material; failures retain the primary error and raw log
tails. Summary/offline commands never dispatch an inference child.

S2 terminal drain uses an explicit `specrhythm.s2-terminal-drain.v1` receipt under
`draft-backend-report.json.s2_work_records[].terminal_drain`. It binds input rows and parent
states to actual results, independently queried Draft device identity, retirement, unrelated
KV scope/digests, and `release.materialized` prefix/block records. The release hook checks
private ownership across the full live pool, including newly materialized blocks, before
sampling `release.resources_released_ns` after the real release. Native state/work logs and
the runtime ledger independently bound this receipt; an operation name alone is insufficient.

New result fields are `stage_dependency_contract`, `active_stage_host_overlap_ms`, and
`terminal_drain.{work_count,work_host_ms,host_overlap_ms,same_cohort_host_overlap_ms,pair_count,
pairs,completions,definition}`. Completion rows keep `token_completion_ns`,
`draft_resources_released_ns` and `active_slot_released_ns` separately. Existing `overlap_ms`
and `physical_overlap` remain proposal-stage metrics. Existing `stage_host_overlap_ms` remains
the opposite-cohort host union, including cross-cohort drain; the three host unions must not
be added. Host-envelope intersections are not exact GPU kernel overlap. New failure details
use `field=s2_stage_dependency`, a specific `artifact`, all related `artifacts`, and an `actual`
record containing operation/terminal evidence and both stages' IDs/cohorts/intervals/IDs in
common. No historical result or seal is rewritten by this contract change.
