# Fixed64/32 b424 timing attribution

Evidence: the operator's `fixed64-stop-b42401585a6b-20260911T085722Z-1496`
small bundle, inspected offline on 2026-09-11. Baseline and initial branch HEAD were
`b42401585a6b32e36fd0b874eb106a8f75ee9ea4`, with a clean worktree. PR #4 was
Draft/Open. Old artifacts are immutable. No AutoDL connection or GPU run occurred.

## What the retained summaries establish

All four continuous points report execution, measurement and cleanup PASS. The bundle
does not contain runtime.json, draft-backend-report.json, or raw event files. Numbers
below are retained aggregates, not independently reconstructed rotations.

| Mode | Window ms | Tokens | tok/s | Mean full64 rotation ms | D proposal / V forward mean ms | Drain ms |
|---|---:|---:|---:|---:|---|---:|
| Target | 4457.589 | 768 | 172.290 | 371.390 | — / 80.021 (B64) | 2234.492 |
| Serial | 13730.581 | 2296 | 167.218 | 1144.140 | 60.626 / 158.800 (B64) | 1832.144 |
| Serial-split | 24857.839 | 2280 | 91.722 | 2071.407 | 110.442 / 111.594 (B32) | 2606.694 |
| PingPong | 19156.602 | 2280 | 119.020 | 1596.297 | 65.698 / 109.892 (B32) | 2716.662 |

Serial progress is 2.989583 tokens per request/verification; grouped progress is
2.968750. These are continuous-state means, not matched initial-state stages. The
Serial/PingPong rotation gap is 452.156 ms. Adding the quoted forward means gives
219.426 ms for Serial and 351.180 ms for two nonoverlapping B32 halves, a difference
of 131.754 ms. The remaining arithmetic difference, 320.402 ms, is **unattributed**:
it includes any uncounted GPU work, synchronization, IPC, host work and idle time.
It is neither measured CPU time nor a predicted removable cost. The 219.785 ms
`2*max(D32,V32)` estimate assumes sufficient overlap and is not a measured rotation.

Split minus PingPong window time is 5701.237 ms. The reported split-only
`wait_draft_owner` union is 5606.630 ms; Target-step unions differ by 48.340 ms.
The residual arithmetic difference is 46.267 ms. Source confirms that split calls
`fixed_runtime.wait_draft()` before each step, whereas PingPong does not. This is
strong evidence of a host-wait accounting relationship, **not** proof of 5.6 seconds
of GPU overlap or removable work. Split also performs 3332 observed IPC calls versus
120, and has different observed D32 durations; waiting, polling and logging interact.

## Observed overhead, not additive critical-path attribution

Each cell is count / clipped cross-process interval union in milliseconds.

| Category | Target | Serial | Serial-split | PingPong |
|---|---:|---:|---:|---:|
| checkpoint_log_write | 2736 / 1258.048 | 2760 / 1545.322 | 14877 / 8221.851 | 11665 / 5003.309 |
| log_fsync | 2736 / 983.962 | 2772 / 1136.231 | 14899 / 5724.828 | 11689 / 3899.186 |
| json_serialization | 9445 / 274.810 | 37537 / 1040.883 | 102120 / 2716.707 | 82764 / 2115.854 |
| live_uuid_validation | 0 / 0 | 0 / 0 | 48 / 2200.928 | 48 / 2078.598 |
| nvidia_smi_subprocess | 0 / 0 | 0 / 0 | 48 / 2199.544 | 48 / 2077.160 |
| ipc | 0 / 0 | 12 / 3357.461 | 3332 / 2702.433 | 120 / 506.516 |
| required_tp_barrier | 0 / 0 | 48 / 11.211 | 96 / 1681.111 | 96 / 1688.077 |
| resident_block_audit | 24 / 23.510 | 72 / 79.612 | 144 / 226.655 | 144 / 198.847 |
| scheduler | 12 / 743.735 | 12 / 726.801 | 24 / 2523.114 | 24 / 2455.278 |

`fixed_observe.install_host_observation()` wraps all modes with the same original-live
configuration. `transport.CheckpointJsonl.append()` hashes a canonical payload, encodes
the checksummed record, opens/writes/flushes and fsyncs. The two JSON encodings have
different content; they are not proven redundant. JSON/fsync timers nest inside log
writes. Dual `for_verification()` nests a real worker snapshot/subprocess query.
`vllm_dual.on_target_verify_end()` invokes it once per TP rank/verification. The Serial
remote proposer has no such per-verification query call: its zero is a path difference,
not evidence of an optimized query. Startup device identity remains checked. Target-only
has no Draft RPC; its zero is expected. Other zero categories in a compact report are
not independently distinguishable from missing observation without raw coverage.

Other wrappers cover both client classes, GroupCoordinator.barrier, ResidentPoolAudit,
and scheduler.schedule. Cross-process union does not add simultaneous TP ranks. It also
does not show which lane was blocked. Legacy timers have no thread, parent-call or call
IDs: exclusive self time and a complete causal critical path remain UNKNOWN. Summing
these nested categories is invalid. In particular, removing a 5-second observed log
union does not establish a 5-second wall-time improvement.

`FixedDraftBackend.propose_many()` bounds D samples around proposal generation; its
first token uses cached logits. K4 normally needs three proposal forwards; commit/prefix
materialization is separate. `fixed_results` selects proposal/commit forward hooks and
bins them by the proposal host span. Raw purpose-tagged forwards are required to verify
membership and report commit, prefill and other purposes separately. CUDA hooks cover
the model call, not all logits/sampling, KV manipulation or fences. Uncovered work must
not be renamed CPU computation.

## What remains unproved and the minimal next evidence

The retained `physical_overlap_status=ZERO` is a reported result. Its implementation
can also return ZERO for an empty Draft event list. The small bundle omits individual
forwards, anchors and associations, so **independent overlap status is UNKNOWN**.
No conclusion about Draft finishing during Target preprocessing, delayed enqueue,
early waits or redundant synchronization follows from this bundle.

Read only these files from each retained Serial and PingPong attempt (split optional):

1. `runtime.json`: actual scheduled rows/internal IDs, stable request IDs/cohorts,
   prefix lengths, committed rounds/versions, output commits, measurement boundaries,
   coordinator/TP host intervals and purpose-tagged CUDA bounds/identity.
2. `draft-backend-report.json`: fixed proposals with IDs/round/context lengths,
   fixed device forwards/clock uncertainty, host intervals and S2 owner work records.
3. `draft-work-events.jsonl`: owner operation/result proposal identity, sync/ready times.
   `draft-transport.jsonl` supplies server RPC intervals only; its schema does not retain
   request/correlation IDs or a distinct enqueue/owner-receive timestamp. Those cannot
   be invented, even with the full root.

The runtime already retains the compact committed rounds/Target rows, so copying full
Target diagnostics, prefixes, KV block structures or all logs is unnecessary. CUDA
anchor brackets are encoded in projected lower/upper bounds plus uncertainty; absolute
anchor timestamps were not separately retained. The export must preserve those fields
and source hashes, without fabricating missing timestamps. Worker log end timestamps
follow publication and may include earlier record writes; they are not exact ready times.

## Scope decision

Implement a bounded CPU-only attribution/export tool and regression tests. Do not
change runtime scheduling, logging cadence, waits, UUID queries or algorithms on this
evidence. This is the request's explicit insufficient-trace fallback. It preserves
K4, 64/32 caps, models, precision, workload/cohorts, proposal/commit accounting, private
KV, prefix/version checks, original-live observation, bounded drain and S1/S2 defaults.
Cross-run equality remains NOT_REQUIRED. No new GPU test is needed to collect these
already-retained files; GPU performance validation remains PENDING.
