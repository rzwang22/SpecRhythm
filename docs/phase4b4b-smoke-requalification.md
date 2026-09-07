# Offline repair of the two-request PingPong smoke qualification

The qualifier at `9d176939673661a0621a28cc2ddccf6700a70afc` incorrectly
computed the smoke end from the maximum `timestamp_ns` among `timing-events.jsonl`
rows whose `event` equals `measured-token-commit`. That collection can legitimately
be empty. The resulting Python exception was caught and serialized as an invalid
qualification, even when runtime execution had succeeded.

## Retained-event contract audit

* `phase4b4b_run smoke pingpong` invokes `phase4b2_run_mode dual`, which enables
  `SR_PHASE4B2_PERFORMANCE=1`. It skips **offline formal measurement**, not all
  runtime performance instrumentation. No `decode-performance.json` is required
  for this smoke.
* `DualBatchRemoteProposer._rank_zero_update` records ordinary verified proposal
  commits in `proposal-events.jsonl`, including `commit_end_ns`. It does **not**
  emit a `measured-token-commit` timing event for these commits.
* The same callback calls `record_performance_commit` for a proposal-free,
  one-token terminal Target tail. That helper emits the timing event only when
  runtime performance instrumentation is enabled. A run completing both requests
  through verified proposals therefore has no such events, despite a real
  performance start timestamp.
* The formal performance reader, `performance._commit_events`, already combines
  proposal commit timestamps with explicit timing commits for Dual. Its behavior
  and formal start/end boundaries are unchanged by this reporting fix.
* Both ordinary verified termination and proposal-free Target-tail termination
  call `_transition(..., "TERMINAL")`. The retained `request-state-events.jsonl`
  contains the actual monotonic timestamp. Existing request-state validation
  checks legal, contiguous transitions ending in TERMINAL. Raw runner validity
  also checks the request lifecycle.
* `plugin-report.json` retains the TP-published `measurement_start_ns` and, when
  enabled, `performance_measurement_start_ns`. Initial Draft proposals are
  enqueued after the applicable start. No timestamp is inferred from a rank,
  file modification time, output count, or the time of offline reporting.

## Smoke reporting semantics

`structural_execution_window` records `start_ns`, `end_ns`, `start_source`,
`end_source`, `terminal_request_count` and a scope label. The start uses the
recorded performance start when present, otherwise the recorded correctness
decode start. A malformed present start fails validation. The end is the latest
TERMINAL timestamp after validating the complete expected request set and its
state transitions. Missing or invalid mandatory evidence still fails closed.

The smoke remains `performance_result=false`. Pipeline fill runs from this
runtime decode start to the first Target forward; structural drain runs from
the first completely terminal cohort to the last request TERMINAL. Subsequent
Draft synchronization and process cleanup are excluded from that interval but
still independently required to succeed. The fill/drain scope strings say so.
`overlap_fraction_of_makespan` is `null` for smoke;
`overlap_fraction_of_structural_execution` uses the explicit structural window.
No throughput, TPOT or formal makespan is manufactured.

Corrected-5 and corrected-100 still use **only** `decode-performance.json` formal
measurement boundaries, retain their prior metric values and scope labels, and
require real multi-request Draft proposal forwards. Corrected-5 still requires
positive physical cross-cohort overlap. Corrected-100 still requires 100 requests,
the 16-token output limit and 1487 measured committed tokens; valid zero overlap
remains descriptive. Runtime scheduling, backend execution, Target, cohort policy,
logging and the five vLLM patches are untouched.

## All min/max collections in the PingPong qualifier

| Reduction | Contract and empty handling |
| --- | --- |
| Smoke terminal maximum | Mandatory exact, nonempty terminal membership and valid state clocks checked first. |
| Initial proposal end maximum / verification start minimum | Both initial cohorts and nonempty verification evidence are required before reduction. |
| Per-verification TP interval min/max | Missing/empty rank intervals produce an explicit mandatory TP timing error; existing TP/device/CUDA checks remain. |
| First Target forward minimum | Empty forward evidence after the start is a material error. |
| Per-cohort terminal maximum / first-cohort drain minimum | Nonempty A and B membership and every request's TERMINAL evidence are required. |
| Interval union, intersections and wait clipping | Two scalar operands; optional intersection/wait/clipping collections may be empty and yield zero counts/durations. |
| Batch distribution min/max (shared `batch_statistics`) | Empty histograms already yield count 0 and null min/quantiles/max/mean. |

Uninstrumented Draft idle time remains null, rather than a guessed zero. Missing
mandatory artifact fields encountered by PingPong qualification are identified
by name instead of a bare Python KeyError.

## Requalify the retained smoke without GPU execution

Use a CPU environment with the fixed reporter installed. Set these to the original
retained directory and the exact original two-request workload file (do not rebuild
or edit the workload). Preserve the original failed `qualification.json`.

```bash
export SR_REQUAL_RUN=/absolute/path/to/retained/smoke/pingpong
export SR_REQUAL_WORKLOAD=/absolute/path/to/original/smoke-2.jsonl
python -m specrhythm.phase4.pingpong_comparison validate \
  --run-root "$SR_REQUAL_RUN" \
  --workload "$SR_REQUAL_WORKLOAD" \
  --request-count 2 --smoke \
  --output "$SR_REQUAL_RUN/qualification-smoke-requalified.json"
```

The output must be fresh. This command only reads retained artifacts and writes
the new qualification; it does not invoke a model, server, runtime or GPU. The
report's `execution_git_commit` remains the original runtime manifest commit.
Record the reporter checkout SHA separately; do not relabel old execution as a
run of the reporting fix. The existing helper's exact-commit prerequisite policy
is unchanged; this command repairs the report, not the stage orchestration.

No new GPU run is needed for this bug if the normal retained execution evidence
is intact. Reaching the old failing maximum implies the earlier raw validity,
request count/output accounting, process/Draft cleanup, assignment and sampled-row
consensus checks passed. It does not by itself certify downstream cohort/overlap
joins: only an offline requalification of the actual retained artifacts can do that.

The CPU regression fixture has two requests (A1/B1), verified terminal commits,
and no measured timing commits or formal performance artifact. Its end-to-end
qualification fails with the reported empty-max error before the fix and passes
after it. Additional tests cover mandatory missing evidence, optional empty
statistics/overlap, original artifact preservation, and unchanged formal gates.
