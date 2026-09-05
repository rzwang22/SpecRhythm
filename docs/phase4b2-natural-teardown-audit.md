# Phase 4B.2: post-coordinator natural teardown audit

## Code root cause at aa80ed

At `aa80ed22dfb13c7969e0944c4b56025ae3f42ca5`, `process_lifecycle.py` immediately
set `leaked_after_coordinator_exit = coordinator_reaped and bool(remaining)`.
Any remaining owned descendant triggered TERM immediately, even when coordinator
status was zero and no fatal evidence existed. The subsequent TERM/KILL waits
were already forced-cleanup waits; they were not natural shutdown grace.

The initial observation remained latched into `cleanup_valid` after all owned
processes had exited. Consequently a normal transient shutdown window could cause
supervisor-generated SIGTERM, `cleanup_valid=false`, `run_valid=false`, status 125
and retention of `process-lifecycle.active`. The shell then correctly reported the
retained guard. Removing the final validity check alone would conceal real leaks;
the fix must defer both the first signal and leak classification.

## Real evidence and audit boundary

The operator reports normal Target initialization/generation, 100 running requests,
no runtime traceback, EngineCore receiving SIGTERM, graceful exit of both workers,
then status 125 and a retained guard. Preserve the exact directory:

```text
/root/autodl-tmp/SpecRhythm-data/results/phase4/aa80ed22dfb13c7969e0944c4b56025ae3f42ca5/phase4b2-row-mapping-20260905T115441Z-1467/target
```

The supplied log excerpts are consistent with the confirmed code defect. They do
not themselves contain `target_exit_status`, `failure_detection`, or the recorded
signal actions. Direct read-only access to the raw files was attempted through the
two configured AutoDL SSH aliases; both returned connection refused. No raw
artifact contents or checksums are claimed as independently inspected, and no
server result files were changed. A working endpoint was requested separately.

Section A of the [fresh runbook](phase4b2-fresh-three-mode-runbook.md) inventories
the entire old root into a separate audit directory, then reads the original
`process-lifecycle.json` and `target.log`, records hashes and relevant line numbers,
and verifies the clean-coordinator/no-fatal/immediate-TERM branch. If those fields
differ, that audit stops for review rather than assuming this cause. Even matching
old artifacts cannot prove natural teardown would have completed within five
seconds: the old supervisor interrupted that interval. New CPU fixtures establish
the state-machine behavior; only a future fresh A800 run can establish its outcome.

## Repaired state machine

| Initial condition | Observation and action | Result |
| --- | --- | --- |
| Coordinator zero, no fatal, no descendants | Record natural completion; no signals | Eligible, status 0, guard removed if Draft cleanup is also valid |
| Coordinator zero, no fatal, transient descendants | Observe/reap during bounded natural grace, without signals | Natural completion remains eligible; no leak latch |
| Coordinator zero, descendants survive natural grace | Classify leak, then TERM, bounded wait, KILL fallback | Status 125 and retained guard even if forced cleanup succeeds |
| Fatal evidence or failed coordinator | Skip natural grace; retain immediate bounded cleanup and Draft cleanup | Nonzero result; existing invalid-cleanup guard policy retained |
| Fatal evidence during natural grace | Interrupt grace before reaping away the evidence | Immediate bounded cleanup; cannot become a clean run |

The natural grace defaults to **5 seconds**, separately configurable through
`--natural-teardown-grace-seconds` / `PHASE4B_NATURAL_TEARDOWN_GRACE_SECONDS`.
All configured waits must be finite and positive. TERM grace remains 5 seconds,
KILL wait 2 seconds, and polling 50 ms. Actual elapsed time also includes process
enumeration and filesystem I/O. A configured natural grace does not delay fatal
cleanup. The 250 ms timestamp-wrapper post-child drain is unchanged.

The same fatal log markers and Linux nonzero owned-child status checks now operate
during both execution and natural teardown. Teardown checks status before reaping
adopted children. Existing PID/start identity checks, session/descendant ownership,
Linux subreaper/launch-token adoption, and Draft socket ownership rules are unchanged.

## Artifact and guard semantics

New evidence explicitly separates these facts:

- `post_coordinator_descendants_observed` and `post_coordinator_owned_pids`: the
  initial observation after coordinator reap, not a failure verdict.
- `natural_teardown_grace_seconds`, start/end timestamps and
  `natural_teardown_completed`: true on natural success, false on timeout or fatal
  interruption, null when natural grace was ineligible and skipped.
- `leaked_after_coordinator_exit`: descendants survived the clean natural grace.
- `failed_coordinator_descendants`: failed/fatal coordinator teardown with owned
  descendants, preserving the earlier fail-closed cleanup policy.
- `owned_cleanup_completed`: coordinator reaped, no owned Target survivors, and
  valid Draft cleanup. This can be true while `cleanup_valid`/`run_valid` are false
  because a real leak required forced cleanup.

The legacy `child_reap_result.wrapper_exited_with_descendants_alive` remains the
unsafe-exit flag consumed by existing readers. It now excludes successful natural
teardown and includes a proven leak or failed-coordinator descendants. The original
initial observation is retained in the new explicit field instead.

Clean completion removes the guard only after the lifecycle artifact is written.
A proven leak retains it even after successful forced cleanup. Fatal failures keep
their previous nonzero result and cleanup-dependent guard behavior; successful
cleanup alone never changes `run_valid` to true. No old guard is removed by this fix.

## CPU coverage and unchanged scope

Real subprocess tests cover no descendants, transient descendants with and without
the actual timestamp wrapper, TERM-handled and TERM-resistant leaks, fatal logs and
nonzero Linux child exit during natural grace, and no surviving owned children.
The existing fatal-worker/blocked-sibling/Draft/socket tests now configure a long
natural grace to prove it is skipped. All three shell modes use the actual shared
helper with CPU fixtures and prove status 0, natural completion, removed guard and
caller survival. Existing shell failure tests retain nonzero/caller-survival coverage.

Only supervision, its shell option, regression tests, and documentation change.
Target/Serial/Dual decoding, all five patch files and their hashes, sampled-row TP
mapping, EOS canonicalization, retired-ready logic, measurement boundaries,
performance formulas, and matched-work gates remain unchanged. No GPU execution,
numerical-divergence investigation or performance optimization is part of this fix.

The next validation requires a new commit and unused root, with Target → Serial →
Target/Serial matched-work → Dual → Dual performance → three-mode comparison under
that common execution commit. The failed aa80ed Target is never retried or relabeled.
