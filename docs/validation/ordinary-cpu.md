# Ordinary P0/P1/S1 CPU comparison

Source execution `4878f8beb8a36713e35ef3e0a1a6cfe1470967b5`, archive
`pingpong-k3-delivery-a100-ordinary-dual-20260918T130732Z-396.tar.gz`, SHA256
`fd25d3f06c8ff155e2da8074c80645fce92805c071dbb83e0d948ed7b4a65897`.
The original archive and historical qualification remain unchanged. Development
continues from `ba9bafa4b682eb0a1649388088dd44aaacc6128e`, not from an older execution.

## Evidence and one selected optimization

The two measured dual-batch windows had claim→GPU means 94.655/93.377 ms,
TP2 envelopes 108.296/108.343 ms, GPU end→feedback RPC 45.689/45.729 ms.
Scheduler includes the whole wrapper. Coordinator exclusive means in those
matched claim→GPU samples include live pool snapshot 15.289/15.084 ms,
block audit 10.016/10.255 ms, resident binding 11.539/11.131 ms, admission
record work 8.316 ms (first window); the stock scheduler itself is about 1.18 ms.
These are nested in the scheduler envelope; do not add them to that envelope or
across worker/owner threads. Unassigned worker-lane time is not called Python/GIL.

`ResidentPoolAudit.check` previously made a `require` call, diagnostic kwargs and
an owner dictionary entry per block. P1's `block-sets` checks each complete group
for duplicates, strictly validates every block's integer/nonnegative type, checks
set disjointness with all prior requests in that group and updates that group.
Different KV groups remain independent. Both complete pre/post snapshots still
read live KV, prompt contents and frontier. Initial/frozen state, staged/queued
identity, active initial-block ownership, peak block and check counters retain
the same semantics. No hash/ownership PASS cache or inspection skipped. The Draft
uses the original audit implementation; only the Target's implementation changes.

A fixed CPU-only 360-request ×128-block synthetic check, 20 visits, gave median
8.40→2.29 ms on this Mac. This is supporting implementation evidence, not a GPU
throughput prediction or a test threshold. Structural tests check the production
scheduler still performs two full audits and faults are rejected equivalently.

## A verified recovery class

`ordinary-cpu-4878-recovery.json` is a derived read-only analysis from original
runtime/device/owner bindings. Of 261/264 unique physical calls containing recovery,
93/94 finish **wholly after another home's claim and before its earliest possible
GPU start**. Those are 35.6% of calls, not a fraction of all recovery GPU duration,
and not complete K3 jobs. Examples are selected by median/P90 finish→Target gap:
one A recovery extension (request c6fd…9828e, work version22, draft-physical-137)
starts 23.508 ms after B's claim, finishes at47.251 ms, and B's GPU starts at87.616 ms.
It finishes 40.365 ms before B GPU starts. No new profiling run was needed.

This class is already running during Target CPU preparation, not blocked waiting
for Target GPU completion. It motivates reducing CPU preparation; delaying Draft
would manufacture overlap. The 168/170 remaining calls are **unclassified** by
this conservative whole-call test: partial intersections, work outside claim→GPU,
late feedback and genuine dependencies are not interchangeable. Earliest legal
eligibility remains unobserved; no claim that all residual time is removable.

## Shared controls, separate mode semantics

`shared_control.py` extracts the already-measured lossless provenance table,
immutable command publisher, call-local control scope and sole engine caller.
`dual_batch.py` retains a compatibility facade and the original PingPong-only
`dual-batch` guard. Serial explicitly selects `shared-command`.

K3 Serial already uses `PingPrePostController`, `PingPrePostScheduler` and
`PingPrePostProposer` in `fixed_runtime.CLASSES`, while its K3Machine geometry and
admission enforce the Serial idle gate. Compact admission delegates to the same
`machine.admit`; it does not bypass the gate. Skipping an unused next-proposal list
is valid for both because their actual FeedbackClient routes to pp_feedback.

The control read scope expires after every synchronous Target call or exception;
a nested reader shares only that immutable command. Thread/context isolation and
lifecycle publication failure behavior are retained. Live KV and generated prefixes
are never cached. No new executor, priority, READY timing or eager change.

|case|mode|control|Target audit|geometry|
|---|---|---|---|---|
|P0|pingpong-k3|dual-batch (measured behavior)|reference|A64/B64 Target64|
|P1|pingpong-k3|shared-command (same compact operations)|block-sets|A64/B64 Target64|
|S1|serial-k3|shared-command|block-sets|A128 Target128, idle gate|

All use active128/Draft128, lean, unified, deferred-window, resident360, greedy
sampling, K3, no bonus, setup900/drain60 absolute budgets, 256 actual warmup
request verification opportunities plus initial population coverage, 30s windows.
Two Serial Target calls and four full PingPong calls give256 opportunities; actual
batch counts, not number of function calls, control the warmup boundary.

## Validation and limits

The foreground experiment capacities P0/P1/S1, then bounded 16-request/max8 output
same-mode smoke P0/P1 and Serial reference-control S0/S1. S0 is not a performance
point. Actual fixed greedy contract is checked before exact comparison; seed alone
is not an equivalence criterion. Mismatches retain first token/termination difference
and both source runs, stop before performance, and are not excused as historical.
No Target-only comparison or cross-geometry output equality requirement is added.

Performance order P0,P1,S1,S1,P1,P0. Report each window, then **sum tokens / sum
actual seconds**, not best run or an unweighted ratio. S1/P1 is a system comparison
including B128/B64 batching efficiency, not a pure overlap ablation. Full-request
TPOT preserves `fixed_results`' formula, applies only to natural completions in
the window and reports missing/censored cases. Completion throughput is not SLO
goodput. Native cross-home intersections remain unioned across bindings/ranks;
TP2 GPU envelope is not SM utilization.

Production CPU regressions cover owner/controller RPC, Serial gate, stale/duplicate
claims/feedback, recovery, scoped exception/thread invalidation, live two-audit
scheduler faults, report/serialization/qualification/export replay, and first-error
single-package local delivery. They are not GPU evidence. The configured A100 SSH
endpoint reset the connection before authentication; no remote job was launched.
New GPU capacity/semantic smoke/cleanup/performance/overlap remain PENDING.

### Local validation record

Full Python3.12 run: **3168 passed, 7 skipped, 1 failed**. The unchanged legacy
`test_shared_shell_accepts_natural_teardown_and_removes_guard[dual]` expected
`post_coordinator_descendants_observed=True`; inspecting its real fixture report
showed no remaining post-coordinator children, natural teardown/cleanup PASS,
exit0 and no signals. The original pytest output is retained in
`ordinary-cpu-local-test-first.log`. Pytest subsequently pruned that temporary run
directory; the inspected lifecycle JSON is not relabeled as a retained raw file.
The fixture child has an0.8s lifetime; a race with process-table observation is
plausible, not a proven implementation fix. The unchanged teardown group in
isolation passed16/skip1. No timeout or assertion changed; the later pass is not
claimed to explain or repair the first failure.

Final Python3.12 affected production/entry/deadline tests:129 passed.
Python3.9 related production/owner/geometry/deadline set:199 passed.
One earlier test invocation named a nonexistent deadline test file (collected
zero); corrected to `test_k3_shutdown_deadline.py` before validation.
Ruff, compileall, Python3.9 grammar and all Bash syntax/diff checks pass.
GPU/remote filesystem validation stays PENDING. CI is reported from actual runs.
