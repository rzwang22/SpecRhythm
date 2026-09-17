# K3 lean dispatch: matched cycles and bounded evidence

Behavior reference: `4ff1170397db0393caa2d8a43c2089a8e602ddfd`, lean Target,
B128, unified/deferred-window/performance-exploration. Output equivalence NOT_RUN.
The original reports and their qualification are unchanged.

## Historical source and boundary audit

Source: `pingpong-k3-delivery-a100-k3-b128-lean-target-20260917T150307Z-19202.tar.gz`,
SHA256 `5409decada7b4d8437393c1be8ead6ed33c7993d263d14bfe63e3b3c13cdbe6b`.
The prior read-only inventory check verified 445 logical paths / 399 deduplicated
objects. Derived analysis is separate from the archive, never written into it.

First-repeat ordinary PingPong, **the same 52 measured steps**:

|Boundary|Mean ms|
|---|---:|
|claim → native TP start lower|237.563|
|TP envelope start lower → end upper|112.083|
|TP end upper → feedback RPC|63.820|
|claim → complete-step start|97.527|
|feedback RPC → complete-step end|78.825|
|complete-step start → end|394.764|

Therefore `413.465 - 97.527 + 78.825 = 394.764`, within rounding;
integer-nanosecond adjacent ledgers have zero residual. The earlier roughly
400/383 ms were equal-weight means across two windows, not one matched cycle.
A claim is made on the owner before its response serialization, coordinator
control publication/deep-copy and `engine.step()` start. A step ends after
feedback RPC, result transfer and output commit. The three intervals are not
identical. GPU endpoints retain calibration bounds, not host-launch substitutes.

The same first repeat has 3,199 complete ordinary request feedback-generated →
next-feedback-generated cycles (mean 1168.748 ms), and 3,071 eager cycles
(mean 1196.197 ms). Initial/refill/missing predecessor and window-crossing pairs
are counted separately, never interpolated. Global Target cadence, same-home
cadence, request cycles and window/opportunity normalization are separate.

Eager service receive → owner dequeue averages 49.917 ms in that repeat.
39.141 ms intersects recorded physical Draft **host call** intervals; the remaining
10.776 ms is unaccounted. This supports in-flight physical work as a major queue
occupant, not a claim that all of it is GPU compute, fence, GIL or lock time.

## Added evidence

`k3_cycle_evidence` joins actual request/version/proposal claims, both native TP
rank request sets, worker feedback references, owner dequeue/feedback and READY.
It reports all complete request ledgers, incomplete reasons, coverage, exact
integer closure residual, and predeclared nearest-rank median/P90 examples.
Large request-ledger arrays are not repeated in the compact audit/compare JSON;
all-sample statistics, declared examples and per-step endpoints are retained,
and the original owner/transport/native records permit full reconstruction.
Feedback generation is the existing CPU payload-ready hook, not the exact device
instant when a sampling decision first becomes mathematically determined.

Per-thread sweeps report inclusive category unions and exclusive innermost spans;
crossing spans are ambiguous, uncovered time is unaccounted. Different producers
are never added. New bounded spans cover control construction/comparison/copy,
atomic control publication, worker state/input preparation, owner background steps
and status publication. They copy no tensors, add no I/O/fences and preserve the
phased recorder's existing limits and strict loss checks.

READY is proposal/version-scoped. It is physically valid until claimed/retired;
Target availability is a coordinator observation, not a proof that home, capacity
and Serial gate already held at the earliest possible instant. The report retains
those bounds and the later authoritative selection observation. It does not call
the historic selection→claim ~0.5 ms an earliest-eligibility latency.

Remaining CPU cost includes both full resident snapshots/audits, binding, stock
scheduler callbacks, admission records, control response/encoding, execution IPC
and worker preparation. Existing scheduler parent/child spans and new subspans
remain nested. No unknown residual is renamed Python or converted into predicted
throughput. READY publication and Target selection policy are unchanged.

## Single independently switchable execution optimization

`target_dispatch=reference` retains streaming `json.dump` into the same-directory
temporary control file. `target_dispatch=encode-once` changes only the coordinator's
`SR_S2_CONTROL` publication: `json.dumps(..., allow_nan=False)` followed by one text
write, close and the same atomic replace. It preserves keys, separators, escaping,
ordering and bytes; control readers still parse each live snapshot. There is **no
cache** and no reduced publication rate. Every control transition and absolute
deadline remains published at its original position. Other state files, diagnostic
buffering, fsync policy, post_prepost, promotion READY and selection are unchanged.
An encoding/write/replace failure still propagates and cannot publish the new state.
An encoded string lives for one call; its size is recorded, and the contents remain
bounded by the existing 360 resident rows / active128 / configured context limits.
No control snapshot is retained by the observer.

Source evidence: `fixed_runtime.drive.publish_control` compares/builds the live
packet, calls `s2_pool.publish`, then deep-copies it; `PingPrePostController.select`
receives the owner's actual claims before this call. `PingPrePostScheduler` and
`PoolScheduler` read control on the Target preparation path; worker proposers read
the same atomic published file. No reader depends on fragments of the temporary
file. Full resident KV checks still run before and after scheduling.

A fixed-input CPU experiment used the first measured ordinary PingPong admission
subtree from the archive, not a generated ideal payload: 554,172 encoded bytes,
51,697 streaming text-write calls versus one. Twenty local samples gave median
8.912 ms (`dump`) versus 1.803 ms (`dumps` plus one write to StringIO). Byte equality
was checked. These are encoding microbenchmarks, not filesystem or GPU throughput
predictions. Existing whole control publication spans inside claim→GPU average
42.291 ms in the first ordinary window; this includes multiple effects and cannot
all be assigned to encoding. New spans separate encoding, write/close and replace;
construction, comparison, copy, scheduler and worker preparation remain visible.

`lean-reference` and `lean-dispatch-opt` use identical lean numerical coverage,
observation, geometry, dispatch scheduling, work and budgets at one final SHA.
Only the control serialization switch differs. No additional claim-selection
policy is justified: real owner/backend/scheduler event-controlled regression
already admits legal other-home work while unrelated recovery is blocked. The
remaining earliest-eligibility delay is not fully observed and is not declared
avoidable. Compare whole-window advancement, not an isolated function benchmark.

## Both historical lean windows, identical matched endpoint definitions

|Window/mode|steps|claim→GPU|TP envelope|GPU→feedback RPC|claim→step start|RPC→step end|complete-step|
|---|---:|---:|---:|---:|---:|---:|---:|
|0-forward/pingpong-k3|52|237.563|112.083|63.820|97.527|78.825|394.764|
|0-forward/pingpong-eager-k3|50|193.615|111.370|62.054|79.196|111.460|399.303|
|1-reverse/pingpong-k3|55|212.910|111.320|62.923|88.780|73.487|371.860|
|1-reverse/pingpong-eager-k3|47|207.533|110.810|63.101|87.663|114.953|408.734|

All values are ms, all columns within a row use the identical sample set.
Ordinary PingPong equal-weight window means reproduce approximately
400.309 − 93.153 + 76.156 = 383.312 ms. These are endpoint identities,
not an additive budget for independent threads. The two ordinary windows
have 3,199 / 3,391 complete request cycles; eager has 3,071 / 2,879.
Reverse eager receive→dequeue is 54.065 ms on the matched samples,
43.167 ms intersecting physical host calls and 10.899 ms unaccounted.

In the first ordinary window the 95.993 ms scheduler parent includes
16.603 ms binding, 9.326 ms resident decisions, 22.266 ms admission record
construction/encoding and **1.757 ms actual pinned stock scheduler**;
these disjoint resident children plus readiness/finish and wrapper residual
total about 50.136 ms. The remaining scheduler time includes the two full
live pool snapshots/audits and wrapper work; it is not all stock scheduling.
The same claim→GPU path also contains 97.527 ms before complete-step,
11.120 ms step entry before scheduler, and 32.923 ms scheduler-end→GPU.
These are mutually adjacent boundaries. Existing evidence does not split
every execution IPC/worker-preparation operation; new worker subspans expose
part of that residual without extra device synchronization.

Report budget check: losslessly factor the identical explanatory `scope`
string out of each request wait row into `dispatch.request_wait_scope`;
all numeric/missing values and identities remain. Together with compact cycle
statistics, the four replayed reports occupy 6.41–7.05 MB, below the unchanged
8 MiB cap. Missing scope prevents factoring; no guessed default is inserted.
The initial pretty-JSON diagnostic exceeded the cap; production uses compact
JSON, and both the correct encoding and structural headroom were checked.


## CPU and delivery validation (2026-09-18)

The production-chain tests drive real owner/backend/controller/scheduler paths
with controlled CPU interfaces, construct runtime/native reports, qualify, export,
and re-read the single archive. Both encoding profiles preserve actual claim and
Serial gate behavior. The additional report projection keeps every complete-step
thread partition and its same-sample aggregate; missing lanes remain missing.
CPU ordering establishes safe interleaving, not native GPU overlap.

Python3.12 related suite:94 passed. Python3.9 related suite:90 passed.
Final fixed-entry/profile tests:13 passed on3.12; fixed-entry/cycle tests:15 passed
on3.9. The final cycle-only regression has8 passed, including thread coverage and
closure. Ruff passes after correcting import order, lambda fixture binding and
one line-length violation; no functional check, timeout or assertion was relaxed.

First full Python3.12 run:3061 passed,10 failed,3 skipped. Nine failures were the
unchanged subprocess20s failure-summary/export harness boundaries (three K3
joint-export cases, one Ping prepost joint-export case and five legacy Serial
prepost runner cases). One unchanged forked logging test exceeded its existing2s
drain deadline. These are actual failed checks, not certified pre-existing causes.
No evidence establishes their original cause. Subsequent results are recorded
separately; a later pass does not erase or explain these failures.

Execution `c1fca49f3f846ef94f0759b585fddaf091d78b62` contains observations and both
encoding profiles. Fixed entry `9feb51a4d36e4acc0f682ea006a6842d327a69d4` reads that
exact execution SHA; the server does not execute moving branch HEAD. New GPU
performance, geometry, cleanup and overlap are PENDING. Output equivalence is
NOT_RUN, and the historical B64 output differences remain unresolved.


Final full local run: **3089 passed,3 failed,3 skipped** in1031.77s. The three
remaining failures are unchanged legacy harnesses:

- `test_execution_pair_runbook... [capacity-serial:runtime]`:20s child timeout.
- `test_phase4_dual::test_target_failure_terminates_draft_without_unbounded_wait`:5s child timeout.
- `test_phase4_natural_teardown... [target]`:15s child timeout.

Their tests and shell helper implementations have no diff against preserved
`bb79d4f`; none activates the new control encoding policy. Their original causes
remain unresolved; this local full suite is not reported green. Git's separate
working-tree refresh was sampled blocked in `read()`, and an initial static-source
read raised OS error89; those observations do not establish the test failures'
cause. Fresh strict syntax/Bash checks passed for406 Python files/20 scripts,
and compileall passed on3.9 and3.12. Machine-readable local check results are in
`k3-dispatch-local-checks.json`. No source test deadline was extended.
