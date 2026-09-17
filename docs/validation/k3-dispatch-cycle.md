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
