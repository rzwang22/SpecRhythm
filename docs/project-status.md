# SpecRhythm project status

Last updated: 2026-09-12

Maintenance rule: every code-changing PR updates this file with its scope, status, evidence,
known limitations, and next gate before that PR is considered complete.

## Goal

Build a reproducible path for evaluating SpecRhythm's SLO-aware speculative-decoding policy,
first as a dependency-free semantic simulator and workload harness, then with measured GPU
calibration, and only afterward as a vLLM or SGLang integration. Simulator outputs are used to
reject inconsistent policies and design controlled experiments; they are not GPU performance
claims.

## Roadmap

1. **Workload and provenance:** deterministic Mooncake replay, R3 proxy construction,
   validation, manifests, and raw-data hygiene.
2. **Simulator semantics:** persistent proposal lifecycle, deterministic candidate trees,
   tree-aware AdaServe/SpecRhythm allocators, queueing-aware SLO accounting, path-dependent eager
   continuation, diagnostics, and constructive counterexamples.
3. **GPU calibration:** measure context/batch/candidate-dependent `D(B,K,C)` and `V(B,K,C)`,
   acceptance traces, confidence calibration, and candidate roof on a pinned model/engine/GPU.
4. **Engine prototype:** implement the validated control plane as a narrow plugin/prototype in
   the selected serving engine.
5. **Evaluation:** paired workload sweeps, ablations, class-level SLO attainment, goodput, waste,
   confidence intervals, and failure analysis.

## Pull request progress

| PR | Status | Scope | Evidence / boundary |
| --- | --- | --- | --- |
| [#1 workload-v0.1](https://github.com/rzwang22/SpecRhythm/pull/1) | merged | strict Mooncake replay, R3 proxy config, validator, manifest, fixture tests and docs | workload plumbing only; proxy payload and illustrative acceptance |
| [#2 simulator-semantics-v0.2](https://github.com/rzwang22/SpecRhythm/pull/2) | frozen draft; Phase 2 complete, not merged | proposal lifecycle, deterministic tree oracle, tree-aware allocators, base-preserving residual controls, Phase-2 nested search pools and common-snapshot oracle replay, path-aware eager and accounting | pure-Python proxy and oracle upper bounds only; no deployable oracle, measured search cost, GPU integration, or performance claim |
| [#3 gpu-integration-v0.1](https://github.com/rzwang22/SpecRhythm/pull/3) | draft; Phase 3B.1 and corrected-20 Phase 3C.2 complete; Phase 3C.3 corrected-100 awaiting server run | hardened multi-rank primitives, corrected R3-real traces, common-prefix replay, request-bootstrap statistics, 2x shell decomposition and diagnostic learned ranker | user-run 3×A800 correctness artifacts plus Mac CPU tests; no packed-tree/serving engine, Dual-Batch, SLO, calibrated latency or speedup claim |
| [#4 vllm-serving-v0.1](https://github.com/rzwang22/SpecRhythm/pull/4) | Draft/Open/unmerged; S0 CLOSED/PASS; S1-P G0–G3 PASS at `5a00049`; S2 G1 at `24b31a9` reported zero process exits/clean cleanup but rejected terminal-tail overlap; S2-only contract refinement awaiting retest | independent prefilled-KV resident pool, dynamic Poisson arrival/admission, Target/Serial/PingPong and engineering SLO/goodput | CPU/source contracts are not GPU qualification; finite-trace ideal PD-delivery boundary only |

## Resident360 PingPong completion/refill repair (GPU retest PENDING)

The retained e39afc1 scan confirms eight valid points: Target/Serial/PingPong at
B16 and B32, plus Target/Serial B64. PingPong B64 execution and cleanup passed,
but a pre-forward selection of1 instead of32 stopped after3780.603119 ms/462 tokens.
It remains INSUFFICIENT and excluded. The pool was not exhausted: one natural
completion, one B refill,295 queued, all360 settled and no owned process remained.

A CPU regression on unmodified e39afc1 reproduces the global-ready-count versus
single-cohort FIFO quota mismatch. The fixed scan-only hooks collect the finite
published FIFO independently of verification quota and wait before stock allocation
when the selected cohort is incomplete. Real release/refill continues between polls;
all waiting remains inside the original30s window. The final partial guard, old
S1/S2/fixed64 defaults, live UUID and actual token/KV/dependency checks remain.
The exact server ready distribution still needs the optional narrow raw-event export;
the constructed31+1 CPU distribution is not asserted as server fact.

New regressions join actual scheduler, ready claims, asynchronous owner, private KV,
natural terminal/refill, window expiry and360-request shutdown rather than testing
the scheduler/owner only in isolation. Compact bounded wait evidence is retained
in snapshots/light reports. The explicit single-point order override enables a new
root's PingPong B64 without borrowing historical B16 PASS files. Foreground strict
mode/ERR trap lives inside a child Bash, preserving the interactive terminal.

Final local Python3.11 full pytest on byte-identical source/tests: **1836 passed,
3 skipped** (234.34s), including Phase4/S1/S2 and pinned source contracts. Ruff,
compileall, Python3.9 grammar for247 files, Bash/runbook checks and diff checks
pass. Exact-commit Linux CI status is recorded in the delivery. No AutoDL,
GPU or full CPU offline audit was run. PR #4 stays Draft/Open, unmerged; PR #2/#3
are untouched. Next gate: new root, capacity + PingPong B64 single point, then only
on PASS B128 Target → Serial → PingPong. No automatic old-point reruns. All scan
PingPong B values use the repair; old B16/B32 need remeasurement only if a uniform
new-commit performance table is required. Preserve all old roots and labels.
See [design/schema](decode-scan-design.md) and [foreground runbook](decode-scan-runbook.md).

## Resident360 fixed-batch/time decode scan (GPU PENDING)

Independent `specrhythm-decode-scan` / `run_decode_scan.sh` implements the requested
Target/Serial/PingPong × B16/32/64/128 scan on top of the `e78e4c7` alias repair.
One deterministic original360 pool (3:3:2:2), fresh engines, real all-request
prefill before measurement, two full warmup rotations and a30s time-primary
window replace the old sample12 stopping rule only in this new entry.

Per-point physical capacity covers resident360 and its actual B/mode/config,
including B128 query positions and private KV/workspace. Full-forward guards stop
before a partial model dispatch; pool shortage is INSUFFICIENT, with no tail
filtering, altered EOS or automatic configuration/retry. Real commit/prefix,
refill, live UUID and buffered-live/bound-prefix checks remain. Final settlement
uses the existing bounded protocol, first-error/exit retention and owned cleanup.
Light JSON/CSV and a bounded small bundle exclude failed/stopped/insufficient
points; no full CPU audit runs in the default chain.

CPU regressions cover all12 scheduler shapes, all360 prefill for each mode,
real serial/async Draft state-machine shutdown and window/accounting/CLI boundaries.
The identical source/test bytes in a temporary local checkout pass full pytest:
1823 passed / 3 skipped (233.69s), including Phase4/S1/S2 and pinned source checks.
The Desktop checkout encountered an unchanged 5s shell-cleanup test timeout and
observed blocked file reads; the temporary checkout passed that same test without
changing its timeout. Ruff, compileall, all244 Python files with Python3.9 grammar,
Bash/runbook syntax and diff checks pass. Exact SHA and Linux CI outcomes are in
the delivery. GPU capacity,
full-window availability and performance are **PENDING**; no server was contacted.
Old fixed64/S1/S2 defaults and prior artifacts remain unchanged. PR #4 stays
Draft/Open/unmerged; PR #2/#3 are untouched. Next gate: user foreground B16 three
modes, then only after PASS the remaining nine points, using the
[scan runbook](decode-scan-runbook.md) and [definition/schema](decode-scan-design.md).

## Fixed64/32 bound-prefix alias repair (GPU revalidation PENDING)

The supplied failure archive confirms Serial at
`5b7081e0eb692822db18afedf3ce9bd36fcd9bee` failed with an unmapped verify-start
request (effective rc=125, measurement INVALID, owned cleanup PASS). That result
remains FAILED/INVALID. The optimized identity constructor copied binding maps,
leaving the real Serial constructor's diagnostic alias stale after later binds.
Dual lifecycle/reports also retained stale forward/reverse aliases.

The only runtime repair preserves both original owner-local binding container
references while replacing the matching strategy. Existing bindings/history,
future binds, hook readers, reverse aliases and reports now share the same state;
independent schedulers/proposers/TP workers remain isolated. Idempotent install
keeps the same optimized object, counters and lock. No checks or algorithms removed.

New core regression, run before the runtime edit on the failing commit: linear
2 PASS, bound-prefix 2 FAIL with the exact server verify-start error. After repair,
both empty/prebound cases pass through real S2Serial construction, fixed install,
later binds, start/end hooks and three rounds of actual acceptance/commit accounting.
The extended startup/identity suite passes 36 tests, including Target and both
Dual-based modes, report/lifecycle aliases, owner isolation and negative cases.
Earlier tests missed the replaced-object-to-legacy-alias consumer boundary.

Local full pytest: 1772 passed / 3 skipped (178.98 s), including Phase4/S1/S2;
Ruff, compileall, Python 3.9 grammar for all 236 source/test files, related Bash
scripts, extracted runbook Bash syntax and diff checks pass. Actual Linux Python
3.9/3.12, Phase4 Python 3.11 and pinned-source CI status plus the final SHA are
recorded in the delivery. No AutoDL,
GPU or full CPU audit was run. PR #4 stays Draft/Open/unmerged; PR #2/#3 untouched.
Next gate: [new-root foreground runbook](fixed-identity-runtime-runbook.md),
buffered-live + bound-prefix, prepare/capacity → Serial → check PASS/effect →
PingPong → summary/bundle; stop on failure. Old roots stay unchanged. GPU performance
PENDING; no performance benefit is claimed for this repair.

## Fixed64/32 bound-prefix matching experiment (historical implementation)

Serving HEAD was clean at c1dd96d8c86e321d66d52aea31eb7396bf06786b. The new supplied
buffered-live archive's independent JSON and nine export hashes were reviewed;
CPU reanalysis retained PingPong POSITIVE (752.749–758.354 ms, 20/24 positive steps)
and Serial ZERO with clean joins/clocks. Actual complete64 rotations are
1150.924 / 1354.666 ms; no new GPU execution or performance improvement is claimed.

One opt-in `--identity-matching bound-prefix` mechanism reuses validated frozen
prompt bindings after an immutable prefix-free proof, while comparing the current
full prompt on every bind and retaining original alias/change/errors. All fixed
scheduler modes and Target-worker proposers use the same definition; default linear
and S1/S2 remain unchanged. KV audits, proposal/prefix/version checks, required
barriers, live UUID, buffered logs, primary-error and bounded drain are retained.
Runtime/light/export metadata include actual owner counters and measured scheduler
deltas; the old 52.70 ms arithmetic residual is explicitly not a critical path.

CPU regressions exercise real fixed/S2/resident/Dual scheduling, physical-pool
checks and TP startup, and demonstrate fewer actual full identity scans with equal
selection/root/candidate accounting. GPU throughput impact remains unknown. See
[design/evidence](fixed-identity-runtime-design.md) and the
[two-mode runbook](fixed-identity-runtime-runbook.md). Next gate: new root,
prepare/capacity, Serial PASS before PingPong, then lightweight summary/bundle.
No AutoDL, GPU, full audit, stages, other short modes, G2/G3 or parameter grid by the
agent. PR #4 Draft/Open/unmerged; PR #2/#3 untouched. Stop after delivery.

Local validation: full pytest and focused fixed/S1/S2/Phase4 contracts PASS; Ruff,
compileall, Python 3.9 grammar, Bash/runbook syntax and git diff checks PASS.
Linux CI is checked against the pushed final SHA and reported in the delivery.

## Earlier fixed64/32 buffered-live runtime experiment

Based on e7452fc (clean serving HEAD), reviewed the new original-event attribution JSON,
not only its Markdown: PingPong ZERO has complete coverage, join errors=0, 12 rotations,
24/24 opposite-cohort owner operations finished before Target model launch. Draft commit
forward was missing from earlier proposal-only arithmetic: recorded per64 sums are
239.3625 ms Serial / 411.3140 ms PingPong; residual gap is 280.2046 ms, not CPU time.

Fixed-only `prepare --observation buffered-live` batches a reviewed post-run JSONL
whitelist within 256 records/1 MiB, preserves checksums/order/immediate contract validation,
and requires four conserving final-flush receipts within the shared drain deadline.
Default original-live and live UUID checks remain. Protocol sockets/control/ready and
release/error evidence stay immediate. No scheduler, model, precision, K, 64/32, KV or
proposal/accounting changes. No metadata cache, barrier removal or artificial overlap.
Summary now separates Draft proposal/commit, Target, other recorded GPU work, window
clipping, per-rank host costs, and final flush/drain. Proposal-only max formula is no
longer presented as an end-to-end prediction.

Independent integer-anchor projection fixes a CPU-reproduced 3 ns double-rounding
width error without adding tolerance or GPU synchronization. The real Serial-split
failure row is absent locally, so its old UNKNOWN remains; bounded bad-row details are
now emitted by CPU analysis into a new output. Full audit remains separate.

Local CPU validation: 1736 passed, 3 GPU skips; Ruff, compileall, Python 3.9 grammar,
Bash/runbook syntax and diff checks passed. Linux CI final status is in the delivery;
these checks do not qualify GPU performance.
Next: operator prepare/capacity, buffered Serial once, then buffered PingPong once only
after all Serial checks pass; stop and return evidence. No AutoDL connection/GPU run by
agent, no merge, no changes to PR2/3. See [design](fixed-buffered-runtime-design.md),
[schema](fixed-concurrency-diagnostic-schema.md), [runbook](fixed-buffered-runtime-runbook.md).

## Historical fixed64/32 b424 timing attribution (proposal-only first analysis)

The first-pass arithmetic below is retained as history; the new purpose-complete
recorded-forward correction above supersedes its 320.402 ms residual.

The operator's four continuous b424 points report execution/measurement/cleanup PASS:
Target 172.290, Serial 167.218, Serial-split 91.722 and PingPong 119.020 tok/s.
Serial/PingPong mean full64 rotations are 1144.140/1596.297 ms. The quoted forward
means explain 131.754 ms of the 452.156 ms arithmetic gap; 320.402 ms remains unattributed,
not CPU-only overhead. Split's 5606.630 ms explicit pre-step owner wait accounts for most
of its 5701.237 ms aggregate window difference from PingPong, without proving GPU overlap.

The small bundle lacks runtime/backend/raw events. Reported ZERO overlap is independently
UNKNOWN until event/clock coverage is checked. No runtime optimization was applied.
The new CPU-only, externally limited 120-second analyzer/export preserves observations,
unavailable causal timing, partial reports and old results. It uses actual IDs/rounds/
prefix commits, TP-safe interval unions and per-purpose Draft counters. Original-live
logging/UUID, required synchronization, S1/S2 defaults, 64/32/K4 and equality NOT_REQUIRED
remain unchanged. See [evidence](fixed64-b424-timing-attribution.md) and
[CPU-first runbook](fixed-attribution-runbook.md). Next action: analyze/export already
retained Serial/PingPong evidence; do not rent GPU merely for this tool-only revision.
GPU performance validation of any future optimization remains **PENDING**. No AutoDL
connection or GPU execution was performed by the agent.

## Fixed64/32 diagnostic stop settlement repair

The operator's real run at `89a962127f2e9a10a2564736e2124dae0abe51d8` failed in
Serial shutdown with 64 unresolved next-round proposals. Target abort had not settled
Draft state, and final runtime/backend artifacts were absent. The retained failure
bundle was inspected read-only; it cannot recover a complete Serial timing report.
Old results remain unchanged.

This repair adds a fixed-only owner settlement protocol: synchronize the final actual
Target commit without proposing, explicitly discard unused proposals, wait for issued
work/terminal drains, validate private KV and release it before logical cancellation.
Unused resident KV is released without inventing logical initialization. Idempotency
receipts prevent duplicate release. Normal Serial shutdown still rejects unresolved
proposals; S1/S2 algorithms, models/workload/order/K4, live UUID, 64/32 limits and token
length/EOS/round equality `NOT_REQUIRED` remain unchanged.

A single absolute drain deadline is enforced by both remaining RPC budgets and the
existing owned process supervisor, including blocked coordinator/worker teardown.
Atomic compact snapshots retain completed measurement through drain failure; primary
errors survive secondary cleanup/report failures. Light summary/bundle accept missing
final reports, retain partial evidence separately and exclude it from comparisons.
Full CPU audit remains an independent optional command.

Reproduction: before implementation, the CPU fixed `drive` -> real Serial server ->
real `DraftStateMachine.shutdown` path rejected the same 64 pending proposals. After
implementation the same case physically releases the 100-request resident pool,
generates no additional proposal and passes the inherited shutdown guard. Additional
CPU cases cover unsynchronized final commits, initial proposals at time expiry,
operator stop, grouped ready/inflight/tail work, physical-only residents, release/RPC
failure, snapshot retention and real subprocess deadline termination. Validation: 268 related S1/S2/fixed CPU regressions pass, including 88 fixed tests.
Full local pytest with pinned vLLM source audit: 1693 passed, 3 skipped. Ruff,
compileall, Python 3.9 grammar (226 files), runbook Bash syntax (9 blocks) and
`git diff --check` pass. Exact-commit Linux CI is reported with the delivered SHA. See the [design](fixed-concurrency-diagnostic-design.md),
[schema](fixed-concurrency-diagnostic-schema.md) and [foreground runbook](fixed-concurrency-diagnostic-runbook.md).

**GPU revalidation PENDING.** No AutoDL connection or GPU execution by the agent.
PR #4 remains Draft/Open/unmerged. Next operator gate: new root, prepare/capacity,
Serial short first, then Target/Serial-split/PingPong individually, light summary and
small bundle. No S2 GPU PASS or four-mode performance improvement is claimed.

## Independent fixed64/32 timing diagnostic

Based on `50025b734ed02086533f2302ed2b2951c262ec45`, PR #4 adds a separate foreground
`fixed_cli`/`specrhythm-fixed` entry and committed server script/runbook. It reuses the
existing mixed100, real resident prefill/KV and production Draft/Target paths. Main
modes are target64, serial64, serial-split32+32 and pingpong32+32; models/TP/K4 stay
fixed. Explicit per-cohort capacity cannot overfill the other group while one is busy;
held terminal slots remain charged through real release. S1/S2 defaults are unchanged.

Initial-state 32/64 shape samples use fresh processes/prefill and are distinct from
short continuous windows. Serial-split waits for actual owner completion before each
Target step. New in-memory host/CUDA evidence records control, UUID, pool audits,
serialization, sync and logs; per-round fences are not added. Actual TP rank times and
clock-bounded event-union overlap are not summed into fake kernel overlap/time savings.

Light execution/accounting/measurement summaries return without the full S2 qualifier.
The full CPU artifact audit is an explicit separate command and initially PENDING.
Window cancellations are not natural completions or full-request SLO attainment.
Cross-mode token/length/EOS/round equality remains NOT_REQUIRED. Split-cost formulas
require actual union/context/K shape evidence; missing samples remain null with reasons.

Validation: 60 focused CPU owner/coordinator/TP startup, shape/accounting, interval and
foreground-error regressions pass; S1/S2/fixed compatibility tests total 240 passes.
Full local Python 3.11 pytest with pinned source audit: 1665 passed, 3 skipped (GPU
opt-in and two Linux-only process cases). Ruff, compileall, Python 3.9 syntax, nine
Bash scripts and eight runbook blocks pass; exact-commit Linux CI is recorded with
the delivered commit. **No AutoDL connection or GPU execution by
the agent; GPU timing/performance PENDING.** Existing S2 failures/results are retained,
and this diagnostic does not confer S2 G1/G2/G3 PASS. Next action is the operator's
short run using [the committed runbook](fixed-concurrency-diagnostic-runbook.md), then
return the small bundle. [Design and measurement contracts](fixed-concurrency-diagnostic-design.md).

## Phase S: Serving Workloads & Arrival Replay

**S2 terminal-drain qualification correction** is based on
`24b31a9e0125d773697d92ec0ea333384616e539`. The operator reports zero Target/coordinator,
Draft and effective exit codes, valid completed cleanup and no owned PID left. G1 rejected a
13.152235 ms host overlap between cohort-A `finish_tail` and another A request's proposal
verification, with an empty request intersection. The reported conflict and exact intervals
are retained in the S2 design; full server artifacts were not accessed in this coding task.

The old S2 qualifier applied the active Draft opposite-cohort rule to every operation.
The source audit establishes that `finish_tail` validates a one-token terminal prefix,
materializes/fences it, frees only that request's private KV and generates no next proposal.
The S2-only refinement requires native terminal/owner evidence, correct prefix/version,
independent actual GPU bindings, whole-pool private-block validation and unchanged unrelated
prefix/block snapshots before allowing disjoint same-cohort verification overlap. Operation
name or different IDs alone never grant an exemption. Active drafting and all same-request
collisions remain blocked. Receipt/failure details identify the actual artifacts.

Terminal work/host overlap and token/KV/slot completion are reported separately. Proposal
pipeline overlap, real terminal forward/GPU cost, arrival-to-drain timing, held slots, owned
cleanup, primary failures and real exits remain intact. `release_finished` only extracts the
existing coordinator release decision for an actual blocked-owner regression. Scheduler,
arrival/capacity/SLO/model/K/budget/numerical rules, UUID live mode, five patches and S1 defaults
are unchanged; cross-run equality remains NOT_REQUIRED. Historical results are not rewritten.

New CPU regressions use real S2 dispatch and production materialization/release with simulated
hardware, then the full qualifier. The base qualifier rejected the same fixture; the refined
qualifier accepts its cohort-A terminal drain and still rejects true conflicts, invalid
terminal/proposal/prefix/KV evidence and early release/drain. A real asynchronous owner blocked
inside release keeps the actual coordinator slot held. Related S2/S1/Dual/PingPong regressions:
**219 passed** (including **23 new tests**). Full local Python 3.11 pytest with the pinned
source audit: **1605 passed / 3 skipped** (GPU opt-in and two Linux-only process cases).
Ruff, compileall, Bash and diff checks pass; exact-head CI is recorded in the handoff and
PR #4. The next operator run uses the SHA-filled runbook and a
new root, remeasures capacity, repeats calibration/G0 and complete three-mode G1, then permits
G2/G3 only after the preceding gates pass. Old failed directories/seals remain preserved.
No AutoDL connection or GPU execution was performed. S2 GPU/performance remains unqualified.

**S2 PingPong UUID startup correction** follows the operator's run at
`c12b3768eaaaeca3ecde03999440b7b91763b128`, retained at
`/root/autodl-tmp/SpecRhythm-data/results/phase-s2/s2-c12b3768eaaa-20260910T012202Z-1476`.
Reported results: capacity PASS (small100/large390, 3:3:2:2), calibration/G0 PASS, G1
Target and Serial each completed ten and passed; PingPong failed at first verification.
There is no passed G1.json, so G2 remains blocked. These partial operator results do not
establish complete S2 GPU/performance qualification, and the old root is preserved.

S2 had called only the ordinary worker snapshot. The inherited Dual verification end hook
requires the `DualVerificationUuidQuery(worker)` installed by `worker_dual_runtime_snapshot`,
which the legacy Dual runner invoked but S2 omitted. S2 now uses an explicit PingPong startup
RPC on both Target ranks before prefill/verification. Later capacity/memory snapshots only
read the query evidence, preserving its identity and counters; Target/Serial remain on the
ordinary startup path. Live UUID behavior, TP binding, algorithms, capacity selection, trace,
SLO, models/K/numerical settings, measurement boundary, failure reporting and cleanup are
unchanged. Probe counters may be zero without satisfying a nonempty UUID A/B experiment.

The new CPU regression executes real S2 configuration, proposer constructors, startup RPC
callbacks, inherited verification start/end on both simulated ranks and final evidence reads.
It reproduced `AttributeError: 'S2PingProposer' object has no attribute 'uuid_queries'` at the
original `vllm_dual.py:659` before the fix. Earlier tests stopped at LLM construction or used a
canned RPC/step result, missing that integration boundary. The shared legacy UUID hardware
fixture now explicitly clears unrelated serving profiles so focused tests are order-independent.
Historical `24b31a9` local validation: S2 **53 passed**, S1 **104 passed**, legacy Dual UUID **37 passed**
(combined **194 passed**); full Python 3.11 pytest with pinned source audit **1582 passed,
3 skipped** (GPU opt-in and two Linux-only process cases). Ruff, compileall, eight repository
shell files, eight runbook Bash blocks and diff checks pass. Exact-head Linux CI is recorded
in the handoff and PR #4. The next operator run uses a new root and newly measured capacity
(390 is not hard-coded), repeats calibration/G0/G1, and
permits G2/G3 only after G1 succeeds. The agent has not connected to AutoDL or run GPU work.

**S2 — GPU-resident Prefilled-KV Dynamic Decode Serving** continues from
`5a00049e2eabf09f535fdd5f187f77406f6dcfe2`. The operator reported S1-P G0–G3 PASS
in `/root/autodl-tmp/SpecRhythm-data/results/phase-s1/s1p-5a00049-20260909T144802Z-1469`.
That user-supplied baseline status supersedes the earlier S1 GPU-pending entries below;
the coding agent has not independently rerun it. S0/S1 evidence remains read-only.

The independent `specrhythm-s2` entry freezes nested 3:3:2:2 small100/large500 inputs,
common capacity-driven decrements of ten from actual Draft/TP Target rank probes,
independent Poisson/order seeds and a pre-main engineering SLO policy. Every new attempt
recreates private Target/Draft KV before one uninterrupted arrival-to-drain observation.
Synchronous EngineCore steps permit FIFO admission only at safe boundaries; a separate
arrival thread queues arrivals while GPU work is busy. The common active limit is 128,
distinct from 512 resident slots and the 4096 query-token cap. PingPong cohorts are filled
dynamically without waiting for a future empty cohort; Serial can use all active slots.

Native worker commits, KV block isolation, Target structure, sampled-row TP consensus,
within-run budgets/EOS/accounting, real exit and owned cleanup remain mandatory. Cross-run
token/length/EOS/round equality remains NOT_REQUIRED. Sparse singleton execution, zero
overlap, lower speed and zero SLO attainment are observations, not failures. Queue-inclusive
decode average latency is distinct from TPOT, and output work/throughput/makespan are
reported together. Foreground tagged raw logs, precise primary failures, sealed evidence,
fresh-state recovery and a small upload JSON are included.

Initial `c12b376` local Python 3.11 validation with the pinned vLLM source audit: S2 contracts **48 passed**,
S1 compatibility **104 passed**, Phase4 **1182 passed / 2 platform skips**, and full pytest
**1577 passed / 3 skips** (one GPU opt-in and two Linux-only process cases). Ruff, compileall,
eight repository shell files, eight S2 runbook Bash blocks and staged diff checks pass.
Linux CI for the delivered commit is recorded in the final handoff and PR #4. The new
contracts cover real adapter startup order, synthetic private
allocator residency/restoration, independent arrivals, dynamic slots/cohorts, common
capacity and traces, native-shaped accounting, offline requalification and real owned CPU
child failure/cleanup. Fixed-source tests inspect vLLM's synchronous client, retained cached
requests and scheduler allocation hooks. The partial operator results reported above supersede
the initial GPU-pending status. PingPong with the startup correction, complete G1, and G2/G3
measurements/performance remain **PENDING operator validation**.
The coding agent has performed no GPU execution and no AutoDL connection.

See [S2 design](phase-s2-design.md), [S2 schema](phase-s2-schema.md) and
[S2 server runbook template](phase-s2-runbook.md). The handoff includes a separate rendered
runbook with the final full SHA; each commit uses a fresh S2 result root. PR #4 remains
Draft/Open/unmerged; PR #2/#3 are unchanged. Earlier Phase S/4 entries below are historical.

**S0 — Mixed Real-Text Workload Foundation** adds independent versioned serving
requests, source acquisition/locking, deterministic selection, real Qwen3
tokenization, timestamp-only Mooncake composition, full read-only validation and
sealed review artifacts. Work started from verified branch head
`0684a29800519c02a7b3c2558952ad344b17a9bd` on `codex/vllm-serving-v0.1` with a clean
tree; PR #4 remains Draft/Open/unmerged. The Phase 4 entries below record the
earlier resident work; S0 changes none of its execution or historical evidence.

The real main1000 quotas are chat/code/summary/reasoning 300/300/200/200;
calibration200 is 50 per class, with joint deduplication before splitting.
Selection seed is 1664, independent class-slot seed 1665, thinking is disabled,
natural EOS is enabled and full template token counts obey the 4096 context
constraint including a four-token reserve. The source lock pins all original
train files, Mooncake trace and remote tokenizer by commit and file SHA256.
No source answers, synthetic replacements, Mooncake anonymous lengths or prefix
identities become request content.

Local Python 3.11 real-data construction and independent full validation are
**PASS**, with 1000/200 requests and source/artifact hashes unchanged across
validation. Separate-directory rebuilding also produces identical JSONL and
core manifest. Semantic workload SHA256 is
`05b5f2efad0a4ac1771c9a18d847c08d95d360432e1c9cfa55a82ba2f63cba02`.
Server-local Qwen3-0.6B/Qwen3-32B tokenizer alignment is now **PASS**, based on
the returned checksum-bound 1200-prompt server report. Synthetic fixture tests are
separate and default CI downloads no dataset or model. Linux Python 3.9/3.12
retain the full suite; Python 3.11 additionally exercises the S0 CLI/contracts.

S0 local checks: 62 fixture tests pass on Python 3.9.6 and 3.11.15; the final
Python 3.9 full suite passes 1425 tests with 3 skips, and the Phase 4 suite passes
1182 with 2 skips, with the pinned vLLM source audit enabled. Ruff, compileall,
staged diff checks and all eight runbook Bash blocks/embedded Python pass.
Linux CI results for the delivered commit are linked from PR #4.

See [S0 design](phase-s0-workload-design.md) for source mappings, algorithms,
schema/hash contracts and the real-data summary, and [S0 CPU runbook](phase-s0-workload-runbook.md)
for exact server paths, fetch/import, full validation, dual-tokenizer checking,
rebuild and review bundle commands. Real data and generated workloads stay outside
Git. The coding agent performed no GPU execution, model inference or AutoDL
connection. [The appended closure review](phase-s0-closure-review.md) records the
returned archive SHA256, all 35 matching inventory entries, split disjointness,
server reports and the assistant's review of all twenty full prompts. S0 is
**CLOSED/PASS**. No additional human signature is asserted. The old sealed JSON
`manual_sample_review=PENDING`, original checksums and tar remain unchanged.

| Stage | Scope | Status |
| --- | --- | --- |
| S0 | Four-class real-text construction and validation | CLOSED/PASS; server machine/tokenizer/rebuild evidence and assistant content review accepted |
| S1 / S1-P | Resident three-mode performance qualification | Previous GPU gate stopped by old exact-output policy; S1-P cancels cross-run equality, CPU/CI recorded below; fresh GPU G0–G3 PENDING |
| S2 | Target dynamic arrival, queueing and streaming timestamps | Future; not started |
| S3 | Serial and dynamic PingPong join/leave/setup | Future; not started |
| S4 | SLO and load calibration on the independent calibration set | Future; not started |
| S5 | Formal 1000-request serving comparison | Future; not started |

Phase 4C retains its Dual-Eager name. The existing Phase 4 primary evaluation is
resident decode-only; no runner consumes the new arrival field yet. Dynamic
mixed prefill/decode, calibrated SLO, PD/KV handoff and a new GPU performance claim
are outside S0/S1. `slo_policy_ref=null` / `calibration_status=pending` is not an
accepted calibrated experiment.

### S1-P implementation handoff

The Serial startup follow-up starts at `645635d5a54d886ac874a9e1046ae8ad957bef9b`.
The operator reports G0/Target complete and Serial failing on missing
`decode-ready-context.json`. Source audit confirms S1 bypassed the legacy Serial
CLI's context creation. The S1 adapter now creates it exclusively before calling
the real Serial runner/LLM, using the existing builder, real provenance parser and
frozen config/patch/workload/Git/execution bindings. The regression failed at the
simulated LLM construction entrance before the fix and passes afterward without
precreating context. Startup exceptions remain primary; missing later reports are
secondary diagnostics. Foreground logs, actual return codes, owned cleanup and
NOT_REQUIRED token-equality policy are retained. Runtime/scheduler/backend/Target
and five-patch implementation files are unchanged. Current focused CPU suite: 104
passed; full local Python 3.11 suite with the pinned source audit: 1529 passed,
3 skipped. Ruff, compileall, helper syntax and 12 runbook Bash blocks pass.
Linux CI evidence for the delivered commit is recorded in PR #4.
No GPU run or AutoDL connection was performed. Old roots/results stay unchanged;
next operator step is a new root and foreground G0/G1 at the delivered SHA.

The backend/frontend follow-up starts at `08cf93a531dc928e02820b14412398b952592a49`.
S1 qualification now checks the report producer's `vllm-batched-paged-kv-draft`
identity, independently from shutdown complete, zero live requests and execution
failure false. Each check retains field/expected/actual/artifact evidence; failures
print those details and bounded child-log tails directly. Regression artifacts use
the actual backend report producer with a CPU worker. Cross-run equality remains
**NOT_REQUIRED** under the same `s1-performance-v1` policy.

The default runbook uses foreground `gate`. A read-only log mirror labels each
mode and Target/Draft source while both child streams still write to their original
files. Real CPU subprocess tests prove output is visible before exit, preserve
Target exit 7 and Draft startup exit 9, and exercise the existing owned cleanup.
No backend/scheduler/Target/measurement/patch implementation changes are included.
Current local Python 3.11 checks: 100 focused tests, 1525 full-suite passes with 3
skips; Ruff, compileall and runbook syntax checks pass. Linux CI for the delivered
commit is recorded in PR #4. No GPU execution or AutoDL connection was performed.
All old failure artifacts stay unchanged; the next server step is fresh-root
foreground G0/G1 using [the runbook](phase-s1-runbook.md).

S0 is **CLOSED / PASS** with its original sealed inputs and the assistant review
scope preserved. S1-P starts at clean `cd36d18c63ac706548fabccb4e5cf6e0f15e5897` on
`codex/vllm-serving-v0.1`; PR #4 remains Draft/Open/unmerged. PR #2/#3 are untouched.
The operator reports that the previous raw Target attempt-002 completed two real
runs but the old gate rejected `repeated_run_deterministic=false`. That old policy
result stays unchanged at `/root/autodl-tmp/SpecRhythm-data/results/phase-s1/cd36d18-20260909T110103Z-1489`.
It is background evidence, not an automatically completed S1-P gate. S1-D is cancelled.

S1-P uses versioned `s1-performance-v1`. **Cross-run/cross-mode output equality is
not required or performed.** Independent tokens, bootstrap, final prefix, output
length, EOS/finish reason, proposal/acceptance and rounds may differ. All requests
still share frozen input IDs/order, prompt tokens, 512/1024 budgets, model/tokenizer,
sampling, K4 and existing numerical/backend/resource configuration. No new async
sweep, algorithm change, patch change, resampling or ignored EOS is introduced.

Each run still validates completion exactly once; its own bootstrap, final tokens
and committed-event reconstruction; EOS/budget; proposal prefix/round, correction,
bonus, Target/Draft KV/sync and TP mapping; real measured boundaries; nonzero exit
and owned cleanup. Old Phase4 validator defaults remain strict. G0, manifest, result,
gate, seal, resume and offline comparison bind the same policy/schema. Old roots
cannot be silently migrated or reused. G1 now runs Target, Serial, PingPong and one
independent PingPong repeat; G2/G3 sizes and fixed rotations stay unchanged.

Each repeat reports actual full/timed/bootstrap tokens, completed requests,
EOS/cap/setup-terminal counts, task length distributions, makespan, actual tok/s,
completion latency, Target B/Q, Draft/acceptance work and physical overlap evidence.
Mode-level median throughput ratios use each run's actual tokens. Makespan ratios
carry actual output counts and are not equal-work speedups. Equality fields are
null/NOT_REQUIRED; count equality is descriptive only. The label is
`resident decode-only three-mode performance observation`, with no unconditional
end-to-end improvement, pure batching, equal-total-GPU or serving TTFT/SLO claim.
Zero overlap or lower throughput does not fail qualification; invalid/partial runs
do not enter complete-run aggregates. G3 capacity shortage remains BLOCKED without
changing N or budgets. Measurement boundaries and instrumentation are unchanged.

The committed helper sets `OMP_NUM_THREADS=1`, `VLLM_ALLOW_INSECURE_SERIALIZATION=1`
for the existing local callable RPC path, clears USE_TORCH/TF/FLAX and pins GPU Python
and visibility per child. Actual values are recorded. Detached launch, status,
owned cleanup, fresh-attempt resume and immutable review bundles remain available.

S1-P CPU contracts: **87 PASS** (synthetic only). Full local pytest with the pinned
vLLM source audit: **1512 passed, 3 GPU/platform skips**. Ruff, compileall, diff checks
and all 11 runbook Bash blocks/helper syntax pass. Final-SHA Linux CI runs after push;
its actual completion status and links are recorded in PR #4 and the delivery handoff.
Existing Python 3.9/3.12 CI and Python 3.11 Phase4/source contracts remain; the 3.11 job
also runs the S1-P regression file. **S1-P GPU / performance: PENDING**, awaiting a
fresh server G0–G3 run. No GPU inference or AutoDL connection was performed by the
agent; no new GPU correctness/performance result is claimed, and S2/S3 have not started.

See [design and audited call chain](phase-s1-design.md), [versioned result/schema](phase-s1-schema.md)
and [copyable server runbook](phase-s1-runbook.md). Stop after this handoff and await
S1-P server results; recovering cross-run token equality is not a prerequisite.

## Phase 4A.0–4A.1: vLLM freeze and Serial Disaggregated correctness

Phase 4 is stacked on the exact frozen PR #3 head
`34c7ea9836c2595c8a8aeaeb5680709520edd3d8` and does not modify Phase 3 algorithms or results.
The serving integration freezes vLLM `v0.25.1` at commit
`752a3a504485790a2e8491cacbb35c137339ad34` in a separate Python 3.11/PyTorch 2.11.0 environment;
vLLM is not a dependency of the Python 3.9 simulator package.

Phase 4A.0 brings up Qwen3-0.6B TP=1 on physical GPU 0 and Qwen3-32B TP=2 on physical GPUs
1–2 as separate stock V1 offline engines. It validates exact source/install provenance, physical
placement, every TP rank's local parameters and memory, selected attention backend, repeated
greedy output and token-level comparison with the frozen HF trajectory on five corrected R3-real
requests. Startup/prefill/decode/wall timestamps are recorded only for bring-up observability.

The Phase 4A.0 adapters freeze future candidate/verification/request-state semantics without
importing simulator policies or proxy latency. vLLM built-in colocated speculative decoding is
not `serial-disaggregated` or SpecRhythm `dual-batch`; vLLM DBO is an intra-model-executor
microbatch overlap and is explicitly disabled. No GPU experiment was run by the Mac coding agent.
Phase 4A.1 changes the serving correctness reference to immutable stock vLLM Target-only greedy
output. The Phase 3 HF trajectory remains advisory provenance only. Before any patch is applied,
the reference command runs the same five corrected R3-real requests twice, verifies token and
termination determinism, records model/tokenizer/runtime pins, and freezes
`stock-target-reference.json` without overwrite. Patched Target-only and two independent Serial
runs must match this reference exactly.

The Serial path uses one persistent Qwen3-0.6B Draft process on GPU 0 with per-request mutable KV,
and the stock vLLM speculative scheduler, batched Target verification, rejection sampler and KV
accounting on Target TP=2. A local Unix-domain-socket custom proposer carries only committed-token
deltas and proposals. The fixed vLLM custom proposer API lacks request identity and exact verify
boundaries, so one zero-fuzz Python patch adds those observer hooks to
`gpu_model_runner.py`. It changes no scheduler/sampler/KV/attention/C++/CUDA code and is inactive
for Target-only generation. Exact base, patch and installed-file hashes are validated.

Every round proves Draft → transfer → Target verify → state sync → next Draft ordering. Draft KV
is cropped after rejection and appends the Target correction/bonus; full-context replay per round
is forbidden. Proposal, accepted/rejected, correction/bonus, bootstrap/tail and final output
accounting are checked independently. The Mac agent ran CPU tests and applied/restored the patch
against the exact vLLM source but did not run CUDA or produce GPU results.

The user-run default-mode A/B gate completed most lifecycle checks, but one of five Serial
sequences diverged from stock Target-only at generated position 1 after an exact BF16
log-probability tie changed under speculative batch expansion. Exact correctness therefore has
not passed. Phase 4A.1.1 adds a symmetric C/D batch-invariant mode, per-rank effective-mode
evidence, actual proposal/logits/position mappings, rejected-KV rollback validation, and
single-request local/remote fixed-proposal controls. Existing A/B artifacts remain immutable
default-mode provenance.

At the exact pinned vLLM commit, the batch-invariance documentation and FlashAttention backend
set the minimum NVIDIA compute capability to 8.0. A800 therefore passes the hardware preflight,
but that preflight intentionally leaves `batch_invariant_effective=false`. The first C attempt on
`a7fe058d` stopped before engine creation because stock runner verification imported vLLM before
mode configuration; the runner verifier now uses package metadata without importing vLLM. The
subsequent C1/C2/D1/D2 artifacts and corrected immutable-artifact validator completed with
`outcome=A`, `valid=true`, exact Target-only repeats, exact Serial repeats, D==C, valid
diagnostics, and identical semantics for all 24 keyed rounds. Their raw cross-request event order
differs, which is scheduler interleaving rather than a semantic change. This Phase 4A.1.1
conclusion is frozen and is not reinterpreted by Phase 4B. See
[phase4-vllm-integration.md](phase4-vllm-integration.md),
[phase4-vllm-source-audit.md](phase4-vllm-source-audit.md), and
[phase4-vllm-server-runbook.md](phase4-vllm-server-runbook.md).

## Phase 4B.0–4B.1: Dual-Batch contracts and GPU correctness readiness

Phase 4B adds a linear, non-eager serving control plane without changing vLLM rejection,
attention, paged-KV, model, TP or default scheduling semantics. The user's A800 artifact at
`96842c8a1e6ffd70c5c1321eecd7384ad74cf542` proved stable/internal identity mapping, but it also
proved two failures: the cadence-based readiness workaround allowed an unproposed Target decode,
and the shell wrapper did not prove that EngineCore/TP descendants had exited. That artifact is
integration-failure provenance, not a GPU correctness result.

Phase 4B.0a replaces cadence mutation with an explicit request-level predicate in independent
patch `0002`. Waiting/Drafting decode is forbidden; a matching prefix-version/hash/round proposal
or legal Target tail is allowed; setup prefill is separate. A blocked request consumes no stock
token/KV budget and does not prevent later work. Scheduler artifacts record the decision for every
request/cycle. Target launch now owns one session/PGID, propagates the coordinator exit code,
records descendants and TERM/KILL actions, verifies Draft/socket cleanup, and blocks the next run
after invalid cleanup.

Phase 4B.0b establishes `DecodeReadyProvider -> DecodeReadyManifest -> consumer` and implements
only `ResidentWarmStartProvider`. Untimed real-KV setup performs Target prompt prefill plus exactly
one bootstrap, then initializes Draft through the same committed prefix. Manifest validation and
a TP barrier precede `measurement_start_ns`; initial proposal generation is forbidden before it.
The first Target forward must consume `[bootstrap]` for Target-only or
`[bootstrap]+proposal` for Serial at exact contiguous positions. A third observer patch records
actual forward boundaries and inputs.

Phase 4 main evaluation is decode-only. Resident warm start is real-KV decode-stage isolation,
not an end-to-end PD deployment. KVConnector handoff remains a future provider and is not
implemented. CPU contract tests pass locally; the coding agent has not run CUDA.

The real-A800 Gate A.1/A.2/A.3 run passed. Gate B then proved that pinned vLLM may legally deliver
only one frozen request in an initial proposer callback; both resident proposers incorrectly
required the whole workload in that callback. The failed `d6c7aa8` directory is preserved.

The corrected contract accumulates immutable stable-ID observations across callbacks. A dedicated
EngineCore scheduler admits prompt/bootstrap work, freezes requests after their first output token,
and releases them only after full-set validation, one TP barrier, manifest creation, measurement
start, and atomic setup-ready publication. The real-A800 `98ec816` run reached the second
incremental request and `_complete_global_setup`, proving the earlier whole-batch assumption was
removed, but failed because the JSON-compatible observation list was reconstructed without tuple
normalization. `ResidentSetupObservation.to_dict/from_dict` is now the only serialized boundary;
Target and Serial both use it and reject malformed token types. The resulting L2 Target run passed
on A800 and is preserved read-only.

The resident Serial attempt at `5db8657` is diagnostic-only because Draft was accidentally
started twice and its live PID provenance was overwritten. Its manifest/proposal hashes and
timing/admission logs nevertheless showed correct round-zero publication and first scheduling,
followed by an erroneous second installation pass after the live prefix advanced. Resident Serial
now owns each initial proposal through `published -> installed -> consumed`, uses pinned vLLM's
`scheduled_spec_decode_tokens` as consumption evidence, preserves detailed fail-closed diagnostics,
and leaves consumed requests exclusively to normal proposer rounds. The subsequent real-A800
resident Target/Serial reruns passed; Phase 4B.0 correctness infrastructure is frozen and is not
reinterpreted by the new Dual path.

GPU 0 runs one persistent Draft service. Heavy Draft model work is serialized on its background
worker queue; Unix-socket calls only enqueue work or poll completed proposals. The Target
scheduler on GPUs 1–2 injects only ready proposals and delegates actual scheduling to the stock
vLLM scheduler. Every proposal carries a canonical ID, round, prefix version/count/SHA256, Draft
KV lengths and token list. A mismatch fails before verification; one request cannot own two
proposals or be drafted through an unverified prefix.

Phase 4B.1 now starts Dual from the same logical decode-ready state. Draft initialization produces
no proposal; rank-zero manifest validation and the Target TP barrier precede measurement; first
proposals are asynchronous and post-boundary. The scheduler emits per-request decisions and a
separate proposal lifecycle, while the unified validator compares Target, Serial and one or more
Dual runs exactly, checks keyed repeats and all logical invariants, and proves that it did not
mutate its inputs. Local CPU tests pass. A complete two-request run at `3ee1c3e` passed Target,
Serial, both Dual executions, controlled Cases A/B/C, the exact output triangle, keyed
repeatability and all underlying semantic components. Read-only replay resolved the remaining
instrumentation issues. A legal Target tail may retain historical proposal metadata only when a
prior `CONSUMED` lifecycle event proves there is no live proposal; Draft readiness and the ordered
one-token terminal transitions remain mandatory. Dual-1 contains a real 57.989848 ms
cross-request temporal overlap; Dual-2 intentionally has none under two-ready coordination. The
historical per-verify rows alias both TP ranks to GPU1 because they used process rank/device zero,
while authoritative worker snapshots already prove Target placement on GPUs 1 and 2. Current
instrumentation uses the actual active CUDA device and cross-validates rank, physical GPU and UUID.
The preserved result remains a correctness artifact only and carries no performance claim. See
[phase4b-decode-ready.md](phase4b-decode-ready.md) and
[phase4b-dual-batch.md](phase4b-dual-batch.md). The active server procedure is
[phase4b1-dual-correctness-runbook.md](phase4b1-dual-correctness-runbook.md).

The first Gate-1-only preparation at `b9a0d6d` froze a deterministic controlled-2 stock reference
and applied all three pinned patches to their exact final hashes, then stopped before any serving
run because its helper incorrectly reused the stock-only checker for the patched installation.
This was not a Dual correctness failure. Patch-state validation now has mutually exclusive exact
`stock` and `patched` modes, immutable success/failure manifests, and explicit helper calls. A
fresh Gate-1-only A800 rerun is required; the earlier root remains immutable provenance.

That rerun at `7e4f871` validated both explicit state checks on A800 and completed resident Target.
Resident Serial then failed only when the common Target-forward observer accessed the Dual-only
`proposal_id` attribute on the legacy Serial `Proposal`; its scheduler had already submitted both
requests for speculative verification. Dual-1/Dual-2 did not run, so Gate1 remains not evaluated.
The observer is now proposal-protocol-aware: it preserves a real canonical Dual ID and records
`null` for Serial or no pending proposal. The Serial schema, execution semantics and all Dual
scheduler/state/lifecycle logic remain unchanged.

Gate semantics are now separated without weakening evidence. Gate1 is controlled two-request
semantic correctness and reports per-run temporal/hardware-qualified overlap independently.
Gate1.5/Gate2 keeps at least one hardware-qualified positive overlap mandatory on at least five
requests through the unchanged asynchronous path. Default validation remains overlap-required;
only controlled Gate1 opts into `separate-gate`. An explicit legacy read-only authority mode
accepts only the exact `3ee1c3e` source, recomputes semantic plus runner-only invariants, and
supersedes only structurally proven historical errors. The helper continues to preserve run and
validator exit codes while always checking cleanup. Subsequent user-run A800 evidence established
Outcome A for controlled Gate1 and default-asynchronous corrected-5 Gate2, including exact
Target/Serial/Dual output equality, correct GPU1/GPU2 TP identity and hardware-qualified overlap
in both Gate2 Dual runs. At that historical checkpoint, Gate3 recovery was the only authorized
next GPU action and Phase 4B.2 remained blocked. The later numerical qualification and explicit
human progression decision documented below supersede that project-level block without changing
the immutable historical artifacts.

The first corrected-100 Gate3 attempt at commit `eba0df4` completed preparation, froze its one
allowed deterministic stock pair and applied the patch stack, then failed in resident Target
setup before global readiness. With 100 requests, pinned vLLM enabled chunked prefill under the
16,384-token budget and delivered one proposer callback containing requests at different setup
stages. Target, Serial and Dual had all assumed every row contained exactly one bootstrap token;
that assumption happened to hold for 2/5 requests. Serial and Dual were not run and Gate3 was not
evaluated. The failed directory remains immutable infrastructure-failure provenance, not stock,
Target-token, Serial, Dual or overlap failure evidence.

All three resident consumers now use one dependency-free row classifier. The authoritative
sampled-token row distinguishes bootstrap from no-bootstrap, while a minimal pinned worker hook
supplies the actual post-forward materialized position count. Partial prefill never binds opaque
identity or initializes Draft; full prompt without a sample remains pending; exactly one sampled
bootstrap records/initializes once; more than one output before global readiness fails closed.
Each wave is logged so an early-bootstrap request can be shown frozen while later requests keep
prefilling. Exception cleanup now snapshots an owned Unix-socket inode, proves the Draft PID dead
before removing the unchanged stale socket, and keeps the lifecycle guard on any live process,
socket identity change or leaked Target descendant. The earlier deterministic stock-100 reference
is reusable byte-for-byte because its stock/model/tokenizer/sampling/runtime/workload contract is
unchanged; reuse records both file hashes and commits and does not measure another stock pair.

The user-run `32b09a6` recovery proved that scale-safe setup and cleanup work for corrected-100:
all bootstrap observations, global decode readiness, the first Target forward, measurement
boundary and TP2 placement passed. Exact output compatibility still failed for four requests at
generated positions 3, 4, 12 and 2; the other 96 were exact. In every divergence the immutable
stock artifact has a `0.125` top-two log-probability margin while resident execution collapses the
same token pair to equal values. This is not accepted as a harmless tie because the stock
preference is nonzero.

The first attempted pair at `c142fa7` failed in both stock TP workers before any numerical
checkpoint because the observer incorrectly treated speculative-only common attention metadata
as the generic block-table authority. Resident and comparator were not run. Its exact directory
is immutable `diagnostic-infrastructure-failed` provenance. The generic observer at `e73e884`
then completed exactly one stock-style and one resident corrected-100 run. Read-only comparison
proved equal actual pre-divergence output history, computed-token boundaries, logical ownership,
and current-token embeddings. Both TP ranks found request-dependent first-different KV layers
4, 21, 24, and 56; earlier layers were exact and later layers differed. Raw logits already differ
and each sampler follows its own argmax, excluding a sampler tie-breaking explanation.

The e73 stock `InputBatch.token_ids_cpu` rows contain pinned-vLLM async `-1` placeholders, so that
field is explicitly non-authoritative metadata. Semantic comparison uses complete run outputs
against the immutable stock reference. The subsequent immutable 8773 per-logical-token run
classified all four requests as `BOOTSTRAP`: every prompt K/V position is bitwise exact on both
TP ranks, and the first difference is the bootstrap token at `prompt_length` in layers 4, 21, 24,
and 56 respectively. The preceding control layers remain exact. The stock endpoint uses ordinary
Target-only async scheduling; the resident endpoint disables async through custom-class resident
execution. This is a causal hypothesis, not proof.

The immutable matched-bootstrap control under commit `efea5c8` subsequently classified
`ASYNC_OFF_MATCHES_STOCK`: ordinary stock Target with async scheduling disabled reproduced the
stock async-ON K/V, raw logits and output for all four divergent requests, not the resident
endpoint. Async scheduling is therefore ruled out as the root cause and no more async, layer,
token-KV, mantissa, class-only, freeze-only or cohort-only micro-diagnostics are authorized. See
[phase4b1-gate3-numerical-diagnostics.md](phase4b1-gate3-numerical-diagnostics.md) and
[phase4b1-gate3-matched-bootstrap.md](phase4b1-gate3-matched-bootstrap.md).

The explicit human engineering decision is now:

- Gate3 structural, semantic-prefix, prompt-KV and logical-KV-ownership correctness: **PASS**.
- Gate3 exact stock trajectory: **NOT ACHIEVED (96/100)**; no tolerance is introduced and the
  four divergent tokens remain visible in immutable artifacts.
- The residual classification is `cross-execution-regime bootstrap numerical divergence`.
- Gate3 numerical qualification: **COMPLETE**; further micro-diagnostics: **DEFERRED**.
- `gate3_exact_stock_equivalence=false` and `phase4b2_progression_permitted=true` coexist by
  design. Historical artifacts retaining `phase4b2_blocked=true` are not rewritten.

## Phase 4B.2: decode-only performance infrastructure

Phase 4B.2 is a measurement layer around the existing resident Target, Serial and Dual-Batch
paths; it is not a second serving implementation. The historical decode-ready manifest boundary
remains unchanged. Performance mode first atomically publishes setup-ready, performs one final
Target-TP barrier and per-rank CUDA synchronization, broadcasts a later monotonic
`performance_measurement_start_ns`, and only then permits the Serial round-zero proposal or Dual
initial enqueue. Target performs no measured Draft proposals. There is no per-token CUDA
synchronization; after generation, one collective RPC synchronizes every Target rank and records
the final completion evidence.

Rank-zero resident callbacks emit explicit semantic token-commit events. Serial proposal commits
use the existing `state_sync_end_ns`, Dual proposal commits use existing `commit_end_ns`, and
proposal-free Target tails use the new explicit commit event. The measurement layer reconstructs
every request as exactly one setup bootstrap plus measured commits and fails if that identity does
not equal the final generated sequence. Per-request latency is final commit minus the shared
boundary. TPOT is `(last_commit-first_commit)/(measured_tokens-1)` and is null for a one-token
request. Makespan is the latest final commit minus the boundary; aggregate throughput is measured
committed tokens divided by makespan.

Standalone mode artifacts cannot produce a speedup. Comparison v2 gates pair and three-mode
speedups on matched work: equal request sets/counts, prompt hashes/counts, bootstrap, maximum
output and measured counts; valid within-mode token accounting and cleanup; equivalent
measurement boundaries, workload/config/model/patch/execution/topology provenance. Exact
generated sequences remain independent diagnostics. Finish/termination differences at the frozen
output length are diagnostic provenance. A matched-work failure suppresses speedup; sequence
differences do not. Timestamped process output is used only to report
post-boundary JIT warnings and `warmup_clean`; it is never a latency authority. The initial
corrected-100 run is functional bring-up, not a final paper workload or result. The Mac coding
agent implemented and CPU-tested this infrastructure without running CUDA. See
[phase4b2-decode-performance-runbook.md](phase4b2-decode-performance-runbook.md).

The first A800 bring-up under `56bd0a5` completed Target and Serial GPU execution. Target derived
successfully; Serial execution returned zero but its derived artifact failed because two
Phase-4B.2 fields were written only to `runtime-manifest.json["phase4a1"]`, not the top-level raw
Serial result. Future Serial runs now publish one canonical evidence block to both locations. A
strict offline compatibility path can reuse only that exact historical execution commit, only
when both raw fields are absent, and only after exact raw/runtime/decode-ready provenance and
two-rank GPU1/GPU2 synchronization validation. It records execution and measurement-code commits
separately and never rewrites the raw GPU artifacts. Target and Dual have no fallback. The
Serial artifact has now recovered successfully: 100 requests, 1487 measured tokens,
50394.65011 ms makespan, 29.50710039169275 tok/s and mean TPOT 2345.39918652 ms;
`warmup_clean=false`, one post-boundary JIT event. Target has 100 requests, 1487 measured tokens,
5813.059543 ms makespan, 255.8033319632212 tok/s and mean TPOT 382.881551485 ms;
`warmup_clean=true`, zero post-boundary JIT events. These are operator-reported GPU results.
Nine Target/Serial trajectories differ after identical bootstrap states; that evidence is retained
without further per-token investigation. The offline comparator verifies per-request counts from
the immutable artifacts before approving the pair. Dual is next, at the same `56bd0a50...`
execution commit; measurement/comparison uses the new commit. PR #4 stays Draft and unmerged.
The resulting claim is preliminary matched-work decode-only bring-up, with no exact-sequence,
output-quality, steady-state or final paper benchmark equivalence claim. Phase 4B.3 sweeps
follow, then Phase 4C Dual-Eager.

After a successful Phase 4B.2 bring-up, Phase 4B.3 adds fixed-output batch/output/context sweeps;
Phase 4C then adds real-GPU Dual-Eager. Arrival-rate, throughput/goodput/SLO and capacity-knee
evaluation remain later work.

## Phase 3.0: GPU readiness and real-trace runner

Phase 3.0 is stacked on the frozen PR #2 head and does not modify its simulator algorithms or
reported results. The default package remains dependency free. An optional GPU extra pins PyTorch
`>=2.7.1,<2.8` and Transformers `>=4.56.1,<4.57`; Transformers is used as a correctness collector
because stable per-candidate draft logits, entropy, and margin are required. It is not the final
serving engine.

The real trace schema separates selector-visible draft features from target-only labels. Completed
request/cycle records are immutable and independently validatable, making interruption/resume
safe. The Phase 3A runner implements deterministic draft-only, target-only, and serial
draft-then-verify collection. A five-GPU coordinator keeps draft TP=1 on GPU 0 and a persistent
target TP=4 worker group on GPUs 1–4. It does not implement Dual-Batch overlap.

The available server has three NVIDIA A800-SXM4-80GB GPUs with NV8 links between every pair, so
the reviewed fallback is 1D2V: Qwen3-0.6B on GPU 0 at TP=1 and Qwen3-32B on GPUs 1–2 at TP=2.
At commit `80d576912028da2d32cd1d8ba5cb593d10a547ae`, a user-run correctness smoke reported Python
3.9.25, PyTorch 2.7.1+cu128, CUDA runtime 12.8, driver 580.126.09, and NCCL 2.26.2. Both model
configs support TP=1/2/4 and correctly reject TP=3 without model surgery. The two-request serial
trace committed 10 accepted candidate tokens plus 6 target-root tokens, and validation reported
`target_only_semantic_equivalence=true`. These observations validate topology, loading, trace,
resume, accounting, and greedy token semantics only; `gpu_measurement=false` is intentional, and
no latency or serving-throughput conclusion follows from this smoke test.

The first user-run primitive smoke exposed a real evidence gap: the TP=2 JSON reported about
34 GB for rank 0 and zero for rank 1 because each process could only observe its own allocator,
while the rank-0 writer never gathered rank-1 state. Two verify runs were also produced by
different commits and therefore are not repeat runs of one implementation. Those v1 files remain
smoke provenance only; their numerical values are not accepted as a latency surface or a
performance result.

Phase 3B.1 replaces that format with a strict v2 schema. Each TP rank now retains its logical and
physical GPU identity, UUID, local parameter count/bytes and device placement, allocated/reserved
memory, forward shapes/checksum, and all CUDA/host samples. Distributed barriers and CUDA
synchronization surround each iteration; rank 0 gathers every rank, and the global sample for an
iteration is the maximum participating-rank latency. Missing ranks, zero model state/memory,
missing samples, device mismatch, invalid statistics, or non-max aggregation fail validation.
Only rank 0 atomically publishes the report.

The hardened statistics retain every sample and report mean, standard deviation, CV, min,
P50/P90/P95/P99/max and flagged outliers, with defaults of five warmups and thirty measured
iterations. Before/after hardware snapshots record observed clocks, temperature, power, P-state,
ECC, memory, PCIe, topology and peer-access state without attempting clock control. Repeated-run
comparison requires an identical commit, config checksum, model revisions, GPU model, TP layout,
backend and operation semantics; raw samples are never pooled across incompatible runs.

All v2 measurements are explicitly `backend=hf_correctness`, `serving_engine=false`,
`kv_cache_reuse=false`, `packed_tree_verification=false`, and
`simulator_latency_surface_compatible=false`. Draft remains serial greedy full-context replay;
verify remains serial full-context replay for `B_cand+1` target forwards. Selector timing still
labels the existing kernel as synthetic `torch.topk`; a dependency-free five-stage selector
interface exists without fake timings. Transfer now covers both draft↔target-leader directions,
the target-leader→TP-peer path when present, and payloads from 4 KiB to 256 MiB, but remains a bare
device-copy primitive rather than complete Draft→Verify transport.

No NVIDIA GPU was available to the Mac agent that implemented Phase 3B.1. The user subsequently
completed the same-commit three-run A800 validation: TP=2 verify run-to-run CV was about
0.9%–1.3% with maximum reported variation 3.04%; draft variation was about 6.7%–8.0%; the
synthetic top-k primitive was about 0.03 ms; large-payload peer copies were stable; and every v2
validation passed. These are repeatability observations for `hf_correctness` primitives, not a
serving latency surface or a performance claim.

## Phase 3C.1–3C.3: real selector diagnosis

Phase 3C.1 adds an isolated, resumable pipeline without changing the simulator. It builds a fixed
100-request public-text pilot with a 60/20/20 code/chat/summarization mixture, binds the first 100
chronological Mooncake arrivals, and stores token IDs and lengths from the actual configured Qwen3
tokenizer. Missing datasets are fatal and no prompts are synthesized as fallback. The 40/50/150 ms
classes remain metadata only.

The draft stage uses Qwen3-0.6B TP=1 to build one shared real 4× forest per request; 1× and 2× are
strict prefix-closed subsets. Node counts 16/32/64 and verification budget 4 are derived from the
frozen Phase-2 width/depth/speculative-budget configuration. The target stage uses Qwen3-32B TP=2
to generate one immutable greedy continuation per request, independent of ratio. Both remain
full-context Transformers correctness paths with `kv_cache_reuse=false`.

Label join keeps draft runtime features and target-only labels in separate objects. Five
target-blind selectors and a within-request target oracle replay the same forest, target and fixed
budget. Reports cover pool/selected target-path coverage, candidate efficiency, oracle regret,
depth/probability/entropy calibration, sibling hits and pool robustness. Stable request-level
train/validation/test splits prevent candidate-node leakage.

The user completed and validated the 60/20/20 real Qwen3 pilot. At budget four,
Residual-Probability accepted 1.92 tokens/proposal at all three pools; the within-request oracle
accepted 2.23/2.30/2.30 at 1x/2x/4x. These are selector-signal observations, not latency or system
performance. The old `target_path_pool_coverage` fell as the nested pool expanded because it used
each pool's own realized maximum depth as its denominator. It was not density and did not have a
fixed recall denominator.

Phase 3C.2 preserves that legacy field and adds fixed-denominator monotonic target-path recall,
target-node density, selected precision/recall, K=4/8/16 horizon coverage, first-missing depth,
16/16/32-node shell decomposition and selection-set stability. Per-task and overall reports now
use request-level stratified bootstrap intervals and paired oracle/Residual-Probability
comparisons. Headroom separates generator coverage, selector regret and budget limits; zero oracle
expansion gain is explicitly not identifiable.

The prompt audit also found that Phase 3C.1 ShareGPT chat prompts were raw first-user text without
the Qwen chat template. Those 20 old chat traces remain legacy diagnostics and cannot be pooled
with corrected data. The v2 builder applies the Qwen tokenizer chat template with
`enable_thinking=false`, retains native HumanEval completion prefixes and the explicit
CNN/DailyMail summarization instruction, and records deidentified rendering/tokenizer metadata.

The corrected 20-request (12/4/4) multi-round mode freezes each at-most-16-token target once, creates one
shared forest at every target-prefix position, and replays every selector sequentially over those
same snapshots. Immutable checkpoint/resume and final-token equality are enforced. The Mac agent
implemented and tested this path without running a GPU model. The user then completed the
corrected-20 server run: Residual-Probability stayed at 2.407 accepted/proposal across pools,
Entropy-Margin reached 2.421, and the Oracle rose from 2.680 at 1x to 2.877 at 2x with no further
4x gain. These are token-efficiency observations only.

Phase 3C.3 promotes the corrected run to 100 requests (60/20/20), adds a strict cross-artifact
validator, request-stratified bootstrap intervals and paired deltas, and restricts formal analysis
to 1x/2x. It decomposes the 2x shell into generator coverage, budget/prefix reachability and
ranking failure. A diagnostic linear `learned-shell-ranker` uses fixed runtime draft features,
70/15/15 task-stratified request splits, immutable model/replay provenance and a predeclared
held-out A/B gate. Target labels are available only during offline training; inference rejects
labeled nodes. The Mac agent implemented and CPU-tested this path but did not run the corrected
100-request Qwen trace. That server run and artifact review are the next gate.

Packed-tree, vLLM/SGLang, Dual-Batch, Eager, SLO evaluation and simulator calibration remain out of
scope. See [phase3-real-trace.md](phase3-real-trace.md) for definitions and boundaries.

## Phase 1.5: residual selection

Four residual policies freeze the exact same-state Dual-Batch request set, budgets, path nodes,
candidate forest, roof, and deterministic target outcome, then differ only in residual selection:
request round-robin, path probability, current SLO-aware two-stage, or feasibility-gated two-stage.
All runs report zero base-preservation violations, and residual roof utilization is aligned within
0.15 percentage points at 2.75×, 0.06 points at 3.0×, and 0.05 points at 3.25×.

The decisive result is Probability versus Shaping. At 3.0×, probability raises goodput
2146.6→2452.3 tokens/s, raises attainment 0.728→0.816, and lowers mean queueing 4.30→2.43 seconds.
At 3.25×, it raises goodput 1414.4→1587.4, raises attainment 0.457→0.502, and lowers queueing
19.17→15.05 seconds. The two are effectively tied below the knee at 2.75×.

This rejects the current SLO-stage formula as a forward mechanism: filling idle roof helps, while
global path-probability selection is more efficient than the SLO-weighted residual stage under
pressure. The rejected policies remain only for provenance and diagnosis. The next mechanism gate
is candidate selection or Overdraft-and-Prune, not further tuning of these SLO weights.

The scalar candidate roof is explicitly not a GPU capacity claim. The proxy charges both request
root positions and candidate positions, while GPU calibration must measure the joint surface
`T_verify(B_req, B_cand, C)`. Full results and definitions are in
[phase1.5-residual-selection.md](phase1.5-residual-selection.md).

## Phase 2: oracle headroom

Phase 2 leaves every existing policy unchanged and exposes two separate diagnostic commands.
`phase2-replay` performs the primary same-snapshot causal comparison, while `phase2-simulate`
reports secondary end-to-end, fully-hidden-search upper bounds. Neither command is part of the
normal policy order.

The historical target oracle samples its next target child from the tree passed to verification,
so changing pool width would also change ground truth. The isolated Phase-2 oracle instead freezes
the historical target trajectory on the immutable 1× tree, then adds deterministic, prefix-closed
branches. This makes `A_1×` strictly comparable with Residual-Probability and keeps target truth
constant across ratios and selectors. It also means the canonical target is always present in the
1× pool: this proxy can measure selector, cross-request residual-allocation, and full-tree gaps,
but it cannot identify real missing-target or better-drafter pool-coverage headroom.

The primary replay uses 10,000 deterministic, stratified snapshots per load and keeps
`B_verify`, request roots, the target outcome, and the proxy verification surface fixed while
expanding metadata-only `B_search` to 1×/2×/4×/8×. Variants A/B/C/D isolate the current
target-blind selector, within-request target oracle, global residual oracle, and full-tree oracle
ceiling respectively. All are marked diagnostic-only; B/C/D explicitly leak target outcomes and
all ratios assume search is fully hidden. Detailed definitions, sampling coverage, results, and
decision boundaries are in [phase2-oracle-headroom.md](phase2-oracle-headroom.md).

The completion audit reused all valid interrupted-run artifacts and executed only nine missing
3.25× cells. Final coverage is 3/3 common replays with exactly 10,000 corrected-queue snapshots
each, 9/9 references, and 48/48 end-to-end oracle cells. A_1× reproduces Phase-1.5
Residual-Probability at all three loads with exact integer equality and floating-point absolute
tolerance `1e-12`.

On common snapshots at 8×, B−A adds 16.36/17.07/17.55 committed candidates per cycle across
2.75×/3.0×/3.25×; C−B adds 1.69/3.18/4.02; D−C adds 2.49/2.83/3.01. All
360,000 ordered dominance checks pass. A loses 9.26/9.34/9.33 candidates per cycle versus A_1×
because added branches are distractors around a target already fully covered by 1×.

End to end, A's 1×→8× goodput falls 2567.0→1761.1, 2452.3→1114.0, and
1587.4→742.1 tokens/s. B largely removes the selector loss; at 3.25× it reaches
2952.6/.9412 goodput/attainment at 4×. C reaches 3030.2/.9963 and D 3033.7/.9988, with C/D
core outcomes unchanged by ratio because their oracle already finds the 1× target. These are
fully-hidden-search system ceilings, not deployable or GPU-measured performance.

## Phase 1: shaping causal diagnosis

Three opt-in diagnostics were added without changing the default algorithms:
`shaping-feasible`, `shaping-residual`, and `shaping-feasible-residual`. One-cycle feasibility uses
the frozen total progress gap, counts one future root exactly once, and does not label a request
globally unsalvageable. Residual variants freeze the same-state Dual-Batch request set, per-request
budget, selected candidate path, and root opportunities before shaping otherwise-unused roof.

The scoped full-R3 proxy results are:

| Load | Policy | Goodput tok/s | Attainment | Queue s | P90 TPOT ms | Total progress/cycle | Infeasible opportunities | Stage-1 → infeasible |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.75× | Dual-Batch | 2557.9 | 0.990 | 0.36 | 28.4 | 63.61 | 0.055 | 0 |
| 2.75× | shaping | 2566.5 | 0.996 | 0.25 | 26.3 | 69.74 | 0.035 | 336,357 (85.7% of stage 1) |
| 2.75× | feasible | 2565.7 | 0.996 | 0.25 | 26.3 | 69.74 | 0.035 | 0 |
| 2.75× | residual | 2567.5 | 0.998 | 0.19 | 25.0 | 69.62 | 0.027 | 128,113 (92.0%) |
| 2.75× | feasible + residual | 2566.8 | 0.997 | 0.19 | 24.9 | 69.61 | 0.027 | 0 |
| 3.0× | Dual-Batch | 1968.3 | 0.673 | 5.91 | 119.9 | 69.93 | 0.409 | 0 |
| 3.0× | shaping | 1746.8 | 0.632 | 11.50 | 228.3 | 74.60 | 0.425 | 3,597,597 (99.0%) |
| 3.0× | feasible | 1750.3 | 0.633 | 11.28 | 224.3 | 74.64 | 0.424 | 0 |
| 3.0× | residual | 2146.6 | 0.728 | 4.30 | 93.6 | 76.70 | 0.353 | 1,177,804 (99.4%) |
| 3.0× | feasible + residual | 2144.7 | 0.727 | 4.31 | 93.6 | 76.71 | 0.354 | 0 |
| 3.25× | Dual-Batch | 1291.6 | 0.421 | 22.70 | 373.5 | 73.24 | 0.640 | 0 |
| 3.25× | shaping | 1117.4 | 0.394 | 37.43 | 612.5 | 76.81 | 0.655 | 5,421,587 (99.6%) |
| 3.25× | feasible | 1116.2 | 0.394 | 37.29 | 609.8 | 76.87 | 0.654 | 0 |
| 3.25× | residual | 1414.4 | 0.457 | 19.17 | 321.8 | 80.57 | 0.609 | 1,761,091 (99.8%) |
| 3.25× | feasible + residual | 1415.0 | 0.457 | 19.14 | 321.6 | 80.57 | 0.608 | 0 |

The feasible-only guard removes essentially all stage-1 allocation to one-cycle-infeasible
requests but leaves goodput almost unchanged. Preserving Dual-Batch breadth/base opportunities is
the materially positive intervention at 3.0× and 3.25×. Combining the guards adds no material gain
over residual preservation alone. This supports breadth/root opportunity cost as the dominant
modeled cause at the proxy knee; it does not validate a new default or a GPU performance claim.

All residual runs report zero base-preservation violations. Their base trees come from the exact
Dual-Batch allocator and sequence-path materializer on each variant's same cycle state; residual
nodes are prefix closed and never evict base work. Because earlier scheduling choices can make
independent runs reach different later states, the invariant is a same-state counterfactual, not
a claim that cycle IDs across divergent runs always contain identical active request sets.

Detailed summaries remain outside Git under
`SpecRhythm-data/results/simulator-semantics-v0.2/phase1-shaping-diagnosis/`.
The full table, class-good-token breakdown, progress accounting, and utilization-denominator audit
are in [phase1-shaping-diagnosis.md](phase1-shaping-diagnosis.md).

## Superseded flat-proxy evidence

- The superseded flat revision had 80 passing tests. The current tree-aware/Phase-2 revision has
  115 passing tests and is Ruff clean, including prefix closure, Figure 5 selection, width-1
  degeneration, path-dependent eager, goodput denominator, proposal/token/tree-node conservation,
  and both Phase-2 CLI commands. The recovery baseline had 114 tests; one integration test was
  added to exercise `phase2-replay` and `phase2-simulate` through the CLI.
- Full Mooncake R3 proxy replay validates at 12,031 requests for 1×, 2×, 4×, and 8×
  time scales, corresponding to 3.401, 6.802, 13.605, and 27.210 requests/s.
- 1× and 2× remain mostly below the configured proxy capacity; 4× and 8× show large queueing
  delay and SLO violations, confirming that pending time is included.
- These results used a **legacy flat-sequence shaping proxy**. They diagnose that proxy only and
  cannot compare AdaServe's or SpecRhythm's tree-aware algorithms.
- Eager compute-waste ratios are high (roughly 0.76–0.88 across the reported proxy sweeps).
  Confidence, admission threshold, latency surfaces, and roof calibration remain open gates.

Generated workloads, validation reports, and comparison JSON remain in the external data tree
and are not committed.

## Tree-aware capacity-knee evidence

The full R3 proxy sweep covers 2.0×, 2.25×, 2.5×, 2.75×, 3.0×, 3.25×, 3.5×, 3.75×, and 4.0×.
The proxy knee is between 2.75× and 3.0×: shaping attainment falls from 0.996 to 0.632 and mean
queueing rises from 0.25 s to 11.50 s. At 3.0×, tree-aware shaping reaches 1746.8 good tokens/s
versus 1968.3 for Dual-Batch; SpecRhythm reaches 1895.8 versus 2645.2 for Dual-Eager. These are
negative proxy results for the frozen tree-aware control plane, not evidence about GPU execution.

At 3.0×, Dual-Batch to shaping adds 2,050,577/522,522 candidate nodes to the 40/50 ms classes,
but realized accepted-node gains are only 99,604/29,814. It also adds 148,942 nodes to the 150 ms
class while realized progress falls by 24,880. There are 3,814 tight requests that receive more
candidates but still miss SLO, and 385 previously attained 150 ms requests lose attainment.
Goodput falls through both numerator (-216,670 SLO-good tokens) and denominator (+29.68 s
makespan). The allocator transfers budget, but proxy expected progress does not translate into
sufficient realized progress.

At the same 3.0× point, flat→tree goodput changes are 498.3→485.3 for AdaServe proxy→tree,
1747.6→1746.8 for shaping, and 1791.9→1895.8 for SpecRhythm with eager. Tree semantics therefore
change both allocation and eager outcomes; the legacy flat result is retained for provenance but
cannot substitute for the tree-aware control plane.

The residual-score ablation at 3.0× gives identical shaping output for path probability and
urgency × path probability under this workload/configuration (1746.8 good tokens/s). The frozen
default remains urgency × path probability; equality here is a diagnostic result, not a reason to
change it.

The complete eager grid at 3.0× spans budgets 1/2/4 and dependency thresholds 0.1/0.2/0.3/0.5;
no cell is filtered. At threshold 0.1, Dual-Eager goodput is 2593.0/2645.2/2651.1 for budgets
1/2/4, while SpecRhythm is 2208.5/1895.8/1827.5. Raising the threshold to 0.5 removes all
Dual-Eager admissions and nearly all SpecRhythm admissions. In the detailed final default run,
SpecRhythm's admitted dependency paths have slightly higher mean probability than Dual-Eager
(0.273 versus 0.258), but consume more eager tokens (910,391 versus 636,371). Counterfactual
same-cycle allocation attributes 846,688 displaced normal nodes to SpecRhythm eager versus only
96 to Dual-Eager. This rejects the "lower-probability admission" explanation and instead
identifies admission volume and normal-budget displacement as the leading mechanism to inspect.
The default is not changed after observing results. These are control-plane sensitivity findings,
not GPU performance claims.

## Current simulator contract

The comparison order includes AR, Serial SD, retained flat-proxy diagnostics, tree-aware
AdaServe, Dual-Batch, Dual-Batch + Rolling Eager, tree-aware shaping, and SpecRhythm. Serial SD
and Dual-Batch share allocation and differ only in exposed cycle latency. Tree-aware AdaServe implements the paper's two-stage
node selection with proxy trees/latencies; it is not the complete AdaServe system. Exact formulas
and root accounting are frozen in [tree-aware-design.md](tree-aware-design.md).

Serial SD and Dual-Batch produce identical allocations for the same fixed logical batch. Their
different cycle duration may alter later trace admissions and queueing, which is an intended
system-level consequence of overlap.

Every request reports queueing, service, and end-to-end decode latency. Every proposal is tracked
to exactly one terminal state, with proposal/token promotion, invalidation, EOS discard, and
compute-waste ratios.

## Open work and known limitations

- `input_tokens` is stored but not used by the current latency model.
- Context-dependent draft/verify latency is not implemented.
- `D(B,K,C)`, `V(B,K,C)`, acceptance, confidence, and candidate roof are proxy inputs until GPU
  calibration.
- R3 proxy lengths are sampled and are not HumanEval, Alpaca, or CNN/DailyMail payloads.
- Phase 4B.1 has a decode-only asynchronous linear Dual runner, complete Gate1/Gate2 A800
  correctness evidence, and completed Gate3 numerical qualification. Phase 4B.2 performance
  infrastructure is implemented but has no A800 result yet. It has no SGLang, packed-tree
  verification, Dual-Eager, KVConnector, arrival scheduling, load/SLO or final-paper evaluation.
  The persistent Draft HF adapter is still a narrow serving prototype.
- No current result may be cited as evidence of real GPU speedup or full AdaServe/SpecRhythm
  reproduction.
