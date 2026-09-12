# Fixed64/32: validated prompt-binding reuse

GPU performance **PENDING**. This change is one opt-in metadata optimization on top
of `c1dd96d8c86e321d66d52aea31eb7396bf06786b`. PR #4 stays Draft/Open; no GPU, AutoDL,
full CPU audit, algorithm change, vLLM patch change or additional experiment grid.

## Binding-alias integration repair (2026-09-12)

The first bound-prefix Serial GPU attempt at
`5b7081e0eb692822db18afedf3ce9bd36fcd9bee` is **FAILED / INVALID**, not a
performance result. The supplied failure archive confirms effective rc=125,
cleanup PASS, owned cleanup complete and no remaining owned PID. Its primary
worker error is `verify-start hook observed an unmapped vLLM request`. Later
TERM/KILL and missing final reports are consequences, not the first cause.
The old root and artifacts remain unchanged.

`RemoteDraftProposer.__init__` stores
`internal_to_stable = identity.internal_to_stable`. Fixed worker startup then
calls `fixed_identity.install(drafter, "identity")`. The old optimized constructor
copied both binding dictionaries before replacing `drafter.identity`. Subsequent
`identity.bind` updated the copies: S2's `identity.stable_id` succeeded, but the
parent Serial verify-start looked in the original empty dictionary and failed.
Verify-end used that same stale alias and could miss its timing update.

The repair changes only the optimized constructor's two binding assignments:
retain `original.internal_to_stable` and `original.stable_to_internal` **by
reference**. The new object changes the matching strategy, not the owner's binding
state. Existing constructor aliases, future binds, reverse lookup, lifecycle
records and reports consequently see one pair of owner-lifetime containers. There
is no per-round copying, alias repair in a hook, fallback to linear or new cache.
The immutable prompt-table copy and prefix-free proof are unchanged.

| Consumer | Actual installation / read path | Impact |
|---|---|---|
| Serial worker | fixed `target_startup` → install S2Serial identity; S2 hook resolves identity, parent start/end read `internal_to_stable` | Direct failure fixed; timing hooks and accounting run unchanged |
| PingPong / Serial-split workers | same fixed startup installs S2Ping identity; Dual verification resolves identity; `_transition` reads `stable_to_internal`; `_write_report` reads `internal_to_stable` | Latent missing lifecycle IDs and binding-report entries fixed; Serial-split uses runtime mode `pingpong` |
| Target-only worker | same startup installs S2Target identity; setup binds and sampled commits resolve `identity` directly | No stale diagnostic alias; same binding history preserved |
| All four schedulers | `FixedBatch.__init__` installs `_resident_identity` or `_dual_identity`; subsequent accesses use that object | No separate alias; selection, root/candidate accounting and KV checks unchanged |

Each original scheduler/proposer/TP worker still creates its own containers. Only
the old and replacement identity facades **within that owner** share them. Startup
installation occurs before verification; repeat install returns the same optimized
object, preserving bindings, lock and counters. Later mutation uses its existing
locked `bind`; legacy aliases are readers. Released-request identity history is
retained exactly as before. Current full prompt, changed-identity, reverse-alias,
proposal/prefix/version and live UUID checks remain mandatory.

### Regression boundary and reproduction

Before the runtime edit, the new core test was run against the actual failing HEAD:

```bash
python -m pytest -o addopts='' -q tests/test_serving_fixed_startup.py \
  -k test_serial_install_bind_verify_and_commit
```

Result: **2 passed (linear), 2 failed (bound-prefix)**. Both optimized cases
(empty installation and installation after one existing binding) failed at the
same `vllm_remote.py` verify-start error as the server. With the two-reference
repair, all four pass. The extended startup/identity suite passes 36 tests.
Full local pytest: 1772 passed, 3 skipped; Ruff, compileall, Python 3.9 grammar,
Bash/runbook syntax and diff checks pass. Linux CI also runs this startup file in
the existing Phase4 Python 3.11 job and both full Python 3.9/3.12 suites; no new
workflow or GPU test is needed. Final CI status is included with the delivered SHA.

The core uses real frozen inputs/provenance, S2 configuration, actual proposer
constructors inside the CPU LLM substitute, actual fixed startup/install, later
bindings, S2 initial-proposal admission, inherited start/end hooks and
`_rank_zero_propose → _finalize_round`. It verifies two requests over three rounds:
batch IDs/timestamps, two accepted candidates plus one Target correction per round,
two rejected candidates, generated-prefix advancement and persisted hook/round
reports. GPU outputs, synchronization and Draft RPC responses are substitutes;
identity installation, binding, hooks and acceptance/accounting are not mocked.
The CPU resident fixture has a 32-token budget so all three rounds remain active;
it does not alter the server workload.

Negative cases retain unbound/change/alias/stale-proposal/duplicate-hook errors.
Additional real startup tests check Target sampled-commit identity, both TP owners,
Dual lifecycle IDs and report bindings, live UUID queries and zero-verification
capacity probes. Existing real fixed scheduler tests cover all modes, 64/32,
refill/cancel/history and physical KV corruption. Independent linear/optimized
comparison fixtures now construct separate owners; they must not share the same
original map when comparing operation counts.

The previous tests compared matching results/costs, exercised scheduler consumers
without legacy aliases, and ran PingPong hooks without asserting the binding and
lifecycle report contents. They never ran Serial verification after replacing its
identity object. That missing adapter-to-consumer boundary allowed the regression.

No scheduling, model, K, logging, device-query frequency, drain, error precedence
or measurement code changes in this repair. Defaults remain linear/original-live,
and cross-run equality remains NOT_REQUIRED. CPU contracts are not GPU acceptance:
new-root Serial then PingPong server verification is still **PENDING**.

## Retained evidence and time positions

Reviewed `analysis/attribution.json`, the projected runtime/backend/event files and
all nine compressed export hashes in the supplied timing archive. Archive SHA256:
`69f4ec4d0cc20aa307707d7100614a43a776037f9d46e591b37e40020777de17`.
Original analysis JSON SHA256:
`b3ea04d20c219e50582c112464d4c58cfaa2c6bf476ac118bde04757d417aaab`.
These hashes verify the supplied export bytes; unavailable original server files
cannot be independently rehashed here. `evidence-export/report.md` is export-only.
Its UNKNOWN is not the independent analysis result.

The existing externally bounded CPU analyzer completed a fresh local reanalysis
in 0.82 s, rc=0, 9 files / 294,378 counted records, no full audit. Both joins and
clock coverage remain clean. PingPong has 12 complete64 rotations / 24 B32 steps,
752.749409–758.353526 ms CUDA-event overlap; interval clipping to each actual
Target step gives 20 POSITIVE and 4 ZERO. Serial has 12 B64 steps and ZERO overlap.
These are CUDA-event bounds, not exact kernel overlap. Old artifacts are unchanged.

Using the rank-0 model hooks, coordinator schedule/step boundaries and the existing
request/cohort/prefix/round/commit joins reproduces these ms per full64 rotation:

| Host position | Serial | PingPong | Difference |
|---|---:|---:|---:|
| Schedule start → schedule end | 63.154 | 112.406 | +49.252 |
| Other pre-model (step → schedule, schedule → model) | 2.930 | 55.974 | +53.044 |
| Model pre-hook → post-hook | 51.490 | 102.271 | +50.780 |
| Model post-hook → step end | 1032.223 | 1067.722 | +35.498 |
| Remaining rotation gaps | 1.126 | 16.294 | +15.167 |
| Actual complete64 rotation | 1150.924 | 1354.666 | +203.742 |

The raw request joins, not adjacent event indices, establish each rotation. The
last row uses actual rotation boundaries; the other rows describe positions, not
exclusive CPU work. Throughput is 166.232 / 140.243 tok/s and committed window tokens
2296 / 2280. These are the user's existing GPU results, not new optimization results.

Clipping observed host categories to those positions gives additional evidence:

* PingPong rank 0 other-pre: JSON union 16.563 ms/rotation, checkpoint union
  15.492, control-read union 0.816, TP barrier union 0.537. JSON/checkpoint overlap
  and must not be added. Serial counterparts are 0.160 / 0 / 0.259 / 0.362.
* PingPong rank 1 barrier union in the same pre-model position is 38.666 ms/rotation
  (Serial 0.543). `DualBatchRemoteProposer.on_target_verify_start` claims proposals,
  broadcasts, checks tokens/prefix/version and advances/logs per-request lifecycle
  on rank 0, then both ranks enter the barrier. The wait is consistent with rank 0
  doing that work. It does not prove the barrier is redundant or every observed
  rank-0 interval is on the critical path.
* Rank-0 nvidia-smi union is entirely post-model: 139.782 ms/rotation in PingPong.
  Rank 1 is 133.902. These ranks must not be added. It explains none of the pre-model
  gap. Serial has no Dual UUID helper, not missing measurements of that helper.
* PingPong reads control 1,632 times in 24 steps: `S2PingProposer._admitted_initial`,
  `DynamicClient.call → assignment`, and `AnnotatedLog → _cohort → assignment`.
  Most read time is post-model (18.531 ms/rotation vs 0.816 pre-model). Control is
  published dynamically; this change does not cache it. Checkpoint payload and
  checksum encodings serve separate purposes and are not removed.
* Both model host calls average about 51 ms per step. PingPong calls twice per
  complete64 rotation. This includes actual model dispatch/launch and hooks;
  recorded JSON/log/query/barrier timers do not explain that interval. There is
  no source/trace proof it is all Python or removable. No engine/graph change.

Forward durations remain purpose-separated: Serial D_proposal=64.338,
D_commit_or_prefix_sync=19.940, V_target=159.132 ms/rotation; PingPong
130.111 / 48.680 / 215.659. Unadjusted sums are 243.410 / 394.450, not critical paths
or exhaustive GPU work. The legacy `unattributed_arithmetic_gap_ms=52.702639` now
explicitly says NOT unobserved time, exclusive overhead, remaining critical path or
potential savings. Positive overlap makes a forward-sum subtraction especially
unsuitable for those interpretations. TP rank times remain separate/max/union.

## Selected mechanism and source proof

`FixedBatch.schedule → PoolScheduler.schedule → ResidentSetupScheduler.schedule
→ _bind_requests → FrozenPromptIdentityMap.bind → match` is the Serial/Target path.
PingPong and Serial-split use `DualBatchScheduler.schedule → _bind_vllm_requests`
at the same point. Both rebind every live resident row before admission decisions,
including the 36 not-yet-active resident rows. Old `match()` compares each current
physical row against all 100 frozen prompt tuples, even with a previous binding.
With the retained unchanged population this is 10,000 candidate comparisons per
Serial schedule, 20,000 per PingPong rotation. Token normalization still reads
the full current physical row and remains in place.

The scheduler also performs two physical-pool snapshots/audits per step; their KV
blocks/materialization can change across stock scheduling, so neither is removed.
Dual builds admissibility snapshots and scans cohort eligibility; time/readiness/
consumption state can change, so no decision/TTL/proposal result is cached.

`--identity-matching bound-prefix` uses only the already proven binding. At each
fixed scheduler/proposer startup, `BoundPromptIdentityMap` takes an immutable copy
of the owner-local frozen prompt table and proves it is prefix-free by checking
adjacent lexicographically sorted prompts. If prompt p prefixes a physical row,
any other matching prompt q must prefix p or have p as its prefix. The prefix-free
proof excludes both, so checking the **entire current p** proves unique identity.

Every bind retains original integer normalization, empty-ID rejection, stable-ID
change rejection, and reverse-map alias rejection. Initial binding, a changed or
short current prompt, unknown ID, or a non-prefix-free workload uses the original
full scan, preserving its precise missing/ambiguous/alias errors. The original
`bind` implements those checks and state updates; only its matching hook is
specialized. No IDs are guessed and no digest collision is trusted.

The same specialization is installed on each fixed Target worker's proposer before
prefill/verification. Serial's `_rank_zero_propose → identity.bind` and
Dual's `_rank_zero_update → identity.bind` receive the same optimization on their
post-model bookkeeping path. Target-only setup bindings are covered too. Draft
execution, scheduler choices, proposal/commit ordering and all vLLM patches stay
unchanged. The original `match()` remains the default S1/S2 implementation.

## Lifetime and invalidation

The owner is one scheduler or one TP worker's proposer; maps/counters are never
shared between them. Installation occurs once, before owner use. Later snapshot
RPCs only read evidence. A reentrant lock protects binding lookup/check/update and
counter snapshots; there is no background task, extra RPC, GPU fence or disk write.
Storage is bounded by the frozen workload plus the original historical binding maps,
not by token/round count. No current row or generated prefix is retained by the reuse.

Commit/version/round changes, refill, cancellation and KV allocation/release cannot
invalidate this immutable proof: **none of that state is cached**. Each bind reads
the current physical tokens. New internal IDs take the original matching path;
retired identity bindings live as long as before, preserving retired-ready handling
and preventing an illegal alias from passing after release. The frozen table is
read-only; replacement is a material error. A new workload/owner builds a new map.
Failure leaves the same binding state as the original path and does not promote a
failed point to PASS. Non-prefix-free workloads remain correct but may show zero
reuse; this is explicit evidence, not a performance improvement.

## Configuration, evidence and limits

Default `prepare --identity-matching linear` retains the original search. Opt-in is
frozen in `diagnostic-config.json` and each manifest's
`fixed_diagnostic.options.identity_matching`, independently of `observation`.
The launcher exports `SR_FIXED_IDENTITY_MATCHING` from that manifest to its owned
children. S1/S2 environment cleaning strips it; a leaked shell variable cannot
override a prepared fixed root. `SR_PHASE4_DUAL_UUID_QUERY_MODE=live` is untouched.

`runtime.identity_matching` and each `target_devices[].identity_matching` report
mode, immutable proof status, prompt count, bind calls, validated reuses, full scans,
candidate comparisons, equivalent linear comparisons, errors and inclusive host ns
inside the binding lock (lock acquisition wait is not included in that counter).
The linear comparison count is a counterfactual operation count, not saved time.
Scheduler step deltas use the same keys and stay in the existing compact step record.
Light results contain `identity_matching.by_owner` (startup→final snapshot) and
`measured_scheduler` (window steps only); comparisons and offline exports preserve
the selection. Rank-1 and capacity zero accesses are valid. No per-request timer
events or token-prefix/KV dumps are introduced.

CPU tests execute fixed → S2 → resident/Dual gates and actual pool audits with only
stock allocation/model interfaces substituted. For a 100-request already-bound
step, the real matching hook makes 100 full-prompt comparisons instead of 10,000;
selected IDs, candidate/root accounting and rows match with the switch off/on.
Separate tests observe actual full `match()` invocations falling from 400 to 100
over four passes, while checking changing physical rows, ambiguous/short/missing
prefixes, alias/history errors, owner isolation, threaded binding, cancellation,
refill, KV corruption, startup TP/live UUIDs and repeated snapshots. Existing
logging, primary-error, bounded drain and full S1/S2/Phase4 tests remain required.

Remaining unknowns: old traces do not isolate identity matching duration; neither
49.252 ms scheduler difference nor any other entire interval is attributed to this
mechanism. CPU operation-count reduction is not GPU throughput evidence. Follow
[the two-mode foreground runbook](fixed-identity-runtime-runbook.md), then stop.
