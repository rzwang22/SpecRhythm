# S2 — GPU-resident Prefilled-KV Dynamic Decode Serving

Status: implementation and CPU contracts; real-GPU qualification is pending. The operator
reported S1-P G0–G3 PASS at `5a00049e2eabf09f535fdd5f187f77406f6dcfe2` in
`/root/autodl-tmp/SpecRhythm-data/results/phase-s1/s1p-5a00049-20260909T144802Z-1469`.
S0 and S1 artifacts remain read-only. S2 uses `specrhythm-s2`, separate schemas and a fresh root.

At `c12b3768eaaaeca3ecde03999440b7b91763b128`, the operator reports capacity PASS
(small100/large390, 3:3:2:2), calibration/G0 PASS and ten completed requests in each of G1
Target and Serial. G1 PingPong failed at its first verification with missing `uuid_queries`;
there is no passed G1 gate, so G2 is blocked. The retained root is
`/root/autodl-tmp/SpecRhythm-data/results/phase-s2/s2-c12b3768eaaa-20260910T012202Z-1476`.
These are operator-reported results, not an agent GPU run or complete S2 qualification.

## S2 terminal-drain concurrency contract

At `24b31a9e0125d773697d92ec0ea333384616e539`, the operator reports that the G1
coordinator/Target, Draft and effective exit codes were zero; cleanup and owned cleanup
completed with no remaining owned PID. Qualification rejected `S2 same-cohort stages overlap`.
The reported pair was cohort A `finish_tail` for
`sr-43f9b5ce7d32f7c6d51db9d5d71ee542a1e9db750c55a65de3b83755671d6495`
overlapping Target proposal verification for
`sr-de0452eb0a569d5a6c4775e2c0df7f56753fe0b9038b3e5d132aa334d5ed9ad7`.
Draft host interval `[17257369905601164, 17257369927652247]` and Target host interval
`[17257369914500012, 17257369966438131]` intersect by **13.152235 ms**, with no common
request ID. This is a reported host-envelope intersection, not measured kernel overlap.
The full retained server artifacts were not accessed by the coding agent.

The source chain is `S2PingProposer` -> inherited `DualBatchRemoteProposer._rank_zero_update`
-> `DynamicClient`/`CohortDraftClient` enqueue -> `BatchedDualDraftController` owner thread
-> `S2DualMachine.execute_batch` -> `BatchedDualDraftMachine._commit_many(target_tail=True)`
-> `S2DraftBackend.commit_many` -> production `VllmBatchedDraftBackend.commit_many` ->
`finish_many` -> `VllmDraftWorker.release`. The Target hook requires a claimed proposal-free
tail and a terminal one-token delta, advances the committed prefix/version and logs TERMINAL
before enqueueing. The Draft dispatcher independently rejects pending proposals, nonterminal
tails, non-monotonic versions and invalid hashes. It materializes the final committed prefix,
fences, updates that request's state and releases its blocks. All tail plans are terminal, so
the active set passed to subsequent proposal generation is empty.

Distinct IDs alone do not establish safety. Draft mutations of the shared model, InputBatch,
request table and KV allocator are serialized on the same checked CUDA owner thread. Worker
materialization receives only the dispatched internal request IDs; release sends only those
IDs through `finished_req_ids`, fences and frees their allocator views. Prefix caching is
disabled and S2 checks private block ownership. Target runs in independently bound GPU1/2
workers while Draft owns GPU0. The new S2-only receipt additionally checks the complete Draft
block table immediately before release, records actual materialized prefix/block identities,
and checks that every unrelated request's prefix/materialized length/block table is unchanged
before, immediately before release, and afterward. Unrelated request IDs remain in scope even
if they belong to the same cohort. No scheduler or shared Phase4/S1 implementation changes.

The exemption applies only to `finish_tail` with a successful owner work result, native Target
TERMINAL evidence preceding dispatch, the exact next-version terminal prefix, no resulting
proposal or proposal forwards, retirement of precisely its own requests, unchanged unrelated
private KV and independent device evidence. Every overlapping Target request must be covered
by that unchanged unrelated pool. Same-cohort overlap requires a single-cohort **verification**;
same-request overlap always fails. Active drafting/commit-and-propose keeps the opposite-cohort
dependency rule, including operations that happen to return no proposal. Later work on a
retired request, premature resource reuse and missing mandatory proof still fail.

Receipts accumulate in memory in `draft-backend-report.json.s2_work_records[].terminal_drain`.
They add CPU scope audits during terminal work, with their cost inside the unchanged observed
work envelope; no new GPU fence or per-verification/fsync log is added. The resource timestamp
is sampled after the actual release returns. Coordinator token completion and active-slot
release remain distinct: the unchanged `release_finished` decision holds a finished request
while the controller still reports it in flight. The slot cannot be reused until the whole
Draft operation returns; final Target fences and all resource drain still precede `end_ns`.
All materialization forwards and GPU event time remain in the existing measured totals.

`result.json.stage_dependency_contract=specrhythm.s2-terminal-drain.v1` labels this refinement.
`physical_overlap`/`overlap_ms` retain the existing proposal CUDA-stage calculation, which
does not include proposal-free `finish_tail`. `stage_host_overlap_ms` retains the original
opposite-cohort host union, including any cross-cohort drain. `active_stage_host_overlap_ms`
excludes validated drain. `terminal_drain` separately reports work count/host cost, all and
same-cohort host overlap unions, pairs and token/KV/slot completion timestamps. These unions
are not additive and do not establish exact kernel overlap or critical-path time saved.
Errors carry the operation, terminal receipt, both request sets/cohorts/intervals, intersection
and the relevant backend/work/state/Target/runtime/capacity artifact paths.

CPU regressions run actual S2 dispatch and production commit/release with simulated hardware,
then use the full offline qualifier without replacing its validators. A same-cohort A tail /
different-request verification fixture was rejected by the base commit's qualifier and passes
the refined contract. Tests also cover invalid terminal/proposal/version/hash/KV evidence,
active same-cohort and same-request conflicts, legal cross-cohort and zero overlap, foreground
errors, preserved work/makespan, and a blocked real owner release holding the coordinator slot.
This does not qualify the retained real run. Old artifacts/seals remain unchanged; missing
terminal receipts do not silently receive the exemption. New code uses a new root, remeasures
capacity and repeats calibration/G0/G1 before G2/G3. Cross-run token/length/EOS/round equality
remains NOT_REQUIRED. GPU retest and complete S2 qualification remain pending.

## PingPong worker startup dependency correction

The inherited `DualBatchRemoteProposer.on_target_verify_end()` calls
`self.uuid_queries.for_verification()`. Its constructor does not create that worker-bound
object. Legacy `dual_runner` first executes `worker_dual_runtime_snapshot` on every TP worker;
that helper creates `DualVerificationUuidQuery(worker)` using the real device snapshot and
assigns it to `worker.model_runner.drafter.uuid_queries`. S2 previously used only the ordinary
`_worker_runtime_snapshot` and omitted this startup side effect.

S2 now dispatches `initialize_pingpong_worker` immediately after LLM creation, before capacity
confirmation, prefill and verification. It reuses the existing Dual initializer and the same
Target TP/device validation. Target-only and Serial continue to dispatch `target_snapshot`.
Later prefill/final memory snapshots call the ordinary worker snapshot and only read
`worker_dual_uuid_evidence`; they never reinitialize the query or reset its counters. A second
initialization remains an error under the existing helper. Live mode and its actual
per-verification UUID subprocess queries are unchanged.

Startup evidence is in `actual-capacity.json.target_worker_ranks[].dual_uuid_query`; prefill
evidence is in `resident-pool.json.target_rank_initial_memory[].dual_uuid_query`; final evidence
is in `runtime.json.target_final_memory[].dual_uuid_query`, read after the existing drain and
Draft shutdown. The inherited verification log retains the actual two-rank UUID/device
intervals. Each rank has one initial validation. Capacity probes have zero verification
queries/accesses/cache hits: they collect startup evidence, not a UUID A/B experiment, and
must not use that experiment's nonempty-verification gate. All artifacts use the existing
attempt seal; no per-verification logging, clock boundary or additional GPU fence is added.

The earlier context test deliberately stopped at LLM construction. The coordinator test's
`collective_rpc` returned canned rows without invoking callbacks, and its fake `step()` never
entered inherited verification hooks. The separate Dual UUID unit test supplied the query to
an isolated observer, so it did not test S2's ownership of initialization. The new CPU contract
crosses that missing boundary: real S2 `run/configure/make_engine`, real proposer constructors,
callbacks executed on two simulated workers, real PingPong verification start/end, and final
counter/report reads. Only hardware, transport and inference inputs are simulated; no test
preassigns `uuid_queries`. Before this correction it reproduced the exact AttributeError at
`vllm_dual.py:659`; afterward the chain, duplicate-init rejection, repeated reads, zero-access
probe, missing-UUID failure and unchanged Target/Serial startup paths pass. This remains CPU
contract evidence, not GPU verification.

## Observation boundary

**GPU-resident prefilled-KV delivery; prefill/import excluded; queue/decode/drain included.**
Arrival models ideal PD delivery: the selected request's private KV is already available on
its decode GPUs. This is a **finite-trace serving observation**. It does not measure network
PD transfer, mixed prefill/decode interference, complete PD end-to-end user latency, kernel
self time, or steady-state service capacity.

Every attempt creates fresh Target and Draft engines. All selected prompts are processed
before the S2 barrier using the qualified `ResidentWarmStartProvider` observations. The
Target has materialized the prompt; its one sampled bootstrap is the pending next input.
Draft has materialized prompt **plus bootstrap**, retaining the corresponding next-logits
GPU view. The existing decode-ready manifest is validated metadata; it is not a KV tensor.
No snapshot tensor copies or cross-process KV file format are introduced. Request IDs,
positions, exact prefix hashes, materialized lengths and physical block tables are checked.

Each allocator retains private blocks for staged/queued requests. Removing an unscheduled
request from vLLM's `InputBatch` does not free its cached request/KV state in the pinned MRV1
source. S2 checks the unchanged staged prefix and block table before every Target step and
around Draft proposal/commit operations. Timed Target preemption is rejected before freeing
blocks. Initial blocks of active requests may grow but cannot be replaced. Live block tables
cannot alias across requests. Fully terminal bootstraps need no continuation KV: setup may
release those blocks, and arrival records their terminal state with zero timed tokens.

## Capacity and identity

The models, GPU0 Draft TP1 / GPUs1,2 Target TP2, BF16, eager model execution, K=4, natural EOS,
greedy sampling, per-request budgets/seeds, UUID live mode, numerical batch-invariant setting
and five vLLM patches remain as frozen by S1. No Shaping or new candidate-selection policy is
introduced. `runtime_profile.py` delegates unchanged to S1/legacy when S2 is absent.

Target scheduler/model-input capacity is 512 **resident pool slots**. Global active decode
capacity is 128 in all three modes; Serial can use all 128. Draft's existing forward capacity
is 128, and its allocator can retain more request views than one forward's input rows. Target
query-token capacity remains 4096. Actual B/Q and active counts are reported separately.

A no-request engine probe is run for each of the three modes. It collects actual cache block
capacity, block size, active device UUID and free CUDA memory after engine initialization on
Draft and each Target rank (nine records). Static estimates do not authorize timing. The
conservative block calculation, separately for each rank and mode, is:

```
all initial prefix blocks
+ sum of the largest min(128, N) growth requirements through full output cap + K4
+ one partial/copy block per possible active request
+ max(32 blocks, ceil(5% of actual cache blocks))
```

Target prefix length is prompt length; Draft prefix length includes bootstrap. Allocation
rounding is applied per request. There are zero snapshot copies. Actual free memory must
also cover an extra 512 MiB workspace margin and, on Draft, `N * vocab_size * 4` bytes of
cached logits. Engine profiling has already reserved model/cache/workspace memory. The
formula is restricted to the pinned dense single-KV-group layout. It does not rely on early
EOS. Each fresh run checks actual capacity again and verifies the complete physical initial
pool before publishing its barrier. A changed or insufficient runtime fails; it never shrinks
its own workload during performance execution.

Small requests 100, large requests 500. The nested order uses the exact S1 mixed100 set first,
then independent seeded hash ranks within each class from remaining main1000. Every prefix
of ten has chat/code/reasoning/summarization counts 3/3/2/2. Prompt bytes/IDs, S0 source and
Mooncake provenance, sampling, budgets and EOS remain untouched. Neither short-request
selection nor calibration/main mixing is permitted. Both desired sizes decrease by ten until
all nine rank/mode checks pass. One plan freezes actual_N for every mode/rate/declared seed.
If large cannot exceed small, G3 explicitly skips the duplicate scale.

## Clock, admission and cohort boundaries

S2 explicitly sets `VLLM_ENABLE_V1_MULTIPROCESSING=0` and `async_scheduling=False`. The pinned
`InprocClient.get_output()` executes exactly one EngineCore step. TP execution remains in the
existing workers. A background EngineCore could advance between coordinator calls, so the
runtime rejects that client type rather than assuming `LLMEngine.step()` controls it.

The arrival observer is a separate monotonic-clock thread, including while the coordinator
waits for Target or Serial Draft GPU work. It retains frozen planned arrivals, observes due
requests, records observation lag, and queues them. Admission happens only between completed
Target steps. The FIFO queue uses the frozen trace order. A finished request continues to
consume an active slot until its real Draft final synchronization/release has drained.

Lifecycle: `STAGED -> QUEUED (ARRIVED event) -> ACTIVE -> FINISHED`, with explicit FAILED rows
and the primary failure on an interrupted observation. Bootstrap-terminal requests go from
QUEUED to FINISHED at observed arrival, with no admission or timed commit. The observation
clock never pauses for trace idle, queueing, admission, control serialization, scheduling,
synchronization or drain. Process teardown and final report serialization occur afterward.

Target and Serial gate timed scheduling on ACTIVE status. S2 Serial generates its first real
Draft proposal on admission and imports that validated proposal at the first verification
hook. Later proposal, acceptance, rollback, EOS and synchronization algorithms are inherited.

PingPong inherits Dual's real readiness, TP sampled-row mapping, verification, acceptance,
retired-ready handling and request/proposal lifecycle. It assigns admitted requests to the
smaller currently held cohort, ties to A. It only modifies a cohort when that cohort has no
in-flight Draft work and Target has finished the preceding step. Cohort membership then stays
stable for that request's lifetime. Target selects one ready cohort at a time, switching after
a scheduled unit; an empty or blocked cohort does not prevent the other eligible cohort from
continuing. It does not wait for future requests or permanently divide the entire pool N/2.
Readiness is decided using the inherited admissibility check, including consumed-proposal
checks, rather than the presence of historical proposal metadata.

Control snapshots are atomically replaced without fsync only when admission/state/initial
proposal metadata changes. Ordinary continuing steps reuse the snapshot. Initial metadata is
retired from the control packet after the first observed commit and retained once in final
raw evidence. Arrival/commit/population statistics are in memory and emitted at completion;
existing native worker diagnostics/logging retain their original behavior. Foreground log
mirroring reads the original Target/Draft logs and cannot replace their exit codes.

## Trace and SLO

Poisson inter-arrivals are generated by `Random(arrival_seed).expovariate(1.0)` and divided by
lambda, with the first arrival exactly zero. A separate `Random(order_seed)` shuffles request
order. Float seconds retain precision; only conversion to the monotonic nanosecond clock is
rounded. Every rate uses the same unit exponential samples and order for its frozen set/seed;
every mode uses the same trace file/hash. Defaults are .25/.5/1 requests/s and one seed pair
1667/1668. These are engineering observation points, not asserted capacity multiples.
Additional seed pairs require explicit `prepare --extra-seed ARRIVAL ORDER` before freezing;
all pairs then share the same capacity-selected IDs and SLO. No repeats are silently added.

SLO can be supplied explicitly per task or derived from a separate calibration200 sample:
five per class, seed 1670 hash ranking, Target-only and active limit one. Calibration uses the
same fresh preloaded KV and observation clock. It reports queueing, but its *baseline* uses
`(completion - admission) / timed tokens` so earlier calibration requests' queueing cannot
inflate the isolated decode reference. The engineering threshold is class median ×1.5.
A class with no timed calibration tokens requires explicit thresholds. Calibration's fixed
20 requests must also pass real capacity; if they cannot fit, use explicit thresholds.
The S0 `slo_policy_ref=null` remains unchanged. The new policy is sealed before main runs and
cannot be altered after seeing main results. It is not an inherited paper SLO.

Main request SLO is **queue-inclusive decode average ms per timed token**:
`(completion - planned_arrival) / n`. Queueing and arrival handling lag remain inside it.
Zero-timed-token requests are reported separately and excluded from the SLO denominator.
Request attainment is good requests / eligible timed requests. Token goodput counts actual
timed tokens of good requests; request goodput counts good requests. Both divide by the same
uninterrupted observation duration used by throughput, including idle and drain.

Commit timestamps are actual coordinator-visible output publication times. A verification
that emits several tokens assigns that one observed timestamp to all of them; no interpolation
is performed. Request TPOT is `(last timed commit - first timed commit)/(n-1)` for n>1; raw
within-request intervals preserve zero gaps inside a batch. Queue wait, first timed-token wait,
TPOT and queue-inclusive average latency have distinct field names and distributions.

## Qualification and limitations

Mandatory: successful real exit and owned cleanup; every request terminal/released; exact
within-run bootstrap + native worker commits = observed output; token/budget/EOS accounting;
request/round/prefix and Target structural evidence; TP sampled-row consensus; private resident
KV and actual capacity; frozen workload/config/model/backend/trace/SLO identity; ordered clocks
and real GPU timing for executed forwards. Empty optional statistics are count zero/None.

Cross-run token, output length, EOS and round equality are **NOT_REQUIRED**. Slow PingPong,
zero overlap, single-cohort execution, lack of multi-request batches under sparse arrivals and
zero SLO attainment are valid observations. Reports include actual output work, tok/s and
makespan together; a shorter run alone does not establish equal-work speedup.

Overlap retains two explicit scopes: the inherited synchronized CUDA-stage envelope
intersection/union with actual Draft/Target TP identity, and a separate broader host-stage
intersection. Neither is kernel self time or automatically time saved on the critical path.
Initial/peak KV block use and GPU memory are reported; memory peaks include engine/setup
allocations, not only timed decode. Full events/block tables/logs remain in raw sealed attempts;
the upload summary contains aggregate metrics and raw paths/hashes, not giant token arrays.

CPU tests use real adapter/provenance/allocator-contract code with explicitly synthetic
workers and inputs. Source tests inspect the pinned vLLM client, scheduler and InputBatch
contracts. They cannot prove real GPU residency, model correctness, capacity, asynchronous
join/leave performance or speedup. Those remain the operator's G0–G3 validation work.

See [schema](phase-s2-schema.md) and [server runbook](phase-s2-runbook.md).

## Independent fixed64/32 diagnostic (2026-09-11)

The new [diagnostic entry](fixed-concurrency-diagnostic-design.md) reuses the S2 private
resident setup and readiness contracts through optional capacity/observer parameters.
It does not change this document's S2 defaults or Poisson/arrival-to-drain interpretation.
Its short-window cancellations, initial-state stages and separate lightweight/audit
statuses must not be passed off as S2 full-trace qualification.
