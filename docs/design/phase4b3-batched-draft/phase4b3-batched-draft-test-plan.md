# Phase 4B.3 — batched Draft test and qualification plan

Status: **tests to implement and gates to run in a follow-up task**. This task creates design/audit documents only. It has not constructed a worker, run pytest against a new backend, run Linux CI, or executed GPU. No AutoDL connection is required for this audit. The user will run server gates after implementation.

Sources and architecture: [MineDraft audit](phase4b3-minedraft-source-audit.md), [pinned API map](phase4b3-vllm025-draft-api-map.md), [backend design](phase4b3-batched-draft-design.md). SpecRhythm base is `64031110e630b6643cc610620f05f8630e896c7c`; pinned vLLM is `752a3a504485790a2e8491cacbb35c137339ad34`.

## 1. CPU/Linux tests before any GPU gate

Separate pure planning/state tests from vLLM source/interface tests. Neither test class may import CUDA runtime or load a model on ordinary Linux CI. Fake tensors/caches/runners must test meaningful independent invariants rather than returning whatever the implementation expects. Tests that require real attention/KV execution belong to D1–D3 and must not be advertised as proved by mocks.

| Test group | Required cases | Independent oracle / failure it detects |
| --- | --- | --- |
| Backend selection | Default remains HF; explicit vLLM opt-in; invalid value; Target-only config unchanged; Serial/Dual selectors recorded | Frozen config manifest and backend factory calls; no implicit backend switch |
| Construction guards | Wrong source hash, MRV2, wrong model revision/device set, TP/PP/DP≠1, native speculative config, hybrid/windowed cache, connector/async/DBO/unsupported sampling | Construction fails before model mutation; no fallback to different semantics |
| Model/cache singleton | Fake worker load/cache-init counters; repeated initialize does not reload model; exactly one allocator/owner | Counts from independent fake resources; prevents two-model native-proposer shortcut |
| Device identity | Real-query adapter represented by explicit validated response; missing/malformed/mismatched UUID/device fails; logical CUDA0 alone is insufficient | Existing device provenance validator; no UUID or identity derived from rank |
| State authority | Stable↔Draft-ID bijection, generation/version checks, duplicate batch IDs, stale proposal/round/hash, unresolved proposal | Control-plane canonical immutable snapshots; validation must precede GPU/allocator side effects |
| Row mapping | Permute input and physical rows independently, compact EOS subset, reorder re-added rows, noncontiguous internal IDs, missing/extra/duplicate IDs, excess capacity rows | ID-tagged logits with independently assigned expected tokens; positional zip must fail |
| Ragged batch packing | B=2/4/8, differing valid lengths, one/two-token commit suffixes, mixed per-row budgets | Query spans cover each suffix exactly once, positions start at R, total scheduled tokens equals sum; no B scalar executes |
| Allocation | Block boundary −1/0/+1, multiple groups, growth, rollback within/across blocks, insufficient pages, finish/reuse | Fake allocator owns unique page IDs; no fabricated IDs, no rejected prefix publication, no premature/double free |
| KV algebra | Enumerate p=0..budget, a=0..p, optional correction/bonus, EOS/output terminal, multi-round chains | Reference token-list model computes C'; independent sentinel KV array proves valid positions equal C' and rejected slots cannot be read |
| Materialization lag | p=1; all accepted with bonus; full acceptance without tail; partial/first reject; empty-suffix nonterminal refresh | Assert `R=min(M_old,L+a)`, positions of actual writes, and `M=len(C')` only after completion; detect last-token/bonus off-by-one |
| EOS/budgets | EOS at proposal step1/interior/final, Target correction EOS, bonus EOS, output remaining0/1/k/k+1, no-proposal tail | Existing stop/acceptance functions and immutable committed token oracle; never propose after terminal or count padding |
| Pinned rebase seam | Existing ID as NewRequestData; shorten context; keep allocated blocks; changed row order; clear worker output state | Fake faithful pinned `_update_streaming_request` transition plus AST guards; no fake finished ID, no misuse of PP-only `new_token_ids` |
| Forward completion seam | Nonempty execute returns pending state; empty cleanup output; missing/unexpected state; double completion; execute-before-completion; output buffer reuse | Stateful fake refuses incorrect protocol; owned logit snapshot survives next runner call; no stock sampled-token append |
| Completion restrictions | Unexpected connector/hidden hybrid/async/drafter state, wrong logits shape, unconsumed completion field | Fail closed; tests may not erase the unexpected state to make a run pass |
| Serial batching | Existing synchronize-and-propose payload with multiple rows | One batched commit execute per nonempty group and p−1 batched proposal executes in homogeneous case; unchanged response order and logical evidence |
| Setup/boundary | Incremental bootstrap, duplicate setup observation, partial prefill, deferred initial proposal, setup failure | Existing DecodeReady/materialization counts and timestamp ordering; no proposal selection before measured boundary |
| Reports | Histogram/quantile edges, actual-forward versus RPC counts, per-purpose sums, no-EOS/early-EOS, shared batch references, incomplete event buffer | Independent count fixture; one physical batch not summed B times; unknown internal synchronization coverage stays null/unknown |
| Shutdown/failure | Terminal, cancel while queued/inflight, backend exception mid-batch, transport failure, repeated finish/shutdown | No successful acknowledgement of partial mutation; all owned resources released through existing lifecycle mechanism |
| D6 queue integration, later | Preserve one enqueue envelope/order; mixed operation compatibility; duplicate inflight, claim/retire race, late result | Existing Dual lifecycle state machine tests; no new accumulation wait, selection or cross-envelope reorder |

Use randomized CPU sequences of proposals/acceptance/terminal actions in addition to fixed edge cases. At every successful commit, compare canonical token lists/hashes to the independent reference and verify all materialized positions. Mock logits should depend on the entire valid input prefix, so an accidentally exposed rejected slot changes the next token and fails the test.

### Source/API tests at the exact vLLM revision

Retrieve/export source only on CPU CI. Record its commit/hash and parse with the Python AST; do not import the vLLM package to inspect definitions.

1. Old classes (`SpecDecodeWorker`, `MultiStepWorker`, `TP1DraftModelRunner`, `SmallerTpProposerWorker`, `Top1Proposer`, `SpeculativeProposals`) remain absent. Runtime backend imports must use present pinned V1 classes.
2. Assert constructor and required dataclass fields for Worker, WorkerWrapperBase, UniProcExecutor, KVCacheManager, Request, SchedulerOutput and New/CachedRequestData. Test optional/default handling from the source rather than copying a stale list into a mock alone.
3. Verify ordinary Draft's same-TP guard; the selected backend must not instantiate native DraftModelProposer or pass Target speculative configuration into the Draft worker.
4. Verify `_update_states`→`_update_streaming_request` handling of an existing ID, replacement of context/block/computed state and clearing of output tokens. Explicitly test that `CachedRequestData.new_token_ids` is not used as the TP1/PP1 correction mechanism.
5. Verify pending `ExecuteModelState` creation/consumption and the execute-before-sample guard. Enumerate completion obligations for the supported feature set; unknown changes fail the adapter's source guard.
6. Test on an exact **five-patch-applied source fixture** as well as pristine source analysis: existing hook guards must stay inactive for ordinary Draft and unchanged for Target. Verify original patch files and expected runner/scheduler hashes; no fuzzy patch acceptance.
7. Verify native rejected-token debit, compaction and slot-mapping anchors used by the design. This guards architectural assumptions; it is not evidence of GPU numerical equivalence.

During the implementation task, run focused new CPU tests, relevant Phase4 tests, full pytest, Ruff and existing Linux CPU CI. Include existing Serial/Dual row-domain, EOS, retired-ready, terminal recovery, lifecycle, request-state and UUID A/B tests. Run broader tests once after the integration changes; repeat only after a relevant new change/failure. All test invocations/versions/results must be recorded against the actual implementation commit.

## 2. GPU gate protocol, executed later by the user

Do not skip directly to corrected-100 or Dual. Each gate consumes the preceding gate's immutable manifest and either passes or stops. Preserve failed artifacts. Do not relax exact-output, lifecycle or overlap validators to pass a performance run.

Common manifest fields: SpecRhythm commit, exact vLLM commit and installed source hashes, five-patch manifest, backend selector, UUID query mode, model/tokenizer/revision/dtype, device UUIDs and mapping, Target TP2, Draft TP1, eager setting, attention/cache configuration, proposal/output budgets, stop policy, workload hash, seed, measurement-boundary identity and diagnostic/logging settings. Freeze these for comparisons except the explicitly named experimental variable.

### D1 — construction and teardown

Construct only the new Draft backend under the existing service lifecycle. This includes actual CUDA initialization/model loading/cache profiling and warm-up; those are future server operations.

Required evidence:

* Exactly one real Qwen3-0.6B model, parameters resident on logical CUDA0 bound to the configured physical GPU0 UUID, no Target weights.
* Expected `UniProcExecutor`, V1 GPU Worker and MRV1 runner classes; independent world size1, distinct rendezvous, no participation in Target TP2 collectives.
* KV specs/config, group/block sizes, initialized storage, real capacity sufficient for the frozen workload; no synthetic capacity fallback.
* No native DraftModelProposer/double load, unsupported features rejected, completion/rebase source guards match the installed five-patch state.
* Clean service exit; no remaining owned model process or live Draft request allocation.

Pass only on actual placement/cache evidence. A configuration string naming GPU0 is insufficient.

### D2 — one request, multi-round KV correctness

Use identical tokenized contexts/model/dtype/greedy settings for HF and vLLM Draft. Test both natural proposals and deliberately constructed external verification outcomes to exercise every commit case without depending on the Target to happen to reject at a desired position.

Required cases: first-token rejection, partial rejection, full acceptance, correction, bonus, p=1, accepted EOS, correction/bonus EOS, output limit, no-proposal tail, multiple rounds, block boundary crossing, finish and repeated cleanup. Injected commits must satisfy the same existing acceptance contract; mark them as correctness fixtures, not performance samples.

At each step compare proposal tokens where deterministic, final canonical committed tokens, prefix hash/version, physical M and logical length. After commit, independently recompute next logits/tokens from C' in a separate diagnostic reference path to expose stale KV; this replay is **outside performance gates**. Inspect selected KV positions or a diagnostic hash against the same model's fresh-prefix reference where useful, acknowledging cross-backend numerical differences. Require no use of rejected tail and no post-EOS proposal.

If HF/vLLM greedy tokens diverge, save the first context, token positions, ranks/device/config and top logits for diagnosis; D2 has not passed. Do not change EOS, precision, budget, tolerances or Gate3 policy as part of this backend task.

### D3 — heterogeneous B=2/4/8

For each B, use different prefix lengths/budgets, reorder request submission and physical batch order, cross a block boundary and include an EOS subset. Require:

* One real batched autoregressive forward for the eligible set, demonstrated by model-call evidence and actual per-forward row map, rather than only a B-row RPC payload.
* Exact proposal equivalence against each request's qualifying reference where deterministic, independent of batch row order; active/EOS masks and public token lengths correct.
* One ragged commit batch mixing first rejection, partial acceptance and all-accept/bonus; valid M and next proposals per row.
* No row alias, shared writable KV between requests, duplicate proposal, extra model copy or leaked allocation.

Do not demand every step has B rows: early EOS and heterogeneous budgets legitimately reduce B_active. Require the recorded histogram to explain the reduction and at least one actual decode/proposal forward with B>1 for each multi-request fixture.

### D4 — corrected-5, HF Serial versus batched Serial

Use the existing corrected-five workflow and validators, identical frozen config except backend selector. Require completed requests, matched bootstrap/work policy, exact/matched semantics under current validation, output-token accounting, valid raw Serial evidence, stable prefix/row identity, lifecycle and request-state validation. Preserve timing boundaries and provenance. No Dual, candidate selection or scheduler tuning.

D4 demonstrates full service/Target integration after isolated Draft correctness. New sampler hooks on Target or different Target forward metadata are not an acceptable fix for a Draft integration failure.

### D5 — corrected-100 Serial performance milestone

Compare **HF Serial vs vLLM-batched Serial** with 100 corrected requests, Qwen3-32B Target, Qwen3-0.6B Draft, same proposal and output budgets, decoding/stop semantics, matched-work policy, Target configuration and eager mode initially. Reproduce the known workload/output accounting (100 completed and 1487 measured committed tokens for the established exact baseline), subject to the existing immutable reference manifest rather than a hard-coded synthetic report count.

Required output: original correctness/performance artifacts plus final Draft backend aggregates. Both runs must pass the current exact/matched workload validators, raw validity, lifecycle, request-state and TP checks. New Draft statistics must demonstrate B>1 in **proposal/commit execution**, not just setup. Verify no setup/proposal computation was moved across the measured boundary, and count total model forwards including KV update separately from proposal-only forwards.

The question is whether the backend change materially reduces the roughly 49 s Serial makespan. Report makespan, throughput, per-purpose Draft forwards, batch p10/p50/p90, proposed tokens, KV materialization, host gaps and explicit synchronization evidence. Do not set an arbitrary absolute makespan target. Use repeated independently recorded pairs with alternating run order after an initial valid pair, under a user-approved server budget; show raw runs/spread rather than only the best run. Compare acceptance/proposal counts and outputs as well as time, so a workload change cannot masquerade as a backend gain.

Correctness plus batching without a measured improvement is a valid finding, but **does not authorize D6 as the performance-success next step**. Investigate the recorded backend costs and define a separate bounded improvement experiment. Do not add Dual tuning to hide an inconclusive Serial result.

### D6 — same backend, current Serial versus current Dual

Only after D5 establishes Serial correctness and backend performance. Integrate the same backend inside existing Dual scheduling with bounded enqueue-envelope batching. Hold backend implementation, Draft/Target configs, proposal budget, Dual microbatch size, ready-window policy, logging/diagnostics and UUID mode fixed.

Require 100 completed requests, matched workload, same output accounting and required exact outputs; Dual raw valid, overlap gate valid, lifecycle valid, sampled-row TP consensus true, request-state validation clean, no stale/retired-ready resurrection and no post-EOS round. Prove physical Draft/Target overlap using the existing boundary/timebase rules and deduplicated actual batch intervals. A successful asynchronous enqueue is not overlap evidence.

Compare batched Serial with batched Dual to determine whether current Dual hides Draft latency. If Dual batches remain small, report actual envelope and model batch sizes. Changing microbatch/ready-window/accumulation/rhythm/shaping is a **later experiment**, never a hidden D6 adjustment.

## 3. Evidence integrity and stop conditions

All final reports are immutable, with run IDs and content hashes. Existing lifecycle/correctness logs remain as configured; new performance aggregates are kept in memory and emitted once. Report counter scope (setup/warm-up/measured), histogram weighting, event coverage and unknown internal synchronization counts explicitly.

Stop the gate on device/rank mismatch, source drift, unsupported runtime feature, missing completion/row evidence, rejected-tail exposure, false materialization length, changed output/stop semantics, unmatched workload, missing committed tokens, duplicated lifecycle events, leaked processes, or incomplete required overlap/TP evidence. Keep failed run artifacts; never regenerate a success report from an incomplete run.

Server command lines should be generated by the follow-up implementation task after the actual selector/CLI and tests exist. This document intentionally does not present invented executable flags. Existing corrected runbooks remain the baseline to extend with a backend selector and new immutable output directory, leaving Target controls and measurement policy fixed.
