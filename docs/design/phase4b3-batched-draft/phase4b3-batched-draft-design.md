# Phase 4B.3 — batched vLLM Draft backend design

**Decision:** retain the separate GPU0 Draft-service process and implement **architecture B, Plan A** first: one ordinary vLLM TP1 Worker/MRV1 runner, one private paged-KV manager, and a restricted SpecRhythm batch/materialization adapter. First qualification is **HF Serial versus vLLM-batched Serial**. Dual integration follows D5; scheduler tuning follows a separately qualified D6.

This is an implementation specification, not implemented runtime behavior. No GPU experiment was run. Baselines are SpecRhythm `64031110e630b6643cc610620f05f8630e896c7c` and vLLM `752a3a504485790a2e8491cacbb35c137339ad34`. The UUID A/B experiment remains orthogonal and unchanged.

## 1. Current data-plane audit and performance motivation

The following are verified in the current source, not inferred from a backend name:

| Current implementation | Source | Observed execution shape |
| --- | --- | --- |
| One resident HF causal LM | [draft_service.py:335–393][sr-hf] | `AutoModelForCausalLM.from_pretrained(...).to("cuda:0")`; one model, per-request `_HFRequest` |
| Persistent request KV | `initialize`, lines 395–412 | Initial full-prefix forward; retain `past_key_values` and last logits; require cache `crop()` |
| Scalar autoregressive proposal | `propose`, lines 414–433 | `argmax(...).item()` for each proposed token; p−1 model calls for p actual tokens; early EOS stop |
| Incremental model input | `_append_token`, lines 464–478 | `tensor([[token_id]])`, effectively `[1,1]` |
| Rollback | `rollback`, lines 435–447 | Crop to old committed length + accepted count; recover saved prefix logits; full acceptance can require forwarding the last proposal token |
| Correction/bonus materialization | `append_target_token`, lines 449–455 | Another scalar forward, then clear pending proposal state |
| Serial batch API | `DraftStateMachine.batch_propose`, lines 92–154 | Python loop calls scalar backend once per request; `draft_microbatch_count=len(rows)` is not evidence of a model batch |
| Serial commit+propose RPC | `synchronize_and_batch_propose`, lines 231–256 | Loop through synchronizations, then loop through proposal requests |
| Dual service | [DualDraftMachine._propose][sr-dual], lines 213–287 | One request's proposal plus existing CUDA timing/synchronization evidence |
| Dual queue | `AsyncDualDraftController._run`, lines 428–478 | Consumes one `_Work`, calls one machine operation, then moves to the next |

The HF backend avoids full-context replay each round, but it does not batch Draft model execution. Current Dual overlaps serial Draft work on GPU0 with Target verification on GPUs1/2. A worker queue alone does not make its model calls batched.

The established real-A800 baseline supplied for this task is:

| Mode | Completed requests / measured committed tokens | Makespan | Throughput | Target forwards | Target request batch p50 | Draft proposals / derived HF forwards |
| --- | --- | --- | --- | --- | --- | --- |
| Target | 100 / 1487 | 5.814 s | 255.77 tok/s | 15 | 99 | — |
| HF Serial | 100 / 1487 | 49.279 s | 30.18 tok/s | 11 | 32 | ≈500 / ≈2027 |
| HF Dual, original baseline | 100 / 1487 | 53.215 s | 27.94 tok/s | ≈267 | 2 | ≈501 / ≈2057 |

The supplied offline Serial attribution is 28.116 s proposal host interval + 14.590 s KV synchronization/update = 42.706 s, about 98% of the 43.465 s Serial-minus-Target gap. These are prior measured/derived values, **not new measurements or kernel-time attribution from this source audit**. They motivate batching both proposal and commit-tail work; they do not predict the speedup of a different backend.

## 2. Architecture comparison

| Criterion | A — full independent Draft LLMEngine/EngineCore | B — dedicated-service Worker + ModelRunner | C — native Target-engine Draft proposer |
| --- | --- | --- | --- |
| Real batching | Built in, with its own scheduler | Explicit batch planner over current runner | Built into Target speculative pipeline |
| KV ownership | Engine scheduler/KV manager | Draft service owns KV manager; runner owns storage | Coupled Target/native-Draft cache and attention metadata |
| Lifecycle isolation | Separate process possible, extra engine machinery | Existing process retained, one CUDA owner | Draft lifetime coupled to Target workers |
| API stability | Public generation API relatively stable; external rollback is not public | Internal APIs pinned and guarded | Native built-in flow supported; remote control/placement would require adaptation |
| External correction/bonus | Public generate/abort/resubmit does not expose retained-KV commit transaction | Explicit suffix-materialization transaction | Native sampler owns correction and progression |
| Integration complexity | External ownership fights autonomous scheduler; abort/replay would defeat purpose | More input/KV plumbing, bounded to one model/configuration | Would replace current remote control and change device/TP assumptions |
| Dual overlap | Possible across processes, but independent scheduling may obscure ordering | Independent GPU0 execution overlaps unchanged Target world | Colocated native path is not SpecRhythm's disaggregated overlap |
| Cross-GPU concurrency | Yes if separately placed | Yes, no shared world required | Current native constructor requires equal TP; no separate GPU0 worker API |
| Stable request identity | Frontend/core ID mapping plus second scheduler | Explicit one-to-one stable→Draft ID; transient runner row map checked each step | Requires continued internal identity adaptation |
| Performance expectations | Good ordinary serving; external commit might cause replay/control overhead | Real B-row forwards; some Python packing/bulk sync remains in MVP | Good native execution, but not an isolated backend replacement |
| Debugging | Engine/scheduler/worker state across layers | Small explicit state table and one allocator | Interleaves Target/Draft internals and existing patches |
| Decision | Do not use as MVP | **Recommended** | Reject for this experiment |

The native TP mismatch guard and internal API boundaries are documented with pinned sources in [the API map](phase4b3-vllm025-draft-api-map.md). A full second engine is **not required**. Neither MineDraft's shared five-rank world nor its custom non-driver groups are needed.

```text
Existing supervisor / owned-process lifecycle
  ├─ Draft service process: only physical GPU0 visible
  │    existing transport/control endpoint
  │    one CUDA-owner thread
  │      VllmBatchedDraftBackend
  │      UniProcExecutor → WorkerWrapperBase → v1 GPU Worker → MRV1 runner
  │      one Qwen3-0.6B model, one independent TP=PP=DP=1 world
  │      one KVCacheManager + one worker-owned paged cache
  └─ Target EngineCore / workers: physical GPUs1/2, TP2
       Qwen3-32B, existing custom proposer and five-patch stack
```

Use the worker/cache initialization recipe in the API map. The Draft service's existing transport remains the interprocess control/data transport for the small token/proposal payloads. `UniProcExecutor.collective_rpc` is local within this process and is not a replacement network protocol. Initialize, execute and tear down the Draft worker on the same owner thread; leave HTTP/IPC responsiveness on the existing service side. Do not inherit Target distributed rendezvous/rank settings accidentally.

## 3. Scope and construction restrictions for Plan A

The initial opt-in backend selector is proposed as configuration `draft_backend: hf-persistent | vllm-batched`, defaulting to the existing HF path. This name is a **proposed interface**, not an available run command. Freeze its value in the plan/runtime report. No change to `SR_PHASE4_DUAL_UUID_QUERY_MODE`, proposal budget, model pair, candidate policy or Target configuration belongs in that implementation.

Plan A supports the present dense text-only Qwen3-0.6B Draft, frozen tokenizer/model revision and dtype, deterministic greedy selection, one visible CUDA device, TP/PP/DP=1, MRV1 and eager execution. Use an ordinary model (`speculative_config=None`) and private full-attention KV with prefix caching disabled. Reject unsupported features during construction: MRV2, hybrid/Mamba or sliding-window layouts, multimodal/encoder inputs, pooling, LoRA, prompt embeddings/adapters, random/logit-modified sampling, structured output, asynchronous scheduling, DBO, KV/EC connectors, EPLB/MoE/routed-expert output, speculative Draft-on-Draft and Target diagnostics inside the Draft worker.

These are bounded implementation constraints, not changes to the selected workload's semantics. If the frozen configuration needs one of them, D1 is blocked until explicitly supported; do not substitute a different configuration silently. Configuration/model max length must accommodate the frozen prompt/output budget and bounded lookahead. If memory profiling auto-fits it lower than that requirement, fail construction. Model load/profiling/warm-up are outside the current measured boundary, with separate counters. Initial request materialization may cache logits as HF does, but must not pre-generate deferred proposals before the existing boundary.

## 4. Concrete ownership and structures

All structures below are proposed SpecRhythm types. The scheduler continues to see existing proposal/evidence objects, not vLLM cache internals.

| Structure | Fields | Sole authority / lifetime |
| --- | --- | --- |
| Existing control request state | stable request ID, canonical committed token IDs/hash, prefix version, next round, output budget, stop policy, proposal lifecycle/terminal state | Existing `DraftStateMachine` / `DualDraftMachine`; the backend receives immutable validated views |
| `DraftRequestHandle` | `stable_id`, unique `draft_request_id`, service generation, current prefix version/length reference, `materialized_length M`, `kv_epoch`, pending proposal reference, owned `next_logits` or valid-logit marker, terminal/released flags | Backend physical/materialization state; no independent acceptance decision |
| `DraftCommitPlan` | stable ID, proposal ID/round, old prefix hash/version, accepted count `a`, canonical Target tail `T`, full expected new prefix/hash, terminal flag | Immutable result of existing control validation, before any GPU mutation |
| `DraftBatchPlan` | monotonically increasing batch ID, ordered stable/Draft IDs, per-row budgets, context snapshots, per-row suffix start/count, purpose (`setup`, `proposal`, `commit`), capacity bounds | Backend planner for one transaction/step; preserves caller order |
| `DraftStepState` | GPU proposed IDs `[B,k_max]`, actual lengths, active mask, per-step batch-row→Draft-ID map; CPU bulk token view in MVP | Backend for one proposal batch, released after immutable proposal construction |
| vLLM `Request` allocation view | Draft ID, immutable current input context, `num_computed_tokens`, sampling metadata, status | Short-lived derived view for allocator; never a second request lifecycle authority |
| KV ownership | Draft ID→per-group allocated blocks, free pool, block capacity | One `KVCacheManager`; keyed by Draft ID across derived views |
| vLLM cached worker state | Draft ID→input context, computed frontier, block IDs; transient InputBatch indices | Worker projection of the current plan, rebuilt/checked per execution |
| `DraftBatchExecutionEvidence` | batch/step IDs, ordered row IDs, query counts, materialized frontiers, block digest/generation, timing reference | Backend result referenced by per-request proposal evidence |

Do not derive an identity from a row number or GPU rank. Allocate an internal ID once per resident request lifetime and maintain a checked bijection. Reject duplicate stable IDs in a batch, stale service generation, mismatched prefix version/hash, multiple unresolved proposals and use-after-finish. Physical row order may change after packing/compaction; map model logits through the actual worker InputBatch IDs immediately after execution. The Draft row map is separate from the existing Target sampled-output-copy domain.

To avoid private mutation of `vllm.Request._all_token_ids`, build an ordinary `Request` view from the exact input context when allocating/materializing. Set its computed frontier explicitly; retain the same Draft ID. `KVCacheManager` keeps ownership by that ID, as shown by `allocate_slots`/`free` in the pinned source. With prefix caching disabled and full attention, speculative input snapshots cannot publish invalid shared prefixes. Check request status deliberately (initial admission versus running), rather than inheriting default WAITING behavior on every view. The backend/control prefix remains the canonical source; snapshots are disposable projections.

## 5. Materialization primitive and pinned adapter

Proposed primitive:

```text
materialize_batch([(handle, context_tokens, valid_prefix_length, suffix_tokens)], purpose)
    → owned last-position logits by Draft ID
    → actual materialized lengths and execution evidence
```

For each row, `context_tokens = valid_prefix + suffix`; its valid prefix already has correct KV. Only the suffix is forwarded. Prefix metadata may be copied on CPU; the prefix must not be recomputed on GPU. Allocate needed pages with `KVCacheManager.allocate_slots(request_view, num_new_tokens=len(suffix))`, obtain full group block IDs using `get_block_ids`, and stop on insufficient capacity. Reserve/prove capacity for the frozen resident cohort and output limits before measurement; do not introduce runtime eviction/replay to make an overcommitted cohort fit.

Create `NewRequestData` for each materialization row, with the full context, those allocated block IDs and `num_computed_tokens=valid_prefix_length`. For an existing Draft ID, the pinned runner invokes `_update_streaming_request` and replaces its context without a fake finish. Use `SchedulerOutput` with:

* `scheduled_new_reqs` = these new/rebased input views; `scheduled_cached_reqs=CachedRequestData.make_empty()`;
* `num_scheduled_tokens[id]=len(suffix)` and total equal to the sum;
* no scheduled speculative tokens, encoders, connectors, structured sampling or common shared prefixes;
* real finished IDs only, including on the final zero-work cleanup output;
* all other required fields initialized to their correct empty/default values at this exact version, preserving KV group dimensions and any allocator-provided initialization metadata.

For homogeneous one-token suffixes this is B tokens over B requests in **one** forward. Ragged commit rows can contain one or two tokens each; ordinary causal attention computes their full suffix in one packed forward, selecting last-position logits per request. For setup, prefill chunking may divide a large frozen batch at the configured token bound; never relabel a partial prefill as complete. MVP decode/commit cohorts that exceed configured capacity fail qualification rather than adding an unreported scheduling policy.

### Completing an execute without inventing a vLLM API

`DraftOnlyForwardAdapter` is a proposed project-local worker extension. A uniquely named method such as `sr_draft_materialize` invokes Worker execution and completes the pending `ExecuteModelState` **instead of** stock `sample_tokens`. This deliberate private adaptation is necessary because setup/commit must not append an unsolicited sampled output to the worker's history. The exact existing seams are listed in the API map.

Its completion contract is:

1. Assert no pending execute state; validate all unsupported features are absent and that the request batch has no speculative sampling metadata.
2. Execute the ordinary Worker input/model path. Empty cleanup work must return the expected empty output. Nonempty supported work must produce the expected pending state and one last-position logit row per actual request.
3. Capture the **post-preparation** row IDs and logit row mapping. Reject missing/extra/duplicate rows. Transfer ownership of valid logits (clone if storage may be reused) before clearing the pending tuple; release hidden-state references no longer needed.
4. Clear exactly the pending execution/completion state. Assert no KV/EC connector output requires deferred processing; clear its empty holder. Check that native drafter/async previous-sampled state is absent. There is no hybrid acceptance correction or EPLB step in the admitted configuration. Do not simply ignore an unfamiliar pending field after a source change.
5. Record completion on the owner's CUDA stream; acknowledge materialization only after the required completion fence. Do not call stock sampling afterward, mutate its output token arrays, or leave a pending state for the next execute.

Use vLLM's current config context in the owner thread and preserve wrapper initialization/forward context. Worker extension methods may not override existing Worker names. Neither `execute_model_state` nor the streaming rebase path is a stable public external-KV API; validate exact source/runtime shapes and test both seams before performance work. No new vLLM source patch is planned for Plan A. Calling an accessor and leaving the stock completion state outstanding is **not** this design.

## 6. Exact KV state transition algorithm

Definitions for one request:

* `C`: control-plane canonical committed tokens, length `L`.
* `M`: number of contiguous positions with valid materialized Draft KV for the current context. Allocated capacity `K ≥ M` is separate.
* `P`: pending proposal, actual length `p`, already truncated by existing Draft EOS/budget policy.
* `a`: accepted prefix length of P after **existing logical canonicalization**, `0 ≤ a ≤ p`.
* `T`: existing canonical correction or bonus token sequence, normally length zero or one. Backend does not infer it from sampler capacity rows.
* `C' = C + P[:a] + T`: required new canonical prefix, verified against the supplied hash/version and output limit.

### Initialize

Allocate private blocks and materialize the supplied committed prefix C, including the current bootstrap convention. Finish with `M=L` and owned next logits. Publish the existing initialization/DecodeReady evidence only after this work completes. Do not create a speculative token or proposal as part of initialization.

### Propose, all rows together

Require `M=L`, a valid next-logit value, and no unresolved proposal. Preserve the current budget `min(candidate_budget, max(remaining_output_tokens - 1, 0))` and existing Target-tail behavior.

For step j, greedily select one token for all eligible rows from their current next logits using one batched argmax. Bulk-transfer the selected vector once in Plan A to establish token IDs, EOS and remaining masks. Record it in P for each row. Remove rows that reached EOS or their individual budget. For continuing rows only, materialize the selected token with one B_active-row forward and obtain next logits. Repeat. There is no per-request `.item()` loop. If p>0, after proposal `M=L+p-1`; the final sampled token has no KV yet. If p=0, `M=L` and no forward or proposal is fabricated.

This preserves the HF convention for proposal work while using true model batches. The existing evidence field describing logical proposal extent may remain `L+p`; it must not be used as the actual materialized frontier. Add actual `M` separately.

### Commit/rollback, all validated rows together

1. Validate all plans before mutating any row: stable/internal IDs, proposal ID, round, parent prefix/hash, accepted-prefix equality, canonical T, version increment, terminal and output budget. Use existing acceptance/stop functions, not a second implementation in the backend.
2. For each row compute `R = min(M_old, L+a)`. Positions below R already match C'. Positions at/after R are invalid until rewritten, regardless of the values still present in allocated pages. Retain private block capacity; no tensor crop or guessed block free.
3. Set the next worker materialization view to C' with valid prefix R. Its suffix is `S = C'[R:]`. Stage every nonempty S into **one ragged commit batch**, subject to the proven capacity bounds. Materialize these suffixes; no full-prefix model replay occurs.
4. If S is empty and the request continues, ensure next logits actually correspond to C'. If the owned logits are from a different speculative frontier, replay only the final token at position `len(C')-1` using valid prefix `len(C')-1` to refresh them. Count this explicit one-token refresh. Do not reuse logits after a rejected suffix. The normal correction/bonus cases have a nonempty S and need no such refresh.
5. After the commit batch's completion fence, assert `M=len(C')`. Publish the existing logical length/hash/version acknowledgement, clear the pending proposal, and only then allow the next proposal. A terminal row is synchronized under the same contract, then freed once, with no sampling or new round.
6. On GPU failure after mutation begins, fail the service/run closed and use existing owned-process cleanup. Do not partially acknowledge a cohort or resume a request with an unknown frontier. Validation/capacity failures before launch publish no successful commit.

| Case | R / suffix materialized | Result |
| --- | --- | --- |
| First rejection, p>0, a=0, correction x | `R=L`; S=`[x]` | Original committed KV retained; rejected suffix overwritten at L |
| Partial acceptance, 0<a<p, correction x | `R=L+a`; S=`[x]` | Accepted KV retained; correction replaces first rejected position |
| All accepted, a=p>0, bonus b | `R=L+p-1`; S=`[P[-1], b]` | Last proposal and bonus written in one two-token query span |
| All accepted, no Target tail | S=`[P[-1]]` if p>0 | Last accepted token materialized; terminal/output-limit logic still controls continuation |
| Accepted EOS within canonical accepted prefix | Use canonical a and T after truncation; materialize only missing accepted suffix | EOS is last logical token, no post-EOS proposal, then release |
| Target correction/bonus is EOS | Materialize normal suffix ending with that EOS | Publish terminal committed state, release, do not sample a next token |
| No proposals / Target-only tail | `R=L`; S=canonical Target tail | No negative frontier or synthetic proposal ID |
| Empty S, nonterminal rollback-only call | Reuse only correctly tagged next logits; otherwise refresh last committed token | No stale rejected-context logits |
| Duplicate commit / stale round | Reject before GPU work | No duplicate append or double free |

Example: `L=20, p=4` leaves `M=23`. Accepting two and adding correction gives `C'` length 23, `R=22`, one-token rewrite at 22. Accepting all four plus bonus gives length 25, `R=23`, two-token materialization at 23/24. Merely setting M to 25 would be a false synchronization acknowledgement.

Native vLLM implements the same **logical frontier plus rejected-tail overwrite** principle in scheduler acceptance debit, base-proposer compaction and slot mapping. It normally tolerates a final unmaterialized token. This design explicitly flushes that tail to preserve SpecRhythm's current post-commit and bootstrap materialization contract; it does not copy the native one-token lag into DecodeReady evidence.

## 7. Stage 1 — Serial integration

The existing Serial `synchronize_and_batch_propose` wire operation already supplies a set of synchronization rows and a set of proposal rows. Preserve its request selection, order, round IDs and Target verification path.

Add a separate optional batched-backend capability (`initialize_many`, `materialize_commits`, `propose_many`, `finish_many` are proposed local methods). Keep the HF scalar backend and its dispatch behavior unchanged for A/B control. In the new path:

1. The state machine validates the complete incoming synchronization cohort with existing rules, constructing immutable `DraftCommitPlan`s.
2. Backend materializes all required commit suffixes as a batch, then the state machine publishes the original per-request commit evidence. This is the needed change to the scalar backend seam, not a new acceptance policy.
3. Validate/collect all requested Draft rows with their existing budgets; call `propose_many` once. Each autoregressive step processes the active cohort. Restore caller order when returning immutable per-request proposals.
4. Target verification proceeds through the current remote custom proposer. Preserve bootstrap deferral, measurement boundary and current strict Serial intervals.
5. Repeated rounds follow the same path; setup/finish remain correct even if initially called with singleton rows. Batched setup may be added through the service capability without generating proposals early; it is not necessary to change Target setup callbacks to establish decode batching.

Do not implement `propose_many` by calling scalar `propose` B times, and do not leave correction/bonus append on B scalar forwards. CPU loops that assemble ragged metadata and contracts are acceptable; model execution and token selection must operate on the cohort.

## 8. Stage 2 — Dual integration after D5

Reuse the same backend and KV algorithm. `AsyncDualDraftController.enqueue` already receives rows together but expands them into individual `_Work` entries. A bounded implementation may preserve the original enqueue **envelope** in the GPU work queue and batch compatible rows inside that envelope. Keep the chosen row order, per-request inflight/claimed/ready checks, proposal lifecycle, retired-ready suppression, and ready publication order. No wait-to-accumulate timer, new window, additional request selection, cross-envelope reordering or microbatch knob belongs in this stage.

Split each machine operation into existing semantic validation, batched physical execution, then original semantic acknowledgement/publication. In particular, `commit_and_propose` may batch validated commit suffixes and then proposals, but a row cannot be ready before its own committed state is materialized. Existing control calls remain responsive; only the owner thread accesses the model/KV.

Batch completion changes latency naturally; do not claim identical timing to the scalar service. Preserve lifecycle/order rules and measure their resulting schedule. Do not release an entire cohort under fabricated identical per-request start times. Shared batch execution events can be referenced by each row with the actual common interval and batch ID. The overlap validator must not sum the same physical interval B times or mistake a queue interval for GPU execution. Existing Target timing, sampled-row mapping, retired-ready handling and UUID query mode are unchanged.

If a legal current enqueue envelope has B=1, run B=1 and report it. Do not manufacture a batch by changing the scheduler. Poor batching under current Dual becomes D6 evidence for a later scheduling experiment, not justification to add accumulation silently.

## 9. Proposal contract and correctness evidence

Map batched device output → actual worker row → Draft ID → stable ID → existing `Proposal`/`DualProposal`. The backend produces tokens and physical evidence; existing state machines assign/validate lifecycle fields and canonical logical semantics.

| Invariant / existing contract | New backend evidence or enforcement |
| --- | --- |
| Stable request identity | Immutable stable↔Draft-ID bijection; service generation; duplicate/missing-ID rejection |
| Request-ID→vLLM row | Actual post-packing InputBatch row map per batch/step; inverse-map equality; row-count bounds |
| Current Target sampled-row domain | Continue `dual_rows.align_sampled_rows` and patch 0005 unchanged; Draft row maps never index Target output arrays |
| `request_id`, `round_id`, `proposal_id`, parent prefix length/hash | Existing control values attached after restoring stable order; checked again before commit |
| Proposal token IDs/EOS | Actual per-row length and existing budget/EOS rules; no capacity padding in public token list |
| No post-EOS round | Terminal validation before batch formation; terminal mask removes row; no next-proposal publication |
| Proposal lifecycle and retired-ready | Existing created/ready/claimed/verified/retired transitions; stale completed batch result cannot revive a retired request |
| Draft/Target prefix agreement | Control C' hash versus worker input-view hash after commit; actual `M=len(C')` after fence |
| DecodeReady/bootstrap | Initial physical materialization count equals supplied logical Draft prefix; preserve Target's distinct bootstrap-materialized convention |
| Logical KV length | Keep current logical fields; add actual materialized frontier and allocator block digest/epoch; allocated capacity is never reported as valid length |
| Matched-work and token accounting | Same frozen request manifest, bootstrap treatment, measured committed tokens, per-request budget, completion/terminal policy; proposed tokens remain a separate counter |
| Exact outputs | HF-versus-vLLM proposal/generation gates where deterministic; no tolerance switch or seed/model substitution on mismatch |
| TP consistency | Draft world=1 actual device/model/UUID evidence; Target per-rank consensus and sampled-row signatures unchanged |
| Physical overlap | GPU event evidence for actual batched Draft work plus existing cross-process host timing/Target evidence; same physical batch counted once |
| Process cleanup | Free all backend handles and allocator blocks once, remove worker request state, drain/fail work, shut down executor under existing owned-process supervisor |
| Model provenance | Existing model/revision/parameter/device evidence plus vLLM source/hash, worker/runner classes, attention/cache layout and backend name |

Existing `draft_physical_request_block_identity` can become observable for the new backend using actual allocator block IDs, group index and allocation epoch; retain HF's unobservable status. Never compare raw block IDs across processes as physical equality. Keep per-request evidence in the existing event/report path; add batch-level references and aggregate performance evidence without changing logging/fsync policy globally.

## 10. Expected launch shape and limits of prediction

Let B requests each produce p tokens and continue without early EOS. Current HF proposal work is `B×(p−1)` model forwards and `B×p` `.item()` synchronizations. Plan A proposal work is `(p−1)` model forwards with B request rows and p **bulk** token-vector transfers/host synchronization points. Selection of token one uses cached logits. With heterogeneous lengths, sum old per-row forwards versus one forward per nonempty step over `B_active`; report the active distribution.

For full accept plus bonus, HF additionally forwards two tokens per request in separate calls: 2B commit forwards. Plan A packs two-token suffixes for B requests into one commit forward if capacity permits. A partial-reject/correction cohort changes B scalar commit forwards into one. CPU metadata/contract construction still contains O(B) loops and may copy O(total context length) token metadata at each rebase in Plan A. This is not a claim of no Python work or no synchronization.

For illustration only, B=32,p=4 gives 96→3 proposal forwards, 128→4 explicit token-transfer synchronization points, and 64→1 full-accept/bonus commit forwards. These are execution-count expectations under stated conditions, **not a predicted makespan**. Ragged numerical behavior, packing overhead, memory, EOS, setup and Target service time require measurement. Do not translate ≈2027 historical HF forwards directly into a promised speedup.

## 11. Aggregate instrumentation

Emit one immutable final `draft-backend-report.json` linked from the existing final report by path/hash. Proposed namespace and fields:

| Field | Definition |
| --- | --- |
| `draft_backend`, `backend_schema_version`, `source_commit`, `source_hashes`, `effective_config_hash` | Exact implementation/config identity; include MRV1 class, dtype, model/tokenizer revisions and device evidence |
| `draft_model_forward_count_total`, `..._setup`, `..._proposal`, `..._commit`, `..._warmup` | Count actual model invocations, not RPCs or proposal rows; measurement-window count separately |
| `draft_forward_request_batch_histogram_by_purpose` | Integer B histogram at each actual forward; report p10/p50/p90 using a documented nearest-rank definition |
| `draft_active_requests_by_step_histogram`, `draft_scheduled_token_count_histogram` | Distinguish logical active requests from packed query tokens and padding; zero-work is separate |
| `draft_proposal_count`, `draft_proposed_token_count`, `draft_proposal_length_histogram` | Logical proposals and actual token counts, excluding padding/setup samples |
| `draft_model_gpu_event_elapsed_ms`, `draft_materialize_gpu_event_elapsed_ms` | Draft-only model pre/post events versus whole materialization events; explicitly distinguish kernel envelope from host wall time |
| `draft_kv_commit_request_count`, `draft_kv_commit_batch_count`, `draft_kv_rollback_request_count`, `draft_kv_invalidated_token_count` | Logical operations and batches; rollback can invalidate zero tokens |
| `draft_correction_token_count`, `draft_bonus_token_count`, `draft_commit_materialized_token_count`, `draft_logit_refresh_forward_count` | Explain catch-up work including unmaterialized last proposals and exceptional zero-suffix refresh |
| `draft_blocks_allocated`, `draft_blocks_freed`, `draft_blocks_peak`, `draft_live_requests_final` | Allocator ownership/cleanup; report block size/groups and capacity separately |
| `draft_step_host_gap_ns_histogram`, `draft_batch_prepare_host_ns_total` | Gap between end of one step and start of next; explicit timebase; do not call host gaps GPU idle without timeline evidence |
| `draft_explicit_scalar_item_count`, `draft_bulk_token_d2h_count`, `draft_explicit_sync_count_by_reason` | Counters at controlled adapter call sites; include commit fences and final report drain separately |
| `draft_internal_sync_count_observed` / `draft_internal_sync_coverage` | Unknown/null unless measured with a separate profiler; never assert all CUDA synchronizations were counted from Python hooks |
| `draft_row_mapping_validation_count`, `draft_prefix_validation_count`, `draft_validation_error_count` | Correctness checks performed, not merely a final boolean |

Use in-memory counters, integer histograms and a bounded event buffer. Eager Draft model pre/post hooks can count and bracket actual model calls; validate that one hook pair corresponds to one invocation and does not include warm-up in measured counters. Record CUDA events on the actual stream and resolve them at existing safe completion points/final drain, not via extra per-step global synchronization. Event elapsed time includes the recorded GPU interval, not a sum of kernel self times. Preserve existing mandatory lifecycle/overlap instrumentation; no per-token fsync stream is added. Label overflow/missing event coverage as incomplete evidence rather than silently extrapolating.

## 12. Two implementation plans and file map

Names below identify expected future additions; none exists because of this task.

| Dimension | Plan A — minimum viable, recommended next | Plan B — production-quality after qualification |
| --- | --- | --- |
| Process | Existing Draft service, one CUDA owner, independent world=1 | Same process/control boundary |
| Model/runner | One ordinary Worker/MRV1 via UniProcExecutor | Same single loaded model/cache; dedicated versioned Draft execution facade |
| KV | Current KVCacheManager, private pages, explicit M/R, ragged suffix materialization | Same logical transaction; persistent row/input buffers, incremental context updates, safe bounded allocation/reuse |
| Batching | Python step loop over B_active; bulk token-vector D2H once per step; batched commit | Device-resident token/length/EOS masks and multi-step buffers; one proposal transfer when ready |
| Native APIs | SchedulerOutput/NewRequestData, internal streaming rebase, ordinary Worker forward, Draft-only completion adapter | Attention metadata builders/CommonAttentionMetadata, current slot-update mechanisms, existing model/cache binding; narrow explicit completion/input-update interface |
| Python costs | Full CPU context snapshots permitted; no GPU full-prefix replay | Avoid repeated full-context metadata copies; compact/restore rows incrementally |
| Patch requirement | **No new vLLM source patch planned**; guarded project-local worker extension accesses private seams | Prefer project-local facade; if an upstream hook is needed, isolate a Draft-only forward-completion/input-rebase method with exact hash verification |
| Risk | Medium/high correctness risk at rebase/completion seams; bounded feature set makes D1–D3 decisive | Higher implementation cost: device masks, buffers, asynchronous lifetime and kernel/layout coupling; qualify separately |
| Immediate value | Prove backend replacement effect quickly in Serial | Reduce residual per-step host overhead after the basic effect is established |

Expected future file changes:

| Path / class | Plan and purpose |
| --- | --- |
| `src/specrhythm/phase4/vllm_draft_backend.py` — `VllmBatchedDraftBackend` | New Plan A backend, model/cache lifecycle, batch proposal and commit orchestration |
| `src/specrhythm/phase4/vllm_draft_worker.py` — `DraftOnlyForwardAdapter` | New Plan A worker extension, construction guards, source-version checks, materialization completion, runtime model/device evidence |
| `src/specrhythm/phase4/draft_batch.py` — plan/handle/row-map types | New CPU-testable plan validation and physical frontier algebra; acceptance remains in existing control functions |
| `src/specrhythm/phase4/draft_backend_metrics.py` | New aggregates and immutable final report schema |
| `src/specrhythm/phase4/draft_service.py` | Optional batched capability dispatch, backend factory, batched commit path; retain HF reference path |
| `src/specrhythm/phase4/config.py` and existing service launch/report plumbing | Default-HF selector and frozen provenance; preserve defaults and measured boundary |
| `src/specrhythm/phase4/dual_service.py` | **D6 only:** preserve enqueue envelopes, batch physical phase, publish original lifecycle results |
| `tests/test_phase4_vllm_draft_*.py` | CPU algebra, fake worker/cache, source/interface guards, reports and lifecycle regression |
| Existing Phase4 Linux CI job | Add focused Draft/source tests without GPU requirement |
| `src/specrhythm/phase4/vllm_draft_runner.py` | Plan B only: persistent GPU batch buffers and incremental Draft input/slot progression facade |

No planned changes to Target/Serial verifier logic, `dual_rows.py`, `dual_commit.py`, retired-ready handling, UUID mode, candidate selection, performance boundary, numerical diagnostics or the five existing vLLM patches. Service/report plumbing changes must be limited to selecting and observing the backend.

Plan B must reuse the worker's already-loaded model and allocated caches. Instantiating `DraftModelProposer` as a second model inside that worker is not a clean shortcut. Native `eagle_step_update_slot_mapping_and_metadata` is useful only with compatible metadata and causal position semantics; it is not permission to copy EAGLE's hidden-state shifts for an ordinary Draft model.

If later implementation demonstrates that a required completion action cannot safely be expressed out of tree, the precise candidate patch location is MRV1 `GPUModelRunner.execute_model`/completion state handling and a corresponding Draft-only Worker extension entry. Add an explicit “materialize without sampling” result/completion hook, default inactive, with all ordinary Target branches unchanged. Document the unsupported obligation that necessitates it, exact pristine/patched hashes and inert-Target tests before accepting it. Do not patch vLLM merely to imitate MineDraft's plugin, loosen current patch verification, or slip a sixth Target behavior change into the backend experiment.

## 13. Qualification and unresolved measurements

The next coding task should implement Plan A and its CPU/source tests, then hand server execution to the user. [The test plan](phase4b3-batched-draft-test-plan.md) defines D1 construction, D2 single-request semantics, D3 B=2/4/8 batch correctness, D4 corrected-5, D5 corrected-100 Serial comparison, and D6 Dual.

D5 asks whether replacing the HF correctness backend materially reduces the approximately 49 s Serial makespan under the frozen workload/configuration, with real Draft batch statistics. It sets no arbitrary 10 s target. HF and vLLM kernels may produce different greedy choices near ties despite deterministic configuration; such a mismatch fails the required equivalence gate and must be reported with the first divergent context. This task does not reopen Gate3 numerical-divergence work or weaken exact validators.

After D5, compare vLLM-batched Serial with vLLM-batched Dual and determine whether the current scheduler hides Draft work. Only a subsequent experiment may vary Dual microbatch size, ready window, accumulation, rhythm policy or shaping. The data plane is a prerequisite for interpreting those policies, not a claim that it alone will fix Dual's small Target batches.

[sr-hf]: https://github.com/rzwang22/SpecRhythm/blob/64031110e630b6643cc610620f05f8630e896c7c/src/specrhythm/phase4/draft_service.py#L335
[sr-dual]: https://github.com/rzwang22/SpecRhythm/blob/64031110e630b6643cc610620f05f8630e896c7c/src/specrhythm/phase4/dual_service.py#L117
