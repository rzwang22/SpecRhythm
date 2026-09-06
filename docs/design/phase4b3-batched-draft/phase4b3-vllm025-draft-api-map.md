# Phase 4B.3 — pinned vLLM 0.25.1 Draft API map

This map targets **`752a3a504485790a2e8491cacbb35c137339ad34`**, not current vLLM main. It distinguishes existing source interfaces from proposed SpecRhythm adapters. All new names in the design are proposals, not APIs claimed to exist in vLLM.

## 1. Architecture changed from V0 to V1

The old `vllm/spec_decode` Worker arrangement is absent from the pinned tree. V1 `EngineCore` owns an executor and scheduler; the scheduler sends `SchedulerOutput` containing new/cached requests, scheduled token counts, block IDs and speculative IDs to workers. The GPU runner prepares packed input and attention metadata, executes the model, then completes sampling/bookkeeping. Native speculative proposers are integrated into that runner's sampling path, with scheduler acceptance accounting on the return path.

Two runner implementations coexist: `vllm/v1/worker/gpu_model_runner.py` (MRV1) and `vllm/v1/worker/gpu/model_runner.py` (MRV2). Both belong to the V1 engine architecture. SpecRhythm's existing custom proposer/five-patch integration selects **MRV1, `VLLM_USE_V2_MODEL_RUNNER=0`**. The proposed first Draft adapter must select MRV1 explicitly as well. An import search finding a second `GPUModelRunner` does not establish API compatibility.

## 2. Migration table

“Absent” below means the class definition was not found by a whole pinned `vllm/` source search; it is not merely absent from one anticipated filename. File hashes and AST findings are recorded in [source-provenance.json](source-provenance.json).

| MineDraft / vLLM 0.9.2 API | vLLM 0.25.1 status | 0.25.1 equivalent | Required adaptation |
| --- | --- | --- | --- |
| `SpecDecodeWorker` | Old class absent | [MRV1 execution/sampling][runner] + [V1 Scheduler][sched] + rejection sampler | Keep SpecRhythm Target verification/control; do not recreate a second SpecDecodeWorker |
| `MultiStepWorker` | Absent | [SpecDecodeBaseProposer.propose][base], or explicit repeated Draft runner execution | Port the step-over-batch loop, not V0 `ExecuteModelRequest` plumbing |
| `TP1DraftModelRunner` | Absent | MRV1 input preparation; base proposer's GPU position/slot metadata updates | Draft-only adapter over one loaded model; no old runner import |
| `SmallerTpProposerWorker` | Absent | No equivalent for this separate-device topology | Independent Draft TP=PP=DP=1 world; do not join Target TP2 |
| `Top1Proposer.get_spec_proposals` | Old class absent | `DraftModelProposer.propose` / custom proposer `propose` | New local batched backend protocol mapped to SpecRhythm proposals |
| `SpeculativeProposals` with token/probability tensors | Old class absent | [DraftTokenIds][outputs], runner's device draft-ID tensors, [SpecDecodeMetadata][metadata] | Return token IDs/lengths by stable ID; no need for dense B×k×V probability output in current greedy contract |
| V0 `ExecuteModelRequest` / `SequenceGroupMetadata` | Old worker protocol removed | [SchedulerOutput, NewRequestData, CachedRequestData][output] | Construct packed ragged scheduled-token spans and authoritative worker context snapshots |
| V0 `SpeculativeConfig` in monolithic config | Moved and substantially expanded | [vllm/config/speculative.py: SpeculativeConfig][config] | Independent Draft worker is an ordinary model with `speculative_config=None`; Target retains current custom-class config |
| V0 `WorkerBase` / CUDA `Worker` | V0 module path removed | [v1/worker/worker_base.py][workerbase], [v1/worker/gpu_worker.py: Worker][worker] | Use current constructor and wrapper; it no longer accepts the old `model_runner_cls` factory argument |
| V0 worker construction wrapper | Replaced | `WorkerWrapperBase`, [UniProcExecutor][executor] | Qualified worker class string, environment/config context, separate rendezvous; no full engine required |
| V0 model runner `prepare_model_input` / `execute_model(..., num_steps)` | Not the current signature | MRV1 `_update_states`, `_prepare_inputs`, `execute_model(SchedulerOutput)` | One materialization batch per call in Plan A; native input/attention preparation retained |
| V0 scheduler `schedule` + sequence groups | Replaced | `Scheduler.schedule`, `update_from_output`, `update_draft_token_ids` | Draft service's deterministic batch planner supplies inputs; no second autonomous admission scheduler |
| V0 block manager / `CacheEngine` | Replaced | [KVCacheManager][kv], per-group cache configs, Worker cache initialization | One Draft allocator owns blocks; worker owns storage; disable prefix sharing in MVP |
| V0 per-sequence block tables | New multigroup representation | [BlockTable / MultiGroupBlockTable][blocks], `NewRequestData.block_ids: tuple[list[int], ...]` | Preserve group dimension, allocation provenance, and transient worker row map |
| Accepted/rejected tokens in V0 speculative worker | Split across V1 components | Scheduler acceptance debit; base proposer input compaction/slot mapping; runner cached-state update | Port logical-frontier model, not private Target state mutation |
| MineDraft asynchronous scorer | No corresponding disaggregated public scorer API | Target engine's existing executor/runner path | Leave current SpecRhythm Target integration unchanged |
| V0 TP override context | Modern distributed module exists, old wrapper absent | Worker distributed initialization under independent world=1 | No global group monkey patch required |
| SpecRhythm `custom_class` proposer hooks | Stock factory exists; identity/verification additions are out-of-tree patches | [create_custom_proposer][custom] + existing five patches | Leave factory, Target hooks, sampled-row domains and TP consensus intact; replace only Draft service implementation |

## 3. Native 0.25.1 Draft KV progression — source, not a guessed API

[`DraftModelProposer`][draft] inherits `SpecDecodeBaseProposer` with `pass_hidden_states_to_model=False`. It loads a Draft model as part of the Target worker's speculative path. It does not share Target embeddings/lm_head for this ordinary Draft-model path. `_raise_if_draft_tp_mismatch`, lines 63–79, **requires Draft TP = Target TP**. The guard is unconditional at construction, even though its comment discusses compile-cache collisions. Turning eager mode on does not make TP1/TP2 pass it.

Consequently this class cannot simply be installed into SpecRhythm's Target TP2 worker to obtain a separate resident Draft TP1 model on GPU0. Nor should a Draft worker load its own ordinary model and then instantiate a native Draft proposer that loads a second copy.

The native correctness mechanism spans these functions:

| Source anchor | What it actually does | Consequence for a separate Draft service |
| --- | --- | --- |
| `Scheduler.update_from_output`, lines 1583–1612 in [scheduler.py][sched] | Derives accepted/rejected speculative lengths from scheduled draft count and returned sampled tokens, then debits `request.num_computed_tokens` and async placeholders for rejections | Valid frontier follows accepted work, not all physically written speculative KV |
| MRV1 `_update_states`, lines 1290–1476 in [gpu_model_runner.py][runner] | Applies scheduler computed counts/block updates to cached worker requests; handles persistent row membership | Worker row order is not an external stable-ID order |
| `GPUModelRunner.propose_draft_token_ids` in [runner][runner] | Selects native/custom proposer branch and supplies sampled tokens/attention state | The custom remote path is different from native Draft-model KV ownership |
| `prepare_next_token_ids_cpu`, lines 1017–1048 in [base proposer][base] | Uses the last valid sampled token, or the defined existing-context fallback | Correction/bonus is an explicit input; it is not fabricated from Draft rank/state |
| `prepare_next_token_ids_padded`, lines 1050 onward | GPU valid-token count and next-token extraction | Avoids copying per-request sampled tokens merely to select the next input |
| `prepare_inputs`, lines 1164–1268 | Computes rejected count as `k + 1 - len(sampled)` for speculative rows; compacts query token indices and reduces sequence lengths | Rejected query positions are excluded from the next Draft input |
| `prepare_inputs_padded`, lines 1102 onward | Builds the corresponding padded/GPU metadata and rejected counts | Layout must match the chosen attention path; CPU and GPU branches are not interchangeable metadata formats |
| `set_inputs_first_pass`, lines 821–960 | For ordinary Draft, reserves extra input slots and copies valid context plus the new sampled token; uses reject masks and slot mapping | Ordinary Draft catch-up is not the hidden-state-shift shortcut used by EAGLE |
| `propose`, lines 502–766 | Runs first-pass materialization and subsequent batched autoregressive steps; adjusts lengths for rejections on the padded path | Native batched progression is real, but coupled to Target attention/request metadata |
| `_update_positions_dependent_metadata`, lines 769–819 | Calls `eagle_step_update_slot_mapping_and_metadata` for stepwise positions, lengths and paged slots | Reusable mechanism after compatible metadata is established, not a public rollback handle |

`KVCacheManager.allocate_slots` distinguishes new tokens and lookahead and caps published prefix-cache blocks at finalized request length. Native rejection does not require a tensor crop of all layer caches. For the proposed dense, causal, private-cache path, shortening the valid frontier and overwriting invalid suffix positions is the analogous mechanism. Hybrid/Mamba state has additional acceptance handling in `_update_states_after_model_execute`; it is outside the Qwen3 MVP and must be rejected rather than silently treated as paged attention only.

## 4. Existing SpecRhythm adaptation that stays in place

The [current five-patch manifest](../../../integrations/vllm/patches/README.md) documents:

| Patch | Existing purpose | This task / first backend implementation |
| --- | --- | --- |
| 0001 | Custom proposer request IDs, actual Target-materialized counts, verification hooks | Unchanged |
| 0002 | Default-off scheduler admissibility predicate | Unchanged |
| 0003 | Target forward host timing observer | Unchanged |
| 0004 | Default-off numerical diagnostic observer | Unchanged |
| 0005 | Dual sampled-output-copy IDs/map alongside physical InputBatch and verification domains | Unchanged |

The stock custom proposer factory checks only that a configured class has callable `propose`; real return-shape and lifecycle requirements reside in its consumers. It does not expose an independent Draft KV transaction. Current `vllm_remote.py`/`vllm_dual.py` remain Target-side bridges to the Draft service. Stable identity and accepted-token evidence must continue through these bridges.

The audited pristine MRV1 source is the architectural baseline. A future Draft adapter running in the shared installed package must additionally validate the **existing patched runner hash**, not falsely demand the pristine runner hash from an installation already required to carry the five patches. Those hooks are inert for a Draft worker with `speculative_config=None` and no Target diagnostics enabled. The expected patched runner is `2905189397b1517659e6606f5bc36c7ca226330f42255c579207fe38f61f9e19`; scheduler is `ffaefd61869589f086e6acdf9a0c4f55f80d5dad145ca3f6fff2379f7a4e2455`. Use the repository patch manager to verify them; do not extend its accepted hashes casually.

## 5. Worker-without-engine construction recipe

These are source-supported building blocks, not a public one-call Draft factory:

1. Construct an ordinary Draft `VllmConfig` through `EngineArgs.create_engine_config` using the frozen model/revision/tokenizer/dtype and single visible GPU. Set TP/PP/DP=1, MRV1, eager, no speculative proposer, no prefix sharing, no asynchronous scheduler/DBO/connectors. Do this before worker initialization/environment caching.
2. Construct `UniProcExecutor(vllm_config)`. [`_init_executor`, lines 46–71][executor], builds `WorkerWrapperBase`, initializes the device and loads one model. Its `_distributed_args` uses a fresh rendezvous and rank zero. Its `collective_rpc` is a **local call through the wrapper**, not a new network data plane.
3. Follow the actual [`EngineCore._initialize_kv_caches`, lines 240–321][core], recipe without constructing EngineCore: register KV specs; get worker KV specs; profile available memory; call `get_kv_cache_configs`; derive `generate_scheduler_kv_cache_config`; propagate any fitted max length; set block counts/sizes and validate; call executor `initialize_from_config` (cache initialization and warm-up).
4. Construct `KVCacheManager` from that scheduler KV config with the matching scheduler/hash block sizes and model limits, `enable_caching=False`, and no speculative/EAGLE-specific allocator mode. For dense Qwen full attention, preserve group structure even if there is only one group. Refuse incompatible hybrid/sliding-window layouts in Plan A.
5. Use `KVCacheManager.allocate_slots`, `get_block_ids`, and `free` as the sole block ownership authority. The worker initializes actual storage with `get_kv_cache_spec` / `initialize_from_config`; the planner does not invent block IDs.
6. Bind all construction/execution/teardown to the service's single CUDA-owner thread. Do not create a second engine, model copy, scheduler thread or Target communication group.

Profiling/warm-up will execute GPU in the **future D1 test**, outside the measurement window. No such calls were run in this task.

## 6. Two non-public seams the MVP must handle explicitly

### External canonical token rebase

At TP1/PP1, `CachedRequestData.new_token_ids` is **not** a general external token-injection API: its declaration says it is used for PP, and MRV1 follows that distinction. Writing a Target correction there would leave the normal last-PP-rank worker token state stale.

There is an existing pinned internal path: `_update_states` sees an existing ID in `scheduled_new_reqs` and calls [`_update_streaming_request`, lines 1580–1609][runner]. That method replaces prompt/context token IDs, block IDs and `num_computed_tokens`, clears worker output IDs and re-adds the row. Plan A deliberately uses it to rebase a Draft-only materialization view with the **same internal ID and allocated blocks**. This repurposes a streaming-context path; it is not an advertised arbitrary rollback API. Tests must cover shortening, retained blocks, reordering and the no-prefix-cache restrictions. Do not pretend to finish/abort a request to achieve this rebase.

### Forward completion without unsolicited sampling

Stock `Worker.execute_model` returning `None` requires `sample_tokens()` before another execute. MRV1 explicitly enforces this at lines 4070–4079. It computes selected-row logits around line 4389 and stores an `ExecuteModelState` tuple at lines 4415–4435. `sample_tokens` consumes that tuple, clears it, samples and then performs ordinary output bookkeeping.

SpecRhythm's setup/commit materialization needs logits without an unsolicited new logical token or proposal. Plan A therefore proposes a **project-local Draft-only completion adapter**, reached through a uniquely named worker extension method, that:

* checks the pinned source/runtime shape and restricted configuration;
* calls ordinary Worker execution, captures the actual current row map and owned last-row logits;
* completes/clears the pending execute state exactly once without the stock sampler's token append;
* handles the corresponding empty connector/output state and rejects any unsupported feature requiring additional completion work;
* lets the backend perform batched greedy selection only when a proposal is requested.

This is an intentional replacement of the stock completion step for a restricted worker. Merely reading `execute_model_state.logits` and calling execute again would violate the contract. A worker extension can access the existing state without editing vLLM source; extension methods must not collide with Worker method names because `WorkerWrapperBase.init_worker` checks this. No new vLLM patch is presently shown to be unavoidable. The interface is version-sensitive and is the highest-risk seam to prove in D1–D3. The [design](phase4b3-batched-draft-design.md) specifies restrictions, state ownership and a longer-term completion interface.

[runner]: https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/worker/gpu_model_runner.py
[sched]: https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/sched/scheduler.py
[base]: https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/spec_decode/llm_base_proposer.py
[draft]: https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/spec_decode/draft_model.py
[outputs]: https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/outputs.py#L310
[metadata]: https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/spec_decode/metadata.py
[config]: https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/config/speculative.py
[workerbase]: https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/worker/worker_base.py
[worker]: https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/worker/gpu_worker.py
[executor]: https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/executor/uniproc_executor.py#L45
[output]: https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/sched/output.py
[core]: https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/engine/core.py#L240
[kv]: https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/kv_cache_manager.py
[blocks]: https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/worker/block_table.py
[custom]: https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/spec_decode/custom_class_proposer.py
