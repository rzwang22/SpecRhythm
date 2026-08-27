# vLLM v0.25.1 source/API audit

This audit is pinned to vLLM tag `v0.25.1`, commit
`752a3a504485790a2e8491cacbb35c137339ad34`. The tag and commit were checked directly from the
repository; the source, not moving documentation, is the basis for the decisions below. vLLM is
not vendored into SpecRhythm.

The freeze requires Python 3.11 for this integration environment. At the pinned commit,
`pyproject.toml` declares Python `>=3.10,<3.15` and build-time PyTorch `2.11.0`. The source
`requirements/common.txt` requires Transformers `>=5.5.3`. The server probe records the actual
CUDA build/runtime, driver, NCCL, Transformers, attention backend, installed-distribution RECORD
checksum and exact source checkout.

## Audit matrix

“Public API” means a documented or directly exposed Python entry point at this exact commit; it
does not promise compatibility with a later vLLM release. “Python patch” means a narrow pinned
fork/change rather than a general plugin.

| Mechanism | Exact source anchor | Public API | General/model/endpoint plugin | Minimum change and kernel impact | Risk and recommendation |
| --- | --- | --- | --- | --- | --- |
| V1 request lifecycle | [`vllm/v1/request.py`: `Request`, `RequestStatus`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/request.py); [`vllm/v1/core/sched/scheduler.py`: `Scheduler.schedule`, `update_from_output`, `add_request`, `_free_request`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/sched/scheduler.py) | Submit/abort/output and `SchedulerConfig.scheduler_cls` are exposed; scheduler internals remain version-pinned | An importable scheduler subclass can add a narrow gate while delegating to stock scheduling | Phase 4B uses a default-off ready gate with no kernel or scheduler-source patch | Medium version-churn risk. Pin v0.25.1 and prove Target-only remains unchanged. |
| Engine core and frontend | [`vllm/v1/engine/core.py`: `EngineCore`, `EngineCoreProc`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/engine/core.py); [`llm_engine.py`: `LLMEngine`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/engine/llm_engine.py); [`entrypoints/llm.py`: `LLM.generate`, `collective_rpc`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/entrypoints/llm.py) | `LLM.generate` and `LLM.collective_rpc` are usable for bring-up | A general plugin can run process initialization; endpoint routers do not expose scheduling | Stock bring-up needs no patch/kernel | Low risk for v0.25.1 bring-up. Use two separately GPU-bound `LLM` instances, never built-in speculative decoding. |
| Draft-model speculative decoding | [`vllm/v1/spec_decode/draft_model.py`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/spec_decode/draft_model.py); [`llm_base_proposer.py`: `SpecDecodeBaseProposer.propose`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/spec_decode/llm_base_proposer.py); [`worker/gpu/spec_decode/speculator.py`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/worker/gpu/spec_decode/speculator.py) | Built-in colocated spec decode is public through engine args | Not a disaggregated engine plugin | Reusing it for two independent engines would require Python runner/scheduler changes; existing kernels remain usable only after layout compatibility is proven | High semantic risk. Do not label built-in speculative decoding as `serial-disaggregated` or `dual-batch`. |
| Custom proposer | [`vllm/v1/spec_decode/custom_class_proposer.py`: `create_custom_proposer`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/spec_decode/custom_class_proposer.py) imports `speculative_config.model`, constructs it with `VllmConfig`, and only checks callable `propose` | Yes, inside the existing speculative pipeline | A general plugin can make a proposer importable; this is not a scheduler/verifier plugin | No kernel for a simple proposer, but no API for an external Target engine or persisted cross-engine proposal | Medium/high risk because the contract is minimal and internal consumers define real shape semantics. Not sufficient for SpecRhythm by itself. |
| Target verification and rejection sampler | [`vllm/v1/sample/rejection_sampler.py`: `RejectionSampler.forward`, `rejection_sample`, greedy/random kernels](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/sample/rejection_sampler.py); modular GPU equivalent under [`vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py) | No public external verification-batch API | No verifier plugin | Prefix-tree packed verification needs a Python model-runner patch and likely attention/input-layout CUDA or Triton work | High risk. Phase 4A defines data contracts only and performs ordinary target generation; it does not claim verification. |
| Scheduler speculative state | [`Scheduler.update_draft_token_ids`, `update_draft_token_ids_in_output`, `update_from_output`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/sched/scheduler.py) attach draft IDs, trim scheduled drafts, and debit rejected computed tokens | Internal mechanics, reachable through a pinned scheduler subclass | `scheduler_cls` can inject a validated ready proposal before delegating | Cross-engine identity/version stays in the adapter; stock scheduler owns tokens/KV after injection | High semantic risk but bounded by fail-closed prefix checks and exact Target-only comparison. |
| KV ownership | [`vllm/v1/core/kv_cache_manager.py`: `KVCacheManager.allocate_slots`, `free`, `get_blocks`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/kv_cache_manager.py); [`block_pool.py`: `BlockPool`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/block_pool.py) | Cache sizing/config is public; per-request ownership is internal | No ownership plugin | A future candidate tree cannot mutate ownership outside scheduler bookkeeping; Python patch first, kernels only if tree layout changes | High correctness risk. Never fabricate block IDs or commit rejected candidate KV. |
| Worker block tables | [`vllm/v1/worker/block_table.py`: `BlockTable`, `MultiGroupBlockTable`, `compute_slot_mapping`, `commit_block_table`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/worker/block_table.py) | Internal | No plugin | Packed-tree slot maps require Python input preparation and probably backend-specific attention support | High kernel/layout coupling. Defer to packed-tree phase. |
| Model runner | [`vllm/v1/worker/gpu/model_runner.py`: `GPUModelRunner.prepare_inputs`, `execute_model`, `sample_tokens`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/worker/gpu/model_runner.py); legacy path [`gpu_model_runner.py`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/worker/gpu_model_runner.py) | Normal inference through `LLM`; runner hooks are internal | Model registry/loader plugins load models, not new batching semantics | Independent stock engines need no patch. Tree verification would patch runner preparation/execution and potentially kernels | High API churn. Freeze `use_v2_model_runner`/runtime evidence before any later patch. |
| Worker RPC and extension | [`vllm/v1/executor/abstract.py`: `Executor.collective_rpc`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/executor/abstract.py); [`entrypoints/llm.py`: `LLM.collective_rpc`, `apply_model`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/entrypoints/llm.py); [`config/parallel.py`: `worker_extension_cls`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/config/parallel.py) | Yes, explicitly intended for small control messages | General plugin/worker extension can register small inspection methods | No kernel. Data-plane candidate transfer should not use control RPC | Medium risk. Phase 4A uses a top-level callable only for rank/parameter/device/memory evidence. |
| Distributed/TP initialization | [`vllm/v1/worker/gpu_worker.py`: `Worker.init_device`, `init_worker_distributed_environment`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/worker/gpu_worker.py); [`vllm/distributed/parallel_state.py`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/distributed/parallel_state.py); multiprocess executor under [`vllm/v1/executor/multiproc_executor.py`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/executor/multiproc_executor.py) | TP size/device visibility are public engine args/environment | Not needed | No patch/kernel for Qwen3 TP1/TP2 | Medium operational risk. Validate every rank, physical/logical mapping, UUID, local shard parameters and CUDA memory. |
| Plugin registration | [`vllm/plugins/__init__.py`: `load_plugins_by_group`, `load_general_plugins`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/plugins/__init__.py); [`model_executor/models/registry.py`: `ModelRegistry.register_model`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/model_executor/models/registry.py); [`model_loader/__init__.py`: `register_model_loader`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/model_executor/model_loader/__init__.py) | General/model-loader/model-registry extension exists | General plugins load in every process. Endpoint/tool-parser hooks affect API presentation, not scheduler execution | Plugins can register imports/types but cannot implement cross-engine scheduling without a Python patch | Medium risk and easy to overstate. Recommended later packaging: general plugin plus a minimal pinned scheduler/runner patch, not endpoint code. |
| Metrics/timestamps | [`vllm/v1/metrics/stats.py`: `RequestStateStats`, `IterationStats`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/metrics/stats.py); [`vllm/v1/engine/output_processor.py`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/engine/output_processor.py); [`entrypoints/generate/base/serving.py`: `build_per_request_timing_metrics`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/entrypoints/generate/base/serving.py) | Request output exposes `metrics`; serving can expose per-request metrics | Stat logger plugin exists; no semantic accounting hook | No kernel. Extra cross-engine events need our monotonic event schema | Low/medium risk. Preserve the timebase distinction: frontend arrival is wall clock; core queue/schedule/token timestamps are monotonic. |
| Attention/CUDA Graph | [`vllm/v1/attention/selector.py`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/attention/selector.py); [`vllm/v1/cudagraph_dispatcher.py`: `CudagraphDispatcher`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/cudagraph_dispatcher.py); `GPUModelRunner.capture_model`/dispatch | Backend/config selection is public; graph keys/layout are internal | Custom ops/backends can be registered but tree layouts must meet backend contracts | Packed-tree work can require Python, attention metadata, Triton/CUDA and new graph-capture keys | High risk. Phase 4A uses `enforce_eager=true` for transparent bring-up and records the selected backend rather than reporting graph performance. |
| vLLM Dual Batch Overlap | [`vllm/config/parallel.py`: `enable_dbo`, decode/prefill thresholds](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/config/parallel.py); [`vllm/v1/worker/ubatch_utils.py`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/worker/ubatch_utils.py); [`ubatching.py`: `UBatchContext`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/worker/ubatching.py) | `enable_dbo` is a public engine option | Not a SpecRhythm plugin | It splits one model-executor batch into microbatches and overlaps compute/communication. No cross-model draft/verify semantics | **Not SpecRhythm Dual-Batch.** It is disabled in Phase 4A. The spec-decode source itself contains DBO-not-implemented guards for EAGLE/extracted hidden states. |

## Integration decision

Phase 4A.0 uses two independent stock vLLM V1 offline engines: Draft TP=1 with physical GPU 0,
and Target TP=2 with physical GPUs 1 and 2. The commands are separate and serialized. Their only
shared object is the already-tokenized five-request workload; no candidate verification or
cross-engine data plane exists yet.

## Phase 4A.1 audit decision

The linear `K<=4` correctness path does not require a new Target kernel. At the pinned commit, the
legacy V1 GPU model runner's `custom_class` proposer returns variable-length per-request token
lists. The scheduler stores those lists as `request.spec_token_ids`; the next scheduled step
materializes candidate positions in the vLLM-managed KV cache, the Target runner executes them as
one speculative batch, `RejectionSampler` applies greedy acceptance, and scheduler bookkeeping
subtracts rejected positions from the logical computed length. This is the reused Target path.

The custom proposer API is insufficient in three narrow ways: the call omits vLLM request IDs,
does not distinguish logical token-row extent from post-forward Target materialization, and has no
observer hook around the Target verification forward. Phase 4A.1 therefore maintains
one pinned Python patch, `integrations/vllm/patches/0001-custom-proposer-request-and-verify-hooks.patch`.
It changes only `vllm/v1/worker/gpu_model_runner.py`, passes existing request IDs and the existing
post-forward materialized count, and invokes optional before/after hooks. At this pinned source,
the proposer runs after `_bookkeeping_sync`: `sampled_token_ids[row]` is the authoritative signal
that this forward sampled a token; `num_tokens_no_spec/token_ids_cpu` is the updated logical row;
and materialized Target positions are the pre-forward computed count plus this scheduler step's
scheduled positions. It exposes no Target logits, does not alter sampling or KV state, and is
inactive for Target-only generation. The exact base and patched file SHA256 values are enforced
by `integrations/vllm/manage_patch.py`; apply/check/restore were exercised on the exact source
commit during Mac development.

The remote Draft side does not use vLLM's colocated DraftModelProposer. It is a separate GPU-0
process with one resident Qwen3-0.6B model and per-request mutable Transformers KV. This avoids
loading the Draft model on Target ranks while preserving cross-round Draft cache semantics. It is
a correctness implementation with recorded per-request microbatches, not a performance claim.

VLLM model runner V2 does not instantiate `custom_class` at this commit, so Phase 4A.1 explicitly
freezes `VLLM_USE_V2_MODEL_RUNNER=0`. This choice is a version-specific compatibility constraint,
not a claim that V1 is generally preferable.

Packed-tree verification remains a separate source gate. It changes position/attention layout
and may require runner, attention-metadata, Triton, or CUDA work. No Phase 4A.1 artifact can be
relabeled as packed-tree, Dual-Batch, Eager, SLO, or performance evidence.

## Phase 4B source audit decision

At pinned commit `752a3a5`, `Scheduler.schedule()` consumes `request.spec_token_ids`, constructs
`scheduled_spec_decode_tokens`, and leaves candidate verification and KV allocation on the stock
path. `EngineCore.post_step()` normally receives proposals from the worker only after a Target
step. A synchronous remote proposer therefore cannot implement Dual-Batch: it blocks Target on
Draft and remains Serial.

The pinned release exposes `SchedulerConfig.scheduler_cls`, but the first GPU construction run
proved that its existing cadence field is not an external-readiness API: `schedule()` increments
`current_step` before checking `next_decode_eligible_step`. The earlier adapter's `current_step+1`
mutation therefore allowed one ordinary Target decode for a request still waiting on Draft.

Phase 4B.0a instead adds a default-off hook in the stock running-request loop, before any token
budget or KV allocation. The out-of-tree scheduler supplies a request-level predicate that allows
setup prefill, a matching unconsumed proposal, or a legal terminal Target tail. A denied request is
skipped with the stock loop's normal `req_index += 1`, so later ready work remains schedulable.
The adapter no longer writes `next_decode_eligible_step`; queue order, allocation, preemption,
paged KV, attention, rejection sampling and internal cadence remain owned by vLLM.

The worker observer remains independent patch `0001`. The explicit scheduler hook is patch
`0002-scheduler-request-admissibility-hook.patch`, applied second. Phase 4B.0b adds
`0003-target-forward-timing-observer.patch` after `0001`; it brackets the existing model forward
with monotonic timestamps and passes them to the already observational diagnostic function.
Restore order is `0003`, `0002`, `0001`. Exact original/intermediate/final SHA256 values are
enforced for both source files by
`integrations/vllm/manage_patch.py`. No patch changes C++/CUDA/Triton, sampler, logits,
attention, model weights or TP partitioning. The hook is absent from the stock scheduler class and
therefore inert unless the explicit Phase-4B subclass implements it. Dual mode remains fail-closed
behind `SR_PHASE4_DUAL_BATCH=1`; default and Target-only behavior are unchanged.

## Phase 4B.1 decode-only source boundary

Phase 4B.1 does not add a fourth vLLM patch and does not alter the sampler, model forward, block
manager, attention backend, C++/CUDA or Triton. It composes the three already-audited hooks with
two out-of-tree classes:

- `DualBatchScheduler` consumes only proposals already published by the persistent GPU-0 service,
  installs them in `request.spec_token_ids`, and delegates the actual batch to the pinned stock
  scheduler. `scheduled_spec_decode_tokens` remains the authoritative consumption evidence.
- `DualBatchRemoteProposer` first performs resident bootstrap observation. Draft initialization
  runs on the Draft worker but generates no proposal. Rank zero creates the immutable
  `DecodeReadyManifest`, performs the Target TP barrier, publishes setup-ready, and only then
  enqueues the first asynchronous Draft proposals.

The scheduler reads setup-ready from an atomic cross-process artifact; no Python object is shared
between EngineCore and TP workers. A bootstrapped request is inadmissible before global readiness,
and remains inadmissible afterward until a matching proposal or legal terminal tail exists. The
test-only `one-ready` mode caps only the first publication cycle, while `two-ready` polls queue
metadata without blocking on Draft GPU work. They construct Gate 1 Cases A and B respectively,
are recorded in run metadata, and are disabled in the production/default `none` path.

The diagnostic observer now carries the proposal ID plus logical pending-token fields. These are
observations of the existing pinned Target input and do not change it. Non-Oracle Draft messages
still contain no Target logits, frozen reference output, acceptance labels or future tokens.

Target process ownership is separate from scheduler semantics. Phase 4B.0a launches the Target
coordinator with Python `start_new_session=True`, records its PID/PGID/session and all observed
members, propagates the coordinator exit code, and checks the complete owned group. A wrapper that
exits while an EngineCore/worker descendant remains is a failed run. Cleanup signals only the
recorded PGID (TERM, then bounded KILL) and leaves an active-run guard if cleanup validation fails.
