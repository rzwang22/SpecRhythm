# Phase 4B.3 Plan A implementation

This implements **Serial + VllmBatchedDraftBackend** on baseline `64031110e630b6643cc610620f05f8630e896c7c`. The authoritative [design directory](design/phase4b3-batched-draft/README.md) is retained verbatim. GPU execution, numerical equivalence, and speedup are **not validated** by this implementation task. The operator must run [D1–D5](phase4b3-plan-a-runbook.md). D6/Dual integration is a separate task.

## Selection and isolation

`SR_PHASE4_DRAFT_BACKEND=hf-persistent|vllm-batched` selects the backend at Draft service startup. The default is `hf-persistent`; unknown values fail before loading a model. The existing HF backend and Target code remain unchanged. Selecting `vllm-batched` for Dual fails before model loading. No fallback to HF occurs after a vLLM failure.

The new backend owns one `UniProcExecutor`, one `Worker`, one MRV1 `GPUModelRunner`, and one Qwen3-0.6B model in the existing separate GPU0 process. It does not construct an LLM/AsyncLLMEngine or use native speculative decoding. Native proposers' shared Target/Draft TP assumptions do not fit this independent Draft world1 / Target TP2 architecture.

Construction admits Python 3.11, the configured PyTorch 2.11.0, pinned vLLM 0.25.1, MRV1, eager dense Qwen3-0.6B full attention, TP/PP/DP1, private prefix-cache-disabled KV, and synchronous execution. It rejects inherited distributed ranks, existing process groups, wrong model architecture/path, wrong parameter device/dtype, sliding-window/hybrid/Mamba, connectors, LoRA, auxiliary output, routed experts, native speculation, and incompatible runner completion state. The real `active_cuda_device_identity` UUID/device query must validate physical GPU0; rank numbers never supply a UUID. World/rank and parameter residency are checked after construction. Source guards run before importing GPU frameworks.

`draft-startup.json` is written exclusively after successful validation and before service readiness. It records actual model/tokenizer manifests, parameter count/bytes/dtype, GPU UUID/binding, world/rank, worker/runner classes, KV capacity/groups/block size, source identities and startup profiling/warm-up evidence. Existing startup/report/socket paths are rejected. Normal RPC shutdown emits one exclusive final `draft-backend-report.json`; the existing Serial raw artifact binds its SHA256 through `draft_shutdown`. Failed service startup/serving also cleans up the worker and records failure. Reports never claim GPU performance qualification automatically.

## Pinned internal APIs

Exact pin: vLLM `752a3a504485790a2e8491cacbb35c137339ad34` (0.25.1). [vllm_draft_api.json](../src/specrhythm/phase4/vllm_draft_api.json) is the packaged machine-readable inventory: **25 source files**, exact base and required installed SHA256, AST class fields and method signatures, source lines, and runtime guard descriptions. The installed runner and scheduler hashes require the already qualified five-patch state. Every inventoried file is checked at runtime; any drift fails closed. CPU source tests compare all listed signatures and apply the existing patches only to a temporary source fixture.

| Source / interface | Use and runtime assertion |
| --- | --- |
| `engine/arg_utils.py`: `EngineArgs.create_engine_config`; `config/{__init__,vllm,model,parallel,scheduler,cache}.py` | Explicit isolated eager config; validate effective topology, cache, scheduling and unsupported features; scoped `set_current_vllm_config`; `validate_block_size` |
| `v1/executor/{abstract,uniproc_executor}.py`: `UniProcExecutor`, `collective_rpc`, `driver_worker` | Local one-worker execution; exact imported Worker source path and MRV1 class; one actual forward per nonempty materialization |
| `v1/worker/worker_base.py`: `WorkerWrapperBase` extension mechanism | Distinct `DraftOnlyForwardAdapter.sr_draft_materialize`; no override of stock Worker methods |
| `v1/worker/gpu_worker.py`: `Worker.execute_model`, `model_runner`, shutdown | Ordinary TP1 execution; no native proposer model, no Target engine in this process |
| `v1/worker/gpu_model_runner.py`: `get_model`, `execute_model`, `execute_model_state`, `ExecuteModelState`, `_update_states`, `_update_streaming_request` | Guard exact pending-state field layout and all skipped completion obligations; consume owned logits once; rebase external context by stable internal ID |
| `v1/worker/gpu_input_batch.py`, `block_table.py`: `InputBatch.req_ids`, cached row/block updates | Map actual logits rows to expected IDs with an exact bijection; never reuse a stale positional mapping |
| `v1/core/sched/output.py`: `SchedulerOutput.make_empty`, `NewRequestData` | One ragged suffix schedule per active cohort; no fake finish during rebase, no PP-only `CachedRequestData.new_token_ids` correction injection |
| `v1/request.py`: `Request`, `RequestStatus.RUNNING`; `sampling_params.py`: `SamplingParams` | Private allocator request view with complete canonical context and explicit computed frontier; deterministic greedy metadata, no logprobs/penalties |
| `v1/core/{kv_cache_manager,kv_cache_utils,single_type_kv_cache_manager}.py`, `v1/kv_cache_interface.py` | Native cache sizing/initialization and `KVCacheManager` page allocation/free; exact full-attention spec, preserved model context, fail on exhaustion |
| `distributed/parallel_state.py` | Private world1 setup through Worker; destroy model-parallel and distributed groups on shutdown |
| `v1/worker/{kv_connector_model_runner_mixin,ec_connector_model_runner_mixin}.py` | Unsupported transfer/completion work must be absent before consuming pending forward |
| `v1/core/sched/scheduler.py` | Existing patched scheduler file identity is guarded; **no Target scheduler execution is replaced** |

No new vLLM patch was added. All five existing patch files remain byte-for-byte unchanged. The adapter consumes a private MRV1 completion seam, not a public vLLM rollback API. Supporting a different source pin requires a fresh audit.

## Proposal execution and row ownership

`DraftProposalPlan` carries stable request ID, round, committed prefix, effective budget and EOS policy. Validate the entire cohort first. Each request stores an opaque internal ID, committed prefix, materialized length, next logits, next round and pending proposal. Retired IDs cannot be resurrected. The service retains the existing `min(candidate_budget, remaining_output_budget - 1)` rule; a remaining budget of one continues through the unchanged proposal-free Target tail.

Setup materializes each full committed prefix once. Its cached next logits yield the first proposal token. Each subsequent step selects all continuing requests, materializes one token per row in **one model call**, then performs batched greedy selection. A length-k proposal therefore needs k selections and at most k−1 proposal forwards, matching the retained HF first-token convention. EOS/budget exhaustion removes rows from the next active cohort. No per-request model-forward loop exists inside a speculative step. Actual runner row order is checked independently and results are restored by ID.

Modern MRV1 input metadata is rebuilt for each active cohort: a `NewRequestData` record provides full CPU token context, existing page IDs and `num_computed_tokens=R`. Already resident requests take MRV1's streaming-context rebase branch, which removes/re-adds the input row, resets external context and clears stale sampled output IDs. Only `len(context)-R` suffix tokens are scheduled on GPU. Thus CPU context copying remains a Plan A cost; GPU full-prefix replay per round does not occur.

Stock `sample_tokens` is not called. The extension validates the exact pending `ExecuteModelState`, checks unsupported speculative/connector/auxiliary obligations are absent, owns one cloned logits batch, maps rows, and clears pending execution state. External Draft selection performs `stack(...).argmax(dim=-1)` on-device and a single `cpu().tolist()` per active step. There is no request-local `.item()` loop.

## Commit and persistent KV

Let C be committed context of length L, P the pending proposal of length p, a the verified accepted count, and M the physically materialized length. The last proposal token is normally selected but not materialized, so `M=L+max(p-1,0)` is checked exactly. The canonical final context is `C'=C+P[:a]+S`, where S is the verified correction/bonus suffix.

1. Check stable ID, round, parent context, pending P, acceptance/tail shape and final prefix hash for the complete cohort.
2. Compute `R=min(M,L+a)`; rejected positions beyond R have no committed authority.
3. Reuse existing private pages. Schedule only `C'[R:]`; mixed one-token correction and two-token accepted-last-plus-bonus tails share one ragged forward.
4. A zero-length missing suffix needs no model execution. If a nonterminal prefix shortens to an already materialized accepted frontier, refresh its last token to obtain the correct next logits; this exceptional refresh is counted explicitly.
5. Fence successful materialization, then publish M and logical committed length as `len(C')`, advance round/hash and clear pending state. A failed GPU operation poisons the backend and requires service restart.
6. Terminal commits release native runner rows and allocator pages after fencing. Shutdown removes hooks, releases model references and destroys the private process groups.

Physical rejected tail storage may stay allocated. Subsequent suffix metadata masks it and overwrites reused positions; it is never counted as committed KV. There is no eviction, preemption/replay or HF fallback. Insufficient pages or cohort/token capacity fails closed. Plan A is configured for 128 requests; it does not silently split an oversized cohort.

`BatchedDraftStateMachine` integrates the existing batched Serial synchronize/propose RPC. The legacy three-operation single-request wire sequence is also supported: synchronize classifies the decision, rollback stages invalidation, and append executes/publishes the complete materialization. Staging is explicitly marked in that response. The Target's existing scheduling, EOS, lifecycle, sampled rows and measurement code are untouched.

## Aggregate evidence

Forward counts come from model pre/post hooks, including setup/warm-up separately. A nonempty adapter materialization must observe exactly one model hook call. A backend name alone cannot certify batching.

| Evidence fields | Meaning |
| --- | --- |
| `backend_name`, `serving_performance_backend`, `execution_failed` | Backend purpose and failure state; qualification is separate |
| `draft_model_forward_count`, `_total`, `_by_purpose` | Actual proposal+commit forwards; total also includes setup/warm-up |
| `draft_batch_count`, `draft_batch_size_{min,p10,p50,p90,max,mean,histogram}`, `draft_batch_statistics_by_purpose` | Request rows per actual forward; nearest-rank quantiles; setup cannot inflate measured batching |
| `draft_autoregressive_step_count`, `draft_active_rows_per_step`, `draft_proposed_token_count`, `draft_proposal_count` | Bulk selection steps, shrinking active-row distribution and emitted proposal tokens |
| `draft_kv_materialization_forward_count`, `draft_correction_bonus_forward_count`, `draft_kv_operations`, `draft_scheduled_token_count_by_purpose` | Actual batched commit calls, missing-tail/refresh/rollback work and query-token counts |
| `draft_host_sync_count`, `_by_reason`, `draft_scalar_item_count`, `draft_sync_coverage` | Explicit adapter fences and bulk D2H; unobserved vLLM-internal synchronization is reported as unknown, not zero |
| `draft_gpu_event_time_ms`, `_by_purpose`, `draft_gpu_event_scope` | CUDA event intervals around actual model calls; excludes output-head computation outside model forward, sampling and other runner work; not kernel self time |
| `draft_step_host_gap_ns_total`, `_count` | Host gaps between successive active-step iterations |
| `worker_resources`, `backend_shutdown_complete`, `draft_live_requests_final` | Page allocations/frees/peak, allocator ownership and final cleanup |

Counters stay in memory. CUDA events are resolved at existing adapter completion fences and emitted in the final report. No per-token fsync instrumentation is added. Existing transport/round logging remains unchanged. Per-proposal `number_of_model_forwards` describes shared batch participation; use final actual-forward totals rather than summing that per-request field.

## Validation and operator interpretation

The CPU suite includes default/explicit/invalid backend selection, Serial factory routing, Dual rejection, heterogeneous active rows, row permutation errors, EOS/limits, zero/partial/full acceptance, correction/bonus/no-tail refresh, multiround HF-algorithm equivalence, stale transitions, failure poisoning, immutable evidence and cleanup. The independent fake worker retains stale physical suffixes and asserts correct frontier reuse.

For B=2,4,8,32,100 and k=4, the fake observes **3 actual B-row proposal forwards and 4 bulk selections**, versus 3B request-local HF forwards. Mixed commit tails require one ragged commit call. This proves CPU call shape and state semantics, not CUDA correctness or speedup. A separate test runs the actual adapter metadata glue with pinned real `SchedulerOutput` classes and CPU allocator/executor substitutes.

The dedicated Linux Python 3.11 source job checks out the exact vLLM pin and validates hashes, signatures, modern request rebase, forward completion and the original five patches in a disposable fixture. The normal Linux matrix runs Ruff/full pytest on Python 3.9/3.12; the existing Phase4 Python 3.11 contract job remains in place. Local test counts and final CI links are recorded in the task delivery.

Local validation on 2026-09-06: focused new/source checks **75 passed**; the Phase4-only run before the final comparator integration test was **724 passed, 6 skipped**; full suite including that test **906 passed, 7 skipped**. Four source checks skip without the explicit pinned-source directory and pass when it is supplied; other skips are pre-existing optional/platform checks. Ruff, all relevant helper Bash syntax, every new runbook Bash block/embedded Python fragment, wheel packaging/API resource equality and revision diff checks pass. A missing `pip` module in the local virtualenv was handled by building with `uv build`; no GPU dependency was installed. Linux CI results must be checked for the delivered SHA independently.

The offline comparator consumes HF Serial and batched Serial artifacts directly. It keeps three independent findings: completed-work performance comparability, exact Draft proposals on **common committed contexts**, and final Target sequence equality. Successful final Target numerical divergence is diagnostic under the existing policy; same-prefix Draft mismatch remains visible and blocks operator readiness. Unmatched contexts provide no Draft equivalence evidence, which is why D2/D3 controlled HF fixtures remain mandatory.

D5 requires 100 completions, identical requested work/accounting, valid raw/lifecycle evidence, final report binding and actual measured batch p50>1. It reports current HF derived forwards, new actual forwards, timing and warm-up/JIT evidence. There is no programmed speedup threshold. The operator assesses whether measured overhead reduction justifies a later D6 task. Neither CPU tests nor a single server pair establish production performance.

## File audit

Added implementation: `src/specrhythm/phase4/{draft_batch.py,draft_metrics.py,vllm_draft_worker.py,vllm_draft_api.json,vllm_draft_backend.py,batched_draft_service.py,draft_gate.py,draft_comparison.py}`.

Added tests: `tests/{test_phase4_vllm_draft.py,test_phase4_vllm_draft_source.py,test_phase4_draft_comparison.py,test_phase4_draft_runbook.py}`.

Added documentation: this file, `docs/phase4b3-plan-a-runbook.md`, and the seven previously completed design artifacts under `docs/design/phase4b3-batched-draft/` (included without revision).

Modified existing files: `src/specrhythm/phase4/draft_service.py` (explicit factory dispatch), `src/specrhythm/phase4/dual_service.py` (Serial-only selection guard), `pyproject.toml` (packaged API JSON), `.github/workflows/ci.yml` (CPU pinned-source job). No other existing runtime files are changed.
