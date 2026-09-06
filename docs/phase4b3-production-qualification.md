# Phase 4B.3 production Draft qualification policy v2

`VllmBatchedDraftBackend` is the authoritative production Draft execution regime. HF remains a bring-up oracle and cross-backend diagnostic reference. Once the pinned four-path evidence qualifies the HF/vLLM numerical boundary, HF proposal equality alone does not block D3, D4 or D5. Internal vLLM state, ownership, mapping, batching, cleanup and provenance requirements remain blocking. No kernel, precision, vLLM patch, proposal budget, scheduler or Dual change implements this decision.

## Evidence and its limits

The operator ran the four-path experiment at `dd02c7ffe7f419134d9134b8feb932c20652ca34` on A800. The retained archive is `D3-logits-20260906T093347Z-1477.tar.gz`, SHA256 `150e2763501eeefce56b7d44439692bf98860ad782f9489fac95603c85fb8cbb`. The agent independently loaded the full four vectors, checked original artifact hashes and recomputed their summaries and classification without GPU execution.

| Execution | Request-7 top1 | Raw margin 16−227 |
| --- | --- | --- |
| A: fresh HF, full 44-token prefix | 16 | +0.0625 |
| B: fresh vLLM singleton, 43-token committed prefill plus token 11 | 227 | −0.1875 |
| C: fresh vLLM B7, exact survivor committed prefixes plus token 11 | 227 | −0.1875 |
| D: persistent vLLM B7, original frame-8 history | 227 | −0.1875 |

B/C and C/D have identical complete 151,936-element float32 vectors: maximum absolute difference 0.0. Their common little-endian float32 SHA256 is `cb401df3aed179688b36643b19e8fcfb843ee8ccd9320af386c083c0d2aa5cef`. HF versus vLLM differs, maximum absolute difference 0.27734375. The outcome is `hf_vllm_execution_numerical_divergence`, `strong_evidence`.

This qualifies the production execution-regime boundary for the frozen model/config/source. It does not identify a particular BF16, linear, attention, fused-operation or reduction mechanism. `mechanistic_root_cause_proven=false` remains explicit. No further numerical diagnosis or K/V content expansion is required for D4/D5 progression under the project decision.

## Versioned reports and D3 execution

The legacy `draft_gate` entry point and `specrhythm.phase4b3-draft-gate.v1` retain their historical HF-exact semantics. Old artifacts are never rewritten or reinterpreted as v2 passes. The new `draft_qualification_gate` entry point emits `specrhythm.phase4b3-draft-qualification.v2` and takes the complete retained four-path directory as an explicit input.

Before GPU execution, the new entry point recomputes all four raw vectors and verifies the original `probe-inputs.json`, path-artifact hashes, classification and all probe structural diagnostics. Both vLLM full-vector equalities must pass, independently of top1. A declared classification flag alone is insufficient. The worker must use matching model weights, tokenizer/config metadata, Python/Torch/transformers/vLLM versions, numerical-source hashes and five patches. The qualification commit and physical GPU UUID may differ from the probe; numerical inputs, installed source identity and the A800/SM80 execution regime may not.

### Current device binding versus retained execution regime

The `10a64da` admission failure was caused by comparing the current startup UUID with the historical probe UUID before production replay. An AutoDL restart can allocate another A800. The initialized worker's existing real UUID query and device-binding check remain unchanged. D3 now separately requires `CUDA_VISIBLE_DEVICES=0`, logical CUDA 0, container-mapped physical GPU0, a real syntactically valid UUID, NVIDIA A800-SXM4-80GB and TP1/world1/rank0. The admission artifact retains both UUIDs and reports `current_device_binding_valid`, `historical_probe_gpu_uuid`, `current_gpu_uuid`, `same_physical_gpu` and `execution_regime_device_compatible`. A different UUID is allowed; a different device class or compute capability is not.

The retained B/C/D runtime metadata must agree on A800/SM80, BF16, MRV1, eager/private-prefix execution, effective batch-invariance, actual FlashAttention implementation/version and other captured stable worker properties. Current metadata must match that regime. All 25 installed API source hashes, source commit, numerical sources, five patches, model weights/metadata, tokenizer, config and Torch/transformers/vLLM versions keep their exact preflight/identity checks. The original probe enforced Python 3.11 but did not capture a patch version: admission preserves that exact major/minor contract and records the current full version without inventing historical data. Allocator capacity, startup timings and physical UUID are not cross-run numerical compatibility criteria. The existing four-way probe and its same-run UUID consistency rule are unchanged.

`specrhythm.phase4b3-draft-admission.v1` is written once as `admission.json` and bound into the v2 gate. Current runtime evidence is revalidated offline against the retained regime and backend provenance. Admission failure before production replay yields `execution_started=false`, `admission_valid=false`, `structural_checks_executed=false`, `vllm_semantic_valid=null` and null structural results. The aggregate preserves the primary admission error instead of inventing 15 structural failures. An interrupted replay retains partial observations with `execution_started=true`; structural results remain unavailable until the suite completes. Actual observed replay/cleanup failures still block qualification.

After this fix, use a new root and the [fresh-server D1–D3 admission rerunbook](phase4b3-device-admission-runbook.md). It reruns D1, D2 and v2 B2/B4/B8, inspects the aggregate and packages artifacts. It does not launch D4/D5. The prior admission-failed roots and all historical probe artifacts remain untouched.

Fresh D3 uses the original heterogeneous initial requests, recorded EOS subset and output limits. Its synthetic zero/partial/full acceptance and correction/bonus progression now uses **actual vLLM proposals**. Removing the old HF equality assertion alone would be incorrect: HF-derived commit deltas cannot serve as vLLM acceptance history after proposals diverge. The implementation keeps original HF proposals and compares them only at equal parent prefixes. Later contexts changed by the production trajectory are recorded as unmatched, never falsely labeled exact or direct same-prefix divergence.

The fresh D3 worker installs no materialization observer, compute-logits wrapper, GPU metadata read or diagnostic model hook. Host-state assertions run at ordinary API boundaries: stable request/internal identity; round and committed-prefix identity; output budgets; EOS termination/subset shrinking; retirement; allocator and runner live sets; private block ownership and no visible aliases; CPU InputBatch row/token/page-table binding; materialized frontier; and committed KV length. Native adapter forward/completion checks remain unchanged. The actual proposal-forward histogram must contain the requested B2/B4/B8 size. Every request must retire and all allocated blocks must be freed.

These host checks do not claim to inspect K/V bytes. The separate full-vector fresh/persistent control supplies the qualified historical-execution evidence. The diagnostic four-path run itself is not a new uninstrumented D3 pass or a performance result.

Each D3 directory contains the unmodified generated HF oracle, detailed `observation.json`, startup, backend report, copied compact regime qualification and final `gate.json`. Reports use exclusive final writes and bind inputs by SHA256. Partial observations and cleanup failures are retained.

Key v2 fields are:

- `vllm_semantic_valid`, `vllm_batch_invariant_valid`, `vllm_persistent_history_valid`;
- `cross_backend_divergence_qualified`, `d3_qualified`, `structural_checks`;
- `hf_draft_exact`, `hf_exact_diagnostic` (including unmatched-context counts);
- `hf_vllm_divergent_request_count`, `hf_vllm_divergent_rounds`;
- `hf_vllm_divergence_classification`, `hf_vllm_divergence_evidence_status`;
- `mechanistic_root_cause_proven=false`, `hf_proposal_equality_is_blocking=false`.

`draft_qualification aggregate` emits `specrhythm.phase4b3-d1-d3-qualification.v2`. It requires D1/D2 success and independently validates fresh v2 B2, B4 and B8 reports, their underlying artifacts and compatible provenance. Compatible retained D1/D2 are supported; the complete runbook reruns all five gates to avoid reliance on incomplete retained evidence. B2/B4 v1 artifacts remain historical evidence but do not replace the new v2 execution reports in this aggregator. D4 admission requires this bound aggregate. D5 additionally requires a valid corrected-five D4 comparison. From D4 onward, completed qualification certificates are reused without recursively repeating D1–D3 checks.

## D4 and D5

The comparison CLI now requires `--qualification` and `--stage D4|D5`; D5 also requires `--d4`. D4 is exactly five completed requests; D5 is exactly 100. The legacy Python comparison function without a stage retains its v1 behavior for historical analysis; it cannot produce a new progression certificate.

The D4/D5 wrapper uses the CPU-only `draft_performance_policy` admission. It reads completed D3/D4 certificates, checks the requested current checkout, qualified model/config and exact workload, and saves request contracts in `serial-admission.v1`. Historical D3 commit, device UUID and full raw-logits provenance are not requalified. D3 entry points and all Draft/Target execution remain unchanged.

Blocking conditions are successful execution return path, expected completed request count, valid token accounting/declared limits, request lifecycle/cleanup, structurally valid Target verification, matching workload/model/config/backend identity, real multi-request vLLM proposal batching, and finite positive performance metrics. Current raw/lifecycle/Target evidence and final backend metrics remain bound by hashes. Actual nonempty errors and `valid=false` still block.

Diagnostic conditions are qualified HF/vLLM nonexactness, final sequence equality, JIT warnings, informational numerical results, duplicated historical provenance checks and nonessential metadata. `errors=null`, `errors=[]`, missing errors and other empty containers all mean no error. They do not override a false validity flag. Batch p50 is reported; actual multi-request proposal forwards are required even if p50 is one. Work differences are quantified instead of disqualifying an otherwise valid end-to-end experiment.

Reports retain schema v2 and add `qualification_policy=specrhythm.phase4b3-performance-validity.v1`, `blocking_conditions`, `blocking_errors`, `diagnostic_conditions`, `diagnostics`, `error_serialization` and `historical_qualification_revalidated=false`. D5 checks experiment identity against the completed D4 report, while allowing a newer comparator commit and a new server allocation.

Admission reads the actual workload with the existing R3 loader and binds each request's prompt identity and `maximum_new_tokens`. Per-round remaining budgets are checked against these declared limits. Historical performance reports can leave `maximum_new_tokens` null; the new comparator uses the bound workload declaration and never guesses an output limit from the observed generated length or changes the performance schema.

`specrhythm.phase4b3-serial-backend-comparison.v2` distinguishes `stage_qualified`, `d4_qualified`, `d5_qualified`, `performance_comparable` and `performance_interpretation_allowed`. D4 qualifies the corrected-five pilot and allows an explicitly scoped pilot performance summary; it does not establish corrected-100 performance. D5 reports the fresh pair's end-to-end results even when qualified cross-backend Draft proposals differ. HF equality and unmatched contexts remain explicit diagnostics, as does final Target sequence equality under the existing matched-work policy.

For **each mode**, `work_accounting` reports completed requests, measured committed tokens, makespan, throughput, TPOT, proposal count, proposed/accepted/rejected tokens, mean accepted length, verification rounds, physical Target forwards and query tokens, actual Draft forwards, batch min/p10/p50/p90/max/mean, Draft GPU event time, scoped host synchronization count, correction/bonus counts and JIT/warmup evidence. Target counts are reconstructed from existing rank-0 diagnostics: shared physical forwards are counted once, without multiplying TP replicas; query tokens sum their request rows, including measured Target-only tails.

vLLM uses its existing in-memory metrics. The HF A/B wrapper opts into `SR_PHASE4B3_HF_METRICS=1`, which delegates all original HF model/proposal/cache operations and adds only nonblocking CUDA events and in-memory counters. It reads events once at final shutdown and adds no measured-path fence or per-forward file write. Setup is separated from proposal/commit forwards. Existing greedy scalar `.item()` calls are counted; framework-internal synchronizations are explicitly unobserved. The ordinary HF service remains unchanged when the flag is absent; Dual uses neither this dispatch nor this observer. The backend selector default is unchanged; the A/B helper selects each backend explicitly.

Performance terminology is **production vLLM Batched Draft end-to-end improvement**. `execution_time_comparison` separates makespan and Draft event-time reduction. `work_deltas` reports absolute and relative proposal, acceptance and Target-work changes. A descriptive, declared 1% rule marks nearly matched work; zero-denominator differences remain explicit rather than being hidden. `pure_batching_speedup_claim=false` always. The historical approximately 49.3 s / 30.2 tok/s HF baseline is context; the fresh HF run is the comparison denominator.

D6 remains out of scope regardless of improvement magnitude. No Dual integration or microbatch/scheduling/accumulation/UUID/eager tuning is authorized by these reports.

For the completed D4 executions, use the [offline D4 and fresh-server D5 runbook](phase4b3-d4-offline-d5-runbook.md). It preserves original D4 artifacts and never reruns D3/D4 GPU. D5 is allowed only after the offline D4 comparison qualifies. The earlier full progression runbook remains historical context.
