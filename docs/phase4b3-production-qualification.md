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

Before GPU execution, the new entry point recomputes all four raw vectors and verifies the original `probe-inputs.json`, path-artifact hashes, classification and all probe structural diagnostics. Both vLLM full-vector equalities must pass, independently of top1. A declared classification flag alone is insufficient. The worker must use matching model weights, tokenizer/config metadata, Python/Torch/transformers/vLLM versions, numerical-source hashes, five patches and GPU identity. The qualification commit may differ from the probe commit; numerical inputs and installed source identity may not.

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

`draft_qualification aggregate` emits `specrhythm.phase4b3-d1-d3-qualification.v2`. It requires D1/D2 success and independently validates fresh v2 B2, B4 and B8 reports, their underlying artifacts and compatible provenance. Compatible retained D1/D2 are supported; the complete runbook reruns all five gates to avoid reliance on incomplete retained evidence. B2/B4 v1 artifacts remain historical evidence but do not replace the new v2 execution reports in this aggregator. D4 admission requires this bound aggregate. D5 additionally requires a valid corrected-five D4 comparison, recomputed from its source artifacts.

## D4 and D5

The comparison CLI now requires `--qualification` and `--stage D4|D5`; D5 also requires `--d4`. D4 is exactly five completed requests; D5 is exactly 100. The legacy Python comparison function without a stage retains its v1 behavior for historical analysis; it cannot produce a new progression certificate.

New Serial wrappers validate prerequisites and fingerprint current weights/config/source before starting either backend, saving a per-mode `serial-admission.v1` artifact. They call the unchanged resident Serial and decode-measurement helpers. The comparator checks the admission, execution commit, startup/shutdown identity, same workload and output limits, token accounting, lifecycle, actual Target diagnostics/round identity, prefix chains, Target authority, Draft/Target committed KV lengths, backend cleanup and actual proposal batching. Draft batch p50 must remain greater than one for the vLLM mode. Unqualified or incompatible numerical evidence cannot waive any of these checks.

Admission reads the actual workload with the existing R3 loader and binds each request's prompt identity and `maximum_new_tokens`. Per-round remaining budgets are checked against these declared limits. Historical performance reports can leave `maximum_new_tokens` null; the new comparator uses the bound workload declaration and never guesses an output limit from the observed generated length or changes the performance schema.

`specrhythm.phase4b3-serial-backend-comparison.v2` distinguishes `stage_qualified`, `d4_qualified`, `d5_qualified`, `performance_comparable` and `performance_interpretation_allowed`. D4 validates semantics and work but does not authorize a performance interpretation from five requests. D5 reports the fresh pair's end-to-end results even when qualified cross-backend Draft proposals differ. HF equality and unmatched contexts remain explicit diagnostics, as does final Target sequence equality under the existing matched-work policy.

For **each mode**, `work_accounting` reports completed requests, measured committed tokens, makespan, throughput, TPOT, proposal count, proposed/accepted/rejected tokens, mean accepted length, verification rounds, physical Target forwards and query tokens, actual Draft forwards, batch min/p10/p50/p90/max/mean, Draft GPU event time, scoped host synchronization count, correction/bonus counts and JIT/warmup evidence. Target counts are reconstructed from existing rank-0 diagnostics: shared physical forwards are counted once, without multiplying TP replicas; query tokens sum their request rows, including measured Target-only tails.

vLLM uses its existing in-memory metrics. The HF A/B wrapper opts into `SR_PHASE4B3_HF_METRICS=1`, which delegates all original HF model/proposal/cache operations and adds only nonblocking CUDA events and in-memory counters. It reads events once at final shutdown and adds no measured-path fence or per-forward file write. Setup is separated from proposal/commit forwards. Existing greedy scalar `.item()` calls are counted; framework-internal synchronizations are explicitly unobserved. The ordinary HF service remains unchanged when the flag is absent; Dual uses neither this dispatch nor this observer. The backend selector default is unchanged; the A/B helper selects each backend explicitly.

Performance terminology is **production vLLM Batched Draft end-to-end improvement**. `execution_time_comparison` separates makespan and Draft event-time reduction. `work_deltas` reports absolute and relative proposal, acceptance and Target-work changes. A descriptive, declared 1% rule marks nearly matched work; zero-denominator differences remain explicit rather than being hidden. `pure_batching_speedup_claim=false` always. The historical approximately 49.3 s / 30.2 tok/s HF baseline is context; the fresh HF run is the comparison denominator.

D6 remains out of scope regardless of improvement magnitude. No Dual integration or microbatch/scheduling/accumulation/UUID/eager tuning is authorized by these reports.

Use the complete [fresh-server runbook template](phase4b3-production-qualification-runbook.md), delivered with the exact tested SHA substituted. The operator returns the full immutable D3→D4→D5 archive for review.
