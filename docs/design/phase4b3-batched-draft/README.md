# Phase 4B.3 batched Draft audit and design

Prepared 2026-09-06 against SpecRhythm `64031110e630b6643cc610620f05f8630e896c7c` and vLLM `752a3a504485790a2e8491cacbb35c137339ad34` (0.25.1). This directory contains analysis/design artifacts only. Serving runtime, UUID experiment and the Target five-patch stack are unchanged; no GPU or AutoDL work was performed.

**Recommended next coding task:** Plan A, an opt-in vLLM-batched Draft backend inside the existing GPU0 service, using one TP1 Worker/MRV1 runner and private paged KV. Qualify Serial before integrating Dual. The two highest-risk interfaces are a pinned Draft-only forward completion without stock token sampling, and rebasing external committed context through the existing MRV1 streaming-request path. Neither is presented as a public rollback API.

| Document | Contents |
| --- | --- |
| [MineDraft source audit](phase4b3-minedraft-source-audit.md) | Actual worker/batch/KV paths, five-rank topology, asynchronous timeline, reuse classification |
| [vLLM 0.25.1 API map](phase4b3-vllm025-draft-api-map.md) | V0→V1 migration table, native accepted-token/KV progression, existing Target patch boundary, exact private adapter seams |
| [Batched Draft design](phase4b3-batched-draft-design.md) | Current HF audit, architecture decision, state structures, explicit commit algorithm, Serial/Dual stages, plans/files/instrumentation |
| [Test plan](phase4b3-batched-draft-test-plan.md) | CPU/source tests to implement; user-executed GPU gates D1–D6; immutable evidence and failure criteria |
| [Source provenance](source-provenance.json) | 64 source files with SHA256, 146 selected AST symbols, pinned revisions, old-class search over 1,966 vLLM Python files |
| [Document validation](document-validation.json) | Checks performed for this docs-only task; no runtime correctness/performance claim |

## Required deliverable index

| Requested item | Answer / location |
| --- | --- |
| 1. MineDraft Draft architecture | Proposer Worker/runner inside one V0 speculative engine topology; audit §1 |
| 2. Exact batching path | `get_spec_proposals → Top1Proposer → sampler_output → TP1DraftModelRunner`; audit §2 |
| 3. Draft KV lifecycle | Worker-owned paged storage, scheduler block tables, logical rollback and bonus catch-up; audit §3 |
| 4. Target/Draft topology | World5; Draft/driver0, Target1–4, non-driver Target TP group; audit §4 |
| 5. Reusable components | Thirteen-component classification; audit §6 |
| 6. 0.9.2→0.25.1 migration | Explicit status/equivalent/adaptation table; API map §2 |
| 7. Current incompatibilities | Scalar HF/service API, removed V0 classes, native same-TP guard, external token/KV ownership; API map §§3–6 and design §§1–5 |
| 8. Recommended architecture | B: isolated Worker/runner backend; design §2 |
| 9. Separate process | Retain GPU0 Draft service, independent world1; design §§2–3 |
| 10. Exact commit/rollback | `R=min(M,L+a)`, materialize only canonical missing suffix, publish only at `M=len(C')`; design §6 |
| 11. Proposal contract | Existing IDs/round/hash/EOS/budget/lifecycle with actual Draft row/physical evidence; design §9 |
| 12. Minimum viable plan | Plan A: pinned ordinary runner, rebase/completion adapter, batched proposal and commit; design §12 |
| 13. Production plan | Plan B: persistent row/input buffers, GPU step/EOS masks and narrower facade; design §12 |
| 14. Recommended plan | Plan A first, gated by single-request and real-batch correctness; design §§12–13 |
| 15. Expected files/classes | Add backend/worker adapter/planner/metrics; small service/config dispatch; Dual service only at D6; design §12 |
| 16. New vLLM patch | None currently required/planned; internal adapter is version-sensitive, conditional minimal patch policy explicit; API map §6 and design §12 |
| 17. Gates | CPU/Linux source/unit/regressions, then D1→D2→D3→D4→D5 Serial→D6 Dual; test plan |
| 18. Instrumentation | Actual model forwards, request/token batch histograms, events, KV work, host gaps, scoped synchronization counts; design §11 |
| 19. Research boundary | Batched data plane versus scheduler/lifecycle policies; source-limited attribution, no novelty conclusion; audit §7 |
| 20. Generated files | Document table above |

The supplied historical A800 measurements are identified as prior results, not rerun here. GPU output equivalence and performance improvement remain gates to establish after implementation. No backend CLI/run flags are claimed to exist yet.
