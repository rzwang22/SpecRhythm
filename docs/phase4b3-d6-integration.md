# Phase 4B.3 D6: production Draft in the existing Dual path

Base: `45725fcd3da3163be0d03889791bccb08fb31688`. The operator reports D1–D5 qualified, including corrected-100 Serial vLLM at 8285.539721 ms / 179.469298328 tok/s versus the fresh HF pair at 52368.990773 ms / 28.3946659665 tok/s. These supplied results establish the starting point; they are not reused as D6 timings. The coding agent runs CPU tests only.

## Source-first implementation map

| Boundary | Existing implementation | D6 integration |
| --- | --- | --- |
| Service launch | `phase4b1_start_draft(dual)` → CLI `phase4-dual-draft-service` → `dual_service.run_dual_draft_service` | Same entry and process placement; dispatch production selector to the D6 adapter. |
| Selector/rejection | `vllm_draft_backend.selected_draft_backend` reads `SR_PHASE4_DRAFT_BACKEND`, default `hf-persistent`; Dual entry explicitly rejected other values | `vllm-batched` now selects production; HF and invalid-selector behavior remain supported. |
| Bootstrap | `vllm_dual._rank_zero_bootstrap` sends setup-only `execute(initialize)` for each complete DecodeReady prefix | Same initialization and barrier, no speculative work before measurement. |
| Initial proposals | `_enqueue_initial_proposals` sends all active initial rows in one existing `enqueue(propose_only, rows)` message | One `propose_many` for that existing cohort. |
| Later work | `on_target_tokens` collects this callback's verified commit rows and sends one `enqueue(commit_and_propose, rows)`; terminal tails use `finish_tail` | One `commit_many`, then one `propose_many` for the surviving members of that same message. |
| Old bottleneck | `AsyncDualDraftController.enqueue` split a message into one `_Work` per request; `_run` called HF scalar `propose` after scalar rollback/append | Production controller queues the original message as one unit. No cross-message merge, delay, accumulation or new ready policy. |
| Ready selection | `DualBatchScheduler.schedule` nonblocking `poll_ready(available)` with existing microbatch=2 | Scheduler source and ready/retired-ready logic unchanged. Initial Draft cohort can exceed 2; subsequent commit cohort is derived from the existing Target scheduling decision. |
| Identity and commit | `DualProposal`, stable request IDs, proposal ID, prefix version/hash, `dual_greedy_acceptance` | Same identities and acceptance rule projected into existing `DraftProposalPlan` / `DraftCommitPlan`. Validate the entire commit cohort before mutation. |
| Retirement | Terminal commits finish per-request KV; one-token proposal-free tail appends its terminal Target token then finishes | `commit_many` releases terminal private blocks. A narrow proposal-free terminal-tail branch allows exactly one Target token with no live proposal; stale rounds/prefixes and nonterminal empty proposals remain invalid. |
| Worker lifecycle | HF model created by service main; scalar work on background queue; shutdown after draining | Production model is created, initialized, run and shut down on the same background owner thread required by the qualified backend. One model/worker only. |
| Shutdown evidence | RPC `shutdown` → queue drain → backend shutdown → service/process exit | Final existing backend metrics written once, hash returned in `draft_shutdown`; startup is immutable. Serve/bind failure also closes the owner worker. |
| Physical overlap | `build_cycle_and_overlap_events` intersects synchronized Draft proposal intervals with Target TP2 verification intervals on disjoint request sets | Same CUDA interval contract around each actual Draft cohort; shared intervals explicitly identified. Reporting unions repeated witness intervals and never calls them critical-path savings. |

The length-prefixed canonical JSON Unix-socket protocol and protocol version are unchanged. No new vLLM patch, worker API, second Draft model, candidate policy, proposal budget, UUID policy, eager mode or scheduler threshold is introduced. The backend uses its qualified `propose_many` / `commit_many` implementation; the only additional accepted commit form is Dual's terminal proposal-free tail. Existing Serial plans cannot select that form, and its entry point/state machine are unchanged.

## Target-only scope discovered during the audit

The existing Target-only resident runner requires a Draft service to materialize the shared DecodeReady prefix **before** measurement. Its measured callback returns no proposals. Completely removing that setup service would change the Target startup contract and shared measurement prerequisites. D6 retains this baseline behavior and reports no measured Draft proposal work for Target-only; it does not claim that GPU0 has no resident setup model. The operator was asked to clarify this distinction. No Target source changes implement D6.

## Evidence and qualification

D6-A is a two-request transport/model/KV smoke, with overlap permitted to be absent and no performance conclusion. D6-B uses the unchanged corrected-five workload, real production batching, ordinary Target verification, native request/lifecycle validation, physical overlap and positive performance metrics. D6-C uses a fresh result root and runs Target, Serial-vLLM, Dual-vLLM in that order on the same final commit/server/corrected-100 inputs.

`draft_dual_comparison validate` consumes the existing runtime, lifecycle, metrics and normal Target evidence; it returns the first material failure instead of emitting failures for unexecuted checks. No numerical observer or per-forward logging is added. Empty `errors` values are equivalent. HF proposal equality, final sequence equality, JIT warnings and historical GPU UUID are not performance gates. Current workload/model/config/backend identity, output limits, token accounting, Target mapping/causality, retirement/cleanup, real batching, and Dual physical overlap remain material.

The per-run report is `qualification.json` (`specrhythm.phase4b3-d6-run.v1`). The final comparison writes `comparison.json` and a concise `comparison.md`. It includes all six requested throughput/makespan ratios, mean/p50/p90 TPOT changes, Target physical forwards/query tokens, Draft work/batch/GPU counters, observed overlap, original Draft cohort sizes, verification batch distribution and fragmentation counts. Cases A–D are selected from measured throughput only; a tie/other order is left unclassified. These descriptions do not prove a scheduling mechanism or authorize tuning.

Use **production vLLM Batched Draft end-to-end improvement**. Work can differ across modes; no pure batching claim is made. Existing D1–D5, Phase 4B.2 and numerical probe artifacts remain immutable.

The complete operator handoff is [the fresh-server D6 runbook](phase4b3-d6-runbook.md). Actual D6-A/B/C GPU results remain pending operator execution.
