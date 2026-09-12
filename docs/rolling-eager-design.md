# Fixed-length Rolling Eager Continuation: shared protocol and Serial GPU adapter

Stage 1 delivered the reusable CPU protocol at
`e4076628b10ccb5fef712dabae645f712c32cb51`. Stage 2 connects that same protocol to
the real Draft worker and an explicitly selected `serial-eager` fixed diagnostic
mode. GPU correctness, overlap and performance remain **PENDING**: server execution
belongs to the operator, and no AutoDL connection or GPU run was performed during
implementation. Existing Target, Serial, PingPong and simulator defaults remain
intact. PingPong GPU mixed admission is still deferred.

The feature branch remains `codex/rolling-eager-v0.1`, created from verified commit
`5a16d00fd10778189db3addbff558a2260944b32`. Its dependent Draft PR uses PR #4's
head branch, `codex/vllm-serving-v0.1`, as its base; the base SHA inspected for
this stage was also `5a16d00fd10778189db3addbff558a2260944b32`. PR #4 remains
unmerged and its branch is unchanged; PR #2 and PR #3 are outside this change.
Continuation work is delivered through dependent Draft PR #5 without merging it.

## Audited existing execution contracts

The following facts come from the local source at the base SHA. No external
engine checkout or GPU session is needed for this audit.

| Area | Source and current behavior | Consequence for continuation |
| --- | --- | --- |
| Target acceptance | [`serial.py`](../src/specrhythm/phase4/serial.py), `greedy_acceptance`: the authoritative Target delta contains the accepted Draft prefix and one correction after rejection, or the complete proposal and one bonus on nonterminal full acceptance. `AcceptanceDecision` conserves the committed-token accounting. | The bonus belongs to the parent commit. A future proposal cannot assume its prefix ends immediately after the four accepted candidates. |
| Serving termination | [`dual_commit.py`](../src/specrhythm/phase4/dual_commit.py), `DualStopPolicy.canonicalize` applies EOS, token-stop and maximum-output trimming; `dual_greedy_acceptance` also permits a terminal truncated accepted prefix without inventing a correction or bonus. [`vllm_dual.py`](../src/specrhythm/phase4/vllm_dual.py), `_rank_zero_update`, uses the canonical sampled result. | Termination wins over promotion. A terminal accepted prefix is legal output, but is never a basis for more continuation. |
| Serial authority | [`vllm_remote.py`](../src/specrhythm/phase4/vllm_remote.py), `_rank_zero_propose` and `_finalize_round`, derives the delta from the logical Target prefix, checks that the prefix extends its previous value, and synchronizes Draft before requesting the next normal proposal. | A future Serial adapter must split verification-start continuation work from verification-result resolution without weakening the existing Target prefix check. |
| Draft proposal state | [`draft_service.py`](../src/specrhythm/phase4/draft_service.py), `DraftStateMachine.batch_propose`, rejects another proposal while one is pending. [`dual_service.py`](../src/specrhythm/phase4/dual_service.py), `DualDraftMachine`, likewise retains one live proposal and calls `_propose` only after `commit_and_propose` has incorporated the correction/bonus. | A continuation needs a distinct dependent slot. Treating it as an ordinary second proposal violates the current machine contract. |
| Physical Draft KV | [`vllm_draft_backend.py`](../src/specrhythm/phase4/vllm_draft_backend.py), `propose_many`, samples the first candidate from cached committed-prefix logits and materializes a sampled token only when another candidate is needed. [`draft_batch.py`](../src/specrhythm/phase4/draft_batch.py), `commit_frontier`, requires `materialized == prefix_length + max(proposal_length - 1, 0)`. | A four-token proposal leaves its fourth token unmaterialized. Existing `commit_frontier` cannot accept the longer frontier produced by continuation and must not be reused unchanged for that case. |
| HF correctness backend | [`draft_service.py`](../src/specrhythm/phase4/draft_service.py), `HFPersistentDraftBackend.propose`, has the same final-token gap. On complete acceptance `rollback` materializes that final candidate before `append_target_token` appends the Target tail. | Cached logits and token generation are separate from physical materialization on both existing Draft backends. |
| Async ownership | [`dual_batched_draft.py`](../src/specrhythm/phase4/dual_batched_draft.py), `BatchedDualDraftController`, constructs the machine/backend and executes batches and shutdown on one owner thread. The backend and [`vllm_draft_worker.py`](../src/specrhythm/phase4/vllm_draft_worker.py) check that thread. | Scheduler callbacks publish immutable messages. They must not mutate Draft KV or transfer its ownership. |
| Async mailboxes | [`dual_service.py`](../src/specrhythm/phase4/dual_service.py), `AsyncDualDraftController`, indexes inflight, ready and claimed work by stable request ID. Production batch completion clears the claimed entry and publishes any returned proposal directly to ready. | Continuation completion must use a separate role/generation mailbox until its parent dependency is resolved; it cannot pass through ordinary ready publication. |
| Verify admission | [`admissibility.py`](../src/specrhythm/phase4/admissibility.py) checks request identities, prefix version/count/hash, round, exact proposal tokens, expiry and consumption. [`vllm_dual_scheduler.py`](../src/specrhythm/phase4/vllm_dual_scheduler.py) rejects stale or second unverified proposals. | Promotion must first establish ordinary, unconsumed proposal evidence at the new authoritative prefix. Unconfirmed continuation never enters the verify-ready queue. |
| A/B policy | [`vllm_pingpong_scheduler.py`](../src/specrhythm/phase4/vllm_pingpong_scheduler.py), `_decision_for`, gates by selected cohort and same-cohort Draft work. [`serving/s2_scheduler.py`](../src/specrhythm/serving/s2_scheduler.py), `S2PingScheduler`, adds active-pool/arrival gates and a similar cohort dependency. [`dual_pingpong_draft.py`](../src/specrhythm/phase4/dual_pingpong_draft.py) requires a single stable cohort per batch. | Eager work must separate home cohort from execution role. A promoted request may qualify in the next Target stage without changing identity, copying Target state or waiting for a complete A/B cycle. |

## Frozen semantics and identities

`normal_candidate_length = 4` and `eager_candidate_length = 4`. These are
candidate counts, not total generated work. A full eager continuation generates
one predicted bridge plus four future candidates. EOS and the remaining output
budget may shorten legal work; the normal Target-only tail remains available
when only one output token remains.

Three independent questions are represented:

1. **Eligibility:** whether a stable request ID belongs to the configured eager
   set. Rejection, resource pressure and the admission switch do not remove
   static membership; the decision records membership separately from `enabled`.
2. **Admission:** whether this execution opportunity may start a Draft task or
   admit a promoted proposal to the next verification stage. Capacity can defer
   an eligible request without changing its configuration.
3. **Validity:** whether this particular continuation has an exact, live,
   confirmed dependency and complete unconsumed data. Eligibility alone never
   makes a continuation valid.

Request identity is independent of physical InputBatch row, active-pool slot and
A/B membership. A replacement request is evaluated using its own stable ID and
receives its own proposal and continuation generations. Owner identity is also
checked; one owner cannot mutate another owner's request state.

The protocol records the authoritative committed prefix, version, length and
hash; current proposal ID and tokens; continuation ID and generation; parent
proposal ID; complete dependency prefix; predicted bridge and future candidates;
materialized KV frontier; lifecycle status; recovery requirement; and eager
decision/version. Hashes are consistency evidence, not a substitute for comparing
the complete dependency token sequence. Independent proposal and continuation
IDs prevent a recovery round from aliasing a discarded predecessor.

At most one future continuation may depend on the currently verifying proposal.
There is no recursively constructed chain of unresolved predictions. Once the
parent is committed and the continuation becomes a legal promoted proposal,
verification of that new proposal may start the next dependent continuation.

## Implemented CPU modules and API

The reusable CPU modules in [`continuation`](../src/specrhythm/continuation) have
no automatic scheduler installation. Their responsibilities remain:

| Module | Implemented responsibility |
| --- | --- |
| [`core.py`](../src/specrhythm/continuation/core.py) | `RollingContinuation`, owner-local request/proposal/continuation state, immutable work and receipt contracts, dependency validation, rolling promotion, normal recovery, cleanup and accounting. |
| [`policy.py`](../src/specrhythm/continuation/policy.py) | `EagerEligibilityProvider.evaluate`, `EagerDecision`, `EagerStepContext` and immutable `StaticEagerEligibility`. The decision separates `eligible`, `enabled` and `decision_version`; `can_admit` combines the first two without making a capacity choice. |
| [`scheduling.py`](../src/specrhythm/continuation/scheduling.py) | Immutable `SchedulingView`, `TargetAdmission` and `DraftTask`; pure selectors for Target admission, recovery-first normal work and dependent Draft work; identity deduplication and dispatch-time intent revalidation. |
| [`adapter.py`](../src/specrhythm/continuation/adapter.py) | `scheduling_view`, `start_admission` and `start_draft_task` connect those scheduler contracts to real core state. They apply current policy and revalidate identity, proposal and versions immediately before dispatch on the existing owner. |
| [`cpu.py`](../src/specrhythm/continuation/cpu.py) | `generate(work, next_token)`, a deterministic token-by-token executor that materializes missing dependency inputs and leaves the final sampled token outside KV. It returns actual generated tokens and frontier evidence, not a preselected protocol outcome. |

Create one `RollingContinuation(owner_id, provider)` on the existing owner's
thread. All its reads and mutations check that thread; worker-side `generate`
operates only on its immutable `DraftWork`. The `state`, `proposal` and
`continuation` accessors return defensive copies.

| API | Effect |
| --- | --- |
| `register(request_id, prefix, ..., committed_prefix_version=0)` | Attach one stable identity, full current prefix, version, A/B home cohort, EOS IDs and remaining output allowance. Duplicate identity, including a retired identity in the same owner, is rejected. |
| `evaluate_eager_eligibility(request_id, step_context)` | Store a provider decision; reject a regressed version or changed decision at the same version. A disabling decision cancels the active dependent continuation. |
| `schedule_normal_recovery(request_id)` | Issue immutable normal or recovery work from the committed prefix. Repeated scheduling while that same normal work is outstanding returns its existing work ID. |
| `record_normal_completion(work, completion)` | Validate exact work identity, dependency, length/EOS and frontier, then create an independently identified legal proposal. A zero-candidate result represents the legal one-token Target tail. |
| `start_verification(request_id, proposal_id, eager=False)` | Consume one current legal ready proposal for Target. Eager admission additionally requires an enabled eligible decision and a promoted source. |
| `begin_continuation(request_id, admitted=True)` | Start at most one dependent future for the verifying parent. A denied capacity admission, ineligibility, EOS parent or insufficient remaining candidate budget creates no work. |
| `record_continuation_completion(work, completion)` | Validate and account generated branch evidence. Wait for the parent, make a confirmed matching continuation promotable, or retire a late invalid/cancelled/released result without reactivation. |
| `record_continuation_abort(work, generated_tokens, frontier)` | Account physically completed partial work after its adapter has stopped and fenced it; require the existing cancelled/invalidated/released dependency and prevent reactivation. |
| `resolve_parent_verification(ParentVerification(...))` | Apply one authoritative, validated Target delta, advance prefix version once, classify accepted/correction/bonus tokens using `dual_greedy_acceptance`, and resolve or discard the dependent continuation. |
| `promote_continuation(request_id, continuation_id)` | Install the confirmed branch as a new legal proposal whose candidates exclude the bridge. Promotion cannot be repeated. |
| `discard_continuation(request_id, continuation_id, reason)` | Preserve the discard reason, clear speculative payload and require normal recovery when needed. A promoted proposal has its own lifecycle and cannot be discarded through this API. |
| `finish_or_cancel(request_id, reason)` / `shutdown()` | Stop admission, clear speculative and KV payloads, retain committed audit state plus ID/digest tombstones, and prevent late notifications from reopening the request. Shutdown also rejects new registrations. |

`register` treats the supplied prefix as the already-established boundary:
`max_output_tokens` is the **additional** output allowance after registration,
and `Accounting.committed_tokens` counts subsequent Target deltas. An adapter
attaching after bootstrap must pass the remaining request budget, not the original
maximum. `committed_prefix_version` can be initialized from the existing owner
and thereafter increments once per accepted Target receipt.

`DraftWork` binds owner, request, unique work ID, parent proposal ID, prefix
version, dependency tokens, candidate length, normal/recovery/continuation kind,
base KV prefix, decision version and EOS IDs. `DraftCompletion` supplies the work
ID, generated tokens and materialized frontier; the original work is supplied
alongside it and compared against its retained digest. `ParentVerification`
separately binds owner, request, proposal ID, parent version/content and the
authoritative delta/terminal reason.

## Rolling and recovery protocol

Normal drafting always starts from the authoritative committed prefix.
Verification and dependent Draft work can coexist for the same request:

| Opportunity | Target role | Draft role | Resolution |
| --- | --- | --- | --- |
| 0 | Verify normal proposal P0 | Generate E1 using P0 and a predicted bridge | Full acceptance plus exact bridge/dependency match promotes E1. |
| 1 | Verify promoted E1 | Generate E2 | Match promotes E2. |
| 2 | Verify promoted E2 | Generate E3 | Rejection invalidates E3 and records recovery required. |
| 3 | Verify other eligible requests | Normally draft R3 from the corrected committed prefix | New independent proposal ID becomes ready. |
| 4 | Verify R3 | Generate E4 | Static eager membership survived rejection; match promotes E4. |
| 5 onward | Verify each promoted proposal | Generate its next dependent continuation | Rolling continues until a mismatch, switch boundary or terminal result. |

Parent rejection and a full-acceptance bridge mismatch have separate discard
reasons. Both discard future dependent work and restore the true Target prefix.
The old parent cannot re-enter verification. Recovery receives normal Draft
admission before this request can supply another proposal; other requests remain
free to use Target while it recovers.

Promotion is a guarded transfer: the request must be live; request, owner, parent
proposal and parent prefix version/content must match; the parent must be fully
accepted; the Target bonus must equal the predicted bridge; data must be complete;
and no prior promotion/consumption may have occurred. The new proposal's prefix
is the actual parent commit including its bonus, and its candidates exclude the
bridge. Promotion does not commit any tokens.

## Bonus bridging and KV frontier

Let `C` be the committed prefix, `L = len(C)`, and `P = (d1,d2,d3,d4)`.
Before continuation, ordinary Draft KV covers `C,d1,d2,d3`, with frontier `L+3`.
The logical proposal also contains `d4`, but sampling it did not materialize it.

To predict the next full proposal, the Draft owner must first materialize `d4`
and obtain its next-token logits. It then samples `predicted_bridge`, materializes
that bridge to predict `e1`, and similarly proceeds through `e4`. Complete work
has this layout:

```text
logical tokens: C | d1 d2 d3 d4 | predicted_bridge | e1 e2 e3 | e4
Draft KV:      C | d1 d2 d3 d4 | predicted_bridge | e1 e2 e3 |
frontier: L + 8
```

When Target commits `d1 d2 d3 d4 b`, promotion requires
`predicted_bridge == b`. The retained KV now covers the complete new committed
prefix `C + P + b` and the first three promoted candidates. The final promoted
candidate `e4` remains unmaterialized, exactly matching the normal four-candidate
frontier convention relative to its new prefix. No bridge token is emitted or
counted again in the next round.

If Target rejects at candidate index `k`, KV at most through `C + P[:k]` is safe,
bounded by the actual materialized frontier. Every later parent, bridge and future
candidate position is invalid. Normal recovery materializes the Target correction
from the retained prefix before generating a new candidate. On a bridge mismatch
after full parent acceptance, at most `C + P` is reusable; the predicted bridge
and everything after it must be invalidated and the actual bonus materialized.
Tail/terminal trimming keeps only canonical output. Termination releases the
request's Draft resources on their owner in the GPU adapter described below.

The current CPU core conservatively retains the common prefix of its **installed**
`materialized_kv_prefix` and the Target's committed prefix. Continuation completion
is separate immutable branch evidence: it does not overwrite installed KV while
its parent is unresolved. Thus a bridge-mismatch recovery may retain `L+3` from the
ordinary parent branch and rematerialize both `d4` and the real bonus, even though
discarded continuation evidence proved that `d4` was computed. Promotion is the
only operation that installs the matching continuation branch. This distinction
allows a delayed completion to be validated after normal recovery without
overwriting the recovered proposal or its frontier.

The CPU core itself expresses frontier validity and suffix work; it does not allocate
paged KV, execute a model, or prove GPU block reuse. `cpu.generate` completes a
private immutable branch model. A production GPU owner must serialize the real
Draft operations, fence or cancel pending writes before rollback/recovery/release,
and return evidence only after the physical frontier is established. A late CPU
completion is not permission for a physical worker to write old KV after recovery.
Existing `DraftCommitPlan` assumptions cannot be silently weakened.

The implemented accounting fields have these boundaries:

| Fields | Counting point and meaning |
| --- | --- |
| `committed_tokens`, `parent_accepted_tokens`, `correction_tokens`, `bonus_tokens` | Target receipt resolution only. `committed_tokens = parent_accepted_tokens + correction_tokens + bonus_tokens`; duplicate receipts add nothing. |
| `normal_generated_tokens`, `recovery_generated_tokens`, `recovery_jobs` | Validated normal-work completion counts all ordinary candidates; recovery candidates are a subset following failed/discarded work. Recovery jobs count issued recovery work once. A normal all-accepted round with eager disabled or deferred remains normal work. |
| `early_generated_tokens`, `bridge_generated_tokens` | Validated continuation completion counts all early tokens, with bridges counted separately as a subset. Early candidate generation is their difference. Late discarded completion is still generated work. |
| `reused_candidate_tokens`, `reused_bridge_tokens` | Promotion counts candidates made available for reuse and the matched parent bridge. These counters describe promotion, not Target acceptance or output, and may include a promoted proposal later cancelled before verification. |
| `discarded_early_tokens` | Counts completed early work invalidated, cancelled or released before promotion, once; if the result arrives late, the count occurs on that completion. |
| `draft_materialized_tokens` | Counts missing dependency inputs plus generated inputs that were actually fed to the deterministic executor, including the parent-last-token/bridge work. This is CPU branch work accounting, not measured GPU forwards or a speedup. |

A promotion never increments committed output. Physical materialization, token
generation, promotion and Target commitment are separate accounting events.

## Asynchronous results, termination and control boundaries

Draft completion and Target feedback may arrive in either order:

| First event | Intermediate meaning | Second event |
| --- | --- | --- |
| Draft completes | Generated continuation, parent still unconfirmed; not verify-ready | Matching Target result makes it promotable. |
| Target fully accepts | Parent confirmed, future Draft data still incomplete; not verify-ready | Matching Draft completion makes it promotable. |
| Target rejects or terminates | Dependent work is invalid/cancelled and cannot be promoted | Late Draft completion is accounted/retired without reactivating the request. |

The implemented `ContinuationStatus` values are `GENERATING`, `WAITING_PARENT`,
`WAITING_DRAFT`, `PROMOTABLE`, `PROMOTED`, `CONSUMED`, `INVALIDATED`, `CANCELLED`
and `RELEASED`. Draft-first completion enters `WAITING_PARENT`; Target-first full
acceptance enters `WAITING_DRAFT`. Neither state is a proposal ready for Target.
`PROMOTED` changes to `CONSUMED` when its derived proposal begins verification.

Identical repeated Draft completion or Target feedback is idempotent; conflicting
replay, stale parent version/content, cross-owner events and duplicate consumption
fail closed. Cleanup clears proposal tokens/parent prefixes, continuation
dependency/candidate/bridge payloads, active work and KV prefix, retaining IDs and
digests to identify late results. The committed prefix/accounting remain available
for audit. Original continuation `reason` and final `release_reason` are separate,
so bridge mismatch and parent rejection remain distinguishable after cleanup.
Late completion may add previously unreported work accounting, but cannot restore
token payload, change committed output or overwrite a recovered proposal.

The provider extension point is
`evaluate(request_state, step_context) -> eager_decision`. Stage 1's production
provider reads only a preconfigured stable-ID set; it does not inspect SLO,
progress gap, acceptance rate or adaptive request budgets. A test provider may
change decision versions to exercise future dynamic control safely.

The owner must call `evaluate_eager_eligibility` when delivering each switch change;
changing a test provider's fields alone is not event delivery. Versioned scheduler
intents must also be revalidated against a current owner snapshot immediately
before dispatch.

When eager is switched off, new early Draft work and new eager verification
admission stop. Already submitted legal Target verification completes normally.
Started Draft work is safely completed or cancelled; stale completion cannot
reinstate a previous decision. A promoted but unsubmitted proposal remains a
legal ordinary proposal and returns to normal cohort scheduling, with the same
tokens, prefix and identity. Re-enabling allows new eager work from the current
legal state; it never changes already committed tokens or revives invalid work.

The CPU scheduling contract deduplicates normal and eager candidates by stable
request identity and checks proposal/version evidence at actual admission.
Home cohort remains metadata; execution role identifies current Target verify,
dependent Draft continuation or normal recovery. The next-stage eager opportunity
does not create another request or another authoritative Target state. Actual
mixed-batch capacity, fairness and GPU allocations are deferred.

## Executable regression evidence

[`test_rolling_eager_protocol.py`](../tests/test_rolling_eager_protocol.py) drives
the real core using deterministic `cpu.generate` tokens and a token-by-token
Target oracle. `test_continuous_success_rejection_recovery_and_continuous_success`
runs both initial feedback orders and alternates them across six verifications:

```text
promote, promote, reject/recover, promote, promote, promote
```

It verifies seven unique proposal IDs and six unique continuation IDs, including
a new normal recovery proposal after the third parent rejects. Exact final
accounting is 28 committed tokens (22 accepted candidates, one correction, five
bonuses), 30 early generated tokens including six bridges, 20 promoted candidates,
five promoted bridges, five discarded early tokens and four recovery candidates
from one recovery job. This exercises repeated eager after recovery, not a single
eager opportunity or precomputed PASS flag.

The same suite covers bridge mismatch in both feedback orders, work/receipt
identity and dependency corruption, incomplete/frontier-corrupt completion,
duplicate messages and consumption, late rejection results after recovery,
owner/request isolation, real owner-thread checks, switch delivery, promoted
normal fallback, KV gaps, EOS at parent/bridge/future candidate positions, short
output budgets including a zero-candidate tail, cancellation and owner shutdown.
[`test_rolling_eager_scheduling.py`](../tests/test_rolling_eager_scheduling.py)
tests static identity eligibility, cross-A/B next-stage admission, deduplication,
changed proposal/version rejection, simultaneous Target/Draft roles, recovery
while another request verifies, terminal filtering and disabled baseline ordering.
[`test_rolling_eager_adapter.py`](../tests/test_rolling_eager_adapter.py) then drives
those selectors through the actual owner adapter: capacity deferral and cross-A/B
rolling, rejection/recovery while another Target role runs, stale intents,
switch-off cancellation and normal fallback, late work after owner shutdown,
ordinary multi-round drafting with zero recovery counts, and recovery-first normal
Draft capacity.
[`test_rolling_eager_failures.py`](../tests/test_rolling_eager_failures.py) covers
provider exceptions, invalid decisions and version violations: failed registration
leaves no request behind, and failed receipt resolution remains retryable without
partially committing tokens.

The repository's full pytest/ruff gates include existing baseline tests. Its CI
matrix executes Python 3.9 and 3.12; the existing Python 3.11 contract and pinned
vLLM source jobs remain separate. Delivery reports the actual local and CI status;
this design document does not substitute CPU protocol evidence for GPU qualification.

## Stage 2 connected execution path

`serial-eager` is an explicit optional fixed/decode-scan mode. Preparing a new
resident360 root seals optional eager points, while the default scan still selects
only its original baseline points. Explicit `--mode serial-eager --batch 16`
routes through `decode_scan_cli` / `fixed_cli` and `s2_cli` into the existing fixed
runtime. No policy in the simulator or existing Target/Serial/PingPong mode changes
its default. The [operator runbook](rolling-eager-gpu-runbook.md) selects only B16.

The connected call chain is:

1. [`serving/fixed_draft.py`](../src/specrhythm/serving/fixed_draft.py) selects
   [`eager_draft.py`](../src/specrhythm/serving/eager_draft.py) only for the eager
   mode. Its `EagerOwner` constructs one `EagerSerialMachine` and one
   `EagerFixedDraftBackend` on the existing Draft owner thread. The backend combines
   `GPUContinuationBackendMixin` with the existing observed `FixedDraftBackend`;
   it retains the same model, private allocation and device identity.
2. The existing pinned worker patch invokes `on_target_verify_start` before
   `_model_forward`. The audited insertion at original runner line 4334 follows
   input preparation and precedes the ordinary forward-context/model call.
   [`EagerSerialProposer`](../src/specrhythm/serving/eager_proposer.py) first uses
   `S2SerialProposer` to install the initial admission-created proposal and validate
   existing TP identity. Target rank zero then freezes and sends request/proposal
   IDs, round, full candidate tokens, parent length and hash. Other TP ranks never
   submit duplicate Draft work. A TP barrier covers enqueue acknowledgment before
   any rank enters the Target forward; it does not await Draft completion.
3. `EagerOwner.call("eager_enqueue", ...)` deep-copies the payload into its mailbox
   and returns immediately. The socket thread neither changes the shared protocol
   nor touches GPU KV. The owner validates the frozen proposal and calls the shared
   core's `start_verification` / `begin_continuation`, then enrolls dependent work.
4. [`eager_owner.py`](../src/specrhythm/serving/eager_owner.py) drains queued feedback
   and control messages before its next bounded token step. Each
   `step_gpu_continuations` issues one actual batched materialization/sample/fence
   across active requests. A launched write reaches its fence before the owner can
   roll back or release any affected KV. This allows queued rejection, cancellation
   and switch-off to stop remaining speculative steps safely.
5. Target's inherited authoritative result/proposer boundary sends immutable
   synchronization rows. [`eager_machine.py`](../src/specrhythm/serving/eager_machine.py)
   uses `RollingContinuation` to resolve them once, including true EOS/length
   trimming. Matching complete continuation is promoted; rejection or bridge
   mismatch selects normal recovery from the actual prefix. An accepted parent
   whose continuation is incomplete may wait for the remaining bounded steps.
   Its actual unhidden wait interval is recorded and included in decode time.
6. The machine returns a mixture of promoted, normally recovered, ordinary and
   legal tail results. Only current legal proposal evidence can reach the next
   corresponding verification. Verification-start repeats the same procedure,
   including after recovery. The Target remains the sole committed-output authority.

The eager proposer uses an explicit timeline type because its next Draft start
can precede parent verification completion. It preserves actual phase intervals
and readiness checks instead of forcing them into the original Serial
non-overlap ordering. Target sampling and the existing worker patch stack remain
unchanged.

### Physical continuation, repair and cached logits

[`gpu_backend.py`](../src/specrhythm/continuation/gpu_backend.py) provides an opt-in
`GPUContinuationBackendMixin` and standalone `RollingVllmDraftBackend`. It does not
weaken ordinary `commit_frontier`. `begin_gpu_continuation` validates parent,
version, candidate budget, current last-token gap and model capacity but launches
no model work. `step_gpu_continuations` materializes the missing parent-last-token
position, then bridge/candidate inputs in successive steps, using the existing
`VllmDraftWorker.materialize`, `greedy` and `fence`. Completion tokens/frontiers
therefore come from real worker operations, not the CPU executor.
Backend `eager_enrolled` and runtime `admissions` count registration only;
`eager_started` / runtime `started` increment after the first physically fenced
forward/greedy step. A branch cancelled before its first step has no start count.

Unlike the stage-1 independent branch model, real continuation extends the same
private request allocation. The committed prefix and parent proposal fields stay
frozen while its physical frontier advances. The GPU adapter's per-work progress
records are the authority for that extended frontier. On promotion,
`rebase_gpu_parent` requires complete work, full parent acceptance, exact bridge,
complete parent dependency and exact promoted tokens. It retains the existing
allocation, frontier and cached logits, and installs the new prefix/proposal.
The cached logits still describe the position immediately before the final
promoted candidate, matching the ordinary sampled-last-token convention.

On rejection or bridge mismatch, the owner first fences/aborts the branch. Repair
retains only the safe accepted parent prefix, clears invalid deeper cached logits,
and materializes the missing correction or actual bonus suffix to rebuild logits
before normal drafting. If no tail supplied fresh logits at a shortened boundary,
the boundary input must be recomputed. Full-prefix prefill is not used for the
live request's recovery. Physical blocks above a shorter logical frontier remain
private high-water capacity: the existing streaming rebase overwrites invalid
positions before reuse. Request release fences the worker and frees its entire
allocation, including high-water blocks. Logical truncation is not falsely
reported as immediate physical block reclamation.

`record_continuation_abort` accounts an interrupted branch's actually generated
tokens/frontier without requiring a complete K4 result. Started GPU writes must
have crossed the worker fence before this evidence is returned. Duplicate work,
completion and settlement are identified by bound IDs/digests; old completion
cannot replace a recovered proposal or its physical prefix. The physical adapter
binds the core prefix version to its existing round counter when attaching a
request; it does not confuse that counter with bootstrap output length.

### Drain and reporting

Fixed runtime preserves the original setup, warmup, decode window and drain
boundaries. The eager server passes the coordinator's single absolute drain
deadline through settlement, shutdown, owner join and buffered log finalization.
Its diagnostic settlement first resolves pending authoritative deltas and safely
aborts outstanding continuation before delegating final private-KV release.
Normal completion and refill still require physical resource-release receipts;
replacements use their own stable identity. Ledger removal follows physical
settlement. An owner thread or worker that misses the deadline remains a failure,
even if the outer supervisor subsequently kills its owned processes.

[`eager_results.py`](../src/specrhythm/serving/eager_results.py) joins retained
counter deltas into lifetime and measured-window reports. It separately reports
admission, actual start/completion, full accepts/rejections, bridge outcomes,
promotion, promoted candidates actually verified, promoted candidates accepted,
early discard, normal recovery, bridge/materialization work and committed
accepted/correction/bonus counts. Promotion is never substituted for output.
Window event counters use completion/update timestamps; unhidden wait uses the
actual interval clipped to the measured window. Throughput retains the existing
coordinator's measured committed tokens and actual elapsed window.

Actual eager Draft device intervals are recorded by the same `FixedDraftBackend`
device timeline as existing work, then correlated with Target device intervals.
Only a positive lower bound from native CUDA clock bounds is reported as observed
GPU overlap. Host enqueue or call overlap by itself remains `UNKNOWN`. Missing
natural promotion, consumption, accepted tokens, rejection or bridge mismatch is
`NOT_OBSERVED`; no automatic scan expansion manufactures those events.

[`gpu_check.py`](../src/specrhythm/continuation/gpu_check.py) is the additional
operator-only physical Draft regression. It runs the real backend/shared-core
path, compares generated/promoted/recovered candidates with independent ordinary
Draft reference KV, checks the existing physical page-table/frontier metadata and
requires complete resource release. Its controlled receipts are explicitly
`INJECTED_DIAGNOSTIC`, distinct from natural Target outcomes. Real EOS can prevent
a bounded construction, which is then `NOT_OBSERVED`. Only diagnostic reference
allocations replay prefixes, outside throughput measurement. The CLI supervisor
owns one child process tree by PID/start identity, offers same-root status/errors/
controlled stop, and retains the first failure. Read-only commands initialize no
GPU. CPU harness tests exercise the same new backend coordination with substituted
device operations; they are not a GPU qualification result.

## Integration boundaries and remaining PingPong work

The stage-1 integration map remains the boundary checklist below. Stage 2 connects
the Serial/fixed subset through the modules above. The PingPong mixed Target batch,
cross-cohort GPU admission and associated fairness/capacity policy remain deferred:

| Modules | Required integration |
| --- | --- |
| `phase4/vllm_remote.py`, `phase4/resident_scheduler.py`, `serving/s2_proposer.py`, `serving/s2_scheduler.py` | Serial verification-start submission, authoritative-result resolution, ready promotion and normal recovery/tail admission. Preserve current serial default behavior and timing/commit authority. |
| `phase4/vllm_dual.py`, `phase4/vllm_dual_scheduler.py` | Carry proposal/continuation IDs and parent prefix versions in immutable messages; resolve Target output once; publish only validated promoted proposals; retain stale/terminal message checks. |
| `phase4/vllm_pingpong.py`, `phase4/vllm_pingpong_scheduler.py`, `serving/s2_scheduler.py` | Separate home cohort from work role; permit a valid promoted request in the next eligible verify stage and merge/deduplicate normal/eager candidates before stock Target allocation. |
| `phase4/dual_service.py`, `phase4/dual_batched_draft.py`, `phase4/dual_pingpong_draft.py`, `phase4/batched_draft_service.py` | Separate current proposal and dependent continuation work/mailboxes, maintain owner-thread mutation, coordinate late messages, cleanup and shutdown, and avoid automatic publication of unresolved futures. |
| `phase4/vllm_draft_backend.py`, `phase4/draft_batch.py`, `phase4/vllm_draft_worker.py`, `serving/s2_draft.py`, `serving/fixed_draft.py` | Add explicit dependent-generation/materialization and truncate/rebase plans for extended speculative KV; retain private request blocks and verify position/frontier evidence. |
| `serving/fixed_settle.py`, `serving/s2_runtime.py`, `serving/fixed_runtime.py` | Drain both request roles and outstanding owner work before releasing slots or recording completion; bind refills to their own stable identities. |

Target sampling, EOS/length behavior, TP identity checks, stock Target budgets and
KV ownership remain their existing authorities. The shared core, immutable
owner messages, incremental physical backend, safe repair and versioned eligibility
interfaces are reusable by the next PingPong stage. Dynamic urgency, per-request
budget adaptation and Shaping remain outside scope. GPU correctness, overlap and
performance are PENDING until the operator returns server evidence.
