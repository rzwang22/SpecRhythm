# Unified K3 (explicit experimental modes)

`serial-k3`, `pingpong-k3`, `pingpong-eager-k3` use three actual candidates,
including the seed, and the existing no-bonus acceptance protocol. Old prepost3
remains P1/P4. Active16, immutable homes8/8, one TP2 Target ceiling8, one Draft.

Initial committed prefix → sample one cached-logit seed → two fenced extensions
→ READY K3. Initial/refill seeds and prefill are separate costs. A claimed proposal
is immutable. Eager generates at most min(3, remaining minus parent length) tokens
from the hypothetical fully accepted parent. Each step yields to queued feedback.
Rejection cancels that dependency, fences writes, rolls back and materializes the
correction; its logits provide candidate one. Two normal extensions complete K3.
Full acceptance retains only the checked dependency: three retained tokens already
form K3, so no fourth-token forward is issued. EOS or budget1/2 permits a tail.
READY is published only after the required candidate count and physical frontier
checks. Committed prefix never includes unverified candidates or an extra bonus.

Accounting distinguishes cached initial seeds, post-verification seeds, normal
extensions and eager candidates. Initial seed + two extensions counts as one
three-candidate proposal. Lifetime candidate conservation is checked separately:
generated = submitted at the validated verify-start hook + discarded conditional
candidates + discarded unsubmitted candidates, once all work is retired. Repeated
settlement cannot add discards twice. Native association independently proves Target
execution; submitted candidates are neither accepted tokens nor committed output.
This lifetime check includes initialization/drain and is never summed with the
measurement-window counters. Private incomplete candidates cancelled before READY
are counted as discards; they never become short Target proposals.

Both PingPong modes use the same stable-home admission policy. Other-home normal
and recovery work share compatible token batches with lookahead, without waiting
to assemble a batch. Serial uses the same code and ceiling but an explicit idle
Draft gate before Target, and never enrolls conditional continuation.

Pipeline investigation: the previous coordinator performs two synchronous status
RPCs and then a synchronous admission RPC before Target. The owner begins a token
step after replying to each command. Thus these three reads can serialize a seed
and two/three extensions even when another home is READY. K3 status reads use an
immutable owner-published snapshot, and admission alone claims current authoritative
state at a fenced boundary. Status is informational, never proof of READY or KV
completion. Snapshot is replaced after every command and token step; admission,
feedback, control and drain still run on the owner. GPU opportunity and actual
overlap for K3 require server evidence. The retained old package independently shows
the next home was already READY in all77 measured adjacent transitions, yet two
status queue waits and an admission boundary preceded Target dispatch; all four
old Draft forwards completed before the next Target GPU interval. The zero-overlap
statistic alone was insufficient; the associated raw execution ordering supports it.

Example: A K3 validates while B extends → B READY/claimed while A correction seeds
→ A recovers K3 during B validation → A accepts, reuses lookahead K3 → repeats.
No rejection creates a whole-cohort recovery barrier in either PingPong mode.

GPU output correctness, pipeline coverage, native overlap and performance: PENDING.

## Implementation and evidence boundaries

The shared PrePost protocol now exposes defaulted length/protocol hooks. Only
`K3Machine`/`K3BackendMixin` select3; old classes still select4 and retain their old
P1/P4 branches. Initialization queues two ordinary token steps after cached seed
sampling. Publication checks the actual physical proposal and remaining budget;
Target scheduler and the pinned bookkeeping projection independently reject an
unexplained short candidate list. The engine actually requests3 speculative tokens.
All commits still pass the existing no-bonus acceptance, prefix/version and ownership
checks. Full/runtime allocator regressions reconcile the complete pool and observe
no new full scans in unchanged runtime token steps.

`K3Owner` publishes a fresh informational status object after factory completion,
every dispatched command and every fenced token step. No cache holds a permanent
"checks passed" result. Snapshot contains no cached control decision. Consumers can
inspect population, but only `pp_admit` may mutate READY/claim; `k3_idle` (Serial),
feedback, stop, rollback and release stay owner commands. Copies are thread isolated.
A synchronous admission can still wait for the currently executing physical step:
that necessary boundary is retained and must be measured. Normal/recovery/eager
rows share a batch immediately when compatible; there is no timer waiting for a
larger batch. Every finite token step returns to queued control/feedback.

For the prior zero-overlap observation, a confirmed implementation serialization
mechanism is repeated status RPCs routed through that same busy owner queue. The
CPU event-coordinated regression proves the old status cannot return during a held
write, the new status can, and B is claimed before A's full recovery. It also proves
A reaches READY before B feedback. It does **not** prove those RPCs alone explain
all historical GPU dispatch latency. The locally retained c0ecc2a raw archive was
subsequently verified and its77 transitions confirm the ordering above; see
[timing derivative](k3-baseline-observations.json). One A→B example, relative to
A feedback, has status enqueue/dequeue9.808/52.179ms and54.817/93.048ms, Bclaim127.138ms,
last A Draft native end upper160.413ms, and B Target start lower233.838ms. B's batch
was already READY before A feedback. Those overlapping intervals are not additive.
Remaining delays include the one required admission boundary, Target
pre-forward work/control publication, lack of other-home legal work and competing
Draft compute. The server report retains waiting inventories and native intervals
to distinguish them without delaying Target or discarding window time.

Serial explicitly waits for all finite Draft work before claiming the next ceiling8
batch. PingPong removes only that Serial gate; both PingPong variants use the same
home selection and informational status implementation. Eager only enables dependent
continuations; it does not add promotion priority. No old mode's scheduling changes.

## Capacity contract and startup failure cleanup (2026-09-14)

The first K3 server attempt (`b2a2d29210d68357590a567d4819f3552175d179`)
failed before preparation: `fixed_runtime.run()` initialized Target, obtained actual
rank capacities, then called `s2_plan.capacity_for(speculative_tokens=3)` for Target.
The shared legacy function deliberately requires an integer reserve of at least4.
The conflict was in the caller's allocation contract, not in K3 acceptance or CUDA.

`k3_capacity.reservation()` now separates logical demand from conservative allocation:

| Mode / resource | Actual candidate limit | Current + conditional demand | Legacy minimum | Final reserve |
| --- | ---: | ---: | ---: | ---: |
| All three K3 modes, each Target TP rank | 3 | 3 + 0 | 4 | 4 |
| Serial / ordinary PingPong Draft | 3 | 3 + 0 | 4 | 4 |
| Eager PingPong Draft | 3 | 3 + 3 | 4 | 6 |

A prompt/committed prefix of length C holds a logical K3 proposal while Draft KV
covers C+2; the final sampled candidate is not yet materialized. Three lookahead
forwards consume that final token and then two lookahead tokens: KV reaches C+5,
with six logical pending candidates total. Successful reuse advances committed C
by3, leaving a complete K3 with the same last-token frontier. It does not sample
candidate four. Rejection fences outstanding writes and truncates invalid slots;
correction replaces the rejected position. One catch-up materializes the corrected
committed frontier and supplies next seed logits, then two extensions complete K3.
Correction is committed output, not an additional simultaneously live speculative
position. Tail budgets/EOS only reduce these bounds. Target's root is already part
of C; root+K3 gives four query positions but only three speculative positions.
The model-context guard's extra cursor margin is unchanged; it is not a generated
candidate or a materialized seventh speculative position.

`capacity_for()` still reserves maximum output growth, including the initial
committed seed, and rounds at physical block boundaries. It retains the private
partial/copy block per active request, max(32 blocks, 5% capacity) safety reserve,
512MiB additional workspace, all resident Draft float32 logits and actual free-memory
validation. The conservative `5 * target_ceiling` query envelope is also unchanged:
it is capacity headroom, not candidate generation. No hardware reserve is reduced.
Old modes call the unchanged capacity function with their original4/7/9 arguments
and retain the old sparse fields. New K3 checks explicitly record
`candidate_length`, `required_speculative_positions`, `reserved_speculative_positions`,
`legacy_minimum_reserve`, reserve-based `speculative_capacity_tokens`, and nonnegative
`extra_speculative_tokens` (0 or2). Target num_speculative_tokens stays3.

`fixed_plan.capacity_metadata → fixed_runtime.run → k3_capacity.check → capacity_for`
is the production path. The capacity report also retains raw per-request prompt and
maximum-output budgets. `fixed_results`, `decode_scan_results` and the device contract
recompute block/workspace arithmetic from those budgets and the original three rank
observations, reject missing/wrong types or conflicting declarations, and compare
against the frozen workload when available. The bounded exporter retains these raw
fields without inventing them. Historical packages are never backfilled. The static
`python -m specrhythm.serving.k3_capacity` preflight exercises all six mode/resource
interfaces before model initialization; it explicitly reports GPU_capacity=PENDING.

For exceptions after entering K3 model startup but before `drive()`, the coordinator
previously shut down only its Target handle. It did not send the idle Draft service
an orderly shutdown. `StartupCleanup` now uses the existing drain deadline to request
the actual owner shutdown, requiring empty pending work, zero physical live requests
and a shutdown receipt, then closes the Target engine. Per-call sockets close through
the existing transport context manager. No retry or process-wide kill is added.
The helper is idempotent, records primary and secondary errors separately, and publishes
its phase deadline for the existing owned-process supervisor. Normal drain and all
old modes remain unchanged.

An acknowledged owner and returned Target shutdown API are not process cleanup PASS.
A model constructor may fail before returning a handle; owned subprocess cleanup then
still depends on the supervisor. The pinned in-process vLLM client accepts a timeout
but does not forward it to every internal shutdown call; the existing external drain
supervision remains necessary. Configuration failures before entering model startup
also remain supervisor-managed. These limits are explicit in startup-cleanup.json;
actual descendant exit, no stale owner/socket, and final cleanup qualification still
require new server evidence. The original capacity exception remains the command's
first failure even when shutdown or recording fails.

## K3 dispatch follow-up: verified 778d run (2026-09-14)

The preceding status-RPC diagnosis concerns c0ecc2a/P1-P4. It is **not** the cause
of the remaining K3 zero overlap after the owner snapshot fix. The returned
`pingpong-k3-delivery-20260914T062632Z-1634.tar.gz` verifies all145 logical objects,
capacity, joint outputs, execution, measurement and cleanup. Its original status and
46.55 / 67.64 / 71.39 tok/s single windows remain unchanged. Native joins confirm
ordinary/recovery cross-request overlap0 in all modes; eager/parent overlap is
6860.925561–6904.568683ms. These observations do not establish stable speedup.

[Read-only derivative with source hashes, request/proposal versions and original
forward indices](k3-pipeline-778d-observations.json) contains the first rejection
cycle and aggregate dispatch decomposition. In ordinary PingPong, all111 measured
claim→Target intervals average99.055ms; scheduler accounts for84.087ms. Its inclusive
`prefix_hash_and_block_record` intervals occupy37.945ms and the separate full block
checks7.911ms. About38.231ms inside scheduler remains outside those two categories,
and14.969ms of dispatch lies outside scheduler. The prefix category includes token
validation, JSON encoding, digest and block-record construction: it is not a pure
hash benchmark. JSON spans nest inside it. These per-lane quantities neither include
all wall time nor imply that removing37.945ms would yield a specified throughput.
The old run lacks finer scheduler phase spans; missing fields remain null.

The first measured A→B cycle, relative to A rank0 start lower bound
7758704266688304ns, illustrates the actual ordering:

| Actual boundary | Relative ms | Source / interpretation |
| --- | ---: | --- |
| A Target native interval | 0–77.100 | TP0; TP1 separately retained |
| A result available | 117.366 | Target rank0 feedback hook |
| A feedback RPC | 118.583–121.057 | inclusive RPC, not added to owner work |
| Owner processes A feedback | 119.201–120.714 | `pp_feedback` dispatch |
| A public Draft repair/seed | 125.656–158.055 | native interval |
| Two coordinator status RPCs | 129.383–129.661;130.167–130.378 | snapshot reads, already fast |
| B admission RPC / owner dequeue | 130.780–166.000 /164.414 | waits for current fenced token only |
| B authoritative claim | about164.450 | B was already READY |
| A two extensions | 169.585–199.492;205.828–228.258 | native intervals |
| B scheduler | 175.326–255.991 | full resident snapshot/check twice |
| B eager-registration RPC | 260.466–260.974 | mailbox ACK,0.508ms |
| B Target native interval | 260.188–339.069 | calibrated native bounds |

The native CUDA anchor has uncertainty; the projected lower GPU endpoint may slightly
precede the host hook. It does not establish GPU execution before the enqueue barrier.
The complete IDs/versions and both native endpoint bounds are in the derivative.
A's version2 proposal `sr-c6fd4c5bc623ca136c924dbe915c49c1837c97e1e42c6e9a1236021c9819828e:prepost:2`
rejects after one accepted candidate. Its correction and two extensions produce
version3 READY; B has entered its own claim while that recovery remains in progress.
The scheduler subsequently consumes the available overlap opportunity on CPU.

Source conclusions:

| Path | Actual dependency | Treatment |
| --- | --- | --- |
| `fixed_runtime.drive` → `K3Owner.status` | informational population copy | Existing snapshot retained; no new wait-idle in PingPong |
| `PingPrePostController.select` → `pp_admit` | authoritative once-only claim at current Draft write fence | Retained; no unfenced backend access |
| `PoolScheduler.schedule` → `physical_rows` → `ResidentPoolAudit.check` twice | all live KV/frontier/ownership; repeated immutable prompt digest | Reuse only validated immutable prompt digest |
| `PingPrePostProposer.on_target_verify_start` | this claim's prefix/version/candidates | Retained; no wait for another request's full proposal |
| `EagerSerialProposer` → `EagerOwner.call(eager_enqueue)` | bounded mailbox acceptance, TP ACK barrier | Already nonblocking with respect to Draft execution; not split again |
| `EagerOwner._run` → `machine.step` | single token step, then queued control/feedback first | Retained; no concurrent backend thread |
| `fixed_runtime.drive` Serial `k3_idle` | deliberate all-Draft idle gate | Serial comparison retains serial execution |

There is no evidence that the B verify-start callback waits for all A recovery,
that eager registration waits for GPU start, or that a global PingPong idle gate
causes this run. The confirmed repeated work is in Target's dispatch path. Other
resident scheduling, serialization, sampling and worker launch costs remain; no
unmeasured residual is assigned to GIL, RPC or fence.

### Narrow execution change and proof invalidation

`PingPrePostScheduler.physical_rows` selects `k3_prompt_proof` only for the three
explicit K3 modes. First observation records `(internal ID, immutable token tuple,
validated digest)`. Every subsequent observation compares the **entire current
prompt**, exact integer types and internal identity, then reads the **current**
materialized frontier and every current block list. It does not cache rows, KV,
allocator ownership, decisions, control files or a permanent checks-passed flag.
Changed prompt/binding fails; finished/removed handles evict the proof. New handles
hash again. Both full `ResidentPoolAudit.check` calls remain synchronous: private
block uniqueness, cross-request conflicts, initial prefix retention, resident states,
frontier and peak/check counters retain their original behavior. Frozen/control
identity checks still run. New CPU negatives reject corrupt prompt, bool/float token,
duplicate blocks, missing/evicted resident and invalid frontier.

This removes repeated normalization-list/JSON/hash work for immutable prompts. It
**does not remove the all-pool KV scans**, change audit mode, batch size, sampling,
logging buffer or protocol. The common optimization applies equally to all three K3
modes. Serial's scheduling policy is unchanged; paired Serial must be remeasured.
Old non-K3 modes use the old snapshot producer. K3 demand3/reserve4 and eager Draft6,
initialization cleanup and strict report/device contracts remain unchanged.

### Observability and CPU proof boundaries

Bounded causal spans now cover Target pre-schedule pool work, stock resident
scheduling and post-schedule pool work. Each snapshot records proof policy v1,
resident count, new hashes, reused proofs and current tokens compared. No disk write,
GPU query, fence, observer lock or extra control read is added by these spans.

`k3_evidence` first binds each Target native forward's internal request IDs to the
actual scheduler rows, then to claim proposal/version and both TP ranks. Draft
physical records bind native forwards, request/work versions and parent claims.
Only then are cross-request roles and homes classified. Mixed physical batches can
participate in more than one role; role event sums and overlaps are not additive.
Parent-eager, cross-request normal/recovery, cross-home and Draft union outside every
Target are reported separately. Missing native/request/parent/rank evidence makes
integrity INCOMPLETE and overlap null, never zero.

`pipeline.dispatch` retains actual READY publication as well as the earlier physical
proposal completion, per-claim scheduler boundaries, native starts and phase spans.
GPU-idle time while a subsequently claimed proposal is READY does not prove the CPU
sampler/dispatcher is free. The bounded rejection example includes READY, publication,
claim, TP forwards, validated owner feedback, ordinary/recovery forwards and next
READY. Host timestamps and calibrated CUDA bounds remain separate. Detailed rows
appear once in the point report; comparison JSON embeds only aggregates. Raw bounded
producer records remain available for rejoining; caps and missing-row checks stay.

The new event-coordinated regression runs actual controller, owner, scheduler and
verify-start/end adapters with a controllable Draft worker and CPU Target boundary.
B reaches execution while A's second extension is deliberately held; A then reaches
READY during B verification, or after a faster B, depending on the tested order.
No sleep or elapsed-time threshold proves it. Restoring the old snapshot producer
fails all four variants at the duplicate-hash structural assertion, while the
interleaving assertion itself also passes on the old producer when GPU work is held.
This distinction matters: there was a CPU dispatch bottleneck, not an unobserved
whole-proposal wait. Earlier tests stopped at claim/owner and missed that bottleneck.
The regression proves safe interleaving, not native GPU overlap or speedup. New GPU
cross-request pipeline, native overlap and performance remain **PENDING**. A returned
zero-overlap run must remain NOT_DEMONSTRATED and be explained from its actual
READY/claim/phase/forward sequence, even if throughput rises.

## Resident dispatch normalization (2026-09-14 follow-up)

Source baseline: execution `8a3aca007fb60f9f67d1dcb7bedb5f8b78b8602e`, preserved delivery
`8b8a2d5fb9ca16f35756429feb19214d544520d6`. The read-only observations from the
082831Z-1891 archive are recorded separately in `k3-resident-8a3aca-observations.json`.
All three latest baseline runs and their output/diagnostic qualifications remain valid.
52.54/75.72/79.70 tok/s are single windows, not a stable speedup claim.

The exact MRO is PingPrePostScheduler → FixedBatch → PoolScheduler →
ResidentSetupScheduler → pinned vLLM Scheduler. The existing
`target_resident_stock_schedule` wraps the ENTIRE ResidentSetupScheduler call. Its
34.6 ms baseline mean includes the following work and is not the vLLM body alone:

| New child span | Production work | Checks / side effects retained |
|---|---|---|
| `target_resident_binding` | Current full token row normalization; identity matching | table-key/internal ID equality, complete current prompt comparison, both binding maps, alias/history checks; whole generated suffix still converted |
| `target_resident_readiness` | setup-ready/deferred refresh; initial lifecycle prepare | existing disk readiness and control semantics, no cached new decision |
| `target_resident_decisions` | live resident decisions and initial proposal validation | claim parent hash/length, K3 budget/EOS, spec tokens installed, same decisions |
| `target_resident_stock` | direct `super().schedule()` into pinned Scheduler | stock allocation/selection and dynamic admissibility callbacks; not isolated callback-free vLLM internals |
| `target_resident_initial_finish` | initial lifecycle completion | existing one-time initial proposal accounting |
| `target_resident_admission_records` | sorted construction and append for every decision | identical fields, row count, ordering and writer; no reduced log volume |

These six spans are disjoint children on one host PID/thread, nested in the existing
wrapper. Prefix/hash/JSON subspans remain inclusive and cannot be added to them.
Wrapper time minus the child interval union is reported as unaccounted. No device
synchronization, GPU query, per-token event, new trace lock or disk write is added.
Six rows per scheduler call use the existing phased budgets. Work counters report
live rows, successful full-row token conversion visits, normalized rows and decision/
admission rows. They do not claim to count token reads in other phases. The baseline
has no such subphase records, so individual baseline child times remain unknown.

The confirmed duplicate is narrow: `_bind_requests` converted every item to int and
built a tuple; BoundPromptIdentityMap converted that already normalized tuple again.
A full-match fallback converted it once more. K3 now constructs a private immutable
`_NormalizedTokenRow` from the CURRENT raw input once and passes it directly through
binding. Its constructor performs exactly the same int conversion over the full row,
including generated suffix; it does not merely validate the prompt or accept a caller's
PASS flag. Matching can reuse this call-local normalized tuple. Current complete prompt
comparison is still performed. New binding, changed identity and non-prefix-free prompts
retain full matching and its ambiguity/no-match failures. Ordinary public sequence
inputs retain the previous normalization/matching path, including non-K3 callers.

The input object is neither retained by the scheduler nor the identity map. Growing
prefixes and refill always create a new row; finished entries keep the legacy skip rule.
Historical bidirectional bindings outlive removal, so retiring a request does not make
an illegal internal/stable alias reusable. int-coercible inputs retain existing coercion
semantics; invalid generated suffix input still fails before stock allocation. The
separate strict prompt proof checks still reject bool/float prompt content as before.

No admission predicate, record schema, batch choice, initial proposal check, hashing of
claimed current prefix, logging buffer or fence changed. Both full pool snapshots and
block audits still run, reading live KV/frontier/ownership. K3 capacities remain demand3/
reserve4 for Target and normal Draft, demand/reserve6 for eager Draft. True K3, no bonus,
active16/home8+8/ceiling8, static eager, Serial idle gate and owner feedback boundaries
are unchanged. `post_prepost()` and promotion READY publication are untouched; earlier
READY publication remains a separate future candidate requiring dependency proof.

The performance report additionally gives window/steps cadence, existing engine-step
statistics and outside-step time; count of Target steps with definite cross-request
ordinary/recovery overlap and uncertainty-only overlap; and recovery coverage fraction.
The latter denominator is the union of native physical Draft forwards participating in
rejection recovery, selected by measurement host launch and clipped to the measurement
window. The numerator unions intersections with Target TP native intervals whose claims
do not contain that recovery request. Repeated bindings, mixed roles and TP ranks never
multiply an interval. Lower/upper fractions use opposite denominator bounds; missing
bounds yield null, a no-recovery window is explicit. OBSERVED means any positive lower
bound, not sufficiently hidden recovery. Neither CPU savings nor audit costs are
subtracted from measured throughput.

## Four-mode matrix (2026-09-15)

The new `specrhythm.k3-four-mode.v1` execution geometry distinguishes Serial and
PingPong. `serial-k3` and `serial-eager-k3` have one immutable A home of 16,
Target ceiling16, and the all-Draft-idle gate before the next claim. Serial-eager
uses the same K3 owner/backend, enabling conditional lookahead only during its
own parent verification. It has no B home and no cross-home pipeline.
`pingpong-k3` and `pingpong-eager-k3` retain A8/B8 and Target ceiling8. All four
permit compatible Draft rows to form physical B16; home capacity is not a Draft
batch limit. Controller, owner claim, live Target scheduler and capacity metadata
all use this geometry. No `post_prepost()` or READY publication order changes.

All use seed + two extensions for ordinary K3, three reusable eager candidates,
no fourth candidate and no Target bonus commit. Capacity v2 records geometry and
checks actual maximum sequence/query demand: Target up to 16*(root+K3)=64 query
positions for Serial versus32 for PingPong; Draft token-step up to16 positions.
Resident KV/logits/workspace arithmetic still charges active16, with ordinary
required3/reserved4 and eager Draft required6/reserved6. No B8 fallback. Legacy
capacity v1 can still be replayed using its recorded geometry; old Serial K3 B8
runs remain B8 and are not the new matched baseline.

Warmup uses 16 completed request-verification opportunities per unit, two units
(32 opportunities), with per-request counts and unique coverage retained. Thus
full Serial needs two Target steps and full PingPong four. Partial steps contribute
only their actual B; no padding or invented opportunities. Window starts only with
full active population, preserving live pipeline state and the unchanged deadline.
