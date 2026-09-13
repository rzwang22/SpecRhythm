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
