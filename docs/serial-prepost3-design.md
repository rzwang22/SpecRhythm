# Serial pre/post3: explicit no-bonus rolling protocol

Status: CPU implementation and source contracts; GPU correctness, native overlap and
performance **PENDING**. This changes the algorithm only in `serial-prepost3` and
`serial-eager-prepost3`. Existing Serial/Serial-eager/PingPong defaults are unchanged.
The branch preserves optimized `c02ee7dec30ceff21e95eea8f3a42b4909d37202` and later
`bd38ae1ba3f2c6acb1de30fa98196f68da8c3ec6` delivery/diagnostic history.

## Contract and physical accounting

```text
eager_lookahead_steps = 3
post_verify_batch_steps = 1
long_proposal_tokens = 4
short_proposal_tokens = 1
serial_extension_steps = 3
```

Let C be the authoritative committed prefix of length n, P the current proposal
of actual length k (1 or 4 in ordinary rounds), and F the exclusive materialized
Draft KV frontier. Initially F=n+k−1: the last proposal token has been sampled
but has not been input to the model. KV beyond F is not valid evidence.

| Physical operation | Model input, at position | Sampled output | Frontier after fence |
| --- | --- | --- | --- |
| lookahead 1 | P[k−1], n+k−1 | a1 | n+k |
| lookahead 2 | a1, n+k | a2 | n+k+1 |
| lookahead 3 | a2, n+k+1 | a3 | n+k+2 |
| common post, successful parent | a3, n+k+2 | a4 | n+k+3 |
| common post, rejected parent after j accepts | Target correction x, n+j | r1 | n+j+1 |

The first three *new* candidates are a1/a2/a3 from three actual forwards; the
fourth is a4 from the common forward. C and P are never counted as lookahead.
On rejection, C'=C+P[:j]+x; everything dependent on the rejected suffix is invalid.
After fencing writes, materialize x at the rollback frontier n+j. Its logits
produce r1 in that **same** common forward, with no three-step recovery extension.
This is possible because the authoritative delta contains at most one correction.
The mixed batch uses ragged contexts and individual write positions in one
`worker.materialize` call. Each row's suffix is exactly one token; a numerical
failure, extra physical call, stale frontier or missing fence is an error.

The previous Serial K4 already samples its first candidate from cached logits
after seed/commit materialization, then performs three proposal forwards. The new
Serial control reuses this division: one `prepost_post` seed/materialization,
then three `prepost_extension` forwards. Its first verification, like the eager
variant, starts from a legal P1 bootstrap. Initial/refill prefix materialization
and cached-logit seed sampling remain separately recorded and inside the existing
setup/refill timing boundaries. Cached sampling is not an extra GPU forward.

Successful full-length eager rounds perform 3 lookahead + 1 common forward.
Feedback arriving early can cancel the remaining lookahead steps; full acceptance
waits for the still-valid steps. The exposed wait is measured. Draining uses
`prepost_repair` for required committed KV writes without next-candidate sampling.
Terminal rows may join a common batch without sampling; already-materialized
terminal rows need no forward. EOS or an output budget below four may yield 2/3
real candidates, explicitly marked terminal-clipped, rather than fake padding.
Noneligible requests keep normal Serial progress; the fixed experiment's immutable
eligible set is the resident request set. No eligibility is removed on rejection.

Source: `continuation/prepost.py`, `continuation/prepost_backend.py`,
`serving/prepost_machine.py`; ordinary cached seed/three extensions originate in
`phase4/draft_batch.py`, `phase4/vllm_draft_backend.py` and
`phase4/vllm_draft_worker.py`.

## Publication, sampling and Target KV

Full accept commits precisely P. Partial accept commits P[:j]+Target correction.
Lookahead is separate from C until it becomes a proposal and is actually verified.
The stock rejection sampler still calculates its additional full-accept bonus;
`prepost_target.install()` discards that unused result **before** custom proposer,
scheduler or client output sees it. The bonus is neither committed nor reused as
a candidate. This is a separate commit protocol, not removal of the old bridge
check. Old `dual_greedy_acceptance` and bridge logic remain in the old modes.

The opt-in wrapper runs after pinned vLLM `_bookkeeping_sync` parses negative
padding, validates the whole batch, then projects canonical tokens into its
returned lists and CPU request/token caches, including `num_tokens_no_spec` and
`is_token_ids`. It preserves list aliases. EOS/max-output clipping uses the same
canonical values in all consumers. Existing packed logprob offsets are retained:
downstream per-request slicing uses the canonical output length and excludes the
unused bonus. Deterministic sampling is the supported experiment setting; the
joint reference gate is required before any performance interpretation.

The pinned engine source contract is vLLM
`752a3a504485790a2e8491cacbb35c137339ad34` (CI checks out that exact revision).
Its scheduler naturally consumes ragged proposal lists. `_prepare_inputs` uses
actual scheduled counts to build positions, attention offsets and slot mapping;
`_calc_spec_decode_metadata` uses per-request draft counts, cumulative query
counts and sampling indices. P1/P4 both go into **one** Target TP2 forward.
For B16 with eight P1 and eight P4: 40 valid candidates and 56 valid query
positions (16 committed roots + 40 candidates), not 80 effective positions.
Internal engine graph/packing padding is not a proposal or accepted token.

The stock scheduler's post-output frontier adjustment uses `len(output)-1`.
With the no-bonus full-accept projection this retains the final committed token
as the next uncached root, so next Q remains actual P+1. That stock speculative
statistics counter is a cache-frontier counter under this variant; protocol
acceptance is reported separately from actual candidate comparisons. Source tests
execute the real bookkeeping, ragged sampling-index function, and scheduler
frontier block; they do not claim to execute CUDA attention/slot writes on CPU.

## Owner state, feedback and safety

Each request binds committed prefix/version, current proposal ID/tokens, and
lookahead ID/dependency/version/frontier. The real Unix adapter and `EagerOwner`
queue are reused. Admission validates the batch on one owner-managed state;
enqueue acknowledges queue insertion without waiting for Draft compute. Target
never waits for the first Draft forward. Control/feedback remains prioritized
between physical batch steps. A rejection already in the queue cancels subsequent
steps, with the required fence before rollback or release. A full accept that
arrives early enters the existing batch wait until valid lookahead completes.

Common settlement validates all plans before writing, does one physical batch,
fences it, then publishes per-request authoritative commits and next proposals.
It never regenerates retained a1/a2/a3. Rejection yields P1, full three-token reuse
yields P4. Exact duplicate feedback is idempotent; conflicting/stale identity,
prefix, version, frontier or feedback fails. Disable/reenable versions invalidate
old work without resurrecting it. Cancellation, EOS, refill, delayed terminal
publication and controlled drain retain physical ownership and release checks.

`runtime` audits reuse actual allocator allocate/free interception and affected
request checks; setup and teardown full audits, ownership conflict detection,
capacity/frontier validation, prefix/dependency checks and fence requirements are
unchanged. There is no long-lived cached “audit passed” boolean.

## Example across six Target steps

R0 is a persistent successful peer, so the batch stays mixed when R1 recovers.
All proposals below are uncommitted until the corresponding Target step.

| Step | R1 verified P | Target decision / commit | Draft during verify | Common output / next P |
| --- | --- | --- | --- | --- |
| 0 | [s] | full / s | a1,a2,a3 | append a4 → P4 |
| 1 | [a1,a2,a3,a4] | reject at a2 / a1,x | b1,b2,b3 invalidated | input x → [r1] P1 |
| 2 | [r1] | reject / y | c1..c3 cancelled/discarded | input y → [r2] P1 |
| 3 | [r2] | full / r2 | d1,d2,d3 retained | append d4 → P4 |
| 4 | [d1,d2,d3,d4] | full / d1..d4 | e1,e2,e3 retained | append e4 → P4 |
| 5 | [e1,e2,e3,e4] | full / e1..e4 | f1,f2,f3 retained | append f4 → P4 |

Authoritative output is `s,a1,x,y,r2,d1..d4,e1..e4`; b/c lookahead, rejected
candidates, unused bonus and the still-unverified f block never enter output.
Steps 2 and 3 verify R1 P1 alongside R0 P4 in the same Target batch. Repetition is
limited by ordinary EOS/output/model capacity, not an eager failure counter.

## Evidence and remaining gate

`prepost_physical` binds each physical forward to request/round/continuation,
purpose, B, materialized positions, generated count and fence.
Legacy `fixed_proposals` publication rows carry `batch_participation_only`; their
per-request GPU duration is null. Physical B/count/time comes from native forwards,
not the number of request publications. Aggregate backend counters cover lifecycle
(including decode warmup/drain); measurement uses actual window timestamps. `prepost` retains
parent feedback, lookahead completion, common-forward boundaries, candidate-ready,
accept/correction counts, next length, retained/discarded counts and lifecycle.
Target `prepost_samples` records real lengths, Q and discarded bonus/stop suffix.
All use bounded phased records (setup8192, warmup4096, measurement65536, drain8192),
with retained/dropped counts and dropped time bounds. Recording reads the existing
phase, adds no I/O, device synchronization or global lock; native device events and
existing logging/file attribution remain enabled. Extra recording cost is shared
by both new modes. Old modes are unaffected.

`prepost_evidence.analyze` checks physical/native call conservation, parent token
accounting, one common step, absence of legacy recovery extension and measured
mixed P1/P4 coverage. It rejects missing/truncated measurement evidence, including
loss in any phase intersecting the window. The parent report retains native TP
union/overlap bounds, work complete at Target end, per-thread exclusive host
categories, log/fsync file costs, step distributions and unaccounted time. New
mode admission/parent/settlement phases are explicitly joined. No host/RPC/GPU
inclusive times are added across lanes. Missing is not zero; terminal, refill,
setup and drain remain distinguishable from complete steady steps.

CPU regressions cover owner queue ordering using Events, both feedback orders,
repeated rejection and renewed rolling, mixed common GPU-call substitutes,
real allocator runtime/full reconciliation, terminal/drain/failure propagation,
version/ownership/frontier negatives, stock Target projection/ragged indices,
full adapter→owner→backend cycles, and collection→qualification/export negatives.
They establish software contracts, not GPU equivalence or hidden execution time.
See [runbook](prepost3-runbook.md) for the required actual joint GPU gate and pair.
