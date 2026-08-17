# Phase 4B decode-ready contract

Phase 4's main evaluation boundary is **decode-only**. `ResidentWarmStartProvider` creates real
resident Target and Draft KV state outside the measured window so the scheduler and speculative
decode path can be isolated without claiming an end-to-end prefill/decode deployment.
`KVConnectorHandoffProvider` is a future provider behind the same contract; this phase does not
implement KVConnector, NIXL, LMCache, Mooncake, disk-backed KV transfer, or production PD.

```text
DecodeReadyProvider
  -> immutable DecodeReadyManifest
  -> Target-only / Serial / future Dual decode consumer
```

## Resident warm start

Pinned vLLM may split the initial prefill into arbitrary non-empty proposer callbacks. For every
request, untimed setup therefore performs these operations incrementally:

1. Target prompt prefill samples exactly one bootstrap token and then stops advancing.
2. Draft initializes real resident KV for `prompt + bootstrap` and generates no proposal.
3. An immutable stable-ID observation and bootstrap/Draft-initialization timestamps are retained.
4. The EngineCore scheduler freezes that request after bootstrap while any frozen request is not
   ready; later prompt/bootstrap work remains admissible.
5. After the complete frozen set validates, all Target TP ranks cross one CUDA-synchronized
   barrier and `measurement_start_ns` is broadcast.
6. The manifest is written. Target-only then publishes its atomic setup-ready artifact. Serial
   first creates all round-zero proposals after measurement start and includes them in its atomic
   setup-ready artifact for exact scheduler installation.

The scheduler and TP proposer workers do not share Python globals. The scheduler reads only the
atomic setup-ready artifact, validates its manifest SHA256 and stable/internal identity mapping,
and uses the existing explicit request predicate. A request with one output token cannot advance
before global readiness. Current-step arithmetic is forbidden.

Resident Serial gives each readiness-published round-zero proposal an explicit one-shot lifecycle:
`published -> installed -> consumed`. Installation validates the live committed-prefix length and
SHA256. An installed but unscheduled proposal remains admissible only while both the live prefix
and `request.spec_token_ids` remain unchanged. Membership in the
[pinned vLLM scheduler's `scheduled_spec_decode_tokens`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/sched/scheduler.py#L548-L563)
is the consumption evidence; after consumption, the scheduler never compares or reinstalls round
zero and normal `RemoteDraftProposer` rounds own subsequent proposal state. Every transition and
fail-closed check is written to a dedicated checkpoint JSONL artifact.

The Target state deliberately uses the standard pending-input convention:

```text
logical committed prefix = prompt + bootstrap
Target materialized KV   = prompt
Target pending input     = bootstrap at position len(prompt)
Draft materialized KV    = prompt + bootstrap
```

Allocated block capacity is not a materialized-token count. The manifest records the scheduler's
`target_num_computed_tokens` separately and states its relation to materialized Target KV. The
first real Target diagnostic must independently confirm that count.

## Manifest and timing invariants

`DecodeReadyManifest` is a frozen dataclass serialized as canonical JSON with a SHA256 over every
field except the hash itself. It records the SpecRhythm and pinned-vLLM commits, ordered patch
hashes, model/tokenizer revisions, workload hash, sampling and batch-invariant configuration,
GPU placement/TP, setup/barrier/measurement timestamps, and every request's exact logical state.

`ResidentSetupObservation` has one canonical JSON-compatible codec. `to_dict()` emits token IDs
as a JSON list; `from_dict()` strictly validates integer fields and restores
`prompt_token_ids` to `tuple[int, ...]`. Target, Serial, and future providers must use this loader
instead of raw dataclass `**mapping` reconstruction. The in-memory tuple invariant is fail closed.

Validation fails if any request does not satisfy:

```text
logical_prefix_count            = prompt_count + 1
Target_materialized_KV + 1      = logical_prefix_count
Target_pending_position         = Target_materialized_KV
Target_pending_token            = bootstrap = logical_prefix[-1]
Draft_materialized_KV           = logical_prefix_count
prefix_version                  = 1
next_round_id                   = 0
initial_proposal_generated      = false
```

The pre-barrier event proves those invariants before admission. Proposal timestamps and Target
forward timestamps must be at or after `measurement_start_ns`; outliers are not waived. This
boundary is correctness evidence only and produces no speedup, latency, throughput, goodput, or
SLO result.

## First Target forward

For prompt length `P` and proposal length `K`, the first timed inputs are:

| Consumer | Token IDs | Positions |
| --- | --- | --- |
| Target-only | `[bootstrap]` | `[P]` |
| Serial / future Dual | `[bootstrap] + proposal[0:K]` | `[P, ..., P+K]` |

The patched observer records the actual Target input tokens, positions, forward start/end,
logical/physical KV counts and logits mapping. The first-forward artifact also records proposal
tokens, accepted/rejected counts, post-forward committed tokens, post-rollback materialized KV,
and the single prefix-version transition. Missing or duplicated bootstrap tokens, position
offsets, invalid acceptance accounting, or KV mismatch fail closed.

## Correctness chain

The corrected Gate B first uses a fresh two-request run:

```text
raw-prompt stock Target output
  == bootstrap + resident Target continuation
  == bootstrap + resident Serial continuation
```

Token IDs, bootstrap, EOS/stop reason, max-token termination and final logical length all belong
to equality. The comparator already accepts an optional future Dual output, but Phase 4B.0b does
not run or claim Dual-Batch Outcome A. L5 remains blocked until the corrected L2 artifacts are
reviewed. The Phase 3 HF trajectory remains diagnostic provenance, not the serving correctness
reference. See [phase4b-resident-l2-rerun.md](phase4b-resident-l2-rerun.md).

## Patch and process boundaries

The pinned vLLM stack is applied in fixed order:

1. `0001-custom-proposer-request-and-verify-hooks.patch`;
2. `0002-scheduler-request-admissibility-hook.patch`;
3. `0003-target-forward-timing-observer.patch`.

Patch 2 is inactive on stock schedulers. Patch 3 adds timestamps around the existing model forward
and passes them to the observational diagnostic hook. None changes attention, sampler, model
weights, TP partitioning, C++/CUDA/Triton, or vLLM DBO. Apply/check/restore validates every patch
and exact original/intermediate/final file hash.

Target execution is owned by one recorded session/PGID. Cleanup signals only that PGID, propagates
the true coordinator status, verifies all descendants and the Draft socket are gone, and leaves a
guard file when cleanup is invalid. A subsequent run must not begin while that guard remains.
