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

For every request, untimed setup performs exactly these operations:

1. Target prompt prefill samples exactly one bootstrap token and then stops advancing.
2. Draft initializes real resident KV for `prompt + bootstrap` and generates no proposal.
3. Request-level prompt/bootstrap/KV invariants are validated.
4. All Target TP ranks cross a CUDA-synchronized barrier.
5. `measurement_start_ns` is broadcast and the immutable manifest is written.
6. Target-only executes its first one-token decode, or Serial begins its first Draft proposal.

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

Gate B uses fresh two-request and five-request runs:

```text
raw-prompt stock Target output
  == bootstrap + resident Target continuation
  == bootstrap + resident Serial continuation
```

Token IDs, bootstrap, EOS/stop reason, max-token termination and final logical length all belong
to equality. The comparator already accepts an optional future Dual output, but Phase 4B.0b does
not run or claim Dual-Batch Outcome A. The Phase 3 HF trajectory remains diagnostic provenance,
not the serving correctness reference.

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
