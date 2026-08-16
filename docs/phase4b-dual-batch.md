# Phase 4B linear Dual-Batch design

Phase 4B.0 defines the execution contracts; Phase 4B.1 makes the frozen 1D+2V stack ready for a
3×A800 correctness run. It does not evaluate serving performance. Packed trees,
Residual-Probability, Eager, Shaping, TP3/TP4 and four-GPU comparisons remain outside this phase.

## Three distinct modes

| Mode | GPU 0 | GPUs 1–2 | Cross-engine overlap |
| --- | --- | --- | --- |
| Target-only | idle | Qwen3-32B TP2 | no |
| Serial Disaggregated | Qwen3-0.6B Draft, then idle | idle, then Target verify | forbidden |
| Dual-Batch | Drafts microbatch B | verifies disjoint microbatch A | required in steady state |

vLLM DBO remains disabled. CUDA graphs, TP collectives, CPU log threads and timestamps without
CUDA synchronization are not Dual-Batch evidence.

## Request lifecycle and proposal identity

The lifecycle is:

```text
BOOTSTRAP → DRAFT_READY → DRAFTING → PROPOSAL_READY
          → VERIFY_READY → VERIFYING → COMMITTING
          → DRAFT_SYNC → DRAFT_READY
```

`FINISHED` and `FAILED` are terminal. The one-token Target tail follows the explicit
proposal-free `DRAFTING → DRAFT_READY → VERIFY_READY → VERIFYING → COMMITTING → FINISHED` path.
One request cannot be in Draft and Verify simultaneously, own two proposals, continue Draft
through an unverified proposal, or re-enter after termination.

Every proposal contains `request_id`, `round_id`, canonical `proposal_id`, `prefix_version`,
prefix token count/SHA256, Draft KV lengths before/after, token IDs, and creation/Draft interval
timestamps. Target validates all parent-prefix fields before verification. Stale proposals are
logged and rejected; they are never repaired, partially reused, or committed.

## Ready-only handoff and stock-vLLM boundary

The GPU-0 service owns one persistent Draft model and per-request KV. A single background worker
serializes model forward, rollback and correction/bonus append operations. Socket handlers only
enqueue work, poll completed work, or return already-claimed metadata. Consequently a Target
scheduler poll never waits for Draft GPU execution.

The Target uses vLLM's supported `scheduler_cls` extension. The plugin injects completed proposal
tokens into `request.spec_token_ids`, temporarily defers decode requests without a ready proposal,
then calls the stock scheduler. Allocation, preemption, continuous batching, paged KV, attention,
rejection sampling and output commit remain stock vLLM behavior. Bootstrap is Target-only; after
it commits, Draft jobs become eligible. A proposal-free final token is explicitly admitted as a
Target tail.

The pinned `0001-custom-proposer-request-and-verify-hooks.patch` remains the only upstream patch.
It supplies stable request IDs and Target verify observers in `gpu_model_runner.py`. Phase 4B
does not add `0002`: the source audit found that `scheduler_cls` is sufficient. The patch is
default-inactive and has no Target-only effect.

## Dynamic microbatches and overlap proof

Ready work is formed deterministically in FIFO order. A and B are dynamic sets, not permanent
partitions. The scheduler never delays ready Target work merely to manufacture overlap. Draft may
idle if no request is Draft-ready; Target may idle if no proposal is ready.

Draft GPU 0 and each Target TP rank record CUDA events bracketed by CUDA synchronization plus a
shared host monotonic interval. For a disjoint pair:

```text
overlap_start = max(draft_start, verify_start)
overlap_end   = min(draft_end, verify_end)
overlap_ns    = max(0, overlap_end - overlap_start)
```

A positive interval is accepted only with Draft physical GPU 0 evidence, both Target physical
GPUs 1 and 2, CUDA-event evidence and disjoint request sets. The ratio is diagnostic only and is
never converted into a speedup claim.

## KV and speculative semantics

The Draft backend performs one full-prefix initialization per admitted request, then keeps KV
across rounds. Rejection crops the Draft proposal suffix, and exactly one Target correction or
bonus is appended when present. A request may draft again only after its logical Draft KV length,
prefix version and prefix hash equal the committed Target prefix.

Target verification reuses vLLM's linear greedy speculative semantics. If proposal token `d_m`
first differs from Target `g_m`, the accepted prefix plus `g_m` commits. Full acceptance commits
the proposal plus the Target bonus, except terminal Draft EOS. Rejected/future proposal positions
must not remain logically visible after commit.

## Artifacts, resume and validation

Each run produces the run JSON, runtime manifest, checksummed request-state/proposal/verification/
transport/Target-diagnostic logs, scheduler and Draft-work logs, derived cycle and overlap logs,
an output checkpoint, validation JSON and Markdown summary. The runner checkpoints complete
requests. `--resume` skips those stable IDs and starts fresh engine/Draft state only for remaining
cohorts; it never reconstructs a live in-flight proposal.

Repeated runs compare proposal rounds by `(request_id, round_id)`. Different cross-request JSONL
order is diagnostic metadata, not a failure. Duplicate/missing keys or any proposal, acceptance,
correction, bonus, commit or terminal mismatch fail.

Outcomes are:

- A: five- and 100-request exact correctness, repeated keyed semantics, state/KV/accounting and
  batch-invariant evidence all pass, with positive real-GPU overlap.
- B: correctness passes but no real overlap; the implementation is functionally serial.
- C: overlap occurs but sequence, termination, prefix, state, KV or accounting correctness fails.
- D: correctness would require a C++/CUDA/Triton change, large scheduler rewrite or replacement
  engine. Work stops before expanding scope.

No Phase 4B GPU outcome exists until the external 3×A800 artifacts pass the read-only validator.
