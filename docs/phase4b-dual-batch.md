# Phase 4B linear Dual-Batch design

Phase 4B.0 defines and has passed the correctness infrastructure gates. Phase 4B.1 implements
the real decode-only 1D+2V Dual-Batch runner and its fail-closed A800 correctness gates. The Mac
development pass proves CPU contracts only; no Phase 4B.1 GPU result exists until the external
Gate 1/2/3 artifacts pass. It does not evaluate serving performance. Packed trees,
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

Every Dual request starts from an immutable `DecodeReadyManifest` entry. Target has
`KV(prefix[:-1])` plus the pending final token, while Draft has the complete `prompt+bootstrap`
prefix. All requests finish Target bootstrap, Draft initialization, manifest validation, global
freeze and a TP barrier before `measurement_start_ns`. Setup-ready contains no proposal. Initial
Draft work is enqueued only after that boundary.

The timed lifecycle is:

```text
DRAFT_READY → DRAFTING → PROPOSAL_READY
          → VERIFY_READY → VERIFYING → COMMITTING
          → DRAFT_SYNC → DRAFT_READY
```

`TARGET_TAIL_READY` names the only proposal-free live Target operation; `TERMINAL` and `FAILED`
are terminal. The compatibility enum name `FINISHED` aliases `TERMINAL` but new evidence always
serializes `TERMINAL`.
One request cannot be in Draft and Verify simultaneously, own two proposals, continue Draft
through an unverified proposal, or re-enter after termination.

Every proposal contains `request_id`, `round_id`, canonical `proposal_id`, `prefix_version`,
prefix token count/SHA256, Draft KV lengths before/after, token IDs, and creation/Draft interval
timestamps. Its lifecycle is `CREATED → PUBLISHED → INSTALLED → CONSUMED`, or the explicit
terminal disposition `DROPPED_STALE`. Target validates all parent-prefix fields before
verification. Stale proposals are logged and rejected; they are never repaired, partially reused,
silently discarded, consumed twice, or committed.

The proposal `request_id` is always the frozen workload ID. vLLM-owned scheduler tables use an
opaque internal ID that may contain an implementation-defined suffix. Both the Target proposer
and scheduler independently match the current physical token row against unique frozen
`prompt_token_ids`, then bind one internal ID to one stable ID. They never split, trim, regex-match
or prefix-match the internal ID text. Zero/multiple prompt matches, identity changes and aliases
fail closed. Boundary events retain both fields where useful, but state, proposal, round,
validation and Draft-service keys remain stable IDs.

## Ready-only handoff and stock-vLLM boundary

The GPU-0 service owns one persistent Draft model and per-request KV. A single background worker
serializes model forward, rollback and correction/bonus append operations. Socket handlers only
enqueue work, poll completed work, or return already-claimed metadata. Consequently a Target
scheduler poll never waits for Draft GPU execution.

The Target uses vLLM's supported `scheduler_cls` extension. The plugin injects completed proposal
tokens into `request.spec_token_ids`, rejects decode requests without a valid ready proposal via
the explicit request-level admissibility hook,
then calls the stock scheduler. Allocation, preemption, continuous batching, paged KV, attention,
rejection sampling and output commit remain stock vLLM behavior. Bootstrap is Target-only; after
it commits, Draft jobs become eligible. A proposal-free final token is explicitly admitted as a
Target tail.

The pinned `0001-custom-proposer-request-and-verify-hooks.patch` supplies opaque vLLM request IDs
and Target verify observers in `gpu_model_runner.py`; the SpecRhythm adapter performs explicit
prompt-token identity translation. Independent `0002` adds the default-off stock-loop
admissibility hook and `0003` records Target-forward timing. The out-of-tree `scheduler_cls`
supplies the actual predicate. The patch stack is default-inactive and has no Target-only semantic
effect.

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

Target verification reuses vLLM's linear greedy speculative semantics. Its exact input is the
pending token followed by the proposal; positions begin at the materialized Target KV count. If
proposal token `d_m`
first differs from Target `g_m`, the accepted prefix plus `g_m` commits. Full acceptance commits
the proposal plus the Target bonus, except terminal Draft EOS. Rejected/future proposal positions
must not remain logically visible after commit.

## Artifacts, resume and validation

Each run produces the run JSON, decode-ready and runtime manifests, checksummed request-state,
proposal-round, proposal-lifecycle, scheduler-decision, verification,
transport/Target-diagnostic logs, scheduler and Draft-work logs, derived cycle and overlap logs,
an output checkpoint, validation JSON and Markdown summary. No runner overwrites an existing
artifact. Process-group cleanup is a separate immutable artifact produced only after the Target
coordinator and Draft service exit.

The server helper uses a short `/tmp` Unix socket and bounds both failure cleanup and graceful
Draft shutdown. It checks the Target exit status before waiting for Draft, terminates Target
leftovers plus Draft after a Target failure, reaps Draft, and returns the Target status instead of
hanging the shell wrapper.

Repeated runs compare proposal rounds by `(request_id, round_id)`. Different cross-request JSONL
order is diagnostic metadata, not a failure. Duplicate/missing keys or any proposal, acceptance,
correction, bonus, commit or terminal mismatch fail.

`phase4b1-dual-correctness-validate` is read-only and compares decode-only Target, Serial and
Dual under the same logical manifest identity. It compares token IDs and every termination field,
then validates request/proposal lifecycle, scheduler admissibility, pending/proposal positions,
Target/Draft logical KV, commit conservation, Target-blind Draft inputs, process cleanup and a
positive cross-request real-GPU overlap witness. It hashes every input before and after. The only
outcomes are `A` with `valid=true, errors=[]`, or `FAIL` with explicit errors. Cross-request JSONL
write order is diagnostic only; keyed round semantics are mandatory.

No Phase 4B GPU outcome exists until the external 3×A800 artifacts pass the read-only validator.
The field-level transition and evidence schema is maintained in
[phase4b1-dual-state-machine.md](phase4b1-dual-state-machine.md).
