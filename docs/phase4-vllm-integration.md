# Phase 4 vLLM serving integration

Phase 4 is a stacked Draft PR based on the frozen Phase 3 branch. It does not change simulator
policies or reinterpret Phase 3 full-context measurements as serving latency.

## Phase 4A.1 scope

The frozen environment is vLLM `v0.25.1` at commit
`752a3a504485790a2e8491cacbb35c137339ad34`, Python 3.11, PyTorch 2.11.0, BF16,
greedy decoding, seed 1664, `enable_thinking=false`, `max_model_len=4096`, and linear proposal
budget `K=4`. The layout is fixed:

| Role | Model | Physical GPUs | TP | Runtime |
| --- | --- | --- | ---: | --- |
| Draft | Qwen3-0.6B | 0 | 1 | one persistent HF model with cross-round mutable KV |
| Target | Qwen3-32B | 1,2 | 2 | patched-hook vLLM V1 engine; stock scheduler/verifier/KV |

The Draft HF adapter is a correctness implementation, not a serving-performance backend. It
loads one model, performs one initial full-prefix prefill per request, then uses single-token
forwards plus cache cropping. Five requests share that one resident engine; per-request KV states
may be processed as internal microbatches. No request owns a separate model or engine.

Phase 4A.1 implements only `serial-disaggregated`:

```text
Draft batch finishes
  -> local Unix-socket proposal transfer finishes
  -> Target batched speculative verification finishes
  -> accept/reject and Draft state synchronization finish
  -> next Draft batch may start
```

There is no cross-round or same-round Draft/Verify overlap. vLLM DBO is disabled. Dual-Batch,
Dual-Eager, packed trees, tree selection, SLO evaluation, goodput, and speedup remain absent.

## Phase 4A.1.1 batch-invariant correctness hardening

The first user-run default-mode A/B experiment passed patched Target-only regression, lifecycle
accounting, strict-serial ordering, GPU placement, persistent Draft KV, and Target-label
isolation. One of five Serial outputs nevertheless differed from stock Target-only at generated
position 1. Stock Target-only had an exact BF16 log-probability tie between token IDs 22570 and
53143 and chose 22570; expanded speculative verification changed the values and accepted the
Draft proposal 53143. This is an unresolved exact-token correctness failure, not a benign-drift
waiver.

Phase 4A.1.1 makes execution configuration a first-class correctness input. Both stock
Target-only C and Serial D accept `--correctness-mode batch-invariant`, which sets
`VLLM_BATCH_INVARIANT=1` before importing vLLM. Each TP rank reports the raw and parsed flag,
dtype, compute capability, attention backend and its support declaration, custom-all-reduce
state, and enabled all-reduce path. The combined manifest records separate
`batch_invariant_requested`, `batch_invariant_effective`, and `batch_invariant_validation`
fields. A requested mode without consistent, complete rank evidence fails; setting the
environment variable alone is never sufficient.

The pinned vLLM v0.25.1 source explicitly documents batch invariance as requiring NVIDIA compute
capability at least 8.0, and its FlashAttention backend both supports batch invariance and accepts
compute capability 8.0 or newer. The available A800 therefore passes the hardware preflight. The
preflight still reports `batch_invariant_effective=false`: C/D may be labelled effective only
after every initialized TP rank proves the resolved environment flag, attention-backend support,
custom-all-reduce disablement, cascade-attention disablement, and DBO disablement. There is no
override or silent fallback for missing worker evidence.

Runner provenance verification is also part of the correctness boundary, but it must not import
vLLM. Both stock and patched runner checks locate
`vllm/v1/worker/gpu_model_runner.py` through installed-distribution metadata and hash the file
directly. `run_stock_smoke` remains the authority that configures the correctness mode before the
first vLLM import. Serial and fixed-control entry points likewise configure the mode before any
engine import; their import-free patched-runner check occurs inside that protected interval. The
failed C-reference attempt at commit `a7fe058d417ae44edea497657e17eef161c09d0e` stopped before
engine creation and is not a GPU correctness result.

Repeated Serial round semantics are keyed by `(request_id, round_id)`. Cross-request scheduler
interleaving may change the raw JSONL order without changing speculative-decoding behavior, so raw
event-order equality is diagnostic metadata only. The validator still fails closed on duplicate,
missing, extra, non-monotonic per-request keys or any keyed difference in proposals, accepted and
rejected counts, Target correction/bonus tokens, committed tokens, or terminal state.

The four conceptual experiments remain distinct and use fresh artifact directories:

| Group | Execution | Mode | Status |
| --- | --- | --- | --- |
| A | stock Target-only | default | retained provenance; never rewritten |
| B | Serial Disaggregated | default | retained failed exact comparison |
| C | stock Target-only, twice | batch-invariant | A800 hardware-supported; requires rank proof |
| D | Serial Disaggregated, twice | batch-invariant | compared only with C |

Outcome A requires both C repeats, both D repeats, termination, accounting, strict timeline, KV
monotonicity, and every `D == C` token sequence to pass. If C and D differ, the Target-only
diagnostic log is matched to the Serial log by stable request ID and committed-prefix SHA256. It
records vLLM's actual `target_logits_indices`, proposal index, flattened input position,
contiguous position IDs, logical/physical KV lengths, raw-logit and log-prob top-k values,
selected Target token, verification shape, attention/all-reduce backend, dtype, and
batch-invariant state. This log is Target-only and its fields are forbidden from every Draft IPC
payload.

Only after a failed C/D comparison may the `K=1,2,4` fixed controls run on the single divergent
request. `LocalStaticProposer` and `RemoteFixedProposer` return the same prefix of
`[53143, 2213, 369, 264]`; the latter adds only local Unix transport. Equal target logits,
accepted prefix, committed tokens, and final trajectory plus a valid prefix/position/KV/mask
proof makes Outcome B eligible: an upstream execution-shape numerical limitation, with exact
correctness still failed. Any control or mapping difference is Outcome C, an integration bug.
No epsilon tie rule, token-specific rewrite, reference-guided acceptance, Target shadow replay,
request deletion, or relaxed text-similarity criterion is permitted.

These evidence levels are not interchangeable:

| Evidence | Meaning in this phase |
| --- | --- |
| Algorithmic speculative correctness | rejection, correction/bonus, termination, and KV/accounting invariants hold |
| Exact token equality | every generated token and termination field in D equals C |
| Hardware numerical batch invariance | pinned mode is requested and proven effective on every TP rank for a supported GPU |
| Serving performance | not measured or reported anywhere in Phase 4A.1.1 |

## Correctness references

The Phase 4 serving correctness reference is an immutable stock vLLM Target-only greedy artifact:

```text
serving_correctness_reference = stock-vllm-target-only
```

`stock-target-reference.json` is created before the vLLM hook patch is applied. The command runs
the five corrected R3-real requests twice, requires deterministic token IDs and termination,
records model/tokenizer/runtime pins, and freezes the file without overwrite. Its
`artifact_sha256` is the SHA256 of canonical JSON before that field is inserted; the run artifacts
also record the final file SHA256.

The pass conditions are:

```text
patched target-only token IDs and termination == frozen stock target-only
serial run 1 token IDs and termination       == frozen stock target-only
serial run 2 token IDs and termination       == frozen stock target-only
serial run 1                                 == serial run 2
```

The old Phase 3 HF trajectory is retained as
`provenance-and-divergence-diagnosis-only`. It can produce a warning but cannot fail Phase 4.
Serial execution never rewrites or regenerates either reference.

## Remote proposer and Target authority

GPU 0 hosts a local Unix-domain-socket service with a versioned, length-prefixed canonical-JSON
protocol. It supports:

- `initialize`;
- `batch_propose`;
- `synchronize_committed_prefix`;
- `rollback_rejected_suffix`;
- `append_target_correction_or_bonus`;
- `cancel_request`;
- `finish_request`;
- `shutdown`.

The combined synchronization/batch call invokes the same checked transitions atomically for one
round. Every proposal records stable request and round IDs, parent length/hash, at most four token
IDs, Draft EOS, Draft monotonic boundaries, serialized payload bytes, and model/runtime
provenance. Duplicate, stale, out-of-order, prefix-mismatched, or post-finish work fails.

Only Target TP rank zero opens the socket. The proposal is broadcast to the second Target rank;
the proposer object has zero model parameters. IPC contains current committed token deltas and
prefix hashes only. Target logits, future Target tokens, top-k values, and oracle labels are
forbidden from the Draft payload. AF_UNIX does not listen on a network interface and the protocol
does not use pickle. If vLLM still requires `VLLM_ALLOW_INSECURE_SERIALIZATION=1` for its own
worker RPC, the patch manifest records a security warning and restricts use to a trusted local
experiment.

## Linear greedy verification

For committed prefix `P` and Draft proposal `D=(d1,...,dk)`, `k<=4`, the stock vLLM speculative
path evaluates the candidate positions in one Target verification step and uses its rejection
sampler. Let `g_i` be the Target argmax after `P,d1,...,d_(i-1)`.

- First mismatch at `m`: commit accepted `d1...d_(m-1)` and Target correction `g_m`.
- All accepted: commit `d1...dk` and Target bonus `g_(k+1)`.
- Accepted Draft EOS may terminate without a bonus.
- If only one output slot remains, no proposal is created and Target commits the final tail token.

The first output token is the stock vLLM bootstrap token produced after prefill. It is reported
separately from proposal rounds so final accounting remains:

```text
proposed = accepted_draft + rejected_draft
committed_round = accepted_draft + target_correction + target_bonus
final_generated = target_bootstrap + target_tail + sum(committed_round)
```

Correction and bonus are mutually exclusive and each has length at most one.

## KV lifecycle

Target KV ownership remains inside vLLM. Its scheduler debits rejected speculative tokens from
the logical computed length; physical blocks that temporarily contain rejected positions are not
part of the subsequent logical request prefix.

Draft keeps one mutable cache per active request. Proposal forwards may materialize candidate KV.
After Target returns:

1. crop to the old committed prefix plus accepted Draft tokens;
2. append a Target correction or bonus if present;
3. compare full committed token prefixes and SHA256 hashes;
4. propose the next round only from that synchronized prefix.

The adapter refuses a Transformers cache without an in-place `crop()` operation. It never falls
back to full-context replay. Finished/cancelled requests release Draft state and cannot re-enter a
batch.

## Minimal vLLM patch boundary

The exact vLLM custom-class proposer already feeds returned token lists into the stock
speculative scheduler, batched Target forward, rejection sampler, and KV accounting. Its public
signature does not expose stable request IDs or exact Target verification boundaries. The single
patch under `integrations/vllm/patches/` therefore changes only
`vllm/v1/worker/gpu_model_runner.py` to:

1. pass existing vLLM request IDs to the custom proposer;
2. invoke optional `on_target_verify_start` and `on_target_verify_end` hooks;
3. invoke an observational Target diagnostic hook, which writes nothing unless an external
   diagnostic path is explicitly configured.

No Target logits or decisions are passed to Draft by the patch. It changes no scheduler, sampler,
KV, attention, C++, CUDA, or Triton code. When speculative decoding is disabled, the proposal and
verification branches are inactive; only an explicitly requested Target-only diagnostic log can
run. The patch manager verifies the exact base file SHA256, applies with zero fuzz, records
patch/pre/post hashes, and can restore stock code. Patched Target-only must still equal the frozen
unmodified reference before Serial is allowed to pass.

## Artifacts and evidence boundary

The two runs keep fsync'd, per-record-checksummed `round-events.jsonl` and
`transport-events.jsonl`. A truncated final line or checksum mismatch fails validation. Each
round contains strict monotonic boundaries:

```text
draft_end <= transfer_start <= transfer_end <= verify_start <= verify_end
          <= state_sync_start <= state_sync_end <= next_round_draft_start
```

Output files set:

```text
gpu_correctness_result=true
gpu_performance_result=false
reports_goodput=false
reports_slo_attainment=false
reports_speedup=false
```

Wall-clock phase observations exist only to prove non-overlap. They are not benchmark samples and
must not be used as a simulator latency surface or performance conclusion.
# Phase 4B.0/4B.1 integration boundary

Phase 4B adds a third, explicitly named `dual-batch` mode alongside `target-only` and
`serial-disaggregated`. Its Target callback does not synchronously request Draft work. GPU-0
model work runs on the asynchronous Draft service; a vLLM `scheduler_cls` plugin consumes only
completed proposals and delegates the actual batch to the stock scheduler. Draft and Verify
request sets must be disjoint in every overlap event.

The existing `0001` patch remains sufficient and no new upstream patch is added. Proposal parent
version/count/SHA256 checks run before verification, one request has at most one proposal, and
Draft cannot pass an unverified prefix. Batch-invariant mode and per-rank worker evidence remain
mandatory; vLLM DBO remains disabled.

Phase 4B.1 artifacts can establish exact-token/termination correctness and the existence of real
GPU0 versus GPU1–2 overlap. They cannot establish speedup, throughput, goodput, SLO attainment,
latency improvement or production readiness. The detailed state, KV and outcome definitions are
in [phase4b-dual-batch.md](phase4b-dual-batch.md).
