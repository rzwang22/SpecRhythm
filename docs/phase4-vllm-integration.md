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
2. invoke optional `on_target_verify_start` and `on_target_verify_end` hooks.

No Target logits or decisions are passed by the patch. It changes no scheduler, sampler, KV,
attention, C++, CUDA, or Triton code. When speculative decoding is disabled the added branches are
inactive. The patch manager verifies the exact base file SHA256, applies with zero fuzz, records
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
