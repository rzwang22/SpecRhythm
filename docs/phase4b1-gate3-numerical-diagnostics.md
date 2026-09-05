# Phase 4B.1 Gate3 numerical localization

This work is diagnostic-only. Exact generated-token equality against the immutable stock-vLLM
reference remains the correctness authority. It does not authorize a tolerance, a replacement
reference, Serial, Dual, Dual-Eager, performance, TPOT, throughput, goodput, SLO, speedup, or
Phase 4B.2 experiment.

## Preserved evidence

The corrected-100 resident Target run at commit `32b09a6` passed chunked setup, all 100 bootstrap
observations, global decode readiness, first-forward contracts, TP2 placement, measurement
boundaries, and cleanup. It matched 96/100 immutable trajectories and diverged at exactly these
zero-based generated positions:

| Request | Position | Stock token | Resident token |
| --- | ---: | ---: | ---: |
| `r3-c7ee1a73ee79dd6dc21cb8dc` | 3 | 600 | 296 |
| `r3-32ae44a69fffd76f0dd4b787` | 4 | 3435 | 15 |
| `r3-646c340a0281105c1c20de27` | 12 | 448 | 323 |
| `r3-e00f5312321ec537a9c716cd` | 2 | 23826 | 1674 |

For every pair the immutable stock path preferred its selected token by exactly `0.125` in raw
logit space, while resident execution made the two values equal. Each path's raw argmax matches
its emitted token, so this is already upstream of sampling rather than a sampler tie-breaking
bug.

The first observer launch at `c142fa7` failed before a valid checkpoint because it read
speculative-only common attention metadata on a stock run. It remains immutable
`diagnostic-infrastructure-failed` provenance. The generic-ownership observer at `e73e884` then
completed one stock-style and one resident corrected-100 run at:

```text
/root/autodl-tmp/SpecRhythm-data/results/phase4/e73e8848904eee7f18e7beba80d1ec2da94e8267/phase4b1-gate3-numerical-20260904T070021Z
```

That root is immutable. It proves equal actual pre-divergence generated history, computed-token
boundary, logical KV ownership, and current-token embedding, followed by different KV bytes,
hidden states, and raw logits. Read-only per-layer comparison found the same boundary on both TP
ranks: layers 3/4 for `c7ee`, 20/21 for `646c`, 23/24 for `32ae`, and 55/56 for `e00f`. Every
layer before the boundary was exact and every later layer differed. This strongly motivates
per-logical-token localization but does not prove a numerical-drift hypothesis or close Gate3.

## Async CPU placeholder correction

The e73 stock records exposed an observer metadata error. In pinned vLLM async scheduling,
`GPUModelRunner._bookkeeping_sync()` keeps the real sample in GPU
`prev_sampled_token_ids` but writes `[-1]` into `InputBatch.token_ids_cpu`. On the next step,
`_prepare_input_ids()` scatters those real GPU values into the model input. Therefore the e73
stock `logical_committed_prefix_token_ids` field is an async CPU placeholder view, not a semantic
history.

The new record calls this field `async_cpu_placeholder_view`, permits its signed `-1` metadata,
and marks `semantic_prefix_authority=false`. The comparator derives semantics only from each
completed run's actual `generated_token_ids` and the immutable stock reference. At a planned
position `p`, all three exact checks are mandatory:

```text
stock_generated[:p]    == immutable_generated[:p]
resident_generated[:p] == immutable_generated[:p]
stock_generated[:p]    == resident_generated[:p]
```

The report names this authority
`final-run-output-vs-immutable-stock-reference`. Placeholder equality never drives a correctness
decision.

## Exact pinned-vLLM mapping audit

The authority is vLLM `v0.25.1`, commit
`752a3a504485790a2e8491cacbb35c137339ad34`:

- [`GPUModelRunner._bookkeeping_sync()`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/worker/gpu_model_runner.py#L3627-L3745)
  documents the real GPU sample and async CPU `-1` placeholder split;
- [`GPUModelRunner._prepare_input_ids()`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/worker/gpu_model_runner.py#L1738-L1858)
  copies `prev_sampled_token_ids` into the executed GPU input;
- [`BlockTable` and `MultiGroupBlockTable`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/worker/block_table.py)
  expose the per-request logical block row, kernel `block_size`, and group tables;
- [`GPUModelRunner._get_slot_mappings()`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/worker/gpu_model_runner.py#L3986-L4034)
  obtains each group's current slot mapping from that same block table;
- [`FlashAttentionBackend.get_kv_cache_shape()`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/attention/backends/flash_attn.py#L123-L153)
  defines semantic shape `(num_blocks, 2, block_size, num_kv_heads, head_size)`, and its forward
  [unbinds dimension 1 into key and value](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/attention/backends/flash_attn.py#L815-L829).

For a logical position `i`, the observer reads physical block
`block_row[i // block_size]`, offset `i % block_size`, then hashes K and V separately from
`cache[physical_block, 0 or 1, offset]`. Physical block numbers are provenance, not comparison
keys. The exact Qwen3-32B TP2 run must prove one KV group and unique ownership of both selected
layers on every rank; any other group/layout/ownership is unsupported and fails closed.

## Per-logical-token design and schemas

`configs/phase4b1_gate3_per_token_kv_diagnostic.json` is an immutable four-checkpoint plan. It
pins the known request, divergence, token pair, exact control layer, exact first-different layer,
the `32b09a6` correctness source, and the e73 coarse source. Loading resolves prompt tokens,
prompt length, maximum output, workload SHA256, and layer names from the frozen corrected-100
workload; changing any established checkpoint or source fails.

At each checkpoint and TP rank, the observer retains the existing all-layer aggregate evidence,
then reconstructs only the control and first-different layer for logical positions
`[0,num_computed_tokens)`. It concatenates each selected layer on GPU, performs one bounded
GPU-to-CPU transfer per selected layer, and hashes each CPU token slice separately for K and V.
It never includes the pending Target input whose KV has not yet been materialized. The selected
aggregate digest must equal that layer's digest in the existing per-rank aggregate record.

The record schema is `specrhythm.phase4b1-gate3-per-token-kv-record.v1`. It includes exact TP,
group, FlashAttention layout, dtype, per-token K/V shapes, materialized interval, transfer count,
separate token hashes, and the explicitly non-authoritative placeholder view. The comparator
schema is `specrhythm.phase4b1-gate3-per-token-kv-comparison.v1`. Its JSON provides per request
and rank:

- exact final-output/reference prefix checks;
- all-exact K/V status for the control layer;
- aggregate equality and K/V mismatch counts for the first-different layer;
- first and last K mismatch, first and last V mismatch, union-first position, and phase;
- a checksum/equality-only window of at most two positions on either side.

The Markdown summary presents the request, output position, rank, control layer status,
first-different layer, first logical mismatch, phase, and K/V mismatch counts. No raw tensors are
written.

## Fail-closed rules and decision tree

Validation fails on incomplete TP evidence, non-one-group or wrong FlashAttention layout,
ambiguous layer ownership, invalid/truncated block mapping, selected/aggregate checksum
inconsistency, non-exact control, an aggregate first-layer difference with no per-token K/V
difference, any reference/prefix mismatch, any changed known divergence, or any new divergence.
No tolerance or tie-equivalence rule exists.

A structurally valid report classifies only the earliest observed location:

- `PROMPT_PREFILL`: at least one first mismatch is below its prompt length;
- `BOOTSTRAP`: no prompt mismatch and at least one first mismatch equals prompt length;
- `DECODE_HISTORY`: every first mismatch is greater than prompt length;
- `FAIL-CLOSED`: any evidence or authority condition fails.

Gate3 remains not closed after classification. A `PROMPT_PREFILL` result calls for the narrowest
prefill execution-history audit; `BOOTSTRAP` localizes onset to bootstrap materialization;
`DECODE_HISTORY` localizes it to later decode progression. A policy decision, Serial/Dual work,
or performance experiment requires a separate reviewed phase.
