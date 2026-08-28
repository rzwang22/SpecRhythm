# Phase 4B.1 Gate3 numerical localization

This phase is diagnostic-only. Exact token equality against the immutable stock reference remains
the correctness authority. No tie-equivalent-token rule, numerical tolerance, replacement stock
reference, Serial, Dual, performance, TPOT or SLO result is authorized here.

## Preserved evidence and question

The first corrected-100 resident Target run at commit `32b09a6` passed chunked setup, all 100
bootstrap observations, global decode readiness, first-forward contracts, TP2 placement,
measurement boundaries and cleanup. It then failed exact output compatibility for four requests.
The common pattern is a stock top-two log-probability margin of exactly `0.125` becoming equal
resident values. Because pairwise log-softmax differences equal pairwise raw-logit differences,
this is pre-log-softmax drift rather than evidence of a different log-softmax tie policy. The old
directory remains immutable; this phase never edits or reclassifies it as a pass.

The first one-shot instrumentation run at commit `c142fa7` is also immutable, at:

```text
/root/autodl-tmp/SpecRhythm-data/results/phase4/c142fa7adbbdf0d81cc02d9244a3be75d4b9d7e7/phase4b1-gate3-numerical-20260828T033035Z
```

Its preparation and four-layer patch application passed, but both stock TP ranks crashed before
producing a valid numerical checkpoint. Resident Target and the comparator were never run. This
is `diagnostic-infrastructure-failed` provenance, not a Gate3 numerical result, and it does not
consume the one authorized successful stock/resident comparison. The later EngineCore request
`KeyError` and shared-memory warning followed the worker failure and are classified as shutdown
aftermath unless independent evidence proves otherwise.

## Exact pinned-vLLM batch-invariance audit

The authority is vLLM `v0.25.1`, commit
`752a3a504485790a2e8491cacbb35c137339ad34`.

The audited pinned files are `docs/features/batch_invariance.md`,
`vllm/model_executor/layers/batch_invariant.py`,
`vllm/v1/attention/backends/flash_attn.py`,
`vllm/model_executor/layers/linear.py`, `vllm/model_executor/layers/layernorm.py`,
`vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen2.py` and
`tests/v1/determinism/test_batch_invariance.py` at that exact commit.

| Dimension | What the pinned source establishes | What it does not establish |
| --- | --- | --- |
| Batch size | The beta documentation promises independence from batch size. Needle and log-probability tests compare one request with larger batches. | The tests do not enumerate every Qwen3-32B TP2 execution shape. |
| Request order | The documentation promises independence from request order; the needle tests vary placement in a batch. | This is not a proof for arbitrary scheduler histories. |
| Mixed prefill/decode | FlashAttention declares batch-invariance support and fixes split count; AOT scheduling is disabled in invariant mode. | No pinned test proves every mixed prefill/decode composition used by resident global freeze. |
| Chunked-prefill decomposition | No affirmative guarantee was found. | The pinned log-probability test contains an explicit TODO saying its prompts do not exercise chunking. |
| Decode cohort after global freeze | Ordinary batch-size variation is intended to be invariant. | The exact early-bootstrap/frozen/late-prefill trajectory is not an upstream test case. |
| Paged-KV block layout | FlashAttention uses the supplied block table with `num_splits=1`. | No source-level theorem or test proves bitwise equality across different physical block allocation histories. |
| LM-head `M` dimension | On SM80 unquantized linear explicitly calls the persistent invariant matmul. It accumulates in FP32 in a fixed K order. | The kernel casts its final output back to the output dtype. The pinned suite does not prove every Qwen3-32B BF16 TP2 `M` shape produces identical final BF16 bins. |

On SM80, invariant mode overrides `mm`, `addmm`, `matmul`, `linear`, `bmm`, softmax,
log-softmax and mean; Qwen RMSNorm explicitly selects the invariant implementation. It disables
reduced-precision reduction and TF32, fixes NCCL settings, and the Phase-4 preflight separately
requires custom all-reduce, cascade attention and DBO to be disabled. Qwen3's unquantized QKV,
MLP, output projection and LM head route through the invariant linear method; final RMSNorm uses
the invariant RMSNorm; FlashAttention reports support and fixes its split count.

Embedding lookup, elementwise activation/gating, rotary operations, KV writes and physical paged
layout are not replaced by the listed ATen overrides. They may be deterministic for identical
inputs, but the pinned source does not elevate that observation into a bitwise guarantee across
chunk decomposition and allocation history. The feature is documented as beta, and Qwen3-32B is
not one of the exact dense Qwen3 checkpoints listed in the pinned tested-model table. Therefore
`batch_invariant_effective=true` is necessary worker evidence, not proof of all dimensions above.

## Diagnostic design

The immutable full corrected-100 workload is retained because a four-request replay would change
prefill/decode cohort size, chunking and LM-head `M`. Exactly one patched-observational
stock-style run and one resident Target run are permitted. The stock-style run uses
`speculative_config=None` and the stock scheduler; the patch is present only so the worker can
observe tensors. It is explicitly ineligible for reference freezing. The resident run is expected
to remain a correctness failure if the four token divergences reproduce.

The plan in `configs/phase4b1_gate3_numerical_diagnostic.json` contains only the four established
request/zero-based-position/token pairs. Before each relevant forward, every TP rank captures its
local evidence through the TP CPU group; rank zero alone writes the combined record. It records:

- committed-prefix tokens and SHA256, computed-token count, pending input token/position;
- logical positions, block size, physical block IDs and current slot mapping;
- raw-byte per-layer and aggregate checksums of every rank's pre-forward logical KV shard;
- request cohort, sampled-row and LM-head `M` dimensions.

### Generic KV ownership authority

The pinned runner builds `slot_mappings_by_group` with `_get_slot_mappings()` before every model
forward. Independently of speculative decoding, `runner.input_batch.block_table` is a
`MultiGroupBlockTable`; each member `BlockTable` supplies the authoritative physical block row,
`block_size`, row capacity and slot mapping. The observer combines those two generic sources with
the pinned `kv_cache_config.kv_cache_groups` layer ownership. It does not read
`spec_decode_common_attn_metadata`: that object is legally `None` when the stock-style run has
`speculative_config=None`.

The observer first identifies an active planned `(request_id, output_position)` using only safe
request, token and schedule state. An irrelevant forward returns before resolving block tables or
hashing KV tensors. For a planned checkpoint, it records every KV group explicitly, requires the
block-table, slot-mapping and KV-cache group counts and IDs to agree, and fails closed on a
truncated row, missing slot, invalid block size or unmapped layer. The exact Qwen3-32B Gate3 run
must formally report one KV group on every TP rank; a different group count is retained in the
schema but is unsupported by this exact comparison and fails validation rather than silently
selecting group zero. Logical ownership comparison canonicalizes group-aware logical offsets and
layer membership while keeping physical block IDs as recorded diagnostics.

Forward hooks retain only one selected hidden row, never a full tensor dump. They summarize:

- decoder input embedding;
- last-layer branch output;
- exact fused input to final RMSNorm;
- final normalized hidden state;
- LM-head input.

Each summary records dtype, shape, raw-byte and FP32-cast SHA256, min/max/norm and deterministic
coordinates. The post-logits hook records raw pre-softmax values for both competing tokens, the
top candidates, argmax, LM-head/quant-method classes and input/output dtypes. All artifacts are
Target-only and forbidden from Draft transport.

The comparison is exact. It first compares logical prefix/KV ownership, then KV raw bytes,
decoder input, pre-norm state, normalized state, LM-head input, competing raw logits and finally
argmax versus the emitted token. It reports the first observed differing boundary but leaves the
scientific classification fail-closed for human review. A structurally valid diagnostic does not
make the resident correctness run valid.

## Interpretation after the A800 pair

- A logical prefix/position/ownership mismatch is a semantic correctness bug.
- Equal logical ownership but differing KV or hidden bytes localizes numerical state drift; the
  execution-shape metadata must support any later claim that shape caused it.
- Equal normalized/LM-head input with different raw logits localizes the difference to the LM-head
  projection/gather path.
- Equal raw logits with a different emitted token is a sampler/tie-breaking bug.
- Missing checkpoints, a changed divergence, or ambiguous evidence remains insufficient and
  fails closed.

No tolerance policy may be implemented from this evidence in the same change.
