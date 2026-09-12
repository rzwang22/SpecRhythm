# Phase 4B.3 D3-B8 investigation at 8709c81

The supplied artifacts establish a semantic failure, but do **not** establish its root cause. This change adds the explicitly requested D3-only diagnostic; it is not a data-plane fix. KV alias, invalid frontier, streaming rebase corruption and logits misassociation must not be classified from equal continuation tokens alone. Exact HF proposal comparison remains mandatory. D4/D5 are blocked pending D3 qualification.

## Evidence and temporal boundary

Implementation: `8709c81e9ab2ded78e8f260cde4862b73e9b855a`.

The user archive `phase4b3-d3-debug-8709c81.tar.gz` has SHA256 `d9dccf27a33a3d04479475abff0ce844f4d729622613523b9d300d2619c5349a`. Its D3-B8 `hf-oracle.json` SHA256 is `4ae08efaef34ff096ba3853963ec76b503b48204ffe23ac6facf652bb82d0d43`; `gate.json` SHA256 is `a2434387008cdf3083d0cf6a39a1b1721b717250c4cd0ef29df84460927a9659`. The token fixture and these provenance hashes are preserved in `tests/fixtures/phase4b3/d3-b8-8709c81.json`.

The archive contains D2, D3-B4 and D3-B8 oracle/startup/backend/gate reports. D1 and D3-B2 passes are operator-reported; their gate artifacts are not in this archive. The 25 source fingerprints in the startup reports match the baseline's packaged API manifest, including the installed five-patch runner. The reported model is dense Qwen3-0.6B, bf16, A800 GPU0, world1, MRV1, block size 16, with 33,868 available KV blocks.

| Gate | Observation |
| --- | --- |
| D2 | All recorded proposal comparisons exact; cleanup succeeds |
| D3-B4 | All comparisons exact; proposal batch p50 3, max 4 |
| D3-B8 | Round 0 exact for 8; round 1 exact for 7; round 2 fails only request 7; proposal batch min/p50 7, max 8 |

At round 2, `batch-8-7` expects `[11,16,17,11]`, but returns `[11,227,11,227]`, identical to request 1's continuation. In `VllmBatchedDraftBackend.propose_many`, token 0 is selected from `state.next_logits`; each subsequent token is selected after `worker.materialize(rows, "proposal")`. Thus request 7's cached entry logits still select the correct first token, and the first wrong token follows the **first round-2 proposal materialization**. The EOS request retired after round 0; round 1 already exercised the surviving seven requests successfully.

This rules out one-time B8-to-B7 compaction *alone* as an explanation. B2/B4 omit request 7 and its length/history, but the artifacts cannot distinguish a length/history trigger from another batch-dependent effect. There is no proven causal explanation yet for B2/B4 passing or B8 failing specifically in round 2.

## Length and block-boundary calculation

Lengths below are counts, not last-token indices. `L0` is the initial prompt length; `M` is materialized length at proposal return; `R` is the retained frontier before correction; `C` is the resulting committed length. Proposal generation materializes `p-1` tokens, so `M = L + p - 1`; commit planning uses `R = min(M, L + accepted)` and materializes only the missing committed tail. Round 0 survivors accept zero plus correction 227; round 1 survivors accept one plus correction 227.

| Request suffix | L0 | M0 | R0 | C1 | M1 | R1 | C2 / round-2 entry | First round-2 materialized length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 17 | 18 | 18 | 19, terminal/free | — | — | — | — |
| 1 | 20 | 23 | 20 | 21 | 24 | 22 | 23 | 24 |
| 2 | 23 | 26 | 23 | 24 | 27 | 25 | 26 | 27 |
| 3 | 26 | 29 | 26 | 27 | 30 | 28 | 29 | 30 |
| 4 | 29 | 32 | 29 | 30 | 33 | 31 | 32 | 33 |
| 5 | 32 | 35 | 32 | 33 | 36 | 34 | 35 | 36 |
| 6 | 36 | 39 | 36 | 37 | 40 | 38 | 39 | 40 |
| 7 | 40 | 43 | 40 | 41 | 44 | 42 | 43 | 44 |

Request 0's terminal path frees its pages; it does not need to materialize the final EOS token for a later proposal.

Request 7 goes **43 → 44**, requiring `ceil(43/16) = ceil(44/16) = 3` blocks. It does not cross a block boundary. Only request 4 crosses a logical boundary at this step, **32 → 33**. Request 4 already materialized length 33 in round 1, so its third block was allocated earlier. The audited full-attention allocator retains the block list after a logical rollback; no new allocation is required for that first round-2 step under this implementation. This is an allocation requirement derived from source, not a measurement of the missing physical block IDs.

For request 7, round 1 materializes positions 41=11, 42=220, 43=16. Partial acceptance retains length 42, correction overwrites position 42 with 227, and the new valid length is 43. The first round-2 step writes 11 at position 43. The stale token at position 43 must not enter the attention context before this write. Other requests also exercise overwrite after partial acceptance; length arithmetic alone does not prove which GPU state was used.

## Pinned raw-logits row-domain proof

Source pin: vLLM `752a3a504485790a2e8491cacbb35c137339ad34` (0.25.1). Line numbers below refer to the **unpatched** export; installed runner SHA256 is `2905189397b1517659e6606f5bc36c7ca226330f42255c579207fe38f61f9e19`. The original five-patch stack is unchanged.

For the admitted dense, eager, text-only TP1/PP1/DP1 configuration, with no vLLM speculative decoding, async scheduling, pooling, LoRA or connectors:

1. `GPUModelRunner._update_states` removes/re-adds streaming requests, then calls `input_batch.condense()` (1470), `_may_reorder_batch` (1472) and `refresh_metadata` (1474). Reordering occurs before input preparation.
2. `execute_model` calls `_update_states` (4112), then reads the resulting `input_batch.req_ids` and gathers scheduled lengths by those IDs (4148–4150). Denote that order `r[0], …, r[n-1]` and positive scheduled lengths `s[i]`.
3. `_prepare_inputs` expands request indices in that order, gathers each row's scheduled tokens starting at its computed frontier, and builds `q = [0, cumsum(s)]` (`query_start_loc`, 2027 onward). The scheduled hidden-state segment for `r[i]` is `[q[i], q[i+1])`.
4. The non-speculative branch sets `logits_indices = query_start_loc[1:] - 1` (2193). Therefore selection index `i` is the final scheduled hidden row of `r[i]`, including ragged batches. It is not the speculative sampled-token row domain.
5. `execute_model` calls `_model_forward` (4353), then `sample_hidden_states = hidden_states[logits_indices]` and `self.model.compute_logits(sample_hidden_states)` (4387–4388). Qwen3's `compute_logits` applies the LM head/logits processor per row; vocabulary slicing and elementwise transforms preserve the row axis. The TP1 path has no cross-rank row redistribution.
6. The same `logits` object is stored in `ExecuteModelState` (4419). The admitted synchronous path has no deferred speculative correction or InputBatch reorder after construction. `complete_draft_forward` reads this raw tensor before any stock `sample_tokens`, clones/unbinds it, and calls `mapped_rows` with the current InputBatch IDs.

Consequently, **the current raw-logits/InputBatch row association is supported by pinned source for the admitted configuration**. `mapped_rows` alone was never sufficient for that proof. A CPU source test executes the pinned index/gather expressions with ragged, permuted row sentinels; another guards the relevant call order and raw-state handoff. These checks cannot prove the actual GPU buffer values or kernel correctness in the failed run.

The additional model/head source audit used `qwen3.py` SHA256 `ff7edb758557fe2a03da50bc234f484e103e698ca522226f36372825eed0eec2`, `logits_processor.py` `d6dfec7287020d587f4a4d475ce113c383b2eb55afe5ae9b0e037df2e2d13310`, and `batch_invariant.py` `1c5d11963d0849a407dee75d650a282664664b5ff409b4dbea79712d5c78ae85`. These three files are from the source pin, not additional runtime fingerprints present in the old startup artifact. The batch-invariant matrix multiply allocates its result and indexes output by matrix row; this audit found no concrete request-1/request-7 alias mechanism. It is not a GPU numerical correctness proof.

## Remaining candidate causes

| Candidate | Audited behavior | Missing proof from the failed run |
| --- | --- | --- |
| A: KV/page alias | External `KVCacheManager` uses stable internal request IDs. `Request` recreation does not replace allocator ownership; `get_block_ids` returns lists of those owned pages. Full attention retains allocation across rollback. InputBatch block add/move/swap/clear updates each row. | Actual external owners, CPU and GPU page rows, and computed physical write slots at each transition; correct IDs alone do not validate KV tensor bytes |
| B: wrong frontier | `commit_many` computes `R=min(M,L+a)`; the correction forward uses that frontier. Worker passes it through `Request` and `NewRequestData`. | Actual runner CPU/GPU computed counts, positions and sequence lengths after each rebase |
| C: stale InputBatch token row | `_update_streaming_request` removes the old row, replaces prompt/context/frontier/blocks, clears its output list and re-adds it. `add_request` copies the new prompt; condense/swap copy active row state. `refresh_metadata` resets pending batch edits each update. | Actual per-request token-row hashes and scheduled CPU/GPU input IDs after repeated updates |
| D: wrong raw-logits association | The source proof above supports current mapping. No concrete source-domain bug found in this non-speculative path. | Measured query spans/indices, IDs at logits construction and completion, returned top1 mapping |
| E: another cause | No evidence currently identifies a different concrete cause. | If all structural checks pass but the mismatch remains, investigate retained KV contents/model execution using that trace; do not relabel automatically as numerical drift |

`self.views` stores newly created request views under the unchanged internal ID; release fences, removes runner state, and frees that ID's external allocation. Resource totals are balanced (B8 allocated/freed 20/20, peak 19), but balanced totals do not rule out transient alias or stale contents.

## Diagnostic implementation and interpretation

`python -m specrhythm.phase4.draft_gate --gate D3 --diagnostic …` installs an instance-local observer after fixture initialization. Without the explicit flag, the observer is not imported or installed. Non-D3 use is rejected before GPU setup. No serving backend, Dual, Target/Serial data plane or vLLM source changes are made.

Every nonempty materialization records its purpose, round and sequence index. The first frame with `round=2` and `purpose="proposal"` is the primary failing boundary. Earlier frames retain round-0/1 transitions for comparison:

- `before_materialize`: stable/internal IDs, context length/hash, valid length, suffix, external block IDs/counts; all live ownership is also recorded.
- `after_update_states`, `req_id_to_index`: actual row IDs/indices, InputBatch and CachedRequestState context hashes, computed/prompt/active counts, scheduled counts/IDs and CPU block-table rows.
- `prepared_logits_domain`: actual GPU query starts and logits indices, post-update IDs and speculative-metadata absence.
- `immediately_before_model`: GPU input IDs, positions, computed counts, sequence lengths, request indices, block-table rows and physical slot mappings. These must match the requested suffix/frontier and private external pages.
- `logits_construction`, `after_forward`: authoritative request order, shape, top1 tokens, top2 scores/tokens and margins. Authoritative top1 uses `argmax`, since `topk` tie order need not match greedy selection.
- `completion_mapping`: InputBatch IDs at completion, returned ID order and top1 by ID; checked against construction-time evidence.
- `errors`, `checks_passed`: failed observations are retained before raising. All requests are checked, including request 7 versus request 1; shared private blocks and identical whole fixture contexts fail closed.

Records remain in memory and are written once to the exclusively created `draft-materialization-diagnostic.json` after observer removal and backend cleanup. There is no per-verification fsync logging. The observer does not edit any input, page, scheduling decision, logits or returned value. Its GPU reads synchronize and perturb timing; both the diagnostic and gate artifacts mark `diagnostic_only=true`. A diagnostic pass does **not** qualify uninstrumented B8 or eliminate a timing-sensitive fault. Structural checks also do not attest the bytes of retained KV tensors.

The companion [D3-B8-only runbook](phase4b3-d3-b8-rerun.md) collects this missing evidence. No GPU execution or AutoDL connection was performed by the coding agent. After the trace proves a cause, add the smallest adapter fix and a regression that fails the old code. After a fresh uninstrumented B8 pass, prepare the compact D1–D3 aggregate validator using retained D1/D2/B2/B4 evidence. Do not advance to D4 automatically.

## CPU regression coverage

`test_phase4_d3_diagnostics.py` replays the archive's exact B8 token/acceptance fixture through the real backend/state machine with an independent CPU KV-array oracle: eight requests, EOS retirement, seven survivors, partial acceptance/correction, then request 7's own round-2 continuation. This test passes the old data plane and therefore guards the transition pattern, **not a reproduced GPU root cause**.

Additional fault tests inject shared pages, request-1 context/frontier/page identity into request 7, wrong GPU page rows, invalid logits indices and wrong completion associations. They check preservation of failing evidence, actual permuted row order, no observer mutation, top1 tie behavior, cleanup and exclusive final report creation. The pinned source tests separately guard the raw-logits domain. CPU results and Linux CI do not constitute a GPU pass.
