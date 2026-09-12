# D3 same-prefix raw-logits discrimination probe

This is a diagnostic experiment, not a correctness fix or a D3 qualification pass. It does not run D4/D5, change Dual/Serial serving, scheduling, proposal budgets, UUID caching or the vLLM five-patch stack. The coding agent runs CPU tests only; the operator performs the A800 experiment.

## Frozen inputs and four paths

The packaged `src/specrhythm/phase4/d3_logits_probe_inputs.json` is derived from the previously frozen same-prefix probe inputs and original D3 oracle, with source diagnostic SHA256 `6483c7c957636d13808570ab8c8538ef145f47a2115bc1ac2fec7e19c2ee3ec4`. Its complete file hash is pinned in `draft_logits_contract.py`; all seven committed/comparison prefix hashes and the original commit history are verified before GPU execution. Tokens are never reconstructed by tokenizing approximate prompts.

The distinction between **committed prefix** and **comparison prefix** is explicit: request 7 enters frame 8 with 43 committed tokens; input token 11 at position 43 produces the 44-token comparison prefix whose raw next logits are measured. Its comparison hash is `6b15e8e5cb12ec01cc344791bbd8649df17fc5a7225e6cb4a6b20233a236d374`.

| Path | Setup | Measured raw-logits boundary |
| --- | --- | --- |
| A: `hf-fresh` | Existing HF correctness backend, empty state, full 44-token request-7 prefix | Fresh prefill's last-position raw logits |
| B: `vllm-fresh-singleton` | Fresh MRV1 backend, only request 7, scratch prefill of its 43-token committed prefix | One B1 forward appending frozen token 11 at computed=43 |
| C: `vllm-fresh-b7` | Another fresh MRV1 backend, exact seven surviving requests in frozen order, each committed prefix prefilled from scratch | One B7 forward appending the frozen token 11 to each request; request 7 computed=43 |
| D: `vllm-persistent-b7` | Another fresh MRV1 backend, exact initial B8 and original proposal/commit/EOS history from the frozen oracle | Original frame-8 first round-2 proposal materialization, measured before completion mapping |

B/C use the same singleton prefill procedure for request 7 and the same final one-token suffix; their final forward differs in batch size. C's other six requests are live with their own fresh pages; it never calls proposal sampling or replays rejected suffixes. Absolute page numbers need not equal D's reused allocation, but per-request binding, computed lengths, positions, private pages and slots are validated by the existing D3 observer.

D replays the original **vLLM request history** using immutable oracle data. It does not regenerate the HF oracle or approximate the prompts. Earlier rounds must remain exact. It completes the original round-2 proposal sequence while retaining the raw frame-8 capture; its known HF mismatch is evidence rather than a probe infrastructure failure. If the first cached token or earlier history changes, the same-prefix guard invalidates the capture instead of silently forcing tokens. If D no longer selects 227, the classification retains the result and flags possible diagnostic perturbation; no automatic retry occurs.

Every path runs in a separate sequential subprocess. A terminates before any vLLM child starts, preventing both simultaneous model residency and contamination of HF by vLLM's process-global torch operator overrides. B/C/D each start with zero live requests/KV, then construct the appropriate fresh or historical state. Worker state, PID, lifetime UUID, setup identity, actual capture rows and cleanup are recorded. D's fresh process differs from the old combined HF-oracle/vLLM gate process; an unreproduced result must therefore be retained and investigated, not treated as a fix.

## Provenance and precision

CPU preflight verifies the exact execution Git commit, clean editable checkout, qualified config hash, model/tokenizer metadata, Python 3.11, PyTorch 2.11.0, vLLM 0.25.1, source Git pin, all 25 installed API file hashes, original five patch files and six additional numerical-source hashes. It fingerprints the actual safetensors weights. Each child repeats these checks against the parent's immutable inputs, preventing changed weights/config/source between paths. Workload/reference or previous server result directories are not dependencies of this probe.

All children require GPU0, BF16, MRV1, eager/private KV, TP1/world1 and `VLLM_BATCH_INVARIANT=1` where applicable. Each path records a real UUID/device identity. HF uses the existing model loader and reports its actual attention implementation; vLLM reports its actual attention implementations/FA version and effective batch-invariant configuration. No kernel or attention implementation is forced to make results agree.

The request-7 tensor is captured directly from raw model logits, before sampling policy or softmax, and converted with `.float().cpu().tolist()`. Every path stores the **full 151,936-element float32 CPU vector**, original dtype, lossless JSON values, little-endian float32 content hash, token 16/227 values, top-10, top1, signed `16−227` margin and absolute delta. Top-k ties use ascending token ID; top1 matches greedy argmax. No display rounding is used.

The classifier recomputes summaries from the stored vector. It checks fixture/prefix/request identity, clean fresh controls, independent lifetimes, cleanup and common model/config/GPU provenance. It reports B−A, C−B and D−C token-score/margin deltas and full-vector equality/max absolute delta. Full vectors remain in per-path JSON files, whose hashes bind them to `classification.json`.

## Classification and conditional follow-up

| A/B/C/D top1 pattern | Report |
| --- | --- |
| 16 / 227 / 227 / 227 | Case A: strong evidence for HF–vLLM execution numerical divergence |
| 16 / 16 / 16 / 227 | Case B: strong evidence for persistent-history/KV/rebase difference; prepare actual K/V content comparison next |
| 16 / 16 / 227 / 227 | Case C: strong evidence for batch-shape-dependent vLLM execution difference |
| Any mixed/unreproduced outcome | Inconclusive, retain exact values and discriminating follow-up |
| Missing/invalid path or failed fairness checks | Invalid evidence, no positive classification |

These labels are evidence categories, not proof of a particular faulty kernel or line of code. `mechanistic_root_cause_proven=false`, `correctness_fix_authorized_by_result=false` and `d4_d5_allowed=false` remain explicit. The original exact D3 gate is not relaxed.

If B/C differ in **any raw-vector value**, even if top1 matches, the result requests the batch-invariance audit and includes its pinned source basis and actual runtime evidence. The audited source declares FlashAttention batch-invariance support, sets `max_num_splits=1`, and on A800/SM80 routes unquantized linear through `linear_batch_invariant → matmul_persistent`, with fixed BF16 tiling and torch mm/linear overrides. For actual FA2, `FlashAttentionImpl.forward → flash_attn_varlen_func → torch.ops._vllm_fa2_C.varlen_fwd` receives the split control. These declarations target supported **vLLM** execution; they do not promise HF equivalence or prove end-to-end raw-logit equality for this streaming-rebase path. A B/C mismatch requires examination of the captured effective flags, actual FA version, source fingerprints and the actual kernel implementation before any patch.

If C selects 16 and D selects 227, the result requests an actual K/V comparison at matched layer/token positions. This probe intentionally contains no K/V content expansion. If A itself differs from the old HF token 16, the next control is fresh HF versus original HF history at the same prefix, before any vLLM overrides. If D does not reproduce 227, retain it and compare D-only original process setup; do not force a classification.

## Operator artifacts

The [fresh-server runbook template](phase4b3-d3-logits-probe-runbook.md) is delivered with its commit marker replaced by the exact new SHA. One invocation creates an exclusive result directory containing:

- `probe-inputs.json` (frozen fixture plus execution/model/source identities);
- `hf-fresh.json`, `vllm-fresh-singleton.json`, `vllm-fresh-b7.json`, `vllm-persistent-b7.json`;
- per-path logs and `classification.json`;
- `run.log` from the enclosing operator command.

The operator root contains a fresh `probe/` directory so the run log can be created before the probe starts without violating exclusive creation. Files are written once after capture/cleanup, not per materialization. Failed paths retain their report or a parent-created failure record plus process log. A zero return code means all four observations completed, **not** that D3 passed. The operator returns the full packaged directory. No GPU execution is performed by the coding agent.
