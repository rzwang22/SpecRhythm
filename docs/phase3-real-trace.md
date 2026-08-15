# Phase 3C.1: R3-real pilot and selector diagnosis

Phase 3C.1 asks one narrow question: when Qwen3-0.6B builds a real candidate forest for prompts
tokenized by the shared Qwen3 tokenizer, do target-blind draft features contain enough signal to
approach a within-request Qwen3-32B oracle at a fixed verification budget? It does not measure
serving latency, goodput, SLO attainment, or speedup.

## Workload contract

The fixed pilot contains 100 requests: 60 code, 20 chat, and 20 summarization. The five-request
smoke uses deterministic largest-remainder apportionment and therefore contains 3/1/1. Public
text and Mooncake timing are deliberately independent:

- HumanEval or MBPP supplies code prompts;
- ShareGPT or OpenAssistant JSONL supplies chat prompts;
- CNN/DailyMail, XSum, or GovReport JSONL supplies source documents;
- the first N valid chronological Mooncake arrivals are rebased to zero in milliseconds and bound to the
  deterministically shuffled 6:2:2 task sequence;
- the configured Qwen3 tokenizer produces the stored token IDs and exact prompt length.

There is no handwritten fallback. A missing source file, insufficient valid source records,
duplicate source ID, malformed JSONL, unresolved path, or insufficient Mooncake arrivals is a hard
error. SLO classes 40/50/150 ms remain request metadata for code/chat/summarization; Phase 3C.1
never reads them for selection or evaluation.

The workload JSONL is byte deterministic for a fixed config, seed, tokenizer and inputs. Its
manifest records portable source filenames, source and arrival SHA256 values, the tokenizer
fingerprint, exact mixture, seed, time scale and output SHA256. The creation timestamp and command
are provenance fields in the manifest and do not enter the workload bytes.

## Frozen candidate-pool definition

`configs/phase3c_r3_real_1d2v.yaml` derives its dimensions from the checked-in Phase-2
`configs/simulator.json`; it does not define a second ratio convention. The frozen width is 2,
depth is 8 and speculative verification budget is 4. Therefore the 1× base contains
`width × depth = 16` non-root nodes, while 2× and 4× contain 32 and 64 nodes.

For each request, the draft stage performs best-first prefix expansion once until the shared 4×
forest is complete. The 1× and 2× pools are exact prefixes of that materialization order. Every
parent precedes its children, so all three pools are prefix closed; the same node retains the same
stable ID at every ratio. The record separates `search_pool_nodes`, the initially empty
`selected_verify_nodes`, and target-path nodes populated only by the later label join.

This is a slow correctness collector. Each expanded parent invokes one full-context draft forward,
and records `kv_cache_reuse=false`, the actual forward count, generation semantics and model
revision. Search cost is not hidden or converted into a latency claim.

## Immutable target and labels

The target stage greedily generates one continuation per request, independently of every pool
ratio. It records token IDs, token log probabilities, EOS position, model revision and forward
count. Its per-request checkpoint is immutable: `--resume` reuses matching completed records and
never overwrites them.

The CPU label join reconstructs each candidate path and adds four target-only fields:
`on_target_path`, `target_prefix_match`, `accepted_if_selected`, and
`committed_if_selected`. Missing target depths are explicit for every ratio. In particular, 1×
coverage is observed data and is never assumed to be 100 percent.

Serialized nodes contain two separate objects: `runtime_features` and `target_only_labels`.
Target-blind selectors accept only the `RuntimeCandidateNode` type. Passing a labeled node or a
runtime mapping containing a target field raises `TargetFeatureLeakageError`; the oracle is the
only selector with a labeled-node interface.

## Frozen selector baselines

Every selector sees the same request, forest, pool and per-request budget of four non-root nodes.
Selection is deterministic, prefix closed and cannot exceed the budget.

| Selector | Available information |
| --- | --- |
| `residual-probability` | descending path probability |
| `local-probability` | descending local token probability |
| `depth-normalized-log-path` | descending `log_path_probability / depth` |
| `round-robin-branch-coverage` | depth, sibling rank, branch rank and deterministic ties |
| `entropy-margin-heuristic` | `log_path + 0.25 × margin - 0.10 × cumulative_entropy/depth` |
| `within-request-target-oracle` | target labels; diagnostic ceiling only |

The entropy/margin coefficients are frozen before the GPU pilot. They must not be tuned on pilot
test rows. Requests receive a stable 70/15/15 diagnostic-train/validation/test split by hashing
their request IDs; candidates inherit their request split and are never randomly split by node.

## Outputs and interpretation

The diagnosis reports target-path pool and selected coverage, accepted and committed tokens per
proposal, accepted/verified, oracle regret and gap recovery, selected-node precision/recall,
first-error depth, prefix-closure overhead and search/verified node counts. It also emits
depth/probability and depth/entropy calibration, sibling-rank hit rate, selected-versus-unselected
calibration, depth-level task metrics and pool-expansion robustness.

The decisive comparison is every target-blind selector against the within-request oracle on the
same immutable target. The 100-request pilot is for schema, leakage, coverage and learnability
signal diagnosis only. It is too small for a final test conclusion and has no packed-tree
verification, serving-engine execution, Dual-Batch overlap, Eager, SLO scheduling or simulator
calibration.

All public data, prompt text, model continuations, checkpoints and reports belong outside Git under
`$SR_PHASE3C_ROOT`. The exact three-GPU commands are in
[phase3-gpu-runbook.md](phase3-gpu-runbook.md#13-phase-3c1-r3-real-pilot-on-three-a800-gpus).
