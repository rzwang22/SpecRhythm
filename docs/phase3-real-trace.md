# Phase 3C.3: corrected multi-round validation and 2x shell learnability

Phase 3C asks whether draft-only Qwen3 features contain useful signal for selecting a fixed budget
of candidate nodes against one frozen Qwen3 target trajectory. It is a correctness and selector
diagnosis pipeline. It does not implement packed-tree verification, a serving engine, Dual-Batch,
Eager, SLO scheduling, GPU performance evaluation, or simulator calibration. Phase 3C.3 adds one
small learned selector as an offline diagnostic; it is not a serving-system mechanism or a new
default.

## Evidence and prompt boundary

The legacy Phase 3C.1 pilot contains 100 requests (60 HumanEval code, 20 ShareGPT chat and 20
CNN/DailyMail summarization), a Qwen3-0.6B draft, a Qwen3-32B target, a four-node verification
budget, and nested 16/32/64-node pools. Its raw forests, target trajectories, labels and selector
checkpoints remain immutable. They may be re-summarized without another model run.

The prompt audit found one incompatibility: the Phase 3C.1 ShareGPT adapter used the first user
turn as raw text and did not apply the Qwen chat template. Phase 3C.2 applies the configured
tokenizer's `apply_chat_template` with `add_generation_prompt=true` and
`enable_thinking=false`. Therefore the old 100-request trace is legacy diagnostic evidence and
must not be presented or pooled as the corrected Phase 3C result. HumanEval remains its native
code-completion `prompt`
without a synthetic instruction. CNN/DailyMail retains the explicit `Summarize the following
document:` instruction.

The v2 workload manifest records, for each task, source fields, instruction semantics, chat-template
use, thinking mode, special-token metadata, truncation direction, maximum context and one
deidentified rendering example. Overlength records are rejected and replaced; text is never
silently truncated. Public text, rendered prompts, token IDs, model outputs and reports remain
outside Git.

## Corrected coverage vocabulary

All nested pools share one eligible denominator:

```text
eligible target nodes = min(frozen target length, shared 4x forest depth)
target_path_recall = target nodes present in pool / eligible target nodes
target_node_density = target nodes present in pool / pool nodes
selected_target_precision = selected target nodes / selected nodes
selected_target_recall = selected target nodes / eligible target nodes
```

`target_path_recall` is monotonic for nested pools. The report also records
`full_target_trajectory_covered`, `first_missing_target_depth`, the first missing depth inside the
four-node verification horizon, and target recall at K=4, 8 and 16.

The v1 field `target_path_pool_coverage` is retained unchanged for schema migration. It divided
target nodes by `min(target length, that pool's own maximum realized depth)`. It was neither
density nor a fixed-denominator recall, so it could decrease when a larger pool reached a deeper
level. The v2 report labels that legacy definition explicitly instead of silently changing it.

Likewise, “88 of 100 requests missing full 1x coverage” would mean that some target depth in the
entire (up to 16-token) trajectory is absent. It does not mean 88 requests fail in the first
four-token proposal. The report separately counts missing full trajectories and missing K=4
horizons.

## Pool shells and selection stability

Every request uses one best-first, prefix-closed 4x forest. Its materialization order defines:

```text
1x base  = nodes [0, 16)
2x shell = nodes [16, 32)
4x shell = nodes [32, 64)
```

Per request and shell, the report stores node and target-node counts, density, minimum/maximum
target depth, budget-four reachability, prefix-enabling nodes, oracle-selected shell nodes and
per-target-blind-selector shell selections/hits. Thus an oracle gain from 1x to 2x can be traced to
specific added nodes; a zero 2x-to-4x gain means no additional prefix-closed budget-four target
continuation was usable on those snapshots, not necessarily that the 4x shell contained no target
labels.

For each target-blind selector, 1x→2x, 2x→4x and 1x→4x comparisons report exact selected-set match,
Jaccard, displaced budget, new-shell selections and target hits. This distinguishes identical
Residual-Probability selections from changed selections with coincidentally equal accepted-token
outcomes or aggregate cancellation.

## Request-level statistics and headroom

Code, chat, summarization and all-request rows are reported separately for every pool and selector.
Accepted/committed tokens, coverage and gap metrics include mean, median, P25/P75, P90, population
standard deviation and a deterministic request-bootstrap 95% interval. The all-request bootstrap
preserves task strata. Candidate nodes are never treated as independent samples. Each row also
contains win/tie/loss versus Residual-Probability, paired delta versus the within-request oracle,
and absolute/relative oracle gaps.

Headroom is separated into generator coverage (oracle gain from pool expansion), selector regret
(budget-four oracle minus target-blind selection), budget constraint (unbudgeted contiguous target
path minus budget-four oracle), and expansion utilization. A zero oracle expansion gain produces
`null` and `identifiable=false`, never a fabricated zero. Cases are named explicitly: no useful
target nodes, useful nodes blocked by budget, budget-reachable nodes missed by the selector, or a
realized selector gain.

## Corrected multi-round common-prefix contract

The completed corrected smoke uses 20 requests (12/4/4); the formal server run uses 100 requests
(60/20/20). Each freezes at most 16 greedy target tokens once per request. For every target prefix
before EOS/limit, the draft stage creates exactly one common
snapshot. It records request/task, prefix position, context length, remaining target tokens, full
target hash, forest hash, nested-pool hashes, target continuation and budget. Stable node IDs
include the prefix position. All selectors consume the same serialized snapshot; no selector can
regenerate a forest or target.

Sequential replay starts at prefix zero, selects four prefix-closed nodes, accepts the longest
matching draft prefix, commits those tokens plus a target correction/bonus when necessary, and
advances along the frozen target. It reports rounds, accepted/committed/verified tokens, total
verified nodes, oracle regret and first-versus-later-round acceptance. Every final committed token
sequence must equal the immutable target-only trajectory. Checkpoints are one atomic file per
snapshot/request and `--resume` never replaces a differing record.

This remains full-context Transformers correctness collection with `kv_cache_reuse=false`. No
latency, goodput, SLO attainment or speedup may be inferred from it. Exact server commands are in
[phase3-gpu-runbook.md](phase3-gpu-runbook.md#15-phase-3c3-corrected-100-request-multi-round-and-learned-pilot).

## Phase 3C.2 corrected-20 observation

The user-run corrected 20-request trace passed target immutability, common-snapshot, nested-pool,
prefix-closure, final-token equality and candidate/root accounting checks. Residual-Probability
accepted 2.407 candidates/proposal at 1x, 2x and 4x; Entropy-Margin accepted 2.421. The
within-request target Oracle accepted 2.680 at 1x and 2.877 at both 2x and 4x. These values support
the narrower hypothesis that the 2x-minus-1x shell contains sparse useful nodes which the frozen
target-blind selectors usually miss, while 4x adds no visible Oracle headroom in this small pilot.
They are token-efficiency diagnostics, not speedup or GPU-performance measurements.

## Formal Phase 3C.3 statistics

The corrected 100-request report treats requests, not snapshots or candidate nodes, as the
sampling unit. For every task and overall, it reports request and snapshot counts, proposal
rounds/request, accepted and committed tokens/proposal, verified candidates/request, first- and
later-round acceptance, and Oracle regret/request. Deterministic 95% bootstrap intervals resample
requests while preserving code/chat/summarization strata. Paired request-level deltas cover
Entropy-Margin minus Residual-Probability, Oracle-minus-Residual at 1x and 2x, Oracle 2x-minus-1x,
and the diagnostic Oracle 4x-minus-2x comparison.

Formal selector evidence is limited to the five frozen target-blind selectors plus the
within-request target Oracle at 1x and 2x. The 4x pool is retained only for the Oracle 4x-minus-2x
diagnostic; target-blind 4x rows are not promoted as formal evidence, and no 8x trace is created.
The Oracle reads target outcomes and is an upper bound, never a deployable selector.

## 2x shell opportunity and feature separation

Every common snapshot is decomposed into its 1x base and 2x-minus-1x shell. The report keeps base
and shell node counts, target-path nodes, budget-four reachability, selector selections and
accepted progress separately. The four flags have distinct meanings:

- `shell_target_present`: generator coverage exists somewhere in the shell;
- `shell_target_budget_reachable`: a shell target node has depth at most four and can be reached
  with its prefix;
- `oracle_uses_shell`: the budget-four Oracle actually spends verification budget in the shell;
- `oracle_shell_gain_positive`: Oracle 2x accepted progress exceeds Oracle 1x progress.

This prevents pool coverage, budget/prefix-closure reachability and ranking failure from being
collapsed into one ambiguous coverage number. Residual-Probability and Entropy-Margin shell
selection recall and gains conditioned on each opportunity flag are reported by task and overall.

The offline feature dataset serializes `runtime_features` and `target_only_labels` as separate
objects. Runtime fields include draft probabilities/log-probabilities, depth, sibling rank,
parent probability, entropy/margin, branching factor, configured remaining generation budget,
prefix/round progress, task indicators and base/shell membership. Target path membership, Oracle
selection and Oracle shell-gain contribution remain labels only. AUROC, AUPRC, precision@4,
recall@4, NDCG@4 and positive prevalence are reported on the held-out request split for all
budget-reachable nodes, reachable 2x shell nodes and Oracle-opportunity snapshots.

## Diagnostic learned-shell-ranker and decision gate

`learned-shell-ranker` is a deterministic, balanced linear logistic model trained with fixed
hyperparameters. Request IDs are split 70/15/15 into task-stratified train/validation/test sets;
rounds from one request never cross splits. Validation and test data do not choose features,
thresholds or hyperparameters. Training may read the offline target-path label, but inference is
typed to accept `RuntimeCandidateNode` only and rejects labeled nodes. Greedy selection remains
budget-four and prefix closed. The split, feature list, coefficients, training-set hash, model
hash, source hashes and immutable replay checkpoints are retained in the artifact manifest.

Held-out comparisons report Residual-Probability, Entropy-Margin, the learned ranker and the
within-request Oracle. Oracle gap recovery is defined only for requests with a positive Oracle
minus Residual denominator. The A/B rule is frozen before server evaluation: Outcome A requires
an overall paired improvement and gap-recovery interval with lower bound above zero, plus no
represented task whose paired-delta interval falls below zero. Outcome A only recommends a later
2x Overdraft-and-Prune packed-tree prototype. Otherwise Outcome B keeps 1x
Residual-Probability and proceeds only to a Dual-Batch/overlap baseline. Neither outcome changes
the current selector automatically.
