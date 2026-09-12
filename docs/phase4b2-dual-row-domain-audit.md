# Phase 4B.2 bookkeeping row domains and bounded failure cleanup

The A800 execution at `66207c521b8b313080583fc47265704bb7a6c66c` failed before a
clean Dual performance result. Its supplied root is immutable historical evidence:

`/root/autodl-tmp/SpecRhythm-data/results/phase4/66207c521b8b313080583fc47265704bb7a6c66c/phase4b2-logical-commit-20260905T102753Z-1526`

The observer raised its aligned-row length check on the first speculative step.
The check remains; its inputs now come from an explicit validated projection.
EOS/stop/max-token canonicalization, proposal acceptance, retired-ready handling,
terminal recovery, metric formulas and cross-mode provenance rules are unchanged.
No GPU run or numerical-divergence investigation is part of this coding repair.

## What the pinned source actually establishes

Audited Git objects at vLLM `752a3a504485790a2e8491cacbb35c137339ad34`, including
the existing four Python patches. Let `N` be this step's request-row count and
`C` the physical storage capacity (`max_num_reqs`), generally `C >= N`.

| Object | Domain and ordering | Unscheduled, discarded and terminal rows |
| --- | --- | --- |
| Engine/model-runner cached requests | Persistent request identities, potentially the whole 100-request cohort | Can retain requests absent from the current step |
| `scheduler_output.num_scheduled_tokens` | One entry per scheduled request; values are input token counts, not row indices | Excludes unscheduled requests |
| `scheduled_spec_decode_tokens` | Subset of scheduled IDs owning speculative candidates | Does not describe every sampled row; bootstrap/tails may have no proposal |
| `input_batch.req_ids` / `req_id_to_index` after `_update_states()` | `N` dense active rows, after removal, addition, compaction and attention-backend reorder | Unscheduled and previously finished rows removed; cached request state can remain elsewhere |
| `sampler_output.sampled_token_ids` | `N` rows in the stabilized input-batch order; one-token width or speculative output width | Discarded sampling rows can still occupy a row |
| `_update_states_after_model_execute()` | Uses sampled counts for hybrid-model state bookkeeping; does not assign a new output row order | Returns immediately for the non-hybrid Qwen models in this workload |
| `req_ids_output_copy` / `req_id_to_index_output_copy` | Bookkeeping copies of the active ID list and its exact row-index map | Includes the row for an empty discarded result and the step that produces a terminal result |
| `valid_sampled_token_ids` | `N` rejection-parsed output lists; row `i` belongs to `req_ids_output_copy[i]`, whose copied index is `i` | Discarded rows become empty lists, not removed rows; serving EOS truncation is still later |
| `num_tokens_no_spec` / `token_ids_cpu` | Storage of `C` counts and `C × max_model_len` tokens; access by current physical request index | Unused capacity is not an active request list; its contents must not be consumed |

Source anchors:

- [`_update_states()` removes unscheduled requests](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/worker/gpu_model_runner.py#L1187)
  and [condenses/reorders before sampling](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/worker/gpu_model_runner.py#L1469).
- [`_update_states_after_model_execute()`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/worker/gpu_model_runner.py#L1522)
  is hybrid-state bookkeeping, not a request-row compaction step.
- [`_bookkeeping_sync()`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/worker/gpu_model_runner.py#L3656)
  copies IDs/maps, parses sampled output, retains empty discarded rows and writes
  valid tokens into the corresponding current physical rows.
- [`InputBatch` allocates the full count capacity](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/worker/gpu_input_batch.py#L149);
  its `req_ids` list is not that capacity-sized array.

This qualifies the initial failure hypothesis: the pinned synchronous path does
not prove that all 100 active requests remain in `input_batch.req_ids` at the
one-request step. It explicitly removes unscheduled requests. In this exact path,
the output copy and current active list ordinarily agree. What is demonstrably
wrong is requiring the full capacity-vector length to equal sampled/request-row
count. The supplied trace does not contain all three lengths, so it cannot identify
every failing operand. Merely substituting output-copy IDs would still leave the
capacity comparison wrong. The repair explicitly distinguishes all three domains.

## Explicit mapping and TP contract

The new `0005-dual-sampled-row-context.patch` carries one structured context from
the post-bookkeeping call through the custom-proposer dispatch:

- `sampled_request_ids = req_ids_output_copy.copy()`
- `req_id_to_sampled_index = req_id_to_index_output_copy.copy()`
- `physical_request_ids = input_batch.req_ids.copy()`
- `req_id_to_physical_index = input_batch.req_id_to_index.copy()`
- scheduled and speculative request-ID sets from the same scheduler output.

For each sampled index `i`, `dual_rows.align_sampled_rows()` proves the output-copy
map identifies `sampled_request_ids[i]` at `i`, then uses that ID to look up physical
slot `j`. Only `num_tokens_no_spec[j]`, `token_ids_cpu[j, :count]` and the corresponding
materialized count are projected. `i == j` is never assumed. The context retains
explicit physical indices for audit, while the observer sees strictly aligned
request/delta/physical rows. Empty discarded results retain their identity.

Duplicate/missing IDs, inconsistent copied indices, aliases, mismatched scheduler
sets, invalid counts/tokens, insufficient physical capacity, or sample-count
mismatch fail before state mutation. The existing `_rank_zero_update()` length,
uniqueness, rejection and physical-prefix checks remain unchanged. No zip truncation,
positional inference after compaction, stale tolerance or physical-prefix-based
logical reconstruction was introduced.

Both TP ranks validate locally, then all-gather success/error and a digest of the
ordered logical projection. Physical slots may differ only if their projections
agree. Any rank's invalid mapping or disagreement is fatal on both ranks before
rank-zero state mutation. Rank-zero commit errors are also broadcast as failures
so the other rank does not wait indefinitely for a success broadcast.

The unchanged EOS regression still maps `[13, 151645, EXTRA]` onto prior `[45596]`,
commits only `[45596, 13, 151645]`, ends at logical length 83 and TERMINAL, performs
final Draft synchronization and creates no round-1 proposal. CPU tests run this
through a physical slot at index 73 with 256-row storage, as well as partial rejection
and max-token completion. No real GPU outcome is claimed by these fixtures.

## Independent process-lifecycle repair

The old supervisor polled only the outer coordinator PID. The timestamp wrapper
blocked reading until pipe EOF before waiting for its child. A live TP worker
could retain a pipe writer after the internal coordinator failed, while fatal
workers could also leave the coordinator itself alive. Consequently the supervisor
never entered its otherwise bounded cleanup branch.

The supervisor now also watches incremental Target output for pinned fatal
messages (`WorkerProc failed`, `EngineCore encountered a fatal error`, unexpected
worker death/shutdown, or `EngineDeadError`) and reads nonzero owned zombie-child
status on Linux. The [pinned worker monitor](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/executor/multiproc_executor.py#L268)
reports unexpected deaths even when the parent reaps a worker before the next poll.
The shared-memory wait warning alone is not treated as a failure, since long
healthy initialization can also wait.

Ownership tracks PID plus process start identity and observed parent/child
relationships, rather than process names. Target starts in a new session. On Linux
the supervisor temporarily enables child-subreaper behavior and adds a unique
inherited Target launch token, allowing adoption, identification and reaping of
orphan workers, including detached descendants. Signals target only proven owned
PIDs; an interactive shell, unrelated process or reused PID is not selected.

On internal failure, cleanup begins while the outer wrapper is still alive:
TERM, bounded wait, KILL fallback, then reaping and verification. Draft and its
observed descendants receive their own bounded TERM/KILL cleanup. The supplied
Draft PID must have the actual Unix socket open at initial observation; removal
also requires process termination and an unchanged device/inode/mode identity.
An unrelated or replaced socket is preserved and cleanup fails closed.

Defaults are 50 ms polling, 5 seconds TERM wait and 2 seconds KILL wait per Target
and Draft cleanup, plus enumeration/I/O overhead. The timestamp wrapper separately
polls its direct child and bounds post-child output draining to 250 ms, so inherited
writers cannot keep it waiting for EOF indefinitely. It preserves timestamped-log
schema and normal child status propagation. A detected failure always yields a
nonzero effective status; it is never promoted by successful cleanup. Lifecycle
artifacts retain detection reason, observed identities, signal actions and survivors.

CPU subprocess tests cover a failed worker with a blocked sibling, coordinator
termination with inherited pipe writers, detached Target children on Linux,
TERM-resistant Target/Draft, KILL fallback, socket ownership/replacement, and the
actual `phase4b2_run_mode` shell path. The interactive caller survives with nonzero
status and no Target/Draft processes remain. Linux-specific exit-status/adoption
tests run in Linux CI and are explicitly skipped on macOS.

## Patch migration and fresh provenance

The first four patch files remain byte-for-byte unchanged. The exact new runner
SHA256 is `2905189397b1517659e6606f5bc36c7ca226330f42255c579207fe38f61f9e19`;
scheduler SHA256 remains `ffaefd61869589f086e6acdf9a0c4f55f80d5dad145ca3f6fff2379f7a4e2455`.
The old four-patch hash is recognized for strict restore only. Current installed
checks and runner validation require all five patches. Local CPU validation applies,
restores and reapplies against the full pinned source with exact hashes, and compiles
the resulting Python without importing vLLM or starting GPU work.

Use the [fresh three-mode runbook](phase4b2-fresh-three-mode-runbook.md). It preserves
the exact failed root, requires old workers to be cleared before installed-code
migration, writes a new apply manifest outside the old result root, and runs future
Target → Serial → matched-work pair → Dual → Dual performance → three-mode comparison
under one common new commit. Every command captures its return code; failures
require manual stopping, not termination of the interactive Bash. No old mode output
or failed Dual directory is reused for the final baseline.
