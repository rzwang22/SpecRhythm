# Phase 4B.2 Dual serving-visible commit audit

The `04e9b6141e3846835e6fdee0a42cdb9e8d021e4e` A800 execution is invalid for
performance use. Its retired-ready protection worked, but the Target observer
committed a token after EOS and caused unnecessary round-1 Draft work. The strict
`d6fe1529...` terminal recovery correctly refused its contradictory prefix. Raw
artifacts remain immutable; no recovery exception or historical commit rewrite
is permitted. This repair has CPU validation only; it does not establish a new
GPU result or perform numerical divergence diagnostics.

## Pinned source and call order

Audited the local Git objects at vLLM commit
`752a3a504485790a2e8491cacbb35c137339ad34`, rather than the current vLLM checkout.
The installed custom-proposer hook patches are unchanged by this repair.

1. The sampler/rejection sampler produces the step's sampled tensor.
2. [`GPUModelRunner._bookkeeping_sync()`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/worker/gpu_model_runner.py#L3682)
   converts the ordinary one-token case to lists and clears discarded rows. For
   speculative verification it calls `RejectionSampler.parse_output()`. The result
   is `valid_sampled_token_ids`: accepted Draft prefix plus the Target correction
   or bonus, with invalid/padded entries removed. These are not raw Draft candidates.
3. [Bookkeeping caches every valid sampled token](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/worker/gpu_model_runner.py#L3725)
   in `input_batch.token_ids_cpu`, advances `num_tokens_no_spec`, and extends the
   model runner's cached request output. It does not perform serving stop checks.
4. The synchronous CPU-token proposer runs [after bookkeeping](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/worker/gpu_model_runner.py#L4636).
   The [`custom_class` dispatch](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/worker/gpu_model_runner.py#L4943)
   passes precisely that `valid_sampled_token_ids` list, plus the updated physical
   buffer/counts. The existing SpecRhythm patch also supplies request IDs and
   verification hooks. No new vLLM patch is required.
5. Later, [`Scheduler._update_request_with_output()`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/sched/scheduler.py#L1897)
   appends tokens one at a time, calls `check_stop()` after each append, and deletes
   the remaining sampled suffix at the first stop. The stopped token is inclusive.
   [`check_stop()`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/core/sched/utils.py#L94)
   checks primary EOS, explicit stop IDs, then output/context length (and optional
   repetition stopping). EOS wins if it coincides with the output budget.

## Failure and logical commit algorithm

The old observer ignored the sampled-delta argument during normal decode. It
rebuilt logical output from the entire physical row, clipped only to the output
budget, then tested whether the last token was EOS. For the supplied real evidence:

| Evidence | Tokens/count |
| --- | --- |
| Frozen prompt | 80 tokens |
| Previous logical generated output | `[45596]` |
| Current rejection-parsed delta | `[13, 151645, EXTRA]` |
| Physical row after bookkeeping | 84 tokens |
| Serving-visible final output | `[45596, 13, 151645]` |
| Required logical committed prefix | 83 tokens |

Because `EXTRA` was not EOS, the observer recorded `VERIFYING -> COMMITTING ->
DRAFT_SYNC` at length 84, and submitted a real second Draft proposal. This was a
runtime semantic error at the commit boundary, not an identity/bootstrap error
or merely missing terminal evidence.

`DualBatchRemoteProposer.propose()` now forwards the per-request sampled rows to
`_rank_zero_update()`. For each live request it:

1. Resolves the frozen prompt binding; retains the previous logical generated
   tokens and committed prefix. Duplicate/missing sampled rows fail closed.
2. Validates the rejection-parsed delta as non-negative integer tokens and checks
   the physical row equals `previous committed prefix + current sampled delta`.
   Previously unobserved tokens, regressions, replays and divergence fail closed.
3. Appends delta tokens in order, stopping inclusively at the first primary EOS,
   supported stop token, or the remaining `maximum_new_tokens` limit. No later
   token enters generated output, the commit event, the prefix hash or Draft work.
4. Validates the full sampled verification against the pending proposal before
   classifying the canonical, possibly shorter commit. Raw tokens after a rejected
   candidate cannot masquerade as accepted output.
5. Advances the committed prefix version once, consumes the current round once,
   records its acceptance accounting, and clears its pending proposal. Terminal
   output goes directly `VERIFYING -> COMMITTING -> TERMINAL`; otherwise it enters
   `DRAFT_SYNC`. The next-round cursor advances to the next unused slot (1 after
   round 0), but a terminal request never allocates that round or advances again.

The invariant is `logical committed prefix` is a prefix of the physical row. For
a live request before this step's stop, the stronger raw-delta equality above is
required. Physical post-terminal tokens may remain in the row. An empty later
callback checks the terminal prefix but creates no events/work; any further
sampled/verified terminal request fails. The proposal-free Target-tail path still
requires exactly one sampled token, terminal output and claimed tail readiness.

## Stop contract

The fixed Phase4 workload supplies tokenized prompts, per-request output limits
and seeds. Both Dual runners use explicit greedy `SamplingParams` (temperature 0,
top-p 1, one completion), with no custom stop strings/tokens, minimum-length or
repetition stopping. Target and Serial parameter construction is unchanged.

The Dual observer uses the same request-parameter factory as the Dual runner and
mirrors pinned [`InputProcessor`](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/engine/input_processor.py#L323):
`model_config.try_get_generation_config()`, `renderer_from_config(...).get_eos_token_id()`,
then `SamplingParams.update_from_generation_config()` and `update_from_tokenizer()`.
This happens during initialization, outside measured decode. It does not alter
the serving parameters. The primary EOS comes from the renderer/tokenizer, not a
guess from `hf_config.eos_token_id`; the generation configuration can supply
additional EOS IDs that [vLLM merges into stop-token IDs](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/sampling_params.py#L627).
The resulting policy is recorded in the proposer report for each frozen request.

Primary EOS is recorded as `eos`; other processed stop-token IDs as `stop`; budget
completion as `max_tokens`. Processed `ignore_eos` information is respected by the
policy, though the fixed runner does not enable it. Custom stop strings are not
inferred from token IDs. Unsupported string/minimum-length/repetition contracts,
asynchronous scheduling, and workloads whose complete output budget would exceed
the model context limit fail closed. Supporting new workload sampling options
requires an explicit adapter change and tests.

## Draft final synchronization and defensive retirement

The terminal commit still enqueues `commit_and_propose(terminal=true)`. Its name
describes the general operation; its terminal branch does not propose. Draft
checks proposal identity/round/version/hash, rolls KV back to the accepted prefix,
materializes any correction/bonus, updates committed state, and calls `finish()`.
HF rollback also materializes the last accepted Draft token when it has not yet
entered the cache, including accepted EOS. These final bookkeeping operations
remain required even when no next proposal will follow.

A serving stop within an accepted Draft prefix requires no correction/bonus.
The new Dual-only acceptance helper retains that prefix, classifies the uncommitted
proposal suffix as rejected/rolled back, and preserves token conservation. The
Serial acceptance helper is unchanged. The async controller removes in-flight and
claimed ownership after final synchronization and publishes no ready proposal/tail.
The scheduler now clears absent/finished requests from `_dual_drafting` even on a
cycle with no available poll slot; this does not change poll cadence or budgets.

All `04e9b` retired-ready checks remain: valid historical binding, strict payload
and proposal identity checks, legal drop lifecycle, consumed-proposal protection,
replay consistency and strict live stale-proposal rejection. Genuinely late work
is still safely dropped. The observed post-EOS round-1 proposal should now be
absent, rather than generated and dropped.

The terminal reconciler and offline recovery implementation are unchanged. Neither
can trim a committed prefix or repair a false commit. Tests retain the original
failure for a state prefix longer than the final output. The real `04e9b` run
must not be promoted: it performed unnecessary work, even though its outputs and
cleanup completed. Use the [fresh three-mode runbook](phase4b2-fresh-three-mode-runbook.md)
for one future Target, Serial and Dual run at one common new execution commit.

## CPU validation scope

Tests exercise the actual proposer entry point, state transitions, Draft machine,
async controller and HF rollback using CPU-only backends/caches. Coverage includes
one-token and full/partial acceptance, EOS as final/interior sampled token, terminal
accepted-prefix truncation, exact budget clipping, post-EOS physical suffixes,
terminal idempotence, malformed/missing evidence, model/stop-token policy sources,
Target-tail behavior, strict retirement and recovery refusal. The supplied
80-prompt-token regression ends with `[45596, 13, 151645]`, length 83, TERMINAL,
and no round-1 proposal. These fixtures are synthetic reproductions of the supplied
facts, not newly collected A800 evidence.
