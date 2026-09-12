# Phase 4B.4A — source-first microbatch audit

Audited before implementation against SpecRhythm
`7b6367efdf81edb61d9a6641d0270bef01920512` and pinned vLLM
`752a3a504485790a2e8491cacbb35c137339ad34` (0.25.1). No GPU executed.

## Origin and propagation

1. `integrations/vllm/phase4b1_gate_helpers.sh:phase4b1_run_mode` hardcodes
   `--microbatch-size 2` in the resident Dual command. D6 calls B.2, which calls
   this helper. This is the effective D6 origin, independent of YAML.
2. `src/specrhythm/cli.py` resident Dual argument and
   `dual_runner.run_resident_dual_batch` also default to 2. The legacy nonresident
   B0 CLI/runner default is 1 and remains unchanged.
3. `dual_runner._configure_resident_environment` delegates to
   `_configure_environment`, setting the existing private transport variable
   `SR_PHASE4_DUAL_MICROBATCH_SIZE`. Each Target scheduler loads it in
   `vllm_dual_scheduler.DualBatchScheduler.__init__`.
4. `DualBatchScheduler.schedule` counts already-ready speculative or Target-tail
   requests in `running`, then calls nonblocking `poll_ready` with
   `max(0, N - already_ready)`. The existing readiness hook controls admissibility
   before stock allocation. Stock `super().schedule()` chooses actual requests;
   `scheduled_spec_decode_tokens` keys are translated from internal IDs to stable
   `verify_request_ids` in the scheduler event. No request partition or waiting
   until N ready is involved. Sweep coordination remains `none`.
5. `vllm_dual.DualBatchRemoteProposer.on_target_verify_start` intersects those keys
   with runner request IDs and claims their proposals. TP verifies the cohort.
   `_rank_zero_update` aligns physical sampled rows and builds one `commit_rows`
   message for verified IDs. Existing `BatchedDualDraftController` preserves the
   message; `BatchedDualDraftMachine._commit_many` calls `commit_many`, then
   `_propose_many` for surviving requests. Terminal retirement/tails are unchanged.

## Independent limits

Ready-set size, setup readiness, retirement/EOS, the stock scheduler's sequence
capacity, token budget, per-request K+1 query demand, remaining context, long
prefill threshold, and KV allocation/preemption can all lower actual cohorts.
The resident Target constructor does not override max_num_seqs or
max_num_batched_tokens. No additional tuning knob is introduced.

Pinned vLLM `engine/arg_utils.py:get_batch_defaults` uses LLM defaults of 16384
batched tokens/1024 sequences on devices with at least 70 GiB whose name excludes
A100; otherwise 8192/256, followed by further config constraints. Do not infer
actual A800 settings from its name alone. Stock
`v1/core/sched/scheduler.py` loads `max_num_running_reqs` from `max_num_seqs` and
`max_num_scheduled_tokens` from its explicit value or `max_num_batched_tokens`.
Scheduling clips to token/context budgets and `allocate_slots` can fail/preempt;
the running sequence limit also bounds admission. The qualified Draft adapter's
separate max_num_seqs=128 already admits the initial B100 and is unchanged.

## Narrow implementation

`SR_PHASE4B_DUAL_MICROBATCH_SIZE` is the single operator control read by the
resident shell helper **only for Dual**, before creating the run directory or
starting Draft. Unset means 2; positive decimal integers are accepted; zero,
negative, fractional or malformed values fail early. Direct resident CLI users
retain the existing `--microbatch-size` interface and default 2. The private
worker environment variable is transport, not a second characterization control.
Target/Serial do not parse the public variable, even if malformed. HF/production
Dual keep 2 absent explicit selection. There is no automatic tuning or new cap.

`requested_dual_microbatch_size` and `effective_dual_microbatch_size` are emitted
by the plugin, scheduler, raw resident report, runtime manifest, measurement,
cell qualification and sweep. Effective means **scheduler-loaded upper bound**,
not attained batch size. Raw finalization cross-checks the CLI request against
scheduler readback and the plugin; qualification joins all four canonical reports
and checks actual verification sizes. A final immutable `dual-microbatch.json`
sidecar binds N to every runtime JSON/JSONL file, including unchanged backend
reports. Scheduler events also record loaded max_num_seqs, max_num_scheduled_tokens,
max_model_len and KV cache block capacity. No new per-verification fsync or logger was introduced.

The explicit `characterization` overlap policy allows zero supported overlap.
It checks timing intervals, synchronized CUDA events and per-rank GPU identities,
then reproduces material existing witness fields. The reported overlap duration
is the union of actual cross-request interval intersections, without summing
shared rows or filling gaps between intervals. Historical witness envelopes and
logging are unchanged. Default D6's required-overlap gate remains intact.

No backend, proposal/acceptance/K, sampled-row mapping, retirement, scheduling
selection/cadence, UUID query mode, measurement boundary, logging/fsync, Target,
Serial or vLLM patch file changed. Only the existing upper bound is varied, with
metadata and CPU characterization reporting. The five-patch stack stays pinned.
