# Phase 4B.1 Gate3 matched-bootstrap control

> Historical diagnostic contract. Its immutable artifacts retain their original
> `phase4b2_blocked=true` field. After the successful `ASYNC_OFF_MATCHES_STOCK` control, the
> 2026-09-05 human engineering decision completed numerical qualification and separately set
> `phase4b2_progression_permitted=true` without claiming exact stock equivalence. See
> [project status](project-status.md) and the
> [Phase 4B.2 runbook](phase4b2-decode-performance-runbook.md).

This is a Target-only numerical diagnostic. It does not close Gate3, change exact-correctness
policy, or authorize Phase 4B.2 or performance work.

## Pinned-vLLM async-scheduling audit

The authority is pinned vLLM `752a3a504485790a2e8491cacbb35c137339ad34` (`v0.25.1`):

- `SchedulerConfig.async_scheduling` is a public `bool | None` field; explicit `False` disables
  async scheduling. `get_scheduler_cls()` resolves false to the ordinary
  `vllm.v1.core.sched.scheduler.Scheduler`, while true resolves `AsyncScheduler` ([pinned
  source](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/config/scheduler.py#L140-L167)).
- `EngineArgs` exposes that field and the CLI exposes `--async-scheduling`; engine-config
  construction passes it directly into `SchedulerConfig` ([field](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/engine/arg_utils.py#L671),
  [CLI](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/engine/arg_utils.py#L1391-L1408),
  [construction](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/engine/arg_utils.py#L2053-L2073)).
- When the field is `None`, pinned vLLM enables async scheduling unless an incompatible option
  intervenes. Explicit `False` skips that defaulting branch and is logged as disabled ([pinned
  resolution](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/config/vllm.py#L894-L977)).
- Offline `LLM` forwards extra keyword arguments to `EngineArgs`, then constructs its normal
  `LLMEngine` ([pinned entry point](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/entrypoints/llm.py#L289-L336)).

The chosen mechanism is therefore the supported
`LLM(async_scheduling=False)` argument. SpecRhythm does not set `scheduler_cls`, enable
speculation, or modify vLLM defaults. The ordinary stock runner still passes
`speculative_config=None`. This option is accepted only for a Target, corrected-100,
batch-invariant, diagnostic-single-run invocation with the existing per-token plan and both
diagnostic outputs.

After engine initialization, rank-zero and every TP worker must prove:

- async requested and effective are both false;
- `speculative_config` is null;
- configured scheduler class is null and the resolved class is
  `vllm.v1.core.sched.scheduler.Scheduler`;
- custom-class proposer and `ResidentSetupScheduler` are absent;
- no global resident setup/freeze or Draft process belongs to the control;
- TP is 2 on physical GPUs 1 and 2.

Any failure stops before a comparison can be valid.

## Artifacts and schemas

The stock-smoke JSON and runtime bundle embed
`specrhythm.phase4b1-gate3-matched-bootstrap-control.v1`. It records the explicit/effective
async state, null speculative configuration, configured/resolved scheduler, absence of resident
and Draft machinery, eager/chunked-prefill/prefix-cache settings, and TP size. Worker snapshots
repeat the execution-mode evidence independently on both ranks.

The per-token records retain
`specrhythm.phase4b1-gate3-per-token-kv-record.v1`, with execution mode
`matched-stock-async-off`. Their `matched_bootstrap_control` block is fail-closed and their
`execution_shape` records active rows, sampled-logit rows, `lm_head_m`, total scheduled tokens,
the planned request query length, decode/prefill row counts, batch kind, TP, and physical GPU
mapping. These are provenance fields, not timings.

The three-way report schema is
`specrhythm.phase4b1-gate3-matched-bootstrap-comparison.v1`. It compares:

- A: immutable ordinary stock Target, async ON;
- B: one new ordinary stock Target, async OFF;
- C: immutable resident Target, async OFF plus resident setup.

Completed output JSON is the semantic-prefix authority. For each planned position `p`, all
three actual prefixes `generated_token_ids[:p]` must equal the immutable reference. The token at
`p` may differ. Prompt K/V at both selected layers and the entire previous control layer must be
bitwise exact. At logical position `prompt_length`, K and V are compared independently on both TP
ranks. Raw competing logits, raw argmax, selected output token, logical ownership, Target
boundary, and execution-shape metadata are retained.

Classification is strict:

- `ASYNC_OFF_MATCHES_RESIDENT`: every endpoint-different bootstrap K/V component in B equals C,
  with no unrelated component change;
- `ASYNC_OFF_MATCHES_STOCK`: every such component in B equals A;
- `ASYNC_OFF_THIRD_STATE`: every request has a valid, neither-endpoint bootstrap state;
- `MIXED_BY_REQUEST`: valid requests do not share one endpoint/third-state result;
- `FAIL-CLOSED`: any prefix, prompt, control-layer, TP, layout, checkpoint, runtime, or
  instrumentation contract fails.

Even the first four classifications leave `gate3_closed=false`, `phase4b2_blocked=true`, exact
token correctness unchanged, and a human decision pending.

## Independent checksum binding

New per-token captures no longer rely only on copying the legacy full-layer digest into a
selected-layer record. After paged K/V is reconstructed once on CPU in logical order, the
observer records:

- `logical_reconstructed_raw_sha256`: SHA256 of the contiguous logical tensor byte sequence
  `[K_or_V, logical_position, ...]` in C order—complete K plane followed by complete V plane;
- `token_digest_sequence_sha256`: a domain-separated SHA256 over ascending logical position,
  then that position's K digest and V digest.

The latter is independently recomputed by validation. The first is computed directly from the
same exact raw K/V payloads used for the individual token hashes. It is intentionally distinct
from the legacy block-by-block digest when a sequence spans physical blocks. Immutable 8773
endpoint records predate these fields and remain accepted without mutation; the new matched
control requires them, and all future captures emit them.

No tolerance, epsilon, tie-equivalence, alternate-token acceptance, or fuzzy K/V comparison is
implemented.
