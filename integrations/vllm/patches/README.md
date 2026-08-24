# Pinned vLLM patch series

Base: vLLM `v0.25.1`, commit
`752a3a504485790a2e8491cacbb35c137339ad34`.

Patch order is fixed and both patches are Python-only:

1. `0001-custom-proposer-request-and-verify-hooks.patch` changes
`vllm/v1/worker/gpu_model_runner.py`. It passes vLLM request IDs to the already supported
`custom_class` proposer and invokes optional before/after Target verification hooks. The hooks are
inactive when speculative decoding is disabled and do not alter Target sampling, rejection, KV,
scheduler, attention, C++ or CUDA semantics.
2. `0002-scheduler-request-admissibility-hook.patch` changes
`vllm/v1/core/sched/scheduler.py`. It adds one default-off request predicate before the stock
running-request allocation path. When absent, stock behavior is unchanged. The Phase-4B scheduler
uses it to skip a request waiting for Draft without consuming token/KV budget or blocking later
ready requests. It does not overload vLLM's `next_decode_eligible_step` cadence field.
3. `0003-target-forward-timing-observer.patch` changes the already-`0001`-patched
`gpu_model_runner.py`. It records host monotonic boundaries immediately around the existing model
forward and passes them, plus existing input metadata, to the Target-only diagnostic hook. It is
observational and does not synchronize, alter logits, or change execution semantics.

The patch is required because the public custom proposer signature at this commit does not expose
stable request identity or exact Target-forward boundaries. The out-of-tree proposer uses the
hooks only for stable IPC correlation and strict-serial correctness timestamps.

Phase 4B reuses the worker hooks for per-rank verification evidence, adds the minimal default-off
scheduler hook in `0002`, and adds observational forward timestamps in `0003`. The Dual-Batch
policy and predicate remain outside vLLM source and are explicitly enabled only for the
correctness run.

Run `python integrations/vllm/manage_patch.py check --expect-state stock ...` before applying the
stack and `check --expect-state patched ...` afterward. Omitting `--expect-state` preserves the
legacy stock-check behavior. The two states are mutually exclusive exact runner/scheduler SHA256
pairs; neither accepts partial or opposite-state installations. When `--manifest` is supplied,
the check publishes a new immutable diagnostic containing the expected state, actual hashes,
pinned source commit, active patch hashes and validity. The manager applies in order without fuzzy
matching, records every patch plus original/patched source checksums, and restores in reverse
order. A Phase-4A installation containing only the prior worker patch can still be restored. Do
not apply the stack to another vLLM commit.
