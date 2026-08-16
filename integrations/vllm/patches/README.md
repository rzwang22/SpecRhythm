# Pinned vLLM patch series

Base: vLLM `v0.25.1`, commit
`752a3a504485790a2e8491cacbb35c137339ad34`.

`0001-custom-proposer-request-and-verify-hooks.patch` changes one Python file only:
`vllm/v1/worker/gpu_model_runner.py`. It passes vLLM request IDs to the already supported
`custom_class` proposer and invokes optional before/after Target verification hooks. The hooks are
inactive when speculative decoding is disabled and do not alter Target sampling, rejection, KV,
scheduler, attention, C++ or CUDA semantics.

The patch is required because the public custom proposer signature at this commit does not expose
stable request identity or exact Target-forward boundaries. The out-of-tree proposer uses the
hooks only for stable IPC correlation and strict-serial correctness timestamps.

Phase 4B reuses the same hooks for per-rank verification evidence. It does not add a second patch:
the pinned release's public `scheduler_cls` extension is sufficient for the out-of-tree ready-only
gate. The Dual-Batch adapter remains outside vLLM source and is explicitly enabled only for the
correctness run.

Run `python integrations/vllm/manage_patch.py check ...` before applying it. The manager checks
the exact base/installed file SHA256, applies the patch without fuzzy matching, records patch and
file checksums, and can restore the stock file. Do not apply it to another vLLM commit.
