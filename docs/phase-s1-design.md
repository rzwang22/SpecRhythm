# Phase S1: long-output resident three-mode baseline

Implementation starts at clean commit `0dd5750384eb63bd4d2ef3b32163d819862e0fbd`
on `codex/vllm-serving-v0.1`. PR #4 stays Draft/Open/unmerged; PR #2/#3 are untouched.
[S0 is CLOSED/PASS](phase-s0-closure-review.md). S1 implementation and CPU contracts
are delivered independently of **GPU qualification, which is PENDING**.

S1 asks whether the frozen four-class schema reaches real resident execution,
whether the three modes produce exactly the same long-budget token sequences, and
what the measured Draft/Target work and waiting evidence explain. Performance gain
and positive overlap are observations, not acceptance requirements. No arrival
runner, dynamic admission/rebalancing, SLO tuning, Shaping, Eager, PD or KVConnector
is introduced. S2/S3 have not started.

## Source audit and integration map

| Stage | S1 entry and handling | Preserved contract |
| --- | --- | --- |
| Data | `serving/s1_workload.py`: explicit `SR_S1_EXECUTION_MANIFEST`; lossless `ResidentServingRequest` around `ServingWorkloadRequest` | Legacy `stock_vllm.load_smoke_requests` still accepts its original classes/counts and mixture |
| Transport | Full frozen row retained; prompt IDs become a tuple where Phase4 needs hashable keys; actual LLM receives token-ID prompts | Stable ID from unique frozen prompt; no suffix parsing, new template or special tokens |
| Setup | Existing `ResidentWarmStartProvider`, incremental setup and DecodeReady artifacts | Logical prompt + actual bootstrap; Target KV excludes pending last token; Draft materializes full prefix |
| Target | `run_resident_target` through `s1_runtime.run_consumer` | Real resident Target-only; no timed Draft proposal; setup Draft resource disclosed |
| Serial | `run_serial_disaggregated`, production `serve_batched_draft` | Same persistent paged-KV vLLM Draft and strict Target→sync→Draft dependencies |
| PingPong | `run_resident_dual_batch` selects existing PingPong classes; production `run_dual_draft_service` | Fixed manifest-order alternating A/B, opposite-cohort dependencies, shrinking/drain, existing scheduler admissibility/no-HOL |
| Raw reference | `run_stock_smoke`, Target role, two ordinary stock runs | G1 exact raw→resident reference chain; dormant installed five-patch stack, no reference continuation injected into Draft |
| Output | Native full token IDs, finish/stop reason, accounting, DecodeReady and per-round artifacts | Natural EOS, correction/bonus/tail, completed sets, actual bootstrap |
| Validation | `s1_results.inspect_run`, `compare_results`, offline CLI `compare` | Independent three-mode contract; old five-mode, corrected-5/100, 16-token/1487-token experiments unchanged |
| Report | `specrhythm.s1-result.v1` / `specrhythm.s1-comparison.v1` | Correctness, measurement, lifecycle and repeatability separate from performance numbers |

The pinned source audit uses vLLM commit
`752a3a504485790a2e8491cacbb35c137339ad34`, especially `SamplingParams`,
`InputProcessor` stop-policy processing, MRV1 `GPUModelRunner.kv_cache_config`,
worker configuration and the existing five patched hooks. G0 uses the repository's
**read-only installed patch check**. Its adapter preserves `operation=check`; it
does not fabricate an apply operation or reapply patches. No vLLM patch changed.

## Input, budgets and setup

Only frozen S0 `build-a/main1000.jsonl` supplies requests. `verify_s0` checks all
35 sealed files, frozen core SHA256s and server tokenizer report bindings.
G0 checks the current server tokenizer JSON, path-independent config and template
against S0, without re-rendering prompts or loading weights. Source resolve,
re-sampling, calibration-set parameter selection and full-1000 resident admission
are absent.

Select each class's first N rows in original JSONL order and merge in that same
order. Freeze the resulting full rows, IDs/hash, quotas and A/B assignment in a
new S1 directory. The manifest separates shared logical identity from per-run
physical setup hashes, timestamps, PIDs and internal IDs.

| Subset | chat/code/summary/reasoning | Request IDs SHA256 | Maximum context including budget/reserve |
| --- | --- | --- | ---: |
| s1-smoke4 | 1/1/1/1 | `8163652fd5045805224feb56f02ba7ffbcf62617f135e16123fdfe393214cbee` | 1685 |
| s1-mixed20 | 6/6/4/4 | `219c4852e2bb24a4a9051098ced14005f09ac4d54a0993b40067dee5f2fd4e35` | 2157 |
| s1-mixed100 | 30/30/20/20 | `4bd0fcbdb87346d348e20f89d5e0509a3bd4e979973f85c988142d12c2b2f31b` | 2213 |

These are offline observations of the accepted input, not GPU capacity results.
The sets are nested. Loader legality is generic (including one/odd request counts);
4/20/100 belong to gate configuration. Arrival/source/SLO provenance remains intact,
with `arrival_replay_enabled=false`.

Chat/summary retain 512-token caps and code/reasoning 1024, including the actual
one-token untimed bootstrap. Temperature 0, top_p 1, per-request seed, natural EOS,
no minimum length or ignored EOS. The engine's processed EOS/stop/sampling parameters
are recorded before generation; disagreement with the frozen policy fails closed.
No matched-bootstrap injection is used; each fresh engine generates its own bootstrap.
G1 checks it against the ordinary raw Target reference.

After all resident rows reach DecodeReady, the existing final TP barrier and CUDA
sync publish the performance start. Serial initial proposals and both PingPong
initial cohort fills start afterward. Bootstrap-terminal requests are recorded
with zero timed tokens and excluded from proposals/verification. A one-token
remaining budget takes a legal Target tail; empty deferred proposal sets are
accepted only with the S1 manifest's terminal/budget proof. Old defaults remain strict.

G0 estimates worst-case KV blocks from prompt + full budget + four speculative
reserve tokens, BF16 KV geometry, model bytes, TP placement and a 4 GiB workspace
reserve per rank. This is an estimate, not a guarantee. Each real engine records
and checks its allocated block count, block size and effective context/sequence/token
capacity before `generate`. S1 uses context 4096, max_num_seqs 128 and query capacity
4096. G3 additionally checks its needs against G2's actual Target and Draft block
capacities. Insufficient capacity is **BLOCKED**, with N and budgets unchanged.

## Exactness and failure semantics

The comparator checks full generated IDs and bootstrap, completion exactly once,
length/EOS/finish/stop, final prefix and actual timed-token counts by stable request
ID. Any token or termination divergence blocks formal speedup. G1 checks both raw
Target repetitions against resident Target. Every mode and repeat shares execution,
workload, sampling, model, patch and config identity; physical setup identity may differ.

Round validation uses actual proposed/accepted/rejected IDs. Nonterminal commits
include accepted prefix plus one correction or bonus; terminal truncation can omit
that extra token. Prefix hashes/versions, logical KV accounting, native Target
diagnostics and PingPong consumed/verified/committed coverage are checked. Existing
runtime guards still reject stale proposals, duplicate consumption, unproposed
advancement, wrong row mapping and invalid scheduler admission. Terminal prefix/state,
Draft synchronization, sampled-row TP consensus and owned cleanup are retained.

`errors=null` and `errors=[]` both mean no errors. A nonempty error list, false raw
validity, failed coordinator, missing requests, bad accounting, invalid lifecycle,
invalid measurement or mismatched execution identity blocks the gate. No HF/numerical
equivalence waiver is inherited from older performance comparison tools. First output
divergence includes request, token position and retained round/prefix evidence.
Repeated runs compare per-request/round semantics, ignoring cross-request write order;
round differences remain visible and are not labeled deterministic merely because
final tokens match. Their scheduler-level cause is reported unavailable unless proven.

## Measurement and interpretation

The start is the established post-setup performance event. The end is the latest
all-Target-rank final synchronization or **necessary measured Draft forward fence
completion**, so outstanding Draft computation cannot fall outside makespan. Cleanup,
offline validation and sealing are outside. No timestamp is guessed.

S1 adds only in-memory per-forward B/Q/purpose/host timestamps to Draft's existing
model hooks. The **existing** CUDA events and fences fill duration/completion fields;
no extra CUDA event, device synchronize or per-forward file write is added. The
records are emitted with the final backend report. Warmup/setup are separate from
proposal/commit. Target and Serial default runtime behavior is unchanged when S1
is absent. The existing Phase4 diagnostics/logger configuration is retained; no new
full-logits probe or per-token fsync is introduced. These timings include the existing
guard/diagnostic overhead and must not be represented as a logging-free experiment.

Metrics include actual length p50/p90/p99 and EOS/cap ratios overall and per class;
actual timed output; makespan/throughput; barrier-to-last-commit completion latency;
and aggregate ms per timed output token. The latter excludes actual bootstrap and
is **not serving TPOT**. Zero measured work is structurally reportable with null
throughput/per-token statistics and `performance_result=false`.

Target forwards deduplicate row/TP evidence by physical forward envelope, with B,
Q, proposal positions, context, committed progress and host duration. Target CUDA
model duration is unavailable in those row diagnostics. Untimed chunked prefill
count is explicitly unavailable. Draft has actual per-forward CUDA event duration
and aggregate purpose counters. Acceptance uses actual proposal lengths, not K as
a forced denominator; mean accepted prefix and mean committed/verification differ.

Physical cross-cohort overlap reuses established synchronized CUDA/host-alignment
witnesses and interval unions; TP intervals are never summed as critical-path time.
Zero overlap and a slower PingPong are valid observations. Host envelopes are not
physical overlap or time saved. Scheduler/IPC/ready-wait/commit/sync exclusive spans
and clipping causes lacking evidence are explicitly unavailable, not guessed from
B alone or subtracted as disjoint timing categories. Native cohort cycles and work
records remain available for further attribution.

Target uses TP2 on GPU1/2; SD adds TP1 Draft GPU0. Resident Target's untimed provider
also reserves/uses Draft GPU0, which is reported. This is a fixed Target-resource
comparison, not an equal-total-GPU fairness result. The retained label is
`production vLLM Batched Draft end-to-end improvement`; no pure batching claim.

## Execution and handoff

The executable [runbook](phase-s1-runbook.md) stops at G3. Every child uses the pinned
GPU Python in its own owned session with a unique PID/start identity and launch token.
The detached supervisor survives SSH closure, writes stage/log/exit-code artifacts,
and stops later gates on failure. Resume verifies seals, cleans interrupted owned
processes first, skips validated completed runs and uses a fresh attempt/engine/KV
for anything incomplete. It never repairs or overwrites a partial measurement.

[Schema](phase-s1-schema.md) describes frozen inputs, result/seal structure and
unavailable fields. CPU fixtures are synthetic contract evidence only. They do not
establish GPU numerical equality, KV correctness, physical overlap or throughput.
