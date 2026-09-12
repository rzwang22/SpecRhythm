# Phase S1-P: resident three-mode performance qualification

S1-P starts at clean commit `cd36d18c63ac706548fabccb4e5cf6e0f15e5897`
on `codex/vllm-serving-v0.1`. PR #4 stays Draft/Open/unmerged; PR #2/#3 are untouched.
[S0 is CLOSED/PASS](phase-s0-closure-review.md). S1 implementation and CPU contracts
are delivered independently of **GPU qualification, which is PENDING**.

S1 asks whether the frozen four-class schema reaches real resident execution,
whether each mode executes and measures its own actual output correctly, and what
the measured Draft/Target work and waiting evidence explain. Independent output
sequences, bootstraps, lengths, termination reasons and round boundaries may differ. Performance gain
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
| Optional raw observation | `run_stock_smoke`, explicit S1 runtime raw-target entry only | Removed from G1; per-run output/accounting checks, no repeated-output comparison or reference passed to resident/Draft |
| Output | Native full token IDs, finish/stop reason, accounting, DecodeReady and per-round artifacts | Natural EOS, correction/bonus/tail, completed sets, actual bootstrap |
| Validation | `s1_results.inspect_run`, `compare_results`, offline CLI `compare` | Independent three-mode contract; old five-mode, corrected-5/100, 16-token/1487-token experiments unchanged |
| Report | `specrhythm.s1-result.v2` / `specrhythm.s1-comparison.v2` | Internal correctness, measurement and lifecycle separate from actual performance ratios; output repeatability NOT_REQUIRED |

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
It is checked against that run’s own final output and committed events only.

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

## S1-P policy and failure semantics

The previous G1 raw Target attempt-002 completed two real generations but was stopped
by `repeated_run_deterministic=false` under the old exact-output policy (operator
report; no new server inspection by the agent). That root remains sealed and unchanged.
This task replaces S1-D; no numerical consistency diagnosis or async/batch-invariant
configuration sweep is performed.

`s1-performance-v1` with policy schema `specrhythm.s1-policy.v1` is mandatory in
execution v2, effective-runtime v2, result v2, comparison v2, G0 v2 and seal v2.
Manifest hashes bind policy and input/config identity. Startup rejects an old root
before writing launcher files; child and Draft-child load the new policy, and
seal/resume/previous-gate/offline comparison reject missing or different policy/schema.
Old passing or failing artifacts cannot silently become S1-P gate evidence. A fresh
G0 and new engine lifecycle are required with the new final execution commit.

No default S1-P path invokes cross-run/cross-mode token, bootstrap, final-prefix,
termination, length, proposal, acceptance or round comparison. `compare_results`
checks common input/config/effective sampling and each result’s internal validity.
Its historical equality fields are null; the explicit comparison status is
`NOT_REQUIRED`, performed=false. Optional historical Python diagnostic helpers remain
outside this call chain. The stock runner’s `compare_repeated_outputs` defaults true
for historical callers; the explicit S1 raw adapter passes false and qualifies actual
completed request sets, per-run token accounting and natural termination instead.
The normal S1-P G1 plan never invokes raw Target.

Within a run, validation still requires completion exactly once, that run's full
committed tokens/bootstrap/final-prefix identity, length/EOS/finish/stop correctness,
and real timed-token counts. Serial/PingPong still use actual proposed/accepted/rejected
IDs, correction/bonus/tail rules, prefix hashes/versions, logical KV accounting,
Target diagnostics and consumed/verified/committed coverage. Existing runtime guards
reject stale proposals, duplicate consumption, unproposed advancement, incorrect row
mapping and invalid scheduler admission. Draft synchronization and TP consensus stay
within each run. No Draft algorithm, scheduler, model, K or patch is changed.

`errors=null` and `errors=[]` mean no errors. Nonempty errors, false raw validity,
nonzero actual/effective exit code, missing/duplicate requests, bad accounting,
invalid lifecycle/cleanup, missing GPU completion, invalid measurement or different
input/config still block. Errors are retained with material reasons; validity is
never forced true to bypass a failure. Raw token/event artifacts remain available
for traceability without introducing output equality as a gate.

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
comparison, not an equal-total-GPU fairness result. The result label is
`resident decode-only three-mode performance observation`. It makes no end-to-end
improvement or pure batching claim. Every run reports its own completed request count,
full/timed/bootstrap tokens, EOS/cap/setup-terminal counts and task length distributions.
Throughput is actual timed tokens divided by that run's decode duration. Compare mode
medians of actual tok/s, with raw samples and population standard deviation. Makespan
ratios carry each mode's actual timed-token counts; they are not equal-work speedups.
`equal_timed_token_counts` is descriptive only; neither equal caps nor equal counts
proves equal tokens or total model work. Differing outputs/lengths/rounds never cause
resampling, ignored EOS, reduced budgets or hidden run exclusion.

## Execution and handoff

The executable [runbook](phase-s1-runbook.md) stops at G3. Every child uses the pinned
GPU Python in its own owned session with a unique PID/start identity and launch token.
Foreground `gate` is the default; an optional detached supervisor survives SSH
closure. Both write stage/log/exit-code artifacts,
and stops later gates on material execution/measurement/cleanup failure. G1 contains
Target, Serial, PingPong and an independent PingPong repeat; G2/G3 retain 20/100 requests
and G3's fixed three rotations. The wrapper and each child explicitly set
`OMP_NUM_THREADS=1`, `VLLM_ALLOW_INSECURE_SERIALIZATION=1` and unset USE_TORCH/TF/FLAX.
The RPC setting enables the existing local callable RPC path; actual values are
recorded in G0 execution, command/Draft-owner and effective-runtime artifacts. Resume verifies seals, cleans interrupted owned
processes first, skips validated completed runs and uses a fresh attempt/engine/KV
for anything incomplete. It never repairs or overwrites a partial measurement.

[Schema](phase-s1-schema.md) describes frozen inputs, result/seal structure and
unavailable fields. CPU fixtures are synthetic contract evidence only. They do not
establish real GPU KV correctness, physical overlap or throughput. Independent
GPU output equality is not required by S1-P.

## Backend report and foreground reporting correction

The follow-up starts at `08cf93a531dc928e02820b14412398b952592a49`. The previous
S1 validator and synthetic fixture confused the selector `vllm-batched` with the
producer report name `vllm-batched-paged-kv-draft`. The validator now references
`VllmBatchedDraftBackend.backend_name`; native CPU fixtures call its actual `report()`
method with only the worker replaced by a CPU fake. Shutdown, live request count and
execution-failure checks remain mandatory and are recorded separately with expected,
actual and artifact path. No runtime backend implementation is modified.

A read-only log mirror shows Target/Draft stdout/stderr live with mode/process tags.
Raw files remain the subprocess output destination, so display cannot replace a
process return code or truncate logs. Failures automatically show field-level errors,
exit/cleanup evidence and both bounded log tails. Real CPU subprocess tests require
both live streams to be visible before permitting Target exit, then check preserved
Target 7 / Draft startup 9 statuses and actual owned process/socket cleanup. Artifact
qualification failures after a failed process preserve its recorded nonzero status.
The decode measurement boundary, GPU synchronization, scheduling and cross-run
NOT_REQUIRED policy remain unchanged. These tests do not constitute a GPU result.

## Serial context startup correction

At `645635d5a54d886ac874a9e1046ae8ad957bef9b`, the operator reports G0 and resident
Target complete, followed by Serial startup failure. Source audit identifies the
missing adapter responsibility: the legacy `phase4-resident-serial-run` CLI builds
the context before `run_serial_disaggregated()`, but S1's `run_consumer("serial")`
passed only a path. The runner exports it as `SR_PHASE4_DECODE_READY_CONTEXT`, then
`LLM(...)` constructs workers whose `RemoteDraftProposer.__init__` reads it. The
runner/proposer do not create the context. Target/PingPong already create theirs.

`prepare_serial_context()` in the S1 adapter now owns creation for the current
Serial attempt. It checks frozen config/patch/workload identity, reuses
`build_decode_ready_context()` with the frozen Git commit and numerical mode,
records the complete S1 execution binding and validates real provenance parsing
before and after exclusive file creation. Only then does it enter the unchanged
Serial runner. No runner, proposer, scheduler, backend, Target or patch code changes.

The CPU regression invokes the real adapter and Serial runner with only the
environment/installed GPU dependencies substituted. Its simulated LLM constructor
asserts that context already exists and parses/binds it using the real
`DecodeReadyProvenance`. The test failed on the original code at that exact entry
with `Serial context missing at LLM construction`, then passed with the fix. It
does not precreate context. Changed frozen inputs fail before LLM, and a second
startup cannot overwrite the first context. A simulated LLM startup exception is
retained as the primary failure through the child CLI and foreground supervisor;
missing reports and later validation errors remain secondary diagnostics.
