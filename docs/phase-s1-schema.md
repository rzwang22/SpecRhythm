# S1 execution and result contracts

This is a separate contract from the Phase4 corrected-5/100 measurement schemas.
Public entry: `python -m specrhythm.serving.s1_cli`; installed alias: `specrhythm-s1`.
The committed server wrapper is `integrations/vllm/phase_s1.sh`.

## Frozen execution

`specrhythm.s1-execution.v1`, one `execution-manifest.json` per subset:

| Field | Meaning |
| --- | --- |
| `logical`, `logical_sha256` | Parent semantic hash, subset, ordered IDs, workload SHA256, actual per-request budgets/seeds, natural-EOS sampling intent, arrival replay disabled |
| `parent` | S0 directory, all 35 inventory entries, checksum-file SHA256, semantic identity, data-producing commit and evidence scope |
| `request_ids_sha256` | Canonical JSON hash of ordered stable IDs |
| `workload_file` | Relative `requests.jsonl`, preserving every ServingWorkloadRequest field |
| `assignment` | Immutable alternating A/B assignment in manifest order |
| `execution` | Execution commit, fixed Python, model metadata/weight inventory, config and checked patch hashes, source path, EOS IDs, K/backend/numerical mode, configured capacity |
| `manifest_sha256` | Canonical JSON hash of the object before this field is added |

Canonical hashes use UTF-8 sorted compact JSON plus newline (`serving.common.digest`).
Byte hashes use file SHA256. Physical DecodeReady manifest hashes are recorded in each
run; they are not shared logical workload hashes. S0's data commit and S1's execution
commit are separate. Weight identity includes metadata/index hashes plus shard
names/sizes/mtime; it does not claim to be a cryptographic full-weight content hash.

`specrhythm.s1-effective-runtime.v1` is written once per real engine before generation:
processed sampling per request, actual Target worker rank/device/batch-invariant evidence,
resolved Target config and allocated KV capacity, actual Draft startup provenance and
capacity, manifest hash and ordinary Target bootstrap source. No reference continuation
is sent to Draft. GPU effective configuration is unavailable until server execution.

## Independent result

`specrhythm.s1-result.v1`, `result.json` per owned attempt:

| Field | Meaning |
| --- | --- |
| `valid`, `errors`, `error_details` | Material run qualification and domain-specific failure details |
| `checks.correctness/measurement/lifecycle` | Separate validity/errors; uncompleted checks do not become PASS |
| `performance_result` | Valid run with positive actual timed work; false for failures or all-setup-terminal runs |
| `requests` | Stable ID/class, full generated IDs/count, actual bootstrap, untimed/timed counts, natural EOS/cap termination, final prefix hash, terminal-in-setup flag, commit events and barrier-to-completion latency |
| `round_semantics` | Per-request/round sorted prefix/hash, proposal, accepted/rejected, correction/bonus, committed IDs and terminal status |
| `measurement` | Real start/end and source; setup/bootstrap/cleanup exclusion and initial fill/drain inclusion |
| `metrics` | Makespan, actual timed output/throughput, latency distribution, output length/EOS/cap by task, explicitly scoped aggregate ms per timed token |
| `work.Target_physical_forwards` | Actual B/Q/proposal positions/context/progress and host envelope, deduplicated by forward; unavailable CUDA row duration labeled |
| `work.Target_forward_count/Target_query_tokens` | Sum of physical forwards and input positions, without summing TP ranks |
| `work.Draft` | Native backend report, all existing aggregate counters, by-purpose counts/batch distributions/Q/GPU event time, cleanup and provenance |
| `work.Draft_physical_forwards` | S1 measured proposal/commit forward records, including actual B/Q/CUDA event duration and existing fence completion |
| `work.proposed_tokens/accepted_tokens/rejected_tokens` | Actual round totals, no fixed K denominator |
| `work.mean_accepted_draft_prefix/mean_committed_tokens_per_verification` | Separate acceptance and committed-progress means |
| `overlap`, `cohort` | Existing physical witness union validity/duration, stable assignment, candidate/admissible/scheduled rows, fill/drain scope |
| `resources`, `attribution` | Setup/timed GPU use and reservation; unavailable exclusive timing/clipping fields; no equal-total-GPU claim |
| `artifact_sha256` | Byte identity of retained native input/evidence files |

Native Draft report `s1_forward_records` is optional outside S1. Each S1 record has
`purpose`, `B`, `Q`, `internal_request_ids`, `host_start_ns`, `host_launch_end_ns`,
`gpu_event_ms`, `cuda_completion_observed_ns`, `extra_cuda_sync=false`,
`per_forward_file_write=false`. It uses existing model events/fences, and is emitted
once at shutdown. Setup/warmup records are not counted as measured proposal/commit.

Empty optional distributions have `count=0` and null quantiles/min/max. Missing
mandatory evidence fails with a material error. Null/empty error lists both mean no
errors; nonempty errors and `valid=false` block. A zero-timed-token request has an
explicit setup-terminal record; an entirely zero-work run has no speedup/throughput.

`specrhythm.s1-comparison.v1`, `<gate>/comparison.json`, contains exact three-mode
and repeat checks, raw-reference scope, first divergences, round semantic differences,
raw matched-work counters, raw/median/population-stdev metrics, seal hashes and fixed
run order. Speedup is the ratio of mode-level medians (makespan and throughput reported
separately), never a mean of per-request speedups. It is null if exactness/validity
fails. The label is `production vLLM Batched Draft end-to-end improvement`;
`pure_batching_claim=false`, `new_serving_SLO_result=false`.

## Files, recovery and immutability

Each run lives at `<root>/<gate>/<repeat>-<mode>/attempt-NNN/`. Native outputs,
DecodeReady/setup/initial-proposal/timing, Target diagnostics, scheduler/request-state,
proposal/verification/Draft work and overlap artifacts remain available as applicable.
Raw Target has `raw.json`, runtime/effective config, command/log/lifecycle/exit evidence,
and a distinct reference-only result. It has no resident performance result.

`command.json`, `ownership.json`, `draft-owner.json`, `draft-socket-owner.json`,
`process-lifecycle.json`, `exit-code.json`, Target log and Draft service log retain
launch identity, return path and ownership. `seal.json` binds every top-level regular
file except itself, including `result.json`; inventory and byte equality are required
before a run can be skipped. Run artifacts are flat; no live socket resides inside them.
Recovery writes a new cleanup sidecar beside the old attempt and uses a new attempt
for reruns. It does not resume live KV or modify sealed failure evidence.

Supervisor status (`stage.json`, latest `launcher.json`, latest `exit-code.json`) is
mutable operational state. Historical launcher logs and per-launch exit files are
retained. `compare --gate G1|G2|G3` revalidates sealed runs offline and writes a new
`offline-comparison-*.json`, never replacing the original comparison. Parent S0 input
and all subset bytes are read-only throughout.

`bundle` refuses active owned processes, snapshots all regular root files into a
review inventory and creates `<root>-review.tar.gz` plus a printed archive SHA256.
The bundle contains no model weights, live KV, original source corpus or credentials.
It includes selected request rows/provenance, manifests, G0 configuration and capacity,
G1–G3 retained attempts/comparisons and failure/recovery evidence. Review conclusions
after sealing belong in new files or project documentation, never in old JSON/tar.
