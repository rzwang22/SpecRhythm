# S1-P execution and result contracts

This is a separate contract from the Phase4 corrected-5/100 measurement schemas.
Public entry: `python -m specrhythm.serving.s1_cli`; installed alias: `specrhythm-s1`.
The committed server wrapper is `integrations/vllm/phase_s1.sh`.

## Versioned acceptance policy

All primary S1-P artifacts carry these fields (fresh root required):

```json
{
  "policy_schema_version": "specrhythm.s1-policy.v1",
  "acceptance_policy": "s1-performance-v1",
  "cross_run_token_equality_required": false,
  "cross_mode_token_equality_required": false,
  "cross_run_output_length_equality_required": false,
  "cross_run_output_comparison": {"performed": false, "status": "NOT_REQUIRED"},
  "execution_validity_required": true,
  "measurement_validity_required": true,
  "cleanup_validity_required": true
}
```

The manifest digest binds these fields. G0 uses `specrhythm.s1-g0.v2`, and seals use
`specrhythm.s1-seal.v2`. Startup, child, gate, previous-gate, seal, resume and offline
verification reject a missing/different policy or old schema. Old roots are not migrated.

## Frozen execution

`specrhythm.s1-execution.v2`, one `execution-manifest.json` per subset:

| Field | Meaning |
| --- | --- |
| `logical`, `logical_sha256` | Parent semantic hash, subset, ordered IDs, workload SHA256, actual per-request budgets/seeds, natural-EOS sampling intent, arrival replay disabled |
| `parent` | S0 directory, all 35 inventory entries, checksum-file SHA256, semantic identity, data-producing commit and evidence scope |
| `request_ids_sha256` | Canonical JSON hash of ordered stable IDs |
| `workload_file` | Relative `requests.jsonl`, preserving every ServingWorkloadRequest field |
| `assignment` | Immutable alternating A/B assignment in manifest order |
| `execution` | Execution commit, fixed Python, model metadata/weight inventory, config and checked patch hashes, source path, EOS IDs, K/backend/numerical mode, configured capacity and actual `launch_environment` |
| `manifest_sha256` | Canonical JSON hash of the object before this field is added |

Canonical hashes use UTF-8 sorted compact JSON plus newline (`serving.common.digest`).
Byte hashes use file SHA256. Physical DecodeReady manifest hashes are recorded in each
run; they are not shared logical workload hashes. S0's data commit and S1's execution
commit are separate. Weight identity includes metadata/index hashes plus shard
names/sizes/mtime; it does not claim to be a cryptographic full-weight content hash.

`specrhythm.s1-effective-runtime.v2` is written once per real engine before generation:
processed sampling per request, actual Target worker rank/device/batch-invariant evidence,
resolved Target config and allocated KV capacity, actual Draft startup provenance and
capacity, manifest hash and ordinary Target bootstrap source. No reference continuation
is sent to Draft. `launch_environment` records OMP_NUM_THREADS=1,
VLLM_ALLOW_INSECURE_SERIALIZATION=1 and absent USE_TORCH/TF/FLAX. GPU effective configuration is unavailable until server execution.

## Independent result

`specrhythm.s1-result.v2`, `result.json` per owned attempt:

| Field | Meaning |
| --- | --- |
| `valid`, `errors`, `error_details` | Material run qualification and domain-specific failure details |
| `checks.correctness/measurement/lifecycle` | Separate validity/errors; uncompleted checks do not become PASS |
| `draft_backend_checks` | Four independent field checks with `valid`, `field`, `expected`, `actual`, `present` and absolute `artifact` path; failures also appear in `errors`/`error_details` |
| `performance_result` | Valid run with positive actual timed work; false for failures or all-setup-terminal runs |
| `requests` | Stable ID/class, full generated IDs/count, actual bootstrap, untimed/timed counts, natural EOS/cap termination, final prefix hash, terminal-in-setup flag, commit events and barrier-to-completion latency |
| `round_semantics` | Per-request/round sorted prefix/hash, proposal, accepted/rejected, correction/bonus, committed IDs and terminal status |
| `measurement` | Real start/end and source; setup/bootstrap/cleanup exclusion and initial fill/drain inclusion |
| `metrics` | Completed requests; full/timed/bootstrap token counts; EOS/cap/setup-terminal counts; makespan and actual tok/s; latency and task length distributions; scoped ms per timed token |
| `work.Target_physical_forwards` | Actual B/Q/proposal positions/context/progress and host envelope, deduplicated by forward; unavailable CUDA row duration labeled |
| `work.Target_forward_count/Target_query_tokens/Target_batch_B/Target_query_Q` | Physical forward count, total query positions, B/Q distributions, without summing TP ranks |
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

`specrhythm.s1-comparison.v2`, `<gate>/comparison.json`, validates shared frozen
input/config identity, effective sampling, completed request sets, run validity and
within-run metric accounting. It does not call output/round equality helpers.

- `repeatability.performed=false`, `status=NOT_REQUIRED`,
  `exact_tokens_and_termination=null`, `round_comparison_performed=false`.
- `raw_reference_checked=false`; raw Target is absent from default G1.
- `matched_work.status=NOT_ASSESSED`, `exact_timed_output_equal=null`;
  `equal_timed_token_counts` independently describes counts without implying matched
  sequences or computation. Different counts do not block.
- `observations[mode][]` retains each repeat's actual metrics, per-request output lengths,
  finish/stop reasons, work, resources, effective configuration and overlap evidence.
- `metrics[mode]` contains raw values, median and population stdev for makespan, actual
  tok/s, full/timed/bootstrap token counts.
- `performance_ratios[mode]` compares mode-level medians to Target. The throughput
  ratio uses actual tok/s. The makespan ratio is accompanied by Target and mode timed
  token arrays, and is not described as equal-work speedup. `throughput_above_target`
  is an observation, never a validity condition. Invalid runs have no ratios.
- `label="resident decode-only three-mode performance observation"`,
  `end_to_end_improvement_claim=false`, `pure_batching_claim=false`,
  `new_serving_SLO_result=false`.

The backend selector is `SR_PHASE4_DRAFT_BACKEND=vllm-batched`. The report producer
`VllmBatchedDraftBackend.report()` uses `backend_name=vllm-batched-paged-kv-draft`.
Validation requires that report name, `backend_shutdown_complete=true`, integer
`draft_live_requests_final=0`, and `execution_failed=false`. All four checks are
reported independently, including missing fields; a selector string is not accepted
as the runtime report name. The S1-P output-equality policy is unchanged.

## Files, recovery and immutability

Each run lives at `<root>/<gate>/<repeat>-<mode>/attempt-NNN/`. Native outputs,
DecodeReady/setup/initial-proposal/timing, Target diagnostics, scheduler/request-state,
proposal/verification/Draft work and overlap artifacts remain available as applicable.
Raw Target has `raw.json`, runtime/effective config, command/log/lifecycle/exit evidence,
and a distinct explicit-observation result; it is not a default S1-P gate. It has no resident performance result.

`command.json`, `ownership.json`, `draft-owner.json`, `draft-socket-owner.json`,
`process-lifecycle.json`, `exit-code.json`, Target log and Draft service log retain
launch identity, return path and ownership. `seal.json` contains schema/policy fields, `execution_sha256` and a `files` mapping
that binds every top-level regular file except itself, including `result.json`.
Inventory, byte equality and the current result policy/schema are required
before a run can be skipped. Run artifacts are flat; no live socket resides inside them.
Recovery writes a new cleanup sidecar beside the old attempt and uses a new attempt
for reruns. It does not resume live KV or modify sealed failure evidence.

Foreground `gate` mirrors the existing raw child logs with mode/Target/Draft tags.
Children still write directly to files; they are never attached to a display pipe.
Failure summaries print result errors, field checks, exit/lifecycle evidence and
bounded log tails, without modifying the retained attempt. Target/Draft real exit
codes remain in `exit-code.json`; later artifact errors do not replace them. Failed
`stage.json` retains the current mode and directory. `start`/`resume` remain optional
detached entries, with the same display routed into their launcher log.

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
