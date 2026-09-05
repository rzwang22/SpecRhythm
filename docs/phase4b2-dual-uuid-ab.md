# Phase 4B.2: Dual verification UUID query A/B

Base: `8997ec9d0053d8a6b40b99b8b9694099a7607f1d`. This experiment changes only
Dual's verification UUID lookup. PR #4 stays Draft/Open/unmerged. GPU execution
is reserved for the operator; CPU tests do not establish a performance result.

## Old call path and implementation

At the base commit, `DualBatchRemoteProposer.on_target_verify_end()` in
`src/specrhythm/phase4/vllm_dual.py` calls
`stock_vllm.active_cuda_device_identity(self.torch)` after the existing CUDA end
event, host-end timestamp and TP barrier. The helper obtains the active logical
CUDA index, parses `CUDA_VISIBLE_DEVICES`, maps to the physical device, reads its
properties, then calls `stock_vllm._nvidia_uuid(physical_gpu_id)`. That executes:

```text
nvidia-smi -i <physical_gpu_id> --query-gpu=uuid --format=csv,noheader,nounits
```

There is one subprocess per verification batch per TP rank, independently of the
number of requests in that batch. With 263 verification batches and TP2, this
means 526 verification queries. The experiment measures its own actual batch
count; 263 is a historical observation, not a required schedule.

`dual_uuid.worker_dual_runtime_snapshot()` wraps the existing Dual startup RPC.
It still executes the original `_worker_runtime_snapshot()` exactly once on each
worker, including the real UUID query, CUDA synchronization and model/device
evidence. Its queried identity is copied into a read-only mapping owned by that
worker's proposer. No caller supplies UUIDs or derives them from a rank. The
existing coordinator `validate_worker_ranks()` and batch-invariant checks still
run before generation; per-verification TP identity checks remain in place.

| Setting | Behavior |
| --- | --- |
| Unset, or `SR_PHASE4_DUAL_UUID_QUERY_MODE=live` | Default. Calls the original active-device identity helper on every verification, with its original validation and error behavior. |
| `SR_PHASE4_DUAL_UUID_QUERY_MODE=cached` | Validates the real startup UUID format and active logical/physical binding, then reuses that identity. Every verification still checks current CUDA index, visible-device mapping and device name. Missing/malformed startup UUID or changed binding fails closed. |
| Any other value | Dual startup fails; no fallback. Values are case-sensitive. |

The mode is read in each Target worker at startup and remains fixed for its
lifetime. A worker restart must query and validate again. Initialization cannot
overwrite an existing cache. Changing the shell setting requires a fresh run.
Target and Serial do not read this setting.

The shared identity/snapshot/final-sync helpers, Draft backend, proposal budget,
microbatch size, scheduler, EOS, sampled-row mapping, retired-ready handling,
lifecycle, performance boundary, logging/fsync, diagnostics and five-patch stack
are unchanged. Faster identity access may affect asynchronous readiness timing;
no scheduling decision or rule is modified, and identical batch counts across
A/B are not assumed.

## Runtime evidence

The existing, once-written `resident-dual.json` contains `uuid_query_evidence`.
The nonresident Dual result has the same field. No extra log file, per-forward
write, fsync, CUDA synchronization or TP collective is added during verification.
One final CPU RPC collects counters after the original decode/final-sync/end
timestamp and Draft shutdown. The original final result write emits the evidence.
`decode-performance.json.artifact_sha256.raw_run` binds that raw result, including
these counters. The frequently rewritten `plugin-report.json` is unchanged.

Fields within `uuid_query_evidence`:

- `uuid_query_mode`
- `uuid_initial_validation_count`
- `uuid_verification_subprocess_query_count`
- `uuid_cache_hit_count`
- `uuid_verification_access_count`
- `uuid_query_by_rank`: all the above counters plus global/local/TP rank,
  logical CUDA index, physical GPU ID, UUID, GPU name and visible devices.
- `verification_batch_count`, `valid`, `errors`

For TP2, both modes must have two startup validations. For **each rank**, live
must have `verification_batch_count` subprocess queries and zero hits; cached
must have zero verification subprocess queries and `verification_batch_count`
hits. Counts must be positive to constitute A/B evidence. The report cross-checks
rank coverage and identity against startup snapshots and the unchanged verification
log, and counts distinct `verify_sequence` values rather than per-request rows.

Counters cover successful startup/verification identity accesses during this
worker lifetime. Failed queries abort execution and cannot produce a valid
completed experiment. Existing final-sync UUID queries and topology probes remain
real queries in both modes and are deliberately outside these verification counts.
An empty verification trace is inconclusive. Use fresh directories, not resumed
logs spanning worker lifetimes. `uuid_query_evidence.valid` is a separate experiment
gate; the established raw-run, overlap, lifecycle and performance gates still apply.

## Operator A/B commands (not run by the coding agent)

Use the established pinned server environment, frozen corrected-100 workload,
reference, config, environment/topology and verified five-patch manifest. Keep the
same common input files for A and B. Retain the successful 8997 baseline unchanged.
Set `SR_PHASE4B_COMMIT` to the **delivered experiment commit**, check out/install
that revision in the existing environment, and use a clean working tree. Both new
runs must execute this same commit. The older runbook's Target/Serial reruns are
not needed for this Dual-only A/B; do not splice old execution artifacts into a
new three-mode comparison.

Run one block at a time in interactive Bash. Stop manually on a nonzero printed
return code, retain that directory, and use a new root for any subsequent attempt.
The existing ownership, socket, teardown and failure guards stay enabled.

```bash
set +e
set +u
set +o pipefail
trap - ERR
source integrations/vllm/phase4b_run_helpers.sh &&
  source integrations/vllm/phase4b1_gate_helpers.sh &&
  source integrations/vllm/phase4b2_run_helpers.sh
RC="$?"
echo "load unchanged run helpers rc=$RC"
```

Choose a new `SR_PHASE4B2_UUID_AB_ROOT`. This check creates only the experiment
root; the run helper creates each mode directory and refuses existing ones.

```bash
export SR_PHASE4B2_UUID_AB_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/$SR_PHASE4B_COMMIT/uuid-ab-$(date -u +%Y%m%dT%H%M%SZ)-$$"
python - <<'PY'
import hashlib, json, os, pathlib, subprocess
assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == os.environ["SR_PHASE4B_COMMIT"]
assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
names = ("SR_PHASE4B_WORKLOAD", "SR_PHASE4B_REFERENCE", "SR_PHASE4B_CONFIG",
         "SR_PHASE4B_ENVIRONMENT", "SR_PHASE4B_TOPOLOGY", "SR_PHASE4B_PATCH_MANIFEST")
inputs = {os.environ[name]: hashlib.sha256(pathlib.Path(os.environ[name]).read_bytes()).hexdigest()
          for name in names}
assert os.environ.get("PHASE4B1_OVERLAP_REQUIREMENT", "required") == "required"
root = pathlib.Path(os.environ["SR_PHASE4B2_UUID_AB_ROOT"])
root.mkdir(parents=True, exist_ok=False)
with (root / "fixed-input-sha256.json").open("x") as handle:
    json.dump(inputs, handle, indent=2, sort_keys=True)
print(root)
PY
RC="$?"
echo "freeze A/B inputs and fresh root rc=$RC"
```

Mode A, then its unchanged offline performance derivation:

```bash
SR_PHASE4_DUAL_UUID_QUERY_MODE=live phase4b2_run_mode dual \
  "$SR_PHASE4B2_UUID_AB_ROOT/live" "$SR_PHASE4B_WORKLOAD" 100 "$SR_PHASE4B_REFERENCE"
RC="$?"
echo "Dual live execution and cleanup rc=$RC"
```

```bash
phase4b2_measure_mode dual-batch "$SR_PHASE4B2_UUID_AB_ROOT/live" "$SR_PHASE4B_WORKLOAD"
RC="$?"
echo "Dual live performance derivation rc=$RC"
```

Mode B uses the same command and inputs, changing only the UUID setting and fresh
artifact destination. Each helper call creates fresh Target workers and a fresh
Draft service. Wait for A's helper/cleanup to finish before starting B.

```bash
SR_PHASE4_DUAL_UUID_QUERY_MODE=cached phase4b2_run_mode dual \
  "$SR_PHASE4B2_UUID_AB_ROOT/cached" "$SR_PHASE4B_WORKLOAD" 100 "$SR_PHASE4B_REFERENCE"
RC="$?"
echo "Dual cached execution and cleanup rc=$RC"
```

```bash
phase4b2_measure_mode dual-batch "$SR_PHASE4B2_UUID_AB_ROOT/cached" "$SR_PHASE4B_WORKLOAD"
RC="$?"
echo "Dual cached performance derivation rc=$RC"
```

Check A/B evidence and matched work offline. This checks the same two **Dual**
modes directly; the existing three-mode comparator is unchanged. Exact generated
outputs are required here for the controlled deterministic A/B. If they differ,
retain the evidence and do not claim a controlled speedup from this pair.

```bash
python - <<'PY'
import hashlib, json, os, pathlib
from specrhythm.phase4.dual_correctness import validate_request_state_events
from specrhythm.phase4.dual_uuid import build_dual_uuid_query_report
from specrhythm.phase4.transport import CheckpointJsonl
root = pathlib.Path(os.environ["SR_PHASE4B2_UUID_AB_ROOT"])
for name, sha in json.loads((root / "fixed-input-sha256.json").read_text()).items():
    assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest() == sha, name
values, outputs = {}, {}
for mode in ("live", "cached"):
    directory = root / mode
    raw_path = directory / "resident-dual.json"
    raw = json.loads(raw_path.read_text())
    perf = json.loads((directory / "decode-performance.json").read_text())
    plugin = json.loads((directory / "plugin-report.json").read_text())
    lifecycle = json.loads((directory / "process-lifecycle.json").read_text())
    assert raw["valid"] is True and raw["errors"] == []
    assert raw["overlap_gate"]["valid"] is True
    assert plugin["sampled_row_tp_consensus"] is True
    assert validate_request_state_events(CheckpointJsonl(directory / "request-state-events.jsonl").read()) == []
    assert lifecycle["run_valid"] is lifecycle["cleanup_valid"] is True
    assert lifecycle["natural_teardown_completed"] is True
    assert lifecycle["leaked_after_coordinator_exit"] is False
    assert lifecycle["target_exit_status"] == lifecycle["effective_exit_status"] == 0
    assert not (directory / "process-lifecycle.active").exists()
    assert perf["valid"] is perf["performance_result"] is perf["cleanup_valid"] is True
    assert perf["errors"] == [] and perf["mode"] == "dual-batch"
    assert perf["execution_git_commit"] == os.environ["SR_PHASE4B_COMMIT"]
    assert perf["artifact_sha256"]["raw_run"] == hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert raw["request_count"] == perf["request_count"] == perf["metrics"]["completed_requests"] == 100
    evidence = raw["uuid_query_evidence"]
    log = CheckpointJsonl(directory / "verification-events.jsonl").read()
    assert evidence == build_dual_uuid_query_report(evidence["uuid_query_by_rank"], raw["worker_ranks"], log)
    assert evidence["valid"] is True and evidence["uuid_query_mode"] == mode
    assert evidence["uuid_initial_validation_count"] == 2
    count = evidence["uuid_verification_access_count"]
    assert count > 0
    assert evidence["uuid_verification_subprocess_query_count"] == (count if mode == "live" else 0)
    assert evidence["uuid_cache_hit_count"] == (count if mode == "cached" else 0)
    outputs[mode] = {row["request_id"]: row["generated_token_ids"] for row in raw["outputs"]}
    assert len(outputs[mode]) == 100
    assert len({row["request_id"] for row in perf["requests"]}) == 100
    for row in perf["requests"]:
        assert row["token_accounting_valid"] is True
        assert row["setup_committed_output_tokens"] == 1
        assert [row["bootstrap_token_id"], *row["measured_committed_output_token_ids"]] == outputs[mode][row["request_id"]]
        assert row["measured_committed_output_token_count"] == len(row["measured_committed_output_token_ids"])
    values[mode] = perf
    print(mode, "UUID evidence:", json.dumps(evidence, sort_keys=True))
a, b = values["live"], values["cached"]
for field in ("workload_sha256", "execution_git_commit", "vllm_commit", "vllm_version",
              "patch_hashes", "models", "placement", "gpu_topology", "correctness_mode", "mode_semantics"):
    assert a[field] == b[field], field
for field in ("workload", "config", "topology", "patch_manifest"):
    assert a["artifact_sha256"][field] == b["artifact_sha256"][field], field
for field in a["measurement"]:
    if field not in ("measurement_start_ns", "measurement_end_ns"):
        assert a["measurement"][field] == b["measurement"][field], field
def work(value):
    fields = ("prompt_token_count", "prompt_token_ids_sha256", "bootstrap_token_id",
              "maximum_new_tokens", "measured_committed_output_token_count")
    return {row["request_id"]: [row[field] for field in fields] for row in value["requests"]}
assert work(a) == work(b), "A/B per-request work differs"
assert outputs["live"] == outputs["cached"], "A/B generated outputs differ"
for mode, value in values.items():
    assert value["metrics"]["total_measured_committed_output_tokens"] == sum(
        row["measured_committed_output_token_count"] for row in value["requests"])
assert a["metrics"]["total_measured_committed_output_tokens"] == b["metrics"]["total_measured_committed_output_tokens"]
print("UUID A/B, 100 completions, exact outputs, matched work and correctness gates: PASS")
print("live/cached decode makespan ratio:", a["metrics"]["decode_makespan_ms"] / b["metrics"]["decode_makespan_ms"])
PY
RC="$?"
echo "UUID A/B evidence, correctness and matched-work check rc=$RC"
```

Keep both complete directories, including JIT/warmup evidence. A single pair is an
exploratory timing result; server evidence is required before attributing a gain
to UUID caching. No server or GPU commands above were executed during development.
