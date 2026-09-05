# Phase 4B.2 matched-work decode-only bring-up

This is the continuation for the existing corrected-100 Target and recovered Serial runs.
Reuse both `decode-performance.json` files without remeasurement or GPU reruns. The next GPU
work is exactly one Dual-Batch execution. PR #4 stays Draft and unmerged. The coding agent
performs CPU/offline work only; the operator runs Stage B on the A800 server.

## Acceptance policy and schema

Comparison v2 (`specrhythm.phase4b2-decode-performance-comparison.v2`) reads immutable v1
per-mode artifacts. `matched_work_comparability.valid` controls performance acceptance.
`exact_sequence_diagnostic` reports exact token comparisons without tolerance, including all
divergent request IDs, matching/divergent counts and the first ten mismatches. The unchanged
`exact_correctness_triangle` is retained as legacy diagnostic evidence; neither diagnostic is
a performance gate or a correctness PASS. Old v1 `requires_exact_cross_mode_comparison=true`
records the historical policy and does not override comparison v2.

Matched work requires a nonempty unique request set, equal request/completion counts and a
canonical request-ID mapping (artifact row order may differ). It checks per-request prompt
count/SHA, bootstrap ID, maximum output length, measured output count, and each mode's
`bootstrap + measured == final` token accounting. Aggregate counts must equal their per-request
sums and each other. Workload/config SHAs, model paths/revisions, vLLM source commit/version,
patch stack and patch-manifest SHA, GPU topology and placement must match. Draft uses GPU0;
Target uses GPU1,2 with TP2. Cleanup and all per-mode artifacts must be valid, every request
must have completed, and measurement-boundary contracts must be equivalent. Dual must retain
its physical overlap witness and have no per-round global CUDA synchronization.

Execution compatibility currently requires the identical execution commit. Historical v1
`git_commit` is the fallback when `execution_git_commit` is absent; contradictory values fail.
`measurement_code_git_commit` may differ (or be unavailable in the original Target artifact).
Requested workload semantics are bound by the equal workload/config digests, correctness mode
and per-request output limits. No equivalence between different execution commits is assumed.

Different post-bootstrap token IDs do not fail matched work. Finish/stop differences are
recorded under `termination_differences` and do not fail when both modes fill their frozen output
limit. `maximum_new_tokens` includes the one setup bootstrap, so the full measured count is
`maximum_new_tokens - 1`. A reason difference before that fixed length, incomplete/failed
requests, or any measured-count difference fails. `warmup_clean=false` and post-boundary JIT
are reported without rejecting this preliminary bring-up.

A valid pair exposes `performance_valid_for_pair=true` and Target/Serial speedup while keeping
`comparison_complete=false` and `performance_valid=false` (the latter reserves the complete
three-mode result). A valid final comparison exposes `performance_valid=true`. Speedups are
null on any matched-work failure. The claim is **preliminary Phase 4B.2 matched-work decode-only
bring-up**: performance-only comparison, without claims of exact generated-token or output
quality equivalence, a final paper benchmark, or steady-state performance.

## Measurement contract (unchanged)

The performance boundary follows global setup-ready publication, one Target TP barrier and
Target setup CUDA synchronization. Setup and bootstrap are excluded. The first post-bootstrap
output is counted; there is no per-token CUDA synchronization. Serial round-zero proposals and
Dual initial enqueue start after the boundary. Final synchronization covers every Target rank.

Explicit commit events remain authoritative: Target and proposal-free tails use resident
commits; Serial proposals use `state_sync_end_ns`; Dual proposals use `commit_end_ns`.
Log timestamps are used only for JIT provenance. Latency is final measured commit minus the
boundary. TPOT is `(last_commit - first_commit) / (measured_count - 1)`, null for one measured
token. Decode makespan is the latest final commit minus the boundary; throughput is total
measured tokens divided by makespan. This change does not alter any metric calculation.

## Stage A — reuse and approve the existing Target/Serial pair (offline)

Run the blocks in the same Bash shell; stop on any nonzero command. The checkout must be clean.
`SR_PHASE4B_MEASUREMENT_COMMIT` pins the new comparison commit before switching to execution
code. Use the full SHA returned with this change if the branch has advanced since delivery.
The automatic root selection below succeeds only when exactly one old result root exists.
If there are several, explicitly export `SR_PHASE4B2_ROOT` to the intended existing root first.

```bash
set -euo pipefail
cd /root/autodl-tmp/src/SpecRhythm
git fetch origin codex/vllm-serving-v0.1
test -z "$(git status --porcelain)"
git switch --detach origin/codex/vllm-serving-v0.1
export SR_PHASE4B_MEASUREMENT_COMMIT="$(git rev-parse HEAD)"
export SR_PHASE4B_EXECUTION_COMMIT="56bd0a50e3b5f33cf30e32564532b1483ea7e34d"
test "$SR_PHASE4B_MEASUREMENT_COMMIT" != "$SR_PHASE4B_EXECUTION_COMMIT"

conda activate /root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1
python -m pip install -e '.[dev]' --no-deps --no-build-isolation

if test -z "${SR_PHASE4B2_ROOT:-}"; then
  mapfile -t SR_PHASE4B2_OLD_ROOTS < <(
    find "/root/autodl-tmp/SpecRhythm-data/results/phase4/$SR_PHASE4B_EXECUTION_COMMIT" \
      -mindepth 1 -maxdepth 1 -type d -name 'phase4b2-decode-performance-*' -print
  )
  test "${#SR_PHASE4B2_OLD_ROOTS[@]}" -eq 1 || {
    printf 'Set SR_PHASE4B2_ROOT explicitly; found %s roots\n' \
      "${#SR_PHASE4B2_OLD_ROOTS[@]}" >&2
    printf '%s\n' "${SR_PHASE4B2_OLD_ROOTS[@]}" >&2
    exit 1
  }
  export SR_PHASE4B2_ROOT="${SR_PHASE4B2_OLD_ROOTS[0]}"
fi
test -d "$SR_PHASE4B2_ROOT"

export SR_INPUT_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/eba0df493a7fd350ef3c8776e06d30e6196b6749/phase4b1-gate2-corrected5-20260827T040244Z"
export SR_PHASE4B_WORKLOAD="$SR_INPUT_ROOT/workloads/corrected-100.jsonl"
export SR_PHASE4B_REFERENCE="$SR_INPUT_ROOT/Gate-3-corrected-100/reference/stock-target-reference.json"
export SR_PHASE4B_CONFIG="$PWD/configs/phase4b_dual_batch_1d2v.yaml"
export SR_PHASE4B_ENVIRONMENT="$SR_PHASE4B2_ROOT/environment.json"
export SR_PHASE4B_TOPOLOGY="$SR_PHASE4B2_ROOT/topology.json"
export SR_PHASE4B_PATCH_MANIFEST="$SR_PHASE4B2_ROOT/patch-stage/vllm-patch-stack.json"

# Bind unchanged input files and execution provenance before approving the pair.
python - "$SR_PHASE4B2_ROOT" "$SR_PHASE4B_EXECUTION_COMMIT" \
  "$SR_PHASE4B_WORKLOAD" "$SR_PHASE4B_CONFIG" \
  "$SR_PHASE4B_TOPOLOGY" "$SR_PHASE4B_PATCH_MANIFEST" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for mode in ("target", "serial"):
    value = json.loads((root / mode / "decode-performance.json").read_text())
    assert value.get("execution_git_commit", value["git_commit"]) == sys.argv[2]
    for key, name in zip(("workload", "config", "topology", "patch_manifest"), sys.argv[3:]):
        assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest() == value["artifact_sha256"][key]
assert sum(bool(line.strip()) for line in pathlib.Path(sys.argv[3]).read_text().splitlines()) == 100
PY

# Preserve all old exact-only failure evidence in place. New outputs have different names.
test ! -e "$SR_PHASE4B2_ROOT/target-serial-matched-work.json"
test ! -e "$SR_PHASE4B2_ROOT/target-serial-matched-work.md"
test ! -e "$SR_PHASE4B2_ROOT/target-serial-before-dual.sha256"
find "$SR_PHASE4B2_ROOT/target" "$SR_PHASE4B2_ROOT/serial" -type f -print0 | \
  sort -z | xargs -0 sha256sum > "$SR_PHASE4B2_ROOT/target-serial-before-dual.sha256"

source integrations/vllm/phase4b2_run_helpers.sh
phase4b2_compare_target_serial "$SR_PHASE4B2_ROOT"
phase4b2_require_matched_work_pair "$SR_PHASE4B2_ROOT" 100
```

Expected for the reported current pair (verified from its files, never hard-coded):

```text
MATCHED WORK TARGET/SERIAL PASS
exact_sequence_equal = false
performance_comparable = true
```

The diagnostic should retain 9 divergent and 91 matching requests. These counts are observations,
not approval assertions. The exact command approving progression is
`phase4b2_require_matched_work_pair "$SR_PHASE4B2_ROOT" 100`; it verifies the comparison's input
hashes again. Do not rederive either mode or investigate individual divergent requests.

Serial's legacy metadata recovery is already complete: execution `56bd0a50...`, measurement
`abe452d3...`, `valid=true`, `errors=[]`. That recovery remains restricted to Serial with both
raw fields absent and matching raw/runtime/decode-ready evidence. The raw Serial artifacts
remain immutable. The earlier exact-only failed pair is retained as `target-serial-exact.*`.

## Stage B — one Dual-Batch execution at the original execution commit (operator only)

Keep the existing result root, environment, topology, workload and patch manifest. Do not restore,
reapply or regenerate the patch stack. Do not run Target, Serial, hardware preflight generation,
Gate3 diagnostics or Dual-Eager. The existing helper uses Draft GPU0 and Target GPU1,2 TP2.

```bash
# Approve with the new comparator helper BEFORE sourcing the old execution helper.
phase4b2_require_matched_work_pair "$SR_PHASE4B2_ROOT" 100
sha256sum -c "$SR_PHASE4B2_ROOT/target-serial-before-dual.sha256"
test ! -e "$SR_PHASE4B2_ROOT/dual"
test -z "$(git status --porcelain)"
git switch --detach "$SR_PHASE4B_EXECUTION_COMMIT"
export SR_PHASE4B_COMMIT="$SR_PHASE4B_EXECUTION_COMMIT"
test "$(git rev-parse HEAD)" = "$SR_PHASE4B_EXECUTION_COMMIT"
python -m pip install -e '.[dev]' --no-deps --no-build-isolation

export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_BATCH_INVARIANT=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export SR_DRAFT_MODEL="/root/autodl-tmp/models/Qwen3-0.6B"
export SR_TARGET_MODEL="/root/autodl-tmp/models/Qwen3-32B"
export SR_VLLM_SOURCE="/root/autodl-tmp/src/vllm-v0.25.1"
export SR_VLLM_ROOT="$(python - <<'PY'
from importlib import metadata
print(metadata.distribution("vllm").locate_file(""))
PY
)"
export PHASE4B1_OVERLAP_REQUIREMENT=required
unset PHASE4B1_NUMERICAL_PLAN PHASE4B1_NUMERICAL_OUTPUT

test -d "$SR_DRAFT_MODEL"
test -d "$SR_TARGET_MODEL"
test -f "$SR_PHASE4B_REFERENCE"
test -f "$SR_PHASE4B_ENVIRONMENT"
test "$(git -C "$SR_VLLM_SOURCE" rev-parse HEAD)" = \
  "752a3a504485790a2e8491cacbb35c137339ad34"
# Config must still be the artifact-bound configuration after switching checkout.
python - "$SR_PHASE4B_CONFIG" "$SR_PHASE4B2_ROOT/target/decode-performance.json" <<'PY'
import hashlib, json, pathlib, sys
assert hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest() == \
    json.loads(pathlib.Path(sys.argv[2]).read_text())["artifact_sha256"]["config"]
PY

# Read-only process/socket and installed-patch checks; no GPU experiment here.
if pgrep -af 'vllm|specrhythm.*draft-service|EngineCore'; then
  echo 'Existing GPU process: stop and resolve ownership before executing Dual' >&2
  exit 1
fi
if test -n "$(find /tmp -maxdepth 1 -type s -name 'sr4b1-*' -print -quit)"; then
  echo 'Existing Phase-4B socket: stop; do not unlink blindly' >&2
  exit 1
fi
test ! -e "$SR_PHASE4B2_ROOT/dual-patched-state-check.json"
python integrations/vllm/manage_patch.py check \
  --vllm-root "$SR_VLLM_ROOT" --source "$SR_VLLM_SOURCE" \
  --expect-state patched --manifest "$SR_PHASE4B2_ROOT/dual-patched-state-check.json"

source integrations/vllm/phase4b_run_helpers.sh
source integrations/vllm/phase4b1_gate_helpers.sh
source integrations/vllm/phase4b2_run_helpers.sh
# This is the ONLY GPU execution command in the continuation. Never retry in place.
phase4b2_run_mode dual "$SR_PHASE4B2_ROOT/dual" \
  "$SR_PHASE4B_WORKLOAD" 100 "$SR_PHASE4B_REFERENCE" || exit 1
sha256sum -c "$SR_PHASE4B2_ROOT/target-serial-before-dual.sha256"
```

Stop on a nonzero run, failed cleanup, missing overlap, or existing Dual directory. Preserve the
failed run if one occurs; do not run again for a favorable result. This stage uses the unchanged
resident Dual-Batch state machine already present at `56bd0a50...`.

## Stage C — offline Dual measurement and final three-mode comparison

Switch back to the pinned comparison commit and re-source its helper. Do not rerun any mode.

```bash
test -z "$(git status --porcelain)"
git switch --detach "$SR_PHASE4B_MEASUREMENT_COMMIT"
test "$(git rev-parse HEAD)" = "$SR_PHASE4B_MEASUREMENT_COMMIT"
python -m pip install -e '.[dev]' --no-deps --no-build-isolation
source integrations/vllm/phase4b2_run_helpers.sh
phase4b2_require_matched_work_pair "$SR_PHASE4B2_ROOT" 100

test ! -e "$SR_PHASE4B2_ROOT/dual/decode-performance.json"
phase4b2_measure_mode dual-batch "$SR_PHASE4B2_ROOT/dual" "$SR_PHASE4B_WORKLOAD"
python - "$SR_PHASE4B2_ROOT/dual/decode-performance.json" \
  "$SR_PHASE4B_EXECUTION_COMMIT" "$SR_PHASE4B_MEASUREMENT_COMMIT" <<'PY'
import json, pathlib, sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert value["valid"] is True and value["errors"] == []
assert value["performance_result"] is True and value["cleanup_valid"] is True
assert value["execution_git_commit"] == sys.argv[2]
assert value["measurement_code_git_commit"] == sys.argv[3]
assert value["request_count"] == value["metrics"]["completed_requests"] == 100
assert value["mode_semantics"]["natural_draft_target_overlap"] is True
assert value["mode_semantics"]["per_round_global_cuda_synchronize"] is False
assert all(row["token_accounting_valid"] for row in value["requests"])
print("DUAL PERFORMANCE ARTIFACT VALID")
PY

test ! -e "$SR_PHASE4B2_ROOT/decode-performance-comparison.json"
test ! -e "$SR_PHASE4B2_ROOT/decode-performance-comparison.md"
phase4b2_compare_all "$SR_PHASE4B2_ROOT"
python - "$SR_PHASE4B2_ROOT/decode-performance-comparison.json" <<'PY'
import json, pathlib, sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert value["valid"] is True and value["errors"] == []
assert value["comparison_complete"] is True
assert value["performance_valid"] is True
assert value["matched_work_comparability"]["valid"] is True
assert all(row["completed_requests"] == 100 for row in value["metrics"].values())
assert value["speedups"] is not None
print("performance_valid = true")
print("matched_work_comparability.valid = true")
print("exact_sequence_diagnostic.all_equal =", json.dumps(value["exact_sequence_diagnostic"]["all_equal"]))
for key in ("metrics", "warmup", "speedups", "exact_sequence_diagnostic"):
    print(key, json.dumps(value[key], indent=2, sort_keys=True))
print(value["claim_boundary"])
PY
sha256sum -c "$SR_PHASE4B2_ROOT/target-serial-before-dual.sha256"
```

No assertion requires exact generated-token equality. The JSON and Markdown include measured
count, decode makespan, aggregate throughput, latency p50/p90/p99, TPOT mean/p50/p90/p99,
`warmup_clean` and post-measurement JIT count for each mode. They also expose:

- `target_vs_serial`: Serial relative to Target.
- `target_vs_dual_batch`: Dual relative to Target.
- `serial_vs_dual_batch`: Dual relative to Serial.

`makespan_speedup = baseline makespan / compared mode makespan`;
`throughput_ratio = compared mode throughput / baseline throughput`. A value below one is a
slowdown. Both ratios agree for equal measured work. No Serial slowdown is hidden or corrected.

Known metrics from the operator's existing artifacts (not a new GPU measurement by this change):

| Mode | Requests | Measured tokens | Makespan ms | Throughput tok/s | Mean TPOT ms | Warmup clean | JIT count |
|---|---:|---:|---:|---:|---:|---|---:|
| Target | 100 | 1487 | 5813.059543 | 255.8033319632212 | 382.881551485 | true | 0 |
| Serial | 100 | 1487 | 50394.65011 | 29.50710039169275 | 2345.39918652 | false | 1 |
| Dual-Batch | pending | pending | pending | pending | pending | pending | pending |

Expected Serial/Target makespan speedup and throughput ratio are approximately `0.115351x`
(Serial takes approximately `8.669x` the Target makespan). Percentiles come directly from the
existing artifacts; they were not supplied in the handoff and are not invented here.

## Historical evidence and stop conditions

Historical Gate3 roots remain immutable, including
`8773a611a555c9c6efcbce146bb722124d0ee513` and
`efea5c8884e93b39114c320a724dc2c768ec1c8d`. Existing
`historical-bootstrap-before.sha256` / `historical-bootstrap-after.sha256` and corresponding
async checksum evidence are retained. This continuation does not write to either historical root,
replace the existing result root, or rewrite Target/Serial files or their metrics.

Hard stops are invalid artifacts or cleanup; missing/duplicate/incomplete requests; unequal
prompt/bootstrap/output limits or measured counts; inconsistent token accounting; unequal or
missing workload/config/model/patch/execution/topology provenance; invalid measurement boundaries;
and absent Dual physical overlap or per-round global synchronization. These failures suppress
speedups. Exact post-bootstrap token differences and fixed-length finish/stop differences are
diagnostic evidence. Warmup/JIT differences are recorded for later steady-state work.

After successful three-mode bring-up, proceed to Phase 4B.3 fixed-output workloads, context-length,
batch-size and output-length sweeps, and repeated steady-state measurements. Phase 4C adds
Dual-Eager afterward. Do not add more exact-token diagnostics before those phases.
