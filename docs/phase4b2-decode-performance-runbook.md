# Phase 4B.2 decode-only performance acceptance runbook

This is the first functional performance bring-up for resident Target, resident Serial and
resident Dual-Batch on 3×A800. It does not rerun Gate3 diagnostics, does not introduce tolerance,
and does not implement Dual-Eager, packed-tree verification, load/SLO evaluation or a paper
workload. The corrected-100 input is used only to validate the measurement infrastructure.

The historical Gate3 conclusion is deliberately split: exact stock equivalence is not achieved
(96/100), while structural/logical and numerical qualification is complete. Phase 4B.2 compares
the three modes inside the same resident execution regime. Any resident cross-mode token or
termination difference remains a hard failure.

## Measurement contract

The Phase 4B.1 decode-ready manifest timestamp remains historical correctness evidence. In
performance mode, setup-ready is published first. All Target TP ranks then enter one barrier,
synchronize outstanding Target CUDA setup work, and receive one rank-zero
`time.monotonic_ns()` performance boundary. Serial creates its round-zero proposal only after
that boundary; Dual enqueues its initial proposal only after it; Target never proposes. No
per-token CUDA synchronization is introduced. At completion, one collective RPC synchronizes
each Target TP rank and timestamps it.

The setup bootstrap is excluded. The token produced by the first Target forward that consumes
the pending bootstrap is the first measured token. Explicit commit events are the timing
authority: Target sampled commits and proposal-free tails use resident commit events, Serial
proposal commits use `state_sync_end_ns`, and Dual proposal commits use `commit_end_ns`. Log
timestamps are used only to identify possible post-boundary JIT messages.

For request `i`:

```text
decode_latency_i = final_measured_commit_ns - measurement_start_ns
TPOT_i = (last_commit_ns - first_commit_ns) / (measured_tokens_i - 1)
```

TPOT is null when a request has exactly one measured token. Batch makespan is the latest final
commit minus the boundary. Throughput is total measured committed tokens divided by makespan.

## One-shot A800 procedure

Run every block in the same shell. Stop immediately on any nonzero command or failed assertion.
Never delete or reuse the newly created result root. Do not retry a mode until it passes.

### 1. Exact checkout, environment and immutable inputs

```bash
cd /root/autodl-tmp/src/SpecRhythm || exit 1
git fetch origin codex/vllm-serving-v0.1 || exit 1
git switch --detach origin/codex/vllm-serving-v0.1 || exit 1
export SR_PHASE4B_COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --porcelain)" || exit 1

conda activate /root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1 || exit 1
python -m pip install -e '.[dev]' --no-deps --no-build-isolation || exit 1

export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_BATCH_INVARIANT=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export SR_DRAFT_MODEL="/root/autodl-tmp/models/Qwen3-0.6B"
export SR_TARGET_MODEL="/root/autodl-tmp/models/Qwen3-32B"
export SR_VLLM_SOURCE="/root/autodl-tmp/src/vllm-v0.25.1"
export SR_PHASE4B_CONFIG="$PWD/configs/phase4b_dual_batch_1d2v.yaml"
export SR_VLLM_ROOT="$(python - <<'PY'
from importlib import metadata
print(metadata.distribution("vllm").locate_file(""))
PY
)"

export SR_INPUT_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/eba0df493a7fd350ef3c8776e06d30e6196b6749/phase4b1-gate2-corrected5-20260827T040244Z"
export SR_PHASE4B_WORKLOAD="$SR_INPUT_ROOT/workloads/corrected-100.jsonl"
export SR_PHASE4B_REFERENCE="$SR_INPUT_ROOT/Gate-3-corrected-100/reference/stock-target-reference.json"
export SR_HIST_BOOTSTRAP="/root/autodl-tmp/SpecRhythm-data/results/phase4/8773a611a555c9c6efcbce146bb722124d0ee513/phase4b1-gate3-per-token-kv-20260904T092503Z"
export SR_HIST_ASYNC="/root/autodl-tmp/SpecRhythm-data/results/phase4/efea5c8884e93b39114c320a724dc2c768ec1c8d/phase4b1-gate3-matched-bootstrap-20260904T161526Z"

test -d "$SR_VLLM_ROOT/vllm" || exit 1
test -d "$SR_VLLM_SOURCE/.git" || exit 1
test "$(git -C "$SR_VLLM_SOURCE" rev-parse HEAD)" = \
  "752a3a504485790a2e8491cacbb35c137339ad34" || exit 1
test -d "$SR_DRAFT_MODEL" || exit 1
test -d "$SR_TARGET_MODEL" || exit 1
test -f "$SR_PHASE4B_WORKLOAD" || exit 1
test "$(wc -l < "$SR_PHASE4B_WORKLOAD")" -eq 100 || exit 1
test -f "$SR_PHASE4B_REFERENCE" || exit 1
test -d "$SR_HIST_BOOTSTRAP" || exit 1
test -d "$SR_HIST_ASYNC" || exit 1
```

Expected: detached checkout is clean, the conda environment resolves pinned vLLM, both model
directories and all immutable inputs exist, and the workload contains exactly 100 rows.

### 2. Check historical evidence, GPU state and exact patch state

```bash
export SR_PHASE4B2_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/$SR_PHASE4B_COMMIT/phase4b2-decode-performance-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$SR_PHASE4B2_ROOT" || exit 1
mkdir -p "$SR_PHASE4B2_ROOT"

find "$SR_HIST_BOOTSTRAP" -type f -print0 | sort -z | \
  xargs -0 sha256sum > "$SR_PHASE4B2_ROOT/historical-bootstrap-before.sha256" || exit 1
find "$SR_HIST_ASYNC" -type f -print0 | sort -z | \
  xargs -0 sha256sum > "$SR_PHASE4B2_ROOT/historical-async-before.sha256" || exit 1

nvidia-smi -L | tee "$SR_PHASE4B2_ROOT/nvidia-smi-L.txt" || exit 1
nvidia-smi topo -m | tee "$SR_PHASE4B2_ROOT/nvidia-smi-topo.txt" || exit 1
if pgrep -af 'vllm|specrhythm.*draft-service|EngineCore' \
    > "$SR_PHASE4B2_ROOT/preflight-processes.txt"; then
  cat "$SR_PHASE4B2_ROOT/preflight-processes.txt" >&2
  exit 1
fi
if find /tmp -maxdepth 1 -type s -name 'sr4b1-*' -print -quit | grep -q .; then
  echo "stale Phase-4B socket exists" >&2
  exit 1
fi

source integrations/vllm/phase4b_run_helpers.sh || exit 1
source integrations/vllm/phase4b1_gate_helpers.sh || exit 1
source integrations/vllm/phase4b2_run_helpers.sh || exit 1
phase4b1_restore_stock "$SR_PHASE4B2_ROOT/patch-stage" || exit 1

env -u CUDA_VISIBLE_DEVICES VLLM_BATCH_INVARIANT=1 specrhythm phase4-probe \
  --config "$SR_PHASE4B_CONFIG" --vllm-source "$SR_VLLM_SOURCE" \
  --environment-output "$SR_PHASE4B2_ROOT/environment.json" \
  --topology-output "$SR_PHASE4B2_ROOT/topology.json" \
  --validation-output "$SR_PHASE4B2_ROOT/probe-validation.json" || exit 1
export SR_PHASE4B_ENVIRONMENT="$SR_PHASE4B2_ROOT/environment.json"
export SR_PHASE4B_TOPOLOGY="$SR_PHASE4B2_ROOT/topology.json"

CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
  specrhythm phase4-batch-invariant-preflight \
    --correctness-mode batch-invariant \
    --output "$SR_PHASE4B2_ROOT/batch-invariant-preflight.json" || exit 1

phase4b1_apply_patch_stack "$SR_PHASE4B2_ROOT/patch-stage" || exit 1
python integrations/vllm/manage_patch.py check \
  --vllm-root "$SR_VLLM_ROOT" --source "$SR_VLLM_SOURCE" \
  --expect-state patched \
  --manifest "$SR_PHASE4B2_ROOT/patched-state-final-check.json" || exit 1
```

Expected: three A800s are visible, no stale owned process/socket exists, the fresh environment
and topology probe validates, the TP=2 batch-invariant hardware preflight passes, and the
patched-state manifest validates the exact pinned runner, scheduler and active patch hashes.
The fresh topology artifact, rather than a historical Gate input, is bound into every mode
result. This procedure does not freeze or rerun a stock reference.

### 3. Target, then validate Target

```bash
phase4b2_run_mode target "$SR_PHASE4B2_ROOT/target" \
  "$SR_PHASE4B_WORKLOAD" 100 "$SR_PHASE4B_REFERENCE" || exit 1
phase4b2_measure_mode target "$SR_PHASE4B2_ROOT/target" \
  "$SR_PHASE4B_WORKLOAD" || exit 1

python - "$SR_PHASE4B2_ROOT/target/decode-performance.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["schema_version"] == "specrhythm.phase4b2-decode-performance.v1"
assert value["mode"] == "target"
assert value["valid"] is True
assert value["performance_result"] is True
assert value["reports_speedup"] is False
assert value["cleanup_valid"] is True
assert value["measurement"]["setup_excluded"] is True
assert value["measurement"]["bootstrap_excluded_from_measured_token_count"] is True
assert value["mode_semantics"]["draft_measured_work"] is False
assert value["request_count"] == 100
assert all(row["token_accounting_valid"] for row in value["requests"])
PY
```

Stop if Target exits nonzero, cleanup is invalid, the Phase-4B.2 boundary is absent/early, any
commit is missing, or bootstrap accounting fails. A historical stock mismatch is retained as
diagnostic evidence but is not a resident cross-mode failure.

### 4. Serial, validate Serial, then exact Target–Serial gate

```bash
phase4b2_run_mode serial "$SR_PHASE4B2_ROOT/serial" \
  "$SR_PHASE4B_WORKLOAD" 100 "$SR_PHASE4B_REFERENCE" || exit 1
phase4b2_measure_mode serial "$SR_PHASE4B2_ROOT/serial" \
  "$SR_PHASE4B_WORKLOAD" || exit 1

python - "$SR_PHASE4B2_ROOT/serial/decode-performance.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["mode"] == "serial"
assert value["valid"] is True
assert value["cleanup_valid"] is True
assert value["mode_semantics"]["initial_proposal_after_measurement_start"] is True
assert all(row["token_accounting_valid"] for row in value["requests"])
PY

phase4b2_compare_target_serial "$SR_PHASE4B2_ROOT" || exit 1
python - "$SR_PHASE4B2_ROOT/target-serial-exact.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["valid"] is True
assert value["comparison_complete"] is False
assert value["performance_valid"] is False
assert value["speedups"] is None
assert value["exact_correctness_triangle"]["target_equals_serial"]["equal"] is True
PY
```

The intentionally incomplete pair comparison reports no speedup. Stop before Dual on any exact
request/prompt/bootstrap/measured-token/final-token/finish/termination or provenance difference.

### 5. Dual-Batch, validate Dual, then exact triangle and metrics

```bash
phase4b2_run_mode dual "$SR_PHASE4B2_ROOT/dual" \
  "$SR_PHASE4B_WORKLOAD" 100 "$SR_PHASE4B_REFERENCE" || exit 1
phase4b2_measure_mode dual-batch "$SR_PHASE4B2_ROOT/dual" \
  "$SR_PHASE4B_WORKLOAD" || exit 1

python - "$SR_PHASE4B2_ROOT/dual/decode-performance.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["mode"] == "dual-batch"
assert value["valid"] is True
assert value["cleanup_valid"] is True
assert value["mode_semantics"]["natural_draft_target_overlap"] is True
assert value["mode_semantics"]["per_round_global_cuda_synchronize"] is False
assert all(row["token_accounting_valid"] for row in value["requests"])
PY

phase4b2_compare_all "$SR_PHASE4B2_ROOT" || exit 1
python - "$SR_PHASE4B2_ROOT/decode-performance-comparison.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["valid"] is True
assert value["comparison_complete"] is True
assert value["performance_valid"] is True
assert value["exact_correctness_triangle"]["valid"] is True
assert value["exact_correctness_triangle"]["target_equals_serial"]["equal"] is True
assert value["exact_correctness_triangle"]["target_equals_dual-batch"]["equal"] is True
assert value["speedups"] is not None
print(json.dumps(value["metrics"], indent=2, sort_keys=True))
print(json.dumps(value["speedups"], indent=2, sort_keys=True))
PY
```

Only the final artifact may report speedup. It reports makespan speedup with an explicit
denominator and the corresponding throughput ratios. `warmup_clean=false` does not silently
discard the run; inspect each mode's `post_measurement_jit_events` before interpreting numbers.

### 6. Prove historical immutability and freeze the new root

```bash
find "$SR_HIST_BOOTSTRAP" -type f -print0 | sort -z | \
  xargs -0 sha256sum > "$SR_PHASE4B2_ROOT/historical-bootstrap-after.sha256" || exit 1
find "$SR_HIST_ASYNC" -type f -print0 | sort -z | \
  xargs -0 sha256sum > "$SR_PHASE4B2_ROOT/historical-async-after.sha256" || exit 1
diff -u "$SR_PHASE4B2_ROOT/historical-bootstrap-before.sha256" \
  "$SR_PHASE4B2_ROOT/historical-bootstrap-after.sha256" || exit 1
diff -u "$SR_PHASE4B2_ROOT/historical-async-before.sha256" \
  "$SR_PHASE4B2_ROOT/historical-async-after.sha256" || exit 1

find "$SR_PHASE4B2_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | \
  xargs -0 sha256sum > "$SR_PHASE4B2_ROOT/SHA256SUMS" || exit 1
chmod -R a-w "$SR_PHASE4B2_ROOT" || exit 1
echo "$SR_PHASE4B2_ROOT"
```

Stop here and return the root, the three `decode-performance.json` files, final comparison JSON
and Markdown, and all three `warmup_clean` values. Do not run again for a favorable number.

## Hard stop conditions

- Any command returns nonzero or any lifecycle/socket/process cleanup check fails.
- A setup/proposal/commit timestamp precedes the Phase-4B.2 boundary.
- A Target-mode Draft proposal is observed, or Serial round-zero/Dual initial Draft starts early.
- Target, Serial or Dual token accounting differs from `bootstrap + measured == final`.
- Target differs exactly from Serial, or either differs exactly from Dual.
- Workload, config, patch stack, placement or topology provenance differs across modes.
- Physical overlap evidence is absent from the Dual run.

Do not continue to Dual after a Target–Serial failure, and never report speedup from an invalid or
incomplete comparison.

## After successful bring-up

Phase 4B.3 will add fixed-output workloads and batch-size, output-length and context-length sweeps.
Only after those baselines are stable does Phase 4C add Dual-Eager. Serving arrival/load sweeps,
throughput/goodput/SLO, the capacity knee and final paper evaluation follow later.
