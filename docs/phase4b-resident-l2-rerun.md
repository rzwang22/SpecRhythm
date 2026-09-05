# Phase 4B incremental resident-setup L2 rerun

> Archived after the real-A800 `98ec816` Target run reached both incremental requests but failed
> while rebuilding a tuple-typed observation from its JSON-compatible list. Preserve that run.
> The later `5db8657` Serial attempt exposed a round-zero proposal lifecycle bug and also started
> Draft twice, invalidating its process-lifecycle provenance. Do not execute this procedure. Use
> `docs/phase4b-resident-l2-serial-rerun.md` after the lifecycle fix.

This is a historical procedure after the real-A800 `d6c7aa8` Gate-B failure. It must not be used
for a new run; the failed `d6c7aa8` directory remains immutable failure provenance.

## 1. Checkout and identify immutable inputs

The discovery deliberately fails when multiple old roots exist. If so, export
`SR_FAILED_D6_ROOT` to the exact preserved root and continue from its first `test`.

```bash
set -euo pipefail
cd /root/autodl-tmp/src/SpecRhythm
git fetch origin codex/vllm-serving-v0.1
git switch --detach origin/codex/vllm-serving-v0.1
export SR_PHASE4B_FIX_COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --short)"

conda activate /root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1
python --version
python -m pip install -e '.[dev]' --no-deps

export SR_FAILED_D6_PARENT="/root/autodl-tmp/SpecRhythm-data/results/phase4/d6c7aa8fe80096d72b7a027c5b6cd37ff7a55410"
mapfile -t failed_roots < <(
  find "$SR_FAILED_D6_PARENT" -mindepth 1 -maxdepth 1 -type d \
    -name 'phase4b0-gates-*' | sort
)
printf '%s\n' "${failed_roots[@]}"
test "${#failed_roots[@]}" -eq 1
export SR_FAILED_D6_ROOT="${failed_roots[0]}"

export SR_VLLM_SOURCE="/root/autodl-tmp/src/vllm-v0.25.1"
export SR_L2_WORKLOAD="$SR_FAILED_D6_ROOT/workloads/r3-corrected-2.jsonl"
export SR_L2_REFERENCE="$SR_FAILED_D6_ROOT/L2/reference/stock-target-reference.json"
test -f "$SR_L2_WORKLOAD"
test -f "$SR_L2_REFERENCE"
test -f "$SR_FAILED_D6_ROOT/environment.json"
test -f "$SR_FAILED_D6_ROOT/topology.json"
test -f "$SR_FAILED_D6_ROOT/vllm-patch-stack.json"
test "$(git -C "$SR_VLLM_SOURCE" rev-parse HEAD)" = \
  "752a3a504485790a2e8491cacbb35c137339ad34"

export SR_PHASE4B_L2_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/$SR_PHASE4B_FIX_COMMIT/resident-l2-incremental-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$SR_PHASE4B_L2_ROOT"
mkdir -p "$SR_PHASE4B_L2_ROOT"
source integrations/vllm/phase4b_run_helpers.sh

sha256sum "$SR_L2_WORKLOAD" "$SR_L2_REFERENCE" \
  "$SR_FAILED_D6_ROOT/environment.json" \
  "$SR_FAILED_D6_ROOT/topology.json" \
  "$SR_FAILED_D6_ROOT/vllm-patch-stack.json" \
  > "$SR_PHASE4B_L2_ROOT/reused-input-sha256-before.txt"
```

The old L2 stock reference is reusable only because the runner proves an exact match for the
pinned stock-vLLM commit/runner hash, model and tokenizer metadata, workload SHA256, sampling,
GPU/TP placement, dtype, context, batch-invariant mode, DBO state, and remaining Target runtime
settings. Its older `specrhythm_commit` is allowed because the reference was intentionally frozen
before integration patches. No failed resident output or L5 reference is reused.

## 2. L2 resident Target

```bash
export SR_L2_TARGET_DIR="$SR_PHASE4B_L2_ROOT/target"
export SR_PHASE4_DRAFT_SOCKET="/tmp/sr4b-${SR_PHASE4B_FIX_COMMIT:0:8}-l2-target.sock"
export PHASE4B_LIFECYCLE_ARTIFACT="$SR_L2_TARGET_DIR/process-lifecycle.json"
export PHASE4B_RUN_GUARD="$SR_L2_TARGET_DIR/process-lifecycle.active"
mkdir -p "$SR_L2_TARGET_DIR"
unlink "$SR_PHASE4_DRAFT_SOCKET" 2>/dev/null || true

CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
specrhythm phase4-draft-service \
  --config configs/phase4b_dual_batch_1d2v.yaml \
  --socket "$SR_PHASE4_DRAFT_SOCKET" \
  --event-log "$SR_L2_TARGET_DIR/draft-service-events.jsonl" \
  --ready "$SR_L2_TARGET_DIR/draft-service-ready.json" \
  >"$SR_L2_TARGET_DIR/draft-service.log" 2>&1 &
draft_pid="$!"
for _ in $(seq 1 600); do
  test -S "$SR_PHASE4_DRAFT_SOCKET" && \
    test -f "$SR_L2_TARGET_DIR/draft-service-ready.json" && break
  kill -0 "$draft_pid" 2>/dev/null || {
    cat "$SR_L2_TARGET_DIR/draft-service.log"
    exit 1
  }
  sleep 1
done
test -S "$SR_PHASE4_DRAFT_SOCKET"

phase4b_run_target_with_cleanup \
  "$draft_pid" "$SR_L2_TARGET_DIR/target.log" \
  "$SR_L2_TARGET_DIR/draft-service.log" -- \
  env CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
  specrhythm phase4-resident-target-run \
    --config configs/phase4b_dual_batch_1d2v.yaml \
    --workload "$SR_L2_WORKLOAD" --request-count 2 \
    --correctness-mode batch-invariant \
    --environment "$SR_FAILED_D6_ROOT/environment.json" \
    --topology "$SR_FAILED_D6_ROOT/topology.json" \
    --reference "$SR_L2_REFERENCE" \
    --patch-manifest "$SR_FAILED_D6_ROOT/vllm-patch-stack.json" \
    --draft-socket "$SR_PHASE4_DRAFT_SOCKET" \
    --draft-ready "$SR_L2_TARGET_DIR/draft-service-ready.json" \
    --context "$SR_L2_TARGET_DIR/decode-ready-context.json" \
    --decode-ready-manifest "$SR_L2_TARGET_DIR/decode-ready-manifest.json" \
    --timing-events "$SR_L2_TARGET_DIR/timing-events.jsonl" \
    --setup-control "$SR_L2_TARGET_DIR/setup-control.json" \
    --setup-ready "$SR_L2_TARGET_DIR/setup-ready.json" \
    --admission-events "$SR_L2_TARGET_DIR/admission-events.jsonl" \
    --target-diagnostics "$SR_L2_TARGET_DIR/target-diagnostics.jsonl" \
    --plugin-report "$SR_L2_TARGET_DIR/plugin-report.json" \
    --first-forward "$SR_L2_TARGET_DIR/first-target-forward.json" \
    --output "$SR_L2_TARGET_DIR/resident-target.json"

python - "$SR_L2_TARGET_DIR" <<'PY'
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
run = json.loads((root / "resident-target.json").read_text())
ready = json.loads((root / "setup-ready.json").read_text())
lifecycle = json.loads((root / "process-lifecycle.json").read_text())
assert run["valid"] is True
assert run["resident_admission"]["valid"] is True
assert run["measurement_boundary_valid"] is True
assert ready["global_decode_ready"] is True
assert ready["consumer"] == "target-only"
assert ready["initial_proposals"] == []
assert lifecycle["run_valid"] is True
assert lifecycle["remaining_owned_pids"] == []
PY
```

Stop and return the Target directory if this fails. Do not run Serial.

## 3. L2 resident Serial

```bash
export SR_L2_SERIAL_DIR="$SR_PHASE4B_L2_ROOT/serial"
export SR_PHASE4_DRAFT_SOCKET="/tmp/sr4b-${SR_PHASE4B_FIX_COMMIT:0:8}-l2-serial.sock"
export PHASE4B_LIFECYCLE_ARTIFACT="$SR_L2_SERIAL_DIR/process-lifecycle.json"
export PHASE4B_RUN_GUARD="$SR_L2_SERIAL_DIR/process-lifecycle.active"
mkdir -p "$SR_L2_SERIAL_DIR"
unlink "$SR_PHASE4_DRAFT_SOCKET" 2>/dev/null || true

CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
specrhythm phase4-draft-service \
  --config configs/phase4b_dual_batch_1d2v.yaml \
  --socket "$SR_PHASE4_DRAFT_SOCKET" \
  --event-log "$SR_L2_SERIAL_DIR/draft-service-events.jsonl" \
  --ready "$SR_L2_SERIAL_DIR/draft-service-ready.json" \
  >"$SR_L2_SERIAL_DIR/draft-service.log" 2>&1 &
draft_pid="$!"
for _ in $(seq 1 600); do
  test -S "$SR_PHASE4_DRAFT_SOCKET" && \
    test -f "$SR_L2_SERIAL_DIR/draft-service-ready.json" && break
  kill -0 "$draft_pid" 2>/dev/null || {
    cat "$SR_L2_SERIAL_DIR/draft-service.log"
    exit 1
  }
  sleep 1
done
test -S "$SR_PHASE4_DRAFT_SOCKET"

phase4b_run_target_with_cleanup \
  "$draft_pid" "$SR_L2_SERIAL_DIR/target.log" \
  "$SR_L2_SERIAL_DIR/draft-service.log" -- \
  env CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
  specrhythm phase4-resident-serial-run \
    --config configs/phase4b_dual_batch_1d2v.yaml \
    --workload "$SR_L2_WORKLOAD" --request-count 2 \
    --correctness-mode batch-invariant \
    --environment "$SR_FAILED_D6_ROOT/environment.json" \
    --topology "$SR_FAILED_D6_ROOT/topology.json" \
    --runtime-manifest "$SR_L2_SERIAL_DIR/runtime-manifest.json" \
    --reference "$SR_L2_REFERENCE" \
    --patch-manifest "$SR_FAILED_D6_ROOT/vllm-patch-stack.json" \
    --draft-socket "$SR_PHASE4_DRAFT_SOCKET" \
    --draft-ready "$SR_L2_SERIAL_DIR/draft-service-ready.json" \
    --round-events "$SR_L2_SERIAL_DIR/round-events.jsonl" \
    --transport-events "$SR_L2_SERIAL_DIR/transport-events.jsonl" \
    --plugin-report "$SR_L2_SERIAL_DIR/plugin-report.json" \
    --context "$SR_L2_SERIAL_DIR/decode-ready-context.json" \
    --decode-ready-manifest "$SR_L2_SERIAL_DIR/decode-ready-manifest.json" \
    --timing-events "$SR_L2_SERIAL_DIR/timing-events.jsonl" \
    --setup-control "$SR_L2_SERIAL_DIR/setup-control.json" \
    --setup-ready "$SR_L2_SERIAL_DIR/setup-ready.json" \
    --admission-events "$SR_L2_SERIAL_DIR/admission-events.jsonl" \
    --initial-proposal-events "$SR_L2_SERIAL_DIR/initial-proposal-events.jsonl" \
    --target-diagnostics "$SR_L2_SERIAL_DIR/target-diagnostics.jsonl" \
    --first-forward "$SR_L2_SERIAL_DIR/first-target-forward.json" \
    --output "$SR_L2_SERIAL_DIR/resident-serial.json"

python - "$SR_L2_SERIAL_DIR" <<'PY'
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
run = json.loads((root / "resident-serial.json").read_text())
ready = json.loads((root / "setup-ready.json").read_text())
lifecycle = json.loads((root / "process-lifecycle.json").read_text())
assert run["valid"] is True
assert run["resident_admission"]["valid"] is True
assert run["measurement_boundary_valid"] is True
assert ready["global_decode_ready"] is True
assert ready["consumer"] == "serial"
assert len(ready["initial_proposals"]) == 2
assert all(
    row["draft_start_ns"] >= ready["measurement_start_ns"]
    for row in ready["initial_proposals"]
)
assert lifecycle["run_valid"] is True
assert lifecycle["remaining_owned_pids"] == []
PY
```

Stop and return both directories if Serial fails.

## 4. Read-only L2 validation and integrity

```bash
specrhythm phase4-resident-validate \
  --target "$SR_L2_TARGET_DIR/resident-target.json" \
  --serial "$SR_L2_SERIAL_DIR/resident-serial.json" \
  --target-manifest "$SR_L2_TARGET_DIR/decode-ready-manifest.json" \
  --serial-manifest "$SR_L2_SERIAL_DIR/decode-ready-manifest.json" \
  --output "$SR_PHASE4B_L2_ROOT/resident-l2-validation.json"

python - "$SR_PHASE4B_L2_ROOT/resident-l2-validation.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["valid"] is True
assert value["target_equals_serial"]["valid"] is True
assert value["dual_evaluated"] is False
assert value["performance_result"] is False
PY

sha256sum "$SR_L2_WORKLOAD" "$SR_L2_REFERENCE" \
  "$SR_FAILED_D6_ROOT/environment.json" \
  "$SR_FAILED_D6_ROOT/topology.json" \
  "$SR_FAILED_D6_ROOT/vllm-patch-stack.json" \
  > "$SR_PHASE4B_L2_ROOT/reused-input-sha256-after.txt"
diff -u "$SR_PHASE4B_L2_ROOT/reused-input-sha256-before.txt" \
  "$SR_PHASE4B_L2_ROOT/reused-input-sha256-after.txt"
find "$SR_PHASE4B_L2_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$SR_PHASE4B_L2_ROOT/artifact-sha256.txt"
echo "$SR_PHASE4B_L2_ROOT"
```

Return the validation, both setup-ready artifacts, manifests, timing/admission events,
first-forward contracts, lifecycle files, logs, and checksums. Then stop for review.
