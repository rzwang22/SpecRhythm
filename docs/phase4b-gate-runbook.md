# Phase 4B.0a/4B.0b 3×A800 gate runbook

> Archived after the real-A800 `d6c7aa8` Gate-B failure. Gate A passed, but this procedure's
> resident Gate B assumed one all-request prefill callback. Do not rerun its Gate B or L5 steps.
> Use `docs/phase4b-resident-l2-rerun.md`, which runs only the corrected L2 gate.

This runbook is the only active Phase-4B server procedure. It uses GPUs `0` for Draft and `1,2`
for Target TP=2. It collects correctness contracts, not performance. Stop on the first nonzero
command. Do not run 100 requests, Phase 4B.1 Dual-Batch correctness, packed trees, Dual-Eager,
KVConnector, SLO evaluation, or performance experiments.

## 1. Exact checkout and inputs

```bash
set -euo pipefail
cd /root/autodl-tmp/src/SpecRhythm
git fetch origin codex/vllm-serving-v0.1
git switch --detach origin/codex/vllm-serving-v0.1
export SR_PHASE4B_COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --short)"

conda activate /root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1
python --version
python -m pip install -e '.[dev]' --no-deps

export SR_DRAFT_MODEL="/root/autodl-tmp/models/Qwen3-0.6B"
export SR_TARGET_MODEL="/root/autodl-tmp/models/Qwen3-32B"
export SR_VLLM_SOURCE="/root/autodl-tmp/src/vllm-v0.25.1"
export SR_PHASE3C_COMMIT="34c7ea9836c2595c8a8aeaeb5680709520edd3d8"
export SR_PHASE3C_RUN="/root/autodl-tmp/SpecRhythm-data/results/phase3c/$SR_PHASE3C_COMMIT/corrected-multiround-100"
export SR_R3_100="$SR_PHASE3C_RUN/workload.jsonl"
export SR_PHASE4B_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/$SR_PHASE4B_COMMIT/phase4b0-gates-$(date -u +%Y%m%dT%H%M%SZ)"
export SR_VLLM_ROOT="$(python - <<'PY'
from importlib import metadata
print(metadata.distribution("vllm").locate_file(""))
PY
)"

test -f "$SR_DRAFT_MODEL/config.json"
test -f "$SR_TARGET_MODEL/config.json"
test -f "$SR_R3_100"
test "$(wc -l < "$SR_R3_100")" -eq 100
test ! -e "$SR_PHASE4B_ROOT"
mkdir -p "$SR_PHASE4B_ROOT/workloads"
source integrations/vllm/phase4b_run_helpers.sh

git -C "$SR_VLLM_SOURCE" checkout --detach \
  752a3a504485790a2e8491cacbb35c137339ad34
test "$(git -C "$SR_VLLM_SOURCE" rev-parse HEAD)" = \
  "752a3a504485790a2e8491cacbb35c137339ad34"
nvidia-smi -L | tee "$SR_PHASE4B_ROOT/nvidia-smi-L.txt"
nvidia-smi topo -m | tee "$SR_PHASE4B_ROOT/nvidia-smi-topo.txt"

specrhythm phase4-dual-contract-dry-run \
  --output "$SR_PHASE4B_ROOT/dual-contract-dry-run.json"
specrhythm phase4-decode-ready-contract-dry-run \
  --output "$SR_PHASE4B_ROOT/decode-ready-contract-dry-run.json"
```

Create only the frozen two- and five-request inputs. The two-request construction is deliberately
short; the five-request order is 3 code, 1 chat, 1 summarization.

```bash
python - "$SR_R3_100" \
  "$SR_PHASE4B_ROOT/workloads/r3-corrected-2.jsonl" \
  "$SR_PHASE4B_ROOT/workloads/r3-corrected-5.jsonl" <<'PY'
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
assert len(rows) == 100
two = []
for task in ("code", "chat"):
    row = dict(next(item for item in rows if item["task_class"] == task))
    row["maximum_new_tokens"] = min(int(row["maximum_new_tokens"]), 8)
    two.append(row)
needed = {"code": 3, "chat": 1, "summarization": 1}
five = []
for row in rows:
    task = row["task_class"]
    if needed.get(task, 0):
        five.append(row)
        needed[task] -= 1
    if not any(needed.values()):
        break
for path, selected in ((sys.argv[2], two), (sys.argv[3], five)):
    with open(path, "x", encoding="utf-8") as handle:
        for row in selected:
            assert row["prompt_length"] == len(row["prompt_token_ids"])
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
PY
find "$SR_PHASE4B_ROOT/workloads" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$SR_PHASE4B_ROOT/workload-sha256.txt"
```

## 2. Stock references, probe, and ordered patch stack

Restore an earlier one-patch installation if present, then create immutable raw-prompt references
before applying the three-patch stack.

```bash
python integrations/vllm/manage_patch.py restore \
  --vllm-root "$SR_VLLM_ROOT" --source "$SR_VLLM_SOURCE"
python integrations/vllm/manage_patch.py check \
  --vllm-root "$SR_VLLM_ROOT" --source "$SR_VLLM_SOURCE" \
  --manifest "$SR_PHASE4B_ROOT/vllm-stock-check.json"

env -u CUDA_VISIBLE_DEVICES VLLM_BATCH_INVARIANT=1 specrhythm phase4-probe \
  --config configs/phase4b_dual_batch_1d2v.yaml \
  --vllm-source "$SR_VLLM_SOURCE" \
  --environment-output "$SR_PHASE4B_ROOT/environment.json" \
  --topology-output "$SR_PHASE4B_ROOT/topology.json" \
  --validation-output "$SR_PHASE4B_ROOT/probe-validation.json"

CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
specrhythm phase4-batch-invariant-preflight \
  --correctness-mode batch-invariant \
  --output "$SR_PHASE4B_ROOT/batch-invariant-preflight.json"

for level in 2 5; do
  ref_dir="$SR_PHASE4B_ROOT/L$level/reference"
  mkdir -p "$ref_dir"
  CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
  specrhythm phase4-stock-reference \
    --config configs/phase4b_dual_batch_1d2v.yaml \
    --correctness-mode batch-invariant \
    --request-count "$level" \
    --workload "$SR_PHASE4B_ROOT/workloads/r3-corrected-$level.jsonl" \
    --environment "$SR_PHASE4B_ROOT/environment.json" \
    --topology "$SR_PHASE4B_ROOT/topology.json" \
    --runtime-manifest "$ref_dir/runtime-manifest.json" \
    --output "$ref_dir/stock-target-reference.json" \
    2>&1 | tee "$ref_dir/stock-reference.log"
  chmod a-w "$ref_dir/stock-target-reference.json"
done

python integrations/vllm/manage_patch.py apply \
  --vllm-root "$SR_VLLM_ROOT" \
  --source "$SR_VLLM_SOURCE" \
  --manifest "$SR_PHASE4B_ROOT/vllm-patch-stack.json"
python - "$SR_PHASE4B_ROOT/vllm-patch-stack.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["patch_stack_applied"] is True
assert [row["order"] for row in value["patch_stack"]] == [1, 2, 3]
assert [row["patch_file"] for row in value["patch_stack"]] == [
    "0001-custom-proposer-request-and-verify-hooks.patch",
    "0002-scheduler-request-admissibility-hook.patch",
    "0003-target-forward-timing-observer.patch",
]
PY
```

## 3. Gate A: explicit admissibility and cleanup

Run only the old two-request mixed construction. Its final sequence is not interpreted as a
Phase-4B.1 result.

```bash
export SR_GATE_A_DIR="$SR_PHASE4B_ROOT/Gate-A/construction"
export SR_PHASE4_DUAL_DRAFT_SOCKET="/tmp/sr4b-${SR_PHASE4B_COMMIT:0:8}-gate-a.sock"
export PHASE4B_LIFECYCLE_ARTIFACT="$SR_GATE_A_DIR/process-lifecycle.json"
export PHASE4B_RUN_GUARD="$SR_GATE_A_DIR/process-lifecycle.active"
mkdir -p "$SR_GATE_A_DIR"
unlink "$SR_PHASE4_DUAL_DRAFT_SOCKET" 2>/dev/null || true

CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
specrhythm phase4-dual-draft-service \
  --config configs/phase4b_dual_batch_1d2v.yaml \
  --socket "$SR_PHASE4_DUAL_DRAFT_SOCKET" \
  --event-log "$SR_GATE_A_DIR/draft-work-events.jsonl" \
  --transport-events "$SR_GATE_A_DIR/transport-events.jsonl" \
  --ready "$SR_GATE_A_DIR/draft-service-ready.json" \
  >"$SR_GATE_A_DIR/draft-service.log" 2>&1 &
draft_pid="$!"
for _ in $(seq 1 600); do
  test -S "$SR_PHASE4_DUAL_DRAFT_SOCKET" && break
  kill -0 "$draft_pid" 2>/dev/null || { cat "$SR_GATE_A_DIR/draft-service.log"; exit 1; }
  sleep 1
done
test -S "$SR_PHASE4_DUAL_DRAFT_SOCKET"

phase4b_run_target_with_cleanup \
  "$draft_pid" "$SR_GATE_A_DIR/target.log" "$SR_GATE_A_DIR/draft-service.log" -- \
  env CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
  specrhythm phase4-dual-batch-run \
    --config configs/phase4b_dual_batch_1d2v.yaml \
    --workload "$SR_PHASE4B_ROOT/workloads/r3-corrected-2.jsonl" \
    --request-count 2 \
    --environment "$SR_PHASE4B_ROOT/environment.json" \
    --topology "$SR_PHASE4B_ROOT/topology.json" \
    --runtime-manifest "$SR_GATE_A_DIR/runtime-manifest.json" \
    --reference "$SR_PHASE4B_ROOT/L2/reference/stock-target-reference.json" \
    --patch-manifest "$SR_PHASE4B_ROOT/vllm-patch-stack.json" \
    --draft-socket "$SR_PHASE4_DUAL_DRAFT_SOCKET" \
    --draft-ready "$SR_GATE_A_DIR/draft-service-ready.json" \
    --scheduler-events "$SR_GATE_A_DIR/scheduler-events.jsonl" \
    --request-state-events "$SR_GATE_A_DIR/request-state-events.jsonl" \
    --proposal-events "$SR_GATE_A_DIR/proposal-events.jsonl" \
    --verification-events "$SR_GATE_A_DIR/verification-events.jsonl" \
    --draft-work-events "$SR_GATE_A_DIR/draft-work-events.jsonl" \
    --transport-events "$SR_GATE_A_DIR/transport-events.jsonl" \
    --target-diagnostics "$SR_GATE_A_DIR/target-diagnostics.jsonl" \
    --plugin-report "$SR_GATE_A_DIR/plugin-report.json" \
    --output-checkpoint "$SR_GATE_A_DIR/output-checkpoint.jsonl" \
    --cycle-events "$SR_GATE_A_DIR/cycle-events.jsonl" \
    --overlap-events "$SR_GATE_A_DIR/overlap-events.jsonl" \
    --microbatch-size 1 --cohort-size 2 \
    --output "$SR_GATE_A_DIR/construction-run.json"
test ! -S "$SR_PHASE4_DUAL_DRAFT_SOCKET"

read -r waiting_id prefill_id < <(python - \
  "$SR_PHASE4B_ROOT/workloads/r3-corrected-2.jsonl" <<'PY'
import json
import sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
print(rows[0]["request_id"], rows[1]["request_id"])
PY
)
specrhythm phase4-gate-a-validate \
  --scheduler-events "$SR_GATE_A_DIR/scheduler-events.jsonl" \
  --lifecycle "$SR_GATE_A_DIR/process-lifecycle.json" \
  --waiting-request-id "$waiting_id" \
  --prefill-request-id "$prefill_id" \
  --output "$SR_GATE_A_DIR/gate-a-validation.json"
```

Run an intentional process-tree failure, verify targeted cleanup, then launch a clean real Target
smoke. This synthetic failure must return nonzero; status `23` is expected.

```bash
export SR_FAIL_DIR="$SR_PHASE4B_ROOT/Gate-A/intentional-failure"
mkdir -p "$SR_FAIL_DIR"
: > "$SR_FAIL_DIR/draft.log"
sleep 600 &
failure_draft_pid="$!"
export PHASE4B_LIFECYCLE_ARTIFACT="$SR_FAIL_DIR/process-lifecycle.json"
export PHASE4B_RUN_GUARD="$SR_FAIL_DIR/process-lifecycle.active"
unset SR_PHASE4_DUAL_DRAFT_SOCKET SR_PHASE4_DRAFT_SOCKET
set +e
phase4b_run_target_with_cleanup \
  "$failure_draft_pid" "$SR_FAIL_DIR/target.log" "$SR_FAIL_DIR/draft.log" -- \
  python -c 'import os,time; p=os.fork(); os._exit(23) if p else time.sleep(600)'
failure_status="$?"
set -e
test "$failure_status" -eq 23
python - "$SR_FAIL_DIR/process-lifecycle.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["target_exit_status"] == 23
assert value["child_reap_result"]["wrapper_exited_with_descendants_alive"] is True
assert value["cleanup_valid"] is True
assert value["run_valid"] is False
assert value["remaining_owned_pids"] == []
assert value["draft_shutdown_result"]["alive_after_cleanup"] is False
PY
test ! -e "$PHASE4B_RUN_GUARD"

export SR_FOLLOWUP_DIR="$SR_PHASE4B_ROOT/Gate-A/clean-followup"
mkdir -p "$SR_FOLLOWUP_DIR"
python -m specrhythm.phase4.process_lifecycle \
  --artifact "$SR_FOLLOWUP_DIR/process-lifecycle.json" \
  --target-log "$SR_FOLLOWUP_DIR/target.log" -- \
  env CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
  specrhythm phase4-stock-smoke \
    --config configs/phase4b_dual_batch_1d2v.yaml \
    --role target --correctness-mode batch-invariant --request-count 2 \
    --workload "$SR_PHASE4B_ROOT/workloads/r3-corrected-2.jsonl" \
    --environment "$SR_PHASE4B_ROOT/environment.json" \
    --topology "$SR_PHASE4B_ROOT/topology.json" \
    --runtime-manifest "$SR_FOLLOWUP_DIR/runtime-manifest.json" \
    --target-diagnostics "$SR_FOLLOWUP_DIR/target-diagnostics.jsonl" \
    --output "$SR_FOLLOWUP_DIR/stock-smoke.json"
python - "$SR_FOLLOWUP_DIR/process-lifecycle.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["run_valid"] is True
assert value["remaining_owned_pids"] == []
PY
```

If any Gate A command fails, stop here and return the Gate A directory. Do not run Gate B.

## 4. Gate B: resident Target and resident Serial

Each consumer gets a fresh persistent Draft service and its own immutable manifest. Run level 2,
validate, then level 5. Never proceed to level 5 if level 2 fails.

```bash
run_resident_consumer () {
  level="$1"
  consumer="$2"
  run_dir="$SR_PHASE4B_ROOT/Gate-B/L$level/$consumer"
  socket_path="/tmp/sr4b-${SR_PHASE4B_COMMIT:0:8}-L${level}-${consumer}.sock"
  mkdir -p "$run_dir"
  unlink "$socket_path" 2>/dev/null || true

  CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
  specrhythm phase4-draft-service \
    --config configs/phase4b_dual_batch_1d2v.yaml \
    --socket "$socket_path" \
    --event-log "$run_dir/draft-service-events.jsonl" \
    --ready "$run_dir/draft-service-ready.json" \
    >"$run_dir/draft-service.log" 2>&1 &
  draft_pid="$!"
  for _ in $(seq 1 600); do
    test -S "$socket_path" && test -f "$run_dir/draft-service-ready.json" && break
    kill -0 "$draft_pid" 2>/dev/null || { cat "$run_dir/draft-service.log"; return 1; }
    sleep 1
  done
  test -S "$socket_path"

  export SR_PHASE4_DRAFT_SOCKET="$socket_path"
  export PHASE4B_LIFECYCLE_ARTIFACT="$run_dir/process-lifecycle.json"
  export PHASE4B_RUN_GUARD="$run_dir/process-lifecycle.active"
  if test "$consumer" = target; then
    phase4b_run_target_with_cleanup \
      "$draft_pid" "$run_dir/target.log" "$run_dir/draft-service.log" -- \
      env CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
      specrhythm phase4-resident-target-run \
        --config configs/phase4b_dual_batch_1d2v.yaml \
        --workload "$SR_PHASE4B_ROOT/workloads/r3-corrected-$level.jsonl" \
        --request-count "$level" \
        --correctness-mode batch-invariant \
        --environment "$SR_PHASE4B_ROOT/environment.json" \
        --topology "$SR_PHASE4B_ROOT/topology.json" \
        --reference "$SR_PHASE4B_ROOT/L$level/reference/stock-target-reference.json" \
        --patch-manifest "$SR_PHASE4B_ROOT/vllm-patch-stack.json" \
        --draft-socket "$socket_path" \
        --draft-ready "$run_dir/draft-service-ready.json" \
        --context "$run_dir/decode-ready-context.json" \
        --decode-ready-manifest "$run_dir/decode-ready-manifest.json" \
        --timing-events "$run_dir/timing-events.jsonl" \
        --setup-control "$run_dir/setup-control.json" \
        --setup-ready "$run_dir/setup-ready.json" \
        --admission-events "$run_dir/admission-events.jsonl" \
        --target-diagnostics "$run_dir/target-diagnostics.jsonl" \
        --plugin-report "$run_dir/plugin-report.json" \
        --first-forward "$run_dir/first-target-forward.json" \
        --output "$run_dir/resident-target.json"
  else
    phase4b_run_target_with_cleanup \
      "$draft_pid" "$run_dir/target.log" "$run_dir/draft-service.log" -- \
      env CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
      specrhythm phase4-resident-serial-run \
        --config configs/phase4b_dual_batch_1d2v.yaml \
        --workload "$SR_PHASE4B_ROOT/workloads/r3-corrected-$level.jsonl" \
        --request-count "$level" \
        --correctness-mode batch-invariant \
        --environment "$SR_PHASE4B_ROOT/environment.json" \
        --topology "$SR_PHASE4B_ROOT/topology.json" \
        --runtime-manifest "$run_dir/runtime-manifest.json" \
        --reference "$SR_PHASE4B_ROOT/L$level/reference/stock-target-reference.json" \
        --patch-manifest "$SR_PHASE4B_ROOT/vllm-patch-stack.json" \
        --draft-socket "$socket_path" \
        --draft-ready "$run_dir/draft-service-ready.json" \
        --round-events "$run_dir/round-events.jsonl" \
        --transport-events "$run_dir/transport-events.jsonl" \
        --plugin-report "$run_dir/plugin-report.json" \
        --context "$run_dir/decode-ready-context.json" \
        --decode-ready-manifest "$run_dir/decode-ready-manifest.json" \
        --timing-events "$run_dir/timing-events.jsonl" \
        --setup-control "$run_dir/setup-control.json" \
        --setup-ready "$run_dir/setup-ready.json" \
        --admission-events "$run_dir/admission-events.jsonl" \
        --initial-proposal-events "$run_dir/initial-proposal-events.jsonl" \
        --target-diagnostics "$run_dir/target-diagnostics.jsonl" \
        --first-forward "$run_dir/first-target-forward.json" \
        --output "$run_dir/resident-serial.json"
  fi
  test ! -S "$socket_path"
  test ! -e "$PHASE4B_RUN_GUARD"
}

validate_resident_level () {
  level="$1"
  level_dir="$SR_PHASE4B_ROOT/Gate-B/L$level"
  specrhythm phase4-resident-validate \
    --target "$level_dir/target/resident-target.json" \
    --serial "$level_dir/serial/resident-serial.json" \
    --target-manifest "$level_dir/target/decode-ready-manifest.json" \
    --serial-manifest "$level_dir/serial/decode-ready-manifest.json" \
    --output "$level_dir/resident-validation.json"
  python - "$level_dir/resident-validation.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["valid"] is True
assert value["target_equals_serial"]["valid"] is True
assert value["dual_evaluated"] is False
assert value["performance_result"] is False
PY
}

run_resident_consumer 2 target
run_resident_consumer 2 serial
validate_resident_level 2

run_resident_consumer 5 target
run_resident_consumer 5 serial
validate_resident_level 5
```

## 5. Integrity and review bundle

```bash
find "$SR_PHASE4B_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$SR_PHASE4B_ROOT/artifact-sha256.txt"
python - "$SR_PHASE4B_ROOT" "$SR_PHASE4B_COMMIT" <<'PY'
import hashlib
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
files = []
for path in sorted(root.rglob("*")):
    if path.is_file() and path.name != "review-manifest.json":
        files.append({
            "path": str(path.relative_to(root)),
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
manifest = {
    "schema_version": "specrhythm.phase4b0-review-bundle.v1",
    "specrhythm_commit": sys.argv[2],
    "gpu_correctness_claim_requires_manual_review": True,
    "gpu_performance_result": False,
    "phase4b1_dual_outcome_claimed": False,
    "files": files,
}
(root / "review-manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
tar -C "$SR_PHASE4B_ROOT" -czf "$SR_PHASE4B_ROOT/review-bundle.tar.gz" \
  review-manifest.json artifact-sha256.txt environment.json topology.json \
  probe-validation.json batch-invariant-preflight.json vllm-patch-stack.json \
  Gate-A Gate-B L2 L5 workloads
sha256sum "$SR_PHASE4B_ROOT/review-bundle.tar.gz" | \
  tee "$SR_PHASE4B_ROOT/review-bundle.sha256"
```

Return `gate-a-validation.json`, all three lifecycle artifacts, both L2 and L5 resident validation
files, both provider manifests per level, first-forward/timing/diagnostic artifacts, patch stack,
review manifest, logs and bundle checksum. Then stop and wait for review.
