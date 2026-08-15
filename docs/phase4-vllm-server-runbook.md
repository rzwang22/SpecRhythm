# Phase 4A.1 1D+2V Serial Disaggregated runbook (3×A800)

This runbook executes a five-request GPU correctness smoke. It does not benchmark performance,
replay arrivals, report SLO/goodput/speedup, or implement Dual-Batch, Eager, packed trees, TP3, or
OPT-66B. Generated artifacts remain outside Git.

The fixed layout is Draft Qwen3-0.6B on physical GPU 0 and Target Qwen3-32B TP=2 on physical GPUs
1 and 2. `VLLM_USE_V2_MODEL_RUNNER=0` is required because custom-class proposers are implemented
only by the pinned V1 runner in vLLM v0.25.1.

## 1. Fetch the Draft PR head without changing PR #3

```bash
set -euo pipefail
cd /root/autodl-tmp/src/SpecRhythm
git fetch origin codex/vllm-serving-v0.1
git switch --detach origin/codex/vllm-serving-v0.1
export SR_PHASE4_COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --short)"
git show --no-patch --oneline "$SR_PHASE4_COMMIT"
```

Compare `SR_PHASE4_COMMIT` with the handoff commit before loading a model.

## 2. Activate the independent pinned environment

Reuse the Phase 4 Python 3.11 environment if Phase 4A.0 already created it:

```bash
conda activate /root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1
cd /root/autodl-tmp/src/SpecRhythm
python -VV
python -m pip install -e '.[dev]' --no-deps
python -m pip check
```

If it does not exist, create it first:

```bash
conda create -y -p /root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1 python=3.11
conda activate /root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1
python -m pip install --upgrade pip setuptools wheel
python -m pip install 'vllm==0.25.1' 'torch==2.11.0' 'transformers>=5.5.3,<5.6'
cd /root/autodl-tmp/src/SpecRhythm
python -m pip install -e '.[dev]' --no-deps
python -m pip check
```

Verify versions and CUDA without setting the insecure-serialization override:

```bash
unset VLLM_ALLOW_INSECURE_SERIALIZATION
export VLLM_USE_V2_MODEL_RUNNER=0
python - <<'PY'
import platform
import torch
import transformers
import vllm

assert platform.python_version().startswith("3.11.")
assert torch.__version__.split("+")[0] == "2.11.0"
assert vllm.__version__ == "0.25.1"
assert torch.cuda.is_available()
print({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "nccl": torch.cuda.nccl.version(),
    "transformers": transformers.__version__,
    "vllm": vllm.__version__,
})
PY
```

## 3. Verify the exact vLLM source and stock installed Python file

```bash
export SR_VLLM_SOURCE="/root/autodl-tmp/src/vllm-v0.25.1"
if test ! -d "$SR_VLLM_SOURCE/.git"; then
  git clone --depth=1 --branch v0.25.1 https://github.com/vllm-project/vllm.git \
    "$SR_VLLM_SOURCE"
fi
git -C "$SR_VLLM_SOURCE" checkout --detach \
  752a3a504485790a2e8491cacbb35c137339ad34
test "$(git -C "$SR_VLLM_SOURCE" rev-parse HEAD)" = \
  "752a3a504485790a2e8491cacbb35c137339ad34"
test "$(git -C "$SR_VLLM_SOURCE" describe --tags --exact-match HEAD)" = "v0.25.1"
test -z "$(git -C "$SR_VLLM_SOURCE" status --porcelain --untracked-files=no)"

export SR_VLLM_ROOT="$(python - <<'PY'
from pathlib import Path
import vllm
print(Path(vllm.__file__).resolve().parents[1])
PY
)"
python integrations/vllm/manage_patch.py check \
  --vllm-root "$SR_VLLM_ROOT" \
  --source "$SR_VLLM_SOURCE"
```

The last command must report `patch_applied=false`. Stop if the stock-file SHA does not match; do
not force or fuzzy-apply the patch.

## 4. Freeze models, workload, reference, and output paths

```bash
export SR_DRAFT_MODEL="/root/autodl-tmp/models/Qwen3-0.6B"
export SR_TARGET_MODEL="/root/autodl-tmp/models/Qwen3-32B"
export SR_PHASE3C_COMMIT="34c7ea9836c2595c8a8aeaeb5680709520edd3d8"
export SR_PHASE3C_RUN="/root/autodl-tmp/SpecRhythm-data/results/phase3c/$SR_PHASE3C_COMMIT/corrected-multiround-100"
export SR_R3_WORKLOAD="$SR_PHASE3C_RUN/workload.jsonl"
export SR_HF_TARGET="$SR_PHASE3C_RUN/target"
export SR_PHASE4_RUN="/root/autodl-tmp/SpecRhythm-data/results/phase4/$SR_PHASE4_COMMIT/serial-1d2v-$(date -u +%Y%m%dT%H%M%SZ)"

test -f "$SR_DRAFT_MODEL/config.json"
test -f "$SR_TARGET_MODEL/config.json"
test -f "$SR_R3_WORKLOAD"
test -d "$SR_HF_TARGET/requests"
test ! -e "$SR_PHASE4_RUN"
mkdir -p "$SR_PHASE4_RUN"
```

Extract the earliest corrected 3/1/1 requests in trace order. No prompt is rewritten or
retokenized:

```bash
python - "$SR_R3_WORKLOAD" "$SR_PHASE4_RUN/r3-real-smoke-5.jsonl" <<'PY'
import json
import sys

needed = {"code": 3, "chat": 1, "summarization": 1}
rows = []
for line in open(sys.argv[1], encoding="utf-8"):
    row = json.loads(line)
    task = row["task_class"]
    if needed.get(task, 0):
        rows.append(row)
        needed[task] -= 1
    if not any(needed.values()):
        break
assert needed == {"code": 0, "chat": 0, "summarization": 0}, needed
assert len(rows) == 5
assert all(row["prompt_length"] == len(row["prompt_token_ids"]) for row in rows)
assert all(row["maximum_new_tokens"] > 0 for row in rows)
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
print([(row["request_id"], row["task_class"], row["prompt_length"]) for row in rows])
PY
```

## 5. Check the earlier Phase 4A.0 artifact, then probe this run

```bash
export SR_PHASE4A0_COMMIT="ba9bded3f16c9b58f13c93a4426c000d523330cd"
export SR_PHASE4A0_RUN="/root/autodl-tmp/SpecRhythm-data/results/phase4/$SR_PHASE4A0_COMMIT/stock-1d2v"
test -f "$SR_PHASE4A0_RUN/validation.json"
python - "$SR_PHASE4A0_RUN/validation.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["valid"], value
print("Phase 4A.0 artifact valid")
PY

nvidia-smi -L | tee "$SR_PHASE4_RUN/nvidia-smi-L.txt"
nvidia-smi topo -m | tee "$SR_PHASE4_RUN/nvidia-smi-topo.txt"
env -u CUDA_VISIBLE_DEVICES specrhythm phase4-probe \
  --config configs/phase4a_target_fair_1d2v.yaml \
  --vllm-source "$SR_VLLM_SOURCE" \
  --environment-output "$SR_PHASE4_RUN/environment.json" \
  --topology-output "$SR_PHASE4_RUN/topology.json" \
  --validation-output "$SR_PHASE4_RUN/probe-validation.json" \
  2>&1 | tee "$SR_PHASE4_RUN/probe.log"
```

## 6. Generate and freeze the unmodified stock Target-only reference

This must happen before applying the patch. The command performs two identical greedy runs and
freezes the output file with exclusive creation.

```bash
CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 \
specrhythm phase4-stock-reference \
  --config configs/phase4a_target_fair_1d2v.yaml \
  --workload "$SR_PHASE4_RUN/r3-real-smoke-5.jsonl" \
  --environment "$SR_PHASE4_RUN/environment.json" \
  --topology "$SR_PHASE4_RUN/topology.json" \
  --runtime-manifest "$SR_PHASE4_RUN/runtime-manifest.json" \
  --legacy-hf-target-dir "$SR_HF_TARGET" \
  --output "$SR_PHASE4_RUN/stock-target-reference.json" \
  2>&1 | tee "$SR_PHASE4_RUN/stock-reference.log"

chmod a-w "$SR_PHASE4_RUN/stock-target-reference.json"
sha256sum "$SR_PHASE4_RUN/stock-target-reference.json" | \
  tee "$SR_PHASE4_RUN/stock-target-reference.sha256"
```

An HF mismatch is advisory. A repeated stock-vLLM mismatch is fatal.

## 7. Apply the pinned one-file Python hook patch

```bash
python integrations/vllm/manage_patch.py apply \
  --vllm-root "$SR_VLLM_ROOT" \
  --source "$SR_VLLM_SOURCE" \
  --manifest "$SR_PHASE4_RUN/vllm-base-and-patch-manifest.json" \
  2>&1 | tee "$SR_PHASE4_RUN/patch-apply.log"

python - "$SR_PHASE4_RUN/vllm-base-and-patch-manifest.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["patch_applied"] is True
assert value["python_only"] is True
assert value["cpp_cuda_modified"] is False
assert value["target_only_behavior_change_when_speculation_disabled"] is False
print(json.dumps(value, indent=2, sort_keys=True))
PY
```

Do not set `VLLM_ALLOW_INSECURE_SERIALIZATION=1` unless vLLM explicitly refuses its local worker
RPC. If it is required, use only this trusted host and regenerate the patch manifest with the
variable set so validation emits the security warning.

## 8. Run patched Target-only regression

```bash
CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 \
specrhythm phase4-target-regression \
  --config configs/phase4a_target_fair_1d2v.yaml \
  --workload "$SR_PHASE4_RUN/r3-real-smoke-5.jsonl" \
  --environment "$SR_PHASE4_RUN/environment.json" \
  --topology "$SR_PHASE4_RUN/topology.json" \
  --runtime-manifest "$SR_PHASE4_RUN/runtime-manifest.json" \
  --reference "$SR_PHASE4_RUN/stock-target-reference.json" \
  --patch-manifest "$SR_PHASE4_RUN/vllm-base-and-patch-manifest.json" \
  --legacy-hf-target-dir "$SR_HF_TARGET" \
  --output "$SR_PHASE4_RUN/patched-target-regression.json" \
  2>&1 | tee "$SR_PHASE4_RUN/patched-target-regression.log"

python - "$SR_PHASE4_RUN/patched-target-regression.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["valid"], value
assert value["comparison"]["all_sequences_equal"] is True
print("patched Target-only == frozen stock Target-only")
PY
```

Stop here if regression fails. Never regenerate the reference to match patched output.

## 9. Run Serial Disaggregated twice

Define a shell function that creates one fresh Draft service and one Target process. Each run uses
fresh sockets and checkpoints; no artifact is resumed or overwritten.

```bash
run_serial_once () {
  run_id="$1"
  run_dir="$SR_PHASE4_RUN/run-$run_id"
  test ! -e "$run_dir"
  mkdir -p "$run_dir"

  CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=0 \
  specrhythm phase4-draft-service \
    --config configs/phase4a_target_fair_1d2v.yaml \
    --socket "$run_dir/draft.sock" \
    --event-log "$run_dir/draft-service-events.jsonl" \
    --ready "$run_dir/draft-service-ready.json" \
    >"$run_dir/draft-service.log" 2>&1 &
  draft_pid="$!"

  ready=0
  for _ in $(seq 1 600); do
    if test -f "$run_dir/draft-service-ready.json" && test -S "$run_dir/draft.sock"; then
      ready=1
      break
    fi
    if ! kill -0 "$draft_pid" 2>/dev/null; then
      cat "$run_dir/draft-service.log"
      return 1
    fi
    sleep 1
  done
  if test "$ready" != 1; then
    kill "$draft_pid" 2>/dev/null || true
    wait "$draft_pid" 2>/dev/null || true
    echo "Draft service readiness timeout" >&2
    return 1
  fi

  if ! CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 \
    specrhythm phase4-serial-run \
      --config configs/phase4a_target_fair_1d2v.yaml \
      --workload "$SR_PHASE4_RUN/r3-real-smoke-5.jsonl" \
      --environment "$SR_PHASE4_RUN/environment.json" \
      --topology "$SR_PHASE4_RUN/topology.json" \
      --runtime-manifest "$SR_PHASE4_RUN/runtime-manifest.json" \
      --reference "$SR_PHASE4_RUN/stock-target-reference.json" \
      --patch-manifest "$SR_PHASE4_RUN/vllm-base-and-patch-manifest.json" \
      --draft-socket "$run_dir/draft.sock" \
      --draft-ready "$run_dir/draft-service-ready.json" \
      --round-events "$SR_PHASE4_RUN/round-events-run-$run_id.jsonl" \
      --transport-events "$SR_PHASE4_RUN/transport-events-run-$run_id.jsonl" \
      --plugin-report "$run_dir/remote-proposer-report.json" \
      --output "$SR_PHASE4_RUN/serial-disaggregated-run-$run_id.json" \
      >"$run_dir/target-serial.log" 2>&1; then
    kill "$draft_pid" 2>/dev/null || true
    wait "$draft_pid" 2>/dev/null || true
    cat "$run_dir/target-serial.log"
    return 1
  fi
  wait "$draft_pid"
  test ! -S "$run_dir/draft.sock"
  python - "$SR_PHASE4_RUN/serial-disaggregated-run-$run_id.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["valid"], value
assert value["exact_sequence_match"] is True
assert value["strict_serial_timeline"]["validated_in_runner"] is True
assert value["accounting"]["valid"] is True
print({
    "requests": value["request_count"],
    "rounds": value["strict_serial_timeline"]["round_events"],
    "accounting": value["accounting"],
})
PY
}

run_serial_once 1
run_serial_once 2
```

## 10. Validate both runs and generate the summary

```bash
specrhythm phase4-serial-validate \
  --config configs/phase4a_target_fair_1d2v.yaml \
  --reference "$SR_PHASE4_RUN/stock-target-reference.json" \
  --patch-manifest "$SR_PHASE4_RUN/vllm-base-and-patch-manifest.json" \
  --target-regression "$SR_PHASE4_RUN/patched-target-regression.json" \
  --run "$SR_PHASE4_RUN/serial-disaggregated-run-1.json" \
  --run "$SR_PHASE4_RUN/serial-disaggregated-run-2.json" \
  --round-events "$SR_PHASE4_RUN/round-events-run-1.jsonl" \
  --round-events "$SR_PHASE4_RUN/round-events-run-2.jsonl" \
  --transport-events "$SR_PHASE4_RUN/transport-events-run-1.jsonl" \
  --transport-events "$SR_PHASE4_RUN/transport-events-run-2.jsonl" \
  --output "$SR_PHASE4_RUN/validation.json" \
  --markdown-output "$SR_PHASE4_RUN/summary.md" \
  2>&1 | tee "$SR_PHASE4_RUN/validation.log"

cat "$SR_PHASE4_RUN/summary.md"
python - "$SR_PHASE4_RUN/validation.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["valid"], value
assert value["checks"]["patched_target_equals_stock"] is True
assert value["checks"]["serial_runs_equal_stock"] == [True, True]
assert value["checks"]["serial_runs_deterministic"] is True
print(json.dumps(value, indent=2, sort_keys=True))
PY
```

## 11. Create the review bundle

```bash
tar -C "$SR_PHASE4_RUN" -czf "$SR_PHASE4_RUN/review-bundle.tar.gz" \
  environment.json \
  topology.json \
  probe-validation.json \
  runtime-manifest.json \
  vllm-base-and-patch-manifest.json \
  stock-target-reference.json \
  stock-target-reference.sha256 \
  patched-target-regression.json \
  serial-disaggregated-run-1.json \
  serial-disaggregated-run-2.json \
  round-events-run-1.jsonl \
  round-events-run-2.jsonl \
  transport-events-run-1.jsonl \
  transport-events-run-2.jsonl \
  validation.json \
  summary.md \
  nvidia-smi-L.txt \
  nvidia-smi-topo.txt \
  probe.log \
  stock-reference.log \
  patch-apply.log \
  patched-target-regression.log \
  validation.log \
  run-1 \
  run-2

sha256sum "$SR_PHASE4_RUN/review-bundle.tar.gz" | \
  tee "$SR_PHASE4_RUN/review-bundle.sha256"
find "$SR_PHASE4_RUN" -maxdepth 2 -type f -print | sort
```

Return `summary.md`, `validation.json`, both Serial JSON files, both round-event logs, the bundle,
and `review-bundle.sha256`. Stop after this correctness run. Do not start a 100-request workload,
arrival replay, latency/capacity sweep, Dual-Batch, Eager, or packed-tree experiment.
