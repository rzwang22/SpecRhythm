# Phase 4A.0 vLLM server runbook (3×A800)

This runbook brings up two independent stock vLLM engines. It does not run speculative
verification, serial-disaggregated execution, SpecRhythm Dual-Batch, packed trees, eager work,
SLO evaluation, or a performance benchmark. Do not add generated artifacts to Git.

`enforce_eager=true` in the vLLM config only disables CUDA Graph execution for an inspectable
stock-engine bring-up. It is not the SpecRhythm rolling-Eager mechanism.

The commands assume the existing server layout:

```text
/root/autodl-tmp/src/SpecRhythm
/root/autodl-tmp/models/Qwen3-0.6B
/root/autodl-tmp/models/Qwen3-32B
/root/autodl-tmp/SpecRhythm-data/results/
```

## 1. Fetch the isolated Phase 4 branch

Use a fresh shell. Do not modify the frozen Phase 3 conda environment.

```bash
set -euo pipefail
cd /root/autodl-tmp/src/SpecRhythm
git fetch origin codex/vllm-serving-v0.1
git switch --detach origin/codex/vllm-serving-v0.1
export SR_PHASE4_COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --short)"
git show --no-patch --oneline "$SR_PHASE4_COMMIT"
```

Compare `SR_PHASE4_COMMIT` with the commit in the Phase 4 handoff before running a model.

## 2. Create the independent Python 3.11 environment

```bash
conda create -y -p /root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1 python=3.11
conda activate /root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1
python -VV
python -m pip install --upgrade pip setuptools wheel
```

Download the exact binary distribution first so its SHA256 is retained. Installation still
resolves the pinned environment dependencies; it is not an editable install of vLLM.

```bash
export SR_PHASE4_WHEELS="/root/autodl-tmp/SpecRhythm-data/wheels/vllm-0.25.1"
mkdir -p "$SR_PHASE4_WHEELS"
python -m pip download --only-binary=:all: --no-deps vllm==0.25.1 -d "$SR_PHASE4_WHEELS"
sha256sum "$SR_PHASE4_WHEELS"/vllm-0.25.1-*.whl | tee "$SR_PHASE4_WHEELS/vllm-wheel-sha256.txt"
python -m pip install \
  "$SR_PHASE4_WHEELS"/vllm-0.25.1-*.whl \
  'torch==2.11.0' \
  'transformers>=5.5.3,<5.6'
python -m pip install -e '.[dev]' --no-deps
python -m pip check
```

## 3. Check out the exact vLLM source for source-to-wheel provenance

The checkout is an audit/provenance source and is not vendored into SpecRhythm.

```bash
export SR_VLLM_SOURCE="/root/autodl-tmp/src/vllm-v0.25.1"
if test ! -d "$SR_VLLM_SOURCE/.git"; then
  git clone --filter=blob:none https://github.com/vllm-project/vllm.git "$SR_VLLM_SOURCE"
fi
git -C "$SR_VLLM_SOURCE" fetch origin tag v0.25.1
git -C "$SR_VLLM_SOURCE" checkout --detach 752a3a504485790a2e8491cacbb35c137339ad34
test "$(git -C "$SR_VLLM_SOURCE" rev-parse HEAD)" = \
  "752a3a504485790a2e8491cacbb35c137339ad34"
test "$(git -C "$SR_VLLM_SOURCE" describe --tags --exact-match HEAD)" = "v0.25.1"
test -z "$(git -C "$SR_VLLM_SOURCE" status --porcelain --untracked-files=no)"
```

Verify the runtime packages without importing any Phase 3 GPU dependencies:

```bash
python - <<'PY'
import platform
import torch
import transformers
import vllm

assert platform.python_version().startswith("3.11.")
assert torch.__version__.split("+")[0] == "2.11.0"
assert vllm.__version__ == "0.25.1"
print({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "nccl": torch.cuda.nccl.version() if torch.cuda.is_available() else None,
    "transformers": transformers.__version__,
    "vllm": vllm.__version__,
})
PY
```

## 4. Freeze paths and inspect the corrected five-request source

The smoke selects the earliest 3 code, 1 chat and 1 summarization rows from the existing corrected
R3-real workload while retaining their trace order. It does not render or tokenize a new prompt.

```bash
export SR_DRAFT_MODEL="/root/autodl-tmp/models/Qwen3-0.6B"
export SR_TARGET_MODEL="/root/autodl-tmp/models/Qwen3-32B"
export SR_PHASE3C_COMMIT="34c7ea9836c2595c8a8aeaeb5680709520edd3d8"
export SR_PHASE3C_RUN="/root/autodl-tmp/SpecRhythm-data/results/phase3c/$SR_PHASE3C_COMMIT/corrected-multiround-100"
export SR_R3_WORKLOAD="$SR_PHASE3C_RUN/workload.jsonl"
export SR_HF_TARGET="$SR_PHASE3C_RUN/target"
export SR_PHASE4_RUN="/root/autodl-tmp/SpecRhythm-data/results/phase4/$SR_PHASE4_COMMIT/stock-1d2v"
mkdir -p "$SR_PHASE4_RUN"

test -f "$SR_DRAFT_MODEL/config.json"
test -f "$SR_TARGET_MODEL/config.json"
test -f "$SR_R3_WORKLOAD"
test -d "$SR_HF_TARGET/requests"
test "$(wc -l < "$SR_R3_WORKLOAD")" -ge 5
python - "$SR_R3_WORKLOAD" "$SR_PHASE4_RUN/r3-real-smoke-5.jsonl" <<'PY'
import json
import sys

required = {"code": 3, "chat": 1, "summarization": 1}
rows = []
for line in open(sys.argv[1], encoding="utf-8"):
    row = json.loads(line)
    task = row["task_class"]
    if required.get(task, 0) > 0:
        rows.append(row)
        required[task] -= 1
    if not any(required.values()):
        break
assert not any(required.values()), required
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
assert len(rows) == 5
assert all(row["prompt_length"] == len(row["prompt_token_ids"]) for row in rows)
assert all(row["maximum_new_tokens"] > 0 for row in rows)
assert [sum(row["task_class"] == task for row in rows) for task in required] == [3, 1, 1]
chat = next(row for row in rows if row["task_class"] == "chat")
assert chat["prompt_text"].startswith("<|im_start|>user")
assert "<|im_start|>assistant" in chat["prompt_text"]
print([(row["request_id"], row["task_class"], row["prompt_length"]) for row in rows])
PY
```

The extracted five-row file is a run artifact under `SpecRhythm-data`; do not commit it.

## 5. Probe environment and physical topology

Run the probe with all three GPUs visible so physical IDs 0, 1 and 2 can be validated.

```bash
cd /root/autodl-tmp/src/SpecRhythm
nvidia-smi -L | tee "$SR_PHASE4_RUN/nvidia-smi-L.txt"
nvidia-smi topo -m | tee "$SR_PHASE4_RUN/nvidia-smi-topo.txt"

env -u CUDA_VISIBLE_DEVICES specrhythm phase4-probe \
  --config configs/phase4a_target_fair_1d2v.yaml \
  --vllm-source "$SR_VLLM_SOURCE" \
  --environment-output "$SR_PHASE4_RUN/environment.json" \
  --topology-output "$SR_PHASE4_RUN/topology.json" \
  --validation-output "$SR_PHASE4_RUN/probe-validation.json" \
  2>&1 | tee "$SR_PHASE4_RUN/phase4a.log"

python - "$SR_PHASE4_RUN/probe-validation.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["valid"], value
print(json.dumps(value, indent=2, sort_keys=True))
PY
```

If the probe exits nonzero, stop. It intentionally has no CPU/synthetic fallback.

## 6. Draft TP=1 stock-engine smoke on physical GPU 0

```bash
CUDA_VISIBLE_DEVICES=0 specrhythm phase4-stock-smoke \
  --config configs/phase4a_target_fair_1d2v.yaml \
  --role draft \
  --workload "$SR_PHASE4_RUN/r3-real-smoke-5.jsonl" \
  --environment "$SR_PHASE4_RUN/environment.json" \
  --topology "$SR_PHASE4_RUN/topology.json" \
  --runtime-manifest "$SR_PHASE4_RUN/runtime-manifest.json" \
  --output "$SR_PHASE4_RUN/draft-smoke.json" \
  2>&1 | tee -a "$SR_PHASE4_RUN/phase4a.log"
```

This is ordinary greedy generation by the Draft model, not candidate-tree generation.

## 7. Target TP=2 stock-engine smoke on physical GPUs 1 and 2

```bash
CUDA_VISIBLE_DEVICES=1,2 specrhythm phase4-stock-smoke \
  --config configs/phase4a_target_fair_1d2v.yaml \
  --role target \
  --workload "$SR_PHASE4_RUN/r3-real-smoke-5.jsonl" \
  --environment "$SR_PHASE4_RUN/environment.json" \
  --topology "$SR_PHASE4_RUN/topology.json" \
  --runtime-manifest "$SR_PHASE4_RUN/runtime-manifest.json" \
  --frozen-hf-target-dir "$SR_HF_TARGET" \
  --output "$SR_PHASE4_RUN/target-tp2-smoke.json" \
  2>&1 | tee -a "$SR_PHASE4_RUN/phase4a.log"
```

If the vLLM and HF token trajectories differ, do not regenerate the HF target. The smoke JSON
will retain the first divergence, vLLM top-k row, token IDs and configuration for review.

## 8. Validate, summarize and inspect

```bash
specrhythm phase4-validate \
  --config configs/phase4a_target_fair_1d2v.yaml \
  --environment "$SR_PHASE4_RUN/environment.json" \
  --topology "$SR_PHASE4_RUN/topology.json" \
  --runtime-manifest "$SR_PHASE4_RUN/runtime-manifest.json" \
  --draft-smoke "$SR_PHASE4_RUN/draft-smoke.json" \
  --target-smoke "$SR_PHASE4_RUN/target-tp2-smoke.json" \
  --output "$SR_PHASE4_RUN/validation.json" \
  --markdown-output "$SR_PHASE4_RUN/summary.md" \
  2>&1 | tee -a "$SR_PHASE4_RUN/phase4a.log"

python - "$SR_PHASE4_RUN" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
validation = json.load(open(root / "validation.json", encoding="utf-8"))
runtime = json.load(open(root / "runtime-manifest.json", encoding="utf-8"))
assert validation["valid"], validation
assert set(runtime["roles"]) == {"draft", "target"}
assert runtime["roles"]["draft"]["engine"]["physical_gpu_ids"] == [0]
assert runtime["roles"]["target"]["engine"]["physical_gpu_ids"] == [1, 2]
assert len(runtime["roles"]["target"]["worker_ranks"]) == 2
for role in ("draft", "target"):
    for rank in runtime["roles"][role]["worker_ranks"]:
        assert rank["parameter_count"] > 0
        assert rank["allocated_memory_bytes"] > 0
        assert rank["gpu_uuid"]
        assert rank["attention_backends"]
print((root / "summary.md").read_text(encoding="utf-8"))
PY

find "$SR_PHASE4_RUN" -maxdepth 2 -type f -print | sort
```

## 9. Package the small review bundle

Do not include model weights, raw logits, full Phase 3 traces or any repository data.

```bash
tar -C "$SR_PHASE4_RUN" -czf "$SR_PHASE4_RUN/phase4a-review-bundle.tar.gz" \
  environment.json \
  topology.json \
  probe-validation.json \
  runtime-manifest.json \
  draft-smoke.json \
  target-tp2-smoke.json \
  validation.json \
  summary.md \
  nvidia-smi-L.txt \
  nvidia-smi-topo.txt
sha256sum "$SR_PHASE4_RUN/phase4a-review-bundle.tar.gz" | \
  tee "$SR_PHASE4_RUN/phase4a-review-bundle.sha256"
```

Return `summary.md`, `validation.json`, the bundle SHA256 and bundle. Stop after this run; do not
start cross-engine verification or latency/SLO experiments.
