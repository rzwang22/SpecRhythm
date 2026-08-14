# Phase 3 GPU server runbook

This runbook starts from the frozen Phase-2 head through the stacked
`codex/gpu-integration-v0.1` branch. Phase 3A is a correctness-first Transformers collector, not a
vLLM/SGLang serving implementation. It performs no Dual-Batch overlap. The latency commands below
must run on real NVIDIA GPUs; the code has no synthetic timing fallback.

## Repository audit at the Phase-2 boundary

The frozen Phase-2 head has no PyTorch, CUDA, NCCL, vLLM, SGLang, Transformers, or custom CUDA
execution path. Its `CandidateTreeOracle` and verifier are deterministic simulator proxies, not
model/tokenizer abstractions or GPU candidate kernels. It also has no measured latency-surface
loader. Its only artifact convention is to keep large simulator outputs outside Git under an
external results tree.

Phase 3 therefore adds a separate optional package instead of reusing simulator objects as if they
were real-model records. Transformers 4.56.x is the first correctness backend because the required
draft logits, entropy, and top-1/top-2 margin are directly observable. The production vLLM versus
SGLang decision remains open until these traces and surfaces have been reviewed.

## 1. Fetch the exact remote head

The following resolves the branch once, records that exact commit, and checks it out detached so a
later branch update cannot change a running experiment.

```bash
export SR_REPO_DIR="$HOME/src/SpecRhythm"
export SR_GPU_RESULTS="/home/rzwang/data/SpecRhythm-data/results"

test -d "$SR_REPO_DIR/.git" || git clone https://github.com/rzwang22/SpecRhythm.git "$SR_REPO_DIR"
cd "$SR_REPO_DIR"
git fetch origin codex/gpu-integration-v0.1
export SR_PHASE3_COMMIT="$(git rev-parse FETCH_HEAD)"
git switch --detach "$SR_PHASE3_COMMIT"
test "$(git rev-parse HEAD)" = "$SR_PHASE3_COMMIT"
git status --short
```

`git status --short` must be empty. Record `SR_PHASE3_COMMIT` with every artifact.

## 2. Create the pinned environment

The pinned baseline uses Python 3.9, PyTorch 2.7.1 with the official CUDA 12.8 wheel, and
Transformers 4.56.x. If the installed NVIDIA driver cannot support the CUDA 12.8 runtime, stop and
choose a compatible official PyTorch wheel; do not silently use CPU PyTorch.

```bash
cd "$SR_REPO_DIR"
python3.9 -m venv .venv-phase3
source .venv-phase3/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[dev,gpu]'
python -m pytest -q
python -m ruff check .
git diff --check
```

## 3. Probe CUDA, NCCL, topology, and the models

Set the two model directories explicitly. They must use the same tokenizer vocabulary and EOS
token for speculative decoding.

```bash
export SR_DRAFT_MODEL="/absolute/path/to/draft-model"
export SR_TARGET_MODEL="/absolute/path/to/target-model"
export SR_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-${SR_PHASE3_COMMIT:0:12}"
export SR_PHASE3_ROOT="$SR_GPU_RESULTS/phase3/$SR_RUN_ID"
mkdir -p "$SR_PHASE3_ROOT"

: "${SR_DRAFT_MODEL:?set SR_DRAFT_MODEL}"
: "${SR_TARGET_MODEL:?set SR_TARGET_MODEL}"
test -f "$SR_DRAFT_MODEL/config.json"
test -f "$SR_TARGET_MODEL/config.json"

nvidia-smi -L
nvidia-smi topo -m
specrhythm gpu-probe --output "$SR_PHASE3_ROOT/gpu-environment.json"

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda runtime", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("device count", torch.cuda.device_count())
print("nccl", torch.cuda.nccl.version())
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 5
PY
```

Probe all requested TP sizes. The validator intentionally returns nonzero if any requested size is
unsupported, so capture the report and then require TP=4 explicitly. TP=3 is inspected but is not
used for a benchmark unless a later reviewed model/engine plan supports it.

```bash
set +e
specrhythm tp-check \
  --model-config "$SR_DRAFT_MODEL/config.json" \
  --tp-sizes 1 2 3 4 \
  --output "$SR_PHASE3_ROOT/draft-tp-compatibility.json"
export SR_DRAFT_TP_STATUS="$?"
specrhythm tp-check \
  --model-config "$SR_TARGET_MODEL/config.json" \
  --tp-sizes 1 2 3 4 \
  --output "$SR_PHASE3_ROOT/target-tp-compatibility.json"
export SR_TARGET_TP_STATUS="$?"
set -e

python - "$SR_PHASE3_ROOT/target-tp-compatibility.json" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1]))
by_tp = {row["tp_size"]: row for row in report["results"]}
print("TP=3", by_tp[3])
print("TP=4", by_tp[4])
assert by_tp[4]["supported"], by_tp[4]["reason"]
PY
```

The validator never truncates heads, pads an incompatible architecture, or edits model structure.

## 4. CPU dry-run

This validates configuration, checkpoint, trace, manifest, selector isolation, and target-token
semantics. It does not emit latency data.

```bash
specrhythm phase3-run \
  --config configs/phase3_trace_1d4v.yaml \
  --backend dry-run \
  --mode target-only \
  --input configs/phase3-smoke-prompts.jsonl \
  --output-dir "$SR_PHASE3_ROOT/dry-target"

specrhythm phase3-run \
  --config configs/phase3_trace_1d4v.yaml \
  --backend dry-run \
  --mode serial \
  --input configs/phase3-smoke-prompts.jsonl \
  --output-dir "$SR_PHASE3_ROOT/dry-serial"

specrhythm phase3-validate \
  --trace-dir "$SR_PHASE3_ROOT/dry-serial" \
  --target-only-dir "$SR_PHASE3_ROOT/dry-target" \
  --output "$SR_PHASE3_ROOT/dry-validation.json"
```

## 5. Single-GPU draft smoke test

```bash
CUDA_VISIBLE_DEVICES=0 specrhythm phase3-run \
  --config configs/phase3_trace_1d4v.yaml \
  --mode draft-only \
  --input configs/phase3-smoke-prompts.jsonl \
  --output-dir "$SR_PHASE3_ROOT/gpu-draft-tp1" \
  --draft-gpus 0 \
  --draft-tp 1 \
  --environment-metadata "$SR_PHASE3_ROOT/gpu-environment.json"
```

## 6. TP=4 target-only smoke test

All four ranks execute identical target forwards; immutable checkpoint creation is race-safe and
rank 0 writes the compact summary and manifest.

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun \
  --standalone \
  --nproc-per-node=4 \
  -m specrhythm.cli phase3-run \
  --config configs/phase3_trace_1d4v.yaml \
  --mode target-only \
  --input configs/phase3-smoke-prompts.jsonl \
  --output-dir "$SR_PHASE3_ROOT/gpu-target-tp4" \
  --target-gpus 1,2,3,4 \
  --target-tp 4 \
  --environment-metadata "$SR_PHASE3_ROOT/gpu-environment.json"
```

## 7. Five-GPU serial trace smoke test

Run this as one coordinator process, not under `torchrun`. It loads the draft model on GPU 0 and
spawns a persistent NCCL TP=4 target group mapped to GPUs 1–4.

```bash
specrhythm phase3-run \
  --config configs/phase3_trace_1d4v.yaml \
  --mode serial \
  --input configs/phase3-smoke-prompts.jsonl \
  --output-dir "$SR_PHASE3_ROOT/gpu-serial-1d4v" \
  --draft-gpus 0 \
  --draft-tp 1 \
  --target-gpus 1,2,3,4 \
  --target-tp 4 \
  --environment-metadata "$SR_PHASE3_ROOT/gpu-environment.json"

specrhythm phase3-validate \
  --trace-dir "$SR_PHASE3_ROOT/gpu-serial-1d4v" \
  --target-only-dir "$SR_PHASE3_ROOT/gpu-target-tp4" \
  --output "$SR_PHASE3_ROOT/gpu-serial-validation.json"
```

The validation must report `target_only_semantic_equivalence: true` before collecting a larger
trace.

## 8. Resume an interrupted trace

Use the exact same commit, config, input, model revisions, seed, and output directory. Completed
cycle files are checked and never overwritten.

```bash
specrhythm phase3-run \
  --config configs/phase3_trace_1d4v.yaml \
  --mode serial \
  --input configs/phase3-smoke-prompts.jsonl \
  --output-dir "$SR_PHASE3_ROOT/gpu-serial-1d4v" \
  --draft-gpus 0 \
  --draft-tp 1 \
  --target-gpus 1,2,3,4 \
  --target-tp 4 \
  --environment-metadata "$SR_PHASE3_ROOT/gpu-environment.json" \
  --resume
```

## 9. Small latency calibration

These commands measure the current correctness primitives. Verification is serial full-context
execution, not a packed-tree serving kernel, and must not be presented as vLLM/SGLang or
Dual-Batch performance.

```bash
CUDA_VISIBLE_DEVICES=0 specrhythm phase3-benchmark \
  --config configs/phase3_latency_1d4v.yaml \
  --operation draft \
  --operation select \
  --output "$SR_PHASE3_ROOT/latency-draft-select.json" \
  --markdown-output "$SR_PHASE3_ROOT/latency-draft-select.md" \
  --environment-metadata "$SR_PHASE3_ROOT/gpu-environment.json"

CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun \
  --standalone \
  --nproc-per-node=4 \
  -m specrhythm.cli phase3-benchmark \
  --config configs/phase3_latency_1d4v.yaml \
  --operation verify \
  --output "$SR_PHASE3_ROOT/latency-verify-tp4.json" \
  --markdown-output "$SR_PHASE3_ROOT/latency-verify-tp4.md" \
  --environment-metadata "$SR_PHASE3_ROOT/gpu-environment.json"

specrhythm phase3-benchmark \
  --config configs/phase3_latency_1d4v.yaml \
  --operation transfer \
  --output "$SR_PHASE3_ROOT/latency-transfer.json" \
  --markdown-output "$SR_PHASE3_ROOT/latency-transfer.md" \
  --environment-metadata "$SR_PHASE3_ROOT/gpu-environment.json"
```

Every measurement contains CUDA and host distributions, peak allocated GPU memory, actual request
roots, search-pool nodes, verified candidates, and target-input positions. `N_search` and
`B_cand` remain separate axes.

## 10. Consolidate and inspect artifacts

```bash
specrhythm phase3-summarize \
  --trace-dir "$SR_PHASE3_ROOT/gpu-serial-1d4v" \
  --trace-output "$SR_PHASE3_ROOT/gpu-serial-1d4v.jsonl" \
  --output "$SR_PHASE3_ROOT/gpu-serial-1d4v-summary.json"

python - "$SR_PHASE3_ROOT" <<'PY'
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
for path in sorted(root.rglob("*.json")):
    value = json.load(open(path))
    print(path.relative_to(root), value.get("schema_version"), value.get("valid", ""))
PY

find "$SR_PHASE3_ROOT" -type f -maxdepth 3 -print | sort
```

The runner-generated `manifest.json` binds the source config, prompt input, environment probe,
commit, model configuration, command, seed, and immutable trace digest. Keep the entire run under
`$SR_GPU_RESULTS/phase3/`; do not add model weights, raw logits, checkpoint JSON, or consolidated
trace JSONL to Git.

After these smoke tests, stop and return the environment probe, TP report, validation JSON,
summary, manifests, and any error logs for review. Do not proceed to Dual-Batch integration from an
unreviewed smoke result.
