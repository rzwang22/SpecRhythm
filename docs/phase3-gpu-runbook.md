# Phase 3 GPU server runbook

This runbook starts from the frozen Phase-2 head through the stacked
`codex/gpu-integration-v0.1` branch. Phase 3A is a correctness-first Transformers collector and
Phase 3B.1 hardens primitive measurement evidence. Neither is a vLLM/SGLang serving
implementation, and neither performs Dual-Batch overlap. GPU latency commands have no synthetic
fallback. Section 12 is the completed three-GPU Phase 3B.1 repeatability protocol; Section 13 is
the current Phase 3C.1 R3-real pilot. The earlier commands remain as historical bootstrap examples.

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

The checked-in topology variants are:

- `phase3_trace_1d4v.yaml` / `phase3_latency_1d4v.yaml`: five GPUs, draft TP=1 and target TP=4.
- `phase3_trace_1d2v.yaml` / `phase3_latency_1d2v.yaml`: three GPUs, draft TP=1 and target TP=2.

Do not use a 1D4V benchmark config on a three-GPU host. Unlike `phase3-run`,
`phase3-benchmark` does not accept model/GPU/TP overrides, so the topology-specific latency file
is mandatory.

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

## 11. Validated three-GPU A800 path (1D2V)

The observed three-GPU host has NV8 connectivity between GPUs 0, 1, and 2. Activate the existing
Conda environment and keep the previously created result root:

```bash
cd /root/autodl-tmp/src/SpecRhythm
conda activate /root/autodl-tmp/envs/specrhythm-phase3-80d5769
export SR_DRAFT_MODEL="/root/autodl-tmp/models/Qwen3-0.6B"
export SR_TARGET_MODEL="/root/autodl-tmp/models/Qwen3-32B"
export SR_PHASE3_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase3/bootstrap-20260814T141034Z-80d576912028"
```

The correctness smoke at commit `80d576912028da2d32cd1d8ba5cb593d10a547ae` passed TP=2 target
loading, serial trace accounting, and target-only token equivalence. After pulling a later reviewed
PR #3 head that contains the 1D2V configs, run the small correctness-backend calibration as three
separate commands:

```bash
CUDA_VISIBLE_DEVICES=0 specrhythm phase3-benchmark \
  --config configs/phase3_latency_1d2v.yaml \
  --operation draft \
  --operation select \
  --output "$SR_PHASE3_ROOT/latency-draft-select-tp1.json" \
  --markdown-output "$SR_PHASE3_ROOT/latency-draft-select-tp1.md" \
  --environment-metadata "$SR_PHASE3_ROOT/gpu-environment.json" \
  2>&1 | tee "$SR_PHASE3_ROOT/latency-draft-select-tp1.log"

CUDA_VISIBLE_DEVICES=1,2 torchrun \
  --standalone \
  --nproc-per-node=2 \
  -m specrhythm.cli phase3-benchmark \
  --config configs/phase3_latency_1d2v.yaml \
  --operation verify \
  --output "$SR_PHASE3_ROOT/latency-verify-tp2.json" \
  --markdown-output "$SR_PHASE3_ROOT/latency-verify-tp2.md" \
  --environment-metadata "$SR_PHASE3_ROOT/gpu-environment.json" \
  2>&1 | tee "$SR_PHASE3_ROOT/latency-verify-tp2.log"

unset CUDA_VISIBLE_DEVICES
specrhythm phase3-benchmark \
  --config configs/phase3_latency_1d2v.yaml \
  --operation transfer \
  --output "$SR_PHASE3_ROOT/latency-transfer-0-to-1.json" \
  --markdown-output "$SR_PHASE3_ROOT/latency-transfer-0-to-1.md" \
  --environment-metadata "$SR_PHASE3_ROOT/gpu-environment.json" \
  2>&1 | tee "$SR_PHASE3_ROOT/latency-transfer-0-to-1.log"
```

These measurements exercise the Transformers correctness primitives. They are not packed-tree,
continuous-batching, vLLM/SGLang, Dual-Batch, or end-to-end serving measurements.

## 12. Phase 3B.1 canonical three-run A800 rerun

The initial v1 smoke proved that the programs ran, but its TP memory dictionary was rank-0-local,
and two verify files came from different commits. Do not compare or average those files. The v2
gate below records and validates every TP rank independently, uses the per-iteration maximum rank
latency, retains raw samples, captures hardware state before and after, and rejects cross-commit or
cross-semantics comparisons. It still measures HF full-context correctness primitives only.

Activate the existing environment, resolve one immutable branch head, and keep all outputs outside
Git:

```bash
conda activate /root/autodl-tmp/envs/specrhythm-phase3-80d5769
set -euo pipefail
cd /root/autodl-tmp/src/SpecRhythm

git fetch origin codex/gpu-integration-v0.1
export SR_PHASE3B1_COMMIT="$(git rev-parse FETCH_HEAD)"
git switch --detach "$SR_PHASE3B1_COMMIT"
test "$(git rev-parse HEAD)" = "$SR_PHASE3B1_COMMIT"
test -z "$(git status --short)"

export SR_DRAFT_MODEL="/root/autodl-tmp/models/Qwen3-0.6B"
export SR_TARGET_MODEL="/root/autodl-tmp/models/Qwen3-32B"
export SR_PHASE3B1_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase3b1/${SR_PHASE3B1_COMMIT}"

test -f "$SR_DRAFT_MODEL/config.json"
test -f "$SR_TARGET_MODEL/config.json"
mkdir -p "$SR_PHASE3B1_ROOT"

python -m pip install -e '.[dev,gpu]'
python -m pytest -q
python -m ruff check .
git diff --check

nvidia-smi -L | tee "$SR_PHASE3B1_ROOT/nvidia-smi-L.txt"
nvidia-smi topo -m | tee "$SR_PHASE3B1_ROOT/nvidia-smi-topo.txt"
specrhythm gpu-probe --output "$SR_PHASE3B1_ROOT/gpu-environment.json"

specrhythm phase3-selector-dry-run \
  --request-count 2 \
  --search-pool-size 16 \
  --candidate-budget 8 \
  --output "$SR_PHASE3B1_ROOT/selector-interface-dry-run.json"
```

Run draft/synthetic-top-k, TP=2 serial-full-context verify, and all configured bare-copy
directions three times. The transfer operation includes GPU 0→1, GPU 1→0, and GPU 1→2, with
4 KiB, 64 KiB, 1 MiB, 16 MiB, 64 MiB, and 256 MiB payloads.

```bash
for sr_repeat in 1 2 3; do
  export SR_PHASE3B1_RUN="$SR_PHASE3B1_ROOT/run-$sr_repeat"
  mkdir -p "$SR_PHASE3B1_RUN"

  CUDA_VISIBLE_DEVICES=0 specrhythm phase3-benchmark \
    --config configs/phase3_latency_1d2v.yaml \
    --operation draft \
    --operation select \
    --output "$SR_PHASE3B1_RUN/draft-select.json" \
    --markdown-output "$SR_PHASE3B1_RUN/draft-select.md" \
    --environment-metadata "$SR_PHASE3B1_ROOT/gpu-environment.json" \
    2>&1 | tee "$SR_PHASE3B1_RUN/draft-select.log"

  CUDA_VISIBLE_DEVICES=1,2 torchrun \
    --standalone \
    --nproc-per-node=2 \
    -m specrhythm.cli phase3-benchmark \
    --config configs/phase3_latency_1d2v.yaml \
    --operation verify \
    --output "$SR_PHASE3B1_RUN/verify-tp2.json" \
    --markdown-output "$SR_PHASE3B1_RUN/verify-tp2.md" \
    --environment-metadata "$SR_PHASE3B1_ROOT/gpu-environment.json" \
    2>&1 | tee "$SR_PHASE3B1_RUN/verify-tp2.log"

  env -u CUDA_VISIBLE_DEVICES specrhythm phase3-benchmark \
    --config configs/phase3_latency_1d2v.yaml \
    --operation transfer \
    --output "$SR_PHASE3B1_RUN/transfer.json" \
    --markdown-output "$SR_PHASE3B1_RUN/transfer.md" \
    --environment-metadata "$SR_PHASE3B1_ROOT/gpu-environment.json" \
    2>&1 | tee "$SR_PHASE3B1_RUN/transfer.log"

  for sr_report in draft-select verify-tp2 transfer; do
    specrhythm phase3-benchmark-validate \
      --input "$SR_PHASE3B1_RUN/$sr_report.json" \
      --output "$SR_PHASE3B1_RUN/$sr_report-validation.json"
  done
done
```

Compare only like-for-like reports from the exact same commit and config:

```bash
specrhythm phase3-benchmark-compare \
  --input "$SR_PHASE3B1_ROOT/run-1/draft-select.json" \
  --input "$SR_PHASE3B1_ROOT/run-2/draft-select.json" \
  --input "$SR_PHASE3B1_ROOT/run-3/draft-select.json" \
  --output "$SR_PHASE3B1_ROOT/draft-select-comparison.json" \
  --markdown-output "$SR_PHASE3B1_ROOT/draft-select-comparison.md"

specrhythm phase3-benchmark-compare \
  --input "$SR_PHASE3B1_ROOT/run-1/verify-tp2.json" \
  --input "$SR_PHASE3B1_ROOT/run-2/verify-tp2.json" \
  --input "$SR_PHASE3B1_ROOT/run-3/verify-tp2.json" \
  --output "$SR_PHASE3B1_ROOT/verify-tp2-comparison.json" \
  --markdown-output "$SR_PHASE3B1_ROOT/verify-tp2-comparison.md"

specrhythm phase3-benchmark-compare \
  --input "$SR_PHASE3B1_ROOT/run-1/transfer.json" \
  --input "$SR_PHASE3B1_ROOT/run-2/transfer.json" \
  --input "$SR_PHASE3B1_ROOT/run-3/transfer.json" \
  --output "$SR_PHASE3B1_ROOT/transfer-comparison.json" \
  --markdown-output "$SR_PHASE3B1_ROOT/transfer-comparison.md"
```

Finally, require two non-empty TP ranks in every verify cell and inspect the compact summaries:

```bash
python - "$SR_PHASE3B1_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for path in sorted(root.glob("run-*/verify-tp2.json")):
    report = json.load(open(path))
    assert report["validation"]["valid"], report["validation"]
    assert report["git_commit"] == root.name
    for cell in report["measurements"]:
        ranks = cell["rank_measurements"]
        assert [rank["global_rank"] for rank in ranks] == [0, 1]
        for rank in ranks:
            assert rank["model_parameter_count"] > 0
            assert rank["parameter_bytes"] > 0
            assert rank["max_allocated_memory_bytes"] > 0
            assert rank["model_parameters_on_expected_device"] is True
            assert len(rank["cuda_samples_ms"]) == 30
            assert len(rank["host_samples_ms"]) == 30
    print(path.relative_to(root), "PASS")

for name in (
    "draft-select-comparison.json",
    "verify-tp2-comparison.json",
    "transfer-comparison.json",
):
    report = json.load(open(root / name))
    assert report["valid"], report["errors"]
    print(name, "cells=", len(report["cells"]), "PASS")
PY

cat "$SR_PHASE3B1_ROOT/verify-tp2-comparison.md"
cat "$SR_PHASE3B1_ROOT/draft-select-comparison.md"
cat "$SR_PHASE3B1_ROOT/transfer-comparison.md"
```

Stop after returning the three comparison reports, three run summaries, validation JSON, and any
warning/error logs. Do not start R3-real, packed-tree verification, simulator calibration,
Dual-Batch, Eager, or end-to-end SLO evaluation from these primitive measurements.

## 13. Phase 3C.1 R3-real pilot on three A800 GPUs

Phase 3B.1 has passed the user's three-run A800 validation. This section starts a separate
Phase 3C.1 trace pilot. It does not load the Phase 3B.1 latency JSON into the simulator and does
not implement packed-tree verification, a serving engine, overlap, Eager, SLO scheduling or
goodput evaluation.

### Required external inputs

Provide exactly these three public-data exports before running the checked configuration:

1. `HumanEval.jsonl` or `HumanEval.jsonl.gz`, one JSON object per line with `task_id` and `prompt`;
2. `sharegpt.jsonl`, one conversation object per line with `id` and `conversations`/`messages`;
3. `cnn_dailymail-test.jsonl`, one JSON object per line with `id`, `article` and `highlights`.

The builder also needs the existing Mooncake `conversation_trace.jsonl`. Do not place any of
these files in the repository. If a downloaded ShareGPT artifact is one large JSON array, export
it to one-object-per-line JSONL before this run; the builder intentionally does not guess or
silently rewrite source formats.

Fetch one exact Draft PR #3 head, activate the existing environment, install that checkout and
bind the external paths:

```bash
conda activate /root/autodl-tmp/envs/specrhythm-phase3-80d5769
set -euo pipefail
cd /root/autodl-tmp/src/SpecRhythm

git fetch origin codex/gpu-integration-v0.1
export SR_PHASE3C_COMMIT="$(git rev-parse FETCH_HEAD)"
git switch --detach "$SR_PHASE3C_COMMIT"
test "$(git rev-parse HEAD)" = "$SR_PHASE3C_COMMIT"
test -z "$(git status --short)"

python -m pip install -e '.[dev,gpu]'
python -m pytest -q
python -m ruff check .
git diff --check

export SR_DRAFT_MODEL="/root/autodl-tmp/models/Qwen3-0.6B"
export SR_TARGET_MODEL="/root/autodl-tmp/models/Qwen3-32B"
export SR_MOONCAKE_TRACE="/root/autodl-tmp/SpecRhythm-data/raw/Mooncake/FAST25-release/traces/conversation_trace.jsonl"
export SR_HUMANEVAL_JSONL="/root/autodl-tmp/SpecRhythm-data/raw/public/HumanEval/HumanEval.jsonl.gz"
export SR_SHAREGPT_JSONL="/root/autodl-tmp/SpecRhythm-data/raw/public/ShareGPT/sharegpt.jsonl"
export SR_CNNDM_JSONL="/root/autodl-tmp/SpecRhythm-data/raw/public/CNN-DailyMail/cnn_dailymail-test.jsonl"
export SR_PHASE3C_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase3c/$SR_PHASE3C_COMMIT"

test -f "$SR_DRAFT_MODEL/config.json"
test -f "$SR_TARGET_MODEL/config.json"
test -f "$SR_MOONCAKE_TRACE"
test -f "$SR_HUMANEVAL_JSONL"
test -f "$SR_SHAREGPT_JSONL"
test -f "$SR_CNNDM_JSONL"
mkdir -p "$SR_PHASE3C_ROOT"
```

The config uses GPU 0 for draft TP=1 and launches a persistent target TP=2 worker group on
physical GPUs 1–2. The target command is a coordinator command and must not be wrapped in
`torchrun`.

### Five-request smoke: 3 code, 1 chat, 1 summarization

Build a real-tokenized workload. This stage loads only the Qwen3 tokenizer and does not run a GPU
model:

```bash
export SR_PHASE3C_RUN="$SR_PHASE3C_ROOT/smoke-5"
mkdir -p "$SR_PHASE3C_RUN"

specrhythm phase3c-workload-build \
  --config configs/phase3c_r3_real_1d2v.yaml \
  --request-count 5 \
  --output "$SR_PHASE3C_RUN/workload.jsonl" \
  --manifest "$SR_PHASE3C_RUN/workload-manifest.json"
```

Generate the shared nested forest on GPU 0, then generate the immutable target trajectory on
GPUs 1–2:

```bash
CUDA_VISIBLE_DEVICES=0 specrhythm phase3c-draft-forest \
  --config configs/phase3c_r3_real_1d2v.yaml \
  --workload "$SR_PHASE3C_RUN/workload.jsonl" \
  --output-dir "$SR_PHASE3C_RUN/draft"

env -u CUDA_VISIBLE_DEVICES specrhythm phase3c-target-trajectory \
  --config configs/phase3c_r3_real_1d2v.yaml \
  --workload "$SR_PHASE3C_RUN/workload.jsonl" \
  --output-dir "$SR_PHASE3C_RUN/target"
```

Join labels and replay all fixed selectors on CPU, then validate and summarize:

```bash
specrhythm phase3c-label-join \
  --workload "$SR_PHASE3C_RUN/workload.jsonl" \
  --forest-dir "$SR_PHASE3C_RUN/draft" \
  --target-dir "$SR_PHASE3C_RUN/target" \
  --output-dir "$SR_PHASE3C_RUN/labeled"

specrhythm phase3c-selector-replay \
  --config configs/phase3c_r3_real_1d2v.yaml \
  --labeled-dir "$SR_PHASE3C_RUN/labeled" \
  --output-dir "$SR_PHASE3C_RUN/selectors"

specrhythm phase3c-validate \
  --workload "$SR_PHASE3C_RUN/workload.jsonl" \
  --forest-dir "$SR_PHASE3C_RUN/draft" \
  --target-dir "$SR_PHASE3C_RUN/target" \
  --labeled-dir "$SR_PHASE3C_RUN/labeled" \
  --selector-dir "$SR_PHASE3C_RUN/selectors" \
  --output "$SR_PHASE3C_RUN/validation.json"

specrhythm phase3c-summary \
  --labeled-dir "$SR_PHASE3C_RUN/labeled" \
  --selector-dir "$SR_PHASE3C_RUN/selectors" \
  --output "$SR_PHASE3C_RUN/selector-diagnosis.json" \
  --markdown-output "$SR_PHASE3C_RUN/selector-diagnosis.md"
```

Require the exact smoke mixture, real tokenizer lengths, all five immutable stage records and no
performance fields:

```bash
python - "$SR_PHASE3C_RUN" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
manifest = json.load(open(root / "workload-manifest.json"))
validation = json.load(open(root / "validation.json"))
diagnosis = json.load(open(root / "selector-diagnosis.json"))
assert manifest["task_counts"] == {"code": 3, "chat": 1, "summarization": 1}
assert manifest["prompt_lengths_are_proxy"] is False
assert hashlib.sha256((root / "workload.jsonl").read_bytes()).hexdigest() == manifest["output_workload_sha256"]
assert validation["valid"], validation["errors"]
assert validation["forest_records"] == validation["target_records"] == validation["labeled_records"] == 5
assert diagnosis["request_count"] == 5
assert diagnosis["gpu_performance_result"] is False
assert diagnosis["reports_goodput"] is False
assert diagnosis["reports_slo_attainment"] is False
assert diagnosis["reports_speedup"] is False
print("Phase 3C.1 five-request smoke PASS")
PY
```

If a model stage is interrupted, rerun only that exact stage with `--resume`. For example:

```bash
env -u CUDA_VISIBLE_DEVICES specrhythm phase3c-target-trajectory \
  --config configs/phase3c_r3_real_1d2v.yaml \
  --workload "$SR_PHASE3C_RUN/workload.jsonl" \
  --output-dir "$SR_PHASE3C_RUN/target" \
  --resume
```

Completed request files are compared by identity and never replaced.

### 100-request pilot: 60 code, 20 chat, 20 summarization

Run this only after the five-request validation passes:

```bash
export SR_PHASE3C_RUN="$SR_PHASE3C_ROOT/pilot-100"
mkdir -p "$SR_PHASE3C_RUN"

specrhythm phase3c-workload-build \
  --config configs/phase3c_r3_real_1d2v.yaml \
  --request-count 100 \
  --output "$SR_PHASE3C_RUN/workload.jsonl" \
  --manifest "$SR_PHASE3C_RUN/workload-manifest.json"

CUDA_VISIBLE_DEVICES=0 specrhythm phase3c-draft-forest \
  --config configs/phase3c_r3_real_1d2v.yaml \
  --workload "$SR_PHASE3C_RUN/workload.jsonl" \
  --output-dir "$SR_PHASE3C_RUN/draft" \
  --resume

env -u CUDA_VISIBLE_DEVICES specrhythm phase3c-target-trajectory \
  --config configs/phase3c_r3_real_1d2v.yaml \
  --workload "$SR_PHASE3C_RUN/workload.jsonl" \
  --output-dir "$SR_PHASE3C_RUN/target" \
  --resume

specrhythm phase3c-label-join \
  --workload "$SR_PHASE3C_RUN/workload.jsonl" \
  --forest-dir "$SR_PHASE3C_RUN/draft" \
  --target-dir "$SR_PHASE3C_RUN/target" \
  --output-dir "$SR_PHASE3C_RUN/labeled" \
  --resume

specrhythm phase3c-selector-replay \
  --config configs/phase3c_r3_real_1d2v.yaml \
  --labeled-dir "$SR_PHASE3C_RUN/labeled" \
  --output-dir "$SR_PHASE3C_RUN/selectors" \
  --resume

specrhythm phase3c-validate \
  --workload "$SR_PHASE3C_RUN/workload.jsonl" \
  --forest-dir "$SR_PHASE3C_RUN/draft" \
  --target-dir "$SR_PHASE3C_RUN/target" \
  --labeled-dir "$SR_PHASE3C_RUN/labeled" \
  --selector-dir "$SR_PHASE3C_RUN/selectors" \
  --output "$SR_PHASE3C_RUN/validation.json"

specrhythm phase3c-summary \
  --labeled-dir "$SR_PHASE3C_RUN/labeled" \
  --selector-dir "$SR_PHASE3C_RUN/selectors" \
  --output "$SR_PHASE3C_RUN/selector-diagnosis.json" \
  --markdown-output "$SR_PHASE3C_RUN/selector-diagnosis.md"
```

The new pilot directory is empty, so using `--resume` on its first model/join/replay invocation is
safe and also makes the commands directly reusable after interruption. Perform the final gate:

```bash
python - "$SR_PHASE3C_RUN" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
manifest = json.load(open(root / "workload-manifest.json"))
validation = json.load(open(root / "validation.json"))
diagnosis = json.load(open(root / "selector-diagnosis.json"))
assert manifest["request_count"] == 100
assert manifest["task_counts"] == {"code": 60, "chat": 20, "summarization": 20}
assert validation["valid"], validation["errors"]
assert validation["forest_records"] == validation["target_records"] == validation["labeled_records"] == 100
assert diagnosis["request_count"] == 100
assert diagnosis["evidence_scope"] == "100-request-pilot-schema-and-selector-signal-only"
print("Phase 3C.1 100-request pilot PASS")
PY

cat "$SR_PHASE3C_RUN/selector-diagnosis.md"
```

Return the workload manifest, all stage summaries, validation JSON, selector diagnosis JSON/MD,
and error logs if any. Stop there. Do not interpret the pilot as a GPU speedup or proceed to
packed-tree, serving-engine, Dual-Batch, Eager, SLO or simulator work without review.
