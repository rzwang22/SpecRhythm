# D3 four-path raw-logits probe: complete fresh-server runbook

Operator execution only, in an interactive Bash shell after the AutoDL restart. No previous exports are required. Run each block separately. If its printed `rc` is nonzero, stop manually, retain the output and inspect it before continuing. This runbook performs only the four D3 diagnostic paths, without changing GPU dependencies or vLLM patches. It does not start Target, Dual, D4 or D5.

The delivery copy replaces `@DELIVERED_COMMIT@` below with the exact tested commit. The repository copy is a template because a commit cannot contain its own SHA. Use the fully rendered delivery copy linked in the response.

## 1. Safe shell and conda bootstrap

```bash
set +e
set +u
set +o pipefail
trap - ERR
export SR_PHASE4B3_PROBE_COMMIT="@DELIVERED_COMMIT@"
export SR_PHASE4B3_PROBE_REPO="/root/autodl-tmp/src/SpecRhythm"
SR_PHASE4B3_CONDA_BASE=""
if command -v conda >/dev/null 2>&1; then
  SR_PHASE4B3_CONDA_BASE="$(conda info --base)"
elif test -x /root/miniconda3/bin/conda; then
  SR_PHASE4B3_CONDA_BASE="/root/miniconda3"
elif test -x /root/miniconda/bin/conda; then
  SR_PHASE4B3_CONDA_BASE="/root/miniconda"
fi
if test -n "$SR_PHASE4B3_CONDA_BASE" && test -f "$SR_PHASE4B3_CONDA_BASE/etc/profile.d/conda.sh"; then
  source "$SR_PHASE4B3_CONDA_BASE/etc/profile.d/conda.sh" &&
    conda activate /root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1
else
  echo "Conda bootstrap could not locate conda.sh; stop and inspect the server installation."
  false
fi
RC="$?"
echo "conda bootstrap rc=$RC"
```

## 2. Fetch, exact checkout and editable install

This rejects a dirty checkout rather than discarding changes. It does not merge or alter PR #4's Draft status.

```bash
cd "$SR_PHASE4B3_PROBE_REPO" &&
  test -z "$(git status --porcelain)" &&
  git fetch origin codex/vllm-serving-v0.1 &&
  git switch --detach "$SR_PHASE4B3_PROBE_COMMIT" &&
  test "$(git rev-parse HEAD)" = "$SR_PHASE4B3_PROBE_COMMIT" &&
  python -m pip install -e . --no-deps --no-build-isolation
RC="$?"
echo "exact checkout and editable install rc=$RC"
```

## 3. Every required environment variable

The target model path only resolves the original config; no Target model is loaded. Workload/reference paths and previous result/provenance directories are not needed: all token inputs, history, metadata fingerprints and patch hashes are committed in the fixture.

```bash
export SR_DRAFT_MODEL="/root/autodl-tmp/models/Qwen3-0.6B"
export SR_TARGET_MODEL="/root/autodl-tmp/models/Qwen3-32B"
export SR_VLLM_SOURCE="/root/autodl-tmp/src/vllm-v0.25.1"
export SR_PHASE4B_CONFIG="$SR_PHASE4B3_PROBE_REPO/configs/phase4b_dual_batch_1d2v.yaml"
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_BATCH_INVARIANT=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export SR_PHASE4_DRAFT_BACKEND=vllm-batched
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
unset MASTER_ADDR MASTER_PORT
unset SR_PHASE4_DRAFT_SOCKET SR_PHASE4_DUAL_DRAFT_SOCKET
unset PHASE4B1_NUMERICAL_PLAN PHASE4B1_NUMERICAL_OUTPUT
SR_VLLM_ROOT="$(python - <<'PY'
from importlib import metadata
print(metadata.distribution('vllm').locate_file(''))
PY
)"
RC="$?"
export SR_VLLM_ROOT
echo "locate installed vLLM rc=$RC"
```

## 4. CPU preflight, five-patch verification and fresh root

This checks Python/Torch/vLLM versions, the pinned vLLM Git source, original five patch file hashes, all 25 installed API file hashes, six numerical-source files, the exact config/token hashes and model/tokenizer metadata. It also hashes the actual safetensors weights. It does not load a model or initialize CUDA. If the installed patch check fails, stop; this runbook never applies or changes patches automatically.

```bash
python -m specrhythm.phase4.draft_logits_probe --check-only \
  --config "$SR_PHASE4B_CONFIG" \
  --expected-commit "$SR_PHASE4B3_PROBE_COMMIT"
RC="$?"
echo "CPU preflight and unchanged five-patch check rc=$RC"
```

After `rc=0`, create the new operator root. Existing attempts are preserved.

```bash
export SR_PHASE4B3_PROBE_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/$SR_PHASE4B3_PROBE_COMMIT/D3-logits-$(date -u +%Y%m%dT%H%M%SZ)-$$"
python - <<'PY'
import os
from pathlib import Path
root = Path(os.environ['SR_PHASE4B3_PROBE_ROOT'])
root.mkdir(parents=True, exist_ok=False)
print(root)
PY
RC="$?"
echo "fresh operator root rc=$RC"
```

## 5. The single GPU command

The parent runs A, B, C and D in four sequential isolated processes; each process revalidates the inputs before loading a model. The fresh `probe` subdirectory is created exclusively by the probe. The same HF mismatch is allowed as diagnostic evidence, never relabeled as a D3 pass.

```bash
python -u -m specrhythm.phase4.draft_logits_probe --allow-gpu \
  --config "$SR_PHASE4B_CONFIG" \
  --expected-commit "$SR_PHASE4B3_PROBE_COMMIT" \
  --output "$SR_PHASE4B3_PROBE_ROOT/probe" \
  >"$SR_PHASE4B3_PROBE_ROOT/run.log" 2>&1
RC="$?"
echo "four-path D3 observation rc=$RC"
tail -n 40 "$SR_PHASE4B3_PROBE_ROOT/run.log"
```

`rc=0` means observation completed; inspect `probe/classification.json` for strong-evidence/inconclusive status. A nonzero result retains invalid-path diagnostics. If D does not reproduce, keep that run too. Do not change flags, patch vLLM, retry automatically or continue to D4/D5.

## 6. Package all artifacts on success or failure

```bash
tar -czf "$SR_PHASE4B3_PROBE_ROOT.tar.gz" \
  -C "$(dirname "$SR_PHASE4B3_PROBE_ROOT")" "$(basename "$SR_PHASE4B3_PROBE_ROOT")"
RC="$?"
echo "package four-path diagnostic rc=$RC"
echo "$SR_PHASE4B3_PROBE_ROOT.tar.gz"
```

Return this archive. It contains the immutable frozen inputs, all four raw-vector reports, row-domain/history evidence, per-path logs, classification/deltas and the enclosing run log. No additional gates or K/V-content expansion should be run at this stage.
