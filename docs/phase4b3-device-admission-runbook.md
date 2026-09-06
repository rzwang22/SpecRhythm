# Phase 4B.3 device admission: fresh-server D1 → D3 rerun

Operator execution only. The agent has run no GPU. Use a fresh interactive **Bash** shell after restarting AutoDL. Execute each block separately. If its printed `rc` is nonzero, stop manually and preserve the root; the commands keep the interactive shell open. This runbook stops at the D1–D3 aggregate. It never applies a patch, changes precision/kernels, runs Dual, or starts D4/D5.

The delivered copy replaces `@DELIVERED_COMMIT@` with the exact tested commit. The repository copy is a template because a commit cannot contain its own SHA. Existing inputs below are explicitly named, not inherited from previous shell exports. If a retained file was moved, change its path to that same artifact; do not substitute a different retained proof or patch manifest.

## 1. Safe shell, conda, exact checkout and install

```bash
set +e
set +u
set +o pipefail
trap - ERR
export SR_PHASE4B_COMMIT="@DELIVERED_COMMIT@"
export SR_PHASE4B3_REPO="/root/autodl-tmp/src/SpecRhythm"
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
  echo "Conda bootstrap missing; stop and inspect the installation."
  false
fi
RC="$?"
echo "conda activation rc=$RC"
```

```bash
cd "$SR_PHASE4B3_REPO" &&
  test -z "$(git status --porcelain)" &&
  git fetch origin codex/vllm-serving-v0.1 &&
  git switch --detach "$SR_PHASE4B_COMMIT" &&
  test "$(git rev-parse HEAD)" = "$SR_PHASE4B_COMMIT" &&
  python -m pip install -e . --no-deps --no-build-isolation
RC="$?"
echo "exact checkout and editable install rc=$RC"
```

## 2. Complete environment and retained input paths

GPU0 is Draft; GPUs1,2 are Target TP2. This qualification process sees only GPU0 and uses TP1/world1. BF16, MRV1, eager, proposal budget 4, batch-invariance and the five patches remain fixed. A new A800 UUID is permitted only when its current binding and SM80 execution regime validate.

```bash
export SR_DRAFT_MODEL="/root/autodl-tmp/models/Qwen3-0.6B"
export SR_TARGET_MODEL="/root/autodl-tmp/models/Qwen3-32B"
export SR_VLLM_SOURCE="/root/autodl-tmp/src/vllm-v0.25.1"
export SR_PHASE4B_CONFIG="$SR_PHASE4B3_REPO/configs/phase4b_dual_batch_1d2v.yaml"
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_BATCH_INVARIANT=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export SR_PHASE4_DRAFT_BACKEND=vllm-batched
export SR_PHASE4_DUAL_UUID_QUERY_MODE=live
export SR_PHASE4B3_HF_METRICS=0
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
unset MASTER_ADDR MASTER_PORT PYTHONPATH
unset SR_PHASE4_DRAFT_SOCKET SR_PHASE4_DUAL_DRAFT_SOCKET
unset PHASE4B1_NUMERICAL_PLAN PHASE4B1_NUMERICAL_OUTPUT
export SR_PHASE4B3_BASELINE_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/8997ec9d0053d8a6b40b99b8b9694099a7607f1d/phase4b2-natural-teardown-20260905T125632Z-1467"
export SR_PHASE4B3_PROBE_SOURCE="/root/autodl-tmp/SpecRhythm-data/results/phase4/dd02c7ffe7f419134d9134b8feb932c20652ca34/D3-logits-20260906T093347Z-1477/probe"
SR_VLLM_ROOT="$(python - <<'PY'
from importlib import metadata
print(metadata.distribution('vllm').locate_file(''))
PY
)"
RC="$?"
export SR_VLLM_ROOT
echo "locate installed vLLM rc=$RC"
```

## 3. Freeze a new result root and copy evidence without modifying originals

The source probe is the operator's completed four-path experiment, not a new GPU run. Its full vectors are recomputed and verified offline. The old five-patch apply manifest is reused byte-for-byte. Use a NEW root because the preceding root failed admission. Rerun all five gates on the current server; preserve old roots and probe files unchanged. The old probe UUID remains historical provenance and does not have to equal the current A800 UUID.

```bash
export SR_PHASE4B3_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/$SR_PHASE4B_COMMIT/draft-admission-$(date -u +%Y%m%dT%H%M%SZ)-$$"
export SR_PHASE4B_ENVIRONMENT="$SR_PHASE4B3_ROOT/environment.json"
export SR_PHASE4B_TOPOLOGY="$SR_PHASE4B3_ROOT/topology.json"
export SR_PHASE4B_PATCH_MANIFEST="$SR_PHASE4B3_ROOT/patch-stage/vllm-patch-stack.json"
python - <<'PY'
import hashlib, json, os, shutil
from pathlib import Path
from specrhythm.phase4.batched_draft_service import write_immutable_report
from specrhythm.phase4.draft_qualification import load_probe_qualification
inputs = {'config': {'path': os.environ['SR_PHASE4B_CONFIG'], 'sha256': hashlib.sha256(Path(os.environ['SR_PHASE4B_CONFIG']).read_bytes()).hexdigest()}}
proof = load_probe_qualification(Path(os.environ['SR_PHASE4B3_PROBE_SOURCE']))
assert proof['valid'], proof['errors']
manifest = Path(os.environ['SR_PHASE4B3_BASELINE_ROOT']) / 'patch-stage/vllm-patch-stack.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == '33c5a75f95a490695d966548d6f638bed810a410ff0d028ee114bfbe854069fc'
root = Path(os.environ['SR_PHASE4B3_ROOT'])
root.mkdir(parents=True, exist_ok=False)
(root / 'patch-stage').mkdir()
shutil.copyfile(manifest, os.environ['SR_PHASE4B_PATCH_MANIFEST'])
shutil.copytree(os.environ['SR_PHASE4B3_PROBE_SOURCE'], root / 'probe-evidence')
write_immutable_report(root / 'regime-qualification.json', proof)
write_immutable_report(root / 'fixed-inputs.json', {'execution_commit': os.environ['SR_PHASE4B_COMMIT'], 'inputs': inputs})
print(root)
PY
RC="$?"
echo "freeze inputs and retained four-path evidence rc=$RC"
```

```bash
python integrations/vllm/manage_patch.py check \
  --vllm-root "$SR_VLLM_ROOT" --source "$SR_VLLM_SOURCE" \
  --expect-state patched --manifest "$SR_PHASE4B3_ROOT/installed-patch-check.json" &&
python -m specrhythm.phase4.draft_logits_probe --check-only \
  --config "$SR_PHASE4B_CONFIG" --expected-commit "$SR_PHASE4B_COMMIT"
RC="$?"
echo "unchanged five patches, source, config and weight checks rc=$RC"
```

Capture current environment/topology with the existing metadata probe. This does not run generation.

```bash
env -u CUDA_VISIBLE_DEVICES VLLM_BATCH_INVARIANT=1 specrhythm phase4-probe \
  --config "$SR_PHASE4B_CONFIG" --vllm-source "$SR_VLLM_SOURCE" \
  --environment-output "$SR_PHASE4B_ENVIRONMENT" \
  --topology-output "$SR_PHASE4B_TOPOLOGY" \
  --validation-output "$SR_PHASE4B3_ROOT/probe-validation.json"
RC="$?"
echo "current environment/topology rc=$RC"
```

## 4. D1/D2 and fresh D3 B2/B4/B8

D1/D2 retain their v1 semantics. D3 uses a distinct v2 entry point; **do not add `--diagnostic`**. No materialization/logits observer is installed. Ordinary API-boundary host-state validation remains mandatory. D3 is a semantic qualification. No performance run is included here.

```bash
python -m specrhythm.phase4.draft_gate --allow-gpu \
  --config "$SR_PHASE4B_CONFIG" --gate D1 \
  --output "$SR_PHASE4B3_ROOT/D1" >"$SR_PHASE4B3_ROOT/D1.log" 2>&1
RC="$?"
echo "D1 construction/cleanup rc=$RC"
```

```bash
python -m specrhythm.phase4.draft_gate --allow-gpu \
  --config "$SR_PHASE4B_CONFIG" --gate D2 \
  --output "$SR_PHASE4B3_ROOT/D2" >"$SR_PHASE4B3_ROOT/D2.log" 2>&1
RC="$?"
echo "D2 single-request semantics rc=$RC"
```

```bash
python -m specrhythm.phase4.draft_qualification_gate --allow-gpu \
  --config "$SR_PHASE4B_CONFIG" --expected-commit "$SR_PHASE4B_COMMIT" \
  --probe-root "$SR_PHASE4B3_ROOT/probe-evidence" --request-count 2 \
  --output "$SR_PHASE4B3_ROOT/D3-B2" >"$SR_PHASE4B3_ROOT/D3-B2.log" 2>&1
RC="$?"
echo "D3 v2 B2 qualification rc=$RC"
```

```bash
python -m specrhythm.phase4.draft_qualification_gate --allow-gpu \
  --config "$SR_PHASE4B_CONFIG" --expected-commit "$SR_PHASE4B_COMMIT" \
  --probe-root "$SR_PHASE4B3_ROOT/probe-evidence" --request-count 4 \
  --output "$SR_PHASE4B3_ROOT/D3-B4" >"$SR_PHASE4B3_ROOT/D3-B4.log" 2>&1
RC="$?"
echo "D3 v2 B4 qualification rc=$RC"
```

```bash
python -m specrhythm.phase4.draft_qualification_gate --allow-gpu \
  --config "$SR_PHASE4B_CONFIG" --expected-commit "$SR_PHASE4B_COMMIT" \
  --probe-root "$SR_PHASE4B3_ROOT/probe-evidence" --request-count 8 \
  --output "$SR_PHASE4B3_ROOT/D3-B8" >"$SR_PHASE4B3_ROOT/D3-B8.log" 2>&1
RC="$?"
echo "fresh uninstrumented D3 v2 B8 qualification rc=$RC"
```

Inspect `admission.json` and `gate.json`. A valid restart may report `same_physical_gpu=false` with `current_device_binding_valid=true` and `execution_regime_device_compatible=true`. The production replay starts only after admission. A prerequisite failure reports `execution_started=false`, `admission_valid=false`, `structural_checks_executed=false`, and null structural results. Actual replay failures remain blocking.

```bash
python -m specrhythm.phase4.draft_qualification aggregate \
  --root "$SR_PHASE4B3_ROOT" --output "$SR_PHASE4B3_ROOT/qualification.json"
RC="$?"
echo "bound D1-D3 aggregate rc=$RC"
```

## 5. Inspect the aggregate and package the new root

```bash
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['SR_PHASE4B3_ROOT'])
for name in ('D3-B2', 'D3-B4', 'D3-B8'):
    r = json.loads((root / name / 'gate.json').read_text())
    print(name, json.dumps({k: r.get(k) for k in (
        'd3_qualified', 'errors', 'execution_started', 'admission_valid',
        'structural_checks_executed', 'current_device_binding_valid',
        'historical_probe_gpu_uuid', 'current_gpu_uuid', 'same_physical_gpu',
        'execution_regime_device_compatible', 'hf_draft_exact')}, indent=2))
a = json.loads((root / 'qualification.json').read_text())
print('aggregate', json.dumps(a, indent=2))
assert a['valid'] and a['d3_qualified'], a['errors']
PY
RC="$?"
echo "D1-D3 aggregate inspection rc=$RC"
```

Stop here. D4/D5 may proceed only after the new aggregate passes; this runbook does not launch them. If any stage failed, preserve the failed evidence and use the packaging block below without proceeding.

```bash
tar -czf "$SR_PHASE4B3_ROOT.tar.gz" \
  -C "$(dirname "$SR_PHASE4B3_ROOT")" "$(basename "$SR_PHASE4B3_ROOT")"
RC="$?"
echo "package D1-D3 admission rerun rc=$RC"
echo "$SR_PHASE4B3_ROOT.tar.gz"
```

Return the complete archive, including current environment/topology, unchanged retained probe copies, admission and gate reports, backend cleanup evidence, logs and aggregate. No GPU run was performed by the agent.
