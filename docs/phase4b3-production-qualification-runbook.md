# Phase 4B.3 production Draft: fresh-server D3 → D4 → D5

For the current UUID admission correction, first use the [fresh-server D1–D3 rerunbook](phase4b3-device-admission-runbook.md) in a new root. That handoff stops at aggregate review. The complete progression reference below does not authorize skipping this rerun or automatically entering D4/D5.

Operator execution only. The agent has run no GPU. Use a fresh interactive **Bash** shell after restarting AutoDL. Execute each block separately. If its printed `rc` is nonzero, stop manually and preserve the root; the commands keep the interactive shell open. The runbook never applies a patch, changes precision/kernels, runs Dual, or starts D6.

The delivered copy replaces `@DELIVERED_COMMIT@` with the exact tested commit. The repository copy is a template because a commit cannot contain its own SHA. Existing inputs below are explicitly named, not inherited from previous shell exports. If a retained file was moved, change its path to that same artifact; do not substitute a different workload or reference.

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

GPU0 is Draft; GPUs1,2 are Target TP2. The existing helpers set each process's visibility. All modes retain BF16/eager, proposal budget 4, batch-invariant mode, the original measurement boundary, logging and five patches. The production authority is vLLM; the global backend selector default is unchanged and each A/B mode selects its backend explicitly.

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
export PHASE4B1_OVERLAP_REQUIREMENT=required
export PHASE4B_NATURAL_TEARDOWN_GRACE_SECONDS=5
unset MASTER_ADDR MASTER_PORT PYTHONPATH
unset SR_PHASE4_DRAFT_SOCKET SR_PHASE4_DUAL_DRAFT_SOCKET
unset PHASE4B1_NUMERICAL_PLAN PHASE4B1_NUMERICAL_OUTPUT
export SR_INPUT_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/eba0df493a7fd350ef3c8776e06d30e6196b6749/phase4b1-gate2-corrected5-20260827T040244Z"
export SR_PHASE4B3_WORKLOAD5="$SR_INPUT_ROOT/workloads/corrected-5.jsonl"
export SR_PHASE4B3_REFERENCE5="$SR_INPUT_ROOT/Gate-2-corrected-5/reference/stock-target-reference.json"
export SR_PHASE4B_WORKLOAD="$SR_INPUT_ROOT/workloads/corrected-100.jsonl"
export SR_PHASE4B_REFERENCE="$SR_INPUT_ROOT/Gate-3-corrected-100/reference/stock-target-reference.json"
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

The source probe is the operator's completed four-path experiment, not a new GPU run. Its full vectors are recomputed and verified offline. The old five-patch apply manifest is reused byte-for-byte. D1/D2/B2/B4 are rerun here for simple, complete provenance; compatible retained D1/D2 reports may instead be supplied to the aggregator after explicit provenance review. Old D3 v1 reports alone never admit D4.

```bash
export SR_PHASE4B3_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/$SR_PHASE4B_COMMIT/production-draft-$(date -u +%Y%m%dT%H%M%SZ)-$$"
export SR_PHASE4B_ENVIRONMENT="$SR_PHASE4B3_ROOT/environment.json"
export SR_PHASE4B_TOPOLOGY="$SR_PHASE4B3_ROOT/topology.json"
export SR_PHASE4B_PATCH_MANIFEST="$SR_PHASE4B3_ROOT/patch-stage/vllm-patch-stack.json"
python - <<'PY'
import hashlib, json, os, shutil
from pathlib import Path
from specrhythm.phase4.batched_draft_service import write_immutable_report
from specrhythm.phase4.draft_qualification import load_probe_qualification
names = ('SR_PHASE4B_CONFIG', 'SR_PHASE4B3_WORKLOAD5', 'SR_PHASE4B3_REFERENCE5',
         'SR_PHASE4B_WORKLOAD', 'SR_PHASE4B_REFERENCE')
inputs = {name: {'path': os.environ[name], 'sha256': hashlib.sha256(Path(os.environ[name]).read_bytes()).hexdigest()} for name in names}
for name, count in (('SR_PHASE4B3_WORKLOAD5', 5), ('SR_PHASE4B_WORKLOAD', 100)):
    assert len([line for line in Path(os.environ[name]).read_text().splitlines() if line.strip()]) == count
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

```bash
source integrations/vllm/phase4b_run_helpers.sh &&
  source integrations/vllm/phase4b1_gate_helpers.sh &&
  source integrations/vllm/phase4b2_run_helpers.sh &&
  source integrations/vllm/phase4b3_qualification_helpers.sh
RC="$?"
echo "load checked D3-D5 helpers rc=$RC"
```

## 4. D1/D2 and fresh D3 B2/B4/B8

D1/D2 retain their v1 semantics. D3 uses a distinct v2 entry point; **do not add `--diagnostic`**. No materialization/logits observer is installed. Ordinary API-boundary host-state validation remains mandatory. D3 is a semantic qualification, not a throughput benchmark; D5 supplies the serving performance measurement.

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

HF difference alone may now coexist with `d3_qualified=true`, but structural errors, unmatched numerical provenance, either vLLM-internal vector inequality, incomplete batching or cleanup remain fatal. Do not assume B8 passes before inspecting the new run.

```bash
python -m specrhythm.phase4.draft_qualification aggregate \
  --root "$SR_PHASE4B3_ROOT" --output "$SR_PHASE4B3_ROOT/qualification.json"
RC="$?"
echo "bound D1-D3 aggregate rc=$RC"
```

## 5. D4 corrected-five pair and validator

Each wrapper invocation independently rechecks the aggregate, model weights, config and installed sources **before** starting a service. The HF mode enables only nonblocking forward events and in-memory counters; it adds no measured-path fence. Both modes retain existing lifecycle supervision and offline performance-boundary validation. Each mode gets its own fresh directory.

```bash
unset RANK LOCAL_RANK WORLD_SIZE
phase4b3_run_serial D4 hf
RC="$?"
echo "D4 HF Serial execution/cleanup/measurement rc=$RC"
```

```bash
phase4b3_run_serial D4 vllm
RC="$?"
echo "D4 production vLLM Serial execution/cleanup/measurement rc=$RC"
```

```bash
phase4b3_compare_serial D4
RC="$?"
echo "D4 structural/work/Target verification qualification rc=$RC"
```

Require `stage_qualified=true`, `d4_qualified=true`, an empty error list and actual proposal batching. HF differences remain in `hf_exact_diagnostic` and the explicit divergence fields. Five-request timing does not establish the performance milestone.

## 6. D5 corrected-100 fresh pair and performance/work comparison

D5 wrappers recompute the bound D4 comparison as well as D1-D3 qualification before each execution. A stale, missing or failed D4 report prevents launch. This is authorized only after the preceding checks pass.

```bash
phase4b3_run_serial D5 hf
RC="$?"
echo "D5 fresh HF Serial execution/cleanup/measurement rc=$RC"
```

```bash
phase4b3_run_serial D5 vllm
RC="$?"
echo "D5 fresh production vLLM Serial execution/cleanup/measurement rc=$RC"
```

```bash
phase4b3_compare_serial D5
RC="$?"
echo "D5 production end-to-end improvement and work comparison rc=$RC"
```

```bash
python - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ['SR_PHASE4B3_ROOT']) / 'D5/comparison.json'
r = json.loads(p.read_text())
print(json.dumps({k: r[k] for k in ('stage_qualified', 'errors', 'hf_draft_exact',
    'hf_vllm_divergent_request_count', 'production_performance_label',
    'execution_time_comparison', 'work_accounting', 'work_deltas',
    'proposal_acceptance_target_work_nearly_matched', 'nearly_matched_rule')}, indent=2))
PY
RC="$?"
echo "D5 complete performance/work summary rc=$RC"
```

The result is a **production vLLM Batched Draft end-to-end improvement**, not automatically a pure batching speedup. Review the actual fresh HF baseline, makespan/throughput/TPOT, proposal and acceptance differences, Target work, Draft forward/batch/event/sync counters, and JIT/warmup evidence together. The historical ≈49.3 s / 30.2 tok/s is context, not a substitute for this fresh HF measurement. No numerical improvement threshold automatically authorizes D6.

## 7. Package on success or failure

After any failed stage, retain its root and run this block instead of proceeding. The package includes immutable probe copies, D3 observations/qualification, all admissions, mode logs, lifecycle records, raw evidence and D4/D5 comparisons that were actually produced. Original artifacts remain untouched.

```bash
tar -czf "$SR_PHASE4B3_ROOT.tar.gz" \
  -C "$(dirname "$SR_PHASE4B3_ROOT")" "$(basename "$SR_PHASE4B3_ROOT")"
RC="$?"
echo "package D3-D5 artifact root rc=$RC"
echo "$SR_PHASE4B3_ROOT.tar.gz"
```

Return the archive for review. Do not integrate Dual, tune batching/scheduling/UUID/eager settings, or start D6.
