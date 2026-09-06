# Completed D4: offline comparison, then fresh-server D5

D3 and both D4 GPU executions are complete. Run no D1–D4 generation commands.
The agent performs CPU work only. D5 GPU commands below are for the operator,
and are admitted only after the existing D4 comparison qualifies offline.

Use a fresh interactive Bash shell; execute blocks separately and stop manually
on a nonzero printed `rc`. No block closes the shell or changes vLLM patches.
The delivered copy substitutes the exact tested SHA for `@DELIVERED_COMMIT@`.
Set the retained-root path to the actual completed D3/D4 root; it is not guessed.

## 1. Fresh-server installation (also sufficient for server-side CPU comparison)


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

## 2. Offline D4 only

This step needs no CUDA runtime, model load, new measurement or server allocation.
On the local CPU checkout, use `.venv/bin/python` in place of `python` and set the
retained root to the extracted archive directory. The directory must contain
`qualification.json`, `D4/hf-admission.json`, `D4/vllm-admission.json`, and both
complete `D4/hf` and `D4/vllm` artifact directories. D3 raw vectors are not needed.

```bash
export SR_PHASE4B3_RETAINED_ROOT="/absolute/path/to/completed-D3-D4-root"
export SR_PHASE4B3_OFFLINE_OUTPUT="$SR_PHASE4B3_RETAINED_ROOT/D4/comparison.offline-@DELIVERED_COMMIT@.json"
CUDA_VISIBLE_DEVICES="" python -m specrhythm.phase4.draft_comparison \
  --qualification "$SR_PHASE4B3_RETAINED_ROOT/qualification.json" \
  --stage D4 --request-count 5 \
  --hf "$SR_PHASE4B3_RETAINED_ROOT/D4/hf" \
  --vllm "$SR_PHASE4B3_RETAINED_ROOT/D4/vllm" \
  --output "$SR_PHASE4B3_OFFLINE_OUTPUT"
RC="$?"
echo "offline D4 comparison rc=$RC"
```

Inspect the recomputed result. Preserve the previous comparison by content hash
before installing the qualifying result as `D4/comparison.json`. All execution
artifacts remain byte-for-byte unchanged. A failed comparison remains available
at the offline output path and does not admit D5.

```bash
python - <<'PYCODE'
import hashlib, json, os
from pathlib import Path
from specrhythm.phase4.batched_draft_service import write_immutable_report
root = Path(os.environ['SR_PHASE4B3_RETAINED_ROOT'])
source = Path(os.environ['SR_PHASE4B3_OFFLINE_OUTPUT'])
value = json.loads(source.read_text())
print(json.dumps({k: value[k] for k in ('d4_qualified', 'blocking_errors', 'diagnostics', 'work_accounting', 'execution_time_comparison')}, indent=2))
assert value['d4_qualified'] and not value['blocking_errors'], 'D4 does not qualify'
destination = root / 'D4/comparison.json'
if destination.exists():
    original = destination.read_bytes()
    backup = destination.with_name('comparison.previous-' + hashlib.sha256(original).hexdigest() + '.json')
    if backup.exists():
        assert backup.read_bytes() == original
    else:
        with backup.open('xb') as stream:
            stream.write(original)
replacement = destination.with_name('comparison.install.json')
write_immutable_report(replacement, value)
replacement.replace(destination)
print('D4 qualified offline; D5 is allowed; no GPU was run by this step')
PYCODE
RC="$?"
echo "preserve and install offline D4 rc=$RC"
```

## 3. Explicit D5 environment and inputs

If the offline comparison was performed locally, copy the corrected certificate
and original qualification to the retained server root before continuing.
`SR_PHASE4B3_RETAINED_ROOT` must now refer to that server directory.
GPU0 remains Draft TP1; GPUs1,2 remain Target TP2. Runtime defaults, BF16/eager,
proposal budget, measurement boundary and cleanup behavior are unchanged.

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
SR_VLLM_ROOT="$(python - <<'PY'
from importlib import metadata
print(metadata.distribution('vllm').locate_file(''))
PY
)"
RC="$?"
export SR_VLLM_ROOT
echo "locate installed vLLM rc=$RC"
```

Create a fresh D5 root and copy only the completed prerequisite certificates and
unchanged five-patch manifest. No D3 probe or D4 execution is repeated.

```bash
export SR_PHASE4B3_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/$SR_PHASE4B_COMMIT/production-draft-D5-$(date -u +%Y%m%dT%H%M%SZ)-$$"
export SR_PHASE4B_ENVIRONMENT="$SR_PHASE4B3_ROOT/environment.json"
export SR_PHASE4B_TOPOLOGY="$SR_PHASE4B3_ROOT/topology.json"
export SR_PHASE4B_PATCH_MANIFEST="$SR_PHASE4B3_ROOT/patch-stage/vllm-patch-stack.json"
unset SR_PHASE4B3_QUALIFICATION SR_PHASE4B3_D4_COMPARISON
python - <<'PYCODE'
import os, shutil
from pathlib import Path
from specrhythm.phase4.draft_performance_policy import require_performance_stage
prior = Path(os.environ['SR_PHASE4B3_RETAINED_ROOT'])
require_performance_stage(prior / 'qualification.json', 'D5', prior / 'D4/comparison.json')
root = Path(os.environ['SR_PHASE4B3_ROOT'])
root.mkdir(parents=True, exist_ok=False)
(root / 'D4').mkdir()
(root / 'patch-stage').mkdir()
shutil.copyfile(prior / 'qualification.json', root / 'qualification.json')
shutil.copyfile(prior / 'D4/comparison.json', root / 'D4/comparison.json')
manifest = Path(os.environ['SR_PHASE4B3_BASELINE_ROOT']) / 'patch-stage/vllm-patch-stack.json'
shutil.copyfile(manifest, os.environ['SR_PHASE4B_PATCH_MANIFEST'])
print(root)
PYCODE
RC="$?"
echo "fresh D5 root and completed certificates rc=$RC"
```

```bash
python integrations/vllm/manage_patch.py check \
  --vllm-root "$SR_VLLM_ROOT" --source "$SR_VLLM_SOURCE" \
  --expect-state patched --manifest "$SR_PHASE4B3_ROOT/installed-patch-check.json" &&
env -u CUDA_VISIBLE_DEVICES VLLM_BATCH_INVARIANT=1 specrhythm phase4-probe \
  --config "$SR_PHASE4B_CONFIG" --vllm-source "$SR_VLLM_SOURCE" \
  --environment-output "$SR_PHASE4B_ENVIRONMENT" \
  --topology-output "$SR_PHASE4B_TOPOLOGY" \
  --validation-output "$SR_PHASE4B3_ROOT/probe-validation.json"
RC="$?"
echo "current metadata and unchanged patch check rc=$RC"
```

```bash
source integrations/vllm/phase4b_run_helpers.sh &&
  source integrations/vllm/phase4b1_gate_helpers.sh &&
  source integrations/vllm/phase4b2_run_helpers.sh &&
  source integrations/vllm/phase4b3_qualification_helpers.sh
RC="$?"
echo "load D5 helpers rc=$RC"
```

## 4. Operator-only D5 GPU runs

Run each backend once with the corrected-100 workload. The CPU admission reads
the workload contracts and existing D3/D4 certificates. It does not require the
historical D3 execution commit to equal this comparator commit.

```bash
phase4b3_run_serial D5 hf
RC="$?"
echo "D5 HF execution and measurement rc=$RC"
```

```bash
phase4b3_run_serial D5 vllm
RC="$?"
echo "D5 vLLM execution and measurement rc=$RC"
```

```bash
phase4b3_compare_serial D5
RC="$?"
echo "D5 offline comparison rc=$RC"
```

## 5. Concise report and artifact handoff

```bash
python - <<'PYCODE'
import json, os
from pathlib import Path
r = json.loads((Path(os.environ['SR_PHASE4B3_ROOT']) / 'D5/comparison.json').read_text())
print(r['production_performance_label'])
print(json.dumps({k: r[k] for k in ('d5_qualified', 'blocking_errors', 'execution_time_comparison', 'proposal_acceptance_target_work_exactly_matched', 'pure_batching_speedup_claim')}, indent=2))
fields = ('completed_requests', 'measured_committed_tokens', 'decode_makespan_ms', 'throughput_tokens_per_second', 'tpot_ms', 'draft_model_forward_count', 'draft_batch_size', 'draft_gpu_event_time_ms', 'accepted_draft_tokens', 'proposed_tokens', 'rejected_draft_tokens', 'target_forward_count', 'target_query_tokens')
for mode, work in r['work_accounting'].items():
    print(mode, json.dumps({k: work[k] for k in fields}, indent=2))
PYCODE
RC="$?"
echo "D5 summary rc=$RC"
```

```bash
tar -czf "$SR_PHASE4B3_ROOT.tar.gz" -C "$(dirname "$SR_PHASE4B3_ROOT")" "$(basename "$SR_PHASE4B3_ROOT")"
RC="$?"
echo "artifact package rc=$RC path=$SR_PHASE4B3_ROOT.tar.gz"
```

Use the label **production vLLM Batched Draft end-to-end improvement**. Quantify
work differences explicitly; matched work alone does not isolate batching from
the backend change. No pure-batching claim or D6 authorization is emitted.
