# Phase 4B.4A — fixed Dual microbatch fresh-server runbook

The agent has not run GPU. These commands are for the operator on AutoDL.
Use the delivered copy with `@DELIVERED_COMMIT@` replaced by its exact tested SHA.
Run each Bash block separately in an interactive **Bash** shell. Every block
prints `rc`; stop manually on any nonzero value and retain its artifacts.
Do not retry into an existing root or continue to the next stage after failure.

Use one fresh server and one commit for Target, Serial, and all seven Dual cells.
Do not restart or change code/config/environment between cells. Target and Serial
run once. Historical D6 numbers are context only, never the sweep controls.

## 1. Safe shell and conda

```bash
set +e
set +u
set +o pipefail
trap - ERR
export SR_PHASE4B_COMMIT="@DELIVERED_COMMIT@"
export SR_PHASE4B3_REPO="/root/autodl-tmp/src/SpecRhythm"
SR_4B4_CONDA_BASE=""
if command -v conda >/dev/null 2>&1; then
  SR_4B4_CONDA_BASE="$(conda info --base)"
elif test -x /root/miniconda3/bin/conda; then
  SR_4B4_CONDA_BASE="/root/miniconda3"
elif test -x /root/miniconda/bin/conda; then
  SR_4B4_CONDA_BASE="/root/miniconda"
fi
if test -n "$SR_4B4_CONDA_BASE" && test -f "$SR_4B4_CONDA_BASE/etc/profile.d/conda.sh"; then
  source "$SR_4B4_CONDA_BASE/etc/profile.d/conda.sh" &&
    conda activate /root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1
else
  echo "Conda bootstrap missing; stop and inspect the installation."
  false
fi
RC="$?"
echo "conda activation rc=$RC"
```

## 2. Exact checkout and editable install

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

## 3. Full environment and frozen input paths

The existing five patches must already be installed in the persistent vLLM
environment. Section 5 checks them and stops on a mismatch; it never adds or
applies a patch. The pinned vLLM source is
`752a3a504485790a2e8491cacbb35c137339ad34`.

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
export PHASE4B1_OVERLAP_REQUIREMENT=characterization
unset SR_PHASE4B_DUAL_MICROBATCH_SIZE SR_PHASE4_DUAL_MICROBATCH_SIZE
unset SR_PHASE4_DUAL_TEST_COORDINATION
export PHASE4B_NATURAL_TEARDOWN_GRACE_SECONDS=5
unset MASTER_ADDR MASTER_PORT PYTHONPATH
unset SR_PHASE4_DRAFT_SOCKET SR_PHASE4_DUAL_DRAFT_SOCKET
unset PHASE4B1_NUMERICAL_PLAN PHASE4B1_NUMERICAL_OUTPUT
export SR_INPUT_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/eba0df493a7fd350ef3c8776e06d30e6196b6749/phase4b1-gate2-corrected5-20260827T040244Z"
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

GPU0 remains Draft TP1/world1/MRV1/BF16/eager with one persistent paged-KV model.
GPUs1,2 remain Target TP2/BF16/eager. Proposal K=4, UUID `live`, batch
invariance, measurement boundary, logging and cleanup remain unchanged.
`SR_PHASE4B_DUAL_MICROBATCH_SIZE` is a helper-only positive-integer control,
default 2. The explicit Dual cell argument sets it for that invocation; Target
and Serial ignore it. N is an upper bound, with no wait to fill a batch.

## 4. New immutable session root

Only a new sweep directory and an unchanged patch manifest copy are created.

```bash
export SR_PHASE4B4_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/$SR_PHASE4B_COMMIT/phase4b4-microbatch-sweep-$(date -u +%Y%m%dT%H%M%SZ)-$$"
export SR_PHASE4B_ENVIRONMENT="$SR_PHASE4B4_ROOT/environment.json"
export SR_PHASE4B_TOPOLOGY="$SR_PHASE4B4_ROOT/topology.json"
export SR_PHASE4B_PATCH_MANIFEST="$SR_PHASE4B4_ROOT/patch-stage/vllm-patch-stack.json"
python - <<'PYCODE'
import os, shutil
from pathlib import Path
from specrhythm.phase4.manifest import sha256_file
root = Path(os.environ['SR_PHASE4B4_ROOT'])
manifest = Path(os.environ['SR_PHASE4B3_BASELINE_ROOT']) / 'patch-stage/vllm-patch-stack.json'
assert sha256_file(manifest) == '33c5a75f95a490695d966548d6f638bed810a410ff0d028ee114bfbe854069fc'
for name in ('SR_PHASE4B_WORKLOAD', 'SR_PHASE4B_REFERENCE'):
    assert Path(os.environ[name]).is_file(), name
root.mkdir(parents=True, exist_ok=False)
(root / 'patch-stage').mkdir()
shutil.copyfile(manifest, os.environ['SR_PHASE4B_PATCH_MANIFEST'])
print(root)
PYCODE
RC="$?"
echo "fresh root and unchanged inputs rc=$RC"
```

## 5. Five-patch check and current environment/topology capture

This operator-side probe queries current GPU metadata; it does not generate
tokens. It runs once per new server, not against historical UUIDs.

```bash
python integrations/vllm/manage_patch.py check \
  --vllm-root "$SR_VLLM_ROOT" --source "$SR_VLLM_SOURCE" \
  --expect-state patched --manifest "$SR_PHASE4B4_ROOT/installed-patch-check.json" &&
env -u CUDA_VISIBLE_DEVICES VLLM_BATCH_INVARIANT=1 specrhythm phase4-probe \
  --config "$SR_PHASE4B_CONFIG" --vllm-source "$SR_VLLM_SOURCE" \
  --environment-output "$SR_PHASE4B_ENVIRONMENT" \
  --topology-output "$SR_PHASE4B_TOPOLOGY" \
  --validation-output "$SR_PHASE4B4_ROOT/probe-validation.json"
RC="$?"
echo "five-patch and current server probe rc=$RC"
```

## 6. One experiment preflight

This CPU step records the commit, models, config, workloads, vLLM and current
server evidence once. It does not load a model, rerun D1–D5, or compare numerical
vectors. Each runtime still enforces its actual device binding and structural
verification contract.

```bash
python - <<'PYCODE'
import json, os, subprocess
from pathlib import Path
from specrhythm.phase4.batched_draft_service import write_immutable_report
from specrhythm.phase4.config import load_phase4_config
from specrhythm.phase4.manifest import model_revision_manifest, sha256_file
from specrhythm.phase4.stock_vllm import load_smoke_requests
root = Path(os.environ['SR_PHASE4B4_ROOT'])
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert commit == os.environ['SR_PHASE4B_COMMIT']
assert json.loads((root / 'probe-validation.json').read_text())['valid'] is True
config = load_phase4_config(os.environ['SR_PHASE4B_CONFIG'])
assert config.proposal_budget == 4 and config.enforce_eager
assert config.draft.tensor_parallel_size == 1 and config.draft.physical_gpu_ids == (0,)
assert config.target.tensor_parallel_size == 2 and config.target.physical_gpu_ids == (1, 2)
assert config.draft.dtype == config.target.dtype == 'bfloat16'
inputs = {}
for key, count in (('SR_PHASE4B_WORKLOAD', 100),):
    path = Path(os.environ[key])
    rows = load_smoke_requests(path, count, require_task_mixture=count in (5, 100))
    assert len(rows) == count and all(r.maximum_new_tokens == 16 for r in rows)
    inputs[key] = {'path': str(path), 'sha256': sha256_file(path), 'request_count': count}
models = {role: model_revision_manifest(engine.resolved_model_path, engine.revision)
          for role, engine in (('draft', config.draft), ('target', config.target))}
write_immutable_report(root / 'preflight.json', {
    'valid': True, 'errors': [], 'execution_git_commit': commit, 'models': models,
    'config_sha256': sha256_file(Path(os.environ['SR_PHASE4B_CONFIG'])),
    'vllm_commit': config.expected_vllm_commit, 'inputs': inputs,
    'evidence_sha256': {name: sha256_file(root / name) for name in
        ('environment.json', 'topology.json', 'installed-patch-check.json', 'patch-stage/vllm-patch-stack.json')},
    'draft_backend': 'vllm-batched', 'dual_uuid_query_mode': 'live',
    'expected_measured_committed_tokens': 1487, 'microbatch_sweep': [2, 4, 8, 16, 32, 64, 100],
    'target_only_scope': 'existing DecodeReady Draft setup; no measured Draft proposals',
})
print('Preflight valid; no generation performed')
PYCODE
RC="$?"
echo "one experiment preflight rc=$RC"
```

## 7. Load existing runners and the sweep wrapper

```bash
source integrations/vllm/phase4b_run_helpers.sh &&
  source integrations/vllm/phase4b1_gate_helpers.sh &&
  source integrations/vllm/phase4b2_run_helpers.sh &&
  source integrations/vllm/phase4b4_microbatch_helpers.sh
RC="$?"
echo "load sweep helpers rc=$RC"
```

## 8. Target-only control — once

Retain the established pre-measurement HF Draft DecodeReady setup. Measured
Target execution has no Draft proposals. No Target implementation is changed.

```bash
phase4b4_run target
RC="$?"
echo "Target baseline rc=$RC; stop manually if nonzero"
```

## 9. Serial-vLLM control — once

```bash
phase4b4_run serial
RC="$?"
echo "Serial baseline rc=$RC; stop manually if nonzero"
```

## 10. Dual-vLLM microbatch=2

```bash
phase4b4_run dual 2
RC="$?"
echo "Dual mb2 rc=$RC; stop manually if nonzero"
```

## 11. Dual-vLLM microbatch=4

```bash
phase4b4_run dual 4
RC="$?"
echo "Dual mb4 rc=$RC; stop manually if nonzero"
```

## 12. Dual-vLLM microbatch=8

```bash
phase4b4_run dual 8
RC="$?"
echo "Dual mb8 rc=$RC; stop manually if nonzero"
```

## 13. Dual-vLLM microbatch=16

```bash
phase4b4_run dual 16
RC="$?"
echo "Dual mb16 rc=$RC; stop manually if nonzero"
```

## 14. Dual-vLLM microbatch=32

```bash
phase4b4_run dual 32
RC="$?"
echo "Dual mb32 rc=$RC; stop manually if nonzero"
```

## 15. Dual-vLLM microbatch=64

```bash
phase4b4_run dual 64
RC="$?"
echo "Dual mb64 rc=$RC; stop manually if nonzero"
```

## 16. Dual-vLLM microbatch=100

```bash
phase4b4_run dual 100
RC="$?"
echo "Dual mb100 rc=$RC; stop manually if nonzero"
```

N=100 is the same existing upper bound: schedule currently eligible requests
without waiting to fill 100 or changing admission. Actual batches can be smaller.
The generic positive-integer runtime also accepts 101, but it is outside this
predefined seven-cell sweep. Endpoint observations use mb100 relative to mb2;
2 and 100 are endpoints when identifying an intermediate peak.

Each cell must return rc=0 and `qualification.json: valid=true`. Each completes
100 requests and 1487 measured committed tokens; raw Target verification,
acceptance/accounting, TP identity/consensus and Draft KV/process cleanup must be
valid. Requested and scheduler-loaded effective bounds must agree. Actual
batches can be smaller. Poor throughput and valid zero overlap do not fail a
cell. Default D6 overlap qualification remains unchanged outside this runbook.

## 17. Offline sweep report and concise Markdown table

```bash
phase4b4_compare
RC="$?"
echo "offline sweep report rc=$RC; stop manually if nonzero"
```

```bash
cat "$SR_PHASE4B4_ROOT/sweep.md"
RC="$?"
echo "read concise sweep table rc=$RC"
```

Columns are **2, 4, 8, 16, 32, 64, 100**. Rows are throughput, makespan, TPOT mean,
Target forwards, Draft forwards, Draft batch p50/p90, verification batch p50/p90,
Draft GPU event time, overlap ms, accepted length, Dual/Target throughput and
Dual/Serial throughput. Same-session control metrics follow the table.

`sweep.json` retains all per-cell metrics: completed requests, measured tokens,
TPOT mean/p50/p90, Target forwards/query tokens, verification batch
count/min/p10/p50/p90/max/mean/histogram/singletons, Draft model/proposal/commit
forwards, batch min/p10/p50/p90/max/mean and per-purpose distributions,
GPU event time, host sync count/reasons, proposed/accepted/rejected tokens,
proposal/correction/bonus counts, accepted length, both microbatch bounds,
physical-overlap existence and structural validity, actual intersection-union
overlap ms, ready/commit-and-propose cohort distributions, fragmentation and
stock scheduler constraints. `dual-microbatch.json` binds the common bound to
all retained JSON/JSONL runtime artifacts, including unchanged Draft reports.

Warnings about substantially different mb2 structure appear **before** the table.
The diagnostic heuristic checks p50=2 and forward counts within 0.5–2x the D6
reference; it never invalidates a cell. No historical timing equality is required.
Compare Draft/verify batch growth, forward/GPU-time reductions, overlap loss and
throughput recovery across the curve. The report identifies the best observed
cell and whether it is intermediate, without declaring an optimal policy.
Interpret quantitative work differences before making causal claims. If large N
still trails Serial, inspect remaining Dual overhead before designing a scheduler.

Label: **production vLLM Batched Draft end-to-end improvement**. No pure batching
claim without quantitatively matched work. Observed overlap is **not**
critical-path time saved. The current scope is a saturated corrected-100 burst
with short outputs (maximum_new_tokens=16), without online arrivals, TTFT or SLO
claims. No MineDraft, adaptive rhythm or next mechanism is implemented.

## 18. Package every retained cell, including any failure

```bash
if test ! -e "$SR_PHASE4B4_ROOT.tar.gz"; then
  tar -czf "$SR_PHASE4B4_ROOT.tar.gz" -C "$(dirname "$SR_PHASE4B4_ROOT")" "$(basename "$SR_PHASE4B4_ROOT")"
else
  echo "Package already exists; retain it and choose a fresh archive name."
  false
fi
RC="$?"
echo "artifact package rc=$RC path=$SR_PHASE4B4_ROOT.tar.gz"
```

On any structural failure, stop the sweep manually and run only the packaging
block. Never overwrite/retry a cell or rewrite D6 artifacts. Return the archive
with logs and the first failing result. After successful CPU/CI delivery, only
the operator executes this runbook; the agent does not connect to AutoDL or run GPU.
