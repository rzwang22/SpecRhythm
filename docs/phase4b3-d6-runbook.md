# Phase 4B.3 D6 — fresh-server operator runbook

The agent has not run GPU. These commands are for the operator on AutoDL.
Use the delivered copy with `@DELIVERED_COMMIT@` replaced by its exact tested SHA.
Run each Bash block separately in an interactive **Bash** shell. Every block
prints `rc`; stop manually on any nonzero value and retain its artifacts.
Do not retry into an existing root or continue to the next stage after failure.

There are two sessions: D6-A/B bring-up, then a fresh server for D6-C.
On **each** server/session, execute sections 1–7 from an empty shell. They define
all exports anew. For C, import the completed B certificate using section 11;
do not repeat A/B generation or use previous Target/Serial timings.

## 1. Safe shell and conda

```bash
set +e
set +u
set +o pipefail
trap - ERR
export SR_PHASE4B_COMMIT="@DELIVERED_COMMIT@"
export SR_PHASE4B3_REPO="/root/autodl-tmp/src/SpecRhythm"
SR_D6_CONDA_BASE=""
if command -v conda >/dev/null 2>&1; then
  SR_D6_CONDA_BASE="$(conda info --base)"
elif test -x /root/miniconda3/bin/conda; then
  SR_D6_CONDA_BASE="/root/miniconda3"
elif test -x /root/miniconda/bin/conda; then
  SR_D6_CONDA_BASE="/root/miniconda"
fi
if test -n "$SR_D6_CONDA_BASE" && test -f "$SR_D6_CONDA_BASE/etc/profile.d/conda.sh"; then
  source "$SR_D6_CONDA_BASE/etc/profile.d/conda.sh" &&
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

GPU0 remains Draft TP1/world1/MRV1/BF16/eager with one persistent paged-KV model.
GPUs1,2 remain Target TP2/BF16/eager. Existing helper microbatch=2, proposal K=4,
candidate policy, measurement boundary, UUID `live`, logging and cleanup remain
unchanged. No test-only readiness coordination or numerical observer is enabled.

## 4. New immutable D6 session root

This writes only a new D6 directory. It copies the immutable patch manifest and
the first two complete corrected-five rows for the smoke; it does not shorten
their output limits or modify any historical artifact.

```bash
export SR_D6_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/$SR_PHASE4B_COMMIT/production-draft-D6-$(date -u +%Y%m%dT%H%M%SZ)-$$"
export SR_PHASE4B_ENVIRONMENT="$SR_D6_ROOT/environment.json"
export SR_PHASE4B_TOPOLOGY="$SR_D6_ROOT/topology.json"
export SR_PHASE4B_PATCH_MANIFEST="$SR_D6_ROOT/patch-stage/vllm-patch-stack.json"
export SR_D6_SMOKE_WORKLOAD="$SR_D6_ROOT/workloads/smoke-2.jsonl"
python - <<'PYCODE'
import os, shutil
from pathlib import Path
from specrhythm.phase4.manifest import sha256_file
root = Path(os.environ['SR_D6_ROOT'])
manifest = Path(os.environ['SR_PHASE4B3_BASELINE_ROOT']) / 'patch-stage/vllm-patch-stack.json'
assert sha256_file(manifest) == '33c5a75f95a490695d966548d6f638bed810a410ff0d028ee114bfbe854069fc'
for name in ('SR_PHASE4B3_WORKLOAD5', 'SR_PHASE4B3_REFERENCE5', 'SR_PHASE4B_WORKLOAD', 'SR_PHASE4B_REFERENCE'):
    assert Path(os.environ[name]).is_file(), name
root.mkdir(parents=True, exist_ok=False)
(root / 'patch-stage').mkdir()
(root / 'workloads').mkdir()
shutil.copyfile(manifest, os.environ['SR_PHASE4B_PATCH_MANIFEST'])
rows = Path(os.environ['SR_PHASE4B3_WORKLOAD5']).read_text().splitlines()
assert len(rows) == 5
Path(os.environ['SR_D6_SMOKE_WORKLOAD']).write_text('\n'.join(rows[:2]) + '\n')
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
  --expect-state patched --manifest "$SR_D6_ROOT/installed-patch-check.json" &&
env -u CUDA_VISIBLE_DEVICES VLLM_BATCH_INVARIANT=1 specrhythm phase4-probe \
  --config "$SR_PHASE4B_CONFIG" --vllm-source "$SR_VLLM_SOURCE" \
  --environment-output "$SR_PHASE4B_ENVIRONMENT" \
  --topology-output "$SR_PHASE4B_TOPOLOGY" \
  --validation-output "$SR_D6_ROOT/probe-validation.json"
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
root = Path(os.environ['SR_D6_ROOT'])
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert commit == os.environ['SR_PHASE4B_COMMIT']
assert json.loads((root / 'probe-validation.json').read_text())['valid'] is True
config = load_phase4_config(os.environ['SR_PHASE4B_CONFIG'])
assert config.proposal_budget == 4 and config.enforce_eager
assert config.draft.tensor_parallel_size == 1 and config.draft.physical_gpu_ids == (0,)
assert config.target.tensor_parallel_size == 2 and config.target.physical_gpu_ids == (1, 2)
assert config.draft.dtype == config.target.dtype == 'bfloat16'
inputs = {}
for key, count in (('SR_PHASE4B3_WORKLOAD5', 5), ('SR_PHASE4B_WORKLOAD', 100), ('SR_D6_SMOKE_WORKLOAD', 2)):
    path = Path(os.environ[key])
    rows = load_smoke_requests(path, count, require_task_mixture=count in (5, 100))
    assert len(rows) == count
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
    'target_only_scope': 'existing DecodeReady Draft setup; no measured Draft proposals',
})
print('Preflight valid; no generation performed')
PYCODE
RC="$?"
echo "one experiment preflight rc=$RC"
```

## 7. Load existing runners and the D6 stage wrapper

```bash
source integrations/vllm/phase4b_run_helpers.sh &&
  source integrations/vllm/phase4b1_gate_helpers.sh &&
  source integrations/vllm/phase4b2_run_helpers.sh &&
  source integrations/vllm/phase4b3_d6_helpers.sh
RC="$?"
echo "load D6 helpers rc=$RC"
```

## 8. D6-A: two-request production Dual smoke

```bash
phase4b3_d6_run A dual
RC="$?"
echo "D6-A Dual smoke rc=$RC"
```

Success: production service ready, real multi-request proposal forwards,
Target proposals verified/committed, two completed requests, both private Draft
KV handles retired, allocator/live-request counts zero, native process cleanup
valid. `qualification.json` must have `valid=true` and
`performance_result=false`. Physical overlap is not a smoke prerequisite.
If it fails, stop at the first actual failure and package this root.

## 9. D6-B: unchanged corrected-five production Dual

```bash
phase4b3_d6_run B dual
RC="$?"
echo "D6-B corrected-five Dual execution and concise validation rc=$RC"
```

Success: five requests complete with valid tokens/output limits; Target
mapping/causality/TP consensus and request lifecycle remain valid; all Draft
requests retire and shutdown is natural; at least one actual proposal forward
has multiple requests; physical overlap is valid; metrics are finite/positive.
The command prints its concise metrics and writes `D6-B/dual/qualification.json`.
JIT warnings, qualified numerical differences and final-sequence differences
do not independently block performance interpretation. A material failure stops
C; preserve the failed root instead of emitting downstream missing-check errors.

## 10. Save the bring-up result before restarting

```bash
phase4b3_d6_require "$SR_D6_ROOT/D6-B/dual/qualification.json" &&
  tar -czf "$SR_D6_ROOT.tar.gz" -C "$(dirname "$SR_D6_ROOT")" "$(basename "$SR_D6_ROOT")"
RC="$?"
echo "D6-A/B package rc=$RC path=$SR_D6_ROOT.tar.gz"
echo "Record this completed B root before restart: $SR_D6_ROOT"
```

## 11. Fresh server for C: repeat sections 1–7, then import B qualification

Restart AutoDL with the persistent environment/models/inputs available. Open a
new Bash shell and run **all** of sections 1–7. This creates a new session root,
new current-server probe and preflight under the same exact final SHA. No export
from the previous shell is assumed. Set the path below to the recorded completed
B root. Only its qualification certificate is copied; no timing is reused.

```bash
export SR_D6_COMPLETED_B_ROOT="/absolute/path/to/recorded-completed-D6-B-root"
export SR_D6_C_ROOT="$SR_D6_ROOT/D6-C-corrected-100"
python - <<'PYCODE'
import json, os, shutil
from pathlib import Path
root = Path(os.environ['SR_D6_ROOT'])
source = Path(os.environ['SR_D6_COMPLETED_B_ROOT']) / 'D6-B/dual/qualification.json'
report = json.loads(source.read_text())
assert report['valid'] is True and not report.get('errors') and report['performance_result'] is True
assert report['metrics']['completed_requests'] == 5
assert report['experiment_identity']['execution_git_commit'] == os.environ['SR_PHASE4B_COMMIT']
destination = root / 'D6-B/dual'
destination.mkdir(parents=True, exist_ok=False)
shutil.copyfile(source, destination / 'qualification.json')
Path(os.environ['SR_D6_C_ROOT']).mkdir(parents=True, exist_ok=False)
print('Fresh corrected-100 root:', os.environ['SR_D6_C_ROOT'])
PYCODE
RC="$?"
echo "completed B certificate and fresh C root rc=$RC"
```

## 12. C Target-only, then Serial-vLLM, then Dual-vLLM

Run these three modes on this same server without editing code, changing config,
changing input limits, or restarting between modes. The wrapper checks the
preceding mode before launching the next. Target-only retains the established
pre-measurement HF Draft DecodeReady setup; measured execution has no Draft
proposals. No Target source or setup contract is changed by D6.

```bash
phase4b3_d6_run C target
RC="$?"
echo "D6-C Target-only rc=$RC"
```

```bash
phase4b3_d6_run C serial
RC="$?"
echo "D6-C production Serial-vLLM rc=$RC"
```

```bash
phase4b3_d6_run C dual
RC="$?"
echo "D6-C production Dual-vLLM rc=$RC"
```

## 13. Concise offline three-mode report

```bash
phase4b3_d6_compare
RC="$?"
echo "D6-C offline comparison rc=$RC"
```

Each mode must complete 100 requests on the same workload/model/config/commit
and current-server identity. Read `comparison.md` for the main table and ratios;
`comparison.json` contains all requested Draft batch quantiles, purpose counters,
GPU event time, host syncs, acceptance, Target work, cohort sizes, verification
fragmentation and JIT observation. All six throughput/makespan ratios and nine
mean/p50/p90 relative TPOT changes are computed. Case A/B/C/D is selected only
after these measurements; an unlisted order/tie stays unclassified.

Use **production vLLM Batched Draft end-to-end improvement**. Quantify work
differences; do not claim pure batching or call observed overlap critical-path
time saved. This runbook performs no scheduler/acceptance/UUID tuning.

## 14. Package the new C session

```bash
tar -czf "$SR_D6_ROOT.tar.gz" -C "$(dirname "$SR_D6_ROOT")" "$(basename "$SR_D6_ROOT")"
RC="$?"
echo "D6-C artifact package rc=$RC path=$SR_D6_ROOT.tar.gz"
```

Return both new session archives, including logs on failure. Historical
Phase 4B.2, D1–D5 and numerical probe directories remain immutable.
