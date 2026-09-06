# Phase 4B.4B — MineDraft-style ping-pong fresh-server runbook

The agent has not run GPU. These commands are for the operator on AutoDL.
Use the delivered copy with `@DELIVERED_COMMIT@` replaced by its exact tested SHA.
Run each Bash block separately in an interactive **Bash** shell. Every block
prints `rc`; stop manually on any nonzero value and retain its artifacts.
Do not retry into an existing root or continue to the next stage after failure.

Run smoke and corrected-5 first. After both pass, restart the server, repeat
sections 1–7 at the same exact B SHA, and run all five formal corrected-100 cells
on that one server. Do not reuse Task A or bring-up timings as formal controls.
Do not change code/config/environment between the five formal cells.

Default policy remains legacy. Pingpong is selected only by its helper cell.
The source/design audit is in `docs/phase4b4b-pingpong-design.md`.

## 1. Safe shell and conda

```bash
set +e
set +u
set +o pipefail
trap - ERR
export SR_PHASE4B_COMMIT="@DELIVERED_COMMIT@"
export SR_PHASE4B3_REPO="/root/autodl-tmp/src/SpecRhythm"
SR_PP_CONDA_BASE=""
if command -v conda >/dev/null 2>&1; then
  SR_PP_CONDA_BASE="$(conda info --base)"
elif test -x /root/miniconda3/bin/conda; then
  SR_PP_CONDA_BASE="/root/miniconda3"
elif test -x /root/miniconda/bin/conda; then
  SR_PP_CONDA_BASE="/root/miniconda"
fi
if test -n "$SR_PP_CONDA_BASE" && test -f "$SR_PP_CONDA_BASE/etc/profile.d/conda.sh"; then
  source "$SR_PP_CONDA_BASE/etc/profile.d/conda.sh" &&
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
export SR_PHASE4B_DUAL_RHYTHM=legacy
unset SR_PHASE4_DUAL_RHYTHM_MANIFEST
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
GPUs1,2 remain Target TP2/BF16/eager. Proposal K=4, UUID `live`, batch
invariance, measurement boundary, logging and cleanup remain unchanged.
`SR_PHASE4B_DUAL_RHYTHM` defaults to legacy. Legacy mb2/mb100 retain their
existing upper-bound behavior. Pingpong uses burst-sized readiness capacity
and schedules eligible members of the selected persistent A/B cohort, subject
to stock capacity and the opposite-stage dependency. There is no wait to fill
a steady-state batch. Only initial fill requires both groups ready.

## 4. New immutable session root

Create a new immutable root every time. The first two unchanged corrected-5
rows populate the two-request smoke. On the formal server, set
`SR_PP_PREREQUISITE` to the completed corrected-5 qualification path as explained
in section 10; this copies only its structural prerequisite, never its timings.
For the first bring-up session, leave that variable unset.

```bash
export SR_PP_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/$SR_PHASE4B_COMMIT/phase4b4b-pingpong-$(date -u +%Y%m%dT%H%M%SZ)-$$"
export SR_PHASE4B_ENVIRONMENT="$SR_PP_ROOT/environment.json"
export SR_PHASE4B_TOPOLOGY="$SR_PP_ROOT/topology.json"
export SR_PHASE4B_PATCH_MANIFEST="$SR_PP_ROOT/patch-stage/vllm-patch-stack.json"
export SR_PP_SMOKE_WORKLOAD="$SR_PP_ROOT/workloads/smoke-2.jsonl"
python - <<'PYCODE'
import os, shutil
from pathlib import Path
from specrhythm.phase4.manifest import sha256_file
root = Path(os.environ['SR_PP_ROOT'])
manifest = Path(os.environ['SR_PHASE4B3_BASELINE_ROOT']) / 'patch-stage/vllm-patch-stack.json'
assert sha256_file(manifest) == '33c5a75f95a490695d966548d6f638bed810a410ff0d028ee114bfbe854069fc'
for name in ('SR_PHASE4B3_WORKLOAD5', 'SR_PHASE4B3_REFERENCE5', 'SR_PHASE4B_WORKLOAD', 'SR_PHASE4B_REFERENCE'):
    assert Path(os.environ[name]).is_file(), name
root.mkdir(parents=True, exist_ok=False)
(root / 'patch-stage').mkdir()
shutil.copyfile(manifest, os.environ['SR_PHASE4B_PATCH_MANIFEST'])
(root / 'workloads').mkdir()
rows = Path(os.environ['SR_PHASE4B3_WORKLOAD5']).read_text().splitlines()
assert len(rows) == 5
Path(os.environ['SR_PP_SMOKE_WORKLOAD']).write_text('\n'.join(rows[:2]) + '\n')
prerequisite = os.environ.get('SR_PP_PREREQUISITE')
if prerequisite:
    import json
    p = json.loads(Path(prerequisite).read_text())
    assert p['valid'] is True and not p.get('errors')
    assert p['execution_git_commit'] == os.environ['SR_PHASE4B_COMMIT']
    assert p['metrics']['completed_requests'] == 5
    assert p['pingpong']['pipeline']['physical_overlap_valid'] is True
    shutil.copyfile(prerequisite, root / 'corrected5-prerequisite.json')
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
  --expect-state patched --manifest "$SR_PP_ROOT/installed-patch-check.json" &&
env -u CUDA_VISIBLE_DEVICES VLLM_BATCH_INVARIANT=1 specrhythm phase4-probe \
  --config "$SR_PHASE4B_CONFIG" --vllm-source "$SR_VLLM_SOURCE" \
  --environment-output "$SR_PHASE4B_ENVIRONMENT" \
  --topology-output "$SR_PHASE4B_TOPOLOGY" \
  --validation-output "$SR_PP_ROOT/probe-validation.json"
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
root = Path(os.environ['SR_PP_ROOT'])
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert commit == os.environ['SR_PHASE4B_COMMIT']
assert json.loads((root / 'probe-validation.json').read_text())['valid'] is True
config = load_phase4_config(os.environ['SR_PHASE4B_CONFIG'])
assert config.proposal_budget == 4 and config.enforce_eager
assert config.draft.tensor_parallel_size == 1 and config.draft.physical_gpu_ids == (0,)
assert config.target.tensor_parallel_size == 2 and config.target.physical_gpu_ids == (1, 2)
assert config.draft.dtype == config.target.dtype == 'bfloat16'
inputs = {}
for key, count in (('SR_PHASE4B3_WORKLOAD5', 5), ('SR_PP_SMOKE_WORKLOAD', 2), ('SR_PHASE4B_WORKLOAD', 100)):
    path = Path(os.environ[key])
    rows = load_smoke_requests(path, count, require_task_mixture=count in (5, 100))
    assert len(rows) == count
    if count == 100:
        assert all(r.maximum_new_tokens == 16 for r in rows)
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
    'expected_measured_committed_tokens': 1487,
    'formal_cells': ['target', 'serial', 'dual-mb2', 'dual-mb100', 'pingpong'],
    'pingpong_assignment': 'frozen alternating A/B; 50/50 for corrected-100',
    'target_only_scope': 'existing DecodeReady Draft setup; no measured Draft proposals',
})
print('Preflight valid; no generation performed')
PYCODE
RC="$?"
echo "one experiment preflight rc=$RC"
```

## 7. Load existing runners and the isolated baseline helper

```bash
source integrations/vllm/phase4b_run_helpers.sh &&
  source integrations/vllm/phase4b1_gate_helpers.sh &&
  source integrations/vllm/phase4b2_run_helpers.sh &&
  source integrations/vllm/phase4b4b_pingpong_helpers.sh
RC="$?"
echo "load pingpong helpers rc=$RC"
```

## 8. Two-request smoke — first bring-up server only

```bash
phase4b4b_run smoke pingpong
RC="$?"
echo "Pingpong smoke rc=$RC; stop manually if nonzero"
```

Require `qualification.json: valid=true`: A/B nonempty, both initial proposals,
at least one alternation, valid commit/prefix/proposal identities, ordinary
Target structure, retirement, zero live Draft KV and clean shutdown. Each
cohort has one request, so this smoke alone does not require a multi-request
Draft forward or a positive overlap witness. No performance conclusion.

## 9. Corrected-5 structural run — first bring-up server only

```bash
phase4b4b_run five pingpong
RC="$?"
echo "Corrected-5 pingpong rc=$RC; stop manually if nonzero"
```

Require all five completions, valid accounting/lifecycle, stable membership,
selected-cohort Target verification, committed-prefix Draft authorization,
TP sampled-row consensus, actual production multi-request Draft batching and
at least one positive physical cross-cohort overlap witness. Any material
failure stops progression. No final-sequence numerical equality gate is added.

## 10. Preserve bring-up, then restart for formal corrected-100

```bash
phase4b4b_require "$SR_PP_ROOT/five/pingpong/qualification.json" &&
  printf 'export SR_PP_PREREQUISITE=%q\n' "$SR_PP_ROOT/five/pingpong/qualification.json"
RC="$?"
echo "corrected-5 handoff rc=$RC"
```

Copy the printed **exact export command**. Package the bring-up directory with
section 18, then restart AutoDL yourself. On the fresh server, run that export
command and repeat **all of sections 1–7**. They activate conda, check out the
same B SHA, reinstall editable code, restore all exports, create a fresh root,
check the unchanged five-patch stack, capture this server's topology and write
one fresh preflight. The prerequisite copy is only permission to proceed;
all five formal timings will be newly measured in this new root. Do not rerun
sections 8–9 on the formal server. Retain both roots.

```bash
phase4b4b_require "$SR_PP_ROOT/corrected5-prerequisite.json"
RC="$?"
echo "fresh formal root prerequisite rc=$RC; stop manually if nonzero"
```

## 11. Target-only — once on formal server

```bash
phase4b4b_run full target
RC="$?"
echo "Target rc=$RC; stop manually if nonzero"
```

## 12. Serial-vLLM — once on formal server

```bash
phase4b4b_run full serial
RC="$?"
echo "Serial rc=$RC; stop manually if nonzero"
```

## 13. Legacy Dual mb2

```bash
phase4b4b_run full dual-mb2
RC="$?"
echo "legacy Dual mb2 rc=$RC; stop manually if nonzero"
```

## 14. Legacy Dual mb100

```bash
phase4b4b_run full dual-mb100
RC="$?"
echo "legacy Dual mb100 rc=$RC; stop manually if nonzero"
```

## 15. MineDraft-style ping-pong Dual

```bash
phase4b4b_run full pingpong
RC="$?"
echo "Pingpong corrected-100 rc=$RC; stop manually if nonzero"
```

All five must complete 100 requests and 1487 measured tokens. Actual execution,
Target verification structure, token accounting, workload/model/config/backend
identity and lifecycle/KV cleanup remain blocking. The pingpong cell additionally
checks the ten cohort/dependency invariants. Poor speed or valid zero overlap
in corrected-100 are observations, not extra qualification failures. The positive
overlap bring-up requirement belongs to corrected-5. Legacy controls permit valid
zero overlap for characterization, as in Task A; ordinary D6 defaults are unchanged.

## 16. Offline five-mode report

```bash
phase4b4b_compare
RC="$?"
echo "offline five-cell comparison rc=$RC; stop manually if nonzero"
```

```bash
cat "$SR_PP_ROOT/comparison.md"
RC="$?"
echo "concise comparison rc=$RC"
```

The table shows throughput, makespan, TPOT mean, Target forwards/batch p50,
Draft forwards/batch p50/GPU time, accepted length and overlap ms. Four ratios
compare PingPong to Target, Serial, mb100 and mb2. JSON includes TPOT p50/p90,
query/proposed/accepted/rejected tokens, per-cohort distributions and forward
counts, attained versus eligible sizes, stock constraints, fill/drain/wait
observations and physical overlap union/count/fraction. Draft owner idle time
is unavailable and explicitly null; it is not inferred GPU idle time.

## 17. Interpretation and scope

Use the label **MineDraft-style large-cohort ping-pong baseline**, with
**production vLLM Batched Draft end-to-end improvement** for performance.
Do not claim pure batching unless work is quantitatively matched. The report
prints PingPong-minus-control work counters. Observed overlap is never
critical-path time saved. No adaptive mechanism is implemented.

P1: better than mb100 and approaches/exceeds Serial; investigate whether large
cohorts retained batching and useful overlap. P2: approximately mb100; coarse
batching dominates this session. P3: below mb100 but above mb2; inspect switching
and dependency cost. P4: inspect residual runtime/synchronization before the
next research phase. The report's 5% “approximately/approaches” band is a disclosed
descriptive convention, not a significance test, tuning rule or validity gate.
These are single-session observations; no positive outcome is preselected.

## 18. Package the current root, including any failure

```bash
if test ! -e "$SR_PP_ROOT.tar.gz"; then
  tar -czf "$SR_PP_ROOT.tar.gz" -C "$(dirname "$SR_PP_ROOT")" "$(basename "$SR_PP_ROOT")"
else
  echo "Package already exists; retain it and choose a fresh archive name."
  false
fi
RC="$?"
echo "artifact package rc=$RC path=$SR_PP_ROOT.tar.gz"
```

On failure, stop manually and package the first failure. Do not retry into an
existing cell or overwrite reports. Return the bring-up archive and formal
archive, including runtime logs and all immutable qualification evidence.
The agent has not run GPU or connected to AutoDL. Only the operator runs this
runbook. PR #4 remains Draft/Open/unmerged.
