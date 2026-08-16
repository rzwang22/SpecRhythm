# Phase 4A.1.1 batch-invariant correctness runbook

Phase 4A.1 default-mode A/B artifacts must remain unchanged. This addendum creates only fresh C/D
artifacts. It does not run Dual-Batch, Eager, packed trees, SLO/goodput, or a performance test.

The C-reference attempt from commit `a7fe058d417ae44edea497657e17eef161c09d0e`
stopped before engine creation because runner verification imported vLLM before correctness-mode
setup. Keep that incomplete directory as failure provenance; do not resume it or place new output
inside it. The corrected commit creates a new commit-keyed result root below.

Runner checks now locate installed files through distribution metadata and hash them without
importing vLLM. The four execution paths preserve this ordering contract:

| Entry path | Required order before engine creation |
| --- | --- |
| stock reference C | import-free stock SHA check → `run_stock_smoke` mode setup → first vLLM import |
| patched Target regression C | `run_stock_smoke` mode setup → first vLLM import → patched SHA recheck |
| Serial D | mode setup → import-free patched SHA check → first vLLM import |
| fixed local/remote controls | mode setup → import-free patched SHA check → first vLLM import |

## 0. Mandatory 3×A800 preflight

Fetch the Draft PR head, activate the existing independent Python 3.11 environment, and record
the exact commit. Do not edit server code.

```bash
set -euo pipefail
cd /root/autodl-tmp/src/SpecRhythm
git fetch origin codex/vllm-serving-v0.1
git switch --detach origin/codex/vllm-serving-v0.1
export SR_PHASE4_COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --short)"
conda activate /root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1
python -m pip install -e '.[dev]' --no-deps

export SR_DRAFT_MODEL="/root/autodl-tmp/models/Qwen3-0.6B"
export SR_TARGET_MODEL="/root/autodl-tmp/models/Qwen3-32B"
export SR_PHASE4_BI_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/$SR_PHASE4_COMMIT/batch-invariant-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$SR_PHASE4_BI_ROOT"
mkdir -p "$SR_PHASE4_BI_ROOT"

CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
specrhythm phase4-batch-invariant-preflight \
  --correctness-mode batch-invariant \
  --output "$SR_PHASE4_BI_ROOT/batch-invariant-preflight.json"
cat "$SR_PHASE4_BI_ROOT/batch-invariant-preflight.json"
python - "$SR_PHASE4_BI_ROOT/batch-invariant-preflight.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["cuda_available"] is True, value
assert value["batch_invariant_requested"] is True, value
assert value["batch_invariant_effective"] is False, value
assert value["effective_requires_initialized_worker_evidence"] is True, value
assert value["devices"] and all(
    row["compute_capability"] == "8.0" for row in value["devices"]
), value
assert all(row["documented_hardware_supported"] is True for row in value["devices"]), value
assert value["pinned_vllm_hardware_contract"].endswith(">= 8.0"), value
assert value["valid"] is True and value["errors"] == [], value
PY
```

On the current A800 server this preflight must return zero because every visible GPU has compute
capability 8.0, which satisfies the exact pinned vLLM v0.25.1 hardware requirement. This is only
permission to initialize C/D workers. `batch_invariant_effective` deliberately remains false in
the preflight: hardware capability and an environment request are not runtime proof.

Every initialized TP rank in C and D must independently prove that
`VLLM_BATCH_INVARIANT` resolved true, every active attention backend reports batch-invariance
support, custom all-reduce is disabled, cascade attention is disabled, and DBO is disabled. A
missing or inconsistent field fails closed. Do not proceed past a worker-validation failure and
do not relabel the preflight as effective evidence.

Reuse the exact five-request file from A/B without modifying it. Point `SR_PHASE4_AB_RUN` at the
old default-mode result directory; checksums are retained as provenance, while all C/D outputs go
under the fresh root.

```bash
export SR_PHASE4_AB_RUN="/root/autodl-tmp/SpecRhythm-data/results/phase4/REPLACE_WITH_AB_COMMIT/REPLACE_WITH_AB_RUN"
export SR_R3_SMOKE="$SR_PHASE4_AB_RUN/r3-real-smoke-5.jsonl"
export SR_VLLM_SOURCE="/root/autodl-tmp/src/vllm-v0.25.1"
export SR_VLLM_ROOT="$(python - <<'PY'
from pathlib import Path
import vllm
print(Path(vllm.__file__).resolve().parents[1])
PY
)"
test -f "$SR_R3_SMOKE"
sha256sum "$SR_R3_SMOKE" | tee "$SR_PHASE4_BI_ROOT/workload.sha256"
```

Restore/check the unmodified Target runner, probe once, then create two independent immutable C
references. Target diagnostic rows are collected later from patched Target-only regressions,
which must exactly equal these unmodified references.

```bash
python integrations/vllm/manage_patch.py restore \
  --vllm-root "$SR_VLLM_ROOT" --source "$SR_VLLM_SOURCE" || \
python integrations/vllm/manage_patch.py check \
  --vllm-root "$SR_VLLM_ROOT" --source "$SR_VLLM_SOURCE"
python integrations/vllm/manage_patch.py check \
  --vllm-root "$SR_VLLM_ROOT" --source "$SR_VLLM_SOURCE"

python - <<'PY'
import sys
from specrhythm.phase4.reference import require_stock_vllm_runner

assert not any(name == "vllm" or name.startswith("vllm.") for name in sys.modules)
print("stock_runner_sha256=" + require_stock_vllm_runner())
assert not any(name == "vllm" or name.startswith("vllm.") for name in sys.modules)
print("runner_verification_import_free=true")
PY

env -u CUDA_VISIBLE_DEVICES VLLM_BATCH_INVARIANT=1 specrhythm phase4-probe \
  --config configs/phase4a_target_fair_1d2v.yaml \
  --vllm-source "$SR_VLLM_SOURCE" \
  --environment-output "$SR_PHASE4_BI_ROOT/environment.json" \
  --topology-output "$SR_PHASE4_BI_ROOT/topology.json" \
  --validation-output "$SR_PHASE4_BI_ROOT/probe-validation.json"

for run_id in 1 2; do
  run_dir="$SR_PHASE4_BI_ROOT/C-$run_id"
  test ! -e "$run_dir"
  mkdir -p "$run_dir"
  CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
  specrhythm phase4-stock-reference \
    --config configs/phase4a_target_fair_1d2v.yaml \
    --correctness-mode batch-invariant \
    --workload "$SR_R3_SMOKE" \
    --environment "$SR_PHASE4_BI_ROOT/environment.json" \
    --topology "$SR_PHASE4_BI_ROOT/topology.json" \
    --runtime-manifest "$run_dir/runtime-manifest.json" \
    --output "$run_dir/stock-target-reference.json" \
    2>&1 | tee "$run_dir/stock-reference.log"
  chmod a-w "$run_dir/stock-target-reference.json"
done
```

Apply the exact zero-fuzz Python observer patch, then run two patched Target-only regressions to
collect C-side target diagnostics without changing token semantics.

```bash
python integrations/vllm/manage_patch.py apply \
  --vllm-root "$SR_VLLM_ROOT" \
  --source "$SR_VLLM_SOURCE" \
  --manifest "$SR_PHASE4_BI_ROOT/vllm-base-and-patch-manifest.json"

for run_id in 1 2; do
  run_dir="$SR_PHASE4_BI_ROOT/C-$run_id"
  CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
  specrhythm phase4-target-regression \
    --config configs/phase4a_target_fair_1d2v.yaml \
    --correctness-mode batch-invariant \
    --workload "$SR_R3_SMOKE" \
    --environment "$SR_PHASE4_BI_ROOT/environment.json" \
    --topology "$SR_PHASE4_BI_ROOT/topology.json" \
    --runtime-manifest "$run_dir/patched-runtime-manifest.json" \
    --reference "$run_dir/stock-target-reference.json" \
    --patch-manifest "$SR_PHASE4_BI_ROOT/vllm-base-and-patch-manifest.json" \
    --target-diagnostics "$run_dir/target-diagnostics.jsonl" \
    --output "$run_dir/patched-target-regression.json" \
    2>&1 | tee "$run_dir/patched-target-regression.log"
done
```

Run D twice. Each run uses a new Draft process, socket, checkpoints, runtime manifest, diagnostic
log, and result file. The Draft process receives no Target diagnostics.

```bash
run_batch_invariant_serial () {
  run_id="$1"
  run_dir="$SR_PHASE4_BI_ROOT/D-$run_id"
  test ! -e "$run_dir"
  mkdir -p "$run_dir"
  CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
  specrhythm phase4-draft-service \
    --config configs/phase4a_target_fair_1d2v.yaml \
    --socket "$run_dir/draft.sock" \
    --event-log "$run_dir/draft-service-events.jsonl" \
    --ready "$run_dir/draft-service-ready.json" \
    >"$run_dir/draft-service.log" 2>&1 &
  draft_pid="$!"
  for _ in $(seq 1 600); do
    test -f "$run_dir/draft-service-ready.json" && test -S "$run_dir/draft.sock" && break
    kill -0 "$draft_pid" 2>/dev/null || { cat "$run_dir/draft-service.log"; return 1; }
    sleep 1
  done
  test -S "$run_dir/draft.sock"
  set +e
  CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
  specrhythm phase4-serial-run \
    --config configs/phase4a_target_fair_1d2v.yaml \
    --correctness-mode batch-invariant \
    --workload "$SR_R3_SMOKE" \
    --environment "$SR_PHASE4_BI_ROOT/environment.json" \
    --topology "$SR_PHASE4_BI_ROOT/topology.json" \
    --runtime-manifest "$run_dir/runtime-manifest.json" \
    --reference "$SR_PHASE4_BI_ROOT/C-1/stock-target-reference.json" \
    --patch-manifest "$SR_PHASE4_BI_ROOT/vllm-base-and-patch-manifest.json" \
    --draft-socket "$run_dir/draft.sock" \
    --draft-ready "$run_dir/draft-service-ready.json" \
    --round-events "$run_dir/round-events.jsonl" \
    --transport-events "$run_dir/transport-events.jsonl" \
    --plugin-report "$run_dir/remote-proposer-report.json" \
    --target-diagnostics "$run_dir/target-diagnostics.jsonl" \
    --output "$run_dir/serial-run.json" \
    >"$run_dir/target-serial.log" 2>&1
  serial_status="$?"
  set -e
  wait "$draft_pid"
  test -f "$run_dir/serial-run.json"
  echo "$serial_status" >"$run_dir/serial-exit-status.txt"
}
run_batch_invariant_serial 1
run_batch_invariant_serial 2
```

Validate C/D. A zero exit and `outcome=A` is the only Phase 4A.1 correctness pass.

```bash
set +e
specrhythm phase4-batch-invariant-validate \
  --stock-reference "$SR_PHASE4_BI_ROOT/C-1/stock-target-reference.json" \
  --stock-reference "$SR_PHASE4_BI_ROOT/C-2/stock-target-reference.json" \
  --target-regression "$SR_PHASE4_BI_ROOT/C-1/patched-target-regression.json" \
  --target-regression "$SR_PHASE4_BI_ROOT/C-2/patched-target-regression.json" \
  --serial-run "$SR_PHASE4_BI_ROOT/D-1/serial-run.json" \
  --serial-run "$SR_PHASE4_BI_ROOT/D-2/serial-run.json" \
  --round-events "$SR_PHASE4_BI_ROOT/D-1/round-events.jsonl" \
  --round-events "$SR_PHASE4_BI_ROOT/D-2/round-events.jsonl" \
  --target-diagnostics "$SR_PHASE4_BI_ROOT/C-1/target-diagnostics.jsonl" \
  --target-diagnostics "$SR_PHASE4_BI_ROOT/C-2/target-diagnostics.jsonl" \
  --serial-diagnostics "$SR_PHASE4_BI_ROOT/D-1/target-diagnostics.jsonl" \
  --serial-diagnostics "$SR_PHASE4_BI_ROOT/D-2/target-diagnostics.jsonl" \
  --output "$SR_PHASE4_BI_ROOT/validation.json" \
  --markdown-output "$SR_PHASE4_BI_ROOT/summary.md"
validation_status="$?"
set -e
cat "$SR_PHASE4_BI_ROOT/summary.md"
```

### Validator-only replay for immutable completed C/D artifacts

If C1/C2/D1/D2 already completed, do not rerun them and do not write new files into their input
root. Check out the corrected validator commit, point `SR_PHASE4_BI_ROOT` at the existing immutable
C/D directory, and write only to a separate validation directory. The before/after digest diff
proves that validation did not mutate any GPU artifact.

```bash
set -euo pipefail
cd /root/autodl-tmp/src/SpecRhythm
git fetch origin codex/vllm-serving-v0.1
git switch --detach origin/codex/vllm-serving-v0.1
export SR_VALIDATOR_COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --short)"
python -m pip install -e '.[dev]' --no-deps

export SR_PHASE4_BI_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/REPLACE_WITH_EXISTING_COMPLETE_CD_ROOT"
test -f "$SR_PHASE4_BI_ROOT/C-1/stock-target-reference.json"
test -f "$SR_PHASE4_BI_ROOT/C-2/stock-target-reference.json"
test -f "$SR_PHASE4_BI_ROOT/D-1/serial-run.json"
test -f "$SR_PHASE4_BI_ROOT/D-2/serial-run.json"

export SR_PHASE4_VALIDATION_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4-validator/$SR_VALIDATOR_COMMIT/$(basename "$SR_PHASE4_BI_ROOT")"
test ! -e "$SR_PHASE4_VALIDATION_ROOT"
mkdir -p "$SR_PHASE4_VALIDATION_ROOT"

find "$SR_PHASE4_BI_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$SR_PHASE4_VALIDATION_ROOT/input-sha256-before.txt"

set +e
specrhythm phase4-batch-invariant-validate \
  --stock-reference "$SR_PHASE4_BI_ROOT/C-1/stock-target-reference.json" \
  --stock-reference "$SR_PHASE4_BI_ROOT/C-2/stock-target-reference.json" \
  --target-regression "$SR_PHASE4_BI_ROOT/C-1/patched-target-regression.json" \
  --target-regression "$SR_PHASE4_BI_ROOT/C-2/patched-target-regression.json" \
  --serial-run "$SR_PHASE4_BI_ROOT/D-1/serial-run.json" \
  --serial-run "$SR_PHASE4_BI_ROOT/D-2/serial-run.json" \
  --round-events "$SR_PHASE4_BI_ROOT/D-1/round-events.jsonl" \
  --round-events "$SR_PHASE4_BI_ROOT/D-2/round-events.jsonl" \
  --target-diagnostics "$SR_PHASE4_BI_ROOT/C-1/target-diagnostics.jsonl" \
  --target-diagnostics "$SR_PHASE4_BI_ROOT/C-2/target-diagnostics.jsonl" \
  --serial-diagnostics "$SR_PHASE4_BI_ROOT/D-1/target-diagnostics.jsonl" \
  --serial-diagnostics "$SR_PHASE4_BI_ROOT/D-2/target-diagnostics.jsonl" \
  --output "$SR_PHASE4_VALIDATION_ROOT/validation.json" \
  --markdown-output "$SR_PHASE4_VALIDATION_ROOT/summary.md"
validator_exit="$?"
set -e

find "$SR_PHASE4_BI_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$SR_PHASE4_VALIDATION_ROOT/input-sha256-after.txt"
diff -u \
  "$SR_PHASE4_VALIDATION_ROOT/input-sha256-before.txt" \
  "$SR_PHASE4_VALIDATION_ROOT/input-sha256-after.txt"
cat "$SR_PHASE4_VALIDATION_ROOT/summary.md"

python - "$SR_PHASE4_VALIDATION_ROOT/validation.json" "$validator_exit" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
assert int(sys.argv[2]) == 0, value
assert value["valid"] is True and value["outcome"] == "A", value
checks = value["checks"]
assert checks["target_only_repeated_exact_equality"] is True, value
assert checks["serial_repeated_exact_equality"] is True, value
assert checks["serial_equals_stock"] == [True, True], value
assert checks["round_semantics_repeated"] is True, value
assert checks["target_diagnostics_valid"] is True, value
print("validator_only_outcome=A")
print("raw_event_order_equal=" + str(checks["raw_event_order_equal"]).lower())
PY
```

If `validation_status=0`, stop with Outcome A. If it is nonzero because C differs from D, create
the first-divergence proof before running controls:

```bash
specrhythm phase4-divergence-diagnose \
  --stock-diagnostics "$SR_PHASE4_BI_ROOT/C-1/target-diagnostics.jsonl" \
  --serial-diagnostics "$SR_PHASE4_BI_ROOT/D-1/target-diagnostics.jsonl" \
  --serial-run "$SR_PHASE4_BI_ROOT/D-1/serial-run.json" \
  --output "$SR_PHASE4_BI_ROOT/first-divergence.json" || true
cat "$SR_PHASE4_BI_ROOT/first-divergence.json"
```

Do not run the following K=1/2/4 matrix unless C != D. First freeze only the divergent request;
no prompt or output limit is rewritten.

```bash
python - "$SR_R3_SMOKE" "$SR_PHASE4_BI_ROOT/D-1/serial-run.json" \
  "$SR_PHASE4_BI_ROOT/divergent-request.jsonl" <<'PY'
import json
import sys
run = json.load(open(sys.argv[2], encoding="utf-8"))
item = next(row for row in run["comparison"]["requests"]
            if row["first_divergence_position"] is not None)
request_id = item["request_id"]
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
row = next(row for row in rows if row["request_id"] == request_id)
with open(sys.argv[3], "x", encoding="utf-8") as handle:
    handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
print(request_id)
PY

for k in 1 2 4; do
  control_dir="$SR_PHASE4_BI_ROOT/controls/K$k"
  mkdir -p "$control_dir"
  CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
  specrhythm phase4-fixed-control-run \
    --config configs/phase4a_target_fair_1d2v.yaml \
    --workload "$SR_PHASE4_BI_ROOT/divergent-request.jsonl" \
    --environment "$SR_PHASE4_BI_ROOT/environment.json" \
    --topology "$SR_PHASE4_BI_ROOT/topology.json" \
    --patch-manifest "$SR_PHASE4_BI_ROOT/vllm-base-and-patch-manifest.json" \
    --proposer local-static --proposal-budget "$k" \
    --target-diagnostics "$control_dir/local-diagnostics.jsonl" \
    --output "$control_dir/local-run.json"

  specrhythm phase4-fixed-proposal-service \
    --socket "$control_dir/fixed.sock" >"$control_dir/fixed-service.log" 2>&1 &
  fixed_pid="$!"
  for _ in $(seq 1 60); do
    test -S "$control_dir/fixed.sock" && break
    kill -0 "$fixed_pid" 2>/dev/null || { cat "$control_dir/fixed-service.log"; exit 1; }
    sleep 1
  done
  CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
  specrhythm phase4-fixed-control-run \
    --config configs/phase4a_target_fair_1d2v.yaml \
    --workload "$SR_PHASE4_BI_ROOT/divergent-request.jsonl" \
    --environment "$SR_PHASE4_BI_ROOT/environment.json" \
    --topology "$SR_PHASE4_BI_ROOT/topology.json" \
    --patch-manifest "$SR_PHASE4_BI_ROOT/vllm-base-and-patch-manifest.json" \
    --proposer remote-fixed --proposal-budget "$k" \
    --remote-socket "$control_dir/fixed.sock" \
    --target-diagnostics "$control_dir/remote-diagnostics.jsonl" \
    --output "$control_dir/remote-run.json"
  wait "$fixed_pid"
done

specrhythm phase4-fixed-control-validate \
  --local-run "$SR_PHASE4_BI_ROOT/controls/K1/local-run.json" \
  --local-run "$SR_PHASE4_BI_ROOT/controls/K2/local-run.json" \
  --local-run "$SR_PHASE4_BI_ROOT/controls/K4/local-run.json" \
  --remote-run "$SR_PHASE4_BI_ROOT/controls/K1/remote-run.json" \
  --remote-run "$SR_PHASE4_BI_ROOT/controls/K2/remote-run.json" \
  --remote-run "$SR_PHASE4_BI_ROOT/controls/K4/remote-run.json" \
  --local-diagnostics "$SR_PHASE4_BI_ROOT/controls/K1/local-diagnostics.jsonl" \
  --local-diagnostics "$SR_PHASE4_BI_ROOT/controls/K2/local-diagnostics.jsonl" \
  --local-diagnostics "$SR_PHASE4_BI_ROOT/controls/K4/local-diagnostics.jsonl" \
  --remote-diagnostics "$SR_PHASE4_BI_ROOT/controls/K1/remote-diagnostics.jsonl" \
  --remote-diagnostics "$SR_PHASE4_BI_ROOT/controls/K2/remote-diagnostics.jsonl" \
  --remote-diagnostics "$SR_PHASE4_BI_ROOT/controls/K4/remote-diagnostics.jsonl" \
  --output "$SR_PHASE4_BI_ROOT/fixed-control-validation.json"
```

Outcome B requires both `first-divergence.json.valid=true` and
`fixed-control-validation.json.local_remote_all_equal=true`; exact C/D correctness still has not
passed. Otherwise classify Outcome C and return to integration debugging.

---

# Phase 4A.1 default-mode A/B provenance (3×A800)

This runbook executes a five-request GPU correctness smoke. It does not benchmark performance,
replay arrivals, report SLO/goodput/speedup, or implement Dual-Batch, Eager, packed trees, TP3, or
OPT-66B. Generated artifacts remain outside Git.

The fixed layout is Draft Qwen3-0.6B on physical GPU 0 and Target Qwen3-32B TP=2 on physical GPUs
1 and 2. `VLLM_USE_V2_MODEL_RUNNER=0` is required because custom-class proposers are implemented
only by the pinned V1 runner in vLLM v0.25.1.

## 1. Fetch the Draft PR head without changing PR #3

```bash
set -euo pipefail
cd /root/autodl-tmp/src/SpecRhythm
git fetch origin codex/vllm-serving-v0.1
git switch --detach origin/codex/vllm-serving-v0.1
export SR_PHASE4_COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --short)"
git show --no-patch --oneline "$SR_PHASE4_COMMIT"
```

Compare `SR_PHASE4_COMMIT` with the handoff commit before loading a model.

## 2. Activate the independent pinned environment

Reuse the Phase 4 Python 3.11 environment if Phase 4A.0 already created it:

```bash
conda activate /root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1
cd /root/autodl-tmp/src/SpecRhythm
python -VV
python -m pip install -e '.[dev]' --no-deps
python -m pip check
```

If it does not exist, create it first:

```bash
conda create -y -p /root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1 python=3.11
conda activate /root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1
python -m pip install --upgrade pip setuptools wheel
python -m pip install 'vllm==0.25.1' 'torch==2.11.0' 'transformers>=5.5.3,<5.6'
cd /root/autodl-tmp/src/SpecRhythm
python -m pip install -e '.[dev]' --no-deps
python -m pip check
```

Verify versions and CUDA without setting the insecure-serialization override:

```bash
unset VLLM_ALLOW_INSECURE_SERIALIZATION
export VLLM_USE_V2_MODEL_RUNNER=0
python - <<'PY'
import platform
import torch
import transformers
import vllm

assert platform.python_version().startswith("3.11.")
assert torch.__version__.split("+")[0] == "2.11.0"
assert vllm.__version__ == "0.25.1"
assert torch.cuda.is_available()
print({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "nccl": torch.cuda.nccl.version(),
    "transformers": transformers.__version__,
    "vllm": vllm.__version__,
})
PY
```

## 3. Verify the exact vLLM source and stock installed Python file

```bash
export SR_VLLM_SOURCE="/root/autodl-tmp/src/vllm-v0.25.1"
if test ! -d "$SR_VLLM_SOURCE/.git"; then
  git clone --depth=1 --branch v0.25.1 https://github.com/vllm-project/vllm.git \
    "$SR_VLLM_SOURCE"
fi
git -C "$SR_VLLM_SOURCE" checkout --detach \
  752a3a504485790a2e8491cacbb35c137339ad34
test "$(git -C "$SR_VLLM_SOURCE" rev-parse HEAD)" = \
  "752a3a504485790a2e8491cacbb35c137339ad34"
test "$(git -C "$SR_VLLM_SOURCE" describe --tags --exact-match HEAD)" = "v0.25.1"
test -z "$(git -C "$SR_VLLM_SOURCE" status --porcelain --untracked-files=no)"

export SR_VLLM_ROOT="$(python - <<'PY'
from pathlib import Path
import vllm
print(Path(vllm.__file__).resolve().parents[1])
PY
)"
python integrations/vllm/manage_patch.py check \
  --vllm-root "$SR_VLLM_ROOT" \
  --source "$SR_VLLM_SOURCE"
```

The last command must report `patch_applied=false`. Stop if the stock-file SHA does not match; do
not force or fuzzy-apply the patch.

## 4. Freeze models, workload, reference, and output paths

```bash
export SR_DRAFT_MODEL="/root/autodl-tmp/models/Qwen3-0.6B"
export SR_TARGET_MODEL="/root/autodl-tmp/models/Qwen3-32B"
export SR_PHASE3C_COMMIT="34c7ea9836c2595c8a8aeaeb5680709520edd3d8"
export SR_PHASE3C_RUN="/root/autodl-tmp/SpecRhythm-data/results/phase3c/$SR_PHASE3C_COMMIT/corrected-multiround-100"
export SR_R3_WORKLOAD="$SR_PHASE3C_RUN/workload.jsonl"
export SR_HF_TARGET="$SR_PHASE3C_RUN/target"
export SR_PHASE4_RUN="/root/autodl-tmp/SpecRhythm-data/results/phase4/$SR_PHASE4_COMMIT/serial-1d2v-$(date -u +%Y%m%dT%H%M%SZ)"

test -f "$SR_DRAFT_MODEL/config.json"
test -f "$SR_TARGET_MODEL/config.json"
test -f "$SR_R3_WORKLOAD"
test -d "$SR_HF_TARGET/requests"
test ! -e "$SR_PHASE4_RUN"
mkdir -p "$SR_PHASE4_RUN"
```

Extract the earliest corrected 3/1/1 requests in trace order. No prompt is rewritten or
retokenized:

```bash
python - "$SR_R3_WORKLOAD" "$SR_PHASE4_RUN/r3-real-smoke-5.jsonl" <<'PY'
import json
import sys

needed = {"code": 3, "chat": 1, "summarization": 1}
rows = []
for line in open(sys.argv[1], encoding="utf-8"):
    row = json.loads(line)
    task = row["task_class"]
    if needed.get(task, 0):
        rows.append(row)
        needed[task] -= 1
    if not any(needed.values()):
        break
assert needed == {"code": 0, "chat": 0, "summarization": 0}, needed
assert len(rows) == 5
assert all(row["prompt_length"] == len(row["prompt_token_ids"]) for row in rows)
assert all(row["maximum_new_tokens"] > 0 for row in rows)
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
print([(row["request_id"], row["task_class"], row["prompt_length"]) for row in rows])
PY
```

## 5. Check the earlier Phase 4A.0 artifact, then probe this run

```bash
export SR_PHASE4A0_COMMIT="ba9bded3f16c9b58f13c93a4426c000d523330cd"
export SR_PHASE4A0_RUN="/root/autodl-tmp/SpecRhythm-data/results/phase4/$SR_PHASE4A0_COMMIT/stock-1d2v"
test -f "$SR_PHASE4A0_RUN/validation.json"
python - "$SR_PHASE4A0_RUN/validation.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["valid"], value
print("Phase 4A.0 artifact valid")
PY

nvidia-smi -L | tee "$SR_PHASE4_RUN/nvidia-smi-L.txt"
nvidia-smi topo -m | tee "$SR_PHASE4_RUN/nvidia-smi-topo.txt"
env -u CUDA_VISIBLE_DEVICES specrhythm phase4-probe \
  --config configs/phase4a_target_fair_1d2v.yaml \
  --vllm-source "$SR_VLLM_SOURCE" \
  --environment-output "$SR_PHASE4_RUN/environment.json" \
  --topology-output "$SR_PHASE4_RUN/topology.json" \
  --validation-output "$SR_PHASE4_RUN/probe-validation.json" \
  2>&1 | tee "$SR_PHASE4_RUN/probe.log"
```

## 6. Generate and freeze the unmodified stock Target-only reference

This must happen before applying the patch. The command performs two identical greedy runs and
freezes the output file with exclusive creation.

```bash
CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 \
specrhythm phase4-stock-reference \
  --config configs/phase4a_target_fair_1d2v.yaml \
  --workload "$SR_PHASE4_RUN/r3-real-smoke-5.jsonl" \
  --environment "$SR_PHASE4_RUN/environment.json" \
  --topology "$SR_PHASE4_RUN/topology.json" \
  --runtime-manifest "$SR_PHASE4_RUN/runtime-manifest.json" \
  --legacy-hf-target-dir "$SR_HF_TARGET" \
  --output "$SR_PHASE4_RUN/stock-target-reference.json" \
  2>&1 | tee "$SR_PHASE4_RUN/stock-reference.log"

chmod a-w "$SR_PHASE4_RUN/stock-target-reference.json"
sha256sum "$SR_PHASE4_RUN/stock-target-reference.json" | \
  tee "$SR_PHASE4_RUN/stock-target-reference.sha256"
```

An HF mismatch is advisory. A repeated stock-vLLM mismatch is fatal.

## 7. Archived Phase-4A patch command

This earlier section predates the ordered Phase-4B patch stack. For new Phase-4B work, use
[phase4b-gate-runbook.md](phase4b-gate-runbook.md), which validates and records all three patches.

```bash
python integrations/vllm/manage_patch.py apply \
  --vllm-root "$SR_VLLM_ROOT" \
  --source "$SR_VLLM_SOURCE" \
  --manifest "$SR_PHASE4_RUN/vllm-base-and-patch-manifest.json" \
  2>&1 | tee "$SR_PHASE4_RUN/patch-apply.log"

python - "$SR_PHASE4_RUN/vllm-base-and-patch-manifest.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["patch_applied"] is True
assert value["python_only"] is True
assert value["cpp_cuda_modified"] is False
assert value["target_only_behavior_change_when_speculation_disabled"] is False
print(json.dumps(value, indent=2, sort_keys=True))
PY
```

Do not set `VLLM_ALLOW_INSECURE_SERIALIZATION=1` unless vLLM explicitly refuses its local worker
RPC. If it is required, use only this trusted host and regenerate the patch manifest with the
variable set so validation emits the security warning.

## 8. Run patched Target-only regression

```bash
CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 \
specrhythm phase4-target-regression \
  --config configs/phase4a_target_fair_1d2v.yaml \
  --workload "$SR_PHASE4_RUN/r3-real-smoke-5.jsonl" \
  --environment "$SR_PHASE4_RUN/environment.json" \
  --topology "$SR_PHASE4_RUN/topology.json" \
  --runtime-manifest "$SR_PHASE4_RUN/runtime-manifest.json" \
  --reference "$SR_PHASE4_RUN/stock-target-reference.json" \
  --patch-manifest "$SR_PHASE4_RUN/vllm-base-and-patch-manifest.json" \
  --legacy-hf-target-dir "$SR_HF_TARGET" \
  --output "$SR_PHASE4_RUN/patched-target-regression.json" \
  2>&1 | tee "$SR_PHASE4_RUN/patched-target-regression.log"

python - "$SR_PHASE4_RUN/patched-target-regression.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["valid"], value
assert value["comparison"]["all_sequences_equal"] is True
print("patched Target-only == frozen stock Target-only")
PY
```

Stop here if regression fails. Never regenerate the reference to match patched output.

## 9. Run Serial Disaggregated twice

Define a shell function that creates one fresh Draft service and one Target process. Each run uses
fresh sockets and checkpoints; no artifact is resumed or overwritten.

```bash
run_serial_once () {
  run_id="$1"
  run_dir="$SR_PHASE4_RUN/run-$run_id"
  test ! -e "$run_dir"
  mkdir -p "$run_dir"

  CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=0 \
  specrhythm phase4-draft-service \
    --config configs/phase4a_target_fair_1d2v.yaml \
    --socket "$run_dir/draft.sock" \
    --event-log "$run_dir/draft-service-events.jsonl" \
    --ready "$run_dir/draft-service-ready.json" \
    >"$run_dir/draft-service.log" 2>&1 &
  draft_pid="$!"

  ready=0
  for _ in $(seq 1 600); do
    if test -f "$run_dir/draft-service-ready.json" && test -S "$run_dir/draft.sock"; then
      ready=1
      break
    fi
    if ! kill -0 "$draft_pid" 2>/dev/null; then
      cat "$run_dir/draft-service.log"
      return 1
    fi
    sleep 1
  done
  if test "$ready" != 1; then
    kill "$draft_pid" 2>/dev/null || true
    wait "$draft_pid" 2>/dev/null || true
    echo "Draft service readiness timeout" >&2
    return 1
  fi

  if ! CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 \
    specrhythm phase4-serial-run \
      --config configs/phase4a_target_fair_1d2v.yaml \
      --workload "$SR_PHASE4_RUN/r3-real-smoke-5.jsonl" \
      --environment "$SR_PHASE4_RUN/environment.json" \
      --topology "$SR_PHASE4_RUN/topology.json" \
      --runtime-manifest "$SR_PHASE4_RUN/runtime-manifest.json" \
      --reference "$SR_PHASE4_RUN/stock-target-reference.json" \
      --patch-manifest "$SR_PHASE4_RUN/vllm-base-and-patch-manifest.json" \
      --draft-socket "$run_dir/draft.sock" \
      --draft-ready "$run_dir/draft-service-ready.json" \
      --round-events "$SR_PHASE4_RUN/round-events-run-$run_id.jsonl" \
      --transport-events "$SR_PHASE4_RUN/transport-events-run-$run_id.jsonl" \
      --plugin-report "$run_dir/remote-proposer-report.json" \
      --output "$SR_PHASE4_RUN/serial-disaggregated-run-$run_id.json" \
      >"$run_dir/target-serial.log" 2>&1; then
    kill "$draft_pid" 2>/dev/null || true
    wait "$draft_pid" 2>/dev/null || true
    cat "$run_dir/target-serial.log"
    return 1
  fi
  wait "$draft_pid"
  test ! -S "$run_dir/draft.sock"
  python - "$SR_PHASE4_RUN/serial-disaggregated-run-$run_id.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["valid"], value
assert value["exact_sequence_match"] is True
assert value["strict_serial_timeline"]["validated_in_runner"] is True
assert value["accounting"]["valid"] is True
print({
    "requests": value["request_count"],
    "rounds": value["strict_serial_timeline"]["round_events"],
    "accounting": value["accounting"],
})
PY
}

run_serial_once 1
run_serial_once 2
```

## 10. Validate both runs and generate the summary

```bash
specrhythm phase4-serial-validate \
  --config configs/phase4a_target_fair_1d2v.yaml \
  --reference "$SR_PHASE4_RUN/stock-target-reference.json" \
  --patch-manifest "$SR_PHASE4_RUN/vllm-base-and-patch-manifest.json" \
  --target-regression "$SR_PHASE4_RUN/patched-target-regression.json" \
  --run "$SR_PHASE4_RUN/serial-disaggregated-run-1.json" \
  --run "$SR_PHASE4_RUN/serial-disaggregated-run-2.json" \
  --round-events "$SR_PHASE4_RUN/round-events-run-1.jsonl" \
  --round-events "$SR_PHASE4_RUN/round-events-run-2.jsonl" \
  --transport-events "$SR_PHASE4_RUN/transport-events-run-1.jsonl" \
  --transport-events "$SR_PHASE4_RUN/transport-events-run-2.jsonl" \
  --output "$SR_PHASE4_RUN/validation.json" \
  --markdown-output "$SR_PHASE4_RUN/summary.md" \
  2>&1 | tee "$SR_PHASE4_RUN/validation.log"

cat "$SR_PHASE4_RUN/summary.md"
python - "$SR_PHASE4_RUN/validation.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["valid"], value
assert value["checks"]["patched_target_equals_stock"] is True
assert value["checks"]["serial_runs_equal_stock"] == [True, True]
assert value["checks"]["serial_runs_deterministic"] is True
print(json.dumps(value, indent=2, sort_keys=True))
PY
```

## 11. Create the review bundle

```bash
tar -C "$SR_PHASE4_RUN" -czf "$SR_PHASE4_RUN/review-bundle.tar.gz" \
  environment.json \
  topology.json \
  probe-validation.json \
  runtime-manifest.json \
  vllm-base-and-patch-manifest.json \
  stock-target-reference.json \
  stock-target-reference.sha256 \
  patched-target-regression.json \
  serial-disaggregated-run-1.json \
  serial-disaggregated-run-2.json \
  round-events-run-1.jsonl \
  round-events-run-2.jsonl \
  transport-events-run-1.jsonl \
  transport-events-run-2.jsonl \
  validation.json \
  summary.md \
  nvidia-smi-L.txt \
  nvidia-smi-topo.txt \
  probe.log \
  stock-reference.log \
  patch-apply.log \
  patched-target-regression.log \
  validation.log \
  run-1 \
  run-2

sha256sum "$SR_PHASE4_RUN/review-bundle.tar.gz" | \
  tee "$SR_PHASE4_RUN/review-bundle.sha256"
find "$SR_PHASE4_RUN" -maxdepth 2 -type f -print | sort
```

Return `summary.md`, `validation.json`, both Serial JSON files, both round-event logs, the bundle,
and `review-bundle.sha256`. This is the historical Phase 4A stop boundary; the separately headed
Phase 4B protocol below supersedes only the old prohibition on starting Dual-Batch. It still does
not authorize an arrival replay, latency/capacity sweep, Eager, or packed-tree experiment.

# Archived Phase 4B prototype procedure — do not execute

The commands below preserve the pre-4B.0a prototype for provenance only. They contain the old
mixed Prefill/Decode flow and do not enforce the current decode-ready boundary. The active,
ordered 3×A800 procedure is [phase4b-gate-runbook.md](phase4b-gate-runbook.md): Gate A first,
then two- and five-request Gate B only. Do not run the archived L2/L5/L100 procedure.

This is a new result root. It does not mutate the frozen Phase 4A Outcome-A artifacts. It proves
correctness and the existence of GPU overlap only; it is not a performance experiment.

## 1. Exact code, environment, models and corrected workloads

```bash
set -euo pipefail
cd /root/autodl-tmp/src/SpecRhythm
git fetch origin codex/vllm-serving-v0.1
git switch --detach origin/codex/vllm-serving-v0.1
export SR_PHASE4B_COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --short)"

conda activate /root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1
python --version
python -m pip install -e '.[dev]' --no-deps

export SR_DRAFT_MODEL="/root/autodl-tmp/models/Qwen3-0.6B"
export SR_TARGET_MODEL="/root/autodl-tmp/models/Qwen3-32B"
export SR_VLLM_SOURCE="/root/autodl-tmp/src/vllm-v0.25.1"
export SR_PHASE3C_COMMIT="34c7ea9836c2595c8a8aeaeb5680709520edd3d8"
export SR_PHASE3C_RUN="/root/autodl-tmp/SpecRhythm-data/results/phase3c/$SR_PHASE3C_COMMIT/corrected-multiround-100"
export SR_R3_100="$SR_PHASE3C_RUN/workload.jsonl"
export SR_PHASE4B_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/$SR_PHASE4B_COMMIT/dual-batch-1d2v-$(date -u +%Y%m%dT%H%M%SZ)"
export SR_PHASE4B_INPUT_ROOT="${SR_PHASE4B_INPUT_ROOT:-$SR_PHASE4B_ROOT}"
source integrations/vllm/phase4b_run_helpers.sh

test -f "$SR_DRAFT_MODEL/config.json"
test -f "$SR_TARGET_MODEL/config.json"
test -f "$SR_R3_100"
test "$(wc -l < "$SR_R3_100")" -eq 100
test ! -e "$SR_PHASE4B_ROOT"
mkdir -p "$SR_PHASE4B_ROOT/workloads"

git -C "$SR_VLLM_SOURCE" checkout --detach \
  752a3a504485790a2e8491cacbb35c137339ad34
test "$(git -C "$SR_VLLM_SOURCE" rev-parse HEAD)" = \
  "752a3a504485790a2e8491cacbb35c137339ad34"

export SR_VLLM_ROOT="$(python - <<'PY'
from importlib import metadata
print(metadata.distribution("vllm").locate_file(""))
PY
)"

nvidia-smi -L | tee "$SR_PHASE4B_ROOT/nvidia-smi-L.txt"
nvidia-smi topo -m | tee "$SR_PHASE4B_ROOT/nvidia-smi-topo.txt"
specrhythm phase4-dual-contract-dry-run \
  --output "$SR_PHASE4B_ROOT/dual-contract-dry-run.json"
```

Build the two-request construction workload and frozen 3/1/1 five-request workload from the
corrected 100 rows. Prompt text and token IDs are unchanged. Only the construction test caps its
output length at eight tokens so it remains short; its own Target-only reference uses the same
cap.

```bash
python - "$SR_R3_100" \
  "$SR_PHASE4B_ROOT/workloads/r3-corrected-2.jsonl" \
  "$SR_PHASE4B_ROOT/workloads/r3-corrected-5.jsonl" <<'PY'
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
assert len(rows) == 100
assert {task: sum(row["task_class"] == task for row in rows)
        for task in ("code", "chat", "summarization")} == {
            "code": 60, "chat": 20, "summarization": 20,
        }

two = []
for task in ("code", "chat"):
    row = dict(next(item for item in rows if item["task_class"] == task))
    row["maximum_new_tokens"] = min(row["maximum_new_tokens"], 8)
    two.append(row)

needed = {"code": 3, "chat": 1, "summarization": 1}
five = []
for row in rows:
    task = row["task_class"]
    if needed.get(task, 0):
        five.append(row)
        needed[task] -= 1
    if not any(needed.values()):
        break

for path, selected in ((sys.argv[2], two), (sys.argv[3], five)):
    with open(path, "x", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    assert all(row["prompt_length"] == len(row["prompt_token_ids"]) for row in selected)
PY

cp --reflink=auto "$SR_R3_100" "$SR_PHASE4B_ROOT/workloads/r3-corrected-100.jsonl" 2>/dev/null || \
cp "$SR_R3_100" "$SR_PHASE4B_ROOT/workloads/r3-corrected-100.jsonl"
find "$SR_PHASE4B_ROOT/workloads" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$SR_PHASE4B_ROOT/workload-sha256-before.txt"
```

## 2. Restore stock vLLM, probe, and freeze two references per workload

```bash
python integrations/vllm/manage_patch.py restore \
  --vllm-root "$SR_VLLM_ROOT" --source "$SR_VLLM_SOURCE" || \
python integrations/vllm/manage_patch.py check \
  --vllm-root "$SR_VLLM_ROOT" --source "$SR_VLLM_SOURCE"

env -u CUDA_VISIBLE_DEVICES VLLM_BATCH_INVARIANT=1 specrhythm phase4-probe \
  --config configs/phase4b_dual_batch_1d2v.yaml \
  --vllm-source "$SR_VLLM_SOURCE" \
  --environment-output "$SR_PHASE4B_ROOT/environment.json" \
  --topology-output "$SR_PHASE4B_ROOT/topology.json" \
  --validation-output "$SR_PHASE4B_ROOT/probe-validation.json"

CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
specrhythm phase4-batch-invariant-preflight \
  --correctness-mode batch-invariant \
  --output "$SR_PHASE4B_ROOT/batch-invariant-preflight.json"

for level in 2 5 100; do
  workload="$SR_PHASE4B_ROOT/workloads/r3-corrected-$level.jsonl"
  for run_id in 1 2; do
    ref_dir="$SR_PHASE4B_ROOT/L$level/reference-$run_id"
    mkdir -p "$ref_dir"
    CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
    specrhythm phase4-stock-reference \
      --config configs/phase4b_dual_batch_1d2v.yaml \
      --correctness-mode batch-invariant \
      --request-count "$level" \
      --workload "$workload" \
      --environment "$SR_PHASE4B_ROOT/environment.json" \
      --topology "$SR_PHASE4B_ROOT/topology.json" \
      --runtime-manifest "$ref_dir/runtime-manifest.json" \
      --output "$ref_dir/stock-target-reference.json" \
      2>&1 | tee "$ref_dir/stock-reference.log"
    chmod a-w "$ref_dir/stock-target-reference.json"
  done
done
```

Archived note: this command block predates the ordered `0001`/`0002`/`0003` patch stack and must
not be used for a new run. Follow the active Gate runbook instead.

```bash
python integrations/vllm/manage_patch.py apply \
  --vllm-root "$SR_VLLM_ROOT" \
  --source "$SR_VLLM_SOURCE" \
  --manifest "$SR_PHASE4B_ROOT/vllm-base-and-patch-manifest.json"
python integrations/vllm/manage_patch.py restore \
  --vllm-root "$SR_VLLM_ROOT" --source "$SR_VLLM_SOURCE"
python integrations/vllm/manage_patch.py apply \
  --vllm-root "$SR_VLLM_ROOT" \
  --source "$SR_VLLM_SOURCE" \
  --manifest "$SR_PHASE4B_ROOT/vllm-base-and-patch-manifest.json"

CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
specrhythm phase4-target-regression \
  --config configs/phase4b_dual_batch_1d2v.yaml \
  --correctness-mode batch-invariant \
  --workload "$SR_PHASE4B_ROOT/workloads/r3-corrected-5.jsonl" \
  --environment "$SR_PHASE4B_ROOT/environment.json" \
  --topology "$SR_PHASE4B_ROOT/topology.json" \
  --runtime-manifest "$SR_PHASE4B_ROOT/patched-target-runtime.json" \
  --reference "$SR_PHASE4B_ROOT/L5/reference-1/stock-target-reference.json" \
  --patch-manifest "$SR_PHASE4B_ROOT/vllm-base-and-patch-manifest.json" \
  --target-diagnostics "$SR_PHASE4B_ROOT/patched-target-diagnostics.jsonl" \
  --output "$SR_PHASE4B_ROOT/patched-target-regression.json"
```

## 3. Run one resumable Dual-Batch collection

The function starts a fresh persistent Draft service. If a Target process is interrupted, kill
the old service, start it again with the same paths, and rerun the Target command with `--resume`.
Only completed request checkpoints are skipped; no in-flight proposal is reconstructed.

For the first-L2 identity-fix recovery, create a new `SR_PHASE4B_ROOT` but export the old immutable
result root as `SR_PHASE4B_INPUT_ROOT` before running this section. If the workload, config,
environment/topology, patch manifest, Target regression and both stock-reference hashes are
unchanged, reuse them through that input root and skip section 2. Never point the new DB run at or
resume the failed L2 directory.

```bash
run_dual_once () {
  level="$1"
  run_id="$2"
  cohort_size="$3"
  run_dir="$SR_PHASE4B_ROOT/L$level/DB-$run_id"
  socket_path="/tmp/sr4b-${SR_PHASE4B_COMMIT:0:8}-L${level}-DB${run_id}.sock"
  mkdir -p "$run_dir"
  unlink "$socket_path" 2>/dev/null || true

  CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
  specrhythm phase4-dual-draft-service \
    --config configs/phase4b_dual_batch_1d2v.yaml \
    --socket "$socket_path" \
    --event-log "$run_dir/draft-work-events.jsonl" \
    --transport-events "$run_dir/transport-events.jsonl" \
    --ready "$run_dir/draft-service-ready.json" \
    >"$run_dir/draft-service.log" 2>&1 &
  draft_pid="$!"

  for _ in $(seq 1 600); do
    test -f "$run_dir/draft-service-ready.json" && test -S "$socket_path" && break
    kill -0 "$draft_pid" 2>/dev/null || { cat "$run_dir/draft-service.log"; return 1; }
    sleep 1
  done
  test -S "$socket_path"

  if phase4b_run_target_with_cleanup \
    "$draft_pid" "$run_dir/target.log" "$run_dir/draft-service.log" -- \
    env CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
    specrhythm phase4-dual-batch-run \
    --config configs/phase4b_dual_batch_1d2v.yaml \
    --workload "$SR_PHASE4B_INPUT_ROOT/workloads/r3-corrected-$level.jsonl" \
    --request-count "$level" \
    --environment "$SR_PHASE4B_INPUT_ROOT/environment.json" \
    --topology "$SR_PHASE4B_INPUT_ROOT/topology.json" \
    --runtime-manifest "$run_dir/runtime-manifest.json" \
    --reference "$SR_PHASE4B_INPUT_ROOT/L$level/reference-1/stock-target-reference.json" \
    --patch-manifest "$SR_PHASE4B_INPUT_ROOT/vllm-base-and-patch-manifest.json" \
    --draft-socket "$socket_path" \
    --draft-ready "$run_dir/draft-service-ready.json" \
    --scheduler-events "$run_dir/scheduler-events.jsonl" \
    --request-state-events "$run_dir/request-state-events.jsonl" \
    --proposal-events "$run_dir/proposal-events.jsonl" \
    --verification-events "$run_dir/verification-events.jsonl" \
    --draft-work-events "$run_dir/draft-work-events.jsonl" \
    --transport-events "$run_dir/transport-events.jsonl" \
    --target-diagnostics "$run_dir/target-diagnostics.jsonl" \
    --plugin-report "$run_dir/dual-proposer-report.json" \
    --output-checkpoint "$run_dir/output-checkpoint.jsonl" \
    --cycle-events "$run_dir/dual-cycle-events.jsonl" \
    --overlap-events "$run_dir/overlap-events.jsonl" \
    --microbatch-size 1 \
    --cohort-size "$cohort_size" \
    --resume \
    --output "$run_dir/dual-batch-run.json"; then
    target_status=0
  else
    target_status="$?"
  fi
  test "$target_status" -eq 0 || return "$target_status"
  test -f "$run_dir/dual-batch-run.json"
  test ! -S "$socket_path"
}

# Level 1: two short requests. Review the cycle log before continuing.
run_dual_once 2 1 2
run_dual_once 2 2 2
```

Stop here until the fresh L2 pair passes the validator below. Do not resume the failed L2
directory, and do not start L5/L100 while this identity-fix gate is unresolved. The helper checks
the Target status first, terminates and reaps Draft on failure, tails both logs, and bounds the
graceful-shutdown wait. Its `/tmp` socket name remains below the AF_UNIX pathname limit; do not
replace it with a result-directory socket.

## 4. Read-only validation for levels 2, 5 and 100

```bash
validate_dual_level () {
  level="$1"
  validation_dir="$SR_PHASE4B_ROOT/L$level/validation"
  mkdir -p "$validation_dir"

  find "$SR_PHASE4B_ROOT/L$level" -type f ! -path '*/validation/*' -print0 | \
    sort -z | xargs -0 sha256sum > "$validation_dir/input-sha256-before.txt"

  set +e
  specrhythm phase4-dual-batch-validate \
    --stock-reference "$SR_PHASE4B_INPUT_ROOT/L$level/reference-1/stock-target-reference.json" \
    --stock-reference "$SR_PHASE4B_INPUT_ROOT/L$level/reference-2/stock-target-reference.json" \
    --target-regression "$SR_PHASE4B_INPUT_ROOT/patched-target-regression.json" \
    --run "$SR_PHASE4B_ROOT/L$level/DB-1/dual-batch-run.json" \
    --run "$SR_PHASE4B_ROOT/L$level/DB-2/dual-batch-run.json" \
    --request-state-events "$SR_PHASE4B_ROOT/L$level/DB-1/request-state-events.jsonl" \
    --request-state-events "$SR_PHASE4B_ROOT/L$level/DB-2/request-state-events.jsonl" \
    --proposal-events "$SR_PHASE4B_ROOT/L$level/DB-1/proposal-events.jsonl" \
    --proposal-events "$SR_PHASE4B_ROOT/L$level/DB-2/proposal-events.jsonl" \
    --cycle-events "$SR_PHASE4B_ROOT/L$level/DB-1/dual-cycle-events.jsonl" \
    --cycle-events "$SR_PHASE4B_ROOT/L$level/DB-2/dual-cycle-events.jsonl" \
    --overlap-events "$SR_PHASE4B_ROOT/L$level/DB-1/overlap-events.jsonl" \
    --overlap-events "$SR_PHASE4B_ROOT/L$level/DB-2/overlap-events.jsonl" \
    --draft-work-events "$SR_PHASE4B_ROOT/L$level/DB-1/draft-work-events.jsonl" \
    --draft-work-events "$SR_PHASE4B_ROOT/L$level/DB-2/draft-work-events.jsonl" \
    --target-diagnostics "$SR_PHASE4B_ROOT/L$level/DB-1/target-diagnostics.jsonl" \
    --target-diagnostics "$SR_PHASE4B_ROOT/L$level/DB-2/target-diagnostics.jsonl" \
    --output "$validation_dir/validation.json" \
    --markdown-output "$validation_dir/summary.md"
  validator_status="$?"
  set -e

  find "$SR_PHASE4B_ROOT/L$level" -type f ! -path '*/validation/*' -print0 | \
    sort -z | xargs -0 sha256sum > "$validation_dir/input-sha256-after.txt"
  diff -u "$validation_dir/input-sha256-before.txt" \
    "$validation_dir/input-sha256-after.txt"
  cat "$validation_dir/summary.md"
  return "$validator_status"
}

validate_dual_level 2

find "$SR_PHASE4B_INPUT_ROOT/workloads" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$SR_PHASE4B_ROOT/workload-sha256-after.txt"
diff -u "$SR_PHASE4B_INPUT_ROOT/workload-sha256-before.txt" \
  "$SR_PHASE4B_ROOT/workload-sha256-after.txt"
```

Only after `L2/validation/validation.json` reports `valid=true` and `outcome=A` may the same
function be used for L5 and L100:

```bash
run_dual_once 5 1 5
run_dual_once 5 2 5
validate_dual_level 5

run_dual_once 100 1 10
run_dual_once 100 2 10
validate_dual_level 100
```

Outcome A requires the five- and 100-request validators to be valid, exact, state/KV/accounting
clean, batch-invariant on every TP rank, and positive-overlap. Outcome B is exact but has no
positive overlap. Outcome C is any correctness failure. Stop on B or C; do not tune timing or
start a performance run.

## 5. Review bundle

```bash
python - "$SR_PHASE4B_ROOT" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
files = []
for path in sorted(root.rglob("*")):
    if path.is_file() and path.stat().st_size < 128 * 1024 * 1024:
        files.append({
            "path": str(path.relative_to(root)),
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
manifest = {
    "schema_version": "specrhythm.phase4b-review-bundle.v1",
    "specrhythm_commit": pathlib.Path("/root/autodl-tmp/src/SpecRhythm/.git/HEAD").read_text().strip(),
    "performance_result": False,
    "files": files,
}
(root / "review-manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

tar -C "$SR_PHASE4B_ROOT" -czf "$SR_PHASE4B_ROOT/review-bundle.tar.gz" \
  review-manifest.json environment.json topology.json probe-validation.json \
  batch-invariant-preflight.json vllm-base-and-patch-manifest.json \
  patched-target-regression.json workload-sha256-before.txt workload-sha256-after.txt \
  L2 L5 L100
sha256sum "$SR_PHASE4B_ROOT/review-bundle.tar.gz" | \
  tee "$SR_PHASE4B_ROOT/review-bundle.sha256"
```

Return all three `validation.json`/`summary.md` pairs, both five-request and both 100-request run
JSONs, overlap/cycle logs, `review-manifest.json`, and the bundle checksum. Then stop. Do not run
Phase 4B.2, packed trees, residual selection, Eager, Shaping, TP3/TP4 or four-GPU evaluation.
