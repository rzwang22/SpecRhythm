# Phase 4B.1 real decode-only Dual-Batch correctness runbook

This is the active 3×A800 procedure after Phase 4B.0. It runs correctness only. It must not be
used to report TPOT, latency, throughput, goodput, SLO, speedup or overlap benefit. Stop on the
first nonzero command. Never reuse a run directory, delete an earlier failure, or run Gate 2/3
after an earlier gate fails.

## 1. Exact checkout and environment

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
export SR_R3_100="/root/autodl-tmp/SpecRhythm-data/results/phase3c/$SR_PHASE3C_COMMIT/corrected-multiround-100/workload.jsonl"
export SR_PHASE4B_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/$SR_PHASE4B_COMMIT/phase4b1-dual-correctness-$(date -u +%Y%m%dT%H%M%SZ)"
export SR_PHASE4B_CONFIG="$PWD/configs/phase4b_dual_batch_1d2v.yaml"
export SR_VLLM_ROOT="$(python - <<'PY'
from importlib import metadata
print(metadata.distribution("vllm").locate_file(""))
PY
)"

test -f "$SR_DRAFT_MODEL/config.json"
test -f "$SR_TARGET_MODEL/config.json"
test -f "$SR_R3_100"
test "$(wc -l < "$SR_R3_100")" -eq 100
test ! -e "$SR_PHASE4B_ROOT"
mkdir -p "$SR_PHASE4B_ROOT/workloads" "$SR_PHASE4B_ROOT/references"
nvidia-smi -L | tee "$SR_PHASE4B_ROOT/nvidia-smi-L.txt"
nvidia-smi topo -m | tee "$SR_PHASE4B_ROOT/nvidia-smi-topo.txt"

source integrations/vllm/phase4b_run_helpers.sh
source integrations/vllm/phase4b1_gate_helpers.sh

specrhythm phase4-dual-contract-dry-run \
  --output "$SR_PHASE4B_ROOT/dual-contract-dry-run.json"
specrhythm phase4-decode-ready-contract-dry-run \
  --output "$SR_PHASE4B_ROOT/decode-ready-contract-dry-run.json"
```

Create one controlled two-request workload, the corrected 3/1/1 five-request workload and the
unchanged corrected 60/20/20 100-request workload. The controlled requests have different output
limits so one terminates while the other remains live.

```bash
python - "$SR_R3_100" "$SR_PHASE4B_ROOT/workloads" <<'PY'
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2])
rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
assert len(rows) == 100
assert {task: sum(row["task_class"] == task for row in rows) for task in (
    "code", "chat", "summarization"
)} == {"code": 60, "chat": 20, "summarization": 20}

two = [
    dict(next(row for row in rows if row["task_class"] == "code")),
    dict(next(row for row in rows if row["task_class"] == "chat")),
]
two[0]["maximum_new_tokens"] = 4
two[1]["maximum_new_tokens"] = 12
needed = {"code": 3, "chat": 1, "summarization": 1}
five = []
for row in rows:
    task = row["task_class"]
    if needed.get(task, 0):
        five.append(row)
        needed[task] -= 1
    if not any(needed.values()):
        break

for name, selected in (
    ("controlled-2.jsonl", two),
    ("corrected-5.jsonl", five),
    ("corrected-100.jsonl", rows),
):
    path = destination / name
    with path.open("x", encoding="utf-8") as handle:
        for row in selected:
            assert row["prompt_length"] == len(row["prompt_token_ids"])
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
PY
find "$SR_PHASE4B_ROOT/workloads" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$SR_PHASE4B_ROOT/workload-sha256.txt"
```

## 2. Probe, immutable stock references and pinned patch stack

```bash
git -C "$SR_VLLM_SOURCE" checkout --detach \
  752a3a504485790a2e8491cacbb35c137339ad34
test "$(git -C "$SR_VLLM_SOURCE" rev-parse HEAD)" = \
  "752a3a504485790a2e8491cacbb35c137339ad34"

python integrations/vllm/manage_patch.py restore \
  --vllm-root "$SR_VLLM_ROOT" --source "$SR_VLLM_SOURCE"
python integrations/vllm/manage_patch.py check \
  --vllm-root "$SR_VLLM_ROOT" --source "$SR_VLLM_SOURCE" \
  --manifest "$SR_PHASE4B_ROOT/vllm-stock-check.json"

env -u CUDA_VISIBLE_DEVICES VLLM_BATCH_INVARIANT=1 specrhythm phase4-probe \
  --config "$SR_PHASE4B_CONFIG" --vllm-source "$SR_VLLM_SOURCE" \
  --environment-output "$SR_PHASE4B_ROOT/environment.json" \
  --topology-output "$SR_PHASE4B_ROOT/topology.json" \
  --validation-output "$SR_PHASE4B_ROOT/probe-validation.json"
export SR_PHASE4B_ENVIRONMENT="$SR_PHASE4B_ROOT/environment.json"
export SR_PHASE4B_TOPOLOGY="$SR_PHASE4B_ROOT/topology.json"

CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
specrhythm phase4-batch-invariant-preflight \
  --correctness-mode batch-invariant \
  --output "$SR_PHASE4B_ROOT/batch-invariant-preflight.json"

for item in gate1:2:controlled-2 gate2:5:corrected-5 gate3:100:corrected-100; do
  IFS=: read -r gate count stem <<<"$item"
  ref="$SR_PHASE4B_ROOT/references/$gate"
  mkdir -p "$ref"
  CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
  specrhythm phase4-stock-reference \
    --config "$SR_PHASE4B_CONFIG" \
    --correctness-mode batch-invariant --request-count "$count" \
    --workload "$SR_PHASE4B_ROOT/workloads/$stem.jsonl" \
    --environment "$SR_PHASE4B_ENVIRONMENT" --topology "$SR_PHASE4B_TOPOLOGY" \
    --runtime-manifest "$ref/runtime-manifest.json" \
    --output "$ref/stock-target-reference.json" \
    2>&1 | tee "$ref/stock-reference.log"
  chmod a-w "$ref/stock-target-reference.json"
done

python integrations/vllm/manage_patch.py apply \
  --vllm-root "$SR_VLLM_ROOT" --source "$SR_VLLM_SOURCE" \
  --manifest "$SR_PHASE4B_ROOT/vllm-patch-stack.json"
export SR_PHASE4B_PATCH_MANIFEST="$SR_PHASE4B_ROOT/vllm-patch-stack.json"
python integrations/vllm/manage_patch.py check \
  --vllm-root "$SR_VLLM_ROOT" --source "$SR_VLLM_SOURCE" \
  --manifest "$SR_PHASE4B_ROOT/vllm-patched-check.json"
```

## 3. Gate 1: two-request controlled correctness

Dual-1 uses the explicitly test-only, non-blocking `one-ready` publication limit to construct
Case A. Dual-2 uses the test-only metadata gate to construct Case B. Neither runs Draft work in
the Target scheduler; both switches are recorded and are not production policy or performance
results. The different output limits construct Case C.

```bash
export SR_GATE1="$SR_PHASE4B_ROOT/Gate-1-controlled-2"
export SR_GATE1_WORKLOAD="$SR_PHASE4B_ROOT/workloads/controlled-2.jsonl"
export SR_GATE1_REFERENCE="$SR_PHASE4B_ROOT/references/gate1/stock-target-reference.json"

phase4b1_run_mode target "$SR_GATE1/target" "$SR_GATE1_WORKLOAD" 2 "$SR_GATE1_REFERENCE"
phase4b1_run_mode serial "$SR_GATE1/serial" "$SR_GATE1_WORKLOAD" 2 "$SR_GATE1_REFERENCE"
phase4b1_run_mode dual "$SR_GATE1/dual-1" "$SR_GATE1_WORKLOAD" 2 "$SR_GATE1_REFERENCE" one-ready
phase4b1_run_mode dual "$SR_GATE1/dual-2" "$SR_GATE1_WORKLOAD" 2 "$SR_GATE1_REFERENCE" two-ready

specrhythm phase4b1-dual-controlled-validate \
  --asynchronous-scheduler "$SR_GATE1/dual-1/scheduler-events.jsonl" \
  --coordinated-scheduler "$SR_GATE1/dual-2/scheduler-events.jsonl" \
  --request-state-events "$SR_GATE1/dual-1/request-state-events.jsonl" \
  --output "$SR_GATE1/controlled-validation.json"
phase4b1_validate_gate "$SR_GATE1" "$SR_GATE1/dual-1" "$SR_GATE1/dual-2"
```

If any command above fails, preserve `$SR_GATE1` and stop.

## 4. Gate 2: corrected R3-real five requests (3/1/1)

```bash
export SR_GATE2="$SR_PHASE4B_ROOT/Gate-2-corrected-5"
export SR_GATE2_WORKLOAD="$SR_PHASE4B_ROOT/workloads/corrected-5.jsonl"
export SR_GATE2_REFERENCE="$SR_PHASE4B_ROOT/references/gate2/stock-target-reference.json"

phase4b1_run_mode target "$SR_GATE2/target" "$SR_GATE2_WORKLOAD" 5 "$SR_GATE2_REFERENCE"
phase4b1_run_mode serial "$SR_GATE2/serial" "$SR_GATE2_WORKLOAD" 5 "$SR_GATE2_REFERENCE"
phase4b1_run_mode dual "$SR_GATE2/dual-1" "$SR_GATE2_WORKLOAD" 5 "$SR_GATE2_REFERENCE" none
phase4b1_run_mode dual "$SR_GATE2/dual-2" "$SR_GATE2_WORKLOAD" 5 "$SR_GATE2_REFERENCE" none
phase4b1_validate_gate "$SR_GATE2" "$SR_GATE2/dual-1" "$SR_GATE2/dual-2"
```

If any command above fails, preserve `$SR_GATE2` and stop.

## 5. Gate 3: corrected R3-real 100 requests (60/20/20)

Gate 3 is permitted only after Gate 1 and Gate 2 both contain `valid=true`, `outcome=A` and an
empty error list. A single 100-request Dual run is not repeatability evidence; repeatability comes
only from Gate 1 and Gate 2 unless a second 100-request run is explicitly added later.

```bash
python - "$SR_GATE1/validation.json" "$SR_GATE2/validation.json" <<'PY'
import json
import sys
for path in sys.argv[1:]:
    value = json.load(open(path, encoding="utf-8"))
    assert value["valid"] is True and value["outcome"] == "A" and value["errors"] == []
PY

export SR_GATE3="$SR_PHASE4B_ROOT/Gate-3-corrected-100"
export SR_GATE3_WORKLOAD="$SR_PHASE4B_ROOT/workloads/corrected-100.jsonl"
export SR_GATE3_REFERENCE="$SR_PHASE4B_ROOT/references/gate3/stock-target-reference.json"

phase4b1_run_mode target "$SR_GATE3/target" "$SR_GATE3_WORKLOAD" 100 "$SR_GATE3_REFERENCE"
phase4b1_run_mode serial "$SR_GATE3/serial" "$SR_GATE3_WORKLOAD" 100 "$SR_GATE3_REFERENCE"
phase4b1_run_mode dual "$SR_GATE3/dual-1" "$SR_GATE3_WORKLOAD" 100 "$SR_GATE3_REFERENCE" none
phase4b1_validate_gate "$SR_GATE3" "$SR_GATE3/dual-1"
```

## 6. Final immutability manifest and expected artifacts

```bash
find "$SR_PHASE4B_ROOT" -type f ! -name 'all-artifacts-sha256.txt' -print0 \
  | sort -z | xargs -0 sha256sum > "$SR_PHASE4B_ROOT/all-artifacts-sha256.txt"
python - "$SR_PHASE4B_ROOT" <<'PY'
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
for gate in ("Gate-1-controlled-2", "Gate-2-corrected-5", "Gate-3-corrected-100"):
    value = json.loads((root / gate / "validation.json").read_text())
    assert value["valid"] is True
    assert value["outcome"] == "A"
    assert value["errors"] == []
    assert value["input_artifacts_immutable"] is True
print(root)
PY
```

Every mode directory contains its run JSON, decode-ready manifest, setup control/ready, timing,
Target diagnostics, plugin report and process-lifecycle artifact. Dual directories additionally
contain request-state, proposal-round, proposal-lifecycle, scheduler, verification, Draft-work,
transport, cycle, overlap and output-checkpoint artifacts. Every gate contains `validation.json`
and `validation.md`; Gate 1 also contains `controlled-validation.json`.

After the commands finish, report the root path and the three validation JSON files, then stop.
Do not start Phase 4B.2, Dual-Eager, packed-tree verification, KVConnector, performance or SLO
experiments.
