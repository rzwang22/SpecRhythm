# Phase 4B.1 real decode-only Dual-Batch correctness runbook

This is the active 3×A800 procedure after Phase 4B.0. It runs correctness only. It must not be
used to report TPOT, latency, throughput, goodput, SLO, speedup or overlap benefit. Stop on the
first nonzero command. Never reuse a run directory, delete an earlier failure, or run Gate 2/3
after an earlier gate fails. In particular, keep the earlier preparation root containing the
intermittent corrected-100 stock nondeterminism failure unchanged as diagnostic provenance. Use a
fresh `SR_PHASE4B_ROOT`; do not repeat that failed freeze in place or in new directories merely to
obtain a favorable pair. A later investigation must be a separately authorized diagnostic run,
not part of this gate procedure.

The real-A800 root ending in
`b9a0d6dc.../phase4b1-gate1-only-20260824T053635Z` is also immutable diagnostic provenance.
Its controlled-2 stock pair was deterministic and its three-layer patch application reached the
expected runner and scheduler hashes. Preparation then stopped because the helper mistakenly ran
a stock-state check against that patched installation. No Target, Serial or Dual consumer was
started, so this root is a control-plane preparation failure, not a Gate 1 correctness failure.
Do not reuse it.

The subsequent fresh root under commit `7e4f8711934fe10cb829e1947e62179c11a0209d`
is likewise immutable. It validates the explicit stock/patched state checks on A800 and contains a
successful resident Target run. Resident Serial then reached its first real speculative Target
verification and stopped in the observational hook because that shared hook accessed the
Dual-only `proposal_id` field on a Serial `Proposal`. Dual-1 and Dual-2 were not started and Gate 1
Outcome was not evaluated. Classify that root exactly as: preparation/control-plane PASS, Target
PASS, Serial diagnostics-compatibility FAIL, both Dual runs NOT RUN, Gate1 NOT EVALUATED.

The observer now records a canonical proposal ID only when the pending protocol actually provides
one. Serial proposals retain `proposal_id=null`; no synthetic Dual identity is created and the
Serial protocol is unchanged. Because this modifies production experiment diagnostics, do not
resume or accept either earlier root. Rerun the complete Section 1–3 chain in a fresh directory
under the new commit, including a new stock pair, Target, Serial, both Dual controls and both
validators.

The complete real-A800 root under `3ee1c3ec4007d3e835bc7d7f385d2d3b5c3c3e8a`
is also immutable. Preparation, Target, Serial diagnostics compatibility, both Dual executions,
the controlled scheduler construction, exact output triangle, keyed repeatability, state/proposal
lifecycle, accounting, verification, Draft sync, target blindness, measurement boundary and
cleanup all passed. Its unified validation still failed: the scheduler validator rejected legal
setup-prefill and the authoritative `legal-target-tail` value, while the controlled two-request
workload produced no positive physical overlap. The helper then masked each nonzero Dual rc after
successful cleanup. Classify this root as controlled construction PASS, exact-token triangle PASS,
semantic components PASS, repeatability PASS, scheduler validator false-positive FAIL, physical
overlap NOT OBSERVED, final Gate1 NOT YET ACHIEVED, performance claim NONE.

The gate structure is now explicit. Gate 1 uses `overlap-requirement=separate-gate`: its outcome
is controlled semantic correctness, while the same report retains `overlap.valid=false` and
forbids an overlap claim. Gate 1.5/Gate 2 uses the default `required` mode with at least five
requests and must produce a positive disjoint-request GPU Draft/Target interval. It is the overlap
existence gate. No sleep, model slowdown, scheduler-state proxy or manufactured interval is
allowed. Gate 3 remains blocked until both earlier gates pass.

The old overlap JSONL stores only actual intersections, so a zero row cannot identify the nearest
non-overlapping pair by itself. `phase4b1-overlap-diagnose` reads the immutable Draft-work,
verification and overlap files together and reports each run's exact nearest host intervals,
signed intersection, separation, ordering and physical placement. Write its output outside the
old root. It is diagnostic-only and never converts scheduler concurrency into physical overlap.

`phase4b1_restore_stock` now emits an immutable `check --expect-state stock` manifest, while
`phase4b1_apply_patch_stack` emits a distinct immutable `check --expect-state patched` manifest.
Each state accepts only its exact pinned runner/scheduler SHA pair; a partial or opposite state
fails closed. For the recovery run, use a fresh root ending in `phase4b1-gate1-only-...`, execute
Sections 1–3 only, confirm Gate 1 Outcome A, and stop before Section 4.

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
export SR_PHASE4B_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/$SR_PHASE4B_COMMIT/phase4b1-gate1-only-$(date -u +%Y%m%dT%H%M%SZ)"
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
mkdir -p "$SR_PHASE4B_ROOT/workloads"
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

## 2. Probe stock vLLM only

This section does not freeze any Gate reference and does not apply the patch stack. References are
measured only immediately before their dependent gate. A failed two-run pair always leaves an
immutable `stock-determinism-diagnostic.json` containing both raw runs and exact per-request first
divergences; it never creates a reference and must not be followed by another freeze attempt in
this gate procedure.

```bash
git -C "$SR_VLLM_SOURCE" checkout --detach \
  752a3a504485790a2e8491cacbb35c137339ad34
test "$(git -C "$SR_VLLM_SOURCE" rev-parse HEAD)" = \
  "752a3a504485790a2e8491cacbb35c137339ad34"

phase4b1_restore_stock "$SR_PHASE4B_ROOT/preparation-stock-probe"

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

```

## 3. Gate 1: two-request controlled correctness

Dual-1 uses the explicitly test-only, non-blocking `one-ready` publication limit to construct
Case A. Dual-2 uses the test-only metadata gate to construct Case B. Neither runs Draft work in
the Target scheduler; both switches are recorded and are not production policy or performance
results. The different output limits construct Case C.

```bash
export SR_GATE1="$SR_PHASE4B_ROOT/Gate-1-controlled-2"
export SR_GATE1_WORKLOAD="$SR_PHASE4B_ROOT/workloads/controlled-2.jsonl"
export SR_GATE1_REFERENCE="$SR_GATE1/reference/stock-target-reference.json"

phase4b1_restore_stock "$SR_GATE1/stock-stage"
phase4b1_freeze_stock_reference "$SR_GATE1/reference" "$SR_GATE1_WORKLOAD" 2
phase4b1_apply_patch_stack "$SR_GATE1/stock-stage"

phase4b1_run_mode target "$SR_GATE1/target" "$SR_GATE1_WORKLOAD" 2 "$SR_GATE1_REFERENCE"
phase4b1_run_mode serial "$SR_GATE1/serial" "$SR_GATE1_WORKLOAD" 2 "$SR_GATE1_REFERENCE"
PHASE4B1_OVERLAP_REQUIREMENT=separate-gate \
phase4b1_run_mode dual "$SR_GATE1/dual-1" "$SR_GATE1_WORKLOAD" 2 \
  "$SR_GATE1_REFERENCE" one-ready
PHASE4B1_OVERLAP_REQUIREMENT=separate-gate \
phase4b1_run_mode dual "$SR_GATE1/dual-2" "$SR_GATE1_WORKLOAD" 2 \
  "$SR_GATE1_REFERENCE" two-ready

specrhythm phase4b1-dual-controlled-validate \
  --asynchronous-scheduler "$SR_GATE1/dual-1/scheduler-events.jsonl" \
  --coordinated-scheduler "$SR_GATE1/dual-2/scheduler-events.jsonl" \
  --request-state-events "$SR_GATE1/dual-1/request-state-events.jsonl" \
  --output "$SR_GATE1/controlled-validation.json"
PHASE4B1_OVERLAP_REQUIREMENT=separate-gate \
phase4b1_validate_gate "$SR_GATE1" "$SR_GATE1/dual-1" "$SR_GATE1/dual-2"
phase4b1_require_outcome_a "$SR_GATE1/validation.json"

python - "$SR_GATE1/validation.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["gate_profile"] == "controlled-correctness"
assert value["overlap_requirement"] == "separate-gate"
assert value["overlap_gate"]["required_for_validation"] is False
assert value["overlap_gate"]["claim_permitted"] is False
PY
```

If any command above fails, preserve `$SR_GATE1` and stop. In particular, a nondeterministic stock
pair is a Gate 1 preparation failure even though its diagnostic artifact was written.
For the current recovery run, also stop after successful `phase4b1_require_outcome_a`; do not run
Sections 4–6. Record a Gate1-only checksum manifest with:

```bash
find "$SR_PHASE4B_ROOT" -type f ! -name 'gate1-artifacts-sha256.txt' -print0 \
  | sort -z | xargs -0 sha256sum > "$SR_PHASE4B_ROOT/gate1-artifacts-sha256.txt"
cat "$SR_GATE1/stock-stage/vllm-stock-check.json"
cat "$SR_GATE1/stock-stage/vllm-patched-check.json"
cat "$SR_GATE1/validation.json"
```

## 4. Gate 1.5 / Gate 2: corrected five-request positive-overlap existence

This gate is not authorized by a Gate 1 correctness result alone. Run it only after review of the
Gate 1 artifacts. The unchanged asynchronous production path must naturally create Draft backlog;
physical overlap remains mandatory and its absence returns nonzero.

```bash
export SR_GATE2="$SR_PHASE4B_ROOT/Gate-2-corrected-5"
export SR_GATE2_WORKLOAD="$SR_PHASE4B_ROOT/workloads/corrected-5.jsonl"
export SR_GATE2_REFERENCE="$SR_GATE2/reference/stock-target-reference.json"

phase4b1_require_outcome_a "$SR_GATE1/validation.json"
phase4b1_restore_stock "$SR_GATE2/stock-stage"
phase4b1_freeze_stock_reference "$SR_GATE2/reference" "$SR_GATE2_WORKLOAD" 5
phase4b1_apply_patch_stack "$SR_GATE2/stock-stage"

phase4b1_run_mode target "$SR_GATE2/target" "$SR_GATE2_WORKLOAD" 5 "$SR_GATE2_REFERENCE"
phase4b1_run_mode serial "$SR_GATE2/serial" "$SR_GATE2_WORKLOAD" 5 "$SR_GATE2_REFERENCE"
PHASE4B1_OVERLAP_REQUIREMENT=required \
phase4b1_run_mode dual "$SR_GATE2/dual-1" "$SR_GATE2_WORKLOAD" 5 \
  "$SR_GATE2_REFERENCE" none
PHASE4B1_OVERLAP_REQUIREMENT=required \
phase4b1_run_mode dual "$SR_GATE2/dual-2" "$SR_GATE2_WORKLOAD" 5 \
  "$SR_GATE2_REFERENCE" none
PHASE4B1_OVERLAP_REQUIREMENT=required \
phase4b1_validate_gate "$SR_GATE2" "$SR_GATE2/dual-1" "$SR_GATE2/dual-2"
phase4b1_require_outcome_a "$SR_GATE2/validation.json"
```

If any command above fails, preserve `$SR_GATE2` and stop. Gate 1 remains valid and untouched.

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
export SR_GATE3_REFERENCE="$SR_GATE3/reference/stock-target-reference.json"

phase4b1_restore_stock "$SR_GATE3/stock-stage"
phase4b1_freeze_stock_reference "$SR_GATE3/reference" "$SR_GATE3_WORKLOAD" 100
phase4b1_apply_patch_stack "$SR_GATE3/stock-stage"

phase4b1_run_mode target "$SR_GATE3/target" "$SR_GATE3_WORKLOAD" 100 "$SR_GATE3_REFERENCE"
phase4b1_run_mode serial "$SR_GATE3/serial" "$SR_GATE3_WORKLOAD" 100 "$SR_GATE3_REFERENCE"
phase4b1_run_mode dual "$SR_GATE3/dual-1" "$SR_GATE3_WORKLOAD" 100 "$SR_GATE3_REFERENCE" none
phase4b1_validate_gate "$SR_GATE3" "$SR_GATE3/dual-1"
phase4b1_require_outcome_a "$SR_GATE3/validation.json"
```

The corrected-100 freeze executes exactly one pair. If it returns nonzero, inspect
`$SR_GATE3/reference/stock-determinism-diagnostic.json`, preserve the complete Gate 3 directory,
and stop. Do not run another freeze to obtain a favorable pair and do not run Gate 3 consumers.

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
and `validation.md`; Gate 1 also contains `controlled-validation.json`. Every attempted gate has
`reference/stock-determinism-diagnostic.json` with both raw stock runs. The immutable stock
reference exists only when that same pair is deterministic. `stock-stage/` records the exact
restore/check and re-applied patch manifests for that gate.

After the commands finish, report the root path and the three validation JSON files, then stop.
Do not start Phase 4B.2, Dual-Eager, packed-tree verification, KVConnector, performance or SLO
experiments.
