# Phase 4B initial-proposal lifecycle fix: L2 Serial-only rerun

This is the only active server procedure after the diagnostic-only resident Serial failure at
`5db8657`. It reuses the successful L2 Target result read-only, runs exactly one fresh L2
resident Serial, and validates the pair only after Serial passes. It contains no Target rerun,
L5, Phase 4B.1, performance, Dual-Eager, or SLO command.

The old Serial attempt is not acceptance evidence: its shell launched Draft twice and overwrote
the live PID with the failed duplicate PID. It remains immutable diagnostic provenance.

The preserved diagnostic chronology localizes the code path without turning the run into a pass:

- A and B bootstrap/Draft-ready timestamps were `5365005651243716` and `5365005765936579`;
- pre-barrier validation and measurement start followed at `5365005766248304` and
  `5365005767987068`;
- the earliest initial Draft began later, at `5365005770096985`;
- setup-ready published at `5365006013244370`;
- both proposal parent lengths and SHA256 values equal their DecodeReady manifest prefixes;
- cycle 2 records both proposals installed and both requests scheduled before the later stale-parent
  exception.

At pinned vLLM commit `752a3a5`, stock scheduling adds an entry to
`scheduled_spec_decode_tokens` only for speculative tokens included in that scheduler output, then
clears the request's installed `spec_token_ids`. The lifecycle fix therefore consumes round zero
from that output mapping and never infers consumption merely from prior installation.

## 1. Exact checkout and immutable inputs

The discovery checks intentionally fail if they find anything other than one matching root. In
that case, export the exact preserved path manually and continue from its corresponding `test`.

```bash
set -euo pipefail
cd /root/autodl-tmp/src/SpecRhythm
git fetch origin codex/vllm-serving-v0.1
git switch --detach origin/codex/vllm-serving-v0.1
export SR_PHASE4B_SERIAL_FIX_COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --short)"

conda activate /root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1
python --version
python -m pip install -e '.[dev]' --no-deps

export SR_D6_PARENT="/root/autodl-tmp/SpecRhythm-data/results/phase4/d6c7aa8fe80096d72b7a027c5b6cd37ff7a55410"
mapfile -t d6_roots < <(
  find "$SR_D6_PARENT" -mindepth 1 -maxdepth 1 -type d \
    -name 'phase4b0-gates-*' | sort
)
printf '%s\n' "${d6_roots[@]}"
test "${#d6_roots[@]}" -eq 1
export SR_D6_ROOT="${d6_roots[0]}"

export SR_5DB_PARENT="/root/autodl-tmp/SpecRhythm-data/results/phase4/5db86575835d58801fdf9575f952730e46eae035"
mapfile -t successful_target_files < <(
  find "$SR_5DB_PARENT" -type f \
    -path '*/resident-l2-target-serialization-fix-*/target/resident-target.json' | sort
)
printf '%s\n' "${successful_target_files[@]}"
test "${#successful_target_files[@]}" -eq 1
export SR_PRESERVED_TARGET_DIR="$(dirname "${successful_target_files[0]}")"

mapfile -t failed_serial_logs < <(
  find "$SR_5DB_PARENT" -type f -path '*/serial/target.log' | sort
)
printf '%s\n' "${failed_serial_logs[@]}"
test "${#failed_serial_logs[@]}" -eq 1
export SR_FAILED_SERIAL_DIR="$(dirname "${failed_serial_logs[0]}")"

export SR_L2_WORKLOAD="$SR_D6_ROOT/workloads/r3-corrected-2.jsonl"
export SR_L2_REFERENCE="$SR_D6_ROOT/L2/reference/stock-target-reference.json"
test -f "$SR_L2_WORKLOAD"
test -f "$SR_L2_REFERENCE"
test -f "$SR_D6_ROOT/environment.json"
test -f "$SR_D6_ROOT/topology.json"
test -f "$SR_D6_ROOT/vllm-patch-stack.json"
test -f "$SR_PRESERVED_TARGET_DIR/resident-target.json"
test -f "$SR_PRESERVED_TARGET_DIR/decode-ready-manifest.json"
test -f "$SR_PRESERVED_TARGET_DIR/process-lifecycle.json"
test -f "$SR_FAILED_SERIAL_DIR/target.log"

test ! -e "$SR_FAILED_SERIAL_DIR/process-lifecycle.active"
if ps -eo args= | grep -E '[s]pecrhythm phase4|[v]llm' >/dev/null; then
  echo 'A Phase-4/vLLM process is still running; stop and inspect it.' >&2
  exit 1
fi

export SR_PHASE4B_SERIAL_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/$SR_PHASE4B_SERIAL_FIX_COMMIT/resident-l2-serial-lifecycle-fix-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$SR_PHASE4B_SERIAL_ROOT"
mkdir -p "$SR_PHASE4B_SERIAL_ROOT"
source integrations/vllm/phase4b_run_helpers.sh

find "$SR_PRESERVED_TARGET_DIR" "$SR_FAILED_SERIAL_DIR" -type f -print0 \
  | sort -z | xargs -0 sha256sum \
  > "$SR_PHASE4B_SERIAL_ROOT/immutable-evidence-sha256-before.txt"
sha256sum "$SR_L2_WORKLOAD" "$SR_L2_REFERENCE" \
  "$SR_D6_ROOT/environment.json" \
  "$SR_D6_ROOT/topology.json" \
  "$SR_D6_ROOT/vllm-patch-stack.json" \
  > "$SR_PHASE4B_SERIAL_ROOT/immutable-input-sha256-before.txt"
```

## 2. Validate the preserved successful Target before reuse

```bash
python - "$SR_PRESERVED_TARGET_DIR" \
  "$SR_PHASE4B_SERIAL_ROOT/preserved-target-validation.json" <<'PY'
import json
import pathlib
import sys

from specrhythm.phase4.manifest import atomic_write_json

root = pathlib.Path(sys.argv[1])
run = json.loads((root / "resident-target.json").read_text())
ready = json.loads((root / "setup-ready.json").read_text())
lifecycle = json.loads((root / "process-lifecycle.json").read_text())
report = {
    "schema_version": "specrhythm.phase4b-preserved-target-reuse.v1",
    "source_commit": "5db86575835d58801fdf9575f952730e46eae035",
    "source_directory": root.name,
    "read_only": True,
    "target_valid": run.get("valid") is True,
    "resident_admission_valid": run.get("resident_admission", {}).get("valid") is True,
    "first_target_forward_valid": run.get("first_target_forward_valid") is True,
    "measurement_boundary_valid": run.get("measurement_boundary_valid") is True,
    "raw_vs_decode_valid": run.get("raw_vs_decode", {}).get("valid") is True,
    "stock_sequences_equal": run.get("stock_comparison", {}).get("all_sequences_equal") is True,
    "target_only_ready": ready.get("consumer") == "target-only",
    "lifecycle_cleanup_valid": lifecycle.get("cleanup_valid") is True,
    "remaining_owned_pids": lifecycle.get("remaining_owned_pids"),
    "gpu_performance_result": False,
}
assert all(
    report[name] is True
    for name in (
        "target_valid",
        "resident_admission_valid",
        "first_target_forward_valid",
        "measurement_boundary_valid",
        "raw_vs_decode_valid",
        "stock_sequences_equal",
        "target_only_ready",
        "lifecycle_cleanup_valid",
    )
)
assert report["remaining_owned_pids"] == []
atomic_write_json(pathlib.Path(sys.argv[2]), report)
print(json.dumps(report, indent=2, sort_keys=True))
PY
```

The fix is confined to the `consumer == "serial"` initial-proposal lifecycle and its Serial
artifact plumbing. Target-only creates no lifecycle object, requires no new artifact, and retains
the same admission and stock scheduling path. The successful Target run is therefore safe to
reuse as read-only correctness evidence.

## 3. Run one fresh L2 resident Serial

This block starts Draft exactly once. `draft_pid` is read-only, and an EXIT trap owns cleanup if
any subsequent assertion fails.

```bash
export SR_L2_SERIAL_DIR="$SR_PHASE4B_SERIAL_ROOT/serial"
export SR_PHASE4_DRAFT_SOCKET="/tmp/sr4b-${SR_PHASE4B_SERIAL_FIX_COMMIT:0:8}-l2-serial.sock"
export PHASE4B_LIFECYCLE_ARTIFACT="$SR_L2_SERIAL_DIR/process-lifecycle.json"
export PHASE4B_RUN_GUARD="$SR_L2_SERIAL_DIR/process-lifecycle.active"
mkdir -p "$SR_L2_SERIAL_DIR"
test ! -S "$SR_PHASE4_DRAFT_SOCKET"
test ! -e "$PHASE4B_RUN_GUARD"

CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
specrhythm phase4-draft-service \
  --config configs/phase4b_dual_batch_1d2v.yaml \
  --socket "$SR_PHASE4_DRAFT_SOCKET" \
  --event-log "$SR_L2_SERIAL_DIR/draft-service-events.jsonl" \
  --ready "$SR_L2_SERIAL_DIR/draft-service-ready.json" \
  >"$SR_L2_SERIAL_DIR/draft-service.log" 2>&1 &
draft_pid="$!"
readonly draft_pid
trap 'phase4b_terminate_and_wait "$draft_pid"' EXIT

for _ in $(seq 1 600); do
  test -S "$SR_PHASE4_DRAFT_SOCKET" && \
    test -f "$SR_L2_SERIAL_DIR/draft-service-ready.json" && break
  kill -0 "$draft_pid" 2>/dev/null || {
    cat "$SR_L2_SERIAL_DIR/draft-service.log"
    exit 1
  }
  sleep 1
done
test -S "$SR_PHASE4_DRAFT_SOCKET"
test -f "$SR_L2_SERIAL_DIR/draft-service-ready.json"
test "$(jobs -pr | grep -cx "$draft_pid")" -eq 1

phase4b_run_target_with_cleanup \
  "$draft_pid" "$SR_L2_SERIAL_DIR/target.log" \
  "$SR_L2_SERIAL_DIR/draft-service.log" -- \
  env CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
  specrhythm phase4-resident-serial-run \
    --config configs/phase4b_dual_batch_1d2v.yaml \
    --workload "$SR_L2_WORKLOAD" --request-count 2 \
    --correctness-mode batch-invariant \
    --environment "$SR_D6_ROOT/environment.json" \
    --topology "$SR_D6_ROOT/topology.json" \
    --runtime-manifest "$SR_L2_SERIAL_DIR/runtime-manifest.json" \
    --reference "$SR_L2_REFERENCE" \
    --patch-manifest "$SR_D6_ROOT/vllm-patch-stack.json" \
    --draft-socket "$SR_PHASE4_DRAFT_SOCKET" \
    --draft-ready "$SR_L2_SERIAL_DIR/draft-service-ready.json" \
    --round-events "$SR_L2_SERIAL_DIR/round-events.jsonl" \
    --transport-events "$SR_L2_SERIAL_DIR/transport-events.jsonl" \
    --plugin-report "$SR_L2_SERIAL_DIR/plugin-report.json" \
    --context "$SR_L2_SERIAL_DIR/decode-ready-context.json" \
    --decode-ready-manifest "$SR_L2_SERIAL_DIR/decode-ready-manifest.json" \
    --timing-events "$SR_L2_SERIAL_DIR/timing-events.jsonl" \
    --setup-control "$SR_L2_SERIAL_DIR/setup-control.json" \
    --setup-ready "$SR_L2_SERIAL_DIR/setup-ready.json" \
    --admission-events "$SR_L2_SERIAL_DIR/admission-events.jsonl" \
    --initial-proposal-events "$SR_L2_SERIAL_DIR/initial-proposal-events.jsonl" \
    --target-diagnostics "$SR_L2_SERIAL_DIR/target-diagnostics.jsonl" \
    --first-forward "$SR_L2_SERIAL_DIR/first-target-forward.json" \
    --output "$SR_L2_SERIAL_DIR/resident-serial.json"

trap - EXIT
test ! -S "$SR_PHASE4_DRAFT_SOCKET"
test ! -e "$PHASE4B_RUN_GUARD"
```

## 4. Serial gate, then read-only Target-vs-Serial validation

```bash
python - "$SR_L2_SERIAL_DIR" <<'PY'
import json
import pathlib
import sys

from specrhythm.phase4.resident_initial_proposal import (
    validate_initial_proposal_lifecycle_events,
)
from specrhythm.phase4.transport import CheckpointJsonl

root = pathlib.Path(sys.argv[1])
run = json.loads((root / "resident-serial.json").read_text())
ready = json.loads((root / "setup-ready.json").read_text())
lifecycle = json.loads((root / "process-lifecycle.json").read_text())
proposal_rows = CheckpointJsonl(root / "initial-proposal-events.jsonl").read()
proposal_errors = validate_initial_proposal_lifecycle_events(
    proposal_rows,
    expected_request_ids=(
        "r3-6a6186801558c5cd3a48869f",
        "r3-86d740144712e45992f62adc",
    ),
)
assert run["valid"] is True
assert run["resident_admission"]["valid"] is True
assert run["resident_initial_proposal_lifecycle"]["valid"] is True
assert run["first_target_forward_valid"] is True
assert run["measurement_boundary_valid"] is True
assert run["raw_vs_decode"]["valid"] is True
assert run["comparison"]["all_sequences_equal"] is True
assert ready["global_decode_ready"] is True
assert ready["consumer"] == "serial"
assert len(ready["initial_proposals"]) == 2
assert proposal_errors == []
assert lifecycle["run_valid"] is True
assert lifecycle["cleanup_valid"] is True
assert lifecycle["remaining_owned_pids"] == []
PY

specrhythm phase4-resident-validate \
  --target "$SR_PRESERVED_TARGET_DIR/resident-target.json" \
  --serial "$SR_L2_SERIAL_DIR/resident-serial.json" \
  --target-manifest "$SR_PRESERVED_TARGET_DIR/decode-ready-manifest.json" \
  --serial-manifest "$SR_L2_SERIAL_DIR/decode-ready-manifest.json" \
  --output "$SR_PHASE4B_SERIAL_ROOT/resident-l2-validation.json"

python - "$SR_PHASE4B_SERIAL_ROOT/resident-l2-validation.json" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["valid"] is True
assert value["target_equals_serial"]["valid"] is True
assert value["dual_evaluated"] is False
assert value["performance_result"] is False
PY

find "$SR_PRESERVED_TARGET_DIR" "$SR_FAILED_SERIAL_DIR" -type f -print0 \
  | sort -z | xargs -0 sha256sum \
  > "$SR_PHASE4B_SERIAL_ROOT/immutable-evidence-sha256-after.txt"
diff -u "$SR_PHASE4B_SERIAL_ROOT/immutable-evidence-sha256-before.txt" \
  "$SR_PHASE4B_SERIAL_ROOT/immutable-evidence-sha256-after.txt"

sha256sum "$SR_L2_WORKLOAD" "$SR_L2_REFERENCE" \
  "$SR_D6_ROOT/environment.json" \
  "$SR_D6_ROOT/topology.json" \
  "$SR_D6_ROOT/vllm-patch-stack.json" \
  > "$SR_PHASE4B_SERIAL_ROOT/immutable-input-sha256-after.txt"
diff -u "$SR_PHASE4B_SERIAL_ROOT/immutable-input-sha256-before.txt" \
  "$SR_PHASE4B_SERIAL_ROOT/immutable-input-sha256-after.txt"

find "$SR_PHASE4B_SERIAL_ROOT" -type f ! -name artifact-sha256.txt -print0 \
  | sort -z | xargs -0 sha256sum \
  > "$SR_PHASE4B_SERIAL_ROOT/artifact-sha256.txt"
echo "$SR_PHASE4B_SERIAL_ROOT"
```

Return the fresh Serial directory, lifecycle events, pair validation, preserved-Target validation,
process lifecycle artifact, logs, and checksums. Stop for review even if every assertion passes.
