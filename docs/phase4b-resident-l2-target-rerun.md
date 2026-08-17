# Phase 4B observation-serialization fix: L2 Target-only rerun

This is the only active server procedure after the real-A800 `98ec816` failure. It verifies the
preserved partial evidence, then runs one fresh L2 resident Target. It contains no Serial, L5,
Phase 4B.1, performance, Dual-Eager, or SLO command.

## 1. Exact checkout and immutable provenance

The two discovery commands fail if multiple candidate roots exist. In that case, manually export
the exact preserved `SR_FAILED_D6_ROOT` or `SR_FAILED_98_ROOT` and continue from its first `test`.

```bash
set -euo pipefail
cd /root/autodl-tmp/src/SpecRhythm
git fetch origin codex/vllm-serving-v0.1
git switch --detach origin/codex/vllm-serving-v0.1
export SR_PHASE4B_FIX_COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --short)"

conda activate /root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1
python --version
python -m pip install -e '.[dev]' --no-deps

export SR_FAILED_D6_PARENT="/root/autodl-tmp/SpecRhythm-data/results/phase4/d6c7aa8fe80096d72b7a027c5b6cd37ff7a55410"
mapfile -t d6_roots < <(
  find "$SR_FAILED_D6_PARENT" -mindepth 1 -maxdepth 1 -type d \
    -name 'phase4b0-gates-*' | sort
)
printf '%s\n' "${d6_roots[@]}"
test "${#d6_roots[@]}" -eq 1
export SR_FAILED_D6_ROOT="${d6_roots[0]}"

export SR_FAILED_98_PARENT="/root/autodl-tmp/SpecRhythm-data/results/phase4/98ec81641a3e1041a8551e1e98d7008d62b0ab86"
mapfile -t failed_98_roots < <(
  find "$SR_FAILED_98_PARENT" -mindepth 1 -maxdepth 1 -type d \
    -name 'resident-l2-incremental-*' | sort
)
printf '%s\n' "${failed_98_roots[@]}"
test "${#failed_98_roots[@]}" -eq 1
export SR_FAILED_98_ROOT="${failed_98_roots[0]}"

export SR_L2_WORKLOAD="$SR_FAILED_D6_ROOT/workloads/r3-corrected-2.jsonl"
export SR_L2_REFERENCE="$SR_FAILED_D6_ROOT/L2/reference/stock-target-reference.json"
test -f "$SR_L2_WORKLOAD"
test -f "$SR_L2_REFERENCE"
test -f "$SR_FAILED_D6_ROOT/environment.json"
test -f "$SR_FAILED_D6_ROOT/topology.json"
test -f "$SR_FAILED_D6_ROOT/vllm-patch-stack.json"
test -f "$SR_FAILED_98_ROOT/target/timing-events.jsonl"
test -f "$SR_FAILED_98_ROOT/target/admission-events.jsonl"

export SR_PHASE4B_TARGET_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/$SR_PHASE4B_FIX_COMMIT/resident-l2-target-serialization-fix-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$SR_PHASE4B_TARGET_ROOT"
mkdir -p "$SR_PHASE4B_TARGET_ROOT"
source integrations/vllm/phase4b_run_helpers.sh

sha256sum "$SR_L2_WORKLOAD" "$SR_L2_REFERENCE" \
  "$SR_FAILED_D6_ROOT/environment.json" \
  "$SR_FAILED_D6_ROOT/topology.json" \
  "$SR_FAILED_D6_ROOT/vllm-patch-stack.json" \
  "$SR_FAILED_98_ROOT/target/timing-events.jsonl" \
  "$SR_FAILED_98_ROOT/target/admission-events.jsonl" \
  > "$SR_PHASE4B_TARGET_ROOT/immutable-input-sha256-before.txt"
```

The old L2 stock reference is reused read-only. The runner will again fail closed unless its
pinned stock-vLLM runner, model/tokenizer metadata, workload SHA256, sampling, placement, dtype,
context, batch-invariant and DBO/runtime settings match exactly.

## 2. Validate the partial `98ec816` evidence without mutation

```bash
python - "$SR_FAILED_98_ROOT" \
  "$SR_PHASE4B_TARGET_ROOT/failed-98-partial-evidence.json" <<'PY'
import json
import pathlib
import sys

from specrhythm.phase4.manifest import atomic_write_json
from specrhythm.phase4.transport import CheckpointJsonl

failed = pathlib.Path(sys.argv[1]) / "target"
timing = CheckpointJsonl(failed / "timing-events.jsonl").read()
admission = CheckpointJsonl(failed / "admission-events.jsonl").read()
bootstrap = [row for row in timing if row.get("event") == "bootstrap-draft-ready"]
bootstrap_ids = sorted({str(row.get("request_id", "")) for row in bootstrap})
frozen = [
    row for row in admission
    if isinstance(row.get("num_output_tokens"), int)
    and row["num_output_tokens"] >= 1
    and row.get("global_decode_ready") is False
    and row.get("admissible") is False
    and row.get("scheduled") is False
]
report = {
    "schema_version": "specrhythm.phase4b-failed-98-partial-evidence.v1",
    "source_commit": "98ec81641a3e1041a8551e1e98d7008d62b0ab86",
    "read_only_source": True,
    "bootstrap_draft_ready_request_ids": bootstrap_ids,
    "both_l2_observations_recorded": len(bootstrap_ids) == 2,
    "frozen_before_global_readiness_count": len(frozen),
    "at_least_one_request_frozen": bool(frozen),
    "manifest_materialization_completed": False,
    "gpu_correctness_result": False,
    "gpu_performance_result": False,
}
assert report["both_l2_observations_recorded"] is True
assert report["at_least_one_request_frozen"] is True
atomic_write_json(pathlib.Path(sys.argv[2]), report)
print(json.dumps(report, indent=2, sort_keys=True))
PY
```

If either assertion fails, stop and return the two preserved JSONL files. Do not reinterpret the
scheduler dump alone as complete artifact validation.

## 3. Fresh L2 resident Target only

```bash
export SR_L2_TARGET_DIR="$SR_PHASE4B_TARGET_ROOT/target"
export SR_PHASE4_DRAFT_SOCKET="/tmp/sr4b-${SR_PHASE4B_FIX_COMMIT:0:8}-l2-target.sock"
export PHASE4B_LIFECYCLE_ARTIFACT="$SR_L2_TARGET_DIR/process-lifecycle.json"
export PHASE4B_RUN_GUARD="$SR_L2_TARGET_DIR/process-lifecycle.active"
mkdir -p "$SR_L2_TARGET_DIR"
unlink "$SR_PHASE4_DRAFT_SOCKET" 2>/dev/null || true

CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
specrhythm phase4-draft-service \
  --config configs/phase4b_dual_batch_1d2v.yaml \
  --socket "$SR_PHASE4_DRAFT_SOCKET" \
  --event-log "$SR_L2_TARGET_DIR/draft-service-events.jsonl" \
  --ready "$SR_L2_TARGET_DIR/draft-service-ready.json" \
  >"$SR_L2_TARGET_DIR/draft-service.log" 2>&1 &
draft_pid="$!"
for _ in $(seq 1 600); do
  test -S "$SR_PHASE4_DRAFT_SOCKET" && \
    test -f "$SR_L2_TARGET_DIR/draft-service-ready.json" && break
  kill -0 "$draft_pid" 2>/dev/null || {
    cat "$SR_L2_TARGET_DIR/draft-service.log"
    exit 1
  }
  sleep 1
done
test -S "$SR_PHASE4_DRAFT_SOCKET"

phase4b_run_target_with_cleanup \
  "$draft_pid" "$SR_L2_TARGET_DIR/target.log" \
  "$SR_L2_TARGET_DIR/draft-service.log" -- \
  env CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
  specrhythm phase4-resident-target-run \
    --config configs/phase4b_dual_batch_1d2v.yaml \
    --workload "$SR_L2_WORKLOAD" --request-count 2 \
    --correctness-mode batch-invariant \
    --environment "$SR_FAILED_D6_ROOT/environment.json" \
    --topology "$SR_FAILED_D6_ROOT/topology.json" \
    --reference "$SR_L2_REFERENCE" \
    --patch-manifest "$SR_FAILED_D6_ROOT/vllm-patch-stack.json" \
    --draft-socket "$SR_PHASE4_DRAFT_SOCKET" \
    --draft-ready "$SR_L2_TARGET_DIR/draft-service-ready.json" \
    --context "$SR_L2_TARGET_DIR/decode-ready-context.json" \
    --decode-ready-manifest "$SR_L2_TARGET_DIR/decode-ready-manifest.json" \
    --timing-events "$SR_L2_TARGET_DIR/timing-events.jsonl" \
    --setup-control "$SR_L2_TARGET_DIR/setup-control.json" \
    --setup-ready "$SR_L2_TARGET_DIR/setup-ready.json" \
    --admission-events "$SR_L2_TARGET_DIR/admission-events.jsonl" \
    --target-diagnostics "$SR_L2_TARGET_DIR/target-diagnostics.jsonl" \
    --plugin-report "$SR_L2_TARGET_DIR/plugin-report.json" \
    --first-forward "$SR_L2_TARGET_DIR/first-target-forward.json" \
    --output "$SR_L2_TARGET_DIR/resident-target.json"
```

## 4. Target gate validation and integrity

```bash
python - "$SR_L2_TARGET_DIR" <<'PY'
import json
import pathlib
import sys

from specrhythm.phase4.decode_ready import load_decode_ready_manifest
from specrhythm.phase4.transport import CheckpointJsonl

root = pathlib.Path(sys.argv[1])
run = json.loads((root / "resident-target.json").read_text())
ready = json.loads((root / "setup-ready.json").read_text())
lifecycle = json.loads((root / "process-lifecycle.json").read_text())
first = json.loads((root / "first-target-forward.json").read_text())
manifest = load_decode_ready_manifest(
    json.loads((root / "decode-ready-manifest.json").read_text())
)
timing = CheckpointJsonl(root / "timing-events.jsonl").read()
admission = CheckpointJsonl(root / "admission-events.jsonl").read()
bootstrap_ids = {
    str(row.get("request_id", ""))
    for row in timing
    if row.get("event") == "bootstrap-draft-ready"
}
frozen = [
    row for row in admission
    if isinstance(row.get("num_output_tokens"), int)
    and row["num_output_tokens"] >= 1
    and row.get("global_decode_ready") is False
    and row.get("admissible") is False
    and row.get("scheduled") is False
]
released = [
    row for row in admission
    if isinstance(row.get("num_output_tokens"), int)
    and row["num_output_tokens"] >= 1
    and row.get("global_decode_ready") is True
    and row.get("admissible") is True
]
assert run["valid"] is True
assert run["resident_admission"]["valid"] is True
assert run["first_target_forward_valid"] is True
assert run["measurement_boundary_valid"] is True
assert run["raw_vs_decode"]["valid"] is True
assert run["stock_comparison"]["all_sequences_equal"] is True
assert len(manifest.requests) == 2
assert isinstance(manifest.requests[0].logical_committed_prefix_token_ids, tuple)
assert ready["global_decode_ready"] is True
assert ready["consumer"] == "target-only"
assert ready["initial_proposals"] == []
assert len(bootstrap_ids) == 2
assert frozen
assert released
assert first["valid"] is True
assert len(first["requests"]) == 2
assert lifecycle["run_valid"] is True
assert lifecycle["cleanup_valid"] is True
assert lifecycle["remaining_owned_pids"] == []
PY

sha256sum "$SR_L2_WORKLOAD" "$SR_L2_REFERENCE" \
  "$SR_FAILED_D6_ROOT/environment.json" \
  "$SR_FAILED_D6_ROOT/topology.json" \
  "$SR_FAILED_D6_ROOT/vllm-patch-stack.json" \
  "$SR_FAILED_98_ROOT/target/timing-events.jsonl" \
  "$SR_FAILED_98_ROOT/target/admission-events.jsonl" \
  > "$SR_PHASE4B_TARGET_ROOT/immutable-input-sha256-after.txt"
diff -u "$SR_PHASE4B_TARGET_ROOT/immutable-input-sha256-before.txt" \
  "$SR_PHASE4B_TARGET_ROOT/immutable-input-sha256-after.txt"
find "$SR_PHASE4B_TARGET_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$SR_PHASE4B_TARGET_ROOT/artifact-sha256.txt"
echo "$SR_PHASE4B_TARGET_ROOT"
```

Return the fresh Target directory, partial-evidence report, logs, manifests, setup-ready,
timing/admission evidence, first-forward contract, lifecycle artifact and checksums. Stop for
review. Do not run Serial even if every assertion passes.
