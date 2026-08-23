#!/usr/bin/env bash

# Source after phase4b_run_helpers.sh. These functions collect correctness
# artifacts only. They intentionally expose no latency/performance switch.

phase4b1_require_environment () {
  for phase4b1_name in \
      SR_PHASE4B_CONFIG SR_PHASE4B_ENVIRONMENT SR_PHASE4B_TOPOLOGY \
      SR_PHASE4B_PATCH_MANIFEST SR_PHASE4B_COMMIT; do
    test -n "${!phase4b1_name:-}" || {
      echo "required environment variable is missing: $phase4b1_name" >&2
      return 2
    }
  done
}

phase4b1_start_draft () {
  phase4b1_kind="$1"
  phase4b1_dir="$2"
  phase4b1_socket="$3"
  unlink "$phase4b1_socket" 2>/dev/null || true
  if test "$phase4b1_kind" = dual; then
    CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
      specrhythm phase4-dual-draft-service \
        --config "$SR_PHASE4B_CONFIG" \
        --socket "$phase4b1_socket" \
        --event-log "$phase4b1_dir/draft-work-events.jsonl" \
        --transport-events "$phase4b1_dir/transport-events.jsonl" \
        --ready "$phase4b1_dir/draft-service-ready.json" \
        >"$phase4b1_dir/draft-service.log" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
      specrhythm phase4-draft-service \
        --config "$SR_PHASE4B_CONFIG" \
        --socket "$phase4b1_socket" \
        --event-log "$phase4b1_dir/draft-service-events.jsonl" \
        --ready "$phase4b1_dir/draft-service-ready.json" \
        >"$phase4b1_dir/draft-service.log" 2>&1 &
  fi
  PHASE4B1_DRAFT_PID="$!"
  export PHASE4B1_DRAFT_PID
  for _ in $(seq 1 600); do
    test -S "$phase4b1_socket" && return 0
    kill -0 "$PHASE4B1_DRAFT_PID" 2>/dev/null || {
      cat "$phase4b1_dir/draft-service.log" >&2
      return 1
    }
    sleep 1
  done
  echo "Draft service did not publish its socket" >&2
  phase4b_terminate_and_wait "$PHASE4B1_DRAFT_PID"
  return 124
}

phase4b1_run_mode () {
  phase4b1_require_environment || return
  phase4b1_mode="$1"
  phase4b1_dir="$2"
  phase4b1_workload="$3"
  phase4b1_count="$4"
  phase4b1_reference="$5"
  phase4b1_coordination="${6:-none}"
  test "$phase4b1_mode" = target || test "$phase4b1_mode" = serial || \
      test "$phase4b1_mode" = dual || {
    echo "mode must be target, serial, or dual" >&2
    return 2
  }
  test ! -e "$phase4b1_dir" || {
    echo "refusing to reuse immutable run directory: $phase4b1_dir" >&2
    return 2
  }
  mkdir -p "$phase4b1_dir"
  phase4b1_socket="/tmp/sr4b1-${SR_PHASE4B_COMMIT:0:8}-$(basename "$phase4b1_dir").sock"
  phase4b1_kind=stock
  test "$phase4b1_mode" = dual && phase4b1_kind=dual
  phase4b1_start_draft "$phase4b1_kind" "$phase4b1_dir" "$phase4b1_socket" || return
  unset SR_PHASE4_DRAFT_SOCKET SR_PHASE4_DUAL_DRAFT_SOCKET
  if test "$phase4b1_mode" = dual; then
    export SR_PHASE4_DUAL_DRAFT_SOCKET="$phase4b1_socket"
  else
    export SR_PHASE4_DRAFT_SOCKET="$phase4b1_socket"
  fi
  export PHASE4B_LIFECYCLE_ARTIFACT="$phase4b1_dir/process-lifecycle.json"
  export PHASE4B_RUN_GUARD="$phase4b1_dir/process-lifecycle.active"
  if test "$phase4b1_mode" = target; then
    phase4b1_command=(
      specrhythm phase4-resident-target-run
      --config "$SR_PHASE4B_CONFIG"
      --workload "$phase4b1_workload" --request-count "$phase4b1_count"
      --correctness-mode batch-invariant
      --environment "$SR_PHASE4B_ENVIRONMENT"
      --topology "$SR_PHASE4B_TOPOLOGY"
      --reference "$phase4b1_reference"
      --patch-manifest "$SR_PHASE4B_PATCH_MANIFEST"
      --draft-socket "$phase4b1_socket"
      --draft-ready "$phase4b1_dir/draft-service-ready.json"
      --context "$phase4b1_dir/decode-ready-context.json"
      --decode-ready-manifest "$phase4b1_dir/decode-ready-manifest.json"
      --timing-events "$phase4b1_dir/timing-events.jsonl"
      --setup-control "$phase4b1_dir/setup-control.json"
      --setup-ready "$phase4b1_dir/setup-ready.json"
      --admission-events "$phase4b1_dir/admission-events.jsonl"
      --target-diagnostics "$phase4b1_dir/target-diagnostics.jsonl"
      --plugin-report "$phase4b1_dir/plugin-report.json"
      --first-forward "$phase4b1_dir/first-target-forward.json"
      --output "$phase4b1_dir/resident-target.json"
    )
  elif test "$phase4b1_mode" = serial; then
    phase4b1_command=(
      specrhythm phase4-resident-serial-run
      --config "$SR_PHASE4B_CONFIG"
      --workload "$phase4b1_workload" --request-count "$phase4b1_count"
      --correctness-mode batch-invariant
      --environment "$SR_PHASE4B_ENVIRONMENT"
      --topology "$SR_PHASE4B_TOPOLOGY"
      --runtime-manifest "$phase4b1_dir/runtime-manifest.json"
      --reference "$phase4b1_reference"
      --patch-manifest "$SR_PHASE4B_PATCH_MANIFEST"
      --draft-socket "$phase4b1_socket"
      --draft-ready "$phase4b1_dir/draft-service-ready.json"
      --round-events "$phase4b1_dir/round-events.jsonl"
      --transport-events "$phase4b1_dir/transport-events.jsonl"
      --plugin-report "$phase4b1_dir/plugin-report.json"
      --context "$phase4b1_dir/decode-ready-context.json"
      --decode-ready-manifest "$phase4b1_dir/decode-ready-manifest.json"
      --timing-events "$phase4b1_dir/timing-events.jsonl"
      --setup-control "$phase4b1_dir/setup-control.json"
      --setup-ready "$phase4b1_dir/setup-ready.json"
      --admission-events "$phase4b1_dir/admission-events.jsonl"
      --initial-proposal-events "$phase4b1_dir/initial-proposal-events.jsonl"
      --target-diagnostics "$phase4b1_dir/target-diagnostics.jsonl"
      --first-forward "$phase4b1_dir/first-target-forward.json"
      --output "$phase4b1_dir/resident-serial.json"
    )
  else
    phase4b1_command=(
      specrhythm phase4b1-resident-dual-run
      --config "$SR_PHASE4B_CONFIG"
      --workload "$phase4b1_workload" --request-count "$phase4b1_count"
      --environment "$SR_PHASE4B_ENVIRONMENT"
      --topology "$SR_PHASE4B_TOPOLOGY"
      --patch-manifest "$SR_PHASE4B_PATCH_MANIFEST"
      --draft-socket "$phase4b1_socket"
      --draft-ready "$phase4b1_dir/draft-service-ready.json"
      --context "$phase4b1_dir/decode-ready-context.json"
      --decode-ready-manifest "$phase4b1_dir/decode-ready-manifest.json"
      --timing-events "$phase4b1_dir/timing-events.jsonl"
      --setup-control "$phase4b1_dir/setup-control.json"
      --setup-ready "$phase4b1_dir/setup-ready.json"
      --scheduler-events "$phase4b1_dir/scheduler-events.jsonl"
      --request-state-events "$phase4b1_dir/request-state-events.jsonl"
      --proposal-events "$phase4b1_dir/proposal-events.jsonl"
      --proposal-lifecycle-events "$phase4b1_dir/proposal-lifecycle-events.jsonl"
      --verification-events "$phase4b1_dir/verification-events.jsonl"
      --draft-work-events "$phase4b1_dir/draft-work-events.jsonl"
      --transport-events "$phase4b1_dir/transport-events.jsonl"
      --target-diagnostics "$phase4b1_dir/target-diagnostics.jsonl"
      --plugin-report "$phase4b1_dir/plugin-report.json"
      --output-checkpoint "$phase4b1_dir/output-checkpoint.jsonl"
      --cycle-events "$phase4b1_dir/cycle-events.jsonl"
      --overlap-events "$phase4b1_dir/overlap-events.jsonl"
      --runtime-manifest "$phase4b1_dir/runtime-manifest.json"
      --microbatch-size 2 --test-coordination "$phase4b1_coordination"
      --output "$phase4b1_dir/resident-dual.json"
    )
  fi
  phase4b_run_target_with_cleanup \
    "$PHASE4B1_DRAFT_PID" "$phase4b1_dir/target.log" \
    "$phase4b1_dir/draft-service.log" -- \
    env CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 \
      VLLM_BATCH_INVARIANT=1 "${phase4b1_command[@]}"
  test ! -S "$phase4b1_socket"
  test ! -e "$PHASE4B_RUN_GUARD"
}

phase4b1_validate_gate () {
  phase4b1_gate="$1"
  shift
  phase4b1_target="$phase4b1_gate/target"
  phase4b1_serial="$phase4b1_gate/serial"
  phase4b1_args=(
    specrhythm phase4b1-dual-correctness-validate
    --target "$phase4b1_target/resident-target.json"
    --serial "$phase4b1_serial/resident-serial.json"
    --target-manifest "$phase4b1_target/decode-ready-manifest.json"
    --serial-manifest "$phase4b1_serial/decode-ready-manifest.json"
    --target-process-lifecycle "$phase4b1_target/process-lifecycle.json"
    --serial-process-lifecycle "$phase4b1_serial/process-lifecycle.json"
  )
  for phase4b1_dual in "$@"; do
    phase4b1_args+=(
      --dual "$phase4b1_dual/resident-dual.json"
      --dual-manifest "$phase4b1_dual/decode-ready-manifest.json"
      --request-state-events "$phase4b1_dual/request-state-events.jsonl"
      --proposal-events "$phase4b1_dual/proposal-events.jsonl"
      --proposal-lifecycle-events "$phase4b1_dual/proposal-lifecycle-events.jsonl"
      --scheduler-events "$phase4b1_dual/scheduler-events.jsonl"
      --verification-events "$phase4b1_dual/verification-events.jsonl"
      --draft-work-events "$phase4b1_dual/draft-work-events.jsonl"
      --target-diagnostics "$phase4b1_dual/target-diagnostics.jsonl"
      --overlap-events "$phase4b1_dual/overlap-events.jsonl"
      --process-lifecycle "$phase4b1_dual/process-lifecycle.json"
    )
  done
  "${phase4b1_args[@]}" \
    --output "$phase4b1_gate/validation.json" \
    --markdown-output "$phase4b1_gate/validation.md"
  python - "$phase4b1_gate/validation.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["valid"] is True
assert value["outcome"] == "A"
assert value["errors"] == []
assert value["input_artifacts_immutable"] is True
PY
}
