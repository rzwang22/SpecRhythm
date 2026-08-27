#!/usr/bin/env bash

# Source after phase4b_run_helpers.sh. These functions collect correctness
# artifacts only. They intentionally expose no latency/performance switch.

phase4b1_require_patch_environment () {
  for phase4b1_name in SR_VLLM_ROOT SR_VLLM_SOURCE; do
    test -n "${!phase4b1_name:-}" || {
      echo "required vLLM patch environment variable is missing: $phase4b1_name" >&2
      return 2
    }
  done
}

phase4b1_require_reference_environment () {
  phase4b1_require_patch_environment || return
  for phase4b1_name in \
      SR_PHASE4B_CONFIG SR_PHASE4B_ENVIRONMENT SR_PHASE4B_TOPOLOGY \
      SR_PHASE4B_COMMIT; do
    test -n "${!phase4b1_name:-}" || {
      echo "required stock-reference environment variable is missing: $phase4b1_name" >&2
      return 2
    }
  done
}

phase4b1_restore_stock () {
  phase4b1_require_patch_environment || return
  phase4b1_stage_dir="$1"
  test ! -e "$phase4b1_stage_dir" || {
    echo "refusing to reuse immutable stock stage: $phase4b1_stage_dir" >&2
    return 2
  }
  mkdir -p "$phase4b1_stage_dir"
  python integrations/vllm/manage_patch.py restore \
    --vllm-root "$SR_VLLM_ROOT" --source "$SR_VLLM_SOURCE"
  python integrations/vllm/manage_patch.py check \
    --vllm-root "$SR_VLLM_ROOT" --source "$SR_VLLM_SOURCE" \
    --expect-state stock \
    --manifest "$phase4b1_stage_dir/vllm-stock-check.json"
}

phase4b1_freeze_stock_reference () {
  phase4b1_require_reference_environment || return
  phase4b1_reference_dir="$1"
  phase4b1_workload="$2"
  phase4b1_count="$3"
  test ! -e "$phase4b1_reference_dir" || {
    echo "refusing to reuse immutable stock reference directory: $phase4b1_reference_dir" >&2
    return 2
  }
  mkdir -p "$phase4b1_reference_dir"
  if ! CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 \
      VLLM_BATCH_INVARIANT=1 \
      python integrations/vllm/phase4b1_stock_reference.py \
        --config "$SR_PHASE4B_CONFIG" \
        --correctness-mode batch-invariant \
        --request-count "$phase4b1_count" \
        --workload "$phase4b1_workload" \
        --environment "$SR_PHASE4B_ENVIRONMENT" \
        --topology "$SR_PHASE4B_TOPOLOGY" \
        --runtime-manifest "$phase4b1_reference_dir/runtime-manifest.json" \
        --determinism-diagnostic \
          "$phase4b1_reference_dir/stock-determinism-diagnostic.json" \
        --output "$phase4b1_reference_dir/stock-target-reference.json" \
        >"$phase4b1_reference_dir/stock-reference.log" 2>&1; then
    cat "$phase4b1_reference_dir/stock-reference.log" >&2
    echo "stock reference failed; preserve this directory and do not retry in place" >&2
    return 1
  fi
  python - "$phase4b1_reference_dir/stock-determinism-diagnostic.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["valid"] is True
assert value["outcome"] == "deterministic"
assert value["retry_count"] == 0
assert value["retry_until_success"] is False
assert value["divergent_request_count"] == 0
assert value["reference_freeze_eligible"] is True
PY
  test -f "$phase4b1_reference_dir/stock-target-reference.json"
}

phase4b1_reuse_stock_reference () {
  phase4b1_source_reference="$1"
  phase4b1_reference_dir="$2"
  phase4b1_workload="$3"
  test -n "${SR_PHASE4B_COMMIT:-}" || {
    echo "SR_PHASE4B_COMMIT is required for stock-reference reuse" >&2
    return 2
  }
  test -f "$phase4b1_source_reference" || {
    echo "source stock reference is missing: $phase4b1_source_reference" >&2
    return 2
  }
  test ! -e "$phase4b1_reference_dir" || {
    echo "refusing to reuse immutable recovery reference directory" >&2
    return 2
  }
  mkdir -p "$phase4b1_reference_dir"
  python - \
      "$phase4b1_source_reference" \
      "$phase4b1_reference_dir/stock-target-reference.json" \
      "$phase4b1_reference_dir/stock-reference-reuse.json" \
      "$phase4b1_workload" \
      "$SR_PHASE4B_COMMIT" <<'PY'
import pathlib
import sys

from specrhythm.phase4.reference import reuse_immutable_stock_reference

reuse_immutable_stock_reference(
    pathlib.Path(sys.argv[1]),
    pathlib.Path(sys.argv[2]),
    pathlib.Path(sys.argv[3]),
    workload_path=pathlib.Path(sys.argv[4]),
    recovery_git_commit=sys.argv[5],
)
PY
}

phase4b1_apply_patch_stack () {
  phase4b1_require_patch_environment || return
  phase4b1_stage_dir="$1"
  test -d "$phase4b1_stage_dir" || {
    echo "patch stage directory does not exist: $phase4b1_stage_dir" >&2
    return 2
  }
  test ! -e "$phase4b1_stage_dir/vllm-patch-stack.json" || {
    echo "refusing to overwrite gate patch manifest" >&2
    return 2
  }
  python integrations/vllm/manage_patch.py apply \
    --vllm-root "$SR_VLLM_ROOT" --source "$SR_VLLM_SOURCE" \
    --manifest "$phase4b1_stage_dir/vllm-patch-stack.json"
  export SR_PHASE4B_PATCH_MANIFEST="$phase4b1_stage_dir/vllm-patch-stack.json"
  python integrations/vllm/manage_patch.py check \
    --vllm-root "$SR_VLLM_ROOT" --source "$SR_VLLM_SOURCE" \
    --expect-state patched \
    --manifest "$phase4b1_stage_dir/vllm-patched-check.json"
}

phase4b1_require_outcome_a () {
  python - "$1" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["valid"] is True
assert value["outcome"] == "A"
assert value["errors"] == []
assert value["input_artifacts_immutable"] is True
PY
}

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
  if test -e "$phase4b1_socket" || test -L "$phase4b1_socket"; then
    echo "refusing to unlink an unverified pre-existing Draft socket: $phase4b1_socket" >&2
    return 2
  fi
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
  phase4b1_overlap_requirement="${PHASE4B1_OVERLAP_REQUIREMENT:-required}"
  test "$phase4b1_overlap_requirement" = required || \
      test "$phase4b1_overlap_requirement" = separate-gate || {
    echo "overlap requirement must be required or separate-gate" >&2
    return 2
  }
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
    if test -n "${PHASE4B1_NUMERICAL_PLAN:-}" || \
        test -n "${PHASE4B1_NUMERICAL_OUTPUT:-}"; then
      test -n "${PHASE4B1_NUMERICAL_PLAN:-}" && \
          test -n "${PHASE4B1_NUMERICAL_OUTPUT:-}" || {
        echo "both PHASE4B1_NUMERICAL_PLAN and PHASE4B1_NUMERICAL_OUTPUT are required" >&2
        return 2
      }
      phase4b1_command+=(
        --numerical-diagnostic-plan "$PHASE4B1_NUMERICAL_PLAN"
        --numerical-diagnostic-output "$PHASE4B1_NUMERICAL_OUTPUT"
      )
    fi
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
      --overlap-requirement "$phase4b1_overlap_requirement"
      --output "$phase4b1_dir/resident-dual.json"
    )
  fi
  if phase4b_run_target_with_cleanup \
      "$PHASE4B1_DRAFT_PID" "$phase4b1_dir/target.log" \
      "$phase4b1_dir/draft-service.log" -- \
      env CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 \
        VLLM_BATCH_INVARIANT=1 "${phase4b1_command[@]}"; then
    phase4b1_run_status=0
  else
    phase4b1_run_status="$?"
  fi
  phase4b1_cleanup_status=0
  if test -S "$phase4b1_socket"; then
    echo "cleanup failed: Draft socket remains: $phase4b1_socket" >&2
    phase4b1_cleanup_status=1
  fi
  if test -e "$PHASE4B_RUN_GUARD"; then
    echo "cleanup failed: lifecycle guard remains: $PHASE4B_RUN_GUARD" >&2
    phase4b1_cleanup_status=1
  fi
  if test "$phase4b1_run_status" -ne 0; then
    return "$phase4b1_run_status"
  fi
  return "$phase4b1_cleanup_status"
}

phase4b1_validate_gate () {
  phase4b1_gate="$1"
  shift
  phase4b1_overlap_requirement="${PHASE4B1_OVERLAP_REQUIREMENT:-required}"
  test "$phase4b1_overlap_requirement" = required || \
      test "$phase4b1_overlap_requirement" = separate-gate || {
    echo "overlap requirement must be required or separate-gate" >&2
    return 2
  }
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
  if "${phase4b1_args[@]}" \
      --overlap-requirement "$phase4b1_overlap_requirement" \
      --output "$phase4b1_gate/validation.json" \
      --markdown-output "$phase4b1_gate/validation.md"; then
    phase4b1_validation_status=0
  else
    phase4b1_validation_status="$?"
  fi
  if test "$phase4b1_validation_status" -ne 0; then
    echo "Gate validator failed with status $phase4b1_validation_status" >&2
    return "$phase4b1_validation_status"
  fi
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
