#!/usr/bin/env bash

# Source the existing B/B.1/B.2 helpers first. No scheduler or runtime flags change.
phase4b3_d6_require () {
  python - "$1" <<'PY'
import json, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text())
assert r.get('valid') is True and not r.get('errors'), r.get('errors')
print('Completed prerequisite valid')
PY
}

phase4b3_d6_run () {
  local sr6_stage="$1" sr6_mode="$2" sr6_count sr6_workload sr6_reference sr6_dir
  local sr6_smoke=()
  phase4b3_d6_require "$SR_D6_ROOT/preflight.json" || return
  case "$sr6_stage:$sr6_mode" in
    A:dual)
      sr6_count=2; sr6_workload="$SR_D6_SMOKE_WORKLOAD"; sr6_reference="$SR_PHASE4B3_REFERENCE5"
      sr6_dir="$SR_D6_ROOT/D6-A/dual"; sr6_smoke=(--smoke) ;;
    B:dual)
      phase4b3_d6_require "$SR_D6_ROOT/D6-A/dual/qualification.json" || return
      sr6_count=5; sr6_workload="$SR_PHASE4B3_WORKLOAD5"; sr6_reference="$SR_PHASE4B3_REFERENCE5"
      sr6_dir="$SR_D6_ROOT/D6-B/dual" ;;
    C:target|C:serial|C:dual)
      phase4b3_d6_require "$SR_D6_ROOT/D6-B/dual/qualification.json" || return
      if test "$sr6_mode" = serial; then
        phase4b3_d6_require "$SR_D6_C_ROOT/target/qualification.json" || return
      elif test "$sr6_mode" = dual; then
        phase4b3_d6_require "$SR_D6_C_ROOT/serial/qualification.json" || return
      fi
      sr6_count=100; sr6_workload="$SR_PHASE4B_WORKLOAD"; sr6_reference="$SR_PHASE4B_REFERENCE"
      sr6_dir="$SR_D6_C_ROOT/$sr6_mode" ;;
    *) echo "Use A/B dual, then C target, C serial, C dual" >&2; return 2 ;;
  esac
  test "$(git rev-parse HEAD)" = "$SR_PHASE4B_COMMIT" || return 2
  if test "$sr6_mode" = target; then
    # Retain historical DecodeReady materialization only; no measured Draft proposals.
    SR_PHASE4_DRAFT_BACKEND=hf-persistent SR_PHASE4B3_HF_METRICS=0 \
      phase4b2_run_mode target "$sr6_dir" "$sr6_workload" "$sr6_count" "$sr6_reference" || return
  elif test "$sr6_stage" = A; then
    # D6-A tests transport/KV/retirement, not speed or overlap. No measurement validator.
    SR_PHASE4_DRAFT_BACKEND=vllm-batched PHASE4B1_OVERLAP_REQUIREMENT=separate-gate \
      phase4b2_run_mode dual "$sr6_dir" "$sr6_workload" "$sr6_count" "$sr6_reference" || return
  else
    SR_PHASE4_DRAFT_BACKEND=vllm-batched PHASE4B1_OVERLAP_REQUIREMENT=required \
      phase4b2_run_mode "$sr6_mode" "$sr6_dir" "$sr6_workload" "$sr6_count" "$sr6_reference" || return
  fi
  if test "$sr6_stage" != A; then
    local sr6_performance_mode="$sr6_mode"
    if test "$sr6_mode" = dual; then
      sr6_performance_mode=dual-batch
    fi
    phase4b2_measure_mode "$sr6_performance_mode" "$sr6_dir" "$sr6_workload" || return
  fi
  python -m specrhythm.phase4.draft_dual_comparison validate \
    --run-root "$sr6_dir" --mode "$sr6_mode" --request-count "$sr6_count" \
    --workload "$sr6_workload" "${sr6_smoke[@]}" --output "$sr6_dir/qualification.json"
}

phase4b3_d6_compare () {
  python -m specrhythm.phase4.draft_dual_comparison compare \
    --root "$SR_D6_C_ROOT" --output "$SR_D6_C_ROOT/comparison.json" \
    --markdown "$SR_D6_C_ROOT/comparison.md"
}
