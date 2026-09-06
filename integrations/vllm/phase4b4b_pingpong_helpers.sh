#!/usr/bin/env bash

# Source the B/B.1/B.2 helpers first. Runtime mode stays dual; measurement is dual-batch.
phase4b4b_require () {
  python - "$1" "$SR_PHASE4B_COMMIT" <<'PY'
import json, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text())
assert r.get('valid') is True and not r.get('errors'), r.get('errors')
commit = r.get('execution_git_commit', r.get('experiment_identity', {}).get('execution_git_commit'))
assert commit == sys.argv[2], 'Prerequisite must use this exact B commit'
PY
}

phase4b4b_run () {
  local pp_stage="$1" pp_cell="$2" pp_count pp_workload pp_reference pp_dir
  local pp_mode=dual pp_perf=dual-batch pp_size=100 pp_rhythm=legacy
  local pp_args=()
  phase4b4b_require "$SR_PP_ROOT/preflight.json" || return
  case "$pp_stage:$pp_cell" in
    smoke:pingpong)
      pp_count=2; pp_workload="$SR_PP_SMOKE_WORKLOAD"; pp_reference="$SR_PHASE4B3_REFERENCE5"
      pp_dir="$SR_PP_ROOT/smoke/pingpong"; pp_rhythm=pingpong; pp_args=(--smoke) ;;
    five:pingpong)
      phase4b4b_require "$SR_PP_ROOT/smoke/pingpong/qualification.json" || return
      pp_count=5; pp_workload="$SR_PHASE4B3_WORKLOAD5"; pp_reference="$SR_PHASE4B3_REFERENCE5"
      pp_dir="$SR_PP_ROOT/five/pingpong"; pp_rhythm=pingpong ;;
    full:target|full:serial|full:dual-mb2|full:dual-mb100|full:pingpong)
      phase4b4b_require "$SR_PP_ROOT/corrected5-prerequisite.json" || return
      pp_count=100; pp_workload="$SR_PHASE4B_WORKLOAD"; pp_reference="$SR_PHASE4B_REFERENCE"
      pp_dir="$SR_PP_ROOT/$pp_cell"
      case "$pp_cell" in
        target|serial) pp_mode="$pp_cell"; pp_perf="$pp_cell" ;;
        dual-mb2) pp_size=2 ;;
        pingpong) pp_rhythm=pingpong ;;
      esac ;;
    *) echo "Use smoke/five pingpong, then full target/serial/dual-mb2/dual-mb100/pingpong" >&2; return 2 ;;
  esac
  test "$(git rev-parse HEAD)" = "$SR_PHASE4B_COMMIT" || return 2
  test ! -e "$pp_dir" || { echo "Preserve existing cell: $pp_dir" >&2; return 2; }
  if test "$pp_mode" = target; then
    SR_PHASE4_DRAFT_BACKEND=hf-persistent SR_PHASE4B3_HF_METRICS=0 \
      phase4b2_run_mode target "$pp_dir" "$pp_workload" "$pp_count" "$pp_reference" || return
  else
    SR_PHASE4B_DUAL_RHYTHM="$pp_rhythm" SR_PHASE4_DRAFT_BACKEND=vllm-batched \
      SR_PHASE4B_DUAL_MICROBATCH_SIZE="$pp_size" PHASE4B1_OVERLAP_REQUIREMENT=characterization \
      phase4b2_run_mode "$pp_mode" "$pp_dir" "$pp_workload" "$pp_count" "$pp_reference" || return
  fi
  if test "$pp_stage" != smoke; then
    phase4b2_measure_mode "$pp_perf" "$pp_dir" "$pp_workload" || return
  fi
  if test "$pp_cell" = pingpong; then
    python -m specrhythm.phase4.pingpong_comparison validate \
      --run-root "$pp_dir" --workload "$pp_workload" --request-count "$pp_count" \
      "${pp_args[@]}" --output "$pp_dir/qualification.json"
  else
    if test "$pp_mode" = dual; then pp_args=(--microbatch-size "$pp_size"); fi
    python -m specrhythm.phase4.pingpong_comparison control \
      --run-root "$pp_dir" --workload "$pp_workload" --mode "$pp_mode" \
      "${pp_args[@]}" --output "$pp_dir/qualification.json"
  fi
}

phase4b4b_compare () {
  python -m specrhythm.phase4.pingpong_comparison compare --root "$SR_PP_ROOT" \
    --output "$SR_PP_ROOT/comparison.json" --markdown "$SR_PP_ROOT/comparison.md"
}
