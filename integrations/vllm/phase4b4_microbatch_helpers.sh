#!/usr/bin/env bash

# Source B/B.1/B.2 helpers first. Runtime and measurement use their existing names.
phase4b4_run () {
  local sr4_mode="$1" sr4_size="${2:-}" sr4_name="$1" sr4_perf="$1"
  local sr4_args=()
  case "$sr4_mode" in
    target|serial) ;;
    dual)
      sr4_size="$(SR_PHASE4B_DUAL_MICROBATCH_SIZE="$sr4_size" python -m specrhythm.phase4.dual_microbatch)" || return
      case "$sr4_size" in 2|4|8|16|32|64|100) ;; *) echo "Use sweep sizes 2/4/8/16/32/64/100" >&2; return 2 ;; esac
      sr4_name="dual-mb$sr4_size"; sr4_perf=dual-batch
      sr4_args=(--microbatch-size "$sr4_size") ;;
    *) echo "Use target, serial, or dual N" >&2; return 2 ;;
  esac
  python - "$SR_PHASE4B4_ROOT/preflight.json" "$SR_PHASE4B_COMMIT" <<'PY'
import json, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text())
assert r.get('valid') is True and not r.get('errors'), r.get('errors')
assert r['execution_git_commit'] == sys.argv[2]
PY
  local sr4_rc="$?"
  test "$sr4_rc" = 0 || return "$sr4_rc"
  test "$(git rev-parse HEAD)" = "$SR_PHASE4B_COMMIT" || return 2
  local sr4_dir="$SR_PHASE4B4_ROOT/$sr4_name"
  test ! -e "$sr4_dir" || { echo "Preserve existing cell: $sr4_dir" >&2; return 2; }
  if test "$sr4_mode" = target; then
    SR_PHASE4_DRAFT_BACKEND=hf-persistent SR_PHASE4B3_HF_METRICS=0 \
      phase4b2_run_mode target "$sr4_dir" "$SR_PHASE4B_WORKLOAD" 100 "$SR_PHASE4B_REFERENCE" || return
  elif test "$sr4_mode" = serial; then
    SR_PHASE4_DRAFT_BACKEND=vllm-batched \
      phase4b2_run_mode serial "$sr4_dir" "$SR_PHASE4B_WORKLOAD" 100 "$SR_PHASE4B_REFERENCE" || return
  else
    SR_PHASE4B_DUAL_RHYTHM=legacy \
      SR_PHASE4_DRAFT_BACKEND=vllm-batched SR_PHASE4B_DUAL_MICROBATCH_SIZE="$sr4_size" \
      PHASE4B1_OVERLAP_REQUIREMENT=characterization \
      phase4b2_run_mode dual "$sr4_dir" "$SR_PHASE4B_WORKLOAD" 100 "$SR_PHASE4B_REFERENCE" || return
  fi
  phase4b2_measure_mode "$sr4_perf" "$sr4_dir" "$SR_PHASE4B_WORKLOAD" || return
  python -m specrhythm.phase4.dual_microbatch_sweep validate \
    --run-root "$sr4_dir" --mode "$sr4_mode" --workload "$SR_PHASE4B_WORKLOAD" \
    "${sr4_args[@]}" --output "$sr4_dir/qualification.json"
}

phase4b4_compare () {
  python -m specrhythm.phase4.dual_microbatch_sweep compare \
    --root "$SR_PHASE4B4_ROOT" --output "$SR_PHASE4B4_ROOT/sweep.json" \
    --markdown "$SR_PHASE4B4_ROOT/sweep.md"
}
