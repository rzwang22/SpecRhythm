#!/usr/bin/env bash

# Source after the existing Phase4B/4B.1/4B.2 helpers. No serving policy changes.
phase4b3_run_serial () {
  local sr3_stage="$1" sr3_backend="$2"
  local sr3_count sr3_workload sr3_reference sr3_directory
  local sr3_prior=()
  case "$sr3_stage" in
    D4)
      sr3_count=5
      sr3_workload="$SR_PHASE4B3_WORKLOAD5"
      sr3_reference="$SR_PHASE4B3_REFERENCE5"
      ;;
    D5)
      sr3_count=100
      sr3_workload="$SR_PHASE4B_WORKLOAD"
      sr3_reference="$SR_PHASE4B_REFERENCE"
      sr3_prior=(--d4 "$SR_PHASE4B3_ROOT/D4/comparison.json")
      ;;
    *) echo "Only qualified D4/D5 Serial is supported" >&2; return 2 ;;
  esac
  case "$sr3_backend" in
    hf|vllm) ;;
    *) echo "Backend must be hf or vllm" >&2; return 2 ;;
  esac
  sr3_directory="$SR_PHASE4B3_ROOT/$sr3_stage/$sr3_backend"
  test ! -e "$sr3_directory" || {
    echo "Preserve the old attempt and use a fresh result root" >&2
    return 2
  }
  # This is a CPU check using the qualified production configuration. The actual
  # backend selection below is separately recorded in the immutable admission.
  CUDA_VISIBLE_DEVICES=0 SR_PHASE4_DRAFT_BACKEND=vllm-batched \
    python -m specrhythm.phase4.draft_qualification prepare-serial \
      --qualification "$SR_PHASE4B3_ROOT/qualification.json" \
      --stage "$sr3_stage" "${sr3_prior[@]}" --backend "$sr3_backend" \
      --expected-commit "$SR_PHASE4B_COMMIT" --config "$SR_PHASE4B_CONFIG" \
      --workload "$sr3_workload" --reference "$sr3_reference" \
      --output "$SR_PHASE4B3_ROOT/$sr3_stage/$sr3_backend-admission.json" || return
  if test "$sr3_backend" = hf; then
    SR_PHASE4_DRAFT_BACKEND=hf-persistent SR_PHASE4B3_HF_METRICS=1 \
      phase4b2_run_mode serial "$sr3_directory" "$sr3_workload" "$sr3_count" "$sr3_reference" || return
  else
    SR_PHASE4_DRAFT_BACKEND=vllm-batched SR_PHASE4B3_HF_METRICS=0 \
      phase4b2_run_mode serial "$sr3_directory" "$sr3_workload" "$sr3_count" "$sr3_reference" || return
  fi
  phase4b2_measure_mode serial "$sr3_directory" "$sr3_workload"
}

phase4b3_compare_serial () {
  local sr3_stage="$1" sr3_count
  local sr3_prior=()
  case "$sr3_stage" in
    D4) sr3_count=5 ;;
    D5) sr3_count=100; sr3_prior=(--d4 "$SR_PHASE4B3_ROOT/D4/comparison.json") ;;
    *) echo "Only D4/D5 comparisons are supported" >&2; return 2 ;;
  esac
  python -m specrhythm.phase4.draft_comparison \
    --qualification "$SR_PHASE4B3_ROOT/qualification.json" \
    --stage "$sr3_stage" "${sr3_prior[@]}" \
    --hf "$SR_PHASE4B3_ROOT/$sr3_stage/hf" \
    --vllm "$SR_PHASE4B3_ROOT/$sr3_stage/vllm" \
    --request-count "$sr3_count" \
    --output "$SR_PHASE4B3_ROOT/$sr3_stage/comparison.json"
}
