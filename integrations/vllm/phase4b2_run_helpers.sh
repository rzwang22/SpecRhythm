#!/usr/bin/env bash

# Source phase4b_run_helpers.sh and phase4b1_gate_helpers.sh first.  These
# wrappers reuse the validated resident runners and add only the Phase-4B.2
# boundary, timestamped JIT provenance, and offline metrics layer.

phase4b2_run_mode () {
  phase4b2_mode="$1"
  phase4b2_dir="$2"
  phase4b2_workload="$3"
  phase4b2_count="$4"
  phase4b2_reference="$5"
  SR_PHASE4B2_PERFORMANCE=1 \
    phase4b1_run_mode \
      "$phase4b2_mode" "$phase4b2_dir" "$phase4b2_workload" \
      "$phase4b2_count" "$phase4b2_reference"
}

phase4b2_measure_mode () {
  phase4b2_mode="$1"
  phase4b2_dir="$2"
  phase4b2_workload="$3"
  test -n "${SR_PHASE4B_CONFIG:-}" || {
    echo "SR_PHASE4B_CONFIG is required" >&2
    return 2
  }
  test -n "${SR_PHASE4B_TOPOLOGY:-}" || {
    echo "SR_PHASE4B_TOPOLOGY is required" >&2
    return 2
  }
  test -n "${SR_PHASE4B_PATCH_MANIFEST:-}" || {
    echo "SR_PHASE4B_PATCH_MANIFEST is required" >&2
    return 2
  }
  specrhythm phase4b2-decode-run \
    --mode "$phase4b2_mode" \
    --run-root "$phase4b2_dir" \
    --workload "$phase4b2_workload" \
    --config "$SR_PHASE4B_CONFIG" \
    --topology "$SR_PHASE4B_TOPOLOGY" \
    --patch-manifest "$SR_PHASE4B_PATCH_MANIFEST" \
    --output "$phase4b2_dir/decode-performance.json"
}

phase4b2_compare_target_serial () {
  phase4b2_root="$1"
  specrhythm phase4b2-decode-compare \
    --target "$phase4b2_root/target/decode-performance.json" \
    --serial "$phase4b2_root/serial/decode-performance.json" \
    --output "$phase4b2_root/target-serial-exact.json" \
    --markdown-output "$phase4b2_root/target-serial-exact.md"
}

phase4b2_compare_all () {
  phase4b2_root="$1"
  specrhythm phase4b2-decode-compare \
    --target "$phase4b2_root/target/decode-performance.json" \
    --serial "$phase4b2_root/serial/decode-performance.json" \
    --dual "$phase4b2_root/dual/decode-performance.json" \
    --output "$phase4b2_root/decode-performance-comparison.json" \
    --markdown-output "$phase4b2_root/decode-performance-comparison.md"
}
