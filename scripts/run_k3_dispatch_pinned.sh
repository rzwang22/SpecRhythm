#!/usr/bin/env bash
# Independent child strict mode; caller's interactive terminal remains open.
if bash -s -- "$@" <<'SR_K3_DISPATCH_CHILD'
set -Eeuo pipefail
[[ $# -le 1 ]] || { printf 'Choose one of lean-reference or lean-dispatch-opt.\n' >&2; exit 2; }
EXPERIMENT="${1:-lean-dispatch-opt}"
case "$EXPERIMENT" in
  lean-reference|lean-dispatch-opt) ;;
  *) printf 'Choose lean-reference or lean-dispatch-opt (both lean).\n' >&2; exit 2 ;;
esac
FINAL_SHA=c1fca49f3f846ef94f0759b585fddaf091d78b62
REPO="${SR_K3_REPO:-/root/autodl-tmp/src/SpecRhythm}"
git -C "$REPO" fetch origin codex/rolling-eager-v0.1
git -C "$REPO" cat-file -e "${FINAL_SHA}^{commit}"
RUN_TREE="${REPO}-k3-b128-${EXPERIMENT}-${FINAL_SHA:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
git -C "$REPO" worktree add --detach "$RUN_TREE" "$FINAL_SHA"
test -f "$RUN_TREE/scripts/run_k3_b128.sh"
# A separate heavy numerical plan would invalidate this paired profile comparison.
if [[ -n "${SR_PHASE4_NUMERICAL_DIAGNOSTIC_PLAN:-}" ]]; then
  printf 'Unset SR_PHASE4_NUMERICAL_DIAGNOSTIC_PLAN for this fixed comparison.\n' >&2
  exit 2
fi
unset SR_K3_MANAGED_LOCAL SR_K3_LOCAL_DELIVERY SR_PING_DELIVERY SR_K3_REPEAT_CHILD
unset SR_K3_REVERSE SR_K3_DRAFT_DISPATCH SR_K3_OBSERVATION SR_K3_SKIP_CAPACITY
unset SR_K3_TARGET_DIAGNOSTICS SR_FIXED_TARGET_DIAGNOSTICS SR_K3_TARGET_DISPATCH
export SR_EXEC_REPO="$RUN_TREE" SR_K3_EXPERIMENT="$EXPERIMENT"
export SR_K3_VALIDATION_PROFILE=performance-exploration
export SR_K3_LOCAL_BASE="${SR_K3_LOCAL_BASE:-/tmp/specrhythm-runs}"
export SR_PING_RESULTS="${SR_PING_RESULTS:-/root/autodl-tmp/SpecRhythm-data/results/rolling-eager}"
export SR_PING_RUN_TAG="a100-k3-b128-${EXPERIMENT}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
bash "$RUN_TREE/scripts/run_k3_b128.sh" "$FINAL_SHA"
SR_K3_DISPATCH_CHILD
then
  printf 'Declared K3 profile finished. Return only the runner archive.\n'
else
  rc=$?
  printf 'K3 stopped (original rc=%s); later points stopped. Interactive terminal remains open.\n' "$rc"
  exit "$rc"
fi
