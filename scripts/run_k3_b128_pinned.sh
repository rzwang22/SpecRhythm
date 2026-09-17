#!/usr/bin/env bash
# The child owns strict mode; the caller's interactive shell stays open.
if bash -s -- "${1:-unified}" "${2:-b128}" <<'SR_K3_B128_CHILD'
set -Eeuo pipefail
EXPERIMENT="$1"
SCALE="$2"
case "$EXPERIMENT" in
  unified|io-only) ;;
  *) printf 'Choose unified or io-only (legacy dispatch).\n' >&2; exit 2 ;;
esac
case "$SCALE" in
  b128|b64) ;;
  *) printf 'Choose b128 or the explicit same-version b64 rerun.\n' >&2; exit 2 ;;
esac
FINAL_SHA=f7bfb43152f5655edbad9888c53c0a81e9d11954
REPO="${SR_K3_REPO:-/root/autodl-tmp/src/SpecRhythm}"
git -C "$REPO" fetch origin codex/rolling-eager-v0.1
git -C "$REPO" cat-file -e "${FINAL_SHA}^{commit}"
RUN_TREE="${REPO}-k3-${SCALE}-${EXPERIMENT}-${FINAL_SHA:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
git -C "$REPO" worktree add --detach "$RUN_TREE" "$FINAL_SHA"
test -f "$RUN_TREE/scripts/run_k3_${SCALE}.sh"
unset SR_K3_MANAGED_LOCAL SR_K3_LOCAL_DELIVERY SR_PING_DELIVERY SR_K3_REPEAT_CHILD
unset SR_K3_REVERSE SR_K3_DRAFT_DISPATCH SR_K3_OBSERVATION SR_K3_SKIP_CAPACITY
export SR_EXEC_REPO="$RUN_TREE" SR_K3_EXPERIMENT="$EXPERIMENT"
export SR_K3_VALIDATION_PROFILE=performance-exploration
export SR_K3_LOCAL_BASE="${SR_K3_LOCAL_BASE:-/tmp/specrhythm-runs}"
export SR_PING_RESULTS="${SR_PING_RESULTS:-/root/autodl-tmp/SpecRhythm-data/results/rolling-eager}"
export SR_PING_RUN_TAG="a100-k3-${SCALE}-${EXPERIMENT}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
bash "$RUN_TREE/scripts/run_k3_${SCALE}.sh" "$FINAL_SHA"
SR_K3_B128_CHILD
then
  printf 'Declared K3 experiment finished. Return only the runner archive.\n'
else
  rc=$?
  printf 'K3 stopped (original rc=%s); later points stopped. Interactive terminal remains open.\n' "$rc"
  exit "$rc"
fi
