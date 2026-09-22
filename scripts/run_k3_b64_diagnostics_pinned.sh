#!/usr/bin/env bash
# Run with bash; the caller's interactive shell keeps its existing options.
if bash -s -- "${1:-}" <<'SR_K3_DIAGNOSTICS_CHILD'
set -Eeuo pipefail
EXPERIMENT="$1"
case "$EXPERIMENT" in
  io-only|unified) ;;
  *) printf 'Choose exactly one configuration: io-only or unified\n' >&2; exit 2 ;;
esac
FINAL_SHA=cc42a501b63623de3e046e6388d0dcab0d2f339c
REPO="${SR_K3_REPO:-/root/autodl-tmp/src/SpecRhythm}"
git -C "$REPO" fetch origin codex/rolling-eager-v0.1
git -C "$REPO" cat-file -e "${FINAL_SHA}^{commit}"
RUN_TREE="${REPO}-k3-b64-${EXPERIMENT}-${FINAL_SHA:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
git -C "$REPO" worktree add --detach "$RUN_TREE" "$FINAL_SHA"
unset SR_K3_MANAGED_LOCAL SR_K3_LOCAL_DELIVERY SR_PING_DELIVERY SR_K3_REPEAT_CHILD
unset SR_K3_REVERSE SR_K3_DRAFT_DISPATCH SR_K3_OBSERVATION
export SR_EXEC_REPO="$RUN_TREE" SR_K3_EXPERIMENT="$EXPERIMENT"
export SR_K3_VALIDATION_PROFILE=performance-exploration
export SR_PING_RUN_TAG="a100-k3-b64-${EXPERIMENT}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
bash "$RUN_TREE/scripts/run_k3_b64.sh" "$FINAL_SHA"
SR_K3_DIAGNOSTICS_CHILD
then
  printf 'Declared K3 B64 repetitions finished; return only the runner archive.\n'
else
  rc=$?
  printf 'K3 B64 stopped (original rc=%s); later points stopped. Interactive terminal remains open.\n' "$rc"
  exit "$rc"
fi
