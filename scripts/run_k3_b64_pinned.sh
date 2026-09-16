#!/usr/bin/env bash
# Execute with bash, never source into the interactive terminal.
if bash <<'SR_K3_CHILD'
set -Eeuo pipefail
FINAL_SHA=899b54a6b58c0ea07046582c5a17934f630ac040
REPO="${SR_K3_REPO:-/root/autodl-tmp/src/SpecRhythm}"
git -C "$REPO" fetch origin codex/rolling-eager-v0.1
git -C "$REPO" cat-file -e "${FINAL_SHA}^{commit}"
RUN_TREE="${REPO}-k3-b64-${FINAL_SHA:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
git -C "$REPO" worktree add --detach "$RUN_TREE" "$FINAL_SHA"
unset SR_K3_MANAGED_LOCAL SR_K3_LOCAL_DELIVERY SR_PING_DELIVERY
export SR_EXEC_REPO="$RUN_TREE"
export SR_K3_VALIDATION_PROFILE="${SR_K3_VALIDATION_PROFILE:-performance-exploration}"
export SR_PING_RUN_TAG="${SR_PING_RUN_TAG:-a100-k3-b64-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
bash "$RUN_TREE/scripts/run_k3_b64.sh" "$FINAL_SHA"
SR_K3_CHILD
then
  printf 'K3 B64 four-point run finished. Return only the archive printed by the runner.\n'
else
  rc=$?
  printf 'K3 B64 stopped (original rc=%s); later points stopped. Interactive terminal remains open.\n' "$rc"
  exit "$rc"
fi
