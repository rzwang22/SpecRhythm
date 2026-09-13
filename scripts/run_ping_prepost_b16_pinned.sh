#!/usr/bin/env bash
# Execute with bash, never source. The interactive command in the runbook
# receives this script's original failure code in an outer if.
if bash <<'SR_PING_CHILD'
set -Eeuo pipefail
FINAL_SHA=c0ecc2a405c8b6c9cb3016c7254739bda5bec2b6
REPO="${SR_PING_REPO:-/root/autodl-tmp/src/SpecRhythm}"
git -C "$REPO" fetch origin codex/rolling-eager-v0.1
git -C "$REPO" cat-file -e "${FINAL_SHA}^{commit}"
RUN_TREE="${REPO}-ping-prepost3-${FINAL_SHA:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
git -C "$REPO" worktree add --detach "$RUN_TREE" "$FINAL_SHA"
export SR_EXEC_REPO="$RUN_TREE"
bash "$RUN_TREE/scripts/run_ping_prepost_b16.sh" "$FINAL_SHA"
SR_PING_CHILD
then
  printf 'PingPong pair finished. Return only the single archive printed by the runner.\n'
else
  rc=$?
  printf 'PingPong stopped (original rc=%s); later points stopped. Interactive terminal remains open.\n' "$rc"
  exit "$rc"
fi
