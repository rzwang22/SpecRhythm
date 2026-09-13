#!/usr/bin/env bash
# Execute with bash, never source into the interactive terminal.
if bash <<'SR_K3_CHILD'
set -Eeuo pipefail
FINAL_SHA=b2a2d29210d68357590a567d4819f3552175d179
REPO="${SR_K3_REPO:-/root/autodl-tmp/src/SpecRhythm}"
git -C "$REPO" fetch origin codex/rolling-eager-v0.1
git -C "$REPO" cat-file -e "${FINAL_SHA}^{commit}"
RUN_TREE="${REPO}-k3-${FINAL_SHA:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
git -C "$REPO" worktree add --detach "$RUN_TREE" "$FINAL_SHA"
export SR_EXEC_REPO="$RUN_TREE"
bash "$RUN_TREE/scripts/run_k3_b16.sh" "$FINAL_SHA"
SR_K3_CHILD
then
  printf 'K3 three-point run finished. Return only the archive printed by the runner.\n'
else
  rc=$?
  printf 'K3 stopped (original rc=%s); later points stopped. Interactive terminal remains open.\n' "$rc"
  exit "$rc"
fi
