#!/usr/bin/env bash
# Execute with bash; never source the strict child into an interactive shell.
if bash <<'SR_PREPOST_CHILD'
set -Eeuo pipefail
FINAL_SHA=f01e8d037007540209999179357c3dd2ff2335a7
REPO="${SR_PREPOST_REPO:-/root/autodl-tmp/src/SpecRhythm}"
git -C "$REPO" fetch origin codex/rolling-eager-v0.1
git -C "$REPO" cat-file -e "${FINAL_SHA}^{commit}"
RUN_TREE="${REPO}-prepost3-${FINAL_SHA:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
git -C "$REPO" worktree add --detach "$RUN_TREE" "$FINAL_SHA"
export SR_EXEC_REPO="$RUN_TREE"
bash "$RUN_TREE/scripts/run_prepost_b16.sh" "$FINAL_SHA"
SR_PREPOST_CHILD
then
  printf 'Prepost3 pair finished; return the printed evidence archives and comparison JSON.\n'
else
  rc=$?
  printf 'Prepost3 stopped at first failure (rc=%s); later points stopped. Interactive terminal remains open.\n' "$rc"
  exit "$rc"
fi
