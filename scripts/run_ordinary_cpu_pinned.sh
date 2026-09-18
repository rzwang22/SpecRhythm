#!/usr/bin/env bash
# All strict/exit logic belongs to this independent child, never the caller's shell.
if bash -s <<'SR_ORDINARY_CPU_CHILD'
set -Eeuo pipefail
FINAL_SHA=944328263b02a915f397bcb03c8c743c71e9a56c
REPO="${SR_K3_REPO:-/root/autodl-tmp/src/SpecRhythm}"
git -C "$REPO" fetch origin codex/rolling-eager-v0.1
git -C "$REPO" cat-file -e "${FINAL_SHA}^{commit}"
TAG="a100-ordinary-cpu-$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_TREE="${REPO}-ordinary-cpu-${FINAL_SHA:0:12}-${TAG}"
git -C "$REPO" worktree add --detach "$RUN_TREE" "$FINAL_SHA"
test -f "$RUN_TREE/scripts/run_ordinary_cpu.sh"
unset SR_K3_MANAGED_LOCAL SR_K3_LOCAL_DELIVERY SR_PING_DELIVERY SR_K3_REPEAT_CHILD
unset SR_K3_REVERSE SR_K3_SKIP_CAPACITY SR_K3_DRAFT_DISPATCH SR_K3_OBSERVATION
unset SR_K3_TARGET_DISPATCH SR_K3_TARGET_CPU SR_K3_EXPERIMENT SR_K3_TARGET_EXPERIMENT
unset SR_K3_TARGET_DIAGNOSTICS SR_FIXED_TARGET_DIAGNOSTICS
export SR_EXEC_REPO="$RUN_TREE" SR_PING_RUN_TAG="$TAG"
export SR_K3_LOCAL_BASE="${SR_K3_LOCAL_BASE:-/tmp/specrhythm-runs}"
export SR_PING_RESULTS="${SR_PING_RESULTS:-/root/autodl-tmp/SpecRhythm-data/results/rolling-eager}"
bash "$RUN_TREE/scripts/run_ordinary_cpu.sh" "$FINAL_SHA"
SR_ORDINARY_CPU_CHILD
then
  printf 'Ordinary P0/P1/S1 comparison finished; return the single archive above.\n'
else
  rc=$?
  printf 'Stopped at first error (rc=%s); interactive shell remains open.\n' "$rc" >&2
  exit "$rc"
fi
