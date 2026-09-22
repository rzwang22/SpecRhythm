#!/usr/bin/env bash
# Export-only: the source run and all previous archives are read-only.
if bash -s -- "${1:-${SR_ORDINARY_CPU_SOURCE:-/tmp/specrhythm-runs/a100-ordinary-cpu-20260919T124903Z-391/pingpong-k3-delivery-a100-ordinary-cpu-20260919T124903Z-391}}" <<'SR_REEXPORT_CHILD'
set -Eeuo pipefail
FINAL_SHA=f83edb43d22dcbb605468b8518bcca7c1bc36f45
SOURCE="$1"
REPO="${SR_K3_REPO:-/root/autodl-tmp/src/SpecRhythm}"
git -C "$REPO" fetch origin codex/rolling-eager-v0.1
git -C "$REPO" cat-file -e "${FINAL_SHA}^{commit}"
TAG="a100-ordinary-cpu-reexport-$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_TREE="${REPO}-reexport-${FINAL_SHA:0:12}-${TAG}"
git -C "$REPO" worktree add --detach "$RUN_TREE" "$FINAL_SHA"
test -f "$RUN_TREE/src/specrhythm/serving/ordinary_reexport.py"
export PYTHONPATH="$RUN_TREE/src" PYTHONUNBUFFERED=1
export SR_K3_LOCAL_BASE="${SR_K3_LOCAL_BASE:-/tmp/specrhythm-runs}"
export SR_PING_RESULTS="${SR_PING_RESULTS:-/root/autodl-tmp/SpecRhythm-data/results/rolling-eager}"
"${SR_FIXED_PYTHON:-/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11}" \
  -m specrhythm.serving.ordinary_reexport --source "$SOURCE" \
  --expected-execution 944328263b02a915f397bcb03c8c743c71e9a56c \
  --exporter-commit "$FINAL_SHA" --tag "$TAG"
SR_REEXPORT_CHILD
then
  printf 'Reexport finished; return the single archive above.\n'
else
  rc=$?
  printf 'Reexport stopped (rc=%s); original run retained; terminal remains open.\n' "$rc" >&2
  exit "$rc"
fi
