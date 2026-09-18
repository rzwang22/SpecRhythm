#!/usr/bin/env bash
# Public foreground entry, or the one managed local child. No second UPLOAD ONLY.
set -Eeuo pipefail
SHA="${1:?full execution SHA required}"
export SR_FIXED_PYTHON="${SR_FIXED_PYTHON:-/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11}"
export SR_FIXED_S1="${SR_FIXED_S1:-/root/autodl-tmp/SpecRhythm-data/results/phase-s1/s1p-5a00049-20260909T144802Z-1469}"
export SR_EXEC_REPO="${SR_EXEC_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
export PYTHONPATH="$SR_EXEC_REPO/src" PYTHONUNBUFFERED=1
export SR_EAGER_CAUSAL_TRACE=light SR_EAGER_CAUSAL_LAYOUT=phased
export SR_FIXED_COMMIT="$SHA" SR_AUDIT_MODE=runtime
if [[ "${SR_K3_MANAGED_LOCAL:-}" != 1 ]]; then
  exec "$SR_FIXED_PYTHON" -m specrhythm.serving.k3_local_run --repo "$SR_EXEC_REPO" \
    --commit "$SHA" --k3-configuration k3-b128-v1 \
    --validation-profile performance-exploration --cpu-comparison
fi
exec "$SR_FIXED_PYTHON" -m specrhythm.serving.dual_batch_run --repo "$SR_EXEC_REPO" \
  --directory "${SR_K3_LOCAL_DELIVERY:?}" --commit "$SHA" --cpu-comparison
