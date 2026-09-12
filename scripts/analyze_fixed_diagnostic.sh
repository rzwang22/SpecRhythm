#!/usr/bin/env bash
# Read-only retained timing analysis/export. No inference, CUDA initialization or audit.
set -euo pipefail
SR_FIXED_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$SR_FIXED_REPO/src"
export PYTHONUNBUFFERED=1
exec "${SR_FIXED_PYTHON:-python3}" -m specrhythm.serving.fixed_attribution_cli "$@"
