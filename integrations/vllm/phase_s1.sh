#!/usr/bin/env bash
set -euo pipefail
# Explicit interpreter: never execute in the S0 CPU environment.
SR_S1_PYTHON=/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11
cd /root/autodl-tmp/src/SpecRhythm
unset USE_TORCH USE_TF USE_FLAX
export HF_HOME=/root/.cache/huggingface
exec "$SR_S1_PYTHON" -m specrhythm.serving.s1_cli "$@"
