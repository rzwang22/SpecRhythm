#!/usr/bin/env bash
# Separate fixed64/32 diagnostic; never invokes S2 G1/G2/G3 or installs patches.
set -euo pipefail
: "${SR_FIXED_ROOT:?Set a new independent result root}"
SR_FIXED_PYTHON="${SR_FIXED_PYTHON:-/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11}"
SR_FIXED_COMMAND="${1:?prepare|capacity|short|stages|status|errors|stop|summary|audit|bundle}"
shift
SR_FIXED_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SR_FIXED_REPO"
export PYTHONPATH="$SR_FIXED_REPO/src"
export PYTHONUNBUFFERED=1
case "$SR_FIXED_COMMAND" in
  prepare|capacity|short|stages)
    : "${SR_FIXED_COMMIT:?Set the exact full tested commit SHA}"
    [[ "$SR_FIXED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo 'Full 40-character SHA required' >&2; exit 2; }
    [[ "$(git -c core.fsmonitor=false rev-parse HEAD)" == "$SR_FIXED_COMMIT" ]] || {
      echo "Checkout differs from SR_FIXED_COMMIT=$SR_FIXED_COMMIT" >&2; exit 2;
    }
    [[ -z "$(git -c core.fsmonitor=false status --porcelain)" ]] || {
      echo 'Execution checkout must be clean; retain changes and use a separate checkout' >&2; exit 2;
    }
    ;;
esac
exec "$SR_FIXED_PYTHON" -m specrhythm.serving.fixed_cli "$SR_FIXED_COMMAND" --root "$SR_FIXED_ROOT" "$@"
