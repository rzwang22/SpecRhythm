#!/usr/bin/env bash
# Independent resident360/time-window entry; inspect commands never invoke inference/audit.
set -euo pipefail
: "${SR_FIXED_ROOT:?Set a new independent scan root}"
SR_FIXED_PYTHON="${SR_FIXED_PYTHON:-/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11}"
SR_SCAN_COMMAND="${1:?prepare|capacity|run|summary|bundle|status|errors|stop}"
shift
SR_SCAN_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SR_SCAN_REPO"
export PYTHONPATH="$SR_SCAN_REPO/src"
export PYTHONUNBUFFERED=1
case "$SR_SCAN_COMMAND" in
  prepare|capacity|run)
    : "${SR_FIXED_COMMIT:?Set the full tested commit SHA}"
    [[ "$SR_FIXED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
    [[ "$(git -c core.fsmonitor=false rev-parse HEAD)" == "$SR_FIXED_COMMIT" ]]
    [[ -z "$(git -c core.fsmonitor=false status --porcelain)" ]] || {
      echo 'Keep changes; use a clean execution checkout' >&2; exit 2;
    }
    ;;
esac
exec "$SR_FIXED_PYTHON" -m specrhythm.serving.decode_scan_cli "$SR_SCAN_COMMAND" --root "$SR_FIXED_ROOT" "$@"
