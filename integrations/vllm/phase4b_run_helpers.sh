#!/usr/bin/env bash

# Source this file from the Phase-4B server shell.  The caller must have
# started Draft in the same shell so this helper can reap draft_pid.

phase4b_terminate_and_wait () {
  phase4b_pid="$1"
  if ! kill -0 "$phase4b_pid" 2>/dev/null; then
    wait "$phase4b_pid" 2>/dev/null || true
    return 0
  fi
  kill "$phase4b_pid" 2>/dev/null || true
  phase4b_poll=0
  phase4b_poll_limit="${PHASE4B_CLEANUP_POLLS:-100}"
  while kill -0 "$phase4b_pid" 2>/dev/null && \
      test "$phase4b_poll" -lt "$phase4b_poll_limit"; do
    sleep "${PHASE4B_CLEANUP_SLEEP_SECONDS:-0.1}"
    phase4b_poll=$((phase4b_poll + 1))
  done
  if kill -0 "$phase4b_pid" 2>/dev/null; then
    kill -KILL "$phase4b_pid" 2>/dev/null || true
  fi
  wait "$phase4b_pid" 2>/dev/null || true
}

phase4b_wait_for_draft_shutdown () {
  phase4b_pid="$1"
  phase4b_poll=0
  phase4b_poll_limit="${PHASE4B_SHUTDOWN_POLLS:-300}"
  while kill -0 "$phase4b_pid" 2>/dev/null && \
      test "$phase4b_poll" -lt "$phase4b_poll_limit"; do
    sleep "${PHASE4B_CLEANUP_SLEEP_SECONDS:-0.1}"
    phase4b_poll=$((phase4b_poll + 1))
  done
  if kill -0 "$phase4b_pid" 2>/dev/null; then
    echo "Draft service did not shut down within the bounded wait" >&2
    phase4b_terminate_and_wait "$phase4b_pid"
    return 124
  fi
  wait "$phase4b_pid"
}

phase4b_run_target_with_cleanup () {
  phase4b_draft_pid="$1"
  phase4b_target_log="$2"
  phase4b_draft_log="$3"
  shift 3
  test "${1:-}" = "--" || {
    echo "usage: phase4b_run_target_with_cleanup DRAFT_PID TARGET_LOG DRAFT_LOG -- COMMAND..." >&2
    return 2
  }
  shift
  test "$#" -gt 0 || {
    echo "Phase-4B Target command is empty" >&2
    return 2
  }

  phase4b_lifecycle_artifact="${PHASE4B_LIFECYCLE_ARTIFACT:-${phase4b_target_log}.lifecycle.json}"
  phase4b_guard="${PHASE4B_RUN_GUARD:-${phase4b_lifecycle_artifact}.active}"
  phase4b_lifecycle_args=(
    -m specrhythm.phase4.process_lifecycle
    --artifact "$phase4b_lifecycle_artifact"
    --target-log "$phase4b_target_log"
    --guard "$phase4b_guard"
    --draft-pid "$phase4b_draft_pid"
    --natural-teardown-grace-seconds "${PHASE4B_NATURAL_TEARDOWN_GRACE_SECONDS:-5}"
    --graceful-seconds "${PHASE4B_CLEANUP_GRACE_SECONDS:-5}"
    --kill-seconds "${PHASE4B_CLEANUP_KILL_SECONDS:-2}"
    --poll-seconds "${PHASE4B_CLEANUP_POLL_SECONDS:-0.05}"
  )
  phase4b_draft_socket="${SR_PHASE4_DUAL_DRAFT_SOCKET:-${SR_PHASE4_DRAFT_SOCKET:-}}"
  if test -n "$phase4b_draft_socket"; then
    phase4b_lifecycle_args+=(--draft-socket "$phase4b_draft_socket")
  fi
  phase4b_lifecycle_args+=(-- "$@")
  phase4b_python="${PHASE4B_PYTHON:-}"
  if test -z "$phase4b_python"; then
    phase4b_python="$(command -v python || command -v python3)"
  fi
  if "$phase4b_python" "${phase4b_lifecycle_args[@]}"; then
    phase4b_target_status=0
  else
    phase4b_target_status="$?"
  fi

  if test "$phase4b_target_status" -ne 0; then
    phase4b_terminate_and_wait "$phase4b_draft_pid"
    echo "Target failed with status $phase4b_target_status" >&2
    tail -n 200 "$phase4b_target_log" >&2 || true
    tail -n 50 "$phase4b_draft_log" >&2 || true
    return "$phase4b_target_status"
  fi

  phase4b_wait_for_draft_shutdown "$phase4b_draft_pid"
}
