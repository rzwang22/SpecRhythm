#!/usr/bin/env bash
# Run in an independent Bash; see docs/rolling-eager-execution-runbook.md.
main() {
set -Eeuo pipefail
CONTROL_SHA="${1:?full evidence-control SHA required}"
OPTIMIZED_SHA="${2:?full execution-optimized SHA required}"
[[ "$CONTROL_SHA" =~ ^[0-9a-f]{40}$ && "$OPTIMIZED_SHA" =~ ^[0-9a-f]{40}$ ]]
test "$CONTROL_SHA" != "$OPTIMIZED_SHA"
REPO="${SR_EXEC_SOURCE_REPO:-/root/autodl-tmp/src/SpecRhythm}"
RESULTS="${SR_EXEC_RESULTS:-/root/autodl-tmp/SpecRhythm-data/results/rolling-eager}"
TAG="${SR_AUDIT_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
export SR_FIXED_PYTHON="${SR_FIXED_PYTHON:-/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11}"
cd "$REPO"
test -z "$(git -c core.fsmonitor=false status --porcelain)"
git cat-file -e "$CONTROL_SHA^{commit}"
git cat-file -e "$OPTIMIZED_SHA^{commit}"
# Both explicit refs may be sibling commits with the same observation fix.
# Preserve history; do not require rewriting a control descendant onto the optimized branch.
git merge-base "$CONTROL_SHA" "$OPTIMIZED_SHA" > /dev/null
ORDER=(control optimized)
case "${SR_EXEC_VERSION_ORDER:-control-first}" in
  control-first) : ;;
  optimized-first) ORDER=(optimized control) ;;
  *) printf 'Unknown version order\n' >&2; exit 2 ;;
esac
LISTS=()
for VERSION in "${ORDER[@]}"; do
  SHA="$CONTROL_SHA"
  if [[ "$VERSION" == optimized ]]; then SHA="$OPTIMIZED_SHA"; fi
  WORKTREE="${REPO%/*}/SpecRhythm-execution-${SHA:0:12}-${TAG}"
  test ! -e "$WORKTREE"
  git worktree add --detach "$WORKTREE" "$SHA"
  export SR_EXEC_REPO="$WORKTREE" SR_AUDIT_RUN_TAG="${TAG}-${VERSION}"
  export SR_EXEC_REPORT_LIST="$RESULTS/execution-reports-${SHA:0:12}-${TAG}-${VERSION}.txt"
  test ! -e "$SR_EXEC_REPORT_LIST"
  # The point child owns failure capture and preserves the original failure code.
  # No conditional around it: its failure stops this loop before the next version.
  bash "$WORKTREE/scripts/run_execution_pair_b16.sh" "$SHA"
  LISTS+=("$SR_EXEC_REPORT_LIST")
done
REPORTS=()
for LIST in "${LISTS[@]}"; do
  while IFS= read -r REPORT; do REPORTS+=("$REPORT"); done < "$LIST"
done
test "${#REPORTS[@]}" = 4
export PYTHONPATH="$SR_EXEC_REPO/src"
"$SR_FIXED_PYTHON" -m specrhythm.serving.execution_evidence --reports "${REPORTS[@]}" \
  --commits "$CONTROL_SHA" "$OPTIMIZED_SHA" \
  --output "$RESULTS/execution-two-version-${TAG}.json"
printf 'RETURN: %s/execution-two-version-%s.json\n' "$RESULTS" "$TAG"
printf 'Detached source worktrees retained for reproducibility. No GPU repeats scheduled.\n'
}
main "$@"
