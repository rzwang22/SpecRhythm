#!/usr/bin/env bash
# Invoke from the outer if/child Bash in docs/rolling-eager-audit-runbook.md.
main() {
set -Eeuo pipefail
FINAL_SHA="${1:?full final SHA required}"
export SR_FIXED_PYTHON="${SR_FIXED_PYTHON:-/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11}"
export SR_FIXED_S1="${SR_FIXED_S1:-/root/autodl-tmp/SpecRhythm-data/results/phase-s1/s1p-5a00049-20260909T144802Z-1469}"
RUN_TAG="${SR_AUDIT_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
RESULTS=/root/autodl-tmp/SpecRhythm-data/results/rolling-eager
export SR_EAGER_CAUSAL_TRACE=light
cd /root/autodl-tmp/src/SpecRhythm
export PYTHONPATH="$PWD/src" PYTHONUNBUFFERED=1
unset SR_FIXED_ROOT
on_failure() {
  rc=$?; trap - ERR; set +e
  printf 'STOP: rc=%s point=%s root=%s\n' "$rc" "${POINT:-unset}" "${SR_FIXED_ROOT:-unset}"
  if [[ -n "${SR_FIXED_ROOT:-}" && -d "$SR_FIXED_ROOT" ]]; then
    "$SR_FIXED_PYTHON" -m specrhythm.continuation.gpu_check status --root "$SR_FIXED_ROOT"
    "$SR_FIXED_PYTHON" -m specrhythm.continuation.gpu_check errors --root "$SR_FIXED_ROOT"
    bash scripts/run_decode_scan.sh status
    bash scripts/run_decode_scan.sh errors
    bash scripts/run_decode_scan.sh bundle --output "${SR_FIXED_ROOT}-failure-small.tar.gz"
    "$SR_FIXED_PYTHON" -m specrhythm.serving.eager_latency_bundle \
      --root "$SR_FIXED_ROOT" --output "${SR_FIXED_ROOT}-failure-evidence.tar.gz"
  fi
  exit "$rc"
}
trap on_failure ERR
[[ "$FINAL_SHA" =~ ^[0-9a-f]{40}$ ]]
test -z "$(git -c core.fsmonitor=false status --porcelain)"
test "$(git rev-parse HEAD)" = "$FINAL_SHA"
export SR_FIXED_COMMIT="$FINAL_SHA"
REPORTS=()
for POINT in serial:full serial-eager:full serial:runtime serial-eager:runtime; do
  export SR_AUDIT_MODE="${POINT#*:}" SR_AUDIT_SERVING_MODE="${POINT%:*}"
  export SR_FIXED_ROOT="$RESULTS/serial-audit-B16-${FINAL_SHA:0:12}-${SR_AUDIT_SERVING_MODE}-${SR_AUDIT_MODE}-${RUN_TAG}"
  test ! -e "$SR_FIXED_ROOT"
  printf 'NEW ROOT: %s\n' "$SR_FIXED_ROOT"
  bash scripts/run_decode_scan.sh prepare --s1 "$SR_FIXED_S1" \
    --draft-audit "$SR_AUDIT_MODE" --observation buffered-live --identity-matching bound-prefix \
    --selection-seed 1666 --warmup-steps 2 --window-seconds 30 --repeats 1 \
    --setup-timeout 900 --drain-timeout 60
  "$SR_FIXED_PYTHON" - <<'PY_CONFIG'
import json, os, pathlib
root=pathlib.Path(os.environ['SR_FIXED_ROOT'])
s=json.loads((root/'scan-config.json').read_text())
assert s['execution']['git_commit']==os.environ['SR_FIXED_COMMIT']
assert s['workload_sha256']=='cdaf71adace15d229f5087b98f9fd162a958456226a660184fe03f5d6ebd8ff4'
assert s['pool_size']==len(set(s['selection']['request_ids']))==360
opts=s['options']
assert opts['draft_audit']==os.environ['SR_AUDIT_MODE']
assert opts['samples'] is None and opts['warmup_steps']==2 and opts['window_seconds']==30
assert opts['repeats']==1 and opts['setup_timeout']==900 and opts['drain_timeout']==60
assert opts['observation']=='buffered-live' and opts['identity_matching']=='bound-prefix'
print(json.dumps({'point':os.environ['SR_AUDIT_SERVING_MODE'],'options':opts}))
PY_CONFIG
  bash scripts/run_decode_scan.sh capacity --single-point --batch 16 --mode "$SR_AUDIT_SERVING_MODE"
  "$SR_FIXED_PYTHON" -m specrhythm.continuation.gpu_check run \
    --root "$SR_FIXED_ROOT" --request-count 3 --draft-audit "$SR_AUDIT_MODE" \
    --timeout 900 --drain-timeout 60
  "$SR_FIXED_PYTHON" - <<'PY_CORRECTNESS'
import json,os,pathlib
p=pathlib.Path(os.environ['SR_FIXED_ROOT'])/'rolling-eager-gpu-check'
s=json.loads((p/'state.json').read_text());r=json.loads((p/'result.json').read_text())
assert s['status']=='COMPLETE' and s['cleanup_status']=='PASS' and not s['remaining_owned_pids']
assert s['exit_code']==0 and r['valid'] and r['physical_gpu_correctness']=='PASS'
b=r['backend']
assert b['draft_audit']['mode']==os.environ['SR_AUDIT_MODE']
assert r['cleanup_status']=='PASS' and b['eager_gpu_live_work']==0 and not b['eager_gpu_write_inflight']
assert b['worker_resources']['blocks_allocated']==b['worker_resources']['blocks_freed']
assert all(c['exact'] for c in r['reference_comparisons'])
if b['draft_audit']['mode']=='runtime':
 g=b['draft_audit']['runtime']
 assert g['blocks_allocated']==g['blocks_freed'] and g['live_blocks']==0
 assert not g['pending_write']
print('Audit-layer physical correctness and cleanup PASS; proceed to performance.')
PY_CORRECTNESS
  bash scripts/run_decode_scan.sh run --single-point --batch 16 --mode "$SR_AUDIT_SERVING_MODE"
  "$SR_FIXED_PYTHON" - <<'PY_MEASUREMENT'
import os,pathlib
from specrhythm.serving.fixed_artifacts import point_reports
rows=[r for r in point_reports(pathlib.Path(os.environ['SR_FIXED_ROOT'])) if not r['point'].get('probe')]
assert len(rows)==1
r=rows[0]
assert r['mode']==os.environ['SR_AUDIT_SERVING_MODE'] and r['batch']==16
assert all(r[k]=='PASS' for k in ('capacity_status','execution_status','measurement_status','cleanup_status'))
assert r['formal_comparison_eligible'] and r['effective_exit_code']==0
assert r['stop_reason']=='time_budget' and r['measured_window_ms']>=30000
assert r['actual_target_batch']['min']==r['actual_target_batch']['max']==16
if r['mode']=='serial-eager':
 e=r['rolling_eager']
 assert e['window_counters']['started']>0 and e['cleanup_status']=='PASS' and e['owner_stopped']
 assert e['pending_work'] in (0,[])
print('Original execution/measurement/cleanup PASS; no timing adjustment.')
PY_MEASUREMENT
  REPORT="${SR_FIXED_ROOT}-audit-report.json"
  "$SR_FIXED_PYTHON" -m specrhythm.serving.audit_layer_report \
    --root "$SR_FIXED_ROOT" --expected-commit "$FINAL_SHA" --output "$REPORT"
  "$SR_FIXED_PYTHON" - "$REPORT" <<'PY_EVIDENCE'
import json,sys
r=json.load(open(sys.argv[1]))
assert r['draft_audit']['schema_version']=='specrhythm.draft-audit.v1'
assert all(t['mode']=='light' and t['status']=='COMPLETE' for t in r['trace_coverage'].values())
assert all(t['native_coverage']=='COMPLETE' for t in r['target_ranks'].values())
assert r['draft_device']['native_coverage']=='COMPLETE'
assert r['coordinator_exclusive']['status']=='COMPLETE'
assert any(b['boundary']=='setup_complete_snapshot' for b in r['draft_audit']['full_boundaries'])
assert any(b['boundary']=='timing_entry' for b in r['draft_audit']['full_boundaries'])
assert any(b['boundary']=='before_final_release' for b in r['draft_audit']['full_boundaries'])
print('Evidence complete; this does not require speedup or nonzero overlap.')
PY_EVIDENCE
  REPORTS+=("$REPORT")
  bash scripts/run_decode_scan.sh summary
  bash scripts/run_decode_scan.sh bundle --output "${SR_FIXED_ROOT}-complete-small.tar.gz"
  "$SR_FIXED_PYTHON" -m specrhythm.serving.eager_latency_bundle \
    --root "$SR_FIXED_ROOT" --output "${SR_FIXED_ROOT}-complete-evidence.tar.gz"
  sha256sum "${SR_FIXED_ROOT}-complete-evidence.tar.gz"
  printf 'RETURN: %s\n' "${SR_FIXED_ROOT}-complete-evidence.tar.gz"
done
COMPARE="$RESULTS/audit-four-point-${FINAL_SHA:0:12}-${RUN_TAG}.json"
"$SR_FIXED_PYTHON" -m specrhythm.serving.audit_layer_report \
  --reports "${REPORTS[@]}" --expected-commit "$FINAL_SHA" --output "$COMPARE"
printf 'RETURN COMPARISON: %s\n' "$COMPARE"
}
main "$@"
