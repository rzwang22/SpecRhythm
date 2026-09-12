#!/usr/bin/env bash
# Run from an outer `if bash ...; then ...; else ...; fi` as documented.
# Define the complete function before checkout: the executing body is in memory.
main() {
set -Eeuo pipefail
export SR_LATENCY_OBSERVE_SHA="${1:?full observation SHA required}"
export SR_LATENCY_FIX_SHA="${2:?full fix SHA required}"
export SR_FIXED_PYTHON="${SR_FIXED_PYTHON:-/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11}"
export SR_FIXED_S1="${SR_FIXED_S1:-/root/autodl-tmp/SpecRhythm-data/results/phase-s1/s1p-5a00049-20260909T144802Z-1469}"
export SR_LATENCY_RUN_TAG="${SR_LATENCY_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
export SR_EAGER_CAUSAL_TRACE=light
cd /root/autodl-tmp/src/SpecRhythm
export PYTHONPATH="$PWD/src" PYTHONUNBUFFERED=1
unset SR_FIXED_ROOT
on_failure() {
  rc=$?; trap - ERR; set +e
  printf 'Stopped at first error: rc=%s, commit=%s\n' "$rc" "${SR_FIXED_COMMIT:-unset}"
  if [[ -n "${SR_FIXED_ROOT:-}" && -d "$SR_FIXED_ROOT" ]]; then
    "$SR_FIXED_PYTHON" -m specrhythm.continuation.gpu_check status --root "$SR_FIXED_ROOT"
    "$SR_FIXED_PYTHON" -m specrhythm.continuation.gpu_check errors --root "$SR_FIXED_ROOT"
    bash scripts/run_decode_scan.sh status
    bash scripts/run_decode_scan.sh errors
    bash scripts/run_decode_scan.sh summary
    bash scripts/run_decode_scan.sh bundle --output "${SR_FIXED_ROOT}-failure-small.tar.gz"
    "$SR_FIXED_PYTHON" -m specrhythm.serving.eager_latency_bundle \
      --root "$SR_FIXED_ROOT" --output "${SR_FIXED_ROOT}-failure-evidence.tar.gz"
    printf 'Preserve failed root and bundle: %s\n' "$SR_FIXED_ROOT"
  fi
  exit "$rc"
}
trap on_failure ERR
[[ "$SR_LATENCY_OBSERVE_SHA" =~ ^[0-9a-f]{40}$ && "$SR_LATENCY_FIX_SHA" =~ ^[0-9a-f]{40}$ ]]
test -z "$(git -c core.fsmonitor=false status --porcelain)"
git fetch origin codex/rolling-eager-v0.1
git merge-base --is-ancestor "$SR_LATENCY_OBSERVE_SHA" "$SR_LATENCY_FIX_SHA"
for SR_FIXED_COMMIT in "$SR_LATENCY_OBSERVE_SHA" "$SR_LATENCY_FIX_SHA"; do
  export SR_FIXED_COMMIT
  export SR_FIXED_ROOT="/root/autodl-tmp/SpecRhythm-data/results/rolling-eager/serial-latency-B16-${SR_FIXED_COMMIT:0:12}-${SR_LATENCY_RUN_TAG}"
  test ! -e "$SR_FIXED_ROOT"
  test "$SR_LATENCY_OBSERVE_SHA" != "$SR_LATENCY_FIX_SHA"
  test -z "$(git -c core.fsmonitor=false status --porcelain)"
  git checkout --detach "$SR_FIXED_COMMIT"
  printf 'NEW ROOT: %s\n' "$SR_FIXED_ROOT"
test "$(git rev-parse HEAD)" = "$SR_FIXED_COMMIT"
bash scripts/run_decode_scan.sh prepare --s1 "$SR_FIXED_S1" \
  --observation buffered-live --identity-matching bound-prefix \
  --selection-seed 1666 --warmup-steps 2 --window-seconds 30 --repeats 1 \
  --setup-timeout 900 --drain-timeout 60
"$SR_FIXED_PYTHON" - <<'PY'
import json, os, pathlib
root = pathlib.Path(os.environ['SR_FIXED_ROOT'])
scan = json.loads((root / 'scan-config.json').read_text())
assert scan['execution']['git_commit'] == os.environ['SR_FIXED_COMMIT']
assert scan['workload_sha256'] == 'cdaf71adace15d229f5087b98f9fd162a958456226a660184fe03f5d6ebd8ff4'
assert scan['pool_size'] == len(set(scan['selection']['request_ids'])) == 360
assert scan['options']['samples'] is None
assert scan['options']['window_seconds'] == 30 and scan['options']['repeats'] == 1
assert scan['options']['warmup_steps'] == 2
assert scan['options']['setup_timeout'] == 900 and scan['options']['drain_timeout'] == 60
assert scan['options']['observation'] == 'buffered-live'
assert scan['options']['identity_matching'] == 'bound-prefix'
print(json.dumps({'root': str(root), 'workload_sha256': scan['workload_sha256'],
                  'options': scan['options']}, indent=2))
PY
# Fresh physical capacity/prefill checks, including eager's extra speculative KV.
if [[ "$SR_FIXED_COMMIT" == "$SR_LATENCY_FIX_SHA" ]]; then
  bash scripts/run_decode_scan.sh capacity --single-point --batch 16 --mode serial
fi
bash scripts/run_decode_scan.sh capacity --single-point --batch 16 --mode serial-eager
# Actual Draft worker + physical page/frontier checks; no CPU executor and no Target model.
"$SR_FIXED_PYTHON" -m specrhythm.continuation.gpu_check run \
  --root "$SR_FIXED_ROOT" --request-count 3 --timeout 900 --drain-timeout 60
"$SR_FIXED_PYTHON" - <<'PY'
import json, os, pathlib
p = pathlib.Path(os.environ['SR_FIXED_ROOT']) / 'rolling-eager-gpu-check'
state = json.loads((p / 'state.json').read_text())
result = json.loads((p / 'result.json').read_text())
assert state['status'] == 'COMPLETE' and state['cleanup_status'] == 'PASS'
assert not state['remaining_owned_pids'] and state['exit_code'] == 0
assert result['valid'] and result['physical_gpu_correctness'] == 'PASS'
assert result['cleanup_status'] == 'PASS'
assert result['backend']['eager_gpu_live_work'] == 0
assert not result['backend']['eager_gpu_write_inflight']
assert result['backend']['worker_resources']['blocks_allocated'] == result['backend']['worker_resources']['blocks_freed']
assert all(row['exact'] for row in result['reference_comparisons'])
print(json.dumps({'controlled_physical_cases': result['constructed_cases'],
                  'real_Target_event_observations': result['target_observed_events'],
                  'GPU_overlap': result['gpu_overlap']}, indent=2))
PY
# On the fixed version Serial must qualify before eager decode.
if [[ "$SR_FIXED_COMMIT" == "$SR_LATENCY_FIX_SHA" ]]; then
bash scripts/run_decode_scan.sh run --single-point --batch 16 --mode serial
"$SR_FIXED_PYTHON" - <<'PY'
import os, pathlib
from specrhythm.serving.fixed_artifacts import point_reports
root = pathlib.Path(os.environ['SR_FIXED_ROOT'])
rows = [r for r in point_reports(root) if not r['point'].get('probe')]
assert len(rows) == 1 and rows[0]['mode'] == 'serial' and rows[0]['batch'] == 16
r = rows[0]
assert all(r[k] == 'PASS' for k in ('capacity_status','execution_status','measurement_status','cleanup_status'))
assert r['formal_comparison_eligible'] and r['effective_exit_code'] == 0
assert r['stop_reason'] == 'time_budget' and r['measured_window_ms'] >= 30000
assert r['actual_target_batch']['min'] == r['actual_target_batch']['max'] == 16
print('Serial B16 execution, measurement and cleanup qualified; proceed to serial-eager B16.')
PY
fi
bash scripts/run_decode_scan.sh run --single-point --batch 16 --mode serial-eager
"$SR_FIXED_PYTHON" - <<'PY'
import json, os, pathlib
from specrhythm.serving.fixed_artifacts import point_reports
root = pathlib.Path(os.environ['SR_FIXED_ROOT'])
rows = [r for r in point_reports(root) if not r['point'].get('probe')]
expected = {'serial', 'serial-eager'} if os.environ['SR_FIXED_COMMIT'] == os.environ['SR_LATENCY_FIX_SHA'] else {'serial-eager'}
assert len(rows) == len(expected) and {r['mode'] for r in rows} == expected
assert len({r['workload_sha256'] for r in rows}) == 1
assert {r['git_commit'] for r in rows} == {os.environ['SR_FIXED_COMMIT']}
for r in rows:
    assert r['batch'] == 16 and r['effective_exit_code'] == 0
    assert all(r[k] == 'PASS' for k in ('capacity_status','execution_status','measurement_status','cleanup_status'))
    assert r['formal_comparison_eligible'] and r['stop_reason'] == 'time_budget'
    assert r['measured_window_ms'] >= 30000 and r['target_steps'] > 0
    assert r['actual_target_batch']['min'] == r['actual_target_batch']['max'] == 16
    life = json.loads((pathlib.Path(r['artifact']) / 'process-lifecycle.json').read_text())
    assert life['cleanup_valid'] and life['owned_cleanup_completed'] and not life['remaining_owned_pids']
eager = next(r for r in rows if r['mode'] == 'serial-eager')['rolling_eager']
c = eager['window_counters']
assert c['started'] > 0, 'serial-eager window did not actually start eager work'
assert eager['cleanup_status'] == 'PASS' and eager['owner_stopped']
assert eager['pending_work'] in (0, [])
life = eager['lifetime_counters']
assert life['completed'] <= life['started'] <= life['admissions']
assert life['accepted_promoted_candidates'] <= life['verified_promoted_candidates']
print(json.dumps({'acceptance': {
    'execution_and_cleanup': 'PASS', 'actual_eager_start': 'OBSERVED',
    'event_classes': eager['observation'],
    'GPU_overlap': eager['GPU_overlap'],
    'comparison_conditions': 'PASS',
    'window_counters': c, 'window_unhidden_wait_ns': eager['window_unhidden_wait_ns'],
}, 'comparison': [{k:r[k] for k in ('mode','artifact','measured_window_ms',
    'committed_window_tokens','decode_throughput_tok_s')} for r in rows]}, indent=2))
PY
report_modes=(serial-eager)
if [[ "$SR_FIXED_COMMIT" == "$SR_LATENCY_FIX_SHA" ]]; then report_modes=(serial serial-eager); fi
"$SR_FIXED_PYTHON" -m specrhythm.serving.eager_retest_report --modes "${report_modes[@]}" \
  --root "$SR_FIXED_ROOT" --expected-commit "$SR_FIXED_COMMIT" \
  --output "${SR_FIXED_ROOT}-attribution.json"
"$SR_FIXED_PYTHON" - <<'PY_CHECK'
import json, os
p = json.load(open(os.environ['SR_FIXED_ROOT']+'-attribution.json'))
for point in p['points'].values():
    coverage = point['critical_path']['trace_coverage']
    assert all(r['mode'] == 'light' and r['status'] == 'COMPLETE' for r in coverage.values())
    assert point['critical_path']['coordinator_exclusive']['status'] == 'COMPLETE'
print('Bounded light causal traces and main-thread partition complete.')
PY_CHECK
bash scripts/run_decode_scan.sh summary
bash scripts/run_decode_scan.sh bundle --output "${SR_FIXED_ROOT}-complete-small.tar.gz"
"$SR_FIXED_PYTHON" -m specrhythm.serving.eager_latency_render \
  --report "${SR_FIXED_ROOT}-attribution.json" --output "${SR_FIXED_ROOT}-timeline.svg"
"$SR_FIXED_PYTHON" -m specrhythm.serving.eager_latency_bundle \
  --root "$SR_FIXED_ROOT" --output "${SR_FIXED_ROOT}-complete-evidence.tar.gz"
sha256sum "${SR_FIXED_ROOT}-complete-evidence.tar.gz"
printf 'Return: %s\n' "${SR_FIXED_ROOT}-complete-evidence.tar.gz"
done
}
main "$@"
