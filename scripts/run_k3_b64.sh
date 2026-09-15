#!/usr/bin/env bash
# Run in the independent child Bash supplied by the pinned launcher/runbook.
main() {
set -Eeuo pipefail
FINAL_SHA="${1:?full execution SHA required}"
export SR_FIXED_PYTHON="${SR_FIXED_PYTHON:-/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11}"
export SR_FIXED_S1="${SR_FIXED_S1:-/root/autodl-tmp/SpecRhythm-data/results/phase-s1/s1p-5a00049-20260909T144802Z-1469}"
RUN_TAG="${SR_PING_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
# Public entry always routes through the local supervisor. The private inner
# path is set by that supervisor, not derived from the persistent results root.
if [[ -z "${SR_K3_LOCAL_DELIVERY:-}" ]]; then
  cd "${SR_EXEC_REPO:-/root/autodl-tmp/src/SpecRhythm}"
  export PYTHONPATH="$PWD/src"
  exec "$SR_FIXED_PYTHON" -m specrhythm.serving.k3_local_run --repo "$PWD" --commit "$FINAL_SHA" --k3-configuration k3-b64-v1
fi
export SR_PING_DELIVERY="$SR_K3_LOCAL_DELIVERY"
ARCHIVE="${SR_PING_DELIVERY}.tar.gz"
export SR_EAGER_CAUSAL_TRACE=light SR_EAGER_CAUSAL_LAYOUT=phased
export SR_AUDIT_MODE=runtime SR_FIXED_COMMIT="$FINAL_SHA"
STAGE=preflight
POINT=not_started
cd "${SR_EXEC_REPO:-/root/autodl-tmp/src/SpecRhythm}"
export PYTHONPATH="$PWD/src" PYTHONUNBUFFERED=1
# Refuse collisions before installing export trap: never touch a prior run.
test ! -e "$SR_PING_DELIVERY"
test ! -e "$ARCHIVE"
mkdir -p "$SR_PING_DELIVERY/points"
if [[ -n "${SR_K3_STORAGE_RECEIPT:-}" ]]; then
  cp "$SR_K3_STORAGE_RECEIPT" "$SR_PING_DELIVERY/storage-preflight.json"
fi
# Before a point starts, never attribute a static failure to an inherited old root.
export SR_FIXED_ROOT="$SR_PING_DELIVERY/not_started"
finish() {
  first_rc=$?; trap - EXIT ERR; set +e
  if [[ "$first_rc" != 0 ]]; then
    printf 'FIRST FAILURE: rc=%s stage=%s; actual mode/run and original qualification follow\n' "$first_rc" "$STAGE"
    "$SR_FIXED_PYTHON" - "$first_rc" "$STAGE" "$POINT" <<'PY_FAILURE'
import json, os, pathlib, sys
from specrhythm.serving.execution_failure import summarize, summarize_joint
from specrhythm.serving.audit_layer_report import write
d=pathlib.Path(os.environ['SR_PING_DELIVERY'])
root=pathlib.Path(os.environ.get('SR_FIXED_ROOT', str(d/'not_started')))
if sys.argv[2]=='joint_gpu_correctness':
    v=summarize_joint(d/'joint', int(sys.argv[1]))
else:
    v=summarize(root, int(sys.argv[1]), sys.argv[2])
    v['point']=sys.argv[3]
v.update(evidence_packages=[str(d)+'.tar.gz'])
print(json.dumps(v), flush=True)
write(v, d/'first-failure.json')
PY_FAILURE
    summary_rc=$?
    printf 'Secondary failure-summary rc=%s; original rc=%s\n' "$summary_rc" "$first_rc"
  fi
  if [[ "${SR_K3_MANAGED_LOCAL:-}" == 1 ]]; then
    "$SR_FIXED_PYTHON" - "$first_rc" "$STAGE" "$POINT" "${summary_rc:-0}" <<'PY_OUTCOME'
import os, pathlib, sys
from specrhythm.phase4.manifest import atomic_write_json
atomic_write_json(pathlib.Path(os.environ['SR_PING_DELIVERY'])/'runner-outcome.json',
    dict(first_exit_code=int(sys.argv[1]), stage=sys.argv[2], mode=sys.argv[3],
         run_root=os.environ['SR_FIXED_ROOT'], summary_exit_code=int(sys.argv[4])))
PY_OUTCOME
    outcome_rc=$?
    printf 'Runner stopped: original rc=%s; outcome write rc=%s; local archive follows.\n' "$first_rc" "$outcome_rc"
    if [[ "$first_rc" != 0 ]]; then exit "$first_rc"; fi
    exit "$outcome_rc"
  fi
  "$SR_FIXED_PYTHON" -m specrhythm.serving.ping_prepost_delivery \
    --directory "$SR_PING_DELIVERY" --output "$ARCHIVE" --first-code "$first_rc" --stage "$STAGE" --k3 \
    > "$SR_PING_DELIVERY/export.log" 2>&1
  export_rc=$?
  printf 'Export rc=%s; original rc=%s; details=%s\n' "$export_rc" "$first_rc" "$SR_PING_DELIVERY/export.log"
  # Exactly one requested upload, including failed runs. No nested evidence archives.
  if [[ -f "$ARCHIVE" ]]; then
    printf 'UPLOAD ONLY: %s\n' "$ARCHIVE"
  else
    printf 'EXPORT FAILED: archive unavailable; source evidence remains at %s\n' "$SR_PING_DELIVERY"
  fi
  if [[ "$first_rc" != 0 ]]; then exit "$first_rc"; fi
  exit "$export_rc"
}
trap finish EXIT
[[ "$FINAL_SHA" =~ ^[0-9a-f]{40}$ ]]
[[ "$RUN_TAG" =~ ^[a-zA-Z0-9._-]+$ ]]
test -z "$(git -c core.fsmonitor=false status --porcelain)"
test "$(git rev-parse HEAD)" = "$FINAL_SHA"
STAGE=static_capacity_contract
"$SR_FIXED_PYTHON" -m specrhythm.serving.k3_capacity \
  --output "$SR_PING_DELIVERY/k3-capacity-contract.json" --k3-configuration k3-b64-v1
MODES=(serial-k3 serial-eager-k3 pingpong-k3 pingpong-eager-k3)
for POINT in "${MODES[@]}"; do
  STAGE=prepare
  export SR_AUDIT_SERVING_MODE="$POINT" SR_FIXED_ROOT="$SR_PING_DELIVERY/points/$POINT"
  bash scripts/run_decode_scan.sh prepare --k3-configuration k3-b64-v1 --s1 "$SR_FIXED_S1" --draft-audit runtime \
    --observation buffered-live --identity-matching bound-prefix --selection-seed 1666 \
    --warmup-steps 2 --window-seconds 30 --repeats 1 --setup-timeout 900 --drain-timeout 60
  "$SR_FIXED_PYTHON" - <<'PY_CONFIG'
import os,pathlib
from specrhythm.serving.common import read_json
root=pathlib.Path(os.environ['SR_FIXED_ROOT']); s=read_json(root/'scan-config.json')
assert s['execution']['git_commit']==os.environ['SR_FIXED_COMMIT']
assert s['workload_sha256']=='cdaf71adace15d229f5087b98f9fd162a958456226a660184fe03f5d6ebd8ff4'
assert s['pool_size']==len(set(s['selection']['request_ids']))==360
o=s['options']
assert o['draft_audit']=='runtime' and o['observation']=='buffered-live'
assert o['identity_matching']=='bound-prefix' and o['samples'] is None
assert (o['warmup_steps'],o['window_seconds'],o['repeats'],o['setup_timeout'],o['drain_timeout'])==(2,30,1,900,60)
m=read_json(root/'inputs/execution-B64.json')
c=m['fixed_diagnostic']['capacity'][os.environ['SR_AUDIT_SERVING_MODE']]
from specrhythm.serving.k3 import geometry, matches_geometry
g=geometry(os.environ['SR_AUDIT_SERVING_MODE'], "k3-b64-v1")
assert matches_geometry(c['execution_geometry'],os.environ['SR_AUDIT_SERVING_MODE'], 'k3-b64-v1')
assert s['k3_configuration']==m['k3_configuration']==c['k3_configuration']=='k3-b64-v1'
assert (m['active_limit'],c['per_cohort_capacity'],c['max_requests_per_target_forward'])==(64,max(g['home_capacities'].values()),g['target_request_ceiling'])
assert c['proposal_budget']==3
PY_CONFIG
  STAGE=capacity
  bash scripts/run_decode_scan.sh capacity --single-point --batch 64 --mode "$POINT"
done
STAGE=joint_gpu_correctness
POINT=joint
export SR_FIXED_ROOT="$SR_PING_DELIVERY/joint"
"$SR_FIXED_PYTHON" -m specrhythm.serving.k3_gpu_check \
  --source "$SR_PING_DELIVERY/points/${MODES[0]}" --output "$SR_PING_DELIVERY/joint" --k3-configuration k3-b64-v1
for POINT in "${MODES[@]}"; do
  export SR_AUDIT_SERVING_MODE="$POINT" SR_FIXED_ROOT="$SR_PING_DELIVERY/points/$POINT"
  STAGE=execution
  bash scripts/run_decode_scan.sh run --single-point --batch 64 --mode "$POINT"
  STAGE=measurement
  "$SR_FIXED_PYTHON" - <<'PY_MEASUREMENT'
import os,pathlib
from specrhythm.serving.common import read_json
from specrhythm.serving.fixed_artifacts import point_reports
from specrhythm.serving.k3 import geometry
from specrhythm.serving.k3_acceptance import measurement
rows=[r for r in point_reports(pathlib.Path(os.environ['SR_FIXED_ROOT'])) if not r['point'].get('probe')]
assert len(rows)==1
r=rows[0]
mode=os.environ['SR_AUDIT_SERVING_MODE']
# Includes formal_comparison_eligible and original run qualification, then native TP checks.
measurement(r,read_json(pathlib.Path(r['artifact'])/'runtime.json'),mode, 'k3-b64-v1')
g=geometry(mode, "k3-b64-v1")
print(f"Original execution/measurement/cleanup PASS; mode={mode}, real Target ceiling{g['target_request_ceiling']}, active{g['active_limit']}.")
PY_MEASUREMENT
  STAGE=diagnostic_evidence
  "$SR_FIXED_PYTHON" -m specrhythm.serving.ping_prepost_evidence \
    --root "$SR_FIXED_ROOT" --commit "$FINAL_SHA" --k3-configuration k3-b64-v1 --output "${SR_FIXED_ROOT}-audit-report.json" \
    --status "${SR_FIXED_ROOT}-evidence-status.json"
done
STAGE=paired_comparison
"$SR_FIXED_PYTHON" - <<'PY_COMPARE'
import os,pathlib
from specrhythm.serving.ping_prepost_delivery import comparison
from specrhythm.serving.k3 import MODES
from specrhythm.serving.audit_layer_report import write
p=pathlib.Path(os.environ['SR_PING_DELIVERY']); v=comparison(p,modes=MODES)
write(v,p/'comparison.json')
assert v['valid'], 'paired execution/diagnostics/configuration/joint correctness incomplete'
PY_COMPARE
STAGE=complete
}
main "$@"
