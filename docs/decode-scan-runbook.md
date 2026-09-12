# Resident360 warmup boundary repair: PingPong B128 only

GPU qualification/performance for the repair is **PENDING**. No AutoDL/GPU/full
CPU audit was run by the agent. Keep the f718 root unchanged: PingPong B64,
Target B128 and Serial B128 remain their three valid historical points;
PingPong B128 stays FAILED/INVALID, excluded from comparisons. Do not copy its
snapshot throughput into a valid result or mix commits into one unexplained table.

The delivery supplies the block below with the final full SHA filled in. For this
repository template, export `SR_FIXED_COMMIT` to that exact SHA first. The outer
`if bash` catches the child status even if the interactive parent already uses
`set -e`. Strict mode and ERR trap stay inside the child; inspection failures do
not replace the original exit code. Any failure/insufficiency stops the sequence
and attempts a small failure bundle, while the interactive terminal remains open.

```bash
if bash <<'BASH'
set -Eeuo pipefail
cd /root/autodl-tmp/src/SpecRhythm
: "${SR_FIXED_COMMIT:?Set the full final SHA from the delivery}"
test -z "$(git -c core.fsmonitor=false status --porcelain)"
git fetch origin codex/vllm-serving-v0.1
git checkout --detach "$SR_FIXED_COMMIT"
test "$(git rev-parse HEAD)" = "$SR_FIXED_COMMIT"
export SR_FIXED_PYTHON=/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11
export PYTHONPATH="$PWD/src" PYTHONUNBUFFERED=1
export SR_FIXED_S1=/root/autodl-tmp/SpecRhythm-data/results/phase-s1/s1p-5a00049-20260909T144802Z-1469
export SR_FIXED_ROOT="/root/autodl-tmp/SpecRhythm-data/results/decode-scan/resident360-warmup-${SR_FIXED_COMMIT:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
printf 'NEW ROOT: %s\n' "$SR_FIXED_ROOT"
on_failure() {
  rc=$?; trap - ERR; set +e
  bash scripts/run_decode_scan.sh status
  bash scripts/run_decode_scan.sh errors
  bash scripts/run_decode_scan.sh summary
  bash scripts/run_decode_scan.sh bundle --output "${SR_FIXED_ROOT}-failure-small.tar.gz"
  exit "$rc"
}
trap on_failure ERR
bash scripts/run_decode_scan.sh prepare --s1 "$SR_FIXED_S1" \
  --observation buffered-live --identity-matching bound-prefix \
  --selection-seed 1666 --warmup-steps 2 --window-seconds 30 --repeats 1 \
  --setup-timeout 900 --drain-timeout 60
"$SR_FIXED_PYTHON" - <<'PY'
import json, os, pathlib
p = pathlib.Path(os.environ['SR_FIXED_ROOT'])
x = json.loads((p/'scan-config.json').read_text())
assert x['execution']['git_commit'] == os.environ['SR_FIXED_COMMIT']
assert x['pool_size'] == len(set(x['selection']['request_ids'])) == 360
assert x['workload_sha256'] == 'cdaf71adace15d229f5087b98f9fd162a958456226a660184fe03f5d6ebd8ff4'
assert x['options']['samples'] is None
assert x['options']['observation'] == 'buffered-live'
assert x['options']['identity_matching'] == 'bound-prefix'
print(json.dumps({k:x[k] for k in ('options','workload_sha256')}, indent=2))
PY
bash scripts/run_decode_scan.sh capacity --single-point --batch 128 --mode pingpong
bash scripts/run_decode_scan.sh run --single-point --batch 128 --mode pingpong
"$SR_FIXED_PYTHON" - <<'PY'
import json, os, pathlib
from specrhythm.serving.fixed_artifacts import point_reports
p = pathlib.Path(os.environ['SR_FIXED_ROOT'])
rows = [r for r in point_reports(p) if not r['point'].get('probe')]
assert len(rows) == 1
r = rows[0]
assert r['point']['test_order'] == 'independent-single-point'
assert r['mode'] == 'pingpong' and r['batch'] == 128
assert r['git_commit'] == os.environ['SR_FIXED_COMMIT']
assert all(r[k] == 'PASS' for k in ('capacity_status','execution_status','measurement_status','cleanup_status'))
assert r['formal_comparison_eligible'] is True and r['stop_reason'] == 'time_budget'
assert r['diagnostic_logging']['observation'] == 'buffered-live'
assert r['identity_matching']['mode'] == 'bound-prefix'
assert r['effective_exit_code'] == 0
assert r['measured_window_ms'] >= 30000 and r['target_steps'] > 0
assert r['actual_target_batch']['min'] == r['actual_target_batch']['max'] == 64
w = r['scan_warmup_boundary']
assert w['schema_version'] == 'specrhythm.decode-scan-warmup-boundary.v1'
assert w['status'] == 'OPEN' and w['completed_rotations'] == w['required_complete_rotations'] == 2
assert w['pending_step'] is None
assert w['completed_steps'] == 4 + len(w['historical_unpaired_steps'])
assert all(s['end_ns'] <= w['measurement_start_ns'] and s['B'] == 64 for s in w['steps'])
assert sum(s['committed_tokens'] for s in w['steps']) == w['committed_tokens_excluded']
assert w['initial_population']['active_requests'] == 128
assert all(len(w['initial_population']['cohorts'][c]) == 64 for c in ('A','B'))
f = pathlib.Path(r['artifact'])
snapshot = json.loads((f/'measurement-snapshot.json').read_text())
assert snapshot['scan_warmup_boundary'] == w
assert snapshot['measurement_complete'] and snapshot['drain_complete']
assert snapshot['actual_target_batch_histogram'] == {'64': r['target_steps']}
life = json.loads((f/'process-lifecycle.json').read_text())
assert life['cleanup_valid'] and life['owned_cleanup_completed'] and not life['remaining_owned_pids']
print(json.dumps({k:r[k] for k in ('artifact','measured_window_ms','committed_window_tokens',
    'decode_throughput_tok_s','complete_rotations','partial_rotations','scan_warmup_boundary')}, indent=2))
PY
bash scripts/run_decode_scan.sh summary
bash scripts/run_decode_scan.sh bundle --output "${SR_FIXED_ROOT}-complete-small.tar.gz"
BASH
then
  printf 'PingPong B128 foreground retest passed; no other GPU points were started.\n'
else
  rc=$?
  printf 'Retest stopped (rc=%s). Keep the failed root/bundle; terminal remains open.\n' "$rc"
fi
```

`prepare` reads configuration/environment and nvidia-smi inventory; no inference.
`capacity` loads the GPU engines, validates current B128 capacity, prefills all360
requests and drains with zero verification. `run` independently checks its own
capacity/prefill, then two complete warmup rotations and one continuous30s window.
Only PingPong B128 (actual B64 forwards) is scheduled, repeats1. Historical extra
warmup steps are retained/excluded; they are allowed only with a closed start
boundary and full active cohorts. A final measurement half-rotation is legitimate.
There is no sample12 cutoff. Setup900, window30 and drain60 are separate boundaries;
a step already issued before expiry can overrun30s and retains its true time/tokens.
Warmup/prefill/drain are excluded from decode throughput, and separately reported.
No full audit, stages, Target, Serial, Serial-split, G2/G3 or extra grid is invoked.

`--single-point` changes test order only. Any failed/insufficient point stays
excluded and blocks further runs in that root. A PASS requires valid execution,
full timed batches, accounting, boundary evidence and owned cleanup; a nonzero
exit, operator stop or insufficient window cannot become performance PASS.

## Inspection and controlled stop (no new inference, no full audit)

Use the exact new root printed by the foreground child in another terminal:

```bash
export SR_FIXED_ROOT=/root/autodl-tmp/SpecRhythm-data/results/decode-scan/REPLACE_WITH_PRINTED_NEW_ROOT
export SR_FIXED_PYTHON=/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11
cd /root/autodl-tmp/src/SpecRhythm
bash scripts/run_decode_scan.sh status
bash scripts/run_decode_scan.sh errors
# Only to request owned controlled stop:
# bash scripts/run_decode_scan.sh stop --wait-seconds 65
# After cleanup, including failure:
bash scripts/run_decode_scan.sh summary
bash scripts/run_decode_scan.sh bundle --output "${SR_FIXED_ROOT}-inspect-$(date -u +%Y%m%dT%H%M%SZ)-$$-small.tar.gz"
```

The small bundle retains `scan_warmup_boundary` in snapshot/light report, including
all compact warmup step records, actual times/token counts, historical-unpaired
indices, current pending index and start population. Runtime/backend raw logs are
not required to classify this boundary. The original360 identity/prefix/KV and
TP/accounting qualification still uses the full retained runtime in the run.
No additional old-root export or GPU experiment is needed for this CPU root-cause
fix. Old failed files are never rewritten by these commands.
