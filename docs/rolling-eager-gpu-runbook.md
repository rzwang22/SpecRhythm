# Rolling Eager stage 2: operator GPU correctness and Serial B16 comparison

GPU correctness, overlap and performance are **PENDING** until this runbook is
executed on the server. This change remains Draft PR #5. It does not modify the
PR #4 branch or historical result roots. Only Serial B16 and explicitly selected
Serial-eager B16 are measured; no PingPong, other batch size or automatic scan is
started.

The repository keeps a commit template because a commit cannot contain its own
hash. The delivery includes a standalone copy of this runbook with
`__FINAL_FULL_SHA__` replaced by the actual full committed SHA. Execute that filled
copy. The commands use the existing server repository, Python and qualified S1
root below, and create a new result root. The same frozen resident360 workload,
models, TP1 Draft GPU0 / TP2 Target GPUs1–2, seed and measurement settings apply
to both modes.

```bash
export SR_FIXED_COMMIT=__FINAL_FULL_SHA__
export SR_FIXED_PYTHON=/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11
export SR_FIXED_S1=/root/autodl-tmp/SpecRhythm-data/results/phase-s1/s1p-5a00049-20260909T144802Z-1469
export SR_FIXED_ROOT="/root/autodl-tmp/SpecRhythm-data/results/rolling-eager/serial-B16-${SR_FIXED_COMMIT:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
if bash <<'BASH'
set -Eeuo pipefail
cd /root/autodl-tmp/src/SpecRhythm
export PYTHONPATH="$PWD/src" PYTHONUNBUFFERED=1
printf 'NEW ROOT: %s\n' "$SR_FIXED_ROOT"
on_failure() {
  rc=$?; trap - ERR; set +e
  "$SR_FIXED_PYTHON" -m specrhythm.continuation.gpu_check status --root "$SR_FIXED_ROOT"
  "$SR_FIXED_PYTHON" -m specrhythm.continuation.gpu_check errors --root "$SR_FIXED_ROOT"
  bash scripts/run_decode_scan.sh status
  bash scripts/run_decode_scan.sh errors
  bash scripts/run_decode_scan.sh summary
  bash scripts/run_decode_scan.sh bundle --output "${SR_FIXED_ROOT}-failure-small.tar.gz"
  exit "$rc"
}
trap on_failure ERR
[[ "$SR_FIXED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
test -z "$(git -c core.fsmonitor=false status --porcelain)"
git fetch origin codex/rolling-eager-v0.1
git checkout --detach "$SR_FIXED_COMMIT"
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
bash scripts/run_decode_scan.sh capacity --single-point --batch 16 --mode serial
bash scripts/run_decode_scan.sh capacity --single-point --batch 16 --mode serial-eager
# Actual Draft worker + physical page/frontier checks; no CPU executor and no Target model.
"$SR_FIXED_PYTHON" -m specrhythm.continuation.gpu_check run \
  --root "$SR_FIXED_ROOT" --request-count 2 --timeout 900 --drain-timeout 60
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
# Serial must qualify first. Any failure stops before serial-eager decode.
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
bash scripts/run_decode_scan.sh run --single-point --batch 16 --mode serial-eager
"$SR_FIXED_PYTHON" - <<'PY'
import json, os, pathlib
from specrhythm.serving.fixed_artifacts import point_reports
root = pathlib.Path(os.environ['SR_FIXED_ROOT'])
rows = [r for r in point_reports(root) if not r['point'].get('probe')]
assert len(rows) == 2 and {r['mode'] for r in rows} == {'serial', 'serial-eager'}
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
bash scripts/run_decode_scan.sh summary
bash scripts/run_decode_scan.sh bundle --output "${SR_FIXED_ROOT}-complete-small.tar.gz"
BASH
then
  printf 'B16 sequence complete. Keep this root and bundle; no additional GPU points were started.\n'
else
  rc=$?
  printf 'Stopped at first failure (rc=%s). Preserve root/bundle; this terminal remains open.\n' "$rc"
fi
```

The parent shell retains `SR_FIXED_ROOT`, so subsequent commands refer to the
same root. Strict mode and the error trap live only in the child Bash; the
parent's `if/then/else` receives the original failure code. Inspection/export
failures do not replace that first error. A failed capacity, correctness or Serial
point prevents later measurement points; there is no automatic rerun or scan
expansion.

`prepare` inventories the environment and seals inputs; it does not perform model
inference. Each `capacity` creates fresh engines, prefills resident360 and drains
without decode measurement. The eager capacity plan includes its larger
speculative KV reservation. Each measured point again owns fresh engines and
physical capacity checks. It excludes model load, prefill and two complete warmup
rotations, then measures one continuous 30-second window with `samples=None` and
`repeats=1`. All window waiting, dependent generation, recovery and protocol costs
remain inside elapsed decode time. The last issued step may overrun the window;
its actual time/tokens remain visible. Drain shares one 60-second deadline across
owner work, feedback, physical release and log flush. Forced cleanup cannot turn
a failed drain into a valid performance result.

## What the two GPU checks establish

`gpu_check` is a small **real Draft GPU** regression, using the shared protocol
and `RollingVllmDraftBackend`. Each live request prefills once. Complete parent
acceptance, parent rejection and bonus mismatch are deliberately constructed
receipts to exercise retained paged KV, cached logits, correction/bonus repair and
subsequent rolling. Promoted and recovered tokens are compared to ordinary Draft
generation on independent private reference allocations at the same prefix.
Only those reference allocations replay complete prefixes, outside performance
measurement. The worker's existing source/device guards, actual request/block
metadata, InputBatch page tables, fenced forwards and allocation/release counters
provide physical evidence. CPU tests substitute device operations only; server
results must come from the real worker.

Controlled receipts are explicitly `INJECTED_DIAGNOSTIC` and never contribute to
real Target observation counts. Natural EOS is honored. If it prevents one of the
bounded construction cases, that case is `NOT_OBSERVED`; the harness does not
invent a completion or enlarge the test. The actual `serial-eager` B16 run is the
source of natural Target rejection, bridge matching, promotion, consumption and
acceptance evidence.

Execution/cleanup qualification, eager activation, reuse observations, overlap
and comparison conditions are separate results. Missing promotion, consumption,
accepted candidates, rejection or bridge mismatch remains `NOT_OBSERVED`.
Promotion count is not committed output. Committed tokens equal accepted parent
tokens plus correction/bonus, each once; the bridge is not emitted again. Summary
reports generation, promotion, actual verification and acceptance separately.
Overlap is `OBSERVED` only when native CUDA clock bounds prove a positive lower
bound of intersection; host enqueue/call interval overlap alone yields `UNKNOWN`.
There is no throughput or speedup pass threshold.

## Same-root inspection, errors, stop and failure export

In the same terminal the exports above remain set. In another terminal, export
`SR_FIXED_ROOT` to the printed new root, set `SR_FIXED_PYTHON` to the same server
Python, and change to `/root/autodl-tmp/src/SpecRhythm` first.

```bash
cd /root/autodl-tmp/src/SpecRhythm
export PYTHONPATH="$PWD/src"
# Read-only Draft correctness inspection, including when it is still running:
"$SR_FIXED_PYTHON" -m specrhythm.continuation.gpu_check status --root "$SR_FIXED_ROOT"
"$SR_FIXED_PYTHON" -m specrhythm.continuation.gpu_check errors --root "$SR_FIXED_ROOT"
# Only while the Draft correctness harness is RUNNING, request its controlled stop:
# "$SR_FIXED_PYTHON" -m specrhythm.continuation.gpu_check stop --root "$SR_FIXED_ROOT"
# Capacity/measurement point inspection:
bash scripts/run_decode_scan.sh status
bash scripts/run_decode_scan.sh errors
# Only while a capacity/measurement point is RUNNING, request its controlled stop:
# bash scripts/run_decode_scan.sh stop --wait-seconds 65
# After the relevant owner has exited and cleanup has completed, including failure:
bash scripts/run_decode_scan.sh summary
bash scripts/run_decode_scan.sh bundle --output "${SR_FIXED_ROOT}-inspect-$(date -u +%Y%m%dT%H%M%SZ)-$$-small.tar.gz"
```

The Draft correctness supervisor records and rechecks process start identity,
handles its own stop request, and preserves worker errors and `worker.log` under
`rolling-eager-gpu-check/`. Runtime stop remains the existing owned-process
diagnostic stop; neither command starts inference or targets processes by name.
The small bundle includes compact harness result/state/errors and runtime
measurement, eager summary and cleanup evidence. Full model/raw timeline evidence
remains in the new root. Keep all failed artifacts; do not copy partial throughput
into a qualified comparison or modify older result roots.
