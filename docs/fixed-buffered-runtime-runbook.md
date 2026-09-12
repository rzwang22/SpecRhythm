# Fixed64/32 buffered-live: CPU review, then two foreground short points

GPU performance PENDING. Operator commands only; the agent did not connect to AutoDL
or run GPU. PR #4 remains Draft/Open. Keep the old b424 root unchanged. This run uses
one new immutable root, one Serial point followed by one PingPong point with the same
buffered-live persistence policy. No full audit, stages, other modes, G2/G3 or grid.
The delivery supplies the final full `SR_FIXED_COMMIT` and a filled copy of these commands.

## Checkout (no GPU)

```bash
set -euo pipefail
cd /root/autodl-tmp/src/SpecRhythm
: "${SR_FIXED_COMMIT:?Set the final full SHA from the delivery}"
[[ "$SR_FIXED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
test -z "$(git -c core.fsmonitor=false status --porcelain)"
git fetch origin codex/vllm-serving-v0.1
git checkout --detach "$SR_FIXED_COMMIT"
test "$(git rev-parse HEAD)" = "$SR_FIXED_COMMIT"
export SR_FIXED_PYTHON=/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src"
export SR_ATTR_OLD=/root/autodl-tmp/SpecRhythm-data/results/fixed-concurrency/fixed64-stop-b42401585a6b-20260911T085722Z-1496
```

## Offline correction and minimal clock export (CPU only)

No CUDA initialization or full audit. New sibling output; old results are not edited.
External 120 s deadline includes parsing and analysis, prints progress, preserves
partial JSON and real analysis exit code on failure/timeout. Default output <=10 MiB.

```bash
export SR_ATTR_OUT="${SR_ATTR_OLD}-cost-correction-${SR_FIXED_COMMIT:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
bash scripts/analyze_fixed_diagnostic.sh --input "$SR_ATTR_OLD" \
  --output "$SR_ATTR_OUT" --modes serial pingpong serial-split --timeout 120
cat "$SR_ATTR_OUT/report.md"
"$SR_FIXED_PYTHON" - <<'PY'
import json, os, pathlib
p = pathlib.Path(os.environ['SR_ATTR_OUT'])
r = json.loads((p / 'attribution.json').read_text())
small = {
    'analysis_exit': json.loads((p / 'analysis-exit.json').read_text()),
    'source_artifacts': r['source_artifacts'],
    'split_clock': [dict(artifact=a['artifact_directory'],
                        overlap=a.get('raw', {}).get('overlap'),
                        clock_failure_details=a.get('raw', {}).get('clock_failure_details'),
                        gaps=a.get('gaps')) for a in r['attempts'] if a['mode']=='serial-split']}
with (p / 'serial-split-clock-failures.json').open('x') as f:
    json.dump(small, f, indent=2)
print(json.dumps(small, indent=2))
PY
```

Return only `serial-split-clock-failures.json` for the clock question. It includes
source hashes, failing forward index/role/rank, real projected bounds, host timestamps,
uncertainty and width errors (at most 16 rows). No request prefixes/KV/models/full logs
or full result root. If raw rows are absent or parsing fails, preserve UNKNOWN; do not
rerun GPU for this question. CPU analysis rc=0 means completed analysis, not GPU PASS.

## New root and prepare/capacity

`prepare` inspects inventory/environment without model inference; `capacity` launches
GPU engines and checks actual resident100 capacity with zero verification. No hardcoded
old capacity substitute. Frozen workload/model/GPU/TP/eager settings come from the
qualified S1 root. Explicit observation selection is only at prepare, then inherited
from the frozen manifest; shell overrides cannot silently change the prepared policy.

```bash
export SR_FIXED_S1=/root/autodl-tmp/SpecRhythm-data/results/phase-s1/s1p-5a00049-20260909T144802Z-1469
export SR_FIXED_ROOT="/root/autodl-tmp/SpecRhythm-data/results/fixed-concurrency/fixed64-buffered-${SR_FIXED_COMMIT:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
bash scripts/run_fixed_diagnostic.sh prepare --s1 "$SR_FIXED_S1" \
  --observation buffered-live --warmup-steps 2 --samples 12 --repeats 1 \
  --window-seconds 30 --setup-timeout 900 --drain-timeout 60
"$SR_FIXED_PYTHON" - <<'PY'
import json, os, pathlib
p=pathlib.Path(os.environ['SR_FIXED_ROOT'])
r=json.loads((p/'diagnostic-config.json').read_text())
assert r['options']['observation']=='buffered-live'
print(json.dumps(r['options'], indent=2))
PY
bash scripts/run_fixed_diagnostic.sh capacity
```

## Serial first (GPU), inspect before PingPong

```bash
bash scripts/run_fixed_diagnostic.sh short --mode serial
bash scripts/run_fixed_diagnostic.sh status
bash scripts/run_fixed_diagnostic.sh errors
```

Only proceed when Serial has execution/measurement/cleanup PASS, requested 12 samples,
64 active/actual B64, correct accounting, and all four diagnostic receipts COMPLETE.
Normal sample_budget may stop before 30 seconds; the issued step's real overshoot is
retained. Thirty seconds is not a model loading/setup/drain total deadline. Drain has
its shared 60 s deadline, with supervisor's existing bounded TERM/KILL grace afterward.
Operator_stop is INSUFFICIENT, not a natural request completion or a comparison PASS.
Failure/nonzero rc, missing receipt, incomplete cleanup or insufficient samples: stop
here and export evidence. Never auto-run the next point to compensate.

This CPU-only metadata check prints counts and actual live-UUID evidence:

```bash
"$SR_FIXED_PYTHON" - <<'PY'
import json, os, pathlib
root=pathlib.Path(os.environ['SR_FIXED_ROOT'])
rows=list(root.glob('runs/continuous-serial-*/light-summary.json'))
assert len(rows)==1, rows
r=json.loads(rows[0].read_text())
assert r['valid'] and all(r[k]=='PASS' for k in
                         ('execution_status','measurement_status','cleanup_status'))
assert r['valid_samples']==12
assert r['actual_target_batch']['mean']==64
assert r['diagnostic_logging']['observation']=='buffered-live'
receipts=r['diagnostic_logging']['receipts']
assert len(receipts)==4 and all(x['status']=='COMPLETE' and x['integrity_complete'] for x in receipts)
print(json.dumps({k:r.get(k) for k in ('diagnostic_logging','uuid_query_by_rank',
    'host_costs_by_process','window_ms','drain_ms','arrival_to_drain_ms',
    'recorded_gpu_costs','actual_rotation_ms','overlap')}, indent=2))
PY
```

Serial has no Dual UUID helper; its existing device/verification identity path is
unchanged. PingPong retains one live UUID subprocess query per verification per rank,
plus startup validation, with zero cache hits. Inspect actual rank evidence and
nvidia-smi host counts; do not compare a missing operation with a missing observer.

## PingPong only after the above passes (GPU)

```bash
bash scripts/run_fixed_diagnostic.sh short --mode pingpong
bash scripts/run_fixed_diagnostic.sh status
bash scripts/run_fixed_diagnostic.sh errors
```

Require execution/measurement/cleanup PASS, 24 B32 steps/12 full64 rotations and the
same four buffered receipts; verify actual per-rank live query counts. Do not require
positive GPU overlap: goal is real throughput/rotation improvement with valid evidence.
Do not repeat Serial, run other modes or add tests automatically. Keep full-request
TPOT/SLO unavailable for diagnostic cancellations.

## Lightweight summary and bundle (CPU only, also on failure)

```bash
bash scripts/run_fixed_diagnostic.sh summary
bash scripts/run_fixed_diagnostic.sh bundle --output "${SR_FIXED_ROOT}-small.tar.gz"
```

Small bundle includes measurement snapshot, drain and logging receipts, exit codes,
primary/secondary errors and compact results even if final runtime/backend is missing.
It excludes raw traces and full audit. Runtime optimization effects remain GPU PENDING
until these operator tests return. Compare new buffered-live/code with old original-live/
b424 explicitly; a difference is not an algorithm improvement. Include final flush,
drain and full arrival-to-drain costs when discussing any change in measured throughput.
Use offline analysis against the new root separately if more timing evidence is needed.

## Another terminal: status/errors/controlled stop (no new GPU launch)

Reuse the same exact root and Python environment printed in the foreground terminal:

```bash
export SR_FIXED_ROOT=/root/autodl-tmp/SpecRhythm-data/results/fixed-concurrency/REPLACE_WITH_ACTUAL_NEW_ROOT
export SR_FIXED_PYTHON=/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11
cd /root/autodl-tmp/src/SpecRhythm
bash scripts/run_fixed_diagnostic.sh status
bash scripts/run_fixed_diagnostic.sh errors
bash scripts/run_fixed_diagnostic.sh stop --wait-seconds 65
```

Stop publishes the existing controlled-stop request; if the wait expires it uses only
owned cleanup. It does not launch another point. An operator stop never promotes a
partial point to PASS. Keep original failure/result directories, then send the small
bundle. The agent stops after delivery and waits for server evidence.
