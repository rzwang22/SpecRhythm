# Fixed64/32 bound-prefix: two foreground points

GPU performance PENDING. No AutoDL/GPU/full CPU audit was run by the agent.
The delivery includes a copy with the final full SHA filled in. Keep all previous
roots unchanged. Default sequence is prepare/capacity → Serial → check → PingPong
→ check → light summary/bundle. It never schedules another mode or a comparison grid.

## Checkout and environment (no inference)

```bash
set -euo pipefail
cd /root/autodl-tmp/src/SpecRhythm
: "${SR_FIXED_COMMIT:?Set the full final SHA supplied with this runbook}"
[[ "$SR_FIXED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
test -z "$(git -c core.fsmonitor=false status --porcelain)"
git fetch origin codex/vllm-serving-v0.1
git checkout --detach "$SR_FIXED_COMMIT"
test "$(git rev-parse HEAD)" = "$SR_FIXED_COMMIT"
export SR_FIXED_PYTHON=/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src"
export SR_FIXED_S1=/root/autodl-tmp/SpecRhythm-data/results/phase-s1/s1p-5a00049-20260909T144802Z-1469
export SR_FIXED_OLD=/root/autodl-tmp/SpecRhythm-data/results/fixed-concurrency/fixed64-buffered-c1dd96d8c86e-20260911T115700Z-1479
```

## Optional retained-evidence reanalysis (CPU only)

This reuses the existing external 120-second deadline and <=10 MiB report budget;
it prints file/event progress, preserves partial reports/rc, and imports no CUDA or
inference engine. The old root is read-only. No export or full audit is implicit.

```bash
export SR_FIXED_ANALYSIS="${SR_FIXED_OLD}-identity-review-${SR_FIXED_COMMIT:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
bash scripts/analyze_fixed_diagnostic.sh --input "$SR_FIXED_OLD" \
  --output "$SR_FIXED_ANALYSIS" --modes serial pingpong --timeout 120
cat "$SR_FIXED_ANALYSIS/report.md"
```

## New root and preparation

`prepare` queries inventory/environment, without model inference. `capacity` launches
the existing GPU capacity probes; no short measurement is hidden inside that command.
It confirms the real resident100 capacity, not a previously measured block count.
Models/tokenizer/GPU0 Draft/GPU1–2 Target TP2/dtype/eager settings are inherited from
the qualified S1 input; no vLLM patch installation or environment reconstruction here.

```bash
export SR_FIXED_ROOT="/root/autodl-tmp/SpecRhythm-data/results/fixed-concurrency/fixed64-identity-${SR_FIXED_COMMIT:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
printf '%s\n' "$SR_FIXED_ROOT"
set -E
trap 'rc=$?; trap - ERR; set +e; bash scripts/run_fixed_diagnostic.sh errors; bash scripts/run_fixed_diagnostic.sh summary; bash scripts/run_fixed_diagnostic.sh bundle --output "${SR_FIXED_ROOT}-failure-$(date -u +%Y%m%dT%H%M%SZ)-$$-small.tar.gz"; exit "$rc"' ERR
bash scripts/run_fixed_diagnostic.sh prepare --s1 "$SR_FIXED_S1" \
  --observation buffered-live --identity-matching bound-prefix \
  --warmup-steps 2 --samples 12 --repeats 1 --window-seconds 30 \
  --setup-timeout 900 --drain-timeout 60
"$SR_FIXED_PYTHON" - <<'PY'
import json, os, pathlib
p = pathlib.Path(os.environ['SR_FIXED_ROOT'])
r = json.loads((p/'diagnostic-config.json').read_text())
assert r['options']['observation'] == 'buffered-live'
assert r['options']['identity_matching'] == 'bound-prefix'
print(json.dumps(r['options'], indent=2))
PY
bash scripts/run_fixed_diagnostic.sh capacity
```

The launcher sets `SR_FIXED_IDENTITY_MATCHING=bound-prefix` and
`SR_FIXED_OBSERVATION=buffered-live` from the frozen manifest in each owned process.
Do not override these env vars later. The existing required UUID mode remains live.

## Compact pass/effect check (CPU only)

Define this once, then use it after each short point. It reads only the small final
summary. Missing reports/partial measurements/operator stop cannot pass. Historical
binding comparisons avoided are an operation count, not milliseconds saved.

```bash
check_fixed_point() {
  "$SR_FIXED_PYTHON" - "$1" <<'PY'
import json, math, os, pathlib, sys
mode = sys.argv[1]
paths = list(pathlib.Path(os.environ['SR_FIXED_ROOT']).glob('runs/continuous-'+mode+'-*/light-summary.json'))
assert len(paths) == 1, paths
r = json.loads(paths[0].read_text())
assert r['valid'] and all(r[k] == 'PASS' for k in ('execution_status','measurement_status','cleanup_status')), r
assert r['valid_samples'] == (12 if mode == 'serial' else 24)
assert r['actual_rotation_ms']['count'] == 12
assert r['actual_target_batch']['mean'] == (64 if mode == 'serial' else 32)
assert math.isclose(r['full_active_load_fraction'], 1.0)
assert r['cross_run_token_equality'] == 'NOT_REQUIRED'
logs = r['diagnostic_logging']
assert logs['observation'] == 'buffered-live'
assert len(logs['receipts']) == 4
assert all(x['status']=='COMPLETE' and x['integrity_complete'] for x in logs['receipts'])
m = r['identity_matching']
assert m['mode'] == 'bound-prefix'
assert set(m['by_owner']) == {'scheduler','target-rank-0','target-rank-1'}
assert all(x['prefix_free'] for x in m['by_owner'].values())
assert m['measured_scheduler_steps'] == r['valid_samples']
q = m['measured_scheduler']
assert q['validated_binding_reuses'] > 0 and q['binding_errors'] == 0
assert q['candidate_comparisons'] < q['linear_candidate_comparisons']
if mode == 'pingpong':
    ranks = r['uuid_query_by_rank']
    assert len(ranks) == 2
    for x in ranks:
        assert x['uuid_query_mode']=='live' and x['uuid_initial_validation_count']==1
        assert x['uuid_verification_subprocess_query_count'] >= r['valid_samples']
        assert x['uuid_cache_hit_count']==0
print(json.dumps({k:r.get(k) for k in ('mode','identity_matching','uuid_query_by_rank',
    'window_throughput_tok_s','committed_window_tokens','actual_rotation_ms',
    'recorded_gpu_costs','overlap','window_ms','drain_ms','arrival_to_drain_ms')}, indent=2))
PY
}
```

## Serial, then PingPong only after Serial passes (GPU)

```bash
bash scripts/run_fixed_diagnostic.sh short --mode serial
check_fixed_point serial
bash scripts/run_fixed_diagnostic.sh status
bash scripts/run_fixed_diagnostic.sh errors

bash scripts/run_fixed_diagnostic.sh short --mode pingpong
check_fixed_point pingpong
bash scripts/run_fixed_diagnostic.sh status
bash scripts/run_fixed_diagnostic.sh errors
```

Budget: warmup2 / samples12 / repeats1 / window30s / setup900s / drain60s. PingPong
uses 24 B32 steps for 12 complete64 rotations; Serial uses 12 B64 steps. The sample
budget can end before 30 seconds; issued step overshoot is recorded. Thirty seconds
is not a loading/setup/drain total deadline. The final flush remains in the common
bounded drain and full execution cost. No natural EOS/completion is invented for
diagnostic cancellations; full-request TPOT/SLO are unavailable.

`sample_budget` with complete evidence is a normal end; operator_stop, insufficient
samples, incomplete flush/cleanup or any failure stops this sequence and excludes
the point from performance comparisons. The error trap retains original rc and
exports a small failure bundle, even when final runtime/backend reports are absent.
It never starts a replacement point. A non-prefix-free workload may execute correctly
with full scans; the effect check will stop instead of claiming the optimization ran.

## Summary and small bundle (CPU only)

```bash
bash scripts/run_fixed_diagnostic.sh summary
bash scripts/run_fixed_diagnostic.sh bundle --output "${SR_FIXED_ROOT}-small.tar.gz"
trap - ERR
```

Compare against c1dd with **buffered-live/linear** explicitly labeled; the new point
is **buffered-live/bound-prefix** plus the new code SHA. Any observed improvement is
a runtime optimization result, not an algorithm improvement. Include final flush,
drain and full execution cost. Overlap need not increase for the optimization to help.
Do not interpret the legacy 52.70 ms arithmetic residual as the remaining bottleneck.

## Status, errors, controlled stop (another terminal; no GPU launch)

Reuse the exact printed root and Python path; these commands do not require a new
checkout or start another point.

```bash
export SR_FIXED_ROOT=/root/autodl-tmp/SpecRhythm-data/results/fixed-concurrency/REPLACE_WITH_PRINTED_NEW_ROOT
export SR_FIXED_PYTHON=/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11
cd /root/autodl-tmp/src/SpecRhythm
bash scripts/run_fixed_diagnostic.sh status
bash scripts/run_fixed_diagnostic.sh errors
bash scripts/run_fixed_diagnostic.sh stop --wait-seconds 65
```

Stop uses the existing owned cleanup, with bounded TERM/KILL grace if needed. Never
delete the previous root. On a failure, summary/bundle remain safe standalone commands.

## Optional same-commit off control; not part of this sequence

Only when separately requested: use another independent root, run `prepare` with
the same budgets and `--observation buffered-live --identity-matching linear`, then
capacity and only the requested single mode. Do not reuse the optimized root or
invoke its bound-prefix effect check on a linear point. The disabled matching path
uses the original full search; logging selection stays identical. No extra point is
automatically run by this runbook. Stop and return the two-mode bundle first.
