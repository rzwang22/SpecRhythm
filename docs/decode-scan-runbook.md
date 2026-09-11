# Resident360 fixed-time decode scan: foreground server runbook

GPU qualification/performance PENDING. The agent did not connect AutoDL, run GPU,
or invoke the full CPU audit. Use the complete SHA from the delivery as
`SR_FIXED_COMMIT`; the copyable delivery includes that literal SHA. Preserve old
roots and do not reuse their capacity evidence.

## Environment and independent root

```bash
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
export SR_FIXED_ROOT="/root/autodl-tmp/SpecRhythm-data/results/decode-scan/resident360-${SR_FIXED_COMMIT:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
printf '%s\n' "$SR_FIXED_ROOT"
trap 'rc=$?; trap - ERR; set +e; bash scripts/run_decode_scan.sh errors; bash scripts/run_decode_scan.sh summary; bash scripts/run_decode_scan.sh bundle --output "${SR_FIXED_ROOT}-failure-$(date -u +%Y%m%dT%H%M%SZ)-$$-small.tar.gz"; exit "$rc"' ERR
```

## Freeze/inventory, capacity, small batch then remaining points

Prepare checks inventory/environment (including nvidia-smi), no inference.
Capacity loads GPU engines and **prefills all360** then drains without verification.
The B16 three-mode capacity probes are a first check; each later run independently
checks its own exact B/mode/360 physical capacity before prefill/decode. B128 checks
real block/workspace/sequence/token-position capacity, never shrinks automatically.

```bash
bash scripts/run_decode_scan.sh prepare --s1 "$SR_FIXED_S1" \
  --observation buffered-live --identity-matching bound-prefix \
  --selection-seed 1666 --warmup-steps 2 --window-seconds 30 --repeats 1 \
  --setup-timeout 900 --drain-timeout 60
"$SR_FIXED_PYTHON" - <<'PY'
import json, os, pathlib
p = pathlib.Path(os.environ['SR_FIXED_ROOT'])
x = json.loads((p/'scan-config.json').read_text())
assert x['pool_size'] == len(set(x['selection']['request_ids'])) == 360
assert len(x['points']) == 12
assert x['options']['samples'] is None
assert x['options']['observation'] == 'buffered-live'
assert x['options']['identity_matching'] == 'bound-prefix'
print(json.dumps({k:x[k] for k in ('selection','options','workload_sha256')}, indent=2))
print((p/'topology.json').read_text())
PY
bash scripts/run_decode_scan.sh capacity --batch 16
bash scripts/run_decode_scan.sh run --batch 16
bash scripts/run_decode_scan.sh status
bash scripts/run_decode_scan.sh run --remaining
bash scripts/run_decode_scan.sh summary
bash scripts/run_decode_scan.sh bundle --output "${SR_FIXED_ROOT}-complete-small.tar.gz"
```

`run --batch 16` executes Target → Serial → PingPong in fresh engines, stopping
immediately on failure, insufficient measurement or stop. `run --remaining` checks
all three B16 points passed, skips them, then runs B32/B64/B128 in the same mode
order. It never retries failed/insufficient points. Each successful measurement
must have execution/measurement/cleanup PASS, actual full forward shape and at
least30s actual elapsed measurement. Pool exhaustion gives INSUFFICIENT and stops
subsequent GPU points; inspect the retained partial evidence and return the bundle.

An explicit individual continuation is `run --batch 16 --mode serial` (or another
selected point); passing points are skipped. Do not auto-retry a failed root.
The12×30s is only measurement budget: engine loads,360 prefill, two warmup rotations,
refill before the boundary and drain add time. Within-window refill and all normal
prefix/KV/commit work are timed. An issued step may overrun30s under supervision;
its actual denominator and tokens are retained. Final flush/drain are reported
separately. There is no sample12 cutoff, full audit, stages, λ scan or extra grid.

## Independent inspection and controlled stop (no new GPU point)

In another shell, export the exact printed `SR_FIXED_ROOT` and `SR_FIXED_PYTHON`,
then run from the repository:

```bash
bash scripts/run_decode_scan.sh status
bash scripts/run_decode_scan.sh errors
bash scripts/run_decode_scan.sh stop --wait-seconds 65
# After cleanup, including failure:
bash scripts/run_decode_scan.sh summary
bash scripts/run_decode_scan.sh bundle --output "${SR_FIXED_ROOT}-manual-$(date -u +%Y%m%dT%H%M%SZ)-$$-small.tar.gz"
```

These inspect small reports or request owned cancellation. They do not launch an
inference point or CPU audit. A stop can wait for the current bounded operation;
forced exit remains failure, not natural EOS or valid throughput. The retained
first error and original exits distinguish execution failure from cleanup status.
Both PARTIAL/INSUFFICIENT and failed data remain excluded from main comparison.
