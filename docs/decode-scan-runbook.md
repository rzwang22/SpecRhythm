# Resident360 readiness repair: foreground server runbook

New-code GPU qualification/performance is **PENDING**. No AutoDL/GPU/full CPU audit
was run by the agent. Preserve the e39 root and its eight valid historical points;
PingPong B64 stays INSUFFICIENT. Never copy its reports into the new root.
The delivery provides this command with the final full SHA filled in. In the
repository template, set `SR_FIXED_COMMIT` to that literal SHA before execution.

## Fresh root: PingPong B64, then only on PASS B128 Target/Serial/PingPong

All strict mode and failure exit logic is inside a **child Bash**. It cannot install
an ERR trap in or call exit on the operator's interactive parent shell. An error
retains its original exit code, prints errors and attempts a small failure bundle.
Every subsequent GPU point is stopped on failure, operator stop or insufficiency.

```bash
bash <<'BASH'
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
export SR_FIXED_ROOT="/root/autodl-tmp/SpecRhythm-data/results/decode-scan/resident360-ready-${SR_FIXED_COMMIT:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
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
bash scripts/run_decode_scan.sh capacity --single-point --batch 64 --mode pingpong
bash scripts/run_decode_scan.sh run --single-point --batch 64 --mode pingpong
"$SR_FIXED_PYTHON" - <<'PY'
import json, os, pathlib
from specrhythm.serving.fixed_artifacts import point_reports
p = pathlib.Path(os.environ['SR_FIXED_ROOT'])
rows = [r for r in point_reports(p) if not r['point'].get('probe')]
assert len(rows) == 1
r = rows[0]
assert r['point']['test_order'] == 'independent-single-point'
assert r['mode'] == 'pingpong' and r['batch'] == 64
assert r['git_commit'] == os.environ['SR_FIXED_COMMIT']
assert all(r[k] == 'PASS' for k in ('capacity_status','execution_status','measurement_status','cleanup_status'))
assert r['formal_comparison_eligible'] is True and r['stop_reason'] == 'time_budget'
assert r['diagnostic_logging']['observation'] == 'buffered-live'
assert r['identity_matching']['mode'] == 'bound-prefix'
e = r['scan_readiness']
assert any(t['value'].get('event') == 'full-batch-ready' and t['value'].get('ready_poll_quota') == 360
           for t in e['transitions'])
print(json.dumps({k:r[k] for k in ('artifact','measured_window_ms','committed_window_tokens',
                                 'decode_throughput_tok_s','scan_readiness')}, indent=2))
PY
# Each run independently checks its own B128 physical capacity and all360 prefill.
# Any failure stops before the next GPU point; no old B16 qualification is borrowed.
bash scripts/run_decode_scan.sh run --single-point --batch 128 --mode target
bash scripts/run_decode_scan.sh run --single-point --batch 128 --mode serial
bash scripts/run_decode_scan.sh run --single-point --batch 128 --mode pingpong
bash scripts/run_decode_scan.sh summary
bash scripts/run_decode_scan.sh bundle --output "${SR_FIXED_ROOT}-complete-small.tar.gz"
BASH
```

`prepare` reads inventory/environment (including nvidia-smi), but runs no inference.
`capacity` loads GPU engines, verifies actual limits, prefills all360, then drains
with zero verification. Each `run` also performs its own capacity/prefill checks,
then two full warmup rotations and the continuous30s decode window. There is no
sample12 cutoff. Setup900 and drain60 are separate bounded costs; an already issued
atomic step may overrun30s, retaining actual commits/time. Readiness waits, normal
commit/prefix work and refill are never subtracted. No full audit runs here.

`--single-point` requires one explicit mode/B, repeats1 and no `--remaining`.
It changes only test order and records that fact in point metadata. Failed roots
still block continuation. Default scan commands retain their original B16 gate.
The new summary contains four points at most; it does not merge the eight old
points into this commit. Old PingPong B16/B32 use the old collection/wait path;
remeasure them only when a uniform new-commit performance table is requested.
No Target/Serial B16/B32/B64, stages, S1/S2 gates or extra grid is run by default.

## Status, errors and owned stop (no new inference or full audit)

In another terminal, use the exact NEW ROOT printed above. Inspection does not
require a commit checkout guard and does not start inference. `stop` requests only
owned-process cleanup and can wait for the bounded current operation. Natural
terminal, diagnostic cancellation, insufficient measurement and failure remain
different states; no partial result becomes a valid performance point.

```bash
bash <<'BASH'
cd /root/autodl-tmp/src/SpecRhythm
export SR_FIXED_PYTHON=/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11
: "${SR_FIXED_ROOT:?Export the exact NEW ROOT printed by the foreground run}"
bash scripts/run_decode_scan.sh status
bash scripts/run_decode_scan.sh errors
# Only when requesting a controlled operator stop:
# bash scripts/run_decode_scan.sh stop --wait-seconds 65
# After cleanup (including failure), optional lightweight inspection/export:
bash scripts/run_decode_scan.sh summary
bash scripts/run_decode_scan.sh bundle --output "${SR_FIXED_ROOT}-inspect-$(date -u +%Y%m%dT%H%M%SZ)-$$-small.tar.gz"
BASH
```

## Optional retained-failure narrow export: CPU only, no rerun or full audit

The provided small bundle cannot prove the old failing cycle's exact ready
partition. This optional command reads at most the last8MiB of its scheduler log,
validates existing checksums, exports admitted-request metadata from at most eight
cycles ending within3s of rejection, and writes a **new** file capped at2MiB.
Offsets, original record hashes and tail hash identify the source region. It does
not copy token prefixes, KV tensors, the result root or full logs. A missing window
stays UNKNOWN; do not infer its missing state. The outer120s timeout stays CPU-only.

```bash
bash <<'BASH'
set -euo pipefail
export SR_FIXED_PYTHON=/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11
export SR_SCAN_OLD=/root/autodl-tmp/SpecRhythm-data/results/decode-scan/resident360-e39afc1b7d90-20260912T054301Z-1475/runs/decode-scan-pingpong-B64-A-20260912T140920-18308041766005622
export SR_SCAN_EXPORT="${SR_SCAN_OLD}-readiness-tail-$(date -u +%Y%m%dT%H%M%SZ)-$$.json"
timeout 120 "$SR_FIXED_PYTHON" - <<'PY'
import hashlib, json, os, pathlib
p = pathlib.Path(os.environ['SR_SCAN_OLD'])
r = json.loads((p/'light-summary.json').read_text())
end = r['rejected_step']['timestamp_ns']
cohorts = {a['request_id']: a['cohort'] for a in r['admission_order']}
f = p/'scheduler-events.jsonl'
size = f.stat().st_size
start = max(0, size - 8*1024**2)
with f.open('rb') as h:
    h.seek(start)
    if start:
        h.readline()
    offset = h.tell()
    data = h.read(8*1024**2)
rows = []
for line in data.splitlines(keepends=True):
    x = json.loads(line)
    checksum = x.pop('record_sha256')
    canonical = json.dumps(x, sort_keys=True, separators=(',', ':')).encode()
    assert hashlib.sha256(canonical).hexdigest() == checksum
    if end - 3_000_000_000 <= x.get('poll_end_ns', 0) <= end:
        fields = ('cycle_id','poll_start_ns','poll_end_ns','scheduled_request_ids',
                  'verify_request_ids','retired_ready_results')
        y = {k:x.get(k) for k in fields}
        y.update(source_offset=offset, source_record_sha256=checksum)
        wanted = ('request_id','internal_request_id','specrhythm_state','prefix_version',
                  'round_id','live_proposal_present','proposal_consumed','proposal_valid',
                  'proposal_ready_timestamp_ns','admissible','inadmissible_reason','scheduled')
        y['requests'] = [{**{k:a.get(k) for k in wanted}, 'cohort':cohorts[a['request_id']],
                          'candidate_count':len(a.get('spec_token_ids', []))}
                         for a in x['request_admissibility'] if a['request_id'] in cohorts]
        rows.append(y)
    offset += len(line)
out = {'source':str(f),'size_bytes':size,'tail_start_offset':size-len(data),
       'tail_sha256':hashlib.sha256(data).hexdigest(), 'rejected_step':r['rejected_step'],
       'status':'EXPORTED_NOT_YET_CLASSIFIED' if rows else 'UNKNOWN_MISSING_WINDOW',
       'cycles':rows[-8:]}
raw = json.dumps(out, indent=2).encode()
assert len(raw) <= 2*1024**2
with pathlib.Path(os.environ['SR_SCAN_EXPORT']).open('xb') as h:
    h.write(raw)
print(os.environ['SR_SCAN_EXPORT'], len(raw), 'bytes;', len(rows[-8:]), 'cycles; no GPU/audit')
PY
BASH
```
