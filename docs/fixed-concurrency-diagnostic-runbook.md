# Fixed64/32 foreground diagnostic runbook

Current opt-in persistence/timing addendum: [design](fixed-buffered-runtime-design.md) and
[two-mode buffered-live runbook](fixed-buffered-runtime-runbook.md). Existing instructions
below describe the original-live baseline and remain available; no old artifacts are changed.

For the b424 timing-attribution follow-up, start with the separate
[CPU-only analysis/export runbook](fixed-attribution-runbook.md). That revision changes
no runtime behavior and requires no new GPU run. Its optional retest is Serial then
PingPong only; do not automatically repeat all four modes or initial-state stages below.

GPU execution is performed by the operator. This entry does not run S2 G0–G3,
capacity search, calibration, patch application, or dependency installation. Existing
S1/S2 result roots are retained. Use the final full SHA delivered with the change.
The default observer is `original-live` in all four modes. Timing/performance remains
PENDING until real results return.

## Fresh checkout and result root

Use the exact full SHA delivered with this fix. The repository form below resolves the
serving branch once and freezes its full SHA; the delivery message supplies a literal
SHA for reproducing this reviewed revision. Preserve the old `fixed64-89a962127f2e-...`
root and its failed Serial attempt.

```bash
set -euo pipefail
cd /root/autodl-tmp/src/SpecRhythm
git fetch origin codex/vllm-serving-v0.1
export SR_FIXED_COMMIT="$(git rev-parse origin/codex/vllm-serving-v0.1)"
git checkout --detach "$SR_FIXED_COMMIT"
export SR_FIXED_PYTHON=/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11
export SR_FIXED_S1=/root/autodl-tmp/SpecRhythm-data/results/phase-s1/s1p-5a00049-20260909T144802Z-1469
export SR_FIXED_ROOT="/root/autodl-tmp/SpecRhythm-data/results/fixed-concurrency/fixed64-stop-${SR_FIXED_COMMIT:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
export PYTHONUNBUFFERED=1
printf 'Keep this result root for status/stop/summary: %s\n' "$SR_FIXED_ROOT"
bash scripts/run_fixed_diagnostic.sh prepare --s1 "$SR_FIXED_S1" \
  --warmup-steps 2 --samples 12 --repeats 1 --window-seconds 30 \
  --setup-timeout 900 --drain-timeout 60
bash scripts/run_fixed_diagnostic.sh capacity
```

`prepare` performs environment/physical-device inventory, freezes the qualified mixed100
rows/order and loads no model weights; it does not run inference. `capacity` starts the
real three-worker GPU engines, checks resident100/active64 capacity, then shuts down
with zero verification. No capacity reduction or parameter search is allowed. These
commands perform neither full CPU audit nor S2 G0–G3/calibration.

## Serial first, then each remaining mode once

```bash
bash scripts/run_fixed_diagnostic.sh short --mode serial
```

Wait for this command to return rc=0 with execution PASS and measurement PASS before
continuing. Do not use bare `short` here: it would repeat the already successful Serial
point. Each mode below is a separate foreground command with a fresh engine in the
same new root. Inspect each result before launching the next; a manual stop means stop
here, with no automatically launched later mode.

```bash
bash scripts/run_fixed_diagnostic.sh short --mode target
```

```bash
bash scripts/run_fixed_diagnostic.sh short --mode serial-split
```

```bash
bash scripts/run_fixed_diagnostic.sh short --mode pingpong
```

All four `short` commands use GPU; no full CPU audit runs. Raw Target/Draft stdout and
stderr remain visible with mode/source tags and retained in their original logs.
`samples=12` means up to 12 nonempty Target steps for target/serial and 24 steps for
grouped modes, with warmup 2/4 respectively. Time can stop the window sooner. The last
issued step's actual time is retained; drain does not extend measurement or generate
new proposals. Fixed bounds remain global64 and cohort32.

Each command prints execution/measurement status, real exit, primary error and path.
A normal sample/time budget with valid measured evidence yields execution PASS and
measurement PASS after successful cancellation/physical release/shutdown. Unfinished
requests remain `DIAGNOSTIC_CANCELLED`, not naturally completed. A clean operator stop
returns rc=0 but measurement INSUFFICIENT and is excluded from comparisons. A zero-sample
or missing shape point is also INSUFFICIENT. Failed drain/RPC/release/cleanup returns
nonzero; deadline expiry uses effective rc=124. Owned cleanup completion alone does not
mean execution or cleanup validity passed. Read `measurement-snapshot.json` for partial
measurements; it is never final PASS. Missing final runtime/backend reports are secondary
to the original failure. Full-request SLO/goodput is not measured by this diagnostic.

## Light summary and small failure-capable bundle

```bash
bash scripts/run_fixed_diagnostic.sh summary
bash scripts/run_fixed_diagnostic.sh bundle --output "${SR_FIXED_ROOT}-small.tar.gz"
```

These commands do not use GPU or run full CPU audit. They also work after a failed
point with missing final reports. The small bundle includes compact measurement/drain,
exit/ownership and first/secondary failure artifacts; failed or incomplete data appears
separately and contributes no numeric comparison. Preserve the full root and return
the small bundle first.

## Optional initial-state stages (separate GPU experiment)

Only after the short retest, if stage measurements are needed:

```bash
bash scripts/run_fixed_diagnostic.sh stages --stage-samples 1 --stage-warmups 0
```

This starts seven fresh GPU points: AR32 A/B, AR64, Serial SD32 A/B, SD64, and A SD32
with B Draft32. It does not run CPU audit. Effective K must actually be 4; shortened
candidates or missing B32/B64 shape evidence remain INSUFFICIENT. Split-cost formulas
still require matching actual 64 context and the union of the two 32 halves. No extra
warmup trials, continuation replay or parameter search are enabled.

## Status, failure and bounded stop (second terminal)

Set `SR_FIXED_ROOT` to the exact root printed above and `SR_FIXED_PYTHON` to the same
interpreter. The script defaults to the documented server Python if it is omitted.
These commands never start inference.

```bash
bash scripts/run_fixed_diagnostic.sh status
bash scripts/run_fixed_diagnostic.sh errors
bash scripts/run_fixed_diagnostic.sh stop --wait-seconds 60
```

`errors` prints primary/secondary diagnostics and at most 40 lines from each raw log.
`stop` requests a safe boundary stop and cancels remaining points in the foreground
suite. If the point does not stop within 60 seconds, only its recorded owned process
tree is cleaned. Forced cleanup is an interrupted execution, not a performance PASS.
No foreign process or server-wide `pkill` is used.

## Optional full CPU audit

The following is a separate potentially expensive CPU scan. It is never invoked by
`short`, `stages`, `summary` or `bundle`. It can run on a CPU machine with the repository
and full retained root; set `SR_FIXED_PYTHON` to that machine's Python, not the GPU venv.
It does not need model weights or an AutoDL connection. The full root includes relative
`inputs/requests.jsonl` and its immutable execution manifest.

```bash
bash scripts/run_fixed_diagnostic.sh audit
bash scripts/run_fixed_diagnostic.sh status
```

Audit creates `offline-audit.json` per point, preserving the original light result.
On a different machine, use `--directory "$SR_FIXED_ROOT/runs/<point-name>"` to select
one point. Do not use S1/S2 full-completion qualifiers for a diagnostic cancellation.
Return the small bundle first. No performance benefit or S2 GPU PASS is assumed in
advance; the intended interpretation is **production vLLM Batched Draft end-to-end
improvement**, with actual work and own per-mode g beside each comparison.
