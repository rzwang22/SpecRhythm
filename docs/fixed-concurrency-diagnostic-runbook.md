# Fixed64/32 foreground diagnostic runbook

GPU execution is performed by the operator. This entry does not run S2 G0–G3,
capacity search, calibration, patch application, or dependency installation. Existing
S1/S2 result roots are retained. Use the final full SHA delivered with the change.
The default observer is `original-live` in all four modes. Timing/performance remains
PENDING until real results return.

## Fresh checkout and result root

```bash
cd /root/autodl-tmp/src/SpecRhythm
export SR_FIXED_COMMIT='<FINAL_FULL_40_CHARACTER_SHA>'
git fetch origin
git checkout --detach "$SR_FIXED_COMMIT"
export SR_FIXED_PYTHON=/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11
export SR_FIXED_S1=/root/autodl-tmp/SpecRhythm-data/results/phase-s1/s1p-5a00049-20260909T144802Z-1469
export SR_FIXED_ROOT="/root/autodl-tmp/SpecRhythm-data/results/fixed-concurrency/fixed64-${SR_FIXED_COMMIT:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
printf 'Keep this result root for status/stop/summary: %s\n' "$SR_FIXED_ROOT"
bash scripts/run_fixed_diagnostic.sh prepare --s1 "$SR_FIXED_S1" \
  --warmup-steps 2 --samples 12 --repeats 1 --window-seconds 30 \
  --setup-timeout 900 --drain-timeout 60
```

`prepare` checks the current pinned model/patch/environment binding and freezes the
existing mixed100 rows/order. It loads no model weights. `capacity` below loads the
three real workers, records physical capacities and checks resident100/active64 only.
The configured 64/32 values are never reduced on failure. Do not proceed on a nonzero rc.

```bash
bash scripts/run_fixed_diagnostic.sh capacity
```

## Initial-state stage samples

Each point uses fresh process/engine/prefill. The seven points are AR32 A, AR32 B,
AR64, Serial SD32 A, SD32 B, SD64, and A SD32 with B Draft32. Initial Serial proposals
measure isolated D32/D64. Effective K must actually be 4 for a K4 shape sample.
Startup profiling/prefill warms the model; `--stage-warmups` controls additional
explicitly discarded fresh-process trials, not same-engine continuation replay.
One sample per point is deliberately short; use `--stage-samples 3` for more precision
only after the first point set executes cleanly.

```bash
bash scripts/run_fixed_diagnostic.sh stages --stage-samples 1 --stage-warmups 0
bash scripts/run_fixed_diagnostic.sh summary
```

A shape marked INSUFFICIENT does not falsify execution: inspect its actual IDs,
candidate lengths and contexts. No missing timing is synthesized. Split-cost formulas
require the actual 64 set/context to equal the union of both 32 halves.

## Four real continuous paths

The default is foreground, one fresh point for each mode. Raw Target/Draft stdout and
stderr are retained and displayed with mode/source tags. `samples=12` means at most
12 rotation units: 12 nonempty Target steps for target/serial and 24 for grouped modes.
`warmup-steps=2` similarly means 2/4 Target steps. The 30-second measured window may
stop sooner. A point can exceed its budget by the currently issued real step; it then
stops new work and drains/cancels under the separate bounded cleanup contract.

```bash
bash scripts/run_fixed_diagnostic.sh short
bash scripts/run_fixed_diagnostic.sh summary
```

To inspect one mode without rerunning others, explicitly select it. This creates a new
attempt directory; no old attempt is overwritten or silently reused.

```bash
bash scripts/run_fixed_diagnostic.sh short --mode serial-split
```

Each point prints execution/measurement status, actual batches, samples, stage timings,
committed progress, throughput, real exit code, primary failure and artifact path.
`light-summary.json` and `.csv` are immediately available. Full offline audit stays
PENDING, not PASS. Natural completions, cancellations and failures are distinct;
window throughput is not full-request goodput/SLO attainment.

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

## Small bundle and optional CPU audit

```bash
bash scripts/run_fixed_diagnostic.sh summary
bash scripts/run_fixed_diagnostic.sh bundle --output "${SR_FIXED_ROOT}-small.tar.gz"
```

The small bundle contains config, compact JSON/CSV and necessary failures, not large
raw token/Target/request-state logs. Retain the full root locally for later audit.

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
