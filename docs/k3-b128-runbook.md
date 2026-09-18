# A100 uniform K3 B128 performance exploration

New explicit geometry `k3-b128-v1`: Serial A128/Target128, PingPong A64/B64/Target64,
Draft physical ceiling128. Resident360, unchanged frozen workload SHA256
`cdaf71adace15d229f5087b98f9fd162a958456226a660184fe03f5d6ebd8ff4`, seed1666,
GPU0 Draft and GPU1/2 one TP2 Target. This is not a change to K3, precision,
Target diagnostics, scheduler, READY semantics or KV protocol.

Execution commit: `f7bfb43152f5655edbad9888c53c0a81e9d11954`.

The fixed entry `scripts/run_k3_b128_pinned.sh` pins that implementation, creates a
new detached worktree, clears inherited private runner flags, and defaults to
`unified`, `deferred-window`, `performance-exploration`. It also accepts `io-only`
(legacy dispatch/same I/O), and an explicit same-version `b64` rerun. It never
runs both scales automatically. Historical B64 comparisons are cross-run exploration,
not an isolated batch-size effect.

The B128 flow is:

1. Static B128 interface/reservation check, GPU capacity PENDING.
2. Four loaded GPU capacity probes; first insufficiency stops, no smaller fallback.
3. Four modes in forward order, then reverse order, independent fresh roots/engines.
   The second repeat reuses the completed initial probes; every fresh performance
   engine still recomputes actual loaded capacity before resident preparation/decode.
4. At least256 actual request verification opportunities and all128 active IDs
   covered before a continuous30s window. Setup900s/drain60s are unchanged.
5. Native per-rank B128/B64 proof, bounded active/home/Draft counts, runtime K3/KV/
   accounting, execution, measurement, cleanup and diagnostic evidence qualification.
6. Original per-repeat results, ratios and min/max ranges; one deduplicated package.

No Target-only or separate full-output run occurs. Output equivalence is NOT_RUN;
the historical B64 Serial10/64 request mismatch is unresolved. Capacity, evidence,
cleanup and geometry failures still stop subsequent points. A successful measurement
does not certify full output equivalence, hidden recovery or stable speedup.

Mutable files remain under `/tmp/specrhythm-runs/<unique-tag>/`; `/tmp` is container
overlay, not evidence of NVMe. Default persistent directory remains
`/root/autodl-tmp/SpecRhythm-data/results/rolling-eager/` (DPC on the current server).
B128 requires at least32GiB free at startup; buffers/archive limits are unchanged
and fail closed. Space is not reserved. Keep local evidence after success/failure.
A local sealed archive is copied to a unique DPC temporary target, re-read for size
and SHA256, then published. Copy failure retains the locally verified package.
Only the outer supervisor prints one `UPLOAD ONLY:` path. Original execution error,
export/copy errors and statuses remain separate; no package means no invented path.

Expected archive: `pingpong-k3-delivery-a100-k3-b128-unified-<tag>.tar.gz`.
`inventory.json` maps logical paths to digest-addressed objects. It contains the
experiment/validation plans, two comparisons, all eight performance runs, four
initial probes, effective device/diagnostic configuration, initial request IDs,
raw TP/Draft timelines, logging/publication receipts, lifecycle/first failure and
export state. Do not send separate subpackages. Each repeat reports raw ΣB,
throughput,128-opportunity normalization, capture/dispatch/pipeline breakdown;
no best-window selection. GPU capacity/correctness/cleanup/overlap/performance
remain PENDING until operator evidence; full output equivalence stays NOT_RUN.

## Fixed foreground command

Entry commit: `23e748c1a1c093984a9b0abdab92d643f71b2624`.
It pins execution `f7bfb43152f5655edbad9888c53c0a81e9d11954` (not branch HEAD).
Run on the existing A100 host, with its existing models, environment and S1 source:

```bash
if bash <<'SR_K3_B128_BOOT'
set -Eeuo pipefail
REPO="${SR_K3_REPO:-/root/autodl-tmp/src/SpecRhythm}"
ENTRY_SHA=23e748c1a1c093984a9b0abdab92d643f71b2624
git -C "$REPO" fetch origin codex/rolling-eager-v0.1
ENTRY="/tmp/specrhythm-k3-b128-entry-$(date -u +%Y%m%dT%H%M%SZ)-$$.sh"
git -C "$REPO" show "$ENTRY_SHA:scripts/run_k3_b128_pinned.sh" > "$ENTRY"
bash "$ENTRY" unified b128
SR_K3_B128_BOOT
then
  printf 'Finished; return only the archive printed by the runner.\n'
else
  rc=$?
  printf 'Stopped with original rc=%s; interactive shell remains open.\n' "$rc"
fi
```

Only the runner prints `UPLOAD ONLY: <verified archive absolute path>`.
For a separately requested same-version B64 repeat, replace the last child command
with `bash "$ENTRY" unified b64`. For the retained I/O-only/legacy dispatch control,
use `bash "$ENTRY" io-only b128`. Neither is run by default. Each invocation creates
its own worktree, local directory and archive; no old results are overwritten.

## Same-SHA lean dispatch controls (2026-09-18)

The independent entry `scripts/run_k3_dispatch_pinned.sh` accepts exactly one of
`lean-reference` / `lean-dispatch-opt`. Each runs four first-repeat capacity probes
and eight performance windows (forward modes, then reverse modes), sequentially.
Both use lean Target diagnostics, unified Draft dispatch, deferred-window I/O,
performance-exploration and unchanged k3-b128-v1. Output equivalence is NOT_RUN.
Reference retains control streaming encoding; opt uses one-shot encoding before
the same atomic publication. No full numerical baseline or output comparison runs.

All original /tmp mutable storage, setup900/drain60 absolute budgets, local archive,
DPC copy/hash verification and local fallback are retained. Each invocation emits
only one UPLOAD ONLY path. Failure stops later points; no retry/grid is added.
Comparison includes compact all-sample cycle statistics and declared median/P90
examples. Raw native/owner/transport records remain in the bundle for reconstructing
all request ledgers; large derived arrays are not embedded repeatedly.
Missing landmarks, clipped/incomplete cycles and unknown eligibility stay explicit.
The first two windows per mode cannot establish a stable speedup by themselves.


The dispatch entry pins execution `c1fca49f3f846ef94f0759b585fddaf091d78b62`.
Its code includes the new observations and both policies; the only execution
switch between the two controls is control encoding. The script itself belongs
to the follow-up fixed-entry commit; fetch that commit before using it.
Choose one configuration for each foreground invocation. Both invocations create
new worktrees/local directories and independently produce one archive. Do not
run them concurrently on the same GPUs. Original buffers, report caps and
setup/drain deadlines are unchanged.


### Fixed lean dispatch foreground commands

Entry commit: `9feb51a4d36e4acc0f682ea006a6842d327a69d4`.
Run each configuration separately on the existing A100 server. This wrapper
contains strict mode in a child Bash and leaves the interactive parent open.
The first command runs the reference; use `lean-dispatch-opt` in the indicated
argument for the independent optimization run. Do not execute both concurrently.

```bash
if bash -s -- lean-reference <<'SR_K3_DISPATCH_BOOT'
set -Eeuo pipefail
REPO="${SR_K3_REPO:-/root/autodl-tmp/src/SpecRhythm}"
ENTRY_SHA=9feb51a4d36e4acc0f682ea006a6842d327a69d4
git -C "$REPO" fetch origin codex/rolling-eager-v0.1
ENTRY="$(mktemp /tmp/specrhythm-k3-dispatch-entry.XXXXXX.sh)"
git -C "$REPO" show "$ENTRY_SHA:scripts/run_k3_dispatch_pinned.sh" > "$ENTRY"
bash "$ENTRY" "$1"
SR_K3_DISPATCH_BOOT
then
  printf 'Finished; return only the archive printed by the runner.\n'
else
  rc=$?
  printf 'Stopped with original rc=%s; interactive shell remains open.\n' "$rc"
fi
```

The complete optimization command is identical except for the independently
selected argument:

```bash
if bash -s -- lean-dispatch-opt <<'SR_K3_DISPATCH_BOOT'
set -Eeuo pipefail
REPO="${SR_K3_REPO:-/root/autodl-tmp/src/SpecRhythm}"
ENTRY_SHA=9feb51a4d36e4acc0f682ea006a6842d327a69d4
git -C "$REPO" fetch origin codex/rolling-eager-v0.1
ENTRY="$(mktemp /tmp/specrhythm-k3-dispatch-entry.XXXXXX.sh)"
git -C "$REPO" show "$ENTRY_SHA:scripts/run_k3_dispatch_pinned.sh" > "$ENTRY"
bash "$ENTRY" "$1"
SR_K3_DISPATCH_BOOT
then
  printf 'Finished; return only the archive printed by the runner.\n'
else
  rc=$?
  printf 'Stopped with original rc=%s; interactive shell remains open.\n' "$rc"
fi
```

Each invocation prints one `UPLOAD ONLY:` for its own verified single archive,
`pingpong-k3-delivery-a100-k3-b128-<configuration>-<tag>.tar.gz`. Capacity and eight
performance windows, both comparisons, raw evidence, cycle projections, first
failure and export/copy outcomes are inside that archive. No extra collection
commands or separate JSON uploads are required.

## Report publication repair (2026-09-18)

The66-step lean-reference ordinary PingPong window exceeded the compact summary's
8MiB limit after successful execution/measurement/cleanup. Old failure remains.
New K3 summaries reference ordered, hashed `*-audit-report-details.NNN.jsonl`
shards; they are automatically inside the same single upload archive. Do not
upload them separately. Missing/corrupt shards still fail comparison/export.
Terminal results now print bounded status/throughput/path summaries; complete
result JSON remains in the run directory and bundle. There is no algorithm or
logging-buffer change. Both lean-reference and lean-dispatch-opt use the repair.
