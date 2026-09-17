# A100 B128 Target critical-path comparison

Use `baseline` and `lean-target` separately on the same execution SHA; never run
these concurrently on the same GPUs. The new pinned entry defaults to lean-target.
The original B16/B64/B128 entries remain available with their original defaults.
No `lean-target-dispatch` configuration is provided: current owner boundary and
cross-home dispatch already satisfy the tested interleaving, and this change does
not claim an unsupported second scheduling optimization.

| Experiment | Target diagnostics | Draft | I/O | Validation |
| --- | --- | --- | --- | --- |
| baseline | full numerical forensics | unified | deferred-window | performance-exploration |
| lean-target | required metadata/native evidence; numerical forensics NOT_COLLECTED_BY_PROFILE | unified | deferred-window | performance-exploration |

Each invocation creates a new worktree and local root. Sequence: static geometry /
capacity contracts → four GPU capacity probes → four modes forward and reverse
(eight separate 30s windows, at least 256 warmup request opportunities and coverage
of all 128 starting active IDs) → per-run geometry/measurement/cleanup/evidence
qualification → comparisons → one package. No Target-only or independent full-output
runs are added. Optional missing numerical evidence is explicit; required evidence
and logging/deadline/cleanup failures still stop the first failing point.

Geometry: Serial A128/Target128, PingPong A64+B64/Target64; all Draft ceiling128,
resident360, K3/no-bonus, same models/TP/numerics/selection/seed. Setup900s/drain60s
absolute deadlines, capacity/headroom, buffer/archive bounds are unchanged.

Mutable files: `/tmp/specrhythm-runs/<unique-tag>/` (container overlay, no NVMe claim).
Local sealed archive → verified temporary DPC copy → final publication. Default DPC:
`/root/autodl-tmp/SpecRhythm-data/results/rolling-eager/`. Failed delivery retains the
verified local archive. Exactly one outer `UPLOAD ONLY:`; keep all local evidence.
Each invocation returns one `pingpong-k3-delivery-a100-k3-b128-<experiment>-<tag>.tar.gz`.

The bundle retains experiment-plan (profile), manifest/options, runtime declared
and effective per-rank profile, raw capture coverage, trace and native evidence,
all eight results, raw failures, hashes and export/copy status. Comparison includes
each repetition and mean/range; no best-run selection or subtraction of diagnostic
time from measured throughput. Compare baseline→lean-target for the combined
optional-forensics/workload-indexing effect, not a separate scheduler effect.
Capture/logits/top-k/contract spans are nested, not additive. Remaining required
metadata copying, hashing/encoding and non-capture postprocessing are not assumed
zero. READY wait partitions are partial observations, not exact causal labels.

Execution SHA: `4ff1170397db0393caa2d8a43c2089a8e602ddfd`.
The executable `scripts/run_k3_target_pinned.sh` fixes this SHA and accepts exactly
`baseline` or `lean-target` (default). A second positional scale or an unsupported
`lean-target-dispatch` value is rejected. It resets inherited inner-run/profile
flags so an earlier session cannot bypass capacity or choose a different profile.
Fixed entry SHA: `7f5d03002d2e24788ec9bedb792d974d8b55ec3f`.
The entry reads the execution SHA from its committed script; it does not run branch HEAD.
New GPU results and performance conclusions remain PENDING.


Run these independently, sequentially on the same idle A100 GPUs. The first block
runs only baseline. The second runs only lean-target. Do not launch both together.
Each uses a new detached worktree, unique local root, four first-repeat probes and
eight windows. The runner rechecks actual loaded capacity for every fresh engine.
A failed point stops its remaining points; preserve the returned single archive.

```bash
if bash -s -- baseline <<'SR_K3_PROFILE_BOOTSTRAP'
set -Eeuo pipefail
PROFILE="$1"
REPO="${SR_K3_REPO:-/root/autodl-tmp/src/SpecRhythm}"
ENTRY_SHA=7f5d03002d2e24788ec9bedb792d974d8b55ec3f
git -C "$REPO" fetch origin codex/rolling-eager-v0.1
ENTRY=$(mktemp /tmp/specrhythm-k3-target-entry.XXXXXX)
git -C "$REPO" show "$ENTRY_SHA:scripts/run_k3_target_pinned.sh" > "$ENTRY"
bash "$ENTRY" "$PROFILE"
SR_K3_PROFILE_BOOTSTRAP
then
  printf 'Configuration complete; upload only its returned archive.\n'
else
  rc=$?
  printf 'Stopped with original rc=%s; terminal stays open.\n' "$rc"
fi
```

```bash
if bash -s -- lean-target <<'SR_K3_PROFILE_BOOTSTRAP'
set -Eeuo pipefail
PROFILE="$1"
REPO="${SR_K3_REPO:-/root/autodl-tmp/src/SpecRhythm}"
ENTRY_SHA=7f5d03002d2e24788ec9bedb792d974d8b55ec3f
git -C "$REPO" fetch origin codex/rolling-eager-v0.1
ENTRY=$(mktemp /tmp/specrhythm-k3-target-entry.XXXXXX)
git -C "$REPO" show "$ENTRY_SHA:scripts/run_k3_target_pinned.sh" > "$ENTRY"
bash "$ENTRY" "$PROFILE"
SR_K3_PROFILE_BOOTSTRAP
then
  printf 'Configuration complete; upload only its returned archive.\n'
else
  rc=$?
  printf 'Stopped with original rc=%s; terminal stays open.\n' "$rc"
fi
```
