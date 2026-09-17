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

Fixed execution/entry SHAs and copyable commands are added at final delivery.
New GPU results and performance conclusions remain PENDING.
