# Ordinary CPU P0/P1/S1, A100 B128

The immutable `scripts/run_ordinary_cpu_pinned.sh` is the authoritative execution
pin. It creates a fresh detached worktree and fresh local run. Delivery records its
full execution SHA in `dual-batch-plan.json`. No historical root is reused.

## Fixed cases and rollback

|case|CLI mode|target-dispatch|target-cpu|
|---|---|---|---|
|P0|pingpong-k3|dual-batch|reference|
|P1|pingpong-k3|shared-command|block-sets|
|S1|serial-k3|shared-command|block-sets|

To turn off only the extra audit optimization, use `--target-cpu reference` with
`--target-dispatch shared-command`. The original `dual-batch` option and its
PingPong-only guard remain compatible. `target-dispatch reference` retains the
old control path, but is not another performance point in this experiment.
No eager mode is supported by the shared command selection.

One foreground entry: static frozen interface/geometry checks → P0/P1/S1 resident360
capacities → four bounded semantic smokes (P0/P1 and Serial S0/S1) →
**P0 → P1 → S1 → S1 → P1 → P0**, 30 seconds each → comparison and one archive.
S0 uses Serial's reference controls only for max8-token/16-request semantics; it is
not a fourth performance case. Smoke compares same-mode full token sequences and
termination under the existing greedy contract. Any new difference stops the run
with first difference/request/source evidence; seed alone is never sufficient.
Full Target-only output equivalence stays NOT_RUN, historical differences unresolved.

All performance cases use lean Target diagnostics, unified Draft dispatch,
deferred-window I/O, runtime Draft audit, K3/no bonus, the same frozen workload and
seed. Serial is A128/Target128, PingPong A64/B64/Target64; Draft128, GPU0 versus one
TP2 Target on GPU1/2. Preserve 256 **actual request verification opportunities**
and active128 coverage for warmup, setup900 and drain60 absolute deadlines.
No GPU grid, retries, added workers, altered READY order or changed sampling.

## Files and interpretation

Mutable roots: `/tmp/specrhythm-runs/a100-ordinary-cpu-<tag>/` (overlay is not evidence
of NVMe). Final destination:
`/root/autodl-tmp/SpecRhythm-data/results/rolling-eager/`.
The existing local archive/size/SHA256 validation, verified DPC copy and local
fallback remain. Local evidence is retained. The public wrapper prints exactly
one `UPLOAD ONLY:`; return only that `pingpong-k3-delivery-<tag>.tar.gz`.

The package includes plan/effective configuration, capacity, smoke first differences,
six original windows, qualification, native timelines, comparison and first/secondary
errors. The aggregate uses summed tokens / summed actual window seconds, alongside
all individual results/ranges. `request_measurement` reports window natural completions,
completion throughput and original full-request TPOT formula; no eligible samples
means null with reason. `recovery_prepare_class` identifies whole physical recovery
calls completed during another home's claim→GPU preparation, leaving others unclassified.
It does not turn GPU envelope into utilization or throughput into SLO goodput.

Window key `case` distinguishes P0/P1/S1. Compare P1/P0 for the extra CPU change,
S1/P1 for same active resources but different batching geometry. Do not attribute
all of S1/P1 to overlap. No performance threshold is predeclared. GPU execution,
cleanup, calibrated overlap and performance are PENDING until this entry is run.

## Foreground command

The delivery commit fills the exact execution and entry SHA below. Always read the
pinned entry from that commit; do not execute an old runbook checkout by accident.
The outer `if` keeps the interactive shell open; the script preserves first failure
and runs the existing bounded failure export. Required paths may be overridden
explicitly via SR_K3_REPO, SR_FIXED_PYTHON, SR_FIXED_S1, SR_K3_LOCAL_BASE and SR_PING_RESULTS.
