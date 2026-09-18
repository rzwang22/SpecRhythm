# Ordinary K3 double-batch control path

Execution commit: `e3726521df751514877b21f819fe17d508fd3741`. The fixed launcher in `scripts/run_dual_batch_pinned.sh`
checks and runs exactly this revision.

The opt-in `target_dispatch=dual-batch` supports **ordinary `pingpong-k3` only**.
The existing `reference` and `encode-once` paths, Serial gates and eager modes remain
available. No new default, Target executor, GPU kernel or admission policy is enabled.

## Fixed experiment

`run_dual_batch_pinned.sh` (delivered in the entry commit) pins the implementation
commit and creates a detached worktree. `run_dual_batch.sh <full-execution-SHA>`
is the implementation entry. It runs, sequentially:

1. Static `k3-b128-v1` capacity contract; real resident360 capacity for reference/new.
2. Bounded ordinary reference/new semantic smoke: 16 identical frozen requests,
   at most 8 output tokens each, natural EOS/budget completion, real proposal/KV
   cleanup. Compare complete tokens and termination between the two paths. This
   is a separate recorded fixture, not Target-only equivalence or a performance point.
3. Four independent 30-second performance windows: **reference, dual-batch,
   dual-batch, reference**. All are ordinary PingPong A64/B64, Target ceiling64,
   Draft ceiling128, resident360, true K3, unified/deferred-window/lean,
   performance-exploration, 256 actual warmup opportunities with active coverage,
   setup900/drain60. No automatic retries or other modes/grid.
4. Existing execution, measurement, cleanup, native TP geometry/identity, causal
   trace, file attribution and evidence qualification. Stop at the first failure.
5. One bounded local archive, verified copy to DPC, local fallback on copy failure.

Source S1 defaults to
`/root/autodl-tmp/SpecRhythm-data/results/phase-s1/s1p-5a00049-20260909T144802Z-1469`.
The performance workload SHA256 must remain
`cdaf71adace15d229f5087b98f9fd162a958456226a660184fe03f5d6ebd8ff4`.
The Python default is the existing `specrhythm-phase4-vllm-0.25.1` Python3.11
runtime. Model paths, precision, sampling and device binding are inherited from
qualified S1/config and individually checked, not overridden by this entry.
Fresh topology/environment hashes contain collection timestamps; they are
validated per run, while actual execution/selection/options are compared across
runs (only `target_dispatch` may differ).

Mutable files stay under `/tmp/specrhythm-runs/<unique-tag>/`; this is container
local storage, not a claim about NVMe. Final persistent directory remains
`/root/autodl-tmp/SpecRhythm-data/results/rolling-eager/`. Set `SR_K3_LOCAL_BASE`,
`SR_PING_RESULTS`, `SR_FIXED_PYTHON`, `SR_FIXED_S1` explicitly only if necessary.
Both local evidence and failed partial files are retained. The outer local runner
prints exactly one `UPLOAD ONLY:` path when a valid archive exists. Report/copy
errors cannot replace the original execution exit code.

## Results and rollback

The package contains `dual-batch-plan.json`, `smoke-comparison.json`, two capacity
roots, two smoke roots and four `windows/<index>-<dispatch>` roots. Every window
has raw native/transport/owner evidence, existing matched-cycle and audit reports,
measurement qualification and comparison; top-level `comparison.json` retains all
four results and existing TPOT/commit-interval/end-to-end fields. Missing evidence
is not filled with zero. Required absent run files cause export integrity failure.
Large details use the existing bounded companion format, not repeated inline arrays.

Compare complete-window throughput, actual committed tokens/opportunities, Target
cadence, matched claim-to-GPU and feedback costs, and calibrated native cross-home
Draft/Target intersections. TP2 is one Target; do not add the two rank times.
CPU safety interleaving is not GPU overlap. A shorter helper or host call does not
prove faster end-to-end progression. All four windows matter, not only the best.

Use `--target-dispatch reference` in the existing prepare CLI to roll back; the
fixed comparison already exercises that path on the same commit. `dual-batch`
refuses Serial/eager. Keep Target-only full output equivalence **NOT_RUN** in
performance reports and preserve the known historical output differences. The
small reference/new output comparison does not resolve those differences.

No GPU result is asserted at delivery. GPU capacity/correctness/cleanup, overlap
and performance remain **PENDING** until the operator returns this one archive.

## Foreground invocation

From the delivered fixed-entry revision of this repository, run:

```bash
if bash scripts/run_dual_batch_pinned.sh; then
  printf 'Return the single UPLOAD ONLY archive.\n'
else
  rc=$?
  printf 'Stopped (rc=%s); later windows did not run. Terminal stays open.\n' "$rc"
fi
```

The final delivery message provides the immutable entry SHA and the complete
`git show <entry-SHA>:scripts/run_dual_batch_pinned.sh` extraction command, so an
older local worktree's script cannot accidentally be used. The pinned script
fetches, verifies the execution commit above and creates the new worktree itself.
