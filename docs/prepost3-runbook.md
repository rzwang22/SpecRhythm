# Pre/post3 joint correctness and fixed B16 pair

Only an operator runs this on AutoDL. CPU validation does not qualify GPU execution,
output equivalence, overlap or performance. The final delivery pins the execution
SHA in `scripts/run_prepost_b16_pinned.sh`; the same literal is used for both modes.
The launcher creates a fresh detached worktree and never resets the saved checkout.
The generic engine entry is `scripts/run_prepost_b16.sh <full-SHA>`.

## Fixed sequence

1. Prepare a new `serial-prepost3/runtime` root and check resident360 capacity.
2. In a separate nested correctness root, run Target-only, Serial-prepost3 and
   Serial-eager-prepost3 through the real production launchers, Draft GPU0 and
   Target TP2 GPU1/2. Use the first 16 frozen requests with a separately recorded
   maximum of 32 output tokens. All 16 must terminate naturally, release KV, and
   have identical complete output lists to the reference. Both Target ranks must
   record exactly one forward per step, with real request identities/positions.
   Eager must exercise an actual P1/P4 mixed Target step; lack of coverage fails
   the gate, rather than being accepted as correctness evidence.
3. Run the Serial-prepost3/runtime B16 performance point; require execution,
   measurement, cleanup and strict diagnostic integrity before advancing.
4. Prepare a second independent Serial-eager-prepost3/runtime root, check capacity,
   run its performance point and qualify/export it. Generate the compact comparison.

The correctness fixture is explicitly **not** a performance workload and is never
included in the decode window. Performance retains all 360 real-prefilled frozen
requests, B16, seed1666, K4 maximum/3 lookahead/1 common, two warmup steps, continuous
30-second decode with no sample12 cutoff, setup900s/drain60s, `runtime` audit,
`buffered-live` observation and `bound-prefix` identity. Workload digest must be
`cdaf71adace15d229f5087b98f9fd162a958456226a660184fe03f5d6ebd8ff4`.
Models, deterministic sampling, GPU allocation and effective engine settings are
checked by the existing launch/identity/capacity gates. No cross-run UUID equality
requirement is introduced.

## Inputs and output paths

Defaults are the previously verified server locations:

- checkout: `/root/autodl-tmp/src/SpecRhythm` (`SR_EXEC_REPO` in the generic runner);
- Python: `/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11`
  (`SR_FIXED_PYTHON`);
- S1 input: `/root/autodl-tmp/SpecRhythm-data/results/phase-s1/s1p-5a00049-20260909T144802Z-1469`
  (`SR_FIXED_S1`);
- outputs: `/root/autodl-tmp/SpecRhythm-data/results/rolling-eager`.

Each point root is
`serial-prepost3-B16-<sha12>-<mode>-runtime-<UTC>-<pid>`.
Any existing path is refused. Return the two `*-complete-evidence.tar.gz` archives,
the first point's `*-complete-evidence-joint.tar.gz`, the paired
`prepost3-runtime-pair-<sha12>-<tag>.json`, and the two evidence-status JSON files.
The small bundles are retained alongside these for quick review.

Main archives contain bounded source inventories and raw measurement producers
once. The joint archive contains its separate runtime/backend/results/lifecycle
files once, at most64 files/256MiB, plus `joint-inventory.json` with SHA256, explicit
missing paths and limits. Its `result.json` stores per-request complete reference
comparisons; a failed gate has `failure.json`. No historic result is rewritten.

## Failure and stop behavior

Use the pinned launcher's **outer if / independent child Bash** command below.
Never source the strict runner in an interactive shell. First failure stops every
later point. The original code/stage/root is printed before auxiliary work;
`failure-summary.json`, small export, full export and their separate logs/statuses
are attempted. A joint child execution error retains its original return code;
secondary recording/export errors do not replace it. `execution`, `measurement`,
`cleanup`, `diagnostic_integrity` and `performance_conclusion` stay separate.
The interactive parent remains open on failure.

If interrupted, use the existing owned-process stop/drain procedure for the exact
printed new root. Do not restart into that root or continue the remaining pair
without resolving the first failure. There is no automatic retry, load grid,
PingPong or parameter scan. One completed pair establishes mechanism evidence only;
repeat later with a new tag and authorized interleaving before claiming a stable
few-percent performance change.

## Reading the pair

- Original measured throughput, committed tokens/step and complete-step mean/P50/P90
  remain unadjusted; compare only the same protocol's two new modes.
- `prepost.cycles` records actual P1/P4/terminal-clipped lengths, valid candidate and
  Target query counts, acceptance/reuse/discard rates, physical purpose/B/positions,
  common/recovery calls and feedback/lookahead/common/candidate-ready timestamps.
- `cycles` and native device sections retain TP union and forward overlap
  lower/upper bounds and lookahead steps completed at Target end. Host overlap
  cannot establish GPU overlap.
- `execution_path` records feedback delivery and owner processing, response to next
  Target, and file-specific fsync cost. Exclusive process/thread lanes keep nested
  spans and RPC waits separate; unaccounted time is retained.
- Check exactly one common physical step when required, at most three lookahead
  forwards, no eager recovery extensions, and actual mixed Target B16. EOS, output
  clipping, initial/refill and drain work are separately visible, not relabeled away.
- Successful lookahead completion-to-common time includes parent-result dependency;
  candidate-ready-to-step-end includes remaining batch/transport/CPU work. Neither
  is automatically “recovery GPU time.” The final cycle may have no next Target
  start; that endpoint stays missing.

See [protocol and source-level accounting](serial-prepost3-design.md).

## Pinned foreground command

Execution SHA: `f01e8d037007540209999179357c3dd2ff2335a7`. Both modes run this exact commit. The delivery
commit adds only this pin, launcher checks and documentation. Copy into the server
terminal (the outer shell remains open even on a nonzero child result):

```bash
if bash <<'SR_PREPOST_CHILD'
set -Eeuo pipefail
FINAL_SHA=f01e8d037007540209999179357c3dd2ff2335a7
REPO="${SR_PREPOST_REPO:-/root/autodl-tmp/src/SpecRhythm}"
git -C "$REPO" fetch origin codex/rolling-eager-v0.1
git -C "$REPO" cat-file -e "${FINAL_SHA}^{commit}"
RUN_TREE="${REPO}-prepost3-${FINAL_SHA:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
git -C "$REPO" worktree add --detach "$RUN_TREE" "$FINAL_SHA"
export SR_EXEC_REPO="$RUN_TREE"
bash "$RUN_TREE/scripts/run_prepost_b16.sh" "$FINAL_SHA"
SR_PREPOST_CHILD
then
  printf 'Prepost3 pair finished; return the printed evidence archives and comparison JSON.\n'
else
  rc=$?
  printf 'Prepost3 stopped at first failure (rc=%s); later points stopped. Interactive terminal remains open.\n' "$rc"
  # Keep the interactive parent open; the original code is printed above.
fi
```

Alternatively, from a checkout containing this delivery, run
`bash scripts/run_prepost_b16_pinned.sh`; its exit code remains the first failure.
