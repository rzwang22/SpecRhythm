# S2 server runbook — operator-run GPU validation

This file is the versioned template. The handoff also provides a rendered copy with
`@S2_COMMIT@` replaced by the final full commit SHA; use that rendered copy. The coding agent
has not connected to AutoDL or executed GPU work. Corrected PingPong and complete S2
GPU/performance qualification remain pending; earlier partial results are recorded below.
Keep PR #4 Draft/Open/unmerged. Existing S0/S1 roots are read-only.

This run validates the S2 terminal-drain concurrency correction based on
`24b31a9e0125d773697d92ec0ea333384616e539`. That G1 run reportedly exited zero in both
processes and completed owned cleanup, but failed same-cohort overlap qualification for a
terminal `finish_tail` against another request's verification. Preserve that failed root and
its original result/seal/logs; its exact path was not supplied with this report. Also preserve
the earlier root
`/root/autodl-tmp/SpecRhythm-data/results/phase-s2/s2-c12b3768eaaa-20260910T012202Z-1476`
unchanged; do not resume it with new code. Its reported small100/large390 capacity,
calibration/G0 and G1 Target/Serial PASS are partial results. Neither historical failure is
silently reclassified or overwritten. Use the new commit/root below and rerun capacity,
calibration, G0 and all of G1.
Capacity is measured again for the new run; 390 is never used as an input or fixed limit.

Run each block in the same foreground Bash session. If a block fails, stop at that gate;
review its automatic primary error/field/expected/actual/artifact/log-tail output. Do not
continue by editing validity, budgets, thresholds, traces, capacity or old artifacts.

## Checkout and environment

```bash
set -euo pipefail
cd /root/autodl-tmp/src/SpecRhythm
export S2_COMMIT='@S2_COMMIT@'
test -z "$(git status --porcelain --untracked-files=normal)"
git fetch origin codex/vllm-serving-v0.1
git checkout --detach "$S2_COMMIT"
test "$(git rev-parse HEAD)" = "$S2_COMMIT"
export GPU_PY=/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11
export PATH="$(dirname "$GPU_PY"):$PATH"
export PYTHONPATH="$PWD/src"
export SR_VLLM_SOURCE=/root/autodl-tmp/src/vllm-v0.25.1
export SR_DRAFT_MODEL=/root/autodl-tmp/models/Qwen3-0.6B
export SR_TARGET_MODEL=/root/autodl-tmp/models/Qwen3-32B
export HF_HOME=/root/.cache/huggingface
export OMP_NUM_THREADS=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_BATCH_INVARIANT=1
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTHONUNBUFFERED=1
unset USE_TORCH USE_TF USE_FLAX RANK WORLD_SIZE LOCAL_RANK
unset SR_S1_EXECUTION_MANIFEST SR_S2_EXECUTION_MANIFEST
"$GPU_PY" -m specrhythm.serving.s2_cli --help
test "$(git -C "$SR_VLLM_SOURCE" rev-parse HEAD)" = 752a3a504485790a2e8491cacbb35c137339ad34
export S2_S0=/root/autodl-tmp/SpecRhythm-data/results/phase-s0/20260909T042148Z-pypi-1837/build-a
export S2_S1=/root/autodl-tmp/SpecRhythm-data/results/phase-s1/s1p-5a00049-20260909T144802Z-1469
export SR_S2_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase-s2/s2-${S2_COMMIT:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
test ! -e "$SR_S2_ROOT"
printf 'S2 commit: %s\nS2 root: %s\n' "$S2_COMMIT" "$SR_S2_ROOT"
```

`PYTHONPATH` selects this checkout for the CLI and every child without reinstalling packages.
No vLLM patch application or framework/model upgrade is part of this runbook. The existing
five-patch installation is checked read-only. Prepare rejects a dirty/wrong checkout and a
changed sealed workload or S1 baseline. The launcher isolates fresh Target and Draft children
and binds their GPUs; do not launch this under torchrun or an inherited distributed rank.

## G0 preparation, actual capacity, independent SLO and trace freeze

The first command collects metadata without loading model weights. `capacity` then loads and
cleans up each mode once without running requests, collecting actual capacity from all nine
mode/rank combinations. Small100 and large500 shrink by ten on one common nested plan.
PingPong startup now initializes the real UUID query once per Target rank. A capacity probe
records one initial validation and zero verification accesses per rank; no verification batch
is required to pass that probe.

```bash
"$GPU_PY" -m specrhythm.serving.s2_cli prepare --root "$SR_S2_ROOT" \
  --s0 "$S2_S0" --s1 "$S2_S1" --selection-seed 1666 --arrival-seed 1667 --order-seed 1668
"$GPU_PY" -m specrhythm.serving.s2_cli capacity --root "$SR_S2_ROOT"
```

Default SLO calibration is a separate Target-only 20-request run, five per class from
calibration200, active limit one. It creates new engines and preloads all its selected KV.
Class thresholds are 1.5× median isolated active decode ms/token. Calibration queueing is
reported separately and does not inflate that baseline. Main SLO includes queueing.

```bash
"$GPU_PY" -m specrhythm.serving.s2_cli calibrate --root "$SR_S2_ROOT"
"$GPU_PY" -m specrhythm.serving.s2_cli freeze --root "$SR_S2_ROOT"
"$GPU_PY" -m specrhythm.serving.s2_cli gate --root "$SR_S2_ROOT" --gate G0
```

Alternative **before calibration/freeze**: supply reviewed positive class thresholds with
`slo --thresholds '{"chat":...,"code":...,"reasoning":...,"summarization":...}'` instead of
`calibrate`. Do not use illustrative thresholds as measured baselines. If the fixed 20-request
calibration pool cannot fit, explicit thresholds are required. Once written, policy changes
require a new root.

Default is one seed pair, with no automatic repeats. To request more seeds, declare explicit
pairs using `prepare --extra-seed ARRIVAL ORDER` (repeatable) in a **new** root before G0.
This intentionally adds those runs; their IDs, actual_N, capacities and SLO are shared with
the first seed. Do not append seeds or change the selected set after observing main results.

## G1 — ten-request dynamic smoke, three modes

```bash
"$GPU_PY" -m specrhythm.serving.s2_cli gate --root "$SR_S2_ROOT" --gate G1
"$GPU_PY" -m specrhythm.serving.s2_cli status --root "$SR_S2_ROOT"
```

G1 uses 3:3:2:2, lambda=.25, true preloaded prefixes, actual observed arrivals and global active
limit 128. Check retained arrival lag, queue/admission, no premature work, bootstrap accounting,
FIFO/cohort join boundaries and resource drain. Sparse traces can legitimately show no queue
beyond handling lag, only one cohort, singleton Draft batches, or zero overlap. Speed is not a
gate. Invalid execution/accounting/KV/measurement/cleanup stops all subsequent runs.
For PingPong, retained `dual_uuid_query` evidence in startup and final Target worker snapshots
must show one startup validation per rank. Verification events retain actual UUID/device
intervals and live-query counters accumulate across snapshots. Do not switch to cached mode.
**Stop here unless the entire three-mode G1 exits zero and writes a valid G1.json.**

The result labels `stage_dependency_contract=specrhythm.s2-terminal-drain.v1`. A same-cohort
`finish_tail` overlap is allowed only with native terminal/prefix evidence, no follow-up
proposal, private unchanged unrelated KV, disjoint requests and ordered real release.
`terminal_drain` reports its host-envelope overlap and token/KV/slot completion times separately;
the proposal `physical_overlap` metric keeps its existing scope. Zero drain work or overlap
can be valid. Do not interpret host intersections as exact kernel overlap or subtract tail
cost from makespan. All materialization forwards/GPU event costs and arrival-to-drain timing
remain included. On failure, the foreground prints operation, terminal proof, both request
sets/cohorts/intervals, intersection and the actual related artifacts. Preserve those failures.

## G2 — actual_small, .25/.5/1 requests/s, three modes each

```bash
"$GPU_PY" -m specrhythm.serving.s2_cli gate --root "$SR_S2_ROOT" --gate G2
"$GPU_PY" -m specrhythm.serving.s2_cli summary --root "$SR_S2_ROOT" \
  --output "$SR_S2_ROOT/G2-light-review.json"
```

G2 requires sealed G1 validity. The same input/trace is reused by Target, Serial and PingPong
at each offered rate. Every attempt freshly recreates and preloads its private KV. Defaults
run each mode once per rate, not a repeatability grid. Idle time remains in makespan.
Do not attempt G2 after a failed PingPong run or a missing G1.json. Resolve G1 first; an earlier
root's Target/Serial results do not qualify this new root. G3 additionally requires valid G2.

## G3 — actual_large only if capacity allows a distinct scale

```bash
"$GPU_PY" -m specrhythm.serving.s2_cli gate --root "$SR_S2_ROOT" --gate G3
"$GPU_PY" -m specrhythm.serving.s2_cli status --root "$SR_S2_ROOT"
```

G3 requires valid G2. If large cannot exceed small, G3 records the capacity duplicate skip and
runs no duplicate GPU experiment. Do not label that skip as a second observed scale.

## Interruption, owned cleanup and foreground resume

Do not reuse the old continuation KV. Keep the exact commit/environment/root printed above.
After interrupting the foreground launcher, resume starts a new attempt with fresh GPU state
for any unfinished run and reuses only matching sealed valid results. All recorded unfinished
process ownership is cleaned before an older valid result can be skipped. Cleanup checks exact
PID/start identity and socket ownership; it never performs a broad process kill.

```bash
"$GPU_PY" -m specrhythm.serving.s2_cli status --root "$SR_S2_ROOT"
"$GPU_PY" -m specrhythm.serving.s2_cli cleanup --root "$SR_S2_ROOT"
# Set this to the gate interrupted before it passed: G1, G2 or G3.
export S2_RESUME_GATE=G2
"$GPU_PY" -m specrhythm.serving.s2_cli resume --root "$SR_S2_ROOT" --gate "$S2_RESUME_GATE"
```

`capacity` and `freeze` can resume matching completed G0 inputs without overwriting them.
`calibrate` can resume its completed attempt if policy writing was interrupted, but refuses
to replace an existing SLO. A changed code/environment/policy uses a new root.

## Offline requalification, summary and upload

These commands read retained evidence and do not launch GPU children. Invalid historical
attempts remain in the upload summary; a successful fresh attempt does not erase their errors.
The default bundle is a compact JSON containing metrics, validity and raw file paths/hashes.
Full tokens, population/block tables and original logs remain in sealed run directories.

```bash
"$GPU_PY" -m specrhythm.serving.s2_cli offline --root "$SR_S2_ROOT" \
  --output "$SR_S2_ROOT/offline-review.json"
"$GPU_PY" -m specrhythm.serving.s2_cli bundle --root "$SR_S2_ROOT" \
  --output "$SR_S2_ROOT/s2-light-summary.json"
sha256sum "$SR_S2_ROOT/s2-light-summary.json"
```

Upload `s2-light-summary.json` first. Preserve the complete root for any follow-up evidence.
Report actual timed/total output quantities, throughput, request/token goodput, SLO attainment,
queue/first-output/TPOT distributions, B/Q/forward/acceptance counters, occupancy and makespan
together. Cross-run token/length/EOS/round equality is NOT_REQUIRED. Keep the label
**finite-trace serving observation at ideal GPU-resident prefilled-KV delivery**, and do not
claim equal-work batching speedup or full PD end-to-end benefit from makespan alone.
