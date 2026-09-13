# Unified K3: three fixed runtime points, one upload

This is an explicit new protocol, not a reinterpretation of old P1/P4 or Serial/B16
results. GPU output correctness, pipeline/native overlap and performance are PENDING.
Only the operator runs GPUs. The implementation and token/forward accounting are in
[k3-design.md](k3-design.md).

`serial-k3`, `pingpong-k3`, `pingpong-eager-k3` share active16, home8/8, Target ceiling8,
Draft GPU0 and a single TP2 Target on GPU1/2, frozen resident360 workload/seed1666,
no-bonus commits, actual K=min(3,remaining), runtime audit and buffered-live recording.
The source S1 config retains its historical K4 capacity envelope; the new mode's
actual engine num_speculative_tokens and proposal budget are explicitly3. GPU capacity
checks reserve3 Target positions and up to6 Draft speculative positions (eager only).
No model, sampling, control/drain budget, graph setting or measurement boundary changes.

`scripts/run_k3_b16.sh <full SHA>` runs three fresh capacity points, then one shared
Target-only + all three modes complete-output check (16 requests, at most32 output
fixture tokens), then three runtime performance points. Each point genuinely prefills
all360, warms up two rotations (four nonempty admissions), measures continuous30s,
and retains setup900s/drain60s. Workload SHA256 remains
`cdaf71adace15d229f5087b98f9fd162a958456226a660184fe03f5d6ebd8ff4`.
Correctness and mechanism coverage must pass before performance. It never retries or
expands B/load grids. All points use independent roots inside a fresh tagged delivery.

The script retains the first error. `joint/failure.json` names the actual mode and
run, the original report/process layer, process exit and command exit; subsequent
summary/export errors are separate. `inventory.json` includes logical_paths, unique
objects, SHA256/byte counts, missing fields, first code and export validation code.
The actual exporter process code is printed separately. Export COMPLETE is not
execution or correctness PASS. Disk failure after archive close cannot rewrite that
archive; it is reported separately, never relabelled success.

Upload only the absolute path printed once as:

```
UPLOAD ONLY: .../pingpong-k3-delivery-<tag>.tar.gz
```

One package contains comparison, joint output checks, first-failure, manifests,
startup/final TP/Draft identities, original runtime/native/owner reports and bounded
point analyses. The existing192-file/512MiB-file/1GiB-total bounds and phased trace
budgets stay in force; omissions/truncation remain explicit failures. There are no
nested subpackages to collect manually. If no archive can be created, the terminal
reports that export failure and retained source directory instead of inventing a path.

Read each point's `pingpong.cycles` for actual P lengths, short reasons, batches,
committed tokens, feedback→READY, READY→admission, admission→native Target, and work
reuse/discard. `pingpong.pipeline` separates physical normal/recovery/lookahead/KV
forwards, B histograms and GPU sums. Mixed-role forwards participate in each role;
those role sums must not be added. Native cross-home overlap uses interval unions.
Recovery GPU union outside any Target is bounded separately; it is not subtracted
from throughput. Full-step wall distributions and outside-step time remain reported.

The bounded example timeline starts16 rows before the first measured recovery,
retains up to128 rows, and reports omissions. Its forward IDs explicitly name array
indices in the original producer (not invented hardware event IDs); raw phased
records retain all request/proposal/version joins. No recorded recovery is labelled
NO_RECORDED_RECOVERY, not fabricated. A zero-overlap point must be read with owner
waiting_inventory, ready_inventory and raw coordinator/owner spans. Output correctness,
execution/measurement/cleanup, evidence integrity, cross-cohort pipeline, native overlap
and performance are separate conclusions. First single windows cannot establish small
stable speedup; a later authorized repeat can use a fresh SR_PING_RUN_TAG.

## Foreground command

Execution SHA: `b2a2d29210d68357590a567d4819f3552175d179`. The delivery commit adds
`scripts/run_k3_b16_pinned.sh` and this command; it does not change execution source.
Never source a strict runner into the interactive shell. Default repository is
`/root/autodl-tmp/src/SpecRhythm`, Python is
`/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11`, S1 is
`/root/autodl-tmp/SpecRhythm-data/results/phase-s1/s1p-5a00049-20260909T144802Z-1469`.

```bash
if bash <<'SR_K3_CHILD'
set -Eeuo pipefail
FINAL_SHA=b2a2d29210d68357590a567d4819f3552175d179
REPO=/root/autodl-tmp/src/SpecRhythm
git -C "$REPO" fetch origin codex/rolling-eager-v0.1
git -C "$REPO" cat-file -e "${FINAL_SHA}^{commit}"
RUN_TREE="${REPO}-k3-${FINAL_SHA:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
git -C "$REPO" worktree add --detach "$RUN_TREE" "$FINAL_SHA"
export SR_EXEC_REPO="$RUN_TREE"
bash "$RUN_TREE/scripts/run_k3_b16.sh" "$FINAL_SHA"
SR_K3_CHILD
then
  printf 'K3 completed; upload only the archive printed as UPLOAD ONLY.\n'
else
  rc=$?
  printf 'K3 stopped, original rc=%s; later points stopped; terminal remains open.\n' "$rc"
fi
```
