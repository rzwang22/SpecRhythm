# Unified K3: three fixed runtime points, one upload

This is an explicit new protocol, not a reinterpretation of old P1/P4 or Serial/B16
results. The returned778d run passed capacity/joint output/execution/measurement/cleanup,
but normal/recovery cross-request native overlap remained zero. This follow-up changes
only repeated Target immutable-prompt digest work and offline dispatch evidence; new
GPU output correctness, pipeline/native overlap and performance remain PENDING.
Only the operator runs GPUs. The implementation and token/forward accounting are in
[k3-design.md](k3-design.md).

`serial-k3`, `pingpong-k3`, `pingpong-eager-k3` share active16, home8/8, Target ceiling8,
Draft GPU0 and a single TP2 Target on GPU1/2, frozen resident360 workload/seed1666,
no-bonus commits, actual K=min(3,remaining), runtime audit and buffered-live recording.
The source S1 config retains its historical K4 capacity envelope; the new mode's
actual engine num_speculative_tokens and proposal budget are explicitly3. GPU capacity
checks require3 Target/ordinary Draft positions but conservatively reserve4; eager
Draft requires and reserves6. `legacy_minimum_reserve=4` is unchanged. The report
records both demand and allocation; reserving4 does not generate a fourth candidate.
No model, sampling, control/drain budget, graph setting or measurement boundary changes.

`scripts/run_k3_b16.sh <full SHA>` first runs the no-GPU static capacity interface
check (`k3-capacity-contract.json`, GPU_capacity=PENDING), then three fresh real
capacity points, then one shared
Target-only + all three modes complete-output check (16 requests, at most32 output
fixture tokens), then three runtime performance points. Each point genuinely prefills
all360, warms up two rotations (four nonempty admissions), measures continuous30s,
and retains setup900s/drain60s. Workload SHA256 remains
`cdaf71adace15d229f5087b98f9fd162a958456226a660184fe03f5d6ebd8ff4`.
Correctness and mechanism coverage must pass before performance. It never retries or
expands B/load grids. All points use independent roots inside a fresh tagged delivery.

The script retains the first error. Before any point starts, failure attribution uses
the new delivery/not_started directory, never an inherited historical SR_FIXED_ROOT. `joint/failure.json` names the actual mode and
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
nested subpackages to collect manually. Capacity reports retain the three raw rank
observations, per-request budgets and demand/reserve schema for offline recomputation.
Pre-drive failures also retain startup-cleanup.json, original/secondary errors and
process exit codes; returned cleanup APIs alone never override supervisor qualification. If no archive can be created, the terminal
reports that export failure and retained source directory instead of inventing a path.

Read each point's `pingpong.cycles` for actual P lengths, short reasons, batches,
committed tokens, feedback→READY, READY→admission, admission→native Target, and work
reuse/discard. `pingpong.pipeline` separates physical normal/recovery/lookahead/KV
forwards, B histograms and GPU sums. Mixed-role forwards participate in each role;
those role sums must not be added. Native associations first verify actual request/proposal/version and TP identity, then
classify other-request and other-home work. `cross_request_native_overlap` separates
ordinary and rejection recovery from `parent_eager_native_overlap`. These are interval
unions, not sums of per-request or TP event times.
Recovery GPU union outside any Target is bounded separately; it is not subtracted
from throughput. Full-step wall distributions and outside-step time remain reported.

`pipeline.dispatch` reports actual owner READY publication→claim, claim→native Target,
pre-pool/stock-resident/post-pool phase spans, hash/proof counts, and GPU-idle intervals
while subsequently claimed READY work exists. The original `ready_ns` is physical
proposal completion; the actual publication timestamp is separately named. A GPU-idle
interval is not proof that CPU sampling or Target dispatch was already available.
Scheduler child spans are nested; unknown fields remain null. Proof policy
`k3-current-prompt-proof.v1` records live row count, newly hashed prompts, reused
immutable digests and current tokens compared. Both full live block checks remain.

`pipeline.rejection_cycle` adds READY/publication/claim, both native TP forwards,
validated owner feedback, physical ordinary/recovery work and next READY (at most128
rows). Target/Draft forward IDs name original producer array indices with rank,
request/proposal versions and calibrated bounds. Comparison embeds only aggregates;
point reports hold detailed rows once, and the projected raw records remain complete.
Its missing/omitted counts are explicit; the previous compact Draft timeline is
retained for compatibility. That bounded example timeline starts16 rows before the first measured recovery,
retains up to128 rows, and reports omissions. Its forward IDs explicitly name array
indices in the original producer (not invented hardware event IDs); raw phased
records retain all request/proposal/version joins. No recorded recovery is labelled
NO_RECORDED_RECOVERY, not fabricated. A zero-overlap point must be read with owner
waiting_inventory, ready_inventory and raw coordinator/owner spans. Output correctness,
execution/measurement/cleanup, evidence integrity, cross-cohort pipeline, native overlap
and performance are separate conclusions. `cross_request_pipeline_behavior` remains
NOT_DEMONSTRATED if only same-parent eager overlaps; INCOMPLETE native evidence is not
zero. If the new ordinary point still has zero overlap, inspect READY eligibility,
one-step admission wait and the three scheduler phases before proposing another
change. Do not relabel it achieved based on throughput. First single windows cannot establish small
stable speedup; a later authorized repeat can use a fresh SR_PING_RUN_TAG.

## Foreground command

Execution SHA: `8a3aca007fb60f9f67d1dcb7bedb5f8b78b8602e`. The delivery commit updates
`scripts/run_k3_b16_pinned.sh` and this command; it does not change execution source.
Never source a strict runner into the interactive shell. Default repository is
`/root/autodl-tmp/src/SpecRhythm`, Python is
`/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11`, S1 is
`/root/autodl-tmp/SpecRhythm-data/results/phase-s1/s1p-5a00049-20260909T144802Z-1469`.

```bash
if bash <<'SR_K3_CHILD'
set -Eeuo pipefail
FINAL_SHA=8a3aca007fb60f9f67d1dcb7bedb5f8b78b8602e
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

## Resident CPU decomposition in this follow-up

The fixed three-mode flow and all model/workload/budget/observation parameters remain
unchanged. The execution diagnostic gate additionally requires all six resident
subphase spans and their policy/work metadata, on the same PID/thread and cycle.
No additional GPU point or retry is introduced. Six bounded records per dispatch
use the existing phased trace budgets and are retained once in the raw runtime host
trace. Missing fields are INCOMPLETE, never zero-filled. The previous run's original
qualification is not revised by the new diagnostic contract.

Read `comparison.json` for window/steps cadence alongside complete_step_wall_ms and
outside_complete_steps_ms. `dispatch.resident_schedule` aggregates binding/readiness/
decisions/stock/initial_finish/admission-record phases and work scales; detailed
per-step records are only in the referenced point audit report. Work totals count
full current-row int visits and normalized rows, including the generated suffix.
The stock child is the actual pinned Scheduler.schedule call with dynamic predicate
callbacks; it is distinct from the inclusive target_resident_stock_schedule wrapper.

`pipeline.cross_request_overlap_steps` reports definite/uncertainty-only counts over
measured Target steps. `recovery_coverage_by_other_requests` gives the clipped native
physical recovery union denominator, covered interval union and conservative fraction
bounds. A mixed physical batch or two TP ranks are not additive copies. OBSERVED only
means some positive overlap; low coverage remains low even if throughput improves.
The numerator/denominator/window semantics are embedded in the report. All raw
request/proposal/version/forward evidence remains in the same single total archive.
GPU correctness, coverage and performance for the new SHA are PENDING until this run.
