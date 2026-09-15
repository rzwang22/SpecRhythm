# K3 four-mode comparison: one foreground entry, one upload

Execution SHA: `fca2118b89d21b11f18f5de43d69657e820087ab`.
Configuration commit: `ed6ff9703765e2c36b9ec4b3d0cb91edc1125d8f`.
The following entry runs both changes. Draft PR5 remains Draft; no merge or other PR
changes. Only the operator runs GPUs. New capacity, correctness, cleanup, native
pipeline overlap and performance are **PENDING**.

| Mode | Total active | Homes | Target request ceiling | Draft physical merge ceiling |
|---|---:|---|---:|---:|
| serial-k3 |16|A16|16|16|
| serial-eager-k3 |16|A16|16|16|
| pingpong-k3 |16|A8/B8|8|16|
| pingpong-eager-k3 |16|A8/B8|8|16|

All modes use trueK3 (seed+two extensions), no extra bonus commit, the same model
and sampling configuration, resident360 workload/seed1666, Draft GPU0 and one TP2
Target on GPU1/2, runtime audit and buffered-live recording. Serial's all-Draft-idle
gate remains; Serial-eager permits only own-parent speculation. PingPong's claim,
feedback priority, owner token-step boundary and READY publication are unchanged.
A short tail must have an EOS/output-budget reason. Refusal or insufficient capacity
never silently lowers B16 toB8. Demand3/reserve4 and eager Draft6/6 remain distinct
from actual candidates and query positions. Capacity v2 checks each mode's geometry.

This is not a relabelling of the old `d007dce448bd2a7d510172222ae166d7bb6f299e`
Serial B8 result. That three-mode run and its original PASS/single-window conclusions
remain historical. See [validation](k3-validation.md) and [design](k3-design.md).

The flow is fixed:

1. No-model static four-mode capacity/interface check.
2. Four independent fresh capacity points, with three physical rank observations each.
3. One Target-only reference and all four complete-output correctness runs:16 requests,
   up to32 output tokens/request, nominal512 per mode, exact per-request comparison.
4. Four independent runtime performance points. Each genuinely prefills all360, warms
   up32 actual request-verification opportunities, then measures continuous30s.
5. Strict report/diagnostic/configuration comparison and a single bounded archive.

Full Serial warmup is two Target steps; full PingPong is four. Actual request opportunity
counts, per-request distribution and unique coverage are retained. A partial step
contributes only its real B. No sample12 cutoff, automatic retry, model change or grid.
Setup900s/drain60s stay unchanged. Workload SHA256:
`cdaf71adace15d229f5087b98f9fd162a958456226a660184fe03f5d6ebd8ff4`.

Copy this **complete block** into the server's foreground terminal. Do not source the
script. Strict mode and exit handling stay inside the child; the parent shell remains
open on failure. A fresh detached worktree and new result root are created.

```bash
if bash <<'SR_K3_FRONTEND'
set -Eeuo pipefail
REPO=/root/autodl-tmp/src/SpecRhythm
EXECUTION_SHA=fca2118b89d21b11f18f5de43d69657e820087ab
git -C "$REPO" fetch origin codex/rolling-eager-v0.1
git -C "$REPO" cat-file -e "${EXECUTION_SHA}^{commit}"
RUN_TREE="${REPO}-k3-four-${EXECUTION_SHA:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
git -C "$REPO" worktree add --detach "$RUN_TREE" "$EXECUTION_SHA"
export SR_EXEC_REPO="$RUN_TREE"
bash "$RUN_TREE/scripts/run_k3_b16.sh" "$EXECUTION_SHA"
SR_K3_FRONTEND
then
  printf 'Four-mode run finished; upload only the archive named by the runner.\n'
else
  rc=$?
  printf 'Stopped with original rc=%s; later points stopped; terminal remains open.\n' "$rc"
fi
```

The repository also provides `scripts/run_k3_b16_pinned.sh` with this exact execution
SHA. It fetches, creates a new worktree, then calls the execution commit's actual
four-mode runner. The source commit contains `run_k3_b16.sh`, `k3_capacity`, joint
correctness, analysis and single-package exporter; the entry does not depend on an
uncommitted file on the Mac.

Defaults remain the existing S1 root and Python environment in the runner. Optional
`SR_FIXED_PYTHON`, `SR_FIXED_S1`, `SR_PING_RESULTS` and a new `SR_PING_RUN_TAG` can select
existing server locations; they do not change the frozen workload/model requirements.
Do not reuse a historical tag. No AutoDL connection or GPU invocation was made locally.

Only upload the single path printed once by the runner:

```text
UPLOAD ONLY: .../pingpong-k3-delivery-<tag>.tar.gz
```

`comparison.json`, `joint/result.json` (or the real joint failure), per-mode reports,
capacity, startup/final device identity, native/owner timelines, original runtime,
first-failure and `inventory.json` all live in that package. Inventory logical_paths
map to deduplicated objects with byte counts/SHA256. The192-file/512MiB-file/1GiB-total
limits and phased trace budgets remain bounded. Extending the previous145 logical
files by the corresponding fourth-mode42 files gives187; actual inventory, missing
fields or size overflow still decide export integrity, not this estimate. There are
no nested subpackages or additional collection commands.

On first failure, no later point runs. The actual phase/mode/run, process code,
report qualification, cleanup and command code are retained; summary/export failures
are secondary. A closed engine API is not proof of process cleanup. Export COMPLETE
means archive integrity, not execution correctness or speedup. If disk failure prevents
archive creation, the runner reports that failure/source directory instead of inventing
an upload path. A later exporter error does not replace the original nonzero code.

The comparison requires the same source, options, workload, models, config/patch,
numerical mode, engine/launch settings and resource limits. Its four-mode geometry
contract explicitly permits Target16 versus8. Runtime UUID checks are within-run only.
Missing common fields do not become zero/PASS. Read the results separately as output
correctness; execution/measurement/cleanup; evidence integrity; cross-home pipeline;
native GPU overlap; and measured performance.

Per-mode reports retain actual batch histograms, tokens/steps/throughput, window/steps
cadence, complete-step distributions and outside-step time, READY publication→claim,
claim→Target, six resident phases, normalization scale, admission records/encoding count,
role-specific physical forwards/B/GPU time and promotion/reuse/discard. The `stock`
child is the actual pinned scheduler call; its parent includes wrapper work. Hash,
JSON and fsync are children, not additive wall-clock columns.

`pipeline.cross_request_overlap_steps`, `cross_home_native_overlap`,
`recovery_coverage_by_other_requests` and `parent_eager_native_overlap` retain distinct
meanings. Recovery denominator is a union of physical Draft intervals with recovery
participants whose host launches are inside the window, device bounds clipped to it.
Numerator is the union intersecting TP verification for other requests, with duplicate
bindings/roles/ranks removed. Lower coverage divides lower intersection by upper
recovery union; upper divides upper by lower, capped at1. Raw home/proposal/version
joins and bounded rejection timelines are retained. Missing endpoints remain missing.

OBSERVED means some nonzero native overlap, not adequate recovery hiding. A single
window's difference is not stable speedup, and CPU audit/record time is never subtracted
from measured throughput. Earlier READY publication remains a separate future change,
to be decided after this package returns.
