# K3 four-mode comparison: one foreground entry, one upload

Execution SHA: `5b50529f3bd3617f29c60ee9de6bf7143f96449e`.
Configuration commit: `ed6ff9703765e2c36b9ec4b3d0cb91edc1125d8f`.
The following entry includes the configuration, CPU optimization and acceptance repair. Draft PR5 remains Draft; no merge or other PR
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

Entry acceptance is mode-specific and uses `geometry(mode)` in configuration and
measurement. It cross-checks actual native TP request sets and batch statistics;
Serial B16 is accepted, Ping B16 is rejected. The joint full16-request fixture
additionally requires a full native B16 forward for each Serial mode (B8 for Ping),
so a hidden internal B8 cap cannot pass. The first full step/rank/host_start_ns/IDs
is retained in `joint/result.json` and `comparison.json`; raw native records remain
in the same archive. Partial performance batches keep original underfill and
lifecycle evidence. This repairs the old fca2118 ceiling8 entry omission, without
changing algorithms or reinterpreting historical results.

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
EXECUTION_SHA=5b50529f3bd3617f29c60ee9de6bf7143f96449e
git -C "$REPO" fetch origin codex/rolling-eager-v0.1
git -C "$REPO" cat-file -e "${EXECUTION_SHA}^{commit}"
RUN_TREE="${REPO}-k3-four-${EXECUTION_SHA:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
git -C "$REPO" worktree add --detach "$RUN_TREE" "$EXECUTION_SHA"
unset SR_K3_MANAGED_LOCAL SR_K3_LOCAL_DELIVERY SR_PING_DELIVERY
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
map to deduplicated objects with byte counts/SHA256. Per-file512MiB and total unique1GiB
budgets remain unchanged. The bounded inventory now allows512 logical/unique objects
for raw logs, partial JSON, ownership and publication receipts; overflow/omission still
fails integrity and is explicit. There are no nested subpackages or extra collection commands.

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


Entry-repair local validation: full pytest2645 passed/3 skipped; Python3.9 and3.12
related suites157 passed each; Ruff, three-version compileall, Python3.9 AST, all21
tracked Bash scripts and diff checks passed. Skips are opt-in CUDA and two Linux
process tests on macOS. These CPU results do not qualify a GPU run.


A100 storage default: all mutable execution files and the completed tar are under
`/tmp/specrhythm-runs/<unique-tag>/`; the verified final copy goes to
`/root/autodl-tmp/SpecRhythm-data/results/rolling-eager/`. Override with
`SR_K3_LOCAL_BASE` and `SR_PING_RESULTS`, respectively; do not set the private
`SR_K3_LOCAL_DELIVERY`/`SR_K3_MANAGED_LOCAL` handoff variables yourself.
`SR_K3_MIN_FREE_BYTES` defaults to8589934592. Startup records resolved paths/mount facts,
space and a write probe, and rejects known remote mutable storage or root overlap.
The server's `/tmp` is overlay, with physical backing unspecified. Model/S1 locations
and the local socket scheme are unchanged. Keep the local root after both outcomes.

The inner runner stops at first error; the outer process seals a single local archive
and makes one verified DPC copy. On copy failure upload the one printed local fallback;
it embeds the copy failure. No archive means an explicit retained-directory error,
not a fabricated upload path. Console delivery receipt records the tar SHA256 and copy
status; the successful sealed archive cannot contain its own later copy hash. Failure
status, partial/corrupt bytes and raw logs remain inside the package within byte limits.
Do not rerun this entry over an old root or collect extra subpackages. Four points on
one A100 with this recording configuration are the comparison; old A800 numbers are
not a controlled estimate of this infrastructure change. GPU/filesystem acceptance PENDING.

A100 local storage/read/report repair execution: `ae5be9a6b318b31931808fd1523064b33e2f0975`.
The entry-only commit pins this complete source; both scripts and the local delivery module
exist in this execution tree. The terminal block above invokes it in a fresh worktree.


Target-only deadline repair: the coordinator's one absolute drain deadline now reaches
shutdown and final publication, as it already reached diagnostic settlement. Receivers
reject missing/malformed/conflicting values before shutdown; only actual expiry is a
timeout. Error/receipt details and partial bytes remain in the same archive. A failed
protocol cannot be repaired into PASS by a later response. The new fixed entry is pinned
in a separate delivery commit to the tested implementation; no new GPU run has occurred.
See the dated sections in [validation](k3-validation.md) and [design](k3-design.md).

Current deadline-repair execution: `5b50529f3bd3617f29c60ee9de6bf7143f96449e`. The pinned launcher resolves exactly
this commit in a new worktree; it contains both runners, the deadline contract, real
service/RPC/publication code, joint correctness, comparison and local single-package
delivery modules. The earlier ae5be9a source is a historical storage-repair baseline,
not the current test execution. Local validation:2756 passed/3 skipped;3.9/3.12 related
432 each and final owner/deadline regression96 each on3.9/3.11/3.12; Ruff, compileall,
3.9 syntax,21 Bash scripts and diff PASS. GPU acceptance still PENDING.

Pinned launcher, actual foreground runner and local archive/delivery regressions:24 PASS.
The first GitHub push encountered a TLS connection error; the subsequent ordinary push
succeeded. New GitHub CI is queried separately from local tests and may still be pending.
