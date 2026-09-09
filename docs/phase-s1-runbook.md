# Phase S1-P — pinned 3×A800 server runbook

**Agent GPU status: PENDING; no GPU run or AutoDL connection was performed.**
Run only G0–G3, on the existing environment. Keep PR #4 Draft. Do not install/upgrade
vLLM/Torch, rerun S0 construction, alter patches/tokenizers, lower budgets or select
different prompts. See [design](phase-s1-design.md) and [schema](phase-s1-schema.md).

S1-P uses `acceptance_policy=s1-performance-v1`; independent output equality is
NOT_REQUIRED. The previous GPU gate stopped under the old exact-output policy.
Preserve this old root unchanged; do not run resume/compare to rewrite its acceptance:
`/root/autodl-tmp/SpecRhythm-data/results/phase-s1/cd36d18-20260909T110103Z-1489`.
Preserve all other failed S1-P roots as well, including runs at `08cf93a…` and
`645635d…`. At `645635d…` the operator completed G0/Target; Serial failed during
startup because its decode-ready context had not been created. Those completed and
failed artifacts remain unchanged. The corrected S1 adapter creates the Serial
context before Target LLM worker construction; use a fresh root at the new commit.
Prepare a fresh root at the delivered commit, then run G1 again. This runbook does
not implement S1-D numerical diagnosis, an output-equality recovery sweep or S2/S3.

Use Bash. Set `SR_S1_COMMIT` to the final full commit in the delivery message once.
The branch update below is fast-forward only and checks the exact delivered SHA.
The model paths, HF cache, source checkout and GPU interpreter are fixed in the helper.
The helper puts this checkout's `src` first on `PYTHONPATH`, so every child imports
the pinned code without reinstalling packages in the GPU environment. No S0 CPU
environment is used for inference.

## G0 — freeze input and check installed environment, without loading weights

```bash
export SR_S1_COMMIT='<full delivered S1-P commit>'
export SR_S1_REPO=/root/autodl-tmp/src/SpecRhythm
export SR_S1_PYTHON=/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11
export SR_S1_S0=/root/autodl-tmp/SpecRhythm-data/results/phase-s0/20260909T042148Z-pypi-1837/build-a
export SR_S1_BASE=/root/autodl-tmp/SpecRhythm-data/results/phase-s1
export SR_S1_RUN_ID="s1p-serial-context-$(date -u +%Y%m%dT%H%M%SZ)-$$"
export SR_S1_ROOT="$SR_S1_BASE/$SR_S1_RUN_ID"
export OMP_NUM_THREADS=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
unset USE_TORCH USE_TF USE_FLAX
export HF_HOME=/root/.cache/huggingface
export PATH="/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin:$PATH"
(
  set -euo pipefail
  cd "$SR_S1_REPO"
  test -z "$(git status --porcelain)"
  git fetch origin codex/vllm-serving-v0.1
  git switch codex/vllm-serving-v0.1
  git merge --ff-only "$SR_S1_COMMIT"
  test "$(git rev-parse HEAD)" = "$SR_S1_COMMIT"
  test -z "$(git status --porcelain)"
  "$SR_S1_PYTHON" --version
  test ! -e "$SR_S1_ROOT"
  mkdir -p "$SR_S1_BASE"
  set +e
  bash integrations/vllm/phase_s1.sh prepare --root "$SR_S1_ROOT" \
    --s0 "$SR_S1_S0" --vllm-source /root/autodl-tmp/src/vllm-v0.25.1 \
    2>&1 | tee "$SR_S1_BASE/$SR_S1_RUN_ID-g0.log"
  SR_S1_G0_STATUS=("${PIPESTATUS[@]}")
  set -e
  if [ "${SR_S1_G0_STATUS[0]}" -ne 0 ]; then exit "${SR_S1_G0_STATUS[0]}"; fi
  test "${SR_S1_G0_STATUS[1]}" -eq 0
  mv "$SR_S1_BASE/$SR_S1_RUN_ID-g0.log" "$SR_S1_ROOT/g0.log"
  cat "$SR_S1_ROOT/g0.json"
)
```

Proceed only if the subshell exits 0, `g0.json.valid=true` and
`acceptance_policy=s1-performance-v1` (G0 schema v2). If prepare fails,
keep the outside `*-g0.log` and partially created root; do not reuse that G0 root.
Correct the environment/input problem first, then choose a fresh run ID.
`environment.json`, `topology.json`, `config.json`, `patch-manifest.json`, capacity
estimates and three subset manifests (execution schema v2) are now frozen. The check uses Python3.11,
Torch2.11.0, vLLM0.25.1/commit `752a3a5…` and the existing five-patch state. It queries
device metadata but loads no model weights. Each actual engine later validates its
allocated capacity before generation. G0's byte estimate alone is not GPU PASS.

Draft is GPU0/TP1/BF16/eager/MRV1; Target is GPU1/2/TP2/BF16/eager, K4,
batch-invariant, live UUID. Helpers clear stale Phase4/S0 switches and set each child
CUDA visibility explicitly. `OMP_NUM_THREADS=1` and
`VLLM_ALLOW_INSECURE_SERIALIZATION=1` are set by the committed helper and each child,
not inherited by chance from another terminal. The RPC flag
enables the project's existing local callable RPC execution path. Actual values and
absent USE_TORCH/TF/FLAX are retained in G0 execution, command/Draft-owner and effective
runtime records. These settings do not upgrade packages or change GPU placement.

## G1 — smoke4 three modes and independent PingPong observation

```bash
(
  set -euo pipefail
  cd "$SR_S1_REPO"
  bash integrations/vllm/phase_s1.sh gate --root "$SR_S1_ROOT" --gate G1
)
```

`gate` is the default foreground entry. It blocks until the gate passes or fails,
and returns the actual effective execution status (or 1 for qualification failure).
Run it directly, without a `tee` pipeline around GPU execution. Target and Draft
stdout/stderr continue to write directly to `target.log` and `draft-service.log`.
A read-only mirror displays both streams live, tagged `[S1-P <mode> Target]` and
`[S1-P <mode> Draft]`; original log bytes are unchanged. No extra GPU synchronization
or per-token fsync is added. Closing the terminal is not a promised detached run;
use the explicit optional background entry below when disconnect survival is needed.

On failure the command automatically prints qualification errors, field-level Draft
checks (`field`, `expected`, `actual`, `artifact`), exit/lifecycle evidence, and the
last 40 lines of each child log (bounded to 64 KiB per file). No separate log search
is needed. The failed stage retains its mode/attempt directory. A later artifact
error cannot replace a recorded nonzero Target/Draft execution status. Old failed
artifacts are retained; current-code retry uses a new root and fresh G0.
On a child startup exception, `child-failure.json` (or `draft-child-failure.json`)
retains the original exception and traceback. The failed result's `primary_error`
and first error retain that exception; absent downstream reports are explicitly
`secondary_diagnostics` / `MISSING_AFTER_EXECUTION_FAILURE`, not replacement causes.

G1 order: resident Target, Serial, PingPong, then a fresh independent PingPong.
Raw Target is not part of the S1-P gate or a prerequisite. Each owned run has a fresh
engine/KV/result directory. All four requests must complete with internally valid
bootstrap/commit/EOS/budget, Target/Draft state, accounting, measurement and cleanup.
Independent output tokens, bootstraps, lengths, finish reasons and rounds may differ.
The second PingPong is a repeated performance observation, not an equality test.

After `gate` returns successfully, inspect its result and optionally requalify offline:

```bash
cat "$SR_S1_ROOT/exit-code.json"
cat "$SR_S1_ROOT/G1/comparison.json"
bash integrations/vllm/phase_s1.sh compare --root "$SR_S1_ROOT" --gate G1
```

Require foreground command exit 0, stage PASS and comparison `valid=true`. A throughput ratio below 1
or no physical overlap does not block. Internal execution/accounting/measurement/cleanup
failure does block; independent output differences do not. In case of a material failure,
preserve it and stop. The comparator emits a new immutable offline report; it never
changes the original sealed artifacts.

## G2 — mixed20, three modes once

```bash
(
  set -euo pipefail
  cd "$SR_S1_REPO"
  bash integrations/vllm/phase_s1.sh gate --root "$SR_S1_ROOT" --gate G2
)
```

`gate` waits for completion. Continue only after it exits 0:

```bash
cat "$SR_S1_ROOT/exit-code.json"
cat "$SR_S1_ROOT/G2/comparison.json"
bash integrations/vllm/phase_s1.sh compare --root "$SR_S1_ROOT" --gate G2
```

G2 independently rechecks G1's seals and comparison before any new run. Require all
20 requests in all three modes, internally valid execution/accounting, valid measurement
and clean owned shutdown. Each run uses its own actual timed tokens for tok/s. The actual engine capacities retained here are inputs to G3 admission.

## G3 — mixed100, three fixed rotations

```bash
(
  set -euo pipefail
  cd "$SR_S1_REPO"
  bash integrations/vllm/phase_s1.sh gate --root "$SR_S1_ROOT" --gate G3
)
```

G3 first checks worst-case 100-request KV/sequence/query needs against both the
estimate and G2's actual Target/Draft allocated blocks. Insufficient capacity records
**BLOCKED** and starts no G3 engine. Do not reduce N or 512/1024 caps to get a PASS.
The fixed orders are Target→Serial→PingPong, Serial→PingPong→Target,
PingPong→Target→Serial. No parameter grid, legacy sweep or full main1000 run follows.

After the foreground command exits 0:

```bash
cat "$SR_S1_ROOT/exit-code.json"
cat "$SR_S1_ROOT/G3/comparison.json"
bash integrations/vllm/phase_s1.sh compare --root "$SR_S1_ROOT" --gate G3
```

Require all nine runs to have valid execution, measurement and cleanup. Generated
outputs, actual lengths and termination may differ across all modes/repeats.
Report raw metrics, median and population stdev, actual-token throughput ratios,
work, length/EOS/cap distributions, B/Q/progress/duration, acceptance and overlap.
Makespan ratios accompany each run's actual output counts and are not equal-work
speedups. Reports use `resident decode-only three-mode performance observation`;
there is no end-to-end improvement/serving TTFT/SLO claim. Output/round equality is
not performed; its fields are null/NOT_REQUIRED. Unavailable attribution stays explicit.
No positive gain is required. Partial/OOM/invalid runs never enter complete-run aggregates.

## Disconnect, failure cleanup and resume

After a disconnect or an interrupted foreground command, reconnect and restore `SR_S1_ROOT` to the **same existing
absolute root**, plus the repo/Python variables above. Do not generate a new root for
resume. `status` is safe while running; do not start another GPU task concurrently.
There is a kernel lock against concurrent S1 supervisors using this root.
Confirm the prior launcher has exited and owned cleanup is complete before retrying.

```bash
cd /root/autodl-tmp/src/SpecRhythm
bash integrations/vllm/phase_s1.sh status --root "$SR_S1_ROOT"
cat "$SR_S1_ROOT/stage.json"
cat "$SR_S1_ROOT/exit-code.json"
```

If the launcher exited or was interrupted, rerun `gate` in the foreground to resume the same gate. The helper first
verifies/cleans owned incomplete attempts, verifies every seal plus the current S1-P
policy/schema and frozen execution identity, skips completed valid runs, and creates
a new attempt for interrupted/failed work. It does not restart from
half-written JSON or lost live KV. Resolve a real runtime/config failure before retrying;
if code must change, begin a new S1 root and repeat its gates with that new frozen code.

```bash
bash integrations/vllm/phase_s1.sh gate --root "$SR_S1_ROOT" --gate G1
```

Replace G1 with the interrupted G2/G3 as appropriate. For explicit cleanup only,
copy the exact attempt directory from `stage.json`, after `alive=false`:

```bash
export SR_S1_ATTEMPT="$SR_S1_ROOT/G1/0-target/attempt-001"
bash integrations/vllm/phase_s1.sh cleanup --root "$SR_S1_ROOT" --directory "$SR_S1_ATTEMPT"
```

Cleanup uses recorded PID/start identities, unique inherited launch tokens and exact
owned socket identity. It writes a new sidecar outside the retained attempt. It never
uses broad `pkill` or touches other jobs. Missing ownership proof or remaining processes
blocks further work and needs inspection. A prior-policy run can never be skipped
as a new-policy PASS. Raw coordinator and effective timeout/cleanup return codes
are retained separately; timeout is 4 hours per GPU child, Draft startup
timeout is 15 minutes. Timeout/failed attempts never enter performance aggregates.

## Optional detached execution

Foreground `gate` above is the default. To deliberately survive SSH closure, use
`start` for a new gate or `resume` for an interrupted gate instead. Both return
immediately with a PID and launcher log; their immediate exit is not gate PASS.
The same live mirror and failure summary appear in the launcher log.

```bash
bash integrations/vllm/phase_s1.sh start --root "$SR_S1_ROOT" --gate G1
bash integrations/vllm/phase_s1.sh status --root "$SR_S1_ROOT"
tail -n 80 "$SR_S1_ROOT"/launcher-*.log
# Only after the old launcher has exited, for the same interrupted gate:
bash integrations/vllm/phase_s1.sh resume --root "$SR_S1_ROOT" --gate G1
```

## Verify and return a review bundle

```bash
bash integrations/vllm/phase_s1.sh verify --root "$SR_S1_ROOT" --directory "$SR_S1_ATTEMPT"
bash integrations/vllm/phase_s1.sh bundle --root "$SR_S1_ROOT"
```

`verify` applies to a completed sealed attempt (set `SR_S1_ATTEMPT` accordingly).
For a failed gate you can still bundle retained failure artifacts once owned processes
are gone. Bundle output prints archive path, SHA256, inventory path and file count.
Return `<root>-review.tar.gz`, its printed SHA256 and the printed inventory JSON. It includes G0 environment,
capacity and frozen subsets, every retained attempt/log/exit/lifecycle/result/seal,
gate comparisons and recovery records; see the [file contract](phase-s1-schema.md).
The inventory binds all included bytes. Archive/manifest/input files are never
rewritten to append an acceptance note. Bundle creation is exclusive: do it once,
then keep that archive unchanged. There is no S1 server bundle hash until this runs.

Stop after the S1-P gate result and handoff; S2/S3 remain out of scope.
