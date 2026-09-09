# Phase S1 — pinned 3×A800 server runbook

**Agent GPU status: PENDING; no GPU run or AutoDL connection was performed.**
Run only G0–G3, on the existing environment. Keep PR #4 Draft. Do not install/upgrade
vLLM/Torch, rerun S0 construction, alter patches/tokenizers, lower budgets or select
different prompts. See [design](phase-s1-design.md) and [schema](phase-s1-schema.md).

Use Bash. Set `SR_S1_COMMIT` to the final full commit in the delivery message once.
The branch update below is fast-forward only and checks the exact delivered SHA.
The model paths, HF cache, source checkout and GPU interpreter are fixed in the helper.
No S0 CPU environment is used for inference.

## G0 — freeze input and check installed environment, without loading weights

```bash
export SR_S1_COMMIT='<full delivered S1 commit>'
export SR_S1_REPO=/root/autodl-tmp/src/SpecRhythm
export SR_S1_PYTHON=/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11
export SR_S1_S0=/root/autodl-tmp/SpecRhythm-data/results/phase-s0/20260909T042148Z-pypi-1837/build-a
export SR_S1_BASE=/root/autodl-tmp/SpecRhythm-data/results/phase-s1
export SR_S1_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
export SR_S1_ROOT="$SR_S1_BASE/$SR_S1_RUN_ID"
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
  bash integrations/vllm/phase_s1.sh prepare --root "$SR_S1_ROOT" \
    --s0 "$SR_S1_S0" --vllm-source /root/autodl-tmp/src/vllm-v0.25.1 \
    > "$SR_S1_BASE/$SR_S1_RUN_ID-g0.log" 2>&1
  mv "$SR_S1_BASE/$SR_S1_RUN_ID-g0.log" "$SR_S1_ROOT/g0.log"
  cat "$SR_S1_ROOT/g0.json"
)
```

Proceed only if the subshell exits 0 and `g0.json.valid=true`. If prepare fails,
keep the outside `*-g0.log` and partially created root; do not reuse that G0 root.
Correct the environment/input problem first, then choose a fresh run ID.
`environment.json`, `topology.json`, `config.json`, `patch-manifest.json`, capacity
estimates and three subset manifests are now frozen. The check uses Python3.11,
Torch2.11.0, vLLM0.25.1/commit `752a3a5…` and the existing five-patch state. It queries
device metadata but loads no model weights. Each actual engine later validates its
allocated capacity before generation. G0's byte estimate alone is not GPU PASS.

Draft is GPU0/TP1/BF16/eager/MRV1; Target is GPU1/2/TP2/BF16/eager, K4,
batch-invariant, live UUID. Helpers clear stale Phase4/S0 switches and set each child
CUDA visibility explicitly. They do not clear the HF cache or modify the GPU environment.

## G1 — smoke4 raw reference, three modes and independent PingPong repeat

```bash
cd "$SR_S1_REPO"
bash integrations/vllm/phase_s1.sh start --root "$SR_S1_ROOT" --gate G1
bash integrations/vllm/phase_s1.sh status --root "$SR_S1_ROOT"
cat "$SR_S1_ROOT/stage.json"
tail -n 80 "$SR_S1_ROOT"/launcher-*.log
```

`start` returns launcher PID/log immediately. It uses a new session, `/dev/null`
stdin and persistent stdout/stderr, so closing SSH does not stop it. `status` checks
PID **and start identity**, rather than trusting a stale PID file. An active GPU child
may spend time loading weights; its continuous log lives under the current attempt
reported in `stage.json` (`target.log`, `draft-service.log`).

G1 order: raw Target (two ordinary reference repetitions in that reference engine),
resident Target, Serial, PingPong, then a fresh independent PingPong. Each owned run
has fresh engine/KV/result directory. All four requests, exact tokens/bootstrap/EOS,
accounting, structural verification, measurement and cleanup must pass. The reference
is read only by the offline comparison, never by the Draft execution process.

After `status` says `alive=false`, check and independently requalify without GPU:

```bash
cat "$SR_S1_ROOT/exit-code.json"
cat "$SR_S1_ROOT/G1/comparison.json"
bash integrations/vllm/phase_s1.sh compare --root "$SR_S1_ROOT" --gate G1
```

Require launcher exit 0, stage PASS and comparison `valid=true`. A speedup below 1
or no physical overlap does not block. Exactness/accounting/measurement/cleanup
failure does block; preserve it and stop. The comparator emits a new immutable
offline report; it never changes the original sealed artifacts.

## G2 — mixed20, three modes once

```bash
bash integrations/vllm/phase_s1.sh start --root "$SR_S1_ROOT" --gate G2
bash integrations/vllm/phase_s1.sh status --root "$SR_S1_ROOT"
tail -n 80 "$SR_S1_ROOT"/launcher-*.log
```

Wait for completion, then:

```bash
cat "$SR_S1_ROOT/exit-code.json"
cat "$SR_S1_ROOT/G2/comparison.json"
bash integrations/vllm/phase_s1.sh compare --root "$SR_S1_ROOT" --gate G2
```

G2 independently rechecks G1's seals and comparison before any new run. Require all
20 requests in all three modes and exact equality, valid measurement and clean owned
shutdown. The actual engine capacities retained here are inputs to G3 admission.

## G3 — mixed100, three fixed rotations

```bash
bash integrations/vllm/phase_s1.sh start --root "$SR_S1_ROOT" --gate G3
bash integrations/vllm/phase_s1.sh status --root "$SR_S1_ROOT"
cat "$SR_S1_ROOT/G3/capacity-check.json"
tail -n 80 "$SR_S1_ROOT"/launcher-*.log
```

G3 first checks worst-case 100-request KV/sequence/query needs against both the
estimate and G2's actual Target/Draft allocated blocks. Insufficient capacity records
**BLOCKED** and starts no G3 engine. Do not reduce N or 512/1024 caps to get a PASS.
The fixed orders are Target→Serial→PingPong, Serial→PingPong→Target,
PingPong→Target→Serial. No parameter grid, legacy sweep or full main1000 run follows.

After completion:

```bash
cat "$SR_S1_ROOT/exit-code.json"
cat "$SR_S1_ROOT/G3/comparison.json"
bash integrations/vllm/phase_s1.sh compare --root "$SR_S1_ROOT" --gate G3
```

Require all nine runs valid and exact, plus repeatability of full tokens/termination.
Report raw metrics, median and population stdev, ratios of mode-level medians, actual
work, length/EOS/cap distributions, B/Q/progress/duration, acceptance and overlap.
Round differences and unavailable attribution remain visible. No positive gain is
required. Do not publish formal speedup from partial/OOM/divergent runs.

## Disconnect, failure cleanup and resume

For an ordinary disconnect, reconnect and restore `SR_S1_ROOT` to the **same existing
absolute root**, plus the repo/Python variables above. Do not generate a new root for
resume. `status` is safe while running; do not start another GPU task concurrently.
There is a kernel lock against concurrent S1 supervisors using this root.

```bash
cd /root/autodl-tmp/src/SpecRhythm
bash integrations/vllm/phase_s1.sh status --root "$SR_S1_ROOT"
cat "$SR_S1_ROOT/stage.json"
cat "$SR_S1_ROOT/exit-code.json"
```

If the launcher exited or was interrupted, resume the same gate. The helper first
verifies/cleans owned incomplete attempts, verifies every seal, skips completed valid
runs, and creates a new attempt for interrupted/failed work. It does not restart from
half-written JSON or lost live KV. Resolve a real runtime/config failure before retrying;
if code must change, begin a new S1 root and repeat its gates with that new frozen code.

```bash
bash integrations/vllm/phase_s1.sh resume --root "$SR_S1_ROOT" --gate G1
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
blocks further work and needs inspection. Raw coordinator and effective timeout/cleanup
return codes are retained separately; timeout is 4 hours per GPU child, Draft startup
timeout is 15 minutes. Timeout/failed attempts never enter performance aggregates.

## Verify and return a review bundle

```bash
bash integrations/vllm/phase_s1.sh verify --root "$SR_S1_ROOT" --directory "$SR_S1_ATTEMPT"
bash integrations/vllm/phase_s1.sh bundle --root "$SR_S1_ROOT"
```

`verify` applies to a completed sealed attempt (set `SR_S1_ATTEMPT` accordingly).
For a failed gate you can still bundle retained failure artifacts once owned processes
are gone. Bundle output prints archive path, SHA256, inventory path and file count.
Return both `<root>-review.tar.gz` and its printed SHA256. It includes G0 environment,
capacity and frozen subsets, every retained attempt/log/exit/lifecycle/result/seal,
gate comparisons and recovery records; see the [file contract](phase-s1-schema.md).
The inventory binds all included bytes. Archive/manifest/input files are never
rewritten to append an acceptance note. Bundle creation is exclusive: do it once,
then keep that archive unchanged. There is no S1 server bundle hash until this runs.

Stop after the S1 gate result and handoff; S2/S3 remain out of scope.
