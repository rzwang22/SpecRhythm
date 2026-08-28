# Phase 4B.1 real decode-only Dual-Batch correctness runbook

This is the active 3×A800 procedure after Phase 4B.0. It runs correctness only. It must not be
used to report TPOT, latency, throughput, goodput, SLO, speedup or overlap benefit. Stop on the
first nonzero command. Never reuse a run directory, delete an earlier failure, or run Gate 2/3
after an earlier gate fails. In particular, keep the earlier preparation root containing the
intermittent corrected-100 stock nondeterminism failure unchanged as diagnostic provenance. Use a
fresh `SR_PHASE4B_ROOT`; do not repeat that failed freeze in place or in new directories merely to
obtain a favorable pair. A later investigation must be a separately authorized diagnostic run,
not part of this gate procedure.

The real-A800 root ending in
`b9a0d6dc.../phase4b1-gate1-only-20260824T053635Z` is also immutable diagnostic provenance.
Its controlled-2 stock pair was deterministic and its three-layer patch application reached the
expected runner and scheduler hashes. Preparation then stopped because the helper mistakenly ran
a stock-state check against that patched installation. No Target, Serial or Dual consumer was
started, so this root is a control-plane preparation failure, not a Gate 1 correctness failure.
Do not reuse it.

The subsequent fresh root under commit `7e4f8711934fe10cb829e1947e62179c11a0209d`
is likewise immutable. It validates the explicit stock/patched state checks on A800 and contains a
successful resident Target run. Resident Serial then reached its first real speculative Target
verification and stopped in the observational hook because that shared hook accessed the
Dual-only `proposal_id` field on a Serial `Proposal`. Dual-1 and Dual-2 were not started and Gate 1
Outcome was not evaluated. Classify that root exactly as: preparation/control-plane PASS, Target
PASS, Serial diagnostics-compatibility FAIL, both Dual runs NOT RUN, Gate1 NOT EVALUATED.

The observer now records a canonical proposal ID only when the pending protocol actually provides
one. Serial proposals retain `proposal_id=null`; no synthetic Dual identity is created and the
Serial protocol is unchanged. The complete `3ee1c3e` root below is the fresh run that followed
that fix; the earlier partial roots remain provenance only.

The complete real-A800 root under `3ee1c3ec4007d3e835bc7d7f385d2d3b5c3c3e8a`
is also immutable. Preparation, Target, Serial diagnostics compatibility, both Dual executions,
the controlled scheduler construction, exact output triangle, keyed repeatability, state/proposal
lifecycle, accounting, verification, Draft sync, target blindness, measurement boundary and
cleanup all passed. Read-only diagnosis established a 57.989848 ms disjoint-request temporal
Draft/Verify overlap in Dual-1. Dual-2 has zero overlap and a 426.967646 ms separation, as expected
from its test-only two-ready coordination. The historical verification rows incorrectly attribute
both TP ranks to GPU1/the same UUID; the independently validated worker snapshots prove the actual
Target workers are GPU1 and GPU2. Never rewrite the old JSONL.

The remaining scheduler error was also instrumentation-only. `proposal_present=true` on the legal
tail means a consumed proposal remains in the scheduler's historical map; it does not mean live
speculative tokens remain. Current validation accepts this only when exact proposal lifecycle,
Draft target-tail readiness and terminal state evidence prove that the proposal was consumed
before the one-position tail. New events explicitly serialize `proposal_consumed`,
`live_proposal_present` and the tail readiness timestamp.

The gate structure is now explicit. Gate 1 uses `overlap-requirement=separate-gate`: its outcome
is controlled semantic correctness and its report preserves per-run temporal and
hardware-qualified overlap evidence. It does not require both controlled runs to overlap and does
not authorize an overlap-benefit claim. Gate 1.5/Gate 2 uses the default `required` mode with at
least five requests and must produce at least one positive disjoint-request GPU0 Draft/GPU1-2
Target witness under default asynchronous coordination. No sleep, model slowdown, scheduler-state
proxy or manufactured interval is allowed. Gate 3 remains blocked until both earlier gates pass.

The unified validator has an explicit legacy authority mode for this one immutable source commit.
It recomputes every semantic and runner-only invariant from raw evidence, records the embedded
historical verdict as provenance, supersedes only exact structurally proven obsolete errors and
fails on every remaining error. It also uses authoritative `worker_ranks` to supersede the known
per-verify aliasing bug while retaining `historical_event_instrumentation_invalid=true`. Invoke
this mode only with `--legacy-source-commit 3ee1c3ec4007d3e835bc7d7f385d2d3b5c3c3e8a`
and write all outputs outside the preserved tree.

The old overlap JSONL stores only actual intersections, so a zero row cannot identify the nearest
non-overlapping pair by itself. `phase4b1-overlap-diagnose` reads the immutable Draft-work,
verification and overlap files together and reports each run's exact nearest host intervals,
signed intersection, separation, ordering and physical placement. Write its output outside the
old root. It is diagnostic-only and never converts scheduler concurrency into physical overlap.

`phase4b1_restore_stock` now emits an immutable `check --expect-state stock` manifest, while
`phase4b1_apply_patch_stack` emits a distinct immutable `check --expect-state patched` manifest.
Each state accepts only its exact pinned runner/scheduler SHA pair; a partial or opposite state
fails closed. The legacy read-only section remains the immutable `3ee1c3e` closure procedure; it
is not part of the active Gate3 recovery and must keep its output outside the old tree.

## Active Gate3 one-shot numerical localization

The `32b09a6` corrected-100 Target recovery passed chunked setup, all 100 bootstrap observations,
global readiness, first-forward contracts, TP2 placement, measurement boundaries and cleanup. It
failed exact output equality for exactly four requests. Preserve that entire recovery root and do
not run Serial or Dual.

The first diagnostic launch at `c142fa7` restored and applied its four-layer patch stack, then
crashed both stock TP workers before a valid checkpoint because the observer read
speculative-only common attention metadata while `speculative_config=None`. Resident and the
comparator were not run. Preserve this exact root as `diagnostic-infrastructure-failed` evidence:

```text
/root/autodl-tmp/SpecRhythm-data/results/phase4/c142fa7adbbdf0d81cc02d9244a3be75d4b9d7e7/phase4b1-gate3-numerical-20260828T033035Z
```

It does not consume the scientific comparison because no valid stock numerical artifact exists.
The next action is exactly one fresh diagnostic-only full-shape pair, never a retry loop:

1. one patched-observational stock-style run with `speculative_config=None` and the stock
   scheduler;
2. one resident Target run, expected to remain nonzero if the exact four divergences reproduce;
3. one offline exact comparison.

The stock-style diagnostic is not a new stock correctness reference and is marked
`reference_freeze_eligible=false`. A four-request replay is forbidden because it changes the
chunked-prefill, decode-cohort and LM-head shapes. The fourth patch is a diagnostic-only call
immediately before the existing forward and is inert without both an explicit plan and output
path. Exact token equality remains mandatory.

Run from the exact reviewed commit supplied with the handoff. Every state transition and command
has an explicit return-code gate; helper correctness must not depend on shell `errexit`. Resolve,
but never write into, the preserved `32b09a6` and `c142fa7` directories:

```bash
cd /root/autodl-tmp/src/SpecRhythm || exit 1
git fetch origin codex/vllm-serving-v0.1 || exit 1
git switch --detach origin/codex/vllm-serving-v0.1 || exit 1
SR_PHASE4B_COMMIT="$(git rev-parse HEAD)" || exit 1
export SR_PHASE4B_COMMIT
test -z "$(git status --porcelain)" || exit 1

conda activate /root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1 || exit 1
python -m pip install -e '.[dev]' --no-deps || exit 1

export SR_DRAFT_MODEL="/root/autodl-tmp/models/Qwen3-0.6B"
export SR_TARGET_MODEL="/root/autodl-tmp/models/Qwen3-32B"
export SR_VLLM_SOURCE="/root/autodl-tmp/src/vllm-v0.25.1"
export SR_PHASE4B_CONFIG="$PWD/configs/phase4b_dual_batch_1d2v.yaml"
export SR_NUMERICAL_PLAN="$PWD/configs/phase4b1_gate3_numerical_diagnostic.json"
SR_VLLM_ROOT="$(python - <<'PY'
from importlib import metadata
print(metadata.distribution("vllm").locate_file(""))
PY
)" || exit 1
export SR_VLLM_ROOT

export SR_INPUT_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/eba0df493a7fd350ef3c8776e06d30e6196b6749/phase4b1-gate2-corrected5-20260827T040244Z"
export SR_GATE3_WORKLOAD="$SR_INPUT_ROOT/workloads/corrected-100.jsonl"
export SR_GATE3_REFERENCE="$SR_INPUT_ROOT/Gate-3-corrected-100/reference/stock-target-reference.json"
export SR_PHASE4B_ENVIRONMENT="$SR_INPUT_ROOT/environment.json"
export SR_PHASE4B_TOPOLOGY="$SR_INPUT_ROOT/topology.json"

readarray -t SR_PRESERVED_MATCHES < <(
  find "/root/autodl-tmp/SpecRhythm-data/results/phase4/32b09a6749dc44200fffe37411002d862ca1098a" \
    -type d -name 'Gate-3-corrected-100' -print | sort
)
test "${#SR_PRESERVED_MATCHES[@]}" -eq 1 || exit 1
export SR_PRESERVED_GATE3="${SR_PRESERVED_MATCHES[0]}"
test -f "$SR_PRESERVED_GATE3/target/resident-target.json" || exit 1

export SR_FAILED_C142_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/c142fa7adbbdf0d81cc02d9244a3be75d4b9d7e7/phase4b1-gate3-numerical-20260828T033035Z"
test -d "$SR_FAILED_C142_ROOT" || exit 1

export SR_DIAG_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/$SR_PHASE4B_COMMIT/phase4b1-gate3-numerical-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$SR_DIAG_ROOT" || exit 1
mkdir -p "$SR_DIAG_ROOT/stock-style" || exit 1

find "$SR_PRESERVED_GATE3" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$SR_DIAG_ROOT/preserved-gate3-sha256.txt" || exit 1
find "$SR_FAILED_C142_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$SR_DIAG_ROOT/preserved-c142-failure-sha256.txt" || exit 1
nvidia-smi -L | tee "$SR_DIAG_ROOT/nvidia-smi-L.txt" || exit 1
nvidia-smi topo -m | tee "$SR_DIAG_ROOT/nvidia-smi-topo.txt" || exit 1

source integrations/vllm/phase4b_run_helpers.sh || exit 1
source integrations/vllm/phase4b1_gate_helpers.sh || exit 1
phase4b1_restore_stock "$SR_DIAG_ROOT/patch-stage" || exit 1
phase4b1_apply_patch_stack "$SR_DIAG_ROOT/patch-stage" || exit 1

python integrations/vllm/manage_patch.py check \
  --expect-state patched \
  --vllm-root "$SR_VLLM_ROOT" \
  --source "$SR_VLLM_SOURCE" \
  --manifest "$SR_DIAG_ROOT/patch-stage/patched-state-confirmation.json" || exit 1
```

The restore accepts the exact retired `c142fa7` runner only as a strict restoration input. The
patched-state check accepts only the new generic-ownership runner hash; it never treats old and new
instrumentation as interchangeable.

Run the stock-style execution exactly once. It uses the patched runner only for observation; no
custom proposer or resident scheduler is configured:

```bash
CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
specrhythm phase4-stock-smoke \
  --config "$SR_PHASE4B_CONFIG" \
  --role target \
  --workload "$SR_GATE3_WORKLOAD" \
  --request-count 100 \
  --environment "$SR_PHASE4B_ENVIRONMENT" \
  --topology "$SR_PHASE4B_TOPOLOGY" \
  --runtime-manifest "$SR_DIAG_ROOT/stock-style/runtime-manifest.json" \
  --correctness-mode batch-invariant \
  --target-diagnostics "$SR_DIAG_ROOT/stock-style/target-diagnostics.jsonl" \
  --diagnostic-single-run \
  --numerical-diagnostic-plan "$SR_NUMERICAL_PLAN" \
  --numerical-diagnostic-output "$SR_DIAG_ROOT/stock-style/numerical.jsonl" \
  --output "$SR_DIAG_ROOT/stock-style/stock-style.json"
SR_STOCK_STATUS="$?"
test "$SR_STOCK_STATUS" -eq 0 || exit "$SR_STOCK_STATUS"

python - "$SR_DIAG_ROOT/stock-style/stock-style.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["diagnostic_only"] is True
assert value["reference_freeze_eligible"] is False
assert value["stock_reference_replaced"] is False
assert len(value["runs"]) == 1
assert value["numerical_diagnostics"]["valid"] is True
assert value["numerical_diagnostics"]["record_count"] == 4
PY
test "$?" -eq 0 || exit 1
```

Run resident Target exactly once. Exit status `1` is expected only because exact stock equality
remains failed; any other status or invalid diagnostics/cleanup stops the experiment:

```bash
PHASE4B1_NUMERICAL_PLAN="$SR_NUMERICAL_PLAN" \
PHASE4B1_NUMERICAL_OUTPUT="$SR_DIAG_ROOT/resident-target/numerical.jsonl" \
phase4b1_run_mode target \
  "$SR_DIAG_ROOT/resident-target" "$SR_GATE3_WORKLOAD" 100 "$SR_GATE3_REFERENCE"
SR_RESIDENT_STATUS="$?"
test "$SR_RESIDENT_STATUS" -eq 1 || exit 1

python - \
  "$SR_DIAG_ROOT/resident-target/resident-target.json" \
  "$SR_DIAG_ROOT/resident-target/process-lifecycle.json" <<'PY'
import json
import sys
run = json.load(open(sys.argv[1], encoding="utf-8"))
lifecycle = json.load(open(sys.argv[2], encoding="utf-8"))
assert run["valid"] is False
assert run["numerical_diagnostics"]["valid"] is True
assert run["numerical_diagnostics"]["record_count"] == 4
assert lifecycle["cleanup_valid"] is True
divergent = {
    (row["request_id"], row["first_divergence_position"])
    for row in run["stock_comparison"]["requests"] if not row["equal"]
}
assert divergent == {
    ("r3-c7ee1a73ee79dd6dc21cb8dc", 3),
    ("r3-32ae44a69fffd76f0dd4b787", 4),
    ("r3-646c340a0281105c1c20de27", 12),
    ("r3-e00f5312321ec537a9c716cd", 2),
}
PY
test "$?" -eq 0 || exit 1
```

Perform the read-only exact comparison and freeze checksums:

```bash
specrhythm phase4b1-gate3-numerical-compare \
  --plan "$SR_NUMERICAL_PLAN" \
  --workload "$SR_GATE3_WORKLOAD" \
  --stock-run "$SR_DIAG_ROOT/stock-style/stock-style.json" \
  --stock-numerical "$SR_DIAG_ROOT/stock-style/numerical.jsonl" \
  --resident-run "$SR_DIAG_ROOT/resident-target/resident-target.json" \
  --resident-numerical "$SR_DIAG_ROOT/resident-target/numerical.jsonl" \
  --output "$SR_DIAG_ROOT/numerical-comparison.json" \
  --markdown-output "$SR_DIAG_ROOT/numerical-comparison.md"
SR_COMPARATOR_STATUS="$?"
test "$SR_COMPARATOR_STATUS" -eq 0 || exit "$SR_COMPARATOR_STATUS"

find "$SR_DIAG_ROOT" -type f ! -name all-artifacts-sha256.txt -print0 \
  | sort -z | xargs -0 sha256sum > "$SR_DIAG_ROOT/all-artifacts-sha256.txt" || exit 1
cat "$SR_DIAG_ROOT/numerical-comparison.md" || exit 1
echo "$SR_DIAG_ROOT"
```

Stop here and return the immutable directory. Do not run Serial, Dual, performance, Dual-Eager,
Phase4B.2 or any retry if the four checkpoints do not reproduce.

## Historical Gate3 corrected-100 recovery after chunked-prefill setup failure

The first corrected-100 attempt under `eba0df4` is immutable at:

```text
/root/autodl-tmp/SpecRhythm-data/results/phase4/eba0df493a7fd350ef3c8776e06d30e6196b6749/phase4b1-gate2-corrected5-20260827T040244Z/Gate-3-corrected-100
```

Its preparation, 60/20/20 workload, single allowed deterministic stock pair and patch application
passed. Resident Target then failed before global setup because the proposer treated every
callback row as bootstrap-ready although chunked prefill had split the 100-request cohort. Serial
and Dual were not run and Gate3 was not evaluated. Never alter or reuse that directory.

At pinned vLLM `752a3a5`, the CPU custom proposer is called after `_bookkeeping_sync`.
`sampled_token_ids[row]` therefore proves whether that forward sampled a bootstrap;
`num_tokens_no_spec/token_ids_cpu` is the post-bookkeeping logical row, not a Target-KV
materialization count. The minimal active patch additionally passes
`num_computed_tokens_cpu + scheduler_output.num_scheduled_tokens` for each request. The shared
classifier uses those facts for `partial-prefill`, `full-prompt-no-bootstrap` and
`bootstrap-ready`; a second output before global readiness remains fatal.

Run the following only after the new commit and CI are reviewed. It reuses the exact stock-100
reference byte-for-byte and does not run another stock pair. The consumer still validates the full
stock/model/tokenizer/sampling/runtime/workload contract before model creation.

```bash
set -euo pipefail
cd /root/autodl-tmp/src/SpecRhythm
git fetch origin codex/vllm-serving-v0.1
git switch --detach origin/codex/vllm-serving-v0.1
export SR_PHASE4B_COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --short)"

conda activate /root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1
python -m pip install -e '.[dev]' --no-deps

export SR_DRAFT_MODEL="/root/autodl-tmp/models/Qwen3-0.6B"
export SR_TARGET_MODEL="/root/autodl-tmp/models/Qwen3-32B"
export SR_VLLM_SOURCE="/root/autodl-tmp/src/vllm-v0.25.1"
export SR_PHASE4B_CONFIG="$PWD/configs/phase4b_dual_batch_1d2v.yaml"
export SR_VLLM_ROOT="$(python - <<'PY'
from importlib import metadata
print(metadata.distribution("vllm").locate_file(""))
PY
)"

export SR_FAILED_PHASE4B_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/eba0df493a7fd350ef3c8776e06d30e6196b6749/phase4b1-gate2-corrected5-20260827T040244Z"
export SR_FAILED_GATE3="$SR_FAILED_PHASE4B_ROOT/Gate-3-corrected-100"
export SR_SOURCE_REFERENCE="$SR_FAILED_GATE3/reference/stock-target-reference.json"
export SR_GATE3_WORKLOAD="$SR_FAILED_PHASE4B_ROOT/workloads/corrected-100.jsonl"
export SR_PHASE4B_ENVIRONMENT="$SR_FAILED_PHASE4B_ROOT/environment.json"
export SR_PHASE4B_TOPOLOGY="$SR_FAILED_PHASE4B_ROOT/topology.json"

test -f "$SR_SOURCE_REFERENCE"
test -f "$SR_GATE3_WORKLOAD"
test "$(wc -l < "$SR_GATE3_WORKLOAD")" -eq 100
test -f "$SR_FAILED_PHASE4B_ROOT/Gate-1-controlled-2/validation.json"
test -f "$SR_FAILED_PHASE4B_ROOT/Gate-2-corrected-5/validation.json"

python - \
  "$SR_FAILED_PHASE4B_ROOT/Gate-1-controlled-2/validation.json" \
  "$SR_FAILED_PHASE4B_ROOT/Gate-2-corrected-5/validation.json" <<'PY'
import json
import sys
for path in sys.argv[1:]:
    value = json.load(open(path, encoding="utf-8"))
    assert value["valid"] is True
    assert value["outcome"] == "A"
    assert value["errors"] == []
PY

export SR_PHASE4B_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/$SR_PHASE4B_COMMIT/phase4b1-gate3-recovery-$(date -u +%Y%m%dT%H%M%SZ)"
export SR_GATE3="$SR_PHASE4B_ROOT/Gate-3-corrected-100"
export SR_GATE3_REFERENCE="$SR_GATE3/reference/stock-target-reference.json"
test ! -e "$SR_PHASE4B_ROOT"
mkdir -p "$SR_PHASE4B_ROOT"

find "$SR_FAILED_GATE3" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$SR_PHASE4B_ROOT/failed-gate3-input-sha256.txt"
nvidia-smi -L | tee "$SR_PHASE4B_ROOT/nvidia-smi-L.txt"
nvidia-smi topo -m | tee "$SR_PHASE4B_ROOT/nvidia-smi-topo.txt"

source integrations/vllm/phase4b_run_helpers.sh
source integrations/vllm/phase4b1_gate_helpers.sh

phase4b1_restore_stock "$SR_GATE3/stock-stage"
phase4b1_reuse_stock_reference \
  "$SR_SOURCE_REFERENCE" "$SR_GATE3/reference" "$SR_GATE3_WORKLOAD"

python - \
  "$SR_PHASE4B_CONFIG" "$SR_GATE3_REFERENCE" "$SR_GATE3_WORKLOAD" <<'PY'
import pathlib
import sys
from specrhythm.phase4.config import load_phase4_config
from specrhythm.phase4.reference import (
    load_reference,
    require_exact_resident_reference_reuse,
)
config = load_phase4_config(sys.argv[1])
reference = load_reference(pathlib.Path(sys.argv[2]))
require_exact_resident_reference_reuse(reference, config, pathlib.Path(sys.argv[3]))
PY

phase4b1_apply_patch_stack "$SR_GATE3/stock-stage"
```

Run each consumer in order and stop immediately on the first nonzero result:

```bash
phase4b1_run_mode target \
  "$SR_GATE3/target" "$SR_GATE3_WORKLOAD" 100 "$SR_GATE3_REFERENCE"

python - "$SR_GATE3/target/resident-target.json" \
  "$SR_GATE3/target/process-lifecycle.json" \
  "$SR_GATE3/target/timing-events.jsonl" <<'PY'
import json
import sys
run = json.load(open(sys.argv[1], encoding="utf-8"))
lifecycle = json.load(open(sys.argv[2], encoding="utf-8"))
rows = [json.loads(line) for line in open(sys.argv[3], encoding="utf-8") if line.strip()]
setup = [row for row in rows if row.get("event") == "setup-row-classified"]
partial = [row for row in setup if row.get("setup_stage") == "partial-prefill"]
ready = [row for row in setup if row.get("setup_stage") == "bootstrap-ready"]
assert run["valid"] is True and run["errors"] == []
assert lifecycle["cleanup_valid"] is True
assert len({row["request_id"] for row in ready}) == 100
assert partial
assert any(
    early["timestamp_ns"] < later["timestamp_ns"]
    and early["internal_target_request_id"] != later["internal_target_request_id"]
    for early in ready for later in partial
)
PY

phase4b1_run_mode serial \
  "$SR_GATE3/serial" "$SR_GATE3_WORKLOAD" 100 "$SR_GATE3_REFERENCE"
python - "$SR_GATE3/serial/resident-serial.json" \
  "$SR_GATE3/serial/process-lifecycle.json" <<'PY'
import json
import sys
run = json.load(open(sys.argv[1], encoding="utf-8"))
lifecycle = json.load(open(sys.argv[2], encoding="utf-8"))
assert run["valid"] is True and run["errors"] == []
assert lifecycle["cleanup_valid"] is True
PY

PHASE4B1_OVERLAP_REQUIREMENT=required \
phase4b1_run_mode dual \
  "$SR_GATE3/dual-1" "$SR_GATE3_WORKLOAD" 100 "$SR_GATE3_REFERENCE" none

PHASE4B1_OVERLAP_REQUIREMENT=required \
phase4b1_validate_gate "$SR_GATE3" "$SR_GATE3/dual-1"
phase4b1_require_outcome_a "$SR_GATE3/validation.json"

find "$SR_PHASE4B_ROOT" -type f ! -name 'all-artifacts-sha256.txt' -print0 \
  | sort -z | xargs -0 sha256sum > "$SR_PHASE4B_ROOT/all-artifacts-sha256.txt"
cat "$SR_GATE3/reference/stock-reference-reuse.json"
cat "$SR_GATE3/validation.json"
echo "$SR_PHASE4B_ROOT"
```

Stop after this recovery. Do not start performance, Dual-Eager, Gate4 or Phase4B.2.

## 0. Close the preserved Gate 1 by read-only revalidation

This section starts no model process and performs no CUDA forward. Resolve exactly one preserved
Gate directory, create a new analysis directory under the current code commit, then run both the
timing diagnostic and the explicit legacy authority mode.

```bash
set -euo pipefail
cd /root/autodl-tmp/src/SpecRhythm
git fetch origin codex/vllm-serving-v0.1
git switch --detach origin/codex/vllm-serving-v0.1
export SR_REVALIDATOR_COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --short)"

conda activate /root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1
python -m pip install -e '.[dev]' --no-deps

export SR_LEGACY_COMMIT="3ee1c3ec4007d3e835bc7d7f385d2d3b5c3c3e8a"
export SR_LEGACY_BASE="/root/autodl-tmp/SpecRhythm-data/results/phase4/$SR_LEGACY_COMMIT"
mapfile -t SR_GATE_MATCHES < <(
  find "$SR_LEGACY_BASE" -type d -name 'Gate-1-controlled-2' -print | sort
)
printf '%s\n' "${SR_GATE_MATCHES[@]}"
test "${#SR_GATE_MATCHES[@]}" -eq 1
export SR_LEGACY_GATE1="${SR_GATE_MATCHES[0]}"
export SR_READ_ONLY_OUT="/root/autodl-tmp/SpecRhythm-data/results/phase4/$SR_REVALIDATOR_COMMIT/read-only-3ee1c3e-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$SR_READ_ONLY_OUT"
mkdir -p "$SR_READ_ONLY_OUT"

specrhythm phase4b1-overlap-diagnose \
  --draft-work-events "$SR_LEGACY_GATE1/dual-1/draft-work-events.jsonl" \
  --draft-work-events "$SR_LEGACY_GATE1/dual-2/draft-work-events.jsonl" \
  --verification-events "$SR_LEGACY_GATE1/dual-1/verification-events.jsonl" \
  --verification-events "$SR_LEGACY_GATE1/dual-2/verification-events.jsonl" \
  --overlap-events "$SR_LEGACY_GATE1/dual-1/overlap-events.jsonl" \
  --overlap-events "$SR_LEGACY_GATE1/dual-2/overlap-events.jsonl" \
  --output "$SR_READ_ONLY_OUT/overlap-diagnosis.json"
```

Build the validator arguments without using `phase4b1_validate_gate`, because that helper writes
inside its gate root:

```bash
SR_VALIDATE=(
  specrhythm phase4b1-dual-correctness-validate
  --target "$SR_LEGACY_GATE1/target/resident-target.json"
  --serial "$SR_LEGACY_GATE1/serial/resident-serial.json"
  --target-manifest "$SR_LEGACY_GATE1/target/decode-ready-manifest.json"
  --serial-manifest "$SR_LEGACY_GATE1/serial/decode-ready-manifest.json"
  --target-process-lifecycle "$SR_LEGACY_GATE1/target/process-lifecycle.json"
  --serial-process-lifecycle "$SR_LEGACY_GATE1/serial/process-lifecycle.json"
)
for SR_DUAL in "$SR_LEGACY_GATE1/dual-1" "$SR_LEGACY_GATE1/dual-2"; do
  SR_VALIDATE+=(
    --dual "$SR_DUAL/resident-dual.json"
    --dual-manifest "$SR_DUAL/decode-ready-manifest.json"
    --request-state-events "$SR_DUAL/request-state-events.jsonl"
    --proposal-events "$SR_DUAL/proposal-events.jsonl"
    --proposal-lifecycle-events "$SR_DUAL/proposal-lifecycle-events.jsonl"
    --scheduler-events "$SR_DUAL/scheduler-events.jsonl"
    --verification-events "$SR_DUAL/verification-events.jsonl"
    --draft-work-events "$SR_DUAL/draft-work-events.jsonl"
    --target-diagnostics "$SR_DUAL/target-diagnostics.jsonl"
    --overlap-events "$SR_DUAL/overlap-events.jsonl"
    --process-lifecycle "$SR_DUAL/process-lifecycle.json"
  )
done
"${SR_VALIDATE[@]}" \
  --overlap-requirement separate-gate \
  --legacy-source-commit "$SR_LEGACY_COMMIT" \
  --output "$SR_READ_ONLY_OUT/revalidation.json" \
  --markdown-output "$SR_READ_ONLY_OUT/revalidation.md"

python - "$SR_READ_ONLY_OUT/revalidation.json" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["valid"] is True and value["outcome"] == "A"
assert value["input_artifacts_immutable"] is True
assert value["triangle"]["valid"] is True
assert value["overlap_gate"]["temporal_observed_per_run"] == [True, False]
assert value["overlap_gate"]["hardware_qualified_observed_per_run"] == [True, False]
assert value["overlap_gate"]["claim_permitted"] is False
for run in value["dual_runs"]:
    assert run["recomputed_semantic_valid"] is True
    authority = run["embedded_verdict_authority"]
    assert authority["remaining_embedded_errors"] == []
PY
```

Stop after this section and return `revalidation.json` and `overlap-diagnosis.json` for review.
Do not continue to Gate 1.5, Gate 2 or Gate 3.

## 1. Exact checkout and environment

```bash
set -euo pipefail
cd /root/autodl-tmp/src/SpecRhythm
git fetch origin codex/vllm-serving-v0.1
git switch --detach origin/codex/vllm-serving-v0.1
export SR_PHASE4B_COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --short)"

conda activate /root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1
python --version
python -m pip install -e '.[dev]' --no-deps

export SR_DRAFT_MODEL="/root/autodl-tmp/models/Qwen3-0.6B"
export SR_TARGET_MODEL="/root/autodl-tmp/models/Qwen3-32B"
export SR_VLLM_SOURCE="/root/autodl-tmp/src/vllm-v0.25.1"
export SR_PHASE3C_COMMIT="34c7ea9836c2595c8a8aeaeb5680709520edd3d8"
export SR_R3_100="/root/autodl-tmp/SpecRhythm-data/results/phase3c/$SR_PHASE3C_COMMIT/corrected-multiround-100/workload.jsonl"
export SR_PHASE4B_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/$SR_PHASE4B_COMMIT/phase4b1-gate1-only-$(date -u +%Y%m%dT%H%M%SZ)"
export SR_PHASE4B_CONFIG="$PWD/configs/phase4b_dual_batch_1d2v.yaml"
export SR_VLLM_ROOT="$(python - <<'PY'
from importlib import metadata
print(metadata.distribution("vllm").locate_file(""))
PY
)"

test -f "$SR_DRAFT_MODEL/config.json"
test -f "$SR_TARGET_MODEL/config.json"
test -f "$SR_R3_100"
test "$(wc -l < "$SR_R3_100")" -eq 100
test ! -e "$SR_PHASE4B_ROOT"
mkdir -p "$SR_PHASE4B_ROOT/workloads"
nvidia-smi -L | tee "$SR_PHASE4B_ROOT/nvidia-smi-L.txt"
nvidia-smi topo -m | tee "$SR_PHASE4B_ROOT/nvidia-smi-topo.txt"

source integrations/vllm/phase4b_run_helpers.sh
source integrations/vllm/phase4b1_gate_helpers.sh

specrhythm phase4-dual-contract-dry-run \
  --output "$SR_PHASE4B_ROOT/dual-contract-dry-run.json"
specrhythm phase4-decode-ready-contract-dry-run \
  --output "$SR_PHASE4B_ROOT/decode-ready-contract-dry-run.json"
```

Create one controlled two-request workload, the corrected 3/1/1 five-request workload and the
unchanged corrected 60/20/20 100-request workload. The controlled requests have different output
limits so one terminates while the other remains live.

```bash
python - "$SR_R3_100" "$SR_PHASE4B_ROOT/workloads" <<'PY'
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2])
rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
assert len(rows) == 100
assert {task: sum(row["task_class"] == task for row in rows) for task in (
    "code", "chat", "summarization"
)} == {"code": 60, "chat": 20, "summarization": 20}

two = [
    dict(next(row for row in rows if row["task_class"] == "code")),
    dict(next(row for row in rows if row["task_class"] == "chat")),
]
two[0]["maximum_new_tokens"] = 4
two[1]["maximum_new_tokens"] = 12
needed = {"code": 3, "chat": 1, "summarization": 1}
five = []
for row in rows:
    task = row["task_class"]
    if needed.get(task, 0):
        five.append(row)
        needed[task] -= 1
    if not any(needed.values()):
        break

for name, selected in (
    ("controlled-2.jsonl", two),
    ("corrected-5.jsonl", five),
    ("corrected-100.jsonl", rows),
):
    path = destination / name
    with path.open("x", encoding="utf-8") as handle:
        for row in selected:
            assert row["prompt_length"] == len(row["prompt_token_ids"])
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
PY
find "$SR_PHASE4B_ROOT/workloads" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$SR_PHASE4B_ROOT/workload-sha256.txt"
```

## 2. Probe stock vLLM only

This section does not freeze any Gate reference and does not apply the patch stack. References are
measured only immediately before their dependent gate. A failed two-run pair always leaves an
immutable `stock-determinism-diagnostic.json` containing both raw runs and exact per-request first
divergences; it never creates a reference and must not be followed by another freeze attempt in
this gate procedure.

```bash
git -C "$SR_VLLM_SOURCE" checkout --detach \
  752a3a504485790a2e8491cacbb35c137339ad34
test "$(git -C "$SR_VLLM_SOURCE" rev-parse HEAD)" = \
  "752a3a504485790a2e8491cacbb35c137339ad34"

phase4b1_restore_stock "$SR_PHASE4B_ROOT/preparation-stock-probe"

env -u CUDA_VISIBLE_DEVICES VLLM_BATCH_INVARIANT=1 specrhythm phase4-probe \
  --config "$SR_PHASE4B_CONFIG" --vllm-source "$SR_VLLM_SOURCE" \
  --environment-output "$SR_PHASE4B_ROOT/environment.json" \
  --topology-output "$SR_PHASE4B_ROOT/topology.json" \
  --validation-output "$SR_PHASE4B_ROOT/probe-validation.json"
export SR_PHASE4B_ENVIRONMENT="$SR_PHASE4B_ROOT/environment.json"
export SR_PHASE4B_TOPOLOGY="$SR_PHASE4B_ROOT/topology.json"

CUDA_VISIBLE_DEVICES=1,2 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
specrhythm phase4-batch-invariant-preflight \
  --correctness-mode batch-invariant \
  --output "$SR_PHASE4B_ROOT/batch-invariant-preflight.json"

```

## 3. Gate 1: two-request controlled correctness

Dual-1 uses the explicitly test-only, non-blocking `one-ready` publication limit to construct
Case A. Dual-2 uses the test-only metadata gate to construct Case B. Neither runs Draft work in
the Target scheduler; both switches are recorded and are not production policy or performance
results. The different output limits construct Case C.

```bash
export SR_GATE1="$SR_PHASE4B_ROOT/Gate-1-controlled-2"
export SR_GATE1_WORKLOAD="$SR_PHASE4B_ROOT/workloads/controlled-2.jsonl"
export SR_GATE1_REFERENCE="$SR_GATE1/reference/stock-target-reference.json"

phase4b1_restore_stock "$SR_GATE1/stock-stage"
phase4b1_freeze_stock_reference "$SR_GATE1/reference" "$SR_GATE1_WORKLOAD" 2
phase4b1_apply_patch_stack "$SR_GATE1/stock-stage"

phase4b1_run_mode target "$SR_GATE1/target" "$SR_GATE1_WORKLOAD" 2 "$SR_GATE1_REFERENCE"
phase4b1_run_mode serial "$SR_GATE1/serial" "$SR_GATE1_WORKLOAD" 2 "$SR_GATE1_REFERENCE"
PHASE4B1_OVERLAP_REQUIREMENT=separate-gate \
phase4b1_run_mode dual "$SR_GATE1/dual-1" "$SR_GATE1_WORKLOAD" 2 \
  "$SR_GATE1_REFERENCE" one-ready
PHASE4B1_OVERLAP_REQUIREMENT=separate-gate \
phase4b1_run_mode dual "$SR_GATE1/dual-2" "$SR_GATE1_WORKLOAD" 2 \
  "$SR_GATE1_REFERENCE" two-ready

specrhythm phase4b1-dual-controlled-validate \
  --asynchronous-scheduler "$SR_GATE1/dual-1/scheduler-events.jsonl" \
  --coordinated-scheduler "$SR_GATE1/dual-2/scheduler-events.jsonl" \
  --request-state-events "$SR_GATE1/dual-1/request-state-events.jsonl" \
  --output "$SR_GATE1/controlled-validation.json"
PHASE4B1_OVERLAP_REQUIREMENT=separate-gate \
phase4b1_validate_gate "$SR_GATE1" "$SR_GATE1/dual-1" "$SR_GATE1/dual-2"
phase4b1_require_outcome_a "$SR_GATE1/validation.json"

python - "$SR_GATE1/validation.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["gate_profile"] == "controlled-correctness"
assert value["overlap_requirement"] == "separate-gate"
assert value["overlap_gate"]["required_for_validation"] is False
assert value["overlap_gate"]["claim_permitted"] is False
PY
```

If any command above fails, preserve `$SR_GATE1` and stop. In particular, a nondeterministic stock
pair is a Gate 1 preparation failure even though its diagnostic artifact was written.
For the current recovery run, also stop after successful `phase4b1_require_outcome_a`; do not run
Sections 4–6. Record a Gate1-only checksum manifest with:

```bash
find "$SR_PHASE4B_ROOT" -type f ! -name 'gate1-artifacts-sha256.txt' -print0 \
  | sort -z | xargs -0 sha256sum > "$SR_PHASE4B_ROOT/gate1-artifacts-sha256.txt"
cat "$SR_GATE1/stock-stage/vllm-stock-check.json"
cat "$SR_GATE1/stock-stage/vllm-patched-check.json"
cat "$SR_GATE1/validation.json"
```

## 4. Gate 1.5 / Gate 2: corrected five-request positive-overlap existence

This gate is not authorized by a Gate 1 correctness result alone. Run it only after review of the
Gate 1 artifacts. The unchanged asynchronous production path must naturally create Draft backlog;
physical overlap remains mandatory and its absence returns nonzero.

```bash
export SR_GATE2="$SR_PHASE4B_ROOT/Gate-2-corrected-5"
export SR_GATE2_WORKLOAD="$SR_PHASE4B_ROOT/workloads/corrected-5.jsonl"
export SR_GATE2_REFERENCE="$SR_GATE2/reference/stock-target-reference.json"

phase4b1_require_outcome_a "$SR_GATE1/validation.json"
phase4b1_restore_stock "$SR_GATE2/stock-stage"
phase4b1_freeze_stock_reference "$SR_GATE2/reference" "$SR_GATE2_WORKLOAD" 5
phase4b1_apply_patch_stack "$SR_GATE2/stock-stage"

phase4b1_run_mode target "$SR_GATE2/target" "$SR_GATE2_WORKLOAD" 5 "$SR_GATE2_REFERENCE"
phase4b1_run_mode serial "$SR_GATE2/serial" "$SR_GATE2_WORKLOAD" 5 "$SR_GATE2_REFERENCE"
PHASE4B1_OVERLAP_REQUIREMENT=required \
phase4b1_run_mode dual "$SR_GATE2/dual-1" "$SR_GATE2_WORKLOAD" 5 \
  "$SR_GATE2_REFERENCE" none
PHASE4B1_OVERLAP_REQUIREMENT=required \
phase4b1_run_mode dual "$SR_GATE2/dual-2" "$SR_GATE2_WORKLOAD" 5 \
  "$SR_GATE2_REFERENCE" none
PHASE4B1_OVERLAP_REQUIREMENT=required \
phase4b1_validate_gate "$SR_GATE2" "$SR_GATE2/dual-1" "$SR_GATE2/dual-2"
phase4b1_require_outcome_a "$SR_GATE2/validation.json"
```

If any command above fails, preserve `$SR_GATE2` and stop. Gate 1 remains valid and untouched.

## 5. Gate 3: corrected R3-real 100 requests (60/20/20)

Gate 3 is permitted only after Gate 1 and Gate 2 both contain `valid=true`, `outcome=A` and an
empty error list. A single 100-request Dual run is not repeatability evidence; repeatability comes
only from Gate 1 and Gate 2 unless a second 100-request run is explicitly added later.

```bash
python - "$SR_GATE1/validation.json" "$SR_GATE2/validation.json" <<'PY'
import json
import sys
for path in sys.argv[1:]:
    value = json.load(open(path, encoding="utf-8"))
    assert value["valid"] is True and value["outcome"] == "A" and value["errors"] == []
PY

export SR_GATE3="$SR_PHASE4B_ROOT/Gate-3-corrected-100"
export SR_GATE3_WORKLOAD="$SR_PHASE4B_ROOT/workloads/corrected-100.jsonl"
export SR_GATE3_REFERENCE="$SR_GATE3/reference/stock-target-reference.json"

phase4b1_restore_stock "$SR_GATE3/stock-stage"
phase4b1_freeze_stock_reference "$SR_GATE3/reference" "$SR_GATE3_WORKLOAD" 100
phase4b1_apply_patch_stack "$SR_GATE3/stock-stage"

phase4b1_run_mode target "$SR_GATE3/target" "$SR_GATE3_WORKLOAD" 100 "$SR_GATE3_REFERENCE"
phase4b1_run_mode serial "$SR_GATE3/serial" "$SR_GATE3_WORKLOAD" 100 "$SR_GATE3_REFERENCE"
phase4b1_run_mode dual "$SR_GATE3/dual-1" "$SR_GATE3_WORKLOAD" 100 "$SR_GATE3_REFERENCE" none
phase4b1_validate_gate "$SR_GATE3" "$SR_GATE3/dual-1"
phase4b1_require_outcome_a "$SR_GATE3/validation.json"
```

The corrected-100 freeze executes exactly one pair. If it returns nonzero, inspect
`$SR_GATE3/reference/stock-determinism-diagnostic.json`, preserve the complete Gate 3 directory,
and stop. Do not run another freeze to obtain a favorable pair and do not run Gate 3 consumers.

## 6. Final immutability manifest and expected artifacts

```bash
find "$SR_PHASE4B_ROOT" -type f ! -name 'all-artifacts-sha256.txt' -print0 \
  | sort -z | xargs -0 sha256sum > "$SR_PHASE4B_ROOT/all-artifacts-sha256.txt"
python - "$SR_PHASE4B_ROOT" <<'PY'
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
for gate in ("Gate-1-controlled-2", "Gate-2-corrected-5", "Gate-3-corrected-100"):
    value = json.loads((root / gate / "validation.json").read_text())
    assert value["valid"] is True
    assert value["outcome"] == "A"
    assert value["errors"] == []
    assert value["input_artifacts_immutable"] is True
print(root)
PY
```

Every mode directory contains its run JSON, decode-ready manifest, setup control/ready, timing,
Target diagnostics, plugin report and process-lifecycle artifact. Dual directories additionally
contain request-state, proposal-round, proposal-lifecycle, scheduler, verification, Draft-work,
transport, cycle, overlap and output-checkpoint artifacts. Every gate contains `validation.json`
and `validation.md`; Gate 1 also contains `controlled-validation.json`. Every attempted gate has
`reference/stock-determinism-diagnostic.json` with both raw stock runs. The immutable stock
reference exists only when that same pair is deterministic. `stock-stage/` records the exact
restore/check and re-applied patch manifests for that gate.

After the commands finish, report the root path and the three validation JSON files, then stop.
Do not start Phase 4B.2, Dual-Eager, packed-tree verification, KVConnector, performance or SLO
experiments.
