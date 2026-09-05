# Phase 4B.2: fresh three-mode run after the logical EOS commit fix

The completed `04e9b614...` Dual run committed a physical token after EOS: logical
length 84 disagreed with final length 83, and an unnecessary round-1 Draft proposal
was generated. Strict terminal recovery correctly refused it. **Do not promote
or recover that run.** Preserve its entire three-mode root unchanged. The
[pinned-vLLM logical commit audit](phase4b2-dual-logical-commit-audit.md) explains
the runtime repair, stop contract and CPU regression coverage.

This supersedes the old Target/Serial reuse and historical-commit Dual continuation.
The coding agent performs CPU validation only. The operator runs Target, Serial, and
Dual-Batch **once each at the same new fix commit**, under a fresh result root. PR #4
remains Draft/Open/unmerged. A failed mode directory is immutable; never retry in place.

## Runtime repair and retained lifecycle protection

The observer now commits the current rejection-parsed sampled delta onto the
previous logical prefix, stopping inclusively at the first EOS/processed stop ID
or output limit. It cross-checks the physical row without importing post-terminal
tokens into logical output. Terminal work performs required Draft synchronization
and release, with no next proposal. Target/Serial semantics, performance formulas
and the strict reconciler are unchanged. The installed vLLM patch stack is unchanged.

The failed A800 Dual run at `56bd0a50e3b5f33cf30e32564532b1483ea7e34d` progressed
from 100 to 99 running requests before a late ready result reached the scheduler.
The permanent frozen-prompt binding still existed, but stock vLLM had removed its
internal request. The strict live lookup raised before the old terminal-request
branch could run. EngineCore then died, so no `resident-dual.json` was produced.

The Dual-only async resolver distinguishes these cases:

| Binding / live table | Behavior |
| --- | --- |
| Unknown, unbound, inconsistent, or aliased | Fatal |
| Valid historical binding; internal key absent | Validate payload, discard as `request-retired-before-ready` |
| Live object present | Preserve ID, prefix count/SHA, version, round, ownership and second-proposal checks |
| Finished object still present | Validate payload, discard as `terminal-request` |

The generic `resolve_stable_ready_request()` remains strict. Retired proposals must
parse as `DualProposal`, match their envelope's stable ID and canonical proposal ID,
and must not be already consumed. New late proposals emit `CREATED`, `PUBLISHED`,
`DROPPED_STALE`; they are never installed or verified. A previously installed,
unconsumed proposal is dropped once, without republishing it. Consumed proposal
history remains complete. Identical late delivery is idempotent; conflicting replay
or different proposals claiming one retired request round remain fatal.

Retired tails require an actual boolean tail flag and positive integer readiness
timestamp, and get explicit drop evidence. Cleanup removes only that stable ID from
`_dual_drafting`, `_dual_tail_ready`, `_dual_tail_ready_ns`, and `_dual_proposals`.
The poll's pending snapshot cannot reintroduce a retired ID. No request is recreated,
and another request can use the next available slot. Poll cadence and budgets are unchanged.

Scheduler cycles contain a `retired_ready_results` event list. The final
`resident-dual.json` contains `retired_ready_results` with `events` and:

- `retired_ready_result_drop_count`
- `retired_proposal_drop_count`
- `retired_tail_drop_count`

These count distinct discarded ready results, including terminal objects still in
the live table. Cleanup of a previously installed proposal additionally completes
its lifecycle; it is not a second ready-result receipt. Zero counts are explicit.
Nonzero counts do not invalidate a run; existing lifecycle, scheduling and physical
overlap validators still apply.

## Unchanged measurement policy

Use the frozen corrected-100 workload, Qwen3-0.6B Draft on GPU0 and Qwen3-32B Target
on GPU1,2 with TP2. Keep the existing config, proposal budget, microbatch size,
baseline scheduling, required overlap and batch-invariant mode. Setup and bootstrap
are excluded, the first post-bootstrap token is counted, and no per-token CUDA sync
is added. Metric definitions and matched-work gates are unchanged. Exact generated
tokens are diagnostic only. Warmup/JIT provenance remains visible.

There is no cross-commit compatibility exception: all three execution commits must
equal the delivered fix SHA. Old Target/Serial artifacts remain historical evidence.
No Gate3 diagnostics, token-divergence investigation, additional GPU preflight
generation, patch reapplication, or optimization belongs to this run.

## Interactive Bash contract

Use one interactive Bash shell. First export `SR_PHASE4B_FIX_COMMIT` to the full
40-character SHA in the delivery message. Run **one command block at a time**.
After every printed `rc`, stop manually if it is nonzero; retain all artifacts and
resolve the failure before any dependent step. Do not paste the whole runbook at once.

The following preamble disables automatic termination on a command failure, unset
variable termination and an inherited error handler in this shell. Every important
command below captures and prints its return code. A failing test, helper or GPU
child returns control to the operator's shell; no command below terminates that
shell merely because a test or run returns nonzero.

```bash
set +e
set +u
set +o pipefail
trap - ERR
test -n "${SR_PHASE4B_FIX_COMMIT:-}"
RC="$?"
echo "fix commit supplied rc=$RC"
```

## A. Locate and preserve the invalid 04e9b three-mode root

If more than one `04e9b` result exists, set `SR_PHASE4B2_FAILED_ROOT` to the full
root containing the reported `dual/resident-dual.json`, then repeat only the
location block. Automatic selection requires exactly one match. These commands
read the historical artifacts and create an inventory in a separate new audit
directory. Keep it even if a later step fails.

```bash
SR_PHASE4B2_FAILED_ROOT="$(python3 - <<'PY'
import json, os, pathlib
base = pathlib.Path("/root/autodl-tmp/SpecRhythm-data/results/phase4/04e9b6141e3846835e6fdee0a42cdb9e8d021e4e")
configured = os.environ.get("SR_PHASE4B2_FAILED_ROOT")
matches = [pathlib.Path(configured)] if configured else sorted({p.parent.parent for p in base.glob("*/dual/resident-dual.json")})
assert len(matches) == 1, "set the exact historical SR_PHASE4B2_FAILED_ROOT; candidates: " + repr(matches)
root = matches[0]
runtime = json.loads((root / "dual/runtime-manifest.json").read_text())
assert runtime["git_commit"] == "04e9b6141e3846835e6fdee0a42cdb9e8d021e4e"
assert (root / "dual/resident-dual.json").is_file()
print(root)
PY
)"
RC="$?"
echo "locate invalid 04e9b run rc=$RC"
```

```bash
export SR_PHASE4B2_FAILED_ROOT
export SR_PHASE4B2_FAILED_DUAL="$SR_PHASE4B2_FAILED_ROOT/dual"
export SR_PHASE4B2_AUDIT_DIR="/root/autodl-tmp/SpecRhythm-data/audits/phase4b2-logical-commit-$$"
python3 - <<'PY'
import hashlib, json, os, pathlib
root = pathlib.Path(os.environ["SR_PHASE4B2_FAILED_ROOT"])
audit = pathlib.Path(os.environ["SR_PHASE4B2_AUDIT_DIR"])
assert root.is_dir(), root
assert root.resolve() not in audit.resolve().parents
rows = {}
for path in sorted(root.rglob("*")):
    key = str(path.relative_to(root))
    if path.is_symlink():
        rows[key] = {"symlink": os.readlink(path)}
    elif path.is_file():
        rows[key] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    elif path.is_dir():
        rows[key] = {"directory": True}
    else:
        raise ValueError(f"unexpected non-file evidence: {path}")
assert rows, "failed evidence directory is empty"
audit.mkdir(parents=True, exist_ok=False)
with (audit / "failed-dual-before.json").open("x") as handle:
    json.dump({"root": str(root), "entries": rows}, handle, indent=2, sort_keys=True)
print("preserved failed Dual inventory:", audit / "failed-dual-before.json")
PY
RC="$?"
echo "failed Dual checksum inventory rc=$RC"
```

## B. Check out the delivered fix commit

```bash
cd /root/autodl-tmp/src/SpecRhythm
RC="$?"
echo "repository directory rc=$RC"
```

```bash
git fetch origin codex/vllm-serving-v0.1
RC="$?"
echo "fetch fix rc=$RC"
```

```bash
python3 - <<'PY'
import os, re, subprocess
commit = os.environ["SR_PHASE4B_FIX_COMMIT"]
assert re.fullmatch(r"[0-9a-f]{40}", commit), "supply the delivered full SHA"
assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
assert subprocess.check_output(["git", "rev-parse", commit + "^{commit}"], text=True).strip() == commit
assert commit not in {"04e9b6141e3846835e6fdee0a42cdb9e8d021e4e", "d6fe1529c484dc57184d1855f97b26dfdeabd2ce"}
assert subprocess.run(["git", "merge-base", "--is-ancestor", "d6fe1529c484dc57184d1855f97b26dfdeabd2ce", commit]).returncode == 0
PY
RC="$?"
echo "clean checkout and explicit fix pin rc=$RC"
```

```bash
git switch --detach "$SR_PHASE4B_FIX_COMMIT"
RC="$?"
echo "checkout new execution commit rc=$RC"
```

```bash
test "$(git rev-parse HEAD)" = "$SR_PHASE4B_FIX_COMMIT"
RC="$?"
echo "HEAD equals delivered fix rc=$RC"
export SR_PHASE4B_COMMIT="$SR_PHASE4B_FIX_COMMIT"
export SR_PHASE4B_EXECUTION_COMMIT="$SR_PHASE4B_FIX_COMMIT"
export SR_PHASE4B_MEASUREMENT_COMMIT="$SR_PHASE4B_FIX_COMMIT"
```

## C. Editable install and fixed inputs

```bash
conda activate /root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1
RC="$?"
echo "activate pinned environment rc=$RC"
```

```bash
python -m pip install -e '.[dev]' --no-deps --no-build-isolation
RC="$?"
echo "editable install fix rc=$RC"
```

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_BATCH_INVARIANT=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export SR_DRAFT_MODEL="/root/autodl-tmp/models/Qwen3-0.6B"
export SR_TARGET_MODEL="/root/autodl-tmp/models/Qwen3-32B"
export SR_VLLM_SOURCE="/root/autodl-tmp/src/vllm-v0.25.1"
export SR_INPUT_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/eba0df493a7fd350ef3c8776e06d30e6196b6749/phase4b1-gate2-corrected5-20260827T040244Z"
export SR_PHASE4B_WORKLOAD="$SR_INPUT_ROOT/workloads/corrected-100.jsonl"
export SR_PHASE4B_REFERENCE="$SR_INPUT_ROOT/Gate-3-corrected-100/reference/stock-target-reference.json"
export SR_PHASE4B_CONFIG="$PWD/configs/phase4b_dual_batch_1d2v.yaml"
export PHASE4B1_OVERLAP_REQUIREMENT=required
unset PHASE4B1_NUMERICAL_PLAN PHASE4B1_NUMERICAL_OUTPUT
SR_VLLM_ROOT="$(python - <<'PY'
from importlib import metadata
print(metadata.distribution("vllm").locate_file(""))
PY
)"
RC="$?"
echo "locate installed vLLM rc=$RC"
export SR_VLLM_ROOT
```

## D. Read-only pinned-source and installed-patch checks

Check the existing installed patch stack; do not restore or reapply it.

```bash
python integrations/vllm/manage_patch.py check \
  --vllm-root "$SR_VLLM_ROOT" --source "$SR_VLLM_SOURCE" \
  --expect-state patched --manifest "$SR_PHASE4B2_AUDIT_DIR/installed-patch-check.json"
RC="$?"
echo "installed pinned patch state rc=$RC"
```

```bash
python - <<'PY'
import hashlib, importlib.util, json, os, pathlib, subprocess
from specrhythm.phase4.config import load_phase4_config
from specrhythm.phase4.serial_runner import load_patch_manifest, validate_installed_patch_stack
from specrhythm.phase4.stock_vllm import load_smoke_requests
repo = pathlib.Path.cwd()
scheduler = pathlib.Path(importlib.util.find_spec("specrhythm.phase4.vllm_dual_scheduler").origin)
assert scheduler.resolve().is_relative_to(repo / "src")
assert "def _drop_retired_ready_result(" in scheduler.read_text()
from specrhythm.phase4.dual_commit import DualStopPolicy
assert DualStopPolicy(8, 151645).canonicalize((45596,), (13, 151645, 151643)) == ((45596, 13, 151645), "eos")
source = os.environ["SR_VLLM_SOURCE"]
assert subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip() == "752a3a504485790a2e8491cacbb35c137339ad34"
assert not subprocess.check_output(["git", "-C", source, "status", "--porcelain", "--untracked-files=no"], text=True).strip()
old = pathlib.Path(os.environ["SR_PHASE4B2_FAILED_ROOT"])
config_path = pathlib.Path(os.environ["SR_PHASE4B_CONFIG"])
config = load_phase4_config(config_path)
manifest = load_patch_manifest(old / "patch-stage/vllm-patch-stack.json", config)
print(json.dumps(validate_installed_patch_stack(manifest), indent=2))
workload = pathlib.Path(os.environ["SR_PHASE4B_WORKLOAD"])
assert len(load_smoke_requests(workload, 100, require_task_mixture=True)) == 100
assert pathlib.Path(os.environ["SR_PHASE4B_REFERENCE"]).is_file()
historical = json.loads((old / "target/decode-performance.json").read_text())
for key, path in (("config", config_path), ("workload", workload)):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == historical["artifact_sha256"][key]
for role, env in (("draft", "SR_DRAFT_MODEL"), ("target", "SR_TARGET_MODEL")):
    assert pathlib.Path(os.environ[env]).is_dir()
    assert historical["models"][role]["path"] == os.environ[env]
print("fixed code, workload, models and installed patch verified")
PY
RC="$?"
echo "source and fixed input validation rc=$RC"
```

## E. Create a new root and capture current environment/topology

The root creation rejects an existing directory and copies only the unchanged
vLLM apply manifest. It does not copy old mode outputs or relabel their provenance.
The environment probe reads hardware/software metadata; it does not run generation.

```bash
SR_PHASE4B2_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RC="$?"
echo "fresh root timestamp rc=$RC"
export SR_PHASE4B2_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/$SR_PHASE4B_COMMIT/phase4b2-logical-commit-$SR_PHASE4B2_STAMP-$$"
export SR_PHASE4B_ENVIRONMENT="$SR_PHASE4B2_ROOT/environment.json"
export SR_PHASE4B_TOPOLOGY="$SR_PHASE4B2_ROOT/topology.json"
export SR_PHASE4B_PATCH_MANIFEST="$SR_PHASE4B2_ROOT/patch-stage/vllm-patch-stack.json"
```

```bash
python - <<'PY'
import os, pathlib, shutil, subprocess
root = pathlib.Path(os.environ["SR_PHASE4B2_ROOT"])
old = pathlib.Path(os.environ["SR_PHASE4B2_FAILED_ROOT"])
commit = os.environ["SR_PHASE4B_FIX_COMMIT"]
assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == commit
assert root.parent.name == commit and root.resolve() != old.resolve()
assert old.resolve() not in root.resolve().parents
root.mkdir(parents=True, exist_ok=False)
(root / "patch-stage").mkdir()
shutil.copyfile(old / "patch-stage/vllm-patch-stack.json", os.environ["SR_PHASE4B_PATCH_MANIFEST"])
print("NEW result root:", root)
PY
RC="$?"
echo "create fresh immutable run root rc=$RC"
```

```bash
env -u CUDA_VISIBLE_DEVICES VLLM_BATCH_INVARIANT=1 specrhythm phase4-probe \
  --config "$SR_PHASE4B_CONFIG" --vllm-source "$SR_VLLM_SOURCE" \
  --environment-output "$SR_PHASE4B_ENVIRONMENT" \
  --topology-output "$SR_PHASE4B_TOPOLOGY" \
  --validation-output "$SR_PHASE4B2_ROOT/probe-validation.json"
RC="$?"
echo "current environment and topology probe rc=$RC"
```

```bash
python - <<'PY'
import hashlib, json, os, pathlib
root = pathlib.Path(os.environ["SR_PHASE4B2_ROOT"])
assert json.loads((root / "probe-validation.json").read_text())["valid"] is True
names = ("SR_PHASE4B_WORKLOAD", "SR_PHASE4B_REFERENCE", "SR_PHASE4B_CONFIG",
         "SR_PHASE4B_ENVIRONMENT", "SR_PHASE4B_TOPOLOGY", "SR_PHASE4B_PATCH_MANIFEST")
rows = {os.environ[name]: hashlib.sha256(pathlib.Path(os.environ[name]).read_bytes()).hexdigest()
        for name in names}
with (root / "fixed-input-sha256.json").open("x") as handle:
    json.dump(rows, handle, indent=2, sort_keys=True)
PY
RC="$?"
echo "freeze new common input inventory rc=$RC"
```

```bash
source integrations/vllm/phase4b_run_helpers.sh
RC="$?"
echo "load process cleanup helpers rc=$RC"
```

```bash
source integrations/vllm/phase4b1_gate_helpers.sh
RC="$?"
echo "load resident run helpers rc=$RC"
```

```bash
source integrations/vllm/phase4b2_run_helpers.sh
RC="$?"
echo "load performance helpers rc=$RC"
```

Define two CPU-only checks for the remaining steps. `fresh_check` verifies the
commit, inputs and unused destination before each GPU run. Existing processes or
Phase-4B sockets require manual ownership review; these checks do not kill/unlink them.
`fresh_validate_mode` prints the validated performance and the Dual drop summary.

```bash
fresh_check () {
  python - "$1" <<'PY'
import hashlib, json, os, pathlib, subprocess, sys
root = pathlib.Path(os.environ["SR_PHASE4B2_ROOT"])
assert not (root / sys.argv[1]).exists(), "immutable mode directory already exists"
assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == os.environ["SR_PHASE4B_FIX_COMMIT"]
assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
for name, expected in json.loads((root / "fixed-input-sha256.json").read_text()).items():
    assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest() == expected, name
check = subprocess.run(["pgrep", "-af", "vllm|specrhythm.*draft-service|EngineCore"], capture_output=True, text=True)
assert check.returncode == 1, "review existing processes manually:\n" + check.stdout + check.stderr
assert not list(pathlib.Path("/tmp").glob("sr4b1-*.sock")), "review existing sockets manually"
print("fresh mode and common provenance verified:", sys.argv[1])
PY
}
fresh_validate_mode () {
  python - "$1" <<'PY'
import hashlib, json, os, pathlib, sys
root = pathlib.Path(os.environ["SR_PHASE4B2_ROOT"])
mode = sys.argv[1]
value = json.loads((root / mode / "decode-performance.json").read_text())
assert value["valid"] is True and value["errors"] == []
assert value["performance_result"] is True and value["cleanup_valid"] is True
assert value["execution_git_commit"] == os.environ["SR_PHASE4B_FIX_COMMIT"]
assert value["measurement_code_git_commit"] == os.environ["SR_PHASE4B_FIX_COMMIT"]
assert value["request_count"] == value["metrics"]["completed_requests"] == 100
for name, expected in json.loads((root / "fixed-input-sha256.json").read_text()).items():
    assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest() == expected, name
print(mode, "metrics:", json.dumps(value["metrics"], indent=2))
print("warmup_clean =", value["warmup_clean"])
if mode == "dual":
    raw = json.loads((root / mode / "resident-dual.json").read_text())
    assert raw["valid"] is True and raw["errors"] == []
    assert raw["overlap_gate"]["valid"] is True
    from specrhythm.phase4.transport import CheckpointJsonl
    from specrhythm.phase4.dual_correctness import validate_request_state_events
    from specrhythm.phase4.serial import token_prefix_hash
    plugin = json.loads((root / mode / "plugin-report.json").read_text())
    assert plugin["logical_commit_source"] == "current-round-rejection-parsed-sampled-token-ids"
    states = CheckpointJsonl(root / mode / "request-state-events.jsonl").read()
    assert validate_request_state_events(states) == []
    outputs = {row["request_id"]: row for row in raw["outputs"]}
    workload = {row["request_id"]: row for row in map(json.loads, pathlib.Path(os.environ["SR_PHASE4B_WORKLOAD"]).read_text().splitlines())}
    for request_id, output in outputs.items():
        final_prefix = workload[request_id]["prompt_token_ids"] + output["generated_token_ids"]
        last = [row for row in states if row["request_id"] == request_id][-1]
        assert last["destination_state"] == "TERMINAL"
        assert last["committed_prefix_length"] == len(final_prefix)
        assert last["committed_prefix_sha256"] == token_prefix_hash(final_prefix)
    affected = "r3-22887f929fd54d97814c2bd3"
    if outputs[affected]["generated_token_ids"] == [45596, 13, 151645]:
        trace = [row for row in states if row["request_id"] == affected]
        assert trace[-1]["committed_prefix_length"] == 83
        assert [row["destination_state"] for row in trace[-2:]] == ["COMMITTING", "TERMINAL"]
        assert all(row["destination_state"] != "DRAFT_SYNC" for row in trace)
        lifecycle = CheckpointJsonl(root / mode / "proposal-lifecycle-events.jsonl").read()
        assert not [row for row in lifecycle if row["request_id"] == affected and row["round_id"] >= 1]
        assert not [row for row in lifecycle if row["proposal_id"] == "868652c02d867275a39cbdf2cdc4c460133b2e26564e351b74cd2b433d9dda78"]
        print("reported EOS trajectory: logical length 83, TERMINAL, no round-1 proposal")
    else:
        print("affected request followed a different trajectory; normal matched-work gates apply")
    summary = raw["retired_ready_results"]
    counts = {key: val for key, val in summary.items() if key.endswith("_count")}
    assert all(type(val) is int and val >= 0 for val in counts.values())
    assert summary["retired_ready_result_drop_count"] == (
        summary["retired_proposal_drop_count"] + summary["retired_tail_drop_count"]
    ) == len(summary["events"])
    print("legitimate retired ready-result drops:", json.dumps(counts, indent=2))
PY
}
RC="$?"
echo "define fresh-run validation functions rc=$RC"
```

## F. Fresh Target, once

```bash
fresh_check target
RC="$?"
echo "Target fresh-run preconditions rc=$RC"
```

```bash
phase4b2_run_mode target "$SR_PHASE4B2_ROOT/target" \
  "$SR_PHASE4B_WORKLOAD" 100 "$SR_PHASE4B_REFERENCE"
RC="$?"
echo "Target execution and cleanup rc=$RC"
```

## G. Derive and validate Target performance

```bash
phase4b2_measure_mode target "$SR_PHASE4B2_ROOT/target" "$SR_PHASE4B_WORKLOAD"
RC="$?"
echo "Target performance derivation rc=$RC"
```

```bash
fresh_validate_mode target
RC="$?"
echo "Target performance validation rc=$RC"
```

## H. Fresh Serial, once

```bash
fresh_check serial
RC="$?"
echo "Serial fresh-run preconditions rc=$RC"
```

```bash
phase4b2_run_mode serial "$SR_PHASE4B2_ROOT/serial" \
  "$SR_PHASE4B_WORKLOAD" 100 "$SR_PHASE4B_REFERENCE"
RC="$?"
echo "Serial execution and cleanup rc=$RC"
```

## I. Derive and validate Serial performance

```bash
phase4b2_measure_mode serial "$SR_PHASE4B2_ROOT/serial" "$SR_PHASE4B_WORKLOAD"
RC="$?"
echo "Serial performance derivation rc=$RC"
```

```bash
fresh_validate_mode serial
RC="$?"
echo "Serial performance validation rc=$RC"
```

## J. Approve matched work for Target/Serial

```bash
phase4b2_compare_target_serial "$SR_PHASE4B2_ROOT"
RC="$?"
echo "Target Serial matched-work comparison rc=$RC"
```

```bash
phase4b2_require_matched_work_pair "$SR_PHASE4B2_ROOT" 100
RC="$?"
echo "Target Serial matched-work approval rc=$RC"
```

Expect `MATCHED WORK TARGET/SERIAL PASS` and `performance_comparable = true`.
The printed exact-sequence diagnostic may be false. A pair deliberately retains
`comparison_complete=false`; the final performance claim requires all three modes.

## K. Fresh Dual-Batch, once

```bash
fresh_check dual
RC="$?"
echo "Dual fresh-run preconditions rc=$RC"
```

```bash
phase4b2_run_mode dual "$SR_PHASE4B2_ROOT/dual" \
  "$SR_PHASE4B_WORKLOAD" 100 "$SR_PHASE4B_REFERENCE"
RC="$?"
echo "Dual execution and cleanup rc=$RC"
```

On failure, stop manually. Preserve this new Dual directory and its logs; do not
derive performance from an incomplete run or invoke this GPU command again in place.

## L. Derive and validate Dual performance

The execution helper uses `dual`; the measurement CLI uses `dual-batch`.

```bash
phase4b2_measure_mode dual-batch "$SR_PHASE4B2_ROOT/dual" "$SR_PHASE4B_WORKLOAD"
RC="$?"
echo "Dual performance derivation rc=$RC"
```

```bash
fresh_validate_mode dual
RC="$?"
echo "Dual performance and retired-drop validation rc=$RC"
```

## M. Final three-mode matched-work comparison

```bash
phase4b2_compare_all "$SR_PHASE4B2_ROOT"
RC="$?"
echo "final three-mode matched-work comparison rc=$RC"
```

```bash
python - <<'PY'
import hashlib, json, os, pathlib
root = pathlib.Path(os.environ["SR_PHASE4B2_ROOT"])
report = json.loads((root / "decode-performance-comparison.json").read_text())
assert report["valid"] is True and report["errors"] == []
assert report["comparison_complete"] is True and report["performance_valid"] is True
assert report["matched_work_comparability"]["valid"] is True
for mode, directory in (("target", "target"), ("serial", "serial"), ("dual-batch", "dual")):
    path = root / directory / "decode-performance.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == report["input_sha256"][mode]
    assert report["execution_provenance"][mode]["execution_git_commit"] == os.environ["SR_PHASE4B_FIX_COMMIT"]
    assert report["metrics"][mode]["completed_requests"] == 100
print("THREE-MODE MATCHED WORK PASS")
for field in ("metrics", "speedups", "warmup", "exact_sequence_diagnostic", "claim_boundary"):
    print(field, "=", json.dumps(report[field], indent=2))
raw = json.loads((root / "dual/resident-dual.json").read_text())
print("retired drops =", json.dumps({key: value for key, value in raw["retired_ready_results"].items() if key.endswith("_count")}))
PY
RC="$?"
echo "final comparison provenance and validity rc=$RC"
```

Finally verify that the entire invalid historical three-mode root has exactly the same
contents. This check can also be run after a failure at any preceding step.

```bash
python - <<'PY'
import hashlib, json, os, pathlib
audit = pathlib.Path(os.environ["SR_PHASE4B2_AUDIT_DIR"])
before = json.loads((audit / "failed-dual-before.json").read_text())
root = pathlib.Path(before["root"])
assert root.is_dir()
rows = {}
for path in sorted(root.rglob("*")):
    key = str(path.relative_to(root))
    if path.is_symlink():
        rows[key] = {"symlink": os.readlink(path)}
    elif path.is_file():
        rows[key] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    elif path.is_dir():
        rows[key] = {"directory": True}
    else:
        raise ValueError(f"unexpected non-file evidence: {path}")
assert rows == before["entries"], "historical failed Dual evidence changed"
print("historical three-mode root unchanged; files and directory inventory match")
print("new three-mode root:", os.environ["SR_PHASE4B2_ROOT"])
PY
RC="$?"
echo "historical failed Dual immutability verification rc=$RC"
```

Keep the new result root, the separate audit inventory, and the original failed root.
The resulting claim remains preliminary matched-work decode-only bring-up, with
the recorded warmup/JIT limitations. GPU completion and speedup are established
only by the operator's new artifacts, not by the CPU lifecycle tests.

Do not tune proposal/microbatch budgets, scheduling, acceptance, eager/graph mode,
selection or Dual-Eager during this baseline. No token-level numerical divergence
diagnostics belong to this run. On any failed return code, stop manually and retain
the failed directory; a retry requires a separate fresh run root and new review.
