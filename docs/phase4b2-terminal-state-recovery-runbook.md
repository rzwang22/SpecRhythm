# Phase 4B.2: terminal evidence closure and offline recovery

**The reported real `04e9b6141e3846835e6fdee0a42cdb9e8d021e4e` run is NOT recoverable
by this procedure. Do not use the commands below to promote it.** New evidence shows
its state committed prefix has length 84 while final output has length 83, ending
at EOS 151645. It also generated unnecessary round-1 Draft work. The strict checker
correctly refused that contradiction and remains unchanged. See the
[runtime semantic audit](phase4b2-dual-logical-commit-audit.md) and use the
[fresh three-mode runbook](phase4b2-fresh-three-mode-runbook.md) for a future clean
baseline. The coding agent does not run GPU.

The remaining sections document the existing closure mechanism for genuinely
missing terminal evidence with an already exact committed prefix. Its synthetic
CPU fixture demonstrated that narrow case; it did not establish that the real
A800 artifacts were eligible. Completion, overlap and cleanup alone cannot justify
repairing an incorrect historical commit or its performance provenance.

## Process-boundary audit and future-run behavior

The scheduler owns the historical stable/internal binding and the live vLLM request
table. Its `request-retired-before-ready` event establishes removal and stale work,
but it does not own final Target output. The rank-zero proposer owns the observed
request-state trace. It can end at `COMMITTING -> DRAFT_SYNC` when stock completion
removes a request before the next callback. The coordinator owns final serialized
outputs after `llm.generate()` returns. `_serialize_outputs()` requires every vLLM
output to be finished and to have exactly one completion. The process supervisor
subsequently owns the coordinator status and cleanup evidence.

The selected architecture joins these facts in the coordinator, after generation,
final synchronization and Draft shutdown. It checks output completion, frozen
prompt/bootstrap identity, stable/internal bindings, and the full committed prefix.
Every state prefix must be an exact prefix of the final output, and the last
committed prefix must already equal the complete final prefix. Missing commits or
conflicting tokens are never inferred or repaired. Length completion must fill the
requested budget. Stock stop completion must be within budget; an explicit token
stop must match the last output token. Default stock EOS stopping may have a null
stop reason. No guessed model EOS set or new numerical diagnostic is introduced.

Only a valid trace with a legal predecessor of `TERMINAL`, no `FAILED` or prior
`TERMINAL`, matching absent-live-request evidence, and a strictly later observation
timestamp receives a closure. For the eligible synthetic case it is `DRAFT_SYNC -> TERMINAL`, reason
`stock-vllm-retired-after-final-output`. A legal `TARGET_TAIL_READY` predecessor is
also supported by the future-run reconciler when the same final-prefix proof exists.
The appended event is labelled `post-generation-terminal-reconciliation`; it is
not a runtime commit or performance timestamp. Already terminal traces are unchanged.
Reconciliation is idempotent, and conflicting duplicate evidence fails closed.

`validate_request_state_events()` is unchanged and must return no errors on the
resulting stream. The scheduler, proposer algorithms, cadence, microbatch size,
selection, speculative semantics, Target/Serial execution and metric formulas are
unchanged. The late proposal's lifecycle remains `CREATED -> PUBLISHED -> DROPPED_STALE`;
reconciliation never installs, verifies, commits or relabels it.

## Recovery restrictions and provenance

Offline recovery is restricted to the exact source execution commit above and
the corrected-100 60/20/20 workload. It requires all of the following:

- Raw resident schema/mode, `valid=false`, 100 complete checkpoint outputs and
  identical embedded serialized/decode-only outputs, plus exact source digests.
- Raw errors equal the errors reproduced from the state stream, and consist only
  of DRAFT_SYNC terminal gaps. Every gap must have a justified reconciliation.
- Recomputed proposal lifecycle, scheduling, round accounting, request identity,
  resident setup/worker/shutdown evidence and physical overlap all pass. Stored
  cycle/overlap rows must match reconstruction from the original physical intervals.
- Retired counts and events match the scheduler log; the affected request is absent
  from that cycle and later live decisions. Its late proposal was never installed,
  consumed or included in commit accounting.
- Both TP final-sync rows, GPU1,2 placement and commit/finalization ordering pass.
- The owned coordinator is the pinned resident Dual CLI through the timestamp wrapper,
  bound to this run's output and state files with the original execution options.
  Its status is exactly 1, with no launch failure, surviving process, leaked process
  group, Target termination action or invalid Draft cleanup. Final output lies
  within that process lifetime, and logs contain no runtime exception evidence.

The pinned CLI writes the complete resident artifact at the end of the runner and
then returns 1 for its invalid verdict. This complete, cross-checked evidence
distinguishes the post-validation return from an arbitrary coordinator status 1.
The original `run_valid=false`, `target_exit_status=1`, `effective_exit_status=1`,
and raw resident `valid=false` are retained in the provenance; none is overwritten.

The recovery command creates a new directory **outside the original three-mode root**:

- `request-state-events.reconciled.jsonl`: verified original states plus the justified
  closure, using normal checksummed checkpoint framing.
- `terminal-state-revalidation.json`: source hashes/inventory, original errors,
  process interpretation, recovery code commit and explicit
  `terminal_state_reconciliation.recovered=true` with reconciled request IDs.

The optional `--terminal-revalidation` measurement input works only for this Dual
case. It recomputes the certificate's proof and source hashes before accepting it;
the certificate's `valid` flag alone grants nothing. It replaces only the explained
resident/process validation rejection. All existing accounting, synchronization,
cleanup, boundary, workload and performance checks still run. Original execution
provenance stays `04e9b614...`; measurement/revalidation provenance uses the new code
commit. No cross-execution-commit exception is added to the comparator.

## Safe interactive Bash commands: offline only

Use the same server and original absolute artifact paths. Export
`SR_PHASE4B2_REVALIDATION_COMMIT` to the full new SHA in the delivery message.
If known, set `SR_PHASE4B2_SOURCE_ROOT` to the completed three-mode result root.
Otherwise the selection below succeeds only if exactly one matching completed Dual
root exists under the source execution commit. It never guesses between candidates.

Run one block at a time. Every important command captures and prints its return
code. **Stop manually on any nonzero code.** Preserve the output and all evidence;
do not continue to measurement after a refused audit, and do not retry in place.
These commands run no GPU generation and do not alter installed vLLM.

```bash
set +e
set +u
set +o pipefail
trap - ERR
test -n "${SR_PHASE4B2_REVALIDATION_COMMIT:-}"
RC="$?"
echo "revalidation code commit supplied rc=$RC"
```

```bash
cd /root/autodl-tmp/src/SpecRhythm
RC="$?"
echo "repository directory rc=$RC"
```

```bash
git fetch origin codex/vllm-serving-v0.1
RC="$?"
echo "fetch terminal reconciliation fix rc=$RC"
```

```bash
python3 - <<'PY'
import os, re, subprocess
commit = os.environ["SR_PHASE4B2_REVALIDATION_COMMIT"]
assert re.fullmatch(r"[0-9a-f]{40}", commit), "use the delivered full fix SHA"
assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
assert subprocess.check_output(["git", "rev-parse", commit + "^{commit}"], text=True).strip() == commit
PY
RC="$?"
echo "clean checkout and explicit commit pin rc=$RC"
```

```bash
git switch --detach "$SR_PHASE4B2_REVALIDATION_COMMIT"
RC="$?"
echo "checkout evidence reconciliation code rc=$RC"
```

```bash
conda activate /root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1
RC="$?"
echo "activate existing Python environment rc=$RC"
```

```bash
python -m pip install -e '.[dev]' --no-deps --no-build-isolation
RC="$?"
echo "editable install reconciliation code rc=$RC"
```

```bash
SR_PHASE4B2_SOURCE_ROOT="$(python - <<'PY'
import json, os, pathlib, subprocess
commit = "04e9b6141e3846835e6fdee0a42cdb9e8d021e4e"
assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == os.environ["SR_PHASE4B2_REVALIDATION_COMMIT"]
explicit = os.environ.get("SR_PHASE4B2_SOURCE_ROOT")
base = pathlib.Path("/root/autodl-tmp/SpecRhythm-data/results/phase4") / commit
candidates = [pathlib.Path(explicit)] if explicit else sorted(
    path.parent.parent for path in base.glob("*/dual/resident-dual.json")
)
assert len(candidates) == 1, "set SR_PHASE4B2_SOURCE_ROOT explicitly: " + str(candidates)
root = candidates[0].resolve()
manifest = json.loads((root / "dual/decode-ready-manifest.json").read_text())
assert manifest["specrhythm_git_commit"] == commit
print(root)
PY
)"
RC="$?"
echo "select completed source run rc=$RC"
export SR_PHASE4B2_SOURCE_ROOT
```

Create an external checksum inventory before revalidation. This inventories the
entire original three-mode root, including the raw invalid Dual artifact and the
successful Target/Serial artifacts. It does not write inside that root.

```bash
export SR_PHASE4B2_AUDIT_DIR="/root/autodl-tmp/SpecRhythm-data/audits/phase4b2-terminal-state-$$"
export SR_PHASE4B2_DERIVED_ROOT="$SR_PHASE4B2_AUDIT_DIR/revalidated"
export SR_INPUT_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/eba0df493a7fd350ef3c8776e06d30e6196b6749/phase4b1-gate2-corrected5-20260827T040244Z"
export SR_PHASE4B_WORKLOAD="$SR_INPUT_ROOT/workloads/corrected-100.jsonl"
export SR_PHASE4B_CONFIG="$PWD/configs/phase4b_dual_batch_1d2v.yaml"
export SR_PHASE4B_TOPOLOGY="$SR_PHASE4B2_SOURCE_ROOT/topology.json"
export SR_PHASE4B_PATCH_MANIFEST="$SR_PHASE4B2_SOURCE_ROOT/patch-stage/vllm-patch-stack.json"
python - <<'PY'
import hashlib, json, os, pathlib
root = pathlib.Path(os.environ["SR_PHASE4B2_SOURCE_ROOT"])
audit = pathlib.Path(os.environ["SR_PHASE4B2_AUDIT_DIR"])
assert root.is_dir() and root.resolve() not in audit.resolve().parents
assert not any(path.is_symlink() for path in root.rglob("*")), "review symlink evidence manually"
inventory = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(root.rglob("*")) if path.is_file()}
assert inventory
audit.mkdir(parents=True, exist_ok=False)
with (audit / "source-before.json").open("x") as handle:
    json.dump({"root": str(root), "sha256": inventory}, handle, indent=2, sort_keys=True)
print("raw evidence inventory:", audit / "source-before.json")
PY
RC="$?"
echo "preserve complete source checksum inventory rc=$RC"
```

Run the strict offline recovery. This is the decision point for whether the actual
A800 run can be reused. A refusal preserves the raw files and creates no accepted
recovery certificate.

```bash
specrhythm phase4b2-reconcile-dual-terminal \
  --run-root "$SR_PHASE4B2_SOURCE_ROOT/dual" \
  --workload "$SR_PHASE4B_WORKLOAD" --config "$SR_PHASE4B_CONFIG" \
  --topology "$SR_PHASE4B_TOPOLOGY" --patch-manifest "$SR_PHASE4B_PATCH_MANIFEST" \
  --output-dir "$SR_PHASE4B2_DERIVED_ROOT"
RC="$?"
echo "strict completed Dual terminal-state recovery rc=$RC"
```

Expect `recovered=true`, the original execution SHA, and the affected stable request
ID in the printed reconciliation. The derived state validator must pass without
special interpretation; the original state log remains unchanged and invalid.

```bash
python - <<'PY'
import json, os, pathlib
from specrhythm.phase4.dual_correctness import validate_request_state_events
from specrhythm.phase4.transport import CheckpointJsonl
root = pathlib.Path(os.environ["SR_PHASE4B2_DERIVED_ROOT"])
report = json.loads((root / "terminal-state-revalidation.json").read_text())
assert report["valid"] is True and report["source_resident_valid"] is False
assert report["terminal_state_reconciliation"]["recovered"] is True
assert validate_request_state_events(CheckpointJsonl(root / "request-state-events.reconciled.jsonl").read()) == []
print("derived TERMINAL closure validated; original invalid evidence preserved")
PY
RC="$?"
echo "derived terminal state validation rc=$RC"
```

Derive Dual performance with the explicit certificate. Execution stays at the
original common commit; the measurement code commit is the newly checked-out fix.

```bash
specrhythm phase4b2-decode-run --mode dual-batch \
  --run-root "$SR_PHASE4B2_SOURCE_ROOT/dual" \
  --workload "$SR_PHASE4B_WORKLOAD" --config "$SR_PHASE4B_CONFIG" \
  --topology "$SR_PHASE4B_TOPOLOGY" --patch-manifest "$SR_PHASE4B_PATCH_MANIFEST" \
  --terminal-revalidation "$SR_PHASE4B2_DERIVED_ROOT/terminal-state-revalidation.json" \
  --output "$SR_PHASE4B2_DERIVED_ROOT/decode-performance.json"
RC="$?"
echo "revalidated Dual performance derivation rc=$RC"
```

```bash
specrhythm phase4b2-decode-compare \
  --target "$SR_PHASE4B2_SOURCE_ROOT/target/decode-performance.json" \
  --serial "$SR_PHASE4B2_SOURCE_ROOT/serial/decode-performance.json" \
  --dual "$SR_PHASE4B2_DERIVED_ROOT/decode-performance.json" \
  --output "$SR_PHASE4B2_DERIVED_ROOT/decode-performance-comparison.json" \
  --markdown-output "$SR_PHASE4B2_DERIVED_ROOT/decode-performance-comparison.md"
RC="$?"
echo "three-mode matched-work comparison rc=$RC"
```

```bash
python - <<'PY'
import json, os, pathlib
root = pathlib.Path(os.environ["SR_PHASE4B2_DERIVED_ROOT"])
dual = json.loads((root / "decode-performance.json").read_text())
report = json.loads((root / "decode-performance-comparison.json").read_text())
assert dual["valid"] is True and dual["errors"] == [] and dual["cleanup_valid"] is True
assert dual["execution_git_commit"] == "04e9b6141e3846835e6fdee0a42cdb9e8d021e4e"
assert dual["measurement_code_git_commit"] == os.environ["SR_PHASE4B2_REVALIDATION_COMMIT"]
assert dual["terminal_state_reconciliation"]["source_target_exit_status"] == 1
assert report["valid"] is True and report["errors"] == []
assert report["comparison_complete"] is True and report["performance_valid"] is True
assert report["matched_work_comparability"]["valid"] is True
for mode in ("target", "serial", "dual-batch"):
    assert report["execution_provenance"][mode]["execution_git_commit"] == dual["execution_git_commit"]
    assert report["metrics"][mode]["completed_requests"] == 100
for key in ("metrics", "speedups", "warmup", "exact_sequence_diagnostic", "claim_boundary"):
    print(key, "=", json.dumps(report[key], indent=2))
print("THREE-MODE MATCHED WORK PASS with explicit terminal-state revalidation provenance")
PY
RC="$?"
echo "final recovered comparison validation rc=$RC"
```

Finally verify that every source file is unchanged. This block is also safe to
run after any refusal or failed measurement above.

```bash
python - <<'PY'
import hashlib, json, os, pathlib
audit = pathlib.Path(os.environ["SR_PHASE4B2_AUDIT_DIR"])
before = json.loads((audit / "source-before.json").read_text())
root = pathlib.Path(before["root"])
assert root.is_dir()
inventory = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(root.rglob("*")) if path.is_file()}
assert inventory == before["sha256"], "original source evidence changed"
print("entire original three-mode source root unchanged")
print("new derived artifacts:", os.environ["SR_PHASE4B2_DERIVED_ROOT"])
PY
RC="$?"
echo "raw source immutability verification rc=$RC"
```

No command above terminates the interactive Bash shell merely because a test,
recovery or measurement returns nonzero. Failures return to the prompt with a
printed return code; the operator decides when to stop. Keep PR #4 Draft/Open/unmerged.
