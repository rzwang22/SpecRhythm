# Phase 4B.3 Plan A — operator D1–D5 runbook

**Operator execution only. These commands were not run by the coding agent.** Use the delivered implementation commit on PR #4, which remains Draft/Open/unmerged. Retain all old A800 artifacts. Every attempt needs a fresh result root. Do not integrate Dual or vary UUID mode, model/config, proposal budget, logging, measurement boundary or scheduling during this experiment.

The implementation is described in [phase4b3-plan-a-implementation.md](phase4b3-plan-a-implementation.md). The previous [design documents](design/phase4b3-batched-draft/README.md) remain unchanged. The new selector is `SR_PHASE4_DRAFT_BACKEND=hf-persistent|vllm-batched`, default `hf-persistent`. The new backend is **Serial-only**; Dual explicitly rejects this selection before loading a model.

## Interactive shell and frozen inputs

Use an interactive Bash shell in the repository with the qualified Python 3.11 / PyTorch 2.11.0 / vLLM 0.25.1 environment active. Run one block at a time. After each printed nonzero `rc`, stop manually, retain the failed directory and inspect the error before continuing. No block terminates the interactive shell on failure.

```bash
set +e
set +u
set +o pipefail
trap - ERR
source integrations/vllm/phase4b_run_helpers.sh &&
  source integrations/vllm/phase4b1_gate_helpers.sh &&
  source integrations/vllm/phase4b2_run_helpers.sh
RC="$?"
echo "load qualified run helpers rc=$RC"
```

Keep the previously qualified server variables: `SR_DRAFT_MODEL`, `SR_TARGET_MODEL`, `SR_VLLM_SOURCE`, `SR_VLLM_ROOT`, `SR_PHASE4B_CONFIG`, `SR_PHASE4B_ENVIRONMENT`, `SR_PHASE4B_TOPOLOGY`, and `SR_PHASE4B_PATCH_MANIFEST`. Set `SR_PHASE4B_COMMIT` to the full **delivered implementation SHA**, not the old `6403111` baseline. Supply `SR_PHASE4B3_WORKLOAD5` and `SR_PHASE4B3_REFERENCE5` for the existing corrected-five fixture/reference, and `SR_PHASE4B_WORKLOAD` and `SR_PHASE4B_REFERENCE` for corrected-100. Do not construct new prompts or copy a different workload merely to fill these variables.

Install the project code without reinstalling the qualified GPU dependencies:

```bash
python -m pip install -e '.[dev]'
RC="$?"
echo "install SpecRhythm implementation rc=$RC"
```

Create and freeze a new root. This verifies exact Git/source identities and required input files; it does not initialize CUDA.

```bash
export SR_PHASE4B3_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/$SR_PHASE4B_COMMIT/phase4b3-$(date -u +%Y%m%dT%H%M%SZ)-$$"
python - <<'PY'
import hashlib, json, os, pathlib, subprocess
assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == os.environ["SR_PHASE4B_COMMIT"]
assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
assert subprocess.check_output(["git", "-C", os.environ["SR_VLLM_SOURCE"], "rev-parse", "HEAD"], text=True).strip() == "752a3a504485790a2e8491cacbb35c137339ad34"
names = ("SR_PHASE4B_CONFIG", "SR_PHASE4B_ENVIRONMENT", "SR_PHASE4B_TOPOLOGY",
         "SR_PHASE4B_PATCH_MANIFEST", "SR_PHASE4B3_WORKLOAD5", "SR_PHASE4B3_REFERENCE5",
         "SR_PHASE4B_WORKLOAD", "SR_PHASE4B_REFERENCE")
inputs = {name: {"path": os.environ[name], "sha256": hashlib.sha256(pathlib.Path(os.environ[name]).read_bytes()).hexdigest()} for name in names}
root = pathlib.Path(os.environ["SR_PHASE4B3_ROOT"])
root.mkdir(parents=True, exist_ok=False)
with (root / "fixed-inputs.json").open("x") as handle:
    json.dump({"execution_commit": os.environ["SR_PHASE4B_COMMIT"], "inputs": inputs,
               "uuid_mode": os.environ.get("SR_PHASE4_DUAL_UUID_QUERY_MODE", "live")}, handle, indent=2)
print(root)
PY
RC="$?"
echo "freeze implementation and corrected workloads rc=$RC"
```

Check the existing five-patch state, then all Draft adapter source hashes. No new patch is applied.

```bash
python integrations/vllm/manage_patch.py check \
  --vllm-root "$SR_VLLM_ROOT" --source "$SR_VLLM_SOURCE" \
  --expect-state patched --manifest "$SR_PHASE4B3_ROOT/installed-patch-check.json"
RC="$?"
echo "existing five-patch check rc=$RC"
```

```bash
python - <<'PY'
import json, os, pathlib
from specrhythm.phase4.vllm_draft_worker import audit_installed_api
from specrhythm.phase4.batched_draft_service import write_immutable_report
report = audit_installed_api()
write_immutable_report(pathlib.Path(os.environ["SR_PHASE4B3_ROOT"]) / "draft-api-check.json", report)
print(json.dumps({"source": report["vllm_commit"], "checked_files": len(report["files"])}))
PY
RC="$?"
echo "Draft internal API source guards rc=$RC"
```

## D1 — construction, one small request

The gate requires explicit `--allow-gpu`. It creates one model/worker/cache, materializes one small prefix, releases it and shuts down. Inspect `D1/draft-startup.json`, `draft-backend-report.json`, `gate.json` and the log. Require actual GPU0 UUID binding, independent world1, MRV1, correct model/dtype, initialized cache, successful cleanup and `gate.valid=true`.

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
  SR_PHASE4_DRAFT_BACKEND=vllm-batched \
  python -m specrhythm.phase4.draft_gate --allow-gpu \
    --config "$SR_PHASE4B_CONFIG" --gate D1 \
    --output "$SR_PHASE4B3_ROOT/D1" >"$SR_PHASE4B3_ROOT/D1.log" 2>&1
RC="$?"
echo "D1 construction/materialization/cleanup rc=$RC"
```

Also check the actual Unix Draft service endpoint, in a separate fresh process after the construction gate has shut down. This checks service startup/dispatch, not a second simultaneous model. The helper owns the PID it launches; retain it until it has been reaped.

```bash
export SR_PHASE4B3_D1_SOCKET="/tmp/sr-phase4b3-D1-$$.sock"
mkdir "$SR_PHASE4B3_ROOT/D1-service"
RC="$?"
echo "D1 fresh service directory rc=$RC"
```

```bash
SR_PHASE4_DRAFT_BACKEND=vllm-batched phase4b1_start_draft serial \
  "$SR_PHASE4B3_ROOT/D1-service" "$SR_PHASE4B3_D1_SOCKET"
RC="$?"
echo "D1 actual Draft service startup rc=$RC"
```

```bash
python - <<'PY'
import json, os, pathlib
from transformers import AutoTokenizer
from specrhythm.phase4.config import load_phase4_config
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.transport import UnixDraftClient
from specrhythm.phase4.batched_draft_service import write_immutable_report
config = load_phase4_config(os.environ["SR_PHASE4B_CONFIG"])
tokenizer = AutoTokenizer.from_pretrained(str(config.draft.resolved_tokenizer_path), revision=config.draft.tokenizer_revision, trust_remote_code=config.draft.trust_remote_code)
prefix = tokenizer.encode("Continue counting: 1, 2, 3,")
root = pathlib.Path(os.environ["SR_PHASE4B3_ROOT"]) / "D1-service"
client = UnixDraftClient(pathlib.Path(os.environ["SR_PHASE4B3_D1_SOCKET"]))
try:
    initialized = client.call("initialize", {"request_id": "construction", "committed_token_ids": prefix, "committed_prefix_hash": token_prefix_hash(prefix)})
    finished = client.call("finish_request", {"request_id": "construction"})
finally:
    shutdown = client.shutdown()
ready = json.loads((root / "draft-service-ready.json").read_text())
evidence = json.loads((root / "draft-backend-report.json").read_text())
assert ready["backend"] == "vllm-batched-paged-kv-draft"
assert evidence["execution_failed"] is False and evidence["backend_shutdown_complete"] is True
assert evidence["worker_resources"]["live_allocator_requests"] == 0
write_immutable_report(root / "service-result.json", {"valid": True, "initialize": initialized, "finish": finished, "shutdown": shutdown})
print("D1 service initialized one request, released it and shut down")
PY
RC="$?"
echo "D1 service initialization, release and final evidence rc=$RC"
```

After successful shutdown, reap the owned service process:

```bash
wait "$PHASE4B1_DRAFT_PID"
RC="$?"
echo "D1 owned service process reaped rc=$RC"
```

If initialization or shutdown failed, retain the log and run this recovery block to stop/reap that owned process before retrying in a fresh directory. Do not continue to D2 after a nonzero rc.

```bash
phase4b_terminate_and_wait "$PHASE4B1_DRAFT_PID"
RC="$?"
echo "D1 failed service owned-process cleanup rc=$RC"
```

## D2 — single-request semantics

This uses an HF oracle model first and destroys it **before** constructing the vLLM model. It compares proposals at identical prefixes through zero/partial/full acceptance, correction, bonus, multiple rounds, EOS and output-limit termination. EOS tests intentionally use a recorded synthetic stop-token policy to guarantee the branch is exercised; they are correctness fixtures, not serving workloads. The final round consumes the exact remaining output budget. No Target GPU participates in D1–D3.

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
  SR_PHASE4_DRAFT_BACKEND=vllm-batched \
  python -m specrhythm.phase4.draft_gate --allow-gpu \
    --config "$SR_PHASE4B_CONFIG" --gate D2 \
    --output "$SR_PHASE4B3_ROOT/D2" >"$SR_PHASE4B3_ROOT/D2.log" 2>&1
RC="$?"
echo "D2 HF/vLLM same-prefix proposal and KV semantics rc=$RC"
```

Any proposal mismatch stops the gate and preserves both token lists and context fixtures. Do not classify a same-prefix Draft mismatch as qualified final-Target divergence. The gate is not a timing benchmark.

## D3 — heterogeneous B=2/4/8

Each gate runs requests with different prompt/committed lengths and remaining budgets, a shrinking EOS subset, and batched commit tails. Require exact identity-mapped proposals and a real multi-request **proposal forward**, not merely a batched setup call. Inspect per-purpose histograms as well as `gate.valid`.

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
  SR_PHASE4_DRAFT_BACKEND=vllm-batched \
  python -m specrhythm.phase4.draft_gate --allow-gpu \
    --config "$SR_PHASE4B_CONFIG" --gate D3 --request-count 2 \
    --output "$SR_PHASE4B3_ROOT/D3-B2" >"$SR_PHASE4B3_ROOT/D3-B2.log" 2>&1
RC="$?"
echo "D3 B2 true batching rc=$RC"
```

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
  SR_PHASE4_DRAFT_BACKEND=vllm-batched \
  python -m specrhythm.phase4.draft_gate --allow-gpu \
    --config "$SR_PHASE4B_CONFIG" --gate D3 --request-count 4 \
    --output "$SR_PHASE4B3_ROOT/D3-B4" >"$SR_PHASE4B3_ROOT/D3-B4.log" 2>&1
RC="$?"
echo "D3 B4 true batching rc=$RC"
```

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
  SR_PHASE4_DRAFT_BACKEND=vllm-batched \
  python -m specrhythm.phase4.draft_gate --allow-gpu \
    --config "$SR_PHASE4B_CONFIG" --gate D3 --request-count 8 \
    --output "$SR_PHASE4B3_ROOT/D3-B8" >"$SR_PHASE4B3_ROOT/D3-B8.log" 2>&1
RC="$?"
echo "D3 B8 true batching rc=$RC"
```

## D4 — corrected-five Serial pair

Use the unchanged qualified resident/performance helper so bootstrap deferral, measurement boundaries, raw validation, matched-work policy and owned-process cleanup remain identical. Only Draft backend selection and fresh output paths differ. Validate semantics first; do not interpret five-request timing as the performance milestone.

```bash
SR_PHASE4_DRAFT_BACKEND=hf-persistent phase4b2_run_mode serial \
  "$SR_PHASE4B3_ROOT/D4/hf" "$SR_PHASE4B3_WORKLOAD5" 5 "$SR_PHASE4B3_REFERENCE5"
RC="$?"
echo "D4 HF Serial execution and cleanup rc=$RC"
```

```bash
phase4b2_measure_mode serial "$SR_PHASE4B3_ROOT/D4/hf" "$SR_PHASE4B3_WORKLOAD5"
RC="$?"
echo "D4 HF offline validity/accounting rc=$RC"
```

```bash
SR_PHASE4_DRAFT_BACKEND=vllm-batched phase4b2_run_mode serial \
  "$SR_PHASE4B3_ROOT/D4/vllm" "$SR_PHASE4B3_WORKLOAD5" 5 "$SR_PHASE4B3_REFERENCE5"
RC="$?"
echo "D4 batched Serial execution and cleanup rc=$RC"
```

```bash
phase4b2_measure_mode serial "$SR_PHASE4B3_ROOT/D4/vllm" "$SR_PHASE4B3_WORKLOAD5"
RC="$?"
echo "D4 batched offline validity/accounting rc=$RC"
```

```bash
python -m specrhythm.phase4.draft_comparison \
  --hf "$SR_PHASE4B3_ROOT/D4/hf" --vllm "$SR_PHASE4B3_ROOT/D4/vllm" \
  --request-count 5 --output "$SR_PHASE4B3_ROOT/D4/comparison.json"
RC="$?"
echo "D4 matched work, same-prefix Draft equivalence and batch evidence rc=$RC"
```

## D5 — fresh corrected-100 Serial pair

Require D1–D4 success first. Recheck frozen input hashes before the pair:

```bash
python - <<'PY'
import hashlib, json, os, pathlib, subprocess
frozen = json.loads((pathlib.Path(os.environ["SR_PHASE4B3_ROOT"]) / "fixed-inputs.json").read_text())
assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == frozen["execution_commit"]
for name, row in frozen["inputs"].items():
    assert os.environ[name] == row["path"]
    assert hashlib.sha256(pathlib.Path(row["path"]).read_bytes()).hexdigest() == row["sha256"], name
assert os.environ.get("SR_PHASE4_DUAL_UUID_QUERY_MODE", "live") == frozen["uuid_mode"]
for name in ("D1", "D2", "D3-B2", "D3-B4", "D3-B8"):
    assert json.loads((pathlib.Path(os.environ["SR_PHASE4B3_ROOT"]) / name / "gate.json").read_text())["valid"] is True
assert json.loads((pathlib.Path(os.environ["SR_PHASE4B3_ROOT"]) / "D1-service/service-result.json").read_text())["valid"] is True
assert json.loads((pathlib.Path(os.environ["SR_PHASE4B3_ROOT"]) / "D4/comparison.json").read_text())["ready_for_operator_review"] is True
print("D5 inputs and preceding gates valid")
PY
RC="$?"
echo "D5 frozen-input and prerequisite check rc=$RC"
```

```bash
SR_PHASE4_DRAFT_BACKEND=hf-persistent phase4b2_run_mode serial \
  "$SR_PHASE4B3_ROOT/D5/hf" "$SR_PHASE4B_WORKLOAD" 100 "$SR_PHASE4B_REFERENCE"
RC="$?"
echo "D5 HF Serial execution and cleanup rc=$RC"
```

```bash
phase4b2_measure_mode serial "$SR_PHASE4B3_ROOT/D5/hf" "$SR_PHASE4B_WORKLOAD"
RC="$?"
echo "D5 HF offline measurement rc=$RC"
```

```bash
SR_PHASE4_DRAFT_BACKEND=vllm-batched phase4b2_run_mode serial \
  "$SR_PHASE4B3_ROOT/D5/vllm" "$SR_PHASE4B_WORKLOAD" 100 "$SR_PHASE4B_REFERENCE"
RC="$?"
echo "D5 batched Serial execution and cleanup rc=$RC"
```

```bash
phase4b2_measure_mode serial "$SR_PHASE4B3_ROOT/D5/vllm" "$SR_PHASE4B_WORKLOAD"
RC="$?"
echo "D5 batched offline measurement rc=$RC"
```

```bash
python -m specrhythm.phase4.draft_comparison \
  --hf "$SR_PHASE4B3_ROOT/D5/hf" --vllm "$SR_PHASE4B3_ROOT/D5/vllm" \
  --request-count 100 --output "$SR_PHASE4B3_ROOT/D5/comparison.json"
RC="$?"
echo "D5 completed-work and batched-backend evidence rc=$RC"
```

The comparator separates three questions: exact Draft proposals on common committed contexts; final Target trajectory equality (diagnostic under the established qualified policy); and performance comparability under equal completed work. A mismatch in comparable Draft proposals remains visible and blocks the ready-for-review gate. Contexts that diverged on Target are reported as unmatched, not falsely treated as Draft equivalence evidence. D2–D3 provide the controlled same-prefix oracle checks.

Require 100 completions, identical requested budgets, matching per-request committed-token accounting, valid raw/lifecycle artifacts and a Draft forward histogram whose p50 is greater than one. If workload conditions prevent that p50, retain the result for explicit review; the backend name cannot certify batching. Report total actual Draft forwards (setup/warm-up separately), GPU event time, proposal/commit distribution, scoped host syncs, warm-up duration/count and original JIT evidence. The old ≈2027 HF calls are a derived reference; the comparator derives the current HF count from its actual round records.

The operator decides whether the measured reduction from the approximately 49.3 s HF Serial regime is material. There is no mandatory numerical speedup threshold. A first pair is exploratory; repeat with fresh roots/pairs and alternating order under an agreed server budget before drawing a stable performance conclusion. D6/Dual remains a separate implementation task even if D5 succeeds. Do not change Dual microbatch=2, accumulation, verification batches or rhythm policy here.
