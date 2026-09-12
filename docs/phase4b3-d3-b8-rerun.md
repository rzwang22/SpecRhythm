# D3-B8 diagnostic rerun only

Run this manually on the same qualified A800 environment. The coding agent did not run GPU or connect to AutoDL. This is evidence collection, not a correctness fix or performance experiment. Do not run D4/D5 or rerun D1/D2/B2/B4. Preserve the old failed artifacts.

Use the full delivered diagnostic commit from the accompanying response on the existing Draft PR #4. Check out that commit and install only SpecRhythm (`python -m pip install -e .`); keep the qualified Python 3.11 / Torch 2.11.0 / vLLM 0.25.1 environment and existing five patches unchanged. Keep the previous `SR_PHASE4B_CONFIG` and model/tokenizer paths. The config SHA256 must remain `ef0d25a2f9d8ef73f54af4407e36515bfc9ad775bbf287089fe31570db9724a3`.

In an interactive Bash shell, set `SR_PHASE4B_COMMIT` to the **full diagnostic SHA**, then run this CPU-only preflight. If any block prints nonzero `rc`, stop manually and retain its output. The blocks do not terminate your interactive shell.

```bash
set +e
set +u
set +o pipefail
trap - ERR
export SR_PHASE4B3_D3_DEBUG_ROOT="/root/autodl-tmp/SpecRhythm-data/results/phase4/$SR_PHASE4B_COMMIT/D3-B8-debug-$(date -u +%Y%m%dT%H%M%SZ)-$$"
python - <<'PY'
import hashlib, json, os, pathlib, subprocess
from specrhythm.phase4.draft_d3_diagnostics import D3MaterializationDiagnostic
commit = os.environ['SR_PHASE4B_COMMIT']
assert len(commit) == 40
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == commit
assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()
config = pathlib.Path(os.environ['SR_PHASE4B_CONFIG'])
digest = hashlib.sha256(config.read_bytes()).hexdigest()
assert digest == 'ef0d25a2f9d8ef73f54af4407e36515bfc9ad775bbf287089fe31570db9724a3'
root = pathlib.Path(os.environ['SR_PHASE4B3_D3_DEBUG_ROOT'])
root.mkdir(parents=True, exist_ok=False)
with (root / 'run-inputs.json').open('x') as handle:
    json.dump({'execution_commit': commit, 'config_path': str(config),
               'config_sha256': digest, 'diagnostic_only': True,
               'baseline_commit': '8709c81e9ab2ded78e8f260cde4862b73e9b855a'}, handle, indent=2)
print('fresh diagnostic root:', root)
PY
RC="$?"
echo "D3-B8 preflight rc=$RC"
```

Only after `rc=0`, launch the single D3-B8 diagnostic gate. This gate creates its own fresh `gate` subdirectory. Its HF oracle runs and is destroyed before the vLLM Draft model is constructed; it does not rerun D2 or load a Target model.

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=1 \
  SR_PHASE4_DRAFT_BACKEND=vllm-batched \
  python -m specrhythm.phase4.draft_gate \
    --allow-gpu --diagnostic --gate D3 --request-count 8 \
    --config "$SR_PHASE4B_CONFIG" \
    --output "$SR_PHASE4B3_D3_DEBUG_ROOT/gate" \
    >"$SR_PHASE4B3_D3_DEBUG_ROOT/run.log" 2>&1
RC="$?"
echo "D3-B8 diagnostic rc=$RC"
tail -n 30 "$SR_PHASE4B3_D3_DEBUG_ROOT/run.log"
```

Retain the directory on either success or failure. `gate/gate.json` must still require exact proposals. The new `gate/draft-materialization-diagnostic.json` contains the per-materialization records. Return it together with the oracle, startup, backend and gate reports and `run.log`; packaging the entire fresh root is sufficient:

```bash
tar -czf "$SR_PHASE4B3_D3_DEBUG_ROOT.tar.gz" \
  -C "$(dirname "$SR_PHASE4B3_D3_DEBUG_ROOT")" "$(basename "$SR_PHASE4B3_D3_DEBUG_ROOT")"
RC="$?"
echo "package D3-B8 diagnostic rc=$RC"
```

The first `round=2, purpose=proposal` frame is the main investigation target; all preceding frames are also needed. Compare the new HF oracle with the retained baseline oracle, without changing or relaxing the comparison. If an assertion fires before the model forward, the report retains the offending metadata. An early startup failure may produce only `run.log`; return that too.

If this instrumented gate passes, stop and review its trace. The GPU reads perturb timing, so a diagnostic pass is not a qualification pass or proof that the original fault is fixed. A fresh **uninstrumented D3-B8 only** run is needed after diagnosis/fix (same command without `--diagnostic`, using another fresh root). Only after B8 is qualified should retained D1/D2/B2/B4 artifacts be combined with it in a compact offline D1–D3 validator. D1 and B2 artifacts were not in the supplied debug archive and must be retained from their original runs. Do not proceed to D4 automatically.
