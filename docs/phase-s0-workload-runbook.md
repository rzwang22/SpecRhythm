# Phase S0 CPU data acceptance

Run this on the real server after checking out the delivered S0 commit on
`codex/vllm-serving-v0.1`. It needs CPU, disk and tokenizer files only. It does not
need three A800s or start Target/Serial/Dual/PingPong. 1000 is the dataset size,
not a simultaneous KV residency setting. The agent's Mac build does not certify
the two server model directories. Keep PR #4 Draft; stop after data review.

All blocks below run sequentially in **one Bash shell** from the repository root.
Set `SR_S0_EXPECTED_COMMIT` to the full SHA in the delivery report. Other variables
may be overridden before the first block. The default paths below are the
confirmed server paths. Use a fresh run ID; existing sources/cache are reused,
but previous build/report directories are never overwritten or cleared.

## 1. Commit, CPU environment, cache and tokenizer preflight

```bash
set -euo pipefail
: "${SR_S0_EXPECTED_COMMIT:?Set this to the delivered full S0 commit SHA}"
test "$(git branch --show-current)" = codex/vllm-serving-v0.1
test "$(git rev-parse HEAD)" = "$SR_S0_EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
python3.11 --version

export SR_S0_CACHE="${SR_S0_CACHE:-/root/.cache/huggingface}"
export SR_S0_DRAFT="${SR_S0_DRAFT:-/root/autodl-tmp/models/Qwen3-0.6B}"
export SR_S0_TARGET="${SR_S0_TARGET:-/root/autodl-tmp/models/Qwen3-32B}"
export SR_S0_RESULTS="${SR_S0_RESULTS:-/root/autodl-tmp/SpecRhythm-data/results/phase-s0}"
export SR_S0_RUN_ID="${SR_S0_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
export SR_S0_RUN="$SR_S0_RESULTS/$SR_S0_RUN_ID"
export SR_S0_SOURCES="${SR_S0_SOURCES:-$SR_S0_RESULTS/locked-sources-v1}"
export SR_S0_CONFIG="$PWD/configs/workloads/mixed-real-v1.json"
export SR_S0_LOCK="$PWD/configs/workloads/mixed-real-v1-source-lock.json"
test ! -e "$SR_S0_RUN"
mkdir -p "$SR_S0_CACHE" "$SR_S0_RESULTS"
mkdir "$SR_S0_RUN"
python3.11 - <<'PY'
import os
import tempfile
from pathlib import Path
with tempfile.TemporaryFile(dir=os.environ['SR_S0_CACHE']) as probe:
    probe.write(b'S0 cache write check')
for key in ('SR_S0_DRAFT', 'SR_S0_TARGET'):
    root = Path(os.environ[key])
    for name in ('tokenizer.json', 'tokenizer_config.json'):
        assert (root / name).is_file(), f'Missing {root / name}'
    print(key, root, 'tokenizer files present')
PY

# A fresh CPU-only environment, separate from the fixed vLLM/PyTorch environment.
python3.11 -m venv "$SR_S0_RUN/cpu-venv"
export SR_S0_PY="$SR_S0_RUN/cpu-venv/bin/python"
export CUDA_VISIBLE_DEVICES=''
export USE_TORCH=0 USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false
"$SR_S0_PY" -m pip install -e '.[workload]' \
  transformers==4.56.2 tokenizers==0.22.2 huggingface_hub==0.36.2 \
  pyarrow==21.0.0 jinja2==3.1.6 \
  > "$SR_S0_RUN/dependencies.stdout.log" 2> "$SR_S0_RUN/dependencies.stderr.log"
"$SR_S0_PY" -m pip freeze > "$SR_S0_RUN/dependencies.txt"
"$SR_S0_PY" -m specrhythm.serving --help
"$SR_S0_PY" -m specrhythm.serving tokenizer-check \
  --draft-tokenizer "$SR_S0_DRAFT" --target-tokenizer "$SR_S0_TARGET" \
  --output "$SR_S0_RUN/tokenizer-preflight.json" \
  > "$SR_S0_RUN/tokenizer-preflight.stdout.log" \
  2> "$SR_S0_RUN/tokenizer-preflight.stderr.log"
```

Preflight compares actual vocabulary IDs and special-token mappings, without
loading weights. `status=READY` means that prompt encoding acceptance remains
pending. The full 1200-prompt check follows construction. Do not modify token IDs,
templates or the source lock to force a mismatch through validation.

## 2. Fetch or import the locked sources

The standard command downloads only missing fixed-revision source files. It
reuses `/root/.cache/huggingface/hub` and never migrates or clears it.

```bash
"$SR_S0_PY" -m specrhythm.serving source-fetch \
  --source-lock "$SR_S0_LOCK" --sources-dir "$SR_S0_SOURCES" \
  --cache-dir "$SR_S0_CACHE" \
  > "$SR_S0_RUN/source-fetch.stdout.log" 2> "$SR_S0_RUN/source-fetch.stderr.log"
```

For already downloaded files, use this command **instead**. The import map is a
JSON object from logical lock paths to real absolute files, for example
`{"code/train.jsonl":"/existing-downloads/apps/train.jsonl"}`. Every mapped file
is hash checked; omitted keys use the existing fixed-revision cache. Set
`SR_S0_IMPORT_MAP` to your actual map, and use `--offline` to prohibit downloads.
All 23 files (including attribution and tokenizer files) must be available.

```bash
: "${SR_S0_IMPORT_MAP:?Set to your actual local import map JSON}"
"$SR_S0_PY" -m specrhythm.serving source-fetch \
  --source-lock "$SR_S0_LOCK" --sources-dir "$SR_S0_SOURCES" \
  --cache-dir "$SR_S0_CACHE" --import-map "$SR_S0_IMPORT_MAP" --offline \
  > "$SR_S0_RUN/source-import.stdout.log" 2> "$SR_S0_RUN/source-import.stderr.log"
```

`source-resolve` is also implemented for an explicitly new source audit, not
needed for this acceptance. It requires a fresh `--output`, `--sources-dir`,
`--cache-dir`, and optionally `--revisions` containing exact commits keyed by
`chat`, `code`, `summarization`, `reasoning`, `arrival`, `tokenizer`. Ordinary
acceptance uses the committed lock and never silently updates it.

## 3. Build the real main and calibration sets

```bash
"$SR_S0_PY" -m specrhythm.serving build \
  --config "$SR_S0_CONFIG" --source-lock "$SR_S0_LOCK" \
  --sources-dir "$SR_S0_SOURCES" --tokenizer-path "$SR_S0_DRAFT" \
  --output "$SR_S0_RUN/build-a" \
  > "$SR_S0_RUN/build-a-cli.stdout.log" 2> "$SR_S0_RUN/build-a-cli.stderr.log"
```

This verifies sources and the configured tokenizer against the fixed tokenizer,
builds exact quotas, and runs the full independent validator. It scans the full
candidate pools and can take several CPU minutes; there is no model inference.
Candidate exhaustion/missing data produces a failure report, never replacement
prompts. A failed directory is retained; correct the identified input issue and
use a fresh run directory.

## 4. Full read-only validation and server tokenizer acceptance

```bash
"$SR_S0_PY" -m specrhythm.serving validate \
  --run-root "$SR_S0_RUN/build-a" --sources-dir "$SR_S0_SOURCES" \
  --config "$SR_S0_CONFIG" --source-lock "$SR_S0_LOCK" \
  --tokenizer-path "$SR_S0_DRAFT" \
  --output "$SR_S0_RUN/build-a/independent-validation.json" \
  > "$SR_S0_RUN/validate-a.stdout.log" 2> "$SR_S0_RUN/validate-a.stderr.log"
"$SR_S0_PY" -m specrhythm.serving tokenizer-check \
  --draft-tokenizer "$SR_S0_DRAFT" --target-tokenizer "$SR_S0_TARGET" \
  --run-root "$SR_S0_RUN/build-a" \
  --output "$SR_S0_RUN/build-a/server-tokenizer-validation.json" \
  > "$SR_S0_RUN/tokenizer-a.stdout.log" 2> "$SR_S0_RUN/tokenizer-a.stderr.log"
```

Require `valid=true`, `real_data_validation=PASS`, and `inputs_unchanged=true`
from full validation. The separate server tokenizer report must have
`status=PASS`, `validated_prompt_count=1200`, identical Draft/Target identity,
and bindings to the exact workload files/manifest. Its status does not rewrite
the original build's PENDING field. Both reports are necessary to claim server
data acceptance; a Mac remote-tokenizer build alone does not suffice.

## 5. Rebuild in a different directory and compare

```bash
"$SR_S0_PY" -m specrhythm.serving build \
  --config "$SR_S0_CONFIG" --source-lock "$SR_S0_LOCK" \
  --sources-dir "$SR_S0_SOURCES" --tokenizer-path "$SR_S0_DRAFT" \
  --output "$SR_S0_RUN/build-b" \
  > "$SR_S0_RUN/build-b-cli.stdout.log" 2> "$SR_S0_RUN/build-b-cli.stderr.log"
"$SR_S0_PY" -m specrhythm.serving rebuild-check \
  --left "$SR_S0_RUN/build-a" --right "$SR_S0_RUN/build-b" \
  --output "$SR_S0_RUN/build-a/rebuild-check.json" \
  > "$SR_S0_RUN/rebuild-check.stdout.log" 2> "$SR_S0_RUN/rebuild-check.stderr.log"
cmp "$SR_S0_RUN/build-a/main1000.jsonl" "$SR_S0_RUN/build-b/main1000.jsonl"
cmp "$SR_S0_RUN/build-a/calibration200.jsonl" "$SR_S0_RUN/build-b/calibration200.jsonl"
cmp "$SR_S0_RUN/build-a/workload-manifest.json" "$SR_S0_RUN/build-b/workload-manifest.json"
```

Build-b includes the full independent validator. Both JSONL files and core
manifest must match; build-info, timestamps, host paths and logs may differ.

## 6. Inspect the data and twenty real prompts

```bash
"$SR_S0_PY" -m specrhythm.serving inspect --run-root "$SR_S0_RUN/build-a"
"$SR_S0_PY" -m json.tool "$SR_S0_RUN/build-a/dataset-summary.json"
"$SR_S0_PY" -m json.tool "$SR_S0_RUN/build-a/independent-validation.json"
"$SR_S0_PY" -m json.tool "$SR_S0_RUN/build-a/server-tokenizer-validation.json"
cat "$SR_S0_RUN/build-a/sample-review.md"
```

`inspect` labels itself inspection-only. Review the five prompts per class for
source fidelity and task plausibility. The summary contains input lengths,
generation caps and arrival statistics, not measured output, TTFT/TPOT or speedup.

## 7. Seal checksums and the small review bundle

Copy completed CLI/dependency reports before sealing so they are included in the
inventory. Originals and existing output artifacts are not overwritten. Build-b
provenance/validation is included in the build-a review bundle; its duplicate
JSONL data need not be transmitted twice.

```bash
"$SR_S0_PY" - <<'PY'
import os
import shutil
from pathlib import Path
root = Path(os.environ['SR_S0_RUN'])
target = root / 'build-a'
for path in sorted(root.iterdir()):
    if path.is_file():
        with (target / ('run-' + path.name)).open('xb') as out, path.open('rb') as inp:
            shutil.copyfileobj(inp, out)
for name in ('build-info.json', 'validation.json'):
    with (target / ('build-b-' + name)).open('xb') as out:
        out.write((root / 'build-b' / name).read_bytes())
PY
"$SR_S0_PY" -m specrhythm.serving seal \
  --run-root "$SR_S0_RUN/build-a" --bundle "$SR_S0_RUN/s0-review.tar.gz" \
  > "$SR_S0_RUN/seal-a.json"
"$SR_S0_PY" -m specrhythm.serving seal --run-root "$SR_S0_RUN/build-b" \
  > "$SR_S0_RUN/seal-b.json"
(cd "$SR_S0_RUN/build-a" && sha256sum -c checksums.sha256)
(cd "$SR_S0_RUN/build-b" && sha256sum -c checksums.sha256)
sha256sum "$SR_S0_RUN/s0-review.tar.gz"
```

The bundle contains `source-lock.json`, `workload-config.json`, `main1000.jsonl`,
`calibration200.jsonl`, `workload-manifest.json`, `validation.json`,
`selection-report.json`, `dataset-summary.json`, `sample-review.md`,
`build-info.json`, `stdout.log`, `stderr.log`, additional acceptance reports/logs,
and `checksums.sha256`. The inventory covers all top-level files present at seal
time except itself; the later external seal report is not self-included. Original
datasets, cache, CPU venv and model weights are excluded from the tar.

Return `s0-review.tar.gz` and its SHA256 after CPU acceptance and sample review.
Do not run S1, any serving runner, correctness or performance GPU experiment for
this task. Subsequent stages need separate authorization after S0 data acceptance.
