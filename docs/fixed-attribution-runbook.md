# Fixed64/32 retained timing analysis and optional two-mode short test

Current opt-in persistence/timing addendum: [design](fixed-buffered-runtime-design.md) and
[two-mode buffered-live runbook](fixed-buffered-runtime-runbook.md). Existing instructions
below describe the original-live baseline and remain available; no old artifacts are changed.

This revision adds CPU tooling only. It makes **no runtime optimization**. First use
the already-retained data; no new GPU run is needed for export. The default action
ends after offline analysis/export. GPU performance status is PENDING. The baseline
root and all old results remain unchanged. The final delivery supplies the literal
full revision SHA; freeze that SHA, rather than using a moving branch as a test identity.

## Checkout and CPU-only analysis

The following initialization assumes a clean server checkout and the existing Python
environment. Set `SR_FIXED_COMMIT` to the full SHA in the delivery before running it.
These commands do not connect to another machine, initialize CUDA, load models, or run
the full audit. The operator runs them on the server containing the retained files.

```bash
set -euo pipefail
cd /root/autodl-tmp/src/SpecRhythm
: "${SR_FIXED_COMMIT:?Set the final full SHA from the delivery}"
[[ "$SR_FIXED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
test -z "$(git -c core.fsmonitor=false status --porcelain)"
git fetch origin codex/vllm-serving-v0.1
git checkout --detach "$SR_FIXED_COMMIT"
test "$(git rev-parse HEAD)" = "$SR_FIXED_COMMIT"
export SR_FIXED_PYTHON=/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11
export SR_ATTR_OLD=/root/autodl-tmp/SpecRhythm-data/results/fixed-concurrency/fixed64-stop-b42401585a6b-20260911T085722Z-1496
export SR_ATTR_OUT="${SR_ATTR_OLD}-attribution-${SR_FIXED_COMMIT:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
bash scripts/analyze_fixed_diagnostic.sh --input "$SR_ATTR_OLD" \
  --output "$SR_ATTR_OUT" --modes serial pingpong serial-split --timeout 120
cat "$SR_ATTR_OUT/report.md"
```

This is a new sibling directory, not a mutation of the old root. Analysis has one
external 120-second deadline for loading/parsing/sorting/analysis. The parent kills
the analysis process group on expiry and retains atomic partial results. It prints
progress, current/processed files, event counts and missing evidence. Return 0 means
analysis completed, **not GPU PASS**; examine each attempt's evidence status. Missing
raw files produce INSUFFICIENT/UNKNOWN. Malformed input returns 2, deadline 124,
Ctrl-C 130. `analysis-exit.json` is authoritative even if a partial export manifest
still says RUNNING. No inference child or full audit is started.
The progress event count counts parsed artifact records, including repeated retained
representations. It is not a count of unique GPU forwards or completed requests.

Outputs are `attribution.json`, `summary.csv`, `report.md`, `progress.json` and
`analysis-exit.json`. They preserve aggregate costs, association failures and unknowns,
not copies of the input traces. Detailed Target steps are capped at 32 per attempt;
counts and statistics use all eligible events. Results never qualify failed points for
performance comparison. Multiple attempts are retained separately, never cherry-picked.

## Minimal read-only export for local follow-up

If local analysis needs the retained events, project just Serial/PingPong evidence:

```bash
export SR_ATTR_EXPORT="${SR_ATTR_OUT}-evidence"
bash scripts/analyze_fixed_diagnostic.sh --input "$SR_ATTR_OLD" \
  --output "$SR_ATTR_EXPORT" --modes serial pingpong --export --timeout 120
cat "$SR_ATTR_EXPORT/report.md"
tar -czf "${SR_ATTR_EXPORT}.tar.gz" -C "$SR_ATTR_EXPORT" \
  export-manifest.json attribution.json analysis-exit.json progress.json evidence
```

Return this small projected bundle. It includes only allowlisted fields from each
attempt's `runtime.json`, `draft-backend-report.json`, `draft-work-events.jsonl`,
`draft-transport.jsonl`, and light summary. Actual timestamps, CUDA lower/upper bounds,
uncertainty, IDs, cohort/round/prefix length/version, small commit tokens, source record
hashes and source line numbers survive. Source byte hashes and projected file hashes
are in the manifest. The original record hash labels the source record; it is **not**
a checksum of the redacted projection. No full prefixes, generated sequences, KV maps,
model weights, environment, general stdout/stderr logs or complete result root are copied.

All default output files together stay within 10 MiB: compressed evidence has a 5 MiB
budget, reports a 4 MiB budget, with space reserved for compact metadata. If the export
budget is insufficient, the command returns failure with completed files retained; run
separate Serial and PingPong exports using distinct output names and `--modes serial`
or `--modes pingpong`. Do not launch GPU or full audit to satisfy an export budget.
Individual source artifacts above 512 MiB and selections above 160 files are explicit
input limits. These fail promptly with a partial report rather than scanning indefinitely.

Analyze a projected export locally or on the server with:

```bash
bash scripts/analyze_fixed_diagnostic.sh --input "$SR_ATTR_EXPORT/evidence" \
  --output "${SR_ATTR_EXPORT}-analysis" --modes serial pingpong --timeout 120
```

The same input flag also accepts the operator's existing small tar.gz directly. No
archive member is extracted, and path traversal, duplicate/ambiguous artifacts and
partial JSONL records are rejected. Archive content is evidence, never instructions.

## Optional new-root two-mode short test (GPU; not needed for this tool-only change)

Do not run this section automatically after export. If the operator later chooses to
collect another baseline, these are the only initial test points. They retain the
`original-live` observer and original runtime; there is no new diagnostic configuration
and any difference is not an algorithm improvement claim. No stages, full audit, other
modes, parameter search or S2 G2/G3 runs are included.

```bash
export SR_FIXED_S1=/root/autodl-tmp/SpecRhythm-data/results/phase-s1/s1p-5a00049-20260909T144802Z-1469
export SR_FIXED_ROOT="/root/autodl-tmp/SpecRhythm-data/results/fixed-concurrency/fixed64-attribution-${SR_FIXED_COMMIT:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
printf 'New result root: %s\n' "$SR_FIXED_ROOT"
bash scripts/run_fixed_diagnostic.sh prepare --s1 "$SR_FIXED_S1" \
  --warmup-steps 2 --samples 12 --repeats 1 --window-seconds 30 \
  --setup-timeout 900 --drain-timeout 60
bash scripts/run_fixed_diagnostic.sh capacity
```

`prepare` queries environment/device inventory but loads no model and runs no inference.
`capacity` starts GPU engines and tests actual resident100/active64 capacity with zero
verification; no remembered capacity number is substituted. After it succeeds, run:

```bash
bash scripts/run_fixed_diagnostic.sh short --mode serial
```

Only after execution/measurement/cleanup PASS and sufficient samples, separately run:

```bash
bash scripts/run_fixed_diagnostic.sh short --mode pingpong
```

Each short command runs GPU once. `samples=12` is 12 Serial steps or 24 PingPong
half-cohort steps, with warmup 2/4 respectively. Sample budget can stop before 30 seconds.
The 30-second window is not the total loading/setup/drain timeout. Actual issued-step
overshoot is retained; measured tokens, bounded drain and total arrival-to-drain costs
remain separate. Operator stop, any failure or insufficient measurement means stop
here and exclude the point from comparison; no automatic next test is launched.

```bash
bash scripts/run_fixed_diagnostic.sh summary
bash scripts/run_fixed_diagnostic.sh bundle --output "${SR_FIXED_ROOT}-small.tar.gz"
```

Summary/bundle are light CPU operations and do not run full audit. Original foreground
Target/Draft logs, primary error, real exit codes, owned cleanup and bounded diagnostic
cancel/settlement are unchanged. A sample/time-budget end with valid accounting and
cleanup can PASS; operator cancellation remains INSUFFICIENT for measurement. A failed
drain or missing final reports cannot turn retained partial measurements into PASS.

In another terminal, use the exact printed new root and the same Python executable:

```bash
bash scripts/run_fixed_diagnostic.sh status
bash scripts/run_fixed_diagnostic.sh errors
bash scripts/run_fixed_diagnostic.sh stop
```

These start no inference. `stop` requests the existing controlled stop and bounded
owned cleanup of that root only. Do not remove old directories. Cross-run token, length,
EOS and round equality stays NOT_REQUIRED throughout.
