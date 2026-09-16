# A100 K3 B64 performance exploration

This is the explicit `k3-b64-v1` configuration, not legacy fixed64 or prepost3/P1–P4.
The B16 entry and `k3-b16-v1` default remain unchanged. Execution SHA: `899b54a6b58c0ea07046582c5a17934f630ac040`.
Repository fixed entry: `scripts/run_k3_b64_pinned.sh`; it pins that implementation
and creates a fresh detached worktree before executing `scripts/run_k3_b64.sh`.
No GPU execution has been performed by the local agent.

| Mode | Active | Immutable homes | Target requests/forward | Draft physical ceiling |
|---|---:|---|---:|---:|
| serial-k3 | 64 | A64 | 64 | 64 |
| serial-eager-k3 | 64 | A64 | 64 | 64 |
| pingpong-k3 | 64 | A32/B32 | 32 | 64 |
| pingpong-eager-k3 | 64 | A32/B32 | 32 | 64 |

`geometry(mode, configuration)` supplies this table to planning, capacity, controller,
owner, scheduler, warmup and native qualification. The CLI requires
`prepare --k3-configuration k3-b64-v1`; the manifest, point, control snapshot and
capacity retain the declaration. Missing declarations mean legacy B16 only; they cannot
qualify a B64 point. There is no automatic downgrade if capacity is insufficient.

The default validation policy is `performance-exploration`, separate from execution
geometry `k3-b64-v1`. The policy is written to `validation-plan.json`, scan-config,
execution manifests, selected points, runtime, summaries, diagnostic status,
comparison and archive inventory. A missing plan retains the old strict policy;
it never silently selects performance exploration. B16 remains strict by default.

The foreground entry runs exactly:

1. Static B64 interface/capacity arithmetic check (no GPU PASS).
2. Four capacity points, with all360 real prefills and normal drain.
3. Four single runtime performance points, with resident360, unchanged seed1666,
   two warmup units, continuous30s window, setup900s/drain60s, no retries or grid.
4. Each point must pass capacity/device, execution, measurement, cleanup, K3/KV/
   ownership/accounting and diagnostic-integrity checks before the next point.
   Its own warmup/runtime native records must contain one actual distinct64-request
   forward for **each** Serial mode, distinct32 for each PingPong mode, both TP ranks,
   both PingPong homes and bounded active/home occupancy. Multiple smaller forwards
   cannot satisfy this proof. Measured partial batches retain their actual distribution
   and links to population/owner unfilled/deferred reasons.
5. Comparison and exactly one content-deduplicated delivery archive.

No Target-only or independent full-output diagnostic is launched in this default flow.
`full_output_comparison_run=false` and `output_equivalence_status=NOT_RUN` are deliberate,
not PASS and not execution failure. Reported throughput/overlap carries the statement
**this run has not verified complete output equivalence**. `measurement_valid` and
`native_geometry_status` are separate. The legacy `formal_comparison_eligible` field
already means run execution/measurement/cleanup eligibility; it has never certified
joint output equivalence. Its value is not forged or relaxed for this policy.

The historical B64 archive `pingpong-k3-delivery-a100-k3-b64-20260915T175251Z-343.tar.gz`
(SHA256 `d9e31c43f66d501451823180638b66e31c0e84a92ac932be2e01b4b47eaa1606`,
execution `f6f67aa1e1d7aea2a81665ec628d0ae857c148ee`) remains a failed strict comparison:
Serial and Serial-eager each54/64 exact with the same10 differing requests; both
PingPong modes64/64 exact. All four formal performance points were NOT_RUN.
Those discrepancies are unresolved; this policy does not change their data or status.

Warmup counts actual request-verification opportunities, not calls: at least128,
with per-ID coverage and the start boundary retained. At full batch this means two
Serial steps or four PingPong steps. Partial batches accumulate their actual size.
Every full request proposal is K3 (seed plus two expansions), with the existing
no-bonus protocol and EOS/output-budget exceptions. READY order, idle gate,
feedback priority, fences, runtime audit and buffered-live recording are unchanged.

Mutable files remain in `/tmp/specrhythm-runs/<unique-tag>/` (container overlay on
the observed server; not a claimed NVMe device). The local archive is copied to
`/root/autodl-tmp/SpecRhythm-data/results/rolling-eager/` via a unique temporary
file, then reread for size/SHA256 before publication. Copy failure retains and
prints the valid local archive. Keep local evidence; do not delete it after failure.
The outer entry prints exactly one `UPLOAD ONLY:` path when an archive exists.
Original execution/qualification exit code, export code and copy code remain separate.

The B64 static receipt explicitly authorizes offline archive byte limits of2GiB per
source file and4GiB unique payload (512 logical/unique file caps unchanged). B16 stays
512MiB/1GiB. This is storage capacity adaptation only: previous verified B16 unique
payload was912,070,238 bytes, and the largest source was150,886,304 bytes. Producer
buffers, dropped-row checks and logging policy are unchanged. A malformed receipt
retains raw bytes and an evidence error under the old bounded fallback limits; it
never creates a synthetic COMPLETE/PASS. Exceeding any bound remains explicit failure.

Upload only `pingpong-k3-delivery-<tag>.tar.gz`. It includes comparison, validation plan, first failure when present,
raw native/request/proposal evidence, capacity ranks and budgets,
B64 execution manifests, original errors/cleanup, inventory size/SHA256, and export
status. For exploration the inventory explicitly declares `joint/` intentionally
not run. Missing required performance evidence still fails; missing/truncated raw
records never become zero/PASS. No separate JSON/log collection commands are needed.

Read `request_verification_opportunities` as Σ actual Target B in the measured
window. `tokens_per_request_opportunity` uses that denominator; `window_ms_per_active_opportunities`
is actual window_ms×64/ΣB. `window_average_cadence_ms` is window_ms/steps;
`complete_step_wall_ms` covers completed engine steps, while `outside_complete_steps_ms`
retains the remaining window. These are different scopes, not additive phase costs.
Raw native rank batch/durations, owner feedback→READY, READY→claim, claim→GPU,
scheduler subspans, recovery/eager forwards and overlap remain in each audit report.

Comparison reports SE/S, P/S, PE/P and PE/SE within this one B64 execution. Native
cross-home and cross-request unions remain separate from own-parent eager overlap.
Recovery coverage uses the union of relevant physical recovery intervals inside
the window, intersected with other-request Target intervals; TP/mixed-role overlaps
are not summed twice. See each `pipeline` field for exact denominator and bounds.
A nonzero OBSERVED overlap is not proof that recovery is fully hidden. One window
is a mechanism check, not a stable speedup claim. Old B16/A800 results remain history;
cross-run/hardware differences cannot establish this change's isolated benefit.

New B64 capacity, execution/measurement/cleanup, evidence integrity, native geometry,
overlap and performance remain **PENDING server evidence**. Output equivalence is
**NOT_RUN by default**; a manually requested new strict diagnostic is pending until run.

For the new comparison specifically, inspect `pipeline.cross_home_overlap_steps` and
`pipeline.recovery_coverage_by_other_homes` alongside the existing other-request
metrics. Same-home parallel work must not be labelled A/B pipeline coverage.
`K3_mechanism.lookahead_rates` records the generated-candidate denominator and measured
parent-request scope; fractions are null when no lookahead was generated.

## Foreground command pinned to the implementation

This equivalent standalone command reads the actual B64 runner from the immutable
implementation commit. The fixed-entry script in the delivery commit performs the
same worktree/runner selection. All strict mode/exit behavior stays in a child Bash;
the parent `if` keeps the interactive terminal open.

```bash
if bash <<'SR_B64_CHILD'
set -Eeuo pipefail
REPO="${SR_K3_REPO:-/root/autodl-tmp/src/SpecRhythm}"
EXEC_SHA=899b54a6b58c0ea07046582c5a17934f630ac040
git -C "$REPO" fetch origin codex/rolling-eager-v0.1
git -C "$REPO" cat-file -e "${EXEC_SHA}:scripts/run_k3_b64.sh"
RUN_TREE="${REPO}-k3-b64-${EXEC_SHA:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
git -C "$REPO" worktree add --detach "$RUN_TREE" "$EXEC_SHA"
unset SR_K3_MANAGED_LOCAL SR_K3_LOCAL_DELIVERY SR_PING_DELIVERY
export SR_EXEC_REPO="$RUN_TREE"
export SR_K3_VALIDATION_PROFILE="${SR_K3_VALIDATION_PROFILE:-performance-exploration}"
export SR_PING_RUN_TAG="${SR_PING_RUN_TAG:-a100-k3-b64-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
bash "$RUN_TREE/scripts/run_k3_b64.sh" "$EXEC_SHA"
SR_B64_CHILD
then
  printf 'B64 flow finished; return only the archive named by UPLOAD ONLY.\n'
else
  rc=$?
  printf 'B64 stopped (rc=%s); later points stopped; terminal remains open.\n' "$rc"
fi
```

No extra model installation, mount change, CUDA reinstallation, retries or parameter
search is included. Optional existing `SR_FIXED_PYTHON`, `SR_FIXED_S1`, `SR_K3_REPO`,
`SR_K3_LOCAL_BASE` and `SR_PING_RESULTS` overrides retain their previous meaning;
resolved storage paths, filesystem types and free space are recorded by the local
supervisor. Preserve the same workload/model/numerical settings on this A100 host.

## Optional strict diagnostic (explicit operator choice)

Set `SR_K3_VALIDATION_PROFILE=strict-output` in the child Bash above, before invoking
the runner. Use a fresh tag/worktree. This selects the retained capacity → Target-only
plus four64-request complete-output comparisons → performance flow. Comparison failure
still stops all later performance points. Do not reuse an exploration directory.

The profile option uses the same fixed entry, new local directory and single-package
delivery. It preserves the complete32-token-per-request fixture, reference run,
termination comparison and all native checks. It does not add tolerance or modify
sampling. The standalone `specrhythm.serving.k3_gpu_check` command also remains
available for explicit diagnostics; the default runner never calls it.

Aggregate output failures now use `failure_layer=comparison`, `point=comparison`,
`mode=null`, list actual mismatched modes/IDs and reference/source runtime paths,
and mark known inequality FAILED. Per-process exit codes remain separate from the
comparison command's exit1. The last completed eager mode is not blamed for another
mode's mismatch. Historical failure files are never rewritten.

## Independent I/O and dispatch comparison

The new `run_k3_b64_diagnostics_pinned.sh` takes exactly one argument: `io-only` or
`unified`. Both use the same fixed execution commit, deferred-window observation
and performance-exploration. The first keeps original Draft dispatch; the second
also enables runnable-set enforcement and independent promotion publication.
The existing `run_k3_b64_pinned.sh` and B16 entry remain historical controls.

Each invocation performs two repetitions: forward and reverse four-mode orders,
with new roots for every capacity and performance point. No independent complete
output comparison runs; output_equivalence_status remains NOT_RUN. Compare all
repetition values/ranges, not the best window. Scheduling net benefit requires
these two same-I/O runs. The driver does not automatically run both configurations.

The local `/tmp/specrhythm-runs` evidence is retained; only the outer delivery
prints UPLOAD ONLY after local archive validation and verified DPC copy or local
fallback. The single package contains `experiment-plan.json`, both repetitions,
comparison/ranges, raw physical inventories, native timelines, final logging
receipts and first/secondary errors. A failed point stops subsequent points and
repetitions. GPU capacity, cleanup, geometry, overlap and performance remain PENDING
until new server evidence is returned. Historical output differences remain open.
