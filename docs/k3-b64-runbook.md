# A100 K3 B64 four-mode check

This is the explicit `k3-b64-v1` configuration, not legacy fixed64 or prepost3/P1–P4.
The B16 entry and `k3-b16-v1` default remain unchanged. The pinned entry and complete
execution SHA are filled after local validation and the implementation commit.
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

The foreground entry runs exactly:

1. Static B64 interface/capacity arithmetic check (no GPU PASS).
2. Four capacity points, with all360 real prefills and normal drain.
3. Target-only and four modes, same first64 frozen requests, complete outputs capped
   at32 tokens/request. Compare full tokens and termination reasons by ID. EOS may
   legitimately reduce the nominal2048 tokens. Require one actual distinct64-request
   forward for **each** Serial mode, distinct32 for each PingPong mode, both TP ranks,
   and bounded home occupancy. Multiple smaller forwards cannot satisfy this proof.
4. Four single runtime performance points, with resident360, unchanged seed1666,
   two warmup units, continuous30s window, setup900s/drain60s, no retries or grid.
5. Comparison and exactly one content-deduplicated delivery archive.

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

Upload only `pingpong-k3-delivery-<tag>.tar.gz`. It includes comparison, joint result
or first failure, raw native/request/proposal evidence, capacity ranks and budgets,
B64 execution manifests, original errors/cleanup, inventory size/SHA256, and export
status. No separate JSON/log collection commands are needed.

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

B64 capacity, output correctness, execution/measurement/cleanup, evidence integrity,
native overlap and performance remain **PENDING server evidence**.

For the new comparison specifically, inspect `pipeline.cross_home_overlap_steps` and
`pipeline.recovery_coverage_by_other_homes` alongside the existing other-request
metrics. Same-home parallel work must not be labelled A/B pipeline coverage.
`K3_mechanism.lookahead_rates` records the generated-candidate denominator and measured
parent-request scope; fractions are null when no lookahead was generated.
