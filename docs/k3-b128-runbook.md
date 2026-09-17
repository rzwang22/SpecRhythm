# A100 uniform K3 B128 performance exploration

New explicit geometry `k3-b128-v1`: Serial A128/Target128, PingPong A64/B64/Target64,
Draft physical ceiling128. Resident360, unchanged frozen workload SHA256
`cdaf71adace15d229f5087b98f9fd162a958456226a660184fe03f5d6ebd8ff4`, seed1666,
GPU0 Draft and GPU1/2 one TP2 Target. This is not a change to K3, precision,
Target diagnostics, scheduler, READY semantics or KV protocol.

The fixed entry (added in the delivery commit) pins the implementation, creates a
new detached worktree, clears inherited private runner flags, and defaults to
`unified`, `deferred-window`, `performance-exploration`. It also accepts `io-only`
(legacy dispatch/same I/O), and an explicit same-version `b64` rerun. It never
runs both scales automatically. Historical B64 comparisons are cross-run exploration,
not an isolated batch-size effect.

The B128 flow is:

1. Static B128 interface/reservation check, GPU capacity PENDING.
2. Four loaded GPU capacity probes; first insufficiency stops, no smaller fallback.
3. Four modes in forward order, then reverse order, independent fresh roots/engines.
   The second repeat reuses the completed initial probes; every fresh performance
   engine still recomputes actual loaded capacity before resident preparation/decode.
4. At least256 actual request verification opportunities and all128 active IDs
   covered before a continuous30s window. Setup900s/drain60s are unchanged.
5. Native per-rank B128/B64 proof, bounded active/home/Draft counts, runtime K3/KV/
   accounting, execution, measurement, cleanup and diagnostic evidence qualification.
6. Original per-repeat results, ratios and min/max ranges; one deduplicated package.

No Target-only or separate full-output run occurs. Output equivalence is NOT_RUN;
the historical B64 Serial10/64 request mismatch is unresolved. Capacity, evidence,
cleanup and geometry failures still stop subsequent points. A successful measurement
does not certify full output equivalence, hidden recovery or stable speedup.

Mutable files remain under `/tmp/specrhythm-runs/<unique-tag>/`; `/tmp` is container
overlay, not evidence of NVMe. Default persistent directory remains
`/root/autodl-tmp/SpecRhythm-data/results/rolling-eager/` (DPC on the current server).
B128 requires at least32GiB free at startup; buffers/archive limits are unchanged
and fail closed. Space is not reserved. Keep local evidence after success/failure.
A local sealed archive is copied to a unique DPC temporary target, re-read for size
and SHA256, then published. Copy failure retains the locally verified package.
Only the outer supervisor prints one `UPLOAD ONLY:` path. Original execution error,
export/copy errors and statuses remain separate; no package means no invented path.

Expected archive: `pingpong-k3-delivery-a100-k3-b128-unified-<tag>.tar.gz`.
`inventory.json` maps logical paths to digest-addressed objects. It contains the
experiment/validation plans, two comparisons, all eight performance runs, four
initial probes, effective device/diagnostic configuration, initial request IDs,
raw TP/Draft timelines, logging/publication receipts, lifecycle/first failure and
export state. Do not send separate subpackages. Each repeat reports raw ΣB,
throughput,128-opportunity normalization, capture/dispatch/pipeline breakdown;
no best-window selection. GPU capacity/correctness/cleanup/overlap/performance
remain PENDING until operator evidence; full output equivalence stays NOT_RUN.
