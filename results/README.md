# Results directory

Local simulation and GPU benchmark outputs belong here and are ignored by Git. Promote only
small, reviewed calibration profiles or aggregate tables into versioned files.

Simulator-semantics v0.2 comparison reports use schema
`specrhythm.comparison.v3`; individual summaries use
`specrhythm.simulation-summary.v5`. The seven primary modes are `ar`, `serial-sd`, `adaserve`,
`dual-batch`, `dual-eager`, `shaping`, and `specrhythm`; three explicitly named flat proxies are
retained for provenance. Each summary identifies execution, allocator, and eager semantics and
includes overall/per-SLO attainment, goodput, queueing/service/end-to-end latency, and
proposal/token/tree-node accounting.

Phase-1 shaping diagnosis additionally exposes `shaping-feasible`, `shaping-residual`, and
`shaping-feasible-residual`. These are causal diagnostics only and remain outside the default
comparison order. Their summaries include one-cycle feasibility, stage allocation, root/candidate
progress rates, same-state base-preservation checks, and optional full allocation-opportunity
JSONL streams.

Phase 1.5 adds `residual-round-robin`, `residual-probability`, and the `feasible-residual` alias to
isolate residual selection under a frozen Dual-Batch base. Detailed JSON stays outside Git; the
reviewed aggregate report is `docs/phase1.5-residual-selection.md`.

Phase 2 adds separate `phase2-replay` and `phase2-simulate` commands. The four oracle-headroom
variants are deliberately absent from the normal policy registry and cumulative ablation. Common
replay reports use `specrhythm.phase2-common-replay.v1`; compact snapshot JSONL, end-to-end upper
bounds, and other large machine-readable outputs stay in the external data tree. Every Phase-2
output identifies target leakage, fully-hidden-search assumptions, search ratio, requested and
realized pool nodes, and selected verification nodes.

The reviewed external Phase-2 set is complete at 3 common replays, 9 references, and 48 oracle
matrix cells. Machine outputs remain outside Git; only the aggregate tables in
`docs/phase2-oracle-headroom.md` are versioned.

These reports remain local because their latency, acceptance, confidence, and roof inputs are
proxy parameters. They are not GPU benchmark artifacts.
