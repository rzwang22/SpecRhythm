# Results directory

Local simulation and GPU benchmark outputs belong here and are ignored by Git. Promote only
small, reviewed calibration profiles or aggregate tables into versioned files.

Simulator-semantics v0.2 comparison reports use schema
`specrhythm.comparison.v3`. The seven primary modes are `ar`, `serial-sd`, `adaserve`,
`dual-batch`, `dual-eager`, `shaping`, and `specrhythm`; three explicitly named flat proxies are
retained for provenance. Each summary identifies execution, allocator, and eager semantics and
includes overall/per-SLO attainment, goodput, queueing/service/end-to-end latency, and
proposal/token/tree-node accounting.

These reports remain local because their latency, acceptance, confidence, and roof inputs are
proxy parameters. They are not GPU benchmark artifacts.
