# Phase 1 shaping causal diagnosis

This report is the scoped pure-Python R3-proxy experiment requested for Draft PR #2. It compares
only `dual-batch`, default `shaping`, and the three guarded diagnostics at 2.75×, 3.0×, and 3.25×.
All runs use the same workload at a given load, deterministic candidate forest/target outcome,
dual execution proxy, latency proxy, roof 192, maximum request budget 8, and seed 1664. These are
simulator results, not GPU performance measurements.

Class cells use `attainment / SLO-good tokens`. Queue and makespan are seconds. `S1 bad` is the
number of stage-1 nodes allocated to one-cycle-infeasible requests, followed by its share of all
stage-1 nodes.

## Outcome and capacity metrics

| Load | Policy | Goodput | Overall | 40 ms | 50 ms | 150 ms | Raw tok/s | Makespan | Queue | P90 TPOT ms | Cycles | Req/cycle | Verify batch | Roof util. |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.75× | dual-batch | 2557.9 | .990 | .987 / 2,289,520 | .992 / 579,524 | 1.000 / 450,859 | 2565.3 | 1297.9 | .36 | 28.4 | 52,342 | .2299 | 26.30 | .543 |
| 2.75× | shaping | 2566.5 | .996 | .994 / 2,295,075 | .996 / 580,174 | 1.000 / 450,859 | 2569.1 | 1296.0 | .25 | 26.3 | 47,742 | .2520 | 25.32 | .912 |
| 2.75× | shaping-feasible | 2565.7 | .996 | .994 / 2,294,774 | .996 / 580,183 | 1.000 / 450,859 | 2568.5 | 1296.3 | .25 | 26.3 | 47,742 | .2520 | 25.32 | .911 |
| 2.75× | shaping-residual | 2567.5 | .998 | .997 / 2,296,362 | .998 / 580,426 | 1.000 / 450,859 | 2568.9 | 1296.1 | .19 | 25.0 | 47,826 | .2516 | 25.05 | .910 |
| 2.75× | shaping-feasible-residual | 2566.8 | .997 | .996 / 2,296,161 | .998 / 580,380 | 1.000 / 450,859 | 2568.4 | 1296.3 | .19 | 24.9 | 47,830 | .2515 | 25.04 | .909 |
| 3.0× | dual-batch | 1968.3 | .673 | .605 / 1,537,104 | .648 / 420,353 | .905 / 428,950 | 2746.2 | 1212.4 | 5.91 | 119.9 | 47,613 | .2527 | 28.91 | .598 |
| 3.0× | shaping | 1746.8 | .632 | .597 / 1,432,729 | .623 / 378,047 | .745 / 358,961 | 2680.5 | 1242.1 | 11.50 | 228.3 | 44,632 | .2696 | 28.50 | .955 |
| 3.0× | shaping-feasible | 1750.3 | .633 | .597 / 1,430,154 | .624 / 379,729 | .753 / 362,883 | 2682.1 | 1241.4 | 11.28 | 224.3 | 44,609 | .2697 | 28.50 | .955 |
| 3.0× | shaping-residual | 2146.6 | .728 | .660 / 1,683,887 | .711 / 462,418 | .951 / 441,585 | 2761.8 | 1205.6 | 4.30 | 93.6 | 43,410 | .2771 | 28.21 | .953 |
| 3.0× | shaping-feasible-residual | 2144.7 | .727 | .657 / 1,679,541 | .712 / 463,419 | .951 / 441,620 | 2762.8 | 1205.1 | 4.31 | 93.6 | 43,406 | .2772 | 28.20 | .953 |
| 3.25× | dual-batch | 1291.6 | .421 | .369 / 945,784 | .400 / 268,414 | .599 / 301,655 | 2837.0 | 1173.6 | 22.70 | 373.5 | 45,457 | .2647 | 30.29 | .626 |
| 3.25× | shaping | 1117.4 | .394 | .363 / 875,756 | .391 / 247,329 | .492 / 240,867 | 2727.6 | 1220.7 | 37.43 | 612.5 | 43,346 | .2776 | 30.10 | .976 |
| 3.25× | shaping-feasible | 1116.2 | .394 | .363 / 874,628 | .390 / 247,052 | .490 / 239,779 | 2729.8 | 1219.7 | 37.29 | 609.8 | 43,311 | .2778 | 30.10 | .976 |
| 3.25× | shaping-residual | 1414.4 | .457 | .403 / 1,032,606 | .438 / 290,319 | .637 / 320,779 | 2865.1 | 1162.1 | 19.17 | 321.8 | 41,324 | .2911 | 29.89 | .974 |
| 3.25× | shaping-feasible-residual | 1415.0 | .457 | .404 / 1,033,321 | .438 / 290,101 | .638 / 320,890 | 2865.1 | 1162.1 | 19.14 | 321.6 | 41,323 | .2911 | 29.90 | .974 |

## Allocation and progress metrics

`S1` and `S2` give nodes followed by share of all shaping-stage nodes. Dual-Batch has no S1 and
places all of its fixed allocation in S2. `Infeasible` is the opportunity ratio. The next four
columns are verified candidates, candidate commits, root progress, and total progress per cycle.
`A/V` is accepted candidate tokens per verified node. `E/op` and `R/op` are selected expected
candidate progress and realized total committed progress per allocation opportunity.

| Load | Policy | S1 nodes/share | S2 nodes/share | Infeasible | S1 bad/share | V/cycle | C/cycle | Root/cycle | Total/cycle | A/V | E/op | R/op |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.75× | dual-batch | 0 / .000 | 5,458,177 / 1.000 | .055 | 0 / .000 | 104.28 | 37.31 | 26.30 | 63.61 | .358 | 1.167 | 2.418 |
| 2.75× | shaping | 392,572 / .047 | 7,958,640 / .953 | .035 | 336,357 / .857 | 174.92 | 44.42 | 25.32 | 69.74 | .254 | 1.856 | 2.754 |
| 2.75× | feasible | 56,357 / .007 | 8,293,702 / .993 | .035 | 0 / .000 | 174.90 | 44.41 | 25.32 | 69.74 | .254 | 1.856 | 2.754 |
| 2.75× | residual | 139,227 / .039 | 3,464,095 / .961 | .027 | 128,113 / .920 | 174.52 | 44.57 | 25.05 | 69.62 | .255 | 1.860 | 2.779 |
| 2.75× | feasible + residual | 10,946 / .003 | 3,592,865 / .997 | .027 | 0 / .000 | 174.50 | 44.57 | 25.04 | 69.61 | .255 | 1.861 | 2.780 |
| 3.0× | dual-batch | 0 / .000 | 5,458,177 / 1.000 | .409 | 0 / .000 | 114.64 | 41.01 | 28.91 | 69.93 | .358 | 1.167 | 2.418 |
| 3.0× | shaping | 3,633,632 / .444 | 4,546,586 / .556 | .425 | 3,597,597 / .990 | 183.28 | 46.10 | 28.50 | 74.60 | .251 | 1.712 | 2.617 |
| 3.0× | feasible | 38,703 / .005 | 8,136,704 / .995 | .424 | 0 / .000 | 183.27 | 46.14 | 28.50 | 74.64 | .252 | 1.713 | 2.619 |
| 3.0× | residual | 1,184,792 / .384 | 1,901,782 / .616 | .353 | 1,177,804 / .994 | 182.80 | 48.49 | 28.21 | 76.70 | .265 | 1.749 | 2.719 |
| 3.0× | feasible + residual | 8,904 / .003 | 3,077,374 / .997 | .354 | 0 / .000 | 182.79 | 48.50 | 28.20 | 76.71 | .265 | 1.749 | 2.720 |
| 3.25× | dual-batch | 0 / .000 | 5,458,177 / 1.000 | .640 | 0 / .000 | 120.07 | 42.96 | 30.29 | 73.24 | .358 | 1.167 | 2.418 |
| 3.25× | shaping | 5,443,873 / .670 | 2,675,607 / .330 | .655 | 5,421,587 / .996 | 187.32 | 46.71 | 30.10 | 76.81 | .249 | 1.642 | 2.552 |
| 3.25× | feasible | 25,050 / .003 | 8,087,745 / .997 | .654 | 0 / .000 | 187.31 | 46.77 | 30.10 | 76.87 | .250 | 1.643 | 2.554 |
| 3.25× | residual | 1,765,487 / .623 | 1,067,122 / .377 | .609 | 1,761,091 / .998 | 186.94 | 50.68 | 29.89 | 80.57 | .271 | 1.710 | 2.695 |
| 3.25× | feasible + residual | 5,445 / .002 | 2,827,238 / .998 | .608 | 0 / .000 | 186.95 | 50.68 | 29.90 | 80.57 | .271 | 1.710 | 2.695 |

## Interpretation and invariants

The feasible-only change removes the targeted stage-1 work but barely changes outcomes. The
residual constraint is decisive near and above the proxy knee: at 3.0× it raises goodput from
1746.8 to 2146.6 tokens/s and lowers mean queueing from 11.50 to 4.30 seconds; at 3.25× it raises
goodput from 1117.4 to 1414.4 and lowers queueing from 37.43 to 19.17 seconds. Combining the guards
does not materially improve residual preservation. The results support batch breadth/base progress
opportunity cost as the dominant modeled mechanism, while allocation to one-cycle-infeasible
requests is common but not independently causal in this proxy.

Every residual run reports zero base-preservation violations. The residual allocator calls the
same Dual-Batch round-robin allocator and sequence-path materializer on the same cycle snapshot;
its final selected IDs are checked as supersets of those base IDs. All proposal, token, and tree
node conservation checks pass.

The old 75.6% Dual-Batch utilization number was not encoded with a reproducible denominator in
the v3 summary. The new online metric uses the same explicit definition for all policies: selected
non-root nodes divided by roof 192 for cycles with a non-empty allocation opportunity. It produces
59.8% for Dual-Batch and 95.5% for shaping at 3.0×. Thus 95.5% is reproducible under this common
definition, while 75.6% is superseded as an offline/reporting mismatch. No scheduler, proposal,
latency, goodput, or SLO result changed: the prior 3.0× values match the regenerated summaries;
only utilization reporting needed regeneration.

The raw summary JSON remains outside Git at
`SpecRhythm-data/results/simulator-semantics-v0.2/phase1-shaping-diagnosis/`.
