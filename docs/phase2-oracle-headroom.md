# Phase 2: Oracle Headroom Experiment

Phase 2 is a diagnostic-only R3-proxy experiment. It measures structural acceptance headroom and
fully-hidden-search system upper bounds. It does not define a deployable selector, charge realistic
large-pool search latency, integrate a GPU engine, or validate real performance.

## Phase-1.5 baseline completion

`Cand. C/V` is committed candidate tokens divided by verified candidate nodes. Per-SLO attainment
is ordered as 40/50/150 ms. All values come from the existing frozen Phase-1.5 runs.

| Load | Policy | Goodput tok/s | Overall | 40/50/150 ms attainment | Raw tok/s | Queue s | Roof | Cand. C/V | Total progress/cycle |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2.75× | Dual-Batch | 2557.9 | .990 | .987/.992/1.000 | 2565.3 | .36 | .543 | .358 | 63.61 |
| 2.75× | Residual-Round-Robin | 2566.9 | .997 | .996/.997/1.000 | 2569.0 | .22 | .911 | .255 | 69.67 |
| 2.75× | Residual-Probability | 2567.0 | .998 | .996/.998/1.000 | 2568.4 | .18 | .909 | .256 | 69.61 |
| 2.75× | Shaping-Residual | 2567.5 | .998 | .997/.998/1.000 | 2568.9 | .19 | .910 | .255 | 69.62 |
| 2.75× | Feasible-Residual | 2566.8 | .997 | .996/.998/1.000 | 2568.4 | .19 | .909 | .255 | 69.61 |
| 3.0× | Dual-Batch | 1968.3 | .673 | .605/.648/.905 | 2746.2 | 5.91 | .598 | .358 | 69.93 |
| 3.0× | Residual-Round-Robin | 2255.5 | .753 | .681/.746/.975 | 2770.4 | 3.53 | .953 | .266 | 76.98 |
| 3.0× | Residual-Probability | 2452.3 | .816 | .756/.820/.993 | 2781.9 | 2.43 | .953 | .268 | 77.22 |
| 3.0× | Shaping-Residual | 2146.6 | .728 | .660/.711/.951 | 2761.8 | 4.30 | .953 | .265 | 76.70 |
| 3.0× | Feasible-Residual | 2144.7 | .727 | .657/.712/.951 | 2762.8 | 4.31 | .953 | .265 | 76.71 |
| 3.25× | Dual-Batch | 1291.6 | .421 | .369/.400/.599 | 2837.0 | 22.70 | .626 | .358 | 73.24 |
| 3.25× | Residual-Round-Robin | 1478.5 | .470 | .413/.451/.662 | 2875.1 | 17.49 | .974 | .272 | 80.87 |
| 3.25× | Residual-Probability | 1587.4 | .502 | .443/.485/.694 | 2892.6 | 15.05 | .974 | .275 | 81.32 |
| 3.25× | Shaping-Residual | 1414.4 | .457 | .403/.438/.637 | 2865.1 | 19.17 | .974 | .271 | 80.57 |
| 3.25× | Feasible-Residual | 1415.0 | .457 | .404/.438/.638 | 2865.1 | 19.14 | .974 | .271 | 80.57 |

Probability and round-robin are effectively tied below the knee. At 3.0×, probability adds
196.8 good tokens/s, 6.36 attainment points, 11.5 raw tokens/s, and 0.24 total progress/cycle while
reducing mean queueing by 1.10 seconds. At 3.25× it adds 108.9 good tokens/s, 3.15 attainment
points, 17.5 raw tokens/s, and 0.45 total progress/cycle while reducing queueing by 2.44 seconds.
The difference is therefore selector efficiency under pressure, not merely residual-roof fill.

## Oracle audit and comparability boundary

The historical `CandidateTreeOracle.verify` chooses a target child from the children in the tree
passed to verification. Changing pool width would therefore change the random-choice range and
ground truth. Phase 2 uses a separate canonical oracle that:

1. projects the exact historical 1× candidate tree;
2. freezes the historical deterministic target trajectory on that tree;
3. constructs nested 2×/4×/8× pools without modifying any 1× node;
4. applies the same frozen target to every ratio and selector.

Every replayed A_1× plan is compared with the historical Residual-Probability plan. A mismatch is
a hard failure, so Phase-1.5 and Phase-2 1× are directly comparable. The historical oracle remains
unchanged for all existing policies.

This compatibility has an important limitation: the historical target trajectory is already
contained in the complete 1× pool. Added branches cannot reveal a target path missing from 1×.
They can test whether the current selector remains effective with more candidates, but they cannot
measure real better-drafter/pool-coverage headroom. Phase-2 pool results must be interpreted within
that simulator boundary.

## Controlled budgets and variants

`B_search` is the non-root candidate pool generated before selection. `B_verify` is the selected
non-root candidate count charged to target verification. Ratios change only `B_search`; the
Dual-Batch base request set, root opportunities, candidate roof, actual `B_verify`, target outcome,
and proxy `T_verify(B_req, B_cand, C)` remain fixed.

| Variant | Base nodes | Residual/candidate choice | Target leakage |
| --- | --- | --- | --- |
| A: Current | frozen Dual-Batch | global path probability | no |
| B: Within-request oracle | frozen Dual-Batch | A's per-request residual vector; target-optimal within request | yes |
| C: Global residual oracle | frozen Dual-Batch | reallocates only residual slots for committed progress | yes |
| D: Full-tree ceiling | base requests/roots only | may replace all candidate nodes at the same global B_verify | yes |

All variants are `diagnostic_only=true` and `assumes_fully_hidden_search=true`; search latency is
`metadata_only`. A cannot call the target oracle during selection. B/C/D explicitly leak target
outcomes and are upper bounds. Each snapshot must satisfy prefix closure, pool membership, roof and
request/root preservation plus realized progress dominance `B >= A`, `C >= B`, `D >= C`.

## Common-snapshot protocol

For each load, Residual-Probability 1× is run twice. The first pass records lightweight descriptors
for every eligible allocation cycle. Exactly 10,000 snapshots are then selected proportionally and
deterministically across temporal quartile, active-batch quartile, and queue-depth bin. The second
identical baseline pass replays A/B/C/D for all four ratios without advancing counterfactual state.
Compact external snapshot records contain active/base IDs, prefix epoch, committed tokens, SLO,
base budget/nodes, residual roof, target trajectory, verification-surface inputs, and a fully
reconstructible forest descriptor with generator version, seed, ratio, node counts, and SHA256.
Queue-depth strata count only requests whose arrival time has passed but which have not entered the
active set; future trace entries are explicitly excluded.

The two dual-batch cohorts remain explicit. `verification_surface_inputs` describes the stored
proposal batch verified in the current slot, whereas `base_request_ids` and `requests` describe the
parallel next-proposal planning cohort replayed by A/B/C/D. Their request counts may differ when a
slot drains or a request finishes; equating them would collapse the dual-batch lifecycle. Within
each cohort, roots are counted once, selected candidate positions never exceed `B_verify`, and the
proxy continues to use the two-dimensional input `T_verify(B_req, B_cand, C)`.

Detailed JSON and compressed snapshots remain outside Git under
`SpecRhythm-data/results/simulator-semantics-v0.2/phase2-oracle-headroom/`.

## Phase-2 results

### Sampling coverage

Each load contributes exactly 10,000 of its eligible planning snapshots. Temporal quartiles are
within two samples of 2,500 each. Active bins are `[0,16)`, `[16,32)`, `[32,48)`, and `48+`
requests; queue bins are `0`, `1–63`, `64–511`, and `512+` arrived-but-not-admitted requests.

| Load | Eligible | Active-bin samples | Queue-bin samples |
| --- | ---: | --- | --- |
| 2.75× | 47,809 | 56/619/3,264/6,061 | 8,105/1,895/0/0 |
| 3.0× | 43,075 | 60/247/1,781/7,912 | 4,686/3,723/1,591/0 |
| 3.25× | 40,916 | 65/96/919/8,920 | 2,746/2,348/4,316/590 |

The SLO strata below count request-opportunities in the sampled planning cohorts, not unique
requests: a long-lived request can appear in multiple snapshots.

| Load | 40 ms opportunities | 50 ms opportunities | 150 ms opportunities |
| --- | ---: | ---: | ---: |
| 2.75× | 183,139 | 40,259 | 27,000 |
| 3.0× | 206,606 | 45,431 | 29,928 |
| 3.25× | 219,196 | 47,936 | 31,615 |

The 1×/2×/4×/8× pools realize their requested ratios exactly. At 8×, maximum depth is
8, maximum and mean width are 16, and mean depth is 7.887–7.889. Canonical target-path pool
coverage is 1.0 at every observed depth for every ratio. This is a compatibility property of the
frozen 1× target oracle, not evidence that a real drafter has perfect target coverage.

### Common-snapshot structural results

`Pool req/cyc` gives candidate-pool nodes per request and cycle. `Verify/cyc` is the selected
candidate count and is fixed across all four variants at a given load. Each variant cell is
`candidate committed / total progress per cycle (candidate committed/verified)`.

| Load | Ratio | Pool req/cyc | Verify/cyc | A | B | C | D |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| 2.75× | 1× | 15.77/394.97 | 174.56 | 44.65/69.68 (.256) | 51.22/76.26 (.293) | 53.44/78.48 (.306) | 55.94/80.98 (.320) |
| 2.75× | 2× | 31.55/789.94 | 174.56 | 37.27/62.31 (.214) | 51.47/76.51 (.295) | 53.44/78.48 (.306) | 55.94/80.98 (.320) |
| 2.75× | 4× | 63.10/1,579.89 | 174.56 | 35.59/60.63 (.204) | 51.69/76.73 (.296) | 53.44/78.48 (.306) | 55.94/80.98 (.320) |
| 2.75× | 8× | 126.19/3,159.77 | 174.56 | 35.39/60.43 (.203) | 51.75/76.79 (.296) | 53.44/78.48 (.306) | 55.94/80.98 (.320) |
| 3.0× | 1× | 15.78/444.83 | 182.82 | 48.99/77.18 (.268) | 55.91/84.10 (.306) | 59.89/88.09 (.328) | 62.72/90.91 (.343) |
| 3.0× | 2× | 31.55/889.67 | 182.82 | 41.41/69.61 (.227) | 56.32/84.52 (.308) | 59.89/88.09 (.328) | 62.72/90.91 (.343) |
| 3.0× | 4× | 63.10/1,779.34 | 182.82 | 39.77/67.96 (.218) | 56.65/84.85 (.310) | 59.89/88.09 (.328) | 62.72/90.91 (.343) |
| 3.0× | 8× | 126.21/3,558.68 | 182.82 | 39.64/67.84 (.217) | 56.71/84.91 (.310) | 59.89/88.09 (.328) | 62.72/90.91 (.343) |
| 3.25× | 1× | 15.78/471.37 | 186.97 | 51.49/81.36 (.275) | 58.68/88.55 (.314) | 63.73/93.61 (.341) | 66.74/96.62 (.357) |
| 3.25× | 2× | 31.56/942.74 | 186.97 | 43.85/73.72 (.235) | 59.18/89.05 (.317) | 63.73/93.61 (.341) | 66.74/96.62 (.357) |
| 3.25× | 4× | 63.11/1,885.48 | 186.97 | 42.23/72.11 (.226) | 59.61/89.49 (.319) | 63.73/93.61 (.341) | 66.74/96.62 (.357) |
| 3.25× | 8× | 126.23/3,770.97 | 186.97 | 42.16/72.04 (.226) | 59.71/89.59 (.319) | 63.73/93.61 (.341) | 66.74/96.62 (.357) |

At 8×, A/B/C/D mean accepted candidate lengths are respectively 1.413/2.067/2.134/2.234
at 2.75×, 1.406/2.011/2.124/2.224 at 3.0×, and 1.411/1.999/2.133/2.234 at
3.25×. The corresponding P50 is 1 for every variant and P90 is 4/5/5/6. Mean residual
budget per request is 3.013, 2.525, and 2.299 for the three loads; across rows P50 ranges from
2–4 and P90 is 4.

### Paired headroom decomposition

The next table reports mean candidate-committed gain per cycle. Because every pair has identical
roots and selected candidate count, the total-progress gain is numerically identical; the detailed
machine report also contains the paired committed/verified difference. Negative pool gain means
the current target-blind selector is harmed by the larger pool.

| Load | Ratio | Pool A_r−A_1 | Selector B−A | Allocation C−B | Base tree D−C |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2.75× | 1× | 0.00 | 6.57 | 2.23 | 2.49 |
| 2.75× | 2× | -7.37 | 14.20 | 1.97 | 2.49 |
| 2.75× | 4× | -9.05 | 16.09 | 1.75 | 2.49 |
| 2.75× | 8× | -9.26 | 16.36 | 1.69 | 2.49 |
| 3.0× | 1× | 0.00 | 6.92 | 3.99 | 2.83 |
| 3.0× | 2× | -7.57 | 14.91 | 3.57 | 2.83 |
| 3.0× | 4× | -9.22 | 16.89 | 3.24 | 2.83 |
| 3.0× | 8× | -9.34 | 17.07 | 3.18 | 2.83 |
| 3.25× | 1× | 0.00 | 7.19 | 5.06 | 3.01 |
| 3.25× | 2× | -7.64 | 15.33 | 4.56 | 3.01 |
| 3.25× | 4× | -9.26 | 17.38 | 4.12 | 3.01 |
| 3.25× | 8× | -9.33 | 17.55 | 4.02 | 3.01 |

For the decisive 8× comparison, selector-gain mean/P50/P90 and improvement fractions are
16.36/16/25 and 99.65% at 2.75×, 17.07/17/25 and 99.75% at 3.0×, and
17.55/17/26 and 99.72% at 3.25×. Allocation-gain mean/P50/P90 and fractions are
1.69/0/6 and 36.27%, 3.18/2/9 and 60.81%, and 4.02/3/9 and 73.95%. Base-tree gain is
2.49/2/6 (61.89%), 2.83/2/7 (66.77%), and 3.01/2/7 (69.18%).

Relative to A_8×, the selector gap is +46.2%, +43.1%, and +41.6% candidate progress per
cycle. Relative to B_8×, global residual reallocation adds +3.3%, +5.6%, and +6.7%; relative
to C_8×, full-tree recomposition adds about +4.7% at every load.

All 120,000 load/ratio snapshot comparisons pass each of the three ordered dominance checks
`B >= A`, `C >= B`, and `D >= C` (360,000 inequalities total). This is a same-snapshot structural
invariant. Independent end-to-end runs may legitimately lose that ordering in goodput, queueing,
or attainment because an earlier choice changes the later active set and trajectory.

## Completion and A_1× equivalence audit

The recovery audit reused all complete artifacts and ran only the nine missing 3.25× cells. The
final manifest is common replay 3/3, reference 9/9, and end-to-end matrix 48/48, with zero running,
missing, or corrupt cells. Every compressed common trace contains exactly 10,000 parseable records
from the corrected arrived-but-not-admitted queue definition.

For all three loads, Phase-2 A_1× equals the Phase-1.5 Residual-Probability reference on request
count, cycles, goodput, overall and per-class attainment, raw throughput, makespan, mean queueing,
P90 TPOT, verified/committed candidates, root and total progress, completions/cycle,
accepted/verified, and roof utilization. Integer comparison is exact; floating-point comparison
uses relative tolerance 0 and absolute tolerance `1e-12`. All comparisons pass with no mismatch.

## End-to-end fully-hidden-search upper bound

These are independent counterfactual trajectories, not the primary same-snapshot causal result.
`attain/good` pairs class attainment with SLO-good tokens. Latencies are means in seconds except
P90 TPOT, which is milliseconds. Search construction time is not included in these values.

| Load | Ratio | Sel. | Goodput | Overall | 40 ms attain/good | 50 ms attain/good | 150 ms attain/good | Raw | Makespan s | Queue s | Service s | Decode s | P90 TPOT ms |
| --- | ---: | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.75× | 1× | A | 2567.0 | .9975 | .9964/2,296,278 | .9983/580,426 | 1.0000/450,859 | 2568.4 | 1296.31 | 0.18 | 5.48 | 5.66 | 24.8 |
| 2.75× | 1× | B | 2570.7 | .9998 | .9997/2,297,958 | .9996/580,525 | 1.0000/450,859 | 2570.8 | 1295.10 | 0.04 | 4.82 | 4.86 | 21.0 |
| 2.75× | 1× | C | 2571.9 | .9999 | .9999/2,298,019 | 1.0000/580,564 | 1.0000/450,859 | 2571.9 | 1294.55 | 0.03 | 4.72 | 4.75 | 20.2 |
| 2.75× | 1× | D | 2572.7 | 1.0000 | 1.0000/2,298,071 | 1.0000/580,564 | 1.0000/450,859 | 2572.7 | 1294.14 | 0.02 | 4.53 | 4.55 | 19.5 |
| 2.75× | 2× | A | 2075.9 | .7543 | .6788/1,771,470 | .7531/495,913 | .9821/448,282 | 2545.2 | 1308.16 | 3.17 | 6.28 | 9.45 | 75.0 |
| 2.75× | 2× | B | 2570.8 | .9999 | .9999/2,298,019 | 1.0000/580,564 | 1.0000/450,859 | 2570.8 | 1295.11 | 0.04 | 4.80 | 4.84 | 20.8 |
| 2.75× | 2× | C | 2571.9 | .9999 | .9999/2,298,019 | 1.0000/580,564 | 1.0000/450,859 | 2571.9 | 1294.55 | 0.03 | 4.72 | 4.75 | 20.2 |
| 2.75× | 2× | D | 2572.7 | 1.0000 | 1.0000/2,298,071 | 1.0000/580,564 | 1.0000/450,859 | 2572.7 | 1294.14 | 0.02 | 4.53 | 4.55 | 19.5 |
| 2.75× | 4× | A | 1762.2 | .6580 | .5842/1,484,699 | .6359/413,586 | .9015/428,111 | 2522.1 | 1320.15 | 5.95 | 6.45 | 12.40 | 121.7 |
| 2.75× | 4× | B | 2571.1 | .9999 | .9999/2,298,019 | 1.0000/580,564 | 1.0000/450,859 | 2571.1 | 1294.96 | 0.04 | 4.79 | 4.83 | 20.7 |
| 2.75× | 4× | C | 2571.9 | .9999 | .9999/2,298,019 | 1.0000/580,564 | 1.0000/450,859 | 2571.9 | 1294.55 | 0.03 | 4.72 | 4.75 | 20.2 |
| 2.75× | 4× | D | 2572.7 | 1.0000 | 1.0000/2,298,071 | 1.0000/580,564 | 1.0000/450,859 | 2572.7 | 1294.14 | 0.02 | 4.53 | 4.55 | 19.5 |
| 2.75× | 8× | A | 1761.1 | .6576 | .5840/1,483,773 | .6355/413,558 | .9007/427,831 | 2521.8 | 1320.30 | 5.97 | 6.46 | 12.43 | 122.0 |
| 2.75× | 8× | B | 2570.8 | .9999 | .9999/2,298,019 | 1.0000/580,564 | 1.0000/450,859 | 2570.9 | 1295.09 | 0.04 | 4.79 | 4.83 | 20.7 |
| 2.75× | 8× | C | 2571.9 | .9999 | .9999/2,298,019 | 1.0000/580,564 | 1.0000/450,859 | 2571.9 | 1294.55 | 0.03 | 4.72 | 4.75 | 20.2 |
| 2.75× | 8× | D | 2572.7 | 1.0000 | 1.0000/2,298,071 | 1.0000/580,564 | 1.0000/450,859 | 2572.7 | 1294.14 | 0.02 | 4.53 | 4.55 | 19.5 |
| 3.0× | 1× | A | 2452.3 | .8164 | .7565/1,957,278 | .8200/527,798 | .9925/449,990 | 2781.9 | 1196.86 | 2.43 | 5.67 | 8.10 | 60.9 |
| 3.0× | 1× | B | 2801.1 | .9979 | .9970/2,296,588 | .9988/580,452 | 1.0000/450,859 | 2802.4 | 1188.07 | 0.19 | 5.02 | 5.20 | 23.5 |
| 3.0× | 1× | C | 2801.9 | .9996 | .9996/2,297,922 | .9992/580,489 | 1.0000/450,859 | 2802.1 | 1188.20 | 0.08 | 4.84 | 4.92 | 21.1 |
| 3.0× | 1× | D | 2803.4 | .9998 | .9997/2,297,958 | 1.0000/580,564 | 1.0000/450,859 | 2803.5 | 1187.64 | 0.05 | 4.65 | 4.70 | 20.1 |
| 3.0× | 2× | A | 1264.4 | .4399 | .3805/985,923 | .4214/283,955 | .6367/319,423 | 2648.9 | 1256.92 | 19.63 | 6.35 | 25.98 | 330.9 |
| 3.0× | 2× | B | 2800.8 | .9983 | .9975/2,296,966 | .9988/580,452 | 1.0000/450,859 | 2801.8 | 1188.34 | 0.17 | 4.99 | 5.15 | 22.9 |
| 3.0× | 2× | C | 2801.9 | .9996 | .9996/2,297,922 | .9992/580,489 | 1.0000/450,859 | 2802.1 | 1188.20 | 0.08 | 4.84 | 4.92 | 21.1 |
| 3.0× | 2× | D | 2803.4 | .9998 | .9997/2,297,958 | 1.0000/580,564 | 1.0000/450,859 | 2803.5 | 1187.64 | 0.05 | 4.65 | 4.70 | 20.1 |
| 3.0× | 4× | A | 1113.6 | .3968 | .3438/873,537 | .3807/256,213 | .5719/289,273 | 2612.8 | 1274.28 | 25.60 | 6.50 | 32.10 | 419.2 |
| 3.0× | 4× | B | 2801.4 | .9985 | .9979/2,297,127 | .9988/580,452 | 1.0000/450,859 | 2802.3 | 1188.14 | 0.15 | 4.96 | 5.11 | 22.6 |
| 3.0× | 4× | C | 2801.9 | .9996 | .9996/2,297,922 | .9992/580,489 | 1.0000/450,859 | 2802.1 | 1188.20 | 0.08 | 4.84 | 4.92 | 21.1 |
| 3.0× | 4× | D | 2803.4 | .9998 | .9997/2,297,958 | 1.0000/580,564 | 1.0000/450,859 | 2803.5 | 1187.64 | 0.05 | 4.65 | 4.70 | 20.1 |
| 3.0× | 8× | A | 1114.0 | .3968 | .3440/874,139 | .3807/256,213 | .5715/289,071 | 2613.1 | 1274.16 | 25.64 | 6.50 | 32.14 | 419.7 |
| 3.0× | 8× | B | 2801.1 | .9986 | .9981/2,297,267 | .9988/580,452 | 1.0000/450,859 | 2801.9 | 1188.32 | 0.15 | 4.96 | 5.11 | 22.6 |
| 3.0× | 8× | C | 2801.9 | .9996 | .9996/2,297,922 | .9992/580,489 | 1.0000/450,859 | 2802.1 | 1188.20 | 0.08 | 4.84 | 4.92 | 21.1 |
| 3.0× | 8× | D | 2803.4 | .9998 | .9997/2,297,958 | 1.0000/580,564 | 1.0000/450,859 | 2803.5 | 1187.64 | 0.05 | 4.65 | 4.70 | 20.1 |
| 3.25× | 1× | A | 1587.4 | .5018 | .4434/1,159,531 | .4846/321,219 | .6941/346,469 | 2892.6 | 1151.06 | 15.05 | 5.76 | 20.81 | 259.8 |
| 3.25× | 1× | B | 2708.6 | .8343 | .7797/2,008,122 | .8387/535,584 | .9933/450,077 | 3012.3 | 1105.30 | 2.36 | 5.20 | 7.56 | 58.1 |
| 3.25× | 1× | C | 3030.2 | .9963 | .9952/2,295,714 | .9963/580,214 | 1.0000/450,859 | 3032.6 | 1097.89 | 0.30 | 4.93 | 5.22 | 24.1 |
| 3.25× | 1× | D | 3033.7 | .9988 | .9985/2,297,562 | .9988/580,452 | 1.0000/450,859 | 3034.2 | 1097.31 | 0.16 | 4.74 | 4.90 | 21.6 |
| 3.25× | 2× | A | 918.8 | .3085 | .2625/689,769 | .3071/211,279 | .4476/226,947 | 2712.0 | 1227.69 | 42.69 | 6.38 | 49.07 | 662.9 |
| 3.25× | 2× | B | 2882.5 | .9053 | .8730/2,166,036 | .9081/560,095 | .9996/450,819 | 3020.9 | 1102.14 | 1.52 | 5.17 | 6.68 | 44.8 |
| 3.25× | 2× | C | 3030.2 | .9963 | .9952/2,295,714 | .9963/580,214 | 1.0000/450,859 | 3032.6 | 1097.89 | 0.30 | 4.93 | 5.22 | 24.1 |
| 3.25× | 2× | D | 3033.7 | .9988 | .9985/2,297,562 | .9988/580,452 | 1.0000/450,859 | 3034.2 | 1097.31 | 0.16 | 4.74 | 4.90 | 21.6 |
| 3.25× | 4× | A | 743.3 | .2454 | .1956/545,826 | .2406/176,887 | .3998/206,120 | 2664.5 | 1249.57 | 51.87 | 6.52 | 58.38 | 787.7 |
| 3.25× | 4× | B | 2952.6 | .9412 | .9197/2,227,001 | .9468/570,638 | 1.0000/450,859 | 3026.2 | 1100.20 | 1.16 | 5.13 | 6.29 | 38.9 |
| 3.25× | 4× | C | 3030.2 | .9963 | .9952/2,295,714 | .9963/580,214 | 1.0000/450,859 | 3032.6 | 1097.89 | 0.30 | 4.93 | 5.22 | 24.1 |
| 3.25× | 4× | D | 3033.7 | .9988 | .9985/2,297,562 | .9988/580,452 | 1.0000/450,859 | 3034.2 | 1097.31 | 0.16 | 4.74 | 4.90 | 21.6 |
| 3.25× | 8× | A | 742.1 | .2449 | .1948/544,407 | .2406/176,887 | .3994/205,935 | 2664.8 | 1249.42 | 51.93 | 6.52 | 58.45 | 788.5 |
| 3.25× | 8× | B | 2950.5 | .9409 | .9202/2,226,361 | .9439/569,731 | 1.0000/450,859 | 3025.5 | 1100.47 | 1.16 | 5.13 | 6.28 | 38.7 |
| 3.25× | 8× | C | 3030.2 | .9963 | .9952/2,295,714 | .9963/580,214 | 1.0000/450,859 | 3032.6 | 1097.89 | 0.30 | 4.93 | 5.22 | 24.1 |
| 3.25× | 8× | D | 3033.7 | .9988 | .9985/2,297,562 | .9988/580,452 | 1.0000/450,859 | 3034.2 | 1097.31 | 0.16 | 4.74 | 4.90 | 21.6 |

### Phase-1.5 references: end-to-end outcomes

| Load | Reference | Goodput | Overall | 40 ms attain/good | 50 ms attain/good | 150 ms attain/good | Raw | Makespan s | Queue s | Service s | Decode s | P90 TPOT ms |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.75× | Dual-Batch | 2557.9 | .9904 | .9867/2,289,520 | .9917/579,524 | 1.0000/450,859 | 2565.3 | 1297.88 | 0.36 | 5.76 | 6.12 | 28.4 |
| 2.75× | Residual-Probability | 2567.0 | .9975 | .9964/2,296,278 | .9983/580,426 | 1.0000/450,859 | 2568.4 | 1296.31 | 0.18 | 5.48 | 5.66 | 24.8 |
| 2.75× | Dual-Eager | 2562.5 | .9957 | .9939/2,294,692 | .9967/580,243 | 1.0000/450,859 | 2565.3 | 1297.88 | 0.30 | 5.73 | 6.02 | 26.9 |
| 3.0× | Dual-Batch | 1968.3 | .6732 | .6045/1,537,104 | .6475/420,353 | .9048/428,950 | 2746.2 | 1212.42 | 5.91 | 5.89 | 11.79 | 119.9 |
| 3.0× | Residual-Probability | 2452.3 | .8164 | .7565/1,957,278 | .8200/527,798 | .9925/449,990 | 2781.9 | 1196.86 | 2.43 | 5.67 | 8.10 | 60.9 |
| 3.0× | Dual-Eager | 2645.2 | .8933 | .8577/2,149,567 | .8940/557,089 | .9992/450,785 | 2789.4 | 1193.63 | 1.75 | 5.77 | 7.52 | 47.6 |
| 3.25× | Dual-Batch | 1291.6 | .4209 | .3686/945,784 | .3998/268,414 | .5989/301,655 | 2837.0 | 1173.61 | 22.70 | 5.95 | 28.65 | 373.5 |
| 3.25× | Residual-Probability | 1587.4 | .5018 | .4434/1,159,531 | .4846/321,219 | .6941/346,469 | 2892.6 | 1151.06 | 15.05 | 5.76 | 20.81 | 259.8 |
| 3.25× | Dual-Eager | 1937.9 | .5891 | .5063/1,367,890 | .5557/384,993 | .8712/420,689 | 2968.6 | 1121.59 | 7.97 | 5.67 | 13.64 | 139.9 |

### End-to-end operation and search accounting

`C/V` is committed candidate tokens per verified candidate. Search nodes are materialized pool
nodes; verify nodes are selected candidate positions. Roots are outside both counts and appear
once in root progress and in the request dimension of the verification surface.

| Load | Ratio | Sel. | Cycles | Req/cyc | Verify batch | Roof | Verify/cyc | Commit/cyc | Root/cyc | Total/cyc | C/V | Search nodes | Verify nodes | Search/verify |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.75× | 1× | A | 47,834 | .252 | 25.02 | .909 | 174.47 | 44.58 | 25.02 | 69.61 | .256 | 18,883,864 | 8,345,710 | 2.26 |
| 2.75× | 1× | B | 49,278 | .244 | 21.93 | .843 | 161.80 | 45.64 | 21.93 | 67.57 | .282 | 17,051,084 | 7,973,168 | 2.14 |
| 2.75× | 1× | C | 49,454 | .243 | 21.52 | .837 | 160.52 | 45.81 | 21.52 | 67.33 | .285 | 16,788,538 | 7,938,546 | 2.11 |
| 2.75× | 1× | D | 49,942 | .241 | 20.64 | .814 | 156.05 | 46.03 | 20.64 | 66.67 | .295 | 16,291,358 | 7,793,210 | 2.09 |
| 2.75× | 2× | A | 46,931 | .256 | 28.61 | .961 | 184.44 | 42.34 | 28.61 | 70.94 | .230 | 42,419,156 | 8,656,019 | 4.90 |
| 2.75× | 2× | B | 49,314 | .244 | 21.85 | .842 | 161.49 | 45.66 | 21.85 | 67.52 | .283 | 34,008,724 | 7,963,868 | 4.27 |
| 2.75× | 2× | C | 49,454 | .243 | 21.52 | .837 | 160.52 | 45.81 | 21.52 | 67.33 | .285 | 33,577,076 | 7,938,546 | 4.23 |
| 2.75× | 2× | D | 49,942 | .241 | 20.64 | .814 | 156.05 | 46.03 | 20.64 | 66.67 | .295 | 32,582,716 | 7,793,210 | 4.18 |
| 2.75× | 4× | A | 47,180 | .255 | 29.14 | .968 | 185.75 | 41.43 | 29.14 | 70.57 | .223 | 86,899,720 | 8,763,745 | 9.92 |
| 2.75× | 4× | B | 49,335 | .244 | 21.80 | .841 | 161.35 | 45.69 | 21.80 | 67.49 | .283 | 67,883,872 | 7,960,148 | 8.53 |
| 2.75× | 4× | C | 49,454 | .243 | 21.52 | .837 | 160.52 | 45.81 | 21.52 | 67.33 | .285 | 67,154,152 | 7,938,546 | 8.46 |
| 2.75× | 4× | D | 49,942 | .241 | 20.64 | .814 | 156.05 | 46.03 | 20.64 | 66.67 | .295 | 65,165,432 | 7,793,210 | 8.36 |
| 2.75× | 8× | A | 47,167 | .255 | 29.19 | .969 | 185.92 | 41.40 | 29.19 | 70.59 | .223 | 174,028,960 | 8,769,482 | 19.84 |
| 2.75× | 8× | B | 49,339 | .244 | 21.79 | .841 | 161.32 | 45.69 | 21.79 | 67.48 | .283 | 135,732,864 | 7,959,257 | 17.05 |
| 2.75× | 8× | C | 49,454 | .243 | 21.52 | .837 | 160.52 | 45.81 | 21.52 | 67.33 | .285 | 134,308,304 | 7,938,546 | 16.92 |
| 2.75× | 8× | D | 49,942 | .241 | 20.64 | .814 | 156.05 | 46.03 | 20.64 | 66.67 | .295 | 130,330,864 | 7,793,210 | 16.72 |
| 3.0× | 1× | A | 43,118 | .279 | 28.17 | .953 | 182.71 | 49.05 | 28.17 | 77.22 | .268 | 19,165,316 | 7,878,093 | 2.43 |
| 3.0× | 1× | B | 43,889 | .274 | 24.99 | .907 | 173.99 | 50.88 | 24.99 | 75.86 | .292 | 17,304,080 | 7,636,315 | 2.27 |
| 3.0× | 1× | C | 44,161 | .272 | 24.10 | .898 | 172.20 | 51.30 | 24.10 | 75.39 | .298 | 16,788,538 | 7,604,694 | 2.21 |
| 3.0× | 1× | D | 44,555 | .270 | 23.13 | .878 | 168.49 | 51.59 | 23.13 | 74.73 | .306 | 16,291,358 | 7,506,885 | 2.17 |
| 3.0× | 2× | A | 44,583 | .270 | 30.20 | .979 | 187.89 | 44.48 | 30.20 | 74.68 | .237 | 42,534,372 | 8,376,585 | 5.08 |
| 3.0× | 2× | B | 43,934 | .274 | 24.84 | .905 | 173.72 | 50.95 | 24.84 | 75.78 | .293 | 34,433,124 | 7,632,092 | 4.51 |
| 3.0× | 2× | C | 44,161 | .272 | 24.10 | .898 | 172.20 | 51.30 | 24.10 | 75.39 | .298 | 33,577,076 | 7,604,694 | 4.42 |
| 3.0× | 2× | D | 44,555 | .270 | 23.13 | .878 | 168.49 | 51.59 | 23.13 | 74.73 | .306 | 32,582,716 | 7,506,885 | 4.34 |
| 3.0× | 4× | A | 45,111 | .267 | 30.50 | .982 | 188.51 | 43.31 | 30.50 | 73.81 | .230 | 86,957,120 | 8,504,011 | 10.23 |
| 3.0× | 4× | B | 43,968 | .274 | 24.72 | .904 | 173.48 | 51.00 | 24.72 | 75.73 | .294 | 68,606,928 | 7,627,630 | 8.99 |
| 3.0× | 4× | C | 44,161 | .272 | 24.10 | .898 | 172.20 | 51.30 | 24.10 | 75.39 | .298 | 67,154,152 | 7,604,694 | 8.83 |
| 3.0× | 4× | D | 44,555 | .270 | 23.13 | .878 | 168.49 | 51.59 | 23.13 | 74.73 | .306 | 65,165,432 | 7,506,885 | 8.68 |
| 3.0× | 8× | A | 45,112 | .267 | 30.52 | .983 | 188.55 | 43.29 | 30.52 | 73.81 | .230 | 174,034,272 | 8,505,905 | 20.46 |
| 3.0× | 8× | B | 43,980 | .274 | 24.70 | .904 | 173.38 | 51.01 | 24.70 | 75.70 | .294 | 137,102,560 | 7,625,385 | 17.98 |
| 3.0× | 8× | C | 44,161 | .272 | 24.10 | .898 | 172.20 | 51.30 | 24.10 | 75.39 | .298 | 134,308,304 | 7,604,694 | 17.66 |
| 3.0× | 8× | D | 44,555 | .270 | 23.13 | .878 | 168.49 | 51.59 | 23.13 | 74.73 | .306 | 130,330,864 | 7,506,885 | 17.36 |
| 3.25× | 1× | A | 40,942 | .294 | 29.87 | .974 | 186.89 | 51.45 | 29.87 | 81.32 | .275 | 19,297,758 | 7,651,581 | 2.52 |
| 3.25× | 1× | B | 39,883 | .302 | 27.99 | .949 | 181.99 | 55.49 | 27.99 | 83.48 | .305 | 17,617,388 | 7,258,160 | 2.43 |
| 3.25× | 1× | C | 39,934 | .301 | 26.65 | .939 | 180.18 | 56.73 | 26.65 | 83.37 | .315 | 16,788,538 | 7,195,247 | 2.33 |
| 3.25× | 1× | D | 40,235 | .299 | 25.62 | .926 | 177.55 | 57.13 | 25.62 | 82.75 | .322 | 16,291,358 | 7,143,736 | 2.28 |
| 3.25× | 2× | A | 43,306 | .278 | 31.14 | .988 | 189.45 | 45.75 | 31.14 | 76.88 | .241 | 42,602,120 | 8,204,460 | 5.19 |
| 3.25× | 2× | B | 39,805 | .302 | 27.86 | .947 | 181.71 | 55.79 | 27.86 | 83.65 | .307 | 34,999,540 | 7,232,965 | 4.84 |
| 3.25× | 2× | C | 39,934 | .301 | 26.65 | .939 | 180.18 | 56.73 | 26.65 | 83.37 | .315 | 33,577,076 | 7,195,247 | 4.67 |
| 3.25× | 2× | D | 40,235 | .299 | 25.62 | .926 | 177.55 | 57.13 | 25.62 | 82.75 | .322 | 32,582,716 | 7,143,736 | 4.56 |
| 3.25× | 4× | A | 44,032 | .273 | 31.25 | .988 | 189.66 | 44.36 | 31.25 | 75.62 | .234 | 86,988,800 | 8,351,327 | 10.42 |
| 3.25× | 4× | B | 39,769 | .303 | 27.71 | .946 | 181.53 | 56.01 | 27.71 | 83.72 | .309 | 69,555,032 | 7,219,118 | 9.63 |
| 3.25× | 4× | C | 39,934 | .301 | 26.65 | .939 | 180.18 | 56.73 | 26.65 | 83.37 | .315 | 67,154,152 | 7,195,247 | 9.33 |
| 3.25× | 4× | D | 40,235 | .299 | 25.62 | .926 | 177.55 | 57.13 | 25.62 | 82.75 | .322 | 65,165,432 | 7,143,736 | 9.12 |
| 3.25× | 8× | A | 44,038 | .273 | 31.26 | .989 | 189.68 | 44.34 | 31.26 | 75.61 | .234 | 174,038,128 | 8,353,200 | 20.83 |
| 3.25× | 8× | B | 39,786 | .302 | 27.68 | .946 | 181.48 | 56.01 | 27.68 | 83.69 | .309 | 139,004,400 | 7,220,342 | 19.25 |
| 3.25× | 8× | C | 39,934 | .301 | 26.65 | .939 | 180.18 | 56.73 | 26.65 | 83.37 | .315 | 134,308,304 | 7,195,247 | 18.67 |
| 3.25× | 8× | D | 40,235 | .299 | 25.62 | .926 | 177.55 | 57.13 | 25.62 | 82.75 | .322 | 130,330,864 | 7,143,736 | 18.24 |

### Phase-1.5 references: operation accounting

| Load | Reference | Cycles | Req/cyc | Verify batch | Roof | Verify/cyc | Commit/cyc | Root/cyc | Total/cyc | C/V | Search nodes | Verify nodes | Search/verify |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.75× | Dual-Batch | 52,342 | .230 | 26.30 | .543 | 104.28 | 37.31 | 26.30 | 63.61 | .358 | 21,755,072 | 5,458,177 | 3.99 |
| 2.75× | Residual-Probability | 47,834 | .252 | 25.02 | .909 | 174.47 | 44.58 | 25.02 | 69.61 | .256 | 18,883,864 | 8,345,710 | 2.26 |
| 2.75× | Dual-Eager | 52,392 | .230 | 26.34 | .550 | 103.49 | 37.21 | 26.34 | 63.55 | .360 | 21,616,098 | 5,422,152 | 3.99 |
| 3.0× | Dual-Batch | 47,613 | .253 | 28.91 | .598 | 114.64 | 41.01 | 28.91 | 69.93 | .358 | 21,755,072 | 5,458,177 | 3.99 |
| 3.0× | Residual-Probability | 43,118 | .279 | 28.17 | .953 | 182.71 | 49.05 | 28.17 | 77.22 | .268 | 19,165,316 | 7,878,093 | 2.43 |
| 3.0× | Dual-Eager | 46,759 | .257 | 29.64 | .645 | 113.69 | 41.56 | 29.64 | 71.21 | .366 | 21,164,403 | 5,315,906 | 3.98 |
| 3.25× | Dual-Batch | 45,457 | .265 | 30.29 | .626 | 120.07 | 42.96 | 30.29 | 73.24 | .358 | 21,755,072 | 5,458,177 | 3.99 |
| 3.25× | Residual-Probability | 40,942 | .294 | 29.87 | .974 | 186.89 | 51.45 | 29.87 | 81.32 | .275 | 19,297,758 | 7,651,581 | 2.52 |
| 3.25× | Dual-Eager | 43,046 | .279 | 32.45 | .749 | 119.25 | 44.90 | 32.45 | 77.35 | .377 | 20,350,081 | 5,133,209 | 3.96 |

### End-to-end interpretation

The current target-blind A selector is highly sensitive to added distractors. From 1× to 8×,
goodput changes 2567.0→1761.1, 2452.3→1114.0, and 1587.4→742.1 tokens/s across the three
loads. This is not evidence that a larger real drafter pool is harmful: the frozen target is
already in 1×, so the added nodes provide no new target coverage in this isolated experiment.

The within-request oracle recovers almost all of this loss at 2.75× and 3.0×. At 3.25× it
continues to benefit through 4×: B goodput/attainment moves from 2708.6/.8343 at 1× to
2952.6/.9412 at 4×, then is effectively flat at 8×. At 3.25× and 8×, cross-request residual
reallocation raises B→C from 2950.5/.9409 to 3030.2/.9963, while full-tree replacement adds only
3030.2/.9963→3033.7/.9988. C and D have identical core outcomes across pool ratios at each load,
because their target-aware choices already saturate the canonical target available in 1×.

Thus the largest identifiable structural opportunity is candidate choice within a request; global
residual reallocation is smaller but grows with pressure, and full-tree replacement is smaller
again. The end-to-end reference runs show that these target-leaking ceilings exceed
Residual-Probability and Dual-Eager under pressure, but they do not supply a deployable mechanism.
The next candidate-selection experiment must use target-independent signals on a drafter whose
larger pool can actually change target coverage, and it must either charge search latency or use a
measured overlap model.

## Interpretation and measurement boundary

1. The canonical target is frozen on the immutable 1× pool.
2. Because 1× target-path coverage is already 100%, 2×/4×/8× cannot add a new target
   trajectory in this experiment.
3. Phase 2 therefore cannot measure real-drafter missing-target coverage or better-drafter
   headroom.
4. A's decline means the current probability selector is vulnerable to added distractors under
   this fixed-target isolation; it does not mean larger candidate pools are generally harmful.
5. B/C/D read the frozen target outcome and are oracle upper bounds, not deployable policies.
6. `assumes_fully_hidden_search=true` means only that search is outside the simulator critical
   path; it is not a claim that search is free on hardware.
7. Offline Python forest construction, SHA256 generation, replay, and experiment wall-clock are
   excluded from serving latency and are not GPU measurements.
8. These results cannot be extrapolated to free 8× search, GPU speedup, or gains on real logits.
9. Search-pool size and selected verification candidates are separate. The scalar candidate roof
   is not hardware capacity; GPU calibration must sweep the two-dimensional verification inputs
   `T_verify(B_req, B_cand, C)` jointly, including root/request positions and context.
