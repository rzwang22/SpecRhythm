# Phase 4B.4B — MineDraft-style large-cohort ping-pong baseline

This is a baseline for characterization, not SpecRhythm Adaptive Rhythm and not
an exact MineDraft reproduction. No GPU execution was performed during delivery.

## Source audit before implementation

The public [MineDraft repository](https://github.com/electron-shaders/MineDraft/tree/272c8556797495fe9c9f6a3008337bcd5cc2d031)
was inspected at **272c8556797495fe9c9f6a3008337bcd5cc2d031** before implementing B.
Its README specifies vLLM 0.9.2 and a five-GPU parallel layout (Target TP4 plus
Draft TP1). Our pinned vLLM 0.25.1 uses GPU0 Draft TP1 and GPUs1/2 Target TP2.

* [Scheduler assignment](https://github.com/electron-shaders/MineDraft/blob/272c8556797495fe9c9f6a3008337bcd5cc2d031/minedraft/plugin/core/scheduler.py#L103):
  PSD maintains a balance counter and assigns `batch_flag`; retirement recycles
  the counter. The source also contains special handling for mixed/prefill work.
* [PSD capacity](https://github.com/electron-shaders/MineDraft/blob/272c8556797495fe9c9f6a3008337bcd5cc2d031/minedraft/plugin/engine/llm_engine.py#L29):
  the PSD path doubles scheduler sequence capacity to hold both groups.
* [Parallel worker](https://github.com/electron-shaders/MineDraft/blob/272c8556797495fe9c9f6a3008337bcd5cc2d031/minedraft/plugin/spec_decode/spec_decode_worker.py#L755):
  `_split_batched_requests` separates groups using `_send_batch_flag` (line 964).
  The speculative step starts scoring (line 1330), drafts the other group
  (1365–1399), then collects scores; previous proposals are retained and the
  send flag flips when the other group has usable proposals (1481–1495).

The adaptation retains two persistent groups, opposite-group Draft/Target
execution, initial proposals and explicit alternation. It does not copy source,
Tetris, PEARL, capacity multipliers, or MineDraft's mixed-arrival special cases.

## Isolated execution

`SR_PHASE4B_DUAL_RHYTHM=legacy|pingpong` defaults to `legacy`. Only resident Dual
reads it. Pingpong requires the existing production `VllmBatchedDraftBackend`.
Target, Serial and the HF path retain their implementations. The fixed 4B.4A
sweep explicitly selects legacy, including mb100. No vLLM patch is added.

Before workers start, an immutable manifest binds the frozen workload hash to
alternating A/B assignment in workload order. There are no arrivals or transfers.
Initial membership differs by at most one (100 gives 50/50); termination reduces
active counts without rebalancing. All three roles load the same manifest.

Pingpong's readiness polling capacity is the whole burst, not legacy mb2.
The separate scheduler subclass admits all currently eligible members of the
selected cohort through the existing admissibility predicate. Stock vLLM still
owns traversal, token/sequence budgets, preemption and KV allocation. Attained
sizes, eligible IDs, active counts and stock limits are recorded. A clipped
cohort is reported as such; the evidence does not guess which internal stock
budget caused clipping when the scheduler does not expose that attribution.

The initial enqueue is split into A then B on the existing asynchronous owner.
Before the first timed decode, both groups' initial proposals/tails must be ready.
This fill occurs after the unchanged measurement boundary. Thereafter A/B flips
after an actual decode scheduling unit, not on empty polling cycles. A live but
unready selected group waits through nonblocking polls. Once a group retires,
the other drains. After capacity clipping, a held ready member also waits if
Draft still owns another active member of that same cohort. This enforces the
opposite-stage dependency; it does not wait to accumulate more ready members.
No steady-state full-cohort readiness barrier is introduced.

Target's unchanged callback submits committed prefixes to the existing owner:
Verify(A), then Verify(B) concurrent with Draft-next(A), then the inverse.
The Draft adapter checks cohort membership before delegating unchanged commit,
acceptance, proposal identity, EOS and paged-KV work. It records counters in
memory and emits one final immutable report. Existing event writes are annotated
in place; no additional per-verification fsync is introduced.

Pinned vLLM source **752a3a504485790a2e8491cacbb35c137339ad34** was also audited:
`vllm/v1/worker/gpu_model_runner.py:1187–1207` removes unscheduled requests from
InputBatch. `vllm/v1/core/sched/scheduler.py:594–609` clears speculative IDs only
for scheduled requests. Thus holding B's ready proposal while A executes does
not let A's empty custom-proposer result erase B's proposal. Existing five
patches supply the predicate, verification callbacks and sampled-row mapping.

## Evidence and interpretation

The offline report joins stable assignments, scheduling decisions, actual
proposal GPU intervals, TP verification intervals, Target commits and next-prefix
versions/hashes. It reuses material execution/accounting/lifecycle validation.
Numerical equality and metadata-only differences are not new performance gates.

Fill and drain stay inside makespan. Observed overlap is the union of measured
cross-cohort interval intersections, never critical-path time saved. Host polling
wait metrics are explicitly scoped; unobservable Draft idle time is reported as
unavailable rather than inferred GPU occupancy. Capacity clipping and both/one
stage cycles are descriptive. Throughput has no pass threshold.

Bring-up requires nonempty A/B, both initial proposals, alternation, commits,
retirement and zero live Draft KV. Corrected-5 additionally requires a positive
physical cross-cohort overlap witness. Formal corrected-100 uses five fresh
same-session cells at the final B SHA: Target, Serial, legacy mb2, legacy mb100,
PingPong. Task A timings are not formal B controls. Outcomes P1–P4 are observations,
not qualification gates or permission to implement the next policy.

## Implementation map

* `dual_rhythm.py`: selector, immutable ordered assignment and initial enqueue split.
* `vllm_pingpong_scheduler.py`: additional cohort admissibility and annotated cycles;
  the inherited stale/retired-ready, tail and stock-allocation logic is unchanged.
  Policy-ineligible ready requests have an explicit reason, so intentional A/B
  waits do not falsely count as unbounded stock-ready deferrals.
* `vllm_pingpong.py`: existing proposer subclass; initial RPC partitioning and
  evidence annotation. Target verify callbacks and sampled-row/acceptance methods
  are inherited unchanged. The original report writer gains only an optional
  metadata dictionary, empty in legacy mode, without adding a write.
* `dual_pingpong_draft.py`: existing production machine subclass, cohort guards
  and one final in-memory metrics report. The model backend and owner controller
  are reused, including owner-thread construction and shutdown.
* `dual_runner.py`, `dual_service.py`, `dual_batched_draft.py` and the resident
  helper select these classes only for pingpong. The generic microbatch default
  stays 2; pingpong transports its burst-sized readiness bound explicitly.
* `pingpong_comparison.py`: new artifact joins, ten dependency invariants, cohort
  metrics and five-mode table. The shared D6 reporter has one default-off option
  limited to the two-request smoke, whose two cohorts necessarily have one member
  each. Corrected-5/100 still require real multi-request Draft proposal forwards.
* `phase4b4b_pingpong_helpers.sh` and the runbook: structural bring-up, fresh-server
  five-cell execution, offline reporting and immutable packaging. The fixed A
  helper explicitly scopes legacy, so its seven-cell behavior remains isolated.

The report retains the existing Target token-budget, sequence, context and KV
block limits alongside eligible/attained request IDs. If stock clips a cohort,
its internal per-request allocation/preemption cause is not exposed by the
existing hook; the report says so instead of claiming one unproven limit.

There is no new numerical qualification, dynamic split, arrival support,
accumulation, K change, UUID mode change or extra vLLM patch. GPU qualification
and any performance conclusion remain pending the operator runs.
