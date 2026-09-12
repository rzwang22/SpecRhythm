# Phase 4B.1 Dual request and proposal state machines

This document is the evidence contract for real decode-only Dual-Batch. It is not a scheduling
policy specification and contains no performance claim.

## Initialization boundary

For each manifest request, `logical_prefix = prompt + bootstrap`. Target materializes
`logical_prefix[:-1]` and retains `logical_prefix[-1]` as the pending input. Draft materializes the
complete logical prefix. `prefix_version=1`, `next_round_id=0`, state is `DRAFT_READY`, and no
proposal is installed. All observations are complete and validated before the Target TP barrier;
`measurement_start_ns` follows the barrier. Every proposal `draft_start_ns` must be at or after
that boundary.

## Request transitions

```text
DRAFT_READY
  -> DRAFTING
  -> PROPOSAL_READY
  -> VERIFY_READY
  -> VERIFYING
  -> COMMITTING
  -> DRAFT_SYNC
  -> DRAFT_READY

DRAFTING or DRAFT_SYNC
  -> TARGET_TAIL_READY
  -> VERIFYING
  -> COMMITTING
  -> TERMINAL
```

Verified EOS or max-token termination may move `COMMITTING -> TERMINAL`. `FAILED` is fail-closed.
New artifacts never serialize `FINISHED`; that compatibility enum aliases `TERMINAL`. Every event
records stable and internal IDs, source/destination, round, prefix version/count/hash, proposal ID,
reason and monotonic timestamp. Per request, timestamps and prefix versions never regress.

Draft synchronization occurs only after Target commit. The next proposal cannot become
`PROPOSAL_READY` until Draft has rolled back rejected suffix positions, consumed the Target
correction or bonus, and reconstructed the complete new committed prefix.

## Proposal transitions

```text
CREATED -> PUBLISHED -> INSTALLED -> CONSUMED
                         \
                          -> DROPPED_STALE
```

The implementation also permits `CREATED -> PUBLISHED -> DROPPED_STALE` when a request terminates
before installation. A `(request_id, round_id)` has one canonical proposal ID. `CONSUMED` comes
only from pinned vLLM's `scheduled_spec_decode_tokens`; an installed proposal is never consumed
twice. Conservation is `published = consumed + dropped`.

## Target input and commit

For parent prefix length `L` and proposal `d[0:K]`, the exact Target input is:

```text
token IDs = [logical_prefix[-1]] + d[0:K]
positions = [L-1, L, ..., L+K-1]
materialized Target KV before forward = L-1
```

Accepted Draft tokens are a strict proposal prefix; rejected tokens are its remaining suffix.
Correction and bonus are mutually exclusive. Absent explicit EOS/stop/max-token truncation,
`committed = accepted + correction_or_bonus`. Logical Target state after commit is
`KV(committed_prefix[:-1]) + pending(committed_prefix[-1])`; logical Draft state is the complete
committed prefix. Physical rejected slots may remain allocated but cannot be referenced or counted
as live KV.

## Scheduler states

`WAITING_DRAFT` and `DRAFTING` are inadmissible and consume no Target budget. `PROPOSAL_READY` and
`VERIFY_READY` require an exact request/round/prefix match. `TARGET_TAIL_READY` permits only the
legal terminal tail. `TERMINAL` never executes. A denied request is skipped before stock budget
or KV allocation, allowing later ready requests to proceed without head-of-line blocking.

The Gate 1 `one-ready` switch publishes at most one proposal on the first ready cycle without
waiting for Draft; this deterministically constructs Case A. The `two-ready` switch withholds
polling until queue metadata reports two ready proposals, constructing Case B. Neither executes
Draft GPU work in the Target scheduler. Both are recorded and disabled outside construction runs.
