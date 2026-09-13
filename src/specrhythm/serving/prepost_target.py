"""Opt-in projection at pinned vLLM's synchronous bookkeeping publication boundary.

Stock ragged rejection sampling still checks every real candidate. On full
accept its additional bonus is unused. Project before custom propose, scheduler
update or client output; do not change model forwards, sampling or old modes.
"""

from __future__ import annotations

import functools
import os
import time

from specrhythm.continuation.prepost import PROTOCOL, canonical_sample
from specrhythm.continuation.prepost_records import Records
from specrhythm.continuation.trace import TRACE
from specrhythm.serving.common import require


def install(runner):
    from specrhythm.serving.k3 import PROTOCOL as K3_PROTOCOL

    uniform = os.environ.get("SR_S2_MODE", "").endswith("-k3")
    protocol = K3_PROTOCOL if uniform else PROTOCOL
    if getattr(runner, "prepost_protocol", None) == protocol:
        return
    require(not runner.use_async_scheduling, "prepost requires synchronous Target bookkeeping")
    original = runner._bookkeeping_sync
    runner.prepost_protocol = protocol
    runner.prepost_samples = Records()

    @functools.wraps(original)
    def bookkeeping(scheduler_output, *args, **kwargs):
        batch = runner.input_batch
        ids = tuple(batch.req_ids)
        before = {
            rid: (int(batch.num_tokens_no_spec[i]), len(runner.requests[rid].output_token_ids))
            for i, rid in enumerate(ids)
        }
        result = original(scheduler_output, *args, **kwargs)
        sampled, output_ids, indices = result[2], result[4], result[5]
        require(tuple(output_ids) == ids, "prepost Target row order changed during bookkeeping")
        changes, events = [], []
        for rid, candidates in scheduler_output.scheduled_spec_decode_tokens.items():
            i = indices[rid]
            raw = sampled[i]
            if not raw:  # Stock discarded/prefill row; not a verified output.
                continue
            state = runner.requests[rid]
            count, generated = before[rid]
            budget = min(3, state.sampling_params.max_tokens - generated)
            if uniform:
                require(len(candidates) == budget or (0 < len(candidates) < budget
                        and candidates[-1] in runner.drafter.eos_token_ids),
                        "Target sampled incomplete K3 proposal")
            decision, counters = canonical_sample(
                candidates,
                raw,
                previous=generated,
                maximum=state.sampling_params.max_tokens,
                eos=runner.drafter.eos_token_ids,
            )
            canonical = list(decision.committed_token_ids)
            require(
                state.output_token_ids[generated:] == raw
                and int(batch.num_tokens_no_spec[i]) == count + len(raw),
                "stock Target token cache differs from parsed output",
            )
            changes.append((i, state, generated, count, len(raw), canonical))
            events.append(
                dict(
                    internal_request_id=rid,
                    actual_candidate_length=len(candidates),
                    **({"remaining_before": state.sampling_params.max_tokens - generated,
                        "short_reason": "candidate_EOS"
                        if candidates[-1] in runner.drafter.eos_token_ids
                        else "output_budget" if budget < 3 else None} if uniform else {}),
                    committed_tokens=len(canonical),
                    accepted_tokens=len(decision.accepted_draft_token_ids),
                    correction_tokens=len(decision.target_correction_token_ids),
                    terminal=decision.terminal,
                    **counters,
                )
            )
        # Validate the entire batch before altering any published row. Logprob
        # offsets refer to stock packed rows and stay valid: slice_request uses
        # the *canonical* count and therefore excludes unused bonus/stop suffix.
        for i, state, generated, count, raw_length, canonical in changes:
            sampled[i] = canonical
            state.output_token_ids[generated:] = canonical
            end = count + len(canonical)
            batch.num_tokens_no_spec[i] = end
            batch.token_ids_cpu[i, count:end] = canonical
            batch.is_token_ids[i, end : count + raw_length] = False
        if events:
            row = dict(
                timestamp_ns=time.monotonic_ns(),
                requests=events,
                actual_candidate_lengths=[r["actual_candidate_length"] for r in events],
                unused_bonus_is_committed=False,
                protocol=protocol,
                target_effective_query_positions={
                    rid: scheduler_output.num_scheduled_tokens[rid] for rid in ids
                },
            )
            runner.prepost_samples.append(row)
            TRACE.event("prepost_target_sample", **row)
        return result

    runner._bookkeeping_sync = bookkeeping
