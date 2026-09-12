"""Deterministic token/KV work executor; no mocked protocol transitions."""

from __future__ import annotations

from specrhythm.continuation.core import DraftCompletion
from specrhythm.phase4.draft_batch import token_tuple


def generate(work, next_token, eos_token_ids=()):
    """Consume missing inputs, then sample, leaving the final sample out of KV.

    The returned frontier is evidence for this immutable branch only. Returning
    it never installs the branch in the owner; the protocol must validate it.
    """
    context = token_tuple(work.dependency_prefix)
    materialized = list(work.base_kv_prefix)
    if tuple(materialized) != context[:len(materialized)]:
        raise ValueError("CPU Draft KV does not match the branch dependency")
    # Models materialize each missing position before obtaining its next logits.
    for token in context[len(materialized):]:
        materialized.append(token)
    generated = []
    budget = work.candidate_length + int(work.kind == "continuation")
    eos = tuple(eos_token_ids) or work.eos_token_ids
    for index in range(budget):
        token = token_tuple((next_token(context + tuple(generated)),))[0]
        generated.append(token)
        if token in eos or index + 1 == budget:
            break
        materialized.append(token)
    return DraftCompletion(work.work_id, tuple(generated), len(materialized))
