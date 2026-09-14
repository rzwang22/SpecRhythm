"""Reuse an immutable prompt digest, never a live KV or admission decision."""

from specrhythm.continuation.trace import TRACE
from specrhythm.serving.common import require
from specrhythm.serving.s2_pool import prefix_record

POLICY = "k3-current-prompt-proof.v1"


def physical_rows(scheduler):
    proofs = getattr(scheduler, "_k3_prompt_proofs", None)
    if proofs is None:
        scheduler._k3_prompt_proofs = proofs = {}
    rows, reused, hashed, compared = {}, 0, 0, 0
    identity = scheduler._identity()
    with TRACE.span("target_pool_snapshot", policy=POLICY):
        for internal, request in scheduler.requests.items():
            if request.is_finished() or not request.num_output_tokens:
                continue
            rid = identity.stable_id(str(internal))
            # Compare the complete CURRENT prompt on every visit. Object identity,
            # length, an old version or a cached 'passed' boolean cannot prove it.
            tokens = tuple(request.prompt_token_ids)
            proof = proofs.get(rid)
            blocks = scheduler.kv_cache_manager.get_block_ids(internal)
            if proof is None:
                row = prefix_record(tokens, request.num_computed_tokens, blocks)
                proofs[rid] = (str(internal), tokens, row["prefix_sha256"])
                hashed += 1
            else:
                require(
                    proof[0] == str(internal)
                    and tokens == proof[1]
                    and all(type(t) is int for t in tokens),
                    "K3 resident prompt identity/content changed",
                    request_id=rid,
                )
                compared += len(tokens)
                row = dict(
                    prefix_sha256=proof[2],
                    materialized_tokens=int(request.num_computed_tokens),
                    block_ids=[list(g) for g in blocks],
                )
                reused += 1
            require(rid not in rows, "K3 duplicate live prompt binding", request_id=rid)
            rows[rid] = row
        # Removed/finished handles cannot supply a proof to a later lifecycle.
        for rid in set(proofs) - set(rows):
            del proofs[rid]
        TRACE.event(
            "target_pool_prompt_proof",
            policy=POLICY,
            resident_rows=len(rows),
            prompt_hashes=hashed,
            prompt_proofs_reused=reused,
            current_prompt_tokens_compared=compared,
        )
    return rows
