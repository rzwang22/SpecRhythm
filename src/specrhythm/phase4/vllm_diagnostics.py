"""Target-only numerical and index diagnostics for Phase 4A.1.1.

The GPU hook in this module is observational.  Its JSONL is never sent over
the Draft-service transport and is forbidden as proposer input.  The pure
validation helpers are exercised on CPU in CI.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from specrhythm.phase4.transport import CheckpointJsonl

DIAGNOSTIC_ENV = "SR_PHASE4_TARGET_DIAGNOSTICS"
DIAGNOSTIC_SCHEMA = "specrhythm.phase4-target-forward-diagnostic.v1"
TARGET_ONLY_FIELDS = frozenset(
    {
        "raw_target_logits",
        "top_raw_logits",
        "top_target_logprobs",
        "selected_target_token_id",
        "target_logits_indices",
        "target_outcome",
        "future_target_tokens",
        "oracle_labels",
    }
)


def token_sha256(token_ids: Sequence[int]) -> str:
    payload = json.dumps(list(token_ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def diagnostic_proposal_id(pending: Any) -> Optional[str]:
    """Return a canonical Dual proposal ID when the proposal protocol has one.

    The Serial ``Proposal`` protocol intentionally has no canonical
    ``proposal_id``.  Diagnostics must not synthesize a Dual identity for it.
    """

    raw_proposal_id = getattr(pending, "proposal_id", None)
    return str(raw_proposal_id) if raw_proposal_id is not None else None


def logits_position_mapping(
    proposal_token_ids: Sequence[int],
    *,
    sampled_logits_offset: int,
    flattened_input_offset: int,
) -> list[dict[str, int]]:
    """Describe vLLM's proposal[j] <- target-logits[j] relation.

    For a linear K-token proposal the first K sampled-logit rows predict the
    K proposal positions; the following row is the bonus-token row.
    """

    return [
        {
            "proposal_index": index,
            "proposal_token_id": int(token_id),
            "target_logits_row": sampled_logits_offset + index,
            "predicts_flattened_input_position": flattened_input_offset + index + 1,
        }
        for index, token_id in enumerate(proposal_token_ids)
    ]


def validate_logits_mapping(value: Mapping[str, Any]) -> list[str]:
    errors = []
    proposal = value.get("proposal_token_ids")
    mapping = value.get("logits_position_mapping")
    proposal = proposal if isinstance(proposal, list) else []
    mapping = mapping if isinstance(mapping, list) else []
    if len(mapping) != len(proposal):
        errors.append("proposal and logits mapping lengths differ")
    for index, row in enumerate(mapping):
        if not isinstance(row, Mapping):
            errors.append("logits mapping row is not an object")
            continue
        if row.get("proposal_index") != index:
            errors.append("proposal index is not contiguous")
        if index < len(proposal) and row.get("proposal_token_id") != proposal[index]:
            errors.append("proposal token does not match its mapped logits row")
        target_row = row.get("target_logits_row")
        input_position = row.get("predicts_flattened_input_position")
        if not isinstance(target_row, int) or not isinstance(input_position, int):
            errors.append("logits mapping indices must be integers")
        if "proposal_matches_vllm_metadata" in row and row.get(
            "proposal_matches_vllm_metadata"
        ) is not True:
            errors.append("proposal token differs from vLLM speculative metadata")
        if "proposal_input_flat_index" in row and row.get(
            "proposal_input_flat_index"
        ) != row.get("sampled_hidden_state_flat_index", -2) + 1:
            errors.append("proposal and target logits input positions are off by one")
    return errors


def validate_target_diagnostic(value: Mapping[str, Any]) -> list[str]:
    errors = []
    if value.get("schema_version") != DIAGNOSTIC_SCHEMA:
        errors.append("unsupported target diagnostic schema")
    for key in (
        "request_id",
        "committed_prefix_token_ids",
        "committed_prefix_sha256",
        "proposal_token_ids",
        "logical_target_kv_length",
        "scheduled_token_count",
        "query_length",
        "sequence_length",
        "logits_position_mapping",
        "position_ids",
        "target_input_token_ids",
        "target_forward_start_ns",
        "target_forward_end_ns",
        "top_raw_logits",
        "top_target_logprobs",
        "target_verification_shape",
        "attention_backend",
        "all_reduce_backend",
        "dtype",
        "batch_invariant_requested",
    ):
        if key not in value:
            errors.append(f"target diagnostic is missing {key}")
    prefix = value.get("committed_prefix_token_ids")
    if not isinstance(prefix, list) or value.get("committed_prefix_sha256") != token_sha256(
        prefix if isinstance(prefix, list) else []
    ):
        errors.append("committed prefix checksum is invalid")
    errors.extend(validate_logits_mapping(value))
    positions = value.get("position_ids")
    if isinstance(positions, list) and positions:
        if positions != list(range(positions[0], positions[0] + len(positions))):
            errors.append("position IDs are not contiguous")
    inputs = value.get("target_input_token_ids")
    if not isinstance(inputs, list) or len(inputs) != value.get("query_length"):
        errors.append("Target input-token count differs from query length")
    start = value.get("target_forward_start_ns")
    end = value.get("target_forward_end_ns")
    if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end:
        errors.append("Target forward timestamps are invalid")
    if value.get("target_kv_contains_rejected_or_future_tokens") is not False:
        errors.append("target KV contains rejected or future tokens")
    if value.get("causal_attention") is not True:
        errors.append("target diagnostic does not prove causal attention")
    mask = value.get("attention_mask_proof")
    if not isinstance(mask, Mapping):
        errors.append("attention mask proof is missing")
    elif value.get("proposal_token_ids"):
        if mask.get("causal") is not True:
            errors.append("speculative attention is not causal")
        if mask.get("query_start_range") is None:
            errors.append("speculative query-start metadata is missing")
        if mask.get("query_length_matches_scheduler") is not True:
            errors.append("attention query length differs from scheduler input")
        if mask.get("positions_contiguous") is not True:
            errors.append("attention positions are not contiguous")
    return errors


def validate_kv_monotonicity(round_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors = []
    previous: dict[str, int] = {}
    finished: set[str] = set()
    for row in round_rows:
        request_id = str(row.get("request_id", ""))
        parent = row.get("parent_prefix_len")
        committed = row.get("committed_tokens")
        target_length = row.get("logical_target_kv_length")
        if not request_id or not all(
            isinstance(item, int) for item in (parent, committed, target_length)
        ):
            errors.append("round KV accounting fields are invalid")
            continue
        if request_id in finished:
            errors.append(f"request {request_id} produced a round after termination")
        if request_id in previous and parent != previous[request_id]:
            errors.append(f"request {request_id} next parent does not equal prior committed KV")
        if target_length != parent + committed:
            errors.append(f"request {request_id} target KV length violates commit accounting")
        if request_id in previous and target_length < previous[request_id]:
            errors.append(f"request {request_id} target KV regressed")
        previous[request_id] = target_length
        if row.get("terminal") is True:
            finished.add(request_id)
    return errors


def compare_divergence_diagnostics(
    stock_rows: Sequence[Mapping[str, Any]],
    speculative_rows: Sequence[Mapping[str, Any]],
    *,
    request_id: str,
    committed_prefix_sha256: str,
) -> dict[str, Any]:
    """Prove prefix/index/position/KV equivalence at one divergence point."""

    def find(rows: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
        matches = [
            row
            for row in rows
            if row.get("request_id") == request_id
            and row.get("committed_prefix_sha256") == committed_prefix_sha256
        ]
        if not matches:
            return None
        def semantic(row: Mapping[str, Any]) -> dict[str, Any]:
            value = dict(row)
            value.pop("internal_request_id", None)
            value.pop("record_sha256", None)
            return value

        first = semantic(matches[0])
        if any(semantic(row) != first for row in matches[1:]):
            return None
        return matches[0]

    stock = find(stock_rows)
    speculative = find(speculative_rows)
    errors = []
    if stock is None:
        errors.append("stock diagnostic has no unique matching prefix")
    if speculative is None:
        errors.append("speculative diagnostic has no unique matching prefix")
    if stock is not None:
        errors.extend(f"stock: {item}" for item in validate_target_diagnostic(stock))
    if speculative is not None:
        errors.extend(
            f"speculative: {item}" for item in validate_target_diagnostic(speculative)
        )
    checks = {
        "same_committed_prefix": False,
        "proposal_logits_mapping_valid": False,
        "positions_and_mask_valid": False,
        "target_kv_rollback_valid": False,
    }
    if stock is not None and speculative is not None:
        checks["same_committed_prefix"] = (
            stock.get("committed_prefix_token_ids")
            == speculative.get("committed_prefix_token_ids")
        )
        checks["proposal_logits_mapping_valid"] = not validate_logits_mapping(speculative)
        positions = speculative.get("position_ids")
        checks["positions_and_mask_valid"] = bool(
            speculative.get("causal_attention") is True
            and isinstance(positions, list)
            and (
                not positions
                or positions == list(range(positions[0], positions[0] + len(positions)))
            )
        )
        checks["target_kv_rollback_valid"] = (
            speculative.get("target_kv_contains_rejected_or_future_tokens") is False
            and speculative.get("logical_target_kv_length")
            == len(speculative.get("committed_prefix_token_ids", ()))
        )
    for name, valid in checks.items():
        if not valid:
            errors.append(f"divergence proof failed: {name}")
    return {
        "schema_version": "specrhythm.phase4-divergence-proof.v1",
        "request_id": request_id,
        "committed_prefix_sha256": committed_prefix_sha256,
        "checks": checks,
        "valid": not errors,
        "errors": errors,
        "target_diagnostics_visible_to_draft": False,
        "stock_diagnostic": dict(stock) if stock is not None else None,
        "speculative_diagnostic": (
            dict(speculative) if speculative is not None else None
        ),
    }


def compare_fixed_proposal_controls(
    local: Mapping[str, Any], remote: Mapping[str, Any]
) -> dict[str, Any]:
    fields = (
        "proposal_token_ids",
        "top_raw_logits",
        "top_target_logprobs",
        "accepted_prefix_length",
        "committed_token_ids",
    )
    checks = {field: local.get(field) == remote.get(field) for field in fields}
    return {
        "schema_version": "specrhythm.phase4-fixed-proposal-comparison.v1",
        "fixed_proposal": [53143, 2213, 369, 264],
        "local_remote_equal": all(checks.values()),
        "checks": checks,
    }


def capture_target_forward(
    runner: Any,
    *,
    scheduler_output: Any,
    logits: Any,
    spec_decode_metadata: Any,
    logits_indices: Any,
    positions: Any,
    num_scheduled_tokens: Sequence[int],
    common_attention_metadata: Any,
    target_forward_start_ns: int,
    target_forward_end_ns: int,
) -> None:
    """GPU-only hook called from the pinned vLLM Target runner patch."""

    output = os.environ.get(DIAGNOSTIC_ENV)
    if not output:
        return
    from vllm.distributed.parallel_state import get_tp_group

    if int(get_tp_group().rank_in_group) != 0:
        return
    workload_path = os.environ.get("SR_PHASE4_WORKLOAD")
    if not workload_path:
        raise RuntimeError("SR_PHASE4_WORKLOAD is required for stable Target diagnostics")
    from specrhythm.phase4.batch_invariant import BATCH_INVARIANT_ENV
    from specrhythm.phase4.stock_vllm import load_smoke_requests

    workload = Path(workload_path).resolve()
    request_count = sum(
        bool(line.strip()) for line in workload.read_text(encoding="utf-8").splitlines()
    )
    requests = load_smoke_requests(
        workload, expected_count=request_count, require_task_mixture=False
    )
    definitions = {tuple(row.prompt_token_ids): row for row in requests}
    scheduled_spec = scheduler_output.scheduled_spec_decode_tokens
    sampled_offsets = []
    flattened_offsets = []
    sample_cursor = 0
    flat_cursor = 0
    for index, internal_id in enumerate(runner.input_batch.req_ids):
        proposal = tuple(int(item) for item in scheduled_spec.get(internal_id, ()))
        sampled_offsets.append(sample_cursor)
        flattened_offsets.append(flat_cursor)
        sample_cursor += len(proposal) + 1
        flat_cursor += int(num_scheduled_tokens[index])
    logits_cpu = logits.detach().float().cpu()
    positions_cpu = positions.detach().cpu().tolist()
    logits_indices_cpu = logits_indices.detach().cpu().tolist()
    target_logits_indices_cpu = (
        spec_decode_metadata.target_logits_indices.detach().cpu().tolist()
        if spec_decode_metadata is not None
        else []
    )
    draft_token_ids_cpu = (
        spec_decode_metadata.draft_token_ids.detach().cpu().tolist()
        if spec_decode_metadata is not None
        else []
    )
    top_count = min(10, int(logits_cpu.shape[-1]))
    logprobs = logits_cpu.log_softmax(dim=-1)
    attn_names = []
    for groups in getattr(runner, "attn_groups", ()):
        for group in groups:
            backend = getattr(group, "backend", None)
            if backend is not None:
                get_name = getattr(backend, "get_name", None)
                attn_names.append(
                    str(get_name()) if callable(get_name) else str(backend.__name__)
                )
    all_reduce = (
        "PYNCCL-or-torch-distributed"
        if runner.vllm_config.parallel_config.disable_custom_all_reduce
        else "vLLM-runtime-dispatch"
    )
    log = CheckpointJsonl(Path(output).resolve())
    draft_cursor = 0
    for index, internal_id in enumerate(runner.input_batch.req_ids):
        count = int(runner.input_batch.num_tokens_no_spec[index])
        tokens = tuple(
            int(item) for item in runner.input_batch.token_ids_cpu[index, :count].tolist()
        )
        matches = [
            definition
            for prefix, definition in definitions.items()
            if tokens[: len(prefix)] == prefix
        ]
        if len(matches) != 1:
            raise RuntimeError("Target diagnostic cannot map a unique stable request")
        definition = matches[0]
        proposal = [int(item) for item in scheduled_spec.get(internal_id, ())]
        sample_offset = sampled_offsets[index]
        flat_offset = flattened_offsets[index]
        rows = list(range(sample_offset, sample_offset + max(len(proposal), 1)))
        raw_top = []
        prob_top = []
        selected = []
        for row_index in rows:
            raw_values, raw_ids = logits_cpu[row_index].topk(top_count)
            prob_values, prob_ids = logprobs[row_index].topk(top_count)
            raw_top.append(
                [
                    {"token_id": int(token_id), "raw_logit": float(value)}
                    for token_id, value in zip(raw_ids.tolist(), raw_values.tolist())
                ]
            )
            prob_top.append(
                [
                    {"token_id": int(token_id), "log_probability": float(value)}
                    for token_id, value in zip(prob_ids.tolist(), prob_values.tolist())
                ]
            )
            selected.append(int(logits_cpu[row_index].argmax().item()))
        query_length = int(num_scheduled_tokens[index])
        position_slice = [
            int(item)
            for item in positions_cpu[flat_offset : flat_offset + query_length]
        ]
        computed = int(runner.input_batch.num_computed_tokens_cpu[index])
        target_input_token_ids = [
            int(item)
            for item in runner.input_batch.token_ids_cpu[
                index, computed : computed + query_length
            ].tolist()
        ]
        attention_query_start = None
        attention_sequence_length = None
        attention_causal = True
        if common_attention_metadata is not None:
            query_start = common_attention_metadata.query_start_loc_cpu.tolist()
            if index + 1 < len(query_start):
                attention_query_start = [
                    int(query_start[index]),
                    int(query_start[index + 1]),
                ]
            sequence_lengths = common_attention_metadata.seq_lens.detach().cpu().tolist()
            if index < len(sequence_lengths):
                attention_sequence_length = int(sequence_lengths[index])
            causal = common_attention_metadata.causal
            attention_causal = causal is True
        mapping = logits_position_mapping(
            proposal,
            sampled_logits_offset=sample_offset,
            flattened_input_offset=flat_offset,
        )
        for local_index, item in enumerate(mapping):
            global_draft_index = draft_cursor + local_index
            if global_draft_index < len(target_logits_indices_cpu):
                target_row = int(target_logits_indices_cpu[global_draft_index])
                item["target_logits_row"] = target_row
                item["sampled_hidden_state_flat_index"] = int(
                    logits_indices_cpu[target_row]
                )
                item["proposal_input_flat_index"] = int(
                    logits_indices_cpu[target_row + 1]
                )
                item["vllm_draft_token_id"] = int(
                    draft_token_ids_cpu[global_draft_index]
                )
                item["proposal_matches_vllm_metadata"] = (
                    item["proposal_token_id"] == item["vllm_draft_token_id"]
                )
        prefix = list(tokens[:count])
        round_id = None
        drafter_state = getattr(getattr(runner, "drafter", None), "requests", {})
        stable_state = (
            drafter_state.get(definition.request_id)
            if isinstance(drafter_state, Mapping)
            else None
        )
        pending = getattr(stable_state, "pending_proposal", None)
        if pending is not None:
            prefix = list(getattr(stable_state, "committed_token_ids", prefix))
            round_id = int(pending.round_id)
        proposal_id = diagnostic_proposal_id(pending)
        row = {
            "schema_version": DIAGNOSTIC_SCHEMA,
            "request_id": definition.request_id,
            "internal_request_id": str(internal_id),
            "round_id": round_id,
            "proposal_id": proposal_id,
            "committed_prefix_token_ids": prefix,
            "committed_prefix_sha256": token_sha256(prefix),
            "logical_committed_prefix_count": len(prefix),
            "target_pending_input_token_id": prefix[-1] if prefix else None,
            "target_pending_input_position": len(prefix) - 1 if prefix else None,
            "proposal_token_ids": proposal,
            "logical_target_kv_length": len(prefix),
            "physical_kv_num_computed_tokens": computed,
            "scheduled_token_count": query_length,
            "query_length": query_length,
            "sequence_length": computed + query_length,
            "logits_position_mapping": mapping,
            "position_ids": position_slice,
            "target_input_token_ids": target_input_token_ids,
            "target_forward_start_ns": int(target_forward_start_ns),
            "target_forward_end_ns": int(target_forward_end_ns),
            "attention_mask_proof": {
                "causal": attention_causal,
                "query_start_range": attention_query_start,
                "query_length_matches_scheduler": (
                    attention_query_start is None
                    or attention_query_start[1] - attention_query_start[0]
                    == query_length
                ),
                "attention_sequence_length": attention_sequence_length,
                "positions_contiguous": (
                    not position_slice
                    or position_slice
                    == list(
                        range(position_slice[0], position_slice[0] + len(position_slice))
                    )
                ),
            },
            "top_raw_logits": raw_top,
            "top_target_logprobs": prob_top,
            "selected_target_token_id": selected,
            "target_verification_shape": {
                "request_count": len(runner.input_batch.req_ids),
                "scheduled_input_positions": int(sum(num_scheduled_tokens)),
                "sampled_logits_rows": int(logits_cpu.shape[0]),
                "vocab_size": int(logits_cpu.shape[-1]),
            },
            "attention_backend": sorted(set(attn_names)),
            "all_reduce_backend": all_reduce,
            "dtype": str(logits.dtype),
            "batch_invariant_requested": os.environ.get(BATCH_INVARIANT_ENV) == "1",
            "causal_attention": attention_causal,
            "target_kv_contains_rejected_or_future_tokens": computed > len(prefix),
            "target_only_artifact": True,
            "visible_to_draft": False,
        }
        log.append(row)
        draft_cursor += len(proposal)
