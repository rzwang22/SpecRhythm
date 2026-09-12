"""Post-generation closure of retired Dual request evidence; no execution hooks.

The coordinator owns final serialized outputs. Scheduler retirement alone cannot
establish successful completion, and this module never changes proposal evidence.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from specrhythm.phase4.decode_ready import DecodeReadyManifest
from specrhythm.phase4.dual import proposal_identity
from specrhythm.phase4.dual_correctness import (
    _LEGAL_STATES,
    validate_proposal_lifecycle_events,
    validate_request_state_events,
)
from specrhythm.phase4.serial import token_prefix_hash

RECONCILIATION_SCHEMA = "specrhythm.phase4b2-terminal-state-reconciliation.v1"
CLOSURE_REASON = "stock-vllm-retired-after-final-output"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _unique(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    result = {}
    for row in rows:
        request_id = row.get("request_id")
        _require(isinstance(request_id, str) and bool(request_id), f"{label}: invalid request ID")
        _require(request_id not in result, f"{label}: duplicate request ID {request_id}")
        result[request_id] = row
    return result


def completed_output_prefixes(
    requests: Sequence[Any],
    outputs: Sequence[Mapping[str, Any]],
    manifest: DecodeReadyManifest,
) -> dict[str, tuple[int, ...]]:
    """Validate serialized stock completion against the frozen requested work.

    The serializer runs only after output.finished and exactly one completion.
    Stock finish_reason is authoritative for EOS stopping (including its default
    stop_reason=None); no guessed model EOS IDs or new token diagnostics are used.
    """

    definitions = {row.request_id: row for row in requests}
    final = _unique(outputs, "final output")
    ready = {row.request_id: row for row in manifest.requests}
    _require(len(definitions) == len(requests), "duplicate workload request ID")
    _require(
        set(definitions) == set(final) == set(ready), "final output request set is incomplete"
    )
    prefixes = {}
    for request_id, definition in definitions.items():
        row = final[request_id]
        tokens = row.get("generated_token_ids")
        _require(
            isinstance(tokens, list) and bool(tokens), f"{request_id}: missing generated tokens"
        )
        token_prefix_hash(tokens)  # Strict integer/token validation, including booleans.
        maximum = definition.maximum_new_tokens
        _require(
            0 < len(tokens) <= maximum, f"{request_id}: output exceeds requested token budget"
        )
        reason = row.get("finish_reason")
        _require(
            reason in {"length", "stop"}, f"{request_id}: final output is not successful/terminal"
        )
        _require(
            row.get("finished", True) is True, f"{request_id}: output is explicitly unfinished"
        )
        if reason == "length":
            _require(
                len(tokens) == maximum, f"{request_id}: length finish does not fill token budget"
            )
            _require(
                row.get("stop_reason") is None, f"{request_id}: length finish has a stop reason"
            )
        else:
            stop = row.get("stop_reason")
            _require(
                stop is None or (type(stop) is int and stop >= 0 and tokens[-1] == stop),
                f"{request_id}: stop reason contradicts final tokens/requested sampling",
            )
        prompt = tuple(definition.prompt_token_ids)
        bootstrap = ready[request_id]
        _require(
            bootstrap.prompt_token_count == len(prompt)
            and bootstrap.prompt_token_ids_sha256 == token_prefix_hash(prompt)
            and tokens[0] == bootstrap.bootstrap_token_id,
            f"{request_id}: output/prompt/bootstrap identity mismatch",
        )
        _require(
            row.get("prompt_length") == len(prompt)
            and row.get("generated_tokens") == len(tokens)
            and row.get("token_accounting")
            == {
                "prompt_tokens": len(prompt),
                "generated_tokens": len(tokens),
                "total_tokens": len(prompt) + len(tokens),
            },
            f"{request_id}: serialized output accounting mismatch",
        )
        prefixes[request_id] = prompt + tuple(tokens)
    return prefixes


def build_terminal_reconciliation(
    *,
    requests: Sequence[Any],
    outputs: Sequence[Mapping[str, Any]],
    manifest: DecodeReadyManifest,
    identity: Mapping[str, Any],
    state_rows: Sequence[Mapping[str, Any]],
    scheduler_rows: Sequence[Mapping[str, Any]],
    lifecycle_rows: Sequence[Mapping[str, Any]],
    proposal_rows: Sequence[Mapping[str, Any]],
    observation_ns: int,
) -> dict[str, Any]:
    """Return only justified missing transitions, or fail without mutating inputs.

    The caller appends events only after this entire validation succeeds. The
    resulting stream is checked by the unchanged terminal-state validator.
    """

    prefixes = completed_output_prefixes(requests, outputs, manifest)
    _require(
        identity.get("mapping_source") == "unique frozen prompt_token_ids"
        and identity.get("suffix_parsing") is False,
        "terminal reconciliation lacks authoritative frozen-prompt bindings",
    )
    bindings = _unique(identity.get("bindings", ()), "historical binding")
    internal_ids = [row.get("internal_request_id") for row in bindings.values()]
    _require(
        set(bindings) == set(prefixes)
        and identity.get("bound_request_count") == len(bindings)
        and all(isinstance(item, str) and item for item in internal_ids)
        and len(set(internal_ids)) == len(internal_ids),
        "historical stable/internal identity bindings are incomplete or aliased",
    )
    for row in manifest.requests:
        _require(
            bindings[row.request_id]["internal_request_id"] == row.internal_target_request_id,
            f"{row.request_id}: setup and historical internal identity mismatch",
        )
    traces: dict[str, list[Mapping[str, Any]]] = {key: [] for key in prefixes}
    for row in state_rows:
        request_id = row.get("request_id")
        _require(request_id in traces, "request state contains an unknown identity")
        _require(
            row.get("internal_request_id") == bindings[request_id]["internal_request_id"],
            f"{request_id}: state internal identity mismatch",
        )
        traces[request_id].append(row)
    _require(all(traces.values()), "a request has no state trace")
    _require(
        type(observation_ns) is int and observation_ns > 0, "invalid closure observation time"
    )
    additions = []
    for request_id, trace in traces.items():
        last = trace[-1]
        prefix = prefixes[request_id]
        for row in trace:
            count = row.get("committed_prefix_length")
            _require(
                all(
                    type(row.get(key)) is int and row[key] >= 0
                    for key in ("prefix_version", "round_id", "timestamp_ns")
                ),
                f"{request_id}: state version/round/timestamp is malformed",
            )
            _require(
                type(count) is int
                and 0 < count <= len(prefix)
                and row.get("committed_prefix_sha256") == token_prefix_hash(prefix[:count]),
                f"{request_id}: state prefix/hash contradicts final output",
            )
        _require(
            last["committed_prefix_length"] == len(prefix),
            f"{request_id}: final output includes tokens absent from the last committed state",
        )
        if last.get("destination_state") == "TERMINAL":
            errors = validate_request_state_events(trace)
            _require(not errors, "; ".join(errors))
            continue
        predecessor = last.get("destination_state")
        _require(
            all(row.get("destination_state") not in {"FAILED", "TERMINAL"} for row in trace)
            and "TERMINAL" in _LEGAL_STATES.get(predecessor, set()),
            f"{request_id}: state trace cannot legally close to TERMINAL",
        )
        retired = _retired_evidence(
            request_id,
            bindings[request_id]["internal_request_id"],
            scheduler_rows,
        )
        _require(bool(retired), f"{request_id}: missing stock-retired ready-result evidence")
        _validate_late_work(request_id, last, retired, lifecycle_rows, proposal_rows)
        latest = max(last["timestamp_ns"], *(row["timestamp_ns"] for row in retired))
        _require(observation_ns > latest, f"{request_id}: closure timestamp is not strictly later")
        additions.append(
            {
                "schema_version": "specrhythm.phase4b-request-state-event.v1",
                "request_id": request_id,
                "internal_request_id": bindings[request_id]["internal_request_id"],
                "source_state": predecessor,
                "destination_state": "TERMINAL",
                "prefix_version": last["prefix_version"],
                "round_id": last["round_id"],
                "committed_prefix_length": len(prefix),
                "committed_prefix_sha256": token_prefix_hash(prefix),
                "proposal_id": None,
                "reason": CLOSURE_REASON,
                "timestamp_ns": observation_ns,
                "observation_phase": "post-generation-terminal-reconciliation",
                "retired_ready_timestamps_ns": [row["timestamp_ns"] for row in retired],
                "final_generated_token_sha256": token_prefix_hash(
                    _unique(outputs, "final output")[request_id]["generated_token_ids"]
                ),
                "proposal_installed": False,
                "proposal_verified": False,
                "proposal_committed": False,
            }
        )
    errors = validate_request_state_events([*state_rows, *additions])
    _require(not errors, "; ".join(errors))
    return {
        "schema_version": RECONCILIATION_SCHEMA,
        "valid": True,
        "errors": [],
        "source_execution_commit": manifest.specrhythm_git_commit,
        "observation_timestamp_ns": observation_ns,
        "reconciled_request_ids": [row["request_id"] for row in additions],
        "events": additions,
    }


def _retired_evidence(
    request_id: str,
    internal_id: str,
    scheduler_rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    result = []
    seen = set()
    for cycle in scheduler_rows:
        for event in cycle.get("retired_ready_results", ()):
            if event.get("request_id") != request_id:
                continue
            _require(
                cycle.get("schema_version") == "specrhythm.phase4b-scheduler-cycle.v1"
                and type(cycle.get("poll_start_ns")) is int
                and type(cycle.get("poll_end_ns")) is int
                and event.get("schema_version") == "specrhythm.phase4b2-retired-ready-result.v1"
                and event.get("internal_request_id") == internal_id
                and event.get("reason") == "request-retired-before-ready"
                and event.get("discarded") is True
                and event.get("installed") is False
                and event.get("verified") is False
                and type(event.get("timestamp_ns")) is int
                and cycle.get("poll_start_ns", -1)
                <= event["timestamp_ns"]
                <= cycle.get("poll_end_ns", -1),
                f"{request_id}: invalid retired-ready evidence",
            )
            key = (
                event.get("result_kind"),
                event.get("proposal_id"),
                event.get("target_tail_ready_ns"),
            )
            _require(key not in seen, f"{request_id}: duplicate retired-ready evidence")
            seen.add(key)
            result.append(event)
        if result:
            _require(
                request_id not in cycle.get("scheduled_request_ids", ())
                and request_id not in cycle.get("verify_request_ids", ())
                and not any(
                    row.get("request_id") == request_id
                    or row.get("internal_request_id") == internal_id
                    for row in cycle.get("request_admissibility", ())
                ),
                f"{request_id}: scheduler shows a live request after retirement",
            )
    return result


def _validate_late_work(
    request_id: str,
    last: Mapping[str, Any],
    retired: Sequence[Mapping[str, Any]],
    lifecycle_rows: Sequence[Mapping[str, Any]],
    proposal_rows: Sequence[Mapping[str, Any]],
) -> None:
    for event in retired:
        if event.get("result_kind") == "target-tail":
            _require(
                event.get("proposal_id") is None
                and type(event.get("target_tail_ready_ns")) is int
                and 0 < event["target_tail_ready_ns"] <= event["timestamp_ns"],
                f"{request_id}: malformed retired tail",
            )
            continue
        _require(
            event.get("result_kind") == "proposal", f"{request_id}: unknown retired result kind"
        )
        proposal_id = event.get("proposal_id")
        rows = [row for row in lifecycle_rows if row.get("proposal_id") == proposal_id]
        errors = validate_proposal_lifecycle_events(rows)
        _require(not errors, "; ".join(errors))
        _require(
            [row["lifecycle_state"] for row in rows] == ["CREATED", "PUBLISHED", "DROPPED_STALE"]
            and rows[-1].get("reason") == "request-retired-before-ready"
            and rows[-1]["timestamp_ns"] <= event["timestamp_ns"]
            and not any(row.get("proposal_id") == proposal_id for row in proposal_rows),
            f"{request_id}: late proposal was installed, verified or committed",
        )
        for row in rows:
            tokens = row.get("proposal_token_ids", ())
            token_prefix_hash(tokens)
            _require(
                row.get("request_id") == request_id
                and row.get("internal_request_id") == event["internal_request_id"]
                and row.get("prefix_token_count") == last["committed_prefix_length"]
                and row.get("prefix_token_sha256") == last["committed_prefix_sha256"]
                and row.get("prefix_version") == last["prefix_version"]
                and row.get("round_id") == last["round_id"]
                and all(
                    type(row.get(key)) is int
                    for key in (
                        "prefix_version",
                        "round_id",
                        "prefix_token_count",
                        "proposal_length",
                    )
                )
                and 1 <= len(tokens) <= 4
                and row.get("proposal_length") == len(tokens)
                and proposal_id
                == proposal_identity(
                    request_id,
                    row["round_id"],
                    row["prefix_version"],
                    tokens,
                ),
                f"{request_id}: retired proposal prefix/identity is inconsistent",
            )
