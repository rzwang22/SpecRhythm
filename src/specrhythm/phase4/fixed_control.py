"""Single-request fixed-proposal controls for a failed C/D comparison."""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from specrhythm.phase4.serial import PROTOCOL_VERSION
from specrhythm.phase4.stock_vllm import load_smoke_requests
from specrhythm.phase4.transport import (
    UnixDraftClient,
    receive_message,
    send_message,
)

FIXED_PROPOSAL = (53143, 2213, 369, 264)


def fixed_proposal_rows(
    num_tokens_no_spec: Any,
    token_ids_cpu: Any,
    *,
    workload_path: Path,
    budget: int,
) -> list[list[int]]:
    if budget not in (1, 2, 4):
        raise ValueError("fixed-proposal control budget must be K=1, K=2, or K=4")
    definitions = load_smoke_requests(
        workload_path, expected_count=1, require_task_mixture=False
    )
    definition = definitions[0]
    rows = []
    for index in range(len(num_tokens_no_spec)):
        count = int(num_tokens_no_spec[index])
        tokens = tuple(int(item) for item in token_ids_cpu[index, :count].tolist())
        if tokens[: len(definition.prompt_token_ids)] != definition.prompt_token_ids:
            raise RuntimeError("fixed control input does not match the frozen single request")
        generated = max(0, count - len(definition.prompt_token_ids))
        remaining = max(0, definition.maximum_new_tokens - generated)
        rows.append(list(FIXED_PROPOSAL[: min(budget, max(remaining - 1, 0))]))
    return rows


class LocalStaticProposer:
    """Target-worker-local fixed proposer; reads no Target logits or labels."""

    def __init__(self, vllm_config: Any) -> None:
        del vllm_config
        from vllm.distributed.parallel_state import get_tp_group

        self.tp_group = get_tp_group()
        self.workload_path = _required_path("SR_PHASE4_WORKLOAD")
        self.budget = _required_budget()

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: Any,
        token_ids_cpu: Any,
        *,
        request_ids: Optional[Sequence[str]] = None,
        slot_mappings: Any = None,
        target_materialized_token_counts: Optional[Sequence[int]] = None,
    ) -> list[list[int]]:
        del (
            sampled_token_ids,
            request_ids,
            slot_mappings,
            target_materialized_token_counts,
        )
        result = None
        if int(self.tp_group.rank_in_group) == 0:
            result = fixed_proposal_rows(
                num_tokens_no_spec,
                token_ids_cpu,
                workload_path=self.workload_path,
                budget=self.budget,
            )
        value = self.tp_group.broadcast_object(result, src=0)
        if not isinstance(value, list):
            raise RuntimeError("local fixed proposal broadcast failed")
        return value

    @property
    def supports_mm_inputs(self) -> bool:
        return False


class RemoteFixedProposer(LocalStaticProposer):
    """Same fixed proposal transported over AF_UNIX for adapter isolation control."""

    def __init__(self, vllm_config: Any) -> None:
        super().__init__(vllm_config)
        self.client = UnixDraftClient(_required_path("SR_PHASE4_FIXED_SOCKET"))

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: Any,
        token_ids_cpu: Any,
        *,
        request_ids: Optional[Sequence[str]] = None,
        slot_mappings: Any = None,
        target_materialized_token_counts: Optional[Sequence[int]] = None,
    ) -> list[list[int]]:
        del (
            sampled_token_ids,
            request_ids,
            slot_mappings,
            target_materialized_token_counts,
        )
        result = None
        if int(self.tp_group.rank_in_group) == 0:
            local_rows = fixed_proposal_rows(
                num_tokens_no_spec,
                token_ids_cpu,
                workload_path=self.workload_path,
                budget=self.budget,
            )
            response = self.client.call("fixed_proposal", {"rows": local_rows})
            result = response.get("rows")
            if result != local_rows:
                raise RuntimeError("remote fixed service changed the fixed proposal")
        value = self.tp_group.broadcast_object(result, src=0)
        if not isinstance(value, list):
            raise RuntimeError("remote fixed proposal broadcast failed")
        return value


def run_fixed_proposal_service(socket_path: Path) -> None:
    if socket_path.exists():
        raise FileExistsError(f"refusing to replace fixed control socket {socket_path}")
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(socket_path))
        server.listen(8)
        try:
            running = True
            while running:
                connection, _ = server.accept()
                with connection:
                    request, _ = receive_message(connection)
                    operation = request.get("operation")
                    payload = request.get("payload")
                    payload = payload if isinstance(payload, Mapping) else {}
                    if operation == "fixed_proposal":
                        rows = payload.get("rows")
                        if not isinstance(rows, list):
                            response = {"ok": False, "error": "rows must be a list"}
                        else:
                            response = {"ok": True, "result": {"rows": rows}}
                    elif operation == "shutdown":
                        response = {"ok": True, "result": {"shutdown": True}}
                        running = False
                    else:
                        response = {"ok": False, "error": "unsupported operation"}
                    send_message(
                        connection,
                        {"protocol_version": PROTOCOL_VERSION, **response},
                    )
        finally:
            socket_path.unlink(missing_ok=True)


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required for the fixed-proposal control")
    return Path(value).resolve()


def _required_budget() -> int:
    try:
        value = int(os.environ.get("SR_PHASE4_FIXED_K", ""))
    except ValueError as error:
        raise RuntimeError("SR_PHASE4_FIXED_K must be an integer") from error
    if value not in (1, 2, 4):
        raise RuntimeError("SR_PHASE4_FIXED_K must be 1, 2, or 4")
    return value
