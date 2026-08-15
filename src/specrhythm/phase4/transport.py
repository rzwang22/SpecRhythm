"""Local-only JSON transport and crash-detecting event logs for Phase 4A.1."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import struct
import time
from pathlib import Path
from typing import Any, Mapping, Optional

from specrhythm.phase4.serial import PROTOCOL_VERSION

MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def payload_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _receive_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("local transport closed during a framed message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(sock: socket.socket, value: Mapping[str, Any]) -> int:
    payload = canonical_json_bytes(value)
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("local transport payload exceeds the Phase-4 safety limit")
    sock.sendall(struct.pack("!Q", len(payload)) + payload)
    return len(payload)


def receive_message(sock: socket.socket) -> tuple[dict[str, Any], int]:
    size = struct.unpack("!Q", _receive_exact(sock, 8))[0]
    if size > MAX_MESSAGE_BYTES:
        raise ValueError("local transport announced an oversized payload")
    payload = _receive_exact(sock, size)
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("local transport message root must be an object")
    return value, len(payload)


class CheckpointJsonl:
    """Append fsync'd checksummed records and reject partial/corrupt logs."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, value: Mapping[str, Any]) -> None:
        payload = dict(value)
        if "record_sha256" in payload:
            raise ValueError("record_sha256 is reserved for checkpoint framing")
        payload["record_sha256"] = payload_sha256(payload)
        line = canonical_json_bytes(payload) + b"\n"
        with self.path.open("ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        raw = self.path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise ValueError(f"partial JSONL record detected in {self.path.name}")
        rows = []
        for line_number, line in enumerate(raw.splitlines(), 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL record {line_number} in {self.path.name}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record {line_number} is not an object")
            expected = value.pop("record_sha256", None)
            if expected != payload_sha256(value):
                raise ValueError(f"JSONL record {line_number} checksum mismatch")
            value["record_sha256"] = expected
            rows.append(value)
        return rows


class UnixDraftClient:
    """One-request-per-connection local IPC client.

    AF_UNIX prevents accidental exposure on an external network interface. The
    transport is JSON, not pickle, and therefore does not require vLLM's
    insecure serialization switch.
    """

    def __init__(
        self,
        socket_path: Path,
        *,
        timeout_seconds: float = 600.0,
        transport_log: Optional[CheckpointJsonl] = None,
    ) -> None:
        self.socket_path = socket_path
        self.timeout_seconds = timeout_seconds
        self.transport_log = transport_log

    def call(self, operation: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not operation:
            raise ValueError("transport operation must not be empty")
        message = {
            "protocol_version": PROTOCOL_VERSION,
            "operation": operation,
            "payload": dict(payload),
        }
        started = time.monotonic_ns()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(self.timeout_seconds)
            sock.connect(str(self.socket_path))
            sent_bytes = send_message(sock, message)
            response, received_bytes = receive_message(sock)
        finished = time.monotonic_ns()
        if response.get("protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError("Draft service returned an incompatible protocol")
        if response.get("ok") is not True:
            raise RuntimeError(str(response.get("error", "Draft service request failed")))
        event = {
            "schema_version": "specrhythm.phase4-transport-event.v1",
            "transport": "unix-domain-socket",
            "serialization": "length-prefixed-canonical-json",
            "direction": "target-rank0-to-draft-and-response",
            "request_direction": "target-rank0-to-draft-control-and-committed-prefix",
            "response_direction": "draft-to-target-rank0-candidate-batch",
            "operation": operation,
            "send_start_ns": started,
            "receive_end_ns": finished,
            "request_payload_bytes": sent_bytes,
            "response_payload_bytes": received_bytes,
            "protocol_version": PROTOCOL_VERSION,
            "loopback_local_only": True,
            "host_staging": True,
            "gpu_kernel_time": False,
        }
        if self.transport_log is not None:
            self.transport_log.append(event)
        result = response.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("Draft service response result is not an object")
        result.setdefault("transport_start_ns", started)
        result.setdefault("transport_end_ns", finished)
        result.setdefault("transport_payload_bytes", sent_bytes + received_bytes)
        return result

    def shutdown(self) -> dict[str, Any]:
        return self.call("shutdown", {})


def validate_transport_event(value: Mapping[str, Any]) -> list[str]:
    errors = []
    required = {
        "transport": "unix-domain-socket",
        "serialization": "length-prefixed-canonical-json",
        "loopback_local_only": True,
        "gpu_kernel_time": False,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            errors.append(f"transport event has invalid {key}")
    if value.get("protocol_version") != PROTOCOL_VERSION:
        errors.append("transport event protocol is incompatible")
    start = value.get("send_start_ns")
    end = value.get("receive_end_ns")
    if not isinstance(start, int) or not isinstance(end, int) or end < start:
        errors.append("transport event timestamps are invalid")
    for key in ("request_payload_bytes", "response_payload_bytes"):
        if not isinstance(value.get(key), int) or value[key] <= 0:
            errors.append(f"transport event {key} must be positive")
    return errors
