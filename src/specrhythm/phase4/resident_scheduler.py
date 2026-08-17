"""Pinned-vLLM scheduler gate for incremental resident setup."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

from specrhythm.phase4.request_identity import FrozenPromptIdentityMap
from specrhythm.phase4.resident_setup import (
    ADMISSION_EVENT_SCHEMA,
    load_setup_ready,
    resident_admission_decision,
)
from specrhythm.phase4.serial import Proposal, token_prefix_hash
from specrhythm.phase4.stock_vllm import load_smoke_requests
from specrhythm.phase4.transport import CheckpointJsonl

try:
    from vllm.v1.core.sched.scheduler import Scheduler
except ImportError as error:  # pragma: no cover - GPU-only import path
    raise RuntimeError("ResidentSetupScheduler requires pinned vLLM v0.25.1") from error


class ResidentSetupScheduler(Scheduler):
    """Freeze bootstrapped requests until one auditable global boundary."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if os.environ.get("SR_PHASE4_RESIDENT_SETUP") != "1":
            raise RuntimeError("resident scheduler is fail-closed unless explicitly enabled")
        consumer = os.environ.get("SR_PHASE4_RESIDENT_CONSUMER")
        if consumer not in {"target-only", "serial"}:
            raise RuntimeError("resident scheduler consumer is missing or invalid")
        self._resident_consumer = consumer
        self._resident_ready_path = _required_path("SR_PHASE4_RESIDENT_SETUP_READY")
        self._resident_manifest_path = _required_path(
            "SR_PHASE4_DECODE_READY_MANIFEST"
        )
        self._resident_events = CheckpointJsonl(
            _required_path("SR_PHASE4_RESIDENT_ADMISSION_EVENTS")
        )
        request_count = int(os.environ.get("SR_PHASE4_REQUEST_COUNT", "5"))
        definitions = load_smoke_requests(
            _required_path("SR_PHASE4_WORKLOAD"),
            request_count,
            require_task_mixture=request_count == 5,
        )
        self._resident_expected_ids = tuple(row.request_id for row in definitions)
        self._resident_identity = FrozenPromptIdentityMap.from_definitions(definitions)
        self._resident_ready: Optional[dict[str, Any]] = None
        self._resident_initial_proposals: dict[str, Proposal] = {}
        self._resident_installed: set[str] = set()
        self._resident_decisions: dict[str, tuple[bool, str, int, str]] = {}
        self._resident_cycle_id = 0

    def schedule(self, *args: Any, **kwargs: Any) -> Any:
        self._bind_requests()
        self._refresh_readiness()
        if self._resident_ready is not None and self._resident_consumer == "serial":
            self._install_initial_proposals()
        self._resident_decisions = {}
        decision_ns = time.monotonic_ns()
        for internal_id, request in self.requests.items():
            if request.is_finished():
                continue
            stable_id = self._resident_identity.stable_id(str(internal_id))
            proposal_installed = stable_id in self._resident_installed or bool(
                request.spec_token_ids
            )
            admissible, reason = resident_admission_decision(
                num_output_tokens=int(request.num_output_tokens),
                global_decode_ready=self._resident_ready is not None,
                consumer=self._resident_consumer,
                has_initial_proposal=proposal_installed,
            )
            self._resident_decisions[str(internal_id)] = (
                admissible,
                reason,
                decision_ns,
                stable_id,
            )
        output = super().schedule(*args, **kwargs)
        scheduled = output.num_scheduled_tokens
        for internal_id, (admissible, reason, timestamp_ns, stable_id) in sorted(
            self._resident_decisions.items()
        ):
            request = self.requests.get(internal_id)
            scheduled_count = int(scheduled.get(internal_id, 0))
            self._resident_events.append(
                {
                    "schema_version": ADMISSION_EVENT_SCHEMA,
                    "cycle_id": self._resident_cycle_id,
                    "timestamp_ns": timestamp_ns,
                    "consumer": self._resident_consumer,
                    "request_id": stable_id,
                    "internal_request_id": internal_id,
                    "num_output_tokens": (
                        int(request.num_output_tokens) if request is not None else None
                    ),
                    "global_decode_ready": self._resident_ready is not None,
                    "measurement_start_ns": (
                        self._resident_ready.get("measurement_start_ns")
                        if self._resident_ready is not None
                        else None
                    ),
                    "initial_proposal_installed": stable_id
                    in self._resident_installed,
                    "admissible": admissible,
                    "reason": reason,
                    "scheduled": scheduled_count > 0,
                    "scheduled_token_count": scheduled_count,
                    "explicit_request_predicate": True,
                    "current_step_arithmetic": False,
                }
            )
        self._resident_cycle_id += 1
        return output

    def _request_admissible_for_schedule(self, request: Any) -> bool:
        internal_id = str(request.request_id)
        decision = self._resident_decisions.get(internal_id)
        if decision is None:
            stable_id = self._resident_identity.stable_id(internal_id)
            decision_value = resident_admission_decision(
                num_output_tokens=int(request.num_output_tokens),
                global_decode_ready=self._resident_ready is not None,
                consumer=self._resident_consumer,
                has_initial_proposal=(
                    stable_id in self._resident_installed or bool(request.spec_token_ids)
                ),
            )
            decision = (*decision_value, time.monotonic_ns(), stable_id)
            self._resident_decisions[internal_id] = decision
        if int(request.num_output_tokens) > 1 and self._resident_ready is None:
            raise RuntimeError("Target advanced beyond bootstrap before global setup-ready")
        return decision[0]

    def _refresh_readiness(self) -> None:
        if self._resident_ready is not None or not self._resident_ready_path.is_file():
            return
        ready = load_setup_ready(
            self._resident_ready_path,
            manifest_path=self._resident_manifest_path,
            consumer=self._resident_consumer,
            expected_request_ids=self._resident_expected_ids,
        )
        self._resident_ready = ready
        proposals = ready.get("initial_proposals", ())
        self._resident_initial_proposals = {
            str(row["request_id"]): Proposal.from_dict(row) for row in proposals
        }

    def _install_initial_proposals(self) -> None:
        mappings = self._resident_ready.get("stable_to_internal_request_id", {})
        if not isinstance(mappings, dict):
            raise RuntimeError("resident setup-ready identity mapping is invalid")
        for stable_id in self._resident_expected_ids:
            internal_id = str(mappings.get(stable_id, ""))
            if self._resident_identity.internal_id(stable_id) != internal_id:
                raise RuntimeError("resident scheduler identity differs from setup-ready")
            request = self.requests.get(internal_id)
            if request is None or request.is_finished():
                raise RuntimeError("resident initial proposal has no live Target request")
            proposal = self._resident_initial_proposals.get(stable_id)
            if proposal is None:
                raise RuntimeError("resident Serial initial proposal is missing")
            prefix = tuple(int(item) for item in request.all_token_ids)
            if (
                proposal.parent_prefix_len != len(prefix)
                or proposal.parent_prefix_hash != token_prefix_hash(prefix)
            ):
                raise RuntimeError("resident Serial initial proposal parent is stale")
            proposal_tokens = list(proposal.proposal_token_ids)
            if not proposal_tokens:
                raise RuntimeError("resident Serial initial proposal is empty")
            if request.spec_token_ids and request.spec_token_ids != proposal_tokens:
                raise RuntimeError("worker/scheduler initial proposal tokens differ")
            request.spec_token_ids = proposal_tokens
            self._resident_installed.add(stable_id)

    def _bind_requests(self) -> None:
        for internal_id, request in self.requests.items():
            if request.is_finished():
                continue
            if str(request.request_id) != str(internal_id):
                raise RuntimeError("vLLM resident request table identity differs")
            self._resident_identity.bind(
                str(internal_id), tuple(int(item) for item in request.all_token_ids)
            )


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required by the resident scheduler")
    return Path(value).resolve()
