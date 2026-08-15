"""Phase-3A deterministic draft, target, and serial trace collection."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from specrhythm.phase3.config import Phase3Config, resolve_runtime_path
from specrhythm.phase3.engine import CausalLMBackend, create_backend
from specrhythm.phase3.trace import (
    CandidateNodeRecord,
    CycleAccounting,
    RealTraceRecord,
    RequestCycle,
    TargetOutcomeRecord,
    TraceStore,
    sha256_file,
    stable_node_id,
    summarize_records,
)


@dataclass(frozen=True)
class PromptRequest:
    request_id: str
    prompt: str
    slo_class: str
    max_new_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("prompt request_id must not be empty")
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("prompt must be a non-empty string")
        if not self.slo_class:
            raise ValueError("prompt SLO_class must not be empty")
        if self.max_new_tokens is not None and (
            not isinstance(self.max_new_tokens, int)
            or isinstance(self.max_new_tokens, bool)
            or self.max_new_tokens < 1
        ):
            raise ValueError("max_new_tokens must be a positive integer")


def load_prompt_requests(path: Path) -> list[PromptRequest]:
    requests = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"prompt JSONL line {line_number}: {error}") from error
            request = PromptRequest(
                request_id=str(value.get("request_id", "")),
                prompt=value.get("prompt", ""),
                slo_class=str(value.get("SLO_class", value.get("slo_class", "unspecified"))),
                max_new_tokens=value.get("max_new_tokens"),
            )
            if request.request_id in seen:
                raise ValueError(f"duplicate prompt request_id: {request.request_id}")
            seen.add(request.request_id)
            requests.append(request)
    if not requests:
        raise ValueError("prompt input contains no requests")
    return requests


@dataclass(frozen=True)
class _CandidateState:
    record: CandidateNodeRecord
    path_tokens: tuple[int, ...]


def generate_candidate_forest(
    backend: CausalLMBackend,
    *,
    request_id: str,
    cycle_id: int,
    context: list[int],
    search_pool_size: int,
    width: int,
) -> tuple[CandidateNodeRecord, ...]:
    """Materialize a deterministic, prefix-closed BFS forest from real draft logits."""

    frontier: list[tuple[Optional[str], tuple[int, ...], float, int]] = [
        (None, (), 1.0, 0)
    ]
    candidates: list[_CandidateState] = []
    while frontier and len(candidates) < search_pool_size:
        parent_id, path_tokens, parent_probability, parent_depth = frontier.pop(0)
        distribution = backend.next_token(context + list(path_tokens), width)
        for sibling_rank, token in enumerate(distribution.ranked_tokens[:width]):
            if len(candidates) >= search_pool_size:
                break
            node_id = stable_node_id(
                request_id, cycle_id, parent_id, token.token_id, sibling_rank
            )
            record = CandidateNodeRecord(
                stable_node_id=node_id,
                parent_id=parent_id,
                depth=parent_depth + 1,
                token_id=token.token_id,
                local_probability=token.probability,
                path_probability=parent_probability * token.probability,
                draft_logit=token.logit,
                entropy=distribution.entropy,
                top1_top2_margin=distribution.top1_top2_margin,
                sibling_rank=sibling_rank,
                prefix_closed=True,
                selected_for_verification=False,
            )
            state = _CandidateState(record, path_tokens + (token.token_id,))
            candidates.append(state)
            if backend.eos_token_id is None or token.token_id != backend.eos_token_id:
                frontier.append(
                    (node_id, state.path_tokens, record.path_probability, record.depth)
                )
    return tuple(state.record for state in candidates)


def select_candidates(
    candidates: Iterable[CandidateNodeRecord], budget: int
) -> tuple[CandidateNodeRecord, ...]:
    """Select highest-path-probability eligible nodes without target-side labels."""

    values = list(candidates)
    selected = set()
    while len(selected) < min(budget, len(values)):
        eligible = [
            node
            for node in values
            if node.stable_node_id not in selected
            and (node.parent_id is None or node.parent_id in selected)
        ]
        if not eligible:
            break
        node = max(
            eligible,
            key=lambda item: (item.path_probability, -item.depth, item.stable_node_id),
        )
        selected.add(node.stable_node_id)
    return tuple(
        replace(node, selected_for_verification=node.stable_node_id in selected)
        for node in values
    )


def _target_trajectory(
    backend: CausalLMBackend,
    context: list[int],
    steps: int,
) -> tuple[list[int], list[float]]:
    tokens = []
    log_probabilities = []
    for _ in range(steps):
        distribution = backend.next_token(context + tokens, 2)
        token = distribution.top1
        tokens.append(token.token_id)
        log_probabilities.append(math.log(max(token.probability, 1e-300)))
        if backend.eos_token_id is not None and token.token_id == backend.eos_token_id:
            break
    return tokens, log_probabilities


def verify_candidate_forest(
    backend: CausalLMBackend,
    *,
    context: list[int],
    candidates: tuple[CandidateNodeRecord, ...],
    remaining_tokens: int,
) -> tuple[tuple[TargetOutcomeRecord, ...], tuple[int, ...], Optional[int]]:
    """Greedy target verification with one target root/bonus token per cycle."""

    by_parent: dict[Optional[str], list[CandidateNodeRecord]] = {}
    for node in candidates:
        by_parent.setdefault(node.parent_id, []).append(node)
    maximum_depth = max((node.depth for node in candidates), default=0)
    target_tokens, target_logs = _target_trajectory(
        backend, context, min(remaining_tokens, maximum_depth + 1)
    )
    target_nodes = []
    parent_id: Optional[str] = None
    for target_token in target_tokens[:maximum_depth]:
        matching = next(
            (
                node
                for node in by_parent.get(parent_id, ())
                if node.token_id == target_token
            ),
            None,
        )
        if matching is None:
            break
        target_nodes.append(matching.stable_node_id)
        parent_id = matching.stable_node_id
    selected = {node.stable_node_id for node in candidates if node.selected_for_verification}
    accepted_ids = []
    for node_id in target_nodes:
        if node_id not in selected or len(accepted_ids) >= remaining_tokens:
            break
        accepted_ids.append(node_id)
    accepted_set = set(accepted_ids)
    accepted_count = len(accepted_ids)
    outcomes = tuple(
        TargetOutcomeRecord(
            stable_node_id=node.stable_node_id,
            target_token_id=target_tokens[node.depth - 1],
            target_log_probability=target_logs[node.depth - 1],
            on_target_path=node.stable_node_id in set(target_nodes),
            accepted=node.stable_node_id in accepted_set,
            committed=node.stable_node_id in accepted_set,
            accepted_prefix_length=accepted_count,
        )
        for node in candidates
        if node.depth <= len(target_tokens)
    )
    accepted_tokens = tuple(
        next(node.token_id for node in candidates if node.stable_node_id == node_id)
        for node_id in accepted_ids
    )
    root_target: Optional[int] = None
    if accepted_count < remaining_tokens and accepted_count < len(target_tokens):
        root_target = target_tokens[accepted_count]
    committed = accepted_tokens + ((root_target,) if root_target is not None else ())
    return outcomes, committed, root_target


def _request_cycle(
    request: PromptRequest,
    *,
    cycle_id: int,
    prompt_length: int,
    generated: tuple[int, ...],
    config: Phase3Config,
    mode: str,
    draft_model: Optional[str] = None,
    target_model: Optional[str] = None,
) -> RequestCycle:
    return RequestCycle(
        request_id=request.request_id,
        cycle_id=cycle_id,
        prompt_length=prompt_length,
        context_length=prompt_length + len(generated),
        generated_tokens=generated,
        slo_class=request.slo_class,
        draft_model=draft_model or config.draft.model_path,
        target_model=target_model or config.target.model_path,
        random_seed=config.random_seed,
        sampling_configuration=config.sampling_configuration,
        mode=mode,
    )


def _serial_record(
    request: PromptRequest,
    cycle_id: int,
    prompt_tokens: list[int],
    generated: tuple[int, ...],
    draft: CausalLMBackend,
    target: CausalLMBackend,
    config: Phase3Config,
    maximum: int,
) -> RealTraceRecord:
    context = prompt_tokens + list(generated)
    candidates = select_candidates(
        generate_candidate_forest(
            draft,
            request_id=request.request_id,
            cycle_id=cycle_id,
            context=context,
            search_pool_size=config.search_pool_size,
            width=config.candidate_width,
        ),
        config.candidate_budget,
    )
    outcomes, committed, root_target = verify_candidate_forest(
        target,
        context=context,
        candidates=candidates,
        remaining_tokens=maximum - len(generated),
    )
    finished = len(generated) + len(committed) >= maximum or (
        target.eos_token_id is not None and target.eos_token_id in committed
    )
    accepted = sum(outcome.accepted for outcome in outcomes)
    return RealTraceRecord(
        request=_request_cycle(
            request,
            cycle_id=cycle_id,
            prompt_length=len(prompt_tokens),
            generated=generated,
            config=config,
            mode="serial",
            draft_model=draft.model_id,
            target_model=target.model_id,
        ),
        candidate_nodes=candidates,
        target_outcomes=outcomes,
        accounting=CycleAccounting(
            request_roots=1,
            search_pool_nodes=len(candidates),
            verified_candidate_nodes=sum(
                node.selected_for_verification for node in candidates
            ),
            accepted_candidate_nodes=accepted,
            committed_candidate_tokens=accepted,
            committed_target_tokens=int(root_target is not None),
        ),
        root_target_token_id=root_target,
        committed_token_ids=committed,
        request_finished=finished,
    )


def _target_record(
    request: PromptRequest,
    cycle_id: int,
    prompt_tokens: list[int],
    generated: tuple[int, ...],
    target: CausalLMBackend,
    config: Phase3Config,
    maximum: int,
) -> RealTraceRecord:
    distribution = target.next_token(prompt_tokens + list(generated), 2)
    token = distribution.top1.token_id
    finished = len(generated) + 1 >= maximum or token == target.eos_token_id
    return RealTraceRecord(
        request=_request_cycle(
            request,
            cycle_id=cycle_id,
            prompt_length=len(prompt_tokens),
            generated=generated,
            config=config,
            mode="target-only",
            target_model=target.model_id,
        ),
        candidate_nodes=(),
        target_outcomes=(),
        accounting=CycleAccounting(1, 0, 0, 0, 0, 1),
        root_target_token_id=token,
        committed_token_ids=(token,),
        request_finished=finished,
    )


def _draft_record(
    request: PromptRequest,
    prompt_tokens: list[int],
    draft: CausalLMBackend,
    config: Phase3Config,
) -> RealTraceRecord:
    candidates = select_candidates(
        generate_candidate_forest(
            draft,
            request_id=request.request_id,
            cycle_id=0,
            context=prompt_tokens,
            search_pool_size=config.search_pool_size,
            width=config.candidate_width,
        ),
        config.candidate_budget,
    )
    return RealTraceRecord(
        request=_request_cycle(
            request,
            cycle_id=0,
            prompt_length=len(prompt_tokens),
            generated=(),
            config=config,
            mode="draft-only",
            draft_model=draft.model_id,
        ),
        candidate_nodes=candidates,
        target_outcomes=(),
        accounting=CycleAccounting(
            1,
            len(candidates),
            sum(node.selected_for_verification for node in candidates),
            0,
            0,
            0,
        ),
        root_target_token_id=None,
        committed_token_ids=(),
        request_finished=True,
    )


def _model_identity(backend: CausalLMBackend) -> tuple[int, Optional[int], str]:
    return backend.vocab_size, backend.eos_token_id, backend.tokenizer_fingerprint


def run_phase3(
    requests: list[PromptRequest],
    config: Phase3Config,
    *,
    mode: str,
    output_dir: Path,
    resume: bool,
    draft_backend: Optional[CausalLMBackend] = None,
    target_backend: Optional[CausalLMBackend] = None,
    cycle_limit: Optional[int] = None,
) -> dict[str, Any]:
    if mode not in {"draft-only", "target-only", "serial"}:
        raise ValueError("Phase-3 mode must be draft-only, target-only, or serial")
    if cycle_limit is not None and cycle_limit < 1:
        raise ValueError("cycle_limit must be positive")
    store = TraceStore(output_dir)
    if not resume and store.records():
        raise FileExistsError("trace directory is non-empty; pass --resume to continue")
    owns_draft = draft_backend is None and mode in {"draft-only", "serial"}
    owns_target = target_backend is None and mode in {"target-only", "serial"}
    draft = draft_backend
    target = target_backend
    if owns_draft:
        draft = create_backend(config.backend, config.draft, config.random_seed)
    if owns_target:
        target = create_backend(config.backend, config.target, config.random_seed)
    if mode == "serial" and draft is not None and target is not None:
        if _model_identity(draft) != _model_identity(target):
            raise ValueError(
                "draft and target tokenizers must share vocab size and EOS token ID"
            )
    runtime_models = {}
    for role, backend in (("draft", draft), ("target", target)):
        if backend is not None:
            runtime_models[role] = {
                "model_id": backend.model_id,
                "vocab_size": backend.vocab_size,
                "eos_token_id": backend.eos_token_id,
                "tokenizer_fingerprint": backend.tokenizer_fingerprint,
            }
    written = 0
    try:
        for request in requests:
            if cycle_limit is not None and written >= cycle_limit:
                break
            tokenizer = target if mode == "target-only" else draft
            if tokenizer is None:
                raise AssertionError("Phase-3 runner backend was not initialized")
            prompt_tokens = tokenizer.encode(request.prompt)
            maximum = request.max_new_tokens or config.max_new_tokens
            if len(prompt_tokens) + maximum > config.context_length:
                raise ValueError(
                    f"request {request.request_id} exceeds configured context_length"
                )
            cycle_id, generated, finished = store.resume_state(request.request_id)
            if finished:
                continue
            if mode == "draft-only":
                if cycle_id:
                    continue
                if draft is None:
                    raise AssertionError("draft backend is unavailable")
                written += int(store.write(_draft_record(request, prompt_tokens, draft, config)))
                continue
            while not finished:
                if mode == "target-only":
                    if target is None:
                        raise AssertionError("target backend is unavailable")
                    record = _target_record(
                        request,
                        cycle_id,
                        prompt_tokens,
                        generated,
                        target,
                        config,
                        maximum,
                    )
                else:
                    if draft is None or target is None:
                        raise AssertionError("serial backends are unavailable")
                    record = _serial_record(
                        request,
                        cycle_id,
                        prompt_tokens,
                        generated,
                        draft,
                        target,
                        config,
                        maximum,
                    )
                if not record.committed_token_ids:
                    raise AssertionError("generation cycle made no progress")
                written += int(store.write(record))
                generated += record.committed_token_ids
                finished = record.request_finished
                cycle_id += 1
                if cycle_limit is not None and written >= cycle_limit:
                    break
    finally:
        if owns_draft and draft is not None:
            draft.close()
        if owns_target and target is not None:
            target.close()
    report = summarize_records(store.records())
    report.update(
        {
            "mode": mode,
            "backend": config.backend,
            "new_records": written,
            "resumed": resume,
            "cycle_limit": cycle_limit,
            "gpu_measurement": False,
            "runner_semantics": "correctness-first-serial-no-overlap",
            "configured_batch_size": config.batch_size,
            "runner_batching": "sequential requests; no serving-engine continuous batching",
            "runtime_models": runtime_models,
        }
    )
    return report


def build_run_manifest(
    *,
    config_path: Path,
    input_path: Path,
    output_dir: Path,
    config: Phase3Config,
    mode: str,
    command: str,
    git_commit: Optional[str],
    environment_metadata_path: Optional[Path] = None,
    runtime_models: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    from specrhythm import __version__

    store = TraceStore(output_dir)
    records = store.records()
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            (json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
    model_config_checksums = {}
    for role, model in (("draft", config.draft), ("target", config.target)):
        resolved = Path(
            resolve_runtime_path(model.model_path, dry_run=config.backend == "dry-run")
        )
        model_config = resolved / "config.json" if resolved.is_dir() else None
        if model_config is not None and model_config.is_file():
            model_config_checksums[role] = {
                "file": model_config.name,
                "sha256": sha256_file(model_config),
            }
    manifest = {
        "schema_version": "specrhythm.phase3-run-manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "python_version": platform.python_version(),
        "specrhythm_version": __version__,
        "mode": mode,
        "backend": config.backend,
        "gpu_experiment_executed": config.backend == "transformers",
        "runner_semantics": "correctness-first-serial-no-overlap",
        "config_file": config_path.name,
        "config_sha256": sha256_file(config_path),
        "input_file": input_path.name,
        "input_sha256": sha256_file(input_path),
        "trace_sha256": digest.hexdigest(),
        "record_count": len(records),
        "request_count": len({record.request.request_id for record in records}),
        "random_seed": config.random_seed,
        "effective_config": asdict(config),
        "draft_model": asdict(config.draft),
        "target_model": asdict(config.target),
        "runtime_models": runtime_models or {},
        "model_config_checksums": model_config_checksums,
        "command": command,
    }
    if environment_metadata_path is not None:
        manifest["environment_metadata_file"] = environment_metadata_path.name
        manifest["environment_metadata_sha256"] = sha256_file(
            environment_metadata_path
        )
    return manifest
