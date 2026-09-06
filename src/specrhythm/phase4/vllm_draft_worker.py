"""Pinned MRV1 adaptation. Importing this module never imports torch/vLLM.

Only the ordinary dense Qwen3 TP1 worker is admitted. Its pending forward is
completed without stock sampling; external committed context is projected
through the pinned streaming-request rebase path. No vLLM source is modified.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from specrhythm.phase4.config import VLLM_COMMIT, VLLM_VERSION, Phase4Config
from specrhythm.phase4.draft_batch import DraftMaterialization, mapped_rows, unique_ids
from specrhythm.phase4.draft_metrics import DraftMetrics
from specrhythm.phase4.manifest import model_revision_manifest, sha256_file
from specrhythm.phase4.stock_vllm import active_cuda_device_identity
from specrhythm.phase4.vllm_installation import locate_installed_vllm_file

API_PATH = Path(__file__).with_name("vllm_draft_api.json")
EXECUTE_FIELDS = (
    "scheduler_output",
    "logits",
    "spec_decode_metadata",
    "spec_decode_common_attn_metadata",
    "hidden_states",
    "sample_hidden_states",
    "aux_hidden_states",
    "ec_connector_output",
    "cudagraph_stats",
    "slot_mappings",
)


def audit_source_files(root: Path, *, patched: bool = True) -> dict[str, Any]:
    """Exact file guards also work on a CPU-only source export."""
    api = json.loads(API_PATH.read_text())
    rows = []
    for entry in api["files"]:
        path = root / entry["path"]
        key = "required_installed_sha256" if patched else "base_sha256"
        if not path.is_file() or sha256_file(path) != entry[key]:
            raise RuntimeError(f"pinned Draft vLLM API source mismatch: {entry['path']}")
        rows.append({"path": entry["path"], "sha256": entry[key]})
    return {
        "vllm_version": api["vllm_version"],
        "vllm_commit": api["vllm_commit"],
        "files": rows,
        "api_inventory_sha256": sha256_file(API_PATH),
    }


def audit_installed_api() -> dict[str, Any]:
    version = importlib.metadata.version("vllm")
    if version.split("+", 1)[0] != VLLM_VERSION:
        raise RuntimeError(f"Draft requires vLLM {VLLM_VERSION}, found {version}")
    source = os.environ.get("SR_VLLM_SOURCE")
    if not source or not Path(source).is_dir():
        raise RuntimeError("SR_VLLM_SOURCE must name the pinned vLLM Git source")
    commit = subprocess.run(
        ["git", "-C", source, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if commit != VLLM_COMMIT:
        raise RuntimeError("Draft vLLM source commit differs from the audited pin")
    root = locate_installed_vllm_file(Path("vllm/v1/worker/gpu_worker.py")).parents[3]
    return {**audit_source_files(root), "installed_version": version, "root": str(root)}


def validate_placement(identity: Mapping[str, Any], config: Phase4Config) -> None:
    if config.draft.physical_gpu_ids != (0,) or config.draft.tensor_parallel_size != 1:
        raise RuntimeError("Plan A requires Draft TP1 on physical GPU0")
    if (
        identity.get("physical_gpu_id") != 0
        or identity.get("logical_cuda_index") != 0
        or identity.get("cuda_visible_devices") != "0"
        or re.fullmatch(
            r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
            str(identity.get("gpu_uuid", "")),
        )
        is None
    ):
        raise RuntimeError("Draft real UUID/device-binding validation failed")


def validate_runner(runner: Any) -> None:
    """Mandatory guards for every completion obligation omitted by this adapter."""
    disabled = (
        "use_async_scheduling",
        "is_pooling_model",
        "use_aux_hidden_state_outputs",
        "routed_experts_initialized",
        "uses_mrope",
    )
    for field in disabled:
        if not hasattr(runner, field) or getattr(runner, field):
            raise RuntimeError(f"unsupported Draft runner feature: {field}")
    for field in ("speculative_config", "lora_config"):
        if not hasattr(runner, field) or getattr(runner, field) is not None:
            raise RuntimeError(f"unsupported Draft runner feature: {field}")
    if getattr(runner, "drafter", None) is not None:
        raise RuntimeError("Draft worker must not contain a second/native proposer model")


def complete_draft_forward(runner: Any, output: Any, expected: Sequence[str]) -> dict:
    """Consume precisely the audited execute state, without appending a sample."""
    validate_runner(runner)
    state = runner.execute_model_state
    if output is not None or state is None or tuple(state._fields) != EXECUTE_FIELDS:
        raise RuntimeError("Draft forward completion state/signature mismatch")
    if (
        any(
            getattr(state, f) is not None
            for f in (
                "spec_decode_metadata",
                "spec_decode_common_attn_metadata",
                "aux_hidden_states",
                "ec_connector_output",
                "cudagraph_stats",
            )
        )
        or runner.kv_connector_output is not None
    ):
        raise RuntimeError("Draft forward has unsupported deferred completion work")
    ids = tuple(runner.input_batch.req_ids)
    if state.logits.ndim != 2 or state.logits.shape[0] != len(ids):
        raise RuntimeError("Draft logits do not contain one row per actual request")
    # A single owned batch copy; views remain valid after the runner reuses buffers.
    rows = mapped_rows(expected, ids, state.logits.clone().unbind(0))
    runner.execute_model_state = None
    runner.kv_connector_output = None
    return rows


class DraftOnlyForwardAdapter:
    """Uniquely named worker-extension entry; never overrides stock methods."""

    def sr_draft_materialize(self, scheduler_output: Any) -> dict:
        validate_runner(self.model_runner)
        if self.model_runner.execute_model_state is not None:
            raise RuntimeError("Draft execute attempted before previous completion")
        output = self.execute_model(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            if self.model_runner.execute_model_state is not None:
                raise RuntimeError("empty Draft cleanup left pending forward state")
            return {}
        return complete_draft_forward(
            self.model_runner, output, tuple(scheduler_output.num_scheduled_tokens)
        )


class VllmDraftWorker:
    def __init__(self, config: Phase4Config, metrics: DraftMetrics) -> None:
        self.metrics = metrics
        self.owner_thread = threading.get_ident()
        self.views: dict[str, Any] = {}
        self.events: list[tuple[str, Any, Any]] = []
        self.hooks: list[Any] = []
        self.executor = None
        self.closed = False
        self.phase = "warmup"
        self.active_rows: tuple[str, ...] = ()
        self.active_tokens = 0
        self.batch_sequence = 0
        self.blocks_allocated = self.blocks_freed = self.blocks_peak = 0
        self.api = audit_installed_api()  # Before importing either GPU framework.
        if sys.version_info[:2] != (3, 11):
            raise RuntimeError("Plan A GPU environment requires Python 3.11")
        if importlib.metadata.version("torch").split("+", 1)[0] != config.expected_pytorch_version:
            raise RuntimeError("Draft PyTorch version differs from frozen Phase4 environment")
        if os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "0") != "0":
            raise RuntimeError("Plan A requires VLLM_USE_V2_MODEL_RUNNER=0")
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            raise RuntimeError("Plan A Draft service must see only physical GPU0")
        if any(os.environ.get(k) not in (None, "0", "1") for k in ("RANK", "WORLD_SIZE")):
            raise RuntimeError("Draft service inherited a distributed Target launch")
        if os.environ.get("RANK", "0") != "0" or os.environ.get("WORLD_SIZE", "1") != "1":
            raise RuntimeError("Draft service requires an independent world=1")
        if config.draft.resolved_model_path == config.target.resolved_model_path:
            raise RuntimeError("Draft service must not load Target weights")
        model_json = json.loads((config.draft.resolved_model_path / "config.json").read_text())
        if (
            model_json.get("architectures") != ["Qwen3ForCausalLM"]
            or model_json.get("hidden_size") != 1024
            or model_json.get("num_hidden_layers") != 28
            or model_json.get("use_sliding_window", False)
        ):
            raise RuntimeError("Plan A admits only the frozen dense Qwen3-0.6B model")
        if not config.enforce_eager or config.enable_prefix_caching:
            raise RuntimeError("Plan A requires eager execution and private uncached prefixes")
        import torch
        from vllm.config import set_current_vllm_config
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.executor.uniproc_executor import UniProcExecutor
        from vllm.v1.worker.gpu_worker import Worker

        actual_module = Path(inspect.getfile(Worker)).resolve()
        if actual_module != Path(self.api["root"]) / "vllm/v1/worker/gpu_worker.py":
            raise RuntimeError("imported vLLM Worker differs from audited installed distribution")
        self.torch = torch
        self.config_context = set_current_vllm_config
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Plan A requires exactly one visible real CUDA device")
        if torch.distributed.is_initialized():
            raise RuntimeError("Draft service already belongs to a distributed world")
        self.vllm_config = EngineArgs(
            model=str(config.draft.resolved_model_path),
            tokenizer=str(config.draft.resolved_tokenizer_path),
            revision=config.draft.revision,
            tokenizer_revision=config.draft.tokenizer_revision,
            dtype=config.draft.dtype,
            trust_remote_code=config.draft.trust_remote_code,
            seed=config.sampling.seed,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.draft.gpu_memory_utilization,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            distributed_executor_backend="uni",
            enforce_eager=True,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            max_num_seqs=128,
            max_num_batched_tokens=max(config.max_model_len, 256),
            async_scheduling=False,
            enable_dbo=False,
            speculative_config=None,
            worker_cls="vllm.v1.worker.gpu_worker.Worker",
            worker_extension_cls=("specrhythm.phase4.vllm_draft_worker.DraftOnlyForwardAdapter"),
        ).create_engine_config()
        self._validate_config()
        started = time.monotonic_ns()
        try:
            with self.config_context(self.vllm_config):
                self.executor = UniProcExecutor(self.vllm_config)
                self.runner = self.executor.driver_worker.worker.model_runner
                if (
                    type(self.runner).__module__ != "vllm.v1.worker.gpu_model_runner"
                    or type(self.runner).__name__ != "GPUModelRunner"
                ):
                    raise RuntimeError("Draft did not construct the pinned MRV1 runner")
                validate_runner(self.runner)
                self.model = self.runner.get_model()
                self._install_model_events()
                self._initialize_cache()
                self.fence("startup")
                if self.runner.get_model() is not self.model:
                    raise RuntimeError("Draft warmup replaced the one resident model")
                identity = active_cuda_device_identity(torch)
                validate_placement(identity, config)
                world = torch.distributed.get_world_size()
                if world != 1 or torch.distributed.get_rank() != 0:
                    raise RuntimeError("Draft worker joined the wrong distributed world")
                parameters = [p for p in self.model.parameters() if p.numel()]
                dtype = getattr(torch, config.draft.dtype)
                if not parameters or any(p.device != torch.device("cuda:0") for p in parameters):
                    raise RuntimeError("Draft weights are not exclusively on GPU0")
                if any(p.dtype != dtype for p in parameters):
                    raise RuntimeError("Draft parameter dtype differs from frozen config")
                self.provenance = {
                    **identity,
                    "model": model_revision_manifest(
                        config.draft.resolved_model_path, config.draft.revision
                    ),
                    "tokenizer": model_revision_manifest(
                        config.draft.resolved_tokenizer_path, config.draft.tokenizer_revision
                    ),
                    "parameter_count": sum(p.numel() for p in parameters),
                    "parameter_bytes": sum(p.numel() * p.element_size() for p in parameters),
                    "dtype": config.draft.dtype,
                    "model_instance_count": 1,
                    "world_size": world,
                    "global_rank": 0,
                    "tensor_parallel_size": 1,
                    "worker_class": (
                        f"{type(self.executor.driver_worker.worker).__module__}.Worker"
                    ),
                    "runner_class": f"{type(self.runner).__module__}.{type(self.runner).__name__}",
                    "cache_initialized": bool(self.runner.kv_caches),
                    "kv_cache_num_blocks": self.cache_config.num_blocks,
                    "kv_cache_group_count": len(self.cache_config.kv_cache_groups),
                    "block_size": self.vllm_config.cache_config.block_size,
                    "max_num_seqs": self.vllm_config.scheduler_config.max_num_seqs,
                    "max_num_batched_tokens": (
                        self.vllm_config.scheduler_config.max_num_batched_tokens
                    ),
                    "enforce_eager": True,
                    "prefix_caching": False,
                    "vllm_api": self.api,
                    "startup_uuid_validation_count": 1,
                    "startup_warmup_ns": time.monotonic_ns() - started,
                    "warmup_model_forward_count": self.metrics.forwards["warmup"],
                }
                if not self.provenance["cache_initialized"]:
                    raise RuntimeError("Draft worker did not initialize real KV storage")
        except Exception:
            self.shutdown()
            raise

    def _validate_config(self) -> None:
        cfg = self.vllm_config
        if any(
            getattr(cfg, f) is not None
            for f in (
                "speculative_config",
                "lora_config",
                "kv_transfer_config",
                "ec_transfer_config",
            )
        ):
            raise RuntimeError("unsupported Draft model/connector configuration")
        if cfg.model_config.is_hybrid or cfg.parallel_config.enable_eplb:
            raise RuntimeError("hybrid/MoE completion is outside Plan A")
        if (
            any(
                getattr(cfg.parallel_config, name) != 1
                for name in (
                    "tensor_parallel_size",
                    "pipeline_parallel_size",
                    "data_parallel_size",
                )
            )
            or cfg.parallel_config.enable_dbo
            or cfg.scheduler_config.async_scheduling
            or cfg.cache_config.enable_prefix_caching
            or not cfg.model_config.enforce_eager
        ):
            raise RuntimeError("Draft effective execution config differs from isolated eager TP1")

    def _initialize_cache(self) -> None:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm.v1.core.kv_cache_utils import (
            generate_scheduler_kv_cache_config,
            get_kv_cache_configs,
            resolve_kv_cache_block_sizes,
        )
        from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        cfg = self.vllm_config
        register_all_kvcache_specs(cfg)
        specs = self.executor.get_kv_cache_specs()
        if len(specs) != 1 or not specs[0]:
            raise RuntimeError("Draft must have exactly one worker with attention KV")
        if any(
            type(s) is not FullAttentionSpec
            or s.sliding_window is not None
            or s.attention_chunk_size is not None
            for s in specs[0].values()
        ):
            raise RuntimeError("Plan A supports only private full-attention KV")
        required_len = cfg.model_config.max_model_len
        configs = get_kv_cache_configs(cfg, specs, self.executor.determine_available_memory())
        if cfg.model_config.max_model_len != required_len:
            raise RuntimeError("Draft cache sizing changed the frozen model context")
        self.cache_config = generate_scheduler_kv_cache_config(configs)
        cfg.cache_config.num_gpu_blocks = self.cache_config.num_blocks
        cfg.cache_config.block_size = min(
            g.kv_cache_spec.block_size for g in self.cache_config.kv_cache_groups
        )
        cfg.validate_block_size()
        self.executor.initialize_from_config(configs)
        block_size, hash_size = resolve_kv_cache_block_sizes(self.cache_config, cfg)
        self.kv = KVCacheManager(
            self.cache_config,
            max_model_len=required_len,
            scheduler_block_size=block_size,
            hash_block_size=hash_size,
            max_num_batched_tokens=cfg.scheduler_config.max_num_batched_tokens,
            enable_caching=False,
            use_eagle=False,
        )

    def _install_model_events(self) -> None:
        def before(_module: Any, _args: Any) -> None:
            event = self.torch.cuda.Event(enable_timing=True)
            event.record()
            self._model_start = event

        def after(_module: Any, _args: Any, _output: Any) -> None:
            event = self.torch.cuda.Event(enable_timing=True)
            event.record()
            self.events.append((self.phase, self._model_start, event))
            self.metrics.forward(
                self.phase, max(len(self.active_rows), 1), max(self.active_tokens, 1)
            )

        self.hooks = [
            self.model.register_forward_pre_hook(before),
            self.model.register_forward_hook(after),
        ]

    def materialize(self, rows: Sequence[DraftMaterialization], purpose: str) -> dict:
        from vllm.sampling_params import SamplingParams
        from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
        from vllm.v1.request import Request, RequestStatus

        self._owner()
        if not rows:
            return {}
        unique_ids([r.request_id for r in rows])
        cfg = self.vllm_config
        if len(rows) > cfg.scheduler_config.max_num_seqs:
            raise RuntimeError("Draft cohort exceeds configured request capacity")
        # Setup can be singleton full-prefix RPCs. No decode cohort is split silently.
        if sum(len(r.suffix) for r in rows) > cfg.scheduler_config.max_num_batched_tokens:
            raise RuntimeError("Draft materialization exceeds configured token capacity")
        self.phase = purpose
        self.active_rows = tuple(r.request_id for r in rows)
        self.active_tokens = sum(len(r.suffix) for r in rows)
        self.kv.new_step_starts()
        output = SchedulerOutput.make_empty()
        for row in rows:
            params = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
            view = Request(row.request_id, list(row.context), params, None)
            view.status = RequestStatus.RUNNING
            view.num_computed_tokens = row.valid_length
            allocated = self.kv.allocate_slots(view, num_new_tokens=len(row.suffix))
            if allocated is None:
                raise RuntimeError("Draft private KV exhausted; no eviction/replay fallback")
            self.blocks_allocated += sum(len(g) for g in allocated.get_block_ids())
            self.views[row.request_id] = view
            block_ids = self.kv.get_block_ids(row.request_id)
            output.scheduled_new_reqs.append(
                NewRequestData(
                    req_id=row.request_id,
                    prompt_token_ids=list(row.context),
                    mm_features=[],
                    sampling_params=params,
                    pooling_params=None,
                    block_ids=block_ids,
                    num_computed_tokens=row.valid_length,
                    lora_request=None,
                )
            )
            output.num_scheduled_tokens[row.request_id] = len(row.suffix)
        output.total_num_scheduled_tokens = self.active_tokens
        output.num_common_prefix_blocks = [0] * len(self.cache_config.kv_cache_groups)
        if self.cache_config.needs_kv_cache_zeroing:
            output.new_block_ids_to_zero = self.kv.take_new_block_ids() or None
        before = self.metrics.forwards[purpose]
        with self.config_context(cfg):
            result = self.executor.collective_rpc(
                "sr_draft_materialize", args=(output,), single_value=True
            )
        if self.metrics.forwards[purpose] != before + 1:
            raise RuntimeError("Draft materialization did not execute exactly one model batch")
        self.blocks_peak = max(self.blocks_peak, self.blocks_allocated - self.blocks_freed)
        self.batch_sequence += 1
        self.metrics.counters["row_mapping_validations"] += len(rows)
        return result

    def greedy(self, logits: Sequence[Any]) -> tuple[int, ...]:
        self._owner()
        with self.torch.inference_mode():
            ids = self.torch.stack(tuple(logits)).argmax(dim=-1)
            self.metrics.syncs["bulk_token_d2h"] += 1
            return tuple(ids.cpu().tolist())

    def fence(self, reason: str) -> None:
        self._owner()
        self.metrics.syncs[reason] += 1
        self.torch.cuda.synchronize()
        for purpose, start, end in self.events:
            self.metrics.gpu_ms[purpose] += start.elapsed_time(end)
        self.events.clear()

    def request_evidence(self, request_id: str) -> dict:
        return {
            "draft_internal_request_id": request_id,
            "draft_batch_id": self.batch_sequence,
            "draft_physical_request_block_identity": self.kv.get_block_ids(request_id),
            "draft_physical_request_block_observable": True,
        }

    def release(self, request_ids: Sequence[str]) -> None:
        from vllm.v1.core.sched.output import SchedulerOutput

        self._owner()
        if not request_ids:
            return
        self.fence("release")
        output = SchedulerOutput.make_empty()
        output.finished_req_ids = set(request_ids)
        with self.config_context(self.vllm_config):
            self.executor.collective_rpc("sr_draft_materialize", args=(output,), single_value=True)
        for request_id in request_ids:
            self.blocks_freed += sum(len(g) for g in self.kv.get_block_ids(request_id))
            self.kv.free(self.views.pop(request_id))

    def resource_evidence(self) -> dict:
        return {
            "blocks_allocated": self.blocks_allocated,
            "blocks_freed": self.blocks_freed,
            "blocks_peak": self.blocks_peak,
            "live_allocator_requests": len(self.views),
            "worker_shutdown_complete": self.closed,
        }

    def _owner(self) -> None:
        if threading.get_ident() != self.owner_thread or self.closed:
            raise RuntimeError("Draft worker is closed or called outside its CUDA owner thread")

    def shutdown(self) -> None:
        if self.closed:
            return
        try:
            if self.executor is not None:
                from vllm.distributed.parallel_state import (
                    destroy_distributed_environment,
                    destroy_model_parallel,
                )

                try:
                    if self.views:
                        self.release(tuple(self.views))
                    self.fence("shutdown")
                finally:
                    for hook in self.hooks:
                        hook.remove()
                    self.hooks.clear()
                    try:
                        with self.config_context(self.vllm_config):
                            self.executor.shutdown()
                    finally:
                        destroy_model_parallel()
                        destroy_distributed_environment()
        finally:
            self.model = None
            self.executor = None
            self.runner = None
            self.closed = True
