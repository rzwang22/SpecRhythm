"""Persistent TP target worker used by the five-GPU serial Phase-3A runner."""

from __future__ import annotations

import multiprocessing
import os
import queue
import socket
from dataclasses import asdict
from typing import Any, Optional

from specrhythm.phase3.config import ModelRuntimeConfig
from specrhythm.phase3.engine import (
    CausalLMBackend,
    EngineUnavailableError,
    NextTokenDistribution,
    RankedToken,
    TransformersBackend,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _target_worker(
    rank: int,
    world_size: int,
    gpu_ids: tuple[int, ...],
    master_port: int,
    config: ModelRuntimeConfig,
    seed: int,
    commands: Any,
    responses: Any,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    backend: Optional[TransformersBackend] = None
    try:
        backend = TransformersBackend(config, seed)
        if rank == 0:
            responses.put(
                {
                    "kind": "ready",
                    "model_id": backend.model_id,
                    "eos_token_id": backend.eos_token_id,
                    "vocab_size": backend.vocab_size,
                    "tokenizer_fingerprint": backend.tokenizer_fingerprint,
                    "transformers_version": backend.transformers_version,
                }
            )
        while True:
            command = commands.get()
            if command["kind"] == "close":
                break
            if command["kind"] != "next_token":
                raise ValueError(f"unknown TP worker command: {command['kind']}")
            distribution = backend.next_token(command["context"], command["top_k"])
            if rank == 0:
                responses.put(
                    {
                        "kind": "result",
                        "job_id": command["job_id"],
                        "ranked_tokens": [
                            asdict(token) for token in distribution.ranked_tokens
                        ],
                        "entropy": distribution.entropy,
                        "top1_top2_margin": distribution.top1_top2_margin,
                    }
                )
    except Exception as error:  # pragma: no cover - exercised only on GPU hosts
        if rank == 0:
            responses.put(
                {
                    "kind": "error",
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
    finally:
        if backend is not None:
            backend.close()
        try:
            import torch  # type: ignore[import-not-found]

            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except (ImportError, RuntimeError):
            pass


class TensorParallelTargetPool(CausalLMBackend):
    """Expose a TP target model to the serial coordinator through persistent workers."""

    def __init__(
        self,
        config: ModelRuntimeConfig,
        seed: int,
        *,
        timeout_seconds: int = 900,
    ) -> None:
        if config.tp_size < 2:
            raise ValueError("TensorParallelTargetPool is only needed for TP greater than one")
        self._context = multiprocessing.get_context("spawn")
        self._commands = [self._context.Queue() for _ in range(config.tp_size)]
        self._responses = self._context.Queue()
        port = _free_port()
        self._processes = [
            self._context.Process(
                target=_target_worker,
                args=(
                    rank,
                    config.tp_size,
                    config.gpu_ids,
                    port,
                    config,
                    seed,
                    self._commands[rank],
                    self._responses,
                ),
                daemon=True,
            )
            for rank in range(config.tp_size)
        ]
        for process in self._processes:
            process.start()
        self._timeout_seconds = timeout_seconds
        ready = self._receive()
        if ready.get("kind") != "ready":
            self.close()
            raise EngineUnavailableError(
                f"TP target worker failed: {ready.get('error_type')}: {ready.get('message')}"
            )
        self.model_id = str(ready["model_id"])
        self.eos_token_id = ready["eos_token_id"]
        self.vocab_size = int(ready["vocab_size"])
        self.tokenizer_fingerprint = str(ready["tokenizer_fingerprint"])
        self.transformers_version = str(ready["transformers_version"])
        self._job_id = 0
        self._closed = False

    def _receive(self) -> dict[str, Any]:
        try:
            return self._responses.get(timeout=self._timeout_seconds)
        except queue.Empty as error:
            states = [process.exitcode for process in self._processes]
            raise EngineUnavailableError(
                f"timed out waiting for TP target workers; exit codes={states}"
            ) from error

    def encode(self, prompt: str) -> list[int]:
        raise RuntimeError("serial TP target uses the tokenizer-compatible draft encoding")

    def next_token(self, context: list[int], top_k: int) -> NextTokenDistribution:
        job_id = self._job_id
        self._job_id += 1
        command = {
            "kind": "next_token",
            "job_id": job_id,
            "context": context,
            "top_k": top_k,
        }
        for commands in self._commands:
            commands.put(command)
        response = self._receive()
        if response.get("kind") == "error":
            raise EngineUnavailableError(
                f"TP target worker failed: {response.get('error_type')}: "
                f"{response.get('message')}"
            )
        if response.get("kind") != "result" or response.get("job_id") != job_id:
            raise EngineUnavailableError("TP target worker returned an invalid response")
        return NextTokenDistribution(
            tuple(RankedToken(**value) for value in response["ranked_tokens"]),
            float(response["entropy"]),
            float(response["top1_top2_margin"]),
        )

    def next_token_batch(
        self, contexts: list[list[int]], top_k: int
    ) -> tuple[NextTokenDistribution, ...]:
        # The online trace runner currently requests one target context at a time.
        # Direct torchrun benchmarks use TransformersBackend's true batch path.
        return tuple(self.next_token(context, top_k) for context in contexts)

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        for commands in self._commands:
            commands.put({"kind": "close"})
        for process in self._processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
