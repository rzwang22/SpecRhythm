"""Observed production Draft; unchanged proposal, commit, EOS and release algorithms."""

from __future__ import annotations

import time

from specrhythm.serving.fixed_observe import TIMERS, DeviceTimeline
from specrhythm.serving.s2_draft import S2DraftBackend


class FixedDraftBackend(S2DraftBackend):
    def __init__(self, config, *, worker=None):
        super().__init__(config, worker=worker)
        self.fixed_proposals = []
        self.fixed_device_report = None
        worker = self.worker
        self._provenance["fixed_engine_limits"] = {
            "max_num_seqs": worker.vllm_config.scheduler_config.max_num_seqs,
            "max_num_batched_tokens": worker.vllm_config.scheduler_config.max_num_batched_tokens,
            "max_model_len": worker.vllm_config.model_config.max_model_len,
            "num_gpu_blocks": self.provenance["kv_cache_num_blocks"],
            "block_size": self.provenance["block_size"],
        }
        self.fixed_device = DeviceTimeline(
            worker.torch,
            worker.model,
            lambda: {
                "role": "draft",
                "purpose": worker.phase,
                "internal_request_ids": list(worker.active_rows),
                "B": len(worker.active_rows),
                "Q": worker.active_tokens,
            },
            identity={
                "role": "draft",
                "global_rank": 0,
                "gpu_uuid": self.provenance["gpu_uuid"],
                "physical_gpu_id": self.provenance["physical_gpu_id"],
                "dtype": str(worker.vllm_config.model_config.dtype),
                "attention_backends": sorted(
                    {g.backend.__name__ for groups in worker.runner.attn_groups for g in groups}
                ),
            },
        )

    def propose_many(self, plans):
        start = time.monotonic_ns()
        result = super().propose_many(plans)
        self.fixed_proposals.append(
            {
                "start_ns": start,
                "end_ns": time.monotonic_ns(),
                "B": len(plans),
                "request_ids": [p.request_id for p in plans],
                "context_lengths": [len(p.prefix) for p in plans],
                "budgets": [p.budget for p in plans],
                "candidate_lengths": [len(result[p.request_id]) for p in plans],
                "round_ids": [p.round_id for p in plans],
                "first_token_from_cached_logits": True,
            }
        )
        return result

    def shutdown(self):
        if not self.closed:
            # Existing shutdown fences the worker before tearing it down. Read our
            # events at this safe boundary, not once per Draft round.
            self.worker.fence("fixed_diagnostic_window_end")
            self.fixed_device_report = self.fixed_device.report()
        super().shutdown()

    def report(self):
        return {
            **super().report(),
            "fixed_proposals": self.fixed_proposals,
            "fixed_device": self.fixed_device_report,
            "fixed_host": TIMERS.report(),
        }
