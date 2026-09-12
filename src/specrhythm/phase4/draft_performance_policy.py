"""D4/D5 admission from completed qualification certificates; CPU and files only.

D3's qualification and execution are unchanged. Historical raw diagnostics are
not recursively requalified before each performance run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from specrhythm.phase4.batched_draft_service import write_immutable_report
from specrhythm.phase4.config import load_phase4_config
from specrhythm.phase4.manifest import sha256_file
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.stock_vllm import load_smoke_requests

POLICY = "specrhythm.phase4b3-performance-validity.v1"
BLOCKING = (
    "successful execution return path",
    "expected completed requests",
    "token accounting and declared output limits",
    "request lifecycle and cleanup",
    "structurally valid Target verification",
    "workload/model/config/backend experiment identity",
    "actual multi-request vLLM Draft proposal batching",
    "finite positive performance metrics",
)
DIAGNOSTIC = (
    "qualified HF/vLLM numerical divergence and final sequence equality",
    "JIT and informational numerical diagnostics",
    "empty error serialization and nonessential artifact metadata",
    "duplicated historical provenance and qualification commit checks",
    "work differences (reported; never imply pure batching)",
)


def read(path):
    return json.loads(Path(path).read_text())


def runtime_identity(provenance):
    """Execution identity, excluding timings, warnings and descriptive metadata."""
    value = provenance or {}
    result = {
        key: value.get(key)
        for key in (
            "model",
            "tokenizer",
            "dtype",
            "runner_class",
            "enforce_eager",
            "prefix_caching",
            "tensor_parallel_size",
            "world_size",
            "global_rank",
            "physical_gpu_id",
            "logical_cuda_index",
            "cuda_visible_devices",
            "gpu_uuid",
        )
    }
    api = value.get("vllm_api") or {}
    result["vllm_api"] = {key: api.get(key) for key in ("vllm_version", "vllm_commit", "files")}
    return result


def require_performance_stage(qualification_path, stage, d4_path=None):
    if stage not in ("D4", "D5"):
        raise ValueError("only D4/D5 performance stages are supported")
    certificate = read(qualification_path)
    if certificate.get("valid") is not True or certificate.get("errors"):
        raise ValueError("performance requires a completed valid D1-D3 qualification")
    if stage == "D5":
        if d4_path is None:
            raise ValueError("D5 requires a qualified D4 comparison")
        d4 = read(d4_path)
        if (
            d4.get("stage") != "D4"
            or d4.get("d4_qualified") is not True
            or d4.get("performance_comparable") is not True
            or d4.get("errors")
        ):
            raise ValueError("D5 requires a qualified D4 comparison")
    return certificate


def prepare_serial(qualification, stage, d4, backend, config_path, workload, reference, commit):
    prior = require_performance_stage(qualification, stage, d4)
    config = load_phase4_config(str(config_path))
    identity = prior["run_identity"]
    if sha256_file(config_path) != identity["config_sha256"]:
        raise ValueError("Serial config differs from the qualified experiment")
    if str(config.draft.resolved_model_path) != identity["model_path"]:
        raise ValueError("Serial Draft model differs from the qualified experiment")
    count = 5 if stage == "D4" else 100
    if len([line for line in workload.read_text().splitlines() if line.strip()]) != count:
        raise ValueError("Serial stage requires the exact corrected workload size")
    requests = load_smoke_requests(workload, count)
    return {
        "schema_version": "specrhythm.phase4b3-serial-admission.v1",
        "qualification_policy": POLICY,
        "valid": True,
        "stage": stage,
        "backend": backend,
        "run_identity": {
            "execution_commit": commit,
            "model_path": str(config.draft.resolved_model_path),
            "config_sha256": sha256_file(config_path),
        },
        "historical_identity_revalidated": False,
        "d3_qualification_sha256": sha256_file(qualification),
        "d4_comparison_sha256": sha256_file(d4) if d4 else None,
        "workload_sha256": sha256_file(workload),
        "reference_sha256": sha256_file(reference),
        "request_count": count,
        "requests": {
            r.request_id: {
                "prompt_token_count": len(r.prompt_token_ids),
                "prompt_token_ids_sha256": token_prefix_hash(r.prompt_token_ids),
                "maximum_new_tokens": r.maximum_new_tokens,
            }
            for r in requests
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--stage", choices=("D4", "D5"), required=True)
    parser.add_argument("--d4", type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--backend", choices=("hf", "vllm"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if commit != args.expected_commit:
        raise ValueError("current checkout differs from requested performance commit")
    result = prepare_serial(
        args.qualification,
        args.stage,
        args.d4,
        args.backend,
        args.config,
        args.workload,
        args.reference,
        commit,
    )
    write_immutable_report(args.output, result)
    print("Serial performance admission valid; no GPU initialized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
