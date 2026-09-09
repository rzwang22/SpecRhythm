"""Server G0: no model weights loaded, no patch mutation, no dependency installation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from specrhythm.phase4.config import load_phase4_config
from specrhythm.phase4.manifest import (
    collect_environment,
    collect_topology,
    model_revision_manifest,
    sha256_file,
    validate_environment,
    validate_topology,
)
from specrhythm.serving.common import digest, require
from specrhythm.serving.s1_runtime import resident_capacity
from specrhythm.serving.s1_workload import (
    QUOTAS,
    create_manifest,
    select_subset,
    verify_s0,
    write_once,
)

REPO = Path(__file__).resolve().parents[3]
GPU_PYTHON = "/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11"
DRAFT_MODEL = Path("/root/autodl-tmp/models/Qwen3-0.6B")
TARGET_MODEL = Path("/root/autodl-tmp/models/Qwen3-32B")


def git_identity():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()

    require(
        not git("status", "--porcelain", "--untracked-files=normal"),
        "S1 execution checkout is dirty",
    )
    return git("rev-parse", "HEAD")


def clean_environment(mode, manifest_path=None):
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("SR_PHASE4", "PHASE4B", "SR_S1"))
    }
    for key in ("USE_TORCH", "USE_TF", "USE_FLAX", "RANK", "WORLD_SIZE", "LOCAL_RANK"):
        env.pop(key, None)
    env.update(
        HF_HOME="/root/.cache/huggingface",
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_BATCH_INVARIANT="1",
        SR_PHASE4_DRAFT_BACKEND="vllm-batched",
        SR_PHASE4_DUAL_UUID_QUERY_MODE="live",
        SR_PHASE4B3_HF_METRICS="0",
        SR_PHASE4B_DUAL_RHYTHM="pingpong" if mode == "pingpong" else "legacy",
        CUDA_VISIBLE_DEVICES="1,2",
        PYTHONUNBUFFERED="1",
    )
    if manifest_path:
        env["SR_S1_EXECUTION_MANIFEST"] = str(manifest_path.resolve())
    return env


def checked_patch_adapter(value, config):
    """Normalize a read-only installed-state check without pretending to apply patches."""
    from specrhythm.phase4.serial_runner import (
        PATCHED_VLLM_RUNNER_SHA256,
        PATCHED_VLLM_SCHEDULER_SHA256,
    )

    require(
        value.get("operation") == "check"
        and value.get("expected_state") == "patched"
        and value.get("valid") is True
        and not value.get("errors")
        and value.get("verified_source_commit") == config.expected_vllm_commit
        and value.get("actual_runner_sha256") == PATCHED_VLLM_RUNNER_SHA256
        and value.get("actual_scheduler_sha256") == PATCHED_VLLM_SCHEDULER_SHA256,
        "S1 installed five-patch state check failed",
    )
    stack = value.get("patch_stack", [])
    require(len(stack) == 5, "S1 requires exactly the existing five patches")
    for row in stack:
        path = REPO / "integrations/vllm/patches" / Path(row["patch_file"]).name
        require(sha256_file(path) == row["patch_sha256"], "S1 patch source identity mismatch")
    return {
        **value,
        "target_file_sha256_after": value["actual_runner_sha256"],
        "scheduler_file_sha256_after": value["actual_scheduler_sha256"],
        "patch_sha256": stack[0]["patch_sha256"],
        "s1_patch_provenance": "read-only installed state check; apply operation not repeated",
    }


def model_files(path):
    value = model_revision_manifest(path, None)
    shards = sorted(path.glob("*.safetensors"))
    require(shards, "missing local model weight files", path=str(path))
    value["weights_inventory"] = {
        p.name: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in shards
    }
    # Metadata plus immutable local-file inventory; not a cryptographic weight-content digest.
    index = path / "model.safetensors.index.json"
    value["weight_index_sha256"] = sha256_file(index) if index.is_file() else None
    value["chat_template_file_sha256"] = {
        p.name: sha256_file(p) for p in sorted(path.glob("*.jinja"))
    }
    return value


def capacity_estimate(rows, model_config, weight_bytes, tp, total_mib, utilization):
    needed = resident_capacity(rows, 16)
    head_dim = model_config.get(
        "head_dim", model_config["hidden_size"] // model_config["num_attention_heads"]
    )
    kv_heads = model_config["num_key_value_heads"]
    require(kv_heads % tp == 0, "KV heads do not divide configured TP")
    bytes_per_token = 2 * 2 * model_config["num_hidden_layers"] * (kv_heads // tp) * head_dim
    kv_bytes = needed["required_blocks"] * needed["block_size"] * bytes_per_token
    workspace_reserve = 4 * 1024**3
    available = int(total_mib * 1024**2 * utilization) - weight_bytes // tp - workspace_reserve
    return {
        **needed,
        "kv_bytes_per_token_per_rank": bytes_per_token,
        "required_kv_bytes_per_rank": kv_bytes,
        "estimated_kv_bytes_per_rank": available,
        "workspace_reserve_bytes_per_rank": workspace_reserve,
        "estimate_only": True,
        "actual_engine_capacity_check_before_generate": True,
        "valid": available >= kv_bytes and len(rows) <= 128 and len(rows) * 5 <= 4096,
    }


def prepare(root: Path, s0: Path, vllm_source: Path):
    from specrhythm.phase4.vllm_installation import locate_installed_vllm_file
    from specrhythm.serving.schema import load_requests

    require(
        sys.platform == "linux" and sys.version_info[:2] == (3, 11),
        "G0 requires the pinned Linux GPU Python 3.11; no Mac GPU execution",
    )
    require(Path(sys.executable).resolve() == Path(GPU_PYTHON).resolve(), "wrong S1 GPU Python")
    commit = git_identity()
    parent = verify_s0(s0)
    root.mkdir(parents=True, exist_ok=False)
    config_data = json.loads((REPO / "configs/phase4b_dual_batch_1d2v.yaml").read_text())
    config_data["draft"].update(model_path=str(DRAFT_MODEL), tokenizer_path=str(DRAFT_MODEL))
    config_data["target"].update(model_path=str(TARGET_MODEL), tokenizer_path=str(DRAFT_MODEL))
    write_once(root / "config.json", config_data)
    config = load_phase4_config(str(root / "config.json"))
    environment, topology = collect_environment(vllm_source), collect_topology()
    write_once(root / "environment.json", environment)
    write_once(root / "topology.json", topology)
    require(validate_environment(environment, config)["valid"], "pinned S1 environment invalid")
    require(validate_topology(topology, config)["valid"], "S1 device topology invalid")
    require(
        all("A800" in r["name"] for r in topology["gpus"] if r["physical_gpu_id"] in (0, 1, 2)),
        "S1 runbook requires the requested 3 x A800 placement",
    )
    installed_root = locate_installed_vllm_file(
        Path("vllm/v1/worker/gpu_model_runner.py")
    ).parents[3]
    subprocess.run(
        [
            sys.executable,
            str(REPO / "integrations/vllm/manage_patch.py"),
            "check",
            "--expect-state",
            "patched",
            "--vllm-root",
            str(installed_root),
            "--source",
            str(vllm_source),
            "--manifest",
            str(root / "patch-manifest.json"),
        ],
        check=True,
    )
    checked_patch_adapter(json.loads((root / "patch-manifest.json").read_text()), config)
    identities = {"draft": model_files(DRAFT_MODEL), "target": model_files(TARGET_MODEL)}
    tokenizer = json.loads((DRAFT_MODEL / "tokenizer.json").read_text())
    tokenizer_cfg = json.loads((DRAFT_MODEL / "tokenizer_config.json").read_text())
    eos_name = tokenizer_cfg["eos_token"]
    if isinstance(eos_name, dict):
        eos_name = eos_name["content"]
    primary = [t["id"] for t in tokenizer["added_tokens"] if t["content"] == eos_name]
    require(len(primary) == 1, "tokenizer primary EOS identity unavailable")
    generation = json.loads((TARGET_MODEL / "generation_config.json").read_text())
    additional = generation.get("eos_token_id", [])
    if isinstance(additional, int):
        additional = [additional]
    eos = sorted(set(primary + additional))
    tokenizer_report = json.loads((s0 / "server-tokenizer-validation.json").read_text())
    for role, model in (("draft", DRAFT_MODEL), ("target", TARGET_MODEL)):
        metadata = json.loads((model / "tokenizer_config.json").read_text())
        for key in ("name_or_path", "_name_or_path"):
            metadata.pop(key, None)
        template_file = model / "chat_template.jinja"
        template = (
            template_file.read_text() if template_file.is_file() else metadata["chat_template"]
        )
        require(
            sha256_file(model / "tokenizer.json")
            == tokenizer_report[role]["tokenizer_json_sha256"]
            and digest(metadata) == tokenizer_report[role]["tokenizer_config_content_sha256"]
            and digest(template) == tokenizer_report[role]["chat_template_sha256"],
            "server tokenizer changed since S0 acceptance",
            role=role,
        )
    execution = {
        "git_commit": commit,
        "python_executable": str(Path(sys.executable)),
        "config_sha256": sha256_file(root / "config.json"),
        "environment_sha256": sha256_file(root / "environment.json"),
        "topology_sha256": sha256_file(root / "topology.json"),
        "models": identities,
        "patch_manifest_sha256": sha256_file(root / "patch-manifest.json"),
        "vllm_source": str(vllm_source.resolve()),
        "eos_token_ids": eos,
        "capacity": {"max_model_len": 4096, "max_num_seqs": 128, "max_num_batched_tokens": 4096},
        "K": 4,
        "uuid_mode": "live",
        "Draft_backend": "vllm-batched",
        "numerical_mode": "batch-invariant",
        "arrival_replay_enabled": False,
    }
    rows = load_requests(s0 / "main1000.jsonl")
    memory = {r["physical_gpu_id"]: r["memory_total_mib"] for r in topology["gpus"]}
    capacities = {}
    for subset, quota in QUOTAS.items():
        create_manifest(root / subset, subset, rows, parent, execution)
        chosen = select_subset(rows, quota)
        capacities[subset] = {}
        for role, engine in (("draft", config.draft), ("target", config.target)):
            identity = identities[role]
            capacities[subset][role] = capacity_estimate(
                chosen,
                json.loads((Path(identity["path"]) / "config.json").read_text()),
                sum(r["size"] for r in identity["weights_inventory"].values()),
                engine.tensor_parallel_size,
                min(memory[i] for i in engine.physical_gpu_ids),
                engine.gpu_memory_utilization,
            )
    write_once(root / "capacity-estimates.json", capacities)
    write_once(
        root / "g0.json",
        {
            "valid": True,
            "stage": "G0",
            "model_weights_loaded": False,
            "execution": execution,
            "S0": parent,
            "capacity": capacities,
            "GPU_inference_performed": False,
        },
    )
    return 0


def validate_execution_files(root, execution):
    require(git_identity() == execution["git_commit"], "S1 execution checkout changed")
    for name, key in (
        ("config.json", "config_sha256"),
        ("patch-manifest.json", "patch_manifest_sha256"),
        ("environment.json", "environment_sha256"),
        ("topology.json", "topology_sha256"),
    ):
        require(
            sha256_file(root / name) == execution[key],
            "S1 execution input hash changed",
            file=name,
        )
    for identity in execution["models"].values():
        require(model_files(Path(identity["path"])) == identity, "S1 local model identity changed")
    # Importing metadata helpers does not create engines or load weights.
    from specrhythm.phase4.serial_runner import validate_installed_patch_stack

    cfg = load_phase4_config(str(root / "config.json"))
    validate_installed_patch_stack(
        checked_patch_adapter(json.loads((root / "patch-manifest.json").read_text()), cfg)
    )
