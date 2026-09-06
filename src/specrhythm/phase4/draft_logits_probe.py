"""Operator-only D3 same-prefix raw-logits experiment; never a serving entry point."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import traceback
import uuid
from pathlib import Path

from specrhythm.phase4.batched_draft_service import (
    BatchedDraftStateMachine,
    write_immutable_report,
)
from specrhythm.phase4.config import load_phase4_config
from specrhythm.phase4.draft_batch import DraftMaterialization
from specrhythm.phase4.draft_d3_diagnostics import D3MaterializationDiagnostic
from specrhythm.phase4.draft_logits_contract import (
    FIXTURE_SHA256,
    PATH_SCHEMA,
    PATHS,
    classify_paths,
    load_probe_fixture,
    require,
    summarize_logits,
)
from specrhythm.phase4.manifest import sha256_file
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.vllm_draft_worker import audit_installed_api

PROBE_ENV = {
    "CUDA_VISIBLE_DEVICES": "0",
    "VLLM_USE_V2_MODEL_RUNNER": "0",
    "VLLM_BATCH_INVARIANT": "1",
    "SR_PHASE4_DRAFT_BACKEND": "vllm-batched",
}


def preflight(config_path, expected_commit, fixture):
    """CPU file/version checks only, repeated before each independent GPU path."""
    for key, expected in PROBE_ENV.items():
        require(os.environ.get(key) == expected, f"probe requires {key}={expected}")
    require(sys.version_info[:2] == (3, 11), "probe requires qualified Python 3.11")
    require(
        not any(n == "vllm" or n.startswith("vllm.") for n in sys.modules),
        "probe preflight must precede vLLM imports",
    )
    require(
        os.environ.get("RANK", "0") == "0" and os.environ.get("WORLD_SIZE", "1") == "1",
        "probe inherited distributed launch",
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    require(
        len(expected_commit) == 40 and commit == expected_commit, "probe execution commit mismatch"
    )
    require(
        not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip(),
        "probe requires a clean checkout",
    )
    repo = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    )
    require(
        Path(__file__).resolve().is_relative_to(repo),
        "probe must use this checkout's editable install",
    )
    for name, digest in fixture["patch_sha256"].items():
        require(
            sha256_file(repo / "integrations/vllm/patches" / name) == digest,
            "original five-patch file changed",
        )
    require(
        sha256_file(Path(config_path)) == fixture["config_sha256"], "frozen config SHA256 mismatch"
    )
    config = load_phase4_config(config_path)
    require(
        config.draft.dtype == "bfloat16"
        and config.enforce_eager
        and not config.enable_prefix_caching,
        "probe requires original BF16/eager/private-KV config",
    )
    require(
        config.draft.physical_gpu_ids == (0,) and config.draft.tensor_parallel_size == 1,
        "probe requires GPU0/TP1",
    )
    require(
        config.draft.resolved_model_path == config.draft.resolved_tokenizer_path,
        "probe model/tokenizer paths differ",
    )
    model = config.draft.resolved_model_path
    for name, digest in fixture["model_metadata_sha256"].items():
        require(
            sha256_file(model / name) == digest, f"frozen model/tokenizer metadata changed: {name}"
        )
    weights = sorted(model.glob("*.safetensors"))
    require(bool(weights), "probe requires frozen safetensors weights")
    model_files = {p.name: sha256_file(p) for p in weights}
    model_files.update(
        {name: sha256_file(model / name) for name in fixture["model_metadata_sha256"]}
    )
    api = audit_installed_api()  # Version, source Git pin and all installed patched files.
    for name, digest in fixture["numerical_source_sha256"].items():
        require(
            sha256_file(Path(api["root"]) / name) == digest,
            f"pinned numerical source changed: {name}",
        )
    require(
        Path(os.environ.get("SR_VLLM_ROOT", "")).resolve() == Path(api["root"]).resolve(),
        "SR_VLLM_ROOT differs from audited installation",
    )
    versions = {
        name: importlib.metadata.version(name) for name in ("torch", "transformers", "vllm")
    }
    require(
        versions["torch"].split("+", 1)[0] == config.expected_pytorch_version,
        "qualified PyTorch version mismatch",
    )
    return config, {
        "execution_commit": commit,
        "fixture_sha256": FIXTURE_SHA256,
        "config_sha256": fixture["config_sha256"],
        "model_path": str(model),
        "model_files_sha256": model_files,
        "model_revision": config.draft.revision,
        "tokenizer_revision": config.draft.tokenizer_revision,
        "versions": versions,
        "vllm_api": api,
        "five_patch_sha256": fixture["patch_sha256"],
        "numerical_source_sha256": fixture["numerical_source_sha256"],
        "environment": {k: os.environ[k] for k in PROBE_ENV},
    }


class ProbeCapture(D3MaterializationDiagnostic):
    """Reuse D3 metadata checks; capture one raw focus row before completion."""

    def __init__(self, backend, fixture, path):
        super().__init__(backend)
        self.fixture, self.path = fixture, path
        self.capture = None

    def _wrap_logits(self, original):
        observed = super()._wrap_logits(original)

        def capture(hidden_states, *args, **kwargs):
            logits = observed(hidden_states, *args, **kwargs)
            frame = self.frame
            if frame is None or frame["round"] != 2 or frame["purpose"] != "proposal":
                return logits
            wanted = 8 if self.path == PATHS[3] else 0
            if frame["index"] != wanted:
                return logits
            require(self.capture is None, "probe captured the same boundary twice")
            expected = self.fixture["rows"] if self.path != PATHS[1] else self.fixture["rows"][-1:]
            ids = ["sr-draft:" + r["request_id"] for r in expected]
            actual = frame["after_forward"]["authoritative_logits_request_ids"]
            require(actual == ids, "probe final request order differs from frozen frame")
            plans = {r["request_id"]: r for r in frame["before_materialize"]}
            for row in expected:
                plan = plans[row["request_id"]]
                require(
                    plan["context_sha256"] == row["comparison_prefix_sha256"],
                    "probe captured a different prefix",
                )
                require(
                    plan["valid_length"] == len(row["committed_prefix"])
                    and plan["suffix"] == [row["append_token"]],
                    "probe captured a different materialization",
                )
            focus_index = actual.index("sr-draft:" + self.fixture["focus_request_id"])
            raw = logits[focus_index].detach()
            values = raw.float().cpu().tolist()
            summary = summarize_logits(values, str(raw.dtype))
            self.capture = {
                "frame_index": frame["index"],
                "logits": summary,
                "top1_by_request": dict(
                    zip(
                        [self._stable(rid) for rid in actual],
                        frame["after_forward"]["top1_token_ids"],
                    )
                ),
            }
            return logits

        return capture


def initialize_fresh(backend, fixture, path):
    """B/C use identical singleton prefills; only the final decode is B1/B7."""
    require(
        not backend.states and not backend.retired and not backend.worker.views,
        "fresh control inherited KV/request state",
    )
    rows = fixture["rows"][-1:] if path == PATHS[1] else fixture["rows"]
    for row in rows:
        backend.initialize(row["request_id"], row["committed_prefix"])
    require(
        set(backend.states) == {r["request_id"] for r in rows}, "fresh request identity mismatch"
    )
    return rows


def run_fresh_forward(backend, rows):
    plans = []
    for row in rows:
        state = backend.states[row["request_id"]]
        require(
            list(state.prefix) == row["committed_prefix"]
            and state.materialized == len(state.prefix),
            "fresh control did not materialize its own committed prefix",
        )
        require(
            state.next_round == 0 and state.proposal is None,
            "fresh control acquired proposal history",
        )
        plans.append(
            DraftMaterialization(
                state.internal_id, tuple(row["comparison_prefix"]), len(state.prefix)
            )
        )
    result = backend.worker.materialize(plans, "proposal")
    backend.worker.fence("d3_probe_capture")
    require(set(result) == {p.request_id for p in plans}, "fresh completion identity mismatch")


def replay_persistent(backend, fixture, observer):
    machine = BatchedDraftStateMachine(backend)
    history = fixture["history"]
    for rid, prefix in history["initial"].items():
        machine.initialize(rid, prefix, token_prefix_hash(prefix))
    observer.attach()
    comparisons = []
    for index, item in enumerate(history["rounds"][:3]):
        observer.round = index
        response = machine.batch_propose(item["proposals"])
        actual = {r["request_id"]: r["proposal_token_ids"] for r in response["proposals"]}
        comparisons.append(
            {
                "round": index,
                "actual": actual,
                "expected": item["expected"],
                "exact": actual == item["expected"],
            }
        )
        if index < 2:
            require(
                actual == item["expected"], "persistent path diverged before the frozen boundary"
            )
            machine.synchronize_and_batch_propose(item["synchronizations"], [])
    # The original round-2 mismatch is evidence, not a probe infrastructure error.
    return comparisons


def run_path(path, config, fixture, identity):
    focus = fixture["rows"][-1]
    order = [focus["request_id"]] if path in PATHS[:2] else fixture["request_order"]
    report = {
        "schema_version": PATH_SCHEMA,
        "path": path,
        "valid": False,
        "errors": [],
        "diagnostic_only": True,
        "performance_result": False,
        "process_pid": os.getpid(),
        "process_lifetime_id": str(uuid.uuid4()),
        "run_identity": identity,
        "fixture_sha256": FIXTURE_SHA256,
        "prefix_token_ids": focus["comparison_prefix"],
        "prefix_sha256": focus["comparison_prefix_sha256"],
        "request_order": order,
        "batch_size": len(order),
        "model_dtype": config.draft.dtype,
        "history_rounds_replayed": 0,
        "initial_live_requests": None,
        "cleanup_complete": False,
    }
    backend = observer = None
    try:
        if path == PATHS[0]:
            # This subprocess has never constructed/imported a vLLM worker.
            require(
                not any(n == "vllm" or n.startswith("vllm.") for n in sys.modules),
                "HF path inherited vLLM global overrides",
            )
            from specrhythm.phase4.draft_service import HFPersistentDraftBackend
            from specrhythm.phase4.stock_vllm import active_cuda_device_identity
            from specrhythm.phase4.vllm_draft_worker import validate_placement

            backend = HFPersistentDraftBackend(config)
            require(not backend.states, "HF fresh path inherited history")
            report["initial_live_requests"] = len(backend.states)
            parameters = [p for p in backend.model.parameters() if p.numel()]
            require(
                all(p.dtype == backend.torch.bfloat16 for p in parameters),
                "HF parameters are not BF16",
            )
            placement = active_cuda_device_identity(backend.torch)
            validate_placement(placement, config)
            report["gpu_identity"] = placement
            report["runtime_provenance"] = {
                **backend.provenance,
                "attention_implementation": getattr(
                    backend.model.config, "_attn_implementation", None
                ),
                "model_class": f"{type(backend.model).__module__}.{type(backend.model).__name__}",
                "vllm_imported_before_hf": False,
            }
            backend.initialize(focus["request_id"], focus["comparison_prefix"])
            raw = backend.states[focus["request_id"]].next_logits[0].detach()
            report["logits"] = summarize_logits(raw.float().cpu().tolist(), str(raw.dtype))
            report["top1_by_request"] = {focus["request_id"]: report["logits"]["top1_token"]}
            report["prefill_token_count"] = len(focus["comparison_prefix"])
        else:
            from specrhythm.phase4.batch_invariant import worker_batch_invariant_evidence
            from specrhythm.phase4.vllm_draft_backend import VllmBatchedDraftBackend

            backend = VllmBatchedDraftBackend(config)
            require(
                not backend.states
                and not backend.worker.views
                and not backend.worker.runner.input_batch.req_ids,
                "new vLLM worker inherited requests",
            )
            report["initial_live_requests"] = len(backend.states)
            evidence = worker_batch_invariant_evidence(
                backend.worker.executor.driver_worker.worker
            )
            require(
                evidence["batch_invariant_env_resolved"] is True,
                "worker did not resolve batch-invariant mode",
            )
            report["gpu_identity"] = {
                k: backend.provenance[k]
                for k in (
                    "gpu_uuid",
                    "physical_gpu_id",
                    "logical_cuda_index",
                    "cuda_visible_devices",
                    "gpu_name",
                )
            }
            report["runtime_provenance"] = {**backend.provenance, "batch_invariance": evidence}
            context = backend.worker.vllm_config.compilation_config.static_forward_context
            report["runtime_provenance"]["attention_implementations"] = [
                {
                    "layer": name,
                    "implementation": f"{type(layer.impl).__module__}.{type(layer.impl).__name__}",
                    "flash_attention_version": getattr(
                        layer.impl, "vllm_flash_attn_version", None
                    ),
                    "batch_invariant_enabled": getattr(
                        layer.impl, "batch_invariant_enabled", None
                    ),
                }
                for name, layer in context.items()
                if hasattr(layer, "impl")
            ]
            observer = ProbeCapture(backend, fixture, path)
            if path == PATHS[3]:
                report["history_comparisons"] = replay_persistent(backend, fixture, observer)
                report["history_rounds_replayed"] = 2
                report["d3_proposals_exact"] = all(
                    r["exact"] for r in report["history_comparisons"]
                )
            else:
                rows = initialize_fresh(backend, fixture, path)
                report["setup_request_order"] = [r["request_id"] for r in rows]
                report["setup_prefix_sha256"] = {
                    r["request_id"]: r["committed_prefix_sha256"] for r in rows
                }
                observer.round = 2
                observer.attach()
                run_fresh_forward(backend, rows)
            require(observer.capture is not None, "probe never captured the frozen boundary")
            report.update(observer.capture)
            report["captured_frame"] = observer.frames[observer.capture["frame_index"]]
            require(
                report["captured_frame"]["checks_passed"],
                "captured materialization did not complete cleanly",
            )
        report["valid"] = True
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
        report["traceback"] = traceback.format_exc()
    finally:
        if observer is not None:
            try:
                observer.close()
            except Exception as error:
                report["valid"] = False
                report["errors"].append(f"observer cleanup: {error}")
            report["diagnostic_frames"] = observer.report()
        try:
            if backend is not None:
                backend.shutdown()
                if path != PATHS[0]:
                    report["backend_report"] = backend.report()
            report["cleanup_complete"] = True
        except Exception as error:
            report["valid"] = False
            report["errors"].append(f"backend cleanup: {error}")
    return report


def run_all(config_path, expected_commit, output):
    fixture = load_probe_fixture()
    _, identity = preflight(config_path, expected_commit, fixture)
    output.mkdir(parents=True, exist_ok=False)
    write_immutable_report(
        output / "probe-inputs.json", {"fixture": fixture, "run_identity": identity}
    )
    reports, processes = {}, []
    for path in PATHS:
        print(f"Starting independent D3 probe path: {path}", flush=True)
        command = [
            sys.executable,
            "-m",
            "specrhythm.phase4.draft_logits_probe",
            "--allow-gpu",
            "--config",
            str(config_path),
            "--expected-commit",
            expected_commit,
            "--output",
            str(output),
            "--path",
            path,
        ]
        with (output / f"{path}.log").open("x") as log:
            with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT) as child:
                code = child.wait()
                processes.append({"path": path, "pid": child.pid, "returncode": code})
        result_path = output / f"{path}.json"
        if result_path.exists():
            report = json.loads(result_path.read_text())
            if report.get("process_pid") != processes[-1]["pid"] or code != 0:
                report = {**report, "valid": False, "process_error": processes[-1]}
        else:
            report = {
                "schema_version": PATH_SCHEMA,
                "path": path,
                "valid": False,
                "errors": ["path process failed before writing its report"],
                "process_pid": processes[-1]["pid"],
                "returncode": code,
            }
            write_immutable_report(result_path, report)
        reports[path] = report
        print(f"Completed {path}: rc={code}, capture_valid={report.get('valid')}", flush=True)
    result = classify_paths(fixture, reports)
    result["processes"] = processes
    result["path_artifact_sha256"] = {p: sha256_file(output / f"{p}.json") for p in PATHS}
    write_immutable_report(output / "classification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--path", choices=PATHS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.check_only:
        _, identity = preflight(args.config, args.expected_commit, load_probe_fixture())
        print(json.dumps({"cpu_preflight_valid": True, "run_identity": identity}, indent=2))
        return 0
    if not args.allow_gpu or args.output is None:
        parser.error("probe requires --allow-gpu and a fresh --output")
    if args.path:
        fixture = load_probe_fixture()
        config, identity = preflight(args.config, args.expected_commit, fixture)
        envelope = json.loads((args.output / "probe-inputs.json").read_text())
        require(
            envelope == {"fixture": fixture, "run_identity": identity},
            "path inputs/model files changed since preflight",
        )
        require(not (args.output / f"{args.path}.json").exists(), "path report already exists")
        report = run_path(args.path, config, fixture, identity)
        write_immutable_report(args.output / f"{args.path}.json", report)
        return 0 if report["valid"] else 1
    result = run_all(args.config, args.expected_commit, args.output)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("classification", "evidence_status", "top1_pattern", "errors")
            },
            indent=2,
        )
    )
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
