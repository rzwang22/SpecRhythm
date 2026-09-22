"""Foreground K3 run on container-local storage, then one verified persistent copy."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path

from specrhythm.phase4.manifest import atomic_write_json, sha256_file
from specrhythm.serving.k3 import MODES
from specrhythm.serving.ping_prepost_delivery import export


def filesystem(path):
    """Read mount facts once at startup; never infer the physical backing medium."""
    path = Path(path).resolve()
    mounts = Path("/proc/self/mountinfo")
    if mounts.exists():
        candidates = []
        for line in mounts.read_text().splitlines():
            left, right = line.split(" - ", 1)
            fields, fs = left.split(), right.split()
            mount = Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[4]))
            if path == mount or mount in path.parents:
                candidates.append(
                    (len(str(mount)), dict(type=fs[0], mount=str(mount), source=fs[1]))
                )
        if candidates:
            return max(candidates, key=lambda r: r[0])[1]
        raise ValueError("resolved path lacks mountinfo evidence: " + str(path))
    # CPU development hosts only; server entry records Linux mountinfo above.
    result = subprocess.run(["/sbin/mount"], text=True, capture_output=True, check=True)
    candidates = []
    for line in result.stdout.splitlines():
        match = re.match(r"(.+) on (.+) \(([^, )]+)", line)
        if match:
            source, mount, kind = match.groups()
            if path == Path(mount) or Path(mount) in path.parents:
                candidates.append((len(mount), dict(type=kind, mount=mount, source=source)))
    if not candidates:
        raise ValueError("cannot identify development host mount: " + str(path))
    return max(candidates, key=lambda r: r[0])[1]


def prepare_local(local_base, persistent, tag, *, minimum_free_bytes=8 * 1024**3):
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", tag) or tag in (".", ".."):
        raise ValueError("invalid unique run tag")
    if type(minimum_free_bytes) is not int or minimum_free_bytes < 1:
        raise ValueError("minimum local free bytes must be a positive integer")
    requested = Path(local_base).absolute()
    requested.mkdir(parents=True, exist_ok=True)
    base, persistent = requested.resolve(), Path(persistent).resolve()
    persistent.mkdir(parents=True, exist_ok=True)
    local_fs, persistent_fs = filesystem(base), filesystem(persistent)
    remote_types = {"dpc", "nfs", "nfs4", "cifs", "smb3", "fuse.dpc"}
    if local_fs["type"].lower() in remote_types:
        raise ValueError("mutable run directory resolves to remote filesystem: " + str(base))
    if base == persistent or base in persistent.parents or persistent in base.parents:
        raise ValueError("local run and persistent delivery roots must be separate")
    free = shutil.disk_usage(base).free
    if free < minimum_free_bytes:
        raise ValueError(f"local free bytes {free} below declared minimum {minimum_free_bytes}")
    root = base / tag
    root.mkdir()  # Never reuse any prior run, including a failed one.
    fd, probe = tempfile.mkstemp(prefix=".write-probe-", dir=root)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(b"specrhythm storage preflight\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        Path(probe).unlink(missing_ok=True)
    receipt = dict(
        schema_version="specrhythm.k3-local-storage.v1",
        tag=tag,
        requested_local_base=str(requested),
        resolved_local_base=str(base),
        local_run_root=str(root),
        persistent_directory=str(persistent),
        local_filesystem=local_fs,
        persistent_filesystem=persistent_fs,
        free_bytes=free,
        minimum_free_bytes=minimum_free_bytes,
        writable_probe="PASS; not a general filesystem consistency proof",
        physical_medium="UNKNOWN; overlay does not imply local NVMe",
        retention="KEEP local directory and archive after success or failure",
        setup_seconds=900,
        drain_seconds=60,
        report_publication_required=True,
    )
    atomic_write_json(root / "storage-preflight.json", receipt)
    return root, persistent, receipt


def validate_archive(path):
    """Archive validity is independent of invalid GPU/cleanup/JSON evidence inside it."""
    with tarfile.open(path, "r:gz") as archive:
        inventory = json.load(archive.extractfile("inventory.json"))
        for item in inventory["inventory"]:
            if item.get("status") != "INCLUDED":
                continue
            data = archive.extractfile(item["object"]).read()
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("archive payload digest/size mismatch: " + item["path"])
    return dict(
        bytes=path.stat().st_size,
        sha256=sha256_file(path),
        archive_integrity="VERIFIED",
        evidence_integrity=inventory.get("evidence_integrity"),
        export_validation_exit_code=inventory["export_validation_exit_code"],
    )


def deliver(local, persistent, expected):
    """Copy across filesystems, re-read the unique temporary target, then publish."""
    destination = persistent / local.name
    temporary = persistent / ("." + local.name + "." + uuid.uuid4().hex + ".copying")
    lock = persistent / ("." + local.name + ".delivery-lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        with local.open("rb") as source, temporary.open("xb") as target:
            shutil.copyfileobj(source, target, 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        if (
            temporary.stat().st_size != expected["bytes"]
            or sha256_file(temporary) != expected["sha256"]
        ):
            raise ValueError("persistent copy size/SHA256 verification mismatch")
        # This rename is only within the destination filesystem, after byte copy.
        temporary.rename(destination)
        return dict(status="VERIFIED", path=str(destination), **expected)
    except BaseException as error:
        error.delivery_temporary = str(temporary)  # Keep failed copies for diagnosis.
        raise
    finally:
        lock.unlink(missing_ok=True)


def run(repo, commit, *, local_base, persistent, tag, minimum_free_bytes=8 * 1024**3,
        configuration="k3-b16-v1", validation_profile=None, dual_batch=False,
        cpu_comparison=False):
    from specrhythm.serving.k3 import geometry
    from specrhythm.serving.k3_validation import fields

    policy = fields(validation_profile, configuration)
    modes = () if dual_batch or cpu_comparison else MODES
    batch = geometry("serial-k3", configuration)["active_limit"]
    root, persistent, receipt = prepare_local(
        local_base, persistent, tag, minimum_free_bytes=minimum_free_bytes
    )
    directory = root / ("pingpong-k3-delivery-" + tag)
    archive = root / (directory.name + ".tar.gz")
    env = {
        **os.environ,
        "SR_K3_MANAGED_LOCAL": "1",
        "SR_K3_LOCAL_DELIVERY": str(directory),
        "SR_K3_STORAGE_RECEIPT": str(root / "storage-preflight.json"),
        "SR_PING_RUN_TAG": tag,
        "SR_EXEC_REPO": str(repo),
        "PYTHONPATH": str(repo / "src"),
    }
    if policy:
        env["SR_K3_VALIDATION_PROFILE"] = policy["validation_profile"]
    print(json.dumps(receipt), flush=True)
    with (root / "runner.log").open("w") as log:
        process = subprocess.Popen(
            ["bash", str(repo / ("scripts/run_ordinary_cpu.sh" if cpu_comparison else
                                "scripts/run_dual_batch.sh" if dual_batch else
                                f"scripts/run_k3_b{batch}.sh")), commit],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            log.write(line)
            sys.stdout.write(line)
            sys.stdout.flush()
        code = process.wait()
    raw_code = code
    code = 128 - code if code < 0 else code
    directory.mkdir(exist_ok=True)
    shutil.copyfile(root / "runner.log", directory / "runner.log")
    shutil.copyfile(root / "storage-preflight.json", directory / "storage-preflight.json")
    outcome = directory / "runner-outcome.json"
    try:
        first = json.loads(outcome.read_text())
        if type(first["first_exit_code"]) is not int or not isinstance(first["stage"], str):
            raise ValueError("invalid runner outcome")
        if first["first_exit_code"] != code and first["first_exit_code"] != 0:
            raise ValueError("runner outcome disagrees with child exit code")
    except (OSError, ValueError, KeyError, TypeError) as error:
        first = dict(first_exit_code=code, stage="runner", outcome_error=str(error))
        # No invented success when the child did not publish its completion.
        if code == 0:
            code = 44
    state = dict(
        schema_version="specrhythm.k3-delivery-state.v1",
        inner_exit_code=code,
        inner_raw_returncode=raw_code,
        execution_first_exit_code=first["first_exit_code"],
        stage=first["stage"],
        runner_outcome=first,
        status="PENDING_AT_ARCHIVE_SEAL",
        local_run_root=str(root),
        persistent_directory=str(persistent),
        retention="KEEP",
        timestamp_ns=time.monotonic_ns(),
    )
    atomic_write_json(directory / "delivery-status.json", state)
    export_code = 0
    upload = None
    try:
        result = export(
            directory,
            archive,
            first_code=first["first_exit_code"],
            stage=first["stage"],
            modes=modes,
        )
        export_code = result["export_validation_exit_code"]
        from specrhythm.serving.delivery_budget import export_summary

        state["export_summary"] = export_summary(result)
        print(json.dumps(state["export_summary"]), flush=True)
        verified = validate_archive(archive)
        state["local_archive"] = dict(path=str(archive), **verified)
        upload = archive
    except Exception as error:
        export_code = 41
        state.update(status="ARCHIVE_FAILED", archive_error=str(error))
    delivery_code = 0
    if upload is not None:
        try:
            state["persistent_copy"] = deliver(archive, persistent, verified)
            upload = Path(state["persistent_copy"]["path"])
            state["status"] = "DELIVERED"
        except Exception as error:
            delivery_code = 43
            state.update(
                status="LOCAL_FALLBACK",
                delivery_error=str(error),
                delivery_exception=type(error).__name__,
                failed_copy=getattr(error, "delivery_temporary", None),
            )
            # Include the actual copy failure in the one fallback package. Retain
            # the first sealed package too; never retry GPU work or the DPC copy.
            fallback = archive.with_suffix(".fallback-partial")
            try:
                atomic_write_json(directory / "delivery-status.json", state)
                export(
                    directory,
                    fallback,
                    first_code=first["first_exit_code"],
                    stage=first["stage"],
                    modes=modes,
                )
                fallback_verified = validate_archive(fallback)
                os.link(archive, archive.with_suffix(".before-delivery"))
                fallback.replace(archive)
                state["local_archive"] = dict(path=str(archive), **fallback_verified)
            except Exception as secondary:
                state["fallback_repack_error"] = str(secondary)
                # The original locally verified package is still usable.
    state.update(
        export_exit_code=export_code,
        delivery_exit_code=delivery_code,
        final_exit_code=code or export_code or delivery_code,
        upload_path=str(upload) if upload is not None else None,
    )
    try:
        atomic_write_json(root / "delivery-status.json", state)
    except Exception as error:
        state["delivery_receipt_error"] = str(error)
        state["final_exit_code"] = state["final_exit_code"] or 44
    print(json.dumps(state), flush=True)
    if upload is not None:
        print("UPLOAD ONLY: " + str(upload), flush=True)
    else:
        print("NO ARCHIVE: retained evidence directory " + str(root), flush=True)
    return state["final_exit_code"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--commit", required=True)
    p.add_argument("--dual-batch", action="store_true")
    p.add_argument("--cpu-comparison", action="store_true")
    from specrhythm.serving.k3 import B16, CONFIGURATIONS

    p.add_argument("--k3-configuration", choices=CONFIGURATIONS, default=B16)
    from specrhythm.serving.k3_validation import PROFILES

    p.add_argument("--validation-profile", choices=PROFILES)
    args = p.parse_args()
    try:
        code = run(
            args.repo.resolve(),
            args.commit, configuration=args.k3_configuration,
            validation_profile=args.validation_profile, dual_batch=args.dual_batch,
            cpu_comparison=args.cpu_comparison,
            local_base=Path(os.environ.get("SR_K3_LOCAL_BASE", "/tmp/specrhythm-runs")),
            persistent=Path(
                os.environ.get(
                    "SR_PING_RESULTS", "/root/autodl-tmp/SpecRhythm-data/results/rolling-eager"
                )
            ),
            tag=os.environ.get(
                "SR_PING_RUN_TAG",
                time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + str(os.getpid()),
            ),
            minimum_free_bytes=int(os.environ.get("SR_K3_MIN_FREE_BYTES",
                (32 if args.k3_configuration == "k3-b128-v1" else 8) * 1024**3)),
        )
    except Exception as error:
        print("Local-run/delivery infrastructure failure: " + repr(error), file=sys.stderr)
        code = 44
    raise SystemExit(code)


if __name__ == "__main__":
    main()
