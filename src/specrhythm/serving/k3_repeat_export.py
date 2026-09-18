"""Flatten two bounded child exports into one digest-addressed operator archive."""

import hashlib
import io
import json
import tarfile
import tempfile
from pathlib import Path

from specrhythm.serving.common import read_json, require
from specrhythm.serving.k3_repeat_run import REPEATS, compare, read_declaration


def export_repeats(directory, output, *, first_code, stage, modes):
    from specrhythm.serving.ping_prepost_delivery import export

    declaration_error = None
    try:
        config = read_declaration(directory)
    except (OSError, ValueError, KeyError, TypeError) as error:
        config = dict(configuration=None, validation_profile=None, output_equivalence_status=None)
        declaration_error = "experiment-plan.json: " + repr(error)
    require(not output.exists(), "new repeat archive required")
    inventory, logical, emitted, errors, children = [], {}, set(), [], []
    total = 0
    if declaration_error:
        errors.append(declaration_error)
    with tempfile.TemporaryDirectory(prefix="k3-export-", dir=directory.parent) as temp:
        with tarfile.open(output, "x:gz") as out:

            def add_bytes(name, data):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                out.addfile(info, io.BytesIO(data))

            def retain(row, data):
                nonlocal total
                if data is not None:
                    digest = hashlib.sha256(data).hexdigest()
                    require(
                        digest == row["sha256"] and len(data) == row["bytes"],
                        "child archive checksum mismatch",
                    )
                    obj = "objects/" + digest
                    if obj not in emitted:
                        require(
                            len(emitted) < 1024 and total + len(data) <= 8 * 1024**3,
                            "two-repeat archive byte/file budget exhausted",
                        )
                        add_bytes(obj, data)
                        emitted.add(obj)
                        total += len(data)
                    row = {**row, "object": obj}
                    logical[row["path"]] = obj
                require(len(inventory) < 1200, "two-repeat logical file budget exhausted")
                inventory.append(row)

            for name in REPEATS:
                root = directory / "repeats" / name
                if not root.exists():
                    children.append(dict(repeat=name, status="NOT_STARTED"))
                    retain(dict(path="repeats/" + name, status="MISSING_NOT_STARTED"), None)
                    if first_code == 0:
                        errors.append(name + ": required repetition not started")
                    continue
                path = Path(temp) / (name + ".tar.gz")
                outcome = (
                    read_json(root / "runner-outcome.json")
                    if (root / "runner-outcome.json").exists()
                    else {}
                )
                result = export(
                    root,
                    path,
                    first_code=outcome.get("first_exit_code", first_code),
                    stage=outcome.get("stage", stage),
                    modes=modes,
                )
                children.append(
                    dict(
                        repeat=name,
                        **{
                            k: result[k]
                            for k in (
                                "export_status",
                                "evidence_integrity",
                                "export_validation_exit_code",
                                "first_exit_code",
                                "failed_stage",
                            )
                        },
                    )
                )
                errors.extend(name + ": " + error for error in result["export_errors"])
                with tarfile.open(path, "r:gz") as child:
                    for row in result["inventory"]:
                        data = (
                            child.extractfile(row["object"]).read()
                            if (row.get("status") == "INCLUDED")
                            else None
                        )
                        retain({**row, "path": "repeats/" + name + "/" + row["path"]}, data)
            try:
                comparison = compare(directory)
            except (OSError, ValueError, KeyError, TypeError) as error:
                comparison = dict(valid=False, error=repr(error), status="INCOMPLETE")
                errors.append("comparison: " + repr(error))
            for name in (
                "experiment-plan.json",
                "validation-plan.json",
                "runner-outcome.json",
                "first-failure.json",
                "delivery-status.json",
                "storage-preflight.json",
                "runner.log", "repeat-wrapper-errors.json",
                "comparison.json",
            ):
                path = directory / name
                if name == "comparison.json":
                    data = json.dumps(comparison, indent=2).encode()
                elif path.exists():
                    require(path.stat().st_size <= 64 * 1024**2, "top-level evidence byte budget")
                    data = path.read_bytes()
                else:
                    continue
                retain(
                    dict(
                        path=name,
                        status="INCLUDED",
                        bytes=len(data),
                        sha256=hashlib.sha256(data).hexdigest(),
                    ),
                    data,
                )
            result = dict(
                **{**config, "schema_version": "specrhythm.k3-repeat-archive.v1"},
                inventory=inventory,
                logical_paths=logical,
                children=children,
                first_exit_code=first_code,
                failed_stage=stage,
                archive_integrity="COMPLETE",
                evidence_integrity="INCOMPLETE" if errors else "COLLECTED",
                export_status="INCOMPLETE" if errors else "COMPLETE",
                export_errors=errors,
                export_validation_exit_code=41
                if errors
                else 42
                if first_code == 0 and not comparison["valid"]
                else 0,
                limits=dict(
                    unique_payload_bytes=8 * 1024**3,
                    unique_files=1024,
                    logical_files=1200,
                    repetitions=2,
                ),
                unique_payload_bytes=total,
                source_results_unchanged=True,
                intentionally_not_run=[
                    dict(
                        path="joint/",
                        output_equivalence_status="NOT_RUN",
                        reason="explicit performance-exploration",
                    )
                ] if config.get("validation_profile") == "performance-exploration" else [],
            )
            add_bytes("inventory.json", json.dumps(result, indent=2).encode())
    return result
